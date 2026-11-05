import streamlit as st
import numpy as np
from scipy.optimize import brentq, minimize_scalar  # [CORRECCIÓN k/solver] sustituye a fsolve
from scipy.stats import norm
import pandas as pd

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Diseño Pavimento Rigido - Subestaciones - AASHTO '93", page_icon="🏗️", layout="wide")

# --- FUNCIONES TÉCNICAS ---
def calcular_w18(tpd, periodo, crecimiento, peso_eje):
    p_patron = 8.2  # Ton (18 kips)
    alfa = 4.0
    fe = (peso_eje / p_patron)**alfa
    r = crecimiento / 100
    f_crec = periodo * 365 if r == 0 else ((1 + r)**periodo - 1) / r * 365
    return fe, tpd * fe * f_crec

def calcular_k_combinado(k_subrasante, espesor_base_cm, tipo_material):
    """
    Estima el k combinado (Losa sobre base) según AASHTO/PCA.
    k_subrasante: pci
    espesor_base_cm: cm
    tipo_material: 'Granular' o 'Tratada con Cemento'
    """
    h_pulg = espesor_base_cm / 2.54

    if h_pulg < 3: return k_subrasante

    if tipo_material == "Base Granular (Zahorra)":
        factor = 1 + (0.15 * np.log(h_pulg))
        k_nuevo = k_subrasante * factor

    elif tipo_material == "Suelo Cemento / Estabilizada":
        factor = 1 + (0.35 * np.log(h_pulg))
        k_nuevo = k_subrasante * factor * 1.25

    else:
        k_nuevo = k_subrasante

    return min(k_nuevo, 800.0)

# =========================================================
# [CORRECCIÓN k/solver] Correlación CBR–k con sus unidades reales
# =========================================================
# 1 MPa/m = 3.684 pci  (1 pci = 0.2714 MPa/m)
FACTOR_MPA_M_A_PCI = 3.684

def k_correlacion_cbr_mpa_m(cbr):
    """
    Correlación CBR–k de la subrasante. RESULTADO EN MPa/m (no en pci).
    Forma publicada (PCA; recogida en el Manual de Carreteras MTC Perú – Suelos,
    Geología, Geotecnia y Pavimentos):
        CBR <= 10 % : k = 2.55 + 52.5·log10(CBR)        [MPa/m]
        CBR >  10 % : k = 46.0 + 9.08·(log10(CBR))^4.34  [MPa/m]
    Nota: la versión anterior usaba 25.5 en lugar de 2.55 (discontinuidad en
    CBR = 10 %: 78.0 vs 55.1). Con 2.55 ambas ramas empalman (55.05 vs 55.08).
    Es una ESTIMACIÓN PRELIMINAR del k de la subrasante; no es el k efectivo
    de diseño que entra en AASHTO 93.
    """
    if cbr <= 10:
        return 2.55 + 52.5 * np.log10(cbr)
    return 46.0 + 9.08 * (np.log10(cbr))**4.34


# =========================================================
# [CORRECCIÓN k/solver] Solucionador robusto de la ecuación AASHTO 93
# =========================================================
def resolver_espesor_aashto(w18, zr, s0, p0, pt, sc, cd, j, ec, k, d_max_pulg=40.0, tol_residuo=1e-6):
    """
    Resuelve la ecuación AASHTO 93 (pavimento rígido) para D [pulg].
    Unidades: S'c [psi], Ec [psi], k [pci], W18 [ESAL], D [pulg].

    Dominio matemático: el logaritmo del término de materiales exige
    D^0.75 > 18.42/(Ec/k)^0.25 (y D^0.75 > 1.132). En D -> D_sing+ ese término
    tiende a +inf, de modo que el residuo g(D) = log10(W18_capacidad) - log10(W18)
    baja desde +inf hasta un mínimo y luego crece. La rama FÍSICA es la creciente
    (más espesor -> más capacidad). Una raíz en la rama decreciente es ESPURIA.

    Devuelve un dict con 'estado':
      'ok'                   -> raíz física hallada y verificada (|residuo| <= tol)
      'parametros_invalidos' -> datos fuera de rango físico / dominio
      'sin_raiz'             -> g_min > 0: todo espesor del dominio supera W18
                                (la ecuación no gobierna el espesor)
      'fuera_de_rango'       -> se requeriría D > d_max_pulg
      'no_convergencia'      -> fallo numérico o raíz no verificada
    """
    res = {"estado": None, "D_pulg": None, "residuo": None, "mensaje": "",
           "D_singular": None, "D_g_min": None, "g_min": None,
           "D_espuria": None, "advertencias": []}

    # --- 1. Validación de parámetros ---
    errores = []
    for nombre, v in [("W18", w18), ("S0", s0), ("S'c", sc), ("Cd", cd),
                      ("J", j), ("Ec", ec), ("k", k)]:
        if v is None or not np.isfinite(v) or v <= 0:
            errores.append(f"{nombre} debe ser > 0 (valor: {v})")
    if zr is None or not np.isfinite(zr):
        errores.append("Zr no es finito")
    if not (0 < pt < p0 <= 5.0):
        errores.append(f"Se requiere 0 < Pt < P0 <= 5 (P0={p0}, Pt={pt})")
    if errores:
        res["estado"] = "parametros_invalidos"
        res["mensaje"] = "; ".join(errores)
        return res

    d_psi = p0 - pt
    log_w18 = np.log10(w18)
    c_k = 18.42 / ((ec / k)**0.25)
    factor_ajuste = 4.22 - 0.32 * pt

    def g(D):
        term_conf = zr * s0
        term_esp = 7.35 * np.log10(D + 1) - 0.06
        term_serv = np.log10(d_psi / 3.0) / (1 + 1.624e7 / ((D + 1)**8.46))
        num = sc * cd * (D**0.75 - 1.132)
        den = 215.63 * j * (D**0.75 - c_k)
        term_mat = factor_ajuste * np.log10(num / den)
        return (term_conf + term_esp + term_serv + term_mat) - log_w18

    # --- 2. Límite inferior del dominio válido ---
    d_sing = max(c_k, 1.132)**(4.0 / 3.0)
    res["D_singular"] = d_sing
    if d_sing >= d_max_pulg:
        res["estado"] = "parametros_invalidos"
        res["mensaje"] = f"El dominio válido (D > {d_sing:.2f} pulg) queda fuera del rango de búsqueda."
        return res
    d_lo = d_sing * (1 + 1e-9) + 1e-9

    try:
        # --- 3. Mínimo de g: separa la rama espuria (izq.) de la física (der.) ---
        malla = d_sing + np.geomspace(1e-6, d_max_pulg - d_sing, 4000)
        g_malla = np.array([g(x) for x in malla])
        i_min = int(np.nanargmin(g_malla))
        a = malla[max(i_min - 1, 0)]
        b = malla[min(i_min + 1, len(malla) - 1)]
        opt = minimize_scalar(g, bounds=(max(a, d_lo), b), method="bounded", options={"xatol": 1e-10})
        d_gmin, g_min = float(opt.x), float(opt.fun)
        res["D_g_min"], res["g_min"] = d_gmin, g_min

        # Raíz espuria (rama decreciente, junto a la singularidad): solo informativa
        if g_min < 0 and g(d_lo) > 0:
            res["D_espuria"] = brentq(g, d_lo, d_gmin, xtol=1e-12)

        # --- 4. Clasificación ---
        if g_min > 0:
            res["estado"] = "sin_raiz"
            res["mensaje"] = (f"No existe raíz: el mínimo del residuo es g = {g_min:+.4f} > 0 "
                              f"(en D = {d_gmin:.2f} pulg = {d_gmin*2.54:.2f} cm). Toda losa del dominio "
                              f"válido (D > {d_sing:.2f} pulg = {d_sing*2.54:.2f} cm) tiene capacidad teórica "
                              f"mayor que W18; AASHTO 93 no define el espesor con estos parámetros.")
            return res

        if g(d_max_pulg) < 0:
            res["estado"] = "fuera_de_rango"
            res["mensaje"] = f"Se requeriría D > {d_max_pulg:.0f} pulg ({d_max_pulg*2.54:.0f} cm)."
            return res

        # Unicidad en la rama física (un único cambio de signo a la derecha del mínimo)
        cambios = int(np.sum(np.diff(np.sign(g_malla[i_min:])) != 0))
        if cambios > 1:
            res["estado"] = "no_convergencia"
            res["mensaje"] = f"La rama física presenta {cambios} cambios de signo; raíz ambigua."
            return res

        # --- 5. Raíz física por método acotado (brentq) y verificación del residuo ---
        D = brentq(g, d_gmin, d_max_pulg, xtol=1e-12, maxiter=200)
        residuo = g(D)
        res["residuo"] = residuo
        if not np.isfinite(residuo) or abs(residuo) > tol_residuo:
            res["estado"] = "no_convergencia"
            res["mensaje"] = f"Raíz no verificada (residuo = {residuo:.2e})."
            return res

        res["estado"] = "ok"
        res["D_pulg"] = D
        res["mensaje"] = f"Raíz física verificada (|residuo| = {abs(residuo):.1e})."
        if not (3.5 <= D <= 12.5):
            res["advertencias"].append(
                f"D = {D:.2f} pulg está fuera del rango de espesores del AASHO Road Test (≈3.5–12.5 pulg): extrapolación.")
        return res
    except Exception as e:  # fallo numérico: no se confunde con ausencia de raíz
        res["estado"] = "no_convergencia"
        res["mensaje"] = f"Fallo numérico del solucionador: {e}"
        return res


def calcular_espesor_aashto(w18, zr, s0, p0, pt, sc, cd, j, ec, k):
    """Interfaz anterior (D en pulg o None). Usa el solucionador robusto."""
    r = resolver_espesor_aashto(w18, zr, s0, p0, pt, sc, cd, j, ec, k)
    return r["D_pulg"] if r["estado"] == "ok" else None

# =========================================================
# [MÓDULO k AASHTO 93] Determinación del módulo de reacción efectivo
# Referencias:
#  [R1] AASHTO (1993). Guide for Design of Pavement Structures. Parte II, Cap. 3, sec. 3.2
#       (k = MR/19.4 sin subbase; Fig. 3.3 k compuesto; Fig. 3.4 estrato rígido; Fig. 3.5 daño relativo;
#       Fig. 3.6 pérdida de soporte; Tabla 2.7 rangos de LS).
#  [R2] FHWA NHI-05-037 (2006). Geotechnical Aspects of Pavements, sec. 5.4.6, Ecs. 5.20–5.24,
#       Tabla 5-45 (LS). Ecs. 5.22–5.23 = regresiones de AASHTO (1986) Apéndice LL.
#  [R3] García Aladín, Pérez Ruíz, Serrano Guzmán & Noguera Calvache (2024). Desarrollo de ecuaciones
#       sustitutivas ... Revista Eduweb 18(1), 9-23. doi:10.46502/issn.1856-7576/2024.18.01.1
#       (regresiones de la Fig. 3.6 por LS, digitalizadas para keff = 5–2000 pci; interpolación lineal
#       entre líneas LS empleada en su ejemplo resuelto, LS = 1.67).
#  [R4] Powell, Potter, Mayhew & Nunn (1984). TRRL LR 1132 (E = 17.6·CBR^0.64 MPa, 2 ≤ CBR ≤ 12);
#       NCHRP 1-37A (2004): MR = 2555·CBR^0.64 psi.
# =========================================================
PSI_POR_MPA = 145.0377
PULG_POR_CM = 1 / 2.54

# Solo correlaciones con fuente primaria identificada y rango declarado
CORRELACIONES_MR = {
    "AASHTO 93 (Heukelom-Klomp): MR = 1500·CBR": {
        "ref": "[R1] AASHTO 1993, Parte I, sec. 1.5; suelos finos con CBR saturado ≤ 10 %",
        "cbr_min": 0.0, "cbr_max": 10.0,
        "f": lambda c: 1500.0 * c},
    "TRRL / NCHRP 1-37A: MR = 2555·CBR^0.64": {
        "ref": "[R4] Powell et al. 1984 (17.6·CBR^0.64 MPa), rango de calibración 2 ≤ CBR ≤ 12 %; adoptada en NCHRP 1-37A",
        "cbr_min": 2.0, "cbr_max": 12.0,
        "f": lambda c: 2555.0 * c**0.64},
}

ESTRUCTURAS = ["Losa directamente sobre subrasante",
               "Losa sobre subbase granular",
               "Losa sobre base tratada con cemento",
               "Losa sobre otra capa de apoyo (propiedades manuales)"]

# Tabla 2.7 AASHTO 93 / Tabla 5-45 FHWA: rangos de LS (y de E) por material de apoyo de la losa
TABLA_LS = {
    "Base granular tratada con cemento (E = 1.000.000–2.000.000 psi)": (0.0, 1.0),
    "Mezclas de agregado-cemento (E = 500.000–1.000.000 psi)": (0.0, 1.0),
    "Base tratada con asfalto (E = 350.000–1.000.000 psi)": (0.0, 1.0),
    "Mezclas estabilizadas bituminosas (E = 40.000–300.000 psi)": (0.0, 1.0),
    "Estabilizado con cal (E = 20.000–70.000 psi)": (1.0, 3.0),
    "Materiales granulares no tratados (E = 15.000–45.000 psi)": (1.0, 3.0),
    "Suelo fino o subrasante natural (E = 3.000–40.000 psi)": (2.0, 3.0),
}
_MAT = list(TABLA_LS.keys())
# Rangos de módulo E de la misma Tabla 2.7 (psi): guía para ESB cuando no hay ensayo
TABLA_E = dict(zip(_MAT, [(1.0e6, 2.0e6), (5.0e5, 1.0e6), (3.5e5, 1.0e6), (4.0e4, 3.0e5),
                          (2.0e4, 7.0e4), (1.5e4, 4.5e4), (3.0e3, 4.0e4)]))
# Materiales de la Tabla 2.7 que corresponden a cada tipo de capa de apoyo
MATERIALES_CAPA = {"Losa sobre subbase granular": [_MAT[5], _MAT[4]],
                   "Losa sobre base tratada con cemento": [_MAT[0], _MAT[1]],
                   "Losa sobre otra capa de apoyo (propiedades manuales)": _MAT}
# Orientación para ubicar LS dentro del intervalo de la Tabla 2.7 (factores que AASHTO 93 asocia a LS:
# erosión del material de apoyo y movimientos verticales diferenciales del suelo)
ORIENTACION_LS = {
    "Erosión baja → límite inferior del intervalo": (0.0,
        "material de apoyo no erosionable (tratado/estabilizado) o granular limpio bien compactado; drenaje bueno "
        "(Cd ≥ 1,0); sin suelos expansivos ni colapsables; bombeo no esperado"),
    "Erosión moderada → valor medio del intervalo": (0.5,
        "condiciones intermedias: algo de finos en el apoyo o drenaje regular; tránsito pesado ocasional en juntas"),
    "Erosión alta → límite superior del intervalo": (1.0,
        "finos erosionables bajo la losa, drenaje deficiente (Cd < 1,0), juntas sin pasadores con tránsito pesado frecuente, "
        "bombeo esperado o asentamientos/expansiones diferenciales"),
}
MATERIAL_LS_POR_ESTRUCTURA = {ESTRUCTURAS[0]: _MAT[6], ESTRUCTURAS[1]: _MAT[5], ESTRUCTURAS[2]: _MAT[0], ESTRUCTURAS[3]: _MAT[5]}

# Puntos digitalizados de la Fig. 3.6 publicados en [R3] (Tabla 1): keff -> k corregido (pci)
NOMOGRAMA_LS_PUNTOS = {
    1: [(5, 3.395), (10, 5.82), (100, 39.93), (500, 162.26), (1000, 274.425), (2000, 500)],
    2: [(5, 2.35), (10, 3.63), (100, 16.65), (500, 44.90), (1000, 68.68), (2000, 114.89)],
    3: [(5, 1.625), (10, 2.3065), (100, 7.772), (500, 19.115), (1000, 27.736), (2000, 39.95)],
}
LS_COEF = {1: (0.8311, -0.0643), 2: (0.6408, -0.0834), 3: (0.538, -0.1806)}  # [R3] Ecs. 2–4


def _linea_ls(keff, n):
    if n == 0:
        return keff
    a, b = LS_COEF[n]
    return 10 ** (a * np.log10(keff) + b)


def k_corregido_ls(keff, ls):
    """Corrección por pérdida de soporte (Fig. 3.6). LS entero: línea del nomograma ([R3] Ecs. 1–4).
    LS no entero: interpolación lineal en k entre las líneas adyacentes, procedimiento del ejemplo
    resuelto de [R3] (LS = 1.67: 189 y 53 pci -> 98 pci). Se aplica UNA sola vez."""
    ls = float(ls)
    lo, hi = int(np.floor(ls)), int(np.ceil(ls))
    if lo == hi:
        return _linea_ls(keff, lo)
    k_lo, k_hi = _linea_ls(keff, lo), _linea_ls(keff, hi)
    return k_lo + (ls - lo) * (k_hi - k_lo)


def k_compuesto_subbase(mr_psi, dsb_pulg, esb_psi):
    """k∞ compuesto (Fig. 3.3), [R2] Ec. 5.22. Unidades: psi, pulg -> pci."""
    l_d, l_e, l_m = np.log(dsb_pulg), np.log(esb_psi), np.log(mr_psi)
    return float(np.exp(-2.807 + 0.1253 * l_d**2 + 1.062 * l_m + 0.1282 * l_d * l_e
                        - 0.4114 * l_d - 0.0581 * l_e - 0.1317 * l_d * l_m))


def k_estrato_rigido(k_inf, mr_psi, dsg_pulg):
    """Corrección por estrato rígido a < 10 ft (Fig. 3.4). Forma con ln(DSG)·ln(MR) ([R3] Ec. 8),
    contrastada con el ejemplo de la Fig. 3.4 (MR = 4000 psi, DSG = 5 ft, k∞ = 230 pci -> ≈300 pci).
    Nota: el texto de [R2] Ec. 5.23 muestra ln(DSB) en ese término; esa forma no reproduce el ejemplo."""
    l_g, l_m = np.log(dsg_pulg), np.log(mr_psi)
    return float(np.exp(5.303 + 0.0710 * l_g * l_m + 1.366 * np.log(k_inf) - 0.9187 * l_g - 0.6837 * l_m))


def dano_relativo(k_pci, d_pulg):
    """Daño relativo (Fig. 3.5), [R2] Ec. 5.24: ur = (D^0.75 − 0.39·k^0.25)^3.42."""
    base = d_pulg**0.75 - 0.39 * k_pci**0.25
    return base**3.42 if base > 0 else np.nan


def k_desde_dano(ur, d_pulg):
    """Inversa de la Ec. 5.24: k efectivo para el daño relativo promedio."""
    return ((d_pulg**0.75 - ur**(1 / 3.42)) / 0.39)**4


def determinar_k_aashto(estructura, mr_estaciones_psi, d_proy_pulg, dsb_pulg=None, esb_psi=None,
                        k_placa_pci=None, estrato="No existe a < 3 m (según estudio geotécnico)", dsg_pulg=None,
                        k_inf_manual=None, k_rf_manual=None, placa_sobre_capa=False):
    """
    Pasos [R1] sec. 3.2: k inicial -> k∞ compuesto -> corrección por estrato rígido -> daño relativo -> keff
    (ANTES de LS; la corrección LS se aplica aparte, una sola vez).
    Devuelve dict con filas por periodo, keff, origen de cada paso, pendientes y advertencias.
    """
    out = {"filas": [], "pendientes": [], "advertencias": [], "keff": None,
           "origen": {"k_ini": "No aplica", "k_inf": "No aplica", "k_rf": "No aplica"}}
    con_capa = estructura != ESTRUCTURAS[0]
    if con_capa and k_inf_manual is None and not (k_placa_pci and placa_sobre_capa) and (not dsb_pulg or not esb_psi):
        out["pendientes"].append("Faltan espesor o módulo de la capa de apoyo.")
        return out
    if (k_inf_manual is not None or k_rf_manual is not None) and len(mr_estaciones_psi) > 1:
        out["advertencias"].append("Valor manual de nomograma aplicado igual a todos los periodos estacionales.")
    urs = []
    for i, mr in enumerate(mr_estaciones_psi):
        # 1. k inicial de la subrasante
        if k_placa_pci and not placa_sobre_capa:
            k_ini = k_placa_pci
            out["origen"]["k_ini"] = "Ensayo de placa sobre subrasante"
            mr_eq = 19.4 * k_ini if mr is None else mr     # relación k = MR/19.4 [R1] invertida
        elif mr:
            k_ini = mr / 19.4
            out["origen"]["k_ini"] = "Calculado: k = MR/19.4 [R1]"
            mr_eq = mr
        else:
            k_ini, mr_eq = None, None
        # 2. k compuesto
        if k_inf_manual is not None:
            k_inf = k_inf_manual
            out["origen"]["k_inf"] = "Manual (nomograma / fuente externa)"
        elif k_placa_pci and placa_sobre_capa:
            k_inf = k_placa_pci
            out["origen"]["k_inf"] = "Ensayo de placa sobre la capa de apoyo"
        elif con_capa:
            k_inf = k_compuesto_subbase(mr_eq, dsb_pulg, esb_psi)
            out["origen"]["k_inf"] = "Calculado: Ec. 5.22 (Fig. 3.3) [R2]"
            if estructura == ESTRUCTURAS[1]:
                if not (15000 <= esb_psi <= 50000):
                    out["advertencias"].append(f"ESB = {esb_psi:,.0f} psi fuera de 15.000–50.000 psi para subbase granular ([R2] Ec. 5.20).")
                if esb_psi > 4 * mr_eq:
                    out["advertencias"].append(f"ESB > 4·MR ([R2] Ec. 5.21): reduzca ESB.")
            if esb_psi < mr_eq:
                out["advertencias"].append("ESB < MR: la capa de apoyo es menos rígida que la subrasante; la Ec. 5.22 "
                                           "reduce k∞ respecto de MR/19.4. Verifique datos o use valor manual del nomograma.")
            if estructura in (ESTRUCTURAS[2], ESTRUCTURAS[3]):
                out["advertencias"].append("Ec. 5.22 aplicada a una capa tratada/otra: confirme el resultado contra la Fig. 3.3 "
                                           "(o ingrese el valor leído) antes de usarlo como sustentado.")
        else:
            k_inf = k_ini
            out["origen"]["k_inf"] = "No aplica (sin capa de apoyo): k∞ = k inicial"
        # 3. estrato rígido
        k_rf = k_inf
        if k_rf_manual is not None:
            k_rf = k_rf_manual
            out["origen"]["k_rf"] = "Manual (Fig. 3.4 / fuente externa)"
        elif estrato.startswith("Existe"):
            if dsg_pulg and dsg_pulg < 120 and mr_eq:
                k_rf = k_estrato_rigido(k_inf, mr_eq, dsg_pulg)
                out["origen"]["k_rf"] = "Calculado: ecuación de la Fig. 3.4 [R3] Ec. 8"
            else:
                out["pendientes"].append("Estrato rígido declarado sin DSG o sin MR: corrección (Fig. 3.4) pendiente.")
                out["origen"]["k_rf"] = "PENDIENTE"
        elif estrato.startswith("Sin verificar"):
            out["pendientes"].append("Profundidad de estrato rígido NO verificada: corrección Fig. 3.4 pendiente.")
            out["origen"]["k_rf"] = "PENDIENTE"
        ur = dano_relativo(k_rf, d_proy_pulg)
        urs.append(ur)
        out["filas"].append({"Periodo": i + 1, "MR (psi)": round(mr_eq) if mr_eq else None,
                             "MR (MPa)": round(mr_eq / PSI_POR_MPA, 1) if mr_eq else None,
                             "k inicial (pci)": round(k_ini, 1) if k_ini else None,
                             "k∞ (pci)": round(k_inf, 1), "k estrato (pci)": round(k_rf, 1), "ur": round(ur, 2)})
    out["pendientes"] = list(dict.fromkeys(out["pendientes"]))
    out["advertencias"] = list(dict.fromkeys(out["advertencias"]))
    if any(np.isnan(u) for u in urs):
        out["pendientes"].append("k fuera del rango de la Fig. 3.5 para el espesor proyectado.")
        return out
    ur_m = float(np.mean(urs))
    out["ur_medio"] = ur_m
    out["keff"] = k_desde_dano(ur_m, d_proy_pulg)
    return out


def verificar_espesor(res_solver, h_proy_cm, pendientes=(), condicionantes=()):
    """
    CUMPLE: espesor requerido <= h de proyecto. NO CUMPLE: requerido > h de proyecto.
    NO CONCLUYENTE: sin raíz, fuera de aplicabilidad o condiciones pendientes.
    Una ecuación sin raíz NUNCA se interpreta como cumplimiento.
    """
    if res_solver["estado"] != "ok":
        return "NO CONCLUYENTE", f"ecuación sin solución válida ({res_solver['estado']})"
    d_cm = res_solver["D_pulg"] * 2.54
    base = "CUMPLE" if d_cm <= h_proy_cm + 1e-9 else "NO CUMPLE"
    if pendientes:
        return "NO CONCLUYENTE", f"indicativo: {base} ({d_cm:.2f} cm vs {h_proy_cm:.1f} cm); hay condiciones pendientes"
    if condicionantes:
        return base, f"{d_cm:.2f} cm vs {h_proy_cm:.1f} cm; condicionado a: " + "; ".join(condicionantes)
    return base, f"{d_cm:.2f} cm vs {h_proy_cm:.1f} cm"


# --- INTERFAZ ---
st.title("🏗️ Diseñador Pavimento Rigido - Subestaciones")

# Visualización de todas las fórmulas de diseño
with st.expander("📝 Ecuaciones de Diseño (AASHTO 93 & Correlaciones)"):
    st.markdown("**1. Ecuación Estructural AASHTO 93 (Rígido):**")
    st.latex(r" \log_{10}(W_{18}) = Z_R S_0 + 7.35 \log_{10}(D + 1) - 0.06 + \frac{\log_{10}[\Delta PSI / (4.5 - 1.5)]}{1 + \frac{1.624 \times 10^7}{(D + 1)^{8.46}}} + (4.22 - 0.32p_t) \log_{10} \left[ \frac{S'_c C_d (D^{0.75} - 1.132)}{215.63 J \left( D^{0.75} - \frac{18.42}{(E_c/k)^{0.25}} \right)} \right] ")

    st.markdown("**2. Correlación de Módulo de Reacción (k) vs CBR — resultado en MPa/m (estimación preliminar):**")
    st.latex(r" \text{Si } CBR \leq 10\%: \quad k\,[\mathrm{MPa/m}] = 2.55 + 52.5 \log_{10}(CBR) ")
    st.latex(r" \text{Si } CBR > 10\%: \quad k\,[\mathrm{MPa/m}] = 46.0 + 9.08 (\log_{10}(CBR))^{4.34} ")
    st.latex(r" k\,[\mathrm{pci}] = 3.684 \cdot k\,[\mathrm{MPa/m}] ")
    st.caption("El k que entra en la ecuación AASHTO 93 es el **k efectivo de diseño (pci)**, definido de forma independiente en 'Parámetros de Diseño'.")

# =========================================================
# [AJUSTE FINAL] Perfil de proyecto: fija solo los valores INICIALES de los widgets.
# Todos los parámetros siguen siendo editables para otros proyectos.
# =========================================================
PERFILES = {
    # Perfil general (Colombia, Perú u otro país): valores típicos, todos editables
    "Genérico (Colombia, Perú u otro país)": {
        "tpd": 5, "cbr": 8.0, "usar_base": True, "idx_j": 0, "idx_fc": 4, "idx_pav": 0,
        "solado_cm": 0.0, "idx_origen_k": 0, "k_efectivo": 100.0,
        # Módulo k AASHTO 93
        "k_estructura": 1, "k_dsb_cm": 15.0, "k_esb_psi": 25000.0, "k_idx_fuente": 0,
        "k_idx_estrato": 0, "k_dsg_m": 2.0, "k_idx_mat_ls": 5, "k_ls": 1, "h_proy_cm": 20.0, "k_idx_modo_ls": 1,
        "fc_solado": 14.0, "suelo": "", "cbr_cond": "", "doc_geo": "", "plano": "",
        "ls_just": "",
    },
    "Hanaqpampa 220/33 kV (Perú)": {
        "tpd": 6,            # -> W18 = 177,296 ESAL (25 años, eje 11 t, sin crecimiento)
        "cbr": 68.9,
        "usar_base": False,  # sin subbase granular convencional
        "idx_j": 3,          # Escenario 4: sin pasadores y sin bermas (J = 4.20)
        "idx_fc": 4,         # 35 MPa
        "idx_pav": 3,        # factor S'c = 8.0
        "solado_cm": 5.0,    # solado de concreto pobre (sin aporte estructural)
        "idx_origen_k": 0,   # k por procedimiento AASHTO 93 (pestaña Módulo k)
        "k_efectivo": 173.5, # solo para la opción manual: HIPÓTESIS PROVISIONAL sin respaldo
        "k_estructura": 0,   # losa directamente sobre subrasante (el solado no se modela como subbase)
        "k_dsb_cm": 15.0, "k_esb_psi": 25000.0,
        "k_idx_fuente": 1,   # sin ensayo de MR/placa: TRRL/NCHRP como referencia (CBR 68.9 % fuera de rango -> NO CONCLUYENTE)
        "k_idx_estrato": 0,  # corrección por estrato rígido no considerada (opcional en la app)
        "k_idx_modo_ls": 0,  # erosión baja -> límite inferior del intervalo
        "k_dsg_m": 1.5, "k_idx_mat_ls": 6, "k_ls": 2, "h_proy_cm": 15.0,
        "fc_solado": 14.0, "suelo": "GP (grava mal gradada)",
        "cbr_cond": "CBR al 95 % de compactación; muestra a 0,80 m",
        "doc_geo": "EEP-HANQ-SEH-CV-ETC-1038", "plano": "EEP-HANQ-SEH-CV-DRW-1049",
        "ls_just": "Losa apoyada (vía solado sin función estructural) sobre subrasante natural GP.",
    },
}
perfil_sel = st.selectbox("📁 Perfil de proyecto (valores iniciales, todos editables):", list(PERFILES.keys()), index=0)
P = PERFILES[perfil_sel]

tab1, tab_k, tab2, tab3, tab4 = st.tabs(["🚛 Tránsito y Carga", "🪨 Módulo k (AASHTO 93)", "🧱 Parámetros de Diseño",
                                         "📐 Geometría y Juntas", "📊 Ábaco"])

with tab1:
    st.header("Análisis de Tránsito (Eje Crítico)")
    c1, c2 = st.columns(2)
    with c1:
        tpd = st.number_input("TPD (Vehículos pesados/día)", min_value=1, value=P["tpd"], step=1)
        periodo = st.number_input("Periodo de diseño (años)", min_value=1, value=25, step=1)
        tasa = st.number_input("Tasa de crecimiento (%)", min_value=0.0, value=0.0, step=0.1)
    with c2:
        peso_eje = st.number_input("Peso eje más pesado (Ton)", min_value=1.0, value=11.0, step=0.5)
        st.caption("ℹ️ El daño se calcula basándose en un **eje patrón de 8.2 Ton** (18 kips).")

    fe, w18_total = calcular_w18(tpd, periodo, tasa, peso_eje)
    st.metric("W18 Acumulado (ESALs)", f"{w18_total:,.0f}")

    if w18_total < 200000:
        st.warning("⚠️ **Nota Técnica:** El tránsito acumulado es bajo para el rango de aplicación original de AASHTO 93. El espesor obtenido está gobernado por criterios mínimos constructivos.")

# =========================================================
# [MÓDULO k AASHTO 93] Pestaña de determinación del k efectivo (entradas y pasos)
# =========================================================
with tab_k:
    st.header("🪨 Determinación del módulo de reacción efectivo – AASHTO 1993")
    st.caption("Procedimiento AASHTO 1993, Parte II, sec. 3.2: k inicial → k∞ compuesto según el apoyo → corrección por "
               "estrato rígido → daño relativo (Fig. 3.5) → keff → corrección por pérdida de soporte LS (una sola vez). "
               "Unidades internas: MR y E en psi, espesores en pulgadas, k en pci (1 MPa/m = 3.684 pci; 1 MPa = 145.04 psi).")
    ck1, ck2 = st.columns(2)

    with ck1:
        st.subheader("A. Propiedades de la subrasante")
        cbr_medido = st.number_input("CBR medido (%)", 1.0, 100.0, P["cbr"], step=0.1, key=f"k_cbr_{perfil_sel}")
        cbr_cond = st.text_input("Condición del ensayo (compactación, saturación, profundidad)", P["cbr_cond"])
        clas_suelo = st.text_input("Clasificación del suelo (SUCS / AASHTO)", P["suelo"])
        # [AJUSTE] El ensayo es opcional: si no se dispone, se estima MR con una correlación documentada
        _tipo_dato = st.radio("¿Se dispone de un ensayo en el nivel de apoyo de la losa?",
                              ["No — estimar MR a partir del CBR (correlación)",
                               "Sí — ensayo de módulo resiliente (MR)",
                               "Sí — ensayo de placa de carga (k)"], index=0, key=f"tipo_dato_{perfil_sel}")
        if _tipo_dato.startswith("No"):
            fuente_mr = st.selectbox("Correlación MR–CBR:", list(CORRELACIONES_MR.keys()),
                                     index=min(P["k_idx_fuente"], len(CORRELACIONES_MR) - 1))
            st.caption("Cuando se disponga del ensayo, selecciónelo arriba: reemplaza a la correlación.")
        elif "módulo resiliente" in _tipo_dato:
            fuente_mr = "Ensayo de módulo resiliente (MR medido)"
        else:
            fuente_mr = "Ensayo de placa de carga (k medido)"
        mr_fuera_rango, mr_hipotesis, k_placa_pci, mr_psi, placa_sobre_capa = False, False, None, None, False
        mr_origen = ""
        if fuente_mr in CORRELACIONES_MR:
            corr = CORRELACIONES_MR[fuente_mr]
            mr_psi = corr["f"](cbr_medido)
            mr_origen = f"Calculado: {fuente_mr} — {corr['ref']}"
            st.caption(f"Fuente: {corr['ref']}")
            if not (corr["cbr_min"] <= cbr_medido <= corr["cbr_max"]):
                mr_fuera_rango = True
                st.error(f"🚨 CBR = {cbr_medido:.1f} % FUERA del rango de validez de la correlación "
                         f"({corr['cbr_min']:.0f}–{corr['cbr_max']:.0f} %). MR = {mr_psi:,.0f} psi es una EXTRAPOLACIÓN: "
                         "no apta para sustentar el diseño. Use un ensayo de MR o de placa.")
        elif fuente_mr.startswith("Ensayo de módulo"):
            mr_mpa_in = st.number_input("MR medido (MPa) – AASHTO T 307 o norma nacional equivalente", 5.0, 1000.0, 60.0)
            mr_psi = mr_mpa_in * PSI_POR_MPA
            mr_origen = "Ingresado: ensayo de módulo resiliente"
        else:
            k_placa_mpa = st.number_input("k de placa medido (MPa/m)", 5.0, 500.0, 40.0)
            k_placa_pci = k_placa_mpa * FACTOR_MPA_M_A_PCI
            placa_sobre_capa = st.radio("Ensayo de placa ejecutado sobre:", ["La subrasante", "La capa de apoyo terminada"],
                                        horizontal=True) == "La capa de apoyo terminada"
            mr_origen = "Ingresado: ensayo de placa" + (" sobre la capa de apoyo (k∞ directo)" if placa_sobre_capa
                                                        else " sobre subrasante (MR equivalente = 19.4·k [R1])")
            st.caption("k de placa estático. Verifique diámetro de placa (30 in en AASHTO T 222) y condición de humedad del ensayo.")
        if mr_psi:
            st.metric("Módulo resiliente MR", f"{mr_psi:,.0f} psi", delta=f"{mr_psi / PSI_POR_MPA:,.1f} MPa", delta_color="off")
            if mr_psi > 20000:
                st.warning("⚠️ MR > 20.000 psi: por encima de la escala habitual de los nomogramas AASHTO (Figs. 3.3 y 3.4). "
                           "Las ecuaciones de regresión se aplican fuera del rango graficado.")
        if _tipo_dato.startswith("No"):
            cbr_repr = st.checkbox("El CBR es representativo del nivel REAL de apoyo de la losa (verificado en el estudio geotécnico)",
                                   value=False)
        else:
            cbr_repr = True  # el ensayo se declara ejecutado en el nivel de apoyo de la losa

    with ck2:
        st.subheader("B. Configuración de la estructura")
        estructura = st.radio("Apoyo de la losa:", ESTRUCTURAS, index=P["k_estructura"])
        _ke = f"{perfil_sel}_{ESTRUCTURAS.index(estructura)}"
        dsb_pulg_k, esb_psi_k, nombre_capa, k_inf_manual, k_inf_man_origen = None, None, "", None, ""
        esb_guia, mat_capa = False, None
        if estructura != ESTRUCTURAS[0]:
            if estructura == ESTRUCTURAS[3]:
                nombre_capa = st.text_input("Descripción de la capa de apoyo", "", key=f"nom_{_ke}")
            dsb_cm_k = st.number_input("Espesor de la capa DSB (cm)", 5.0, 60.0, P["k_dsb_cm"], step=1.0, key=f"dsb_{_ke}")
            dsb_pulg_k = dsb_cm_k * PULG_POR_CM
            # [AJUSTE] Guía de ESB por material (Tabla 2.7) o valor manual
            _ops_mat = MATERIALES_CAPA[estructura]
            mat_capa = st.selectbox("Material de la capa (Tabla 2.7 AASHTO 93):", _ops_mat, key=f"matc_{_ke}")
            _emin, _emax = TABLA_E[mat_capa]
            modo_esb = st.radio("Valor de ESB:", ["Guía: límite inferior (conservador)", "Guía: valor medio",
                                                  "Guía: límite superior", "Manual (ensayo o especificación)"],
                                key=f"modo_esb_{_ke}")
            if modo_esb.startswith("Manual"):
                esb_psi_k = st.number_input("ESB (psi)", 3000.0, 3000000.0, float(_emin), step=1000.0, key=f"esb_{_ke}")
                esb_origen = st.text_input("Origen del ESB (ensayo, especificación técnica, informe)", "", key=f"esbo_{_ke}")
                esb_guia = False
            else:
                esb_psi_k = {"Guía: límite inferior (conservador)": _emin, "Guía: valor medio": (_emin + _emax) / 2,
                             "Guía: límite superior": _emax}[modo_esb]
                esb_origen = f"Guía del programa: Tabla 2.7 AASHTO 93, {mat_capa.split(' (')[0]} — {modo_esb.split(': ')[1]}"
                esb_guia = True
            st.caption(f"DSB = {dsb_pulg_k:.2f} pulg | ESB = {esb_psi_k:,.0f} psi = {esb_psi_k / PSI_POR_MPA:,.0f} MPa "
                       f"| Rango Tabla 2.7: {_emin:,.0f}–{_emax:,.0f} psi")
            if esb_guia:
                st.info("ESB de **referencia** (rango típico de la Tabla 2.7), no medido: el resultado queda condicionado a "
                        "confirmarlo con ensayo o especificación del material.")
            elif not (_emin <= esb_psi_k <= _emax):
                st.warning("ESB fuera del rango de la Tabla 2.7 para el material seleccionado.")
            if st.checkbox("Ingresar k∞ leído en la Fig. 3.3 u obtenido de otra fuente (en lugar de la Ec. 5.22)",
                           value=(estructura == ESTRUCTURAS[3]), key=f"kinfm_{_ke}"):
                k_inf_manual = st.number_input("k∞ compuesto (pci)", 20.0, 3000.0, 300.0, step=5.0, key=f"kinfv_{_ke}")
                k_inf_man_origen = st.text_input("Origen del valor (figura, lectura, informe)", "", key=f"kinfo_{_ke}")
        else:
            esb_origen = ""
        esp_solado_k = st.number_input("Solado de concreto pobre bajo la losa (cm)", 0.0, 15.0, P["solado_cm"], step=1.0, key=f"k_solado_{perfil_sel}")
        fc_solado = st.number_input("f'c del solado (MPa)", 5.0, 30.0, P["fc_solado"], step=1.0)
        if esp_solado_k > 0:
            st.info(f"ℹ️ **Simplificación:** el solado de {esp_solado_k:.0f} cm (f'c = {fc_solado:.0f} MPa) es una capa de limpieza: "
                    "NO se modela como capa de apoyo ni se le asigna capacidad estructural. Si se quiere considerar, "
                    "selecciónelo explícitamente como 'otra capa de apoyo' con sus propiedades.")

        st.subheader("C. Estrato rígido o roca cercana")
        estrato = st.radio("Estrato rígido (roca o capa muy rígida) a menos de 3 m (10 ft) del nivel de apoyo:",
                           ["No se considera (no hay estrato rígido a < 3 m)",
                            "Existe a < 3 m (indicar profundidad DSG)"], index=P["k_idx_estrato"],
                           help="Corrección opcional de la Fig. 3.4 AASHTO 93. Solo se aplica si el estudio geotécnico identifica "
                                "un estrato rígido a menos de 3 m.")
        dsg_pulg_k, k_rf_manual, k_rf_man_origen = None, None, ""
        if estrato.startswith("Existe"):
            dsg_m_k = st.number_input("DSG: profundidad desde el nivel de apoyo hasta el estrato rígido (m)", 0.3, 3.0, P["k_dsg_m"], step=0.1)
            dsg_pulg_k = dsg_m_k * 100 * PULG_POR_CM
            st.caption(f"DSG = {dsg_pulg_k:.1f} pulg. Ecuación de la Fig. 3.4 con ln(DSG)·ln(MR), "
                       "comprobada con el ejemplo de la Guía (230 → ≈300 pci).")
            if st.checkbox("Ingresar k corregido leído en la Fig. 3.4 (en lugar de la ecuación)", value=False):
                k_rf_manual = st.number_input("k corregido por estrato rígido (pci)", 20.0, 3000.0, 300.0, step=5.0)
                k_rf_man_origen = st.text_input("Origen del valor (Fig. 3.4)", "")

        st.subheader("Daño relativo (Fig. 3.5)")
        h_proy_cm = st.number_input("Espesor de losa de proyecto a verificar (cm)", 10.0, 45.0, P["h_proy_cm"], step=0.5)
        estacional = st.checkbox("Considerar variación estacional del MR (factores por periodo)", value=False)
        factores = [1.0]
        if estacional:
            df_fac = st.data_editor(pd.DataFrame({"Periodo": [f"P{i+1}" for i in range(12)], "Factor MR": [1.0] * 12}),
                                    num_rows="dynamic", hide_index=True, key=f"k_factores_{perfil_sel}")
            factores = [float(f) for f in df_fac["Factor MR"].dropna() if float(f) > 0] or [1.0]
        else:
            st.caption("Una sola estación representativa: keff = k (la Fig. 3.5 no modifica el valor).")

    # --- Cálculo de pasos A–C ---
    if mr_psi:
        _mrs = [mr_psi * f for f in factores]
    elif k_placa_pci and not placa_sobre_capa:
        _mrs = [19.4 * k_placa_pci * f for f in factores]
    else:
        _mrs = [None]
    res_k = determinar_k_aashto(estructura, _mrs, h_proy_cm * PULG_POR_CM, dsb_pulg_k, esb_psi_k,
                                k_placa_pci, estrato, dsg_pulg_k, k_inf_manual, k_rf_manual, placa_sobre_capa)

    st.divider()
    st.subheader("D. Pérdida de soporte LS (Tabla 2.7 y Fig. 3.6)")
    cl1, cl2 = st.columns(2)
    with cl1:
        _mat_def = mat_capa if mat_capa else MATERIAL_LS_POR_ESTRUCTURA[estructura]
        if estructura == ESTRUCTURAS[0]:
            _mat_def = _MAT[P["k_idx_mat_ls"]] if P["k_estructura"] == 0 else _mat_def
        mat_ls = st.selectbox("Material sobre el que apoya la losa (Tabla 2.7):", _MAT, index=_MAT.index(_mat_def),
                              key=f"mat_{_ke}_{mat_capa}")
        ls_min, ls_max = TABLA_LS[mat_ls]
        st.caption(f"Intervalo AASHTO para este material: LS = {ls_min:.1f} – {ls_max:.1f}")
        # [AJUSTE] Criterio guiado por erosionabilidad o manual
        _ops_ls = list(ORIENTACION_LS.keys()) + ["Manual (valor y justificación propios)"]
        modo_ls = st.selectbox("Criterio para fijar LS dentro del intervalo:", _ops_ls, index=P["k_idx_modo_ls"], key=f"modo_ls_{_ke}")
        if modo_ls.startswith("Manual"):
            ls_def = st.slider("LS de diseño (dentro del intervalo)", ls_min, ls_max,
                               float(min(max(P["k_ls"], ls_min), ls_max)), step=0.1, key=f"ls_{_ke}_{mat_ls}")
        else:
            _frac, _desc_ls = ORIENTACION_LS[modo_ls]
            ls_def = round(ls_min + _frac * (ls_max - ls_min), 1)
            st.info(f"LS = **{ls_def:.1f}**. Corresponde a: {_desc_ls}.")
        with st.expander("¿Cómo elegir LS? (orientación)"):
            st.markdown("AASHTO 93 asocia LS a la **erosión potencial del material de apoyo** y a los **movimientos verticales "
                        "diferenciales del suelo**; el intervalo depende del material (Tabla 2.7). Dentro del intervalo:\n\n"
                        + "\n".join(f"- **{k}**: {v[1]}." for k, v in ORIENTACION_LS.items())
                        + "\n\nLa ubicación dentro del intervalo es criterio del diseñador; la orientación del programa no sustituye "
                          "el juicio geotécnico.")
        if ls_def != int(ls_def):
            st.caption("LS no entero: interpolación lineal en k entre las líneas adyacentes de la Fig. 3.6, como en el ejemplo resuelto de [R3].")
        k_ls_manual, k_ls_man_origen = None, ""
        if st.checkbox("Ingresar k corregido leído directamente en la Fig. 3.6 (en lugar de la regresión)", value=False):
            k_ls_manual = st.number_input("k efectivo corregido por LS (pci)", 5.0, 3000.0, 100.0, step=1.0)
            k_ls_man_origen = st.text_input("Origen del valor (Fig. 3.6)", "")
    with cl2:
        if modo_ls.startswith("Manual"):
            ls_just = st.text_area("Justificación técnica del LS adoptado (obligatoria para la memoria)", P["ls_just"], height=130)
        else:
            _obs_ls = st.text_area("Observaciones adicionales (opcional)", P["ls_just"], height=110, key=f"obs_ls_{_ke}")
            ls_just = (f"Criterio: {modo_ls.split(' →')[0].lower()} ({ORIENTACION_LS[modo_ls][1]}). Intervalo Tabla 2.7 para "
                       f"{mat_ls.split(' (')[0].lower()}: {ls_min:.1f}–{ls_max:.1f} → LS = {ls_def:.1f}."
                       + (f" {_obs_ls.strip()}" if _obs_ls.strip() else ""))

    # --- Pendientes globales y k final (LS aplicado UNA vez) ---
    pendientes_k = list(res_k["pendientes"])
    if mr_fuera_rango:
        pendientes_k.append("MR obtenido por correlación fuera de su rango de validez (extrapolación).")
    if not cbr_repr:
        pendientes_k.append("Representatividad del CBR/MR en el nivel real de apoyo de la losa no verificada.")
    if not ls_just.strip():
        pendientes_k.append("LS sin justificación documentada.")
    condicionantes_k = (["ESB de referencia de la Tabla 2.7 (no medido)"] if esb_guia else [])
    if estructura != ESTRUCTURAS[0] and k_inf_manual is None and not (k_placa_pci and placa_sobre_capa) and not esb_origen.strip():
        pendientes_k.append("ESB manual sin origen documentado (ensayo o especificación).")
    if estructura == ESTRUCTURAS[3] and not nombre_capa.strip():
        pendientes_k.append("Capa de apoyo 'otra' sin descripción.")
    for _v, _o, _n in [(k_inf_manual, k_inf_man_origen, "k∞ (Fig. 3.3)"), (k_rf_manual, k_rf_man_origen, "k estrato (Fig. 3.4)"),
                       (k_ls_manual, k_ls_man_origen, "k corregido (Fig. 3.6)")]:
        if _v is not None and not _o.strip():
            pendientes_k.append(f"Valor manual de {_n} sin origen registrado.")
    keff_k = res_k["keff"]
    if keff_k is not None and k_ls_manual is None and ls_def > 0 and not (5 <= keff_k <= 2000):
        pendientes_k.append(f"keff = {keff_k:.0f} pci fuera del rango digitalizado de la Fig. 3.6 (5–2000 pci).")
    if k_ls_manual is not None:
        k_aashto = k_ls_manual
        k_ls_origen = "Manual: leído en la Fig. 3.6 — " + (k_ls_man_origen or "sin origen")
    else:
        k_aashto = k_corregido_ls(keff_k, ls_def) if keff_k else None
        k_ls_origen = "Calculado: regresión de la Fig. 3.6 [R3]" + (" con interpolación lineal" if ls_def != int(ls_def) else "")
    st.caption("🔒 La corrección por LS se aplica una única vez, sobre keff. El k de diseño resultante no vuelve a corregirse "
               "en ninguna otra pestaña.")

    if res_k["filas"]:
        with st.expander("Detalle por periodo (Fig. 3.5)", expanded=False):
            st.dataframe(pd.DataFrame(res_k["filas"]), hide_index=True, use_container_width=True)
    f0 = res_k["filas"][0] if res_k["filas"] else {}
    for a in res_k["advertencias"]:
        st.warning(a)
    if pendientes_k:
        st.error("**Condiciones pendientes de verificación (el resultado no es concluyente mientras existan):**\n\n- "
                 + "\n- ".join(pendientes_k))

with tab2:
    st.header("Configuración AASHTO '93")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🛡️ Confiabilidad y Desviación")
        tipo_infra = st.selectbox("Clasificación de la Vía / Infraestructura:",
                                ["Subestación Extra Alta Tensión (230-500 kV)",
                                 "Subestación Alta Tensión (66-115 kV)",
                                 "Autopistas y Vías Expresas", "Arterias Principales", "Colectoras", "Locales / Industriales"])

        map_conf = {
            "Subestación Extra Alta Tensión (230-500 kV)": 95, "Subestación Alta Tensión (66-115 kV)": 85,
            "Autopistas y Vías Expresas": 90, "Arterias Principales": 85, "Colectoras": 75, "Locales / Industriales": 55
        }

        conf = st.select_slider("Confiabilidad R (%)", [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99], value=map_conf.get(tipo_infra, 95))
        zr = norm.ppf(1 - (conf / 100))
        st.info(f"**Confiabilidad adoptada:** {conf}% (Zr = {zr:.3f})")

        s0_opt = st.selectbox("Guía para S0 (Desviación Estándar):",
                             ["Construcción Rígida Estándar (0.35)", "Mayor incertidumbre (0.39)", "Personalizado"])
        s0 = st.number_input("Valor S0", 0.30, 0.45, 0.35) if s0_opt == "Personalizado" else (0.35 if "0.35" in s0_opt else 0.39)

        p0 = st.slider("Serviciabilidad Inicial (P0)", 4.0, 5.0, 4.5)
        pt = st.slider("Serviciabilidad Final (Pt)", 2.0, 3.0, 2.5)

        # =========================================================
        # 🌡️ MÓDULO TÉRMICO Y CLIMÁTICO (EVALUACIÓN DE AMBIENTE Y SITIO)
        # =========================================================
        st.subheader("🌡️ Evaluación Térmica del Sitio")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            temp_max = st.number_input("Temp. Máxima Día (°C)", -10.0, 50.0, 32.0, step=1.0)
            t_col = st.number_input("Temperatura de colocación del concreto, Tcol (°C)", -10.0, 50.0, 20.0, step=1.0)
            cte_val = st.number_input("Coeficiente de expansión térmica CTE (×10⁻⁶/°C)", 5.0, 15.0, 10.0, step=0.1)
        with col_t2:
            temp_min = st.number_input("Temp. Mínima Noche (°C)", -20.0, 35.0, 12.0, step=1.0)
            gradiente_t = st.number_input("Gradiente térmico estimado ΔTg (°C)", 0.0, 30.0, 8.0, step=0.5)

        delta_t = temp_max - temp_min
        st.caption(f"Oscilación térmica ambiente: **ΔT = {delta_t:.1f} °C**")
        st.warning("⚠️ **Aviso:** Valor preliminar. Debe sustentarse con información climática, mediciones, bibliografía aplicable o criterio de ingeniería del proyecto.")

        st.markdown("**Restricciones de Borde / Confinamiento:**")
        restriccion_borde = st.selectbox(
            "¿Existen elementos rígidos que restrinjan el movimiento de la losa?",
            ["Ninguna", "Canaletas", "Bordillos", "Fundaciones", "Pedestales", "Estructuras / elementos rígidos", "Múltiple restricciones"]
        )

        # =========================================================
        st.subheader("🧱 Propiedades del Concreto")
        k_modo = st.radio("Definición del factor k (S'c = k * √f'c):",
                          ["Valores Recomendados (Memoria)", "Ingreso Manual"], horizontal=True)

        if k_modo == "Valores Recomendados (Memoria)":
            tipo_pav = st.selectbox("Tipo de Pavimento:",
                                     ["Autopistas/Carreteras (k=10.8)",
                                      "Zonas Industriales (k=10.1)",
                                      "Urbanos Secundarios (k=9.4)",
                                      "Subestaciones / Estándar (k=8.0)"], index=P["idx_pav"])

            map_k = {
                "Autopistas/Carreteras (k=10.8)": 10.8,
                "Zonas Industriales (k=10.1)": 10.1,
                "Urbanos Secundarios (k=9.4)": 9.4,
                "Subestaciones / Estándar (k=8.0)": 8.0
            }
            k_final = map_k[tipo_pav]
        else:
            k_final = st.number_input("Ingrese valor de k personalizado:", 7.0, 12.0, 8.0, step=0.1)

        # [CORRECCIÓN f'c] Selector en MPa (valores usuales de proyecto); 1 MPa = 145.0377 psi
        fc_mpa = st.selectbox("Resistencia f'c (MPa)", [21, 24, 28, 31, 35], index=P["idx_fc"])
        fc_psi = fc_mpa * 145.0377
        st.caption(f"f'c = {fc_mpa} MPa = {fc_psi:,.2f} psi")

        sc = k_final * np.sqrt(fc_psi)
        ec = 57000 * np.sqrt(fc_psi)

        st.success(f"**Módulo de Ruptura (S'c):** {sc:.2f} psi")
        st.info(f"**Módulo de Elasticidad (Ec):** {ec:,.0f} psi")

        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px;">
            <small><b>Fórmulas aplicadas:</b><br>
            S'c = {k_final} × √f'c (psi)<br>
            Ec = 57000 × √f'c (psi)</small>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.subheader("🌍 Soporte del Suelo (Sistema Multicapa)")
        st.markdown("#### 1. Subrasante Natural")

        metodo_k = st.radio(
            "Método para definir k natural:",
            ["Correlación AASHTO (CBR)", "Ensayo de Placa de Carga (Manual)"],
            horizontal=True
        )

        if metodo_k == "Correlación AASHTO (CBR)":
            # [MÓDULO k AASHTO 93] un solo CBR en toda la app: se toma de la pestaña "Módulo k"
            cbr = cbr_medido
            st.caption(f"CBR de la subrasante = **{cbr:.1f} %** (definido en la pestaña 🪨 Módulo k).")

            # [CORRECCIÓN k] La correlación entrega MPa/m; se convierte a pci solo como estimación
            k_corr_mpa = k_correlacion_cbr_mpa_m(cbr)
            k_natural = k_corr_mpa * FACTOR_MPA_M_A_PCI

            st.caption(f"k correlacionado (estimación preliminar): **{k_corr_mpa:.1f} MPa/m ≈ {k_natural:.1f} pci**")
            st.warning("⚠️ **Aviso Técnico:** La correlación CBR–k es una aproximación teórica. Se recomienda validar con **placa de carga**. Este valor NO se usa automáticamente como k efectivo de diseño (ver apartado 3).")

            with st.expander("📝 Ver justificación metodológica"):
                st.info("""
                **Unidades:** la correlación CBR–k (PCA / Manual MTC Perú) está expresada en **MPa/m**. Su resultado se convierte a pci (×3.684) solo como estimación de referencia.

                **Criterio de Diseño:** el k que entra en AASHTO '93 es el **k efectivo de diseño (pci)**, definido de forma independiente en el apartado 3. Con CBR altos la correlación produce k muy elevados para una subrasante natural; el valor efectivo debe sustentarse con ensayo de placa o criterio geotécnico.
                """)
        else:
            col_k1, col_k2 = st.columns(2)
            with col_k1:
                k_manual_mpa = st.number_input("k del Ensayo (MPa/m)", 10.0, 150.0, 40.0)
            with col_k2:
                k_natural = k_manual_mpa * 3.684
                st.metric("k Natural (pci)", f"{k_natural:.1f}")

            st.success("✅ Usando valor real de ensayo de placa.")

        st.divider()
        st.markdown("#### 2. Mejoramiento / Sub-base")
        st.caption("El uso de una base incrementa el valor de reacción (k combinado).")

        usar_base = st.checkbox("¿Incluir capa de Base/Mejoramiento?", value=P["usar_base"])

        if usar_base:
            c_b1, c_b2 = st.columns(2)
            with c_b1:
                tipo_base = st.selectbox("Material de Base:", ["Base Granular (Zahorra)", "Suelo Cemento / Estabilizada"])
            with c_b2:
                esp_base = st.number_input("Espesor Base (cm):", 10.0, 60.0, 15.0, step=5.0)

            # [CORRECCIÓN k] k combinado = ESTIMACIÓN de referencia (ya no es automáticamente el k de diseño)
            k_estimado = calcular_k_combinado(k_natural, esp_base, tipo_base)
            mejora_pct = ((k_estimado - k_natural) / k_natural) * 100

            st.metric("k Combinado Estimado (referencia)", f"{k_estimado:.1f} pci", delta=f"+{mejora_pct:.0f}% Incremento")

            if tipo_base == "Suelo Cemento / Estabilizada" and esp_base < 15:
                st.warning("⚠️ Recomendación: Para bases estabilizadas use espesores ≥ 15 cm.")

        else:
            k_estimado = k_natural
            st.metric("k Estimado sobre subrasante (referencia)", f"{k_estimado:.1f} pci")
            st.info("Diseño directo sobre subrasante natural (No recomendado para cargas altas).")

        # [AJUSTE FINAL] Solado de concreto pobre: se declara en la sección, SIN aporte estructural
        esp_solado = esp_solado_k  # [MÓDULO k AASHTO 93] un solo dato de solado (pestaña Módulo k)
        if esp_solado > 0:
            st.caption(f"Solado de {esp_solado:.0f} cm: **no se le asigna capacidad estructural** "
                       "(no incrementa k ni reduce el espesor de losa).")

        # =========================================================
        # [CORRECCIÓN k] 3. k EFECTIVO DE DISEÑO — dato independiente de la correlación
        # =========================================================
        st.divider()
        st.markdown("#### 3. Módulo de reacción EFECTIVO de diseño (k, pci)")
        st.caption("Es el único k que entra en la ecuación AASHTO 93, en geometría (21ℓ) y en las notas de plano.")
        origen_k = st.radio(
            "Origen del k efectivo de diseño:",
            ["Procedimiento AASHTO 1993 (pestaña 🪨 Módulo k)",
             "Adoptado manualmente (hipótesis de proyecto)",
             "Igual a la estimación anterior (correlación/ensayo + base)"],
            index=P["idx_origen_k"]
        )
        if origen_k.startswith("Procedimiento"):
            # [MÓDULO k AASHTO 93] k corregido por LS, calculado en la pestaña Módulo k
            k_val = k_aashto if k_aashto else float("nan")
            k_origen_txt = (f"AASHTO 1993: {estructura.upper()}, keff = {keff_k:.1f} pci, LS = {ls_def}"
                            if keff_k else "AASHTO 1993: PENDIENTE")
            k_origen_txt += " - CONDICIONES PENDIENTES DE VERIFICACIÓN GEOTÉCNICA" if pendientes_k else ""
            if k_aashto:
                st.info(f"k efectivo corregido (AASHTO 93, LS = {ls_def}): **{k_aashto:.1f} pci** "
                        f"({k_aashto / FACTOR_MPA_M_A_PCI:.1f} MPa/m).")
            else:
                st.error("El k AASHTO no pudo calcularse: complete los datos de la pestaña 🪨 Módulo k.")
            if pendientes_k:
                st.warning(f"⚠️ {len(pendientes_k)} condición(es) pendiente(s) en la determinación de k (ver pestaña 🪨 Módulo k).")
        elif origen_k.startswith("Adoptado"):
            k_val = st.number_input("k efectivo adoptado (pci)", 25.0, 2000.0, P["k_efectivo"], step=0.5,
                                    help="Hanaqpampa: 173.5 pci = HIPÓTESIS PROVISIONAL, requiere validación geotécnica.")
            k_origen_txt = "HIPÓTESIS PROVISIONAL, ADOPTADO PRELIMINARMENTE - REQUIERE VALIDACIÓN GEOTÉCNICA"
            st.warning("⚠️ **Hipótesis provisional: k efectivo adoptado preliminarmente, requiere validación geotécnica** "
                       "(placa de carga o criterio del especialista). No es resultado directo de la correlación CBR–k "
                       "ni un valor validado.")
        else:
            k_val = k_estimado
            k_origen_txt = ("ESTIMADO POR CORRELACIÓN CBR–k" if metodo_k == "Correlación AASHTO (CBR)"
                            else "ENSAYO DE PLACA") + (" + APORTE DE BASE" if usar_base else "")
            st.info("k efectivo = estimación anterior. Verifique que el valor sea representativo del apoyo real.")
        if np.isfinite(k_val):
            st.metric("k efectivo de diseño (AASHTO 93)", f"{k_val:.1f} pci",
                      delta=f"{k_val - k_estimado:+.1f} pci vs. estimación ({k_estimado:.1f} pci)", delta_color="off")
        st.subheader("🔗 Transferencia de Carga (J)")

        j_manual = st.toggle("Ingresar J manualmente", False)

        if j_manual:
            j_val = st.number_input("Valor J personalizado", 2.0, 5.0, 3.2, step=0.1)
            st.info(f"Valor J manual activo: **{j_val}**")
            tiene_dovelas = "No"
            tiene_soporte = "No"
        else:
            escenarios_j = {
                "Escenario 1: Con Dovelas y con Bermas/Bordillo (J: 2.7)": {
                    "valor": 2.7,
                    "sustento": "Es el escenario ideal. Las dovelas (pasajuntas) transfieren hasta el 50% de la carga. Las bermas o bordillos integrales evitan que el neumático circule por el borde libre, reduciendo el agrietamiento.",
                    "nota_bordillo": True, "dovelas": "Sí", "soporte": "Sí"
                },
                "Escenario 2: Con Dovelas y Sin Bermas/Bordillo (J: 3.2)": {
                    "valor": 3.2,
                    "sustento": "Valor estándar AASHTO '93. Excelente transferencia de carga por dovelas, pero sin soporte lateral. Las cargas en el borde generan mayores tensiones.",
                    "nota_bordillo": False, "dovelas": "Sí", "soporte": "No"
                },
                "Escenario 3: Sin Dovelas pero Con Bordillo/Berma (J: 3.8)": {
                    "valor": 3.8,
                    "sustento": "Transferencia por trabazón de agregados (interlock). El bordillo/berma ayuda a que el esfuerzo en el borde no sea crítico.",
                    "nota_bordillo": True, "dovelas": "No", "soporte": "Sí"
                },
                "Escenario 4: Sin Dovelas y Sin Bermas (J: 4.2)": {
                    "valor": 4.2,
                    "sustento": "Caso crítico. Sin dovelas la junta es propensa al escalonamiento y sin bermas el camión circula por el borde libre sin apoyo lateral.",
                    "nota_bordillo": False, "dovelas": "No", "soporte": "No"
                }
            }

            seleccion = st.radio("Seleccione escenario:", list(escenarios_j.keys()), index=P["idx_j"])
            datos_esc = escenarios_j[seleccion]
            j_val = datos_esc["valor"]
            tiene_dovelas = datos_esc["dovelas"]
            tiene_soporte = datos_esc["soporte"]
            st.write(f"**Sustento:** {datos_esc['sustento']}")

            if datos_esc["nota_bordillo"]:
                st.warning("⚠️ **Nota sobre el Bordillo:** Para ser efectivo, debe ser integral (vaciado monolítico) o anclado con barras de amarre. Si es sobrepuesto, no aporta soporte lateral.")

            st.info(f"Valor J: **{j_val}**")

        st.subheader("💧 Coeficiente de Drenaje (Cd)")
        st.markdown("Responda las siguientes preguntas para estimar el valor Cd:")

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("**1. Velocidad de Drenaje:**")
            st.caption("Si el pavimento se inunda, ¿qué tan rápido desaparece el agua?")
            calidad_input = st.selectbox("Seleccione velocidad:", [
                "Excelente (2 horas)", "Bueno (1 día)", "Regular (1 semana)", "Pobre (1 mes)", "Muy Pobre (Nunca)"
            ])
            mapa_calidad = {
                "Excelente (2 horas)": 0, "Bueno (1 día)": 1, "Regular (1 semana)": 2, "Pobre (1 mes)": 3, "Muy Pobre (Nunca)": 4
            }
            idx_calidad = mapa_calidad[calidad_input]

        with col_d2:
            st.markdown("**2. Clima y Saturación:**")
            st.caption("¿Con qué frecuencia llueve o el suelo está saturado de agua?")
            exposicion_input = st.selectbox("Seleccione condición climática:", [
                "Seco / Árido (< 1% del tiempo)", "Moderado / Estándar (1% - 5% del tiempo)",
                "Húmedo / Lluvioso (5% - 25% del tiempo)", "Saturación Constante (> 25% del tiempo)"
            ])
            mapa_exposicion = {
                "Seco / Árido (< 1% del tiempo)": 0, "Moderado / Estándar (1% - 5% del tiempo)": 1,
                "Húmedo / Lluvioso (5% - 25% del tiempo)": 2, "Saturación Constante (> 25% del tiempo)": 3
            }
            idx_exposicion = mapa_exposicion[exposicion_input]

        matriz_cd = [
            [1.25, 1.20, 1.15, 1.10],
            [1.15, 1.10, 1.00, 1.00],
            [1.05, 1.00, 0.95, 0.80],
            [0.95, 0.90, 0.80, 0.70],
            [0.80, 0.75, 0.65, 0.55]
        ]

        cd_recomendado = matriz_cd[idx_calidad][idx_exposicion]

        st.write("---")
        cd_val = st.number_input(f"Valor Cd Recomendado: {cd_recomendado}", 0.50, 1.40, cd_recomendado, step=0.01)

        if cd_val < 1.0:
            st.error(f"📉 Drenaje Lento (Cd={cd_val}). El agua atrapada debilitará la base, requiriendo mayor espesor de concreto.")
        elif cd_val > 1.0:
            st.success(f"📈 Buen Drenaje (Cd={cd_val}). El agua fluye rápido, permitiendo optimizar el espesor.")
        else:
            st.info("Drenaje Estándar (Cd = 1.0).")

    st.divider()
    if st.button("🚀 CALCULAR ESTRUCTURA"):
        # [CORRECCIÓN solver] Resultado detallado: distingue raíz, ausencia de raíz, rango y convergencia
        res_aashto = resolver_espesor_aashto(w18_total, zr, s0, p0, pt, sc, cd_val, j_val, ec, k_val)
        esp_pulg = res_aashto["D_pulg"] if res_aashto["estado"] == "ok" else None

        if not esp_pulg:
            # No se conservan resultados anteriores: evita que geometría/ábaco usen un cálculo obsoleto
            for _clave in ('esp_final_cm', 'esp_pulg_base', 'w18_res', 'k_res'):
                st.session_state.pop(_clave, None)
            titulos = {
                "sin_raiz": "ℹ️ La ecuación AASHTO 93 NO tiene raíz con estos parámetros",
                "fuera_de_rango": "🚨 Espesor requerido fuera del rango de búsqueda",
                "parametros_invalidos": "🚨 Parámetros fuera de rango",
                "no_convergencia": "🚨 Fallo de convergencia numérica",
            }
            st.error(f"**{titulos.get(res_aashto['estado'], 'Sin resultado')}** (k efectivo = {k_val:.1f} pci)")
            if res_aashto["estado"] == "sin_raiz":
                # [AJUSTE FINAL] Sin raíz ≠ "cualquier espesor es suficiente"
                st.write(f"Diagnóstico numérico: el residuo de la ecuación no se anula en el dominio válido "
                         f"(mínimo g = {res_aashto['g_min']:+.4f} en D = {res_aashto['D_g_min']*2.54:.2f} cm; "
                         f"dominio D > {res_aashto['D_singular']*2.54:.2f} cm).")
                st.warning("⚠️ **Esto NO demuestra que cualquier espesor sea suficiente.** Indica que la combinación de "
                           "parámetros (en especial el **k efectivo**) queda fuera del rango en que la ecuación empírica "
                           "AASHTO 93 es representativa, por lo que el método no permite dimensionar la losa. "
                           "No se asigna espesor: revise k y los demás datos, o justifique el espesor con otro criterio "
                           "documentado por el diseñador.")
            else:
                st.write(res_aashto["mensaje"])

        if esp_pulg:
            esp_exacto_cm = esp_pulg * 2.54
            esp_comercial_cm = np.ceil(esp_exacto_cm)
            esp_final_cm = max(esp_comercial_cm, 15.0)

            st.session_state['esp_final_cm'] = esp_final_cm
            st.session_state['esp_pulg_base'] = esp_pulg
            st.session_state['ec_res'] = ec
            st.session_state['k_res'] = k_val            # k EFECTIVO de diseño (pci)
            st.session_state['k_origen'] = k_origen_txt  # [CORRECCIÓN k] trazabilidad del origen de k
            st.session_state['k_estimado_ref'] = k_estimado
            st.session_state['w18_res'] = w18_total
            st.session_state['conf_res'] = conf
            st.session_state['tiene_dovelas'] = tiene_dovelas
            st.session_state['tiene_soporte'] = tiene_soporte
            st.session_state['j_res'] = j_val
            st.session_state['cd_res'] = cd_val
            st.session_state['esp_solado'] = esp_solado
            st.session_state['fc_mpa_val'] = fc_mpa
            st.session_state['fc_psi_val'] = fc_psi
            st.session_state['sc_psi_val'] = sc
            st.session_state['cbr_input'] = cbr if metodo_k == "Correlación AASHTO (CBR)" else "Suelo Natural"

            # --- DATOS TÉRMICOS NUEVOS EN SESSION STATE ---
            st.session_state['temp_max'] = temp_max
            st.session_state['temp_min'] = temp_min
            st.session_state['delta_t'] = delta_t
            st.session_state['t_col'] = t_col
            st.session_state['cte_val'] = cte_val
            st.session_state['gradiente_t'] = gradiente_t
            st.session_state['restriccion_borde'] = restriccion_borde
            st.session_state['periodo_diseno'] = periodo

            st.session_state['usar_base'] = usar_base
            if usar_base:
                st.session_state['tipo_base_guardado'] = tipo_base
                st.session_state['esp_base_guardado'] = esp_base
            else:
                st.session_state['tipo_base_guardado'] = ""
                st.session_state['esp_base_guardado'] = 0

            st.toast("✅ Cálculo realizado y datos guardados correctamente.")
            st.success(f"### Espesor de Losa Recomendado: {esp_final_cm:.1f} cm")
            st.info(f"*(Valor exacto AASHTO: {esp_exacto_cm:.2f} cm | k efectivo de diseño: {k_val:.1f} pci — {k_origen_txt.lower()})*")
            st.caption(f"Solucionador: {res_aashto['mensaje']} Dominio válido D > {res_aashto['D_singular']*2.54:.2f} cm."
                       + (f" Raíz espuria descartada (rama no física): {res_aashto['D_espuria']*2.54:.2f} cm." if res_aashto['D_espuria'] else ""))
            for _adv in res_aashto["advertencias"]:
                st.warning(_adv)

        # [MÓDULO k AASHTO 93] Verificación frente al espesor de proyecto
        _pend = pendientes_k if origen_k.startswith("Procedimiento") else ["k adoptado sin procedimiento AASHTO sustentado"] if origen_k.startswith("Adoptado") else []
        _cond = (condicionantes_k if origen_k.startswith("Procedimiento")
                 else ["k estimado por correlación CBR–k / combinación de base (no procedimiento AASHTO)"] if origen_k.startswith("Igual") else [])
        verif_estado, verif_txt = verificar_espesor(res_aashto, h_proy_cm, _pend, _cond)
        _fn = {"CUMPLE": st.success, "NO CUMPLE": st.error}.get(verif_estado, st.warning)
        _fn(f"**Verificación del espesor de proyecto h = {h_proy_cm:.1f} cm: {verif_estado}** — {verif_txt}")
        if esp_pulg:
            st.session_state['verif_estado'] = verif_estado
            st.session_state['verif_txt'] = verif_txt
            st.session_state['h_proy_cm'] = h_proy_cm
            st.session_state['estructura_k'] = estructura if origen_k.startswith("Procedimiento") else None
            st.session_state['soporte_k'] = ("SUBRASANTE PREPARADA (SIN SUBBASE ESTRUCTURAL)" if estructura == ESTRUCTURAS[0] else
                                             f"{(nombre_capa or estructura.replace('Losa sobre ', '')).upper()} DE {dsb_pulg_k * 2.54:.0f} cm "
                                             f"(ESB = {esb_psi_k:,.0f} psi) SOBRE SUBRASANTE" if dsb_pulg_k else
                                             f"{(nombre_capa or estructura.replace('Losa sobre ', '')).upper()} SOBRE SUBRASANTE")
            st.session_state['docs_proyecto'] = (P["doc_geo"], P["plano"])

# =========================================================
# [MÓDULO k AASHTO 93] Resultados, escenarios LS, nomograma y memoria (usan los parámetros de 'Parámetros de Diseño')
# =========================================================
with tab_k:
    st.divider()
    st.subheader("E. Resultados trazables")
    st.caption(f"Parámetros tomados de las pestañas Tránsito y Parámetros: W18 = {w18_total:,.0f}; R = {conf} % (Zr = {zr:.3f}); "
               f"S0 = {s0}; P0 = {p0}; Pt = {pt}; f'c = {fc_mpa} MPa; S'c = {sc:.2f} psi; Ec = {ec:,.0f} psi; J = {j_val}; Cd = {cd_val}.")
    _con_capa = estructura != ESTRUCTURAS[0]
    _res_dis = resolver_espesor_aashto(w18_total, zr, s0, p0, pt, sc, cd_val, j_val, ec, k_aashto) if k_aashto else None
    _ve_dis, _vt_dis = verificar_espesor(_res_dis, h_proy_cm, pendientes_k, condicionantes_k) if _res_dis else ("NO CONCLUYENTE", "k no determinado")
    _D_dis = _res_dis["D_pulg"] * 2.54 if (_res_dis and _res_dis["estado"] == "ok") else None
    _pend_tag = "PENDIENTE DE VALIDACIÓN"
    _fila = lambda c, v, u, o, r="": {"Concepto": c, "Valor": v, "Unidad": u, "Origen": o, "Referencia": r}
    resumen = [
        _fila("Subrasante: clasificación", clas_suelo or "—", "", "Ingresado" if clas_suelo else _pend_tag),
        _fila("Subrasante: CBR", f"{cbr_medido:.1f}", "%", "Ingresado" + ("" if cbr_repr else f" — {_pend_tag} (representatividad)"), cbr_cond),
        _fila("Subrasante: MR", f"{f0['MR (psi)']:,.0f} psi = {f0['MR (MPa)']:.1f} MPa" if f0.get("MR (psi)") else "—", "",
              (mr_origen + (f" — {_pend_tag} (extrapolación)" if mr_fuera_rango else "")) if f0.get("MR (psi)") else "No aplica", "[R1]/[R4]"),
        _fila("Capa de apoyo", (nombre_capa or estructura.replace("Losa sobre ", "")) if _con_capa else "Ninguna", "", "Ingresado"),
        _fila("Capa de apoyo: DSB / ESB", f"{dsb_pulg_k * 2.54:.1f} cm / {esb_psi_k:,.0f} psi" if _con_capa and dsb_pulg_k else "No aplica", "",
              ((esb_origen + " (referencia, no medido)") if esb_guia else ("Ingresado — " + (esb_origen or _pend_tag))) if _con_capa else "No aplica", "Tabla 2.7"),
        _fila("k inicial de la subrasante", f"{f0['k inicial (pci)']:.1f}" if f0.get("k inicial (pci)") else "—", "pci", res_k["origen"]["k_ini"], "[R1] k = MR/19.4"),
        _fila("k compuesto k∞", f"{f0['k∞ (pci)']:.1f}" if f0 else "—", "pci",
              res_k["origen"]["k_inf"] + (f" — {k_inf_man_origen}" if k_inf_manual is not None else ""), "Fig. 3.3 / [R2] Ec. 5.22"),
        _fila("Corrección por estrato rígido", f"{f0['k estrato (pci)']:.1f}" if f0 and res_k["origen"]["k_rf"] not in ("No aplica", "PENDIENTE") else res_k["origen"]["k_rf"],
              "pci", res_k["origen"]["k_rf"] + (f" — {k_rf_man_origen}" if k_rf_manual is not None else ""), "Fig. 3.4"),
        _fila("k efectivo antes de LS (Fig. 3.5)", f"{keff_k:.1f}" if keff_k else "—", "pci",
              f"Calculado: {len(factores)} periodo(s)" if keff_k else _pend_tag, "[R2] Ec. 5.24"),
        _fila("LS", f"{ls_def:.1f}" if k_ls_manual is None else "incluido en valor manual", "",
              f"Seleccionado en {ls_min:.0f}–{ls_max:.0f} ({mat_ls.split(' (')[0]})" + ("" if ls_just.strip() else f" — {_pend_tag} (sin justificación)"), "Tabla 2.7"),
        _fila("k efectivo de diseño", f"{k_aashto:.1f} pci = {k_aashto / FACTOR_MPA_M_A_PCI:.1f} MPa/m" if k_aashto else "—", "",
              k_ls_origen if k_aashto else _pend_tag, "Fig. 3.6 / [R3]"),
        _fila("Espesor teórico AASHTO", f"{_D_dis:.2f}" if _D_dis else "Sin raíz / no calculado", "cm", "Calculado" if _D_dis else _pend_tag, "Ec. AASHTO 93"),
        _fila("Espesor adoptado (redondeo hacia arriba, mín. 15 cm)", f"{max(np.ceil(_D_dis), 15.0):.0f}" if _D_dis else "No definido", "cm", "Calculado" if _D_dis else "—"),
        _fila(f"Verificación h de proyecto = {h_proy_cm:.1f} cm", _ve_dis, "", _vt_dis),
    ]
    st.dataframe(pd.DataFrame(resumen), hide_index=True, use_container_width=True)
    st.caption("Origen: **Calculado** = ecuación documentada; **Ingresado** = dato del usuario; **Manual** = lectura de nomograma o "
               "fuente externa registrada; **PENDIENTE DE VALIDACIÓN** = dato faltante o fuera de aplicabilidad.")

    st.subheader("F. Escenarios de pérdida de soporte (sensibilidad)")
    filas_esc = []
    if keff_k:
        _cond_k = condicionantes_k
        _ls_list = sorted(set([0.0, 1.0, 2.0, 3.0, float(ls_def)]))
        for _ls in _ls_list:
            _nom = {0.0: "Sin pérdida de soporte", 1.0: "Pérdida moderada", 2.0: "Pérdida mayor", 3.0: "Pérdida severa"}.get(_ls, "LS intermedio")
            _k = k_corregido_ls(keff_k, _ls)
            _pend_ls = [p for p in pendientes_k if not p.startswith("keff =") and not p.startswith("LS sin")]
            if _ls > 0 and not (5 <= keff_k <= 2000):
                _pend_ls.append("keff fuera del rango de la Fig. 3.6")
            if not (ls_min <= _ls <= ls_max):
                _pend_ls.append("LS fuera del intervalo de la Tabla 2.7 para el material")
            _rs = resolver_espesor_aashto(w18_total, zr, s0, p0, pt, sc, cd_val, j_val, ec, _k)
            _ve, _vt = verificar_espesor(_rs, h_proy_cm, _pend_ls, _cond_k)
            filas_esc.append({"Escenario": _nom + (" ◀ adoptado" if (_ls == ls_def and k_ls_manual is None) else ""),
                              "k∞ (pci)": round(f0.get("k∞ (pci)", 0), 1),
                              "keff antes LS (pci)": round(keff_k, 1), "LS": _ls,
                              "En intervalo Tabla 2.7": "Sí" if ls_min <= _ls <= ls_max else "No",
                              "k corregido (pci)": round(_k, 1), "k corregido (MPa/m)": round(_k / FACTOR_MPA_M_A_PCI, 1),
                              "Espesor teórico (cm)": f"{_rs['D_pulg']*2.54:.2f}" if _rs["estado"] == "ok" else "Sin raíz",
                              f"Verificación {h_proy_cm:.0f} cm": _ve, "Detalle": _vt})
        df_esc = pd.DataFrame(filas_esc)
        st.dataframe(df_esc, hide_index=True, use_container_width=True)
        try:
            _dk = lambda kk: resolver_espesor_aashto(w18_total, zr, s0, p0, pt, sc, cd_val, j_val, ec, kk)["D_pulg"]
            _k_hi = max(kk for kk in np.geomspace(10, 2000, 80) if _dk(kk))  # mayor k con raíz válida
            _kmin = brentq(lambda kk: _dk(kk) * 2.54 - h_proy_cm, 10, _k_hi)
            st.info(f"📌 Para que el espesor requerido sea ≤ {h_proy_cm:.1f} cm se necesita **k de diseño ≥ {_kmin:.1f} pci** "
                    f"({_kmin / FACTOR_MPA_M_A_PCI:.1f} MPa/m), con los demás parámetros fijos.")
        except Exception:
            _kmin = None
        st.caption("Los escenarios son de **sensibilidad**; el LS de diseño lo fija el diseñador con justificación. "
                   "Un escenario sin raíz se reporta NO CONCLUYENTE: no demuestra que cualquier espesor sea suficiente.")

        # --- Nomograma Fig. 3.6 reproducido ---
        import altair as alt
        _kk = np.geomspace(5, 2000, 60)
        _lin = pd.DataFrame([{"keff (pci)": x, "k corregido (pci)": k_corregido_ls(x, l), "Curva": f"LS = {l}"} for l in range(4) for x in _kk])
        _pts = pd.DataFrame([{"keff (pci)": a, "k corregido (pci)": b, "Curva": f"LS = {l}"} for l, v in NOMOGRAMA_LS_PUNTOS.items() for a, b in v])
        _usr = pd.DataFrame([{"keff (pci)": keff_k, "k corregido (pci)": k_corregido_ls(keff_k, l), "Curva": f"LS = {l}"} for l in range(4)])
        _ex = alt.Scale(type="log")
        _ch = (alt.Chart(_lin).mark_line().encode(x=alt.X("keff (pci):Q", scale=_ex), y=alt.Y("k corregido (pci):Q", scale=_ex), color="Curva:N")
               + alt.Chart(_pts).mark_point(shape="cross", size=60).encode(x="keff (pci):Q", y="k corregido (pci):Q", color="Curva:N")
               + alt.Chart(_usr).mark_point(filled=True, size=120, shape="diamond").encode(x="keff (pci):Q", y="k corregido (pci):Q", color="Curva:N",
                                                                                           tooltip=["Curva", "keff (pci)", "k corregido (pci)"])
               ).properties(height=380, title="Fig. 3.6 AASHTO 93 – corrección por LS (líneas: regresión [R3]; cruces: puntos digitalizados [R3]; rombos: este proyecto)")
        st.altair_chart(_ch, use_container_width=True)
        if keff_k > 2000:
            st.warning("El keff de este proyecto está fuera del rango graficado del nomograma (5–2000 pci).")
        st.caption("Las Figs. 3.3 y 3.4 se resuelven con sus ecuaciones de regresión (AASHTO 1986, Apéndice LL) o con el valor "
                   "leído e ingresado manualmente; no se grafican.")

        # --- Memoria de cálculo ---
        _docs = f"Estudio geotécnico: {P['doc_geo']}. Plano de pavimentos: {P['plano']}." if P["doc_geo"] else ""
        _tabla_res = "| Concepto | Valor | Unidad | Origen |\n|---|---|---|---|\n" + \
            "\n".join(f"| {r['Concepto']} | {r['Valor']} | {r['Unidad']} | {r['Origen']} |" for r in resumen)
        _tabla_md = "| Escenario | keff (pci) | LS | k corregido (pci) | Espesor teórico (cm) | Verificación |\n|---|---|---|---|---|---|\n" + \
            "\n".join(f"| {r['Escenario']} | {r['keff antes LS (pci)']} | {r['LS']} | {r['k corregido (pci)']} | {r['Espesor teórico (cm)']} | {r[f'Verificación {h_proy_cm:.0f} cm']} |" for r in filas_esc)
        memoria = f"""## Determinación del módulo de reacción efectivo y verificación del espesor de losa (AASHTO 1993)

**Datos.** {_docs} Subrasante: {clas_suelo or 'sin clasificar'}; CBR = {cbr_medido:.1f} % ({cbr_cond or 'condición no indicada'}).
Apoyo de la losa: {estructura.lower()}{f'; solado de concreto pobre de {esp_solado_k:.0f} cm (f′c = {fc_solado:.0f} MPa), sin capacidad estructural asignada' if esp_solado_k > 0 else ''}.
Diseño: W18 = {w18_total:,.0f} ESAL; R = {conf} % (Zr = {zr:.3f}); S0 = {s0}; P0 = {p0}; Pt = {pt}; f′c = {fc_mpa} MPa;
S′c = {sc:.2f} psi; Ec = {ec:,.0f} psi; J = {j_val}; Cd = {cd_val}; espesor de proyecto h = {h_proy_cm:.1f} cm.

**Metodología (Guía AASHTO 1993, Parte II, sec. 3.2).**
1. k inicial de la subrasante: k = MR/19,4 [R1] (o ensayo de placa). MR: {mr_origen}{' — EXTRAPOLACIÓN fuera del rango de la correlación' if mr_fuera_rango else ''}.
2. k compuesto: {res_k['origen']['k_inf']}. Ec. 5.22 [R2]: ln k∞ = −2,807 + 0,1253(ln DSB)² + 1,062 ln MR + 0,1282 ln DSB·ln ESB − 0,4114 ln DSB − 0,0581 ln ESB − 0,1317 ln DSB·ln MR.
3. Estrato rígido (Fig. 3.4): {estrato.lower()}; {res_k['origen']['k_rf']}.
4. Daño relativo (Fig. 3.5): ur = (D^0,75 − 0,39·k^0,25)^3,42 con D = {h_proy_cm * PULG_POR_CM:.2f} pulg; {len(factores)} periodo(s) → keff = {keff_k:.1f} pci.
5. Pérdida de soporte (Tabla 2.7, Fig. 3.6): material «{mat_ls}», intervalo LS {ls_min:.1f}–{ls_max:.1f}; LS de diseño = {ls_def:.1f}. {ls_just}
   {k_ls_origen}. LS=1: k = 10^(0,8311·log keff − 0,0643); LS=2: k = 10^(0,6408·log keff − 0,0834); LS=3: k = 10^(0,538·log keff − 0,1806) [R3]. La corrección se aplica una sola vez.

**Resumen trazable.**

{_tabla_res}

**Sensibilidad a LS.**

{_tabla_md}

{f'k de diseño mínimo para h ≤ {h_proy_cm:.1f} cm: {_kmin:.1f} pci.' if _kmin else ''}

**Condiciones pendientes.** {'; '.join(pendientes_k) if pendientes_k else 'Ninguna registrada.'}

**Referencias.**
[R1] AASHTO (1993). *Guide for Design of Pavement Structures*. Parte I sec. 1.5; Parte II cap. 3; Tabla 2.7; Figs. 3.3–3.6.
[R2] FHWA (2006). *Geotechnical Aspects of Pavements*, FHWA NHI-05-037, sec. 5.4.6, Ecs. 5.20–5.24, Tabla 5-45. https://www.fhwa.dot.gov/engineering/geotech/pubs/05037/05c.cfm
[R3] García Aladín, M.F. et al. (2024). Desarrollo de ecuaciones sustitutivas… *Revista Eduweb* 18(1), 9-23. https://doi.org/10.46502/issn.1856-7576/2024.18.01.1
[R4] Powell, W.D. et al. (1984). *The structural design of bituminous roads*. TRRL LR 1132; NCHRP 1-37A (2004).
"""
        with st.expander("📄 Texto para la memoria de cálculo", expanded=False):
            st.markdown(memoria)
            st.download_button("⬇️ Descargar memoria (.md)", memoria, file_name="memoria_k_efectivo_AASHTO93.md", mime="text/markdown")
    else:
        st.error("No es posible evaluar escenarios: el k efectivo no está determinado (ver pendientes).")

with tab3:
    st.header("📐 Recomendaciones Geométricas y Modulación")

    if 'esp_final_cm' not in st.session_state:
        st.info("⚠️ Realice el cálculo en la pestaña 'Parámetros de Diseño' para habilitar esta sección.")
    else:
        st.warning("⚠️ **Tránsito Excéntrico:** En subestaciones, el tránsito suele circular cerca del borde. Como recomendación genérica (no detalle aprobado) puede evaluarse borde engrosado +25% en perímetros; aplicar solo si coincide con los planos del proyecto.")

        esp_cm_actual = st.session_state['esp_final_cm']

        # 1. Recuperación dinámica de oscilación térmica para recalcular límites
        t_max_v = st.session_state.get('temp_max', 32.0)
        t_min_v = st.session_state.get('temp_min', 12.0)
        dt_sitio = t_max_v - t_min_v

        # 2. Evaluación estricta del factor L/h según oscilación
        if dt_sitio > 20.0:
            factor_lh = 20.0
            msj_clima = f"Alta Oscilación (ΔT={dt_sitio:.1f}°C) → Límite L/h ≤ 20"
        elif dt_sitio > 12.0:
            factor_lh = 22.0
            msj_clima = f"Oscilación Moderada (ΔT={dt_sitio:.1f}°C) → Límite L/h ≤ 22"
        else:
            factor_lh = 24.0
            msj_clima = f"Oscilación Baja (ΔT={dt_sitio:.1f}°C) → Límite L/h ≤ 24"

        # Límite térmico en metros exactos
        limit_termico = (factor_lh * esp_cm_actual) / 100.0

        # 3. Radio de rigidez relativa (21ℓ en metros)
        esp_pulg_calculo = st.session_state.get('esp_pulg_base', esp_cm_actual / 2.54)
        nu = 0.15
        l_pulg = ((st.session_state.get('ec_res', 3500000) * (esp_pulg_calculo**3)) / (12 * (1 - nu**2) * st.session_state.get('k_res', 215)))**0.25
        limit_rigidez = (21 * l_pulg) * 0.0254

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            ancho_carril = st.number_input("Ancho total de carril (m)", 2.5, 7.0, 4.5, step=0.1)
            num_juntas_long = 1 if ancho_carril > 4.5 else 0
            ancho_losa = ancho_carril / (num_juntas_long + 1)
            st.metric("Ancho de Losa Efectivo (B)", f"{ancho_losa:.2f} m")

        with col_g2:
            # Límite por aspecto L/B <= 1.25
            limit_aspecto = ancho_losa * 1.25

            # El largo sugerido parte del más restrictivo
            largo_sug_teorico = min(limit_aspecto, limit_rigidez, limit_termico)

            largo_sug = st.number_input(
                f"Largo de Losa Adoptado - L (m)",
                min_value=1.0,
                max_value=10.0,
                value=float(round(largo_sug_teorico, 2)),
                step=0.10,
                help="Ajuste manualmente la longitud de la losa si desea probar otros paños."
            )
            st.caption(f"📌 {msj_clima} (L_máx térmico = {limit_termico:.2f} m | L_máx aspecto = {limit_aspecto:.2f} m)")

        st.divider()
        st.subheader("🔍 Verificaciones Técnicas y Alertas para el Diseñador")
        c_v1, c_v2, c_v3 = st.columns(3)

        with c_v1:
            relacion_lb = largo_sug / ancho_losa
            st.write(f"**1. Relación de Aspecto (L/B):** {relacion_lb:.2f}")
            if relacion_lb <= 1.25: st.success("✅ Relación ideal (≤ 1.25)")
            elif relacion_lb <= 1.5: st.warning("⚠️ Relación aceptable (1.25 - 1.50)")
            else: st.error("🚨 Relación crítica (> 1.50). Reducir largo L.")

        with c_v2:
            relacion_lh = (largo_sug * 100) / esp_cm_actual
            st.markdown("**2. 🔥 Control preliminar de modulación térmica:**")
            st.caption("Criterio interno de predimensionamiento térmico. No constituye una verificación mecanicista de curling/warping.")
            st.write(f"Relación Esbeltez (L/h): **{relacion_lh:.1f}**")
            if relacion_lh <= factor_lh:
                st.success(f"✅ Cumple relación L/h (≤ {factor_lh:.0f})")
            else:
                st.error(f"🚨 **ALERTA DE ALABEO:** L/h ({relacion_lh:.1f}) excede el criterio preliminar ({factor_lh:.0f}). **Considere reducir L a ≤ {limit_termico:.2f} m**.")

        with c_v3:
            st.write(f"**3. Rigidez Relativa (21ℓ):** {limit_rigidez:.2f} m")
            if largo_sug <= limit_rigidez: st.success("✅ Cumple límite de rigidez")
            else: st.error("🚨 Excede límite de rigidez.")

        # =========================================================
        # 🌡️ EVALUACIÓN DE RESPUESTA TÉRMICA INDICATIVA
        # =========================================================
        st.divider()
        st.subheader("🌡️ Respuesta Térmica e Indicadores de Junta")

        t_col_val = st.session_state.get('t_col', 20.0)
        cte_val_inp = st.session_state.get('cte_val', 10.0)
        grad_t_inp = st.session_state.get('gradiente_t', 8.0)
        restriccion_val = st.session_state.get('restriccion_borde', "Ninguna")

        alpha = cte_val_inp * 1e-6

        delta_t_exp = t_max_v - t_col_val
        delta_t_con = t_col_val - t_min_v

        epsilon_exp = alpha * max(delta_t_exp, 0.0)
        epsilon_con = alpha * max(delta_t_con, 0.0)

        movimiento_exp_mm = alpha * max(delta_t_exp, 0.0) * largo_sug * 1000.0
        movimiento_con_mm = alpha * max(delta_t_con, 0.0) * largo_sug * 1000.0

        h_m = esp_cm_actual / 100.0
        curvatura_t = (alpha * grad_t_inp) / h_m

        col_res_t1, col_res_t2 = st.columns(2)
        with col_res_t1:
            st.markdown("##### 📏 Deformación y Movimiento Térmico Libre Teórico")
            st.write(f"• **Oscilación térmica ambiente (ΔT):** {dt_sitio:.1f} °C (Máx: {t_max_v:.1f} °C, Mín: {t_min_v:.1f} °C)")
            st.write(f"• **ΔT Expansión:** {max(delta_t_exp, 0.0):.1f} °C | **ΔT Contracción:** {max(delta_t_con, 0.0):.1f} °C")
            st.write(f"• **Deformación Térmica Libre (Exp/Con):** {epsilon_exp*1e6:.0f} με / {epsilon_con*1e6:.0f} με")
            st.write(f"• **Movimiento Térmico Libre Teórico (Expansión):** {movimiento_exp_mm:.2f} mm")
            st.write(f"• **Movimiento Térmico Libre Teórico (Contracción):** {movimiento_con_mm:.2f} mm")
            st.caption("📌 *El valor corresponde al movimiento libre teórico. La apertura/cierre real de la junta depende de las restricciones, fricción, temperatura de colocación, configuración de juntas y condiciones de borde.*")

        with col_res_t2:
            st.markdown("##### 🌀 Gradiente y Curvatura Térmica Libre Idealizada")
            st.write(f"• **Gradiente Térmico Estimado (ΔTg):** {grad_t_inp:.1f} °C")
            st.write(f"• **Curvatura Térmica Libre Idealizada:** {curvatura_t:.6f} 1/m")
            st.caption("📌 *Indicador teórico de curling/warping. No representa por sí solo la curvatura real ni las tensiones desarrolladas en la losa.*")

            if grad_t_inp < 5.0:
                st.success("🟢 **Gradiente Bajo:** Potencial de alabeo bajo.")
            elif grad_t_inp < 10.0:
                st.warning("🟡 **Gradiente Moderado:** Considerar en la modulación.")
            else:
                st.error("🔴 **Gradiente Alto:** El gradiente térmico elevado puede incrementar el potencial de alabeo/curling. Se recomienda verificar mediante análisis específico cuando las condiciones del proyecto lo requieran.")

        if restriccion_val != "Ninguna":
            st.warning(f"⚠️ **RESTRICCIÓN TÉRMICA ({restriccion_val.upper()}):** Las restricciones pueden reducir el movimiento térmico libre y aumentar las tensiones inducidas por restricción. Revisar juntas de aislamiento, separación respecto de fundaciones, pedestales, canaletas, bordillos y otros elementos rígidos.")
            st.error("🚨 **ALERTA DE DETALLE:** Evitar el confinamiento directo de la losa contra fundaciones, pedestales, canaletas u otros elementos rígidos sin considerar una junta de aislamiento/separación compatible con los movimientos previstos.")

        st.divider()
        st.subheader("🔩 Diseño de Acero (Dovelas y Amarre)")

        D = st.session_state['esp_final_cm']
        st_dovelas = st.session_state.get('tiene_dovelas', "No")
        st_soporte = st.session_state.get('tiene_soporte', "No")
        es_doble_losa = (num_juntas_long > 0)

        # Pasadores (Dovelas)
        if st_dovelas == "No":
            dov_info = "SIN DOVELAS (TRABAZÓN DE AGREGADOS)"
            dov_check = False
        else:
            dov_check = True
            if D < 15: dov_info = "Espesor <15cm (Revisar dovelas)"
            elif D < 20: dov_info = "Ø 3/4\" (19mm) | Largo: 40 cm @ 0.30 m"
            elif D < 25: dov_info = "Ø 1\" (25mm) | Largo: 45 cm @ 0.30 m"
            elif D < 30: dov_info = "Ø 1 1/4\" (32mm) | Largo: 50 cm @ 0.30 m"
            else: dov_info = "Ø 1 1/2\" (38mm) | Largo: 50 cm @ 0.30 m"

        # Barras de Amarre
        if D < 20: specs_amarre = "Ø 1/2\" (12mm) | Largo: 60 cm @ 0.75 m"
        elif D < 25: specs_amarre = "Ø 1/2\" (12mm) | Largo: 70 cm @ 0.65 m"
        else: specs_amarre = "Ø 5/8\" (16mm) | Largo: 80 cm @ 0.60 m"

        lista_amarres = []
        if es_doble_losa: lista_amarres.append("Entre Losas (Junta Central)")
        if st_soporte == "Sí": lista_amarres.append("Losa-Bordillo (Borde)")

        if not lista_amarres:
            ama_info = "NO REQUIERE BARRAS DE AMARRE"
            ama_check = False
        else:
            ubicacion = " + ".join(lista_amarres)
            ama_info = f"Ubicación: {ubicacion} | Acero: {specs_amarre}"
            ama_check = True

        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.markdown("#### 🚀 Pasadores (Dovelas)")
            st.caption("Transferencia de carga transversal")
            if dov_check: st.success(dov_info)
            else: st.info(dov_info)

        with col_a2:
            st.markdown("#### 🔗 Barras de Amarre")
            st.caption("Anclaje junta longitudinal")
            if ama_check: st.success(ama_info)
            else: st.info(ama_info)

        st.divider()
        with st.expander("📋 ESPECIFICACIONES TÉCNICAS Y NOTAS PARA PLANO (AutoCAD)", expanded=True):

            p_diseno = st.session_state.get('periodo_diseno', 25)
            w18_esals = st.session_state.get('w18_res', 0)
            cbr_str = st.session_state.get('cbr_input', 'N/A')
            k_comb = st.session_state.get('k_res', 0)
            fc_val = st.session_state.get('fc_mpa_val', 35)
            fc_psi_txt = st.session_state.get('fc_psi_val', fc_val * 145.0377)
            sc_val = st.session_state.get('sc_psi_val', 0)
            usa_base = st.session_state.get('usar_base', False)
            tipo_base_txt = st.session_state.get('tipo_base_guardado', 'Base Granular')
            esp_base_txt = st.session_state.get('esp_base_guardado', 15)

            cbr_linea = f"CBR SUBRASANTE = {cbr_str}%" if isinstance(cbr_str, (int, float)) else f"SUELO NATURAL ({cbr_str})"
            j_txt = st.session_state.get('j_res', 0)
            esp_solado_txt = st.session_state.get('esp_solado', 0.0)

            # [AJUSTE FINAL] Sección real: losa / solado (sin aporte estructural) / soporte
            _est_k = st.session_state.get('estructura_k')
            if _est_k:  # [MÓDULO k AASHTO 93] la sección sigue la configuración usada para determinar k
                soporte_linea = st.session_state.get('soporte_k', "SUBRASANTE")
            elif usa_base:
                soporte_linea = f"SUB-BASE DE {esp_base_txt} cm DE {tipo_base_txt.upper()} SOBRE SUBRASANTE"
            else:
                soporte_linea = "SUBRASANTE PREPARADA (SIN SUBBASE GRANULAR CONVENCIONAL)"
            if esp_solado_txt > 0:
                seccion_linea = (f"LOSA DE CONCRETO h = {D:.1f} cm SOBRE SOLADO DE CONCRETO POBRE DE {esp_solado_txt:.0f} cm, "
                                 f"SOBRE {soporte_linea}. AL SOLADO NO SE LE ASIGNA CAPACIDAD ESTRUCTURAL EN EL DISEÑO (NO MODIFICA k).")
            else:
                seccion_linea = f"LOSA DE CONCRETO h = {D:.1f} cm SOBRE {soporte_linea}."

            # [AJUSTE FINAL] Junta transversal coherente con el escenario seleccionado
            if dov_check:
                junta_transv = (f"PASAJUNTAS (DOVELAS) DE ACERO LISO GRADO 60 {dov_info}. MITAD DE LA BARRA "
                                f"ENGRASADA / ENFUNDADA PARA PERMITIR LIBRE MOVIMIENTO.")
            else:
                junta_transv = (f"SIN PASADORES. TRANSFERENCIA DE CARGA POR TRABAZÓN DE AGREGADOS "
                                f"(J = {j_txt:.2f} ADOPTADO EN EL DISEÑO).")
            if ama_check:
                junta_long = f"BARRAS DE AMARRE CORRUGADAS GRADO 60 ({specs_amarre}). UBICACIÓN: {' + '.join(lista_amarres).upper()}."
            else:
                junta_long = "NO SE REQUIEREN BARRAS DE AMARRE SEGÚN LA CONFIGURACIÓN SELECCIONADA (SIN JUNTA LONGITUDINAL NI BORDILLO ANCLADO)."

            texto_planos = f"""
ESPECIFICACIONES TÉCNICAS Y NOTAS DE PAVIMENTO RÍGIDO:

1. PARÁMETROS BASE DE DISEÑO (METODOLOGÍA AASHTO 93):
   − PERIODO DE DISEÑO: {p_diseno} AÑOS.
   − TRÁNSITO DE DISEÑO ACUMULADO: {w18_esals:,.0f} ESALs (W18).
   − RESISTENCIA DEL SUELO: {cbr_linea}.
   − MÓDULO DE REACCIÓN EFECTIVO DE DISEÑO: k = {k_comb:.1f} pci ({st.session_state.get('k_origen', 'ORIGEN NO REGISTRADO')}).
   − RESISTENCIA DEL CONCRETO: f'c = {fc_val} MPa ({fc_psi_txt:,.2f} psi) | MÓDULO DE RUPTURA S'c = {sc_val:.1f} psi.
   − TRANSFERENCIA DE CARGA J = {j_txt:.2f} | COEFICIENTE DE DRENAJE Cd = {st.session_state.get('cd_res', 0):.2f}.

2. ESTRUCTURA Y GEOMETRÍA DEL PAVIMENTO:
   − ESPESOR DE LOSA REQUERIDO: h = {D:.1f} cm (TEÓRICO AASHTO {st.session_state.get('esp_pulg_base', 0)*2.54:.2f} cm, REDONDEADO HACIA ARRIBA).
   − VERIFICACIÓN DEL ESPESOR DE PROYECTO h = {st.session_state.get('h_proy_cm', 0):.1f} cm: {st.session_state.get('verif_estado', 'N/D')} ({st.session_state.get('verif_txt', '').upper()}).{chr(10) + '   − DOCUMENTOS DE REFERENCIA: ' + ' / '.join(d for d in st.session_state.get('docs_proyecto', ('', '')) if d) + '.' if any(st.session_state.get('docs_proyecto', ('', ''))) else ''}
   − SECCIÓN: {seccion_linea}
   − MODULACIÓN DE LOSAS: PAÑOS CUADRADOS/RECTANGULARES DE {ancho_losa:.2f} m x {largo_sug:.2f} m.
   − CONTROL TÉRMICO PRELIMINAR: VERIFICAR MODULACIÓN, MOVIMIENTO DE JUNTAS Y RESTRICCIONES SEGÚN CONDICIONES TÉRMICAS DEL PROYECTO.

3. JUNTAS Y REFUERZO DE ACERO:
   − JUNTA TRANSVERSAL DE CONTRACCIÓN: {junta_transv}
   − JUNTA LONGITUDINAL DE CONSTRUCCIÓN/CENTRAL: {junta_long}

4. RECOMENDACIONES GENÉRICAS DEL PROGRAMA (NO CONSTITUYEN DETALLE APROBADO;
   VERIFICAR Y AJUSTAR SEGÚN PLANOS Y ESPECIFICACIONES TÉCNICAS DEL PROYECTO):
   − BORDES LIBRES: EVALUAR BORDE ENGROSADO (REFERENCIA h = {D*1.25:.1f} cm, +25%, FRANJA 0.50 m) SOLO SI LO INDICAN LOS PLANOS.
   − COMPACTACIÓN: SEGÚN ESPECIFICACIÓN DEL PROYECTO (REFERENCIA GENÉRICA PARA CAPAS GRANULARES: 98% DEL PROCTOR MODIFICADO).
   − ASERRADO DE JUNTAS: REFERENCIA GENÉRICA, CORTE ENTRE 4 h Y 12 h POSTERIORES AL VACIADO, SEGÚN CONDICIONES DE OBRA.
   − SELLADO DE JUNTAS: REFERENCIA GENÉRICA, CORDÓN DE RESPALDO (BACKER ROD) Y SELLANTE ELASTOMÉRICO, SEGÚN ESPECIFICACIÓN DEL PROYECTO.
   − DIÁMETROS, LONGITUDES Y SEPARACIONES DE ACERO INDICADOS SON VALORES DE REFERENCIA: CONFIRMAR CONTRA EL DETALLE DE PLANOS.
"""
            st.warning("⚠️ Las notas del apartado 4 y los diámetros de acero son **recomendaciones genéricas** del programa: "
                       "no son detalles aprobados. Compárelas con los planos y especificaciones del proyecto antes de usarlas.")
            st.code(texto_planos, language="text")

with tab4:
    st.header("📊 Ábaco de Sensibilidad: Espesor vs CBR")

    st.markdown("""
    ### ¿Qué es el ábaco de diseño?
    Permite evaluar la sensibilidad del espesor frente a variaciones del **CBR del suelo natural**.
    *Nota: Si configuraste una Sub-base en la pestaña anterior, el cálculo considera el aporte estructural de esa capa sobre cada CBR evaluado.*
    *El límite máximo de diseño recomendado para subestaciones es de **25 cm**.*
    """)

    if 'w18_res' not in st.session_state:
        st.info("💡 Por favor, realice el cálculo en la pestaña **'🧱 Parámetros de Diseño'**.")
    else:
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            with c1: cbr_ini = st.number_input("CBR Inicial (%)", 1.0, 50.0, 3.0, key="aba_cbr_ini")
            with c2: cbr_fin = st.number_input("CBR Final (%)", 5.0, 100.0, 20.0, key="aba_cbr_fin")
            with c3: cbr_inc = st.number_input("Incremento (%)", 0.5, 5.0, 5.0, key="aba_cbr_inc")

        rango_cbr = np.arange(cbr_ini, cbr_fin + cbr_inc, cbr_inc)
        datos_abaco = []
        fuera_de_rango = False
        alerta_detectada = False

        usa_base_sim = st.session_state.get('usar_base', False)
        tipo_base_sim = st.session_state.get('tipo_base_guardado', "")
        esp_base_sim = st.session_state.get('esp_base_guardado', 0)

        # [CORRECCIÓN k] Punto de diseño con el k EFECTIVO seleccionado (no proviene del barrido de CBR)
        st.info(f"📌 **Punto de diseño:** k efectivo = {st.session_state.get('k_res', 0):.1f} pci "
                f"({st.session_state.get('k_origen', 'N/D').lower()}) → espesor adoptado "
                f"{st.session_state.get('esp_final_cm', 0):.1f} cm. "
                "El barrido es solo de sensibilidad.")
        # [MÓDULO k AASHTO 93] el barrido usa la MISMA cadena que el k de diseño cuando éste es AASHTO
        usar_cadena_aashto = (origen_k.startswith("Procedimiento") and fuente_mr in CORRELACIONES_MR
                              and k_inf_manual is None and k_rf_manual is None and k_ls_manual is None)
        if usar_cadena_aashto:
            st.caption(f"Cadena por CBR: {fuente_mr} → {estructura.lower()} → estrato: {estrato.lower()} → Fig. 3.5 → LS = {ls_def}. "
                       "Filas fuera del rango de la correlación se marcan.")
        else:
            st.caption("Cadena por CBR: correlación CBR–k (MPa/m → pci) + aporte de base (método de estimación anterior).")

        for c_val in rango_cbr:
            if usar_cadena_aashto:
                _c = CORRELACIONES_MR[fuente_mr]
                _r = determinar_k_aashto(estructura, [_c["f"](c_val) * f for f in factores], h_proy_cm * PULG_POR_CM,
                                         dsb_pulg_k, esb_psi_k, None, estrato, dsg_pulg_k)
                ki_final = k_corregido_ls(_r["keff"], ls_def) if _r["keff"] else np.nan
                _fuera = not (_c["cbr_min"] <= c_val <= _c["cbr_max"])
            else:
                # [CORRECCIÓN k] correlación en MPa/m convertida a pci
                k_nat_iter = k_correlacion_cbr_mpa_m(c_val) * FACTOR_MPA_M_A_PCI
                if usa_base_sim:
                    ki_final = calcular_k_combinado(k_nat_iter, esp_base_sim, tipo_base_sim)
                else:
                    ki_final = k_nat_iter
                _fuera = False

            res_iter = resolver_espesor_aashto(
                st.session_state['w18_res'], zr, s0, p0, pt, sc, cd_val, j_val, st.session_state['ec_res'], ki_final
            )
            esp_pulg = res_iter["D_pulg"] if res_iter["estado"] == "ok" else None

            if not esp_pulg:
                # [CORRECCIÓN solver] la ausencia de raíz se muestra, no se omite en silencio
                datos_abaco.append({
                    "CBR Suelo (%)": f"{c_val:.1f}%",
                    "k Comb. (pci)": round(ki_final, 1),
                    "Espesor Numérico": np.nan,
                    "Espesor Calc. (cm)": "Sin raíz AASHTO" if res_iter["estado"] == "sin_raiz" else res_iter["estado"],
                    "Espesor Adoptado (cm)": "No definido",
                    "Estado": ("⚠️ Sin raíz: revisar k (no implica espesor suficiente)" if res_iter["estado"] == "sin_raiz" else "🚨 Revisar")
                              + (" | CBR fuera de rango de la correlación MR" if _fuera else ""),
                })

            if esp_pulg:
                esp_cm = esp_pulg * 2.54
                k_mpa = ki_final / 3.684

                row = {
                    "CBR Suelo (%)": f"{c_val:.1f}%",
                    "k Comb. (pci)": round(ki_final, 1),
                    "Espesor Numérico": round(esp_cm, 2)
                }

                if esp_cm <= 25.0:
                    # [AJUSTE FINAL] redondeo SIEMPRE hacia arriba (mismo criterio que el cálculo principal)
                    adoptado = max(float(np.ceil(esp_cm)), 15.0)
                    row["Espesor Calc. (cm)"] = round(esp_cm, 2)
                    row["Espesor Adoptado (cm)"] = adoptado
                    row["Estado"] = "✅ OK"
                    if 23.0 <= adoptado <= 25.0:
                        row["Estado"] = "⚠️ Revisar"
                        alerta_detectada = True
                else:
                    fuera_de_rango = True
                    row["Espesor Calc. (cm)"] = f"Excede ({round(esp_cm,1)})"
                    row["Espesor Adoptado (cm)"] = "> 25cm"
                    row["Estado"] = "🚨 Crítico"
                if _fuera:
                    row["Estado"] += " | CBR fuera de rango de la correlación MR"

                datos_abaco.append(row)

        if datos_abaco:
            df = pd.DataFrame(datos_abaco)
            st.subheader("📋 Tabla de Sensibilidad (Considerando Estructura de Base)")
            # Columnas mixtas (número/texto) se muestran como texto para evitar errores de serialización
            st.table(df.drop(columns=["Espesor Numérico"]).astype({"Espesor Calc. (cm)": str, "Espesor Adoptado (cm)": str}))

            if alerta_detectada:
                st.warning("""
                🚨 **ALERTA DE OPTIMIZACIÓN TÉCNICA (Espesor > 23 cm):**
                Para espesores superiores a 23-25 cm, la metodología AASHTO indica que el diseño se vuelve poco eficiente.

                **Recomendaciones antes de aumentar el espesor:**
                1. **Mejorar la Sub-base:** En lugar de una losa más gruesa, considere una sub-base tratada con cemento para elevar el valor de 'k'.
                2. **Revisar Transferencia de Carga:** Verifique si el uso de dovelas de mayor diámetro puede optimizar el coeficiente 'J'.
                3. **Resistencia del Concreto:** Evalúe subir el f'c a 28, 31 o 35 MPa para mejorar el Módulo de Ruptura (S'c).
                """)

            if fuera_de_rango:
                st.error("⚠️ **LÍMITE EXCEDIDO:** El espesor calculado supera los 25 cm.")
                st.warning("""
                **🔍 Recomendaciones de Optimización:**
                Cuando el espesor calculado resulta tan elevado (ej. > 25 cm), la AASHTO '93 sugiere que el diseño debe optimizarse mediante:

                1. **Mejorar el Valor k:** No diseñe sobre la subrasante natural. Considere una sub-base granular o estabilizada con cemento para alcanzar valores de k cercanos a 250 pci.
                2. **Incrementar Resistencia (f'c):** Use un concreto de mayor desempeño (f'c 28, 31 o 35 MPa).
                3. **Verificar Tránsito:** Revise si el número de repeticiones del eje pesado es realista para una subestación.
                """)

            st.subheader("📈 Curva de Sensibilidad del Espesor")
            chart_data = df.set_index("CBR Suelo (%)")[["Espesor Numérico"]]
            chart_data.columns = ["Espesor Calculado (cm)"]
            st.line_chart(chart_data)
