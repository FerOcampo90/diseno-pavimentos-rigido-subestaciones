import streamlit as st
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm
import pandas as pd

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Diseño Pavimento Rigido - Subestaciones - AASHTO '93", page_icon="🏗️", layout="wide")

# ==========================================
# --- 1. FUNCIONES TÉCNICAS (MATEMÁTICAS) ---
# ==========================================

def calcular_w18(tpd, periodo, crecimiento, peso_eje):
    """Calcula el tránsito acumulado (ESALs)"""
    p_patron = 8.2  # Ton (18 kips)
    alfa = 4.0
    fe = (peso_eje / p_patron)**alfa
    r = crecimiento / 100
    f_crec = periodo * 365 if r == 0 else ((1 + r)**periodo - 1) / r * 365
    return fe, tpd * fe * f_crec

def calcular_k_combinado(k_subrasante, espesor_base_cm, tipo_material):
    """Estima el k combinado (Losa sobre base) según aproximaciones AASHTO/PCA."""
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

def calcular_espesor_aashto(w18, zr, s0, p0, pt, sc, cd, j, ec, k):
    """Resuelve la ecuación diferencial de AASHTO 93"""
    d_psi = p0 - pt
    def ecuacion(D):
        if D <= 0: return 1e10
        term_conf = zr * s0
        term_esp = 7.35 * np.log10(D + 1) - 0.06
        term_serv = np.log10(max(d_psi, 0.01) / 3.0) / (1 + (1.624 * 10**7) / ((D + 1)**8.46))
        factor_ajuste = 4.22 - 0.32 * pt
        num = sc * cd * (D**0.75 - 1.132)
        den = 215.63 * j * (D**0.75 - (18.42 / ((ec / k)**0.25)))
        if num / den <= 0: return 1e10
        term_mat = factor_ajuste * np.log10(num / den)
        return (term_conf + term_esp + term_serv + term_mat) - np.log10(max(w18, 1))
    
    for guess in [6, 8, 10, 12, 14, 18]:
        sol, _, ier, _ = fsolve(ecuacion, guess, full_output=True)
        if ier == 1 and sol[0] > 0: return sol[0]
    return None

# ==========================================
# --- 2. INTERFAZ DE USUARIO (UI) ---
# ==========================================

st.title("🏗️ Diseñador Pavimento Rigido - Subestaciones")

with st.expander("📝 Ecuaciones de Diseño (AASHTO 93 & Correlaciones)"):
    st.markdown("**1. Ecuación Estructural AASHTO 93 (Rígido):**")
    st.latex(r" \log_{10}(W_{18}) = Z_R S_0 + 7.35 \log_{10}(D + 1) - 0.06 + \frac{\log_{10}[\Delta PSI / (4.5 - 1.5)]}{1 + \frac{1.624 \times 10^7}{(D + 1)^{8.46}}} + (4.22 - 0.32p_t) \log_{10} \left[ \frac{S'_c C_d (D^{0.75} - 1.132)}{215.63 J \left( D^{0.75} - \frac{18.42}{(E_c/k)^{0.25}} \right)} \right] ")
    st.markdown("**2. Correlación de Módulo de Reacción (k) vs CBR:**")
    st.latex(r" \text{Si } CBR \leq 10\%: \quad k = 25.5 + 52.5 \log_{10}(CBR) ")
    st.latex(r" \text{Si } CBR > 10\%: \quad k = 46.0 + 9.08 (\log_{10}(CBR))^{4.34} ")

tab1, tab2, tab3, tab4 = st.tabs(["🚛 Tránsito y Carga", "🧱 Parámetros de Diseño", "📐 Geometría y Acero", "📊 Ábaco Sensibilidad"])

# --- TAB 1: TRÁNSITO ---
with tab1:
    st.header("Análisis de Tránsito (Eje Crítico)")
    c1, c2 = st.columns(2)
    with c1:
        tpd = st.number_input("TPD (Vehículos pesados/día)", min_value=1, value=5, step=1)
        periodo = st.number_input("Periodo de diseño (años)", min_value=1, value=25, step=1)
        tasa = st.number_input("Tasa de crecimiento (%)", min_value=0.0, value=0.0, step=0.1)
    with c2:
        peso_eje = st.number_input("Peso eje más pesado (Ton)", min_value=1.0, value=11.0, step=0.5)
        st.caption("ℹ️ El daño se calcula basándose en un **eje patrón de 8.2 Ton** (18 kips).")
    
    fe, w18_total = calcular_w18(tpd, periodo, tasa, peso_eje)
    st.metric("W18 Acumulado (ESALs)", f"{w18_total:,.0f}")
    
    if w18_total < 200000:
        st.warning("⚠️ **Nota Técnica:** El tránsito acumulado es bajo. El diseño estará gobernado por espesores mínimos constructivos.")

# --- TAB 2: PARÁMETROS AASHTO ---
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

        st.subheader("🧱 Propiedades del Concreto")
        k_modo = st.radio("Definición del factor k (S'c = k * √f'c):", 
                          ["Valores Recomendados (Memoria)", "Ingreso Manual"], horizontal=True)
        
        if k_modo == "Valores Recomendados (Memoria)":
            tipo_pav = st.selectbox("Tipo de Pavimento:", 
                                     ["Autopistas/Carreteras (k=10.8)", 
                                      "Zonas Industriales (k=10.1)", 
                                      "Urbanos Secundarios (k=9.4)",
                                      "Subestaciones / Estándar (k=8.0)"])
            map_k = {"Autopistas/Carreteras (k=10.8)": 10.8, "Zonas Industriales (k=10.1)": 10.1, "Urbanos Secundarios (k=9.4)": 9.4, "Subestaciones / Estándar (k=8.0)": 8.0}
            k_final = map_k[tipo_pav]
        else:
            k_final = st.number_input("Ingrese valor de k personalizado:", 7.0, 12.0, 8.0, step=0.1)
        
        fc_kg = st.selectbox("Resistencia f'c (kg/cm²)", [210, 245, 280, 315, 350], index=2)
        fc_psi = fc_kg * 14.2233
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
        metodo_k = st.radio("Método para definir k natural:",
                            ["Correlación AASHTO (CBR)", "Ensayo de Placa de Carga (Manual)"], horizontal=True)
        
        if metodo_k == "Correlación AASHTO (CBR)":
            cbr = st.number_input("CBR de la Subrasante (%)", 1.0, 100.0, 3.0, step=0.5, help="Soporte suelo natural")
            if cbr <= 10: k_natural = 25.5 + 52.5 * np.log10(cbr)
            else: k_natural = 46.0 + 9.08 * (np.log10(cbr))**4.34
            st.caption(f"Valor k natural calculado: **{k_natural:.1f} pci**")
            
            st.warning("⚠️ **Aviso Técnico:** La correlación CBR–k es una aproximación teórica. Se recomienda validar con **placa de carga**.")
            with st.expander("📝 Ver justificación metodológica"):
                st.info("""
                **Criterio de Diseño:** Se utiliza la correlación matemática CBR–k expresada en pci para mantener la coherencia con el modelo empírico de la AASHTO '93. 
                
                Gráficos referenciales (como la Fig. 1 de la norma) suelen sobreestimar la capacidad de soporte en subrasantes naturales al no considerar el confinamiento real de la losa. Para un diseño estructural seguro, se prioriza la consistencia con el *AASHTO Road Test*.
                """)
        else:
            col_k1, col_k2 = st.columns(2)
            with col_k1: k_manual_mpa = st.number_input("k del Ensayo (MPa/m)", 10.0, 150.0, 40.0)
            with col_k2:
                k_natural = k_manual_mpa * 3.684
                st.metric("k Natural (pci)", f"{k_natural:.1f}")
            st.success("✅ Usando valor real de ensayo de placa.")

        st.divider()
        st.markdown("#### 2. Mejoramiento / Sub-base")
        usar_base = st.checkbox("¿Incluir capa de Base/Mejoramiento?", value=True)
        
        if usar_base:
            c_b1, c_b2 = st.columns(2)
            with c_b1: tipo_base = st.selectbox("Material de Base:", ["Base Granular (Zahorra)", "Suelo Cemento / Estabilizada"])
            with c_b2: esp_base = st.number_input("Espesor Base (cm):", 10.0, 60.0, 15.0, step=5.0)
            
            k_diseno = calcular_k_combinado(k_natural, esp_base, tipo_base)
            mejora_pct = ((k_diseno - k_natural) / k_natural) * 100
            
            st.metric("Módulo k Combinado (Diseño)", f"{k_diseno:.1f} pci", delta=f"+{mejora_pct:.0f}% Incremento")
            if tipo_base == "Suelo Cemento / Estabilizada" and esp_base < 15:
                st.warning("⚠️ Recomendación: Para bases estabilizadas use espesores ≥ 15 cm.")
        else:
            k_diseno = k_natural
            st.metric("Módulo k de Diseño", f"{k_diseno:.1f} pci")
            st.info("Diseño directo sobre subrasante natural.")

        k_val = k_diseno 

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
                    "sustento": "Ideal. Las dovelas transfieren carga y el bordillo da soporte lateral.",
                    "nota_bordillo": True, "dovelas": "Sí", "soporte": "Sí"
                },
                "Escenario 2: Con Dovelas y Sin Bermas/Bordillo (J: 3.2)": {
                    "valor": 3.2,
                    "sustento": "Estándar AASHTO. Dovelas presentes, pero sin soporte lateral.",
                    "nota_bordillo": False, "dovelas": "Sí", "soporte": "No"
                },
                "Escenario 3: Sin Dovelas pero Con Bordillo/Berma (J: 3.8)": {
                    "valor": 3.8,
                    "sustento": "Trabazón agregados + Soporte Lateral.",
                    "nota_bordillo": True, "dovelas": "No", "soporte": "Sí"
                },
                "Escenario 4: Sin Dovelas y Sin Bermas (J: 4.2)": {
                    "valor": 4.2,
                    "sustento": "Crítico. Sin dovelas + Borde Libre.",
                    "nota_bordillo": False, "dovelas": "No", "soporte": "No"
                }
            }
            seleccion = st.radio("Seleccione escenario:", list(escenarios_j.keys()))
            datos_esc = escenarios_j[seleccion]
            j_val = datos_esc["valor"]
            tiene_dovelas = datos_esc["dovelas"]
            tiene_soporte = datos_esc["soporte"]

            st.write(f"**Sustento:** {datos_esc['sustento']}")
            if datos_esc["nota_bordillo"]:
                st.warning("⚠️ **Nota:** El bordillo debe ser integral o anclado.")
            st.info(f"Valor J: **{j_val}**")

        st.subheader("💧 Coeficiente de Drenaje (Cd)")
        cd_val = st.number_input("Valor Cd Seleccionado", 0.50, 1.30, 1.00, step=0.01)

    st.divider()
    if st.button("🚀 CALCULAR ESTRUCTURA"):
        esp_pulg = calcular_espesor_aashto(w18_total, zr, s0, p0, pt, sc, cd_val, j_val, ec, k_val)
        
        if esp_pulg:
            esp_exacto_cm = esp_pulg * 2.54
            esp_comercial_cm = np.ceil(esp_exacto_cm) 
            esp_final_cm = max(esp_comercial_cm, 15.0)
            
            st.session_state['esp_final_cm'] = esp_final_cm
            st.session_state['esp_pulg_base'] = esp_pulg
            st.session_state['ec_res'] = ec
            st.session_state['k_res'] = k_val
            st.session_state['w18_res'] = w18_total
            st.session_state['conf_res'] = conf
            st.session_state['tiene_dovelas'] = tiene_dovelas
            st.session_state['tiene_soporte'] = tiene_soporte
            
            st.session_state['usar_base'] = usar_base
            if usar_base:
                st.session_state['tipo_base_guardado'] = tipo_base
                st.session_state['esp_base_guardado'] = esp_base
            else:
                 st.session_state['tipo_base_guardado'] = ""
                 st.session_state['esp_base_guardado'] = 0

            st.success(f"### Espesor de Losa Recomendado: {esp_final_cm:.1f} cm")
            st.info(f"*(Valor exacto AASHTO: {esp_exacto_cm:.2f} cm | k diseño: {k_val:.1f} pci)*")

# --- TAB 3: GEOMETRÍA Y ACERO ---
with tab3:
    st.header("📐 Geometría y Acero de Refuerzo")
    
    if 'esp_final_cm' not in st.session_state:
        st.info("⚠️ Realice el cálculo en la pestaña 'Parámetros de Diseño' primero.")
    else:
        D = st.session_state['esp_final_cm']
        st_dovelas = st.session_state.get('tiene_dovelas', "No")
        st_soporte = st.session_state.get('tiene_soporte', "No")

        st.warning("⚠️ **Tránsito Excéntrico:** En subestaciones, considerar bordes engrosados +25% en perímetros.")
        
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            ancho_carril = st.number_input("Ancho total de carril (m)", 2.5, 7.0, 4.5, step=0.1)
            num_juntas_long = 1 if ancho_carril > 4.5 else 0
            ancho_losa = ancho_carril / (num_juntas_long + 1)
            st.metric("Ancho de Losa Efectivo (B)", f"{ancho_losa:.2f} m")
            es_doble_losa = (num_juntas_long > 0)
        
        with col_g2:
            esp_pulg_calculo = st.session_state['esp_pulg_base']
            nu = 0.15 
            l_pulg = ((st.session_state['ec_res'] * (esp_pulg_calculo**3)) / (12 * (1 - nu**2) * st.session_state['k_res']))**0.25
            limit_rigidez = (21 * l_pulg) * 0.0254
            largo_sug = round((min(ancho_losa * 1.25, limit_rigidez, 5.0)) * 2) / 2
            st.metric("Largo Sugerido de Losa (L)", f"{largo_sug} m")

        st.divider()
        c_v1, c_v2 = st.columns(2)
        with c_v1:
            relacion_lb = largo_sug / ancho_losa
            st.write(f"**Relación L/B:** {relacion_lb:.2f}")
            if relacion_lb <= 1.25: st.success("✅ Ideal (≤ 1.25)")
            elif relacion_lb <= 1.5: st.warning("⚠️ Aceptable (1.25 - 1.50)")
            else: st.error("🚨 Crítica (> 1.50)")
        
        with c_v2:
            st.write(f"**Límite Rigidez (21ℓ):** {limit_rigidez:.2f} m")
            if largo_sug <= limit_rigidez: st.success("✅ Cumple límite rigidez")
            else: st.error("🚨 Excede límite rigidez")

        st.divider()
        st.subheader("🔩 Diseño de Acero (Dovelas y Amarre)")

        # 1. CÁLCULO DOVELAS
        if st_dovelas == "No":
            dov_info = "🚫 No requiere (Según escenario seleccionado: Sin Dovelas)."
            dov_check = False
        else:
            dov_check = True
            if D < 15: dov_info = "Espesor < 15cm: No requiere."
            elif D < 20: dov_info = "Ø 3/4\" (19mm) | Largo: 40 cm | Sep: 30 cm"
            elif D < 25: dov_info = "Ø 1\" (25mm) | Largo: 45 cm | Sep: 30 cm"
            elif D < 30: dov_info = "Ø 1 1/4\" (32mm) | Largo: 50 cm | Sep: 30 cm"
            else: dov_info = "Ø 1 1/2\" (38mm) | Largo: 50 cm | Sep: 30 cm"

        # 2. CÁLCULO AMARRES
        if D < 20: specs_amarre = "Ø 1/2\" (12mm) | Largo: 60 cm | Sep: 75 cm"
        elif D < 25: specs_amarre = "Ø 1/2\" (12mm) | Largo: 70 cm | Sep: 65 cm"
        else: specs_amarre = "Ø 5/8\" (16mm) | Largo: 80 cm | Sep: 60 cm"

        lista_amarres = []
        if es_doble_losa: lista_amarres.append("Entre Losas (Central)")
        if st_soporte == "Sí": lista_amarres.append("Losa-Bordillo (Borde)")

        if not lista_amarres:
            ama_info = "🚫 No requiere acero de amarre."
            ama_nota = "Caso: Una sola losa sin bordillo anclado."
            ama_check = False
        else:
            ubicacion = " + ".join(lista_amarres)
            ama_info = f"**Ubicación:** {ubicacion}\n\n**Acero:** {specs_amarre}"
            ama_nota = "Barras corrugadas grado 60."
            ama_check = True

        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.markdown("#### 🚀 Pasadores (Dovelas)")
            if dov_check: st.success(dov_info)
            else: st.info(dov_info)
        with col_a2:
            st.markdown("#### 🔗 Barras de Amarre")
            if ama_check: 
                st.success(ama_info)
                st.caption(f"📝 {ama_nota}")
            else: st.info(f"{ama_info}\n\n*{ama_nota}*")

# --- TAB 4: ÁBACO ---
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

        for c_val in rango_cbr:
            if c_val <= 10: k_nat_iter = 25.5 + 52.5 * np.log10(c_val)
            else: k_nat_iter = 46.0 + 9.08 * (np.log10(c_val))**4.34
            
            if usa_base_sim: ki_final = calcular_k_combinado(k_nat_iter, esp_base_sim, tipo_base_sim)
            else: ki_final = k_nat_iter

            esp_pulg = calcular_espesor_aashto(
                st.session_state['w18_res'], zr, s0, p0, pt, sc, cd_val, j_val, st.session_state['ec_res'], ki_final
            )

            if esp_pulg:
                esp_cm = esp_pulg * 2.54
                row = {
                    "CBR Suelo (%)": f"{c_val:.1f}%",
                    "k Comb. (pci)": round(ki_final, 1),
                    "Espesor Numérico": round(esp_cm, 2)
                }

                if esp_cm <= 25.0:
                    adoptado = max(round(esp_cm, 0), 15.0)
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
                
                datos_abaco.append(row)
        
        if datos_abaco:
            df = pd.DataFrame(datos_abaco)
            st.subheader("📋 Tabla de Sensibilidad (Considerando Estructura de Base)")
            st.table(df.drop(columns=["Espesor Numérico"]))
            
            if alerta_detectada:
                st.warning("""
                🚨 **ALERTA DE OPTIMIZACIÓN TÉCNICA (Espesor > 23 cm):**
                Para espesores superiores a 23-25 cm, la metodología AASHTO indica que el diseño se vuelve poco eficiente. 
                
                **Recomendaciones antes de aumentar el espesor:**
                1. **Mejorar la Sub-base:** En lugar de una losa más gruesa, considere una sub-base tratada con cemento para elevar el valor de 'k'.
                2. **Revisar Transferencia de Carga:** Verifique si el uso de dovelas de mayor diámetro puede optimizar el coeficiente 'J'.
                3. **Resistencia del Concreto:** Evalúe subir el f'c a 280 o 315 kg/cm² para mejorar el Módulo de Ruptura (S'c).
                """)
            
            if fuera_de_rango:
                st.error("⚠️ **LÍMITE EXCEDIDO:** El espesor calculado supera los 25 cm.")
                st.warning("""
                **🔍 Recomendaciones de Optimización:**
                Cuando el espesor calculado resulta tan elevado (ej. > 25 cm), la AASHTO '93 sugiere que el diseño debe optimizarse mediante:
                
                1. **Mejorar el Valor k:** No diseñe sobre la subrasante natural. Considere una sub-base granular o estabilizada con cemento para alcanzar valores de k cercanos a 250 pci.
                2. **Incrementar Resistencia (f'c):** Use un concreto de mayor desempeño (f'c 280 o 315 kg/cm²).
                3. **Verificar Tránsito:** Revise si el número de repeticiones del eje pesado es realista para una subestación.
                """)

            st.subheader("📈 Curva de Sensibilidad del Espesor")
            chart_data = df.set_index("CBR Suelo (%)")[["Espesor Numérico"]]
            chart_data.columns = ["Espesor Calculado (cm)"]
            st.line_chart(chart_data)
