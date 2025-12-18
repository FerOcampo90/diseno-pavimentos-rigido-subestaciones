import streamlit as st
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm
import pandas as pd

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Pavimentos Subestaciones - AASHTO '93", page_icon="🏗️", layout="wide")

# --- FUNCIONES TÉCNICAS ---
def calcular_w18(tpd, periodo, crecimiento, peso_eje):
    p_patron = 8.2  # Ton (18 kips)
    alfa = 4.0
    fe = (peso_eje / p_patron)**alfa
    r = crecimiento / 100
    f_crec = periodo * 365 if r == 0 else ((1 + r)**periodo - 1) / r * 365
    return fe, tpd * fe * f_crec

def calcular_espesor_aashto(w18, zr, s0, p0, pt, sc, cd, j, ec, k):
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

# --- INTERFAZ ---
st.title("🏗️ Diseñador Profesional de Pavimentos - Subestaciones")

# Visualización de todas las fórmulas de diseño
with st.expander("📝 Ecuaciones de Diseño (AASHTO 93 & Correlaciones)"):
    st.markdown("**1. Ecuación Estructural AASHTO 93 (Rígido):**")
    st.latex(r" \log_{10}(W_{18}) = Z_R S_0 + 7.35 \log_{10}(D + 1) - 0.06 + \frac{\log_{10}[\Delta PSI / (4.5 - 1.5)]}{1 + \frac{1.624 \times 10^7}{(D + 1)^{8.46}}} + (4.22 - 0.32p_t) \log_{10} \left[ \frac{S'_c C_d (D^{0.75} - 1.132)}{215.63 J \left( D^{0.75} - \frac{18.42}{(E_c/k)^{0.25}} \right)} \right] ")
    
    st.markdown("**2. Correlación de Módulo de Reacción (k) vs CBR:**")
    st.latex(r" \text{Si } CBR \leq 10\%: \quad k = 25.5 + 52.5 \log_{10}(CBR) ")
    st.latex(r" \text{Si } CBR > 10\%: \quad k = 46.0 + 9.08 (\log_{10}(CBR))^{4.34} ")

tab1, tab2, tab3, tab4 = st.tabs(["🚛 Tránsito y Carga", "🧱 Parámetros de Diseño", "📐 Geometría y Juntas", "📊 Ábaco"])

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
        st.warning("⚠️ **Nota Técnica:** El tránsito acumulado es bajo para el rango de aplicación original de AASHTO 93. El espesor obtenido está gobernado por criterios mínimos constructivos.")

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
        # 1. Definición del Factor k (Correlación para S'c)
        k_modo = st.radio("Definición del factor k (S'c = k * √f'c):", 
                          ["Valores Recomendados (Memoria)", "Ingreso Manual"], horizontal=True)
        
        if k_modo == "Valores Recomendados (Memoria)":
            tipo_pav = st.selectbox("Tipo de Pavimento:", 
                                     ["Autopistas/Carreteras (k=10.8)", 
                                      "Zonas Industriales (k=10.1)", 
                                      "Urbanos Secundarios (k=9.4)",
                                      "Subestaciones / Estándar (k=8.0)"])
            
            map_k = {
                "Autopistas/Carreteras (k=10.8)": 10.8, 
                "Zonas Industriales (k=10.1)": 10.1,
                "Urbanos Secundarios (k=9.4)": 9.4,
                "Subestaciones / Estándar (k=8.0)": 8.0
            }
            k_final = map_k[tipo_pav]
        else:
            k_final = st.number_input("Ingrese valor de k personalizado:", 7.0, 12.0, 8.0, step=0.1)
        # 2. Resistencia a la Compresión
        fc_kg = st.selectbox("Resistencia f'c (kg/cm²)", [210, 245, 280, 315, 350], index=2)
        fc_psi = fc_kg * 14.2233
        # 3. Cálculos Finales
        sc = k_final * np.sqrt(fc_psi)
        ec = 57000 * np.sqrt(fc_psi)
        # 4. Visualización de Resultados
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
        st.subheader("🌍 Soporte del Suelo (Subrasante)")
        
        # Selector de método para k
        metodo_k = st.radio(
            "Método para definir el Módulo k:",
            ["Correlación AASHTO (CBR)", "Ensayo de Placa de Carga (Manual)"],
            horizontal=True
        )
        
        if metodo_k == "Correlación AASHTO (CBR)":
            cbr = st.number_input("CBR de diseño (%)", 1.0, 100.0, 20.0, help="CBR de la subrasante natural")
            
            # Fórmulas de correlación técnica (AASHTO / pci)
            if cbr <= 10:
                k_val = 25.5 + 52.5 * np.log10(cbr)
            else:
                k_val = 46.0 + 9.08 * (np.log10(cbr))**4.34
            
            st.metric("Módulo k Estimado", f"{k_val:.1f} pci")
            
            # Advertencia solicitada y Nota Técnica
            st.warning("⚠️ **Aviso Técnico:** La correlación CBR–k es una aproximación teórica. Se recomienda validar con **placa de carga**.")
            
            with st.expander("📝 Ver justificación metodológica"):
                st.info("""
                **Criterio de Diseño:** Se utiliza la correlación matemática CBR–k expresada en pci para mantener la coherencia con el modelo empírico de la AASHTO '93. 
                
                Gráficos referenciales (como la Fig. 1 de la norma) suelen sobreestimar la capacidad de soporte en subrasantes naturales al no considerar el confinamiento real de la losa. Para un diseño estructural seguro, se prioriza la consistencia con el *AASHTO Road Test*.
                """)
        
        else:
            # Opción manual para cuando hay prueba de placa (ASTM D1196)
            col_k1, col_k2 = st.columns(2)
            with col_k1:
                k_manual_mpa = st.number_input("k del Ensayo (MPa/m)", 10.0, 150.0, 70.0)
            with col_k2:
                # Conversión técnica: 1 MPa/m = 3.684 pci
                k_val = k_manual_mpa * 3.684
                st.metric("k para Diseño (pci)", f"{k_val:.1f}")
            
            st.success("✅ Usando valor real de ensayo de placa (prevalece sobre estimaciones teóricas).")
        
        # El valor 'k_val' se guarda automáticamente para el cálculo AASHTO

        st.subheader("🔗 Transferencia de Carga (J)")
        j_manual = st.toggle("Ingresar J manualmente", False)
        if j_manual:
            j_val = st.number_input("Valor J personalizado", 2.0, 5.0, 3.2, step=0.1)
            j_txt = "Manual"
        else:
            j_opt = st.radio("Configuración de Juntas:", 
                            ["Con Pasadores (Dovelas) - J: 3.2", "Sin Pasadores (Trabazón) - J: 4.2", "Losa unida a berma - J: 2.7"])
            j_val = 3.2 if "3.2" in j_opt else (4.2 if "4.2" in j_opt else 2.7)
            j_txt = j_opt
        st.info(f"Valor J activo: **{j_val}**")

        st.subheader("💧 Coeficiente de Drenaje (Cd)")
        # --- TABLA DE DRENAJE RESTAURADA ---
        tabla_cd = pd.DataFrame({
            "Calidad de Drenaje": ["Excelente", "Bueno", "Regular", "Pobre", "Muy Pobre"],
            "Agua removida en": ["2 horas", "1 día", "1 semana", "1 mes", "Nunca"],
            "<1% de exposición": [1.25, 1.15, 1.05, 0.95, 0.80],
            "1-5% de exposición": [1.20, 1.10, 1.00, 0.90, 0.75],
            "5-25% de exposición": [1.15, 1.05, 0.95, 0.80, 0.65],
            ">25% de exposición": [1.10, 1.00, 0.80, 0.70, 0.55]
        })
        st.table(tabla_cd)
        cd_val = st.number_input("Valor Cd Seleccionado", 0.50, 1.30, 1.00, step=0.01)

    st.divider()
    if st.button("🚀 CALCULAR ESTRUCTURA"):
            esp_pulg = calcular_espesor_aashto(w18_total, zr, s0, p0, pt, sc, cd_val, j_val, ec, k_val)
            
            if esp_pulg:
                # 1. Convertimos el valor exacto del solver a cm inmediatamente
                esp_exacto_cm = esp_pulg * 2.54
                
                # 2. Aplicamos el redondeo comercial directamente en cm (múltiplos de 1 cm o 0.5 cm según prefieras)
                # Aquí lo redondeamos al entero superior para facilitar la construcción
                esp_comercial_cm = np.ceil(esp_exacto_cm) 
                
                # 3. Validamos contra el mínimo constructivo de la memoria (15 cm)
                esp_final_cm = max(esp_comercial_cm, 15.0)
                
                # Guardamos en session_state para las otras pestañas
                st.session_state['esp_final_cm'] = esp_final_cm
                st.session_state['esp_pulg_base'] = esp_pulg # Para cálculos internos de rigidez
                st.session_state['ec_res'] = ec
                st.session_state['k_res'] = k_val
                st.session_state['w18_res'] = w18_total
                st.session_state['conf_res'] = conf
    
                # Mostramos el resultado priorizando cm
                st.success(f"### Espesor de Losa Recomendado: {esp_final_cm:.1f} cm")
                st.info(f"*(Valor exacto calculado por AASHTO: {esp_exacto_cm:.2f} cm)*")
with tab3:
    st.header("📐 Recomendaciones Geométricas")
    
    # Cambiamos la validación al nuevo nombre de variable
    if 'esp_final_cm' not in st.session_state:
        st.info("⚠️ Realice el cálculo en la pestaña 'Parámetros de Diseño' para habilitar esta sección.")
    else:
        st.warning("⚠️ **Tránsito Excéntrico:** En subestaciones, el tránsito suele circular cerca del borde. Se recomienda considerar bordes engrosados +25% del espesor en perímetros.")
        
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            ancho_carril = st.number_input("Ancho total de carril (m)", 2.5, 7.0, 4.5, step=0.1)
            num_juntas_long = 1 if ancho_carril > 4.5 else 0
            ancho_losa = ancho_carril / (num_juntas_long + 1)
            st.metric("Ancho de Losa Efectivo (B)", f"{ancho_losa:.2f} m")
        
        with col_g2:
            # Recuperamos el espesor en pulgadas original para la fórmula del radio de rigidez (l)
            # La fórmula técnica del radio de rigidez RELATIVA (ℓ) requiere unidades en pulgadas
            esp_pulg_calculo = st.session_state['esp_pulg_base']
            nu = 0.15 
            
            # Radio de rigidez relativa (ℓ) en pulgadas
            l_pulg = ((st.session_state['ec_res'] * (esp_pulg_calculo**3)) / (12 * (1 - nu**2) * st.session_state['k_res']))**0.25
            
            # Límite de rigidez (21 veces l) convertido a metros
            limit_rigidez = (21 * l_pulg) * 0.0254
            
            # Largo sugerido (L) redondeado a múltiplos de 0.5m
            largo_sug = round((min(ancho_losa * 1.25, limit_rigidez, 5.0)) * 2) / 2
            st.metric("Largo Sugerido de Losa (L)", f"{largo_sug} m")
            st.write("📌 **Corte de juntas:** Aserrado temprano (4–12 h después del vaciado).")

        st.divider()
        st.subheader("🔍 Verificaciones Técnicas")
        c_v1, c_v2 = st.columns(2)
        with c_v1:
            relacion_lb = largo_sug / ancho_losa
            st.write(f"**1. Relación de Aspecto (L/B):** {relacion_lb:.2f}")
            if relacion_lb <= 1.25: st.success("✅ Relación ideal (≤ 1.25).")
            elif relacion_lb <= 1.5: st.warning("⚠️ Relación aceptable (1.25 - 1.50).")
            else: st.error("🚨 Relación crítica (> 1.50).")

        with c_v2:
            st.write(f"**2. Radio de Rigidez Relativa (ℓ):** {l_pulg:.2f} pulg")
            st.write(f"**3. Espaciamiento Máximo (21ℓ):** {limit_rigidez:.2f} m")
            if largo_sug <= limit_rigidez: st.success("✅ Cumple límite de rigidez.")
            else: st.error("🚨 Excede límite de rigidez.")

        st.divider()
        st.subheader("📝 Resumen de Memoria Técnica")
        resumen_texto = f"""
        El pavimento rígido fue diseñado para un tránsito acumulado de {st.session_state['w18_res']:,.0f} ESALs, 
        con una confiabilidad del {st.session_state['conf_res']}%. 
        
        **Espesor Adoptado:** {st.session_state['esp_final_cm']:.1f} cm. 
        La modulación propuesta ({ancho_losa:.2f} m x {largo_sug:.2f} m) cumple criterios técnicos de rigidez. 
        """
        st.info(resumen_texto)

st.markdown("---")
st.markdown("<p style='color: gray; font-size: 0.8em;'>Nota: El ancho de carril define la geometría constructiva; no es una variable de entrada estructural en la ecuación de la metodología AASHTO 93.</p>", unsafe_allow_html=True)

with tab4:
    st.header("📊 Ábaco de Sensibilidad: Espesor vs CBR")
    
    st.markdown("""
    ### ¿Qué es el ábaco de diseño?
    Permite evaluar la sensibilidad del espesor frente a variaciones del **CBR**, manteniendo constante el tránsito. 
    *El límite máximo de diseño recomendado para subestaciones es de **25 cm**.*
    """)

    if 'w18_res' not in st.session_state:
        st.info("💡 Por favor, realice el cálculo en la pestaña **'🧱 Parámetros de Diseño'**.")
    else:
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            with c1: cbr_ini = st.number_input("CBR Inicial (%)", 1.0, 50.0, 3.0, key="aba_cbr_ini")
            with c2: cbr_fin = st.number_input("CBR Final (%)", 5.0, 100.0, 20.0, key="aba_cbr_fin")
            with c3: cbr_inc = st.number_input("Incremento (%)", 0.5, 5.0, 1.0, key="aba_cbr_inc")

        rango_cbr = np.arange(cbr_ini, cbr_fin + cbr_inc, cbr_inc)
        datos_abaco = []
        fuera_de_rango = False
        alerta_detectada = False # Inicialización para evitar Error

        for c_val in rango_cbr:
            # Cálculo de k (pci)
            ki = 25.5 + 52.5 * np.log10(c_val) if c_val <= 10 else 46.0 + 9.08 * (np.log10(c_val))**4.34
            
            esp_pulg = calcular_espesor_aashto(
                st.session_state['w18_res'], zr, s0, p0, pt, sc, cd_val, j_val, st.session_state['ec_res'], ki
            )

            if esp_pulg:
                esp_cm = esp_pulg * 2.54
                k_mpa = ki / 3.684 # Conversión para tu memoria
                
                # Guardamos siempre el valor numérico para el gráfico
                row = {
                    "CBR (%)": f"{c_val:.1f}%",
                    "k (pci)": round(ki, 1),
                    "k (MPa/m)": round(k_mpa, 1),
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
                    row["Espesor Adoptado (cm)"] = "Excede 25cm"
                    row["Estado"] = "🚨 Crítico"
                
                datos_abaco.append(row)
        
        if datos_abaco:
                    df = pd.DataFrame(datos_abaco)
                    st.subheader("📋 Tabla de Sensibilidad CBR vs Espesor")
                    st.table(df.drop(columns=["Espesor Numérico"]))
                    
                    # --- LA NOTA DE ADVERTENCIA QUE SE HABÍA PERDIDO ---
                    if alerta_detectada:
                        st.warning("""
                        🚨 **ALERTA DE OPTIMIZACIÓN TÉCNICA (Espesor > 23 cm):**
                        Para espesores superiores a 23-25 cm, la metodología AASHTO indica que el diseño se vuelve poco eficiente. 
                        
                        **Recomendaciones antes de aumentar el espesor:**
                        1. **Mejorar la Sub-base:** En lugar de una losa más gruesa, considere una sub-base tratada con cemento para elevar el valor de 'k'.
                        2. **Revisar Transferencia de Carga:** Verifique si el uso de pasajeros (dovelas) de mayor diámetro puede optimizar el coeficiente 'J'.
                        3. **Resistencia del Concreto:** Evalúe subir el f'c a 280 o 315 kg/cm² para mejorar el Módulo de Ruptura (S'c).
                        """)
                    
                    if fuera_de_rango:
                        st.error("⚠️ **LÍMITE EXCEDIDO:** Algunos valores calculados superan los 25 cm. Esto indica un tránsito extremadamente pesado o un suelo muy pobre que requiere estabilización obligatoria.")
        
                    # --- GRÁFICO ---
                    st.subheader("📈 Curva de Sensibilidad del Espesor")
                    chart_data = df.set_index("CBR (%)")[["Espesor Numérico"]]
                    chart_data.columns = ["Espesor Calculado (cm)"]
                    st.line_chart(chart_data)
                        










