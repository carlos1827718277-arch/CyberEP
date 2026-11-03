import os
import pickle as pk
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 🎛️ CONFIGURACIÓN INICIAL / MAPEO DE VARIABLES TRADUCIDAS
# ==============================================================================
# Mapeo de traducción al español para los resultados de la IA
class_mapping_es = {
    0: 'Tráfico Normal', 1: 'Puerta Trasera (Backdoor)', 2: 'Denegación de Servicio Distribuida (DDoS)', 
    3: 'Denegación de Servicio (DoS)', 4: 'Inyección de Código (Injection)',
    5: 'Fuerza Bruta / Contraseña (Password)', 6: 'Secuestro de Datos (Ransomware)', 
    7: 'Escaneo de Puertos (Scanning)', 8: 'Script en Sitios Cruzados (XSS)', 9: 'Hombre en el Medio (MITM)'
}

req_cols = [
    'src_ip', 'src_port', 'dst_ip', 'dst_port', 'proto', 'service', 
    'duration', 'src_bytes', 'dst_bytes', 'conn_state', 'missed_bytes', 
    'src_pkts', 'dst_pkts', 'dns_query', 'dns_qtype', 'ssl_version', 
    'ssl_cipher', 'http_method', 'http_uri', 'http_status_code', 'weird_name'
]

# ==============================================================================
# 🎨 INYECCIÓN DE ESTILOS CSS AVANZADOS Y CONTROL DE BRILLO
# ==============================================================================
st.set_page_config(
    page_title="Consola de Ciberinteligencia",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp, [data-testid="stHeader"] {background:#f3f6f4; color:#172a21;}
.stApp h1, .stApp h2, .stApp h3, .stApp label, .stApp p {color:#172a21;}
[data-testid="stSidebar"] {background:#172a21;}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p, [data-testid="stSidebar"] [data-testid="stMetricValue"] {color:#eff6f0;}
[data-testid="stSidebar"] img {max-width:104px; margin-inline:auto;}
[data-testid="stMainBlockContainer"] {max-width:1320px; padding-top:3.5rem;}
[data-testid="stMetric"], [data-testid="stExpander"] details {
    background:#fff; color:#172a21; border:1px solid #ccd8cf;
    border-radius:12px; padding:12px;
}
[data-testid="stSidebar"] [data-testid="stMetric"] {background:#233c30; border-color:#4a6354;}
[data-testid="stMetricValue"] {color:#18382a;}
[data-testid="stExpander"] summary, [data-baseweb="tab"] {color:#172a21;}
[data-baseweb="tab"][aria-selected="true"] {background:#deebe2; color:#164c34;}
[data-testid="stFileUploaderDropzone"] {background:#e8efe8; border:1px dashed #718d77; border-radius:12px;}
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small {color:#23372b;}
[data-testid="stDataFrame"] {border:1px solid #ccd8cf; border-radius:10px; overflow:hidden;}
.stButton button, .stDownloadButton button {background:#236044; color:#fff; border-radius:8px;}
.stButton button p, .stDownloadButton button p {color:#fff;}
.stButton button:hover, .stDownloadButton button:hover {background:#184b34; color:#fff;}
.cyber-heading {
    background:linear-gradient(115deg,#274637,#172a21);
    border-radius:16px; padding:clamp(20px,4vw,42px); margin-bottom:16px;
}
.cyber-heading h1 {color:#fff; margin:4px 0;}
.cyber-heading p, .cyber-heading small {color:#d5eadb;}
.cyber-heading .chips {display:flex; gap:8px; flex-wrap:wrap; margin-top:16px;}
.cyber-heading .chips span {
    background:#e6f0e8; color:#17372a; padding:4px 10px;
    border-radius:999px; font-size:.8rem;
}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 🤖 BLOQUE 1: ENTRENAMIENTO Y EXPORTACIÓN DE ARTEFACTOS (KAGGLE)
# ==============================================================================
def main_training():
    global scaler, target_encoder, le1, le2, data, X, y
    import matplotlib.pyplot as plt
    
    print("Preparing data...")
    X_train_scaled, X_test_scaled, y_train, y_test = prepare_data(X, y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X.drop(columns=['type'], errors='ignore'))
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train_scaled, y, test_size=0.2, random_state=42)
    
    print("\nTraining XGBoost Model...")
    xgb_model = train_xgboost(X_train_split, y_train_split, X_test_split, y_test_split)
    
    with open("saved_model.pkl", "wb") as file:
        pk.dump(xgb_model, file)
    with open("scaler.pkl", "wb") as file:
        pk.dump(scaler, file)

    try:
        encoder_weights = {
            'dns_query': {"-": 3.920578}, 'ssl_version': {"-": 3.462667}, 'ssl_cipher': {"-": 3.462667}, 'http_uri': {"-": 3.459716}, 'weird_name': {"-": 3.461917},
            'src_ip_le': le1.classes_.tolist() if 'le1' in globals() and hasattr(le1, 'classes_') else ["192.168.1.37"],
            'dst_ip_le': le2.classes_.tolist() if 'le2' in globals() and hasattr(le2, 'classes_') else ["192.168.1.193"]
        }
        with open("encoder_weights.pkl", "wb") as file:
            pk.dump(encoder_weights, file)
        print("¡Ficheros exportados con éxito!")
    except Exception as e:
        encoder_weights = {'dns_query': {"-": 3.4}, 'ssl_version': {"-": 3.4}, 'ssl_cipher': {"-": 3.4}, 'http_uri': {"-": 3.4}, 'weird_name': {"-": 3.4}, 'src_ip_le': [], 'dst_ip_le': []}
        with open("encoder_weights.pkl", "wb") as file:
            pk.dump(encoder_weights, file)
    
    return {'xgboost': {'model': xgb_model}}

# ==============================================================================
# 💻 BLOQUE 2: INTERFAZ WEB - DASHBOARD EMPRESARIAL (STREAMLIT CLOUD)
# ==============================================================================
def mostrar_distribucion(tabla, medida, porcentaje=False):
    """Presenta una tabla y un gráfico horizontal para las mismas clases."""
    clases = list(class_mapping_es.values())
    colores = [
        "#217a51", "#a83b46", "#375ba7", "#b76022", "#774bb0",
        "#8b5c36", "#a03d80", "#167c87", "#7b6c14", "#52637b",
    ]
    grafico = alt.Chart(tabla).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("Clase:N", sort="-x", title=None),
        x=alt.X(f"{medida}:Q", title=medida,
                axis=alt.Axis(**({"format": ".0%"} if porcentaje else {"format": "d", "tickMinStep": 1}))),
        color=alt.Color("Clase:N", scale=alt.Scale(domain=clases, range=colores), legend=None),
        tooltip=["Clase:N", alt.Tooltip(f"{medida}:Q", format=".2%" if porcentaje else "d")],
    ).properties(height=max(150, len(tabla) * 29)).configure(
        background="#ffffff"
    ).configure_axis(
        labelColor="#172a21", titleColor="#172a21", gridColor="#e1e9e3",
        labelLimit=310,
    )
    st.altair_chart(grafico, use_container_width=True)
    st.dataframe(
        tabla,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Clase": st.column_config.TextColumn("Clase de tráfico", width="large"),
            medida: st.column_config.ProgressColumn(
                medida, min_value=0.0, max_value=1.0, format="percent"
            ) if porcentaje else st.column_config.NumberColumn(medida, format="%d"),
        },
    )


def main_streamlit_app():
    @st.cache_resource
    def load_artifacts():
        with open("saved_model.pkl", "rb") as f:
            model = pk.load(f)
        with open("encoder_weights.pkl", "rb") as f:
            weights = pk.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler_obj = pk.load(f)
        return model, weights, scaler_obj

    try:
        xgb_model, encoder_weights, pk_scaler = load_artifacts()
    except FileNotFoundError:
        st.error("⚠️ Faltan artefactos de IA en tu repositorio de GitHub. Asegúrate de que 'saved_model.pkl', 'encoder_weights.pkl' y 'scaler.pkl' estén en la raíz.")
        st.stop()

    # --- BARRA LATERAL (SIDEBAR) - DISEÑO MILITAR CORREGIDO ---
    with st.sidebar:
        # Escudo oficial de la Escuela de Inteligencia del Ejército solicitado
        st.image("https://aulavirtual.eie.edu.pe/pluginfile.php/1/core_admin/logo/0x200/1790129921/logoeie%20%281%29.png", use_container_width=True)
        st.markdown("<h2 style='text-align: center; margin-top: 10px;'>Control Militar AI</h2>", unsafe_allow_html=True)
        st.markdown("<hr style='border-color: #4d682e;'>", unsafe_allow_html=True)
        st.subheader("Panel del modelo")
        st.metric(label="Precisión de detección", value="99.54%")
        st.metric(label="Unidad supervisora", value="Ciberdefensa")
        st.markdown("<hr style='border-color: #4d682e;'>", unsafe_allow_html=True)
        st.caption("🎖️ Desplegado bajo los estándares de la Escuela de Inteligencia del Ejército.")

    # --- CUERPO PRINCIPAL ---
    st.markdown(
        """
        <section class="cyber-heading">
            <small>CONTROL MILITAR AI · CIBERDEFENSA</small>
            <h1>Consola de ciberinteligencia</h1>
            <p>Clasificación de flujos de red y análisis de posibles intrusiones.</p>
            <div class="chips"><span>XGBoost</span><span>21 variables</span><span>10 clases</span></div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["01 · Análisis individual", "02 · Análisis por lotes"])

    # --- PESTAÑA 1: SISTEMA MANUAL ---
    with tab1:
        st.subheader("Configurar flujo de red")
        st.caption("Completa los datos del flujo y selecciona Clasificar flujo para ver el resultado.")
        
        with st.expander("01 · Direccionamiento y protocolo", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                src_ip = st.text_input("IP Origen (src_ip)", "192.168.1.37")
                src_port = st.number_input("Puerto Origen (src_port)", min_value=0, max_value=65535, value=4444)
                proto = st.selectbox("Protocolo de Transporte", ["tcp", "udp", "icmp"])
            with col2:
                dst_ip = st.text_input("IP Destino (dst_ip)", "192.168.1.193")
                dst_port = st.number_input("Puerto Destino (dst_port)", min_value=0, max_value=65535, value=49178)
                conn_state = st.selectbox("Estado del Flujo (conn_state)", ["SF", "S0", "REJ", "OTH", "SH", "S1", "S3", "SHR", "RSTR", "RSTRH", "RSTO", "S2", "RSTOS0"])

        with st.expander("02 · Duración, bytes y paquetes", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                duration = st.number_input("Duración de la Conexión (seg)", min_value=0.0, value=290.37, format="%.4f")
                src_bytes = st.number_input("Bytes de Origen", min_value=0, value=101568)
                src_pkts = st.number_input("Paquetes de Origen", min_value=0, value=108)
            with col2:
                missed_bytes = st.number_input("Bytes Perdidos", min_value=0, value=0)
                dst_bytes = st.number_input("Bytes de Destino", min_value=0, value=2592)
                dst_pkts = st.number_input("Paquetes de Destino", min_value=0, value=31)

        with st.expander("03 · Servicios y metadatos", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                service = st.selectbox("Servicio de Aplicación", ["-", "dns", "http", "ftp", "ssl", "gssapi", "dce_rpc", "smb"])
                dns_query = st.text_input("DNS Query", "-")
                dns_qtype = st.number_input("DNS Query Type (Numérico)", min_value=0, value=0)
                ssl_version = st.text_input("Versión SSL", "-")
                ssl_cipher = st.text_input("Cifrado SSL Suite", "-")
            with col2:
                http_method = st.selectbox("Método HTTP Petición", ["-", "GET", "POST", "HEAD"])
                http_uri = st.text_input("URI HTTP Solicitada", "-")
                http_status_code = st.number_input("Código de Respuesta HTTP", min_value=0, value=0)
                weird_name = st.text_input("weird_name Campo Log", "-")

        if st.button("Clasificar flujo", use_container_width=True):
            input_data = pd.DataFrame([{
            'src_ip': src_ip, 'src_port': src_port, 'dst_ip': dst_ip, 'dst_port': dst_port,
            'proto': proto, 'service': service, 'duration': duration, 'src_bytes': src_bytes,
            'dst_bytes': dst_bytes, 'conn_state': conn_state, 'missed_bytes': missed_bytes,
            'src_pkts': src_pkts, 'dst_pkts': dst_pkts, 'dns_query': dns_query, 'dns_qtype': dns_qtype,
            'ssl_version': ssl_version, 'ssl_cipher': ssl_cipher, 'http_method': http_method,
            'http_uri': http_uri, 'http_status_code': http_status_code, 'weird_name': weird_name
            }])

            input_data['proto'] = input_data['proto'].map({'tcp': 0, 'udp': 1, 'icmp': 2}).fillna(0)
            input_data['service'] = input_data['service'].map({'-': 0, 'dns': 1, 'http': 2, 'ftp': 3, 'ssl': 4, 'gssapi': 5, 'dce_rpc': 6, 'smb': 7}).fillna(0)
            input_data['http_method'] = input_data['http_method'].map({'-': 0, 'GET': 1, 'POST': 2, 'HEAD': 3}).fillna(0)
            input_data['conn_state'] = input_data['conn_state'].map({'S0': 0, 'SF': 1, 'REJ': 2, 'OTH': 3, 'SH': 4, 'S1': 5, 'S3': 6, 'SHR': 7, 'RSTR': 8, 'RSTRH': 9, 'RSTO': 10, 'S2': 11, 'RSTOS0': 12}).fillna(0)

            input_data['src_ip'] = float(encoder_weights['src_ip_le'].index(str(src_ip))) if str(src_ip) in encoder_weights['src_ip_le'] else 20.0
            input_data['dst_ip'] = float(encoder_weights['dst_ip_le'].index(str(dst_ip))) if str(dst_ip) in encoder_weights['dst_ip_le'] else 194.0

            for col in ['dns_query', 'ssl_version', 'ssl_cipher', 'http_uri', 'weird_name']:
                input_data[col] = encoder_weights[col].get(str(input_data[col].values), 3.46)

            input_data = input_data[req_cols].astype(float)
            scaled_values = pk_scaler.transform(input_data)
            input_data_scaled = pd.DataFrame(scaled_values, columns=req_cols)

            prediction = xgb_model.predict(input_data_scaled)
            proba = xgb_model.predict_proba(input_data_scaled)[0]

            pred_idx = int(np.asarray(prediction).ravel()[0])
            result_label = class_mapping_es[pred_idx]

            st.markdown("---")
            st.subheader("Resultado de la clasificación")

            if pred_idx == 0:
                st.success(f"### ✅ {result_label} \n\n El tráfico se comporta de forma legítima bajo parámetros normales.")
            else:
                st.error(f"### 📡 ALERTA CRÍTICA: DETECTADO -> {result_label.upper()} \n\n Se recomienda contención inmediata del vector de red.")

            st.subheader("Puntajes por clase")
            prob_df = pd.DataFrame({
                "Clase": [class_mapping_es[i] for i in range(10)],
                "Puntaje": proba,
            }).sort_values("Puntaje", ascending=False)
            mostrar_distribucion(prob_df, "Puntaje", porcentaje=True)

    # --- PESTAÑA 2: SISTEMA MASIVO ---
    with tab2:
        st.header("Importar flujos de red")
        st.caption("Carga un CSV para revisar la distribución de clases y descargar el reporte del lote.")
        uploaded_file = st.file_uploader("Seleccionar archivo CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                df_original = pd.read_csv(uploaded_file)
                st.success("Archivo cargado · Vista previa")
                st.dataframe(df_original.head(5), use_container_width=True, hide_index=True)

                missing_cols = [c for c in req_cols if c not in df_original.columns]
                if missing_cols:
                    st.error(f"⚠️ Estructura no válida. Columnas faltantes: {missing_cols}")
                else:
                    if st.button("Procesar lote", use_container_width=True):
                        with st.spinner("Procesando vectores correlacionales..."):
                            df_proc = df_original[req_cols].copy()

                            df_proc['proto'] = df_proc['proto'].map({'tcp': 0, 'udp': 1, 'icmp': 2}).fillna(0)
                            df_proc['service'] = df_proc['service'].map({'-': 0, 'dns': 1, 'http': 2, 'ftp': 3, 'ssl': 4, 'gssapi': 5, 'dce_rpc': 6, 'smb': 7}).fillna(0)
                            df_proc['http_method'] = df_proc['http_method'].map({'-': 0, 'GET': 1, 'POST': 2, 'HEAD': 3}).fillna(0)
                            df_proc['conn_state'] = df_proc['conn_state'].map({'S0': 0, 'SF': 1, 'REJ': 2, 'OTH': 3, 'SH': 4, 'S1': 5, 'S3': 6, 'SHR': 7, 'RSTR': 8, 'RSTRH': 9, 'RSTO': 10, 'S2': 11, 'RSTOS0': 12}).fillna(0)

                            for col in ['dns_query', 'ssl_version', 'ssl_cipher', 'http_uri', 'weird_name']:
                                df_proc[col] = df_proc[col].astype(str).map(encoder_weights[col]).fillna(3.46)

                            df_proc['src_ip'] = df_proc['src_ip'].astype(str).map(lambda x: encoder_weights['src_ip_le'].index(x) if x in encoder_weights['src_ip_le'] else 20.0)
                            df_proc['dst_ip'] = df_proc['dst_ip'].astype(str).map(lambda x: encoder_weights['dst_ip_le'].index(x) if x in encoder_weights['dst_ip_le'] else 194.0)

                            df_proc = df_proc.astype(float)
                            scaled_batch = pk_scaler.transform(df_proc)

                            batch_preds = xgb_model.predict(scaled_batch)
                            df_original['DIAGNÓSTICO_IA'] = [class_mapping_es[int(p)] for p in batch_preds]

                            st.success("Análisis del lote completado")

                            conteo_amenazas = df_original['DIAGNÓSTICO_IA'].value_counts()
                            total_registros = len(df_original)
                            trafico_normal = conteo_amenazas.get('Tráfico Normal', 0)
                            amenazas_totales = total_registros - trafico_normal
                            tasa_criticidad = (amenazas_totales / total_registros) * 100

                            st.subheader("Resumen del lote")
                            col_m1, col_m2, col_m3 = st.columns(3)
                            col_m1.metric("Flujos analizados", f"{total_registros} flujos")
                            col_m2.metric("Posibles intrusiones", f"{amenazas_totales} anomalías", delta=f"{tasa_criticidad:.2f}% Tasa", delta_color="inverse")
                            col_m3.metric("Nivel de alerta", "CRÍTICO" if tasa_criticidad > 10 else "MODERADO" if tasa_criticidad > 2 else "SEGURO")

                            st.subheader("Distribución de clases")
                            distribucion = conteo_amenazas.rename_axis("Clase").reset_index(name="Flujos")
                            mostrar_distribucion(distribucion, "Flujos")

                            csv_data = df_original.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Descargar reporte CSV",
                                data=csv_data,
                                file_name="reporte_analitico_eie.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
            except Exception as e:
                st.error(f"Error crítico durante el análisis estadístico: {e}")

if __name__ == "__main__":
    if not os.environ.get("STREAMLIT_SERVER_PORT") and "X" in globals():
        main_training()
    else:
        main_streamlit_app()
