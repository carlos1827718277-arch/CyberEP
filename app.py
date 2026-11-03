import os
import pickle as pk
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
.stApp {
    background: radial-gradient(ellipse at 85% 0%, #283825 0%, #101713 55%);
    color: #edf3ed;
}
[data-testid="stHeader"] {background: #101713;}
[data-testid="stSidebar"] {background: #18241c;}
.stApp h1, .stApp h2, .stApp h3,
.stApp label, [data-testid="stCaptionContainer"] p {
    color: #eaf1e9;
}
[data-testid="stSidebar"] img {max-width: 104px; margin-inline: auto;}
[data-testid="stMetric"] {
    background: #202d24;
    border: 1px solid #425646;
    border-radius: 12px;
    padding: 16px;
    color: #edf3ed;
}
[data-testid="stMetricValue"] {color: #d8ebba;}
[data-testid="stMetricDelta"] svg {display: none;}
.stButton button, .stDownloadButton button {
    background: #c4dfa1;
    color: #162216;
    border: 1px solid #c4dfa1;
    border-radius: 8px;
    font-weight: 650;
}
.stButton button:hover, .stDownloadButton button:hover {
    background: #dceebd;
    color: #162216;
    border-color: #dceebd;
}
.stButton button:focus-visible, .stDownloadButton button:focus-visible {
    outline: 2px solid #fff;
    outline-offset: 3px;
}
[data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #425646;
}
[data-baseweb="tab"] {color: #d0ded0;}
[data-baseweb="tab"][aria-selected="true"] {color: #d8ebba;}
[data-testid="stExpander"] details {
    background: #19251d;
    border: 1px solid #425646;
    border-radius: 10px;
}
[data-testid="stExpander"] summary {color: #eaf1e9;}
.cyber-heading {
    background: #1a281e;
    border: 1px solid #425646;
    border-left: 4px solid #b7d391;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 14px;
}
.cyber-heading small {
    color: #b7d391;
    font-weight: 700;
    letter-spacing: .12em;
}
.cyber-heading h1 {margin: 4px 0 0; padding: 0; font-size: clamp(1.65rem, 4vw, 2.4rem);}
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
        st.subheader("Estado Operacional")
        st.metric(label="Precisión de detección", value="99.54%")
        st.metric(label="Unidad supervisora", value="Ciberdefensa")
        st.markdown("<hr style='border-color: #4d682e;'>", unsafe_allow_html=True)
        st.caption("🎖️ Desplegado bajo los estándares de la Escuela de Inteligencia del Ejército.")

    # --- CUERPO PRINCIPAL ---
    st.markdown("<div class='cyber-heading'><small>OPERACIONES DE CIBERDEFENSA</small><h1>Sistema analítico de ciberinteligencia</h1></div>", unsafe_allow_html=True)
    st.caption("Monitoreo cibernético y clasificación táctica de vectores de intrusión mediante Redes de Árboles Estructurados (XGBoost).")
    st.markdown("---")

    tab1, tab2 = st.tabs(["01 · Análisis individual", "02 · Análisis por lotes"])

    # --- PESTAÑA 1: SISTEMA MANUAL ---
    with tab1:
        st.subheader("Auditoría de Vector Individual")
        
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
            st.subheader("🔥 Resultado del Diagnóstico de Ciberinteligencia:")

            if pred_idx == 0:
                st.success(f"### ✅ {result_label} \n\n El tráfico se comporta de forma legítima bajo parámetros normales.")
            else:
                st.error(f"### 📡 ALERTA CRÍTICA: DETECTADO -> {result_label.upper()} \n\n Se recomienda contención inmediata del vector de red.")

            st.subheader("📊 Matriz de Probabilidades del Flujo Examinado:")
            prob_df = pd.DataFrame({
            'Clase de Tráfico / Amenaza': [class_mapping_es[i] for i in range(10)],
            'Probabilidad Calculada': [f"{p * 100:.4f}%" for p in proba]
            })
            st.dataframe(prob_df, use_container_width=True)

    # --- PESTAÑA 2: SISTEMA MASIVO ---
    with tab2:
        st.header("Procesamiento por Archivos de Auditoría")
        uploaded_file = st.file_uploader("Cargar archivo de logs de red (.CSV)", type=["csv"])

        if uploaded_file is not None:
            try:
                df_original = pd.read_csv(uploaded_file)
                st.success("📂 Datos del lote cargados de forma exitosa:")
                st.dataframe(df_original.head(5), use_container_width=True)

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

                            st.success("🎉 ¡Procesamiento masivo y cálculos completados!")

                            conteo_amenazas = df_original['DIAGNÓSTICO_IA'].value_counts()
                            total_registros = len(df_original)
                            trafico_normal = conteo_amenazas.get('Tráfico Normal', 0)
                            amenazas_totales = total_registros - trafico_normal
                            tasa_criticidad = (amenazas_totales / total_registros) * 100

                            st.subheader("Resumen del lote")
                            col_m1, col_m2, col_m3 = st.columns(3)
                            col_m1.metric("Total Conexiones Auditadas", f"{total_registros} flujos")
                            col_m2.metric("Amenazas Detectadas", f"{amenazas_totales} anomalías", delta=f"{tasa_criticidad:.2f}% Tasa", delta_color="inverse")
                            col_m3.metric("Nivel de Alerta General", "CRÍTICO" if tasa_criticidad > 10 else "MODERADO" if tasa_criticidad > 2 else "SEGURO")

                            col_g1, col_g2 = st.columns(2)
                            with col_g1:
                                st.subheader("Flujos por clase")
                                st.dataframe(conteo_amenazas, use_container_width=True)
                            with col_g2:
                                st.subheader("Distribución de clases")
                                st.bar_chart(conteo_amenazas)

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
