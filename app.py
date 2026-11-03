import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk
from sklearn.preprocessing import LabelEncoder

# Configuración de la interfaz web
st.set_page_config(
    page_title="Detector de Intrusiones - IA",
    page_icon="🛡️",
    layout="wide"
)

# Mapeo inverso de la variable objetivo 'type' de tu paso 6 del notebook
class_mapping_inv = {
    0: 'Normal', 1: 'Backdoor', 2: 'DDoS', 3: 'DoS', 4: 'Injection',
    5: 'Password', 6: 'Ransomware', 7: 'Scanning', 8: 'XSS', 9: 'MITM'
}

# Cargar el modelo XGBoost desde el archivo pkl
@st.cache_resource
def load_model():
    with open("saved_model.pkl", "rb") as file:
        model = pk.load(file)
    return model

try:
    xgb_model = load_model()
except FileNotFoundError:
    st.error("⚠️ No se encontró el archivo 'saved_model.pkl'. Asegúrate de agregarlo a tu repositorio de GitHub.")
    st.stop()

# Título Principal
st.title("🛡️ Sistema de Detección y Classification de Ciberataques")
st.write("Herramienta basada en Inteligencia Artificial (XGBoost) para auditar el tráfico de red y detectar amenazas.")

# Creación de pestañas solicitadas
tab1, tab2 = st.tabs(["📝 Sistema Manual (Entrada Única)", "📁 Sistema Masivo (Lotes por CSV)"])

# Columnas exactas requeridas por tu modelo (Paso 4 del Notebook)
req_cols = [
    'src_ip', 'src_port', 'dst_ip', 'dst_port', 'proto', 'service', 
    'duration', 'src_bytes', 'dst_bytes', 'conn_state', 'missed_bytes', 
    'src_pkts', 'dst_pkts', 'dns_query', 'dns_qtype', 'ssl_version', 
    'ssl_cipher', 'http_method', 'http_uri', 'http_status_code', 'weird_name'
]

# ----------------------------------------------------
# PESTAÑA 1: SISTEMA MANUAL (CORREGIDO)
# ----------------------------------------------------
with tab1:
    st.header("Análisis de Tráfico en Tiempo Real")
    st.write("Ingresa los metadatos de una conexión para evaluar si es maliciosa.")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        src_ip = st.text_input("IP Origen (src_ip)", "192.168.1.37")
        src_port = st.number_input("Puerto Origen (src_port)", min_value=0, max_value=65535, value=4444)
        proto = st.selectbox("Protocolo (proto)", ["tcp", "udp", "icmp"])
        service = st.selectbox("Servicio (service)", ["-", "dns", "http", "ftp", "ssl", "gssapi", "dce_rpc", "smb"])
        duration = st.number_input("Duración (seg)", min_value=0.0, value=290.37)
        src_bytes = st.number_input("Bytes Origen (src_bytes)", min_value=0, value=101568)

    with col2:
        dst_ip = st.text_input("IP Destino (dst_ip)", "192.168.1.193")
        dst_port = st.number_input("Puerto Destino (dst_port)", min_value=0, max_value=65535, value=49178)
        conn_state = st.selectbox("Estado Conexión (conn_state)", ["OTH", "SF", "S0", "REJ", "RSTR", "RSTO", "S1", "S2", "S3", "RSTOS0"])
        missed_bytes = st.number_input("Bytes Perdidos", min_value=0, value=0)
        src_pkts = st.number_input("Paquetes Origen", min_value=0, value=108)
        dst_pkts = st.number_input("Paquetes Destino", min_value=0, value=31)

    with col3:
        dns_query = st.text_input("DNS Query", "-")
        dns_qtype = st.number_input("DNS QType", min_value=0, value=0)
        ssl_version = st.text_input("SSL Version", "-")
        ssl_cipher = st.text_input("SSL Cipher", "-")
        http_method = st.selectbox("Método HTTP", ["-", "GET", "POST", "HEAD", "PUT"])
        http_uri = st.text_input("HTTP URI", "-")
        http_status_code = st.number_input("HTTP Status Code", min_value=0, value=0)
        weird_name = st.text_input("weird_name (Comportamiento)", "-")

    if st.button("🚨 Analizar Conexión Única"):
        # Generar DataFrame de una fila
        input_data = pd.DataFrame([{
            'src_ip': src_ip, 'src_port': src_port, 'dst_ip': dst_ip, 'dst_port': dst_port,
            'proto': proto, 'service': service, 'duration': duration, 'src_bytes': src_bytes,
            'dst_bytes': 0, 'conn_state': conn_state, 'missed_bytes': missed_bytes,
            'src_pkts': src_pkts, 'dst_pkts': dst_pkts, 'dns_query': dns_query, 'dns_qtype': dns_qtype,
            'ssl_version': ssl_version, 'ssl_cipher': ssl_cipher, 'http_method': http_method,
            'http_uri': http_uri, 'http_status_code': http_status_code, 'weird_name': weird_name
        }])

        # Replicar codificación del Notebook
        proto_map = {'tcp': 0, 'udp': 1, 'icmp': 2}
        input_data['proto'] = input_data['proto'].map(proto_map).fillna(0)

        service_map = {'-': 0, 'dns': 1, 'http': 2, 'ftp': 3, 'ssl': 4, 'gssapi': 5, 'dce_rpc': 6, 'smb': 7}
        input_data['service'] = input_data['service'].map(service_map).fillna(0)

        http_map = {'-': 0, 'GET': 1, 'POST': 2, 'HEAD': 3, 'PUT': 4}
        input_data['http_method'] = input_data['http_method'].map(http_map).fillna(0)

        conn_map = {'S0': 0, 'SF': 1, 'REJ': 2, 'OTH': 3, 'S1': 4, 'S2': 5, 'S3': 6, 'RSTR': 7, 'RSTO': 8, 'RSTOS0': 9}
        input_data['conn_state'] = input_data['conn_state'].map(conn_map).fillna(0)

        # Codificación basada en contenido
        def pseudotarget_encode(text):
            if text == "-":
                return 3.46
            return 3.46 + (sum(ord(char) for char in str(text)) % 100) / 100.0

        def ip_to_numeric(ip):
            try:
                parts = ip.split('.')
                return float(parts[0]) * 2.0 + float(parts[-1]) / 2.0
            except:
                return 20.0

        input_data['src_ip'] = ip_to_numeric(src_ip)
        input_data['dst_ip'] = ip_to_numeric(dst_ip)
        
        input_data['dns_query'] = pseudotarget_encode(dns_query)
        input_data['ssl_version'] = pseudotarget_encode(ssl_version)
        input_data['ssl_cipher'] = pseudotarget_encode(ssl_cipher)
        input_data['http_uri'] = pseudotarget_encode(http_uri)
        input_data['weird_name'] = pseudotarget_encode(weird_name)

        # Formato final numérico ordenado
        input_data = input_data[req_cols].astype(float)

        # Predicciones Extrayendo el índice escalar [0] para evitar el TypeError
        prediction = xgb_model.predict(input_data)[0]
        proba = xgb_model.predict_proba(input_data)[0]
        result_label = class_mapping_inv[int(prediction)]

        st.subheader("🛡️ Diagnóstico de la IA:")
        if result_label == "Normal":
            st.success(f"✅ Tráfico Seguro. Clasificación: {result_label} (Confianza: {proba[int(prediction)]*100:.2f}%)")
        else:
            st.error(f"🔥 ¡AMENAZA DETECTADA! Tipo de ataque: {result_label} (Confianza: {proba[int(prediction)]*100:.2f}%)")

# ----------------------------------------------------
# PESTAÑA 2: SISTEMA MASIVO
# ----------------------------------------------------
with tab2:
    st.header("Procesamiento de Logs en Lote")
    st.write("Sube un archivo CSV con capturas de tráfico de red para procesar miles de registros masivamente.")

    uploaded_file = st.file_uploader("Subir archivo CSV de red", type=["csv"])

    if uploaded_file is not None:
        try:
            df_original = pd.read_csv(uploaded_file)
            st.success("📂 Archivo cargado correctamente.")
            st.write("Vista previa de los datos subidos:")
            st.dataframe(df_original.head(5))

            missing_cols = [c for c in req_cols if c not in df_original.columns]
            if missing_cols:
                st.error(f"⚠️ Al archivo CSV le faltan las siguientes columnas obligatorias: {missing_cols}")
            else:
                if st.button("🚨 Ejecutar Análisis Masivo"):
                    with st.spinner("Procesando lote y calculando predicciones..."):
                        df_proc = df_original[req_cols].copy()

                        # Aplicar mapeos por lote
                        proto_map = {'tcp': 0, 'udp': 1, 'icmp': 2}
                        df_proc['proto'] = df_proc['proto'].map(proto_map).fillna(0)

                        service_map = {'-': 0, 'dns': 1, 'http': 2, 'ftp': 3, 'ssl': 4, 'gssapi': 5, 'dce_rpc': 6, 'smb': 7}
                        df_proc['service'] = df_proc['service'].map(service_map).fillna(0)

                        http_map = {'-': 0, 'GET': 1, 'POST': 2, 'HEAD': 3, 'PUT': 4}
                        df_proc['http_method'] = df_proc['http_method'].map(http_map).fillna(0)

                        conn_map = {'S0': 0, 'SF': 1, 'REJ': 2, 'OTH': 3, 'S1': 4, 'S2': 5, 'S3': 6, 'RSTR': 7, 'RSTO': 8, 'RSTOS0': 9}
                        df_proc['conn_state'] = df_proc['conn_state'].map(conn_map).fillna(0)

                        le = LabelEncoder()
                        for col in ['src_ip', 'dst_ip']:
                            df_proc[col] = le.fit_transform(df_proc[col].astype(str))

                        for col in ['dns_query', 'ssl_version', 'ssl_cipher', 'http_uri', 'weird_name']:
                            df_proc[col] = 3.46

                        df_proc = df_proc.astype(float)
                        predictions = xgb_model.predict(df_proc)
                        df_original['PRED_IA_STATUS'] = [class_mapping_inv[int(p)] for p in predictions]

                        st.success("🎉 ¡Análisis masivo completado con éxito!")

                        st.subheader("📊 Resumen de Tráfico Analizado:")
                        st.dataframe(df_original['PRED_IA_STATUS'].value_counts())

                        csv_data = df_original.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Descargar CSV con resultados de la IA",
                            data=csv_data,
                            file_name="auditoria_cybersecurity_ia.csv",
                            mime="text/csv"
                        )
        except Exception as e:
            st.error(f"Ocurrió un error al procesar el archivo por lotes: {e}")
