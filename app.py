import os
import pickle as pk

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class_mapping_inv = {
    0: "Normal",
    1: "Backdoor",
    2: "DDoS",
    3: "DoS",
    4: "Injection",
    5: "Password",
    6: "Ransomware",
    7: "Scanning",
    8: "XSS",
    9: "MITM",
}

req_cols = [
    "src_ip", "src_port", "dst_ip", "dst_port", "proto", "service",
    "duration", "src_bytes", "dst_bytes", "conn_state", "missed_bytes",
    "src_pkts", "dst_pkts", "dns_query", "dns_qtype", "ssl_version",
    "ssl_cipher", "http_method", "http_uri", "http_status_code", "weird_name",
]

categorical_maps = {
    "proto": {"tcp": 0, "udp": 1, "icmp": 2},
    "service": {
        "-": 0, "dns": 1, "http": 2, "ftp": 3, "ssl": 4,
        "gssapi": 5, "dce_rpc": 6, "smb": 7,
    },
    "http_method": {"-": 0, "GET": 1, "POST": 2, "HEAD": 3},
    "conn_state": {
        "S0": 0, "SF": 1, "REJ": 2, "OTH": 3, "SH": 4,
        "S1": 5, "S3": 6, "SHR": 7, "RSTR": 8, "RSTRH": 9,
        "RSTO": 10, "S2": 11, "RSTOS0": 12,
    },
}

target_cols = ["dns_query", "ssl_version", "ssl_cipher", "http_uri", "weird_name"]


def main_training():
    """Ejecutar en el notebook, después de las celdas que definen X, y y las funciones."""
    import matplotlib.pyplot as plt

    # Dividir antes de ajustar el escalador.
    X_train, X_test, y_train, y_test = train_test_split(
        X[req_cols], y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Entrenando XGBoost...")
    xgb_model = train_xgboost(
        X_train_scaled, y_train, X_test_scaled, y_test
    )

    print("Evaluando XGBoost...")
    xgb_results = evaluate_model(
        xgb_model, "XGBoost", X_test_scaled, y_test
    )
    print(f"Precision: {xgb_results['precision']:.4f}")
    print(f"Recall: {xgb_results['recall']:.4f}")
    print(f"F1 Score: {xgb_results['f1_score']:.4f}")

    cv_results = perform_cv(xgb_model, X_train_scaled, y_train, cv=5)

    plot_confusion_matrix(xgb_results["confusion_matrix"])
    plt.title("XGBoost Confusion Matrix")
    plt.tight_layout()
    plt.show()

    plot_feature_importance(xgb_model, X_train.columns)

    # Valores de ejemplo del adjunto. Para reproducir la inferencia,
    # sustituir por los mapas completos de los codificadores ajustados.
    encoder_weights = {
        "dns_query": {"-": 3.920578},
        "ssl_version": {"-": 3.462667},
        "ssl_cipher": {"-": 3.462667},
        "http_uri": {"-": 3.459716},
        "weird_name": {"-": 3.461917},
        "src_ip_le": le1.classes_.tolist(),
        "dst_ip_le": le2.classes_.tolist(),
    }

    for nombre, artefacto in (
        ("saved_model.pkl", xgb_model),
        ("scaler.pkl", scaler),
        ("encoder_weights.pkl", encoder_weights),
    ):
        with open(nombre, "wb") as archivo:
            pk.dump(artefacto, archivo)
        print(f"Guardado: {nombre}")

    return {
        "xgboost": {
            "model": xgb_model,
            "results": xgb_results,
            "cv_results": cv_results,
        }
    }


@st.cache_resource
def load_artifacts():
    with open("saved_model.pkl", "rb") as archivo:
        modelo = pk.load(archivo)
    with open("encoder_weights.pkl", "rb") as archivo:
        pesos = pk.load(archivo)
    with open("scaler.pkl", "rb") as archivo:
        escalador = pk.load(archivo)
    return modelo, pesos, escalador


def preprocess(df, encoder_weights, scaler):
    """Aplica la misma preparación al formulario y al CSV."""
    missing_cols = [col for col in req_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columnas faltantes: {', '.join(missing_cols)}")
    if df.empty:
        raise ValueError("El archivo debe incluir al menos un flujo de red.")

    df_proc = df[req_cols].copy()

    for col, mapping in categorical_maps.items():
        df_proc[col] = df_proc[col].map(mapping).fillna(0)

    for col in target_cols:
        df_proc[col] = (
            df_proc[col].astype(str).map(encoder_weights[col]).fillna(3.46)
        )

    # Valores de respaldo del adjunto para IP desconocidas.
    for col, key, fallback in (
        ("src_ip", "src_ip_le", 20.0),
        ("dst_ip", "dst_ip_le", 194.0),
    ):
        mapping = {ip: i for i, ip in enumerate(encoder_weights[key])}
        df_proc[col] = df_proc[col].astype(str).map(mapping).fillna(fallback)

    return scaler.transform(df_proc.astype(float))


def main_streamlit_app():
    st.set_page_config(
        page_title="Control CyberAI",
        page_icon="🛡️",
        layout="wide",
    )

    try:
        xgb_model, encoder_weights, scaler = load_artifacts()
    except FileNotFoundError as error:
        st.error(f"Falta el archivo {error.filename}. Colócalo junto a app.py.")
        st.stop()

    with st.sidebar:
        st.title("Control CyberAI")
        st.metric("Algoritmo", "XGBoost")
        st.info("Analiza un flujo individual o carga un archivo CSV por lote.")

    st.title("🛡️ Sistema de Detección de Intrusiones en Red")
    st.caption("Clasificación de flujos de red con XGBoost")
    tab1, tab2 = st.tabs(["📝 Análisis manual", "📁 Análisis masivo"])

    with tab1:
        st.subheader("Análisis de flujo individual")

        with st.form("flujo_individual"):
            with st.expander("1. Direccionamiento y protocolo", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    src_ip = st.text_input("IP de origen", "192.168.1.37")
                    src_port = st.number_input("Puerto de origen", 0, 65535, 4444)
                    proto = st.selectbox("Protocolo", list(categorical_maps["proto"]))
                with col2:
                    dst_ip = st.text_input("IP de destino", "192.168.1.193")
                    dst_port = st.number_input("Puerto de destino", 0, 65535, 49178)
                    conn_state = st.selectbox(
                        "Estado de conexión",
                        list(categorical_maps["conn_state"]),
                    )

            with st.expander("2. Duración, bytes y paquetes", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    duration = st.number_input(
                        "Duración en segundos", min_value=0.0,
                        value=290.37, format="%.4f",
                    )
                    src_bytes = st.number_input("Bytes de origen", min_value=0, value=101568)
                    src_pkts = st.number_input("Paquetes de origen", min_value=0, value=108)
                with col2:
                    missed_bytes = st.number_input("Bytes perdidos", min_value=0, value=0)
                    dst_bytes = st.number_input("Bytes de destino", min_value=0, value=2592)
                    dst_pkts = st.number_input("Paquetes de destino", min_value=0, value=31)

            with st.expander("3. Servicio y metadatos DNS, SSL y HTTP"):
                col1, col2 = st.columns(2)
                with col1:
                    service = st.selectbox("Servicio", list(categorical_maps["service"]))
                    dns_query = st.text_input("Consulta DNS", "-")
                    dns_qtype = st.number_input("Tipo de consulta DNS", min_value=0, value=0)
                    ssl_version = st.text_input("Versión SSL", "-")
                    ssl_cipher = st.text_input("Cifrado SSL", "-")
                with col2:
                    http_method = st.selectbox(
                        "Método HTTP", list(categorical_maps["http_method"])
                    )
                    http_uri = st.text_input("URI HTTP", "-")
                    http_status_code = st.number_input("Código HTTP", min_value=0, value=0)
                    weird_name = st.text_input("Anomalía reportada (weird_name)", "-")

            submitted = st.form_submit_button("Clasificar flujo", type="primary")

        if submitted:
            input_data = pd.DataFrame([{
                "src_ip": src_ip,
                "src_port": src_port,
                "dst_ip": dst_ip,
                "dst_port": dst_port,
                "proto": proto,
                "service": service,
                "duration": duration,
                "src_bytes": src_bytes,
                "dst_bytes": dst_bytes,
                "conn_state": conn_state,
                "missed_bytes": missed_bytes,
                "src_pkts": src_pkts,
                "dst_pkts": dst_pkts,
                "dns_query": dns_query,
                "dns_qtype": dns_qtype,
                "ssl_version": ssl_version,
                "ssl_cipher": ssl_cipher,
                "http_method": http_method,
                "http_uri": http_uri,
                "http_status_code": http_status_code,
                "weird_name": weird_name,
            }])

            try:
                scaled_values = preprocess(input_data, encoder_weights, scaler)
                prediction = xgb_model.predict(scaled_values)
                proba = xgb_model.predict_proba(scaled_values)
                pred_idx = int(np.asarray(prediction).ravel()[0])
                result_label = class_mapping_inv[pred_idx]
                confidence = proba[0, pred_idx] * 100

                st.subheader("Resultado de la clasificación")
                if result_label == "Normal":
                    st.success("Clase estimada: Normal")
                else:
                    st.warning(f"Posible intrusión: {result_label}")
                st.metric("Puntaje de la clase estimada", f"{confidence:.2f}%")
            except ValueError as error:
                st.error(f"No se pudo procesar el flujo: {error}")

    with tab2:
        st.subheader("Procesamiento masivo")
        uploaded_file = st.file_uploader(
            "Selecciona el CSV con registros de red",
            type=["csv"],
        )

        if uploaded_file is not None:
            try:
                df_original = pd.read_csv(uploaded_file, keep_default_na=False)
                st.dataframe(df_original.head(), use_container_width=True)

                if st.button("Clasificar lote", type="primary"):
                    with st.spinner("Procesando los flujos de red..."):
                        scaled_batch = preprocess(df_original, encoder_weights, scaler)
                        batch_preds = xgb_model.predict(scaled_batch)

                    resultado = df_original.copy()
                    resultado["PRED_IA_STATUS"] = [
                        class_mapping_inv[int(pred)] for pred in batch_preds
                    ]
                    distribution = resultado["PRED_IA_STATUS"].value_counts()
                    st.session_state["reporte_lote"] = resultado, distribution

                if "reporte_lote" in st.session_state:
                    resultado, distribution = st.session_state["reporte_lote"]
                    st.caption("Resultados del último lote procesado.")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Flujos por clase")
                        st.dataframe(distribution, use_container_width=True)
                    with col2:
                        st.subheader("Distribución")
                        st.bar_chart(distribution)

                    st.download_button(
                        label="Descargar reporte CSV",
                        data=resultado.to_csv(index=False).encode("utf-8"),
                        file_name="reporte_auditoria_cybersecurity_ia.csv",
                        mime="text/csv",
                    )
            except (ValueError, OSError) as error:
                st.error(f"Error al procesar el CSV: {error}")


if __name__ == "__main__":
    if not os.environ.get("STREAMLIT_SERVER_PORT") and "X" in globals():
        main_training()
    else:
        main_streamlit_app()
