import streamlit as st
import numpy as np
import joblib

# ===============================
# 🎯 Load Model dan Scaler
# ===============================
model = joblib.load('knn_model.joblib')         # Ubah sesuai nama file model kamu
scaler = joblib.load('scaler.joblib')     # Ubah sesuai nama file scaler kamu
labels = ['Home Win', 'Away Win', 'Draw']

# ===============================
# 🎨 Layout Header
# ===============================
st.set_page_config(page_title="Prediksi Hasil Sepak Bola", layout="centered")
st.title("⚽ Prediksi Hasil Pertandingan Sepak Bola")
st.markdown("Aplikasi ini menggunakan Machine Learning untuk memprediksi hasil pertandingan berdasarkan statistik 4 laga terakhir masing-masing tim.")

st.markdown("---")

# ===============================
# 🧮 Formulir Input Prediksi
# ===============================
st.subheader("📌 Masukkan Statistik Pertandingan")

with st.form("prediction_form"):
    st.markdown("Masukkan statistik dari tim tuan rumah dan tim tamu:")

    col1, col2 = st.columns(2)

    with col1:
        hp = st.number_input("🏠 Home Points (last 4 matches)", 0.0, 12.0, step=0.5)
        hg = st.number_input("🏠 Home Goals (last 4 matches)", 0.0, 20.0, step=0.5)
        hxg = st.number_input("🏠 Home xG (mean of last 4)", 0.0, 5.0, step=0.1)
        har = st.number_input("🏠 Home Attack Rating", 0.0, 100.0, step=1.0)
        hdr = st.number_input("🏠 Home Defense Rating", 0.0, 100.0, step=1.0)

    with col2:
        ap = st.number_input("🛫 Away Points (last 4 matches)", 0.0, 12.0, step=0.5)
        ag = st.number_input("🛫 Away Goals (last 4 matches)", 0.0, 20.0, step=0.5)
        axg = st.number_input("🛫 Away xG (mean of last 4)", 0.0, 5.0, step=0.1)
        aar = st.number_input("🛫 Away Attack Rating", 0.0, 100.0, step=1.0)
        adr = st.number_input("🛫 Away Defense Rating", 0.0, 100.0, step=1.0)

    submitted = st.form_submit_button("🔮 Prediksi Hasil")

    # ===============================
    # 🔍 Proses Prediksi
    # ===============================
    if submitted:
        input_data = np.array([[hp, ap, hg, ag, hxg, axg, har, aar, hdr, adr]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        prediction_label = labels[prediction]

        st.markdown("### ✅ Hasil Prediksi:")
        st.success(f"Hasil pertandingan diprediksi: **{prediction_label}**")
        st.balloons()

# ===============================
# 📌 Footer
# ===============================
st.markdown("---")
st.caption("Dibuat dengan ❤️ menggunakan Streamlit dan Machine Learning")
