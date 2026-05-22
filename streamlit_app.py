import pickle
import streamlit as st
import pandas as pd
import altair as alt

st.title("Dashboard Analitik DJPb")

# Sidebar Filter
st.sidebar.header("Menu Filter")
opsi = st.sidebar.selectbox("Pilih Model:", ["Klasifikasi", "Regresi"])

# Load Data
data = pd.read_csv("Data/02_realisasi_anggaran_klasifikasi.csv")

tab_overview, tab_prediction = st.tabs(["Overview", "Prediksi"])

with tab_overview:
    st.subheader("Data Sampel")
    st.dataframe(data.head())

    st.subheader("Scatter Plot: Skor IKPA vs Deviasi RPD")
    scatter = alt.Chart(data).mark_circle(size=60).encode(
        x=alt.X("skor_ikpa", title="Skor IKPA"),
        y=alt.Y("deviasi_rpd_persen", title="Deviasi RPD (%)"),
        color=alt.value("#1f77b4"),
        tooltip=["kode_satker", "nama_kementerian", "skor_ikpa", "deviasi_rpd_persen"]
    ).interactive()
    st.altair_chart(scatter, use_container_width=True)

with tab_prediction:
    st.subheader("Menu Prediksi Model")
    st.write("Isi input berikut untuk menjalankan prediksi menggunakan model di folder `model`.")

    jumlah_spm = st.number_input(
        "Jumlah SPM",
        min_value=0,
        max_value=500,
        value=110,
        step=1,
        help="Jumlah Surat Perintah Membayar (SPM)."
    )
    revisi_dipa = st.number_input(
        "Revisi DIPA",
        min_value=0,
        max_value=10,
        value=3,
        step=1,
        help="Jumlah revisi DIPA."
    )
    deviasi_rpd_persen = st.number_input(
        "Deviasi RPD (%)",
        min_value=0.0,
        max_value=40.0,
        value=15.0,
        step=0.1,
        format="%.2f",
        help="Persentase deviasi RPD."
    )
    skor_ikpa = st.number_input(
        "Skor IKPA",
        min_value=70.0,
        max_value=100.0,
        value=85.0,
        step=0.01,
        format="%.2f",
        help="Skor IKPA."
    )
    tipe_satker = st.selectbox(
        "Tipe Satker",
        options=["Kantor Pusat", "Tugas Pembantuan", "Dekonsentrasi", "Kantor Daerah"],
        help="Pilih tipe satuan kerja."
    )

    if st.button("Jalankan Prediksi"):
        try:
            import Orange

            model_path = "model/Best_model.pkcls"
            with open(model_path, "rb") as file:
                model = pickle.load(file)

            tipe_mapping = {
                "Dekonsentrasi": [1, 0, 0, 0],
                "Kantor Daerah": [0, 1, 0, 0],
                "Kantor Pusat": [0, 0, 1, 0],
                "Tugas Pembantuan": [0, 0, 0, 1],
            }
            tipe_vector = tipe_mapping[tipe_satker]
            values = [jumlah_spm, revisi_dipa, deviasi_rpd_persen, skor_ikpa] + tipe_vector
            table = Orange.data.Table.from_list(model.domain, [values])
            prediction = model(table)

            predicted_index = int(prediction[0])
            class_values = list(model.domain.class_var.values)
            predicted_label = class_values[predicted_index] if 0 <= predicted_index < len(class_values) else str(predicted_index)

            st.success(f"Hasil prediksi: {predicted_label}")
        except FileNotFoundError:
            st.error("File model tidak ditemukan di folder model. Pastikan Best_model.pkcls ada.")
        except ImportError:
            st.error("Orange belum terpasang. Instal Orange3 agar model dapat dimuat.")
        except Exception as exc:
            st.error(f"Gagal memuat model atau melakukan prediksi: {exc}")
