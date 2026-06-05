import pickle
import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(
    page_title="Dashboard Analitik DJPb",
    page_icon="📊",
    layout="wide",
)

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

@st.cache_data
def load_model(path: str):
    with open(path, "rb") as file:
        return pickle.load(file)

DATA_PATH = "Data/02_realisasi_anggaran_klasifikasi.csv"
MODEL_PATH = "model/Best_model.pkcls"

st.markdown("""
<style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
    h1 { color: #0f172a; letter-spacing: -0.03em; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 42px; padding: 0 14px; border-radius: 10px; background: #f1f5f9; color: #0f172a;
    }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #2563eb, #1d4ed8); color: white; }
    .card {
        border: 1px solid #e5e7eb; border-radius: 16px; padding: 1rem; background: linear-gradient(180deg, #ffffff, #f8fbff);
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
    }
    .tiny-note { color: #475569; font-size: 0.92rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
# Dashboard Analitik DJPb
**Visualisasi interaktif untuk analisis realisasi anggaran dan prediksi klasifikasi satker.**
""")

st.caption("Platform monitoring performa satker berbasis data realisasi anggaran, skor IKPA, dan deviasi RPD.")

data = load_data(DATA_PATH)

with st.sidebar:
    st.header("Filter Data")
    selected_provinsi = st.multiselect(
        "Provinsi",
        options=sorted(data["provinsi"].unique()),
        default=sorted(data["provinsi"].unique())[:5],
    )
    selected_tipe = st.multiselect(
        "Tipe Satker",
        options=sorted(data["tipe_satker"].unique()),
        default=sorted(data["tipe_satker"].unique()),
    )
    selected_jenis = st.multiselect(
        "Jenis Belanja Utama",
        options=sorted(data["jenis_belanja_utama"].unique()),
        default=sorted(data["jenis_belanja_utama"].unique()),
    )
    filter_tercapai = st.checkbox("Tampilkan hanya yang tercapai 95%", value=False)
    st.divider()
    st.caption("Gunakan filter untuk mengeksplorasi data dan memperbarui visualisasi secara real time.")

    st.header("Model Prediksi")
    model_type = st.selectbox("Pilih Model:", ["Klasifikasi"], help="Hanya model klasifikasi tersedia sekarang.")
    st.info("Model klasifikasi DJPb menggunakan data fitur utama dan tipe satker.")

filtered_data = data[
    data["provinsi"].isin(selected_provinsi)
    & data["tipe_satker"].isin(selected_tipe)
    & data["jenis_belanja_utama"].isin(selected_jenis)
]
if filter_tercapai:
    filtered_data = filtered_data[filtered_data["realisasi_tercapai_95persen"] == "Ya"]

tab_overview, tab_analytics, tab_prediction = st.tabs(["Overview", "Analytics", "Prediksi"])

with tab_overview:
    st.subheader("Ringkasan Data")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Satker", f"{len(filtered_data):,}")
    col2.metric("Rata-rata Skor IKPA", f"{filtered_data['skor_ikpa'].mean():.2f}")
    col3.metric("Rata-rata Deviasi RPD", f"{filtered_data['deviasi_rpd_persen'].mean():.2f}%")
    percent_tercapai = (
        filtered_data["realisasi_tercapai_95persen"].value_counts(normalize=True).get("Ya", 0) * 100
    )
    col4.metric("Tercapai 95%", f"{percent_tercapai:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Tabel Data Terfilter")
    st.dataframe(filtered_data.reset_index(drop=True), use_container_width=True)

    st.subheader("Visualisasi Geografis dan Kinerja")
    chart1, chart2 = st.columns([2, 3])

    tipo_color = alt.Color("tipe_satker:N", legend=alt.Legend(title="Tipe Satker"))
    scatter = (
        alt.Chart(filtered_data)
        .mark_circle(opacity=0.7, size=80)
        .encode(
            x=alt.X("skor_ikpa", title="Skor IKPA"),
            y=alt.Y("deviasi_rpd_persen", title="Deviasi RPD (%)"),
            color=tipo_color,
            tooltip=[
                "kode_satker",
                "nama_kementerian",
                "provinsi",
                "tipe_satker",
                "skor_ikpa",
                "deviasi_rpd_persen",
                "realisasi_tercapai_95persen",
            ],
        )
        .interactive()
    )
    chart1.altair_chart(scatter, use_container_width=True)

    bar = (
        alt.Chart(filtered_data)
        .mark_bar()
        .encode(
            x=alt.X("mean(skor_ikpa)", title="Rata-rata Skor IKPA"),
            y=alt.Y("tipe_satker:N", title="Tipe Satker", sort="-x"),
            color=alt.Color("tipe_satker:N", legend=None),
            tooltip=["tipe_satker", "mean(skor_ikpa)"],
        )
    )
    chart2.altair_chart(bar, use_container_width=True)

with tab_analytics:
    st.subheader("Analisis Distribusi dan Tren")
    st.caption("Gunakan panel ini untuk melihat pola distribusi, komposisi belanja, dan tren realisasi TW3.")
    col1, col2 = st.columns(2)

    hist = (
        alt.Chart(filtered_data)
        .mark_bar(color="#2a9d8f")
        .encode(
            alt.X("realisasi_tw3_persen", bin=alt.Bin(maxbins=25), title="Realisasi TW3 (%)"),
            y="count()",
            tooltip=["count()"],
        )
    )
    col1.altair_chart(hist, use_container_width=True)

    pie = (
        alt.Chart(filtered_data)
        .mark_arc(innerRadius=50)
        .encode(
            theta=alt.Theta("count()", title="Jumlah Satker"),
            color=alt.Color("jenis_belanja_utama:N", legend=alt.Legend(title="Jenis Belanja")),
            tooltip=["jenis_belanja_utama", "count()"],
        )
    )
    col2.altair_chart(pie, use_container_width=True)

    st.markdown("---")
    st.subheader("Rata-rata Realisasi TW3 per Provinsi")
    bar_prov = (
        alt.Chart(filtered_data)
        .mark_bar(color="#264653")
        .encode(
            x=alt.X("mean(realisasi_tw3_persen)", title="Rata-rata TW3 (%)"),
            y=alt.Y("provinsi:N", sort="-x"),
            tooltip=["provinsi", "mean(realisasi_tw3_persen)"],
        )
    )
    st.altair_chart(bar_prov, use_container_width=True)

with tab_prediction:
    st.subheader("Prediksi Klasifikasi Satker")
    st.caption("Masukkan nilai fitur utama untuk mendapatkan prediksi klasifikasi satker berbasis model Orange3.")
    st.markdown('<div class="card">', unsafe_allow_html=True)

    with st.form(key="prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            jumlah_spm = st.number_input(
                "Jumlah SPM",
                min_value=0,
                max_value=500,
                value=110,
                step=1,
                help="Jumlah Surat Perintah Membayar (SPM).",
            )
            revisi_dipa = st.number_input(
                "Revisi DIPA",
                min_value=0,
                max_value=10,
                value=3,
                step=1,
                help="Jumlah revisi DIPA.",
            )
            deviasi_rpd_persen = st.number_input(
                "Deviasi RPD (%)",
                min_value=0.0,
                max_value=40.0,
                value=15.0,
                step=0.1,
                format="%.2f",
                help="Persentase deviasi RPD.",
            )
        with col2:
            skor_ikpa = st.number_input(
                "Skor IKPA",
                min_value=70.0,
                max_value=100.0,
                value=85.0,
                step=0.01,
                format="%.2f",
                help="Skor IKPA.",
            )
            tipe_satker = st.selectbox(
                "Tipe Satker",
                options=["Kantor Pusat", "Tugas Pembantuan", "Dekonsentrasi", "Kantor Daerah"],
                help="Pilih tipe satuan kerja.",
            )

        submit = st.form_submit_button("Jalankan Prediksi")

    if submit:
        try:
            import Orange

            model = load_model(MODEL_PATH)
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
            predicted_label = "Ya" if predicted_index == 1 else "Tidak"

            st.success(f"Hasil prediksi: {predicted_label}")
            st.info("Hasil prediksi didasarkan pada model Orange3 yang dilatih pada data satker.")
        except FileNotFoundError:
            st.error("File model tidak ditemukan di folder model. Pastikan Best_model.pkcls ada.")
        except ImportError:
            st.error("Orange belum terpasang. Instal Orange3 agar model dapat dimuat.")
        except Exception as exc:
            st.error(f"Gagal memuat model atau melakukan prediksi: {exc}")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Catatan: Jika tombol prediksi tidak bekerja, periksa apakah file model dan library Orange3 sudah tersedia.")
