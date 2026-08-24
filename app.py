"""
Dashboard Perbandingan ML vs DL
================================
Dashboard TAMBAHAN, terpisah dari dashboard TA utama. Isinya perbandingan
Machine Learning (Naive Bayes, Random Forest, Logistic Regression -- fitur
TF-IDF) vs Deep Learning (DistilBERT, locked dari TA, tidak di-retrain di
sini), untuk dua task: Aspek (multi-label, 4 kelas) dan Sentimen
(multi-class, 3 kelas). Dataset sama dengan dataset TA.

Semua kode dalam SATU file (`app.py`) sesuai preferensi user -- lebih
gampang di-track, mirip pola notebook yang selalu dikirim utuh.

Ditulis ulang dari awal (2026-08-21) menggantikan draft sebelumnya yang
sempat kepecah multi-file lalu digabung lagi. Angka statis di section 2
LOCKED dari `hasil_perbandingan_ml_dl.json` (diambil langsung dari Google
Drive, termasuk breakdown per-aspek untuk SEMUA model ML, bukan cuma
DistilBERT). Halaman live prediction (section 7) berisi logic lengkap.

Update 2026-08-24 (Fase A selesai): model sudah diupload user ke HF Hub lewat
notebook Colab (`upload_script/FaseA_Upload_ke_HF_Hub.ipynb`, dibuat karena
sandbox Cowork tidak bisa akses huggingface.co langsung). `HF_MODEL_REPO`,
`DISTILBERT_ASPEK_REPO`, `DISTILBERT_SENTIMEN_REPO` di bawah sudah diisi repo
asli (`Jauzaa/...`, public) -- halaman Live Prediksi sekarang seharusnya bisa
jalan end-to-end.

Update 2026-08-21 (redesign kedua): sidebar dipecah jadi 4 menu terpisah --
Machine Learning (ML), Deep Learning (DL), ML vs DL, dan Live Prediksi.
Halaman ML dan DL masing-masing sekarang punya section "Dataset",
"Konfigurasi Model" (hyperparameter persis, diambil ulang dari cell kode
notebook `ML_Baseline_Aspek_Sentimen.ipynb`, bukan dari ringkasan chat),
dan "Ringkasan Hasil" (visualisasi khusus model tipe itu saja). Sekalian
ketauan & DIPERBAIKI bug di `_tag_text()`: notebook aslinya menempelkan tag
`[ASPEK_<NAMA>]` di AKHIR teks (`text.lower() + " [ASPEK_" + aspek + "]"`),
bukan di depan seperti draft pertama -- kalau tidak disamakan, TF-IDF n-gram
di sekitar tag jadi beda dari yang dipelajari model saat training.

Update 2026-08-21 (sore, setelah user kasih screenshot folder Drive):
notebook asli `05a_modeling_aspek.ipynb` & `05b_modeling_sentimen.ipynb`
TERNYATA ADA di Drive (sebelumnya dikira 05b tidak pernah diupload --
salah, sudah dicek ulang & didownload langsung). Ini mengoreksi 2 hal
penting yang sempat salah di draft sebelumnya:
1. Nama folder checkpoint final BUKAN `ckpt_<rasio>` (itu cuma output_dir
   sementara Trainer, dihapus `shutil.rmtree()` selesai training) --
   yang persis disimpan permanen (`trainer.save_model(...)`) adalah
   `model_rasio_<rasio>`. `DISTILBERT_ASPEK_CKPT` / `DISTILBERT_SENTIMEN_CKPT`
   di bawah sudah diperbaiki.
2. Format input sentimen DistilBERT BUKAN tag `[ASPEK_<NAMA>]` ala ML --
   notebook 05b pakai frasa alami per aspek digabung `" [SEP] "` + teks
   (lihat `FRASA_ASPEK` & `predict_sentimen_distilbert()`). Draft
   sebelumnya salah asumsi format ini sama dengan ML -- sudah diperbaiki.
DL_CONFIG di bawah sekarang isi hyperparameter ASLI (base model, LR,
epoch, batch size, dst), bukan lagi placeholder "belum tercatat".

Update 2026-08-21 (redesign ketiga, permintaan user): tabel konfigurasi di
halaman ML & DL dirapikan -- section "Dataset" (sumber file, nama kolom)
DIHAPUS total dari kedua halaman karena bukan konfigurasi training, cuma
bikin ramai. `ML_CONFIG`/`DL_CONFIG` diringkas jadi `ML_TRAINING_CONFIG`/
`DL_TRAINING_CONFIG` -- satu tabel flat per halaman, murni parameter
training, tanpa jargon kode. Sidebar juga diurutkan ulang: Dashboard
(ringkasan ML vs DL, dulu namanya halaman "ML vs DL") sekarang paling atas,
baru ML, DL, Live Prediksi.
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =====================================================================
# 1. KONFIGURASI & KONSTANTA
# =====================================================================

st.set_page_config(
    page_title="Dashboard Perbandingan ML vs DL",
    page_icon="\U0001F4CA",
    layout="wide",
    initial_sidebar_state="expanded",
)

ASPECTS = ["Individual", "Technical", "Social", "Financial"]
SENT_LABELS = ["positif", "negatif", "netral"]

ASPECT_AXIS_LABEL = {
    "Individual": "Individual",
    "Technical": "Technical",
    "Social": "Social",
    "Financial": "Financial",
}

# Threshold multi-label aspek DistilBERT, sesuai keputusan notebook 05a
# (threshold-tuning grid 0.30-0.85 tidak diadopsi karena tidak memenuhi
# kriteria perbaikan >0.005 F1 macro). Dipakai sama untuk aspek ML supaya
# comparable.
ASPECT_THRESHOLD = 0.5

# ---- Repo HF Hub tempat model di-upload (Fase A SELESAI 2026-08-24) ----
# Bisa di-override lewat environment variable dengan nama yang sama, supaya
# tidak perlu edit kode saat testing repo private/staging.
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "Jauzaa/dashboard-ml-vs-dl-models")
DISTILBERT_ASPEK_REPO = os.environ.get("DISTILBERT_ASPEK_REPO", "Jauzaa/distilbert-aspek-ta")
DISTILBERT_SENTIMEN_REPO = os.environ.get("DISTILBERT_SENTIMEN_REPO", "Jauzaa/distilbert-sentimen-ta")

# Nama file di dalam HF_MODEL_REPO (model ML + TF-IDF, hasil notebook
# ML_Baseline_Aspek_Sentimen.ipynb, folder Drive ML_Baseline_Output/)
ML_MODEL_FILES = {
    "aspek": {
        "Naive Bayes": "model_aspek_naive_bayes.joblib",
        "Random Forest": "model_aspek_random_forest.joblib",
        "Logistic Regression": "model_aspek_logistic_regression.joblib",
    },
    "sentimen": {
        "Naive Bayes": "model_sentimen_naive_bayes.joblib",
        "Random Forest": "model_sentimen_random_forest.joblib",
        "Logistic Regression": "model_sentimen_logistic_regression.joblib",
    },
}
TFIDF_FILES = {
    "aspek": "tfidf_aspek.joblib",
    "sentimen": "tfidf_sentimen.joblib",
}

# Sub-folder checkpoint DistilBERT di masing-masing repo. Nama PERSIS dari
# `tr.save_model(os.path.join(OUT_ROOT, f"model_rasio_{td}"))` di notebook
# 05a/05b -- BUKAN "ckpt_<rasio>" (itu output_dir sementara Trainer,
# dihapus otomatis abis training). Rasio dipilih berdasarkan F1 Weighted
# vs Gold tertinggi (kedua kriteria F1 macro & F1 weighted sepakat, lihat
# output cell "Rasio terbaik" di notebook):
#   aspek   -> 80:10:10 (F1 weighted GOLD 0.7098, berhenti epoch 7)
#   sentimen-> 60:20:20 (F1 weighted GOLD 0.8631, berhenti epoch 5)
DISTILBERT_ASPEK_CKPT = "model_rasio_80-10-10"
DISTILBERT_SENTIMEN_CKPT = "model_rasio_60-20-20"

# Base model HuggingFace yang di-fine-tune (persis dari MODEL_NAME di
# notebook 05a & 05b -- sama untuk aspek maupun sentimen)
DISTILBERT_BASE_MODEL = "cahya/distilbert-base-indonesian"

# Panjang token maksimum saat training (MAXLEN di notebook) -- dipakai lagi
# saat inference supaya konsisten dengan apa yang dipelajari model.
DISTILBERT_MAXLEN = 96

# Frasa natural per aspek untuk input model sentimen DistilBERT (persis
# dari dict FRASA_ASPEK di notebook 05b). BEDA dari format ML yang pakai
# tag [ASPEK_<NAMA>] -- lihat predict_sentimen_distilbert().
FRASA_ASPEK = {
    "Individual": "manfaat aplikasi bagi pengguna",
    "Technical": "kinerja teknis aplikasi",
    "Social": "fitur komunitas aplikasi",
    "Financial": "biaya atau langganan aplikasi",
}

MAX_KOMENTAR_CHARS = 500
MIN_KATA = 3

# ---- Konfigurasi training ML -- ringkas, cuma parameter training (bukan
# sumber data), dicek persis dari cell kode notebook
# ML_Baseline_Aspek_Sentimen.ipynb, dicek ulang 2026-08-21) ----
ML_TRAINING_CONFIG = {
    "Fitur teks": "TF-IDF — 10.000 fitur, 1–2 gram, minimal muncul 2x",
    "Split data (Aspek)": "80% latih / 10% validasi / 10% uji",
    "Split data (Sentimen)": "60% latih / 20% validasi / 20% uji (stratifikasi kelas)",
    "Threshold Aspek": "0,5 (sama dengan DistilBERT)",
    "Random Seed": "42 (sama dengan DistilBERT)",
    "Naive Bayes": "Multinomial, parameter default",
    "Random Forest": "200 pohon, kedalaman tanpa batas",
    "Logistic Regression": "maks. 1000 iterasi, regularisasi default",
}

# Catatan singkat kenapa Random Forest tidak dibatasi kedalamannya (relevan
# secara training, bukan soal data) -- ditampilkan sebagai info box, bukan
# baris tabel, supaya tabel utama tetap ringkas.
ML_RF_NOTE = (
    "Random Forest sempat dicoba dibatasi kedalamannya supaya file lebih kecil, tapi "
    "akurasi kelas minoritas (Financial) anjlok jauh -- jadi tetap dipakai tanpa batas "
    "kedalaman, walau ukuran filenya lebih besar."
)

# ---- Konfigurasi training DL -- ringkas, cuma parameter training, dicek
# persis dari cell kode notebook 05a_modeling_aspek.ipynb &
# 05b_modeling_sentimen.ipynb (didownload & dicek ulang 2026-08-21) ----
DL_TRAINING_CONFIG = {
    "Model dasar": "DistilBERT Bahasa Indonesia",
    "Learning rate": "5e-6",
    "Maksimum epoch": "12 (berhenti otomatis kalau 2 epoch tidak membaik)",
    "Batch size": "16",
    "Panjang teks maksimum": "96 token",
    "Jadwal learning rate": "Cosine, pemanasan 20% awal training",
    "Precision": "fp16",
    "Label smoothing": "0,1",
    "Random Seed": "42 (sama dengan ML)",
    "Split terpilih (Aspek)": "80% / 10% / 10% — berhenti di epoch 7",
    "Split terpilih (Sentimen)": "60% / 20% / 20% — berhenti di epoch 5",
    "Threshold Aspek": "0,5 (rentang lain sudah dicoba, tidak signifikan lebih baik)",
    "Penyeimbangan kelas": "Pembobotan otomatis untuk kelas minoritas",
}

# Breakdown F1 per-kelas sentimen DistilBERT vs Gold (378 pasangan) --
# classification_report cell 25 notebook 05b, TAHAP B (dilaporkan).
DISTILBERT_SENTIMEN_PER_KELAS = {
    "positif": {"f1": 0.94, "support": 292},
    "negatif": {"f1": 0.79, "support": 56},
    "netral": {"f1": 0.22, "support": 30},
}


# =====================================================================
# 2. HASIL STATIS (LOCKED, vs Gold Standard -- apple-to-apple dgn DistilBERT)
# =====================================================================
# Sumber: MyDrive/Revisi Skripsi/ML_Baseline_Output/hasil_perbandingan_ml_dl.json
# (diambil langsung dari Google Drive tanggal 2026-08-21, angka persis,
# JANGAN diketik ulang manual -- kalau model di-retrain, generate ulang
# JSON-nya dan copy-paste blok ini lagi).

STATIC_RESULTS = {
    "aspek": {
        "Naive Bayes": {
            "vs_gold": {
                "f1_weighted": 0.6643777367675524,
                "f1_macro": 0.5566737454288736,
                "f1_per_aspek": {
                    "Individual": 0.872870249017038,
                    "Technical": 0.5016949152542373,
                    "Social": 0.6176470588235294,
                    "Financial": 0.23448275862068965,
                },
            },
            "internal_test": {"f1_weighted": 0.8560894240451734, "f1_macro": 0.7150070097722196},
        },
        "Random Forest": {
            "vs_gold": {
                "f1_weighted": 0.6770048387911696,
                "f1_macro": 0.5896986330034888,
                "f1_per_aspek": {
                    "Individual": 0.8780487804878049,
                    "Technical": 0.4702194357366771,
                    "Social": 0.8,
                    "Financial": 0.21052631578947367,
                },
            },
            "internal_test": {"f1_weighted": 0.8883630120026422, "f1_macro": 0.7269037205369405},
        },
        "Logistic Regression": {
            "vs_gold": {
                "f1_weighted": 0.6712581013283877,
                "f1_macro": 0.5825629886271242,
                "f1_per_aspek": {
                    "Individual": 0.8703465982028241,
                    "Technical": 0.46835443037974683,
                    "Social": 0.725925925925926,
                    "Financial": 0.265625,
                },
            },
            "internal_test": {"f1_weighted": 0.8964260632618111, "f1_macro": 0.7603000235594257},
        },
        "DistilBERT (DL)": {
            "vs_gold": {
                "f1_weighted": 0.7098,
                "f1_macro": 0.6553,
                # Per-aspek DistilBERT TIDAK ada di hasil_perbandingan_ml_dl.json
                # (json itu cuma nyimpen weighted/macro) -- angka ini locked dari
                # skripsi, dikutip dari ringkasan handoff (Individual 0.874,
                # Technical 0.476, Social 0.864, Financial 0.407).
                "f1_per_aspek": {
                    "Individual": 0.874,
                    "Technical": 0.476,
                    "Social": 0.864,
                    "Financial": 0.407,
                },
            },
        },
    },
    "sentimen": {
        "Naive Bayes": {
            "vs_gold": {"f1_weighted": 0.844338999455668, "f1_macro": 0.5773570695856544},
            "internal_test": {"f1_weighted": 0.9020055368112194, "f1_macro": 0.5664090471984976},
        },
        "Random Forest": {
            "vs_gold": {"f1_weighted": 0.8265935570038294, "f1_macro": 0.5502348888192922},
            "internal_test": {"f1_weighted": 0.91663177680531, "f1_macro": 0.6299868911091104},
        },
        "Logistic Regression": {
            "vs_gold": {"f1_weighted": 0.8497697804762447, "f1_macro": 0.6084230458305628},
            "internal_test": {"f1_weighted": 0.9196292947979366, "f1_macro": 0.6394629430356444},
        },
        "DistilBERT (DL)": {
            "vs_gold": {"f1_weighted": 0.8631, "f1_macro": 0.6518},
        },
    },
}

ML_MODEL_NAMES = ["Naive Bayes", "Random Forest", "Logistic Regression"]
BEST_MODEL = {
    "aspek": {"vs_gold": "Random Forest"},
    "sentimen": {"vs_gold": "Logistic Regression"},
}

MODEL_COLORS = {
    "Naive Bayes": "#0e7490",
    "Random Forest": "#0891b2",
    "Logistic Regression": "#06b6d4",
    "DistilBERT (DL)": "#f97316",
}


# =====================================================================
# 3. UI HELPERS -- CSS, hero(), section(), kpi_card()
# =====================================================================

def inject_css():
    st.markdown(
        """
        <style>
        /* Paksa tema TERANG selalu, apapun preferensi dark-mode browser/OS
           viewer atau default tema Streamlit Cloud -- config.toml [theme]
           kadang ke-override oleh setting per-viewer, jadi dikunci lagi di
           sini supaya kontras teks/background selalu aman dibaca. */
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
        [data-testid="stMain"], .main {
            background-color: #ffffff !important;
        }
        [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
            background-color: #f0f9ff !important;
        }
        .stApp p:not(.hero *), .stApp span:not(.hero *):not(.badge),
        .stApp label:not(.hero *), .stApp div:not(.hero *):not(.kpi-card *),
        [data-testid="stMarkdownContainer"]:not(.hero *),
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
        [data-testid="stWidgetLabel"] {
            color: #0f172a !important;
        }
        [data-testid="stSidebar"] *:not(.hero *) {
            color: #0f172a !important;
            opacity: 1 !important;
        }
        h1:not(.hero h1), h2:not(.hero h2), h3:not(.hero h3),
        h4:not(.hero h4), h5:not(.hero h5), h6:not(.hero h6) {
            color: #0f172a !important;
        }
        [data-testid="stDataFrame"] {background-color: #ffffff !important;}

        /* st.caption() sengaja dibikin Streamlit redup/opacity rendah buat
           kesan "teks sekunder" -- di tema gelap itu masih kebaca (abu-abu
           muda di atas gelap), tapi di tema terang paksaan kita ini jadi
           abu-abu pucat di atas background terang = nyaris nggak kebaca.
           Dikunci lagi warna solid + opacity penuh. */
        [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *,
        .stCaption, .stCaption * {
            color: #475569 !important;
            opacity: 1 !important;
        }
        /* Label radio (menu sidebar) yang belum dipilih juga sering di-dim
           Streamlit -- kunci opacity penuh supaya semua opsi sama-sama jelas
           kebaca, bukan cuma yang lagi aktif. */
        [data-testid="stSidebar"] [role="radiogroup"] label,
        [data-testid="stSidebar"] [role="radiogroup"] label * {
            opacity: 1 !important;
        }

        /* Komponen custom (hero/badge/kpi) punya warna sendiri yang sudah
           dipilih buat kontras di atas background berwarna -- dikunci pakai
           !important supaya tidak ketiban aturan paksa-terang di atas. */
        .hero h1 {color: #ffffff !important;}
        .hero p, .hero-chip {color: #e0f2fe !important;}
        .kpi-value {color: #0e7490 !important;}
        .kpi-label {color: #475569 !important;}
        .kpi-sub {color: #94a3b8 !important;}
        .badge-ml {color: #0e7490 !important;}
        .badge-dl {color: #c2410c !important;}

        .block-container {padding-top: 2rem; padding-bottom: 3rem;}

        .hero {
            background: linear-gradient(135deg, #0e7490 0%, #0891b2 60%, #06b6d4 100%);
            border-radius: 18px;
            padding: 2.2rem 2.4rem;
            color: #ffffff;
            margin-bottom: 1.6rem;
        }
        .hero h1 {color: #ffffff; margin-bottom: 0.35rem; font-size: 1.9rem;}
        .hero p {color: #e0f2fe; margin: 0; font-size: 0.98rem;}
        .hero-chips {margin-top: 0.9rem; display: flex; gap: 0.5rem; flex-wrap: wrap;}
        .hero-chip {
            background: rgba(255,255,255,0.16);
            border: 1px solid rgba(255,255,255,0.35);
            border-radius: 999px;
            padding: 0.28rem 0.85rem;
            font-size: 0.83rem;
            color: #ffffff;
        }

        .section-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #0e7490;
            margin: 1.6rem 0 0.6rem 0;
            padding-bottom: 0.4rem;
            border-bottom: 2px solid #cffafe;
        }
        .section-desc {color: #475569; font-size: 0.92rem; margin-bottom: 0.9rem;}

        .kpi-card {
            background: #f0f9ff;
            border: 1px solid #cffafe;
            border-radius: 14px;
            padding: 1.1rem 1.2rem;
            text-align: center;
        }
        .kpi-value {font-size: 1.9rem; font-weight: 800; color: #0e7490; line-height: 1.1;}
        .kpi-label {font-size: 0.85rem; color: #475569; margin-top: 0.3rem;}
        .kpi-sub {font-size: 0.75rem; color: #94a3b8; margin-top: 0.15rem;}

        .badge {
            display: inline-block; border-radius: 999px; padding: 0.15rem 0.65rem;
            font-size: 0.78rem; font-weight: 600;
        }
        .badge-ml {background: #cffafe; color: #0e7490;}
        .badge-dl {background: #ffedd5; color: #c2410c;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, chips: list[str] | None = None):
    chips_html = ""
    if chips:
        chips_html = '<div class="hero-chips">' + "".join(
            f'<span class="hero-chip">{c}</span>' for c in chips
        ) + "</div>"
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
            {chips_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, desc: str = ""):
    desc_html = f'<div class="section-desc">{desc}</div>' if desc else ""
    st.markdown(
        f"""
        <div class="section-title">{title}</div>
        {desc_html}
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str = ""):
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def config_table(d: dict) -> pd.DataFrame:
    """Render dict key-value (termasuk nested dict satu level) jadi tabel
    2 kolom Parameter/Nilai -- dipakai untuk section "Dataset" & "Konfigurasi
    Model" di halaman ML dan DL supaya formatnya konsisten.
    """
    rows = []
    for k, v in d.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict):
            for k2, v2 in v.items():
                rows.append({"Parameter": f"{k} — {k2}", "Nilai": str(v2)})
        else:
            rows.append({"Parameter": k, "Nilai": str(v)})
    return pd.DataFrame(rows)


# =====================================================================
# 4. CHART BUILDERS (dipakai bersama oleh halaman ML, DL, dan ML vs DL)
# =====================================================================

def _bar_chart_weighted_macro(task: str, models: list[str]):
    weighted = [STATIC_RESULTS[task][m]["vs_gold"]["f1_weighted"] for m in models]
    macro = [STATIC_RESULTS[task][m]["vs_gold"]["f1_macro"] for m in models]

    fig = go.Figure()
    fig.add_bar(
        name="F1 Weighted",
        x=models,
        y=weighted,
        marker_color="#0e7490",
        text=[f"{v:.3f}" for v in weighted],
        textposition="outside",
    )
    fig.add_bar(
        name="F1 Macro",
        x=models,
        y=macro,
        marker_color="#f97316",
        text=[f"{v:.3f}" for v in macro],
        textposition="outside",
    )
    fig.update_layout(
        barmode="group",
        height=420,
        xaxis=dict(tickfont=dict(color="#0f172a")),
        yaxis=dict(
            title=dict(text="F1 Score (vs Gold Standard)", font=dict(color="#0f172a")),
            range=[0, 1],
            tickfont=dict(color="#0f172a"),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color="#0f172a"),
        ),
        margin=dict(t=40, b=10, l=10, r=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#0f172a"),
    )
    fig.update_traces(textfont=dict(color="#0f172a"))
    return fig


def _grouped_chart_per_aspek(models: list[str]):
    fig = go.Figure()
    for m in models:
        per_aspek = STATIC_RESULTS["aspek"][m]["vs_gold"]["f1_per_aspek"]
        fig.add_bar(
            name=m,
            x=ASPECTS,
            y=[per_aspek[a] for a in ASPECTS],
            marker_color=MODEL_COLORS[m],
            text=[f"{per_aspek[a]:.3f}" for a in ASPECTS],
            textposition="outside",
        )
    fig.update_layout(
        barmode="group",
        height=440,
        xaxis=dict(tickfont=dict(color="#0f172a")),
        yaxis=dict(
            title=dict(text="F1 (vs Gold Standard)", font=dict(color="#0f172a")),
            range=[0, 1],
            tickfont=dict(color="#0f172a"),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color="#0f172a"),
        ),
        margin=dict(t=40, b=10, l=10, r=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#0f172a"),
    )
    fig.update_traces(textfont=dict(color="#0f172a"))
    return fig


def _sentimen_per_kelas_chart():
    """F1 per-kelas sentimen DistilBERT vs Gold (data cuma ada utk DistilBERT --
    ML tidak generate classification_report per-kelas di notebook baseline)."""
    kelas = list(DISTILBERT_SENTIMEN_PER_KELAS.keys())
    f1 = [DISTILBERT_SENTIMEN_PER_KELAS[k]["f1"] for k in kelas]
    support = [DISTILBERT_SENTIMEN_PER_KELAS[k]["support"] for k in kelas]
    fig = go.Figure()
    fig.add_bar(
        x=kelas,
        y=f1,
        marker_color="#f97316",
        text=[f"{v:.2f} (n={s})" for v, s in zip(f1, support)],
        textposition="outside",
    )
    fig.update_layout(
        height=360,
        xaxis=dict(tickfont=dict(color="#0f172a")),
        yaxis=dict(
            title=dict(text="F1 (vs Gold Standard)", font=dict(color="#0f172a")),
            range=[0, 1],
            tickfont=dict(color="#0f172a"),
        ),
        margin=dict(t=20, b=10, l=10, r=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#0f172a"),
        showlegend=False,
    )
    fig.update_traces(textfont=dict(color="#0f172a"))
    return fig


def _full_table(task: str, models: list[str]) -> pd.DataFrame:
    rows = []
    for m in models:
        r = STATIC_RESULTS[task][m]
        row = {
            "Model": m,
            "Tipe": "DL" if m == "DistilBERT (DL)" else "ML",
            "F1 Weighted (vs Gold)": r["vs_gold"]["f1_weighted"],
            "F1 Macro (vs Gold)": r["vs_gold"]["f1_macro"],
        }
        if "internal_test" in r:
            row["F1 Weighted (test-split internal)"] = r["internal_test"]["f1_weighted"]
            row["F1 Macro (test-split internal)"] = r["internal_test"]["f1_macro"]
        else:
            row["F1 Weighted (test-split internal)"] = None
            row["F1 Macro (test-split internal)"] = None
        rows.append(row)
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c.startswith("F1"):
            df[c] = df[c].apply(lambda v: f"{v:.4f}" if pd.notna(v) else "— (locked dari skripsi, n/a)")
    return df


# =====================================================================
# 5. HALAMAN -- Machine Learning (ML) / Deep Learning (DL) / ML vs DL
# =====================================================================

def halaman_ml():
    hero(
        "Machine Learning (ML)",
        "Naive Bayes, Random Forest, Logistic Regression -- fitur TF-IDF. Dataset sama "
        "dengan dashboard TA, dievaluasi terhadap Gold Standard yang sama dengan DistilBERT.",
        chips=["\U0001F916 3 Model ML", "\U0001F4D0 Fitur: TF-IDF", "\U0001F3AF Evaluasi: vs Gold Standard"],
    )

    section("Konfigurasi Training")
    st.dataframe(config_table(ML_TRAINING_CONFIG), width="stretch", hide_index=True)
    st.info(ML_RF_NOTE, icon="\U0001F333")

    section("Ringkasan Hasil", "Angka locked, vs Gold Standard -- 3 model ML saja (lihat halaman \"Dashboard\" untuk perbandingan dengan DistilBERT).")
    tab_aspek, tab_sentimen = st.tabs(["\U0001F4CB Aspek", "\U0001F60A Sentimen"])
    for tab, task, task_label in [
        (tab_aspek, "aspek", "Aspek (multi-label, 4 kelas)"),
        (tab_sentimen, "sentimen", "Sentimen (multi-class, 3 kelas)"),
    ]:
        with tab:
            best = BEST_MODEL[task]["vs_gold"]
            best_score = STATIC_RESULTS[task][best]["vs_gold"]["f1_weighted"]
            best_macro = STATIC_RESULTS[task][best]["vs_gold"]["f1_macro"]

            c1, c2, c3 = st.columns(3)
            with c1:
                kpi_card("Model Terbaik", best, "berdasarkan F1 Weighted vs Gold")
            with c2:
                kpi_card("F1 Weighted", f"{best_score:.4f}", best)
            with c3:
                kpi_card("F1 Macro", f"{best_macro:.4f}", best)

            st.plotly_chart(_bar_chart_weighted_macro(task, ML_MODEL_NAMES), width="stretch", theme=None)

            if task == "aspek":
                section(
                    "Breakdown per Aspek",
                    "Financial adalah kelas minoritas (2.912 sampel di data training) -- F1 rendah di semua model ML.",
                )
                st.plotly_chart(_grouped_chart_per_aspek(ML_MODEL_NAMES), width="stretch", theme=None)

            st.dataframe(_full_table(task, ML_MODEL_NAMES), width="stretch", hide_index=True)


def halaman_dl():
    hero(
        "Deep Learning (DL)",
        "DistilBERT -- locked dari TA, TIDAK di-retrain di sini. Dataset sama dengan "
        "dashboard TA, dievaluasi terhadap Gold Standard yang sama dengan ML.",
        chips=["\U0001F9E0 DistilBERT", "\U0001F512 Locked dari TA", "\U0001F3AF Evaluasi: vs Gold Standard"],
    )

    section("Konfigurasi Training")
    st.dataframe(config_table(DL_TRAINING_CONFIG), width="stretch", hide_index=True)

    section("Ringkasan Hasil", "Angka locked dari skripsi, vs Gold Standard.")
    tab_aspek, tab_sentimen = st.tabs(["\U0001F4CB Aspek", "\U0001F60A Sentimen"])
    with tab_aspek:
        d = STATIC_RESULTS["aspek"]["DistilBERT (DL)"]["vs_gold"]
        c1, c2 = st.columns(2)
        with c1:
            kpi_card("F1 Weighted", f"{d['f1_weighted']:.4f}", "vs Gold Standard")
        with c2:
            kpi_card("F1 Macro", f"{d['f1_macro']:.4f}", "vs Gold Standard")
        st.plotly_chart(_grouped_chart_per_aspek(["DistilBERT (DL)"]), width="stretch", theme=None)
    with tab_sentimen:
        d = STATIC_RESULTS["sentimen"]["DistilBERT (DL)"]["vs_gold"]
        c1, c2 = st.columns(2)
        with c1:
            kpi_card("F1 Weighted", f"{d['f1_weighted']:.4f}", "vs Gold Standard")
        with c2:
            kpi_card("F1 Macro", f"{d['f1_macro']:.4f}", "vs Gold Standard")
        st.plotly_chart(_sentimen_per_kelas_chart(), width="stretch", theme=None)
        st.caption("Breakdown per-kelas dari classification_report notebook 05b (TAHAP B, vs 378 pasangan Gold).")


def halaman_dashboard():
    """Halaman pertama yang dilihat user -- ringkasan perbandingan ML vs DL
    (dulu namanya halaman "ML vs DL", di-redesign jadi menu "Dashboard" &
    dipindah ke urutan pertama sidebar atas permintaan user)."""
    hero(
        "Dashboard -- Ringkasan Perbandingan ML vs DL",
        "Naive Bayes, Random Forest, Logistic Regression (TF-IDF) dibandingkan dengan "
        "DistilBERT -- dievaluasi terhadap Gold Standard yang sama supaya apple-to-apple. "
        "Dataset sama dengan dashboard TA.",
        chips=[
            "\U0001F4C4 Task: Aspek & Sentimen",
            "\U0001F3AF Evaluasi: vs Gold Standard",
            "⚖️ F1 Weighted (primer) + F1 Macro",
        ],
    )

    all_models = ML_MODEL_NAMES + ["DistilBERT (DL)"]
    tab_aspek, tab_sentimen = st.tabs(["\U0001F4CB Aspek", "\U0001F60A Sentimen"])

    for tab, task, task_label in [
        (tab_aspek, "aspek", "Aspek (multi-label, 4 kelas)"),
        (tab_sentimen, "sentimen", "Sentimen (multi-class, 3 kelas)"),
    ]:
        with tab:
            best = BEST_MODEL[task]["vs_gold"]
            best_score = STATIC_RESULTS[task][best]["vs_gold"]["f1_weighted"]
            dl_score = STATIC_RESULTS[task]["DistilBERT (DL)"]["vs_gold"]["f1_weighted"]
            gap = dl_score - best_score

            section(f"Ringkasan -- {task_label}")
            c1, c2, c3 = st.columns(3)
            with c1:
                kpi_card("Model ML Terbaik", best, f"F1 Weighted vs Gold: {best_score:.4f}")
            with c2:
                kpi_card("DistilBERT (DL)", f"{dl_score:.4f}", "F1 Weighted vs Gold")
            with c3:
                kpi_card("Selisih DL vs ML terbaik", f"+{gap:.4f}", "DistilBERT masih unggul" if gap > 0 else "ML menyusul DL")

            section("F1 Weighted & Macro per Model")
            st.plotly_chart(_bar_chart_weighted_macro(task, all_models), width="stretch", theme=None)

            if task == "aspek":
                section(
                    "Breakdown per Aspek",
                    "Financial adalah kelas minoritas (2.912 sampel di data training) -- F1 rendah "
                    "di semua model, ML maupun DL, konsisten dengan temuan skripsi.",
                )
                st.plotly_chart(_grouped_chart_per_aspek(all_models), width="stretch", theme=None)

            section(
                "Tabel Lengkap",
                "'vs Gold' = metrik utama (apple-to-apple dengan DistilBERT). "
                "'test-split internal' = evaluasi terhadap split internal masing-masing model ML "
                "(bukan angka pembanding utama, hanya konteks tambahan).",
            )
            st.dataframe(_full_table(task, all_models), width="stretch", hide_index=True)


# =====================================================================
# 6. HALAMAN LIVE PREDICTION -- ML vs DistilBERT side-by-side
# =====================================================================
# CATATAN STATUS (2026-08-24): Fase A selesai -- model sudah diupload user
# ke HF Hub via notebook Colab, HF_MODEL_REPO / DISTILBERT_ASPEK_REPO /
# DISTILBERT_SENTIMEN_REPO di section 1 sudah diisi repo asli (bukan
# placeholder lagi). Sudah dicek lewat HF Hub API (bukan download penuh,
# sandbox Cowork tidak bisa akses huggingface.co) bahwa isi ketiga repo
# sesuai yang diharapkan (nama file .joblib persis cocok ML_MODEL_FILES/
# TFIDF_FILES, folder model_rasio_80-10-10 & model_rasio_60-20-20 ada).
# Belum sempat dites RUN END-TO-END beneran (perlu jalan di lingkungan yang
# bisa akses HF Hub, misal HF Space nanti di Fase D) -- tapi satu bug
# krusial sempat ketemu & diperbaiki di load_distilbert_models() di bawah:
# tokenizer HARUS di-load dari base model, bukan dari folder checkpoint
# (checkpoint tidak punya file tokenizer sama sekali, lihat komentar di
# fungsinya).

@st.cache_resource(show_spinner="Mengunduh & memuat model ML...")
def load_ml_models():
    """Download (via huggingface_hub) & load 3 model ML + 2 TF-IDF vectorizer.

    Di-cache st.cache_resource supaya cuma didownload sekali per proses
    server, bukan setiap ada request prediksi.
    """
    from huggingface_hub import hf_hub_download
    import joblib

    models = {"aspek": {}, "sentimen": {}}
    vectorizers = {}

    for task, files in ML_MODEL_FILES.items():
        for model_name, filename in files.items():
            path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=filename)
            models[task][model_name] = joblib.load(path)

    for task, filename in TFIDF_FILES.items():
        path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=filename)
        vectorizers[task] = joblib.load(path)

    return models, vectorizers


@st.cache_resource(show_spinner="Mengunduh & memuat DistilBERT...")
def load_distilbert_models():
    """Download 2 checkpoint DistilBERT (aspek + sentimen) langsung dari HF
    Hub -- BUKAN lewat XNLI (beda dari dashboard TA yang sekarang pakai
    XNLI buat sentimen; di dashboard ini DistilBERT dipakai apa adanya
    sebagai pembanding DL, sesuai scope yang disepakati).
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from huggingface_hub import snapshot_download

    aspek_dir = snapshot_download(repo_id=DISTILBERT_ASPEK_REPO, allow_patterns=[f"{DISTILBERT_ASPEK_CKPT}/*"])
    sentimen_dir = snapshot_download(repo_id=DISTILBERT_SENTIMEN_REPO, allow_patterns=[f"{DISTILBERT_SENTIMEN_CKPT}/*"])

    aspek_ckpt = os.path.join(aspek_dir, DISTILBERT_ASPEK_CKPT)
    sentimen_ckpt = os.path.join(sentimen_dir, DISTILBERT_SENTIMEN_CKPT)

    # PENTING: tokenizer di-load dari base model (DISTILBERT_BASE_MODEL), BUKAN dari
    # folder checkpoint. Di notebook training (05a/05b cell 12), Trainer() dibuat TANPA
    # argumen tokenizer=..., jadi tr.save_model() cuma nyimpen config.json +
    # model.safetensors + training_args.bin -- tidak ada tokenizer.json/vocab.txt di
    # checkpoint. Notebook aslinya sendiri juga load tokenizer dari base model saat
    # scoring (cell 31: `tok_skor = AutoTokenizer.from_pretrained(MODEL_NAME)`), jadi ini
    # konsisten, bukan workaround. Tokenizer-nya sendiri tidak berubah selama fine-tuning
    # (tidak ada token baru ditambahkan), jadi aman dipakai apa adanya dari base model.
    tok_aspek = AutoTokenizer.from_pretrained(DISTILBERT_BASE_MODEL)
    model_aspek = AutoModelForSequenceClassification.from_pretrained(aspek_ckpt)
    model_aspek.eval()

    tok_sentimen = AutoTokenizer.from_pretrained(DISTILBERT_BASE_MODEL)
    model_sentimen = AutoModelForSequenceClassification.from_pretrained(sentimen_ckpt)
    model_sentimen.eval()

    return {
        "aspek": (tok_aspek, model_aspek),
        "sentimen": (tok_sentimen, model_sentimen),
    }


def _tag_text(text: str, aspek: str) -> str:
    """Format input untuk model SENTIMEN ML: teks + tag [ASPEK_<NAMA>] di
    AKHIR, persis formula notebook ML_Baseline_Aspek_Sentimen.ipynb (cell 18):
        df_sent["text_input"] = text.lower() + " [ASPEK_" + aspect.upper() + "]"
    Awalnya draft ini nulis tag di DEPAN teks -- ketauan salah saat
    ngecek ulang cell training, sudah diperbaiki di sini (2026-08-21) supaya
    match persis dengan apa yang dipelajari model ML saat training.

    HANYA dipakai untuk model ML -- DistilBERT sentimen pakai format
    BEDA sama sekali (frasa natural + " [SEP] " + teks, lihat
    `_format_sentimen_distilbert()`), dikonfirmasi dari notebook 05b yang
    ternyata ADA di Drive (sempat dikira tidak pernah diupload).
    """
    return f"{text.lower()} [ASPEK_{aspek.upper()}]"


def _format_sentimen_distilbert(text: str, aspek: str) -> str:
    """Format input untuk model SENTIMEN DistilBERT: frasa natural per aspek
    + " [SEP] " + teks (lowercase), persis formula notebook
    05b_modeling_sentimen.ipynb (cell 10, fungsi `bentuk()`):
        fr = aspect.map(FRASA_ASPEK); tx = bert_text.lower()
        return fr + " [SEP] " + tx
    BEDA dari format ML (`_tag_text`) -- draft sebelumnya salah asumsi
    keduanya sama, sudah diperbaiki (2026-08-21) setelah notebook 05b
    ketemu & dicek langsung.
    """
    return f"{FRASA_ASPEK[aspek]} [SEP] {text.lower()}"


def predict_aspek_ml(text: str, models: dict, vectorizer) -> dict:
    """Multi-label: tiap model punya OneVsRestClassifier (lihat notebook,
    cell import `sklearn.multiclass.OneVsRestClassifier`). Ambil
    predict_proba per kelas, threshold 0.5.
    """
    X = vectorizer.transform([text])
    hasil = {}
    for model_name, clf in models.items():
        try:
            proba = clf.predict_proba(X)
            # OneVsRestClassifier.predict_proba -> shape (1, n_classes)
            scores = np.asarray(proba)[0]
        except AttributeError:
            # fallback kalau model tidak expose predict_proba
            scores = np.asarray(clf.predict(X))[0].astype(float)
        hasil[model_name] = {
            aspek: {
                "score": float(scores[i]),
                "label": bool(scores[i] >= ASPECT_THRESHOLD),
            }
            for i, aspek in enumerate(ASPECTS)
        }
    return hasil


def predict_sentimen_ml(text: str, aspek_terdeteksi: list[str], models: dict, vectorizer) -> dict:
    """Untuk tiap aspek yang lolos threshold di prediksi aspek, prediksi
    sentimen aspek-specific dengan input teks+tag [ASPEK_<NAMA>].
    """
    hasil = {}
    for model_name, clf in models.items():
        hasil[model_name] = {}
        for aspek in aspek_terdeteksi:
            tagged = _tag_text(text, aspek)
            X = vectorizer.transform([tagged])
            pred = clf.predict(X)[0]
            label = pred if isinstance(pred, str) else SENT_LABELS[int(pred)]
            try:
                proba = clf.predict_proba(X)[0]
                conf = float(np.max(proba))
            except AttributeError:
                conf = None
            hasil[model_name][aspek] = {"label": label, "confidence": conf}
    return hasil


def predict_aspek_distilbert(text: str, tok_model) -> dict:
    """Input di-lowercase dulu (sama seperti X_all di notebook 05a:
    `data_awal["bert_text"].astype(str).str.lower()`), max_length=96
    (MAXLEN training) -- keduanya sempat salah di draft pertama (belum
    lowercase, max_length=256)."""
    import torch

    tok, model = tok_model
    inputs = tok(text.lower(), return_tensors="pt", truncation=True, padding="max_length", max_length=DISTILBERT_MAXLEN)
    with torch.inference_mode():
        logits = model(**inputs).logits
        scores = torch.sigmoid(logits)[0].tolist()
    return {
        aspek: {"score": float(scores[i]), "label": scores[i] >= ASPECT_THRESHOLD}
        for i, aspek in enumerate(ASPECTS)
    }


def predict_sentimen_distilbert(text: str, aspek_terdeteksi: list[str], tok_model) -> dict:
    """Pakai `_format_sentimen_distilbert()` (frasa + [SEP] + teks), BUKAN
    `_tag_text()` ala ML -- lihat docstring masing-masing fungsi. max_length=96
    sama seperti training (MAXLEN)."""
    import torch

    tok, model = tok_model
    hasil = {}
    for aspek in aspek_terdeteksi:
        formatted = _format_sentimen_distilbert(text, aspek)
        inputs = tok(formatted, return_tensors="pt", truncation=True, padding="max_length", max_length=DISTILBERT_MAXLEN)
        with torch.inference_mode():
            logits = model(**inputs).logits
            proba = torch.softmax(logits, dim=-1)[0].tolist()
        idx = int(np.argmax(proba))
        hasil[aspek] = {"label": SENT_LABELS[idx], "confidence": float(proba[idx])}
    return hasil


def _validasi_komentar(text: str) -> str | None:
    text = text.strip()
    if not text:
        return "Komentar tidak boleh kosong."
    if len(text) > MAX_KOMENTAR_CHARS:
        return f"Komentar maksimal {MAX_KOMENTAR_CHARS} karakter (sekarang {len(text)})."
    if len(text.split()) < MIN_KATA:
        return f"Komentar minimal {MIN_KATA} kata."
    return None


def _render_hasil_side_by_side(text: str, ml_models, ml_vectorizers, dl_models, ml_pilihan: str):
    aspek_ml = predict_aspek_ml(text, ml_models["aspek"], ml_vectorizers["aspek"])
    aspek_dl = predict_aspek_distilbert(text, dl_models["aspek"])

    aspek_terdeteksi_ml = [a for a, v in aspek_ml[ml_pilihan].items() if v["label"]]
    aspek_terdeteksi_dl = [a for a, v in aspek_dl.items() if v["label"]]
    aspek_gabungan = sorted(set(aspek_terdeteksi_ml) | set(aspek_terdeteksi_dl))

    sentimen_ml = predict_sentimen_ml(text, aspek_gabungan, ml_models["sentimen"], ml_vectorizers["sentimen"])
    sentimen_dl = predict_sentimen_distilbert(text, aspek_gabungan, dl_models["sentimen"])

    col_ml, col_dl = st.columns(2)

    with col_ml:
        st.markdown(f'<span class="badge badge-ml">ML — {ml_pilihan}</span>', unsafe_allow_html=True)
        st.write("**Skor Aspek**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Aspek": ASPECTS,
                    "Skor": [f"{aspek_ml[ml_pilihan][a]['score']:.3f}" for a in ASPECTS],
                    "Terdeteksi": ["✅" if aspek_ml[ml_pilihan][a]["label"] else "" for a in ASPECTS],
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.write("**Sentimen per Aspek**")
        if aspek_gabungan:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Aspek": a,
                            "Sentimen": sentimen_ml[ml_pilihan][a]["label"],
                            "Confidence": (
                                f"{sentimen_ml[ml_pilihan][a]['confidence']:.3f}"
                                if sentimen_ml[ml_pilihan][a]["confidence"] is not None
                                else "—"
                            ),
                        }
                        for a in aspek_gabungan
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("Tidak ada aspek yang terdeteksi (skor semua di bawah threshold 0.5).")

    with col_dl:
        st.markdown('<span class="badge badge-dl">DistilBERT (DL)</span>', unsafe_allow_html=True)
        st.write("**Skor Aspek**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Aspek": ASPECTS,
                    "Skor": [f"{aspek_dl[a]['score']:.3f}" for a in ASPECTS],
                    "Terdeteksi": ["✅" if aspek_dl[a]["label"] else "" for a in ASPECTS],
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.write("**Sentimen per Aspek**")
        if aspek_gabungan:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Aspek": a,
                            "Sentimen": sentimen_dl[a]["label"],
                            "Confidence": f"{sentimen_dl[a]['confidence']:.3f}",
                        }
                        for a in aspek_gabungan
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("Tidak ada aspek yang terdeteksi (skor semua di bawah threshold 0.5).")


def halaman_prediksi_live():
    hero(
        "Prediksi Live -- ML vs DistilBERT",
        "Ketik satu komentar, hasil ML dan DistilBERT tampil side-by-side untuk komentar yang sama. "
        "Hasil murni output model (tanpa nilai Gold Standard skripsi dicampur di sini).",
        chips=["\U0001F916 3 Model ML (pilih salah satu)", "\U0001F9E0 DistilBERT (DL)", "⚡ Real-time"],
    )

    if HF_MODEL_REPO.startswith("USERNAME/") or DISTILBERT_ASPEK_REPO.startswith("USERNAME/"):
        st.warning(
            "Model belum diupload ke HF Hub -- `HF_MODEL_REPO`, `DISTILBERT_ASPEK_REPO`, "
            "`DISTILBERT_SENTIMEN_REPO` di section 1 `app.py` masih placeholder (\"USERNAME/...\"). "
            "Halaman ini akan error kalau dicoba sekarang. Selesaikan Fase A (upload ke HF Hub) dulu, "
            "lalu ganti 3 baris itu -- tidak perlu ubah kode lain.",
            icon="⚠️",
        )
        st.stop()

    section("Input Komentar")
    ml_pilihan = st.selectbox("Model ML yang dibandingkan", ML_MODEL_NAMES, index=2)
    text = st.text_area(
        "Komentar",
        max_chars=MAX_KOMENTAR_CHARS,
        placeholder="Tulis komentar di sini (maks 500 karakter, min 3 kata)...",
        height=120,
    )

    if st.button("Prediksi", type="primary"):
        error = _validasi_komentar(text)
        if error:
            st.error(error)
            return

        with st.spinner("Memuat model (pertama kali bisa agak lama)..."):
            ml_models, ml_vectorizers = load_ml_models()
            dl_models = load_distilbert_models()

        section("Hasil Prediksi")
        with st.spinner("Memproses prediksi ML & DistilBERT..."):
            _render_hasil_side_by_side(text.strip(), ml_models, ml_vectorizers, dl_models, ml_pilihan)


# =====================================================================
# 7. ROUTING UTAMA
# =====================================================================

# Urutan sesuai permintaan user: Dashboard (ringkasan perbandingan) duluan,
# baru ML, DL, Live Prediksi -- Python dict mempertahankan urutan insert,
# jadi urutan di sini = urutan tampil di sidebar.
MENU_ITEMS = {
    "\U0001F4CA Dashboard": halaman_dashboard,
    "\U0001F916 Machine Learning (ML)": halaman_ml,
    "\U0001F9E0 Deep Learning (DL)": halaman_dl,
    "\U0001F4AC Live Prediksi": halaman_prediksi_live,
}


def main():
    inject_css()

    st.sidebar.markdown("### \U0001F4CA Dashboard ML vs DL")
    st.sidebar.caption("Terpisah dari dashboard TA utama -- dataset sama.")
    menu = st.sidebar.radio(
        "Menu",
        list(MENU_ITEMS.keys()),
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Model ML: Naive Bayes, Random Forest, Logistic Regression (TF-IDF).\n\n"
        "Model DL: DistilBERT (locked dari TA, tidak di-retrain).\n\n"
        "Evaluasi headline: vs Gold Standard (bukan test-split biasa)."
    )

    MENU_ITEMS[menu]()


if __name__ == "__main__":
    main()
