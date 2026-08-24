# Dashboard Perbandingan ML vs DL

Dashboard tambahan (terpisah dari dashboard TA utama) yang membandingkan model Machine
Learning (Naive Bayes, Random Forest, Logistic Regression -- fitur TF-IDF) dengan Deep
Learning (DistilBERT, locked dari TA, tidak di-retrain), untuk task Aspek (multi-label, 4
kelas) dan Sentimen (multi-class, 3 kelas). Dataset sama dengan dashboard TA.

Ditulis ulang dari awal 2026-08-21, lalu di-redesign 2x hari yang sama: pertama jadi 4
halaman terpisah di sidebar dengan konfigurasi ML dan DL dicek ulang persis dari notebook
asli (bukan ringkasan chat), lalu dirapikan lagi -- tabel konfigurasi disederhanakan
(cuma parameter training, tanpa detail sumber data) dan sidebar diurutkan ulang. Semua
kode ada di satu file `app.py` (7 section, ditandai komentar nomor).

## Sidebar (4 menu, urutan sesuai permintaan user)

1. **Dashboard** -- ringkasan perbandingan ML vs DL (dulu namanya halaman "ML vs DL"),
   4 model sekaligus. Halaman pertama yang dilihat user.
2. **Machine Learning (ML)** -- tabel konfigurasi training (TF-IDF, split, hyperparameter
   per model), ringkasan hasil 3 model ML saja.
3. **Deep Learning (DL)** -- tabel konfigurasi training DistilBERT (base model, LR,
   epoch, batch size), ringkasan hasil DistilBERT saja (termasuk breakdown F1 per-kelas
   sentimen).
4. **Live Prediksi** -- prediksi real-time, ML vs DistilBERT side-by-side.

Tabel konfigurasi di halaman ML dan DL sengaja HANYA berisi parameter training (split,
hyperparameter, threshold, dst) -- detail sumber data (nama file, path, nama kolom) TIDAK
ditampilkan lagi di tabel, supaya ringkas dan gampang dibaca.

## Status

- **Halaman Dashboard, ML, DL**: SELESAI & sudah dites jalan.
- **Halaman "Live Prediksi"**: SELESAI & sudah dites end-to-end di lokal (2026-08-24) --
  model ML + DistilBERT diambil dari HF Hub (`Jauzaa/dashboard-ml-vs-dl-models`,
  `Jauzaa/distilbert-aspek-ta`, `Jauzaa/distilbert-sentimen-ta`, ketiganya public),
  hasil prediksi ML vs DistilBERT tampil bener side-by-side.
- **Deploy online**: SEDANG JALAN, lihat "Next Steps".

## Menjalankan lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

Semua halaman termasuk "Live Prediksi" sudah bisa langsung jalan (model sudah di HF Hub,
bukan placeholder lagi).

## Next Steps

1. ~~**Fase A -- Upload model ke HF Hub**~~ **SELESAI (2026-08-24)**. 3 repo public:
   `Jauzaa/dashboard-ml-vs-dl-models` (6 model ML + 2 TF-IDF vectorizer),
   `Jauzaa/distilbert-aspek-ta` (folder `model_rasio_80-10-10`),
   `Jauzaa/distilbert-sentimen-ta` (folder `model_rasio_60-20-20`). Terverifikasi jalan
   end-to-end di lokal.
2. ~~**Fase B -- Bikin HF Space**~~ **DIBATALKAN (2026-08-24)**. Rencana awal pakai HF
   Space, tapi ternyata: (a) SDK Streamlit bawaan HF sudah dideprecate (harus lewat SDK
   Docker + Dockerfile), DAN (b) Docker/Gradio Space di HF sekarang butuh langganan PRO
   ($9/bulan) meski hardware-nya gratis -- cuma Static Space yang masih gratis. User
   pilih pindah ke **Streamlit Community Cloud** (gratis, resmi dari Streamlit, dibuat
   khusus buat app Streamlit) sebagai gantinya -- lihat Fase B baru di bawah.
3. **Fase B (baru) -- Deploy ke Streamlit Community Cloud**:
   a. Push project ini (`app.py`, `requirements.txt`, `.streamlit/config.toml`) ke
      repo GitHub baru (public) -- lewat GitHub web UI ("Add file -> Upload files"),
      tidak perlu git/command line.
   b. Buka [share.streamlit.io](https://share.streamlit.io), login pakai akun GitHub,
      klik "New app", pilih repo & branch, "Main file path" isi `app.py`, Deploy.
   c. Streamlit Cloud otomatis install `requirements.txt` & jalanin app -- tidak butuh
      Dockerfile sama sekali (sudah dihapus dari project ini).
4. **Fase C -- Testing** end-to-end di URL Streamlit Cloud (bukan cuma lokal).
5. **Fase D -- Kirim**: upload ke OneDrive + link Streamlit Cloud, kirim ke WA dospem.

## Catatan teknis penting

- `scikit-learn` di-pin range `>=1.5,<1.7` -- versi training persis tidak tercatat
  eksplisit di notebook ML, tapi ada `FutureWarning` di cell Logistic Regression yang
  memastikan rentang ini (`'multi_class' was deprecated in version 1.5 and will be
  removed in 1.7`). Kompatibel dengan Python 3.10 lokal.
- `torch` pakai `--extra-index-url https://download.pytorch.org/whl/cpu` -- JANGAN
  dihapus (target deploy CPU-only, cegah pip narik wheel CUDA ~2GB+ yang tidak kepake).
- **Nama folder checkpoint DistilBERT final adalah `model_rasio_<rasio>`**, BUKAN
  `ckpt_<rasio>` -- `ckpt_*` cuma `output_dir` sementara `Trainer` yang dihapus otomatis
  (`shutil.rmtree`) begitu training selesai.
- Threshold aspek = 0.5 flat untuk ML maupun DistilBERT -- grid tuning
  `np.arange(0.30, 0.85, 0.02)` sempat dicoba di notebook 05a, delta F1 macro cuma
  +0.0005 (ambang keputusan >0.005) sehingga DITOLAK, 0.5 tetap dipakai.
- **Format input sentimen BEDA antara ML dan DistilBERT**:
  - ML: teks (lowercase) + tag di AKHIR, `text.lower() + " [ASPEK_" + aspek.upper() + "]"`
    (persis cell 18 `ML_Baseline_Aspek_Sentimen.ipynb`).
  - DistilBERT: frasa natural per aspek + `" [SEP] "` + teks (lowercase), misal aspek
    Technical -> `"kinerja teknis aplikasi [SEP] <komentar>"` (persis fungsi `bentuk()`
    cell 10 `05b_modeling_sentimen.ipynb`, lihat `FRASA_ASPEK` di section 1 `app.py`).
- Base model DistilBERT: `cahya/distilbert-base-indonesian`. Hyperparameter training:
  `learning_rate=5e-6, num_train_epochs<=12 (early stopping patience=2), batch_size=16,
  max_length=96, weight_decay=0.01, warmup_ratio=0.2, lr_scheduler='cosine', fp16=True,
  label_smoothing=0.1, seed=42` -- sama persis untuk aspek & sentimen (beda cuma jumlah
  label & fungsi loss: BCEWithLogitsLoss+pos_weight untuk aspek multi-label,
  CrossEntropyLoss+class_weight untuk sentimen multi-class).
- DistilBERT di sini dipakai APA ADANYA sebagai pembanding DL (load langsung 2
  checkpoint) -- BUKAN lewat XNLI, beda dari dashboard TA yang sekarang pakai XNLI
  untuk sentimen.
- **Tokenizer DistilBERT di-load dari base model (`DISTILBERT_BASE_MODEL`), BUKAN dari
  folder checkpoint** -- checkpoint HF Hub (`model_rasio_80-10-10`/`model_rasio_60-20-20`)
  cuma berisi `config.json`, `model.safetensors`, `training_args.bin` (tidak ada file
  tokenizer sama sekali), karena `Trainer` di notebook training dibuat tanpa argumen
  `tokenizer=...` sehingga `tr.save_model()` tidak pernah menyimpan tokenizer. Ini
  konsisten dengan cara notebook asli sendiri melakukan scoring (cell 31 05a/05b: load
  tokenizer dari `MODEL_NAME`, bukan dari `model_path`) -- bukan workaround, dan aman
  karena tokenizer tidak berubah selama fine-tuning classification head.
- **HF Space DIBATALKAN sebagai target deploy** -- SDK Streamlit bawaan HF dideprecate
  (wajib Docker sekarang), DAN Docker/Gradio Space butuh PRO $9/bulan meski hardware
  gratis (cuma Static Space yang gratis). Pindah ke Streamlit Community Cloud (gratis).
  `upload_script/FaseA_Upload_ke_HF_Hub.ipynb` tetap relevan (model masih di HF Hub, itu
  gratis). `upload_script/FaseB_Buat_Space_dan_Push.ipynb` sudah TIDAK DIPAKAI lagi
  (dibiarkan ada sebagai riwayat, bukan langkah aktif).
