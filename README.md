# Dashboard Perbandingan ML vs DL

Dashboard tambahan (terpisah dari dashboard TA utama) yang membandingkan model Machine
Learning (Naive Bayes, Random Forest, Logistic Regression -- fitur TF-IDF) dengan Deep
Learning (DistilBERT, locked dari TA, tidak di-retrain), untuk task Aspek (multi-label, 4
kelas) dan Sentimen (multi-class, 3 kelas). Dataset sama dengan dashboard TA.

## Menjalankan lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```
