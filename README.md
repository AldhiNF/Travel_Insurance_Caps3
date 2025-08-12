# **Business Understanding (Pemahaman Bisnis)**
Capstone Project – Travel Insurance Claim Prediction


## **Business Problem Understanding**
Perusahaan asuransi perjalanan menanggung risiko klaim tak terduga (darurat medis, pembatalan, bagasi hilang). Mayoritas nasabah **tidak** melakukan klaim, namun sebagian kecil yang melakukan klaim bisa menimbulkan kerugian besar. Tujuan proyek ini adalah memahami pola yang membedakan nasabah yang mengajukan klaim dan membangun model prediksi untuk mengelola risiko.


## **Business Problem Statement**
Perusahaan kesulitan memprediksi nasabah yang akan klaim pada data yang sangat tidak seimbang (minoritas klaim). Kegagalan mengidentifikasi calon pengklaim meningkatkan eksposur biaya dan mengganggu arus kas. Kita membutuhkan model dengan **Recall tinggi** terhadap kelas *Claim* agar meminimalkan **False Negative**.


## **Goals**
- Mengembangkan model klasifikasi untuk memprediksi **Claim (1)** vs **Not Claim (0)**.
- Meminimalkan **False Negative** (klaim terlewat) → fokus pada **Recall** dan **F2-score**.
- Memberi insight faktor risiko untuk mendukung kebijakan underwriting dan alokasi dana.


## **Dataset**
- Sumber: Tingginya false positive dipengaruhi oleh ketimpangan data, di mana hanya 1,71% pemegang polis yang mengajukan klaim. Hal ini membuat model cenderung memprediksi klaim lebih sering.
- Target: `Claim` (0/1)
- Fitur inti umum: `Age`, `Duration`, `Net Sales`, `Commission`, `Agency`, `Agency Type`, `Product Name`, `Distribution Channel`, `Destination`, dll.


**Distribusi Kelas (terdeteksi dari notebook):**


- ('NOT CLAIM', '90', 'claim', '448')


## **Analytical Approach (Models & Metrics)**
- **Preprocessing**: handling missing values, encoding kategori, scaling numerik, outlier check.
- **Menangani Imbalance**: SMOTE, RandomOverSampler, RandomUnderSampler, NearMiss, ADASYN, SMOTEENN, SMOTETomek, EasyEnsemble, BalancedRandomForest.
- **Modeling**: LogisticRegression, RandomForest, XGBClassifier, AdaBoost, GradientBoosting.
- **Evaluasi**: cross-validation dengan metrik **Recall**, **F2**, dan **PR AUC**.


## **Results & Model Selection**
- **Best Recall** (CV/validasi): ~0.963
- **Best F2-score**: ~0.242
- **Best PR AUC**: ~0.087
- Model terbaik dipilih berdasarkan *trade-off* antara Recall tinggi dan stabilitas (std) antar fold.


## **Model Limitations**
1. **Imbalance**: Proporsi klaim sangat kecil → rawan **False Positive/False Negative**.
2. **Feature Sufficiency**: Fitur mungkin belum merepresentasikan seluruh faktor risiko (riwayat klaim, biaya polis, musim, rute perjalanan, dll.).
3. **Domain Validity**: Prediksi valid pada rentang fitur yang terlihat saat training; out-of-range menurunkan reliabilitas.
4. **Data Drift**: Pola klaim dapat berubah musiman; perlu *monitoring* dan *retraining* berkala.


## **Cost Sensitivity & Business Impact**
Kesalahan **False Negative** lebih mahal (klaim terlewat) daripada **False Positive** (alokasi dana menganggur). Oleh karena itu, threshold akan disetel untuk mengutamakan **Recall**, dan opsi **cost-sensitive learning** dipertimbangkan.


## **Recommendations**
### Untuk Model
- Tambahkan fitur: biaya polis, riwayat klaim, frekuensi perjalanan, musim, demografi lanjutan.
- Coba teknik imbalance lanjut: **SMOTEENN**, **SMOTETomek**, **ADASYN**, **EasyEnsemble**, **BalancedRandomForest**.
- Lakukan **hyperparameter tuning** lebih luas untuk model-model kandidat.
- Monitoring model dan retraining berkala.

### Untuk Bisnis
- Underwriting lebih ketat pada segmen berisiko tinggi (durasi lama, usia lebih tua, komisi/net sales tinggi, kanal online, produk komprehensif).
- Insentif untuk segmen berisiko rendah (diskon, loyalty, proses klaim ringkas).
- Edukasi pencegahan risiko & komunikasi transparan profil risiko.


## **Reproducibility**
1. Buat environment (Python 3.10+). Install requirements (scikit-learn, imbalanced-learn, xgboost, pandas, numpy, matplotlib).
2. Jalankan notebook dari awal hingga akhir.
3. Ekspor model terbaik (jika ada) dan catat threshold prediksi yang dipakai.
