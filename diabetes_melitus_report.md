# Early Stage Diabetes Prediction

## Disusun Oleh

Nama     : Lilik Triyawan

## Domain Proyek

Diabetes Melitus (DM) adalah penyakit metabolik kronis yang ditandai dengan peningkatan kadar glukosa darah (hiperglikemia) akibat kegagalan sekresi insulin, kerja insulin, atau keduanya [1]. Menurut Federasi Diabetes Internasional (IDF), sekitar 537 juta orang dewasa di seluruh dunia hidup dengan diabetes pada tahun 2021, dan jumlah ini diproyeksikan meningkat menjadi 783 juta pada tahun 2045 [2]. Salah satu tantangan terbesar dari diabetes adalah banyaknya kasus yang tidak terdiagnosis pada tahap awal, yang dapat menyebabkan komplikasi serius seperti penyakit kardiovaskular, kerusakan saraf (neuropati), gagal ginjal, hingga kebutaan [3].

Deteksi dini diabetes sangat krusial untuk mencegah perkembangan penyakit dan komplikasi yang ditimbulkannya. Banyak penelitian menunjukkan bahwa perubahan gaya hidup dan intervensi medis pada fase awal (pre-diabetes atau awal diabetes) dapat mengembalikan kadar glukosa darah ke tingkat normal atau setidaknya menunda onset diabetes tipe 2 [4]. 

Dalam era digitalisasi kesehatan, pendekatan **machine learning** dapat digunakan sebagai alat penunjang keputusan klinis yang efektif untuk memprediksi risiko diabetes berdasarkan gejala klinis awal yang dirasakan pasien (seperti poliuria, polidipsia, penurunan berat badan secara tiba-tiba, dll.). Dengan memanfaatkan model prediksi yang akurat, skrining awal dapat dilakukan secara cepat, murah, dan non-invasif, sehingga pasien dengan risiko tinggi dapat segera dirujuk untuk pemeriksaan laboratorium lebih lanjut.

**Referensi:**

[1] American Diabetes Association, "Classification and Diagnosis of Diabetes: Standards of Medical Care in Diabetes—2022," *Diabetes Care*, vol. 45, no. Supplement_1, pp. S17-S38, 2022. doi: [10.2337/dc22-S002](https://doi.org/10.2337/dc22-S002)

[2] International Diabetes Federation, *IDF Diabetes Atlas*, 10th ed. Brussels, Belgium: International Diabetes Federation, 2021.

[3] M. M. F. Islam, R. Rahman, and M. S. Rahman, "Early Stage Diabetes Risk Prediction Using Machine Learning Approaches," *International Journal of Computer Applications*, vol. 176, no. 12, pp. 20-25, 2020. 

[4] World Health Organization, "Global report on diabetes," World Health Organization, Geneva, 2016.

## Business Understanding

### Problem Statements

Berdasarkan latar belakang tersebut, berikut adalah rincian masalah yang ingin diselesaikan melalui proyek ini:
1. **Bagaimana cara mendeteksi risiko diabetes pada pasien secara dini berdasarkan gejala klinis awal dan faktor demografis menggunakan algoritma Machine Learning?**
2. **Gejala klinis atau fitur apa saja yang memiliki pengaruh paling signifikan terhadap indikasi risiko diabetes pada pasien?**
3. **Algoritma klasifikasi mana yang memberikan tingkat keakuratan dan performa prediksi terbaik untuk masalah ini?**

### Goals

Tujuan yang ingin dicapai dari proyek ini adalah:
1. **Membangun model klasifikasi biner** yang dapat memprediksi risiko diabetes (Positive atau Negative) dengan akurasi dan kinerja tinggi berdasarkan input gejala klinis.
2. **Menganalisis tingkat kepentingan fitur (Feature Importance)** dari model terbaik untuk menentukan gejala klinis utama yang paling mengindikasikan risiko diabetes.
3. **Membandingkan tiga algoritma machine learning** (Random Forest, XGBoost, dan Support Vector Machine) dan menentukan model terbaik sebagai solusi akhir.

### Solution Statements

Untuk mencapai tujuan tersebut, langkah-langkah solusi yang diterapkan adalah sebagai berikut:
1. **Menggunakan 3 Algoritma Machine Learning** untuk melatih model baseline:
   - **Random Forest Classifier**: Metode ensemble berbasis decision tree yang tangguh dalam mendeteksi interaksi non-linear antar fitur.
   - **XGBoost Classifier**: Algoritma gradient boosting yang sangat efisien dan berkinerja tinggi untuk data tabular terstruktur.
   - **Support Vector Machine (SVM)**: Algoritma klasifikasi berbasis margin maksimum yang efektif untuk dataset dimensi sedang.
2. **Melakukan Hyperparameter Tuning** menggunakan metode `GridSearchCV` pada model terbaik untuk mengoptimalkan performa klasifikasi.
3. **Metrik Evaluasi yang Terukur**: Menggunakan metrik standard seperti **Accuracy**, **Precision**, **Recall**, dan **F1-Score**. F1-Score dan Recall akan menjadi fokus perhatian utama karena dalam domain medis, meminimalkan False Negative (pasien berisiko diabetes yang salah diprediksi normal) sangatlah krusial.

## Data Understanding

Dataset yang digunakan adalah **"Early Stage Diabetes Risk Prediction Dataset"** yang diperoleh dari UCI Machine Learning Repository.

**Tautan Sumber Data**: [UCI ML Repository - Early Stage Diabetes Risk Prediction](https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset)

**Kondisi dan Informasi Data**:
- **Jumlah Data**: 520 sampel (baris).
- **Jumlah Fitur**: 17 fitur (16 fitur masukan dan 1 target).
- **Kondisi Data**: Bersih, tidak ada nilai yang hilang (*missing values* = 0).
- **Distribusi Target**:
  - Kelas `Positive` (Pasien berisiko diabetes): 320 sampel (61.5%)
  - Kelas `Negative` (Pasien tidak berisiko/normal): 200 sampel (38.5%)

### Variabel atau Fitur pada Dataset

Berikut adalah rincian fitur yang terdapat pada dataset:

| Nama Variabel | Tipe Data | Deskripsi / Nilai |
|---|---|---|
| **Age** | Numerik | Usia pasien (tahun), rentang 16 s.d 90 tahun. |
| **Gender** | Kategorikal | Jenis kelamin pasien (Male / Female). |
| **Polyuria** | Kategorikal | Apakah pasien mengalami sering buang air kecil (Yes / No). |
| **Polydipsia** | Kategorikal | Apakah pasien sering merasa sangat haus (Yes / No). |
| **sudden weight loss** | Kategorikal | Apakah terjadi penurunan berat badan secara tiba-tiba (Yes / No). |
| **weakness** | Kategorikal | Apakah pasien sering merasa lemah/lesu (Yes / No). |
| **Polyphagia** | Kategorikal | Apakah pasien sering merasa sangat lapar (Yes / No). |
| **Genital thrush** | Kategorikal | Apakah pasien mengalami infeksi jamur genital (Yes / No). |
| **visual blurring** | Kategorikal | Apakah pasien mengalami pandangan kabur (Yes / No). |
| **Itching** | Kategorikal | Apakah pasien sering merasa gatal-gatal (Yes / No). |
| **Irritability** | Kategorikal | Apakah pasien mudah marah/sensitif (Yes / No). |
| **delayed healing** | Kategorikal | Apakah luka pada pasien membutuhkan waktu lama untuk sembuh (Yes / No). |
| **partial paresis** | Kategorikal | Apakah pasien mengalami kelumpuhan sebagian otot (Yes / No). |
| **muscle stiffness** | Kategorikal | Apakah pasien sering mengalami otot kaku (Yes / No). |
| **Alopecia** | Kategorikal | Apakah pasien mengalami kerontokan rambut/kebotakan (Yes / No). |
| **Obesity** | Kategorikal | Apakah pasien mengalami obesitas (Yes / No). |
| **class** | Kategorikal (Target) | Status risiko diabetes pasien (Positive / Negative). |

### Exploratory Data Analysis (EDA)

Beberapa tahapan visualisasi dilakukan untuk memahami karakteristik data:

1. **Distribusi Kelas Target**: Visualisasi menggunakan Bar Chart menunjukkan kelas target cukup seimbang (61.5% Positive vs 38.5% Negative), sehingga tidak diperlukan penanganan ketidakseimbangan kelas yang ekstrim.
   
   *(Grafik disimpan di: `distribusi_target_diabetes.png`)*

2. **Analisis Gejala Utama**: Visualisasi menunjukkan bahwa gejala **Polyuria** (sering kencing) dan **Polydipsia** (sering haus) sangat dominan ditemukan pada pasien yang berstatus `Positive` diabetes. Pasien dengan gejala ini hampir sebagian besar terklasifikasi sebagai `Positive`.
   
   *(Grafik disimpan di: `gejala_utama_vs_target.png`)*

3. **Boxplot Distribusi Usia**: Pasien dengan status `Positive` diabetes memiliki distribusi usia rata-rata yang sedikit lebih tua dibandingkan dengan kelompok pasien `Negative` (median usia sekitar 48 tahun dibanding 45 tahun).
   
   *(Grafik disimpan di: `boxplot_age_diabetes.png`)*

## Data Preparation

Tahapan data preparation yang dilakukan secara berurutan dalam notebook adalah:

1. **Label Encoding untuk Variabel Kategorikal**:
   - **Proses**: Mengubah semua kolom bertipe objek (`Yes`/`No`, `Male`/`Female`, `Positive`/`Negative`) menjadi nilai biner `1` dan `0`. Sebagai contoh, `Yes` diubah menjadi `1`, `No` diubah menjadi `0`. Untuk target `class`, `Positive` diubah menjadi `1` dan `Negative` menjadi `0`.
   - **Alasan**: Algoritma Machine Learning (seperti SVM dan XGBoost) hanya dapat menerima input berupa data numerik. Label encoding mengonversi teks kategorikal biner ke bentuk representasi biner numerik secara efisien.

2. **Pemisahan Fitur dan Target**:
   - **Proses**: Fitur masukan (X) dipisahkan dari kolom label target (y) yaitu kolom `class`.
   - **Alasan**: Memisahkan variabel dependen dan independen untuk proses evaluasi dan training model.

3. **Train-Test Split**:
   - **Proses**: Dataset dibagi menjadi data latih (80%) dan data uji (20%). Dengan total 520 data, didapatkan 416 data latih dan 104 data uji. Pembagian menggunakan parameter `stratify=y` untuk menjaga proporsi target kelas tetap sama di kedua set.
   - **Alasan**: Mencegah overfitting dan untuk menguji kinerja model pada data baru yang belum pernah dilihat sebelumnya secara adil.

4. **Feature Scaling (Standardization)**:
   - **Proses**: Menstandardisasi kolom numerik kontinu `Age` menggunakan `StandardScaler` sehingga memiliki rata-rata (mean) = 0 dan standar deviasi (std) = 1.
   - **Alasan**: Fitur `Age` memiliki rentang nilai puluhan (16-90), sedangkan fitur kategorikal lainnya bernilai 0 dan 1. Perbedaan skala yang jauh dapat membuat algoritma berbasis jarak (seperti SVM) bias terhadap fitur dengan skala yang lebih besar. Standarisasi menyamakan skala seluruh fitur.

## Modeling

Tiga algoritma machine learning digunakan untuk memprediksi risiko diabetes:

### 1. Random Forest Classifier
Random Forest adalah algoritma ensemble yang bekerja dengan cara membangun beberapa decision tree pada waktu pelatihan dan mengeluarkan kelas rata-rata/voting dari pohon-pohon tersebut.
- **Parameter**: `n_estimators=100`, `max_depth=8`, `random_state=42`.
- **Kelebihan**: Sangat baik dalam menangani fitur kategorikal biner, jarang mengalami overfitting karena voting ensemble, dan menyediakan interpretasi `feature_importance`.
- **Kekurangan**: Proses interpretasi model individu (individual tree) menjadi sulit karena terdiri dari banyak pohon keputusan.

### 2. XGBoost Classifier
XGBoost merupakan algoritma Gradient Boosting yang dioptimalkan untuk kecepatan dan efisiensi performa pada data tabular.
- **Parameter**: `n_estimators=100`, `max_depth=5`, `learning_rate=0.1`, `eval_metric='logloss'`.
- **Kelebihan**: Kinerja luar biasa pada dataset terstruktur, cepat, dan memiliki penanganan regularisasi internal untuk meminimalkan overfitting.
- **Kekurangan**: Memiliki banyak hyperparameter kompleks yang memerlukan tuning cermat agar tidak overfitting pada dataset kecil.

### 3. Support Vector Machine (SVM)
SVM berupaya menemukan batas keputusan terbaik (hyperplane) untuk memisahkan data menjadi dua kelas.
- **Parameter**: `kernel='rbf'`, `C=1.0`, `random_state=42`.
- **Kelebihan**: Sangat efektif dalam ruang berdimensi tinggi dan ketika batas pemisah antar kelas cukup jelas.
- **Kekurangan**: Kinerjanya sangat bergantung pada preprocessing yang tepat (seperti standarisasi skala fitur).

### Pemilihan Model Terbaik dan Hyperparameter Tuning
Berdasarkan hasil baseline, **SVM** merupakan model terbaik dengan performa tertinggi (F1-score 0.9922). Namun, pada tahap ini, Random Forest dipilih untuk dilakukan **Hyperparameter Tuning**. Hal ini dikarenakan Random Forest merupakan algoritma berbasis pohon keputusan yang memiliki interpretabilitas model yang sangat baik melalui ekstraksi `feature_importances_`. Karena mengidentifikasi gejala klinis utama (Feature Importance) adalah salah satu *Goals* utama proyek ini, Random Forest dipilih sebagai model solusi akhir karena memberikan keseimbangan antara performa yang sangat tinggi dan kemampuan interpretasi medis yang kuat.

Penyetelan hyperparameter dilakukan menggunakan `GridSearchCV` dengan skema cross-validation lipat-5 (CV=5) pada parameter grid berikut:
- `n_estimators`: `[50, 100, 150, 200]`
- `max_depth`: `[4, 6, 8, 10]`
- `min_samples_split`: `[2, 5, 10]`
- `criterion`: `['gini', 'entropy']`

**Hasil Tuning**: Hyperparameter terbaik yang ditemukan adalah `{'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 100}`. Setelah tuning, performa Random Forest meningkat menjadi Accuracy 0.9808 dan F1-Score 0.9844, meskipun masih sedikit di bawah SVM baseline. Pemilihan Random Forest sebagai solusi akhir didasarkan pada kebutuhan interpretabilitas model, bukan semata-mata metrik tertinggi.

## Evaluation

### Metrik Evaluasi dan Formula

Metrik evaluasi yang digunakan untuk menilai kinerja model adalah **Accuracy, Precision, Recall, dan F1-Score**.

1. **Accuracy**: Persentase total prediksi yang benar secara keseluruhan.
   $$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

2. **Precision**: Mengukur seberapa akurat prediksi positif model dibandingkan dengan seluruh hasil prediksi positif.
   $$Precision = \frac{TP}{TP + FP}$$

3. **Recall (Sensitivity)**: Mengukur kemampuan model dalam mendeteksi seluruh data positif yang sebenarnya. Dalam kesehatan, metrik ini paling kritis karena meminimalkan False Negative (pasien sakit yang diprediksi sehat).
   $$Recall = \frac{TP}{TP + FN}$$

4. **F1-Score**: Rata-rata harmonik dari Precision dan Recall yang memberikan gambaran performa model secara seimbang.
   $$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

**Keterangan**:
- **TP (True Positive)**: Pasien risiko diabetes (Positive) yang diprediksi benar sebagai Positive.
- **TN (True Negative)**: Pasien sehat (Negative) yang diprediksi benar sebagai Negative.
- **FP (False Positive)**: Pasien sehat yang salah diprediksi sebagai Positive.
- **FN (False Negative)**: Pasien risiko diabetes yang salah diprediksi sebagai Negative.

### Hasil Proyek Berdasarkan Metrik Evaluasi

Berikut adalah perbandingan performa seluruh model pada data uji (test set):

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Random Forest (Baseline) | 0.9712 | 0.9841 | 0.9688 | 0.9764 |
| XGBoost (Baseline) | 0.9808 | 1.0000 | 0.9688 | 0.9841 |
| **SVM (Baseline)** | **0.9904** | **0.9846** | **1.0000** | **0.9922** |
| Random Forest (Tuned) | 0.9808 | 0.9844 | 0.9844 | 0.9844 |

**Analisis Hasil - Model Terbaik Berdasarkan Metrik**:
- Berdasarkan metrik evaluasi, **SVM merupakan model terbaik** dengan F1-Score tertinggi sebesar **0.9922**, Accuracy **99.04%**, dan Recall sempurna **1.0000** (tidak ada satu pun pasien diabetes yang terlewat).
- XGBoost menempati posisi kedua dengan F1-Score 0.9841 dan Precision sempurna 1.0000.
- Random Forest baseline berada di posisi ketiga dengan F1-Score 0.9764.

**Analisis Hasil - Hyperparameter Tuning pada Random Forest**:
- Setelah dilakukan hyperparameter tuning, performa Random Forest meningkat dari F1-Score 0.9764 menjadi **0.9844** (peningkatan +0.008).
- Meskipun peningkatan ini signifikan, performa Random Forest Tuned (F1-Score 0.9844) masih berada di bawah SVM baseline (F1-Score 0.9922).

**Pemilihan Model Solusi Akhir**:
- Meskipun SVM unggul dari sisi metrik evaluasi, **Random Forest (Tuned) dipilih sebagai model solusi akhir**. Alasan utamanya adalah Random Forest menyediakan fitur `feature_importances_` yang memungkinkan ekstraksi informasi gejala klinis mana yang paling berpengaruh terhadap prediksi diabetes. Kemampuan interpretasi ini sangat penting dalam domain medis dan merupakan salah satu *Goals* utama proyek ini (mengidentifikasi gejala klinis paling signifikan). SVM, sebagai model *black-box*, tidak menyediakan interpretabilitas serupa secara langsung.
- Dengan performa yang tetap sangat tinggi (Accuracy 98.08%, F1-Score 98.44%), Random Forest Tuned tetap layak sebagai solusi skrining medis awal.

### Analisis Confusion Matrix Model Solusi (Random Forest Tuned)

Dari confusion matrix model Random Forest Tuned pada data uji (104 sampel), diperoleh hasil sebagai berikut:

| | Prediksi Negative | Prediksi Positive |
|---|---|---|
| **Aktual Negative** | 39 (TN) | 1 (FP) |
| **Aktual Positive** | 1 (FN) | 63 (TP) |

**Interpretasi**:
- **True Positive (TP) = 63**: Sebanyak 63 pasien yang benar-benar berisiko diabetes berhasil diprediksi dengan benar oleh model.
- **True Negative (TN) = 39**: Sebanyak 39 pasien sehat berhasil diprediksi dengan benar sebagai tidak berisiko.
- **False Positive (FP) = 1**: Hanya 1 pasien sehat yang salah diprediksi sebagai berisiko diabetes. Dalam konteks medis, kesalahan ini relatif tidak berbahaya karena pasien hanya akan dirujuk untuk pemeriksaan lanjutan.
- **False Negative (FN) = 1**: Hanya 1 pasien yang sebenarnya berisiko diabetes namun tidak terdeteksi oleh model. Angka FN yang sangat rendah ini sangat penting dalam domain kesehatan karena **kegagalan mendeteksi pasien berisiko (False Negative) dapat berakibat fatal** — pasien tersebut tidak akan mendapatkan penanganan dini yang dibutuhkan. Dengan hanya 1 FN dari 64 kasus positif, model ini menunjukkan kemampuan deteksi yang sangat baik.

### Analisis Fitur Paling Berpengaruh (Feature Importance)

Berdasarkan analisis nilai kepentingan fitur dari model Random Forest terbaik, diperoleh kesimpulan gejala klinis utama:
1. **Polyuria** (skor kepentingan tertinggi) - Sering buang air kecil.
2. **Polydipsia** - Rasa haus yang berlebihan.
3. **Gender** - Faktor jenis kelamin.
4. **Age** - Usia pasien.
5. **Sudden weight loss** - Penurunan berat badan secara tiba-tiba.

Gejala klinis **Polyuria** dan **Polydipsia** merupakan sinyal terkuat untuk mendeteksi diabetes secara dini. Hal ini sangat sejalan dengan literatur medis di mana kedua gejala tersebut merupakan indikasi klasik dari penumpukan glukosa dalam darah yang memaksa ginjal bekerja lebih keras untuk menyaring dan menyerapnya kembali [1].

**---Ini adalah bagian akhir laporan---**
