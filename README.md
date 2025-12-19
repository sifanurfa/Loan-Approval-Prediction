# Loan Approval Prediction App (AdaBoost Model)

Aplikasi ini memprediksi apakah **pengajuan pinjaman layak disetujui** berdasarkan data calon peminjam.  
Dibangun menggunakan **Python, Scikit-Learn, AdaBoost, dan Streamlit** dengan pendekatan **machine learning pipeline**,  
sehingga proses *preprocessing* dan *inference* berjalan konsisten antara training dan deployment.

Aplikasi: [Loan Approval Prediction](https://sifanurfa-loan-approval.streamlit.app/)

---

## **Dataset**
Dataset: [Loan Data](https://raw.githubusercontent.com/sifanurfa/dataset/refs/heads/main/loan_data.csv)

| Kolom | Deskripsi |
|-------|------------|
| Gender | Jenis kelamin pemohon |
| Married | Status pernikahan |
| Dependents | Jumlah tanggungan |
| Education | Pendidikan terakhir |
| Self_Employed | Wirausaha atau tidak |
| ApplicantIncome | Pendapatan utama |
| CoapplicantIncome | Pendapatan tambahan |
| LoanAmount | Jumlah pinjaman (ribu) |
| Loan_Amount_Term | Lama pinjaman (bulan) |
| Credit_History | Riwayat kredit (1 = baik, 0 = buruk) |
| Property_Area | Lokasi properti |
| Loan_Status | Target (Y = disetujui, N = tidak) |

---

## **Struktur Proyek**
```
Loan-Approval-Prediction
│
├── app.py                         # Aplikasi Streamlit
├── Loan_Approval_Prediction.ipynb # Notebook untuk training & evaluasi model
├── AdaBoost_best_pipeline.joblib  # Pipeline terbaik (preprocessor + model AdaBoost)
├── requirements.txt               # Dependencies untuk Streamlit Cloud
└── README.md                      # Dokumentasi proyek
```

---

## **Langkah Training Model**

1. Buka file `training_notebook.ipynb`
2. Jalankan semua sel (read data, preprocessing, modelling, evaluasi)
3. Model terbaik (berdasarkan F1-score tertinggi) otomatis disimpan sebagai:
   ```
   AdaBoost_best_pipeline.joblib
   ```
4. File `.joblib` ini sudah berisi **preprocessor + AdaBoost model**,  
   jadi bisa langsung digunakan untuk prediksi tanpa transformasi manual.

---

## **Hasil Evaluasi Model**

| Model | Accuracy | Precision | Recall | F1-score |
|--------|-----------|------------|----------|-----------|
| **AdaBoost (Best)** | **0.852** | **0.828** | **1.000** | **0.906** |
| KNN | 0.852 | 0.828 | 1.000 | 0.906 |
| Logistic Regression | 0.843 | 0.820 | 1.000 | 0.901 |
| Random Forest | 0.843 | 0.827 | 0.988 | 0.900 |
| Decision Tree | 0.757 | 0.838 | 0.817 | 0.827 |

**Model terbaik: AdaBoost (F1 = 0.906)**  
Nilai recall = 1.0 menunjukkan semua data positif berhasil terdeteksi

---

## **Penggunaan di Aplikasi Streamlit**

1. Masukkan data calon peminjam:
   - Gender: Male  
   - Married: Yes  
   - Dependents: 1  
   - Education: Graduate  
   - Self Employed: No  
   - ApplicantIncome: 5000  
   - CoapplicantIncome: 1500  
   - LoanAmount: 100  
   - Loan_Amount_Term: 360  
   - Credit_History: 1.0  
   - Property_Area: Urban  

2. Klik tombol **“Prediksi”**

3. Hasil muncul seperti ini:
   ```
   Disetujui! (Probabilitas: 91.5%)
   ```
   atau
   ```
   Tidak Disetujui. (Probabilitas: 22.7%)
   ```

---

## **Pipeline**
Pipeline ini menggabungkan semua tahapan preprocessing dan model dalam satu objek:

```
ColumnTransformer
├── num → SimpleImputer(median) → StandardScaler
└── cat → SimpleImputer(most_frequent) → OneHotEncoder
↓
AdaBoostClassifier (base_estimator=DecisionTree, n_estimators=50, learning_rate=1.0)
```

Dengan pipeline ini, data input di aplikasi akan otomatis:
1. Mengisi missing values  
2. Menormalkan fitur numerik  
3. Melakukan one-hot encoding pada fitur kategorikal  
4. Diteruskan ke model AdaBoost untuk prediksi  

---
