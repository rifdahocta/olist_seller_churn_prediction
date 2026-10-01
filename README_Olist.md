# Olist Seller Churn Prediction

Final project **Tim Alpha** untuk mengidentifikasi seller berisiko churn pada platform e-commerce Olist dan mendukung penentuan campaign retensi yang lebih terukur.

## Anggota Tim Alpha

| No. | Nama |
|---|---|
| 1 | Rifdah Octavi Azzahra |
| 2 | Dzakwan Taufiq Nur Muhammad |
| 3 | Aris Sando Hamzah |

## Dashboard

📊 **[Buka Dashboard Olist — Looker Studio](https://datastudio.google.com/reporting/81bca56e-5d11-4f64-a322-d5e099e64841)**

## Latar Belakang dan Tujuan

Keberlanjutan aktivitas seller berperan dalam menjaga variasi produk dan transaksi marketplace. Proyek ini menganalisis pola aktivitas seller serta membangun model klasifikasi untuk menghasilkan prediksi dan probabilitas churn sebagai pendukung strategi retensi.

F2-score dan recall kelas churn menjadi metrik utama karena kegagalan mendeteksi seller churn (False Negative) dianggap lebih mahal dalam simulasi bisnis daripada pemberian campaign kepada seller yang sebenarnya aktif (False Positive).

## Data dan Definisi Target

Data historis Olist periode 2016–2018 mencakup orders, order items, customers, sellers, payments, products, geolocation, dan terjemahan kategori produk. Data diintegrasikan pada tingkat **satu baris per seller**.

Pelabelan pada notebook menggunakan aktivitas pengiriman order ke carrier (`order_delivered_carrier_date`). Cutoff ditetapkan 30 hari sebelum tanggal pengiriman terakhir dalam dataset. Seller dengan nol order valid atau aktivitas pertama pada 30 hari terakhir sebelum cutoff dikategorikan **not eligible** dan dikeluarkan dari pemodelan. Seller eligible dengan aktivitas terakhir sebelum cutoff diberi label churn; aktivitas terakhir pada atau setelah cutoff diberi label no churn.

| Label | Arti | Jumlah seller eligible |
|---|---|---:|
| 1 | Churn | 1.818 |
| 0 | No churn | 874 |
| **Total** | **Data pemodelan** | **2.692** |

Proporsi churn pada data pemodelan adalah **67,53%**. Sebanyak 403 seller tidak eligible dikeluarkan sebelum pemodelan.

## Alur Analisis

1. Extract, Transform, Load (ETL): penyesuaian tipe data, pemeriksaan missing value, duplikasi, dan koordinat geografis.
2. Integrasi data dan agregasi fitur pada tingkat seller, dengan fitur historis dibatasi hingga cutoff.
3. Exploratory Data Analysis (EDA), analisis hubungan fitur dengan target, dan pemeriksaan korelasi antarfitur.
4. Pembagian data train–test sebesar 80:20 dengan stratifikasi target.
5. Preprocessing melalui median imputation dan RobustScaler di dalam pipeline.
6. Perbandingan model menggunakan cross-validation, kemudian tuning Logistic Regression melalui GridSearchCV dengan 5 fold dan scoring F2.
7. Evaluasi akhir, simulasi biaya kesalahan prediksi, interpretasi koefisien, dan aplikasi Streamlit.

## Fitur Model Final

| Fitur | Deskripsi |
|---|---|
| `total_orders` | Jumlah order historis seller |
| `days_since_last_order` | Jeda hari dari order terakhir hingga cutoff |
| `avg_gap_antar_order` | Rata-rata jeda hari antarorder |
| `total_active_order_days` | Jumlah hari berbeda dengan aktivitas order |
| `unique_products` | Jumlah produk unik |
| `total_product_price` | Total nilai harga item, tanpa ongkos kirim |
| `min_product_price` | Harga item minimum |
| `max_payment_value` | Nilai pembayaran maksimum pada agregasi seller |
| `max_payment_installments` | Jumlah cicilan maksimum |
| `avg_product_name_length` | Rata-rata panjang nama produk |
| `unique_product_categories` | Jumlah kategori produk unik |
| `min_seller_customer_distance_km` | Jarak minimum seller–customer dalam km |
| `median_seller_customer_distance_km` | Median jarak seller–customer dalam km |
| `max_seller_customer_distance_km` | Jarak maksimum seller–customer dalam km |

## Model dan Hasil Evaluasi

Model final menggunakan **Logistic Regression dengan regularisasi L2 dan `C=0.001`**. Pemilihan mempertimbangkan performa validasi dan kestabilan hasil. Angka berikut mengacu pada output evaluasi yang tersimpan dalam notebook final.

| Metrik | Nilai |
|---|---:|
| F2 train | 90,44% |
| Rata-rata F2 cross-validation terbaik | 90,41% |
| F2 test | 89,61% |
| Recall churn — test | 94,78% |
| Precision churn — test | 73,56% |
| Accuracy — test | 73,47% |

Confusion matrix data test, dengan **kelas positif = churn**:

| Aktual / Prediksi | No churn (0) | Churn (1) |
|---|---:|---:|
| No churn (0) | TN = 51 | FP = 124 |
| Churn (1) | FN = 19 | TP = 345 |

Model menangkap 345 dari 364 seller churn pada data test. Sebanyak 124 seller no churn ikut ditandai churn, sehingga kapasitas dan biaya campaign perlu diperhitungkan.

## Simulasi Biaya

Simulasi menggunakan asumsi biaya **$500 per FN** dan **$100 per FP**:

`Loss = (500 × FN) + (100 × FP)`

| Skenario | FN | FP | Estimasi loss |
|---|---:|---:|---:|
| Tanpa model: seluruh seller diprediksi no churn | 364 | 0 | $182.000 |
| Dengan model | 19 | 124 | $21.900 |

Penurunan estimasi loss sebesar **87,97%**. Angka ini merupakan simulasi penalti kesalahan klasifikasi berdasarkan asumsi proyek, bukan biaya aktual Olist, total anggaran campaign, atau bukti penghematan setelah implementasi.

## Aplikasi Streamlit

Aplikasi mendukung:

- Prediksi satu seller melalui formulir manual.
- Prediksi massal melalui unggah CSV serta unduh hasil prediksi.
- Tampilan probabilitas churn dan label prediksi.
- Batas input mengikuti minimum–maksimum pada data train.
- Threshold tetap **0,5 (50%)**, tanpa pengaturan pengguna: probabilitas ≥50% diklasifikasikan sebagai churn, sedangkan <50% sebagai active/no churn.
- Logo Olist dan template CSV.

Ekstrak paket aplikasi, buka terminal di folder yang berisi `app.py`, lalu jalankan:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.port 8502
```

Buka [http://localhost:8502](http://localhost:8502).

Versi aplikasi portabel menggunakan parameter yang diekspor dari model final melalui `model_parameters.json`. Mengganti file SAV saja tidak memperbarui prediksi aplikasi; parameter perlu diekspor ulang ketika model berubah.

## Berkas Utama

| Berkas | Fungsi |
|---|---|
| `Olist_Seller_Churn_Prediction.ipynb` | Notebook ETL, EDA, pemodelan, dan evaluasi |
| `olist_seller_churn_model_final.sav` | Pipeline model final yang disimpan |
| `olist_seller_churn_X_train.csv` | Fitur data train dan referensi batas input |
| `olist_seller_churn_streamlit_final.zip` | Paket aplikasi Streamlit |
| `README.md` | Dokumentasi proyek |

## Rekomendasi

Prioritaskan campaign retensi berdasarkan risiko seller churn yang diprediksi model agar penargetan lebih terukur dan anggaran lebih efisien. Sesuaikan prioritas dengan kapasitas tim, nilai bisnis seller, dan hasil evaluasi campaign.

Probabilitas merupakan estimasi model. Threshold 0,5 digunakan sebagai baseline; kualitas kalibrasi dan dampak retensi perlu dievaluasi sebelum penggunaan operasional yang lebih luas. Untuk menguji kemampuan peringatan dini, lakukan validasi waktu pada periode berikutnya dengan fitur yang tersedia sebelum periode target.

## Teknologi

Python, pandas, NumPy, seaborn, Matplotlib, scikit-learn, Google BigQuery, Looker Studio, dan Streamlit.
