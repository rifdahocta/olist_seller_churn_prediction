# Olist Seller Churn — Model Final

Ekstrak ZIP, lalu buka terminal pada folder aplikasi:

```bat
cd /d "C:\Users\Aris Sando\Documents\Olist\olist_seller_churn_streamlit"
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.port 8502
```

Buka http://localhost:8502. Hentikan aplikasi lama dengan Ctrl+C sebelum menjalankan versi baru.

## Pembaruan
- Parameter prediksi berasal dari olist_seller_churn_model_final.sav.
- Nama fitur mengikuti model final, termasuk avg_gap_antar_order dan total_active_order_days.
- Input manual dan CSV divalidasi terhadap min–max data train (2.153 baris).
- Nilai di luar batas ditolak sebelum prediksi. Nilai kosong diisi median model.
- Threshold tetap 0,5 (50%), tanpa pengaturan pengguna; probabilitas tetap ditampilkan.
- Template CSV terbaru dapat diunduh dalam aplikasi.
- Logo Olist memerlukan koneksi internet.

Prediksi dijalankan dari parameter JSON, sehingga aplikasi tidak memerlukan scikit-learn atau imbalanced-learn. Tidak ada proses pelatihan ulang. Jangan mengganti SAV saja untuk memperbarui prediksi; parameter JSON harus diekspor kembali dari model baru.
