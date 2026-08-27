# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan

- **Nama:** Mohamad Rangga Nur Faizin
- **Email:** mohammad.rangga.n.f@gmail.com
- **Id Dicoding:** rangfaziii19

## Business Understanding

Jaya Jaya Institut adalah institusi pendidikan tinggi yang telah berdiri sejak tahun 2000 dan memiliki reputasi baik dalam mencetak lulusan berkualitas. Namun, institusi ini menghadapi masalah serius: tingkat dropout (siswa tidak menyelesaikan studi) yang cukup tinggi. Tingginya angka dropout berdampak pada reputasi institusi, efektivitas pembelajaran, dan kerugian finansial.

### Permasalahan Bisnis

1. Tingginya angka siswa yang dropout sebelum menyelesaikan pendidikan.
2. Sulit mengidentifikasi siswa berisiko tinggi secara dini untuk mendapatkan intervensi.
3. Belum ada sistem monitoring berbasis data untuk memantau performa siswa secara efektif.

### Cakupan Proyek

1. Menganalisis data historis siswa untuk memahami pola dan tren dropout.
2. Membangun model machine learning untuk memprediksi risiko dropout siswa.
3. Mengidentifikasi faktor-faktor kunci yang memengaruhi keputusan dropout.
4. Mengembangkan dashboard interaktif untuk memudahkan stakeholder memahami data dan memantau performa siswa.
5. Membangun prototype sistem prediksi berbasis web (Streamlit) yang siap digunakan.
6. Memberikan rekomendasi action items untuk meningkatkan retensi siswa.

---

## Persiapan

### Sumber data

Dataset: [Students' Performance](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv)

Data yang digunakan pada proyek ini adalah dataset Students' Performance dari Dicoding Academy yang berisi informasi akademik, demografis, dan sosial-ekonomi siswa yang terdaftar di berbagai program studi di Jaya Jaya Institut. Dataset ini digunakan untuk membangun model klasifikasi yang memprediksi apakah seorang siswa akan dropout, tetap terdaftar (enrolled), atau lulus (graduate).

### Setup environment

```
# Setup virtual environment (hanya perlu dijalankan sekali)
uv venv .venv
uv pip install --python .venv/Scripts/python.exe pandas==2.2.3 numpy==2.1.3 matplotlib==3.9.2 seaborn==0.13.2 scikit-learn==1.5.2 joblib==1.4.2 jupyter==1.1.1 nbformat==5.10.4 streamlit==1.41.0
```

---

## Business Dashboard

Dashboard interaktif telah dibuat menggunakan Metabase untuk membantu Jaya Jaya Institut memahami data dan memonitor performa siswa. Dashboard ini menampilkan metrik utama seperti total siswa, persentase dropout, distribusi status siswa per jurusan, dan faktor-faktor yang mempengaruhi dropout.

**Kredensial akses dashboard:**

- **URL Dashboard:** Diakses via aplikasi Streamlit (lihat bagian Menjalankan Sistem Machine Learning).
- **Metabase credentials (alternatif):**
  - Email: `root@mail.com`
  - Password: `root123`

---

## Menjalankan Sistem Machine Learning

Prototype sistem prediksi dropout siswa berbasis Streamlit dapat dijalankan secara lokal maupun cloud.

### Menjalankan secara lokal

```bash
# 1. Pastikan virtual environment sudah aktif
uv venv .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt

# 2. Jalankan aplikasi
streamlit run app.py

# 3. Buka browser di http://localhost:8501
```

### Prototype di Streamlit Community Cloud

Link prototype: `https://student-performance-<username>.streamlit.app/`

Prototype ini menyediakan dua menu utama:
- **Dashboard**: Visualisasi data dan faktor-faktor yang mempengaruhi dropout.
- **Predict**: Formulir interaktif untuk memprediksi risiko dropout siswa secara real-time beserta probabilitas dan rekomendasi intervensi.

---

## Conclusion

Berdasarkan analisis data dan model machine learning yang telah dikembangkan, dapat disimpulkan bahwa:

1. **Tingkat dropout di Jaya Jaya Institut cukup tinggi.** Dari 4.424 siswa, 32.1% mengalami dropout, 49.9% graduate, dan 18.0% masih enrolled. Angka ini menunjukkan perlunya intervensi segera.
2. **Performa akademik semester awal adalah prediktor terkuat.** Siswa yang memiliki nilai semester 1 dan 2 yang rendah, serta jumlah unit yang disetujui sedikit, memiliki risiko dropout yang jauh lebih tinggi.
3. **Kondisi finansial mempengaruhi dropout.** Siswa dengan tunggakan biaya (Debtor=1) atau pembayaran tidak terkini menunjukkan tingkat dropout lebih tinggi. Sebaliknya, pemegang beasiswa (Scholarship_holder=1) cenderung lebih bisa menyelesaikan studi.
4. **Usia pendaftaran juga berpengaruh.** Siswa yang mendaftar di usia lebih tua (di atas 23 tahun) menunjukkan risiko dropout yang lebih tinggi.
5. **Model Random Forest berhasil memprediksi dropout dengan akurasi tinggi** (ROC-AUC > 0.90), menunjukkan bahwa pendekatan machine learning dapat menjadi alat deteksi dini yang efektif.

---

### Rekomendasi Action Items

Berdasarkan temuan di atas, berikut adalah rekomendasi untuk Jaya Jaya Institut:

1. **Program Bimbingan Khusus untuk Siswa Berisiko Tinggi**  
   Siswa yang terdeteksi berisiko tinggi (probabilitas dropout > 60%) diberikan bimbingan akademik dan konseling sejak semester pertama.

2. **Peningkatan Dukungan Finansial**  
   Memperluas program beasiswa dan bantuan biaya kuliah, terutama untuk siswa yang memiliki tunggakan atau kesulitan pembayaran.

3. **Monitoring Performa Akademik Semester Awal**  
   Membangun sistem alert otomatis jika nilai semester 1 atau jumlah unit yang disetujui di bawah threshold tertentu.

4. **Kampanye Kesejahteraan dan Keterlibatan Siswa**  
   Meningkatkan program kegiatan ekstrakurikuler dan dukungan psikologis untuk meningkatkan keterlibatan dan retensi siswa.

5. **Evaluasi Program Studi dengan Dropout Tinggi**  
   Jurusan dengan tingkat dropout tertinggi (misal: Management) perlu dievaluasi ulang kurikulum dan metode pengajarannya.
