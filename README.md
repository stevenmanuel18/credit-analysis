# South German Credit Prediction using Logistic Regression

## Project Domain

Pembiayaan kredit adalah sebuah penyedia dana berdasarkan perjanjian pinjaman antara peminjam dan bank. Bagian marketing dalam perbankan harus memilih calon pelanggan yang diberikan kredit berdasarkan beberapa hal seperti tingkat resiko, jatuh tempo, dan lain-lain. 
Pemilihan calon pelanggan penting untuk meminimalkan resiko terjadinya pinjaman kredit bermasalah dengan cara mempelajari pola pada data pelanggan yang tersedia. Salah satu cara yang dapat digunakan adalah menggunakan teknik penambangan data [1]. 
Pada penelitian sebelumnya digunakan algoritma random forest untuk memprediksi pinjaman bermasalah, sehingga dalam proyek ini akan digunakan algoritma logistic regression.

## Business Understanding

### Problem Statement(s)
 - Bagaimana mengenali pola pada data South German Credit untuk mengklasifikasikan sampel yang memiliki kecenderungan untuk terjadi pinjaman kredit bermasalah?

### Goals
 - Membuat model pembelajaran mesin klasifikasi yaitu algoritma logistic regression yang dapat mengenali pola data kredit dan memprediksi adanya indikasi pinjaman kredit bermasalah sebelum persetujuan pinjaman.

### Solution Statement(s)
 - Membangun model pembelajaran mesin logistic regression dan pembuatan baseline model
 - Meningkatkan performa model logistic regression dengan hyperparameter tuning dengan memperhatikan akurasi sebagai metrik penilaian

## Data Understanding
Data yang digunakan pada proyek ini adalah "South German Credit" yaitu data pelanggan dari tahun 1973 sampai 1975 dengan 20 variabel independen dan 1000 sampel yang terdiri dari 700 sampel kredit baik dan 300 kredit bermasalah [2].

### Variabel
 - laufkont (status)			: status akun bank peminjam (categorical)
 - laufzeit (duration)			: durasi kredit dalam bulan (quantitative)
 - moral (credit_history)		: riwayat pemenuhan kontrak kredit sebelumnya atau bersamaan (categorical)
 - verw (purpose)			: tujuan pengajuan kredit (categorical)
 - hoehe (amount)			: jumlah kredit dalam Deutsche Mark (quantitative)
 - sparkont (savings)			: tabungan peminjam (categorical)
 - beszeit (emplyoment_duration)	: lamanya masa kerja peminjam pada pemberi kerja saat ini (ordinal)
 - rate (installment_rate)		: angsuran kredit sebagai persentase pendapatan yang dapat dibelanjakan debitur (ordinal)
 - famges (personal_status_sex)		: informasi gabungan jenis kelamin dan perkawinan (categorical)
 - buerge (other_debtors)		: apakah ada peminjam atau penjamin lain dalam kredit? (categorical)
 - wohnzeit (present_resident)		: jangka waktu dalam tahun debitur tinggal di tempat tinggalnya sekarang (ordinal)
 - verm (property)			: properti paling harga peminjam (ordinal)
 - alter (age)				: usia dalam tahun (quantitative)
 - weitkred (other_installment_plans)	: rencana cicilan dari penyedia selain bank pemberi kredit (categorical)
 - wohn (housing)			: tipe rumah peminjam (categorical)
 - bishkred (number_of_credits)		: jumlah kredit yang dimiliki peminjam saat ini atau pernah punya pada bank ini (ordinal)
 - beruf (job)				: kualitas pekerjaan peminjam (ordinal)
 - pers (person_liable)			: jumlah orang yang bergantung pada peminjam secara finansial (binary)
 - telef (telephone)			: apakah ada telepon rumah yang terdaftar atas nama peminjam? (binary)
 - gastarb (foreign_worker)		: apakah peminjam pekerja asing? (binary)
 - kredit (credit_risk)			: apakah kontrak kredit dipenuhi dengan baik atau buruk ? (binary)

### Exploratory Data Analysis (EDA)

Pada tahapan ini data diperiksa menggunakan perintah .info() untuk memeriksa apakah adanya missing value dan menggunakan .describe() jika ada nilai yang tidak semestinya. Ditemukan bahwa pada data ini tidak terdapat missing value atau anomali, sehingga siap dipakai untuk prediksi.

## Data Preparation

Pada tahap ini data disiapkan untuk masuk ke model pembelajaran mesin. Beberapa variabel disimpan dalam bentuk nilai diskrit berurutan seolah-olah adalah variabel ordinal yang semestinya adalah variabel nominal, sehingga variabel-veriabel yang tercakup dalam hal tersebut diubah menggunakan one hot encoding. Setelah itu data dipisahkan menjadi data latih dan data uji dengan perbandingan 8:2, dimana data latih digunakan untuk membangun model dan data uji untuk mengevaluasi model. Data yang ada tidak langsung dinormalisasikan untuk mencegah data leakage pada cross-validation, melainkan pembuatan pipeline untuk normalisasi pada setiap fold. 

## Modeling

Pada proyek ini digunakan algoritma logistic regression. Meskipun dinamakan regression, model ini sebenarnya adalah model pembelajaran mesin untuk klasifikasi. 
Model ini mengestimasi probabilitas terjadinya suatu kejadian berdasarkan variabel independent yang tersedia [2]. Setiap variabel independent diasumsikan memiliki hubungan linier. Setiap variabel lalu diberikan bobot untuk dicari pada setiap iterasinya. Jumlahan antara bobot dan variabel independen akan disubstitusikan kedalam fungsi logistik yang menghasilkan nilai antara 0 dan 1, dengan 0.5 sebagai threshold, yang berarti setiap nilai yang jatuh diatas nilai 0.5 diprediksi positif, dan sebaliknya adalah negatif.

Beberapa kelebihan dan kelemahan model logistic regression adalah sebagai berikut [4]
Kelebihan:
 - Mudah diimplementasi dan efisien
 - Tidak mengasumsi distribusi pada data
 - Cepat
 - Baik dalam memprediksi data sederhana
 - Tidak cenderung untuk over-fitting

Kekurangan:
 - Buruk dalam data berdimensi tinggi
 - Menggunakan fungsi pemisah linier
 - Asumsi kelinieran antara fitur dan target
 - Membutuhkan Tidak terjadinya multicollinearity antara fitur
 - Susah dalam mencari hubungan yang lebih kompleks

Setelah logistic regression dibangun menggunakan scikit-learn, akurasi model dijadikan baseline untuk improvement berikutnya. Pipeline yang sudah dibuat digunakan dalam hyperparameter tuning menggunakan random search cv dengan aturan stratifiedkfold untuk mencari hyperparameter yang lebih optimal dan performa yang lebih baik. Hyperparameter yang dicari dan digunakan adalah C = 0.4385722446796244

## Evaluation

Pada tahap ini model yang sudah dibuat akan digunakan untuk memprediksi pada data uji. Metrik yang digunakan adalah akurasi, karena kedua label positif dan negatif sama pentingnya dalam kasus ini. Label positif penting untuk mencari calon pelanggan yang sehat dan negatif untuk memperkecil resiko pinjaman yang tidak sehat.
Akurasi adalah metrik penilaian yang membandingkan jumlah sampel yang terprediksi benar dengan jumlah sampel yang ada pada data.
Pada proyek ini, baseline yang berhasil dibuat memiliki nilai akurasi latih dan validasi 79.47% dan 75.63% berturut-turut. Sedangkan setelah melalui hyperparameter tuning, variance pada model mengecil dan terjadi peningkatan akurasi pada data validasi dengan akurasi latih dan validasi 79.31% dan 76%. Akurasi uji yang didapat pada akhir proyek adalah 77%.

## Referensi
[1] Religia, Y., Pranoto, G. T., & Santosa, E. D. (2020). South German Credit Data Classification Using Random Forest Algorithm to Predict Bank Credit Receipts. JISA (Jurnal Informatika dan Sains), 3(2), 62-66. <br>
[2] https://archive.ics.uci.edu/dataset/573/south+german+credit+update <br>
[3] https://aws.amazon.com/what-is/logistic-regression/ <br>
[4] https://www.geeksforgeeks.org/advantages-and-disadvantages-of-logistic-regression/ <br>
