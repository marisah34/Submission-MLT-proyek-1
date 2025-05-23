# Laporan Proyek Machine Learning Terapan - Marisah Lofiana

## 📌 Domain Proyek
![c4d50819-270f-4e09-8a8b-ea46136d5caf](https://github.com/user-attachments/assets/752b8a79-1b5b-4398-9383-bcfabd89ba57)

Anemia merupakan kondisi medis yang ditandai dengan kadar hemoglobin dalam darah yang lebih rendah dari nilai normal, sehingga menghambat kemampuan darah dalam mengangkut oksigen ke jaringan tubuh. Dampak dari kondisi ini tidak dapat disepelekan karena dapat menurunkan kualitas hidup, mengganggu produktivitas, serta meningkatkan risiko komplikasi serius, seperti penyakit jantung. Anemia sering kali disebabkan oleh kekurangan zat besi, vitamin B12, atau gangguan kesehatan lainnya. Berdasarkan data World Health Organization (WHO), pada tahun 2023 sekitar 30,7% wanita usia 15–49 tahun mengalami anemia, dengan 35,5% di antaranya merupakan ibu hamil. Selain itu, pada tahun 2019 tercatat bahwa 39,8% anak-anak usia 6–59 bulan di dunia juga mengalami kondisi ini.

WHO juga mencatat bahwa lebih dari 30% populasi global terdampak oleh anemia, menjadikannya salah satu masalah kesehatan masyarakat yang mendesak. Kelompok yang paling rentan adalah anak-anak dan perempuan usia produktif, karena kondisi ini dapat berdampak besar terhadap pertumbuhan, kesehatan reproduksi, dan angka kematian. Oleh karena itu, deteksi dini terhadap anemia sangat penting agar pasien bisa segera mendapatkan penanganan medis yang sesuai.

Dalam praktik klinis, diagnosis anemia umumnya dilakukan melalui uji laboratorium untuk memeriksa berbagai parameter darah seperti hemoglobin dan jumlah sel darah merah. Namun, di daerah dengan akses terbatas terhadap fasilitas medis, proses ini bisa menjadi lambat dan mahal. Selain itu, keterlambatan dalam diagnosis dapat menyebabkan perburukan kondisi pasien.

Sebagai solusi atas tantangan ini, pendekatan berbasis teknologi seperti machine learning dapat dimanfaatkan untuk membangun model prediktif yang mampu mendeteksi anemia secara lebih cepat dan efisien. Dengan memanfaatkan data medis seperti kadar hemoglobin, jumlah sel darah merah, serta parameter lainnya seperti MCV (Mean Corpuscular Volume), MCH (Mean Corpuscular Hemoglobin), dan MCHC (Mean Corpuscular Hemoglobin Concentration), algoritma klasifikasi dapat dilatih untuk mengidentifikasi pola yang mengindikasikan kemungkinan seseorang menderita anemia.

Proyek ini bertujuan mengembangkan model machine learning yang mampu memprediksi status anemia seseorang hanya berdasarkan data hasil pemeriksaan darah yang relatif sederhana. Dengan demikian, tenaga medis, terutama di daerah dengan keterbatasan fasilitas, dapat dibantu dalam proses pengambilan keputusan medis secara lebih cepat dan akurat. Pendekatan ini diharapkan dapat mempercepat proses diagnosis, memungkinkan penanganan lebih awal, dan pada akhirnya berkontribusi pada peningkatan kualitas layanan kesehatan masyarakat.

## 📌 Business Understanding
### 📖 Problem Statements
Berdasarkan apa yagn telah dipaparkan dalam latar belakang diatas, maka dapat disimpulkan rumusan masalah yang dirumuskan sebagai berikut: 
- Bagaimana membangun model machine learning untuk memprediksi status anemia berdasarkan parameter darah?
- Apakah hemoglobin menjadi fitur yang paling berpengaruh dalam menentukan status anemia?
- Algoritma klasifikasi mana yang paling efektif untuk digunakan dalam kasus ini?
### 📖 Goals 
Berdasarkan Rumusan Masalah diatas, maka proyek kali ini memiliki tujuan yaitu sebagai berikut: 
- Mengembangkan model prediktif untuk mengklasifikasi status anemia.
- Mengidentifikasi fitur paling relevan dalam prediksi anemia.
- Membandingkan performa beberapa algoritma klasifikasi untuk memilih model terbaik.
### 📖 Solution Statements
Dengan dipaparkan tujuan dari dibuatnya proyek ini diatas, maka proyek ini memiliki beberapa solusi sebagai berikut: 
- Membangun dan membandingkan performa dari beberapa algoritma seperti Logistic Regression, Random Forest, dan K-Nearest Neighbors.
- Melakukan evaluasi model berdasarkan metrik akurasi, precision, recall, dan F1-score untuk memilih model terbaik.
- Melakukan visualisasi dan analisis korelasi untuk memahami peran setiap fitur dalam prediksi.

## 📌 Data Understanding
Dataset yang digunakan dalam proyek ini adalah [Anemia Dataset](https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset) yang diperoleh dari Kaggle. Dataset ini berisi informasi medis tentang berbagai individu, termasuk data darah yang digunakan untuk mendiagnosis anemia yang berisi 1421 1.421 baris dan 6 kolom. Tidak terdapat missing value, namun terdapat 887 baris duplikat yang dihapus. Outlier ditemukan hanya pada kolom Hemoglobin. Dataset ini digunakan untuk memprediksi apakah seorang pasien menderita anemia berdasarkan beberapa fitur darah dan informasi jenis kelamin. Dataset ini digunakan dengan tujuan untuk mengklasifikasikan individu menjadi dua kategori: anemia dan tidak anemia.

### Variabel-variabel pada Anemia dataset:
| # | Column | Dtype |
| ------ | ------ | ------ |
| 0 | Gender | int64 |
| 1 | Hemoglobin | float64 |
| 2 | MCH | float64 |
| 3 | MCHC | float64 |
| 4 | MCV | float64 |
| 5 | Result | int64 |

Dataset yang digunakan merupakan hasil pemeriksaan darah individu dengan atribut:

- Gender: 0 = Male, 1 = Female
- Hemoglobin: kadar hemoglobin
- MCH: Mean Corpuscular Hemoglobin
- MCHC: Mean Corpuscular Hemoglobin Concentration
- MCV: Mean Corpuscular Volume
- Result: 0 = Tidak Anemia, 1 = Anemia (label klasifikasi)

Seluruh kolom dalam dataset memiliki tipe data numerik, dengan empat fitur berupa float64 yaitu Hemoglobin, MCH, MCHC, dan MCV, serta dua fitur bertipe int64, yakni Gender dan Result. Berdasarkan hal tersebut, dapat disimpulkan bahwa tipe data pada setiap kolom sudah sesuai sehingga proses encoding tambahan tidak diperlukan untuk pelatihan model. Meski demikian, saat proses Exploratory Data Analysis (EDA), kolom Gender dan Result akan diubah sementara menjadi tipe kategorikal guna memudahkan analisis visualisasi, lalu dikembalikan ke bentuk semula sebelum pemodelan dilakukan.

![Screenshot 2025-05-22 132025](https://github.com/user-attachments/assets/c5ceee4f-03ae-49e1-813f-9968089466e6)
Berdasarkan statistik deskriptif, kolom Gender dan Result menunjukkan distribusi yang cukup seimbang antara kelas 0 dan 1. Sementara itu, fitur numerik seperti Hemoglobin, MCH, MCHC, dan MCV memiliki rentang nilai dan standar deviasi yang relatif besar, mencerminkan adanya variasi signifikan antar individu. Hal ini wajar mengingat dataset mencakup kondisi fisiologis yang berbeda antara individu yang menderita anemia dan yang tidak.

### Pengecekan Missing Value

Untuk pengecekan missing values, kode berikut digunakan:
  ```python
  # Memeriksa missing value
  df.isnull().sum()
  ```
![image](https://github.com/user-attachments/assets/bf1f9c47-29f0-4ae3-b1af-b4cd7b5d92c4)

Dikarekan dalam Dataset ini tidak terdapat missing value jadi tidak diperlukan penanganan missing value. 

### Pengecekan Data Duplikat 

Langkah selanjutnya adalah mengecek apakah ada data duplikat dalam dataset. 

 Untuk pengecekan data duplikat, kode berikut digunakan:
  ```python
  # Memeriksa duplikasi data
  jumlah_duplikat = df.duplicated().sum()
    print(f"Jumlah baris duplikat: {jumlah_duplikat}")
  ```
  Pada dataset ini, ditemukan 887 baris duplikat.

  ### Pengecekan Outlier 

Untuk mendeteksi outlier atau nilai ekstrem, teknik boxplot dan IQR digunakan untuk mengidentifikasi data yang berada di luar batas normal distribusi. Penananganan outlier diperlukan karena outlier yang tidak sesuai dengan pola data dapat mengganggu model, menghasilkan prediksi yang tidak akurat, dan menyebabkan overfitting.
![Screenshot 2025-05-22 141631](https://github.com/user-attachments/assets/89a4aa23-e060-4217-8f26-fa2ccff59072)

Dari hasil deteksi outlier menggunakan metode Interquartile Range (IQR), ditemukan hanya satu nilai outlier pada fitur Hemoglobin. Variasi kadar hemoglobin antar individu memang wajar terjadi, terutama dalam kaitannya dengan deteksi anemia. Karena kadar hemoglobin yang berada di luar batas normal bisa mengindikasikan kondisi medis tertentu, maka nilai outlier tersebut tetap dipertahankan. Hal ini penting karena data tersebut dapat menjadi informasi diagnostik yang bernilai, sekaligus membantu model dalam mengenali kasus medis yang jarang terjadi.

### Exploratory Data Analysis
#### Univariate Analysis
- **Analisis Distribusi Data Kategorikal**
  ![Screenshot 2025-05-22 161529](https://github.com/user-attachments/assets/a472c221-5aae-4c0e-bc70-a96bc5b06cd2)

Berdasarkan hasil visualisasi, terlihat bahwa jumlah individu berjenis kelamin perempuan lebih besar dibandingkan laki-laki, dengan dominasi yang cukup signifikan pada kategori perempuan. Sementara itu, pada variabel Result, jumlah individu yang tidak mengalami anemia tampak lebih banyak dibandingkan dengan yang terdiagnosis anemia, di mana kategori Not Anemic mendominasi. Meskipun secara umum distribusi data terlihat cukup seimbang, perbedaan jumlah antara kelas Anemic dan Not Anemic tetap menjadi perhatian. Oleh karena itu, dalam proses pembangunan model, penting untuk memperhitungkan ketimpangan kelas ini agar model yang dihasilkan tidak condong terhadap kelas yang jumlahnya lebih besar dan tetap memiliki performa yang baik.
- **Analisis Distribusi Data Numerik**
  ![image](https://github.com/user-attachments/assets/48052017-faaa-45ef-895f-4c0a0d19eecb)

  Berdasarkan hasil analisis distribusi melalui histogram, kadar Hemoglobin menunjukkan pola distribusi yang condong ke kiri (left-skewed), dengan mayoritas nilai berada dalam rentang normal 10 hingga 16 g/dL. Namun, terdapat sejumlah individu dengan kadar Hemoglobin yang sangat rendah, mengindikasikan kemungkinan kondisi anemia pada kelompok tersebut. Sementara itu, variabel MCH, MCHC, dan MCV menunjukkan sebaran yang lebih merata. Nilai MCH terdistribusi relatif simetris dengan konsentrasi tertinggi di rentang 20–22, sedangkan MCHC cenderung terkonsentrasi pada kisaran 28–32. Untuk MCV, distribusinya lebih bervariasi dan mencakup rentang yang cukup luas, terutama antara 70 hingga 100, yang mencerminkan perbedaan ukuran volume sel darah merah antar individu.
#### Bivariate Analysis
![image](https://github.com/user-attachments/assets/3b8df5c4-c22b-4107-a475-b235e739e7f9)

Berdasarkan boxplot, distribusi nilai MCH pada individu yang menderita anemia dan yang tidak terlihat cukup mirip, dengan tingkat variasi yang hampir setara. Hal serupa juga tampak pada distribusi MCV dan MCHC, di mana kedua kelompok menunjukkan pola yang relatif sebanding, meskipun terdapat sedikit perbedaan. Hal ini mengindikasikan bahwa meskipun MCH, MCV, dan MCHC tetap relevan sebagai fitur, perbedaan antara kelompok Anemic dan Not Anemic tidak terlalu mencolok pada ketiga variabel tersebut. Sebaliknya, fitur Hemoglobin menunjukkan perbedaan yang paling jelas, di mana individu dengan anemia memiliki kadar hemoglobin yang secara signifikan lebih rendah dibandingkan mereka yang tidak mengalami anemia.
#### Multivarate Analysis
- **Analisis Distribusi**
  ![image](https://github.com/user-attachments/assets/a5fd4a03-c584-43ef-a3a3-3029390a6bf7)

  Hasil eksplorasi lanjutan terhadap distribusi Hemoglobin berdasarkan gender dan status anemia menunjukkan bahwa perempuan dengan anemia (ditandai kotak merah) memiliki kadar Hemoglobin yang lebih rendah dibandingkan laki-laki dengan kondisi serupa. Menariknya, perempuan tanpa anemia (kotak biru) justru menunjukkan kadar Hemoglobin yang cenderung lebih tinggi daripada laki-laki yang juga tidak menderita anemia. Secara keseluruhan, individu yang terdiagnosis anemia—baik laki-laki maupun perempuan—memiliki kadar Hemoglobin yang secara signifikan lebih rendah dibandingkan mereka yang sehat. Temuan ini menegaskan bahwa Hemoglobin merupakan fitur krusial dalam identifikasi anemia, dan adanya perbedaan berdasarkan jenis kelamin, di mana perempuan cenderung memiliki kadar yang lebih rendah, mencerminkan karakteristik umum dari kondisi anemia.
- **Analisis Korelasi antar Fitur**
  ![image](https://github.com/user-attachments/assets/1aafe8c4-b0e1-4597-91fe-ba6ac707af83)

  Berdasarkan matriks korelasi, terdapat hubungan positif sedang antara Gender dan status anemia (Result), dengan nilai sekitar 0,25. Hal ini menunjukkan kecenderungan bahwa perempuan sedikit lebih berisiko mengalami anemia, sejalan dengan temuan medis yang umum. Di sisi lain, Hemoglobin menunjukkan korelasi negatif yang sangat kuat dengan Result (-0,80), mengindikasikan bahwa semakin rendah kadar Hemoglobin, semakin besar kemungkinan individu menderita anemia. Ini menegaskan peran penting Hemoglobin sebagai indikator utama dalam diagnosis anemia. Sementara itu, korelasi antara MCH, MCHC, dan Result tergolong lemah, sehingga kontribusinya terhadap prediksi lebih kecil dibanding Hemoglobin. Namun, mengingat keterbatasan jumlah data, semua fitur tetap dipertahankan agar tidak mengurangi informasi yang dapat digunakan oleh model. Dengan demikian, Hemoglobin menjadi fitur paling signifikan dalam pemodelan, sedangkan MCH, MCV, dan MCHC tetap dimasukkan untuk melengkapi informasi meski pengaruhnya relatif lebih kecil.
## 📌 Data Preparation
- **Pengahapusan Data Duplikat**
  
  Data duplikat bisa muncul karena kesalahan pengumpulan atau input data. Baris yang identik di setiap fitur akan diidentifikasi dan dihapus untuk menghindari model terlalu menekankan informasi yang sama, yang dapat menyebabkan overfitting atau kesalahan saat pelatihan. Dengan demikian, penghapusan data duplikat memastikan model belajar dari data yang unik dan relevan.
  Untuk pengahapusan data duplikat, kode yang digunakan yaitu:
   ```python
    # Menghapus baris duplikat
    df = df.drop_duplicates()
    ```
  Sisa data setelah pembersihan baris duplikat adalah 534.
- **Data Splitting**
  
  Membagi dataset menjadi dua bnagian yaitu data tranning dan testing. Data training akan digunakan untuk melatih model, sedangkan data testing akan digunakan untuk mengevaluasi kinerja model yang sudah dibangun. Pemisahan data ini penting untuk menghindari overfitting dan memastikan model dapat diuji pada data yang tidak digunakan selama proses pelatihan.
  Berikut adalah data setelah dilakukan proses splitting
  
  |Data              |Jumlah  |
  |------------------|--------|
  |Data Keseluruhan  |534     |
  |Data Train        |427     |
  |Data Test         |107     |
  
- **Penanganan Imbalance Classes**
  
  Karena variabel target Result menunjukkan ketidakseimbangan kelas, di mana jumlah individu Not Anemic lebih dominan, perlu dilakukan penanganan agar model tidak bias terhadap kelas mayoritas. Salah satu metode yang efektif adalah SMOTE (Synthetic Minority Over-sampling Technique), yang menghasilkan sampel sintetis dari kelas Anemic untuk menyeimbangkan distribusi. Tanpa penanganan ini, model cenderung mengabaikan kelas minoritas, sehingga akurasi tidak mencerminkan performa sebenarnya. Oleh karena itu, penggunaan SMOTE membantu menciptakan model yang lebih seimbang, adil, dan akurat. Berikut ditampilkan perbandingan data sebelum dan sesudah diterapkan SMOTE.
  
  ![image](https://github.com/user-attachments/assets/acb64db7-662f-4207-811f-566433db50cc)
- **Feature Scaling**
  
  Scaling digunakan untuk menyamakan skala antar fitur agar tidak ada fitur dengan nilai besar yang mendominasi model. Dalam proyek ini, digunakan metode standarisasi karena distribusi data mendekati normal. Proses ini dilakukan menggunakan StandardScaler() dari sklearn, yang mengubah nilai fitur agar memiliki rata-rata nol dan standar deviasi satu.

  Penting untuk dicatat, standarisasi hanya diterapkan pada data latih guna mencegah kebocoran informasi ke data uji. Berikut ditampilkan hasil dari proses standarisasi.
  ![image](https://github.com/user-attachments/assets/51712d09-6de6-467c-9d14-0468350b533c)

## Modeling 

Pada tahap ini, beberapa algoritma machine learning digunakan untuk menyelesaikan klasifikasi anemia, yaitu Random Forest, Decision Tree, Logistic Regression, dan K-Nearest Neighbors. Awalnya, setiap model dilatih menggunakan parameter default pada data latih yang telah diskalakan dan diseimbangkan (X_train_scaled dan y_train_resampled), untuk memperoleh baseline atau kinerja awal model tanpa optimasi.

Selanjutnya, dilakukan proses tuning hyperparameter menggunakan GridSearchCV guna mencari kombinasi parameter terbaik yang mampu meningkatkan performa model. Setelah kombinasi optimal ditemukan, model dilatih ulang menggunakan parameter tersebut dan dievaluasi kembali untuk melihat peningkatan hasil dibandingkan dengan versi default.

Berikut adalah penjelasan dari model yang akan digunakan:

### 1. Model Random Forest (RF)
![random_forest](https://github.com/user-attachments/assets/6f585d3b-b922-4c1e-84ca-15ed3710169a)

Random forest adalah algoritma machine learning yang umum digunakan, dengan merek dagang milik Leo Breiman dan Adele Cutler, yang menggabungkan hasil dari beberapa decision trees untuk mencapai satu hasil. Algoritma ini banyak diadopsi karena mudah digunakan dan fleksibel, dan mampu menangani masalah klasifikasi dan regression.

Berikut merupakan code pelatihan model: 
```python
rf = RandomForestClassifier().fit(X_train_scaled, y_train_resampled)
```

**Tahapan:**

**Random Forest (RF)** adalah algoritma ensemble yang menggunakan teknik pembelajaran berbasis pohon keputusan. Proses ini dimulai dengan **pembagian data secara acak** menjadi beberapa subset yang berbeda. Setiap subset data kemudian digunakan untuk membangun **beberapa pohon keputusan** secara terpisah. Setiap pohon keputusan ini dilatih pada data yang berbeda, dengan beberapa fitur yang dipilih secara acak pada setiap percabangan untuk meningkatkan keragaman antar pohon. Setelah semua pohon selesai dilatih, algoritma **menggunakan mayoritas suara** (voting) dari seluruh pohon untuk menghasilkan prediksi akhir. Dengan cara ini, Random Forest mengurangi risiko overfitting yang sering terjadi pada pohon keputusan tunggal dan meningkatkan akurasi model dengan menggabungkan prediksi dari banyak pohon keputusan yang berbeda. 

**Parameter yang Digunakan:**

- `n_estimators`: Jumlah pohon dalam hutan.
- `max_depth`: Kedalaman maksimum pohon.
- `min_samples_split`: Jumlah minimum sampel yang diperlukan untuk membagi internal pohon.
- `min_samples_leaf`: Jumlah minimum sampel yang diperlukan di setiap daun pohon.

**Kelebihan:**

- Mengurangi risiko **overfitting** dibandingkan pohon keputusan tunggal.
- Cocok untuk dataset besar dan dapat menangani data yang tidak linier.
- Memberikan nilai **feature importance** yang berguna untuk analisis lebih lanjut.

**Kekurangan:**

- Proses pelatihan lebih lama jika jumlah pohon sangat besar.
- Kurang interpretatif dibandingkan dengan model pohon keputusan tunggal.

### 1. Model Decision Tree
![decision-tree-1](https://github.com/user-attachments/assets/06dafe10-41e9-4a0a-81fc-1cc978bd7bff)

Decision tree adalah sebuah metode atau algoritma pembelajaran yang diawasi (supervised learning) dan bersifat non-parametrik yang digunakan untuk memprediksi atau memodelkan data. Ia menggunakan struktur pohon hierarkis untuk membuat keputusan berdasarkan serangkaian aturan, mulai dari simpul akar hingga simpul daun. Decision tree dapat digunakan untuk tugas klasifikasi dan regresi.

Berikut merupakan code pelatihan model: 
```python
dt = DecisionTreeClassifier().fit(X_train_scaled, y_train_resampled)
```
