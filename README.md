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

## 📌 Modeling 

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

### 2. Model Decision Tree
![decision-tree-1](https://github.com/user-attachments/assets/06dafe10-41e9-4a0a-81fc-1cc978bd7bff)

Decision tree adalah sebuah metode atau algoritma pembelajaran yang diawasi (supervised learning) dan bersifat non-parametrik yang digunakan untuk memprediksi atau memodelkan data. Ia menggunakan struktur pohon hierarkis untuk membuat keputusan berdasarkan serangkaian aturan, mulai dari simpul akar hingga simpul daun. Decision tree dapat digunakan untuk tugas klasifikasi dan regresi.

Berikut merupakan code pelatihan model: 
```python
dt = DecisionTreeClassifier().fit(X_train_scaled, y_train_resampled)
```

**Tahapan** 

**Decision Tree** merupakan metode machine learning yang membentuk model berbentuk pohon keputusan untuk melakukan prediksi berdasarkan fitur-fitur input. Proses diawali dengan memilih fitur yang paling efektif dalam membagi dataset, menggunakan kriteria seperti Gini Impurity atau Entropy. Pembagian ini dilakukan secara berulang pada setiap node, sehingga data terbagi menjadi subset yang semakin homogen. Pemilihan fitur terbaik untuk setiap node dilakukan secara rekursif, hingga mencapai kedalaman maksimum pohon atau ketika tidak ada lagi pembagian yang lebih optimal. Proses tersebut berlanjut sampai model mencapai kondisi di mana pemisahan lebih lanjut tidak memberikan keuntungan, atau mencapai batas penghentian yang telah ditentukan, seperti jumlah minimum sampel dalam satu node atau batas kedalaman pohon.

**Parameter yang Digunakan:**

- `max_depth`: Batas maksimal kedalaman pohon.

- `min_samples_split`: Jumlah minimum data yang diperlukan untuk membagi node.

- `min_samples_leaf`: Jumlah minimum data yang harus ada pada setiap daun pohon.

- `criterion`: Metode evaluasi untuk menentukan kualitas pemisahan data (misalnya gini atau entropy).

**Kelebihan**

- Mudah dipahami dan dijelaskan.

- Cepat dalam proses pelatihan dan prediksi.

- Mampu menangani tipe data numerik maupun kategorikal.

**Kekurangan**

- Berisiko mengalami overfitting, terutama jika pohon terlalu dalam.

- Kurang stabil jika diterapkan pada data yang mengandung noise.

### 3. Model Logistic Regression
![images](https://github.com/user-attachments/assets/09c263aa-f387-489e-b3b0-f847533091ad)


Logistic regression adalah metode analisis statistik untuk memprediksi hasil biner, seperti ya atau tidak, berdasarkan pengamatan sebelumnya dari kumpulan data.

Berikut merupakan code pelatihan model: 
```python
lr = LogisticRegression().fit(X_train_scaled, y_train_resampled)
```

**Logistic Regression** adalah algoritma klasifikasi yang digunakan untuk memprediksi probabilitas suatu kejadian dengan model berbasis fungsi logit. Prosesnya dimulai dengan menghitung bobot awal untuk masing-masing fitur, kemudian memperbarui bobot tersebut menggunakan teknik optimasi seperti Gradient Descent guna meminimalkan fungsi kerugian, yaitu log loss. Model ini menghasilkan nilai probabilitas, yang kemudian diklasifikasikan ke dalam kelas tertentu berdasarkan ambang batas (threshold) yang telah ditentukan.

**Parameter yang Digunakan:**

- `penalty`: Jenis regulasi yang digunakan untuk mencegah overfitting (misalnya l1, l2, atau elasticnet).

- `C`: Parameter regulasi yang mengontrol kekuatan regularisasi (nilai lebih kecil = regulasi lebih kuat).

- `solver`: Metode optimasi yang digunakan untuk memperkirakan parameter (misalnya liblinear, saga, atau lbfgs).

- `max_iter`: Jumlah iterasi maksimum saat proses optimasi berlangsung.

**Kelebihan**

- Proses pelatihan cepat dan sederhana.

- Memberikan probabilitas prediksi, sehingga bisa digunakan untuk berbagai analisis risiko.

- Cocok untuk data dengan jumlah fitur tidak terlalu banyak.

**Kekurangan**

- Kurang efektif untuk masalah non-linear tanpa transformasi fitur.

- Sensitif terhadap outlier.

### 4. Model K-Nearest Neighbors (KNN)

Algoritma k-tetangga terdekat (KNN) adalah pengklasifikasi pembelajaran non-parametrik dan terawasi, yang menggunakan kedekatan untuk membuat klasifikasi atau prediksi tentang pengelompokan titik data individu.

Berikut adalah kode pelatihan model.

```python
knn = KNeighborsClassifier().fit(X_train_scaled, y_train_resampled)
```

**Tahapan**

**K-Nearest Neighbors** merupakan algoritma klasifikasi yang melakukan prediksi berdasarkan kedekatan data. Saat diberikan data baru, algoritma ini menghitung jarak antara data tersebut dengan seluruh data pada dataset pelatihan, menggunakan metrik seperti Euclidean atau Manhattan distance. Data kemudian diklasifikasikan berdasarkan mayoritas label dari k tetangga terdekatnya. Proses ini tidak memerlukan pelatihan model sebelumnya, sehingga termasuk metode lazy learning.

**Parameter yang Digunakan**

- `n_neighbors`: Jumlah tetangga terdekat yang akan digunakan untuk menentukan label.

- `weights`: Bobot yang diberikan pada tetangga (misalnya uniform atau distance).

- `metric`: Metode perhitungan jarak antar data (seperti euclidean, manhattan, atau minkowski).

**Kelebihan**

- Sederhana dan mudah diimplementasikan.

- Dapat digunakan untuk klasifikasi maupun regresi.

- Tidak memerlukan asumsi khusus terhadap distribusi data.

**Kekurangan**

- Lambat pada dataset berukuran besar karena harus menghitung jarak ke seluruh data.

- Sensitif terhadap skala fitur dan nilai k yang dipilih.

- Rentan terhadap outlier.

## 📌 Hyperparameter Tuning

Agar model dapat bekerja lebih optimal, dilakukan pencarian kombinasi hyperparameter terbaik melalui proses **hyperparameter tuning**. **GridSearchCV** digunakan untuk menguji berbagai kemungkinan nilai parameter dan menentukan yang paling optimal.

Contoh kode tuning untuk **Random Forest**:

```python
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5)
grid_rf.fit(X_train_scaled, y_train_resampled)

print("Best Params for Random Forest:", grid_rf.best_params_)
best_rf = grid_rf.best_estimator_
```

Berikut adalah parameter yang digunakan untuk pelatihan tiap model hasil dari hyperparameter tuning
![image](https://github.com/user-attachments/assets/f68bfde4-4056-4fd6-8641-9d4b58c2de49)

### 📌 Pemilihan Model Terbaik

Random Forest (RF) dipilih sebagai metode terbaik untuk menganalisis dataset anemia yang terdiri dari 534 data karena algoritma ini mampu meminimalkan risiko overfitting menggunakan pendekatan ensemble, sehingga model yang dihasilkan lebih andal dan akurat. Selain bisa mengatasi hubungan non-linier antar fitur, RF juga memberikan informasi tentang pentingnya tiap fitur untuk membantu analisis lanjutan. Keunggulan lainnya, model ini tidak terpengaruh oleh skala fitur, bisa langsung menangani data yang hilang tanpa perlu standarisasi, dan tetap menghasilkan performa yang stabil meskipun jumlah datanya relatif sedikit. Meski begitu, hasil evaluasi tetap harus diperhatikan dengan memeriksa metrik seperti accuracy, precision, recall, dan F1-score guna memastikan performa model tetap optimal.

## 📌 Evaluasi 

Pada tahap evaluasi model, digunakan metrik **Akurasi, Precision, Recall, dan F1-Score** untuk menilai kinerja model. Pemilihan metrik ini didasarkan pada relevansinya terhadap permasalahan klasifikasi biner dalam proyek ini, yakni untuk memprediksi apakah seorang individu teridentifikasi mengalami anemia atau tidak.

**1. Akurasi**

Akurasi adalah Mengukur seberapa banyak prediksi yang benar dibandingkan dengan total jumlah prediksi. Semakin tinggi akurasi, semakin banyak prediksi yang tepat.

 $$
  \text{Akurasi} = \frac{\text{True Positives} + \text{True Negatives}}{\text{Total Observations}}
  $$

Dimana **True Positives (TP)** adalah jumlah individu yang memang mengalami anemia dan berhasil diprediksi dengan benar sebagai penderita anemia, sedangkan **True Negatives (TN)** merupakan jumlah individu yang sebenarnya tidak mengalami anemia dan diprediksi dengan tepat sebagai tidak menderita anemia. Nilai akurasi yang tinggi menunjukkan bahwa model mampu melakukan prediksi yang benar terhadap sebagian besar data secara keseluruhan.
  
**2. Precision**

Precision adalah Menghitung proporsi data positif yang diprediksi benar dari semua data yang diprediksi positif. Artinya, seberapa tepat model saat memprediksi kasus positif.

 $$
  \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
  $$

Precision yang tinggi menunjukkan bahwa model memiliki ketelitian yang baik dalam mengklasifikasikan individu sebagai penderita anemia, dengan jumlah kesalahan klasifikasi positif (false positives) yang lebih sedikit. Artinya, ketika model memprediksi seseorang menderita anemia, kemungkinan besar prediksi tersebut benar. Tingginya precision penting dalam situasi di mana kesalahan dalam memberikan diagnosis harus dihindari, karena dapat berdampak serius jika individu sehat didiagnosis secara keliru.

**3. Recall**

Recall adalah Mengukur seberapa banyak data positif yang berhasil diprediksi benar dari seluruh data positif yang ada. Semakin tinggi recall, semakin sedikit kasus positif yang terlewat.

 $$
  \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
  $$

Recall yang tinggi berarti model berhasil mendeteksi sebagian besar individu yang memang mengalami anemia, meskipun mungkin masih terdapat sejumlah kesalahan prediksi negatif (false negatives). Dalam konteks medis, recall yang tinggi lebih diutamakan karena penting untuk menangkap semua pasien yang benar-benar sakit, dibandingkan sekadar mengurangi jumlah prediksi positif yang salah. Ini bertujuan meminimalkan kemungkinan penderita anemia yang tidak terdeteksi.

**4. F1-Score**

F1-Score adalah Rata-rata harmonis dari precision dan recall. F1-Score digunakan untuk menyeimbangkan keduanya, khususnya saat dataset tidak seimbang (imbalanced).

$$
  \text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
  $$

F1-Score yang tinggi menunjukkan bahwa model memiliki keseimbangan yang baik antara ketepatan prediksi (precision) dan kemampuan dalam mendeteksi seluruh kasus positif (recall). Nilai F1-Score yang baik menandakan bahwa model tidak hanya akurat dalam memprediksi penderita anemia, tetapi juga mampu mengenali sebagian besar individu yang benar-benar sakit. Hal ini sangat penting dalam bidang kesehatan, karena memastikan prediksi tepat dan tidak banyak kasus yang terlewat.
 
## 📌 Hasil Proyek Berdasarkan Metrik Evaluasi

Setelah proses pelatihan dan evaluasi model melalui cross-validation, diperoleh hasil performa yang bervariasi tergantung algoritma yang diterapkan. Secara umum, model yang telah melalui proses penyetelan hyperparameter menunjukkan peningkatan performa meskipun tidak terlalu signifikan dibandingkan dengan kondisi sebelum tuning.
![image](https://github.com/user-attachments/assets/b1d2a3b5-5066-42df-9c79-a96d1097233c)

Berikut adalah ringkasan metrik evaluasi untuk model yang diuji sebelum dilakukan hyperparameter tuning:
![image](https://github.com/user-attachments/assets/fc6ce0f7-1d56-417d-ac19-a3f15fe1c9a9)

- **Random Forest (RF)**
    - Akurasi: 99.07%
    - Precision: 99.08%
    - Recall: 99.07%
    - F1-Score: 99.07%
  
  Penjelasan:

Random Forest menunjukkan performa terbaik dalam mendeteksi anemia, dengan nilai akurasi, precision, dan recall yang sangat tinggi. Model ini mampu mengenali hampir seluruh individu yang benar-benar menderita anemia, sekaligus menjaga tingkat kesalahan prediksi tetap rendah. Keseimbangan antara ketepatan prediksi positif dan kemampuan mendeteksi semua kasus anemia membuat Random Forest menjadi pilihan yang sangat unggul untuk konteks medis seperti ini.

- **Decision Tree (DT)**
    - Akurasi: 97.20%
    - Precision: 97.36%
    - Recall: 97.20%
    - F1-Score: 97.20%

  Penjelasan:

  Decision Tree masih memberikan performa yang sangat baik, meskipun sedikit di bawah Random Forest. Precision yang tinggi menunjukkan bahwa sebagian besar prediksi positifnya akurat. Namun, nilai recall yang sedikit lebih rendah dibandingkan Random Forest menunjukkan bahwa model ini sedikit lebih sering melewatkan individu yang benar-benar mengalami anemia. Meskipun demikian, hasilnya tetap solid dan dapat diandalkan.
  
- **Logistic Regression (LR)**
    - Akurasi: 96.26%
    - Precision: 96.54%
    - Recall: 96.26%
    - F1-Score: 96.27%

  Penjelasan:

  Logistic Regression memperlihatkan performa yang cukup baik dengan nilai precision dan recall yang seimbang di angka sekitar 96%. Model ini mampu melakukan prediksi positif dengan baik dan cukup efektif dalam mendeteksi penderita anemia. Meski performanya tidak sebaik Random Forest dan Decision Tree, model ini tetap layak dipertimbangkan, apalagi karena interpretasinya yang sederhana dalam dunia medis.

- **K-Nearest Neighbors (KNN):**
    - Akurasi: 90.65%
    - Precision: 91.69%
    - Recall: 90.66%
    - F1-Score: 90.66%

  Penjelasan:

  KNN menunjukkan performa paling rendah dibanding model lainnya. Meskipun precision-nya cukup baik, nilai akurasi, recall, dan F1-Score-nya di kisaran 90% menunjukkan bahwa model ini kurang optimal dalam mendeteksi semua individu yang benar-benar mengalami anemia. Nilai recall yang lebih rendah menandakan KNN cenderung melewatkan beberapa kasus positif. Ini bisa disebabkan oleh sensitivitas KNN terhadap jumlah data dan jarak antar data yang mempengaruhi hasil klasifikasinya.

Secara keseluruhan, **Random Forest** menjadi model yang paling unggul dalam hal akurasi, recall, dan keseimbangan metrik evaluasi, diikuti oleh **Decision Tree**, **Logistic Regression**, dan **K-Nearest Neighbors**. Model ini memberikan hasil yang optimal pada dataset yang diuji.

Berikut adalah ringkasan metrik evaluasi untuk model yang diuji **setelah dilakukan hyperparameter tuning**:
![image](https://github.com/user-attachments/assets/1893fed7-9d9e-4481-b7e1-57243f6cbbc7)

- **Random Forest (RF):**
    - Akurasi: 99.07%
    - Precision: 99.08%
    - Recall: 99.07%
    - F1-Score: 99.07%

  Penjelasan:

  Setelah dilakukan hyperparameter tuning, performa Random Forest tetap menjadi yang terbaik. Nilai akurasi, precision, recall, dan F1-Score yang sangat tinggi menunjukkan bahwa model ini mampu mendeteksi hampir seluruh kasus anemia dengan tingkat kesalahan prediksi yang sangat rendah. Keseimbangan antara precision dan recall yang tinggi sangat penting dalam konteks medis karena dapat meminimalkan risiko kesalahan diagnosis.

- **Decision Tree (DT):**
    - Akurasi: 99.07%
    - Precision: 99.08%
    - Recall: 99.07%
    - F1-Score: 99.07%

  Penjelasan:

  Menariknya, setelah hyperparameter tuning, Decision Tree mampu menyamai performa Random Forest dengan nilai metrik yang identik. Hal ini menunjukkan bahwa tuning parameter seperti kedalaman pohon dan jumlah minimum sampel dapat secara signifikan meningkatkan performa Decision Tree hingga setara Random Forest. Model ini menjadi alternatif yang baik karena lebih sederhana secara interpretasi dibanding Random Forest, tanpa harus mengorbankan performa.

- **Logistic Regression (LR):**
    - Akurasi: 97.20%
    - Precision: 97.36%
    - Recall: 97.20%
    - F1-Score: 97.20%

  Penjelasan:

  Logistic Regression juga mengalami peningkatan performa setelah tuning, meskipun masih di bawah Random Forest dan Decision Tree. Nilai precision dan recall yang seimbang di kisaran 97% menunjukkan model ini cukup andal dalam memprediksi dan mendeteksi kasus anemia. Kelebihan Logistic Regression tetap pada interpretasi model yang lebih mudah dipahami, yang bisa jadi nilai plus dalam dunia medis.

- **K-Nearest Neighbors (KNN):**
    - Akurasi: 90.65%
    - Precision: 91.69%
    - Recall: 90.65%
    - F1-Score: 90.66%

  Penjelasan:

  Setelah tuning, performa KNN tetap menjadi yang terendah di antara keempat model. Meskipun ada sedikit peningkatan, nilai recall yang masih di angka 90% menunjukkan bahwa model ini cenderung melewatkan beberapa kasus anemia. KNN memang sensitif terhadap pemilihan jumlah tetangga (k) dan skala data, yang bisa menjadi tantangan dalam pengaplikasian pada data medis.

Secara keseluruhan, **Random Forest** tetap menjadi model yang paling disarankan setelah tuning dengan nilai Recall yang sangat tinggi dan metrik evaluasi lainnya yang juga tinggi, diikuti oleh Decision Tree, dengan Logistic Regression dan KNN lebih cocok digunakan dalam skenario yang lebih sederhana.

## 📌 Model Terbaik Berdasarkan Metrik Evaluasi 

Dalam konteks diagnosis medis, nilai **recall** yang tinggi sangat penting karena fokus utamanya adalah mengidentifikasi sebanyak mungkin pasien yang benar-benar mengalami anemia. Kesalahan dalam bentuk **false negative**, yaitu kasus di mana penderita anemia tidak terdeteksi, dapat berdampak serius karena individu tersebut tidak akan mendapatkan penanganan yang dibutuhkan. Oleh sebab itu, model dengan performa recall yang baik, seperti **Random Forest**, menjadi pilihan utama karena mampu mengenali lebih banyak kasus anemia secara akurat. Selain itu, **Random Forest** menunjukkan hasil evaluasi model yang sangat baik. Walaupun **Decision Tree** memberikan hasil yang kompetitif, Random Forest dinilai sedikit lebih unggul berkat kemampuannya dalam meminimalkan risiko overfitting melalui pendekatan ensemble. Oleh karena itu, Random Forest dipilih sebagai model yang paling optimal dalam proyek ini.


## 📌 Evaluasi Pemahaman Bisnis

- Model klasifikasi yang dikembangkan **mampu mengidentifikasi apakah seseorang menderita anemia berdasarkan data medis sederhana** seperti kadar hemoglobin, MCH, MCV, dan MCHC. Penerapan algoritma machine learning seperti Random Forest, Decision Tree, Logistic Regression, serta K-Nearest Neighbors memungkinkan prediksi status anemia (Anemic atau Not Anemic) tanpa memerlukan pemeriksaan laboratorium yang mahal. Solusi ini memberikan alternatif yang lebih efisien, hemat biaya, dan lebih mudah dijangkau dibandingkan metode konvensional berbasis tes laboratorium.

- Melalui proses Exploratory Data Analysis (EDA), diketahui bahwa kadar **hemoglobin memiliki pengaruh paling signifikan dalam menentukan status anemia**. Hubungan yang sangat kuat antara hemoglobin dan status anemia menjadikan fitur ini sebagai indikator utama. Fitur lain seperti MCV dan MCHC juga memberikan kontribusi meskipun tidak sebesar hemoglobin.

- Kinerja dari model diukur melalui berbagai metrik seperti akurasi, presisi, recall, dan F1-score. Optimalisasi model dilakukan melalui teknik hyperparameter tuning menggunakan Grid Search untuk memperoleh konfigurasi parameter terbaik. Hasil tuning menunjukkan adanya peningkatan performa model pada seluruh metrik evaluasi, yang membuat model ini semakin andal dalam mendeteksi anemia secara tepat.

## 📌 Kesimpulan 

Dalam proyek ini, telah dikembangkan model machine learning untuk mendeteksi anemia menggunakan data medis sederhana seperti hemoglobin, MCV, MCH, dan MCHC. Berbagai algoritma seperti Random Forest, Logistic Regression, dan Decision Tree digunakan, serta performanya ditingkatkan melalui proses hyperparameter tuning dengan Grid Search. Berdasarkan hasil evaluasi, algoritma Random Forest menunjukkan kinerja paling unggul dalam memprediksi anemia dengan nilai akurasi, presisi, recall, dan F1-score yang tinggi. Model ini menawarkan alternatif diagnosis dini anemia yang lebih cepat dan terjangkau, sehingga dapat menjadi solusi efektif untuk mengatasi keterlambatan dalam proses deteksi penyakit tersebut.


