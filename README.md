# 📘 Wine Quality Prediction

Proyek prediksi kualitas wine menggunakan Machine Learning dan Deep Learning berdasarkan karakteristik physicochemical properties dari Portuguese "Vinho Verde" wine.

## 👤 Informasi
- **Nama:** Arisatya Abhirama
- **NIM:** 233307036
- **Repo:** [Link GitHub Repository]
- **Video:** [Link Video Penjelasan]

---

## 1. 🎯 Ringkasan Proyek
Proyek ini bertujuan untuk memprediksi kualitas wine (score 0-10) berdasarkan 11 fitur physicochemical. Pendekatan yang digunakan:
- Melakukan Exploratory Data Analysis (EDA) pada red dan white wine
- Melakukan data preparation dan feature engineering
- Membangun 3 model: **Linear Regression (Baseline)**, **XGBoost (Advanced)**, **Neural Network (Deep Learning)**
- Melakukan evaluasi dan menentukan model terbaik untuk prediksi kualitas wine

---

## 2. 📄 Problem & Goals

**Problem Statements:**
1. Bagaimana cara memprediksi kualitas wine secara akurat berdasarkan properti fisikokimia?
2. Fitur apa yang paling berpengaruh terhadap kualitas wine?
3. Apakah ada perbedaan karakteristik antara red wine dan white wine?
4. Model mana yang paling efektif untuk prediksi kualitas wine (regression task)?

**Goals:**
1. Membangun sistem prediksi kualitas wine dengan MAE < 0.6 dan R² > 0.35
2. Membandingkan performa 3 jenis model (baseline, advanced, deep learning)
3. Mengidentifikasi fitur-fitur penting yang menentukan kualitas wine
4. Membuat model yang reproducible dan dapat membantu wine producers meningkatkan kualitas produk

---

## 📁 Struktur Folder
```
wine-quality-prediction/
│
├── data/                      # Dataset
│   ├── winequality-red.csv   # Red wine data (1599 samples)
│   └── winequality-white.csv # White wine data (4898 samples)
│
├── notebooks/                 # Jupyter notebooks
│   └── ML_Project.ipynb      # Notebook utama proyek
│
├── src/                       # Source code (opsional)
│
├── models/                    # Saved models
│   ├── model_baseline.pkl    # Linear Regression
│   ├── model_xgboost.pkl     # XGBoost
│   └── model_nn.h5           # Neural Network
│
├── images/                    # Visualizations
│   ├── wine_distribution.png
│   ├── correlation_heatmap.png
│   ├── feature_distributions.png
│   ├── quality_distribution.png
│   ├── feature_importance.png
│   └── training_history.png
│
├── requirements.txt           # Dependencies
├── .gitignore
└── README.md
```

---

## 3. 📊 Dataset

- **Sumber:** [UCI Machine Learning Repository - Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Paper:** Cortez et al., 2009 - "Modeling wine preferences by data mining from physicochemical properties"
- **Jumlah Data:** 
  - Red Wine: 1,599 samples
  - White Wine: 4,898 samples
  - **Total: 6,497 samples**
- **Jumlah Fitur:** 11 input features + 1 output (quality)
- **Tipe:** Tabular Data (Regression Task)
- **Target:** Quality score (0-10, tetapi praktisnya 3-9)

### Fitur Utama
| Fitur | Tipe | Deskripsi | Satuan |
|-------|------|-----------|--------|
| fixed acidity | Float | Asam tetap | g/dm³ |
| volatile acidity | Float | Asam volatil (asam asetat) | g/dm³ |
| citric acid | Float | Asam sitrat | g/dm³ |
| residual sugar | Float | Gula sisa setelah fermentasi | g/dm³ |
| chlorides | Float | Kandungan garam | g/dm³ |
| free sulfur dioxide | Float | SO2 bebas | mg/dm³ |
| total sulfur dioxide | Float | Total SO2 | mg/dm³ |
| density | Float | Densitas wine | g/cm³ |
| pH | Float | Tingkat keasaman | 0-14 |
| sulphates | Float | Aditif wine | g/dm³ |
| alcohol | Float | Persentase alkohol | % vol |
| quality | Integer | Kualitas wine (target) | 0-10 |

---

## 4. 🔧 Data Preparation

### 4.1 Data Cleaning
- Menggabungkan red dan white wine dataset
- Menambah kolom 'wine_type' untuk membedakan red/white
- Handling outliers menggunakan IQR method
- Tidak ada missing values

### 4.2 Feature Engineering
- Normalisasi semua fitur numerik (StandardScaler)
- Encoding wine_type (Red=0, White=1)
- Feature interaction analysis
- Binning quality menjadi 3 kategori untuk analisis tambahan

### 4.3 Data Splitting
- Training set: 80% (5,197 samples)
- Test set: 20% (1,300 samples)
- Random state: 42 untuk reproducibility

---

## 5. 🤖 Modeling

### Model 1 – Baseline: Linear Regression
- Model sederhana untuk regression task
- Mudah diinterpretasi (coefficients)
- Baseline untuk perbandingan

### Model 2 – Advanced: XGBoost Regressor
- Gradient boosting algorithm
- Robust terhadap outliers dan missing values
- Feature importance analysis built-in
- Hyperparameter: n_estimators=200, max_depth=5, learning_rate=0.1

### Model 3 – Deep Learning: Neural Network (MLP)
- Multilayer Perceptron for regression
- Arsitektur: Input(11) → Dense(128, ReLU) → Dropout(0.3) → Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dense(1, Linear)
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam
- Training dengan early stopping dan learning rate reduction

---

## 6. 🧪 Evaluation

**Metrik:** MAE (Mean Absolute Error), MSE, RMSE, R² Score

### Hasil Perbandingan Model
| Model | MAE | RMSE | R² Score | Training Time |
|-------|-----|------|----------|---------------|
| Linear Regression | ~0.65 | ~0.85 | ~0.28 | <1s |
| XGBoost | ~0.48 | ~0.63 | ~0.48 | ~5s |
| Neural Network | ~0.52 | ~0.68 | ~0.42 | ~45s |

**Note:** Hasil dapat bervariasi tergantung data split dan hyperparameter tuning

---

## 7. 🏁 Kesimpulan

### Model Terbaik: XGBoost Regressor
- **MAE terendah (~0.48)**: Prediksi rata-rata meleset ±0.48 poin dari actual quality
- **R² Score tertinggi (~0.48)**: Menjelaskan 48% variansi data
- **Training time efisien**: Hanya ~5 detik untuk 6,497 samples

### Alasan:
1. XGBoost memberikan balance terbaik antara akurasi dan efisiensi
2. Feature importance membantu interpretasi hasil
3. Robust terhadap outliers dalam data wine
4. Neural Network akurasi bagus tapi training lebih lama

### Key Insights:
- **Fitur paling penting**: Alcohol, volatile acidity, sulphates, citric acid
- **Alcohol content** adalah prediktor terkuat kualitas wine
- **White wine** cenderung memiliki kualitas score lebih konsisten
- **Quality distribution** tidak seimbang (kebanyakan score 5-6)
- **Model regression** lebih sulit daripada classification karena granularity target

### Business Impact:
- Wine producers dapat fokus meningkatkan alcohol content dan mengurangi volatile acidity
- Prediksi membantu quality control sebelum wine dirilis ke pasar
- Cost-effective alternative untuk sensory evaluation oleh wine experts

---

## 8. 🔮 Future Work

- [x] Collect more data untuk extreme quality (score 3, 9, 10)
- [x] Try ensemble methods (stacking multiple models)
- [x] Feature engineering: polynomial features, interaction terms
- [x] Convert to classification task (Low/Medium/High quality)
- [ ] Hyperparameter optimization dengan Optuna/Bayesian Search
- [ ] Deploy model ke web application (Streamlit)
- [ ] Create REST API untuk wine quality prediction
- [ ] A/B testing dengan wine experts

---

## 9. 🔁 Reproducibility

### Instalasi Dependencies
```bash
pip install -r requirements.txt
```

### Menjalankan Project
```bash
# Clone repository
git clone [URL_REPO_TEMAN_ANDA]
cd wine-quality-prediction

# Install dependencies
pip install -r requirements.txt

# Download dataset dari UCI atau gunakan file yang sudah ada
# Letakkan winequality-red.csv dan winequality-white.csv di folder data/

# Jalankan notebook
jupyter notebook notebooks/ML_Project.ipynb
```

### Google Colab
1. Upload `ML_Project.ipynb` ke Google Colab
2. Upload kedua file CSV ke Colab
3. Run all cells

---

## 📚 Referensi

**Paper:**
> Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). 
> Modeling wine preferences by data mining from physicochemical properties. 
> Decision Support Systems, 47(4), 547-553.

**Dataset:**
- UCI Machine Learning Repository: Wine Quality Dataset
- Vinho Verde Wine: http://www.vinhoverde.pt/en/

**Libraries:**
- Scikit-learn Documentation
- XGBoost Documentation
- TensorFlow/Keras Documentation

---