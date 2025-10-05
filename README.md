# 📊 Linear Regresyon: Sıfırdan İleri Seviye

> **Makine Öğrenmesi'nin Temel Taşı: Linear Regresyon Ders Notları**
> 
> *Hazırlayan: Dr. Buket Toptaş*  
> *Tarih: 2025*

---

## 📚 İçindekiler

1. [Giriş](#-giriş)
2. [Linear Regresyon Nedir?](#-linear-regresyon-nedir)
3. [Matematiksel Temel](#-matematiksel-temel)
4. [Kod Örnekleri](#-kod-örnekleri)
5. [Görselleştirme](#-görselleştirme)
6. [Performans Metrikleri](#-performans-metrikleri)
7. [Gerçek Dünya Uygulamaları](#-gerçek-dünya-uygulamaları)
8. [İleri Seviye Konular](#-i̇leri-seviye-konular)
9. [Kaynaklar](#-kaynaklar)

---

## 🎯 Giriş

Linear Regresyon, **makine öğrenmesinin alfabesi**dir. İki değişken arasındaki doğrusal ilişkiyi modelleyen bu algoritma, karmaşık modellerin temelini oluşturur.

### 🎓 Bu Derste Öğrenecekleriniz

- ✅ Linear regresyonun matematiksel temelleri
- ✅ Python ve Sklearn ile uygulama
- ✅ Model performansını değerlendirme
- ✅ Overfitting ve underfitting kavramları
- ✅ Gerçek dünya problemlerini çözme

### 📋 Ön Koşullar

```python
# Gerekli kütüphaneler
numpy>=1.21.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
pandas>=1.3.0
seaborn>=0.11.0
```

---

## 🔍 Linear Regresyon Nedir?

Linear regresyon, **bağımlı değişken (y)** ile bir veya daha fazla **bağımsız değişken (x)** arasındaki **doğrusal ilişkiyi** modelleyen istatistiksel bir yöntemdir.

### 📐 Basit Formül

```
y = mx + b
```

Burada:
- **y**: Tahmin etmek istediğimiz değer (bağımlı değişken)
- **x**: Bildiğimiz değer (bağımsız değişken)
- **m**: Eğim (slope) - x değişince y ne kadar değişir?
- **b**: Kesim noktası (intercept) - x=0 olduğunda y'nin değeri

### 🎯 Örnek Senaryo: Dondurma Satışları

> Ahmet Amca bir sahil kasabasında dondurma satıyor. Sıcaklık arttıkça satışları da artıyor. Yarın hava 35°C olacaksa, kaç dondurma hazırlamalı?

| Sıcaklık (°C) | Satış (Adet) |
|---------------|--------------|
| 20            | 30           |
| 22            | 35           |
| 25            | 45           |
| 28            | 55           |
| 30            | 60           |
| 32            | 70           |
| 35            | 80           |

**Soru:** 35°C'de kaç dondurma satılır?

---

## 📊 Matematiksel Temel

### En Küçük Kareler Yöntemi (Ordinary Least Squares - OLS)

Linear regresyon, **hata karelerinin toplamını minimize eden** en iyi doğruyu bulur.

#### Eğim (m) Hesaplama

```
m = Σ[(xi - x̄)(yi - ȳ)] / Σ[(xi - x̄)²]
```

#### Kesim (b) Hesaplama

```
b = ȳ - m × x̄
```

Burada:
- `x̄` = Ortalama x değeri
- `ȳ` = Ortalama y değeri
- `Σ` = Toplam sembolü

### 🧮 Adım Adım Hesaplama

```python
import numpy as np

# Veri
sicaklik = np.array([20, 22, 25, 28, 30, 32, 35])
satis = np.array([30, 35, 45, 55, 60, 70, 80])

# Ortalamalar
x_mean = np.mean(sicaklik)  # 27.43
y_mean = np.mean(satis)      # 53.57

# Eğim (m)
numerator = np.sum((sicaklik - x_mean) * (satis - y_mean))
denominator = np.sum((sicaklik - x_mean) ** 2)
m = numerator / denominator  # 3.42

# Kesim (b)
b = y_mean - m * x_mean      # -40.14

print(f"Denklem: y = {m:.2f}x + {b:.2f}")
# Output: y = 3.42x + -40.14
```

**Sonuç:** Her 1°C artışta satış ~3.42 adet artıyor!

---

## 💻 Kod Örnekleri

### 🔰 Başlangıç: Manuel Uygulama

```python
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    """Manuel Linear Regresyon Sınıfı"""
    
    def __init__(self):
        self.m = None  # Eğim
        self.b = None  # Kesim
    
    def fit(self, X, y):
        """Modeli eğit"""
        x_mean = np.mean(X)
        y_mean = np.mean(y)
        
        # Eğim hesapla
        numerator = np.sum((X - x_mean) * (y - y_mean))
        denominator = np.sum((X - x_mean) ** 2)
        self.m = numerator / denominator
        
        # Kesim hesapla
        self.b = y_mean - self.m * x_mean
        
        return self
    
    def predict(self, X):
        """Tahmin yap"""
        return self.m * X + self.b
    
    def score(self, X, y):
        """R² skorunu hesapla"""
        y_pred = self.predict(X)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        return 1 - (ss_res / ss_tot)

# Kullanım
model = LinearRegression()
model.fit(sicaklik, satis)

print(f"Eğim: {model.m:.4f}")
print(f"Kesim: {model.b:.4f}")
print(f"R² Skoru: {model.score(sicaklik, satis):.4f}")

# Tahmin
yarin_sicaklik = 35
tahmin = model.predict(yarin_sicaklik)
print(f"35°C'de tahmini satış: {tahmin:.0f} dondurma")
```

### 🚀 Profesyonel: Sklearn ile Uygulama

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Veriyi hazırla
X = sicaklik.reshape(-1, 1)  # 2D array'e çevir
y = satis

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model oluştur ve eğit
model = LinearRegression()
model.fit(X_train, y_train)

# Tahmin
y_pred = model.predict(X_test)

# Performans
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")
print(f"Eğim: {model.coef_[0]:.4f}")
print(f"Kesim: {model.intercept_:.4f}")
```

### 📈 Çok Değişkenli Linear Regresyon

```python
import pandas as pd

# Veri (sıcaklık, nem, rüzgar → satış)
data = pd.DataFrame({
    'sicaklik': [20, 22, 25, 28, 30, 32, 35],
    'nem': [60, 55, 50, 45, 40, 35, 30],
    'ruzgar': [10, 15, 12, 8, 5, 7, 3],
    'satis': [30, 35, 45, 55, 60, 70, 80]
})

# Özellikler ve hedef
X = data[['sicaklik', 'nem', 'ruzgar']]
y = data['satis']

# Model
model = LinearRegression()
model.fit(X, y)

print("Katsayılar:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:.2f}")
print(f"Kesim: {model.intercept_:.2f}")

# Tahmin
yeni_veri = pd.DataFrame({
    'sicaklik': [35],
    'nem': [30],
    'ruzgar': [5]
})
tahmin = model.predict(yeni_veri)
print(f"\nTahmin: {tahmin[0]:.0f} dondurma")
```

---

## 📉 Görselleştirme

### Basit Scatter Plot ve Regresyon Çizgisi

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Stil ayarları
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))

# Scatter plot
plt.scatter(sicaklik, satis, color='blue', s=100, 
            alpha=0.6, edgecolors='black', linewidth=2,
            label='Gerçek Veriler')

# Regresyon çizgisi
X_line = np.linspace(sicaklik.min(), sicaklik.max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)
plt.plot(X_line, y_line, color='red', linewidth=3, 
         label='Regresyon Çizgisi')

# Hata çizgileri
for x, y_true in zip(sicaklik, satis):
    y_pred = model.predict([[x]])[0]
    plt.plot([x, x], [y_true, y_pred], 'g--', alpha=0.5, linewidth=1)

plt.xlabel('Sıcaklık (°C)', fontsize=12, fontweight='bold')
plt.ylabel('Satış (Adet)', fontsize=12, fontweight='bold')
plt.title('🍦 Dondurma Satış Tahmini - Linear Regresyon', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Residual Plot (Hata Analizi)

```python
# Tahminler
y_pred = model.predict(X)
residuals = y - y_pred

# Residual plot
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_pred, residuals, color='green', s=100, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Tahmin Edilen Değer', fontweight='bold')
plt.ylabel('Hata (Residual)', fontweight='bold')
plt.title('Residual Plot', fontweight='bold')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(y, y_pred, color='purple', s=100, alpha=0.6)
perfect_line = np.linspace(y.min(), y.max(), 100)
plt.plot(perfect_line, perfect_line, 'r--', linewidth=2, 
         label='Mükemmel Tahmin')
plt.xlabel('Gerçek Değer', fontweight='bold')
plt.ylabel('Tahmin', fontweight='bold')
plt.title('Gerçek vs Tahmin', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 🎯 Performans Metrikleri

### 1. R² (R-Squared) - Determinasyon Katsayısı

**Ne ölçer?** Modelin veriyi ne kadar iyi açıkladığı

```python
r2 = r2_score(y_true, y_pred)
```

**Yorumlama:**
- **R² = 1.0**: Mükemmel model (tüm varyansı açıklıyor)
- **R² = 0.8-0.9**: Çok iyi
- **R² = 0.6-0.7**: İyi
- **R² < 0.5**: Zayıf
- **R² < 0**: Modeliniz ortalamanın altında!

### 2. MSE (Mean Squared Error)

**Ne ölçer?** Hataların karelerinin ortalaması

```python
mse = mean_squared_error(y_true, y_pred)
```

**Özellikler:**
- Büyük hataları daha çok cezalandırır
- Asla negatif olamaz
- Orijinal birimle aynı değil (kare)

### 3. RMSE (Root Mean Squared Error)

**Ne ölçer?** MSE'nin karekökü (orijinal birimle)

```python
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
```

**Avantajları:**
- Orijinal birimle çalışır (yorumlaması kolay)
- Outlier'lara duyarlı

### 4. MAE (Mean Absolute Error)

**Ne ölçer?** Hataların mutlak değerlerinin ortalaması

```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred)
```

**Avantajları:**
- Outlier'lara daha az duyarlı
- Basit yorumlama

### 📊 Karşılaştırma

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(y_true, y_pred):
    """Model performansını değerlendir"""
    metrics = {
        'R²': r2_score(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred)
    }
    
    print("📊 Model Performansı:")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"{metric:10s}: {value:.4f}")
    
    return metrics

# Kullanım
metrics = evaluate_model(y_test, y_pred)
```

---

## 🌍 Gerçek Dünya Uygulamaları

### 1. 🏠 Ev Fiyat Tahmini

```python
# Örnek veri
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
X = data.data
y = data.target

# Model
model = LinearRegression()
model.fit(X, y)

# En önemli özellikler
feature_importance = pd.DataFrame({
    'Feature': data.feature_names,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print(feature_importance)
```

### 2. 📈 Satış Tahmini

```python
# Reklam harcaması → Satış tahmini
advertising = pd.DataFrame({
    'TV': [230, 44, 17, 151, 180],
    'Radio': [37, 39, 45, 41, 10],
    'Newspaper': [69, 45, 69, 58, 58],
    'Sales': [22, 10, 9, 18, 12]
})

X = advertising[['TV', 'Radio', 'Newspaper']]
y = advertising['Sales']

model = LinearRegression()
model.fit(X, y)

# Yeni kampanya tahmini
new_campaign = [[200, 40, 60]]
predicted_sales = model.predict(new_campaign)
print(f"Tahmini satış: {predicted_sales[0]:.2f} bin $")
```

### 3. 🎓 Öğrenci Not Tahmini

```python
# Çalışma saati → Sınav notu
study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
exam_scores = np.array([45, 50, 55, 65, 70, 80, 85, 90])

model = LinearRegression()
model.fit(study_hours, exam_scores)

# 9 saat çalışan öğrencinin tahmini notu
predicted_score = model.predict([[9]])
print(f"9 saat çalışma → Tahmini not: {predicted_score[0]:.1f}")
```

---

## 🚀 İleri Seviye Konular

### 1. Polynomial Regression (Polinom Regresyon)

Doğrusal olmayan ilişkiler için:

```python
from sklearn.preprocessing import PolynomialFeatures

# Polinom özellikleri oluştur (derece=2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Model
model = LinearRegression()
model.fit(X_poly, y)

# x² ve xy terimleri otomatik eklendi!
```

### 2. Regularization (Ridge & Lasso)

Overfitting'i önlemek için:

```python
from sklearn.linear_model import Ridge, Lasso

# Ridge (L2 Regularization)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Lasso (L1 Regularization)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
```

### 3. Feature Scaling

Özellikleri normalize etme:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)
```

### 4. Cross-Validation

Modeli güvenilir şekilde değerlendirme:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"Ortalama R²: {scores.mean():.4f} (±{scores.std():.4f})")
```

---


### 🌐 Online Kaynaklar
- [Scikit-learn Dokümantasyonu](https://scikit-learn.org/)
- [Khan Academy - Statistics](https://www.khanacademy.org/math/statistics-probability)
- [StatQuest YouTube](https://www.youtube.com/c/joshstarmer)


## 🤝 Katkıda Bulunma

Bu proje açık kaynak kodludur. Katkılarınızı bekliyoruz!

```bash
# Repo'yu fork edin
git clone https://github.com/bukettoptas/linear-regression-tutorial.git

# Branch oluşturun
git checkout -b feature/yeni-özellik

# Commit yapın
git commit -m "Yeni özellik eklendi"

# Push edin
git push origin feature/yeni-özellik

# Pull request açın
```

---


**Buket Toptaş**  
💼 LinkedIn: [linkedin.com/in/bukettoptas]  
🐙 GitHub: [@bukettoptas](https://github.com/bukettoptas)

---

<div align="center">

**⭐ Bu repo işinize yaradıysa yıldız vermeyi unutmayın! ⭐**

Made with ❤️ for Machine Learning Students

</div>
