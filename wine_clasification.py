import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import shap

#veri setini yükleme adim1

print("1. ADIM: Veri Seti Yükleme")
wine = load_wine()

# Veri Seti ve DataFrame
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target)

print(f"Özellik Matrisi (X) Şekli: {X.shape}")
print(f"Hedef Değişken (y) Şekli: {y.shape}")

#veri seti kalite kontrolleri adim2

print("\n 2. ADIM: Kalite Kontrol")

#Eksik Değer Analizi
print("\n Eksik Değer Sayıları:")
print(X.isnull().sum().sum())
# Tüm sütunlarda 0 eksik değer 

#Aykırı Değer Analizi IQR Yöntemi
def detect_iqr_outliers(df):
    outlier_counts = {}
    print("\n Aykırı Değer Sayıları (IQR):")
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_counts[col] = len(outliers)
        if len(outliers) > 0:
            print(f"  {col}: {len(outliers)} adet")
    return outlier_counts

detect_iqr_outliers(X)

#Veri Tipi İncelemesi
print("\n Veri Tipleri Özeti:")
X.info(verbose=False)


#Keşifsel Veri Analizi (EDA) adim3

print("\n Keşifsel Veri Analizi")

#İstatistiksel Özellikler
print("\n3.1 İstatistiksel Özellikler (X.describe()):")
print(X.describe())

#Korelasyon Matrisi
correlation_matrix = X.corr(method='pearson')

# En yüksek korelasyonlu çiftleri bulma
corr_pairs = correlation_matrix.unstack()
sorted_pairs = corr_pairs.sort_values(kind="quicksort", ascending=False)
strong_pairs = sorted_pairs[sorted_pairs != 1.0].drop_duplicates()

print("\nEn Yüksek Korelasyonlu 3 Çift Özellik:")
print(strong_pairs.head(3))

# Korelasyon Isı Haritası Görselleştirmesi
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Pearson Korelasyon Matrisi')
plt.show() # 

#Veri Ölçeklendirme adim4 

print("\nVeri Ölçeklendirme")
scaler = StandardScaler()
X_scaled_array = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns)
print(f"Ölçeklendirilmiş Veri Şekli: {X_scaled.shape}")

#Veri setinin bölünmesi adim5

print("\nVeri Setinin Bölümü")
# %70 Eğitim, %30 Kalan (Validation + Test)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)

# Kalanı %10 Validation ve %20 Test olacak şekilde bölme
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=(0.10 / 0.80), random_state=42, stratify=y_train_val
)

print(f"Eğitim Seti Şekli: {X_train.shape}")
print(f"Doğrulama Seti Şekli: {X_val.shape}")
print(f"Test Seti Şekli: {X_test.shape}")

# Özellik seçimi ve boyut indirgeme adim6

print("\nBoyut Indirgeme")

#PCA (Principal Component Analysis)
pca = PCA(n_components=None)
pca.fit(X_train)
explained_variance_ratio = pca.explained_variance_ratio_
avg_variance = explained_variance_ratio.mean()
n_components_pca = sum(explained_variance_ratio > avg_variance) # 4 bileşen çıkar

pca_final = PCA(n_components=n_components_pca)
X_train_pca = pca_final.fit_transform(X_train)

print(f"PCA İndirgenmiş Veri Şekli: {X_train_pca.shape} ({n_components_pca} Bileşen)")

#PCA Görselleştirme
pca_2d = PCA(n_components=2)
X_train_pca_2d = pca_2d.fit_transform(X_train)
plt.figure(figsize=(8, 6))
plt.scatter(X_train_pca_2d[:, 0], X_train_pca_2d[:, 1], c=y_train, cmap='viridis')
plt.title('PCA (PC1 vs PC2) ile Sınıf Ayrımı')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(ticks=np.unique(y_train))
plt.show() # 

#LDA (Linear Discriminant Analysis)
lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)

print(f"LDA İndirgenmiş Veri Şekli: {X_train_lda.shape} (2 Bileşen)")

#LDA Görselleştirme
plt.figure(figsize=(8, 6))
plt.scatter(X_train_lda[:, 0], X_train_lda[:, 1], c=y_train, cmap='viridis')
plt.title('LDA (LD1 vs LD2) ile Sınıf Ayrımı')
plt.xlabel('Linear Discriminant 1')
plt.ylabel('Linear Discriminant 2')
plt.colorbar(ticks=np.unique(y_train))
plt.show() # 

#Model kurulumu ve doğrulama adim7ve8

# En iyi model seçimi için LDA dönüşümlerini kullanma
X_val_lda = lda.transform(X_val)

model_lr_lda = LogisticRegression(random_state=42)
model_lr_lda.fit(X_train_lda, y_train)

# Doğrulama Seti Tahminleri
y_val_pred = model_lr_lda.predict(X_val_lda)
y_val_proba = model_lr_lda.predict_proba(X_val_lda)

print("\nDoğrulama Performansi (LDA - LR) ---")
print(f"Accuracy (Val): {accuracy_score(y_val, y_val_pred):.4f}")
# Diğer metrikler (Macro ortalama)
print(f"F1-score (Val): {f1_score(y_val, y_val_pred, average='macro', zero_division=0):.4f}")


# Test üzerinde degerlendirme adim9

print("\nTest Performansi")

# Test Seti Dönüşümü
X_test_lda = lda.transform(X_test)

# Test Seti Tahminleri
y_test_pred = model_lr_lda.predict(X_test_lda)
y_test_proba = model_lr_lda.predict_proba(X_test_lda)

#Performans Metrikleri
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
test_recall = recall_score(y_test, y_test_pred, average='macro', zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
test_roc_auc = roc_auc_score(y_test, y_test_proba, multi_class='ovr')

print(f"Accuracy (Test): {test_accuracy:.4f}")
print(f"Precision (Test): {test_precision:.4f}")
print(f"Recall (Test): {test_recall:.4f}")
print(f"F1-score (Test): {test_f1:.4f}")
print(f"ROC-AUC (Test): {test_roc_auc:.4f}")

#Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix:\n", cm)

# Confusion Matrix Görselleştirme
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=wine.target_names, yticklabels=wine.target_names)
plt.title('Confusion Matrix (Test Seti)')
plt.ylabel('Gerçek Sınıf')
plt.xlabel('Tahmin Edilen Sınıf')
plt.show() # 

# XAI – SHAP Analizi adim10 
print("\nShap aciklanabilirlik adim10 ---")

# SHAP Explainer oluşturma
# Modelimiz LDA bileşenleri üzerine eğitildi: X_train_lda
explainer = shap.Explainer(model_lr_lda, X_train_lda, feature_names=["LD1", "LD2"])

# Test seti tahminlerini açıklama
shap_values = explainer(X_test_lda)

#En Önemli Özellikler Yorumu (Bar Plot)
print("\nLD1 ve LD2'nin global önemini gösteren SHAP Bar Plot:")
shap.summary_plot(shap_values, X_test_lda, feature_names=["LD1", "LD2"], plot_type="bar", show=False)
plt.title("SHAP Global Feature Importance (LDA Components)")
plt.show() # 

# SHAP Summary Plot (Detaylı Etki - Sınıf Bazlı)
# Bu plot, her bir sınıf için bileşenlerin tahmin üzerindeki dağılımını gösterir.
print("\nSHAP Summary Plot (Detaylı Sınıf Etkisi):")
# shap_values bir list of Explanation objesidir (Her sınıf için bir tane). 
# Çok sınıflı problemde tüm sınıfları bir arada göstermek için [::, ::, 1] gibi bir indeksleme veya 
# sınıf adlarını elle atama gerekebilir. Standart çok sınıflı gösterim kullanılır.
shap.summary_plot(shap_values, X_test_lda, feature_names=["LD1", "LD2"], class_names=wine.target_names, show=False)
plt.title("SHAP Summary Plot (Multi-Class)")
plt.show()
