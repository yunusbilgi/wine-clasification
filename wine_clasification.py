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

# --- BÖLÜM 1: VERİ SETİNİN YÜKLENMESİ (ADIM 1) ---

print("--- 1. ADIM: VERİ SETİ YÜKLEME ---")
wine = load_wine()

# Veri Seti ve DataFrame Oluşturma
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target)

print(f"Özellik Matrisi (X) Şekli: {X.shape}")
print(f"Hedef Değişken (y) Şekli: {y.shape}")

# --- BÖLÜM 2: VERİ SETİ KALİTE KONTROLLERİ (ADIM 2) ---

print("\n--- 2. ADIM: KALİTE KONTROLLERİ ---")

# 2.1 Eksik Değer Analizi
print("\n2.1 Eksik Değer Sayıları:")
print(X.isnull().sum().sum())
# Tüm sütunlarda 0 eksik değer olduğu varsayılır.

# 2.2 Aykırı Değer (Outlier) Analizi (IQR Yöntemi)
def detect_iqr_outliers(df):
    outlier_counts = {}
    print("\n2.2 Aykırı Değer Sayıları (IQR):")
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

# 2.3 Veri Tipi İncelemesi
print("\n2.3 Veri Tipleri Özeti:")
X.info(verbose=False)


# --- BÖLÜM 3: KEŞİFSEL VERİ ANALİZİ (EDA) (ADIM 3) ---

print("\n--- 3. ADIM: KEŞİFSEL VERİ ANALİZİ (EDA) ---")

# 3.1 İstatistiksel Özellikler
print("\n3.1 İstatistiksel Özellikler (X.describe()):")
print(X.describe())

# 3.2 Korelasyon Matrisi
correlation_matrix = X.corr(method='pearson')

# En yüksek korelasyonlu çiftleri bulma
corr_pairs = correlation_matrix.unstack()
sorted_pairs = corr_pairs.sort_values(kind="quicksort", ascending=False)
strong_pairs = sorted_pairs[sorted_pairs != 1.0].drop_duplicates()

print("\n3.2 En Yüksek Korelasyonlu 3 Çift Özellik:")
print(strong_pairs.head(3))

# Korelasyon Isı Haritası Görselleştirmesi
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Pearson Korelasyon Matrisi')
plt.show() # 

# --- BÖLÜM 4: VERİ ÖLÇEKLENDİRME (SCALING) (ADIM 4) ---

print("\n--- 4. ADIM: VERİ ÖLÇEKLENDİRME ---")
scaler = StandardScaler()
X_scaled_array = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns)
print(f"Ölçeklendirilmiş Veri Şekli: {X_scaled.shape}")

# --- BÖLÜM 5: VERİ SETİNİN BÖLÜNMESİ (ADIM 5) ---

print("\n--- 5. ADIM: VERİ SETİ BÖLÜNÜMÜ ---")
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

# --- BÖLÜM 6: ÖZELLİK SEÇİMİ VE BOYUT İNDİRGEME (ADIM 6) ---

print("\n--- 6. ADIM: BOYUT İNDİRGEME ---")

# 6.1 PCA (Principal Component Analysis)
pca = PCA(n_components=None)
pca.fit(X_train)
explained_variance_ratio = pca.explained_variance_ratio_
avg_variance = explained_variance_ratio.mean()
n_components_pca = sum(explained_variance_ratio > avg_variance) # 4 bileşen çıkar

pca_final = PCA(n_components=n_components_pca)
X_train_pca = pca_final.fit_transform(X_train)

print(f"PCA İndirgenmiş Veri Şekli: {X_train_pca.shape} ({n_components_pca} Bileşen)")

# 2D PCA Görselleştirme
pca_2d = PCA(n_components=2)
X_train_pca_2d = pca_2d.fit_transform(X_train)
plt.figure(figsize=(8, 6))
plt.scatter(X_train_pca_2d[:, 0], X_train_pca_2d[:, 1], c=y_train, cmap='viridis')
plt.title('PCA (PC1 vs PC2) ile Sınıf Ayrımı')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(ticks=np.unique(y_train))
plt.show() # 

# 6.2 LDA (Linear Discriminant Analysis)
lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)

print(f"LDA İndirgenmiş Veri Şekli: {X_train_lda.shape} (2 Bileşen)")

# 2D LDA Görselleştirme
plt.figure(figsize=(8, 6))
plt.scatter(X_train_lda[:, 0], X_train_lda[:, 1], c=y_train, cmap='viridis')
plt.title('LDA (LD1 vs LD2) ile Sınıf Ayrımı')
plt.xlabel('Linear Discriminant 1')
plt.ylabel('Linear Discriminant 2')
plt.colorbar(ticks=np.unique(y_train))
plt.show() # 

# --- BÖLÜM 7 & 8: MODEL KURULUMU VE DOĞRULAMA (ADIM 7 & 8) ---

# En iyi model seçimi için LDA dönüşümlerini kullanma
X_val_lda = lda.transform(X_val)

model_lr_lda = LogisticRegression(random_state=42)
model_lr_lda.fit(X_train_lda, y_train)

# Doğrulama Seti Tahminleri
y_val_pred = model_lr_lda.predict(X_val_lda)
y_val_proba = model_lr_lda.predict_proba(X_val_lda)

print("\n--- 7 & 8. ADIM: DOĞRULAMA PERFORMANSI (LDA - LR) ---")
print(f"Accuracy (Val): {accuracy_score(y_val, y_val_pred):.4f}")
# Diğer metrikler (Macro ortalama)
print(f"F1-score (Val): {f1_score(y_val, y_val_pred, average='macro', zero_division=0):.4f}")


# --- BÖLÜM 9: TEST ÜZERİNDE DEĞERLENDİRME (ADIM 9) ---

print("\n--- 9. ADIM: TEST PERFORMANSI (LDA - LR) ---")

# Test Seti Dönüşümü
X_test_lda = lda.transform(X_test)

# Test Seti Tahminleri
y_test_pred = model_lr_lda.predict(X_test_lda)
y_test_proba = model_lr_lda.predict_proba(X_test_lda)

# 9.1 Performans Metrikleri
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

# 9.2 Confusion Matrix
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

# --- BÖLÜM 10: XAI – SHAP ANALİZİ (ADIM 10) ---

print("\n--- 10. ADIM: SHAP AÇIKLANABİLİRLİK ANALİZİ ---")

# SHAP Explainer oluşturma
# Modelimiz LDA bileşenleri üzerine eğitildi: X_train_lda
explainer = shap.Explainer(model_lr_lda, X_train_lda, feature_names=["LD1", "LD2"])

# Test seti tahminlerini açıklama
shap_values = explainer(X_test_lda)

# 10.1 En Önemli Özellikler Yorumu (Bar Plot)
print("\nLD1 ve LD2'nin global önemini gösteren SHAP Bar Plot:")
shap.summary_plot(shap_values, X_test_lda, feature_names=["LD1", "LD2"], plot_type="bar", show=False)
plt.title("SHAP Global Feature Importance (LDA Components)")
plt.show() # 

# 10.2 SHAP Summary Plot (Detaylı Etki - Sınıf Bazlı)
# Bu plot, her bir sınıf için bileşenlerin tahmin üzerindeki dağılımını gösterir.
print("\nSHAP Summary Plot (Detaylı Sınıf Etkisi):")
# shap_values bir list of Explanation objesidir (Her sınıf için bir tane). 
# Çok sınıflı problemde tüm sınıfları bir arada göstermek için [::, ::, 1] gibi bir indeksleme veya 
# sınıf adlarını elle atama gerekebilir. Standart çok sınıflı gösterim kullanılır.
shap.summary_plot(shap_values, X_test_lda, feature_names=["LD1", "LD2"], class_names=wine.target_names, show=False)
plt.title("SHAP Summary Plot (Multi-Class)")
plt.show()