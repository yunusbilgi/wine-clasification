# Gerekli kütüphaneleri yükleyelim
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from itertools import cycle
import shap
import warnings
# XGBoost ve sklearn uyarılarını kapat
warnings.filterwarnings('ignore')

# Wine veri setini yükle
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Eksik Değer Kontrolü:")
print(X.isnull().sum())

# IQR Yöntemi ile aykırı değer tespiti için fonksiyon
def find_outliers_iqr(df, feature):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
    return outliers.shape[0], lower_bound, upper_bound

print("\nIQR Yöntemi ile Aykırı Değer Tespiti:")
outlier_counts = {}
for col in X.columns:
    count, lower, upper = find_outliers_iqr(X, col)
    outlier_counts[col] = count
    if count > 0:
        print(f"- {col}: {count} adet aykırı değer (Alt Sınır: {lower:.2f}, Üst Sınır: {upper:.2f})")

print("\nÖzet: Toplam Aykırı Değer Sayıları:")
print(pd.Series(outlier_counts))

print("\nVeri Tipleri (Dtype) Bilgisi:")
print(X.dtypes)

print("\nSayısal ve Kategorik Değişken Sayıları:")
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
target_type = "Kategorik/Sınıflandırma (int64)"
print(f"- Sayısal Değişken Sayısı: {len(numerical_cols)} (Tüm Özellikler)")
print(f"- Kategorik Değişken Sayısı: {len(categorical_cols)} (Özelliklerde yok, hedef değişken kategorik)")
print(f"- Hedef Değişken (y) Tipi: {target_type}")

print("Tüm Özellikler İçin İstatistiksel Özet (describe metodu):")
print(X.describe())

# StandardScaler nesnesini oluştur
scaler = StandardScaler()

# Veriyi eğit ve dönüştür
X_scaled_array = scaler.fit_transform(X)

# Ölçeklendirilmiş veriyi DataFrame formatına geri dönüştür ve X_scaled olarak kaydet
X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns)

print("Ölçeklendirilmiş X_scaled Verisinin İlk 5 Satırı:")
print(X_scaled.head())
print(f"\nÖlçeklendirilmiş Veri Seti Şekli: {X_scaled.shape}")

# 1. Adım: Veriyi Test setinden ayır (%20 Test, %80 Kalan)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Adım: Kalan veriyi Eğitim ve Doğrulama setlerine ayır (%70 Eğitim, %10 Doğrulama)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
)

# Setlerin boyutlarını kontrol edelim
print("\nVeri Seti Bölünme Boyutları:")
print(f"- Eğitim Seti (70%): X_train={X_train.shape}, y_train={y_train.shape}")
print(f"- Doğrulama Seti (10%): X_val={X_val.shape}, y_val={y_val.shape}")
print(f"- Test Seti (20%): X_test={X_test.shape}, y_test={y_test.shape}")

# PCA modelini oluştur ve eğit (tüm bileşenler)
pca = PCA(n_components=X_train.shape[1])
pca.fit(X_train)

# Açıklanan Varyans Oranlarını al
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

# Açıklanan Varyans grafiğini çiz
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='--')
plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance, marker='o', linestyle='-')
plt.title('PCA Açıklanan Varyans Oranları')
plt.xlabel('Temel Bileşen Sayısı (Component)')
plt.ylabel('Açıklanan Varyans Oranı')
plt.grid(True)
plt.legend(['Bireysel Açıklanan Varyans', 'Kümülatif Açıklanan Varyans'])
plt.show()

# Açıklanan varyans oranının ortalamasını hesapla
avg_explained_variance = np.mean(explained_variance_ratio)
# Ortalama varyanstan büyük olan bileşen sayısını bul
optimal_n_components = np.sum(explained_variance_ratio > avg_explained_variance)

print(f"Açıklanan Varyans Ortalaması: {avg_explained_variance:.4f}")
print(f"Optimal PCA Bileşen Sayısı (Ortalamadan Büyükler): {optimal_n_components}")

# Yeni PCA modelini optimal bileşen sayısı ile eğit
pca_optimal = PCA(n_components=optimal_n_components)
pca_optimal.fit(X_train)

# Eğitim, Doğrulama ve Test setlerini indirge
X_train_pca = pca_optimal.transform(X_train)
X_val_pca = pca_optimal.transform(X_val)
X_test_pca = pca_optimal.transform(X_test)

print(f"\nPCA Sonrası Eğitim Verisi Şekli: {X_train_pca.shape}")

# Tüm veriyi (eğitim+doğrulama+test) indirgeyip görselleştirelim (2D)
X_pca_2d = PCA(n_components=2).fit_transform(X_scaled)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.xlabel('Temel Bileşen 1 (PC1)')
plt.ylabel('Temel Bileşen 2 (PC2)')
plt.title('PCA İndirgenmiş Veride Sınıf Ayrışması (PC1 vs PC2)')
legend1 = plt.legend(*scatter.legend_elements(), title="Şarap Sınıfı")
plt.gca().add_artist(legend1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# LDA modelini oluştur ve eğit
n_components_lda = 2
lda = LinearDiscriminantAnalysis(n_components=n_components_lda)
lda.fit(X_train, y_train)

# Eğitim, Doğrulama ve Test setlerini dönüştür
X_train_lda = lda.transform(X_train)
X_val_lda = lda.transform(X_val)
X_test_lda = lda.transform(X_test)

print(f"\nLDA Sonrası Eğitim Verisi Şekli: {X_train_lda.shape}")

# Tüm veriyi indirgeyip görselleştirelim (2D)
X_lda_2d = lda.transform(X_scaled)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_lda_2d[:, 0], X_lda_2d[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.xlabel('Lineer Ayırıcı (Discriminant) 1 (LD1)')
plt.ylabel('Lineer Ayırıcı (Discriminant) 2 (LD2)')
plt.title('LDA İndirgenmiş Veride Sınıf Ayrışması (LD1 vs LD2)')
legend1 = plt.legend(*scatter.legend_elements(), title="Şarap Sınıfı")
plt.gca().add_artist(legend1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Tüm modelleri bir sözlükte tanımlayalım
models = {
    "LogisticRegression": LogisticRegression(random_state=42, multi_class='ovr', solver='liblinear'),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "GaussianNB": GaussianNB()
}

# Veri temsillerini bir sözlükte toplayalım
data_representations = {
    "Ham (Ölçeklendirilmiş)": (X_train, X_val),
    "PCA": (X_train_pca, X_val_pca),
    "LDA": (X_train_lda, X_val_lda)
}

# Sonuçları tutacak bir DataFrame oluşturalım
results_df = pd.DataFrame(columns=['Model', 'Veri Temsili', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC'])

def evaluate_model(y_true, y_pred, y_proba):
    """Çok sınıflı sınıflandırma metriklerini hesaplar."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    if y_proba is not None and y_proba.shape[1] > 1:
        roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr')
    else:
        roc_auc = np.nan
    return accuracy, precision, recall, f1, roc_auc

row_list = []
for data_name, (X_train_data, X_val_data) in data_representations.items():
    for model_name, model in models.items():
        model.fit(X_train_data, y_train)
        y_val_pred = model.predict(X_val_data)
        y_val_proba = None
        if hasattr(model, 'predict_proba'):
            y_val_proba = model.predict_proba(X_val_data)

        accuracy, precision, recall, f1, roc_auc = evaluate_model(y_val, y_val_pred, y_val_proba)

        row_list.append({
            'Model': model_name,
            'Veri Temsili': data_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1,
            'ROC-AUC': roc_auc
        })

results_df = pd.DataFrame(row_list)
results_df = results_df.round(4)

print("### 8. Doğrulama (Validation) Seti Performans Karşılaştırması ###")
print(results_df)

# En iyi modelin test performansı (LDA Logistic Regression)
best_model = models['LogisticRegression']

y_test_pred = best_model.predict(X_test_lda)
y_test_proba = best_model.predict_proba(X_test_lda)

test_accuracy, test_precision, test_recall, test_f1, test_roc_auc = evaluate_model(
    y_test, y_test_pred, y_test_proba
)

print("\n### 9.1 En İyi Modelin (LDA Logistic Regression) Test Performansı ###")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1-score: {test_f1:.4f}")
print(f"ROC-AUC: {test_roc_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.title('Test Seti Confusion Matrix (LDA - Logistic Regression)')
plt.show()

# ROC Eğrileri
n_classes = len(data.target_names)
y_test_binarized = label_binarize(y_test, classes=range(n_classes))

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
colors = cycle(['blue', 'red', 'green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='Sınıf {0} (AUC={1:0.4f})'.format(data.target_names[i], roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Yanlış Pozitif Oranı (False Positive Rate)')
plt.ylabel('Doğru Pozitif Oranı (True Positive Rate)')
plt.title('Test Seti ROC Eğrileri (One-vs-Rest)')
plt.legend(loc="lower right")
plt.show()

mean_roc_auc = np.mean(list(roc_auc.values()))
print(f"\nOrtalama ROC-AUC Değeri: {mean_roc_auc:.4f}")

# SHAP Analizi
lda_feature_names = [f"LD{i+1}" for i in range(X_train_lda.shape[1])]
explainer_lda = shap.LinearExplainer(best_model, X_train_lda)
shap_values_lda = explainer_lda.shap_values(X_val_lda)
shap_values_list = shap_values_lda
shap_values_array = np.array(shap_values_list)

# SUMMARY PLOT (Her Bir Sınıfı Tek Tek Çizme)
print("LDA SHAP Summary Plotları çiziliyor (Her Sınıf Ayrı Ayrı):")
for i in range(len(data.target_names)):
    shap_i = shap_values_array[i]

    if shap_i.shape[1] > len(lda_feature_names):
        shap_i = shap_i[:, :-1]

    plt.figure()
    shap.summary_plot(
        shap_i,
        feature_names=lda_feature_names,
        show=False,
        title=f'LDA SHAP Summary Plot Sınıf: {data.target_names[i]}'
    )
    plt.show()

# BAR PLOT (Ortalama Mutlak Etki)
print("\nLDA SHAP Bar Plot çiziliyor (Ortalama Mutlak Etki):")
mean_abs_shap_values = []
for i in range(len(data.target_names)):
    shap_i = shap_values_array[i]
    if shap_i.shape[1] > len(lda_feature_names):
        shap_i = shap_i[:, :-1]
    mean_abs_shap_values.append(np.abs(shap_i))
final_mean_abs = np.mean(mean_abs_shap_values, axis=0)

shap.summary_plot(
    final_mean_abs,
    feature_names=lda_feature_names,
    plot_type="bar",
    show=False
)
plt.title('LDA Logistic Regression SHAP Bar Plot (Ortalama Mutlak Etki)')
plt.show()
