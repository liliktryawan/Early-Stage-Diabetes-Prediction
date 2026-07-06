# Import library yang diperlukan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Pengaturan visualisasi
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

colors_dict = {'Positive': '#e74c3c', 'Negative': '#2ecc71'}

print("Semua library berhasil diimport!")

# Memuat dataset
df = pd.read_csv('diabetes_data.csv')

# Menampilkan ukuran dataset
print(f"Jumlah baris: {df.shape[0]}")
print(f"Jumlah kolom: {df.shape[1]}")
print(f"\nPreview data:")
df.head()

# Informasi dataset
print("=" * 60)
print("INFORMASI DATASET")
print("=" * 60)
df.info()

# Statistik deskriptif untuk kolom numerik (Age)
df.describe().round(2)

# Cek missing values
print("Total missing values:", df.isnull().sum().sum())

# Cek duplikat
print(f"Jumlah data duplikat: {df.duplicated().sum()}")

# Distribusi target
print("Distribusi target:")
val_counts = df['class'].value_counts()
print(val_counts)
print()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot Bar dengan warna konsisten
colors_bar = [colors_dict[idx] for idx in val_counts.index]
val_counts.plot(kind='bar', ax=axes[0], color=colors_bar, edgecolor='black')
axes[0].set_title('Distribusi Kelas Target', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Kelas')
axes[0].set_ylabel('Jumlah')
axes[0].tick_params(axis='x', rotation=0)

# Plot Pie dengan warna konsisten
val_counts.plot(kind='pie', ax=axes[1], autopct='%1.1f%%',
                                  colors=colors_bar, startangle=90)
axes[1].set_title('Proporsi Kelas Target', fontsize=14, fontweight='bold')
axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig('distribusi_target_diabetes.png', dpi=150, bbox_inches='tight')
plt.show()

# Visualisasi hubungan gejala utama dengan kelas target dengan warna konsisten
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Hubungan Polyuria dengan Class
sns.countplot(data=df, x='Polyuria', hue='class', palette=colors_dict, ax=axes[0], edgecolor='black')
axes[0].set_title('Hubungan Gejala Polyuria dengan Risiko Diabetes', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Polyuria (Sering Buang Air Kecil)')
axes[0].set_ylabel('Jumlah')

# Hubungan Polydipsia dengan Class
sns.countplot(data=df, x='Polydipsia', hue='class', palette=colors_dict, ax=axes[1], edgecolor='black')
axes[1].set_title('Hubungan Gejala Polydipsia dengan Risiko Diabetes', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Polydipsia (Sering Merasa Haus)')
axes[1].set_ylabel('Jumlah')

plt.tight_layout()
plt.savefig('gejala_utama_vs_target.png', dpi=150, bbox_inches='tight')
plt.show()

# Hubungan Usia (Age) dengan Risiko Diabetes dengan warna konsisten
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='class', y='Age', palette=colors_dict, order=['Negative', 'Positive'])
plt.title('Distribusi Usia Berdasarkan Risiko Diabetes', fontsize=14, fontweight='bold')
plt.xlabel('Status')
plt.ylabel('Usia')
plt.tight_layout()
plt.savefig('boxplot_age_diabetes.png', dpi=150, bbox_inches='tight')
plt.show()

# Inisialisasi LabelEncoder
le = LabelEncoder()

# Duplikasi dataframe agar data asli aman
df_encoded = df.copy()

# Terapkan label encoding ke semua fitur bertipe objek
for col in df_encoded.select_dtypes(include=['object']).columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])

print("Data setelah Label Encoding:")
df_encoded.head()

# Pisahkan X dan y
X = df_encoded.drop('class', axis=1)
y = df_encoded['class']

print(f"Shape fitur (X): {X.shape}")
print(f"Shape target (y): {y.shape}")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Jumlah data latih: {X_train.shape[0]} sampel")
print(f"Jumlah data uji  : {X_test.shape[0]} sampel")

# Inisialisasi Scaler
scaler = StandardScaler()

# Standarisasi kolom Age saja
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# Fit transform data latih, dan transform data uji
X_train_scaled['Age'] = scaler.fit_transform(X_train[['Age']])
X_test_scaled['Age'] = scaler.transform(X_test[['Age']])

print("Statistik kolom Age setelah standarisasi:")
print(f"Mean data latih: {X_train_scaled['Age'].mean():.6f}")
print(f"Std data latih : {X_train_scaled['Age'].std():.6f}")

# Model 1: Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

print("Random Forest - Classification Report:")
print(classification_report(y_test, rf_pred, target_names=['Negative', 'Positive']))

# Model 2: XGBoost
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_test_scaled)

print("XGBoost - Classification Report:")
print(classification_report(y_test, xgb_pred, target_names=['Negative', 'Positive']))

# Model 3: SVM
svm_model = SVC(
    kernel='rbf',
    C=1.0,
    random_state=42
)

svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)

print("SVM - Classification Report:")
print(classification_report(y_test, svm_pred, target_names=['Negative', 'Positive']))

# Buat dataframe ringkasan metrik
models = {
    'Random Forest': rf_pred,
    'XGBoost': xgb_pred,
    'SVM': svm_pred
}

results = []
for name, pred in models.items():
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred),
        'Recall': recall_score(y_test, pred),
        'F1-Score': f1_score(y_test, pred)
    })

results_df = pd.DataFrame(results).set_index('Model')
print("=" * 65)
print("PERBANDINGAN MODEL")
print("=" * 65)
print(results_df.round(4).to_string())
print()

# Pilih model terbaik berdasarkan F1-Score
best_model_name = results_df['F1-Score'].idxmax()
print(f"Model Terbaik: {best_model_name}")

# Visualisasi Perbandingan Model
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Plot metrik (Legend diubah ke kanan atas)
results_df.plot(kind='bar', ax=axes[0], colormap='viridis', edgecolor='black')
axes[0].set_title('Metrik Perbandingan Model', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Skor')
axes[0].set_ylim(0.8, 1.05)
axes[0].tick_params(axis='x', rotation=0)
axes[0].legend(loc='upper right')

# Heatmap Confusion Matrix Model Terbaik
best_pred = models[best_model_name]
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
axes[1].set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Prediksi')
axes[1].set_ylabel('Aktual')

plt.tight_layout()
plt.savefig('perbandingan_model_diabetes.png', dpi=150, bbox_inches='tight')
plt.show()

# Hyperparameter Tuning pada Random Forest
# Random Forest dipilih untuk tuning karena kebutuhan interpretabilitas (feature importance)
# Meskipun SVM memiliki F1-score tertinggi, RF dipilih agar bisa mengekstrak fitur penting

param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [4, 6, 8, 10],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

print("Melakukan GridSearchCV pada Random Forest...")
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best F1-Score (CV): {grid_search.best_score_:.4f}")

# Evaluasi model setelah tuning
best_tuned_model = grid_search.best_estimator_
tuned_pred = best_tuned_model.predict(X_test_scaled)

print("=" * 65)
print("EVALUASI MODEL SETELAH TUNING")
print("=" * 65)
print(classification_report(y_test, tuned_pred, target_names=['Negative', 'Positive']))

print(f"Akurasi Model Tuned: {accuracy_score(y_test, tuned_pred):.4f}")
print(f"F1-Score Model Tuned: {f1_score(y_test, tuned_pred):.4f}")

# Confusion matrix perbandingan sebelum vs sesudah tuning
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sebelum tuning
cm_before = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm_before, annot=True, fmt='d', cmap='Oranges', ax=axes[0],
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
axes[0].set_title('Random Forest Baseline', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Prediksi')
axes[0].set_ylabel('Aktual')

# Setelah tuning
cm_after = confusion_matrix(y_test, tuned_pred)
sns.heatmap(cm_after, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
axes[1].set_title('Random Forest Tuned', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Prediksi')
axes[1].set_ylabel('Aktual')

plt.suptitle('Confusion Matrix - Sebelum vs Sesudah Tuning', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('tuning_comparison_diabetes.png', dpi=150, bbox_inches='tight')
plt.show()

# Feature importance dari model Random Forest Tuned
feature_importance = best_tuned_model.feature_importances_
feat_imp_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(12, 8))
plt.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color='#2ecc71', edgecolor='black')
plt.title('Gejala Paling Berpengaruh dalam Deteksi Risiko Diabetes (Random Forest Tuned)', fontsize=14, fontweight='bold')
plt.xlabel('Skor Kepentingan (Importance)')
plt.tight_layout()
plt.savefig('feature_importance_diabetes.png', dpi=150, bbox_inches='tight')
plt.show()

# Cetak hasil metrik final
final_accuracy = accuracy_score(y_test, tuned_pred)
final_precision = precision_score(y_test, tuned_pred)
final_recall = recall_score(y_test, tuned_pred)
final_f1 = f1_score(y_test, tuned_pred)

print(f"Akurasi Final  : {final_accuracy:.4f}")
print(f"Precision Final: {final_precision:.4f}")
print(f"Recall Final   : {final_recall:.4f}")
print(f"F1-Score Final : {final_f1:.4f}")

