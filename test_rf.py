import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, r2_score, mean_absolute_error, mean_absolute_percentage_error,
    matthews_corrcoef, balanced_accuracy_score, cohen_kappa_score, log_loss
)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# -------------------
# Load CSV
# -------------------
filename = r'C:\Users\bus_c\Desktop\ran\lmix3.csv'
df = pd.read_csv(filename)
file_names = df['filename'].values

# -------------------
# Prepare Data
# -------------------
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# Label Encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale Features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test, train_files, test_files = train_test_split(
    X_scaled, y_encoded, file_names, test_size=0.2, random_state=42, shuffle=True, stratify=y_encoded
)

# -------------------
# Train Random Forest
# -------------------
rf_model = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42)
start_time = time.perf_counter()
rf_model.fit(X_train, y_train)
end_time = time.perf_counter()
print(f"Training time: {end_time - start_time:.2f} seconds")

# -------------------
# Predictions
# -------------------
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)

# -------------------
# Evaluation Metrics
# -------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba[:, 1])
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
cohen_kappa = cohen_kappa_score(y_test, y_pred)
log_loss_value = log_loss(y_test, y_proba)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
g_mean = np.sqrt(recall * specificity)

# Plot Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Print Metrics
print("Evaluation Results:")
metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall (Sensitivity)": recall,
    "Specificity": specificity,
    "G-Mean": g_mean,
    "F1-Score": f1,
    "ROC-AUC": roc_auc,
    "Balanced Accuracy": balanced_acc,
    "Log Loss": log_loss_value,
    "MAE": mae,
    "MCC": mcc,
    "Cohen's Kappa": cohen_kappa
}
for k, v in metrics.items():
    print(f"{k}: {v:.2f}")

# Class-specific accuracy
correct_class_0 = cm[0, 0] / cm[0].sum() * 100
correct_class_1 = cm[1, 1] / cm[1].sum() * 100
print(f"Class 0 Correctly Predicted: {correct_class_0:.2f}%")
print(f"Class 1 Correctly Predicted: {correct_class_1:.2f}%")

# -------------------
# SHAP Analysis
# -------------------
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Top 10 features for Class 1
shap.summary_plot(
    shap_values[1], X_test, feature_names=df.columns[1:-1],
    plot_type="bar", max_display=10, color="royalblue"
)

# Top 10 features for Class 0
shap.summary_plot(
    shap_values[0], X_test, feature_names=df.columns[1:-1],
    plot_type="bar", max_display=10, color="royalblue"
)
