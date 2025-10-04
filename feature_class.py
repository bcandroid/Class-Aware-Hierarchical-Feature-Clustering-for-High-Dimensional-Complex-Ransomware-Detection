import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, r2_score, mean_absolute_error, mean_absolute_percentage_error,
    matthews_corrcoef, balanced_accuracy_score, cohen_kappa_score, log_loss
)
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the CSV file
file_name = r'C:\...\mix.csv'
dff = pd.read_csv(file_name)
df = dff.iloc[:, 1:-1]

first_column = dff.iloc[:, 0]
last_column = dff.iloc[:, -1]

ones = df[dff.iloc[:, -1] == 1]
zeros = df[dff.iloc[:, -1] == 0]

one_cor = ones.corr()
zero_cor = zeros.corr()

# Combine clusters function
def combine(df, clusters):
    if not clusters:
        return pd.DataFrame()
    combined_data = [df[[df.columns[i] for i in cluster]].sum(axis=1) for cluster in clusters]
    combined_df = pd.concat(combined_data, axis=1)
    combined_df.columns = [f"Cluster_{i}" for i in range(len(combined_df.columns))]
    return combined_df

# Delete clusters function
def dell(t, h):
    for cluster in t:
        for elem in cluster:
            for l in h:
                if elem in l:
                    l.remove(elem)
    return h

# Find classes from dataframe
def classes(df):
    G = nx.Graph()
    for row, col in zip(df["Row"], df["Column"]):
        G.add_edge(row, col)
    clusters = list(nx.connected_components(G))
    return [sorted(cluster) for cluster in clusters if len(cluster) > 1]

# Write clusters to txt
def write_to_txt(filename, data, df):
    with open(filename, "w", encoding="utf-8") as f:
        for cluster in data:
            column_names = [df.columns[i] for i in cluster]
            f.write(" ".join(column_names) + "\n")

# Evaluate parameters
def evaluate_params(all_1, all_2, one_1, one_2, zero_1, zero_2):
    try:
        one_high = set(zip(*((one_cor >= one_1).values).nonzero()))
        zero_low = set(zip(*((zero_cor <= zero_2).values).nonzero()))
        one_low = set(zip(*((one_cor <= one_2).values).nonzero()))
        zero_high = set(zip(*((zero_cor >= zero_1).values).nonzero()))
        all_one_high = set(zip(*((one_cor >= all_1).values).nonzero()))
        all_zero_high = set(zip(*((zero_cor >= all_2).values).nonzero()))

        common_high = all_one_high & all_zero_high
        overlap_df = pd.DataFrame(one_high & zero_low, columns=["Row", "Column"])
        mismatch_df = pd.DataFrame(one_low & zero_high, columns=["Row", "Column"])
        common_df = pd.DataFrame(common_high, columns=["Row", "Column"])

        t = classes(common_df)
        h = classes(overlap_df)
        y = classes(mismatch_df)
        w = dell(t, h)
        q = dell(t, y)

        new_df = pd.concat([first_column, combine(df, t), combine(df, w), combine(df, q), last_column], axis=1)
        x = new_df.iloc[:, 1:-1]
        y_label = new_df.iloc[:, -1]

        le = LabelEncoder()
        y_encoded = le.fit_transform(y_label)
        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(x)

        x_train, x_test, y_train, y_test = train_test_split(
            x_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

        model = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        y_proba = model.predict_proba(x_test)

        kappa = cohen_kappa_score(y_test, y_pred)
        loss = log_loss(y_test, y_proba)

        print(f"Evaluating parameters: one_1={one_1}, one_2={one_2}, zero_1={zero_1}, zero_2={zero_2}, all_1={all_1}, all_2={all_2}")
        print(f"Kappa: {kappa:.4f}, Log Loss: {loss:.4f}")
        return kappa, loss, t, w, q
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None, None, None

# Main correlation-based parameter search
def correlation(all_1, all_2, one_1, one_2, zero_1, zero_2):
    best_params = {}
    best_kappa = -float("inf")
    best_log = float("inf")

    for param_set, name in zip([all_1, all_2, one_1, one_2, zero_1, zero_2],
                               ["all_1", "all_2", "one_1", "one_2", "zero_1", "zero_2"]):
        for i, val in enumerate(param_set):
            kappa, log, _, _, _ = evaluate_params(
                best_params.get("all_1", all_1[0]),
                best_params.get("all_2", all_2[0]),
                best_params.get("one_1", one_1[0]),
                best_params.get("one_2", one_2[0]),
                best_params.get("zero_1", zero_1[0]),
                best_params.get("zero_2", zero_2[0])
            )
            if kappa is not None and (kappa > best_kappa or (kappa == best_kappa and log < best_log)):
                best_kappa = kappa
                best_log = log
                best_params[name] = val

    return best_kappa, best_log, best_params

# Evaluate and save results
def evaluate_final(best_params):
    _, _, t, w, q = evaluate_params(
        best_params["all_1"], best_params["all_2"],
        best_params["one_1"], best_params["one_2"],
        best_params["zero_1"], best_params["zero_2"]
    )

    write_to_txt("all_clusters.txt", t, df)
    write_to_txt("rans_clusters.txt", w, df)
    write_to_txt("good_clusters.txt", q, df)

    new_df = pd.concat([first_column, combine(df, t), combine(df, w), combine(df, q), last_column], axis=1)
    x = new_df.iloc[:, 1:-1]
    y_label = new_df.iloc[:, -1]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_label)
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    model = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)

    # Metrics
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    g_mean = np.sqrt(recall * specificity)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    loss = log_loss(y_test, y_proba)

    # Visualizations
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.hist(y_proba[:, 1], bins=50, alpha=0.7, label="Positive (1) Probabilities")
    plt.hist(y_proba[:, 0], bins=50, alpha=0.7, label="Negative (0) Probabilities")
    plt.xlabel("Prediction Probability")
    plt.ylabel("Sample Count")
    plt.title("Predicted Class Probability Distribution")
    plt.legend()
    plt.show()

    # Print results
    print("\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, Specificity: {specificity:.2f}")
    print(f"G-Mean: {g_mean:.2f}, F1-Score: {f1:.2f}, ROC-AUC: {roc_auc:.2f}, Balanced Accuracy: {balanced_acc:.2f}")
    print(f"Log Loss: {loss:.2f}, Cohen's Kappa: {kappa:.2f}")

# Define parameter ranges
one_1 = np.arange(0.95, 0.5, -0.01)
one_2 = np.arange(0.5, -0.15, -0.01)
zero_1 = np.arange(0.95, 0.5, -0.01)
zero_2 = np.arange(0.5, -0.15, -0.01)
all_1 = np.arange(0.95, 0.5, -0.01)
all_2 = np.arange(0.95, 0.5, -0.01)

# Run correlation search
best_kappa, best_loss, best_params = correlation(all_1, all_2, one_1, one_2, zero_1, zero_2)

# Print best results
print("\n🌟 Best Results:")
print(f"Log Loss: {best_loss:.4f}, Cohen Kappa: {best_kappa:.4f}")
print("Parameters:")
for k, v in best_params.items():
    print(f"{k}: {v:.2f}")

# Evaluate final model with best parameters
evaluate_final(best_params)
