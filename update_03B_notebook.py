"""
Script to update 03B_Extended_ML.ipynb:
1. Add explanatory comments distinguishing Regression vs Classification
2. Add LogisticRegression and GaussianNB to the imports and classifier phase
"""
import json
import os

NB_PATH = os.path.join('notebooks', '03B_Extended_ML.ipynb')

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# --- 1. Update title cell to explain the two approaches ---
for cell in cells:
    if cell['cell_type'] == 'markdown' and any('# Extended ML Tasks' in line for line in cell['source']):
        cell['source'] = [
            "# Extended ML Tasks\n",
            "\n",
            "This notebook covers **two complementary approaches** to the NASA C-MAPSS predictive maintenance problem:\n",
            "\n",
            "1. **Regression** — Predict the *exact* Remaining Useful Life (RUL) in cycles. Models: KNN Regressor, SVR.\n",
            "2. **Classification** — Predict *whether* an engine will fail within 30 cycles (binary yes/no). Models: Random Forest, XGBoost, KNN, SVC, Logistic Regression, Naive Bayes.\n",
            "\n",
            "> **Note:** Some algorithms (such as Logistic Regression and Naive Bayes) are **classification-only** algorithms — they cannot predict continuous RUL values. To use them, we re-frame the problem as binary classification using the label `will_fail_30 = (RUL <= 30)`.\n",
            "> This re-framing does **not** affect the separate regression analysis.\n",
        ]
        break

# --- 2. Add LogisticRegression and GaussianNB to imports ---
for cell in cells:
    if cell['cell_type'] == 'code' and any('from sklearn.svm import SVR, SVC' in line for line in cell['source']):
        new_source = []
        for line in cell['source']:
            new_source.append(line)
            if 'from sklearn.svm import SVR, SVC' in line:
                new_source.append("from sklearn.linear_model import LogisticRegression\n")
                new_source.append("from sklearn.naive_bayes import GaussianNB\n")
        cell['source'] = new_source
        break

# --- 3. Update Phase 4 title ---
for cell in cells:
    if cell['cell_type'] == 'markdown' and any('## Phase 4' in line for line in cell['source']):
        cell['source'] = [
            "## Phase 4: Binary Classification (6 models)\n",
            "\n",
            "Since the core task is **regression** (predicting exact RUL), we also evaluate a **classification** formulation:\n",
            "engines with `RUL <= 30` are labeled as `1` (failing), others as `0` (healthy).\n",
            "\n",
            "This allows us to use **classification-only** algorithms like **Logistic Regression** and **Naive Bayes**,\n",
            "which cannot produce continuous RUL predictions but can answer the question: *\"Will this engine fail soon?\"*\n",
            "\n",
            "> **Important:** Naive Bayes assumes feature independence, which is strongly violated by correlated sensor data.\n",
            "> It is included here for completeness but is expected to perform poorly on this dataset.\n",
        ]
        break

# --- 4. Update Phase 4 code to include 6 classifiers ---
for cell in cells:
    if cell['cell_type'] == 'code' and any("'SVC': svc_clf" in line for line in cell['source']):
        cell['source'] = [
            "# --- Classification Models ---\n",
            "# These models predict whether an engine will fail within 30 cycles (binary classification).\n",
            "# Note: Logistic Regression and Naive Bayes are CLASSIFICATION-ONLY algorithms.\n",
            "# They cannot predict continuous RUL values, only class labels.\n",
            "\n",
            "rf_clf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)\n",
            "xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)\n",
            "knn_clf = KNeighborsClassifier(n_neighbors=5)\n",
            "svc_clf = SVC(probability=True, random_state=42)\n",
            "lr_clf = LogisticRegression(max_iter=1000, random_state=42)  # Classification only\n",
            "nb_clf = GaussianNB()  # Classification only; assumes feature independence\n",
            "\n",
            "models_clf = {\n",
            "    'Random Forest Classifier': rf_clf,\n",
            "    'XGBoost Classifier': xgb_clf,\n",
            "    'KNN Classifier': knn_clf,\n",
            "    'SVC': svc_clf,\n",
            "    'Logistic Regression': lr_clf,\n",
            "    'Naive Bayes (GaussianNB)': nb_clf\n",
            "}\n",
            "\n",
            "fig, axes = plt.subplots(2, 3, figsize=(18, 8))\n",
            "axes = axes.flatten()\n",
            "\n",
            "for i, (name, model) in enumerate(models_clf.items()):\n",
            "    # Fit and Predict\n",
            "    model.fit(X_train, y_train_class)\n",
            "    preds = model.predict(X_test)\n",
            "    \n",
            "    # Calculate Metrics\n",
            "    acc = accuracy_score(y_test_class, preds)\n",
            "    prec = precision_score(y_test_class, preds)\n",
            "    rec = recall_score(y_test_class, preds)\n",
            "    f1 = f1_score(y_test_class, preds)\n",
            "    \n",
            "    # Save Metrics\n",
            "    all_metrics.append({\n",
            "        'Model': name,\n",
            "        'Task': 'Classification',\n",
            "        'RMSE': np.nan, 'MAE': np.nan, 'NASA_Score': np.nan,\n",
            "        'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1\n",
            "    })\n",
            "    \n",
            "    # Plot CM\n",
            "    cm = confusion_matrix(y_test_class, preds)\n",
            "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)\n",
            "    axes[i].set_title(f\"{name}\\nF1: {f1:.2f}\")\n",
            "    axes[i].set_xlabel(\"Predicted\")\n",
            "    axes[i].set_ylabel(\"Actual\")\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
        ]
        break

# --- 5. Clear all outputs so the notebook runs fresh ---
for cell in cells:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

# --- Save ---
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Successfully updated {NB_PATH}")
print("  - Added regression vs. classification explanation to title")
print("  - Added LogisticRegression and GaussianNB imports")
print("  - Updated Phase 4 to include 6 classifiers with comments")
print("  - Cleared all outputs for fresh execution")
