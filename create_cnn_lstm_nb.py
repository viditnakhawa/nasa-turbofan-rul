"""
Creates notebooks/04B_CNN_LSTM_Hybrid.ipynb — a Colab-ready notebook
that implements a 1D-CNN + LSTM hybrid model for RUL prediction on FD001.
Run once from the project root:  python create_cnn_lstm_nb.py
"""
import json, pathlib

NB_PATH = pathlib.Path(__file__).parent / "notebooks" / "04B_CNN_LSTM_Hybrid.ipynb"

def md(cell_id, source):
    """Markdown cell helper."""
    if isinstance(source, str):
        source = source.split("\n")
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": [line if line.endswith("\n") else line + "\n" for line in source[:-1]] + [source[-1]],
    }

def code(cell_id, source):
    """Code cell helper."""
    if isinstance(source, str):
        source = source.split("\n")
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line if line.endswith("\n") else line + "\n" for line in source[:-1]] + [source[-1]],
    }

cells = []

# ── 0. Title ─────────────────────────────────────────────────────────────────
cells.append(md("title", [
    "# 04B. Deep Learning – CNN-LSTM Hybrid",
    "",
    "**Goal:** Use a 1D-CNN to automatically extract local degradation features ",
    "from the raw sensor window, then feed those features into an LSTM to capture ",
    "long-term temporal dependencies — achieving *best-of-both-worlds* performance.",
    "",
    "**Framework:** PyTorch  ",
    "**Environment:** Google Colab (GPU recommended)"
]))

# ── 1. Colab Setup ───────────────────────────────────────────────────────────
cells.append(md("setup_header", ["## 1. Colab Setup"]))

cells.append(code("setup_code", [
    "# ============================================================",
    "# COLAB SETUP — Run this cell first every session",
    "# ============================================================",
    "import pandas as pd",
    "import numpy as np",
    "import matplotlib.pyplot as plt",
    "import sys, os, shutil",
    "import torch",
    "import torch.nn as nn",
    "import torch.optim as optim",
    "from torch.utils.data import DataLoader, TensorDataset",
    "from sklearn.model_selection import train_test_split",
    "from sklearn.metrics import mean_squared_error, mean_absolute_error",
    "",
    "# Mount Google Drive",
    "from google.colab import drive",
    "drive.mount('/content/drive')",
    "",
    "# Base path — adjust if your project folder differs",
    "BASE = '/content/drive/MyDrive/nasa_turbofan_project'",
    "",
    "# Make utils importable",
    "os.makedirs('/content/utils', exist_ok=True)",
    "shutil.copy(f'{BASE}/utils/nasa_score.py', '/content/utils/nasa_score.py')",
    "sys.path.append('/content')",
    "from utils.nasa_score import nasa_score",
    "",
    "# Device",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
    "print(f'Device: {device}')",
    "",
    "# Load data",
    "train_df = pd.read_csv(f'{BASE}/data/processed_train.csv')",
    "test_df  = pd.read_csv(f'{BASE}/data/processed_test.csv')",
    "rul_test = pd.read_csv(f'{BASE}/data/RUL_FD001.txt',",
    "                        sep=r'\\s+', header=None, names=['RUL'])",
    "",
    "print(f'Train: {train_df.shape}, Test: {test_df.shape}')",
    "print('Setup complete!')",
]))

# ── 2. Sliding Window ────────────────────────────────────────────────────────
cells.append(md("window_header", [
    "## 2. Reshape Data into Sliding Windows (30 Timesteps)",
    "",
    "Reuses the same windowing approach from `04_Deep_Learning_LSTM.ipynb`."
]))

cells.append(code("window_code", [
    "SEQUENCE_LENGTH = 30",
    "",
    "feature_cols = [c for c in train_df.columns",
    "                if c not in ['unit', 'cycle', 'RUL', 'op1', 'op2', 'op3', 'will_fail_30']]",
    "",
    "def gen_sequence(df, seq_cols, seq_length):",
    "    data_array = df[seq_cols].values",
    "    num_elements = data_array.shape[0]",
    "    for start, stop in zip(range(0, num_elements - seq_length),",
    "                           range(seq_length, num_elements)):",
    "        yield data_array[start:stop, :]",
    "",
    "def gen_labels(df, label_col, seq_length):",
    "    data_array = df[label_col].values",
    "    num_elements = data_array.shape[0]",
    "    return data_array[seq_length:num_elements]",
    "",
    "# ── Training sequences ──",
    "X_train_seq, y_train_seq = [], []",
    "for unit in train_df['unit'].unique():",
    "    unit_data = train_df[train_df['unit'] == unit]",
    "    if len(unit_data) >= SEQUENCE_LENGTH:",
    "        X_train_seq.extend(list(gen_sequence(unit_data, feature_cols, SEQUENCE_LENGTH)))",
    "        y_train_seq.extend(list(gen_labels(unit_data, 'RUL', SEQUENCE_LENGTH)))",
    "",
    "X_train_seq = np.array(X_train_seq)",
    "y_train_seq = np.array(y_train_seq)",
    "print(f'X_train Sequence Shape: {X_train_seq.shape}')",
    "print(f'y_train Sequence Shape: {y_train_seq.shape}')",
    "",
    "# ── Test sequences (last window per engine) ──",
    "X_test_seq = []",
    "for unit in test_df['unit'].unique():",
    "    unit_data = test_df[test_df['unit'] == unit]",
    "    if len(unit_data) >= SEQUENCE_LENGTH:",
    "        seq = unit_data[feature_cols].values[-SEQUENCE_LENGTH:]",
    "    else:",
    "        pad_size = SEQUENCE_LENGTH - len(unit_data)",
    "        seq = np.pad(unit_data[feature_cols].values,",
    "                     ((pad_size, 0), (0, 0)), mode='constant')",
    "    X_test_seq.append(seq)",
    "",
    "X_test_seq = np.array(X_test_seq)",
    "y_test = rul_test['RUL'].values",
    "print(f'X_test Sequence Shape: {X_test_seq.shape}')",
]))

# ── 3. Train / Val Split ─────────────────────────────────────────────────────
cells.append(md("split_header", ["## 3. Train / Validation Split"]))

cells.append(code("split_code", [
    "X_t, X_v, y_t, y_v = train_test_split(",
    "    X_train_seq, y_train_seq, test_size=0.2, random_state=42)",
    "",
    "print(f'Train subset: {X_t.shape}, Val subset: {X_v.shape}')",
    "",
    "# Convert to PyTorch tensors",
    "X_train_tensor = torch.tensor(X_t, dtype=torch.float32).to(device)",
    "y_train_tensor = torch.tensor(y_t, dtype=torch.float32).view(-1, 1).to(device)",
    "",
    "X_val_tensor = torch.tensor(X_v, dtype=torch.float32).to(device)",
    "y_val_tensor = torch.tensor(y_v, dtype=torch.float32).view(-1, 1).to(device)",
    "",
    "X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32).to(device)",
    "",
    "train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),",
    "                          batch_size=1024, shuffle=True)",
    "val_loader   = DataLoader(TensorDataset(X_val_tensor, y_val_tensor),",
    "                          batch_size=1024, shuffle=False)",
]))

# ── 4. Model Definition ──────────────────────────────────────────────────────
cells.append(md("model_header", [
    "## 4. CNN-LSTM Hybrid Architecture",
    "",
    "```",
    "Input  (batch, 30, F)",
    "  │",
    "  ├── permute → (batch, F, 30)       # Conv1d needs channels-first",
    "  ├── Conv1d  → (batch, 64, 30)      # local feature extraction",
    "  ├── ReLU",
    "  ├── MaxPool1d(2) → (batch, 64, 15) # downsample",
    "  ├── Dropout(0.2)",
    "  ├── permute → (batch, 15, 64)      # back to seq-first for LSTM",
    "  │",
    "  ├── LSTM(hidden=100, layers=2, dropout=0.2)",
    "  ├── take last hidden state → (batch, 100)",
    "  ├── BatchNorm1d(100)",
    "  ├── Dropout(0.2)",
    "  │",
    "  └── Linear(100, 1) → RUL prediction",
    "```"
]))

cells.append(code("model_code", [
    "class CNNLSTMModel(nn.Module):",
    "    def __init__(self, input_size, cnn_out=64, kernel_size=5,",
    "                 lstm_hidden=100, lstm_layers=2, dropout=0.2):",
    "        super().__init__()",
    "",
    "        # ── CNN branch ──",
    "        self.conv1 = nn.Conv1d(",
    "            in_channels=input_size,",
    "            out_channels=cnn_out,",
    "            kernel_size=kernel_size,",
    "            padding=kernel_size // 2   # 'same' padding",
    "        )",
    "        self.relu = nn.ReLU()",
    "        self.pool = nn.MaxPool1d(kernel_size=2)",
    "        self.cnn_dropout = nn.Dropout(dropout)",
    "",
    "        # ── LSTM branch ──",
    "        self.lstm = nn.LSTM(",
    "            input_size=cnn_out,",
    "            hidden_size=lstm_hidden,",
    "            num_layers=lstm_layers,",
    "            batch_first=True,",
    "            dropout=dropout",
    "        )",
    "        self.bn = nn.BatchNorm1d(lstm_hidden)",
    "        self.lstm_dropout = nn.Dropout(dropout)",
    "",
    "        # ── Output ──",
    "        self.fc = nn.Linear(lstm_hidden, 1)",
    "",
    "    def forward(self, x):",
    "        # x: (batch, seq_len, features)",
    "",
    "        # Conv1d expects (batch, channels, seq_len)",
    "        x = x.permute(0, 2, 1)",
    "        x = self.conv1(x)          # → (batch, cnn_out, seq_len)",
    "        x = self.relu(x)",
    "        x = self.pool(x)           # → (batch, cnn_out, seq_len // 2)",
    "        x = self.cnn_dropout(x)",
    "",
    "        # Back to (batch, seq_len // 2, cnn_out) for LSTM",
    "        x = x.permute(0, 2, 1)",
    "        out, _ = self.lstm(x)",
    "        out = out[:, -1, :]        # last timestep",
    "        out = self.bn(out)",
    "        out = self.lstm_dropout(out)",
    "        out = self.fc(out)",
    "        return out",
    "",
    "input_size = len(feature_cols)",
    "model = CNNLSTMModel(input_size).to(device)",
    "print(model)",
    "print(f'\\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}')",
]))

# ── 5. Training Loop ─────────────────────────────────────────────────────────
cells.append(md("train_header", [
    "## 5. Training with Early Stopping & LR Scheduler"
]))

cells.append(code("train_code", [
    "def train_model(model, name, epochs=150, patience=15, lr=0.001):",
    "    criterion = nn.MSELoss()",
    "    optimizer = optim.Adam(model.parameters(), lr=lr)",
    "    scheduler = optim.lr_scheduler.ReduceLROnPlateau(",
    "        optimizer, mode='min', factor=0.5, patience=7)",
    "",
    "    train_losses, val_losses = [], []",
    "    best_val_loss = float('inf')",
    "    early_stop_counter = 0",
    "    best_epoch = 0",
    "",
    "    for epoch in range(epochs):",
    "        # ── Train ──",
    "        model.train()",
    "        running = 0.0",
    "        for X_batch, y_batch in train_loader:",
    "            optimizer.zero_grad()",
    "            preds = model(X_batch)",
    "            loss = criterion(preds, y_batch)",
    "            loss.backward()",
    "            optimizer.step()",
    "            running += loss.item() * X_batch.size(0)",
    "        train_loss = running / len(train_loader.dataset)",
    "        train_losses.append(train_loss)",
    "",
    "        # ── Validate ──",
    "        model.eval()",
    "        running = 0.0",
    "        with torch.no_grad():",
    "            for X_batch, y_batch in val_loader:",
    "                preds = model(X_batch)",
    "                loss = criterion(preds, y_batch)",
    "                running += loss.item() * X_batch.size(0)",
    "        val_loss = running / len(val_loader.dataset)",
    "        val_losses.append(val_loss)",
    "",
    "        scheduler.step(val_loss)",
    "",
    "        if (epoch + 1) % 10 == 0:",
    "            print(f'[{name}] Epoch {epoch+1}/{epochs} | '",
    "                  f'Train: {train_loss:.2f} | Val: {val_loss:.2f}')",
    "",
    "        # ── Early stopping & checkpointing ──",
    "        if val_loss < best_val_loss:",
    "            best_val_loss = val_loss",
    "            early_stop_counter = 0",
    "            best_epoch = epoch",
    "            torch.save(model.state_dict(), f'{BASE}/best_{name}.pt')",
    "        else:",
    "            early_stop_counter += 1",
    "",
    "        if early_stop_counter >= patience:",
    "            print(f'[{name}] Early stopping at epoch {epoch+1}. '",
    "                  f'Best epoch was {best_epoch+1}')",
    "            break",
    "",
    "    model.load_state_dict(",
    "        torch.load(f'{BASE}/best_{name}.pt', map_location=device))",
    "    print(f'[{name}] Done. Best Val Loss: {best_val_loss:.4f}')",
    "    return train_losses, val_losses, best_epoch",
    "",
    "print('Training CNN-LSTM Hybrid...')",
    "tl, vl, best_ep = train_model(model, 'CNNLSTMModel')",
]))

# ── 6. Training Curves ───────────────────────────────────────────────────────
cells.append(md("curves_header", ["## 6. Training Curves"]))

cells.append(code("curves_code", [
    "fig, ax = plt.subplots(figsize=(10, 5))",
    "epochs_range = range(1, len(tl) + 1)",
    "ax.plot(epochs_range, tl, label='Train Loss')",
    "ax.plot(epochs_range, vl, label='Val Loss')",
    "ax.axvline(best_ep + 1, color='red', linestyle='--', label='Best Epoch')",
    "ax.set_title('CNN-LSTM Hybrid — Training Curves')",
    "ax.set_xlabel('Epoch')",
    "ax.set_ylabel('MSE Loss')",
    "ax.legend()",
    "plt.tight_layout()",
    "plt.show()",
]))

# ── 7. Evaluation ────────────────────────────────────────────────────────────
cells.append(md("eval_header", ["## 7. Test-Set Evaluation"]))

cells.append(code("eval_code", [
    "model.eval()",
    "with torch.no_grad():",
    "    preds = model(X_test_tensor).cpu().numpy().flatten()",
    "",
    "rmse  = np.sqrt(mean_squared_error(y_test, preds))",
    "mae   = mean_absolute_error(y_test, preds)",
    "nasa  = nasa_score(y_test, preds)",
    "",
    "print(f'CNN-LSTM  RMSE:       {rmse:.2f}')",
    "print(f'CNN-LSTM  MAE:        {mae:.2f}')",
    "print(f'CNN-LSTM  NASA Score: {nasa:.2f}')",
]))

# ── 8. Residual Analysis ─────────────────────────────────────────────────────
cells.append(md("residual_header", ["## 8. Residual & Error Analysis"]))

cells.append(code("residual_code", [
    "residuals = y_test - preds",
    "",
    "fig, axes = plt.subplots(1, 2, figsize=(16, 6))",
    "",
    "# Predicted vs Actual",
    "axes[0].scatter(y_test, preds, alpha=0.6, edgecolors='black', s=40)",
    "axes[0].plot([0, max(y_test)], [0, max(y_test)], 'r--', label='Perfect')",
    "axes[0].set_xlabel('Actual RUL')",
    "axes[0].set_ylabel('Predicted RUL')",
    "axes[0].set_title('Predicted vs Actual RUL')",
    "axes[0].legend()",
    "",
    "# Residual histogram",
    "axes[1].hist(residuals, bins=25, edgecolor='black', alpha=0.7)",
    "axes[1].set_xlabel('Residual (Actual − Predicted)')",
    "axes[1].set_ylabel('Count')",
    "axes[1].set_title('Residual Distribution')",
    "",
    "plt.tight_layout()",
    "plt.show()",
]))

# ── 9. Save Metrics ──────────────────────────────────────────────────────────
cells.append(md("save_header", ["## 9. Save Metrics for Cross-Model Comparison"]))

cells.append(code("save_code", [
    "metrics = pd.DataFrame([{",
    "    'Model': 'CNN-LSTM Hybrid',",
    "    'RMSE':  round(rmse, 2),",
    "    'MAE':   round(mae, 2),",
    "    'NASA_Score': round(nasa, 2),",
    "}])",
    "",
    "metrics_path = f'{BASE}/data/cnn_lstm_metrics.csv'",
    "metrics.to_csv(metrics_path, index=False)",
    "print(f'Metrics saved to {metrics_path}')",
    "print(metrics.to_string(index=False))",
]))

# ── Assemble notebook ─────────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        },
        "colab": {
            "provenance": [],
            "gpuType": "T4"
        },
        "accelerator": "GPU"
    },
    "cells": cells,
}

NB_PATH.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"✓  Created {NB_PATH}")
