import json
import os

notebook = {
 "cells": [],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}

def add_markdown(source):
    notebook["cells"].append({
        "cell_type": "markdown", 
        "id": str(len(notebook["cells"])), 
        "metadata": {}, 
        "source": [line + '\n' for line in source.split('\n')]
    })
    notebook["cells"][-1]["source"][-1] = notebook["cells"][-1]["source"][-1].rstrip('\n')

def add_code(source):
    notebook["cells"].append({
        "cell_type": "code", 
        "execution_count": None, 
        "id": str(len(notebook["cells"])), 
        "metadata": {}, 
        "outputs": [], 
        "source": [line + '\n' for line in source.split('\n')]
    })
    notebook["cells"][-1]["source"][-1] = notebook["cells"][-1]["source"][-1].rstrip('\n')

add_markdown("# 01B. Outlier Detection & Descriptive Statistics\n**Goal:** Calculate descriptive statistics (mean, median, mode, RMS, Std) and perform outlier detection on sensor data.")

add_code("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport os\n\nos.makedirs('plots', exist_ok=True)\n\n# Column names\ncols = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]\n\n# Load Data\ntrain_df = pd.read_csv('../data/train_FD001.txt', sep='\\\\s+', header=None, names=cols)\nsensors = [f's{i}' for i in range(1, 22)]\nprint(f\"Shape: {train_df.shape}\")\nif not train_df.empty:\n    display(train_df.head())")

add_markdown("## 1. Descriptive Statistics\nCalculating **Mean**, **Median**, **Mode**, **RMS** (Root Mean Square), and **Standard Deviation** for all 21 sensors.")
add_code("stats_df = pd.DataFrame(index=sensors)\n\n# 1. Mean\nstats_df['Mean'] = train_df[sensors].mean()\n\n# 2. Median\nstats_df['Median'] = train_df[sensors].median()\n\n# 3. Mode (taking the first mode if multiple exist)\nstats_df['Mode'] = train_df[sensors].mode().iloc[0]\n\n# 4. Standard Deviation\nstats_df['Std_Dev'] = train_df[sensors].std()\n\n# 5. RMS (Root Mean Square) = sqrt(mean(x^2))\nstats_df['RMS'] = np.sqrt((train_df[sensors]**2).mean())\n\ndisplay(stats_df)")

add_markdown("## 2. Visual Outlier Detection (Boxplots)\nUsing boxplots to visualize the spread and outliers for each sensor.")

add_code("plt.figure(figsize=(20, 8))\nsns.boxplot(data=train_df[sensors])\nplt.title('Sensor Value Distributions (Detecting Outliers)')\nplt.xticks(rotation=45)\nplt.savefig('plots/sensor_boxplots.png')\nplt.show()")

add_markdown("## 3. Z-Score Outlier Detection\nDetecting points that are more than 3 standard deviations away from the mean.")
add_code("z_scores = np.abs((train_df[sensors] - train_df[sensors].mean()) / train_df[sensors].std())\noutliers_z = (z_scores > 3)\n\nprint(\"Number of outliers per sensor using Z-Score > 3:\")\ndisplay(outliers_z.sum().to_frame(name=\"Outlier Count\").T)")

add_markdown("## 4. IQR (Interquartile Range) Method\nCalculating boundaries Q1 - 1.5*IQR and Q3 + 1.5*IQR.")
add_code("Q1 = train_df[sensors].quantile(0.25)\nQ3 = train_df[sensors].quantile(0.75)\nIQR = Q3 - Q1\n\nlower_bound = Q1 - 1.5 * IQR\nupper_bound = Q3 + 1.5 * IQR\n\noutliers_iqr = ((train_df[sensors] < lower_bound) | (train_df[sensors] > upper_bound))\n\nprint(\"Number of outliers per sensor using IQR method:\")\ndisplay(outliers_iqr.sum().to_frame(name=\"Outlier Count\").T)")

add_markdown("## 5. Handling Outliers (Clipping)\nInstead of dropping rows (which destroys the time-series sequence), we cap the extreme values to the IQR or Z-score boundaries so that noise is reduced.")
add_code("df_clipped = train_df.copy()\n\nfor col in sensors:\n    df_clipped[col] = np.clip(df_clipped[col], lower_bound[col], upper_bound[col])\n\nprint(\"Outliers have been clipped to the IQR boundaries. Validation of maximum values after clipping:\")\ndisplay(df_clipped[sensors].max().to_frame(name='Max After Clipping').T)")

nb_path = r'c:\Users\Sunil\.vscode\NASA CMaps\notebooks\01B_Outlier_Detection_and_Stats.ipynb'
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print(f'Notebook created at {nb_path}')
