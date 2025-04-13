# We used GPT 4o to speed up the process of visualizing our results
# The code in this file was assisted by the use of this LLM

import matplotlib.pyplot as plt
import numpy as np
import re


def load_road_metrics(filepath):
    morf_values = {}
    lerf_values = {}
    morf_avg = None
    lerf_avg = None
    combined_avg = None

    with open(filepath, 'r') as f:
        for line in f:
            match = re.match(r"(ROAD_\w+?): \[([-0-9eE.+]+)\]", line.strip())
            if match:
                key, value = match.groups()
                value = float(value)
                if "MoRF_" in key and "Avg" not in key:
                    p = int(key.split('_')[-1])
                    morf_values[p] = value
                elif "LeRF_" in key and "Avg" not in key:
                    p = int(key.split('_')[-1])
                    lerf_values[p] = value
                elif key == "ROAD_MoRF_Avg":
                    morf_avg = value
                elif key == "ROAD_LeRF_Avg":
                    lerf_avg = value
                elif key == "ROAD_Combined":
                    combined_avg = value


    percentiles = sorted(morf_values.keys(), reverse=True)
    morf_scores = [morf_values[p] for p in percentiles]
    lerf_scores = [lerf_values[p] for p in percentiles]

    return percentiles, morf_scores, lerf_scores, morf_avg, lerf_avg, combined_avg


filepath = 'output/ablationcam_road_scores.txt'
percentiles, morf_values, lerf_values, morf_avg, lerf_avg, combined_avg = load_road_metrics(filepath)


fig, ax = plt.subplots(figsize=(12, 7))

bar_width = 0.35
index = np.arange(len(percentiles))

morf_bars = ax.bar(index - bar_width/2, morf_values, bar_width, label='MoRF', color='steelblue', alpha=0.8)
lerf_bars = ax.bar(index + bar_width/2, lerf_values, bar_width, label='LeRF', color='firebrick', alpha=0.8)


ax.axhline(y=morf_avg, color='steelblue', linestyle='--', linewidth=2, alpha=0.7, label='MoRF Avg')
ax.axhline(y=lerf_avg, color='firebrick', linestyle='--', linewidth=2, alpha=0.7, label='LeRF Avg')


def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        y_pos = height + 0.2 if height >= 0 else height - 0.4
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{height:.2f}', ha='center', va='bottom', rotation=0, fontsize=8)

add_labels(morf_bars)
add_labels(lerf_bars)


ax.set_xlabel('Percentile', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('ROAD Metrics (ScoreCAM)', fontsize=14, fontweight='bold')
ax.set_xticks(index)
ax.set_xticklabels(percentiles)
ax.legend(fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.3)


y_min = min(min(morf_values), min(lerf_values)) - 1
y_max = max(max(morf_values), max(lerf_values)) + 1
ax.set_ylim(y_min, y_max)

plt.tight_layout()
plt.savefig('road_metrics_bar_plot_reversed.png', dpi=300)
plt.show()

methods = ['GradCAM', 'ScoreCAM', 'FinerCAM', 'AblationCAM']
scores = [2.6702116, 2.6773772, 2.7132516, 2.717256]


plt.figure(figsize=(8, 5))
bars = plt.bar(methods, scores, color='steelblue')
plt.ylabel('Combined ROAD Score')
plt.title('Overall ROAD performance across CAM Methods')


for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.001, f'{yval:.4f}', ha='center', va='bottom')

plt.ylim(0, max(scores) + 0.5)  # y-axis starts at 0
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
