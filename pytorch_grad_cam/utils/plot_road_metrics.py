import matplotlib.pyplot as plt
import numpy as np

# ROAD scores from the four CAM methods
scores = {
    "GradCAM": {
        "ROAD_MoRF": -4.436559,
        "ROAD_LeRF": 0.0710516,
        "ROAD_MoRF_Avg": -5.5910883,
        "ROAD_LeRF_Avg": -0.24704061,
        "ROAD_Combined": 2.6629207
    },
    "ScoreCAM": {
        "ROAD_MoRF": -4.601909,
        "ROAD_LeRF": 0.09860516,
        "ROAD_MoRF_Avg": -5.6209316,
        "ROAD_LeRF_Avg": -0.26004547,
        "ROAD_Combined": 2.6834533
    },
    "AblationCAM": {
        "ROAD_MoRF": -4.9001856,
        "ROAD_LeRF": 0.06025076,
        "ROAD_MoRF_Avg": -5.654843,
        "ROAD_LeRF_Avg": -0.22118847,
        "ROAD_Combined": 2.7183824
    },
    "FinerCAM": {
        "ROAD_MoRF": -4.49748,
        "ROAD_LeRF": 0.01859283,
        "ROAD_MoRF_Avg": -5.562819,
        "ROAD_LeRF_Avg": -0.14141873,
        "ROAD_Combined": 2.7146535
    }
}

# # Plotting
# metrics_to_plot = ["ROAD_MoRF", "ROAD_LeRF", "ROAD_Combined"]
# bar_width = 0.2
# x = np.arange(len(scores))
#
# fig, ax = plt.subplots(figsize=(10, 6))
#
# for idx, metric in enumerate(metrics_to_plot):
#     values = [scores[method][metric] for method in scores]
#     ax.bar(x + idx * bar_width, values, width=bar_width, label=metric)
#
# ax.set_xticks(x + bar_width)
# ax.set_xticklabels(scores.keys())
# ax.set_ylabel("Score")
# ax.set_title("ROAD Evaluation Metrics Across CAM Methods")
# ax.axhline(0, color='black', linewidth=0.8)
# ax.legend()
# plt.tight_layout()
# plt.grid(axis='y', linestyle='--', linewidth=0.5)
# plt.show()

# Create subplots for ROAD_MoRF_Avg, ROAD_LeRF_Avg, and ROAD_Combined with shared y-axis for the first two
fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={'width_ratios': [1, 1, 1]})

metrics_to_plot = ["ROAD_MoRF_Avg", "ROAD_LeRF_Avg", "ROAD_Combined"]
titles = ["Average MoRF", "Average LeRF", "Combined Score"]

# Get common y-limits for the first two plots
morfs = [scores[method]["ROAD_MoRF_Avg"] for method in scores]
lerfs = [scores[method]["ROAD_LeRF_Avg"] for method in scores]
common_ylim = (min(min(morfs), min(lerfs)) - 0.5, max(max(morfs), max(lerfs)) + 0.5)

for i, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
    values = [scores[method][metric] for method in scores]
    axes[i].bar(scores.keys(), values, color='lightcoral' if i == 0 else 'lightseagreen' if i == 1 else 'steelblue', edgecolor='black')
    axes[i].set_title(title)
    axes[i].set_ylabel("Score")
    axes[i].set_xlabel("CAM Method")
    axes[i].axhline(0, color='gray', linewidth=0.8, linestyle='--')
    axes[i].grid(axis='y', linestyle='--', linewidth=0.5)
    if i < 2:
        axes[i].set_ylim(common_ylim)

fig.suptitle("ROAD Evaluation (Average Scores) by CAM Method", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
