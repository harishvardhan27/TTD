"""Generate comparison charts from results_cnn.json produced by demo_cnn_pipeline.py."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("results_cnn.json") as f:
    data = json.load(f)

results = data["results"]
names = [r["name"] for r in results]
size = [r["size_mb"] for r in results]
comp_ratio = [r["compression_ratio"] for r in results]
latency = [r["mean_latency_ms"] for r in results]
err = [r["relative_output_error"] for r in results]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

axes[0, 0].bar(names, size, color="#4C72B0")
axes[0, 0].set_ylabel("Model size (MB)")
axes[0, 0].set_title("Model size")

axes[0, 1].bar(names, comp_ratio, color="#55A868")
axes[0, 1].set_ylabel("Compression ratio (x)")
axes[0, 1].set_title("Compression ratio vs. dense baseline")

axes[1, 0].bar(names, latency, color="#C44E52")
axes[1, 0].set_ylabel("Mean latency (ms) / batch=16, CPU")
axes[1, 0].set_title("Inference latency")

axes[1, 1].bar(names, err, color="#8172B2")
axes[1, 1].set_ylabel("Relative output error vs. dense teacher")
axes[1, 1].set_title("Accuracy proxy (lower is better)")

for ax in axes.flat:
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")

fig.tight_layout()
fig.savefig("tt_cnn_results.png", dpi=150)
print("Saved tt_cnn_results.png")
