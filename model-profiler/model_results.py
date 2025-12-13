import json
import matplotlib.pyplot as plt

with open("model-data/power_profiles.json", "r") as f:
    data = json.load(f)

models = list(data.keys())
latency = [data[model]["avg_inference_time_seconds"] for model in models]
accuracy = [data[model]["accuracy"] for model in models]

plt.figure(figsize=(10, 6))
plt.scatter(latency, accuracy, s=100, alpha=0.7)

for i, model in enumerate(models):
    plt.annotate(
        model, (latency[i], accuracy[i]), xytext=(5, 5), textcoords="offset points"
    )

plt.xlabel("Latency (seconds)")
plt.ylabel("Accuracy")
plt.title("Latency vs Accuracy for YOLOv10 Models")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("model_results.png", dpi=300, bbox_inches="tight")
plt.show()
