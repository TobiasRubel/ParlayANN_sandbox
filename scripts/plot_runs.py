import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Hardcoded list of CSV files, corresponding index names, and build times
csv_data = [
    {"path": "/ssd2/richard/anndata/deep/logs/vamana.log", "name": "vamana-R64-L128-a1.05", "build_time": "1227.1s"},
    {"path": "/ssd2/richard/anndata/deep/logs/navhcnng_c500_l001_f5-3.log", "name": "navhcnng_c500_tll10000_fl0.001_f5-3", "build_time": "997.6s"},
    {"path": "/ssd2/richard/anndata/deep/logs/navhcnng_c500_tll10000_fl0005_f5-3-2.log", "name": "navhcnng_c500_tll10000_fl0.0005_f5-3-2", "build_time": "1450s"},
    {"path": "/ssd2/richard/anndata/deep/logs/navhcnng_c500_tll10000_fl0005_f8-3.log", "name": "navhcnng_c500_tll10000_fl0.0005_f8-3", "build_time": "1363s"},
]

plot_type = 1
plot_types = ["qps", "distcmps"]


plt.figure(figsize=(10, 6))

for entry in csv_data:
    file = entry["path"]
    name = entry["name"]
    build_time = entry["build_time"]

    try:
        # Load CSV without headers
        df = pd.read_csv(file, header=None)

        # Ensure there are at least 5 columns
        if df.shape[1] < 5:
            print(f"Skipping {file}: less than 5 columns")
            continue

        # Convert to NumPy array before filtering
        data_array = df.to_numpy()

        # Filter rows where recall (4th column) is >= 0.8
        filtered_rows = data_array[data_array[:, 3] >= 0.8]

        if filtered_rows.size == 0:
            print(f"Skipping {file}: no recall >= 0.8")
            continue

        # Extract recall and QPS
        recall_values = filtered_rows[:, 3]
        qps_values = filtered_rows[:, 4]
        dist_comps = filtered_rows[:, 5]

        # Sort by recall
        sorted_indices = np.argsort(recall_values)
        recall_values = recall_values[sorted_indices]
        qps_values = qps_values[sorted_indices]
        dist_comps = dist_comps[sorted_indices]

        # Plot with label including name and build time

        if plot_type == 0:
            plt.plot(recall_values, qps_values, marker='o', label=f"{name} ({build_time})")
        else:
            plt.plot(recall_values, dist_comps, marker='o', label=f"{name} ({build_time})")

    except Exception as e:
        print(f"Error processing {file}: {e}")

# Configure plot
plt.xlabel("Recall")
if plot_type == 0:
    plt.ylabel("QPS (log scale)")
else:
    plt.ylabel("Dist Comps (log scale)")
plt.yscale("log")
if plot_type == 0:
    plt.title("QPS vs Recall Trade-off")
else:
    plt.title("Recall vs Dist Comps Trade-off")
plt.legend()
plt.grid(True, which="both", linestyle="--")

# Show plot
plt.savefig("deep-100M.png", dpi=300)