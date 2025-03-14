import struct
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Generate a line plot of average centrality for each edge length from a binary file.")
parser.add_argument(
    "--in-file",
    "-i",
    type=str,
    default="centrality.bin",
    help="Path to the input binary file (default: centrality.bin)."
)
parser.add_argument(
    "--out-file",
    "-o",
    type=str,
    default="centrality_avg.png",
    help="Path to save the output plot image (default: centrality_avg.png)."
)

args = parser.parse_args()
in_path = args.in_file
out_path = args.out_file

with open(in_path, "rb") as f:
    num_edges = struct.unpack("I", f.read(4))[0]
    print(f"File loaded")
    print(f"Number of edges: {num_edges}")

    print(f"Reading data...")
    sources = np.frombuffer(f.read(num_edges * 4), dtype=np.uint32)
    targets = np.frombuffer(f.read(num_edges * 4), dtype=np.uint32)
    distances = np.frombuffer(f.read(num_edges * 4), dtype=np.float32)
    centralities = np.frombuffer(f.read(num_edges * 4), dtype=np.uint32)

num_buckets = 100
max_distance = np.max(distances)
average_centralities = np.zeros(num_buckets)
counts = np.zeros(num_buckets)
for i in range(num_edges):
    bucket = int(distances[i] / max_distance * num_buckets)
    if bucket >= num_buckets:
        bucket = num_buckets - 1
    average_centralities[bucket] += centralities[i]
    counts[bucket] += 1
for i in range(num_buckets):
    if counts[i] > 0:
        average_centralities[i] /= counts[i]
bucket_centers = np.linspace(0, max_distance, num_buckets)

print(f"Plotting data...")
plt.figure(figsize=(10, 6))
plt.plot(bucket_centers, average_centralities, color="blue")
plt.title("Edge Length vs Average Centrality")
plt.xlabel("Edge Length")
plt.ylabel("Centrality")
plt.grid(False)

plt.xlim(left=0)
plt.ylim(bottom=0)

plt.savefig(out_path, dpi=450)
print(f"Plot saved to '{out_path}'")