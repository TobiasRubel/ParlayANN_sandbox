import struct
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Generate a scatterplot of edge length vs centrality from a binary file, including edge usage rates.")
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
    default="centrality_usage.png",
    help="Path to save the output plot image (default: centrality_usage.png)."
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
    usages = np.frombuffer(f.read(num_edges * 4), dtype=np.uint32)

print(f"Plotting data...")
plt.figure(figsize=(10, 6))
plt.scatter(distances, centralities, s=6, c='blue', edgecolors='none', alpha=0.2)
plt.scatter(distances, usages, s=6, c='red', edgecolors='none', alpha=0.2)
plt.title("Edge Length vs Usage Frequency")
plt.xlabel("Edge Length")
plt.ylabel("Usage Frequency")
plt.grid(False)

plt.xlim(left=0)
plt.ylim(bottom=0)

plt.savefig(out_path, dpi=450)
print(f"Plot saved to '{out_path}'")