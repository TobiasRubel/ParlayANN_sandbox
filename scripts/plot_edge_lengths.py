import struct
import numpy as np
import matplotlib.pyplot as plt

in_path = "centrality.bin"
out_path = "edge_lengths.png"

with open(in_path, "rb") as f:
    num_edges = struct.unpack("I", f.read(4))[0]
    print(f"File loaded")
    print(f"Number of edges: {num_edges}")

    print(f"Reading data...")
    sources = np.frombuffer(f.read(num_edges * 4), dtype=np.uint32)
    targets = np.frombuffer(f.read(num_edges * 4), dtype=np.uint32)
    distances = np.frombuffer(f.read(num_edges * 4), dtype=np.float32)

print(f"Checking for duplicates...")
unique_edges = set()
unique_distances = []
for source, target, distance in zip(sources, targets, distances):
    if target > source:
        source, target = target, source
    if (source, target) not in unique_edges:
        unique_edges.add((source, target))
        unique_distances.append(distance)

print(f"Plotting data...")
plt.figure(figsize=(10, 6))
plt.hist(distances, bins=100, color='blue', edgecolor='none', alpha=0.7)
plt.hist(unique_distances, bins=100, color='blue', edgecolor='none', alpha=1.0)
plt.title("Edge Length Distribution (bottom bars are undirected)")
plt.xlabel("Edge Length")
plt.ylabel("Frequency")
plt.grid(False)

plt.xlim(left=0)
plt.ylim(bottom=0)

plt.savefig(out_path, dpi=450)
print(f"Plot saved to '{out_path}'")