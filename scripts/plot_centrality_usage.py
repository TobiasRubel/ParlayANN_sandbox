import struct
import numpy as np
import matplotlib.pyplot as plt

in_path = "centrality.bin"
out_path = "centrality_usage.png"

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