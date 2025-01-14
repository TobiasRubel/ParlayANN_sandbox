import struct
import numpy as np
import matplotlib.pyplot as plt

file_path = "centrality.bin"

with open(file_path, "rb") as f:
    num_pairs_data = f.read(4)
    num_pairs = struct.unpack("I", num_pairs_data)[0]
    print(f"File loaded")
    print(f"Number of pairs: {num_pairs}")

    print(f"Reading data...")
    distances = np.frombuffer(f.read(num_pairs * 4), dtype=np.float32)
    centralities = np.frombuffer(f.read(num_pairs * 4), dtype=np.uint32)

print(f"Plotting data...")
plt.figure(figsize=(10, 6))
plt.scatter(distances, centralities, s=6, c='blue', edgecolors='none', alpha=0.2)
plt.title("Edge Length vs Usage Frequency")
plt.xlabel("Edge Length")
plt.ylabel("Centrality")
plt.grid(False)

plt.xlim(left=0)
plt.ylim(bottom=0)

plt.savefig("centrality.png", dpi=450)
print(f"Plot saved to 'centrality.png'")