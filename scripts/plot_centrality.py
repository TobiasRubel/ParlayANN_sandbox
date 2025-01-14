import struct
import matplotlib.pyplot as plt

file_path = "centrality.bin"

pairs = []
with open(file_path, "rb") as f:
    num_pairs_data = f.read(4)
    num_pairs = struct.unpack("I", num_pairs_data)[0]
    print(f"File loaded")
    print(f"Number of pairs: {num_pairs}")

    print(f"Reading data...")
    for _ in range(num_pairs):
        value_t_data = f.read(4)
        centrality_data = f.read(4)

        value_t = struct.unpack("I", value_t_data)[0]
        centrality = struct.unpack("I", centrality_data)[0]
        pairs.append((value_t, centrality))
    print(f"Done")

print(f"Plotting data...")
values, centralities = zip(*pairs)

plt.figure(figsize=(10, 6))
plt.scatter(values, centralities, s=4, c='blue', edgecolors='none', alpha=0.5)
plt.title("Scatterplot of Edge Lengths and Centralities")
plt.xlabel("Edge Length")
plt.ylabel("Centrality")
plt.grid(True)

plt.xlim(left=0)
plt.ylim(bottom=0)

plt.savefig("centrality.png", dpi=600)

print(f"Plot saved to 'centrality.png'")