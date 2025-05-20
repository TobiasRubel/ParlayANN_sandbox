import struct
import matplotlib.pyplot as plt
import numpy as np


def read_binary_file(filename):
    with open(filename, "rb") as f:
        num_points = struct.unpack("I", f.read(4))[0]
        data = np.fromfile(f, dtype=np.float32, count=2 * num_points)
        points = data.reshape((num_points, 2))
    return points


def plot_points(points, label=None):
    x, y = points[:, 0], points[:, 1]
    plt.scatter(x, y, s=3, alpha=0.4, label=label)


def setup_plot():
    plt.figure(figsize=(8, 8))
    plt.xlabel("Straight-line distance")
    plt.ylabel("Shortest path distance")
    plt.title("Navigability of local neighborhoods")
    plt.grid(alpha=0.3)


def finalize_plot(output_image):
    all_points = np.concatenate([line.get_offsets() for line in plt.gca().collections])
    min_val, max_val = all_points.min(), all_points.max()
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, alpha=0.3)

    plt.legend()
    plt.savefig(output_image, bbox_inches="tight")
    plt.close()


setup_plot()

vamana_binary_filename = "../build/analysis/deep100M_vamana_local_shortest_paths.bin"
plot_points(read_binary_file(vamana_binary_filename), label="Vamana")

navhcnng_binary_filename = "../build/analysis/deep100M_navhcnng_local_shortest_paths.bin"
plot_points(read_binary_file(navhcnng_binary_filename), label="NavHCNNG")

output_image_filename = "local_shortest_paths.png"
finalize_plot(output_image_filename)
print(f"Scatterplot saved to {output_image_filename}")
