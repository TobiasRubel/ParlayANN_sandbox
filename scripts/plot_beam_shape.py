import struct
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Argument parser for input and output file paths
parser = argparse.ArgumentParser(description="Generate a scatterplot of projection length vs perpendicular length from a binary file.")
parser.add_argument(
    "--in-file",
    "-i",
    type=str,
    default="../../output/beam_shape.bin",
    help="Path to the input binary file (default: ../../output/beam_shape.bin)."
)
parser.add_argument(
    "--out-file",
    "-o",
    type=str,
    default="projection_vs_perpendicular.png",
    help="Path to save the output plot image (default: projection_vs_perpendicular.png)."
)

args = parser.parse_args()
in_path = args.in_file
out_path = args.out_file

# Read binary file
with open(in_path, "rb") as f:
    # Read number of sightings
    num_sightings = struct.unpack("I", f.read(4))[0]
    print(f"File loaded")
    print(f"Number of sightings: {num_sightings}")

    print(f"Reading data...")
    edge_lengths = np.frombuffer(f.read(num_sightings * 4), dtype=np.float32)
    projection_lengths = np.frombuffer(f.read(num_sightings * 4), dtype=np.float32)
    perpendicular_lengths = np.frombuffer(f.read(num_sightings * 4), dtype=np.float32)
    used = np.frombuffer(f.read(num_sightings), dtype=np.bool_)

# Separate data points based on 'used' flag
used_indices = used.nonzero()[0]
unused_indices = (~used).nonzero()[0]

print(f"Plotting data...")
plt.figure(figsize=(10, 6))

# Plot unused points first (background layer)
plt.scatter(projection_lengths[unused_indices], perpendicular_lengths[unused_indices], 
            s=6, c='blue', edgecolors='none', alpha=0.2, label="Not Used", zorder=1)

# Plot used points on top (foreground layer)
plt.scatter(projection_lengths[used_indices], perpendicular_lengths[used_indices], 
            s=10, c='red', edgecolors='black', linewidth=0.5, alpha=0.8, label="Used", zorder=2)

plt.title("Projection Length vs Perpendicular Length")
plt.xlabel("Projection Length")
plt.ylabel("Perpendicular Length")
plt.grid(False)

plt.xlim(left=0)
plt.ylim(bottom=0)

plt.legend()
plt.savefig(out_path, dpi=450)
print(f"Plot saved to '{out_path}'")
