"""Convert SceneFactor semantic chunk to PLY point cloud for visualization."""
import numpy as np
from scipy.special import softmax
import sys

# Class names for 3D-FRONT
CLASS_NAMES = {
    0: "empty",
    1: "floor",
    2: "wall",
    3: "cabinet",
    4: "bed",
    5: "chair",
    6: "sofa",
    7: "table",
    8: "door",
    9: "window",
    10: "other",
}
# BGR colors for each class
COLORS = {
    0: (200, 200, 200),   # empty - gray
    1: (180, 150, 100),   # floor - brown
    2: (220, 220, 220),   # wall - white
    3: (150, 100, 50),    # cabinet - dark brown
    4: (100, 150, 200),   # bed - blue
    5: (50, 200, 100),    # chair - green
    6: (200, 100, 100),   # sofa - red
    7: (150, 150, 50),    # table - yellow
    8: (100, 50, 50),     # door - dark red
    9: (200, 200, 100),   # window - light yellow
    10: (100, 100, 100),  # other - dark gray
}


def sem_to_ply(sem_path, output_path, threshold=0.3, max_points=50000):
    data = np.load(sem_path)

    # Get class probabilities and argmax
    probs = softmax(data, axis=0)
    classes = np.argmax(probs, axis=0)
    max_probs = np.max(probs, axis=0)

    # Only keep non-empty voxels with high confidence
    mask = (classes != 0) & (max_probs > threshold)
    coords = np.array(np.where(mask)).T  # (N, 3): z, y, x in voxel space

    if len(coords) == 0:
        print("No non-empty voxels found, lowering threshold...")
        mask = max_probs > 0.1
        coords = np.array(np.where(mask)).T

    classes_filtered = classes[mask]

    # Subsample if too many points
    if len(coords) > max_points:
        idx = np.random.choice(len(coords), max_points, replace=False)
        coords = coords[idx]
        classes_filtered = classes_filtered[idx]

    print(f"Exporting {len(coords)} points...")

    # Generate PLY
    with open(output_path, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(coords)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")

        for i, (z, y, x) in enumerate(coords):
            cls = int(classes_filtered[i])
            # Scale coordinates for better visualization
            sx, sy, sz = x * 0.5, y * 0.5, z * 0.5
            r, g, b = COLORS.get(cls, (255, 255, 255))
            f.write(f"{sx:.1f} {sy:.1f} {sz:.1f} {r} {g} {b}\n")

    print(f"Class distribution:")
    unique, counts = np.unique(classes_filtered, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  {CLASS_NAMES.get(int(cls), 'unknown')}: {cnt}")


if __name__ == '__main__':
    sem_to_ply(
        '/home/lijiarui/Desktop/scene_factor/output/semantic_chunk.npy',
        '/home/lijiarui/Desktop/scene_factor/output/semantic_chunk.ply'
    )
    print("Done! Output: /home/lijiarui/Desktop/scene_factor/output/semantic_chunk.ply")
