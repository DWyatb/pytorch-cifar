import numpy as np
import matplotlib.pyplot as plt

# === Load file ===
file_path = "../cifar10_fin_new.npz"
data = np.load(file_path)

print(f"File path: {file_path}")
print(f"Total {len(data.files)} items:\n")
for key in data.files:
    arr = data[key]
    print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")

# === Keys to show ===
keys_to_show = [
    "x_test",
    "x_test_key12",
    "x_test_key123",
    "x_test_key1234",
    "x_test_key12345"
]

# === Display function ===
def show_img(ax, img_arr, title):
    if img_arr.size == 3072:
        img = img_arr.reshape(32, 32, 3)
        ax.imshow(img.astype(np.uint8))
    else:
        raise ValueError(f"Unexpected image size: {img_arr.size}")
    ax.set_title(title, fontsize=10)
    ax.axis("off")

# === Create figure (3 rows × 5 columns) ===
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
fig.suptitle("CIFAR-10 Key Comparison", fontsize=16)

# === Fill images ===
for col, key in enumerate(keys_to_show):
    if key not in data:
        print(f"Warning: {key} not found in file.")
        continue
    imgs = data[key]
    for row in range(3):  # first 3 images
        show_img(axes[row, col], imgs[row], f"{key}\nimg#{row}")

# === Adjust layout ===
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
