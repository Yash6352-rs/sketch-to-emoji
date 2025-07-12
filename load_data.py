import numpy as np
import os
from sklearn.model_selection import train_test_split

data_path = "data"
class_names = [
    'smiley_face',
    'moon',
    'map',
    'cloud',
    'star',
    'sun',
    'umbrella',
    'flower',
    't-shirt',
    'square'
]

label_map = {name: idx for idx, name in enumerate(class_names)}

X = []
y = []

limit = 2000  # Max samples per class

for label_name in class_names:
    file_path = os.path.join(data_path, f"{label_name}.npy")

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    images = np.load(file_path)

    if images.ndim == 2:
        images = images.reshape(-1, 28, 28)
    elif images.ndim == 4 and images.shape[-1] == 1:
        images = images.reshape(-1, 28, 28)
    elif images.ndim != 3:
        print(f"Unexpected shape for {label_name}: {images.shape}")
        continue

    images = images[:limit]
    print(f"{label_name}: loaded {images.shape[0]} samples")

    X.append(images)
    y += [label_map[label_name]] * len(images)

# Merge arrays
X = np.concatenate(X, axis=0)
y = np.array(y)

# Normalize and reshape
X = X.astype('float32') / 255.0
X = X.reshape(-1, 28, 28, 1)

# Split for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save processed arrays
np.save("data/X.npy", X)
np.save("data/y.npy", y)

print("\nData Ready!")
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print("Saved to data/X.npy and data/y.npy")
