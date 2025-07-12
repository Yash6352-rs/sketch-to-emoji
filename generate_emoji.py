from quickdraw import QuickDrawData
import numpy as np
import os
from PIL import Image

# Initialize QuickDraw access
qd = QuickDrawData()

# Define emoji classes with QuickDraw names
classes = {
    'smiley_face': 'smiley face',
    'moon': 'moon',
    'map': 'map',
    'cloud': 'cloud',
    'star': 'star',
    'sun': 'sun',
    'umbrella': 'umbrella',
    'flower': 'flower',
    't-shirt': 't-shirt',
    'square': 'square'
}

num_samples = 2000  # Only save 2000 per class
output_dir = 'data'
os.makedirs(output_dir, exist_ok=True)

def is_blank(image_array, threshold=50):
    """Filter out nearly blank images based on total pixel intensity."""
    return np.sum(image_array) < threshold

for label, qd_name in classes.items():
    filename = os.path.join(output_dir, f"{label}.npy")
    images = []

    print(f"Downloading samples for '{label}' ({qd_name})...")

    attempts = 0
    while len(images) < num_samples and attempts < num_samples * 3:
        try:
            drawing = qd.get_drawing(qd_name)
            img = drawing.image.resize((28, 28)).convert('L')
            img_arr = np.array(img)

            if not is_blank(img_arr):
                images.append(img_arr)
        except Exception as e:
            print(f"Skipped due to error: {e}")
        finally:
            attempts += 1

    images = np.array(images[:num_samples])  # Trim in case extras
    np.save(filename, images)
    print(f"Saved {len(images)} samples to '{filename}'\n")

# Optionally save the class-to-index mapping
label_map_file = os.path.join(output_dir, "label_map.txt")
with open(label_map_file, "w") as f:
    for idx, key in enumerate(classes.keys()):
        f.write(f"{idx},{key}\n")

print("All class data downloaded and saved.")
