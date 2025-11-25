import os
import glob
from typing import Tuple

import tensorflow as tf


# ==========================
# CONFIG – EDIT THESE
# ==========================
MODEL_PATH = "training_checkpoints/dcanet_best.h5"  # your trained model
INPUT_DIR  = "inference_input"                      # folder with noisy test images
OUTPUT_DIR = "inference_output"                     # folder to save denoised images

IMG_SIZE: Tuple[int, int] = (128, 128)   # MUST match training input shape
CHANNELS = 3                             # RGB


# ==========================
# GPU SETUP (optional)
# ==========================
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Using {len(gpus)} GPU(s).")
else:
    print("No GPU detected, running on CPU.")


# ==========================
# MODEL LOADING
# ==========================
def load_dcanet_model(model_path: str) -> tf.keras.Model:
    """
    Load DCANet from a saved .h5 file for inference.

    We use compile=False because we don't need optimizer/loss.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )
    return model


# ==========================
# IMAGE IO
# ==========================
def load_image(path: str) -> tf.Tensor:
    """
    Load an image and preprocess exactly like training:

    Training:
      - image_dataset_from_directory -> float32 in [0, 255]
      - then /255. -> [0, 1]

    Here:
      - decode image
      - convert_image_dtype -> float32 in [0, 1]
      - resize to (128, 128)
      - add batch dimension
    """
    image_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(image_bytes, channels=CHANNELS, expand_animations=False)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)  # [0, 1]

    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    img = tf.expand_dims(img, axis=0)  # (1, H, W, C)

    return img


def save_image(tensor: tf.Tensor, path: str):
    """
    Save a single image tensor to disk.

    - tensor expected in [0, 1]
    - we convert to uint8 0–255 before saving
    """
    tensor = tf.clip_by_value(tensor, 0.0, 1.0)
    tensor = tensor * 255.0
    tensor = tf.cast(tensor, tf.uint8)
    tf.keras.utils.save_img(path, tensor)


# ==========================
# INFERENCE LOOP
# ==========================
def run_inference_on_folder(model: tf.keras.Model,
                            input_dir: str,
                            output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(input_dir, e)))

    if not files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(files)} image(s) in {input_dir}")

    for idx, img_path in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] Processing: {img_path}")

        # 1. Load & preprocess noisy image (→ [0, 1])
        noisy = load_image(img_path)

        # 2. Forward pass
        #    Your DCANet outputs: [restored, conv_tail, noise_map]
        restored, conv_tail, noise_map = model(noisy, training=False)

        # 3. Take first element in batch
        restored_img = restored[0]  # (H, W, C), in [0, 1]

        # 4. Save
        basename = os.path.basename(img_path)
        name, ext = os.path.splitext(basename)
        out_path = os.path.join(output_dir, f"{name}_denoised.png")
        save_image(restored_img, out_path)

        print(f"Saved: {out_path}")


if __name__ == "__main__":
    model = load_dcanet_model(MODEL_PATH)
    run_inference_on_folder(model, INPUT_DIR, OUTPUT_DIR)
