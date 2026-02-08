import time
from pathlib import Path

import numpy as np
import tensorflow as tf

NUM_IMAGES = 500
IMAGES_PATH = Path("t10k-images-idx3-ubyte")
LABELS_PATH = Path("t10k-labels-idx1-ubyte")
MODEL_PATH = Path("mnist_cnn.tflite")


def load_images(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) < 16:
        raise ValueError(f"{path} is too small for MNIST image header")
    data = np.frombuffer(raw, dtype=np.uint8, offset=16)
    if data.size % (28 * 28) != 0:
        raise ValueError("Image data size is not a multiple of 28x28")
    images = data.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0
    return images


def load_labels(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"{path} is too small for MNIST label header")
    labels = np.frombuffer(raw, dtype=np.uint8, offset=8)
    return labels


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    if not IMAGES_PATH.exists():
        raise FileNotFoundError(IMAGES_PATH)
    if not LABELS_PATH.exists():
        raise FileNotFoundError(LABELS_PATH)

    images = load_images(IMAGES_PATH)
    labels = load_labels(LABELS_PATH)

    if images.shape[0] < NUM_IMAGES or labels.shape[0] < NUM_IMAGES:
        raise ValueError("Not enough samples in MNIST test set")

    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if len(input_details) != 1 or len(output_details) != 1:
        raise RuntimeError("Expected single input and output tensor")

    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    predictions = np.empty(NUM_IMAGES, dtype=np.uint8)

    start = time.perf_counter()
    for i in range(NUM_IMAGES):
        img = images[i : i + 1]
        interpreter.set_tensor(input_index, img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)[0]
        predictions[i] = int(np.argmax(output))
    elapsed = time.perf_counter() - start

    correct = int(np.sum(predictions == labels[:NUM_IMAGES]))
    accuracy_pct = (correct / NUM_IMAGES) * 100.0

    per_image_us = (elapsed / NUM_IMAGES) * 1_000_000.0

    print("MNIST Inference (TFLite)")
    print("========================")
    print(f"Images:     {NUM_IMAGES}")
    print(f"Total time: {elapsed * 1000.0:.2f} ms")
    print(f"Per image:  {per_image_us:.2f} us")
    print(f"Accuracy:   {correct}/{NUM_IMAGES} ({accuracy_pct:.1f}%)")


if __name__ == "__main__":
    main()
