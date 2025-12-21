import os
import io
import yaml
import base64
import logging
from datetime import datetime

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

import torch
from torchvision import datasets, transforms


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

HOST = config["service"]["host"]
PORT = config["service"]["port"]

BATCH_SIZE = config["dataset"]["batch_size"]
DATA_DIR = config["dataset"]["data_dir"]
USER_DATA_DIR = config["storage"]["user_data_dir"]
LOG_FILE = config["logging"]["log_file"]

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(USER_DATA_DIR, exist_ok=True)


logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger("Collector")


app = Flask(__name__)

transform = transforms.Compose([
    transforms.ToTensor()
])

mnist_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform
)

data_loader = torch.utils.data.DataLoader(
    mnist_dataset,
    batch_size=BATCH_SIZE,
    shuffle=config["dataset"]["shuffle"]
)

data_iter = iter(data_loader)

def get_next_batch():
    global data_iter
    try:
        images, labels = next(data_iter)
    except StopIteration:
        data_iter = iter(data_loader)
        images, labels = next(data_iter)

    return images, labels


def tensor_to_base64(tensor):
    img = tensor.squeeze(0).numpy() * 255
    img = Image.fromarray(img.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/batch", methods=["GET"])
def get_batch():
    try:
        images, labels = get_next_batch()

        batch = []
        for img, label in zip(images, labels):
            batch.append({
                "image": tensor_to_base64(img),
                "label": int(label)
            })

        logger.info(f"Batch of size {len(batch)} served")

        return jsonify({
            "batch_size": len(batch),
            "data": batch
        }), 200

    except Exception as e:
        logger.error(f"Error while getting batch: {str(e)}")
        return jsonify({"error": "Failed to get batch"}), 500


@app.route("/user-data", methods=["POST"])
def upload_user_data():
    try:
        data = request.json

        image_base64 = data.get("image")
        label = data.get("label")  # может быть None

        if image_base64 is None:
            return jsonify({"error": "Image is required"}), 400

        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        image = image.resize((28, 28))

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"user_{timestamp}.png"
        image_path = os.path.join(USER_DATA_DIR, filename)
        image.save(image_path)

        meta_path = image_path.replace(".png", ".txt")
        with open(meta_path, "w") as f:
            f.write(str(label))

        logger.info(f"User data saved: {filename}, label={label}")

        return jsonify({
            "message": "User data saved",
            "filename": filename
        }), 201

    except Exception as e:
        logger.error(f"Error while saving user data: {str(e)}")
        return jsonify({"error": "Failed to save user data"}), 500


if __name__ == "__main__":
    logger.info("Collector service started")
    app.run(host=HOST, port=PORT)