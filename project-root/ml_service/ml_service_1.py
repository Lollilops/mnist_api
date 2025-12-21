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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MNISTCNN


# =========================
# Конфигурация
# =========================
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

HOST = config["service"]["host"]
PORT = config["service"]["port"]

MODELS_DIR = config["model"]["models_dir"]
BATCH_SIZE = config["training"]["batch_size"]
EPOCHS = config["training"]["epochs"]
LR = config["training"]["learning_rate"]

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs('./data', exist_ok=True)

# =========================
# Логирование
# =========================
logging.basicConfig(
    filename=config["logging"]["log_file"],
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("MLService")


# =========================
# Flask
# =========================
app = Flask(__name__)


# =========================
# PyTorch
# =========================
device = torch.device("cpu")
model = MNISTCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()


# =========================
# Utils
# =========================
def save_model():
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(MODELS_DIR, f"model_{ts}.pth")
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved: {path}")
    return path


def load_latest_model(model):
    files = sorted(os.listdir(MODELS_DIR))
    if not files:
        return False
    latest = files[-1]
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, latest)))
    logger.info(f"Loaded model: {latest}")
    return model


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    # print(image[0:4, 0:4])
    tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    # print(tensor.shape)
    return tensor


# =========================
# API
# =========================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/models", methods=["GET"])
def list_models():
    models = sorted(os.listdir(MODELS_DIR))
    return jsonify(models), 200


@app.route("/train", methods=["POST"])
def train():
    try:
        logger.info("Training started")

        transform = transforms.ToTensor()
        dataset = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        logger.info("Dataset loaded")
        model = model.to(device)

        model.train()
        for epoch in range(EPOCHS):
            mean_loss = []
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                mean_loss.append(loss)

                loss.backward()
                optimizer.step()
            logger.info(f"{epoch}/{EPOCHS}, loss: {np.mean(mean_loss)}")

        path = save_model()

        logger.info("Training finished, model saved successfully")

        return jsonify({
            "status": "trained",
            "model_path": path
        }), 200

    except Exception as e:
        logger.error(str(e))
        return jsonify({"error": "Training failed"}), 500



@app.route("/predict", methods=["POST"])
def predict():
    logger.info("Prediction started")
    print("Prediction started")
    try:
        data = request.json
        image_base64 = data["image"]

        image_bytes = base64.b64decode(image_base64)
        tensor = preprocess_image(image_bytes).to(device)
        
        logger.info("Image preprocessed")
        print("Image preprocessed")

        model = MNISTCNN()
        model = load_latest_model(model)
        # print(model.parameters)
        model = model.to(device)
        model.eval()
        print("model.eval()")
        with torch.no_grad():
            print("with torch.no_grad():")
            output = model(tensor)
            print("output")
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, dim=1)
        
        logger.info("Prediction successful")
        print("Prediction successful", pred, conf)

        return jsonify({
            "predicted_label": int(pred.item()),
            "confidence": float(conf.item())
        }), 200

    except Exception as e:
        logger.error(str(e))
        print("[ERROR] " + str(e))
        return jsonify({"error": "Prediction failed"}), 500


# =========================
# Init
# =========================
load_latest_model(model)

if __name__ == "__main__":
    logger.info("ML Service started")
    app.run(host=HOST, port=PORT)
    # train()