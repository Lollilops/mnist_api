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
import matplotlib.pyplot as plt


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

logging.basicConfig(
    filename=config["logging"]["log_file"],
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("MLService")

app = Flask(__name__)

device = torch.device("cpu")
model = MNISTCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

def save_model():
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(MODELS_DIR, f"model_{ts}.pth")
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved: {path}")
    return path


def load_latest_model():
    files = sorted(os.listdir(MODELS_DIR))
    if not files:
        return False
    latest = files[-1]
    logger.info(f"Loading model: {latest}")
    print(f"Loading model: {latest}")
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, latest)))
    return True


def preprocess_image(image_bytes, need_to_validate=False):
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    if need_to_validate:
        save_dir = "./images_plt"
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(4, 4))
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"image_orgin_{ts}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    image = image.resize((28, 28))
    if need_to_validate:
        plt.figure(figsize=(4, 4))
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.show()
        save_path = os.path.join(save_dir, f"image_resize_{ts}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    image = np.array(image) / 255.0
    tensor = 1 - torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor

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
        print("Training started")

        transform = transforms.ToTensor()
        dataset = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        model.train()
        for epoch in range(EPOCHS):
            mean_loss = []
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                mean_loss.append(loss.detach())

                loss.backward()
                optimizer.step()
            logger.info(f"{epoch}/{EPOCHS}, loss: {np.mean(mean_loss)}")
            print(f"{epoch}/{EPOCHS}, loss: {np.mean(mean_loss)}")

        path = save_model()

        logger.info("Training finished, model saved successfully")
        print("Training finished, model saved successfully")

        return jsonify({
            "status": "trained",
            "model_path": path
        }), 200

    except Exception as e:
        logger.error(str(e))
        print("[ERROR] " + str(e))
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

        model.eval()
        with torch.no_grad():
            output = model(tensor)
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

load_latest_model()

if __name__ == "__main__":
    logger.info("ML Service started")
    app.run(host=HOST, port=PORT)