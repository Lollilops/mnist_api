import yaml
import logging
import requests
from flask import Flask, request, jsonify


# =========================
# Конфигурация
# =========================
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

HOST = config["service"]["host"]
PORT = config["service"]["port"]

COLLECTOR_URL = config["services"]["collector_url"]
ML_URL = config["services"]["ml_service_url"]
STORAGE_URL = config["services"]["storage_url"]


# =========================
# Логирование
# =========================
logging.basicConfig(
    filename=config["logging"]["log_file"],
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("WebMaster")


# =========================
# Flask
# =========================
app = Flask(__name__)


# =========================
# API
# =========================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# ---------- Training ----------

@app.route("/train", methods=["POST"])
def train_model():
    try:
        logger.info("Training requested")

        # 1. Запуск обучения в ML Service
        resp = requests.post(f"{ML_URL}/train", timeout=300)
        resp.raise_for_status()

        model_path = resp.json().get("model_path")

        # 2. Регистрация модели в Storage
        requests.post(
            f"{STORAGE_URL}/models",
            json={
                "name": model_path.split("/")[-1],
                "path": model_path
            },
            timeout=5
        )

        return jsonify({
            "status": "training_completed",
            "model_path": model_path
        }), 200

    except Exception as e:
        logger.error(str(e))
        return jsonify({"error": "Training failed"}), 500


# ---------- Prediction ----------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # 1. Отправляем изображение в ML Service
        ml_resp = requests.post(
            f"{ML_URL}/predict",
            json={"image": data["image"]},
            timeout=5
        )
        ml_resp.raise_for_status()

        prediction = ml_resp.json()

        # 2. Сохраняем изображение в Storage
        img_resp = requests.post(
            f"{STORAGE_URL}/images",
            json={
                "image": data["image"],
                "label": data.get("true_label")
            },
            timeout=5
        )
        img_resp.raise_for_status()
        image_id = img_resp.json()["image_id"]

        # 3. Сохраняем предсказание
        requests.post(
            f"{STORAGE_URL}/predictions",
            json={
                "image_id": image_id,
                "predicted_label": prediction["predicted_label"],
                "confidence": prediction["confidence"]
            },
            timeout=5
        )

        return jsonify(prediction), 200

    except Exception as e:
        logger.error(str(e))
        return jsonify({"error": "Prediction failed"}), 500


# ---------- History ----------

@app.route("/history", methods=["GET"])
def history():
    try:
        resp = requests.get(f"{STORAGE_URL}/predictions", timeout=5)
        resp.raise_for_status()
        return jsonify(resp.json()), 200

    except Exception as e:
        logger.error(str(e))
        return jsonify({"error": "Failed to fetch history"}), 500


# ---------- Models ----------

@app.route("/models", methods=["GET"])
def models():
    try:
        resp = requests.get(f"{STORAGE_URL}/models", timeout=5)
        resp.raise_for_status()
        return jsonify(resp.json()), 200

    except Exception as e:
        logger.error(str(e))
        return jsonify({"error": "Failed to fetch models"}), 500


# =========================
# Run
# =========================
if __name__ == "__main__":
    logger.info("Web Master started")
    app.run(host=HOST, port=PORT)