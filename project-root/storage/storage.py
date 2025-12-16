import os
import io
import yaml
import base64
import sqlite3
import logging
from datetime import datetime
from PIL import Image
from flask import Flask, request, jsonify


# =========================
# Конфигурация
# =========================
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

HOST = config["service"]["host"]
PORT = config["service"]["port"]

BASE_DIR = config["storage"]["base_dir"]
IMAGES_DIR = os.path.join(BASE_DIR, config["storage"]["images_dir"])
MODELS_DIR = os.path.join(BASE_DIR, config["storage"]["models_dir"])
DB_PATH = os.path.join(BASE_DIR, config["storage"]["database"])
LOG_FILE = config["logging"]["log_file"]

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# =========================
# Логирование
# =========================
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("Storage")


# =========================
# Flask
# =========================
app = Flask(__name__)


# =========================
# База данных
# =========================
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            label INTEGER,
            created_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER,
            predicted_label INTEGER,
            confidence REAL,
            created_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            path TEXT,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


init_db()


# =========================
# API
# =========================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# ---------- Images ----------

@app.route("/images", methods=["POST"])
def save_image():
    try:
        data = request.json
        image_base64 = data["image"]
        label = data.get("label")

        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("L")

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"img_{ts}.png"
        path = os.path.join(IMAGES_DIR, filename)
        image.save(path)

        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO images (filename, label, created_at) VALUES (?, ?, ?)",
            (filename, label, datetime.utcnow().isoformat())
        )
        conn.commit()
        image_id = cur.lastrowid
        conn.close()

        logger.info(f"Image saved: {filename}")

        return jsonify({"image_id": image_id}), 201

    except Exception as e:
        logger.error(str(e))
        return jsonify({"error": "Failed to save image"}), 500


@app.route("/images", methods=["GET"])
def list_images():
    conn = get_db()
    cur = conn.cursor()
    rows = cur.execute("SELECT * FROM images").fetchall()
    conn.close()

    return jsonify([dict(row) for row in rows]), 200


# ---------- Predictions ----------

@app.route("/predictions", methods=["POST"])
def save_prediction():
    try:
        data = request.json

        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO predictions (image_id, predicted_label, confidence, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                data["image_id"],
                data["predicted_label"],
                data["confidence"],
                datetime.utcnow().isoformat()
            )
        )
        conn.commit()
        conn.close()

        logger.info("Prediction saved")

        return jsonify({"status": "saved"}), 201

    except Exception as e:
        logger.error(str(e))
        return jsonify({"error": "Failed to save prediction"}), 500


@app.route("/predictions", methods=["GET"])
def list_predictions():
    conn = get_db()
    cur = conn.cursor()
    rows = cur.execute("""
        SELECT p.*, i.filename
        FROM predictions p
        JOIN images i ON p.image_id = i.id
        ORDER BY p.created_at DESC
    """).fetchall()
    conn.close()

    return jsonify([dict(row) for row in rows]), 200


# ---------- Models ----------

@app.route("/models", methods=["POST"])
def register_model():
    try:
        data = request.json

        name = data["name"]
        path = data["path"]

        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO models (name, path, created_at) VALUES (?, ?, ?)",
            (name, path, datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()

        logger.info(f"Model registered: {name}")

        return jsonify({"status": "registered"}), 201

    except Exception as e:
        logger.error(str(e))
        return jsonify({"error": "Failed to register model"}), 500


@app.route("/models", methods=["GET"])
def list_models():
    conn = get_db()
    cur = conn.cursor()
    rows = cur.execute("SELECT * FROM models ORDER BY created_at DESC").fetchall()
    conn.close()

    return jsonify([dict(row) for row in rows]), 200


# =========================
# Run
# =========================
if __name__ == "__main__":
    logger.info("Storage service started")
    app.run(host=HOST, port=PORT)