import yaml
import requests
from flask import Flask, render_template, jsonify


# =========================
# Config
# =========================
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

HOST = config["service"]["host"]
PORT = config["service"]["port"]
WEB_MASTER_URL = config["web_master"]["url"]


# =========================
# Flask
# =========================
app = Flask(__name__)


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/history")
def history():
    resp = requests.get(f"{WEB_MASTER_URL}/history", timeout=5)
    data = resp.json()
    return render_template("history.html", history=data)


if __name__ == "__main__":
    app.run(host=HOST, port=PORT)