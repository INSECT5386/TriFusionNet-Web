import os
import gdown

# 모델이 없으면 다운로드
if not os.path.exists("model.h5"):
    print("🔽 모델 다운로드 중...")
    gdown.download("https://drive.google.com/uc?id=★파일ID★", "model.h5", quiet=False)

from flask import Flask, render_template, request, jsonify
from model import respond  # respond 함수 불러오기

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.json["message"]
    bot_reply = respond(user_input)
    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)
