from flask import Flask, render_template, jsonify, request
import mysql.connector
import requests
import os
from dotenv import load_dotenv

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
env_path = os.path.join(base_dir, ".env")

# 加载 .env 文件
load_dotenv(env_path)

app = Flask(__name__)

def connect_db():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "sensor_user"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "reservoir_db")
    )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def get_data():
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT timestamp, water_level, temperature, humidity, windpower FROM sensor_data ORDER BY id DESC LIMIT 100")
    data = cursor.fetchall()
    conn.close()
    return jsonify(data)

@app.route('/deepseek', methods=['POST'])
def deepseek():
    question = request.json.get("question", "")
    if not question:
        return jsonify({"answer": "请输入您的问题！"})

    deepseek_url = os.getenv("DEEPSEEK_URL")
    response = requests.post(deepseek_url, json={"model": "deepseek-r1:7b", "prompt": question, "stream": False})
    
    return jsonify({"answer": response.json().get("response", "查询失败！")})

if __name__ == '__main__':
    app.run(debug=True)