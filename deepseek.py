import requests
import json
import os
import datetime
import pymysql
from dotenv import load_dotenv
import re

# 加载环境变量
load_dotenv()

# 获取DeepSeek API URL
deepseek_url = os.getenv("DEEPSEEK_URL")

# 数据库连接函数
def connect_db():
    return pymysql.connect(
        host = os.getenv("DB_HOST", "localhost"),
        user = os.getenv("DB_USER", "sensor_data"),
        password = os.getenv("DB_PASSWORD", ""),
        database = os.getenv("DB_NAME", "reservoir_db")
    )

# 获取历史数据进行分析
def get_history_data():
    try:
        conn = connect_db()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        # 返回需要的时间 温度 湿度 风力 降雨量和水位
        cursor.execute("SELECT * FROM sensor_data ORDER BY timestamp DESC LIMIT 100")
        data = cursor.fetchall()

        for row in data:
            if isinstance(row["timestamp"], datetime.datetime):
                row["timestamp"] = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")

        cursor.close()
        conn.close()
        return data
    except pymysql.Error as e:
        print(f"数据库错误: {e}")
        return []

# 调用DeepSeek查询API
def ask_deepseek(question):
    if not question:
        return "请输入问题后再查询。"
    
    # 获取最近的水文数据
    history_data = get_history_data()

    # 将历史数据添加到问题中作为上下文
    question_with_context = str(history_data) + question
    
    try:
        response = requests.post(
            url = deepseek_url,
            json = {
                "model": "deepseek-r1:7b",
                "prompt": question_with_context,
                "stream": False  # False 等待大模型生成结束后一起输出，适合只提问一次的提问场景 True 会多次返回文本片段，适合多轮次的对话场景
            },
        )

        response.raise_for_status()
        response_data = response.json()
        response_text = response_data.get("response", "API 未返回数据。").strip()

        reasoning = re.search(r"<think(.*?)</think>>", response_text, re.DOTALL)  # 使用正则表达式分离思考和结果
        reasoning = reasoning.group(1).strip() if reasoning else ""

        answer = response_text.split("</think>")[-1].strip()

        return answer
    
    except requests.exceptions.RequestException as e:
        return f"请求失败：{e}"
    
    except json.JSONDecodeError:
        return "API 返回了无法解析的数据格式。"