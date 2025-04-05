import requests
import json
import os
import datetime
import pymysql
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 获取DeepSeek API URL
deepseek_url = os.getenv("DEEPSEEK_URL")

# 数据库连接函数
def connect_db():
    return pymysql.connect(
        host = os.getenv("DB_HOST", "localhost"),
        user = os.getenv("DB_USER", "sensor_user"),
        password = os.getenv("DB_PASSWORD", ""),
        database = os.getenv("DB_NAME", "reservoir_db")
    )

# 获取历史数据进行分析
def get_history_data():
    try:
        conn = connect_db()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT * FROM sensor_data ORDER BY timestamp DESC LIMIT 100")  # 获取最新的100条数据
        data = cursor.fetchall()

        for row in data:
            if isinstance(row["timestamp"], datetime.datetime):
                row["timestamp"] = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")  # 格式化时间

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
                "stream": False
            },
        )

        response.raise_for_status()
        response_data = response.json()
        response_text = response_data.get("response", "API 未返回数据。")
        return response_text
    
    except requests.exceptions.RequestException as e:
        return f"请求失败：{e}"
    
    except json.JSONDecodeError:
        return "API 返回了无法解析的数据格式。"