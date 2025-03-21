# -*- coding: utf-8 -*-
# @Time    : 2025/03/21 8:19
# @Author  : Bruam1
# @Email   : grey040612@gmail.com
# @File    : app.py
# @Software: Vscode


from flask import Flask, jsonify, request, send_from_directory
import os
import pymysql
from datetime import datetime, timedelta
import configparser
from dotenv import load_dotenv
import requests

# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# 加载环境变量
load_dotenv()

# 数据库连接函数
def connect_db():
    try:
        return pymysql.connect(
            host=os.getenv("DB_HOST", "localhost"),
            user=os.getenv("DB_USER", "sensor_user"),
            password=os.getenv("DB_PASSWORD", ""),
            database=os.getenv("DB_NAME", "reservoir_db")
        )
    except Exception as e:
        print(f"数据库连接失败: {e}")
        return None

# 静态文件路由
@app.route('/')
def index():
    return send_from_directory(current_dir, 'index.html')

@app.route('/static/css/<path:path>')
def send_css(path):
    return send_from_directory(os.path.join(current_dir, 'static', 'css'), path)

@app.route('/static/js/<path:path>')
def send_js(path):
    return send_from_directory(os.path.join(current_dir, 'static', 'js'), path)

# API路由
@app.route('/api/realtime-data')
def get_realtime_data():
    try:
        # 获取偏移量和限制数量参数
        offset = request.args.get('offset', default=0, type=int)
        limit = request.args.get('limit', default=100, type=int)
        
        conn = connect_db()
        if conn is None:
            return jsonify({'error': '数据库连接失败'}), 500
            
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        
        # 从指定偏移量开始获取数据
        cursor.execute("""
            SELECT timestamp, water_level, temperature, humidity, windpower
            FROM sensor_data
            ORDER BY timestamp ASC
            LIMIT %s OFFSET %s
        """, (limit, offset))
        
        data = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # 确保时间戳是字符串格式
        for row in data:
            row['timestamp'] = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify(data)
        
    except Exception as e:
        print('Error:', str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/api/history-data')
def get_history_data():
    start_date = request.args.get('start')
    end_date = request.args.get('end')

    if not start_date or not end_date:
        return jsonify({'error': '缺少开始或结束日期'}), 400

    try:
        conn = connect_db()
        if conn is None:
            return jsonify({'error': '数据库连接失败'}), 500

        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute("""
            SELECT timestamp, water_level, temperature, humidity, 
                   windpower, winddirection, rains 
            FROM sensor_data 
            WHERE timestamp BETWEEN %s AND %s
            ORDER BY timestamp DESC
        """, (start_date, end_date))
        data = cursor.fetchall()
        cursor.close()
        conn.close()

        # 转换时间戳格式
        for row in data:
            row['timestamp'] = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')

        return jsonify(data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/deepseek', methods=['POST'])
def deepseek_query():
    try:
        data = request.get_json()
        question = data.get('question')
        print("收到问题:", question, flush=True)  # 添加问题输出
        
        if not question:
            return jsonify({'error': '请提供问题'}), 400

        # 获取最近的数据作为上下文
        conn = connect_db()
        if not conn:
            print("数据库连接失败", flush=True)  # 添加数据库连接状态输出
            return jsonify({'error': '数据库连接失败'}), 500

        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute("""
                SELECT timestamp, water_level, temperature, humidity, windpower
                FROM sensor_data
                ORDER BY timestamp DESC
                LIMIT 100
            """)
            history_data = cursor.fetchall()
            print(f"获取到 {len(history_data)} 条历史数据:{history_data}", flush=True)  # 添加数据获取状态输出
            
        conn.close()

        # 构建发送到 DeepSeek API 的请求
        deepseek_url = os.getenv('DEEPSEEK_URL')
        if not deepseek_url:
            print("DeepSeek URL 未配置", flush=True)  # 添加配置检查输出
            return jsonify({'error': 'DeepSeek API 配置缺失'}), 500

        prompt = str(history_data) + question

        # print("发送到 DeepSeek 的提示信息:", prompt, flush=True)  # 添加提示信息输出
        print(f"正在发送请求到 DeepSeek API: {deepseek_url}", flush=True)
        
        response = requests.post(
            url=deepseek_url,
            json={
                'model': 'deepseek-r1:7b',
                'prompt': prompt,
                "stream": True
            },
        )
        
        print(f"DeepSeek API 响应状态码: {response.status_code}", flush=True)
        if response.status_code != 200:
            print(f"DeepSeek API 错误响应: {response.text}", flush=True)
            return jsonify({'error': 'DeepSeek API 请求失败'}), 500

        result = response.json()
        print("DeepSeek 返回结果:", result, flush=True)  # 添加结果输出
        return jsonify({
            'response': result.get('response', '未能获取有效回答')
        })

    except requests.Timeout:
        print("DeepSeek API 请求超时", flush=True)
        return jsonify({'error': 'DeepSeek API 请求超时'}), 504
    except requests.RequestException as e:
        print(f"请求错误: {str(e)}", flush=True)
        return jsonify({'error': f'请求错误: {str(e)}'}), 500
    except Exception as e:
        print(f"意外错误: {str(e)}", flush=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(f"应用已启动，请访问: http://localhost:5000")
    app.run(debug=True, port=5000) 