# -*- coding: utf-8 -*-
# @Time    : 2025/3/12 14:03
# @Author  : Bruam1
# @Email   : grey040612@gmail.com
# @File    : main.py
# @Software: Vscode


import configparser
import pymysql
import tkinter as tk
from tkinter import ttk
import threading
import time
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import numpy as np
import mplcursors
import datetime
import pymysql.cursors
import requests
import json

# 设定水位警戒值
water_level_threshold = 85.45  # 水位高于85米时报警
alerted_timestamps = set()  # 存储报警时间戳，防止重复弹窗
alerted_window = None  # 存储当前弹窗对象
avg_data = 100  # 每100条数据更新一次图表
last_email_sent_time = None  # 记录上一次邮件发送的邮件
email_lock = threading.Lock()  # 线程锁，防止并发访问

# 可视化界面中文显示
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

config = configparser.ConfigParser()
config.read("./config.cfg", encoding = "UTF-8")  # 读取配置文件
conf_email = config["Email_Setting"]

Email_sender = conf_email['Email_sender']
Email_password = conf_email['Email_password']  # 邮箱SMTP授权码
Email_receiver = conf_email['Email_receiver']  # 接收邮箱
SMTP_server = conf_email['SMTP_server']  # 邮箱SMTP服务器
SMTP_port = int(conf_email['SMTP_port'])  # 邮箱服务器端口

# 加载环境变量
load_dotenv()

# 数据库连接函数
def connect_db():
    return pymysql.connect(
        host = os.getenv("DB_HOST", "localhost"),
        user = os.getenv("DB_USER", "sensor_user"),
        password = os.getenv("DB_PASSWORD", ""),
        database = os.getenv("DB_NAME", "reservoir_db")
    )

# 发送邮件警告
def send_email_alert(timestamp, water_level):
    global last_email_sent_time
    with email_lock:
        current_time = datetime.datetime.now()

        # 检查是否已经过了30分钟
        if last_email_sent_time and (current_time - last_email_sent_time).total_seconds() < 1800:
            print("30分钟内已经发送过邮件，跳过此次发送。")
            return  # 跳过发送
        try:
            subject = "【警告】水利工程系统水位过高"
            body = f"警告！\n时间：{timestamp}\n水位过高：{water_level}米\n请立即处理！"
            msg = MIMEMultipart()
            msg["From"] = Email_sender
            msg["To"] = Email_receiver
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain", "utf-8"))

            # 连接SMTP服务器并发送邮件
            server = smtplib.SMTP(SMTP_server, SMTP_port)
            server.starttls()  # 启用TLS加密
            server.login(Email_sender, Email_password)
            server.sendmail(Email_sender, Email_receiver, msg.as_string())
            server.quit()

            last_email_sent_time = current_time  # 更新邮件发送时间
            print("警告邮件已发送")
        except Exception as e:
            print(f"警告邮件发送失败：{e}")

# 检测弹窗是否关闭，超时发送邮件
def check_alert_window(alert_window, timestamp, water_level):
    time.sleep(30)  # 等待30秒
    if alert_window.winfo_exists():  # 弹窗30秒内未关闭
        print(f"警告弹窗在30秒内未关闭，将发送邮件通知（时间：{timestamp}）")
        send_email_alert(timestamp, water_level)
        alert_window.destroy()  # 关闭弹窗
        alert_window = None  # 清空弹窗对象

# 显示历史数据窗口
def show_history():
    history_window = tk.Toplevel(root)
    history_window.title("历史数据")
    history_window.geometry("800x400")  # 加宽窗口以适应新列

    # 创建Frame以容纳TreeView和滚动条
    frame = tk.Frame(history_window)
    frame.pack(expand = True, fill = "both")

    # 添加垂直滚动条
    y_scroll = tk.Scrollbar(frame, orient = "vertical")
    y_scroll.pack(side = "right", fill = "y")

    # 创建TreeView，并绑定滚动条
    tree = ttk.Treeview(
        frame, 
        columns = ("时间", "水位(m)", "温度(℃)", "湿度(%)", "风力(m/s)", "风向", "降雨量(mm)"), 
        show = "headings",
        yscrollcommand = y_scroll.set)
    tree.heading("时间", text = "时间")
    tree.heading("水位(m)", text = "水位 (m)")
    tree.heading("温度(℃)", text = "温度 (℃)")
    tree.heading("湿度(%)", text = "湿度 (%)")
    tree.heading("风力(m/s)", text = "风力 (m/s)")
    tree.heading("风向", text = "风向")
    tree.heading("降雨量(mm)", text = "降雨量 (mm)")

    tree.column("时间", width = 150, anchor = "center")
    tree.column("水位(m)", width = 80, anchor = "center")
    tree.column("温度(℃)", width = 80, anchor = "center")
    tree.column("湿度(%)", width = 80, anchor = "center")
    tree.column("风力(m/s)", width = 80, anchor = "center")
    tree.column("风向", width = 100, anchor = "center")
    tree.column("降雨量(mm)", width = 80, anchor = "center")

    tree.pack(expand = True, fill = "both")
    y_scroll.config(command = tree.yview)

    try:
        conn = connect_db()
        cursor = conn.cursor()
        # 修改查询字段
        cursor.execute("SELECT timestamp, water_level, temperature, humidity, windpower, winddirection, rains FROM sensor_data ORDER BY id DESC LIMIT 100")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        for row in rows:
            tree.insert("", "end", values=row)
    except pymysql.Error as err:
        print(f"MySQL 错误: {err}")

# 更新实时数据曲线
def update_plot():
    global alerted_timestamps, alerted_window, avg_data
    while True:
        try:
            conn = connect_db()
            cursor = conn.cursor()
            cursor.execute("""
                           SELECT timestamp, COALESCE(water_level, 0), COALESCE(temperature, 0), 
                           COALESCE(humidity, 0), COALESCE(windpower, 0)
                           FROM sensor_data 
                           WHERE water_level IS NOT NULL 
                           ORDER BY id ASC 
                           """)
            rows = cursor.fetchall()
            cursor.close()
            conn.close()

            if rows:
                timestamps, water_levels, temperatures, humidities, windpowers = zip(*rows)

                # 转换为DataFrame以便进行滚动均值计算
                df = pd.DataFrame({
                    "timestamp": timestamps,
                    "water_level": water_levels,
                    "temperature": temperatures,
                    "humidity": humidities,
                    "windpower": windpowers
                })

                # 获取年份
                year = df["timestamp"].iloc[0].year

                # 计算滚动均值
                df['water_level_smooth'] = df['water_level'].rolling(window=3, min_periods=1).mean()
                df['temperature_smooth'] = df['temperature'].rolling(window=3, min_periods=1).mean()
                df['humidity_smooth'] = df['humidity'].rolling(window=3, min_periods=1).mean()
                df['windpower_smooth'] = df['windpower'].rolling(window=3, min_periods=1).mean()

                for i in range(0, len(df)):
                    # 清除旧图表
                    ax1.clear()
                    ax2.clear()
                    ax3.clear()
                    ax4.clear()

                    subset = df.iloc[i:i + avg_data]  # 每次只取100条数据
                    # 显示年份
                    fig.suptitle(f"{year} 年数据分析", fontsize = 14, fontweight = "bold", x = 0.1, y = 0.99)

                    # 水位趋势图
                    ax1.plot(subset['timestamp'], subset['water_level'], label = "水位 (m)", color = "blue", alpha = 0.3)
                    ax1.plot(subset['timestamp'], subset['water_level_smooth'], label = "水位 (平滑)", color = "blue")
                    ax1.set_title("水位变化", fontsize = 15)
                    ax1.set_ylabel("水位 (m)", fontsize = 15)
                    ax1.legend()
                    ax1.grid()

                    # 温度趋势图
                    ax2.plot(subset['timestamp'], subset['temperature'], label = "温度 (℃)", color = "red", alpha = 0.3)
                    ax2.plot(subset['timestamp'], subset['temperature_smooth'], label = "温度 (平滑)", color = "red")
                    ax2.set_title("温度变化", fontsize = 15)
                    ax2.set_ylabel("温度 (℃)", fontsize = 15)
                    ax2.legend()
                    ax2.grid()

                    # 湿度趋势图
                    ax3.plot(subset['timestamp'], subset['humidity'], label="湿度 (%)", color = "green", alpha = 0.3)
                    ax3.plot(subset['timestamp'], subset['humidity_smooth'], label = "湿度 (平滑)", color = "green")
                    ax3.set_title("湿度变化", fontsize = 15)
                    ax3.set_ylabel("湿度 (%)", fontsize = 15)
                    ax3.legend()
                    ax3.grid()

                    # 风力趋势图
                    ax4.plot(subset['timestamp'], subset['windpower'], label="风力 (m/s)", color = "purple", alpha = 0.3)
                    ax4.plot(subset['timestamp'], subset['windpower_smooth'], label = "风力 (平滑)", color = "purple")
                    ax4.set_title("风力变化", fontsize = 15)
                    ax4.set_ylabel("风力 (m/s)", fontsize = 15)
                    ax4.legend()
                    ax4.grid()

                    canvas.draw()
                    time.sleep(0.1)  # 等待0.1秒再绘制下一张

                    # 监测水位是否低于警戒值
                    for i, water_level in enumerate(water_levels):
                        if water_level > water_level_threshold and timestamps[i] not in alerted_timestamps:
                            if alerted_window:
                                alerted_window.destroy()
                                alerted_window = None

                            warning_message = f"警告！\n时间：{timestamps[i]}\n水位过低：{water_level}米\n请检查系统！"
                            alerted_timestamps.add(timestamps[i])
                            alert_window = tk.Toplevel(root)
                            alert_window.title("水位警告")
                            alert_label = tk.Label(alert_window, text = warning_message, font = ("SimHei", 12))
                            alert_label.pack(padx = 50, pady = 50)

                            def close_alert():
                                alert_window.destroy()
                                global alerted_window
                                alerted_window = None

                            confirm_button = tk.Button(alert_window, text="确认", command = close_alert)
                            confirm_button.pack(pady = 10)

                            timer = threading.Timer(30, check_alert_window, args = (alert_window, timestamps[i], water_level))
                            timer.start()
                            # send_email_alert(timestamps[i], water_level)
                            # 启动30秒后检查弹窗状态的线程
                            threading.Thread(target = check_alert_window, args = (alert_window, timestamps[i], water_level), daemon = True).start()
                            alerted_window = alert_window
                            break

        except pymysql.Error as err:
            print(f"MySQL 错误: {err}")

# 获取历史数据进行分析
def get_histroy_data():
    try:
        conn = connect_db()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT * FROM sensor_data ORDER BY timestamp DESC LIMIT 100")  # 获取最新的100条数据
        data = cursor.fetchall()

        for row in data:
            if isinstance(row["timestamp"], datetime.datetime):
                row["timestamp"] = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")  # 格式化时间

        cursor.close()
        return data
    except pymysql.Error as e:
        return []

# 调用DeepSeek查询
def ask_deepseek():
    question = deepseep_input.get("1.0", "end-1c").strip()
    if not question:
        deepseek_output.config(state = tk.NORMAL)
        deepseek_output.delete(1.0, tk.END)
        deepseek_output.insert(tk.END, "请输入问题后再查询。")
        deepseek_output.config(state = tk.DISABLED)
        return
    
    # 获取最近的水文数据
    history_data = get_histroy_data()

    question = str(history_data) + question
    try:
        # print("正在发送请求到....")
        response = requests.post(
            url = deepseek_url,
            json = {
                "model": "deepseek-r1:7b",
                "prompt": question,
                "stream": False
                },
        )

        response.raise_for_status()
        response_data = response.json()
        response_text = response_data.get("response", "API 未返回数据。")
    except requests.exceptions.RequestException as e:
        response_text = f"请求失败：{e}"

    except json.JSONDecodeError:
        response_text = "API 返回了无法解析的数据格式。"

    deepseek_output.config(state = tk.NORMAL)
    deepseek_output.delete(1.0, tk.END)
    deepseek_output.insert(tk.END, response_text)
    deepseek_output.config(state = tk.DISABLED)



# 创建主窗口
root = tk.Tk()
root.title("水利工程数据可视化与监控系统")
root.geometry("2400x960")

# 创建主框架
main_frame = tk.Frame(root)
main_frame.pack(fill = tk.BOTH, expand = True)

# 创建图表区域
chart_frame = tk.Frame(main_frame)
chart_frame.grid(row = 0, column = 0, sticky = "nsew")

# Matplotlib 图表
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (12, 8), dpi = 100)
fig.tight_layout(pad = 3.0)
canvas = FigureCanvasTkAgg(fig, master = chart_frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(fill = tk.BOTH, expand = True)

# 启用交互式图表
mplcursors.cursor()

# DeepSeek模块
deepseek_url = os.getenv("DEEPSEEK_URL")
deepseek_frame = tk.Frame(main_frame, width = 500, padx = 10, pady = 10)
deepseek_frame.grid(row = 0, column = 1, sticky = "nsew")

# 提问框
deepseek_label = tk.Label(deepseek_frame, text = "请输入您的问题：", font = ("SimHei", 14))
deepseek_label.pack(anchor = "w")

deepseep_input = tk.Text(deepseek_frame, height = 5, width = 60, font = ("SimHei", 14))
deepseep_input.pack(pady = 5)

# 提交按钮
deepseek_button = tk.Button(deepseek_frame, text = "查询", font = ("SimHei", 14), command = ask_deepseek)
deepseek_button.pack(pady = 5)

# 结果显示框
deepseek_output = tk.Text(deepseek_frame, height = 40, width = 60, font = ("SimHei", 14), state = tk.DISABLED)
deepseek_output.pack(pady = 5)

# 设置布局权重
main_frame.columnconfigure(0, weight = 4)
main_frame.columnconfigure(1, weight = 2)
main_frame.rowconfigure(0, weight = 1)

# 按钮：查看历史数据
history_button = tk.Button(chart_frame, text = "查看历史数据", command = show_history, font = ("SimHei", 12))
history_button.pack(pady = 10)

# 启动数据更新线程
threading.Thread(target = update_plot, daemon = True).start()

# 运行 GUI
root.mainloop()