# -*- coding: utf-8 -*-
# @Time    : 2025/3/2 18:12
# @Author  : Bruam1
# @Email   : grey040612@gmail.com
# @File    : main.py
# @Software: Vscode
import configparser
import mysql.connector
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

# 设定压力警戒值
pressure_threshold = 1.8
alerted_timestamps = set()  # 存储报警时间戳， 防止重复弹窗
alerted_windows = {}  # 存储弹窗对象

#  可视化界面中文显示
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
    return mysql.connector.connect(
        host = os.getenv("DB_HOST", "localhost"),
        user = os.getenv("DB_USER", "sensor_user"),
        password = os.getenv("DB_PASSWORD", ""),
        database = os.getenv("DB_NAME", "reservoir_db")
    )

# 发送邮件警告
def send_email_alert(timestamp, pressure):
    try:
        subject = "【警告】水利工程系统压力过高"
        body = f"警告！\n时间：{timestamp}\n压力过大：{pressure}Mpa\n请立即处理！"
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

        print("警告邮件已发送")
    except Exception as e:
        print(f"警告邮件发送失败：{e}")

# 检测弹窗是否关闭，超时发送邮件
def check_alert_window(timestamp, pressure):
    if timestamp in alerted_windows:  # 弹窗长时间未关闭
        print(f"警告弹窗未关闭，将发送邮件通知（时间：{timestamp}）")
        send_email_alert(timestamp, pressure)
        del alerted_windows[timestamp]  # 清理记录

# 显示历史数据窗口
def show_history():
    history_window = tk.Toplevel(root)
    history_window.title("历史数据")
    history_window.geometry("600x400")

    tree = ttk.Treeview(history_window, columns=("时间", "水位(m)", "流量(m³/s)", "流速(m/s)"), show="headings")
    tree.heading("时间", text="时间")
    tree.heading("水位(m)", text="水位 (m)")
    tree.heading("流量(m³/s)", text="流量 (m³/s)")
    tree.heading("流速(m/s)", text="流速 (m/s)")

    tree.column("时间", width=150, anchor="center")
    tree.column("水位(m)", width=100, anchor="center")
    tree.column("流量(m³/s)", width=100, anchor="center")
    tree.column("流速(m/s)", width=100, anchor="center")

    tree.pack(expand=True, fill="both")

    try:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, water_level, flow_rate, flow_velocity FROM sensor_data ORDER BY id DESC LIMIT 100")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        for row in rows:
            tree.insert("", "end", values=row)
    except mysql.connector.Error as err:
        print(f"MySQL 错误: {err}")

# 更新实时数据曲线
def update_plot():
    while True:
        try:
            conn = connect_db()
            cursor = conn.cursor()
            cursor.execute("SELECT timestamp, water_level, flow_rate, flow_velocity FROM sensor_data ORDER BY id DESC LIMIT 20")
            rows = cursor.fetchall()
            cursor.close()
            conn.close()

            if rows:
                timestamps, water_levels, flow_rates, flow_velocities = zip(*rows)

                ax1.clear()
                ax2.clear()
                ax3.clear()

                ax1.plot(timestamps, water_levels, label="水位 (m)", color="blue")
                ax1.set_title("水位变化")
                ax1.set_ylabel("水位 (m)")
                ax1.legend()
                ax1.grid()

                ax2.plot(timestamps, flow_rates, label="流量 (m³/s)", color="green")
                ax2.set_title("流量变化")
                ax2.set_ylabel("流量 (m³/s)")
                ax2.legend()
                ax2.grid()

                ax3.plot(timestamps, flow_velocities, label="流速 (m/s)", color="purple")
                ax3.set_title("流速变化")
                ax3.set_ylabel("流速 (m/s)")
                ax3.legend()
                ax3.grid()

                canvas.draw()
        except mysql.connector.Error as err:
            print(f"MySQL 错误: {err}")

        time.sleep(2)  # 每 2 秒更新一次图表

# 测试邮件
# send_email_alert("2025-03-05", 2.1)

# 创建主窗口
root = tk.Tk()
root.title("水利工程数据可视化与监控系统")
root.geometry("1280x960")

# Matplotlib 图表
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (6, 8), dpi = 100)
fig.tight_layout(pad = 3.0)
canvas = FigureCanvasTkAgg(fig, master = root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(fill = tk.BOTH, expand = True)

# 按钮：查看历史数据
history_button = tk.Button(root, text = "查看历史数据", command = show_history, font = ("SimHei", 12))
history_button.pack(side = tk.BOTTOM, pady = 10)

# 启动数据更新线程
threading.Thread(target = update_plot, daemon = True).start()

# 运行 GUI
root.mainloop()