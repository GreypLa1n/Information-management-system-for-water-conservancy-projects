import configparser
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import datetime
import threading

# 初始化邮件线程锁
email_lock = threading.Lock()
last_email_sent_time = None  # 记录上一次邮件发送的时间

# 读取配置文件
config = configparser.ConfigParser()
config.read("./config.cfg", encoding="UTF-8")
conf_email = config["Email_Setting"]

Email_sender = conf_email['Email_sender']
Email_password = conf_email['Email_password']  # 邮箱SMTP授权码
Email_receiver = conf_email['Email_receiver']  # 接收邮箱
SMTP_server = conf_email['SMTP_server']  # 邮箱SMTP服务器
SMTP_port = int(conf_email['SMTP_port'])  # 邮箱服务器端口

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
    import time
    time.sleep(30)  # 等待30秒
    if alert_window.winfo_exists():  # 弹窗30秒内未关闭
        print(f"警告弹窗在30秒内未关闭，将发送邮件通知（时间：{timestamp}）")
        send_email_alert(timestamp, water_level)
        alert_window.destroy()  # 关闭弹窗