import configparser
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import datetime
import threading
import logging

# 配置日志
logger = logging.getLogger('email_alerts')

# 初始化邮件线程锁
email_lock = threading.Lock()
last_email_sent_time = None  # 记录上一次邮件发送的时间

# 读取配置文件
config = configparser.ConfigParser()
config.read("./config.cfg", encoding="UTF-8")
config_email = config["Email_Setting"]

Email_sender = config_email["Email_sender"]
Email_password = config_email["Email_password"]  # 邮箱SMTP授权码
Email_receiver = config_email["Email_receiver"]  # 接收邮箱
SMTP_server = config_email["SMTP_server"]  # 邮箱SMTP服务器
SMTP_port = int(config_email["SMTP_port"])  # 邮箱服务器端口
WATER_LEVEL_THRESHOLD = float(config_email["WarningLevel"])  # 警戒水位

# 发送邮件警告
def send_email_alert(timestamp, water_level):
    global last_email_sent_time
    with email_lock:
        current_time = datetime.datetime.now()

        # 检查是否已经过了30分钟
        if last_email_sent_time and (current_time - last_email_sent_time).total_seconds() < 1800:
            logger.info("30分钟内已经发送过邮件，跳过此次发送。")
            return  # 跳过发送
        try:
            # 邮件标题
            subject = f"【紧急水位警报】水利工程水位过高"

            # 创建HTML邮件内容
            current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

            html_body = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 0; padding: 20px; color: #333; }}
                    .container {{ border: 1px solid #ddd; border-radius: 5px; padding: 20px; max-width: 600px; }}
                    .header {{ background-color: #d9534f; color: white; padding: 10px; text-align: center; font-size: 24px; border-radius: 4px 4px 0 0; }}
                    .content {{ padding: 20px; background-color: #f9f9f9; }}
                    .info-row {{ margin: 10px 0; }}
                    .label {{ font-weight: bold; display: inline-block; width: 160px; }}
                    .value {{ display: inline-block; }}
                    .warning {{ color: #d9534f; font-weight: bold; }}
                    .footer {{ margin-top: 20px; font-size: 12px; color: #777; text-align: center; border-top: 1px solid #eee; padding-top: 10px; }}
                    .action-needed {{ background-color: #fcf8e3; border: 1px solid #faebcc; padding: 15px; margin: 15px 0; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        水位过高警报
                    </div>
                    <div class="content">
                        <div class="info-row"><span class="label">监测时间：</span><span class="value">{timestamp}</span></div>
                        <div class="info-row"><span class="label">报警时间：</span><span class="value">{current_time_str}</span></div>
                        <div class="info-row"><span class="label">当前水位：</span><span class="value warning">{water_level} 米</span></div>
                        <div class="info-row"><span class="label">警戒水位：</span><span class="value">{WATER_LEVEL_THRESHOLD} 米</span></div>
                        
                        <div class="action-needed">
                            <h3>需要采取的措施：</h3>
                            <ol>
                                <li>立即核实水位监测数据的准确性</li>
                                <li>通知相关责任人员到岗到位</li>
                                <li>启动应急预案，准备必要的防汛物资</li>
                                <li>增加监测频率，密切关注水位变化趋势</li>
                                <li>如水位持续上涨，考虑采取泄洪等应对措施</li>
                            </ol>
                        </div>
                    </div>
                    <div class="footer">
                        此邮件由水利工程监控系统自动发送，请勿直接回复。如有疑问，请联系系统管理员。<br>
                        发送时间：{current_time_str}
                    </div>
                </div>
            </body>
            </html>
            """

            # 创建多部分邮件
            msg = MIMEMultipart('alternative')
            msg["From"] = Email_sender
            msg["To"] = Email_receiver
            msg["Subject"] = subject

            msg.attach(MIMEText(html_body, "html", "utf-8"))

            # 连接SMTP服务器并发送邮件
            server = smtplib.SMTP(SMTP_server, SMTP_port)
            server.starttls()  # 启用TLS加密
            server.login(Email_sender, Email_password)
            server.sendmail(Email_sender, Email_receiver, msg.as_string())
            server.quit()

            last_email_sent_time = current_time  # 更新邮件发送时间

            logger.info(f"水位警报邮件已发送至 {Email_receiver}，水位={water_level}米，超出警戒值{WATER_LEVEL_THRESHOLD:.2f}%")
        except Exception as e:
            logger.error(f"警告邮件发送失败：{e}")

# 检测弹窗是否关闭，超时发送邮件
def check_alert_window(alert_window, timestamp, water_level):
    import time
    time.sleep(30)  # 等待30秒
    if alert_window.winfo_exists():  # 弹窗30秒内未关闭
        logger.info(f"警告弹窗在30秒内未关闭，发送邮件通知。时间：{timestamp}，水位：{water_level}米")
        send_email_alert(timestamp, water_level)
        alert_window.destroy()  # 关闭弹窗