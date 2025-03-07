import mysql.connector
import random
import time
import os
import logging
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置日志
logging.basicConfig(filename = "data_log.log", level = logging.INFO, 
                    format = "%(asctime)s - %(levelname)s - %(message)s")

# 数据库连接函数
def connect_db():
    try:
        return mysql.connector.connect(
            host = os.getenv("DB_HOST", "localhost"),
            user = os.getenv("DB_USER", "sensor_user"),
            password = os.getenv("DB_PASSWORD", ""),
            database = os.getenv("DB_NAME", "reservoir_db")
        )
    except mysql.connector.Error as err:
        logging.error(f"数据库连接失败: {err}")
        return None

# 生成并存储数据
def generate_data(sample_interval = 1, max_entries = None):
    entry_count = 0

    while max_entries is None or entry_count < max_entries:
        new_entry = {
            "水位(m)": round(random.uniform(1.0, 10.0), 2),
            "压力(MPa)": round(random.uniform(0.5, 2.0), 2),
            "流量(m³/s)": round(random.uniform(5.0, 50.0), 2),
        }

        conn = connect_db()
        if conn is None:
            logging.error("无法连接到数据库，跳过当前数据存储")
            time.sleep(sample_interval)
            continue

        try:
            cursor = conn.cursor()
            sql = "INSERT INTO sensor_data (water_level, pressure, flow_rate) VALUES (%s, %s, %s)"
            cursor.execute(sql, (new_entry["水位(m)"], new_entry["压力(MPa)"], new_entry["流量(m³/s)"]))
            conn.commit()
            logging.info(f"插入数据: {new_entry}")
            entry_count += 1
            print(f"插入数据：{new_entry}")
        except mysql.connector.Error as err:
            logging.error(f"MySQL 错误: {err}")
        finally:
            cursor.close()
            conn.close()

        time.sleep(sample_interval)

if __name__ == "__main__":
    generate_data(sample_interval=1, max_entries=100)  # 运行100次，每秒生成一次数据