import mysql.connector
import hydrofunctions as hf
import time
import os
import logging
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置日志
logging.basicConfig(filename="data_log.log", level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

# 数据库连接
def connect_db():
    try:
        return mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            user=os.getenv("DB_USER", "sensor_user"),
            password=os.getenv("DB_PASSWORD", ""),
            database=os.getenv("DB_NAME", "reservoir_db")
        )
    except mysql.connector.Error as err:
        logging.error(f"数据库连接失败: {err}")
        return None

# 获取 USGS 数据
def fetch_usgs_data(site_id="01646500"):
    try:
        site = hf.NWIS(site_id, 'iv', period='P1H')  # 获取最近 1 小时数据
        df = site.df()
        if df.empty:
            print("未获取到 USGS 数据")
            return None

        latest = df.iloc[-1]  # 获取最新一条数据
        water_level = latest.get("00065:00000", None)  # 水位（米）
        flow_rate = latest.get("00060:00000", None)  # 流量（m³/s）
        flow_velocity = latest.get("72255:00000", None)  # 流速（m/s），部分站点提供

        if water_level is None or flow_rate is None:
            print("数据不完整，跳过存储")
            return None
        
        return {
            "水位(m)": round(water_level, 2),
            "流量(m³/s)": round(flow_rate, 2),
            "流速(m/s)": round(flow_velocity, 2) if flow_velocity else None,
        }
    except Exception as e:
        print(f"获取 USGS 数据失败: {e}")
        return None

# 存储数据
def store_data(data):
    if data is None:
        return
    try:
        conn = connect_db()
        cursor = conn.cursor()
        sql = "INSERT INTO sensor_data (water_level, flow_rate, flow_velocity) VALUES (%s, %s, %s)"
        cursor.execute(sql, (data["水位(m)"], data["流量(m³/s)"], data["流速(m/s)"]))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"插入数据: {data}")
    except mysql.connector.Error as err:
        logging.error(f"MySQL 错误: {err}")

# 运行数据获取
def main():
    while True:
        data = fetch_usgs_data()
        store_data(data)
        time.sleep(3600)  # 每小时更新一次数据

if __name__ == "__main__":
    main()