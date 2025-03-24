# -*- coding: utf-8 -*-
# @Time    : 2025/3/12 11:31
# @Author  : Bruam1
# @Email   : grey040612@gmail.com
# @File    : db.py
# @Software: Vscode


import pymysql
import pandas as pd
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
        return pymysql.connect(
            host = os.getenv("DB_HOST", "localhost"),
            user = os.getenv("DB_USER", "sensor_user"),
            password = os.getenv("DB_PASSWORD", ""),
            database = os.getenv("DB_NAME", "reservoir_db")
        )
    except pymysql.Error as err:
        logging.error(f"数据库连接失败: {err}")
        return None

# 读取 CSV 并存入数据库
def insert_csv_data(csv_path):
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except Exception as e:
        logging.error(f"读取 CSV 失败: {e}")
        return

    # 处理数据，确保列名匹配数据库
    df.rename(columns = {
        "times": "timestamp",
        "waterlevels": "water_level"
    }, inplace = True)

    # **缺失值处理**
    df = df.where(pd.notnull(df), None)  # NaN 替换为 None 适用于数据库
    missing_values = df.isnull().sum()
    logging.info(f"数据缺失情况: {missing_values}")

    # **数据类型转换**
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors = "coerce")  # 解析时间格式
        df["temperature"] = pd.to_numeric(df["temperature"], errors = "coerce")
        df["humidity"] = pd.to_numeric(df["humidity"], errors = "coerce")
        df["windpower"] = pd.to_numeric(df["windpower"], errors = "coerce")
        df["rains"] = pd.to_numeric(df["rains"], errors = "coerce")
        df["water_level"] = pd.to_numeric(df["water_level"], errors = "coerce")
    except Exception as e:
        logging.error(f"数据类型转换失败: {e}")
        return

    # **异常值检测**
    df = df.dropna(subset = ["timestamp", "water_level"])  # 删除关键字段为空的行
    df = df[(df["water_level"] > 0) & (df["water_level"] < 300)]  # 水位应在合理范围
    df = df[(df["temperature"] > -50) & (df["temperature"] < 60)]  # 温度合理范围
    df = df[(df["humidity"] >= 0) & (df["humidity"] <= 100)]  # 湿度范围
    df["timestamp"] = df["timestamp"].dt.strftime('%Y-%m-%d %h:%m:%s')
    logging.info(f"清洗后数据行数: {len(df)}")

    # **连接数据库**
    conn = connect_db()
    if conn is None:
        logging.error("无法连接数据库，数据未存储")
        return

    cursor = conn.cursor()
    sql = """INSERT INTO sensor_data 
            (timestamp, temperature, humidity, winddirection, windpower, rains, water_level) 
            VALUES (%s, %s, %s, %s, %s, %s, %s)"""

    # **插入数据**
    try:
        for _, row in df.iterrows():
            cursor.execute(sql, (
                row["timestamp"], row["temperature"], row["humidity"],
                row["winddirection"], row["windpower"], row["rains"],
                row["water_level"]
            ))

        conn.commit()
        logging.info("CSV 数据成功插入数据库")
        print("CSV 数据已存入数据库")

    except pymysql.Error as err:
        logging.error(f"MySQL 错误: {err}")

    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    data_file_path = "water_data\\fallraw_63000200.csv"
    insert_csv_data(data_file_path)