import os
import pymysql
import datetime
import logging
from dotenv import load_dotenv

# 配置日志
logger = logging.getLogger('database')

class Database:
    """数据库访问类，处理所有数据库操作"""
    
    def __init__(self):
        """初始化数据库连接参数"""
        # 加载环境变量
        load_dotenv()
        
        # 从环境变量中获取数据库连接信息
        self.db_config = {
            'host': os.getenv("DB_HOST", "localhost"),
            'user': os.getenv("DB_USER", "sensor_data"),
            'password': os.getenv("DB_PASSWORD", ""),
            'database': os.getenv("DB_NAME", "reservoir_db")
        }
        logger.info(f"数据库配置已加载, 连接到: {self.db_config['host']}/{self.db_config['database']}")
    
    def connect(self):
        """创建并返回数据库连接"""
        try:
            return pymysql.connect(**self.db_config)
        except pymysql.Error as e:
            logger.error(f"数据库连接失败: {e}")
            raise
    
    def get_sensor_data(self, limit=100, order_by="id DESC"):
        """获取传感器数据"""
        try:
            conn = self.connect()
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            
            query = f"""SELECT * FROM sensor_data 
                ORDER BY {order_by} 
                LIMIT {limit}
            """
            
            cursor.execute(query)
            data = cursor.fetchall()
            
            # 格式化时间戳
            for row in data:
                if isinstance(row["timestamp"], datetime.datetime):
                    row["timestamp"] = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            
            cursor.close()
            conn.close()
            return data
            
        except pymysql.Error as e:
            logger.error(f"获取传感器数据失败: {e}")
            return []
    
    def get_data_for_visualization(self):
        """获取可视化数据"""
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            query = """
                SELECT timestamp, COALESCE(water_level, 0), COALESCE(temperature, 0), 
                COALESCE(humidity, 0), COALESCE(windpower, 0), COALESCE(rains, 0)
                FROM sensor_data 
                WHERE water_level IS NOT NULL 
                ORDER BY timestamp ASC LIMIT 200
            """
            
            cursor.execute(query)
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if not rows:
                return None
            
            # 解析数据
            timestamps, water_levels, temperatures, humidities, windpowers, rains = zip(*rows)
            
            
            return timestamps, water_levels, temperatures, humidities, windpowers, rains
            
        except pymysql.Error as e:
            logger.error(f"获取可视化数据失败: {e}")
            return None
    
    def get_history_data(self):
        """获取历史数据"""
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            query = """
                SELECT timestamp, water_level, temperature, humidity, 
                windpower, winddirection, rains 
                FROM sensor_data 
                ORDER BY timestamp DESC
            """
            cursor.execute(query)
                
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            return rows
            
        except pymysql.Error as e:
            logger.error(f"获取历史数据失败: {e}")
            return []

# 创建全局数据库实例
db = Database()

def connect_db():
    """创建并返回数据库连接"""
    return db.connect()

def get_sensor_data(limit=100):
    """获取传感器数据"""
    return db.get_sensor_data(limit)

def get_data_for_visualization():
    """获取可视化数据"""
    return db.get_data_for_visualization()

def get_history_data():
    """获取历史数据"""
    return db.get_history_data()