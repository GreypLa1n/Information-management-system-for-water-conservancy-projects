# -*- coding: utf-8 -*-
# @Time    : 2025/3/12 14:03
# @Author  : Bruam1
# @Email   : grey040612@gmail.com
# @File    : main.py
# @Software: Vscode

import tkinter as tk
import threading
import logging
import pandas as pd
import matplotlib
from dotenv import load_dotenv
import datetime
import sys
import time
from deepseek import ask_deepseek
from database import get_data_for_visualization, get_history_data
from ui_components import (
    HistoryDataViewer, 
    DataPlotter, 
    WaterLevelMonitor,
    DeepSeekPanel,
    create_charts
)

# 配置中文显示
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('main')


# 加载环境变量
load_dotenv()

class Water_Data_Visualization:
    """水利工程数据可视化与监控系统主应用类"""
    
    def __init__(self):
        """初始化应用程序"""
        self.root = None
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("水利工程数据可视化与监控系统")
        self.root.geometry("2400x960")
        
        # 创建主框架
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建图表区域
        chart_frame = tk.Frame(main_frame)
        chart_frame.grid(row=0, column=0, sticky="nsew")
        
        # 创建图表
        fig, axes, canvas = create_charts(chart_frame)
        
        # 添加历史数据查看按钮
        history_viewer = HistoryDataViewer(self.root)
        history_button = tk.Button(
            chart_frame, 
            text="查看历史数据", 
            command=history_viewer.show, 
            font=("SimHei", 12)
        )
        history_button.pack(pady=10)
        
        # 创建DeepSeek查询面板
        deepseek_frame = tk.Frame(main_frame, width=500, padx=10, pady=10)
        deepseek_frame.grid(row=0, column=1, sticky="nsew")
        deepseek_panel = DeepSeekPanel(deepseek_frame, ask_deepseek)
        
        # 设置布局权重
        main_frame.columnconfigure(0, weight=4)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(0, weight=1)
        
        # 创建数据绘图器
        data_plotter = DataPlotter(fig, axes)
        
        # 创建水位监控器
        water_monitor = WaterLevelMonitor(self.root)
        
        # 启动数据更新线程
        threading.Thread(
            target=self.update_data_thread,
            args=(data_plotter, water_monitor),
            daemon=True
        ).start()
    
    def update_data_thread(self, data_plotter, water_monitor):
        """数据更新线程"""
        logger.info("数据更新线程已启动")
        
        while True:
            try:
                # 获取可视化数据
                data = get_data_for_visualization()
                
                if data:
                    # 解析数据
                    timestamps, water_levels, temperatures, humidities, windpowers = data
                    
                    # 转换为DataFrame以便处理
                    df = pd.DataFrame({"timestamp": timestamps, "water_level": water_levels, "temperature": temperatures, "humidity": humidities, "windpower": windpowers})
                    
                    # 获取年份
                    year = df["timestamp"].iloc[0].year if not df.empty else datetime.datetime.now().year
                    
                    # 计算滚动均值（平滑处理）
                    df['water_level_smooth'] = df['water_level'].rolling(window=3, min_periods=1).mean()
                    df['temperature_smooth'] = df['temperature'].rolling(window=3, min_periods=1).mean()
                    df['humidity_smooth'] = df['humidity'].rolling(window=3, min_periods=1).mean()
                    df['windpower_smooth'] = df['windpower'].rolling(window=3, min_periods=1).mean()
                    
                    # 更新图表
                    data_plotter.update_plots(df, year)
                    
                    # 检查是否需要发出水位警报
                    water_monitor.check_alerts(timestamps, water_levels)
                    
            except Exception as e:
                logger.error(f"更新数据线程出错: {e}")
                # 休眠一段时间后重试
                time.sleep(5)
    
    def run(self):
        """运行应用程序"""
        logger.info("应用程序已启动")
        self.root.mainloop()

if __name__ == "__main__":
    app = Water_Data_Visualization()
    app.run()