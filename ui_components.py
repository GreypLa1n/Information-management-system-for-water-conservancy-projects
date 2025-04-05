import tkinter as tk
from tkinter import ttk, messagebox
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import logging
import time
import mplcursors
from database import get_history_data
from email_alerts import check_alert_window, send_email_alert

# 配置日志
logger = logging.getLogger('ui_components')

# 全局常量
WATER_LEVEL_THRESHOLD = 85.45  # 水位警戒值（米）
AVG_DATA_POINTS = 100  # 数据平滑窗口大小
REFRESH_INTERVAL = 0.1  # 刷新间隔（秒）

# 全局状态变量
alerted_timestamps = set()  # 存储已报警的时间戳
alerted_window = None  # 当前警报窗口对象

class HistoryDataViewer:
    """查看历史数据"""
    
    def __init__(self, parent):
        self.parent = parent
        self.window = None
        
    def show(self):
        """显示历史数据窗口"""
        if self.window and self.window.winfo_exists():
            self.window.focus_force()
            return
            
        # 创建新窗口
        self.window = tk.Toplevel(self.parent)
        self.window.title("历史数据")
        self.window.geometry("800x400")
        
        # 创建Frame容器
        frame = tk.Frame(self.window)
        frame.pack(expand=True, fill="both")
        
        # 添加垂直滚动条
        y_scroll = tk.Scrollbar(frame, orient="vertical")
        y_scroll.pack(side="right", fill="y")
        
        # 创建TreeView控件
        tree = ttk.Treeview(
            frame, 
            columns=("时间", "水位(m)", "温度(℃)", "湿度(%)", "风力(m/s)", "风向", "降雨量(mm)"), 
            show="headings",
            yscrollcommand=y_scroll.set
        )
        
        # 设置列标题
        tree.heading("时间", text="时间")
        tree.heading("水位(m)", text="水位 (m)")
        tree.heading("温度(℃)", text="温度 (℃)")
        tree.heading("湿度(%)", text="湿度 (%)")
        tree.heading("风力(m/s)", text="风力 (m/s)")
        tree.heading("风向", text="风向")
        tree.heading("降雨量(mm)", text="降雨量 (mm)")
        
        # 设置列宽度
        tree.column("时间", width=150, anchor="center")
        tree.column("水位(m)", width=80, anchor="center")
        tree.column("温度(℃)", width=80, anchor="center")
        tree.column("湿度(%)", width=80, anchor="center")
        tree.column("风力(m/s)", width=80, anchor="center")
        tree.column("风向", width=100, anchor="center")
        tree.column("降雨量(mm)", width=80, anchor="center")
        
        # 设置滚动条
        tree.pack(expand=True, fill="both")
        y_scroll.config(command=tree.yview)
        
        # 添加导出按钮
        export_button = tk.Button(
            self.window, 
            text="导出数据", 
            command=self.export_data
        )
        export_button.pack(pady=5)
        
        # 加载数据
        self.load_data(tree)
        
        # 每30秒自动刷新一次数据
        self.schedule_refresh(tree)
    
    def load_data(self, tree):
        """从数据库加载历史数据并填充到TreeView"""
        try:
            # 清空现有数据
            for item in tree.get_children():
                tree.delete(item)
                
            # 获取历史数据
            rows = get_history_data(limit=100)
            
            # 填充数据
            for row in rows:
                tree.insert("", "end", values=row)
                
            logger.info("历史数据已加载")
            
        except Exception as e:
            logger.error(f"加载历史数据失败: {e}")
            messagebox.showerror("数据错误", f"无法加载历史数据：{e}")
    
    def schedule_refresh(self, tree):
        """定期刷新数据"""
        if self.window and self.window.winfo_exists():
            self.load_data(tree)
            # 30秒后再次刷新
            self.window.after(30000, lambda: self.schedule_refresh(tree))
    
    def export_data(self):
        """导出数据到CSV文件"""
        try:
            import pandas as pd
            from datetime import datetime
            
            # 获取数据
            rows = get_history_data(limit=1000)
            
            # 转换为DataFrame
            df = pd.DataFrame(rows, columns=["时间", "水位(m)", "温度(℃)", "湿度(%)", "风力(m/s)", "风向", "降雨量(mm)"])
            
            # 生成文件名
            filename = f"水利数据_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            # 保存到文件
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            
            messagebox.showinfo("导出成功", f"数据已成功导出到: {filename}")
            logger.info(f"数据已导出到: {filename}")
            
        except Exception as e:
            logger.error(f"导出数据失败: {e}")
            messagebox.showerror("导出错误", f"导出数据失败: {e}")


class DataPlotter:
    """数据可视化绘图器"""
    
    def __init__(self, figure, axes):
        self.fig = figure
        self.ax1, self.ax2, self.ax3, self.ax4 = axes
        
    def update_plots(self, df, year, subset_size=AVG_DATA_POINTS):
        """更新所有图表"""
        for i in range(0, len(df)):
            # 清除旧图表
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            self.ax4.clear()
            
            # 获取当前数据子集
            subset = df.iloc[i:i + subset_size]
            if subset.empty:
                continue
                
            # 更新图表标题
            self.fig.suptitle(f"{year} 年数据分析", fontsize=14, fontweight="bold", x=0.1, y=0.99)
            
            # 绘制各种图表
            self._plot_water_level(subset)
            self._plot_temperature(subset)
            self._plot_humidity(subset)
            self._plot_windpower(subset)
            
            # 刷新画布
            self.fig.canvas.draw()
            time.sleep(REFRESH_INTERVAL)
            
    def _plot_water_level(self, data):
        """绘制水位趋势图"""
        self.ax1.plot(data['timestamp'], data['water_level'], 
                     label="水位 (m)", color="blue", alpha=0.3)
        self.ax1.plot(data['timestamp'], data['water_level_smooth'], 
                     label="水位 (平滑)", color="blue")
        self.ax1.set_title("水位变化", fontsize=15)
        self.ax1.set_ylabel("水位 (m)", fontsize=15)
        self.ax1.legend()
        self.ax1.grid()
        
    def _plot_temperature(self, data):
        """绘制温度趋势图"""
        self.ax2.plot(data['timestamp'], data['temperature'], 
                     label="温度 (℃)", color="red", alpha=0.3)
        self.ax2.plot(data['timestamp'], data['temperature_smooth'], 
                     label="温度 (平滑)", color="red")
        self.ax2.set_title("温度变化", fontsize=15)
        self.ax2.set_ylabel("温度 (℃)", fontsize=15)
        self.ax2.legend()
        self.ax2.grid()
        
    def _plot_humidity(self, data):
        """绘制湿度趋势图"""
        self.ax3.plot(data['timestamp'], data['humidity'], 
                     label="湿度 (%)", color="green", alpha=0.3)
        self.ax3.plot(data['timestamp'], data['humidity_smooth'], 
                     label="湿度 (平滑)", color="green")
        self.ax3.set_title("湿度变化", fontsize=15)
        self.ax3.set_ylabel("湿度 (%)", fontsize=15)
        self.ax3.legend()
        self.ax3.grid()
        
    def _plot_windpower(self, data):
        """绘制风力趋势图"""
        self.ax4.plot(data['timestamp'], data['windpower'], 
                     label="风力 (m/s)", color="purple", alpha=0.3)
        self.ax4.plot(data['timestamp'], data['windpower_smooth'], 
                     label="风力 (平滑)", color="purple")
        self.ax4.set_title("风力变化", fontsize=15)
        self.ax4.set_ylabel("风力 (m/s)", fontsize=15)
        self.ax4.legend()
        self.ax4.grid()


class WaterLevelMonitor:
    """水位监控系统"""
    
    def __init__(self, parent, threshold=WATER_LEVEL_THRESHOLD):
        self.parent = parent
        self.threshold = threshold
        
    def check_alerts(self, timestamps, water_levels):
        """监测水位是否需要发出警报"""
        global alerted_timestamps, alerted_window
        
        for i, water_level in enumerate(water_levels):
            # 检查水位是否超过警戒值且未报警过
            if water_level > self.threshold and timestamps[i] not in alerted_timestamps:
                # 如果有之前的弹窗未关闭，先关闭
                if alerted_window and alerted_window.winfo_exists():
                    alerted_window.destroy()
                
                # 记录报警时间戳，防止重复报警
                alerted_timestamps.add(timestamps[i])
                
                # 创建警报弹窗
                self._create_alert_window(timestamps[i], water_level)
                break
                
    def _create_alert_window(self, timestamp, water_level):
        """创建警报弹窗"""
        global alerted_window
        
        # 创建警报信息
        warning_message = f"警告！\n时间：{timestamp}\n水位过高：{water_level}米\n请检查系统！"
        
        # 创建弹窗
        alert_window = tk.Toplevel(self.parent)
        alert_window.title("水位警告")
        alert_window.lift()  # 保证弹窗显示在最前面
        alert_window.attributes('-topmost', True)  # 置顶
        
        # 警告标签
        alert_label = tk.Label(
            alert_window, 
            text=warning_message, 
            font=("SimHei", 12),
            fg="red",
            padx=50, 
            pady=50
        )
        alert_label.pack()
        
        # 确认按钮
        def close_alert():
            alert_window.destroy()
            global alerted_window
            alerted_window = None
            
        confirm_button = tk.Button(
            alert_window, 
            text="确认", 
            command=close_alert,
            width=10,
            height=2
        )
        confirm_button.pack(pady=10)
        
        # 启动30秒后检查弹窗状态的线程
        threading.Thread(
            target=check_alert_window, 
            args=(alert_window, timestamp, water_level), 
            daemon=True
        ).start()
        
        # 更新全局弹窗对象
        alerted_window = alert_window
        
        # 记录日志
        logger.warning(f"水位警告已触发：水位={water_level}米，时间={timestamp}")


class DeepSeekPanel:
    """DeepSeek查询模块"""
    
    def __init__(self, parent, query_handler):
        self.parent = parent
        self.query_handler = query_handler
        self.input_text = None
        self.output_text = None
        
        self._create_panel()
        
    def _create_panel(self):
        query_label = tk.Label(self.parent, text="请输入您的问题：", font=("SimHei", 14))
        query_label.pack(anchor="w")
        
        # 输入框
        self.input_text = tk.Text(
            self.parent, 
            height=5, 
            width=60, 
            font=("SimHei", 14)
        )
        self.input_text.pack(pady=5)
        
        button_frame = tk.Frame(self.parent)
        button_frame.pack(pady=5)
        
        # 提交按钮
        submit_button = tk.Button(button_frame, text="查询", font=("SimHei", 14), command=self._handle_query,width=10)
        submit_button.pack(side="left", padx=5)
        
        # 清除按钮
        clear_button = tk.Button(button_frame, text="清除", font=("SimHei", 14), command=self._clear_input,width=10)
        clear_button.pack(side="left", padx=5)
        
        # 输出框
        self.output_text = tk.Text(self.parent, height=40, width=60, font=("SimHei", 14), state=tk.DISABLED,wrap="word"
        )
        self.output_text.pack(pady=5)
        
        # 添加滚动条
        scrollbar = tk.Scrollbar(self.parent, command=self.output_text.yview)
        scrollbar.place(relx=1.0, rely=0.5, relheight=0.9, anchor="e")
        self.output_text.config(yscrollcommand=scrollbar.set)
    
    def _handle_query(self):
        """处理查询按钮点击事件"""
        # 获取输入文本
        question = self.input_text.get("1.0", "end-1c").strip()
        
        if not question:
            self._update_output("请输入问题后再查询。")
            return
            
        # 更新输出显示处理中状态
        self._update_output("正在处理您的问题，请稍候...")
        
        # 在单独线程中处理查询
        threading.Thread(target=self._process_query, args=(question, ), daemon=True).start()
    
    def _process_query(self, question):
        """在单独线程中处理查询请求"""
        try:
            # 调用查询处理函数
            response = self.query_handler(question)
            # 更新输出显示
            self._update_output(response)
            
        except Exception as e:
            logger.error(f"处理查询时出错: {e}")
            self._update_output(f"处理查询时出错: {e}")
    
    def _update_output(self, text):
        """更新输出文本框内容"""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, text)
        self.output_text.config(state=tk.DISABLED)
    
    def _clear_input(self):
        """清除输入框内容"""
        self.input_text.delete("1.0", tk.END)


# 创建图表组件
def create_charts(parent):
    """创建图表组件"""
    # 创建Matplotlib图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8), dpi=100)
    fig.tight_layout(pad=3.0)
    
    # 创建画布
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)
    
    # 启用交互式图表
    mplcursors.cursor()
    
    return fig, [ax1, ax2, ax3, ax4], canvas 