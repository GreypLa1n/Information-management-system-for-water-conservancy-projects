import tkinter as tk
from tkinter import ttk, messagebox
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import configparser
import logging
import time
import mplcursors
from database import get_history_data
from email_alerts import check_alert_window, send_email_alert
from Transformer_predict import transformer_water_anomaly

# 配置日志
logger = logging.getLogger('ui_components')

# 读取配置文件
config = configparser.ConfigParser()
config.read("./config.cfg", encoding="UTF-8")
config_email = config["Email_Setting"]
WATER_LEVEL_THRESHOLD = float(config_email["WarningLevel"])  # 警戒水位

# 全局常量
AVG_DATA_POINTS = 100  # 数据平滑窗口大小
REFRESH_INTERVAL = 0.1  # 刷新间隔（秒）

# 全局状态变量
alerted_timestamps = set()  # 存储已报警的时间戳
alerted_window = None  # 当前警报窗口对象

class HistoryDataViewer:
    """查看历史数据"""
    
    def __init__(self, parent, data_plotter=None):
        self.parent = parent
        self.window = None
        self.data_plotter = data_plotter  # 数据可视化绘图器引用
        
    def set_data_plotter(self, plotter):
        """设置数据绘图器引用"""
        self.data_plotter = plotter
        
    def show(self):
        """显示历史数据窗口"""
        if self.window and self.window.winfo_exists():
            self.window.focus_force()
            return
            
        # 创建新窗口
        self.window = tk.Toplevel(self.parent)
        self.window.title("历史数据")
        self.window.geometry("1200x600")
        
        # 创建Frame容器
        frame = tk.Frame(self.window)
        frame.pack(expand=True, fill="both", padx=10, pady=10)
        
        # 添加当前位置信息
        position_info = "未连接到可视化模块"
        if self.data_plotter:
            position = self.data_plotter.get_current_position()
            current_index = position["current_index"]
            max_index = position["max_iterations"]
            current_timestamp = position["current_timestamp"]
            
            if current_timestamp:
                position_info = f"当前位置: 索引 {current_index}/{max_index}, 时间: {current_timestamp}"
            else:
                position_info = f"当前位置: 索引 {current_index}/{max_index}"
        
        # 添加标题标签
        title_label = tk.Label(
            frame, 
            text="水利工程历史数据", 
            font=("SimHei", 14, "bold")
        )
        title_label.pack(pady=(0, 5))
        
        # 添加位置信息标签
        self.position_label = tk.Label(
            frame,
            text=position_info,
            font=("SimHei", 10)
        )
        self.position_label.pack(pady=(0, 10))
        
        # 创建数据展示区域
        data_frame = tk.Frame(frame)
        data_frame.pack(expand=True, fill="both")
        
        # 添加垂直滚动条
        y_scroll = tk.Scrollbar(data_frame, orient="vertical")
        y_scroll.pack(side="right", fill="y")
        
        # 添加水平滚动条
        x_scroll = tk.Scrollbar(data_frame, orient="horizontal")
        x_scroll.pack(side="bottom", fill="x")
        
        # 创建TreeView控件
        tree = ttk.Treeview(
            data_frame, 
            columns=("时间", "水位(m)", "温度(℃)", "湿度(%)", "风力(m/s)", "风向", "降雨量(mm)"), 
            show="headings",
            yscrollcommand=y_scroll.set,
            xscrollcommand=x_scroll.set
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
        x_scroll.config(command=tree.xview)
        
        # 创建按钮区域
        btn_frame = tk.Frame(frame)
        btn_frame.pack(pady=10)
        
        # 添加导出按钮
        export_button = tk.Button(btn_frame, text="导出数据", command=self.export_data,width=15,height=2)
        export_button.pack(side=tk.LEFT, padx=5)
        
        # 添加刷新按钮
        refresh_button = tk.Button(btn_frame, text="刷新数据", command=lambda: self.refresh_data(tree), width=15, height=2)
        refresh_button.pack(side=tk.LEFT, padx=5)
        
        # 加载数据 - 只加载一次
        self.load_data(tree)
    
    def load_data(self, tree):
        """从数据库加载历史数据并填充到TreeView"""
        try:
            # 清空现有数据
            for item in tree.get_children():
                tree.delete(item)
            
            # 计算要显示的数据条数
            limit = 100  # 默认显示100条
            if self.data_plotter:
                position = self.data_plotter.get_current_position()
                current_index = position["current_index"]
                # 根据当前可视化索引计算显示的数据量
                limit += current_index
                
            # 获取历史数据
            all_rows = get_history_data()  # 数据已经是按时间降序排列的（从新到旧）
            
            if not all_rows:
                messagebox.showinfo("无数据", "没有找到历史数据")
                return
            
            # 需要反转顺序，获取最早的记录
            all_rows_reversed = list(reversed(all_rows))  # 反转为从旧到新
            
            # 截取最早的limit条数据
            earliest_rows = all_rows_reversed[:limit]
            
            # 再次反转，使其按照时间从新到旧排序显示
            rows = list(reversed(earliest_rows))
            
            # 显示加载提示
            progress_window = tk.Toplevel(self.window)
            progress_window.title("加载进度")
            progress_window.geometry("300x100")
            progress_window.transient(self.window)
            progress_window.grab_set()
            
            progress_label = tk.Label(progress_window, text=f"正在加载 {len(rows)}/{len(all_rows)} 条数据...", pady=10)
            progress_label.pack()
            
            # 添加详细进度显示
            detail_label = tk.Label(progress_window, text="", pady=5)
            detail_label.pack()
            
            # 更新UI
            self.window.update()
            
            # 使用批量插入提高性能
            batch_size = 100
            total_rows = len(rows)
            
            for i in range(0, total_rows, batch_size):
                # 更新进度
                current_position = min(i + batch_size, total_rows)
                progress_label.config(text=f"正在加载... {current_position}/{total_rows}")
                
                # 显示详细读取信息
                detail_label.config(text=f"当前读取索引: {i}/{total_rows}, 读取范围: {i} 到 {current_position}")
                
                progress_window.update()
                
                # 插入一批数据
                batch = rows[i:i + batch_size]
                for row in batch:
                    tree.insert("", "end", values=row)
                
                # 更新UI
                self.window.update()
            
            # 关闭进度窗口
            progress_window.destroy()
            
            # 获取显示的数据日期范围
            if rows:
                first_row = rows[0]  # 显示的最晚读入的数据
                last_row = rows[-1]  # 显示的最早读入的数据
                first_date = first_row[0]
                last_date = last_row[0]
            
            # 显示加载完成信息
            if self.data_plotter:
                position = self.data_plotter.get_current_position()
                current_index = position["current_index"]
                self.window.title(f"历史数据 (显示 {len(rows)}/{len(all_rows)} 条记录，基于当前可视化索引 {current_index})")
                # 更新位置信息标签
                if rows:
                    self.position_label.config(text=f"当前位置: 索引 {current_index}/{position['max_iterations']}, 显示数据量: {len(rows)}, 日期范围: {last_date} 至 {first_date}")
                else:
                    self.position_label.config(text=f"当前位置: 索引 {current_index}/{position['max_iterations']}, 显示数据量: {len(rows)}")
            else:
                self.window.title(f"历史数据 (显示 {len(rows)}/{len(all_rows)} 条记录)")
                
            logger.info(f"历史数据已加载，显示 {len(rows)}/{len(all_rows)} 条记录")
            
        except Exception as e:
            logger.error(f"加载历史数据失败: {e}")
            messagebox.showerror("数据错误", f"无法加载历史数据：{e}")
    
    def export_data(self):
        """导出数据到CSV文件"""
        try:
            import pandas as pd
            from datetime import datetime
            
            # 计算要导出的数据条数
            limit = 100  # 默认导出100条
            if self.data_plotter:
                position = self.data_plotter.get_current_position()
                current_index = position["current_index"]
                # 根据当前可视化索引计算导出的数据量
                limit = 100 + current_index
            
            # 获取历史数据
            all_rows = get_history_data()  # 数据已经是按时间降序排列的（从新到旧）
            
            if not all_rows:
                messagebox.showinfo("无数据", "没有找到可导出的数据")
                return
            
            # 需要反转顺序，获取最早的记录
            all_rows_reversed = list(reversed(all_rows))  # 反转为从旧到新
            
            # 截取最早的limit条数据
            earliest_rows = all_rows_reversed[:limit]
            
            # 再次反转，使其按照时间从新到旧排序显示
            rows = list(reversed(earliest_rows))
                
            # 显示进度提示
            progress_window = tk.Toplevel(self.window)
            progress_window.title("导出进度")
            progress_window.geometry("300x100")
            progress_window.transient(self.window)
            progress_window.grab_set()
            
            progress_label = tk.Label(progress_window, text=f"正在导出数据，共 {len(rows)}/{len(all_rows)} 条...", pady=10)
            progress_label.pack()
            
            # 添加详细进度显示
            detail_label = tk.Label(progress_window, text="正在准备导出...", pady=5)
            detail_label.pack()
            
            # 更新UI
            self.window.update()
            
            # 显示处理进度
            total_rows = len(rows)
            detail_label.config(text=f"正在处理 {total_rows} 条数据...")
            progress_window.update()
            
            # 转换为DataFrame
            df = pd.DataFrame(rows, columns=["时间", "水位(m)", "温度(℃)", "湿度(%)", "风力(m/s)", "风向", "降雨量(mm)"])
            
            # 获取日期范围并添加到文件名
            first_date = ""
            last_date = ""
            if rows:
                first_row = rows[0]  # 最新数据
                last_row = rows[-1]  # 最早数据
                first_date = str(first_row[0]).replace("-", "").replace(":", "").replace(" ", "_")[:8]
                last_date = str(last_row[0]).replace("-", "").replace(":", "").replace(" ", "_")[:8]
            
            # 生成文件名
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            if first_date and last_date:
                filename = f"水利数据_{first_date}至{last_date}_{current_time}.csv"
            else:
                filename = f"水利数据_{current_time}.csv"
            
            # 更新进度
            detail_label.config(text=f"正在写入文件: {filename}")
            progress_window.update()
            
            # 保存到文件
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            
            # 更新完成状态
            detail_label.config(text=f"导出完成: {filename}")
            progress_window.update()
            
            # 延迟一小段时间显示完成状态
            self.window.after(1000, progress_window.destroy)
            
            # 准备显示的消息
            date_range_info = ""
            if rows:
                first_date_display = str(first_row[0])
                last_date_display = str(last_row[0])
                date_range_info = f"日期范围: {first_date_display} 至 {last_date_display}\n"
            
            if self.data_plotter:
                position = self.data_plotter.get_current_position()
                current_index = position["current_index"]
                message = f"数据已成功导出到: {filename}\n{date_range_info}基于当前可视化索引 {current_index}，共导出 {len(rows)}/{len(all_rows)} 条记录"
            else:
                message = f"数据已成功导出到: {filename}\n{date_range_info}共导出 {len(rows)}/{len(all_rows)} 条记录"
                
            messagebox.showinfo("导出成功", message)
            logger.info(f"数据已导出到: {filename}，共导出 {len(rows)}/{len(all_rows)} 条记录")
            
        except Exception as e:
            logger.error(f"导出数据失败: {e}")
            messagebox.showerror("导出错误", f"导出数据失败: {e}")

    def refresh_data(self, tree):
        """刷新数据，根据当前可视化索引重新加载数据"""
        if self.data_plotter:
            # 获取最新的可视化索引
            position = self.data_plotter.get_current_position()
            current_index = position["current_index"]
            
            # 更新位置信息标签
            self.position_label.config(text=f"当前位置: 索引 {current_index}/{position['max_iterations']}, 正在刷新数据...")
            self.window.update()
            
            # 重新加载数据
            self.load_data(tree)
        else:
            messagebox.showinfo("提示", "未连接到可视化模块，无法获取最新索引")
            # 仍然刷新数据
            self.load_data(tree)


class DataPlotter:
    """数据可视化绘图器"""
    
    def __init__(self, figure, axes):
        self.fig = figure
        self.ax1, self.ax2, self.ax3, self.ax4 = axes
        self.current_index = 0  # 当前数据索引位置
        self.max_iterations = 0  # 总数据点数量
        self.current_timestamp = None  # 当前显示的时间戳
        
    def update_plots(self, df, year, subset_size=AVG_DATA_POINTS):
        """更新所有图表"""
        
        # 处理所有数据
        self.max_iterations = len(df)
        
        for i in range(0, self.max_iterations):
            try:
                # 更新当前索引
                self.current_index = i
                
                # 清除旧图表
                self.ax1.clear()
                self.ax2.clear()
                self.ax3.clear()
                self.ax4.clear()
                
                # 获取当前数据子集
                subset = df.iloc[i:i + subset_size]
                if subset.empty:
                    continue
                
                # 记录当前时间戳
                if not subset.empty:
                    self.current_timestamp = subset['timestamp'].iloc[-1]
                
                # 输出所有日志，不限制频率
                print(f"当前循环索引: {i}/{self.max_iterations}, 子集范围: {i} 到 {i + subset_size}")
                    
                # 更新图表标题
                self.fig.suptitle(f"{year} 年", fontsize=14, fontweight="bold", x=0.1, y=0.99)
                
                # 绘制各种图表
                self._plot_water_level(subset)
                self._plot_temperature(subset)
                self._plot_humidity(subset)
                self._plot_windpower(subset)
                
                # 检查是否需要发出水位警报
                # 获取最新的水位数据
                latest_water_level = subset['water_level'].iloc[-1] if not subset.empty else None
                latest_timestamp = subset['timestamp'].iloc[-1] if not subset.empty else None
                
                if latest_water_level is not None and latest_timestamp is not None:
                    # 与警戒水位进行比较
                    print(f"索引 {i} 的最新水位: {latest_water_level}, 时间: {latest_timestamp}")
                    if latest_water_level > WATER_LEVEL_THRESHOLD:
                        # 创建临时列表传递给警报检测函数
                        self._check_water_alert(latest_timestamp, latest_water_level)
                
                # 刷新画布
                self.fig.canvas.draw()
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"绘图过程中出错: {e}")
                # 如果发生错误，不要让整个进程卡死
                break
                
    def get_current_position(self):
        """获取当前数据位置信息"""
        return {
            "current_index": self.current_index,
            "max_iterations": self.max_iterations,
            "current_timestamp": self.current_timestamp
        }
    
    def _check_water_alert(self, timestamp, water_level):
        """检查单个水位数据点是否需要发出警报"""
        global alerted_timestamps, alerted_window
        
        # 检查水位是否超过警戒值且未报警过
        if water_level > WATER_LEVEL_THRESHOLD and timestamp not in alerted_timestamps:
            # 如果有之前的弹窗未关闭，先关闭
            if alerted_window and alerted_window.winfo_exists():
                alerted_window.destroy()
            
            # 记录报警时间戳，防止重复报警
            alerted_timestamps.add(timestamp)
            
            # 创建警报弹窗
            warning_message = f"警告！\n时间：{timestamp}\n水位过高：{water_level}米\n请检查系统！"
            
            # 创建弹窗
            alert_window = tk.Toplevel(self.fig.canvas.get_tk_widget().master)
            alert_window.title("水位警告")
            alert_window.lift()  # 保证弹窗显示在最前面
            alert_window.attributes('-topmost', True)  # 置顶
            
            # 警告标签
            alert_label = tk.Label(
                alert_window, 
                text=warning_message, 
                font=("SimHei", 12),
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
        """绘制降雨量趋势图"""
        self.ax4.plot(data['timestamp'], data['rains'], 
                     label="降雨量 (mm)", color="purple", alpha=0.3)
        self.ax4.plot(data['timestamp'], data['rains_smooth'], 
                     label="降雨量 (平滑)", color="purple")
        self.ax4.set_title("降雨量变化", fontsize=15)
        self.ax4.set_ylabel("降雨量 (mm)", fontsize=15)
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
            # 传入最近100条水位数据检测是否异常，返回True表示异常，False表示正常
            is_anomaly = transformer_water_anomaly(i, water_levels) if i > 99 else False
            if (water_level > self.threshold or is_anomaly) and timestamps[i] not in alerted_timestamps:
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