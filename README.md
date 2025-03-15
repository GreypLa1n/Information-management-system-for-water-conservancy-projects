# 水利工程的信息管理系统
此项目通过读取给定文件中的水利信息并将其进行可视化处理，当可能出现警戒情况时发出警告

未来期望:添加爬虫模块爬取水利网信息和deepseek本地部署+api调用
调整布局，添加背景图片，将GUI界面更加美观<br><br>

3/14 增加了水利数据集，将数据读入数据库，修改main.py使每100条数据更新一次图表，优化视觉体验<br><br>
3/16 警告弹窗若30秒内未关闭则发送警告邮件，同时检测半个小时内是否发送过警告邮件，若发送过则不再发送。<br><br><br>
项目概览<br>
![项目概览](https://github.com/GreypLa1n/Information-management-system-for-water-conservancy-projects/blob/main/images/project_overview.png?raw=true)<br><br>
警告页面<br>
![警告页面](https://github.com/GreypLa1n/Information-management-system-for-water-conservancy-projects/blob/main/images/project_warning.png?raw=true)
