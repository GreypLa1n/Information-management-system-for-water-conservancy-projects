# 水利工程的信息管理系统
此项目通过读取给定文件中的水利信息并将其进行本地可视化处理，当可能出现警戒情况时发出弹窗和邮件警告，同时部署了本地deepseek大模型，可进行简单的问答。<br><br>

3/14 增加了水利数据集，将数据读入数据库，修改main.py使每100条数据更新一次图表，优化视觉体验<br><br>
3/16 警告弹窗若30秒内未关闭则发送警告邮件，同时检测半个小时内是否发送过警告邮件，若发送过则不再发送。<br><br>
3/17 添加了deepseek查询模块<br><br>
3/21 添加了web页面，提供数据实时显示和历史数据查询功能，实现了网页deepseek调用<br><br><br>
**项目概览**<br>
![项目概览](https://github.com/GreypLa1n/Information-management-system-for-water-conservancy-projects/blob/main/images/project_overview.png?raw=true)<br><br>
**警告页面**<br>
![警告页面](https://github.com/GreypLa1n/Information-management-system-for-water-conservancy-projects/blob/main/images/project_warning.png?raw=true)<br><br>
**网页页面**<br>
![网页页面](https://github.com/GreypLa1n/Information-management-system-for-water-conservancy-projects/blob/main/images/web_view1.png?raw=true)<br><br><br>
![网页页面](https://github.com/GreypLa1n/Information-management-system-for-water-conservancy-projects/blob/main/images/web_view2.png?raw=true)<br><br><br>
![网页页面](https://github.com/GreypLa1n/Information-management-system-for-water-conservancy-projects/blob/main/images/web_view3.png?raw=true)<br><br><br>

# TO-DO List<br>
增加水位数据预测，将水位预测结果和当前结果相比对，差距过大则报警
