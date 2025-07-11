// 全局变量
const WATER_LEVEL_THRESHOLD = 85.45;  // 水位警戒值
const DATA_WINDOW_SIZE = 100;  // 显示的数据点数量
let dataOffset = 0;  // 数据偏移量，从0开始
let charts = {
    waterLevel: null,
    temperature: null,
    humidity: null,
    windPower: null
};
let lastUpdateTime = null;
let isFirstLoad = true;  // 标记是否是第一次加载

// 创建图表配置
function createChartConfig(label, color) {
    return {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: label,
                data: [],
                borderColor: color,
                tension: 0.1,
                fill: false,
                pointRadius: 4,
                pointHoverRadius: 6,
                pointBackgroundColor: color,
                pointBorderColor: 'white',
                pointBorderWidth: 2,
                hitRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            layout: {
                padding: {
                    top: 20,     // 增加顶部内边距，为tooltip预留空间
                    right: 10,
                    bottom: 10,
                    left: 10
                }
            },
            scales: {
                y: {
                    beginAtZero: false
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        maxRotation: 0,  // 防止标签旋转
                        autoSkip: true,  // 自动跳过重叠的标签
                        maxTicksLimit: 8,  // 限制显示的标签数量
                        callback: function (value, index, values) {
                            // 只显示每隔几个点的时间
                            if (index % Math.ceil(values.length / 8) === 0) {
                                return this.getLabelForValue(value);
                            }
                            return '';
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    enabled: true,
                    mode: 'nearest',
                    intersect: true,
                    yAlign: 'bottom',
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: 'white',
                    bodyColor: 'white',
                    borderColor: 'white',
                    borderWidth: 1,
                    padding: 10,
                    displayColors: false,
                    position: 'nearest',
                    callbacks: {
                        label: function (context) {
                            return `${context.dataset.label}: ${context.parsed.y.toFixed(2)}`;
                        }
                    }
                }
            },
            hover: {
                mode: 'nearest',
                intersect: true
            },
            interaction: {
                intersect: true,
                mode: 'nearest'
            },
            animation: {
                duration: 0
            },
            clip: false  // 禁用裁剪，允许内容超出图表边界
        }
    };
}

// 初始化图表
function initCharts() {
    // 水位图表
    const waterLevelCtx = document.getElementById('water-level-chart').getContext('2d');
    charts.waterLevel = new Chart(waterLevelCtx, createChartConfig('水位(m)', 'rgb(13, 110, 253)'));

    // 温度图表
    const temperatureCtx = document.getElementById('temperature-chart').getContext('2d');
    charts.temperature = new Chart(temperatureCtx, createChartConfig('温度(℃)', 'rgb(220, 53, 69)'));

    // 湿度图表
    const humidityCtx = document.getElementById('humidity-chart').getContext('2d');
    charts.humidity = new Chart(humidityCtx, createChartConfig('湿度(%)', 'rgb(25, 135, 84)'));

    // 风力图表
    const windPowerCtx = document.getElementById('wind-power-chart').getContext('2d');
    charts.windPower = new Chart(windPowerCtx, createChartConfig('风力(m/s)', 'rgb(111, 66, 193)'));
}

// 计算滚动平均值
function calculateRollingAverage(values, windowSize = 3) {
    const result = [];
    for (let i = 0; i < values.length; i++) {
        const start = Math.max(0, i - windowSize + 1);
        const count = i - start + 1;
        const sum = values.slice(start, i + 1).reduce((a, b) => a + b, 0);
        result.push(sum / count);
    }
    return result;
}

// 计算y轴范围
function calculateYAxisRange(smoothData) {
    if (!smoothData || smoothData.length === 0) return { min: 0, max: 100 };

    const mean = smoothData.reduce((a, b) => a + b, 0) / smoothData.length;
    const stdDev = Math.sqrt(
        smoothData.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / smoothData.length
    );

    // 使用均值±5个标准差作为范围，覆盖更大范围的数据
    const min = Math.floor(mean - 5 * stdDev);
    const max = Math.ceil(mean + 5 * stdDev);

    // 对特殊情况进行处理
    if (min === max) {
        return { min: min - 1, max: max + 1 };  // 确保有一定的显示范围
    }

    // 为湿度特殊处理，确保范围在0-100之间
    if (mean >= 0 && mean <= 100) {  // 判断是否是湿度数据
        return {
            min: Math.max(0, min),
            max: Math.min(100, max)
        };
    }

    return { min, max };
}

// 更新实时数据
async function updateRealtimeData() {
    try {
        const response = await fetch(`/api/realtime-data?offset=${dataOffset}&limit=${DATA_WINDOW_SIZE}`);
        const data = await response.json();

        if (Array.isArray(data) && data.length > 0) {
            // 更新卡片数据（显示最新一条）
            const latestData = data[data.length - 1];  // 使用最后一条数据而不是第一条
            document.getElementById('water-level').textContent = latestData.water_level.toFixed(2);
            document.getElementById('temperature').textContent = latestData.temperature.toFixed(1);
            document.getElementById('humidity').textContent = latestData.humidity.toFixed(1);
            document.getElementById('wind-power').textContent = latestData.windpower.toFixed(1);

            // 获取最新数据的完整日期，用于显示在左上角
            const latestDate = new Date(latestData.timestamp);
            const dateStr = latestDate.toLocaleDateString('zh-CN', {
                year: 'numeric',
                month: '2-digit',
                day: '2-digit'
            });
            // 更新左上角的日期显示
            document.getElementById('current-date').textContent = dateStr;

            // 准备数据，只使用时间部分
            const timestamps = data.map(item => {
                const date = new Date(item.timestamp);
                return date.toLocaleTimeString('zh-CN', {
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit',
                    hour12: false
                });
            });
            const waterLevels = data.map(item => item.water_level);
            const temperatures = data.map(item => item.temperature);
            const humidities = data.map(item => item.humidity);
            const windPowers = data.map(item => item.windpower);

            // 计算滚动平均值
            const waterLevelsSmooth = calculateRollingAverage(waterLevels);
            const temperaturesSmooth = calculateRollingAverage(temperatures);
            const humiditiesSmooth = calculateRollingAverage(humidities);
            const windPowersSmooth = calculateRollingAverage(windPowers);

            // 计算每个图表的y轴范围
            const waterLevelRange = calculateYAxisRange(waterLevelsSmooth);
            const temperatureRange = calculateYAxisRange(temperaturesSmooth);
            const humidityRange = calculateYAxisRange(humiditiesSmooth);
            const windPowerRange = calculateYAxisRange(windPowersSmooth);

            // 更新各个图表
            updateChart(charts.waterLevel, timestamps, waterLevelsSmooth, waterLevelRange);
            updateChart(charts.temperature, timestamps, temperaturesSmooth, temperatureRange);
            updateChart(charts.humidity, timestamps, humiditiesSmooth, humidityRange);
            updateChart(charts.windPower, timestamps, windPowersSmooth, windPowerRange);

            // 检查水位警告（使用最新数据）
            if (latestData.water_level > WATER_LEVEL_THRESHOLD) {
                showAlert(`警告！当前水位 ${latestData.water_level.toFixed(2)}米 已超过警戒值 ${WATER_LEVEL_THRESHOLD}米`);
            }

            // 如果数据少于窗口大小，说明已经到达末尾，重置offset
            if (data.length < DATA_WINDOW_SIZE) {
                dataOffset = 0;
            } else {
                // 每次只增加1，移动一个数据点
                dataOffset += 1;
            }

            lastUpdateTime = new Date();
        }
    } catch (error) {
        console.error('获取实时数据失败:', error);
    }
}

// 更新单个图表
function updateChart(chart, labels, data, range) {
    if (!chart) return;  // 确保图表对象存在

    // 更新数据
    chart.data.labels = labels;
    chart.data.datasets[0].data = data;

    // 更新y轴范围
    if (range && chart.options.scales.y) {
        chart.options.scales.y.min = range.min;
        chart.options.scales.y.max = range.max;
    }

    // 强制更新图表
    chart.update('none');  // 使用 'none' 模式来立即更新，不使用动画
}

// 获取历史数据
async function fetchHistoryData() {
    const startDate = document.getElementById('start-date').value;
    const endDate = document.getElementById('end-date').value;

    if (!startDate || !endDate) {
        alert('请选择开始和结束日期');
        return;
    }

    try {
        // 显示加载提示
        const tbody = document.getElementById('history-data');
        tbody.innerHTML = '<tr><td colspan="7" class="text-center">数据加载中，请稍候...</td></tr>';

        const response = await fetch(`/api/history-data?start=${startDate}&end=${endDate}`);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        if (!Array.isArray(data)) {
            throw new Error('服务器返回的数据格式不正确');
        }

        tbody.innerHTML = '';

        if (data.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="text-center">没有找到符合条件的数据</td></tr>';
            return;
        }

        // 使用文档片段来优化DOM操作
        const fragment = document.createDocumentFragment();
        data.forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${new Date(row.timestamp).toLocaleString()}</td>
                <td>${row.water_level?.toFixed(2) ?? '--'}</td>
                <td>${row.temperature?.toFixed(1) ?? '--'}</td>
                <td>${row.humidity?.toFixed(1) ?? '--'}</td>
                <td>${row.windpower?.toFixed(1) ?? '--'}</td>
                <td>${row.winddirection || '--'}</td>
                <td>${row.rains?.toFixed(1) ?? '--'}</td>
            `;
            fragment.appendChild(tr);
        });
        tbody.appendChild(fragment);

    } catch (error) {
        console.error('获取历史数据失败:', error);
        const tbody = document.getElementById('history-data');
        tbody.innerHTML = `<tr><td colspan="7" class="text-center text-danger">获取数据失败: ${error.message}</td></tr>`;
    }
}

// 显示警告模态框
function showAlert(message) {
    document.getElementById('alert-message').textContent = message;
    const modal = new bootstrap.Modal(document.getElementById('alertModal'));
    modal.show();
}

// 页面切换
function switchPanel(panelId) {
    document.getElementById('realtime-panel').style.display = panelId === 'realtime' ? 'block' : 'none';
    document.getElementById('history-panel').style.display = panelId === 'history' ? 'block' : 'none';

    document.getElementById('realtime-link').classList.toggle('active', panelId === 'realtime');
    document.getElementById('history-link').classList.toggle('active', panelId === 'history');
}

// DeepSeek 查询功能
async function queryDeepseek() {
    const input = document.getElementById('deepseek-input');
    const output = document.getElementById('deepseek-output');
    const question = input.value.trim();

    console.log("发送问题:", question);

    if (!question) {
        output.value = "请输入问题后再查询。";
        return;
    }

    try {
        // 显示加载状态
        output.value = "正在分析数据，请稍候...";

        // 获取最近的数据作为上下文
        const response = await fetch('/api/deepseek', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: question })
        });

        const data = await response.json();
        console.log("收到的响应:", data);

        if (!response.ok) {
            throw new Error(data.error || `HTTP error! status: ${response.status}`);
        }

        if (data.error) {
            throw new Error(data.error);
        }

        output.value = data.response || "未能获取有效回答";
    } catch (error) {
        console.error('查询失败:', error);
        output.value = `查询失败: ${error.message}`;
    }
}

// 事件监听器
document.addEventListener('DOMContentLoaded', () => {
    // 初始化图表
    initCharts();

    // 开始定时更新数据
    updateRealtimeData();
    setInterval(updateRealtimeData, 1000);  // 每5秒更新一次

    // 导航切换
    document.getElementById('realtime-link').addEventListener('click', (e) => {
        e.preventDefault();
        switchPanel('realtime');
    });

    document.getElementById('history-link').addEventListener('click', (e) => {
        e.preventDefault();
        switchPanel('history');
    });

    // 历史数据查询
    document.getElementById('query-btn').addEventListener('click', fetchHistoryData);

    // DeepSeek 查询按钮点击事件
    const queryButton = document.getElementById('query-deepseek');
    if (queryButton) {
        queryButton.addEventListener('click', queryDeepseek);
        console.log('已绑定 DeepSeek 查询按钮点击事件');
    } else {
        console.error('未找到 DeepSeek 查询按钮');
    }

    // 添加回车键支持
    const inputArea = document.getElementById('deepseek-input');
    if (inputArea) {
        inputArea.addEventListener('keypress', function (e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                queryDeepseek();
            }
        });
        console.log('已绑定 DeepSeek 输入框回车事件');
    } else {
        console.error('未找到 DeepSeek 输入框');
    }
}); 