// 全局变量
const WATER_LEVEL_THRESHOLD = 85.45;  // 水位警戒值
let charts = {
    waterLevel: null,
    temperature: null,
    humidity: null,
    windPower: null
};
let lastUpdateTime = null;

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
            scales: {
                y: {
                    beginAtZero: false
                },
                x: {
                    grid: {
                        display: false
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
            }
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

    // 使用均值±3个标准差作为范围，覆盖99.7%的数据
    const min = Math.floor(mean - 3 * stdDev);
    const max = Math.ceil(mean + 3 * stdDev);

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
        const response = await fetch('/api/realtime-data');
        const data = await response.json();

        if (Array.isArray(data) && data.length > 0) {
            // 更新卡片数据（显示最新一条）
            const latestData = data[0];
            document.getElementById('water-level').textContent = latestData.water_level.toFixed(2);
            document.getElementById('temperature').textContent = latestData.temperature.toFixed(1);
            document.getElementById('humidity').textContent = latestData.humidity.toFixed(1);
            document.getElementById('wind-power').textContent = latestData.windpower.toFixed(1);

            // 反转数组以便按时间顺序显示
            const sortedData = data.reverse();

            // 准备数据
            const timestamps = sortedData.map(item => new Date(item.timestamp).toLocaleTimeString());
            const waterLevels = sortedData.map(item => item.water_level);
            const temperatures = sortedData.map(item => item.temperature);
            const humidities = sortedData.map(item => item.humidity);
            const windPowers = sortedData.map(item => item.windpower);

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

            lastUpdateTime = new Date();
        }
    } catch (error) {
        console.error('获取实时数据失败:', error);
    }
}

// 更新单个图表
function updateChart(chart, labels, data, range) {
    chart.data.labels = labels;
    chart.data.datasets[0].data = data;

    // 更新y轴范围
    chart.options.scales.y.min = range.min;
    chart.options.scales.y.max = range.max;

    chart.update();
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
        const response = await fetch(`/api/history-data?start=${startDate}&end=${endDate}`);
        const data = await response.json();

        const tbody = document.getElementById('history-data');
        tbody.innerHTML = '';

        data.forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${new Date(row.timestamp).toLocaleString()}</td>
                <td>${row.water_level.toFixed(2)}</td>
                <td>${row.temperature.toFixed(1)}</td>
                <td>${row.humidity.toFixed(1)}</td>
                <td>${row.windpower.toFixed(1)}</td>
                <td>${row.winddirection}</td>
                <td>${row.rains.toFixed(1)}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (error) {
        console.error('获取历史数据失败:', error);
        alert('获取历史数据失败，请稍后重试');
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

// 事件监听器
document.addEventListener('DOMContentLoaded', () => {
    // 初始化图表
    initCharts();

    // 开始定时更新数据
    updateRealtimeData();
    setInterval(updateRealtimeData, 5000);  // 每5秒更新一次

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
}); 