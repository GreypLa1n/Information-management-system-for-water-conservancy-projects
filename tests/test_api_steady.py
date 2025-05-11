
# from database import get_data_for_visualization

# def test_data_loading():
#     data = get_data_for_visualization()
#     assert data is not None
#     assert isinstance(data, tuple)
#     assert len(data) == 5


import requests
import time

base_url = "http://localhost:5000"

# 1. 登录（模拟时间戳）
timestamp = int(time.time() * 1000)
login_resp = requests.post(f"{base_url}/api/login", json={
    "username": "admin",
    "password": "admin",
    "timestamp": timestamp
})

assert login_resp.status_code == 200
cookies = login_resp.cookies
print("登录响应：", login_resp.json())

# 2. 获取实时数据（使用登录后的 session cookie）
realtime_resp = requests.get(f"{base_url}/api/realtime-data", cookies=cookies)
assert realtime_resp.status_code == 200
print("实时数据：", realtime_resp.json())

#  3. 获取历史数据（设定时间范围）
history_resp = requests.get(
    f"{base_url}/api/history-data",
    params={"start": "2025-01-01", "end": "2025-05-11"},
    cookies=cookies
)
print("历史数据：", history_resp.json())
