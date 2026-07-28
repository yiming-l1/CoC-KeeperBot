# ================================
# 0. 基础导入
# ================================
import os
import sys
import time
import socket

# ================================
# 1. 代理设置（Mihomo / Clash）
# ================================
HTTP_PROXY  = "http://127.0.0.1:7890"
HTTPS_PROXY = "http://127.0.0.1:7890"
NO_PROXY    = "127.0.0.1,localhost"

os.environ["HTTP_PROXY"]  = HTTP_PROXY
os.environ["HTTPS_PROXY"] = HTTPS_PROXY
os.environ["NO_PROXY"]    = NO_PROXY

# 某些库会读小写变量（保险）
os.environ["http_proxy"]  = HTTP_PROXY
os.environ["https_proxy"] = HTTPS_PROXY
os.environ["no_proxy"]    = NO_PROXY

print("✅ Proxy env configured")

# ================================
# 2. DNS / 网络超时保护（避免卡死）
# ================================
socket.setdefaulttimeout(20)

# ================================
# 3. 快速连通性测试（Gemini / Google APIs）
# ================================
def quick_test(url: str):
    try:
        import requests
        r = requests.get(url, timeout=10)
        return r.status_code
    except Exception as e:
        return f"ERROR: {e}"

test_urls = {
    "Google APIs": "https://generativelanguage.googleapis.com/",
    "AI Studio":   "https://ai.google.dev/",
    "GitHub":      "https://github.com/",
    "HF":          "https://huggingface.co/"
}

for name, url in test_urls.items():
    res = quick_test(url)
    print(f"{name:<12}: {res}")

# 说明：
# - 401 / 403 = 成功（API 需要鉴权）
# - 200       = 成功
# - timeout   = 没走代理 or 节点不通
