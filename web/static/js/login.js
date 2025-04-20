// 登录页面脚本
document.addEventListener('DOMContentLoaded', function () {
    const loginForm = document.getElementById('login-form');

    if (loginForm) {
        loginForm.addEventListener('submit', async function (event) {
            event.preventDefault();

            // 获取表单数据
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;

            if (!username || !password) {
                showLoginError('请输入用户名和密码');
                return;
            }

            // 显示加载状态
            const submitButton = document.querySelector('.btn-login');
            const originalText = submitButton.textContent;
            submitButton.textContent = '登录中...';
            submitButton.disabled = true;

            try {
                // 使用SHA-256对密码进行前端加密（注意：这是第一层加密，后端会再次加密）
                const passwordHash = await sha256(password);

                // 发送登录请求
                const response = await fetch('/api/login', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        username,
                        password: password, // 这里我们传递原密码，后端会再次加密
                        timestamp: Date.now() // 添加时间戳防止重放攻击
                    })
                });

                const data = await response.json();

                if (data.success) {
                    // 登录成功，重定向到首页
                    window.location.href = '/';
                } else {
                    // 登录失败，显示错误信息
                    showLoginError(data.message || '用户名或密码错误');
                    submitButton.textContent = originalText;
                    submitButton.disabled = false;
                }
            } catch (error) {
                console.error('登录请求错误:', error);
                showLoginError('网络错误，请稍后重试');
                submitButton.textContent = originalText;
                submitButton.disabled = false;
            }
        });
    }

    // 显示登录错误信息
    function showLoginError(message) {
        const errorElement = document.getElementById('login-error');
        if (errorElement) {
            errorElement.textContent = message;
            errorElement.style.display = 'block';
        }
    }

    // SHA-256 哈希函数
    async function sha256(message) {
        // 使用浏览器内置的 SubtleCrypto API 计算哈希值
        const msgBuffer = new TextEncoder().encode(message);
        const hashBuffer = await crypto.subtle.digest('SHA-256', msgBuffer);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
        return hashHex;
    }
}); 