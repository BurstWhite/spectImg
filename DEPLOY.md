# 部署说明（VPS）

本项目分两部分：`server/`（FastAPI，包装 img2spec.py）和 `web/`（React + Vite 前端）。
后端无状态、不落盘、不缓存；转换结果只以 blob URL 存在于浏览器内存。

## 1. 后端

```bash
# 建议独立虚拟环境
python3 -m venv /opt/spectimg-venv
/opt/spectimg-venv/bin/pip install -r server/requirements.txt

# 生产运行（放在 repo 根目录下启动，或用 --app-dir 指定）
cd /path/to/spectImg
/opt/spectimg-venv/bin/uvicorn server.main:app --host 127.0.0.1 --port 8000 --workers 2
```

必须设置的环境变量：

| 变量 | 说明 |
|---|---|
| `TURNSTILE_SECRET_KEY` | Cloudflare Turnstile 的 Secret Key。**不设置时使用官方 always-pass 测试密钥（等于关闭校验），生产必须设置** |

注意：Griffin-Lim 是 CPU 密集型（一次转换数秒到十几秒），`--workers` 按核数给，
不要给太多以免同时多路转换把 CPU 打满。

systemd 单元示例（`/etc/systemd/system/img2spec.service`）：

```ini
[Unit]
Description=img2spec API
After=network.target

[Service]
WorkingDirectory=/opt/spectimg
Environment=TURNSTILE_SECRET_KEY=0x4AAAAAAA...你的secret...
ExecStart=/opt/spectimg-venv/bin/uvicorn server.main:app --host 127.0.0.1 --port 8000 --workers 2
Restart=always

[Install]
WantedBy=multi-user.target
```

## 2. 前端

```bash
cd web
bun install
# 生产构建前把 web/.env 里的测试 site key 换成真实值
bun run build        # 产物在 web/dist/
```

`web/.env` 中的 `VITE_TURNSTILE_SITE_KEY` 是 Cloudflare 的 **Site Key**（公开的），
当前默认是官方 always-pass 测试密钥，上线前必须替换。
Site Key 和 Secret Key 在 Cloudflare Dashboard → Turnstile 里同一个小组件的详情页。

`web/dist/` 是纯静态文件，用任意静态服务器（nginx / caddy）托管即可。
注意 Vite 构建时把 `VITE_TURNSTILE_SITE_KEY` 内联进了 JS——**改 key 必须重新 build**。

## 3. nginx 反代示例

前端和 API 同域可以完全避开 CORS（后端虽已开 CORS，但同域更省事）：

```nginx
server {
    listen 443 ssl http2;
    server_name your.domain;

    ssl_certificate     /etc/letsencrypt/live/your.domain/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your.domain/privkey.pem;

    root /opt/spectimg/web/dist;
    index index.html;

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        # 转换耗时较长，放宽超时
        proxy_read_timeout 120s;
        client_max_body_size 16m;
    }

    location / {
        try_files $uri /index.html;
    }
}
```

## 4. Cloudflare Turnstile 配置

1. Cloudflare Dashboard → Turnstile → Add site，域名填你的域名，得到 Site Key + Secret Key
2. Site Key → `web/.env` 的 `VITE_TURNSTILE_SITE_KEY`，重新 `bun run build`
3. Secret Key → 后端环境变量 `TURNSTILE_SECRET_KEY`，重启服务

域名必须开启 HTTPS（Turnstile 的强制要求；经 Cloudflare 代理自动满足）。

## 5. 本地开发

```bash
# 后端（测试密钥下 dummy token 会通过 siteverify）
uvicorn server.main:app --port 8000

# 前端（vite 已配置 /api 代理到 localhost:8000）
cd web && bun run dev
```
