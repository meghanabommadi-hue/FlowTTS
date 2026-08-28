#!/usr/bin/env bash
# Serve TensorBoard (and a small status page) through nginx on :80.
#
# TensorBoard binds to loopback only and is proxied, so it is never directly
# exposed. --path_prefix is required or its asset URLs break behind the proxy.
set -uo pipefail
RUN=${RUN:-/opt/omnivoice-train/run}
TB=${TB:-/opt/omnivoice-train/.venv/bin/tensorboard}
# 6006 belongs to another tenant's tensorboard on this shared box - pick the
# first free port at or above 6007 rather than fighting them for it.
PORT=${PORT:-6007}
while ss -ltn 2>/dev/null | grep -q ":$PORT "; do PORT=$((PORT+1)); done
LOGDIR="$RUN/exp/tensorboard"

mkdir -p "$LOGDIR" "$RUN/ui" "$RUN/logs"

cat > /tmp/omnivoice.nginx <<'NGINX'
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;
    client_max_body_size 64m;

    location = /healthz { return 200 "omnivoice-ok\n"; add_header Content-Type text/plain; }

    # status json + preview wavs, served straight off disk
    location /status/ {
        alias /opt/omnivoice-train/run/ui/;
        add_header Cache-Control "no-store, must-revalidate";
        autoindex on;
    }
    location /wav/ {
        alias /opt/omnivoice-train/run/eval_wav/;
        add_header Accept-Ranges bytes;
        autoindex on;
    }

    # TensorBoard needs websockets + a long read timeout for big event files
    location /tb/ {
        proxy_pass http://127.0.0.1:__TBPORT__/tb/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 300s;
        proxy_buffering off;
    }
    location = / { return 302 /tb/; }
}
NGINX
sed "s/__TBPORT__/$PORT/" /tmp/omnivoice.nginx > /etc/nginx/sites-available/omnivoice

rm -f /etc/nginx/sites-enabled/default
ln -sf /etc/nginx/sites-available/omnivoice /etc/nginx/sites-enabled/omnivoice
nginx -t || { echo "nginx config invalid"; exit 1; }
(nginx -s reload 2>/dev/null) || nginx
echo "nginx serving :80 -> /tb/"

if pgrep -f "tensorboard.*--port $PORT " >/dev/null; then
  echo "tensorboard already running"
else
  setsid "$TB" --logdir "$LOGDIR" --host 127.0.0.1 --port "$PORT" \
      --path_prefix /tb --reload_interval 30 --samples_per_plugin "audio=200,scalars=2000" \
      >> "$RUN/logs/tensorboard.log" 2>&1 < /dev/null &
  sleep 4
  echo "tensorboard started on 127.0.0.1:$PORT (logdir=$LOGDIR)"
  sleep 3
fi
curl -s -o /dev/null -w "  local /tb/ -> %{http_code}\n" http://127.0.0.1/tb/
