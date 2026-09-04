#!/usr/bin/env bash
# Install Chaashini's nginx pieces on the CPU box (idempotent, validated, graceful reload).
#  1. sites-enabled/chaashini-internal   (own file; Host-routed internal API for the GPU workers)
#  2. sites-enabled/chaashini            (own file; standalone dashboard listener on :8979)
#  3. one appended `location ^~ /chaashini/` block inside the existing port-80 default server
#     (the only shared file we touch; same pattern the other dashboards on this box use)
set -euo pipefail
APP=/opt/chaashini/app
SE=/etc/nginx/sites-enabled
if ! command -v nginx >/dev/null; then
  echo "nginx missing: installing (additive system package)"; DEBIAN_FRONTEND=noninteractive apt-get install -y -q nginx >/dev/null
fi
mkdir -p $SE /etc/nginx/chaashini-extra
# find an existing default server owned by someone else; the stock placeholder site does not count
DEFAULT_SITE=${DEFAULT_SITE:-}
if [ -z "$DEFAULT_SITE" ]; then
  for f in $SE/*; do [ -f "$f" ] || continue; case "$(basename "$f")" in default|chaashini*) continue;; esac; grep -q "default_server" "$f" && DEFAULT_SITE=$f && break; done
fi
if [ -z "$DEFAULT_SITE" ]; then
  # nobody owns port 80: the dashboard becomes the default server (stock placeholder removed)
  rm -f $SE/default
  cp $APP/deploy/nginx/chaashini-default.conf $SE/chaashini-default
  echo "installed chaashini-default (no other default server on this box)"
fi
cp $APP/deploy/nginx/chaashini-internal.conf $SE/chaashini-internal
cp $APP/deploy/nginx/chaashini-standalone.conf $SE/chaashini
if [ -n "$DEFAULT_SITE" ] && [ -f "$DEFAULT_SITE" ] && ! grep -q "location ^~ /chaashini/" "$DEFAULT_SITE"; then
  cp "$DEFAULT_SITE" "/root/$(basename "$DEFAULT_SITE").bak.chaashini.$(date +%s)"   # backups must NOT live in sites-enabled
  # insert the location block before the final closing brace of the last server block
  python3 - "$DEFAULT_SITE" "$APP/deploy/nginx/chaashini-dashboard.location" <<'PY'
import sys
site, snippet = sys.argv[1], sys.argv[2]
s = open(site).read().rstrip()
assert s.endswith("}"), "unexpected site file ending"
block = open(snippet).read().strip()
out = s[:-1].rstrip() + "\n\n" + "\n".join("    " + l if l else "" for l in block.splitlines()) + "\n}\n"
open(site, "w").write(out)
print("dashboard location appended to", site)
PY
else
  echo "dashboard location already present (or default site missing)"
fi
nginx -t
if pgrep -x nginx >/dev/null; then nginx -s reload && echo "nginx reloaded"; else (nginx && echo "nginx started") || service nginx start; fi
curl -s -o /dev/null -w "internal healthz via Host header: %{http_code}\n" -H "Host: chaashini-internal" http://127.0.0.1/healthz || true
curl -s -o /dev/null -w "dashboard /chaashini/: %{http_code}\n" http://127.0.0.1/chaashini/ || true
