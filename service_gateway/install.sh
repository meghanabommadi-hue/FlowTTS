#!/usr/bin/env bash
# Install FlowTTS as a systemd service + nginx reverse proxy
set -euo pipefail

echo "[install] Copying systemd service..."
cp flowtts.service /etc/systemd/system/flowtts.service

echo "[install] Reloading systemd..."
systemctl daemon-reload
systemctl enable flowtts.service

echo "[install] Installing nginx config..."
cp flowtts-nginx /etc/nginx/sites-available/flowtts
ln -sf /etc/nginx/sites-available/flowtts /etc/nginx/sites-enabled/flowtts
rm -f /etc/nginx/sites-enabled/default

echo "[install] Testing nginx config..."
nginx -t

echo "[install] Reloading nginx..."
systemctl reload nginx

echo ""
echo "Done. Run these to start FlowTTS:"
echo "  systemctl start flowtts"
echo "  systemctl status flowtts"
echo ""
echo "Logs:"
echo "  journalctl -u flowtts -f"
