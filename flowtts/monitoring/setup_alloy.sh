#!/bin/bash
# ============================================================
# FlowTTS → Grafana Cloud monitoring setup via Grafana Alloy
# Run this script on each GPU VM (or bake into the VM image)
#
# BEFORE RUNNING: fill in your Grafana Cloud credentials below
#
# USER_ID:  grafana.com → My Account → your stack → Details
#           (numeric ID shown next to each endpoint)
#
# API_KEY:  meghanabommadi.grafana.net → Connections →
#           Add new connection → Linux node
#           (pre-filled token already scoped to your region)
# ============================================================

LOKI_URL="https://logs-prod-028.grafana.net/loki/api/v1/push"
PROM_URL="https://prometheus-prod-43-prod-ap-south-1.grafana.net/api/prom/push"
LOKI_USER_ID="1500272"
LOKI_API_KEY="glc_eyJvIjoiMTY4NDA3NyIsIm4iOiJzdGFjay0xNTQyNDgzLWhsLXJlYWQtYWxsb3kiLCJrIjoiM3ozTFE3Nzk5N1lOazRRRWczY2FEZzlYIiwibSI6eyJyIjoicHJvZC1hcC1zb3V0aC0xIn19"
PROM_USER_ID="3009194"
PROM_API_KEY="glc_eyJvIjoiMTY4NDA3NyIsIm4iOiJzdGFjay0xNTQyNDgzLWhtLXdyaXRlLXByb21fdHRzIiwiayI6IjBnNVhNUDNRMzJjUjd5THV4MTMzY1BLOSIsIm0iOnsiciI6InByb2QtYXAtc291dGgtMSJ9fQ=="

# ============================================================
# DO NOT EDIT BELOW THIS LINE
# ============================================================

set -e

# Verify credentials are filled in
if [[ "$LOKI_API_KEY" == "<YOUR_API_KEY>" || "$PROM_API_KEY" == "<YOUR_API_KEY>" ]]; then
    echo "ERROR: Please fill in API keys before running this script."
    exit 1
fi

echo "[1/6] Installing Grafana Alloy..."
# Grafana APT repo — uncomment if not already added:
# sudo mkdir -p /etc/apt/keyrings
# wget -q -O - https://apt.grafana.com/gpg.key | gpg --dearmor | sudo tee /etc/apt/keyrings/grafana.gpg > /dev/null
# echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main" | sudo tee /etc/apt/sources.list.d/grafana.list
# sudo apt-get update -qq
sudo apt-get install -y alloy

echo "[2/6] Adding alloy user to required groups..."
sudo usermod -aG systemd-journal alloy   # read journald
sudo usermod -aG ubuntu alloy            # read JSONL files

echo "[3/6] Setting JSONL file permissions..."
sudo chmod -R g+r /home/ubuntu/FlowTTS/flowtts/monitoring/ 2>/dev/null || true

echo "[4/6] Writing Alloy config..."
sudo tee /etc/alloy/config.alloy > /dev/null <<ALLOY_EOF
// ── Logs: journald → Grafana Cloud Loki ──────────────────────────────────
loki.source.journal "flowtts" {
  max_age       = "12h"
  relabel_rules = loki.relabel.flowtts_filter.rules
  forward_to    = [loki.write.cloud.receiver]
  labels        = {job = "flowtts"}
}

loki.relabel "flowtts_filter" {
  rule {
    source_labels = ["__journal__systemd_unit"]
    regex         = "flowtts(@\\\\w+)?\\\\.service"
    action        = "keep"
  }
  rule {
    source_labels = ["__journal__systemd_unit"]
    target_label  = "unit"
  }
  forward_to = []
}

// ── Logs: calls.jsonl → Grafana Cloud Loki ───────────────────────────────
loki.source.file "flowtts_calls" {
  targets = [{
    __path__ = "/home/ubuntu/FlowTTS/flowtts/monitoring/calls.jsonl",
    job      = "flowtts_calls",
  }]
  forward_to = [loki.process.parse_calls.receiver]
}

loki.process "parse_calls" {
  stage.json {
    expressions = {port = "port", call_id = "call_id"}
  }
  stage.labels {
    values = {port = "port"}
  }
  forward_to = [loki.write.cloud.receiver]
}

// ── Logs: llm_outputs.jsonl → Grafana Cloud Loki ─────────────────────────
loki.source.file "flowtts_llm" {
  targets = [{
    __path__ = "/home/ubuntu/FlowTTS/flowtts/monitoring/llm_outputs.jsonl",
    job      = "flowtts_llm",
  }]
  forward_to = [loki.process.parse_llm.receiver]
}

loki.process "parse_llm" {
  stage.json {
    expressions = {port = "port", call_id = "call_id"}
  }
  stage.labels {
    values = {port = "port"}
  }
  forward_to = [loki.write.cloud.receiver]
}

// ── Metrics: FlowTTS /metrics → Grafana Cloud Prometheus ─────────────────
prometheus.scrape "flowtts" {
  targets         = [{"__address__" = "localhost:8764"}]
  forward_to      = [prometheus.remote_write.cloud.receiver]
  scrape_interval = "15s"
}

// ── Loki write endpoint (Grafana Cloud) ──────────────────────────────────
loki.write "cloud" {
  endpoint {
    url = "${LOKI_URL}"
    basic_auth {
      username = "${LOKI_USER_ID}"
      password = "${LOKI_API_KEY}"
    }
  }
  external_labels = {host = sys.env("HOSTNAME")}
}

// ── Prometheus remote_write endpoint (Grafana Cloud) ─────────────────────
prometheus.remote_write "cloud" {
  endpoint {
    url = "${PROM_URL}"
    basic_auth {
      username = "${PROM_USER_ID}"
      password = "${PROM_API_KEY}"
    }
  }
}
ALLOY_EOF

echo "[5/6] Enabling and starting Alloy..."
sudo systemctl daemon-reload
sudo systemctl enable alloy
sudo systemctl restart alloy

echo "[6/6] Verifying Alloy started..."
sleep 5
if sudo systemctl is-active --quiet alloy; then
    echo "Alloy is running."
    echo ""
    echo "Checking for errors in first 10 seconds..."
    sudo journalctl -u alloy --since "10 sec ago" --no-pager | grep -i "error" || echo "No errors found."
else
    echo "ERROR: Alloy failed to start. Check logs:"
    sudo journalctl -u alloy --no-pager -n 20
    exit 1
fi

echo ""
echo "======================================================"
echo " Done! Alloy is running and enabled on boot."
echo ""
echo " Live logs:   sudo journalctl -u alloy -f"
echo " Check ship:  curl -s http://localhost:12345/metrics | grep loki_write_sent_entries_total"
echo ""
echo " In Grafana Cloud → Explore → Loki, run:"
echo '   {job="flowtts"}'
echo "======================================================"
