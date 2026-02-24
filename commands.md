# FlowTTS Commands

## Launch server

```bash
cd /root/FlowTTS && bash run.sh --ctrl-port 8764
```

---

## Open N ports

```bash
/root/CleanTTSData/.venv/bin/python3 -m flowtts.test.open_ports --n 40
```

- Starts from the next port after the highest already open
- To start from a specific port: `--base-port 8765`
- To open specific ports: `--ports 8900,8901,8902`

---

## Send N requests (1 per port)

```bash
cd /root/FlowTTS && /root/CleanTTSData/.venv/bin/python3 -m flowtts.test.test_pipeline --ctrl-port 8764 --requests 40
```

- Auto-discovers all open ports from the server
- Assigns exactly 1 request per port (round-robin matches when requests == ports)

---

## Open M more ports

```bash
/root/CleanTTSData/.venv/bin/python3 -m flowtts.test.open_ports --n 50
```

- Continues from the next port after the highest already open

---

## Send M+N requests (1 per port)

```bash
cd /root/FlowTTS && /root/CleanTTSData/.venv/bin/python3 -m flowtts.test.test_pipeline --ctrl-port 8764 --requests 90
```

---

## Kill server (closes all ports)

```bash
kill $(ss -tlnp | grep :8764 | grep -oP 'pid=\K[0-9]+')
```

Or by PID directly:
```bash
kill -9 <pid>
```

---

## Check open ports

```bash
ss -tlnp | grep python3 | awk '{print $4}' | sort -t: -k2 -n
```

---

## Launch server without decoder (LLM only, faster)

```bash
cd /root/FlowTTS && bash run.sh --ctrl-port 8764 --skip-decoder
```

Returns `audio_tokens` only, no WAV. Use for LLM latency benchmarking.

---

## Notes

- Server ctrl API runs on `127.0.0.1:8764`
- WS ports start at `8765` by default
- All requests run fully parallel (sglang batches LLM, decoder runs in threads)
- WAV output saved to `/root/FlowTTS/test/pipeline_test_YYYYMMDD_HHMMSS/`
- `--skip-decoder` skips ONNX/FASR decode — returns tokens only, no audio_base64
