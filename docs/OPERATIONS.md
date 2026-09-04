# Operations runbook

## Boxes

| role | host | paths |
|---|---|---|
| orchestrator + dashboard | `root@101.53.139.186` (CPU) | `/opt/chaashini/{app,venv,data,logs,models}` |
| inference workers | `root@101.53.138.193` (GPU, shared L4) | `/opt/chaashini/{app,venv-asr,venv-enhance,models,logs}` |

Other production services run on both boxes (TTS engines on the GPU box; several dashboards on
the CPU box). Chaashini is fully self-contained under `/opt/chaashini` and only *adds* nginx
config; it never edits the other services' files except for one appended location block on the
CPU box's default port-80 server (the dashboard), following the same pattern the other
dashboards used.

## Day-to-day

```bash
# CPU box
chaashinictl status            # processes + pipeline summary
chaashinictl logs process-0    # any worker: discover-0, download-0.., process-0.., publish-0, api, supervisor
chaashinictl restart
chaashinictl nginx             # nginx -t && reload
cd /opt/chaashini/app && ../venv/bin/python -m chaashini push        # force a Hub push at the next check
cd /opt/chaashini/app && ../venv/bin/python -m chaashini add hi "https://www.youtube.com/playlist?list=..."   # queue a known-good playlist
cd /opt/chaashini/app && ../venv/bin/python -m chaashini test-lid "यह एक परीक्षण है"

# GPU box
gpuctl status
gpuctl logs asr | enhance
gpuctl restart
```

Dashboard: `http://101.53.139.186/chaashini/` (also `:8979` from networks where that port is open).

## Configuration

* `configs/chaashini.yaml` — everything tunable (languages and weights, quality thresholds, worker counts, push cadence).
* `/opt/chaashini/chaashini.env` — secrets on the CPU box: `HF_TOKEN`, `CHAASHINI_INTERNAL_TOKEN`.
* `/opt/chaashini/gpu.env` — GPU box: orchestrator URL/host header, the same internal token, model dirs.
* `configs/cookies.txt` (optional) — browser cookies for the source site; lifts bot-check limits.

Changes to YAML need `chaashinictl restart`. Changes to code: `deploy/sync.sh all` from the repo, then restart both sides.

## Scaling the source side

The source site throttles datacenter IPs. Levers, in order:
1. Keep `download_concurrency` at 4–6 and `sleep_requests` ≥ 0.5 s (defaults).
2. Export cookies from a logged-in browser to `configs/cookies.txt` (`cookies_file` in YAML). Use a
   throw-away account; never a personal one.
3. For real scale, a rotating residential proxy (`proxy:` in YAML) and/or a PO-token provider plugin
   for yt-dlp (`bgutil-ytdlp-pot-provider`) — see the yt-dlp wiki (EJS / PO Token guides).
4. Channel expansion is the biggest free lever: every well-yielding channel is crawled fully.

## Shared-GPU etiquette

The L4 also serves two production TTS engines (~10–12 GB VRAM, bursty 100 % utilisation). The
workers therefore:

* cap their own allocator (`CHAASHINI_GPU_MEM_FRACTION=0.25` for ASR+diarization, `CHAASHINI_GPU_ENH_MEM_FRACTION=0.17`
  for the enhancer) and release cached memory after every job (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`);
* run the enhancer in 8 s windows (`CHAASHINI_ENH_CHUNK_S`) and refuse sources longer than 2 h (`source.max_duration_s`),
  because diarization memory grows with file length;
* fail a job (and retry once) rather than fight for memory. If `gpuctl logs asr` shows repeated `OutOfMemoryError`,
  lower the fractions or the ASR batch (`CHAASHINI_ASR_BATCH`).

## Disk hygiene (nothing may fill the shared disks)

* per-source work dirs are deleted the moment a source finishes or fails; a 48 h TTL sweeper catches leftovers;
* staged FLACs are deleted once packed into a parquet shard; shards are deleted only after the Hub listing confirms them;
* downloads pause below `storage.min_free_gb` (40 GB) and processing pauses below half of that;
* worker logs rotate (5 × 50 MB each); watchdog `*.out` files are truncated past 200 MB; library caches older than a day are swept;
* the dashboard "Disk free" tile and the `system` events show all of the above.

## Failure modes

| symptom | where to look | fix |
|---|---|---|
| downloads all `rate limited` | dashboard source cooldown, `download-*.log` | wait (auto back-off), add cookies/proxy |
| `diarize_queued` piling up | GPU box `gpuctl status`, `asr_worker.log` | GPU worker down or orchestrator unreachable (`curl -H 'Host: chaashini-internal' http://101.53.139.186/healthz` from the GPU box) |
| push failed | `publish-0.log`, dashboard push history | token/permissions/network; shards stay `built` and retry |
| disk low | dashboard Disk tile | downloads pause automatically at `storage.min_free_gb`; clear `/opt/chaashini/data/work` of terminal videos (janitor does this) |
| dashboard 404 at /chaashini/ | `/etc/nginx/sites-enabled/vartalaap` lost the location | re-run `deploy/install_nginx.sh` on the CPU box |
