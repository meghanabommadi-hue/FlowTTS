# FlowTTS

Multi-backend TTS gateway. `flowtts/server.py` (via `./run.sh`) is the
recommended production entry point — one process, one WebSocket layer, one
`model_type` selected at a time via `FLOWTTS_MODEL_TYPE`.

See `flowtts/synthesis/base.py` for the plugin interface every backend
implements (`BaseSynthesizer`), and `flowtts/synthesis/engine.py` for the
model registry.

## Model dependency matrix

Each backend has its own Python-version/package story, because each was
integrated from a different upstream project with its own pins. **A backend's
dependencies are never installed into FlowTTS's main venv if they conflict
with what another backend already needs there** — that's why several backends
run as separate child processes proxied over HTTP instead of importing
in-process.

| `model_type` | Runs | Venv | Setup script | Requirements file | Why separate |
|---|---|---|---|---|---|
| `mira` | In-process (sglang engine + ncodec) | `.venv_mira` | *(part of original setup)* | `requirements-mira.txt` | Original/default path; FlowTTS's Mira-specific pins live here (`transformers==4.57.3`, `sglang`, `flashinfer`, etc.) |
| `voxcpm` | In-process (`nanovllm_voxcpm` diffusion engine) | external `flow_voxcpm` checkout's own venv (path hardcoded in `synthesis/voxcpm.py:21`, not present on every host) | — | `requirements-voxcpm.txt` (placeholder — see file) | Diffusion (Continuous Flow Matching) engine with its own dependency set; not verified installable alongside Mira's pins in this environment |
| `omnivoice` | Child process, HTTP-proxied | `~/omnivoice_scaled/.venv` | `./setup_omni.sh` | `requirements-omnivoice.txt` | OmniVoice needs `transformers>=5.3.0`, which directly conflicts with the `transformers==4.57.3` pin Mira/sglang need in FlowTTS's own venv |
| `miotts` | Two child processes (vLLM server + codec server), both HTTP-proxied | `.venv_mio` | `./flowtts/setup/setup_mio.sh` | `requirements-miotts.txt` | See below |

### Why miotts needs its own venv (but only one, not two)

miotts's **own** checkout (`~/miotts`) splits into two venvs:
- `.venv_vllm` (Python 3.10) — `vllm==0.8.5` + `transformers==4.51.3`
- `.venv` (Python 3.12) — `miocodec` (the audio codec)

This split exists because **miocodec's package metadata declares
`Requires-Python >=3.12`**, which is simply incompatible with `.venv_vllm`'s
3.10 interpreter in that checkout — not because vllm and miocodec's actual
package requirements conflict with each other.

FlowTTS verified this directly rather than assuming the split was load-bearing:
`vllm==0.8.5`, `transformers==4.51.3`, `miocodec` (from git), and `torch==2.6.0`
all installed into **one** Python 3.12 venv (`.venv_mio`) with no pip resolver
conflicts, no flashinfer pulled in (matching miotts's own note that flashinfer
should be excluded — ABI mismatch), and all four imported together in the same
process successfully (`torch.cuda.is_available()` returns `True`, `vllm`,
`transformers`, and `miocodec` all load with matching pinned versions).

So `model_type=miotts` uses exactly one venv (`.venv_mio`, via `flowtts/setup/setup_mio.sh`).
`synthesis/miotts.py` still launches **two separate child processes** from
that one interpreter — a vLLM server and a codec server — matching miotts's
own `run.sh` design (independent GPU residency, restart isolation, and
avoiding a ~3s codec reload per request), not because of any remaining
Python-version constraint.

### Adding a new model

1. Create `flowtts/synthesis/<name>.py` → `class <Name>Synthesizer(BaseSynthesizer)`.
2. Register it in `flowtts/synthesis/engine.py`'s `_REGISTRY`.
3. Add its settings block to `flowtts/core/config.py` (`Settings.model_type` Literal + a new `<Name>Settings` field).
4. If it uses the generic (non-Mira) streaming path, add its `model_type` string to `flowtts/server.py`'s `_GENERIC_SYNTHESIZER_MODEL_TYPES` frozenset.
5. If its dependencies conflict with FlowTTS's main venv or another backend's venv, give it its own venv + `setup_<name>.sh` + `requirements-<name>.txt`, and spawn it as a child process proxied over HTTP (see `synthesis/omnivoice.py` or `synthesis/miotts.py` for the pattern) rather than importing it directly.

`server.py` itself needs zero changes beyond step 4 — everything else
(`handle_connection`, WAV caching, OOM recovery, the control API, metrics)
dispatches generically through `BaseSynthesizer.synthesize()` /
`synthesize_stream()` / `.sample_rate`.

## miotts backend specifics

`model_type=miotts` wraps [SPRINGLab/Indic-Mio](https://huggingface.co/SPRINGLab/Indic-Mio)
(LLM) + MioCodec (audio decoder) — see `~/miotts` for the standalone project
this backend proxies to.

**No true audio streaming.** MioCodec's `decode()` has no incremental/causal
mode — `forward_wave()` sizes every interpolation/upsampling step off the
*complete* target audio length, computed from the *entire* token sequence, up
front. Decoding a token prefix produces measurably different audio for the
same tokens (an earlier test found max abs waveform diff ~0.66 on a ~[-1,1]
scale), not just a truncated version of the final result — so it's not safe
to chunk. `MiottsSynthesizer.synthesize_stream()` therefore yields exactly one
final `SynthChunk`, same as `OmniVoiceSynthesizer`.

**Post-decode smoothing is on by default and scoped to this backend only.**
`synthesis/miotts.py` loads `~/miotts/miotts/postprocess.py`'s
`smooth_glitches()` directly by file path (via `importlib`, bypassing the
`miotts` package's own `__init__.py`, which otherwise pulls in
torch/transformers into FlowTTS's process) and runs it on the decoded
waveform before returning. It crossfades isolated codec glitches — sparse,
randomly-placed sharp transients from occasional outlier speech tokens — via
a short linear-ramp crossfade around each detected discontinuity. Disable per
deployment with `FLOWTTS_MIOTTS__SMOOTH_GLITCHES=false`.

Setup:
```bash
cd ~/FlowTTS
./flowtts/setup/setup_mio.sh        # creates .venv_mio, installs requirements-miotts.txt
FLOWTTS_MODEL_TYPE=miotts ./run.sh
```
