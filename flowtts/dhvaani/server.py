"""Pipeline position: ENTRYPOINT — one process, one model, three servers.

Role in pipeline:
  Boots the whole DhVaani stack in a single asyncio loop:

      DhvaaniEngine        model load + warmup, owns the GPU
      Control API          aiohttp, legacy ops routes            (--ctrl-port)
      WebSocket gateway    legacy FlowTTS wire protocol          (--ports)
      FastAPI + uvicorn    OpenAI-compatible REST + voice CRUD   (--http-port)

Boot order matters: the engine is fully loaded and warmed BEFORE any port is
bound, so a health check can never pass while the first request would still pay
a cold-start cost.

Usage:
    python -m flowtts.dhvaani.server --ports 4 --profile balanced
    python -m flowtts.dhvaani.server --ports 1 --backend trt --profile fast
    ./run_dhvaani.sh --ports 4 --ctrl-port 8764
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

import structlog

from flowtts.dhvaani.config import PROFILES, apply_profile, dhv_settings

logger = structlog.get_logger(__name__)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        format="%(message)s", stream=sys.stdout, level=getattr(logging, level.upper(), 20)
    )
    for noisy in ("websockets", "aiohttp.access", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty()),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), 20)
        ),
        cache_logger_on_first_use=True,
    )


def build_parser() -> argparse.ArgumentParser:
    s = dhv_settings
    p = argparse.ArgumentParser(description="DhVaani TTS server")
    p.add_argument("--ports", type=int, default=s.server.ws_num_ports,
                   help="Number of WebSocket ports to bind")
    p.add_argument("--base-port", type=int, default=s.server.ws_base_port)
    p.add_argument("--http-port", type=int, default=s.server.http_port,
                   help="REST API port (0 disables)")
    p.add_argument("--ctrl-port", type=int, default=s.server.ctrl_port,
                   help="Control API port (0 disables)")
    p.add_argument("--profile", choices=sorted(PROFILES), default=None,
                   help="fast | balanced | quality (see config.PROFILES)")
    p.add_argument("--backend", choices=["torch", "trt", "triton"], default=None)
    p.add_argument("--voices-dir", default=None)
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p


async def run(args) -> int:
    from flowtts.dhvaani.api.app import create_app
    from flowtts.dhvaani.api.control import start_control_api
    from flowtts.dhvaani.api.ws import WebSocketGateway
    from flowtts.dhvaani.engine.engine import DhvaaniEngine

    s = dhv_settings
    if args.profile:
        apply_profile(s, args.profile)
    if args.backend:
        s.backend.kind = args.backend
    if args.voices_dir:
        s.voice.store_dir = args.voices_dir
    if args.no_warmup:
        s.server.warmup_enabled = False

    engine = DhvaaniEngine(s)
    await engine.start()

    gateway = WebSocketGateway(engine, s)
    ctrl_runner = None
    if args.ctrl_port:
        ctrl_runner = await start_control_api(engine, gateway, s.server.host, args.ctrl_port)

    ports = [args.base_port + i for i in range(max(0, args.ports))]
    if ports:
        await gateway.serve(ports)

    http_task = None
    if args.http_port:
        import uvicorn

        app = create_app(engine, s)
        config = uvicorn.Config(
            app, host=s.server.host, port=args.http_port,
            log_level=args.log_level.lower(), access_log=False, loop="none",
        )
        # uvicorn.Server.serve() rather than uvicorn.run(): the latter creates
        # its own event loop, and the engine already owns this one.
        http_task = asyncio.create_task(uvicorn.Server(config).serve(), name="dhvaani-http")

    _banner(s, ports, args)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # pragma: no cover - Windows
            pass

    exit_code = 0
    watchdog = engine._watchdog
    try:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
            # Two OOMs inside the recovery window mean soft recovery is not
            # working. Exit non-zero so the supervisor (run_dhvaani.sh) restarts
            # a clean process rather than limping along.
            if watchdog is not None and watchdog.restart_requested:
                logger.error("restarting_after_repeated_oom")
                exit_code = 1
                break
    finally:
        logger.info("shutting_down")
        if http_task is not None:
            http_task.cancel()
        if ctrl_runner is not None:
            await ctrl_runner.cleanup()
        await engine.stop()
    return exit_code


def _banner(s, ports, args) -> None:
    lines = ["", "=" * 66, "  DhVaani TTS  --  ARTPARK-IISc/DhVaani-0.5 (ZipVoice flow-matching)", "=" * 66]
    lines.append(f"  profile         : {args.profile or 'config defaults'}")
    lines.append(f"  backend         : {s.backend.kind}")
    lines.append(f"  num_step / CFG  : {s.flow.num_step} / {s.flow.guidance_scale}")
    lines.append(f"  output rate     : {s.audio.output_sample_rate} Hz")
    if ports:
        lines.append(f"  websocket       : ws://{s.server.host}:{ports[0]}"
                     + (f" .. :{ports[-1]}" if len(ports) > 1 else ""))
    if args.http_port:
        lines.append(f"  rest api        : http://{s.server.host}:{args.http_port}/docs")
        lines.append(f"  openai speech   : POST http://{s.server.host}:{args.http_port}/v1/audio/speech")
    if args.ctrl_port:
        lines.append(f"  control api     : http://{s.server.host}:{args.ctrl_port}/ready")
    lines.append("=" * 66)
    lines.append("")
    print("\n".join(lines), flush=True)


def main() -> None:
    args = build_parser().parse_args()
    _configure_logging(args.log_level)
    try:
        code = asyncio.run(run(args))
    except KeyboardInterrupt:
        code = 0
        print("\n[DhVaani] stopped.", flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
