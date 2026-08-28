#!/usr/bin/env python3
"""Training entry point with eval-time audio previews and HF checkpoint push.

OmniTrainer has no callback system (it is a hand-written loop), so the hooks are
added by subclassing `evaluate()` rather than patching upstream source. Extra
settings arrive through environment variables on purpose: TrainingConfig.from_json
silently DROPS unknown keys, so putting them in the train config JSON would make
them vanish without warning.

Env:
  OHUN_EVAL_SET          json list of eval prompts (id/text/language[/ref_audio])
  OHUN_EVAL_INFER_EVERY  run audio preview every N evals (default 1, 0=off)
  OHUN_EVAL_WAV_DIR      also write preview wavs here
  OHUN_INFER_TIMEOUT     seconds before the preview subprocess is killed (900)
  OHUN_HF_REPO           repo id to push improved checkpoints to (optional)
  OHUN_HF_TOKEN          write token for that repo
  OHUN_HF_MIN_DELTA      min eval-loss improvement to trigger a push (0.002)
  OHUN_HF_MIN_STEP       don't push before this step (default 0)
"""
from __future__ import annotations

import argparse, json, os, shutil, subprocess, sys, threading, time

import torch

from omnivoice.training.builder import build_dataloaders, build_model_and_tokenizer
from omnivoice.training.config import TrainingConfig
from omnivoice.training.trainer import OmniTrainer

HERE = os.path.dirname(os.path.abspath(__file__))


def _env(k, d=None):
    v = os.environ.get(k)
    return v if v not in (None, "") else d


class HookedTrainer(OmniTrainer):
    """OmniTrainer + audio previews to TensorBoard + best-checkpoint HF push."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.best_eval = float("inf")
        self.best_composite = -1.0
        self.n_evals = 0
        self.eval_set = _env("OHUN_EVAL_SET")
        self.infer_every = int(_env("OHUN_EVAL_INFER_EVERY", "1"))
        self.wav_dir = _env("OHUN_EVAL_WAV_DIR")
        self.infer_timeout = int(_env("OHUN_INFER_TIMEOUT", "900"))
        self.hf_repo = _env("OHUN_HF_REPO")
        self.hf_token = _env("OHUN_HF_TOKEN")
        self.hf_min_delta = float(_env("OHUN_HF_MIN_DELTA", "0.002"))
        self.hf_min_step = int(_env("OHUN_HF_MIN_STEP", "0"))
        self.asr_url = _env("OHUN_ASR_URL")          # enables eval WER/CER
        self.asr_model = _env("OHUN_ASR_MODEL", "Axiveri/NaijaVox-2.0")
        self._push_thread = None
        # Persist the best score so a supervisor restart doesn't re-push a
        # checkpoint that was already the best.
        self._state_p = os.path.join(self.config.output_dir, "hook_state.json")
        try:
            _s = json.load(open(self._state_p))
            self.best_eval = _s.get("best_eval", float("inf"))
            self.best_composite = _s.get("best_composite", -1.0)
            print(f"[hook] resumed best_eval={self.best_eval:.4f} "
                  f"best_composite={self.best_composite:.4f}", flush=True)
        except Exception:
            pass

    # ---- helpers ---------------------------------------------------------
    def _is_main(self):
        return self.accelerator.is_main_process

    def _tb_logdir(self):
        return os.path.join(self.config.output_dir, "tensorboard")

    def _save_state(self):
        try:
            with open(self._state_p, "w") as f:
                json.dump({"best_eval": self.best_eval,
                           "best_composite": self.best_composite,
                           "step": self.global_step}, f)
        except OSError:
            pass

    def _snapshot(self, step=None):
        """Write an inference-loadable copy of the current weights.

        Only save_pretrained + tokenizer - no optimizer state. Written to a temp
        dir and renamed so a crash mid-write can never leave a half-snapshot
        that from_pretrained would choke on.
        """
        step = self.global_step if step is None else step
        dst = os.path.join(self.config.output_dir, f"eval_snapshot-{step}")
        tmp = dst + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp, exist_ok=True)
        model = self.accelerator.unwrap_model(self.model)
        model.save_pretrained(tmp, safe_serialization=True)
        if self.tokenizer is not None:
            try:
                self.tokenizer.save_pretrained(tmp)
            except Exception as e:
                print(f"[hook] tokenizer save skipped: {e!r}", flush=True)
        shutil.rmtree(dst, ignore_errors=True)
        os.rename(tmp, dst)
        # retire older snapshots, but never the two most recent (one may still
        # be uploading in the push thread)
        try:
            snaps = sorted(
                (d for d in os.listdir(self.config.output_dir)
                 if d.startswith("eval_snapshot-") and not d.endswith(".tmp")),
                key=lambda d: int(d.rsplit("-", 1)[1]))
            for old in snaps[:-2]:
                shutil.rmtree(os.path.join(self.config.output_dir, old),
                              ignore_errors=True)
        except (OSError, ValueError):
            pass
        return dst

    def _audio_preview(self, snap):
        """Synthesize the eval set out-of-process and log audio to TensorBoard."""
        cmd = [sys.executable, os.path.join(HERE, "eval_infer.py"),
               "--checkpoint", snap, "--eval-set", self.eval_set,
               "--logdir", self._tb_logdir(), "--step", str(self.global_step)]
        if self.wav_dir:
            cmd += ["--wav-out", self.wav_dir]
        if self.asr_url:
            cmd += ["--asr-url", self.asr_url, "--asr-model", self.asr_model,
                    "--metrics-out",
                    os.path.join(self.config.output_dir, "last_eval_metrics.json")]
        t0 = time.time()
        try:
            r = subprocess.run(cmd, timeout=self.infer_timeout,
                               capture_output=True, text=True)
            tail = (r.stdout or "").strip().splitlines()[-3:]
            print(f"[hook] audio preview rc={r.returncode} "
                  f"({time.time()-t0:.0f}s): {' | '.join(tail)}", flush=True)
            if r.returncode != 0:
                print(f"[hook] preview stderr: {(r.stderr or '')[-600:]}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"[hook] audio preview TIMED OUT after {self.infer_timeout}s",
                  flush=True)
        except Exception as e:
            print(f"[hook] audio preview failed: {e!r}", flush=True)

    def _push(self, snap, score, step, metric="composite"):
        """Upload a snapshot to the Hub. Runs in a thread so training continues."""
        def work():
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=self.hf_token)
                api.create_repo(self.hf_repo, repo_type="model",
                                private=True, exist_ok=True)
                arrow = "higher is better" if metric == "composite" else "lower is better"
                readme = (f"# OmniNaija - OmniVoice fine-tuned for Nigerian languages\n\n"
                          f"Auto-pushed at step **{step}** with **{metric} "
                          f"{score:.4f}** ({arrow}).\n\n"
                          f"`composite` is a weighted tts-bench score: "
                          f"0.40*(1-WER) + 0.20*(1-CER) + 0.15*(1-SER) + "
                          f"0.25*(MOS-1)/4, computed on a held-out eval set.\n\n"
                          f"Languages: Igbo (ig), Yoruba (yo), Hausa (ha), "
                          f"Nigerian Pidgin (pcm).\n\n"
                          f"Base: `{self.config.llm_name_or_path}`, "
                          f"init from `{self.config.init_from_checkpoint}`.\n")
                with open(os.path.join(snap, "README.md"), "w") as f:
                    f.write(readme)
                api.upload_folder(folder_path=snap, repo_id=self.hf_repo,
                                  repo_type="model",
                                  commit_message=f"step {step} {metric} {score:.4f}")
                print(f"[hook] pushed step {step} -> {self.hf_repo}", flush=True)
            except Exception as e:
                # Never let a Hub problem (rate limit, network) kill training.
                print(f"[hook] HF push failed: {e!r}", flush=True)
        if self._push_thread and self._push_thread.is_alive():
            print("[hook] previous push still running, skipping this one", flush=True)
            return
        self._push_thread = threading.Thread(target=work, daemon=True)
        self._push_thread.start()

    # ---- the hook --------------------------------------------------------
    def evaluate(self):
        metrics = super().evaluate()
        if not metrics or not self._is_main():
            return metrics
        self.n_evals += 1
        loss = metrics.get("eval/loss")
        improved = loss is not None and loss < self.best_eval - self.hf_min_delta

        snap = None
        want_audio = self.eval_set and self.infer_every and \
            (self.n_evals % self.infer_every == 0)
        want_push = self.hf_repo and improved and self.global_step >= self.hf_min_step
        if want_audio or want_push:
            try:
                snap = self._snapshot()
            except Exception as e:
                print(f"[hook] snapshot failed: {e!r}", flush=True)

        if snap and want_audio:
            torch.cuda.empty_cache()      # hand the preview process real headroom
            self._audio_preview(snap)
        # eval_infer writes the tts-bench metrics; prefer the composite for
        # deciding what is worth publishing
        comp = None
        mp = os.path.join(self.config.output_dir, "last_eval_metrics.json")
        try:
            if want_audio and os.path.exists(mp):
                em = json.load(open(mp))
                if em.get("step") == self.global_step:
                    comp = em.get("composite")
                    w = self.accelerator.get_tracker("tensorboard", unwrap=True)
                    for k in ("composite", "wer", "cer", "mos",
                              "sentence_error_rate"):
                        if em.get(k) is not None and w is not None:
                            w.add_scalar(f"eval_quality/{k}", float(em[k]),
                                         self.global_step)
        except Exception as e:
            print(f"[hook] could not read eval metrics: {e!r}", flush=True)

        if comp is not None:
            want_push = (self.hf_repo and comp > self.best_composite + 0.001
                         and self.global_step >= self.hf_min_step)
            print(f"[hook] composite={comp:.4f} (best {self.best_composite:.4f}) "
                  f"-> {'PUSH' if want_push else 'hold'}", flush=True)

        if snap and want_push:
            self._push(snap, comp if comp is not None else loss,
                       self.global_step,
                       metric="composite" if comp is not None else "eval_loss")
        if comp is not None and comp > self.best_composite:
            self.best_composite = comp
            self._save_state()

        if loss is not None and loss < self.best_eval:
            self.best_eval = loss
            self._save_state()
        try:
            w = self.accelerator.get_tracker("tensorboard", unwrap=True)
            if w is not None:
                w.add_scalar("eval/best_loss", self.best_eval, self.global_step)
        except Exception:
            pass
        return metrics


    def train(self):
        """Run training, then let any in-flight Hub upload finish.

        The push runs on a daemon thread so it never blocks the training loop -
        but that means process exit would kill an upload mid-flight, which is
        exactly what happens at the end of a run. Join it here.
        """
        try:
            super().train()
        finally:
            t = self._push_thread
            if t and t.is_alive():
                print("[hook] waiting up to 10 min for the final HF push …",
                      flush=True)
                t.join(timeout=600)
                print(f"[hook] final push {'done' if not t.is_alive() else 'still running - abandoned'}",
                      flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_config", required=True)
    p.add_argument("--data_config", required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    frac = float(_env("OHUN_VRAM_FRACTION", "0") or 0)
    if frac > 0 and torch.cuda.is_available():
        # Leaves room for the co-resident aligner + audio tokenizer; without
        # this the allocator expands until they can no longer allocate.
        torch.cuda.set_per_process_memory_fraction(frac, 0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[hook] trainer capped at {frac:.0%} of {total:.0f} GB "
              f"= {frac*total:.0f} GB", flush=True)

    config = TrainingConfig.from_json(args.train_config)
    config.output_dir = args.output_dir
    config.data_config = args.data_config
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = build_model_and_tokenizer(config)
    train_loader, eval_loader = build_dataloaders(config, tokenizer)
    trainer = HookedTrainer(model=model, config=config,
                            train_dataloader=train_loader,
                            eval_dataloader=eval_loader, tokenizer=tokenizer)
    trainer.train()


if __name__ == "__main__":
    main()
