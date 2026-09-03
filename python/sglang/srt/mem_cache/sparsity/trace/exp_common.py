"""Shared runner for the HiSparse trace/AB experiment CLIs (plain Python).

Replaces the former ``*.sbatch`` wrappers: every experiment is now a plain
Python script that can run directly on a GPU node (interactively, under
``srun``, or wrapped by any scheduler). All knobs keep their old names and are
still overridable via environment variables, e.g. ``TP=8 python -m
sglang.srt.mem_cache.sparsity.trace.ab_sweep``, or via CLI flags.

Provides:

* :func:`setup_environment` -- the CUDA/HF/DeepGEMM env fixes the sbatch
  scripts carried (bundled cu13 toolkit as CUDA_HOME, CCCL compatibility-check
  disable, NVRTC preference, DSv4 fp4 dequant path).
* :class:`Server` -- launch ``sglang.launch_server`` in its own process group,
  wait for readiness, capture server_info / resolved buffer sizes, and tear
  down the whole group (TERM, then KILL, then wait for the port to free).
* :func:`run_bench` -- one ``sglang.benchmark.serving`` invocation.
* :func:`sweep_arm` -- launch + readiness + bench sweep + teardown for one arm.
* :func:`build_buffer_configs` -- matched uniform/DP per-layer buffer JSONs
  from a ``dp_allocations.csv``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import msgspec

SGLANG_DIR = Path(
    os.environ.get("SGLANG_DIR", str(Path(__file__).resolve().parents[6]))
)
RESULTS_ROOT = SGLANG_DIR / "python/sglang/srt/mem_cache/sparsity/trace/results"
TRACE_PKG = "sglang.srt.mem_cache.sparsity.trace"

DEFAULT_MODEL_PATH = (
    "/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/hf_cache/hub/"
    "models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/"
    "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
)
DEFAULT_HF_CACHE = (
    "/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/hf_cache"
)
HOST = "127.0.0.1"


def env(name: str, default):
    """Environment override with the same names the sbatch scripts used."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


def job_tag() -> str:
    """Suffix for output dirs: slurm job id when present, else 'local'."""
    return os.environ.get("SLURM_JOB_ID", "local")


def add_common_args(
    parser: argparse.ArgumentParser, *, output_len_default: Optional[int] = 256
) -> None:
    """Server/bench knobs shared by every sweep script (env-overridable)."""
    parser.add_argument("--model-path", default=env("MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--hf-cache", default=env("HF_CACHE", DEFAULT_HF_CACHE))
    parser.add_argument("--tp", type=int, default=env("TP", 4))
    parser.add_argument("--dp", type=int, default=env("DP", 4))
    parser.add_argument(
        "--max-total-tokens", type=int, default=env("MAX_TOTAL_TOKENS", 262144)
    )
    parser.add_argument(
        "--max-running-requests", type=int, default=env("MAX_RUNNING_REQUESTS", 256)
    )
    parser.add_argument("--num-prompts", type=int, default=env("NUM_PROMPTS", 200))
    parser.add_argument(
        "--output-len",
        type=int,
        default=env("OUTPUT_LEN", output_len_default),
        help="Fixed output length; omit/None to follow the dataset "
        "(gold-patch) length per request.",
    )
    parser.add_argument(
        "--swe-jsonl",
        default=env("SWE_JSONL", str(RESULTS_ROOT / "_shared/swe_bench_300.jsonl")),
    )


def setup_environment(hf_cache: str) -> None:
    """Apply the env fixes the sbatch scripts carried (idempotent).

    * ``HF_HOME`` / offline mode for compute nodes without HF network.
    * Bundled pip cu13 toolkit as ``CUDA_HOME`` (deep_gemm asserts on import
      without one) + lib paths + unversioned libcudart symlink for nvcc JIT.
    * ``-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK``: bundled nvcc 13.3 vs cudart
      headers 13.0 is a benign skew but trips cccl's guard.
    * ``DG_JIT_USE_NVRTC=true``: bundled libnvrtc matches the runtime headers.
    * ``SGLANG_DSV4_FP4_DEQUANT=1``: select the fp4->fp8 dequant MoE path for
      DSv4-Flash's fp4-packed experts (plain Triton fp8 kernel asserts).
    """
    os.environ["HF_HOME"] = hf_cache
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    cu13 = SGLANG_DIR / ".venv/lib/python3.12/site-packages/nvidia/cu13"
    if not os.environ.get("CUDA_HOME") and (cu13 / "bin/nvcc").exists():
        os.environ["CUDA_HOME"] = str(cu13)
        os.environ["PATH"] = f"{cu13}/bin:" + os.environ.get("PATH", "")
        for var in ("LD_LIBRARY_PATH", "LIBRARY_PATH"):
            os.environ[var] = f"{cu13}/lib:" + os.environ.get(var, "")
        cudart = cu13 / "lib/libcudart.so"
        if not cudart.exists():
            try:
                cudart.symlink_to("libcudart.so.13")
            except OSError:
                pass
    flags = os.environ.get("NVCC_APPEND_FLAGS", "")
    if "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK" not in flags:
        os.environ["NVCC_APPEND_FLAGS"] = (
            flags + " " if flags else ""
        ) + "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
    os.environ.setdefault("DG_JIT_USE_NVRTC", "true")
    os.environ.setdefault("SGLANG_DSV4_FP4_DEQUANT", "1")


def log(msg: str) -> None:
    print(f"=== {msg} ({time.strftime('%c')}) ===", flush=True)


def _http_ok(url: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _fetch(url: str, timeout: float = 10.0) -> Optional[bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read()
    except (urllib.error.URLError, OSError, ValueError):
        return None


class Server(msgspec.Struct, kw_only=True):
    """One ``sglang.launch_server`` instance in its own process group."""

    model_path: str
    port: int
    tp: int
    dp: int
    max_total_tokens: int
    max_running_requests: int
    log_path: Path
    extra_flags: Sequence[str] = []
    proc: Optional[subprocess.Popen] = None

    @property
    def base_url(self) -> str:
        return f"http://{HOST}:{self.port}"

    def launch(self) -> None:
        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.model_path,
            "--trust-remote-code",
            "--tp-size",
            str(self.tp),
            "--dp-size",
            str(self.dp),
            "--disable-radix-cache",
            "--disable-custom-all-reduce",
            "--max-total-tokens",
            str(self.max_total_tokens),
            "--max-running-requests",
            str(self.max_running_requests),
            "--enable-metrics",
            "--host",
            HOST,
            "--port",
            str(self.port),
        ]
        if self.dp > 1:
            cmd.append("--enable-dp-attention")
        cmd += list(self.extra_flags)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log(f"launching server: {shlex.join(cmd)}")
        with open(self.log_path, "wb") as logf:
            # start_new_session gives the server its own process group so
            # teardown can reap every DP/TP scheduler subprocess.
            self.proc = subprocess.Popen(
                cmd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True
            )

    def wait_ready(self, attempts: int = 120, interval: float = 15.0) -> bool:
        assert self.proc is not None
        for _ in range(attempts):
            if _http_ok(f"{self.base_url}/get_model_info"):
                return True
            if self.proc.poll() is not None:
                print(f"server died; see {self.log_path}", flush=True)
                return False
            time.sleep(interval)
        return False

    def save_server_info(self, out_path: Path) -> None:
        data = _fetch(f"{self.base_url}/get_server_info")
        if data:
            out_path.write_bytes(data)

    def save_resolved_buffer_sizes(self, out_path: Path) -> None:
        """Persist the per-layer config the server actually resolved."""
        needle = "hisparse per-layer device buffer sizes"
        last = ""
        try:
            for line in self.log_path.read_text(errors="replace").splitlines():
                if needle in line.lower():
                    last = line
        except OSError:
            return
        if last:
            out_path.write_text(last + "\n")

    def teardown(self) -> None:
        """TERM the whole group, escalate to KILL, wait for the port to free."""
        if self.proc is None:
            return
        log("stopping server (full process-group teardown)")
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(self.proc.pid, sig)
            except (ProcessLookupError, PermissionError):
                break
            time.sleep(5)
        for _ in range(24):
            if not _http_ok(f"{self.base_url}/get_model_info"):
                break
            time.sleep(5)
        time.sleep(10)
        self.proc = None


def run_module(module: str, *args: str, log_file: Optional[Path] = None) -> int:
    """Run ``python -m module args...``, optionally teeing output to a file."""
    cmd = [sys.executable, "-m", module, *args]
    print(f"+ {shlex.join(cmd)}", flush=True)
    if log_file is None:
        return subprocess.call(cmd)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "wb") as f:
        return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)


def run_bench(
    base_url: str,
    dataset_path: str,
    num_prompts: int,
    max_concurrency: int,
    output_file: Optional[Path] = None,
    output_len: Optional[int] = None,
    warmup_requests: int = 2,
    extra_args: Sequence[str] = (),
    log_file: Optional[Path] = None,
) -> int:
    args = [
        "--backend",
        "sglang",
        "--base-url",
        base_url,
        "--dataset-name",
        "custom",
        "--dataset-path",
        dataset_path,
        "--num-prompts",
        str(num_prompts),
        "--max-concurrency",
        str(max_concurrency),
        "--request-rate",
        "inf",
        "--warmup-requests",
        str(warmup_requests),
    ]
    if output_len is not None:
        args += ["--sharegpt-output-len", str(output_len)]
    if output_file is not None:
        args += ["--output-file", str(output_file)]
    args += list(extra_args)
    return run_module("sglang.benchmark.serving", *args, log_file=log_file)


def hisparse_config(top_k: int, buffer_sizes_path: str) -> str:
    return json.dumps({"top_k": top_k, "device_buffer_sizes_path": buffer_sizes_path})


def build_buffer_configs(
    dp_csv: str,
    cfg_dir: Path,
    total_ratio: float,
    b_mean: int,
    top_k: int,
    page_size: int,
) -> Dict[str, Path]:
    """Matched uniform/DP per-layer buffer JSONs from a dp_allocations.csv."""
    cfg_dir.mkdir(parents=True, exist_ok=True)
    rc = run_module(
        f"{TRACE_PKG}.make_buffer_sizes",
        "--dp-csv",
        dp_csv,
        "--total-ratio",
        str(total_ratio),
        "--b-mean",
        str(b_mean),
        "--top-k",
        str(top_k),
        "--page-size",
        str(page_size),
        "--out-dir",
        str(cfg_dir),
    )
    if rc != 0:
        raise RuntimeError("make_buffer_sizes failed")
    return {
        "uniform": cfg_dir / "buffer_sizes_uniform.json",
        "dp": cfg_dir / "buffer_sizes_dp.json",
    }


def sweep_arm(
    arm: str,
    out_dir: Path,
    port: int,
    args: argparse.Namespace,
    concurrency_list: Sequence[int],
    server_flags: Sequence[str] = (),
    output_len: Optional[int] = None,
    bench_extra_args: Sequence[str] = (),
) -> bool:
    """Launch one server arm, sweep concurrency, and fully tear down."""
    arm_dir = out_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    server = Server(
        model_path=args.model_path,
        port=port,
        tp=args.tp,
        dp=args.dp,
        max_total_tokens=args.max_total_tokens,
        max_running_requests=args.max_running_requests,
        log_path=arm_dir / "server.log",
        extra_flags=server_flags,
    )
    log(f"ARM={arm} port={port} flags={list(server_flags)}")
    server.launch()
    try:
        if not server.wait_ready():
            print(f"[{arm}] server not ready in time", flush=True)
            return False
        log(f"[{arm}] server ready")
        server.save_resolved_buffer_sizes(arm_dir / "resolved_buffer_sizes.txt")
        server.save_server_info(arm_dir / "server_info.json")
        for c in concurrency_list:
            log(f"[{arm}] bench concurrency={c} num_prompts={args.num_prompts}")
            rc = run_bench(
                server.base_url,
                args.swe_jsonl,
                args.num_prompts,
                c,
                output_file=arm_dir / f"bench_c{c}.jsonl",
                output_len=output_len,
                extra_args=bench_extra_args,
                log_file=arm_dir / f"bench_c{c}.log",
            )
            if rc != 0:
                print(f"[{arm}] bench c={c} returned nonzero (see log)", flush=True)
        return True
    finally:
        server.teardown()


def parse_concurrency(value: str) -> List[int]:
    return [int(tok) for tok in value.replace(",", " ").split()]


def require_file(path: str, what: str) -> None:
    if not Path(path).is_file() or Path(path).stat().st_size == 0:
        raise SystemExit(f"missing {what}: {path}")
