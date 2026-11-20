import os
import sys
import json
import time
import hashlib
import platform
import threading
import subprocess
import numpy as np
from numba import njit

try:
    import resource
except ImportError:
    resource = None

GIT_USERNAME = "230103010-wq"
NUM_THREADS = 4
ITERATIONS_PER_THREAD = 60_000_000
STRIDES = (1, 4, 8, 16)
LCG_MUL = np.uint64(6364136223846793005)


def detect_cache_line_size() -> int:
    linux_path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(linux_path):
        with open(linux_path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    if sys.platform == "darwin":
        out = subprocess.check_output(["sysctl", "-n", "hw.cachelinesize"], encoding="utf-8")
        return int(out.strip())
    return 64


def read_cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], encoding="utf-8").strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


def pick_cores(n):
    """Prefer one logical CPU per physical core (hyper-thread siblings share an L1)."""
    if hasattr(os, "sched_getaffinity"):
        avail = sorted(os.sched_getaffinity(0))
        seen, chosen = set(), []
        for c in avail:
            try:
                with open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list") as f:
                    sib = f.read().strip()
            except OSError:
                sib = str(c)
            if sib not in seen:
                seen.add(sib)
                chosen.append(c)
        return chosen[:n] if len(chosen) >= n else avail[:n]
    return [None] * n


def pin_thread(core) -> bool:
    if core is not None and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {core})   # on Linux pid 0 = calling thread
            return True
        except OSError:
            return False
    return False


# ==============================================================================
# TASK 1
# ==============================================================================
def init_system_context(git_user: str) -> dict:
    assert git_user != "REPLACE_WITH_YOUR_GIT_USERNAME" and len(git_user.strip()) > 0, (
        "Set GIT_USERNAME to your real Git handle before running.")
    logical_cores = os.cpu_count() or 1
    if logical_cores < NUM_THREADS:
        raise RuntimeError(f"Requires at least {NUM_THREADS} logical cores; detected {logical_cores}.")
    line_size = detect_cache_line_size()
    cpu_model = read_cpu_model()
    cpu_sig = f"{platform.machine()}::{platform.processor()}::{logical_cores}::{cpu_model}"
    digest = hashlib.sha256(f"{git_user.strip()}::{cpu_sig}".encode("utf8")).hexdigest()
    seed = int(digest[:16], 16)
    cores = pick_cores(NUM_THREADS)
    return {
        "git_user": git_user, "logical_cores": logical_cores, "cache_line_size": line_size,
        "cpu_model": cpu_model, "cpu_sig": cpu_sig, "instance_seed": seed,
        "affinity_supported": hasattr(os, "sched_setaffinity"), "cores": cores,
    }


# ==============================================================================
# TASK 2: in-place RMW kernel
# The store index depends on the loaded value (v & guard), where guard is a
# runtime zero (buffer.shape[0] >> 40). The compiler cannot prove idx == slot,
# so it cannot keep buffer[slot] in a register across iterations.
# ==============================================================================
@njit(nogil=True)
def worker_strided_accumulate(buffer, slot, iters, seed):
    guard = np.uint64(buffer.shape[0] >> 40)
    for _ in range(iters):
        v = buffer[slot]
        idx = slot + np.int64(v & guard)
        buffer[idx] = v * LCG_MUL + seed


# ==============================================================================
# TASK 4: register-privatized control
# ==============================================================================
@njit(nogil=True)
def worker_private_accumulate(buffer, slot, iters, seed):
    v = buffer[slot]
    for _ in range(iters):
        v = v * LCG_MUL + seed
    buffer[slot] = v


def make_buffer(stride):
    n = NUM_THREADS * stride
    raw = np.zeros(n + 16, dtype=np.uint64)
    off = ((-raw.ctypes.data) % 64) // 8
    buf = raw[off:off + n]
    assert buf.ctypes.data % 64 == 0, "buffer is not 64-byte aligned"
    return raw, buf


def thread_seeds(instance_seed):
    return [np.uint64((instance_seed + (t + 1) * 0x9E3779B97F4A7C15) % 2**64)
            for t in range(NUM_THREADS)]


def run_threads(kernel, stride, seeds, cores, iters):
    raw, buf = make_buffer(stride)
    barrier = threading.Barrier(NUM_THREADS + 1)

    def work(t):
        pin_thread(cores[t])
        barrier.wait()
        kernel(buf, t * stride, iters, seeds[t])

    threads = [threading.Thread(target=work, args=(t,)) for t in range(NUM_THREADS)]
    for th in threads:
        th.start()
    ru0 = resource.getrusage(resource.RUSAGE_SELF) if resource else None
    barrier.wait()
    t0 = time.perf_counter_ns()
    for th in threads:
        th.join()
    t1 = time.perf_counter_ns()
    ru1 = resource.getrusage(resource.RUSAGE_SELF) if resource else None
    finals = [int(buf[t * stride]) for t in range(NUM_THREADS)]
    return {
        "elapsed_ns": t1 - t0,
        "ns_per_update": (t1 - t0) / iters,    # per-thread update, threads run concurrently
        "vol_ctx_switches": (ru1.ru_nvcsw - ru0.ru_nvcsw) if ru0 else None,
        "invol_ctx_switches": (ru1.ru_nivcsw - ru0.ru_nivcsw) if ru0 else None,
        "finals": finals,
    }


# ==============================================================================
# TASK 3
# ==============================================================================
def run_stride_benchmark(ctx, seeds):
    results = {}
    for stride in STRIDES:
        r = run_threads(worker_strided_accumulate, stride, seeds, ctx["cores"], ITERATIONS_PER_THREAD)
        results[f"stride_{stride}"] = r
        print(f"stride={stride:2d} ({stride*8:3d} B): {r['elapsed_ns']:>14,d} ns  "
              f"{r['ns_per_update']:.3f} ns/update  "
              f"ctx_sw vol={r['vol_ctx_switches']} invol={r['invol_ctx_switches']}", flush=True)
    r = run_threads(worker_private_accumulate, 8, seeds, ctx["cores"], ITERATIONS_PER_THREAD)
    results["private"] = r
    print(f"private baseline  : {r['elapsed_ns']:>14,d} ns  {r['ns_per_update']:.3f} ns/update  "
          f"ctx_sw vol={r['vol_ctx_switches']} invol={r['invol_ctx_switches']}", flush=True)
    return results


# ==============================================================================
# TASK 5
# ==============================================================================
def python_reference(seed, iters):
    v, M = 0, 2**64
    s = int(seed)
    for _ in range(iters):
        v = (v * 6364136223846793005 + s) % M
    return v


def audit_llvm_and_parity(ctx, results, seeds):
    # 1) bit-exact parity across all layouts + privatized baseline
    ref = results["stride_8"]["finals"]
    parity = all(results[k]["finals"] == ref for k in results)
    # 2) independent pure-Python check of the recurrence on a short run
    rawk, bufk = make_buffer(1)
    worker_strided_accumulate(bufk, 0, 1000, seeds[0])
    reference_ok = int(bufk[0]) == python_reference(seeds[0], 1000)
    # 3) LLVM IR inspection
    asm = worker_strided_accumulate.inspect_llvm()
    ir = "\n".join(asm.values()) if isinstance(asm, dict) else str(asm)
    body, grab = [], False
    for line in ir.splitlines():
        if line.startswith("define ") and "worker_strided_accumulate" in line and "cpython" not in line:
            grab = True
        if grab:
            body.append(line)
            if line.startswith("}"):
                break
    kernel_ir = "\n".join(body) if body else ir
    ir_info = {
        "store_i64": kernel_ir.count("store i64"),
        "load_i64": kernel_ir.count("load i64"),
        "kernel_body_found": bool(body),
        "pyobject_calls_in_kernel": kernel_ir.count("PyObject"),
    }
    ir_ok = ir_info["store_i64"] > 0 and ir_info["pyobject_calls_in_kernel"] == 0
    return parity, reference_ok, ir_info, ir_ok


def main():
    ctx = init_system_context(GIT_USERNAME)
    print(f"cores={ctx['logical_cores']} line={ctx['cache_line_size']}B cpu='{ctx['cpu_model']}'")
    print(f"affinity_supported={ctx['affinity_supported']} pinned_cores={ctx['cores']}")
    print(f"INSTANCE_SEED={ctx['instance_seed']}")
    seeds = thread_seeds(ctx["instance_seed"])

    # JIT warm-up (outside timed regions)
    _, wb = make_buffer(8)
    worker_strided_accumulate(wb, 0, 10, seeds[0])
    worker_private_accumulate(wb, 0, 10, seeds[0])

    results = run_stride_benchmark(ctx, seeds)

    t1 = results["stride_1"]["elapsed_ns"]
    t8 = results["stride_8"]["elapsed_ns"]
    slowdown = t1 / t8
    print(f"slowdown stride1/stride8 = {slowdown:.2f}x")

    parity, reference_ok, ir_info, ir_ok = audit_llvm_and_parity(ctx, results, seeds)
    print(f"bit-exact parity across layouts + private: {parity}")
    print(f"matches pure-Python reference (1000 iters): {reference_ok}")
    print(f"LLVM IR kernel stats: {ir_info}  ir_ok={ir_ok}")

    passed = parity and reference_ok and ir_ok
    manifest = {
        "git_username": GIT_USERNAME,
        "instance_seed": ctx["instance_seed"],
        "cpu_model": ctx["cpu_model"],
        "logical_cores": ctx["logical_cores"],
        "cache_line_size": ctx["cache_line_size"],
        "num_threads": NUM_THREADS,
        "iterations_per_thread": ITERATIONS_PER_THREAD,
        "affinity_supported": ctx["affinity_supported"],
        "pinned_cores": ctx["cores"],
        "timings_ns": {k: v["elapsed_ns"] for k, v in results.items()},
        "ns_per_update": {k: v["ns_per_update"] for k, v in results.items()},
        "context_switches": {k: [v["vol_ctx_switches"], v["invol_ctx_switches"]]
                             for k, v in results.items()},
        "slowdown_stride1_vs_stride8": slowdown,
        "final_values": {k: [str(x) for x in v["finals"]] for k, v in results.items()},
        "llvm_ir": ir_info,
        "parity_bit_exact": parity,
        "verification_passed": passed,
    }
    with open("cache_lab_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"verification_passed={passed}  (cache_lab_manifest.json written)")
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
    