import os
import sys
import json
import math
import time
import hashlib
import platform
import subprocess
import numpy as np
from mpi4py import MPI
from numba import cuda, njit, float64, uint64

TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2
GLOBAL_NX = 512
GLOBAL_NY = 512
STENCIL_STEPS = 300
SHM_ITERS = 40_000_000
LAMBDA_DAMP = 0.15

NY_LOC = GLOBAL_NY // 2
NX_LOC = GLOBAL_NX // 2
DX = DY = 1.0 / (GLOBAL_NX + 1)
HALO_N = 256
HALO_BLOCKS = (4 * HALO_N + 127) // 128
TAG_UP, TAG_DOWN, TAG_LEFT, TAG_RIGHT = 0, 1, 2, 3
LCG_MUL = np.uint64(6364136223846793005)


def get_gpu_uuid() -> str:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], encoding="utf-8")
    return out.strip().splitlines()[0].strip()


def detect_cpu_line_size() -> int:
    path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    return 64


def read_cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


# TASK 1
def init_hybrid_context(github_user: str) -> dict:
    comm = MPI.COMM_WORLD
    if comm.Get_size() != 4:
        raise RuntimeError(f"Requires exactly 4 MPI ranks (2x2 grid), got {comm.Get_size()}.")

    cart_comm = comm.Create_cart(dims=[2, 2], periods=[False, False], reorder=False)
    rank = cart_comm.Get_rank()
    row_rank, col_rank = cart_comm.Get_coords(rank)
    nbr_up, nbr_down = cart_comm.Shift(0, 1)
    nbr_left, nbr_right = cart_comm.Shift(1, 1)

    cuda.select_device(0)
    gpu_uuid = comm.bcast(get_gpu_uuid() if rank == 0 else None, root=0)

    cpu_model = read_cpu_model()
    cpu_sig = f"{platform.machine()}::{os.cpu_count()}::{cpu_model}"
    line_bytes = detect_cpu_line_size()
    digest = hashlib.sha256(
        f"{github_user.strip()}::{gpu_uuid}::{cpu_sig}".encode("utf8")).hexdigest()
    instance_seed = int(digest[:16], 16)

    # node-local shared memory window: 64 uint64 slots allocated by node rank 0
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    node_rank = node_comm.Get_rank()
    if node_comm.Get_size() != 4:
        raise RuntimeError("All 4 ranks must share one node.")
    nbytes = 64 * 8 if node_rank == 0 else 0
    win = MPI.Win.Allocate_shared(nbytes, 8, comm=node_comm)
    buf, _ = win.Shared_query(0)
    shm = np.frombuffer(buf, dtype=np.uint64, count=64)
    shm_aligned = (shm.ctypes.data % 64 == 0)
    if node_rank == 0:
        shm[:] = 0
    node_comm.Barrier()

    return {
        "comm": comm, "cart_comm": cart_comm, "node_comm": node_comm, "win": win,
        "rank": rank, "node_rank": node_rank, "row_rank": row_rank, "col_rank": col_rank,
        "nbr_up": nbr_up, "nbr_down": nbr_down, "nbr_left": nbr_left, "nbr_right": nbr_right,
        "gpu_uuid": gpu_uuid, "cpu_model": cpu_model, "cpu_sig": cpu_sig,
        "line_bytes": line_bytes, "instance_seed": instance_seed,
        "shm": shm, "shm_aligned": bool(shm_aligned),
    }


# TASK 2
def initialize_fields(ctx):
    gi = ctx["row_rank"] * NY_LOC + np.arange(NY_LOC + 2)
    gj = ctx["col_rank"] * NX_LOC + np.arange(NX_LOC + 2)
    y = (gi * DY)[:, None]
    x = (gj * DX)[None, :]

    kappa = 0.16 + 0.08 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)

    rng = np.random.default_rng(ctx["instance_seed"])
    n_pulses = 3
    cx = rng.uniform(0.2, 0.8, n_pulses)
    cy = rng.uniform(0.2, 0.8, n_pulses)
    amp = rng.uniform(0.5, 1.5, n_pulses)
    sigma = 0.08
    u = np.zeros((NY_LOC + 2, NX_LOC + 2), dtype=np.float64)
    for k in range(n_pulses):
        u += amp[k] * np.exp(-((x - cx[k]) ** 2 + (y - cy[k]) ** 2) / (2 * sigma ** 2))
    S = 0.05 * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / (2 * 0.05 ** 2))

    if ctx["nbr_up"] == MPI.PROC_NULL:    u[0, :] = 0.0
    if ctx["nbr_down"] == MPI.PROC_NULL:  u[-1, :] = 0.0
    if ctx["nbr_left"] == MPI.PROC_NULL:  u[:, 0] = 0.0
    if ctx["nbr_right"] == MPI.PROC_NULL: u[:, -1] = 0.0

    return (np.ascontiguousarray(u), np.ascontiguousarray(kappa),
            np.ascontiguousarray(S), float(kappa.max()))


@cuda.jit
def pack_halos_kernel(u, send, ny, nx, N):
    t = cuda.grid(1)
    if t >= 4 * N:
        return
    seg = t // N
    k = t - seg * N
    if seg == 0:
        send[t] = u[1, k + 1]
    elif seg == 1:
        send[t] = u[ny, k + 1]
    elif seg == 2:
        send[t] = u[k + 1, 1]
    else:
        send[t] = u[k + 1, nx]


@cuda.jit
def unpack_halos_kernel(u, recv, ny, nx, N):
    t = cuda.grid(1)
    if t >= 4 * N:
        return
    seg = t // N
    k = t - seg * N
    if seg == 0:
        u[0, k + 1] = recv[t]
    elif seg == 1:
        u[ny + 1, k + 1] = recv[t]
    elif seg == 2:
        u[k + 1, 0] = recv[t]
    else:
        u[k + 1, nx + 1] = recv[t]


class SolverState:
    def __init__(self, u0, kappa, S, dt):
        self.dt = dt
        self.d_u = cuda.to_device(u0)
        self.d_un = cuda.to_device(u0.copy())
        self.d_kappa = cuda.to_device(kappa)
        self.d_S = cuda.to_device(S)
        self.d_send = cuda.device_array(4 * HALO_N, dtype=np.float64)
        self.d_recv = cuda.device_array(4 * HALO_N, dtype=np.float64)
        self.h_send = cuda.pinned_array(4 * HALO_N, dtype=np.float64)
        self.h_recv = cuda.pinned_array(4 * HALO_N, dtype=np.float64)
        self.h_send[:] = 0.0
        self.h_recv[:] = 0.0
        self.d_send.copy_to_device(self.h_send)
        self.d_recv.copy_to_device(self.h_recv)
        self.stream_boundary = cuda.stream()
        self.stream_interior = cuda.stream()


def exchange_halos_async(st, ctx):
    comm = ctx["cart_comm"]
    sb = st.stream_boundary
    N = HALO_N
    pack_halos_kernel[HALO_BLOCKS, 128, sb](st.d_u, st.d_send, NY_LOC, NX_LOC, N)
    st.d_send.copy_to_host(st.h_send, stream=sb)
    sb.synchronize()
    hs, hr = st.h_send, st.h_recv
    reqs = [
        comm.Irecv(hr[0:N],     source=ctx["nbr_up"],    tag=TAG_DOWN),
        comm.Irecv(hr[N:2*N],   source=ctx["nbr_down"],  tag=TAG_UP),
        comm.Irecv(hr[2*N:3*N], source=ctx["nbr_left"],  tag=TAG_RIGHT),
        comm.Irecv(hr[3*N:4*N], source=ctx["nbr_right"], tag=TAG_LEFT),
        comm.Isend(hs[0:N],     dest=ctx["nbr_up"],      tag=TAG_UP),
        comm.Isend(hs[N:2*N],   dest=ctx["nbr_down"],    tag=TAG_DOWN),
        comm.Isend(hs[2*N:3*N], dest=ctx["nbr_left"],    tag=TAG_LEFT),
        comm.Isend(hs[3*N:4*N], dest=ctx["nbr_right"],   tag=TAG_RIGHT),
    ]
    return reqs


def complete_halo_exchange(st, reqs):
    MPI.Request.Waitall(reqs)
    sb = st.stream_boundary
    st.d_recv.copy_to_device(st.h_recv, stream=sb)
    unpack_halos_kernel[HALO_BLOCKS, 128, sb](st.d_u, st.d_recv, NY_LOC, NX_LOC, HALO_N)


# ==============================================================================
# TASK 3: shared-memory tiled stencil + step orchestrators
# ==============================================================================
@cuda.jit
def tiled_stencil_kernel(u, un, kappa, S, r_start, r_end, c_start, c_end,
                         dt, inv_dx2, inv_dy2, lam):
    smem = cuda.shared.array((SHARED_DIM, SHARED_DIM), dtype=float64)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    i = r_start + cuda.blockIdx.y * TILE_DIM + ty
    j = c_start + cuda.blockIdx.x * TILE_DIM + tx
    H = u.shape[0]
    W = u.shape[1]

    if i < H and j < W:
        smem[ty + 1, tx + 1] = u[i, j]
        if ty == 0 and i - 1 >= 0:
            smem[0, tx + 1] = u[i - 1, j]
        if ty == TILE_DIM - 1 and i + 1 < H:
            smem[SHARED_DIM - 1, tx + 1] = u[i + 1, j]
        if tx == 0 and j - 1 >= 0:
            smem[ty + 1, 0] = u[i, j - 1]
        if tx == TILE_DIM - 1 and j + 1 < W:
            smem[ty + 1, SHARED_DIM - 1] = u[i, j + 1]

    cuda.syncthreads()

    if i <= r_end and j <= c_end:
        c = smem[ty + 1, tx + 1]
        n = smem[ty, tx + 1]
        s = smem[ty + 2, tx + 1]
        w = smem[ty + 1, tx]
        e = smem[ty + 1, tx + 2]
        k0 = kappa[i, j]
        kn = 0.5 * (kappa[i - 1, j] + k0)
        ks = 0.5 * (kappa[i + 1, j] + k0)
        kw = 0.5 * (kappa[i, j - 1] + k0)
        ke = 0.5 * (kappa[i, j + 1] + k0)
        diff = (ks * (s - c) - kn * (c - n)) * inv_dx2 \
             + (ke * (e - c) - kw * (c - w)) * inv_dy2
        un[i, j] = c + dt * (diff - lam * c * c * c + S[i, j])


def launch_stencil(st, stream, r0, r1, c0, c1):
    gy = (r1 - r0 + TILE_DIM) // TILE_DIM
    gx = (c1 - c0 + TILE_DIM) // TILE_DIM
    tiled_stencil_kernel[(gx, gy), (TILE_DIM, TILE_DIM), stream](
        st.d_u, st.d_un, st.d_kappa, st.d_S, r0, r1, c0, c1,
        st.dt, 1.0 / (DX * DX), 1.0 / (DY * DY), LAMBDA_DAMP)


def step_serialized(st, ctx):
    sb = st.stream_boundary
    reqs = exchange_halos_async(st, ctx)
    complete_halo_exchange(st, reqs)
    launch_stencil(st, sb, 1, NY_LOC, 1, NX_LOC)
    sb.synchronize()
    st.d_u, st.d_un = st.d_un, st.d_u


def step_overlapped(st, ctx):
    sb, si = st.stream_boundary, st.stream_interior
    ny, nx = NY_LOC, NX_LOC
    reqs = exchange_halos_async(st, ctx)
    launch_stencil(st, si, 2, ny - 1, 2, nx - 1)
    complete_halo_exchange(st, reqs)
    launch_stencil(st, sb, 1, 1, 1, nx)
    launch_stencil(st, sb, ny, ny, 1, nx)
    launch_stencil(st, sb, 2, ny - 1, 1, 1)
    launch_stencil(st, sb, 2, ny - 1, nx, nx)
    sb.synchronize()
    si.synchronize()
    st.d_u, st.d_un = st.d_un, st.d_u


# TASK 4
@njit(nogil=True)
def host_shm_accumulate(shm_view, slot, iters, seed):
    guard = np.uint64(shm_view.shape[0] >> 40)
    for _ in range(iters):
        v = shm_view[slot]
        idx = slot + np.int64(v & guard)
        shm_view[idx] = v * LCG_MUL + seed


def benchmark_shm_strides(ctx):
    nc, nr, shm = ctx["node_comm"], ctx["node_rank"], ctx["shm"]
    seed = np.uint64((ctx["instance_seed"] + (nr + 1) * 0x9E3779B97F4A7C15) % 2**64)
    res = {}
    for stride in (1, 8, 16):
        if nr == 0:
            shm[:] = 0
        nc.Barrier()
        t0 = time.perf_counter_ns()
        host_shm_accumulate(shm, nr * stride, SHM_ITERS, seed)
        t1 = time.perf_counter_ns()
        nc.Barrier()
        elapsed = nc.allreduce(t1 - t0, op=MPI.MAX)
        finals = nc.allgather(int(shm[nr * stride]))
        res[stride] = {"elapsed_ns": int(elapsed),
                       "ns_per_update": elapsed / SHM_ITERS,
                       "finals": finals}
    return res


def llvm_audit():
    asm = host_shm_accumulate.inspect_llvm()
    ir = "\n".join(asm.values()) if isinstance(asm, dict) else str(asm)
    body, grab = [], False
    for line in ir.splitlines():
        if line.startswith("define ") and "host_shm_accumulate" in line and "cpython" not in line:
            grab = True
        if grab:
            body.append(line)
            if line.startswith("}"):
                break
    k = "\n".join(body) if body else ir
    info = {"store_i64": k.count("store i64"), "load_i64": k.count("load i64"),
            "kernel_body_found": bool(body), "pyobject_in_kernel": k.count("PyObject")}
    ok = info["store_i64"] > 0 and (not body or info["pyobject_in_kernel"] == 0)
    return info, ok


def ptx_audit():
    asm = tiled_stencil_kernel.inspect_asm()
    ptx = "\n".join(asm.values()) if isinstance(asm, dict) else str(asm)
    counts = {"st.shared.f64": ptx.count("st.shared.f64"),
              "ld.shared.f64": ptx.count("ld.shared.f64"),
              "bar.sync": ptx.count("bar.sync")}
    return counts, all(v > 0 for v in counts.values())


# TASK 5
def global_sum(comm, val):
    s = np.array([val], dtype=np.float64)
    r = np.zeros(1, dtype=np.float64)
    comm.Allreduce(s, r, op=MPI.SUM)
    return float(r[0])


def verify_halo_exchange(ctx, u0, kappa, S, dt):
    N = NY_LOC
    uz = u0.copy()
    uz[0, :] = 0.0; uz[-1, :] = 0.0; uz[:, 0] = 0.0; uz[:, -1] = 0.0
    st = SolverState(uz, kappa, S, dt)
    complete_halo_exchange(st, exchange_halos_async(st, ctx))
    cuda.synchronize()
    out = st.d_u.copy_to_host()
    d = max(float(np.max(np.abs(out[0, 1:N+1] - u0[0, 1:N+1]))),
            float(np.max(np.abs(out[N+1, 1:N+1] - u0[N+1, 1:N+1]))),
            float(np.max(np.abs(out[1:N+1, 0] - u0[1:N+1, 0]))),
            float(np.max(np.abs(out[1:N+1, N+1] - u0[1:N+1, N+1]))))
    return ctx["cart_comm"].allreduce(d, op=MPI.MAX)


def run_solver(ctx, u0, kappa, S, dt, step_fn):
    comm = ctx["cart_comm"]
    st = SolverState(u0, kappa, S, dt)
    dxy = DX * DY
    I = (slice(1, NY_LOC + 1), slice(1, NX_LOC + 1))

    def checkpoint(step):
        u = st.d_u.copy_to_host()
        un = st.d_un.copy_to_host()
        E = global_sum(comm, float(np.sum(u[I]) * dxy))
        res = math.sqrt(global_sum(comm, float(np.sum((u[I] - un[I]) ** 2) * dxy)))
        return (step, E, res)

    trace = [checkpoint(0)]
    total_ms = 0.0
    for seg in range(STENCIL_STEPS // 100):
        comm.Barrier()
        cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            step_fn(st, ctx)
        cuda.synchronize()
        total_ms += (time.perf_counter() - t0) * 1e3
        trace.append(checkpoint((seg + 1) * 100))
    return {"u": st.d_u.copy_to_host(), "ms": total_ms, "trace": trace}


def main():
    github_user = sys.argv[1]
    ctx = init_hybrid_context(github_user)
    comm = ctx["cart_comm"]
    rank = ctx["rank"]
    u0, kappa, S, kmax_local = initialize_fields(ctx)
    kmax = comm.allreduce(kmax_local, op=MPI.MAX)
    dt = 0.5 * DX ** 2 / (4.0 * kmax)

    # JIT warm-up (GPU + CPU kernels)
    for fn in (step_serialized, step_overlapped):
        w = SolverState(u0, kappa, S, dt)
        for _ in range(3):
            fn(w, ctx)
        cuda.synchronize()
    tmp = np.zeros(64, dtype=np.uint64)
    host_shm_accumulate(tmp, 0, 10, np.uint64(1))

    halo_err = verify_halo_exchange(ctx, u0, kappa, S, dt)
    ser = run_solver(ctx, u0, kappa, S, dt, step_serialized)
    ovl = run_solver(ctx, u0, kappa, S, dt, step_overlapped)
    parity = comm.allreduce(float(np.max(np.abs(ser["u"] - ovl["u"]))), op=MPI.MAX)
    ser_ms = comm.allreduce(ser["ms"], op=MPI.MAX)
    ovl_ms = comm.allreduce(ovl["ms"], op=MPI.MAX)

    shm_res = benchmark_shm_strides(ctx)
    shm_parity = all(shm_res[s]["finals"] == shm_res[8]["finals"] for s in shm_res)

    if rank == 0:
        print(f"mpi_ranks=4 grid=2x2 user={github_user} shm_aligned_64B={ctx['shm_aligned']}")
        print(f"GPU UUID={ctx['gpu_uuid']}  CPU='{ctx['cpu_model']}' cores={os.cpu_count()}")
        print(f"INSTANCE_SEED={ctx['instance_seed']}")
        print(f"kappa_max={kmax:.6f} dx={DX:.6e} dt_max={DX**2/(4*kmax):.6e} dt={dt:.6e}")
        print(f"halo exchange check vs analytic ghosts: max err = {halo_err:.3e}")
        print("Allreduce trace (overlapped): step, E, L2_residual")
        for s, E, r in ovl["trace"]:
            print(f"  step {s:3d}  E={E:.10f}  L2res={r:.6e}")
        Es = [t[1] for t in ovl["trace"]]
        monotonic = all(Es[i + 1] < Es[i] for i in range(len(Es) - 1))
        print(f"energy monotonically decreasing: {monotonic}")
        print(f"gpu_serialized_ms={ser_ms:.2f}  gpu_overlapped_ms={ovl_ms:.2f}  "
              f"ratio={ser_ms/ovl_ms:.3f}")
        print(f"parity max|ser-ovl| after {STENCIL_STEPS} steps: {parity:.3e}")
        for s in (1, 8, 16):
            print(f"shm stride={s:2d} ({s*8:3d} B): {shm_res[s]['elapsed_ns']:>14,d} ns  "
                  f"{shm_res[s]['ns_per_update']:.3f} ns/update")
        slow = shm_res[1]["elapsed_ns"] / shm_res[8]["elapsed_ns"]
        print(f"false-sharing slowdown stride1/stride8 = {slow:.2f}x   shm parity={shm_parity}")

        counts, ptx_ok = ptx_audit()
        ir_info, ir_ok = llvm_audit()
        print(f"PTX counts: {counts} audit_passed={ptx_ok}")
        print(f"LLVM IR: {ir_info} ir_ok={ir_ok}")

        ok = ptx_ok and ir_ok and parity < 1e-12 and halo_err < 1e-12 and shm_parity
        manifest = {
            "GITHUB_USERNAME": github_user, "gpu_uuid": ctx["gpu_uuid"],
            "cpu_model": ctx["cpu_model"], "logical_cores": os.cpu_count(),
            "INSTANCE_SEED": ctx["instance_seed"], "mpi_ranks": 4, "process_grid": [2, 2],
            "stencil_steps": STENCIL_STEPS, "dt": dt, "kappa_max": kmax,
            "gpu_serialized_ms": ser_ms, "gpu_overlapped_ms": ovl_ms,
            "parity_max_diff": parity, "halo_check_max_err": halo_err,
            "energy_trace": [{"step": s, "E": E, "l2_residual": r} for s, E, r in ovl["trace"]],
            "shm_iters": SHM_ITERS, "shm_aligned_64B": ctx["shm_aligned"],
            "shm_timings_ns": {f"stride_{s}": shm_res[s]["elapsed_ns"] for s in shm_res},
            "shm_ns_per_update": {f"stride_{s}": shm_res[s]["ns_per_update"] for s in shm_res},
            "false_sharing_slowdown": slow, "shm_parity": shm_parity,
            "ptx_counts": counts, "ptx_audit_passed": ptx_ok,
            "llvm_ir": ir_info, "llvm_audit_passed": ir_ok,
            "verification_passed": ok,
        }
        with open("hybrid_lab_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"verification_passed={ok} (hybrid_lab_manifest.json written)")
        if not ok:
            sys.stdout.flush()
            comm.Abort(1)
    comm.Barrier()
    ctx["win"].Free()


if __name__ == "__main__":
    main()
    #code structure was partly generated by ai)
