"""
Memory-safe реконструкция дыхания для EIT под большие mesh.

Ключевые идеи:
- НИКАКИХ плотных J^T J;
- Jacobian считается только по активным элементам (по умолчанию лёгкие);
- J хранится в float32;
- Tikhonov решается итерационно через scipy.sparse.linalg.lsmr
  на расширенной системе [J; sqrt(lambda) I] x = [dV; 0];
- реконструкция идёт по кадрам, чтобы не раздувать память;
- GIF и MP4 собираются ИЗ ГОТОВЫХ PNG, а не через хранение всей анимации в RAM.
"""

import glob
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as spla
from matplotlib.colors import LinearSegmentedColormap
from scipy.sparse.linalg import LinearOperator


# ============================================================
# 1. CONFIG
# ============================================================
@dataclass
class ReconConfig:
    dataset_name: str = "thorax_breath"
    lambda_tikhonov: float = 5e-2
    use_lung_mask_only: bool = True
    reference_frame: int = 0
    current_A_for_jacobian: float = 1.0

    jacobian_dtype: str = "float32"
    max_lsmr_iter: int = 150
    lsmr_atol: float = 1e-5
    lsmr_btol: float = 1e-5

    save_npz: bool = True
    save_frames: bool = True
    save_gif: bool = True
    save_mp4: bool = True
    frame_dpi: int = 110
    gif_fps: int = 10


# ============================================================
# 2. LOAD
# ============================================================
def load_dataset(path: str, name: str = "thorax_breath"):
    print(f"[Load] Загрузка из: {path}")

    with open(os.path.join(path, f"{name}_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    voltages = np.load(os.path.join(path, f"{name}_voltages.npy"))
    breath_phase = np.load(os.path.join(path, f"{name}_breath_phase.npy"))

    mesh_data = np.load(os.path.join(path, f"{name}_mesh.npz"))
    points = mesh_data["points"]
    tris = mesh_data["tris"]
    elem_physicals = mesh_data["elem_physicals"]

    electrodes = np.loadtxt(
        os.path.join(path, f"{name}_electrodes.csv"),
        delimiter=",",
        skiprows=1,
    )
    electrodes_xy = electrodes[:, :2]
    electrode_nodes = electrodes[:, 2].astype(int)

    print("[Load] ✓ Загружено:")
    print(f"  - {len(points)} узлов, {len(tris)} элементов")
    print(f"  - {meta['n_electrodes']} электродов")
    print(f"  - {voltages.shape[0]} кадров")
    print(f"  - {voltages.shape[1]} измерений на кадр")

    return {
        "meta": meta,
        "voltages": voltages,
        "breath_phase": breath_phase,
        "points": points,
        "tris": tris,
        "elem_physicals": elem_physicals,
        "electrodes_xy": electrodes_xy,
        "electrode_nodes": electrode_nodes,
    }


# ============================================================
# 3. FEM / PROTOCOL
# ============================================================
def build_drive_pattern(n_elec: int, pattern: str = "opposite"):
    frames = []
    for k in range(n_elec):
        if pattern == "adjacent":
            inj = (k, (k + 1) % n_elec)
        elif pattern == "opposite":
            inj = (k, (k + n_elec // 2) % n_elec)
        elif pattern == "skip1":
            inj = (k, (k + 2) % n_elec)
        else:
            raise ValueError(f"Неизвестный drive pattern: {pattern}")

        meas_pairs = [((k + 1 + m) % n_elec, (k + 2 + m) % n_elec) for m in range(n_elec - 2)]
        frames.append((inj, meas_pairs))
    return frames



def assemble_stiffness(points: np.ndarray, tris: np.ndarray, sigma: np.ndarray):
    import scipy.sparse as sp

    n_pts = len(points)
    rows, cols, vals = [], [], []

    for e, idx in enumerate(tris):
        xy = points[idx]
        x = xy[:, 0]
        y = xy[:, 1]
        area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
        if area < 1e-12:
            continue

        b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]], dtype=np.float64)
        c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]], dtype=np.float64)
        ke = sigma[e] * (np.outer(b, b) + np.outer(c, c)) / (4.0 * area)

        for i_loc in range(3):
            for j_loc in range(3):
                rows.append(int(idx[i_loc]))
                cols.append(int(idx[j_loc]))
                vals.append(float(ke[i_loc, j_loc]))

    return sp.coo_matrix((vals, (rows, cols)), shape=(n_pts, n_pts)).tocsr()



def solve_forward(k_global, inj_nodes: Tuple[int, int], meas_nodes: np.ndarray, current: float = 1.0) -> np.ndarray:
    n = k_global.shape[0]
    current_vec = np.zeros(n, dtype=np.float64)
    current_vec[inj_nodes[0]] += current
    current_vec[inj_nodes[1]] -= current

    available = np.setdiff1d(np.arange(n), np.array(inj_nodes, dtype=int), assume_unique=False)
    ground_node = int(available[0])

    keep = np.arange(n) != ground_node
    k_red = k_global[keep][:, keep]
    i_red = current_vec[keep]
    v_red = spla.spsolve(k_red.tocsc(), i_red)

    v = np.zeros(n, dtype=np.float64)
    v[keep] = v_red
    v -= np.mean(v[meas_nodes])
    return v



def compute_element_geometry(points: np.ndarray, tris: np.ndarray):
    n_elem = len(tris)
    elem_areas = np.zeros(n_elem, dtype=np.float64)
    elem_grads = np.zeros((n_elem, 2, 3), dtype=np.float64)

    for e, idx in enumerate(tris):
        xy = points[idx]
        x = xy[:, 0]
        y = xy[:, 1]
        area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
        elem_areas[e] = area
        if area < 1e-12:
            continue
        b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]], dtype=np.float64)
        c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]], dtype=np.float64)
        elem_grads[e, 0, :] = b / (2.0 * area)
        elem_grads[e, 1, :] = c / (2.0 * area)

    return elem_areas, elem_grads


# ============================================================
# 4. REFERENCE / MASK
# ============================================================
def build_sigma_reference(meta: Dict, elem_physicals: np.ndarray) -> np.ndarray:
    phys_mapping = meta["phys_mapping"]
    cond = meta["conductivity"]

    tag_to_sigma = {
        phys_mapping["background"]: cond["background"],
        phys_mapping["bone"]: cond["bone"],
        phys_mapping["muscle"]: cond["muscle"],
        phys_mapping["lung"]: cond["lung_exhale"],
        phys_mapping["fat"]: cond["fat"],
    }
    return np.array([tag_to_sigma.get(int(tag), cond["background"]) for tag in elem_physicals], dtype=np.float64)



def get_reconstruction_mask(meta: Dict, elem_physicals: np.ndarray, use_lung_mask_only: bool) -> np.ndarray:
    if not use_lung_mask_only:
        return np.ones(len(elem_physicals), dtype=bool)
    lung_tag = meta["phys_mapping"]["lung"]
    mask = elem_physicals == lung_tag
    if np.sum(mask) == 0:
        print("[Recon] ⚠ В mesh нет элементов лёгких, реконструирую по всем элементам")
        return np.ones(len(elem_physicals), dtype=bool)
    return mask


# ============================================================
# 5. MEMORY-SAFE JACOBIAN
# ============================================================
def compute_jacobian_active_only(
    points: np.ndarray,
    tris: np.ndarray,
    sigma_ref: np.ndarray,
    electrode_nodes: np.ndarray,
    drive_pattern: str,
    current_A: float,
    active_idx: np.ndarray,
    out_dtype=np.float32,
):
    n_electrodes = len(electrode_nodes)
    drive = build_drive_pattern(n_electrodes, drive_pattern)
    n_meas = n_electrodes * (n_electrodes - 2)
    n_active = len(active_idx)

    print(f"[Jacobian] Активный размер: {n_meas} × {n_active}")

    k_global = assemble_stiffness(points, tris, sigma_ref)
    elem_areas, elem_grads = compute_element_geometry(points, tris)

    J = np.zeros((n_meas, n_active), dtype=out_dtype)

    print("[Jacobian] Решаю forward-поля...")
    inj_fields: List[np.ndarray] = []
    meas_fields: List[List[np.ndarray]] = []

    for inj, meas_pairs in drive:
        inj_nodes = (int(electrode_nodes[inj[0]]), int(electrode_nodes[inj[1]]))
        v_inj = solve_forward(k_global, inj_nodes, electrode_nodes, current=current_A)
        inj_fields.append(v_inj)

        local_meas = []
        for m_plus, m_minus in meas_pairs:
            meas_nodes = (int(electrode_nodes[m_plus]), int(electrode_nodes[m_minus]))
            v_meas = solve_forward(k_global, meas_nodes, electrode_nodes, current=current_A)
            local_meas.append(v_meas)
        meas_fields.append(local_meas)

    print("[Jacobian] Собираю матрицу чувствительности только по активным элементам...")
    row = 0
    for k, (_, meas_pairs) in enumerate(drive):
        v_inj = inj_fields[k]
        for m_idx, _ in enumerate(meas_pairs):
            v_meas = meas_fields[k][m_idx]
            row_vals = np.zeros(n_active, dtype=np.float64)
            for j, e in enumerate(active_idx):
                idx = tris[e]
                area = elem_areas[e]
                if area < 1e-12:
                    continue
                grad_u = elem_grads[e] @ v_inj[idx]
                grad_v = elem_grads[e] @ v_meas[idx]
                row_vals[j] = -area * np.dot(grad_u, grad_v) / (sigma_ref[e] + 1e-12)
            J[row, :] = row_vals.astype(out_dtype, copy=False)
            row += 1

    return J


# ============================================================
# 6. ITERATIVE TIKHONOV
# ============================================================
def make_tikhonov_operator(J: np.ndarray, lam: float) -> LinearOperator:
    m, n = J.shape
    sqrt_lam = np.sqrt(lam).astype(np.float64)

    def matvec(x):
        x = np.asarray(x, dtype=np.float64)
        top = J @ x
        bottom = sqrt_lam * x
        return np.concatenate([top, bottom])

    def rmatvec(y):
        y = np.asarray(y, dtype=np.float64)
        y_top = y[:m]
        y_bottom = y[m:]
        return J.T @ y_top + sqrt_lam * y_bottom

    return LinearOperator((m + n, n), matvec=matvec, rmatvec=rmatvec, dtype=np.float64)



def solve_tikhonov_lsmr(J: np.ndarray, dV: np.ndarray, lam: float, maxiter: int, atol: float, btol: float) -> np.ndarray:
    m, n = J.shape
    Aop = make_tikhonov_operator(J, lam)
    rhs = np.concatenate([dV.astype(np.float64), np.zeros(n, dtype=np.float64)])
    sol = spla.lsmr(Aop, rhs, atol=atol, btol=btol, maxiter=maxiter)
    x = sol[0]
    istop = sol[1]
    itn = sol[2]
    normr = sol[3]
    return x, istop, itn, normr


# ============================================================
# 7. RECON
# ============================================================
def reconstruct_difference_eit(dataset: Dict, cfg: ReconConfig):
    meta = dataset["meta"]
    voltages = dataset["voltages"]
    points = dataset["points"]
    tris = dataset["tris"]
    elem_physicals = dataset["elem_physicals"]
    electrode_nodes = dataset["electrode_nodes"]

    sigma_ref = build_sigma_reference(meta, elem_physicals)
    mask = get_reconstruction_mask(meta, elem_physicals, cfg.use_lung_mask_only)
    active_idx = np.where(mask)[0]

    print(f"[Recon] Активных элементов для reconstruction: {len(active_idx)} / {len(tris)}")

    out_dtype = np.float32 if cfg.jacobian_dtype == "float32" else np.float64
    J = compute_jacobian_active_only(
        points=points,
        tris=tris,
        sigma_ref=sigma_ref,
        electrode_nodes=electrode_nodes,
        drive_pattern=meta.get("drive_pattern", "opposite"),
        current_A=cfg.current_A_for_jacobian,
        active_idx=active_idx,
        out_dtype=out_dtype,
    )

    V_ref = voltages[cfg.reference_frame].astype(np.float64)
    n_frames = voltages.shape[0]
    delta_sigma = np.zeros((n_frames, len(tris)), dtype=np.float32)

    print(f"[Recon] Решаю Tikhonov через LSMR, λ={cfg.lambda_tikhonov:.4g}")
    for f in range(n_frames):
        dV = voltages[f].astype(np.float64) - V_ref
        x, istop, itn, normr = solve_tikhonov_lsmr(
            J=J,
            dV=dV,
            lam=cfg.lambda_tikhonov,
            maxiter=cfg.max_lsmr_iter,
            atol=cfg.lsmr_atol,
            btol=cfg.lsmr_btol,
        )
        delta_sigma[f, active_idx] = x.astype(np.float32, copy=False)

        if (f + 1) % 10 == 0 or f == 0:
            print(f"  frame {f+1}/{n_frames}: istop={istop}, itn={itn}, normr={normr:.3e}")

    return delta_sigma, J, active_idx


# ============================================================
# 8. SAVE FRAMES / GIF / MP4
# ============================================================
def make_cmap():
    colors = ["#000000", "#0000ff", "#f0f0f0", "#ff0000"]
    return LinearSegmentedColormap.from_list("eit_cmap", colors, N=256)



def save_frames(dataset: Dict, delta_sigma: np.ndarray, output_dir: str, dpi: int = 110):
    os.makedirs(output_dir, exist_ok=True)
    meta = dataset["meta"]
    breath_phase = dataset["breath_phase"]
    points = dataset["points"]
    tris = dataset["tris"]
    electrodes_xy = dataset["electrodes_xy"]
    fps = meta["breath_fps"]
    t = np.arange(len(delta_sigma)) / fps

    vmax = np.percentile(np.abs(delta_sigma), 99)
    vmax = max(vmax, 1e-12)
    cmap = make_cmap()

    for f in range(len(delta_sigma)):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        ax_recon = axes[0]
        trip = ax_recon.tripcolor(
            points[:, 0], points[:, 1], tris,
            facecolors=delta_sigma[f],
            cmap=cmap, shading="flat",
            vmin=-vmax, vmax=vmax,
        )
        ax_recon.plot(electrodes_xy[:, 0], electrodes_xy[:, 1], "go", markersize=8, zorder=10)
        ax_recon.text(electrodes_xy[:, 0].min() - 10, 0, "L", fontsize=20, color="white", fontweight="bold", ha="center")
        ax_recon.text(electrodes_xy[:, 0].max() + 10, 0, "R", fontsize=20, color="white", fontweight="bold", ha="center")
        ax_recon.set_aspect("equal")
        ax_recon.set_title(f"Кадр {f+1}/{len(delta_sigma)} | t={t[f]:.2f}с | фаза={breath_phase[f]:.3f}")
        ax_recon.set_facecolor("black")
        plt.colorbar(trip, ax=ax_recon, label="Δσ")

        ax_breath = axes[1]
        ax_breath.plot(t, breath_phase, "b-", linewidth=2)
        ax_breath.axvline(t[f], color="r", linestyle="--", alpha=0.7)
        ax_breath.plot(t[f], breath_phase[f], "ro", markersize=10, zorder=10)
        ax_breath.set_xlabel("Время, с")
        ax_breath.set_ylabel("Фаза")
        ax_breath.set_title("Дыхательный цикл")
        ax_breath.set_ylim(-1.2, 1.2)
        ax_breath.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"frame_{f:04d}.png"), dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    print(f"[Frames] ✓ Сохранено {len(delta_sigma)} кадров в {output_dir}")



def build_gif_from_frames(frames_dir: str, output_gif: str, fps: int = 10):
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not frame_paths:
        print("[GIF] ⚠ PNG-кадры не найдены, GIF не создан")
        return

    try:
        import imageio.v2 as imageio
        duration = 1.0 / max(fps, 1)
        with imageio.get_writer(output_gif, mode="I", duration=duration, loop=0) as writer:
            for fp in frame_paths:
                writer.append_data(imageio.imread(fp))
        print(f"[GIF] ✓ Сохранён: {output_gif}")
    except Exception as e:
        print(f"[GIF] ⚠ Ошибка сборки GIF: {e}")



def build_mp4_from_frames(frames_dir: str, output_mp4: str, fps: int = 10):
    import subprocess

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_mp4,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"[MP4] ✓ Сохранён: {output_mp4}")
    except Exception as e:
        print(f"[MP4] ⚠ Ошибка сборки MP4: {e}")


# ============================================================
# 9. MAIN
# ============================================================
def main():
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = input("Укажите путь к папке с результатами: ").strip()

    if not os.path.exists(data_path):
        print(f"❌ Папка не найдена: {data_path}")
        return

    cfg = ReconConfig()
    dataset = load_dataset(data_path, cfg.dataset_name)

    delta_sigma, J, active_idx = reconstruct_difference_eit(dataset, cfg)

    if cfg.save_npz:
        out_npz = os.path.join(data_path, f"{cfg.dataset_name}_reconstruction_memory_safe.npz")
        np.savez_compressed(
            out_npz,
            delta_sigma=delta_sigma,
            jacobian_active=J,
            active_idx=active_idx,
            breath_phase=dataset["breath_phase"],
        )
        print(f"[Save] ✓ Сохранено: {out_npz}")

    frames_dir = os.path.join(data_path, "recon_frames_memory_safe")
    if cfg.save_frames:
        save_frames(dataset, delta_sigma, frames_dir, dpi=cfg.frame_dpi)

    if cfg.save_gif:
        output_gif = os.path.join(data_path, "eit_reconstruction_memory_safe.gif")
        build_gif_from_frames(frames_dir, output_gif, fps=cfg.gif_fps)

    if cfg.save_mp4:
        output_mp4 = os.path.join(data_path, "eit_reconstruction_memory_safe.mp4")
        build_mp4_from_frames(frames_dir, output_mp4, fps=cfg.gif_fps)

    print("\n✓ Реконструкция завершена")
    print(f"  Δσ shape: {delta_sigma.shape}")
    print(f"  J_active shape: {J.shape}")
    print(f"  Активных элементов: {len(active_idx)}")


if __name__ == "__main__":
    main()
