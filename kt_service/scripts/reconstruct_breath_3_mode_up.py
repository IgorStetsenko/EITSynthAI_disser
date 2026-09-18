"""
Реконструкция difference-EIT для датасетов, созданных generate_eit.py.

Ключевая договорённость с генератором:
- используется фактически сохранённый breath_phase.npy, а не заново
  сгенерированные синусоидальные/трапециевидные сигналы;
- reference_frame соответствует проводимости этого кадра;
- Jacobian строится в этой же reference-проводимости;
- в измерениях сохраняется разность потенциалов V(m+) - V(m-), поэтому
  adjoint-поле измерения задаётся током +1 в m+ и -1 в m-;
- ток Jacobian согласован с током генерации через масштабирование dV.

Поддерживаются патологические паттерны generate_eit.py:
normal, kussmaul, biot, cheyne_stokes, agonal/gasping.
"""

import gc
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
import scipy.sparse as sp
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

    # Если None, берётся inject_current_A из meta.json генератора.
    current_A_for_jacobian: float | None = None

    jacobian_dtype: str = "float32"
    max_lsmr_iter: int = 150
    lsmr_atol: float = 1e-5
    lsmr_btol: float = 1e-5

    save_npz: bool = True
    save_frames: bool = True
    save_gif: bool = True
    save_mp4: bool = True
    frame_dpi: int = 110
    gif_fps: int | None = None


# ============================================================
# 2. LOAD
# ============================================================
def _as_scalar_str(value) -> str:
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError("Ожидалось скалярное строковое поле в NPZ")
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def load_dataset(path: str, name: str = "thorax_breath") -> Dict:
    print(f"[Load] Загрузка из: {path}")

    with open(os.path.join(path, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    required = ["voltages.npy", "breath_phase.npy", "mesh.npz", "electrodes.csv"]
    missing = [file_name for file_name in required if not os.path.isfile(os.path.join(path, file_name))]
    if missing:
        raise FileNotFoundError(
            "В папке нет обязательных файлов датасета generate_eit.py: " + ", ".join(missing)
        )

    voltages = np.load(os.path.join(path, "voltages.npy")).astype(np.float64, copy=False)
    breath_phase = np.load(os.path.join(path, "breath_phase.npy")).astype(np.float64, copy=False)

    mesh_data = np.load(os.path.join(path, "mesh.npz"))
    points = mesh_data["points"].astype(np.float64, copy=False)
    tris = mesh_data["tris"].astype(np.int64, copy=False)
    elem_physicals = mesh_data["elem_physicals"]

    electrodes = np.loadtxt(
        os.path.join(path, "electrodes.csv"),
        delimiter=",",
        skiprows=1,
        ndmin=2,
    )
    if electrodes.shape[1] < 3:
        raise ValueError("electrodes.csv должен содержать столбцы x_mm,y_mm,node_id")

    electrodes_xy = electrodes[:, :2].astype(np.float64, copy=False)
    electrode_nodes = electrodes[:, 2].astype(np.int64)

    n_electrodes = int(meta.get("n_electrodes", len(electrode_nodes)))
    expected_n_meas = n_electrodes * (n_electrodes - 2)
    if len(electrode_nodes) != n_electrodes:
        raise ValueError(
            f"meta.json: n_electrodes={n_electrodes}, но electrodes.csv содержит {len(electrode_nodes)} электродов"
        )
    if voltages.ndim != 2 or voltages.shape[1] != expected_n_meas:
        raise ValueError(
            f"Некорректная форма voltages: {voltages.shape}; ожидается (n_frames, {expected_n_meas}) "
            "для протокола generate_eit.py"
        )
    if len(breath_phase) != len(voltages):
        raise ValueError(
            f"breath_phase.npy содержит {len(breath_phase)} кадров, voltages.npy — {len(voltages)}"
        )

    print("[Load] ✓ Загружено:")
    print(f" - {len(points)} узлов, {len(tris)} элементов")
    print(f" - {n_electrodes} электродов")
    print(f" - {voltages.shape[0]} кадров")
    print(f" - {voltages.shape[1]} измерений на кадр")
    print(f" - breathing_type: {meta.get('breathing_type', 'не указан в meta.json')}")

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
def build_drive_pattern(n_elec: int, pattern: str = "opposite") -> List[Tuple[Tuple[int, int], List[Tuple[int, int]]]]:
    """Точная копия протокола измерений из generate_eit.py."""
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

        meas_pairs = [
            ((k + 1 + m) % n_elec, (k + 2 + m) % n_elec)
            for m in range(n_elec - 2)
        ]
        frames.append((inj, meas_pairs))
    return frames


def assemble_stiffness(points: np.ndarray, tris: np.ndarray, sigma: np.ndarray) -> sp.csr_matrix:
    """Векторизованная сборка матрицы жёсткости, согласованная с генератором."""
    n_pts = len(points)
    if len(tris) == 0:
        return sp.csr_matrix((n_pts, n_pts))

    xy = points[tris]
    x = xy[:, :, 0]
    y = xy[:, :, 1]
    x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]
    y0, y1, y2 = y[:, 0], y[:, 1], y[:, 2]

    area = 0.5 * np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
    valid = area >= 1e-12
    if not np.any(valid):
        return sp.csr_matrix((n_pts, n_pts))

    x0, x1, x2 = x0[valid], x1[valid], x2[valid]
    y0, y1, y2 = y0[valid], y1[valid], y2[valid]
    area = area[valid]
    sigma_valid = sigma[valid]
    tris_valid = tris[valid]

    b = np.column_stack((y1 - y2, y2 - y0, y0 - y1))
    c = np.column_stack((x2 - x1, x0 - x2, x1 - x0))
    ke = sigma_valid[:, None, None] * (
        b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :]
    ) / (4.0 * area[:, None, None])

    rows = np.broadcast_to(tris_valid[:, :, None], (len(tris_valid), 3, 3)).ravel()
    cols = np.broadcast_to(tris_valid[:, None, :], (len(tris_valid), 3, 3)).ravel()
    return sp.coo_matrix((ke.ravel(), (rows, cols)), shape=(n_pts, n_pts)).tocsr()


def solve_forward(
    k_global: sp.csr_matrix,
    inj_nodes: Tuple[int, int],
    meas_nodes: np.ndarray,
    current: float = 1.0,
) -> np.ndarray:
    """Решение с тем же выбором ground node и нормировкой, что в генераторе."""
    n = k_global.shape[0]
    current_vec = np.zeros(n, dtype=np.float64)
    current_vec[inj_nodes[0]] += current
    current_vec[inj_nodes[1]] -= current

    ground_node = int(np.setdiff1d(np.arange(n), np.asarray(inj_nodes, dtype=int))[0])
    keep = np.arange(n) != ground_node
    v_red = spla.spsolve(k_global[keep][:, keep].tocsc(), current_vec[keep])

    v = np.zeros(n, dtype=np.float64)
    v[keep] = v_red
    v -= np.mean(v[meas_nodes])
    return v


def compute_element_geometry(points: np.ndarray, tris: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xy = points[tris]
    x = xy[:, :, 0]
    y = xy[:, :, 1]
    x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]
    y0, y1, y2 = y[:, 0], y[:, 1], y[:, 2]

    areas = 0.5 * np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
    grads = np.zeros((len(tris), 2, 3), dtype=np.float64)
    valid = areas >= 1e-12
    if np.any(valid):
        grads[valid, 0, :] = np.column_stack((y1 - y2, y2 - y0, y0 - y1))[valid] / (2.0 * areas[valid, None])
        grads[valid, 1, :] = np.column_stack((x2 - x1, x0 - x2, x1 - x0))[valid] / (2.0 * areas[valid, None])
    return areas, grads


# ============================================================
# 4. REFERENCE / JACOBIAN
# ============================================================
def build_sigma_from_lung_value(meta: Dict, elem_physicals: np.ndarray, lung_sigma: float) -> np.ndarray:
    phys_mapping = meta["phys_mapping"]
    cond = meta["conductivity"]
    tag_to_sigma = {
        int(phys_mapping["background"]): float(cond["background"]),
        int(phys_mapping["bone"]): float(cond["bone"]),
        int(phys_mapping["muscle"]): float(cond["muscle"]),
        int(phys_mapping["lung"]): float(lung_sigma),
        int(phys_mapping["fat"]): float(cond["fat"]),
        int(phys_mapping["skin"]): float(cond.get("skin", cond["background"])),
    }
    return np.asarray(
        [tag_to_sigma.get(int(tag), float(cond["background"])) for tag in elem_physicals],
        dtype=np.float64,
    )


def get_reference_sigma(dataset: Dict, reference_frame: int) -> Tuple[np.ndarray, float, float]:
    meta = dataset["meta"]
    phases = dataset["breath_phase"]
    if not 0 <= reference_frame < len(phases):
        raise IndexError(
            f"reference_frame={reference_frame} вне диапазона 0..{len(phases) - 1}"
        )

    cond = meta["conductivity"]
    sigma_exhale = float(cond["lung_exhale"])
    sigma_inhale = float(cond["lung_inhale"])
    ref_phase = float(phases[reference_frame])
    ref_lung_sigma = sigma_exhale + (sigma_inhale - sigma_exhale) * ref_phase

    sigma_ref = build_sigma_from_lung_value(
        meta, dataset["elem_physicals"], ref_lung_sigma
    )
    return sigma_ref, ref_phase, ref_lung_sigma


def get_reconstruction_mask(meta: Dict, elem_physicals: np.ndarray, use_lung_mask_only: bool) -> np.ndarray:
    if not use_lung_mask_only:
        return np.ones(len(elem_physicals), dtype=bool)

    lung_tag = int(meta["phys_mapping"]["lung"])
    mask = elem_physicals == lung_tag
    if not np.any(mask):
        print("[Recon] ⚠ В mesh нет элементов лёгких; реконструкция выполнится по всем элементам")
        return np.ones(len(elem_physicals), dtype=bool)
    return mask


def compute_jacobian_active_only(
    points: np.ndarray,
    tris: np.ndarray,
    sigma_ref: np.ndarray,
    electrode_nodes: np.ndarray,
    drive_pattern: str,
    current_A: float,
    active_idx: np.ndarray,
    out_dtype=np.float32,
) -> np.ndarray:
    """
    Jacobian для dV = J dσ.

    Для измерения V(m+) - V(m-) adjoint-задача имеет единичный ток
    в m+ и -единичный ток в m-. Деление на sigma_ref здесь не нужно:
    оно относится к параметризации d(log sigma), тогда как генератор
    меняет непосредственно sigma элементов.
    """
    n_electrodes = len(electrode_nodes)
    drive = build_drive_pattern(n_electrodes, drive_pattern)
    n_meas = n_electrodes * (n_electrodes - 2)
    n_active = len(active_idx)

    print(f"[Jacobian] Активный размер: {n_meas} × {n_active}")
    k_global = assemble_stiffness(points, tris, sigma_ref)
    elem_areas, elem_grads = compute_element_geometry(points, tris)

    print("[Jacobian] Решаю forward- и adjoint-поля...")
    inj_fields: List[np.ndarray] = []
    meas_fields: List[List[np.ndarray]] = []
    for inj, meas_pairs in drive:
        inj_nodes = (int(electrode_nodes[inj[0]]), int(electrode_nodes[inj[1]]))
        inj_fields.append(solve_forward(k_global, inj_nodes, electrode_nodes, current=current_A))

        local_meas = []
        for m_plus, m_minus in meas_pairs:
            meas_nodes = (int(electrode_nodes[m_plus]), int(electrode_nodes[m_minus]))
            local_meas.append(solve_forward(k_global, meas_nodes, electrode_nodes, current=1.0))
        meas_fields.append(local_meas)

    print("[Jacobian] Собираю матрицу чувствительности...")
    J = np.empty((n_meas, n_active), dtype=out_dtype)
    active_tris = tris[active_idx]
    active_areas = elem_areas[active_idx]
    active_grads = elem_grads[active_idx]

    row = 0
    for drive_idx, (_, meas_pairs) in enumerate(drive):
        u_local = inj_fields[drive_idx][active_tris]
        grad_u = np.einsum("eij,ej->ei", active_grads, u_local, optimize=True)

        for meas_idx, _ in enumerate(meas_pairs):
            w_local = meas_fields[drive_idx][meas_idx][active_tris]
            grad_w = np.einsum("eij,ej->ei", active_grads, w_local, optimize=True)
            J[row, :] = (-active_areas * np.einsum("ei,ei->e", grad_u, grad_w)).astype(
                out_dtype, copy=False
            )
            row += 1

    return J


# ============================================================
# 5. TIKHONOV / RECONSTRUCTION
# ============================================================
def make_tikhonov_operator(J: np.ndarray, lam: float) -> LinearOperator:
    m, n = J.shape
    sqrt_lam = float(np.sqrt(lam))

    def matvec(x):
        x = np.asarray(x, dtype=np.float64)
        return np.concatenate((J @ x, sqrt_lam * x))

    def rmatvec(y):
        y = np.asarray(y, dtype=np.float64)
        return J.T @ y[:m] + sqrt_lam * y[m:]

    return LinearOperator((m + n, n), matvec=matvec, rmatvec=rmatvec, dtype=np.float64)


def solve_tikhonov_lsmr(
    J: np.ndarray,
    dV: np.ndarray,
    lam: float,
    maxiter: int,
    atol: float,
    btol: float,
) -> Tuple[np.ndarray, int, int, float]:
    m, n = J.shape
    rhs = np.concatenate((dV.astype(np.float64, copy=False), np.zeros(n, dtype=np.float64)))
    solution = spla.lsmr(
        make_tikhonov_operator(J, lam), rhs, atol=atol, btol=btol, maxiter=maxiter
    )
    return solution[0], int(solution[1]), int(solution[2]), float(solution[3])


def reconstruct_difference_eit(dataset: Dict, cfg: ReconConfig):
    meta = dataset["meta"]
    voltages = dataset["voltages"]
    points = dataset["points"]
    tris = dataset["tris"]
    elem_physicals = dataset["elem_physicals"]
    electrode_nodes = dataset["electrode_nodes"]

    sigma_ref, ref_phase, ref_lung_sigma = get_reference_sigma(dataset, cfg.reference_frame)
    mask = get_reconstruction_mask(meta, elem_physicals, cfg.use_lung_mask_only)
    active_idx = np.flatnonzero(mask)

    generator_current = float(meta.get("inject_current_A", 5e-3))
    jacobian_current = (
        generator_current if cfg.current_A_for_jacobian is None else float(cfg.current_A_for_jacobian)
    )
    if jacobian_current == 0.0:
        raise ValueError("current_A_for_jacobian не может быть равен нулю")

    print(f"[Recon] Reference frame: {cfg.reference_frame}")
    print(f"[Recon] Reference phase: {ref_phase:.6f}")
    print(f"[Recon] σ_lung(reference): {ref_lung_sigma:.6g} S/m")
    print(f"[Recon] Активных элементов: {len(active_idx)} / {len(tris)}")
    print(
        f"[Recon] Ток генератора: {generator_current:.6g} A; "
        f"ток Jacobian: {jacobian_current:.6g} A"
    )

    out_dtype = np.float32 if cfg.jacobian_dtype == "float32" else np.float64
    J = compute_jacobian_active_only(
        points=points,
        tris=tris,
        sigma_ref=sigma_ref,
        electrode_nodes=electrode_nodes,
        drive_pattern=meta.get("drive_pattern", "opposite"),
        current_A=jacobian_current,
        active_idx=active_idx,
        out_dtype=out_dtype,
    )

    V_ref = voltages[cfg.reference_frame]
    dV_current_scale = jacobian_current / generator_current
    n_frames = len(voltages)
    delta_sigma = np.zeros((n_frames, len(tris)), dtype=np.float32)

    print(f"[Recon] Решаю Tikhonov через LSMR, λ={cfg.lambda_tikhonov:.4g}")
    for frame_idx in range(n_frames):
        # V пропорционально току. Приводим dV к току, на котором был вычислен J.
        dV = (voltages[frame_idx] - V_ref) * dV_current_scale
        x, istop, itn, normr = solve_tikhonov_lsmr(
            J, dV, cfg.lambda_tikhonov, cfg.max_lsmr_iter, cfg.lsmr_atol, cfg.lsmr_btol
        )
        delta_sigma[frame_idx, active_idx] = x.astype(np.float32, copy=False)

        if frame_idx == 0 or (frame_idx + 1) % 10 == 0 or frame_idx + 1 == n_frames:
            print(
                f" frame {frame_idx + 1}/{n_frames}: "
                f"istop={istop}, itn={itn}, normr={normr:.3e}"
            )

    return delta_sigma, J, active_idx, sigma_ref


# ============================================================
# 6. SAVE / VISUALIZATION
# ============================================================
def make_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "eit_cmap", ["#000000", "#0000ff", "#f0f0f0", "#ff0000"], N=256
    )


def save_frames(
    dataset: Dict,
    delta_sigma: np.ndarray,
    output_dir: str,
    dpi: int = 110,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    meta = dataset["meta"]
    points = dataset["points"]
    tris = dataset["tris"]
    electrodes_xy = dataset["electrodes_xy"]
    breath_phase = dataset["breath_phase"]

    fps = float(meta.get("breath_fps", 10.0))
    t = np.arange(len(delta_sigma), dtype=float) / fps
    vmax = max(float(np.percentile(np.abs(delta_sigma), 99)), 1e-12)
    cmap = make_cmap()
    breathing_type = meta.get("breathing_type", "dataset")

    for frame_idx in range(len(delta_sigma)):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        ax_recon, ax_breath = axes

        trip = ax_recon.tripcolor(
            points[:, 0], points[:, 1], tris,
            facecolors=delta_sigma[frame_idx],
            cmap=cmap, shading="flat", vmin=-vmax, vmax=vmax,
        )
        ax_recon.plot(electrodes_xy[:, 0], electrodes_xy[:, 1], "go", markersize=7, zorder=10)
        ax_recon.set_aspect("equal")
        ax_recon.set_facecolor("black")
        ax_recon.set_title(
            f"Кадр {frame_idx + 1}/{len(delta_sigma)} | "
            f"t={t[frame_idx]:.2f} c | phase={breath_phase[frame_idx]:.3f}"
        )
        plt.colorbar(trip, ax=ax_recon, label="Δσ, S/m")

        ax_breath.plot(t, breath_phase, "b-", linewidth=2, label="Фаза из generate_eit.py")
        ax_breath.axvline(t[frame_idx], color="r", linestyle="--", alpha=0.7)
        ax_breath.plot(t[frame_idx], breath_phase[frame_idx], "ro", markersize=8)
        ax_breath.set_xlabel("Время, c")
        ax_breath.set_ylabel("Фаза вентиляции [0, 1]")
        ax_breath.set_title(f"Дыхательный паттерн: {breathing_type}")
        ax_breath.set_ylim(-0.05, 1.05)
        ax_breath.grid(True, alpha=0.3)
        ax_breath.legend(loc="best")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"frame_{frame_idx:04d}.png"), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        if (frame_idx + 1) % 10 == 0:
            gc.collect()

    print(f"[Frames] ✓ Сохранено {len(delta_sigma)} кадров в {output_dir}")


def build_gif_from_frames(frames_dir: str, output_gif: str, fps: int) -> bool:
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not frame_paths:
        print("[GIF] ⚠ PNG-кадры не найдены; GIF не создан")
        return False
    try:
        import imageio.v2 as imageio
        with imageio.get_writer(output_gif, mode="I", duration=1.0 / max(fps, 1), loop=0) as writer:
            for frame_path in frame_paths:
                writer.append_data(imageio.imread(frame_path))
        print(f"[GIF] ✓ Сохранён: {output_gif}")
        return True
    except Exception as exc:
        print(f"[GIF] ⚠ Ошибка сборки GIF: {exc}")
        return False


def build_mp4_from_frames(frames_dir: str, output_mp4: str, fps: int) -> bool:
    import subprocess

    command = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", output_mp4,
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"[MP4] ✓ Сохранён: {output_mp4}")
        return True
    except Exception as exc:
        print(f"[MP4] ⚠ Ошибка сборки MP4: {exc}")
        return False


def save_reconstruction(
    data_path: str,
    dataset: Dict,
    cfg: ReconConfig,
    delta_sigma: np.ndarray,
    jacobian: np.ndarray,
    active_idx: np.ndarray,
    sigma_ref: np.ndarray,
) -> None:
    output_dir = os.path.join(data_path, "reconstruction")
    os.makedirs(output_dir, exist_ok=True)

    if cfg.save_npz:
        out_npz = os.path.join(output_dir, f"{cfg.dataset_name}_reconstruction.npz")
        np.savez_compressed(
            out_npz,
            delta_sigma=delta_sigma,
            jacobian_active=jacobian,
            active_idx=active_idx,
            sigma_reference=sigma_ref.astype(np.float32),
            breath_phase=dataset["breath_phase"].astype(np.float32),
            reference_frame=np.asarray(cfg.reference_frame, dtype=np.int32),
            breathing_type=np.asarray(dataset["meta"].get("breathing_type", "unknown")),
        )
        print(f"[Save] ✓ Сохранено: {out_npz}")

    frames_dir = os.path.join(output_dir, "recon_frames")
    if cfg.save_frames:
        save_frames(dataset, delta_sigma, frames_dir, dpi=cfg.frame_dpi)

    fps = int(round(float(dataset["meta"].get("breath_fps", 10.0))))
    fps = cfg.gif_fps if cfg.gif_fps is not None else max(fps, 1)
    if cfg.save_gif and cfg.save_frames:
        build_gif_from_frames(frames_dir, os.path.join(output_dir, "eit_reconstruction.gif"), fps)
    if cfg.save_mp4 and cfg.save_frames:
        build_mp4_from_frames(frames_dir, os.path.join(output_dir, "eit_reconstruction.mp4"), fps)


# ============================================================
# 7. MAIN
# ============================================================
def main() -> None:
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = input("Укажите путь к папке датасета (optimized/uniform/random): ").strip()

    if not os.path.isdir(data_path):
        print(f"❌ Папка не найдена: {data_path}")
        return

    cfg = ReconConfig()
    dataset = load_dataset(data_path, cfg.dataset_name)
    delta_sigma, jacobian, active_idx, sigma_ref = reconstruct_difference_eit(dataset, cfg)
    save_reconstruction(data_path, dataset, cfg, delta_sigma, jacobian, active_idx, sigma_ref)

    print("\n" + "=" * 60)
    print("✓ РЕКОНСТРУКЦИЯ ЗАВЕРШЕНА")
    print(f" Δσ shape: {delta_sigma.shape}")
    print(f" J_active shape: {jacobian.shape}")
    print(f" Результаты: {os.path.join(data_path, 'reconstruction')}")
    print("=" * 60)


if __name__ == "__main__":
    main()