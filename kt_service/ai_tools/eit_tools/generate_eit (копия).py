import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union
import gmsh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
from shapely.affinity import translate
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from shapely.prepared import prep
import random

GeometryLike = Union[Polygon, MultiPolygon]




# ============================================================
# 1. КОНФИГУРАЦИЯ
# ============================================================
@dataclass
class EITConfig:
    conductivity: Dict[str, float] = field(default_factory=lambda: {
        "fat": 0.02,
        "bone": 0.01,
        "muscle": 0.35,
        "lung_exhale": 0.25,
        "lung_inhale": 0.08,
        "background": 0.20,
        "skin": 0.30,
    })
    breath_period_sec: float = 4.0
    breath_insp_sec: float = 1.2   # Длительность вдоха
    breath_exp_sec: float = 1.8    # Длительность выдоха
    breath_pause_sec: float = 1.0  # Длительность паузы (апное)
    breath_fps: float = 10.0
    breath_n_cycles: int = 3
    breathing_type: str = "biot"  # normal, cheyne_stokes, biot, kussmaul, gasping

    mesh_characteristic_length: float = 3.5
    mesh_order: int = 1
    mesh_min_length_factor: float = 0.35

    n_electrodes: int = 16
    electrode_radius_mm: float = 5.0
    inject_current_A: float = 5e-3

    drive_pattern: str = "opposite"
    overlap_threshold_default: float = 0.30
    overlap_threshold_lung: float = 0.15

    skin_thickness_mm: float = 3.0

    output_dir: str = "/app/generation_results"
    dataset_name: str = "thorax_breath"

    debug_plot_each_class: bool = True
    debug_plot_mesh: bool = True
    save_ground_truth_series: bool = True


# ============================================================
# 2. ПАРСИНГ И ГЕОМЕТРИЯ
# ============================================================
def parse_list_crd(list_crd: List[str], pixel_spacing: float) -> Dict[int, List[np.ndarray]]:
    tissues: Dict[int, List[np.ndarray]] = {}
    for item in list_crd:
        parts = item.strip().split()
        if len(parts) < 7:
            continue
        cls_id = int(parts[0])
        coords = np.array([float(x) for x in parts[1:]], dtype=float).reshape(-1, 2)
        tissues.setdefault(cls_id, []).append(coords * pixel_spacing)
    return tissues


def _clean_polygon_from_coords(coords: np.ndarray) -> Union[Polygon, MultiPolygon, None]:
    try:
        pts = [(float(x), float(y)) for x, y in coords]
        if len(pts) < 3:
            return None
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        poly = Polygon(pts).buffer(0)
        if poly.is_empty or poly.area <= 0:
            return None
        if isinstance(poly, (Polygon, MultiPolygon)):
            return poly
    except Exception:
        pass
    return None


def _collect_polygon_parts(geom) -> List[Polygon]:
    parts: List[Polygon] = []
    if geom is None or geom.is_empty:
        return parts
    if isinstance(geom, Polygon):
        if geom.area > 0:
            parts.append(geom)
    elif isinstance(geom, MultiPolygon):
        parts.extend([g for g in geom.geoms if g.area > 0])
    elif isinstance(geom, GeometryCollection):
        for g in geom.geoms:
            parts.extend(_collect_polygon_parts(g))
    return parts


def _normalize_geometry(geom) -> Union[GeometryLike, None]:
    if geom is None:
        return None
    geom = geom.buffer(0)
    if geom.is_empty:
        return None
    parts = _collect_polygon_parts(geom)
    if not parts:
        return None
    merged = unary_union(parts).buffer(0)
    if isinstance(merged, (Polygon, MultiPolygon)) and not merged.is_empty:
        return merged
    parts2 = _collect_polygon_parts(merged)
    if not parts2:
        return None
    merged2 = unary_union(parts2).buffer(0)
    if isinstance(merged2, (Polygon, MultiPolygon)) and not merged2.is_empty:
        return merged2
    return None


def build_class_geometry(polygons_data: Dict[int, List[np.ndarray]]) -> Dict[int, GeometryLike]:
    result: Dict[int, GeometryLike] = {}
    for class_id, coord_lists in polygons_data.items():
        polys = []
        for coords in coord_lists:
            poly = _clean_polygon_from_coords(coords)
            if poly is not None:
                polys.append(poly)
        if not polys:
            continue
        merged = _normalize_geometry(unary_union(polys))
        if merged is not None:
            result[class_id] = merged
    return result


def _largest_polygon(geom: GeometryLike) -> Polygon:
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda p: p.area)
    raise TypeError("Ожидался Polygon или MultiPolygon")


def add_polygon_to_gmsh(poly: Polygon, char_length: float, max_points: int = 500) -> int:
    ext = np.array(poly.exterior.coords[:-1], dtype=float)
    if len(ext) > max_points:
        ls = LineString(np.vstack([ext, ext[0]]))
        xs = np.linspace(0, ls.length, max_points, endpoint=False)
        ext = np.array([ls.interpolate(x).coords[0] for x in xs], dtype=float)
    point_tags = [gmsh.model.geo.addPoint(float(x), float(y), 0.0, char_length) for x, y in ext]
    line_tags = []
    for i in range(len(point_tags)):
        line_tags.append(gmsh.model.geo.addLine(point_tags[i], point_tags[(i + 1) % len(point_tags)]))
    return gmsh.model.geo.addCurveLoop(line_tags)


# ============================================================
# 3. MESH И КЛАССИФИКАЦИЯ
# ============================================================
def build_mesh_gmsh(
    body_poly: Polygon,
    inclusions: Dict[int, GeometryLike],
    char_length: float,
    min_length_factor: float,
    mesh_order: int,
    overlap_threshold_default: float,
    overlap_threshold_lung: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    PHYS_BACKGROUND = 1
    PHYS_BONE = 2
    PHYS_MUSCLE = 3
    PHYS_LUNG = 4
    PHYS_FAT = 5
    PHYS_SKIN = 6

    DICOM_TO_PHYS = {
        0: PHYS_BONE,
        1: PHYS_MUSCLE,
        2: PHYS_LUNG,
        3: PHYS_FAT,
        4: PHYS_SKIN,
    }

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("eit_model")

        print("[gmsh] Создание геометрии...")
        print(f"  Тело (класс 4): площадь {body_poly.area:.1f} мм²")

        outer_loop = add_polygon_to_gmsh(body_poly, char_length, max_points=500)
        gmsh.model.geo.addPlaneSurface([outer_loop])
        gmsh.model.geo.synchronize()

        print("[gmsh] Генерация mesh...")
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_length)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", max(char_length * min_length_factor, 0.5))
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.ElementOrder", mesh_order)
        gmsh.model.mesh.generate(2)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_tags = np.array(node_tags, dtype=int)
        node_coords = np.array(node_coords, dtype=float).reshape(-1, 3)
        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}
        points = node_coords[:, :2]

        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(2)
        triangles = []
        for etype, enodes in zip(elem_types, elem_node_tags):
            if etype == 2:
                arr = np.array(enodes, dtype=int).reshape(-1, 3)
                triangles.append(np.array([[tag_to_idx[int(n)] for n in row] for row in arr], dtype=int))
        triangles = np.vstack(triangles) if triangles else np.zeros((0, 3), dtype=int)
        if len(triangles) == 0:
            raise RuntimeError("[gmsh] Нет элементов!")

        print("[gmsh] Классификация элементов...")
        triangle_polys = [Polygon(points[tri]).buffer(0) for tri in triangles]
        triangle_areas = np.array([tp.area if not tp.is_empty else 0.0 for tp in triangle_polys])
        centroids = points[triangles].mean(axis=1)

        element_physicals = np.full(len(triangles), PHYS_BACKGROUND, dtype=int)

        priority_order = [4, 0, 2, 1, 3]
        prepared = {cls_id: prep(geom) for cls_id, geom in inclusions.items()}
        cls_names = {0: "кость", 1: "мышцы", 2: "лёгкие", 3: "жир", 4: "кожа"}

        for cls_id in priority_order:
            if cls_id not in inclusions:
                print(f"  Класс {cls_id}: нет в данных")
                continue

            geom = inclusions[cls_id]
            geom_prep = prepared[cls_id]
            phys_tag = DICOM_TO_PHYS[cls_id]
            threshold = overlap_threshold_lung if cls_id == 2 else overlap_threshold_default

            assigned_centroid = 0
            assigned_overlap = 0

            for i, c in enumerate(centroids):
                if element_physicals[i] != PHYS_BACKGROUND:
                    continue
                if geom_prep.contains(Point(float(c[0]), float(c[1]))):
                    element_physicals[i] = phys_tag
                    assigned_centroid += 1

            remaining = np.where(element_physicals == PHYS_BACKGROUND)[0]
            for i in remaining:
                tri_area = triangle_areas[i]
                if tri_area <= 1e-12:
                    continue
                try:
                    inter_area = triangle_polys[i].intersection(geom).area
                    if inter_area / tri_area >= threshold:
                        element_physicals[i] = phys_tag
                        assigned_overlap += 1
                except Exception:
                    continue

            print(
                f"  Класс {cls_id} ({cls_names.get(cls_id, '?')}): {assigned_centroid + assigned_overlap} элементов "
                f"(центроид={assigned_centroid}, overlap={assigned_overlap}, thr={threshold:.2f})"
            )

        phys_mapping = {
            "background": PHYS_BACKGROUND,
            "bone": PHYS_BONE,
            "muscle": PHYS_MUSCLE,
            "lung": PHYS_LUNG,
            "fat": PHYS_FAT,
            "skin": PHYS_SKIN,
        }

        print(f"\n[gmsh] Итого: {len(points)} узлов, {len(triangles)} элементов")
        unique, counts = np.unique(element_physicals, return_counts=True)
        tag_names_inv = {v: k for k, v in phys_mapping.items()}
        for tag, count in zip(unique.tolist(), counts.tolist()):
            print(f"  {tag_names_inv.get(tag, str(tag)):15s}: {count:5d} элементов ({count / len(triangles) * 100:5.1f}%)")

        return points, triangles, element_physicals, phys_mapping

    finally:
        gmsh.finalize()


# ============================================================
# 4. FEM
# ============================================================
def assign_conductivity(elem_physicals: np.ndarray, cfg: EITConfig, lung_sigma: float, phys_mapping: Dict[str, int]) -> np.ndarray:
    tag_to_sigma = {
        phys_mapping["background"]: cfg.conductivity["background"],
        phys_mapping["bone"]: cfg.conductivity["bone"],
        phys_mapping["muscle"]: cfg.conductivity["muscle"],
        phys_mapping["lung"]: lung_sigma,
        phys_mapping["fat"]: cfg.conductivity["fat"],
        phys_mapping["skin"]: cfg.conductivity["skin"],
    }
    return np.array([tag_to_sigma.get(int(tag), cfg.conductivity["background"]) for tag in elem_physicals], dtype=float)


# ============================================================
# ГЕНЕТИЧЕСКИЙ ОПТИМИЗАТОР
# ============================================================
GA_GENERATIONS = 200
GA_POPULATION_SIZE = 100
GA_MUTATION_RATE = 0.01
GA_ELITE_COUNT = 20
ELECTRODE_SIZE_MM = 15.0


class EITElectrodeOptimizer:
    def __init__(self, body_poly, inclusions: dict, n_electrodes: int, electrode_size_mm: float):
        self.body_poly = body_poly
        self.inclusions = inclusions
        ext_coords = np.array(body_poly.exterior.coords, dtype=float)
        self.contour_coords = ext_coords[:-1]
        self.contour_length = body_poly.length

        self.n_electrodes = n_electrodes
        self.electrode_size = electrode_size_mm

        self.lung_centers = []
        if 2 in self.inclusions:
            geom = self.inclusions[2]
            if isinstance(geom, Polygon):
                self.lung_centers.append(np.array([geom.centroid.x, geom.centroid.y]))
            elif isinstance(geom, MultiPolygon):
                for g in geom.geoms:
                    self.lung_centers.append(np.array([g.centroid.x, g.centroid.y]))

        self.bone_geoms = []
        if 0 in self.inclusions:
            geom = self.inclusions[0]
            if isinstance(geom, (Polygon, MultiPolygon)):
                if isinstance(geom, Polygon):
                    self.bone_geoms.append(geom)
                else:
                    self.bone_geoms.extend(list(geom.geoms))

        print(f"[INFO] Оптимизатор: Контур {self.contour_length:.1f}мм, Лёгкие: {len(self.lung_centers)}, Кости: {len(self.bone_geoms)}")

    def _get_point_on_contour(self, t: float) -> np.ndarray:
        t = t % 1.0
        target_dist = t * self.contour_length
        diffs = np.diff(self.contour_coords, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
        idx = np.searchsorted(cum, target_dist, side="right") - 1
        idx = min(max(idx, 0), len(seg_lens) - 1)
        t_local = (target_dist - cum[idx]) / (seg_lens[idx] + 1e-12)
        return self.contour_coords[idx] * (1.0 - t_local) + self.contour_coords[idx + 1] * t_local

    def _check_overlap(self, positions: List[float]) -> bool:
        if len(positions) < 2:
            return False
        dists_mm = sorted([p * self.contour_length for p in positions])
        n = len(dists_mm)
        for i in range(n):
            d1 = dists_mm[i]
            d2 = dists_mm[(i + 1) % n]
            if i == n - 1:
                distance = self.contour_length - dists_mm[-1] + dists_mm[0]
            else:
                distance = d2 - d1
            if distance < self.electrode_size:
                return True
        return False

    def _dist_point_to_line(self, p, a, b):
        ap = p - a
        ab = b - a
        ab_sq = np.dot(ab, ab)
        if ab_sq == 0:
            return np.linalg.norm(p - a)
        proj = np.dot(ap, ab) / ab_sq
        proj = np.clip(proj, 0.0, 1.0)
        closest = a + proj * ab
        return np.linalg.norm(p - closest)

    def _calculate_fitness(self, positions: List[float]) -> float:
        if not self.lung_centers:
            return self._calc_uniformity(positions)

        electrodes = [self._get_point_on_contour(t) for t in positions]
        lung_center = np.mean(self.lung_centers, axis=0)

        total_score = 0.0
        n_pairs = 0

        for i in range(len(electrodes)):
            for j in range(i + 1, len(electrodes)):
                e1, e2 = electrodes[i], electrodes[j]
                mid = (e1 + e2) / 2.0

                dist_lung = np.linalg.norm(mid - lung_center)
                proximity = 1.0 / (1.0 + dist_lung / 100.0)

                bone_penalty = 0.0
                for bone in self.bone_geoms:
                    dist_bone = self._dist_point_to_line(np.array([bone.centroid.x, bone.centroid.y]), e1, e2)
                    if dist_bone < 20.0:
                        bone_penalty += 0.5 * (1.0 - dist_bone / 20.0)

                path_len = np.linalg.norm(e1 - e2)
                optimal_len = self.contour_length / 4.0
                len_score = np.exp(-((path_len - optimal_len) ** 2) / (2 * 50 ** 2))

                pair_score = proximity * (1.0 - min(bone_penalty, 1.0)) * len_score
                total_score += pair_score
                n_pairs += 1

        if n_pairs > 0:
            total_score /= n_pairs

        uniformity = self._calc_uniformity(positions)
        return 0.7 * total_score + 0.3 * uniformity

    def _calc_uniformity(self, positions: List[float]) -> float:
        distances = []
        n = len(positions)
        for i in range(n):
            t1, t2 = positions[i], positions[(i + 1) % n]
            dist = (t2 - t1) if t2 >= t1 else (1.0 - t1 + t2)
            distances.append(dist)
        ideal = 1.0 / n
        score = 1.0 - np.std(distances) / ideal
        return max(0.0, min(1.0, score))

    def optimize(self, verbose=True) -> List[np.ndarray]:
        print(f"[GA] Запуск оптимизации ({GA_GENERATIONS} поколений)...")

        population = []
        for _ in range(GA_POPULATION_SIZE):
            for _ in range(100):
                pos = sorted([random.random() for _ in range(self.n_electrodes)])
                if not self._check_overlap(pos):
                    population.append(pos)
                    break
            else:
                population.append(pos)

        best_fitness_ever = -1.0
        best_pos_ever = []

        for gen in range(GA_GENERATIONS):
            fitnesses = [self._calculate_fitness(ind) for ind in population]
            max_fit = max(fitnesses)
            best_idx = fitnesses.index(max_fit)

            if max_fit > best_fitness_ever:
                best_fitness_ever = max_fit
                best_pos_ever = population[best_idx].copy()

            if verbose and gen % 50 == 0:
                print(f"  Gen {gen}: Best Fit {best_fitness_ever:.4f}")

            sorted_idx = np.argsort(fitnesses)[::-1]
            new_pop = [population[i].copy() for i in sorted_idx[:GA_ELITE_COUNT]]

            while len(new_pop) < GA_POPULATION_SIZE:
                parent = population[random.randint(0, min(19, len(population) - 1))].copy()
                child = parent.copy()

                idx_mut = random.randint(0, self.n_electrodes - 1)
                child[idx_mut] += random.gauss(0, GA_MUTATION_RATE)
                child[idx_mut] = child[idx_mut] % 1.0
                child = sorted(child)

                if not self._check_overlap(child):
                    new_pop.append(child)
                else:
                    new_pop.append(parent)

            population = new_pop

        final_coords = [self._get_point_on_contour(t) for t in best_pos_ever]
        print(f"[GA] Оптимизация завершена. Fitness: {best_fitness_ever:.4f}")
        return np.array(final_coords)


# ============================================================
# РАССТАНОВКА ЭЛЕКТРОДОВ
# ============================================================
def place_electrodes_uniform(body_poly: Polygon, n: int) -> np.ndarray:
    """Расставляет электроды строго равномерно по длине контура."""
    if n == 0:
        return np.empty((0, 2), dtype=float)
    ext = np.array(body_poly.exterior.coords, dtype=float)
    diffs = np.diff(ext, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = cum[-1]

    def get_point_at_dist(s: float) -> np.ndarray:
        idx = np.searchsorted(cum, s, side="right") - 1
        idx = min(max(idx, 0), len(seg_len) - 1)
        t = (s - cum[idx]) / (seg_len[idx] + 1e-12)
        return ext[idx] * (1.0 - t) + ext[idx + 1] * t

    step = total_len / n
    electrodes = []
    for i in range(n):
        s = i * step
        electrodes.append(get_point_at_dist(s))
    return np.array(electrodes, dtype=float)


def place_electrodes_random(body_poly: Polygon, n: int, min_dist: float = 5, max_attempts: int = 10000) -> np.ndarray:
    """Расставляет электроды случайно, проверяя минимальное расстояние между ними."""
    if n == 0:
        return np.empty((0, 2), dtype=float)
    ext = np.array(body_poly.exterior.coords, dtype=float)
    diffs = np.diff(ext, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = cum[-1]

    def get_point_at_dist(s: float) -> np.ndarray:
        idx = np.searchsorted(cum, s, side="right") - 1
        idx = min(max(idx, 0), len(seg_len) - 1)
        t = (s - cum[idx]) / (seg_len[idx] + 1e-12)
        return ext[idx] * (1.0 - t) + ext[idx + 1] * t

    electrodes = []
    attempts = 0
    while len(electrodes) < n:
        attempts += 1
        if attempts > max_attempts:
            raise ValueError(
                f"Не удалось разместить {n} электродов с min_dist={min_dist:.4f}. "
                f"Удалось разместить только {len(electrodes)}."
            )
        s = np.random.uniform(0.0, total_len)
        pt = get_point_at_dist(s)
        if not electrodes:
            electrodes.append(pt)
            continue
        pts_array = np.array(electrodes)
        dists = np.linalg.norm(pts_array - pt, axis=1)
        if np.all(dists >= min_dist):
            electrodes.append(pt)
    return np.array(electrodes, dtype=float)


def find_nearest_nodes(points: np.ndarray, electrodes: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    _, idx = tree.query(electrodes)
    return np.asarray(idx, dtype=int)


def assemble_stiffness(points: np.ndarray, tris: np.ndarray, sigma: np.ndarray) -> sp.csr_matrix:
    """
    Векторизованная сборка глобальной матрицы жесткости.
    Работает в десятки раз быстрее циклового аналога и избегает ошибок форм массивов.
    """
    points = np.asarray(points)
    tris = np.asarray(tris)
    n_pts = len(points)
    
    if len(tris) == 0:
        return sp.csr_matrix((n_pts, n_pts))
    
    # Получаем координаты вершин для всех треугольников сразу (N, 3, 2)
    xy = points[tris]
    x = xy[:, :, 0]
    y = xy[:, :, 1]
    
    x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]
    y0, y1, y2 = y[:, 0], y[:, 1], y[:, 2]
    
    # Вычисляем площади всех треугольников
    area = 0.5 * np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
    
    # Фильтруем треугольники с нулевой или очень маленькой площадью
    valid = area >= 1e-12
    if not np.any(valid):
        return sp.csr_matrix((n_pts, n_pts))
        
    x0, x1, x2 = x0[valid], x1[valid], x2[valid]
    y0, y1, y2 = y0[valid], y1[valid], y2[valid]
    area = area[valid]
    sigma_valid = sigma[valid]
    tris_valid = tris[valid]
    
    # Вычисляем коэффициенты b и c для всех треугольников
    b = np.array([y1 - y2, y2 - y0, y0 - y1]).T  # (N, 3)
    c = np.array([x2 - x1, x0 - x2, x1 - x0]).T  # (N, 3)
    
    # Вычисляем внешние произведения для всех треугольников
    bb = b[:, :, None] * b[:, None, :]  # (N, 3, 3)
    cc = c[:, :, None] * c[:, None, :]  # (N, 3, 3)
    
    # Матрица жесткости для каждого элемента
    ke = sigma_valid[:, None, None] * (bb + cc) / (4.0 * area[:, None, None])
    
    # Индексы строк и столбцов для разреженной матрицы
    rows = np.broadcast_to(tris_valid[:, :, None], (len(tris_valid), 3, 3))
    cols = np.broadcast_to(tris_valid[:, None, :], (len(tris_valid), 3, 3))
    
    # Flatten для передачи в coo_matrix
    rows = rows.flatten()
    cols = cols.flatten()
    vals = ke.flatten()
    
    return sp.coo_matrix((vals, (rows, cols)), shape=(n_pts, n_pts)).tocsr()


def solve_forward(k_global: sp.csr_matrix, inj_nodes: Tuple[int, int], meas_nodes: np.ndarray, current: float = 5e-3) -> np.ndarray:
    n = k_global.shape[0]
    current_vec = np.zeros(n, dtype=float)
    current_vec[inj_nodes[0]] += current
    current_vec[inj_nodes[1]] -= current
    ground_node = int(np.setdiff1d(np.arange(n), np.array(inj_nodes, dtype=int))[0])
    keep = np.arange(n) != ground_node
    k_red = k_global[keep][:, keep]
    i_red = current_vec[keep]
    v_red = spla.spsolve(k_red.tocsc(), i_red)
    v = np.zeros(n, dtype=float)
    v[keep] = v_red
    v -= np.mean(v[meas_nodes])
    return v[meas_nodes]


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


# ============================================================
# 5. ВИЗУАЛИЗАЦИЯ
# ============================================================
def _conductivity_by_tag(elem_physicals: np.ndarray, cfg: EITConfig, phys_mapping: Dict[str, int]) -> np.ndarray:
    sigma = np.zeros(len(elem_physicals), dtype=float)
    for i, tag in enumerate(elem_physicals):
        if tag == phys_mapping["bone"]:
            sigma[i] = cfg.conductivity["bone"]
        elif tag == phys_mapping["muscle"]:
            sigma[i] = cfg.conductivity["muscle"]
        elif tag == phys_mapping["lung"]:
            sigma[i] = cfg.conductivity["lung_exhale"]
        elif tag == phys_mapping["fat"]:
            sigma[i] = cfg.conductivity["fat"]
        elif tag == phys_mapping["skin"]:
            sigma[i] = cfg.conductivity["skin"]
        else:
            sigma[i] = cfg.conductivity["background"]
    return sigma



def _plot_debug(points, tris, elem_physicals, electrodes_final, cfg, phys_mapping,
                all_electrodes: dict, lung_elems=None, save_dir=None, folder_name=""):
    if save_dir is None:
        save_dir = cfg.output_dir

    # ---------- общие данные ----------
    tag_colors = {
        phys_mapping["background"]: "#000000",
        phys_mapping["bone"]:       "#FFFFFF",
        phys_mapping["muscle"]:     "#FF0000",
        phys_mapping["lung"]:       "#00D9FF",
        phys_mapping["fat"]:        "#BBFF00",
        phys_mapping["skin"]:       "#FFA500",
    }
    tag_names = {
        phys_mapping["background"]: "Фон/тело",
        phys_mapping["bone"]:       "Кость",
        phys_mapping["muscle"]:     "Мышцы",
        phys_mapping["lung"]:       "Лёгкие",
        phys_mapping["fat"]:        "Жир",
        phys_mapping["skin"]:       "Кожа",
    }

    patches = [MplPolygon(points[tri], closed=True) for tri in tris]
    face_colors = [tag_colors.get(int(elem_physicals[i]), "#808080") for i in range(len(tris))]

    colors_map = {
        'uniform':   ('ro', 'Равномерное'),
        'random':    ('yo', 'Случайное'),
        'optimized': ('go', 'Оптимизированное')
    }

    # ---------- какие электроды показывать ----------
    if folder_name == "optimized":
        plot_keys = ['optimized']
    elif folder_name == "random":
        plot_keys = ['random', 'optimized']
    elif folder_name == "uniform":
        plot_keys = ['uniform', 'optimized']
    else:
        plot_keys = [key for key in ['optimized', 'random', 'uniform']
                     if key in all_electrodes and all_electrodes[key] is not None]
        if not plot_keys and electrodes_final is not None:
            plot_keys = ['final']

    # ---------- вспомогательные функции ----------
    def _make_tissue_patches_collection():
        return PatchCollection(patches, facecolor=face_colors,
                               edgecolor="none", alpha=0.95)

    def _draw_electrodes(ax):
        """Рисует нужные электроды на указанной оси."""
        plotted_any = False
        for key in plot_keys:
            if key == 'final':
                if electrodes_final is not None and len(electrodes_final) > 0:
                    ax.plot(electrodes_final[:, 0], electrodes_final[:, 1], "go",
                            markersize=10, markeredgecolor="black", markeredgewidth=1.5,
                            zorder=12)
                    plotted_any = True
            else:
                fmt, _ = colors_map[key]
                if key in all_electrodes and all_electrodes[key] is not None:
                    elec = all_electrodes[key]
                    ax.plot(elec[:, 0], elec[:, 1], fmt,
                            markersize=10, markeredgecolor="black", markeredgewidth=1.5,
                            zorder=11)
                    plotted_any = True

        if not plotted_any and electrodes_final is not None and len(electrodes_final) > 0:
            ax.plot(electrodes_final[:, 0], electrodes_final[:, 1], "go",
                    markersize=10, markeredgecolor="black", markeredgewidth=1.5,
                    zorder=12)

    def _build_electrode_legend_handles():
        handles = []
        for key in plot_keys:
            if key in colors_map:
                fmt, label = colors_map[key]
                marker_color = fmt[0]
                handles.append(Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=marker_color, markersize=10,
                                      markeredgecolor='black', markeredgewidth=1.5,
                                      label=label))
        return handles

    def _build_tissue_legend_handles():
        return [
            Line2D([0], [0], marker="s", color="w",
                   markerfacecolor=tag_colors[tag], markersize=10,
                   label=tag_names[tag])
            for tag in sorted(tag_colors.keys())
        ]

    def _setup_axes(ax, title):
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(points[:, 0].min() - 20, points[:, 0].max() + 20)
        ax.set_ylim(points[:, 1].min() - 20, points[:, 1].max() + 20)

    # ======================================================
    # КАРТИНКА 1: ТКАНИ + электроды + легенда электродов + легенда тканей
    # ======================================================
    fig1, ax_t = plt.subplots(figsize=(11, 9))  # Увеличил размер
    ax_t.add_collection(_make_tissue_patches_collection())
    _draw_electrodes(ax_t)
    _setup_axes(ax_t, " ")

    electrode_handles = _build_electrode_legend_handles()
    tissue_handles = _build_tissue_legend_handles()

    # Создаем объединенную легенду
    all_handles = electrode_handles + tissue_handles
    if all_handles:
        # Разделяем легенду на две части
        leg1 = ax_t.legend(handles=electrode_handles,
                          loc="upper left", bbox_to_anchor=(0.02, 0.98),
                          fontsize=10, framealpha=0.95, title="Электроды",
                          borderaxespad=0.)
        ax_t.add_artist(leg1)  # Добавляем первую легенду как художественный объект
        
        leg2 = ax_t.legend(handles=tissue_handles,
                          loc="center left", bbox_to_anchor=(1.02, 0.5),
                          fontsize=9, framealpha=0.95, title="",
                          borderaxespad=0.)
        ax_t.add_artist(leg2)

    # Сохраняем с увеличенным bbox
    fig1.savefig(os.path.join(save_dir, "mesh_preview_tissues.png"),
                 dpi=220, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig1)

    # ======================================================
    # КАРТИНКА 2: ПРОВОДИМОСТИ + электроды + легенда электродов
    # ======================================================
    fig2, ax_c = plt.subplots(figsize=(10, 8))  # Увеличил размер

    sigma = _conductivity_by_tag(elem_physicals, cfg, phys_mapping)
    trip = ax_c.tripcolor(
        points[:, 0], points[:, 1], tris,
        facecolors=np.log10(np.clip(sigma, 1e-8, None)),
        cmap="viridis", shading="flat",
    )
    if lung_elems is not None and len(lung_elems) > 0:
        lung_points = points[tris[lung_elems]].mean(axis=1)
        ax_c.scatter(lung_points[:, 0], lung_points[:, 1],
                     c="red", s=5, alpha=0.35, zorder=5)

    _draw_electrodes(ax_c)
    _setup_axes(ax_c, "Проводимости (log scale)")

    plt.colorbar(trip, ax=ax_c, label="log10(σ)", pad=0.02)

    if electrode_handles:
        ax_c.legend(handles=electrode_handles,
                   loc="upper left", bbox_to_anchor=(0.02, 0.98),
                   fontsize=10, framealpha=0.95, title="Электроды",
                   borderaxespad=0.)

    fig2.savefig(os.path.join(save_dir, "mesh_preview_conductivity.png"),
                 dpi=220, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig2)




def _plot_class_masks(points, tris, elem_physicals, electrodes_final, cfg, phys_mapping,
                      all_electrodes: dict, save_dir=None, folder_name=""):
    if save_dir is None:
        save_dir = cfg.output_dir
    if not cfg.debug_plot_each_class:
        return
    class_info = [
        ("skin",   "Кожа"),
        ("bone",   "Кость"),
        ("muscle", "Мышцы"),
        ("lung",   "Лёгкие"),
        ("fat",    "Жир"),
    ]
    
    colors_map = {
        'uniform': ('ro', 'Равномерное'),
        'random': ('yo', 'Случайное'),
        'optimized': ('go', 'Оптимизированное')
    }
    zorder_map = {'uniform': 10, 'random': 11, 'optimized': 12}

    # Определяем, какие электроды рисовать в зависимости от папки
    if folder_name == "optimized":
        plot_keys = ['optimized']
    elif folder_name == "random":
        plot_keys = ['random', 'optimized']
    elif folder_name == "uniform":
        plot_keys = ['uniform', 'optimized']
    else:
        # Если папка не указана или другая — рисуем все доступные
        plot_keys = [key for key in ['optimized', 'random', 'uniform'] 
                     if key in all_electrodes and all_electrodes[key] is not None]
        if not plot_keys and electrodes_final is not None:
            plot_keys = ['final']

    for key, title in class_info:
        tag = phys_mapping[key]
        idx = np.where(elem_physicals == tag)[0]

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        ax1, ax2 = axes

        ax1.triplot(points[:, 0], points[:, 1], tris,
                    color="lightgray", lw=0.2, alpha=0.6)

        plotted_any = False
        for k in plot_keys:
            if k == 'final':
                if electrodes_final is not None and len(electrodes_final) > 0:
                    ax1.plot(electrodes_final[:, 0], electrodes_final[:, 1], "go",
                             markersize=12, markeredgecolor="black", markeredgewidth=1.5,
                             label="Электроды", zorder=10)
                    plotted_any = True
            else:
                fmt, label = colors_map[k]
                if k in all_electrodes and all_electrodes[k] is not None:
                    elec = all_electrodes[k]
                    ax1.plot(elec[:, 0], elec[:, 1], fmt,
                             markersize=12, markeredgecolor="black", markeredgewidth=1.5,
                             label=label, zorder=zorder_map[k])
                    plotted_any = True
        
        # Если вообще ничего не нарисовали — рисуем electrodes_final
        if not plotted_any and electrodes_final is not None and len(electrodes_final) > 0:
            ax1.plot(electrodes_final[:, 0], electrodes_final[:, 1], "go",
                     markersize=12, markeredgecolor="black", markeredgewidth=1.5,
                     label="Электроды", zorder=10)

        if len(idx) == 0:
            ax1.text(0.5, 0.5,
                     f"НЕТ ЭЛЕМЕНТОВ\nкласса {title}",
                     transform=ax1.transAxes,
                     ha="center", va="center",
                     color="red", fontsize=20, fontweight="bold")
        else:
            pts = points[tris[idx]].mean(axis=1)
            ax1.scatter(pts[:, 0], pts[:, 1],
                        c="red", s=10, alpha=0.6, label="Центроиды")

        ax1.set_aspect("equal")
        ax1.set_title(f"{title}", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper left", fontsize=9)

        sigma = np.full(len(tris), cfg.conductivity["background"], dtype=float)
        if key == "bone":
            sigma[idx] = cfg.conductivity["bone"]
        elif key == "muscle":
            sigma[idx] = cfg.conductivity["muscle"]
        elif key == "lung":
            sigma[idx] = cfg.conductivity["lung_exhale"]
        elif key == "fat":
            sigma[idx] = cfg.conductivity["fat"]
        elif key == "skin":
            sigma[idx] = cfg.conductivity["skin"]

        trip = ax2.tripcolor(
            points[:, 0], points[:, 1], tris,
            facecolors=np.log10(np.clip(sigma, 1e-8, None)),
            cmap="viridis", shading="flat",
        )

        plotted_any2 = False
        for k in plot_keys:
            if k == 'final':
                if electrodes_final is not None and len(electrodes_final) > 0:
                    ax2.plot(electrodes_final[:, 0], electrodes_final[:, 1], "go",
                             markersize=10, markeredgecolor="white", markeredgewidth=1.5,
                             label="Электроды", zorder=10)
                    plotted_any2 = True
            else:
                fmt, label = colors_map[k]
                if k in all_electrodes and all_electrodes[k] is not None:
                    elec = all_electrodes[k]
                    ax2.plot(elec[:, 0], elec[:, 1], fmt,
                             markersize=10, markeredgecolor="white", markeredgewidth=1.5,
                             label=label, zorder=zorder_map[k])
                    plotted_any2 = True
        
        if not plotted_any2 and electrodes_final is not None and len(electrodes_final) > 0:
            ax2.plot(electrodes_final[:, 0], electrodes_final[:, 1], "go",
                     markersize=10, markeredgecolor="white", markeredgewidth=1.5,
                     label="Электроды", zorder=10)

        plt.colorbar(trip, ax=ax2, label="log10(σ)")
        ax2.set_aspect("equal")
        ax2.set_title(f"Проводимость: {title}", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right", fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"debug_{key}.png"), dpi=220, bbox_inches="tight")
        plt.close(fig)


# ============================================================
# 6. ГЛАВНАЯ ФУНКЦИЯ
# ============================================================
def generate_eit_dataset(list_crd: List[str], cfg: EITConfig = None, use_optimization: bool = True):
    if cfg is None:
        cfg = EITConfig()

    os.makedirs(cfg.output_dir, exist_ok=True)

    if not list_crd or len(list_crd) < 3:
        raise ValueError("list_crd пустой или повреждён")

    pixel_spacing = float(list_crd[0])
    tissues = parse_list_crd(list_crd[2:], pixel_spacing)

    print(f"\n[Main] Найдено классов: {sorted(tissues.keys())}")
    for cls_id, polys in tissues.items():
        total_area = 0.0
        for p in polys:
            poly = _clean_polygon_from_coords(p)
            if poly is not None:
                total_area += poly.area
        print(f"  Класс {cls_id}: {len(polys)} полигонов, общая площадь {total_area:.1f} мм²")

    if 4 not in tissues:
        raise ValueError("Класс 4 (body) отсутствует!")

    body_geoms = build_class_geometry({4: tissues[4]})
    if 4 not in body_geoms:
        raise ValueError("Не удалось построить body_poly")

    body_poly = _largest_polygon(body_geoms[4]).buffer(0)
    if body_poly.is_empty or body_poly.area <= 0:
        raise ValueError("body_poly пустой или невалидный")

    raw_geoms = build_class_geometry({k: v for k, v in tissues.items() if k in [0, 1, 2, 3]})

    def clip_to_body(geom):
        if geom is None:
            return None
        g = _normalize_geometry(geom.intersection(body_poly))
        if g is None or g.is_empty or g.area <= 10.0:
            return None
        return g

    def geom_subtract(base_geom, subtract_list):
        if base_geom is None:
            return None
        subtract_list = [g for g in subtract_list if g is not None and not g.is_empty]
        if not subtract_list:
            return _normalize_geometry(base_geom)
        try:
            diff = base_geom.difference(unary_union(subtract_list))
        except Exception:
            diff = base_geom
            for g in subtract_list:
                try:
                    diff = diff.difference(g)
                except Exception:
                    pass
        diff = _normalize_geometry(diff)
        if diff is None or diff.is_empty or diff.area <= 10.0:
            return None
        return diff

    bone_raw = clip_to_body(raw_geoms.get(0))
    muscle_raw = clip_to_body(raw_geoms.get(1))
    lung_raw = clip_to_body(raw_geoms.get(2))
    fat_raw = clip_to_body(raw_geoms.get(3))

    skin_geom = None
    if cfg.skin_thickness_mm > 0:
        inner_body = body_poly.buffer(-cfg.skin_thickness_mm)
        if not inner_body.is_empty:
            skin_geom = body_poly.difference(inner_body)
            skin_geom = _normalize_geometry(skin_geom)
            if skin_geom is not None:
                print(f"\n[Main] Слой кожи: толщина={cfg.skin_thickness_mm}мм, "
                      f"площадь={skin_geom.area:.1f} мм²")

    skin_list = [skin_geom] if skin_geom is not None else []

    bone = geom_subtract(bone_raw, skin_list)
    lung = geom_subtract(lung_raw, skin_list + [bone])
    muscle = geom_subtract(muscle_raw, skin_list + [bone, lung])
    fat = geom_subtract(fat_raw, skin_list + [bone, lung, muscle])

    inclusions: Dict[int, GeometryLike] = {}
    if skin_geom is not None:
        inclusions[4] = skin_geom
    if bone is not None:
        inclusions[0] = bone
    if muscle is not None:
        inclusions[1] = muscle
    if lung is not None:
        inclusions[2] = lung
    if fat is not None:
        inclusions[3] = fat

    print("\n[Main] Геометрии включений после mutual exclusion:")
    for cls_id in [4, 0, 1, 2, 3]:
        geom = inclusions.get(cls_id)
        name = {0: "кость", 1: "мышцы", 2: "лёгкие", 3: "жир", 4: "кожа"}.get(cls_id, "?")
        if geom is None:
            print(f"  class {cls_id} ({name}): None")
        else:
            print(f"  class {cls_id} ({name}): area={geom.area:.1f}, type={geom.geom_type}")

    cx, cy = body_poly.centroid.x, body_poly.centroid.y
    body_poly = translate(body_poly, xoff=-cx, yoff=-cy)
    inclusions = {k: translate(v, xoff=-cx, yoff=-cy) for k, v in inclusions.items()}

    points, tris, elem_physicals, phys_mapping = build_mesh_gmsh(
        body_poly=body_poly,
        inclusions=inclusions,
        char_length=cfg.mesh_characteristic_length,
        min_length_factor=cfg.mesh_min_length_factor,
        mesh_order=cfg.mesh_order,
        overlap_threshold_default=cfg.overlap_threshold_default,
        overlap_threshold_lung=cfg.overlap_threshold_lung,
    )

    print("\n[Main] Расстановка электродов...")
    electrodes_uniform = place_electrodes_uniform(body_poly, cfg.n_electrodes)
    electrodes_random = place_electrodes_random(body_poly, cfg.n_electrodes)

    if use_optimization:
        print("[Main] Запуск оптимизации размещения электродов (ГА)...")
        optimizer = EITElectrodeOptimizer(
            body_poly=body_poly,
            inclusions=inclusions,
            n_electrodes=cfg.n_electrodes,
            electrode_size_mm=15.0,
        )
        electrodes_optimized = optimizer.optimize()
        print("[Main] Электроды оптимизированы.")
    else:
        electrodes_optimized = electrodes_uniform
        print("[Main] Используются равномерные электроды без оптимизации.")

    all_electrodes = {
        'uniform': electrodes_uniform,
        'random': electrodes_random,
        'optimized': electrodes_optimized
    }

    lung_mask = elem_physicals == phys_mapping["lung"]
    lung_elems = np.where(lung_mask)[0]
    print(f"\n[Main] Элементов лёгких: {len(lung_elems)} из {len(tris)} "
          f"({len(lung_elems) / len(tris) * 100:.1f}%)")


    def _generate_breath_pattern(pattern_type: str, t: np.ndarray, period: float) -> np.ndarray:
        """
        Генерирует различные паттерны дыхания.
        Возвращает breath_phase в диапазоне [0, 1]:
        0 = полный выдох/апноэ, 1 = максимальный вдох
        """
        if pattern_type == "normal":
            # Нормальное дыхание - плавная синусоида от 0 до 1
            return 0.5 * (1.0 + np.sin(2.0 * np.pi * t / period))
        
        elif pattern_type == "cheyne_stokes":
            cycle_period = period * 3.0
            t_cycle = t % cycle_period

            hyperpnea_duration = cycle_period * 0.6
            breath_phase = np.zeros_like(t)

            mask_hyperpnea = t_cycle < hyperpnea_duration
            if np.any(mask_hyperpnea):
                t_hyper = t_cycle[mask_hyperpnea]

                envelope_raw = np.sin(np.pi * t_hyper / hyperpnea_duration)
                envelope = 0.15 + 0.85 * envelope_raw   # старт не из нуля

                rapid_breath_period = period * 0.33
                rapid_phase = np.sin(2.0 * np.pi * t_hyper / rapid_breath_period)
                rapid_phase_normalized = 0.5 * (1.0 + rapid_phase)

                breath_phase[mask_hyperpnea] = envelope * rapid_phase_normalized
                breath_phase = np.clip(breath_phase, 0.0, 1.0)

            return breath_phase
        
        elif pattern_type == "biot":
            # Дыхание Биота по формуле: y = 2 + 2*sin(4x)*(1 + sign(sin(x/4)))/2
            # sin(4x) - быстрые вдохи внутри пачки
            # sign(sin(x/4)) - включение/выключение пачек (огибающая)
            # Для 3+ циклов на интервале 12 секунд:
            
            total_duration = t[-1] if len(t) > 0 else 12.0
            
            # Параметры для обеспечения минимум 3 циклов
            n_cycles = max(3, int(total_duration / 4.0))  # минимум 3 цикла
            cycle_duration = total_duration / n_cycles    # длительность одного цикла
            
            # Внутри цикла: пачка вдохов (60%) + апноэ (40%)
            burst_duration = cycle_duration * 0.6
            apnea_duration = cycle_duration * 0.4
            
            # Частота вдохов внутри пачки (3-4 вдоха за burst_duration)
            n_breaths_per_burst = 4
            breath_period = burst_duration / n_breaths_per_burst
            
            breath_phase = np.zeros_like(t)
            
            for cycle_idx in range(n_cycles):
                cycle_start = cycle_idx * cycle_duration
                cycle_end = cycle_start + cycle_duration
                burst_end = cycle_start + burst_duration
                
                # Маска для текущей пачки вдохов
                mask_burst = (t >= cycle_start) & (t < burst_end)
                
                if np.any(mask_burst):
                    t_burst = t[mask_burst] - cycle_start
                    
                    # Быстрые вдохи внутри пачки
                    for i in range(n_breaths_per_burst):
                        breath_start = i * breath_period
                        breath_end = breath_start + breath_period
                        
                        mask_breath = (t_burst >= breath_start) & (t_burst < breath_end)
                        if np.any(mask_breath):
                            tau = (t_burst[mask_breath] - breath_start) / breath_period
                            # Плавный вдох-выдох
                            breath_phase_burst = 0.5 * (1.0 - np.cos(np.pi * tau))
                            
                            # Индекс в исходном массиве
                            idx = np.where(mask_burst)[0][mask_breath]
                            breath_phase[idx] = breath_phase_burst
            
            return np.clip(breath_phase, 0.0, 1.0)
        
        elif pattern_type == "kussmaul":
            # Дыхание Куссмауля: глубокое, частое, регулярное
            rapid_period = period * 0.5  # В 2 раза чаще
            return 0.5 * (1.0 + np.sin(2.0 * np.pi * t / rapid_period))
        
        elif pattern_type == "gasping":
            # Гаспинг: редкие отдельные вдохи (агональное дыхание)
            breath_phase = np.zeros_like(t)
            
            gasp_interval = 12.0
            gasp_duration = 0.8
            
            for i in range(len(t)):
                t_mod = t[i] % gasp_interval
                if t_mod < gasp_duration:
                    breath_phase[i] = np.sin(np.pi * t_mod / gasp_duration)
            
            return breath_phase
        
        else:
            raise ValueError(f"Неизвестный тип дыхания: {pattern_type}")


    def compute_voltages_for_electrodes(electrodes: np.ndarray):
        elec_nodes = find_nearest_nodes(points, electrodes)
        drive = build_drive_pattern(cfg.n_electrodes, cfg.drive_pattern)
        dt = 1.0 / cfg.breath_fps

        if cfg.breathing_type == "cheyne_stokes":
            full_cycle_sec = cfg.breath_period_sec * 3.0
            n_frames = int(full_cycle_sec * cfg.breath_fps * cfg.breath_n_cycles)
        else:
            n_frames = int(cfg.breath_period_sec * cfg.breath_fps * cfg.breath_n_cycles)

        sigma_lung_ex = cfg.conductivity["lung_exhale"]
        sigma_lung_in = cfg.conductivity["lung_inhale"]

        t = np.arange(n_frames, dtype=float) * dt
        breath_phase = _generate_breath_pattern(cfg.breathing_type, t, cfg.breath_period_sec)
        
        # Для нормального дыхания используем старую формулу
        if cfg.breathing_type == "normal":
            lung_sigma_series = (0.5 * (sigma_lung_ex + sigma_lung_in)
                                - 0.5 * (sigma_lung_ex - sigma_lung_in) * breath_phase)
        else:
            # Для патологических типов: breath_phase от -1 до 1 или 0 до 1
            # Нормализуем к диапазону [0, 1] где 0 = выдох, 1 = вдох
            breath_normalized = (breath_phase + 1) / 2.0  # от 0 до 1
            breath_normalized = np.clip(breath_normalized, 0, 1)
            
            # Проводимость: от lung_exhale (выдох) до lung_inhale (вдох)
            lung_sigma_series = sigma_lung_ex + (sigma_lung_in - sigma_lung_ex) * breath_normalized
        
        sigma_base = assign_conductivity(elem_physicals, cfg, sigma_lung_ex, phys_mapping)
        n_meas = cfg.n_electrodes * (cfg.n_electrodes - 2)
        voltages = np.zeros((n_frames, n_meas), dtype=float)
        
        if cfg.save_ground_truth_series:
            sigma_series = np.zeros((n_frames, len(tris)), dtype=np.float32)
            delta_sigma_series = np.zeros((n_frames, len(tris)), dtype=np.float32)
        else:
            sigma_series = None
            delta_sigma_series = None
            
        for f in range(n_frames):
            sigma = sigma_base.copy()
            sigma[lung_mask] = lung_sigma_series[f]
            
            if sigma_series is not None:
                sigma_series[f] = sigma.astype(np.float32)
                delta_sigma_series[f] = (sigma - sigma_base).astype(np.float32)
                
            k_global = assemble_stiffness(points, tris, sigma)
            row = 0
            for inj, meas_pairs in drive:
                inj_nodes = (int(elec_nodes[inj[0]]), int(elec_nodes[inj[1]]))
                v = solve_forward(k_global, inj_nodes, elec_nodes, current=cfg.inject_current_A)
                for m_plus, m_minus in meas_pairs:
                    voltages[f, row] = v[m_plus] - v[m_minus]
                    row += 1
                    
            if ((f + 1) % 10 == 0) or (f == 0):
                print(f"  frame {f + 1}/{n_frames}, σ_lung={lung_sigma_series[f]:.4f} S/m, phase={breath_phase[f]:.3f}")
                
        return voltages, breath_phase, sigma_series, delta_sigma_series



    def save_dataset_to_folder(folder_name: str, electrodes, voltages, breath_phase,
                               sigma_series, delta_sigma_series, all_electrodes: dict):
        folder_path = os.path.join(cfg.output_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        print(f"\n[Save] Сохранение в папку: {folder_name}/")

        np.savez_compressed(
            os.path.join(folder_path, "mesh.npz"),
            points=points.astype(np.float32),
            tris=tris.astype(np.int32),
            elem_physicals=elem_physicals.astype(np.int16),
        )

        if voltages is not None:
            np.save(
                os.path.join(folder_path, "voltages.npy"),
                voltages.astype(np.float32)
            )
            np.save(
                os.path.join(folder_path, "breath_phase.npy"),
                breath_phase.astype(np.float32)
            )
            if sigma_series is not None:
                np.save(
                    os.path.join(folder_path, "sigma_series.npy"),
                    sigma_series
                )
                np.save(
                    os.path.join(folder_path, "delta_sigma_series.npy"),
                    delta_sigma_series
                )

        if electrodes is not None and len(electrodes) > 0:
            elec_nodes = find_nearest_nodes(points, electrodes)
            np.savetxt(
                os.path.join(folder_path, "electrodes.csv"),
                np.hstack([electrodes, elec_nodes[:, None]]),
                delimiter=",",
                header="x_mm,y_mm,node_id",
                comments=""
            )

        meta = {
            "pixel_spacing_mm": pixel_spacing,
            "n_nodes": int(len(points)),
            "n_elements": int(len(tris)),
            "n_electrodes": int(cfg.n_electrodes) if (electrodes is not None and len(electrodes) > 0) else 0,
            "n_frames": int(len(voltages)) if voltages is not None else 0,
            "drive_pattern": cfg.drive_pattern,
            "conductivity": cfg.conductivity,
            "lung_element_count": int(len(lung_elems)),
            "class_present": sorted(list(inclusions.keys())),
            "phys_mapping": phys_mapping,
            "breath_fps": cfg.breath_fps,
            "breath_period_sec": cfg.breath_period_sec,
            "skin_thickness_mm": cfg.skin_thickness_mm,
        }
        with open(os.path.join(folder_path, "meta.json"),
                  "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        if cfg.debug_plot_mesh:
            _plot_debug(
                points=points,
                tris=tris,
                elem_physicals=elem_physicals,
                electrodes_final=electrodes,
                cfg=cfg,
                phys_mapping=phys_mapping,
                all_electrodes=all_electrodes,
                lung_elems=lung_elems,
                save_dir=folder_path,
                folder_name=folder_name,
            )

        if cfg.debug_plot_each_class:
            _plot_class_masks(
                points=points,
                tris=tris,
                elem_physicals=elem_physicals,
                electrodes_final=electrodes,
                cfg=cfg,
                phys_mapping=phys_mapping,
                all_electrodes=all_electrodes,
                save_dir=folder_path,
                folder_name=folder_name,
            )

        print(f"[Save] ✓ {folder_name}/ сохранён")

    print("\n[Main] Расчёт напряжений для оптимизированных электродов...")
    voltages_opt, breath_phase, sigma_series_opt, delta_sigma_opt = \
        compute_voltages_for_electrodes(electrodes_optimized)

    print("\n[Main] Расчёт напряжений для равномерных электродов...")
    voltages_uni, _, sigma_series_uni, delta_sigma_uni = \
        compute_voltages_for_electrodes(electrodes_uniform)

    print("\n[Main] Расчёт напряжений для случайных электродов...")
    voltages_rand, _, sigma_series_rand, delta_sigma_rand = \
        compute_voltages_for_electrodes(electrodes_random)

    save_dataset_to_folder(
        folder_name="optimized",
        electrodes=electrodes_optimized,
        voltages=voltages_opt,
        breath_phase=breath_phase,
        sigma_series=sigma_series_opt,
        delta_sigma_series=delta_sigma_opt,
        all_electrodes=all_electrodes,
    )

    save_dataset_to_folder(
        folder_name="uniform",
        electrodes=electrodes_uniform,
        voltages=voltages_uni,
        breath_phase=breath_phase,
        sigma_series=sigma_series_uni,
        delta_sigma_series=delta_sigma_uni,
        all_electrodes=all_electrodes,
    )

    save_dataset_to_folder(
        folder_name="random",
        electrodes=electrodes_random,
        voltages=voltages_rand,
        breath_phase=breath_phase,
        sigma_series=sigma_series_rand,
        delta_sigma_series=delta_sigma_rand,
        all_electrodes=all_electrodes,
    )

    save_dataset_to_folder(
        folder_name="no_electrodes",
        electrodes=None,
        voltages=None,
        breath_phase=None,
        sigma_series=None,
        delta_sigma_series=None,
        all_electrodes=all_electrodes,
    )

    print(f"\n[Main] ✓ Все датасеты сохранены в {cfg.output_dir}")
    print(f"  - optimized/       (оптимизированные электроды)")
    print(f"  - uniform/         (равномерные)")
    print(f"  - random/          (случайные)") 
    print(f"  - no_electrodes/   (только mesh)")


# ============================================================
# 7. ТОЧКА ВХОДА
# ============================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = input("Укажите путь к файлу list_crd: ").strip()

    if not os.path.exists(input_file):
        print(f"❌ Файл не найден: {input_file}")
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        list_crd = f.read().splitlines()

    cfg = EITConfig()
    generate_eit_dataset(list_crd, cfg=cfg, use_optimization=True)
