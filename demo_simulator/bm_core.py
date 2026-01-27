# bm_core.py
import numpy as np
import pandas as pd
import re
from scipy.special import lambertw

# --- helpers (из ноутбука, с небольшими правками) ---


def _norm(s: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    norm_cols = { _norm(c): c for c in df.columns }
    # 1) точное совпадение после нормализации
    for cand in candidates:
        key = _norm(cand)
        if key in norm_cols:
            return norm_cols[key]
    # 2) частичное совпадение (если в заголовке много текста)
    for cand in candidates:
        key = _norm(cand)
        for nc, orig in norm_cols.items():
            if key and key in nc:
                return orig
    return None

def standardize_lithotypes_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    targets = {
        "Lithology type": ["Lithology type", "Lithology name", "Lithology", "Rock type"],
        "Initial porosity": ["Initial porosity", "Initial Porosity", "Initial porosity (%)", "Initial porosity %", "Phi0", "phi0"],
        "Athy factor k (depth)": ["Athy factor k (depth)", "Athy factor", "Athy k", "Compaction coefficient", "k (depth)", "kdepth", "c"],
        "Density": ["Density", "Grain density", "Rock density", "rho", "ρ"],
        "Specific surface": ["Specific surface", "Surface area", "S", "SSA"],
        "Permeability factor": ["Permeability factor", "k factor", "Perm factor"],
        "Constant Value 2": ["Constant Value 2", "Bulk modulus", "K", "K (GPa)", "Bulk Modulus"],
        "Shear Modulus": ["Shear Modulus", "G", "G (GPa)", "Shear modulus (GPa)"],
    }

    rename_map = {}
    for std_name, cands in targets.items():
        col = _find_col(df, cands)
        if col and col != std_name:
            rename_map[col] = std_name

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


LITHOLOGY_REPLACE = {
    "Sandstones": "Sandstone (typical)",
    "Shales": "Shale (typical)",
    "Limestones": "Limestone (organic rich - typical)",
    "Dolomite": "Dolomite (typical)",
    "Chalk": "Chalk (typical)",
    "Anhydrite": "Anhydrite",
    "Quartzite": "Quartzite",
}

DEFAULT_R = 1e6
DEFAULT_TAU = 1.5
DEFAULT_S = 1e6
DEFAULT_K = 1.0

# Ваши “жёсткие” S по литологиям (как в ноутбуке)
LITHOLOGY_S_VALUES = {
    "Shale (typical)": 1e10,
    "Sandstone (typical)": 1e5,
    "Chalk (typical)": 1e20,
    "Limestone (organic rich - typical)": 1e6,
    "Dolomite (typical)": 1e10,
    "Anhydrite": 1e20,
    "Quartzite": 1e20,
}

def normalize_inputs(df_well: pd.DataFrame) -> pd.DataFrame:
    df = df_well.copy()
    if "Lithology_type" in df.columns:
        df["Lithology_type"] = df["Lithology_type"].replace(LITHOLOGY_REPLACE)
    return df

def ensure_depths_km(df_well: pd.DataFrame, assume_meters: bool = True) -> pd.DataFrame:
    """
    В ноутбуке вы делили Depth top/bottom на 1000.
    Если assume_meters=True — считаем, что вход в метрах и переводим в км.
    """
    df = df_well.copy()
    for col in ["Depth top, m", "Depth bottom, m"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if assume_meters:
                df[col] = df[col] / 1000.0
    if "Thickness, m" in df.columns:
        df["Thickness, m"] = pd.to_numeric(df["Thickness, m"], errors="coerce")
        if assume_meters:
            df["Thickness, m"] = df["Thickness, m"] / 1000.0
    return df

def corrected(depth_km: float, rho_mantle=3300, rho_water=1000, sea_level_km=0.0, sea_level_today_km=0.0) -> float:
    return depth_km - sea_level_today_km * (rho_water / (rho_mantle - rho_water)) + (sea_level_km - sea_level_today_km)

def porosity_func(phi0: float, c: float, y1_d: float, y2_d: float, thickness_km: float) -> float:
    thickness_km = max(float(thickness_km), 1e-12)
    c = max(float(c), 1e-12)
    return phi0 / c * ((np.exp(-c * y1_d) - np.exp(-c * y2_d)) / thickness_km)

def decomp_func(y1: float, y2: float, y1_d: float, phi0: float, c: float) -> float:
    # ваша формула с lambertw (как в ноутбуке)
    e = np.e
    y2_d = (
        (np.exp(-np.log(e) * y1_d * c) * phi0 * np.log(e)
         - np.exp(-np.log(e) * y1 * c) * phi0 * np.log(e)
         + np.exp(-np.log(e) * y2 * c) * phi0 * np.log(e)
         - np.log(e) * y1 * c + np.log(e) * y1_d * c + np.log(e) * y2 * c
         + lambertw(
            -np.log(e) * phi0 * np.exp(
                -np.exp(-np.log(e) * y1_d * c) * phi0 * np.log(e)
                + np.exp(-np.log(e) * y1 * c) * phi0 * np.log(e)
                - np.exp(-np.log(e) * y2 * c) * phi0 * np.log(e)
                + np.log(e) * y1 * c - np.log(e) * y1_d * c - np.log(e) * y2 * c
            )
         )) / (c * np.log(e))
    )
    return float(np.real(y2_d))

def bulk_density(phi: float, rho_grain: float, rho_water: float = 1000.0) -> float:
    return phi * rho_water + (1 - phi) * rho_grain

def kozeny_carman_lithology(phi: float, S: float, k_factor: float) -> float:
    phi_c = float(phi) - 3.1e-9
    phi_c = max(phi_c, 1e-12)
    S = max(float(S), 1e-12)
    if phi_c < 0.1:
        return 2e16 * k_factor * (phi_c**5 / (S**2 * (1 - phi_c)**2))
    return 2e14 * k_factor * (phi_c**3 / (S**2 * (1 - phi_c)**2))

def sedimentation_rate_mm_per_yr(depth_km: float, age_ma: float) -> float:
    # 1 km/Ma == 1 mm/yr
    if age_ma == 0:
        return 0.0
    return float(depth_km) / float(age_ma)

# bm_core.py
import pandas as pd

def _make_unique_columns(cols):
    seen = {}
    out = []
    for i, c in enumerate(cols):
        if pd.isna(c) or str(c).strip() == "":
            base = f"col_{i}"
        else:
            base = str(c).strip()

        k = seen.get(base, 0)
        seen[base] = k + 1
        out.append(base if k == 0 else f"{base}.{k}")
    return out

def load_lithotypes_sheet(excel_file, sheet_name: str) -> pd.DataFrame:
    import pandas as pd

    raw = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)

    # 1) находим строку, где реально перечислены поля (porosity/density/athy и т.п.)
    header_row = None
    for r in range(min(40, len(raw))):
        row = raw.iloc[r].astype(str).str.lower()
        if (
            row.str.contains("porosity").any()
            and (row.str.contains("athy").any() or row.str.contains("density").any() or row.str.contains("compaction").any())
        ):
            header_row = r
            break

    if header_row is None:
        # запасной вариант: часто названия полей на 2-й строке
        header_row = 1

    # 2) ставим эту строку как заголовки
    df = raw.iloc[header_row + 1 :].copy()
    df.columns = raw.iloc[header_row].tolist()

    # 3) убираем пустые строки
    df = df.dropna(how="all").reset_index(drop=True)

    # 4) приводим названия колонок к строкам, заменяем NaN, делаем уникальными
    cols = []
    seen = {}
    for i, c in enumerate(df.columns):
        base = str(c).strip() if not pd.isna(c) and str(c).strip() != "" else f"col_{i}"
        k = seen.get(base, 0)
        seen[base] = k + 1
        cols.append(base if k == 0 else f"{base}.{k}")
    df.columns = cols

    # 5) унифицируем ключевую колонку литологии
    if "Lithology type" not in df.columns:
        if "Lithology name" in df.columns:
            df = df.rename(columns={"Lithology name": "Lithology type"})
        else:
            df = df.rename(columns={df.columns[0]: "Lithology type"})

    return df


def run_decompaction_por_perm(
    work_layers: pd.DataFrame,
    lithotypes: pd.DataFrame,
    rho_water: float = 1000.0,
    rho_mantle: float = 3300.0,
    basement_age_special: float | None = 260.0,
    basement_depth_km: float = 0.3,
) -> dict[str, pd.DataFrame]:
    """
    Главный “цикл” из вашего ноутбука (по age -> по слоям).
    Возвращает таблицы (как у вас: decompaction_df, decompaction_corrected_df, porosity_df, ...)
    """
    wl = work_layers.copy().reset_index(drop=True)

    # возраст (важно сохранить порядок по скважине)
    age_list = list(pd.to_numeric(wl["Age (Ma)"], errors="coerce").dropna().unique())

    n = len(wl)

    decompaction_df = pd.DataFrame()
    bottom_depth_km_df = pd.DataFrame()         # то, что у вас называлось decompaction_corrected_df (по факту — глубины)
    porosity_df = pd.DataFrame()
    density_df = pd.DataFrame()
    permeability_df = pd.DataFrame()
    sediment_rate_df = pd.DataFrame()
    vp_df = pd.DataFrame()
    vs_df = pd.DataFrame()
    density_column_df = pd.DataFrame()

    for i, age in enumerate(age_list):
        thickness_list = []
        bottom_depth_list = []
        por_list = []
        dens_list = []
        perm_list = []
        sedr_list = []
        vp_list = []
        vs_list = []
        density_column_values = []

        y2_d = None
        y2_d_corr = None

        for row_index, row in wl.iloc[i:].iterrows():
            lith = str(row["Lithology_type"])
            match = lithotypes[lithotypes["Lithology type"] == lith]
            if match.empty:
                raise ValueError(f"Литология '{lith}' не найдена в базе Lithotypes")

            # phi0 = float(match["Initial porosity"].iloc[0]) / 100.0

            phi0_raw = float(match["Initial porosity"].iloc[0])
            phi0 = phi0_raw / 100.0 if phi0_raw > 1.5 else phi0_raw   # если задано в %, делим на 100

            c = float(match["Athy factor k (depth)"].iloc[0])
            c = 1e-10 if c == 0 else c
            rho_grain = float(match["Density"].iloc[0])

            # (параметры проницаемости)
            if lith in LITHOLOGY_S_VALUES:
                S = float(LITHOLOGY_S_VALUES[lith])
            else:
                if "Specific surface" in match.columns:
                    sval = match["Specific surface"].iloc[0]
                    S = DEFAULT_S if pd.isna(sval) else (1e20 if float(sval) == 0 else float(sval))
                else:
                    S = DEFAULT_S
            k_factor = float(match["Permeability factor"].iloc[0]) if "Permeability factor" in match.columns else DEFAULT_K

            # м/км в ваших колонках:
            PWD_km = float(row.get("Paleobathymetry, Ma", 0.0)) / 1000.0
            eust_km = float(row.get("Sea level, m", 0.0)) / 1000.0

            y1 = float(row["Depth top, m"])      # уже в км (после ensure_depths_km)
            y2 = float(row["Depth bottom, m"])   # уже в км

            if row_index == i:
                y1_d = 0.0
                y1_d_corr = corrected(y1_d, rho_mantle, rho_water, PWD_km, eust_km) if age != 0 else 0.0
            else:
                y1_d = float(y2_d)
                y1_d_corr = float(y2_d_corr)

            if age == 0:
                y2_d = y2 + PWD_km
                y2_d_corr = y2_d
            elif basement_age_special is not None and float(age) == float(basement_age_special):
                y2_d_corr = abs(float(np.round(corrected(basement_depth_km, rho_mantle, rho_water, PWD_km, eust_km), 6)))
                y2_d = y2_d_corr
            else:
                y2_d = decomp_func(y1, y2, y1_d, phi0, c)
                y2_d_corr = corrected(y2_d, rho_mantle, rho_water, PWD_km, eust_km)

            thickness_km = y2_d - y1_d
            thickness_list.append(float(np.round(thickness_km, 6)))
            bottom_depth_list.append(float(np.round(y2_d_corr, 6)))

            phi = float(np.round(porosity_func(phi0, c, y1_d, y2_d, thickness_km), 6))
            por_list.append(phi)

            dens = float(np.round(bulk_density(phi, rho_grain, rho_water), 6))
            dens_list.append(dens)

            density_column_values.append(dens * thickness_km)

            perm = float(kozeny_carman_lithology(phi, S, k_factor))
            perm_list.append(perm)

            sedr_list.append(float(sedimentation_rate_mm_per_yr(y2_d, float(age))))

            # Vp/Vs (как у вас — через K/G из базы, если есть)
            K = float(match["Constant Value 2"].iloc[0]) if "Constant Value 2" in match.columns else 0.0
            G = float(match["Shear Modulus"].iloc[0]) if "Shear Modulus" in match.columns else 0.0
            if dens > 0 and K > 0 and G > 0:
                vp = np.sqrt((K * 1e6 + (4/3) * G * 1e6) / dens) / 1000.0
                vs = np.sqrt((G * 1e6) / dens) / 1000.0
            else:
                vp, vs = 0.0, 0.0
            vp_list.append(float(np.round(vp, 6)))
            vs_list.append(float(np.round(vs, 6)))

        col = f"{float(age)}"
        def right_align(values: list[float]) -> np.ndarray:
            arr = np.zeros(n, dtype=float)
            arr[n - len(values):] = np.asarray(values, dtype=float)
            return arr

        decompaction_df[col] = right_align(thickness_list)
        bottom_depth_km_df[col] = right_align(bottom_depth_list)
        porosity_df[col] = right_align(por_list)
        density_df[col] = right_align(dens_list)
        permeability_df[col] = right_align(perm_list)
        sediment_rate_df[col] = right_align(sedr_list)
        vp_df[col] = right_align(vp_list)
        vs_df[col] = right_align(vs_list)

        if thickness_list:
            dens_col = float(np.sum(density_column_values) / max(np.sum(thickness_list), 1e-12))
        else:
            dens_col = 0.0
        density_column_df[col] = [dens_col]

    return {
        "decompaction_thickness_km": decompaction_df,
        "bottom_depth_km": bottom_depth_km_df,
        "porosity": porosity_df,
        "density": density_df,
        "permeability": permeability_df,
        "sedimentation_rate_mm_yr": sediment_rate_df,
        "vp_km_s": vp_df,
        "vs_km_s": vs_df,
        "density_column_avg": density_column_df,
    }
