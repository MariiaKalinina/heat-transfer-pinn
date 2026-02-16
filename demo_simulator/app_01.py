# # app.py
# import io
# import difflib
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import streamlit as st

# from pathlib import Path

# LITHO_XLSX_PATH = Path(__file__).parent / "data" / "lithotypes.xlsx"

# from bm_core import (
#     normalize_inputs,
#     ensure_depths_km,
#     load_lithotypes_sheet,
#     run_decompaction_por_perm,
# )

# plt.rcParams["figure.dpi"] = 100
# plt.rcParams["savefig.dpi"] = 100


# st.set_page_config(page_title="Basin Modeling Demo Simulator", layout="wide")
# st.title("Demo simulator (Basin Modeling / Paleo / Thermal / Petroleum)")

# # ----------------------------
# # Helpers
# # ----------------------------
# def read_csv_smart(upload) -> pd.DataFrame:
#     raw = upload.getvalue()
#     try:
#         return pd.read_csv(io.BytesIO(raw), sep=";", header=0)
#     except Exception:
#         return pd.read_csv(io.BytesIO(raw), sep=",", header=0)

# def make_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     raw_cols = [str(c).strip() if not pd.isna(c) else "" for c in df.columns]
#     seen = {}
#     new_cols = []
#     for i, c in enumerate(raw_cols):
#         base = c if c else f"col_{i}"
#         k = seen.get(base, 0)
#         seen[base] = k + 1
#         new_cols.append(base if k == 0 else f"{base}.{k}")
#     df.columns = new_cols
#     return df

# def norm_txt(x) -> str:
#     return str(x).replace("\xa0", " ").strip().lower()

# # ----------------------------
# # Data loaders (Paleo)
# # ----------------------------
# @st.cache_data(show_spinner=False)
# def load_inputs_paleo(well_upload, assume_meters: bool):
#     df_well = read_csv_smart(well_upload)

#     if "Lithology_type" not in df_well.columns and "Lithology type" in df_well.columns:
#         df_well = df_well.rename(columns={"Lithology type": "Lithology_type"})

#     df_well = normalize_inputs(df_well)
#     df_well = ensure_depths_km(df_well, assume_meters=assume_meters)

#     if not LITHO_XLSX_PATH.exists():
#         raise FileNotFoundError(f"Bundled lithotypes file not found: {LITHO_XLSX_PATH}")

#     xl = pd.ExcelFile(LITHO_XLSX_PATH)
#     sheet_default = "Lithotypes" if "Lithotypes" in xl.sheet_names else xl.sheet_names[0]

#     lithotypes = load_lithotypes_sheet(LITHO_XLSX_PATH, sheet_default)
#     lithotypes = make_unique_columns(lithotypes)

#     # unify lithology column name
#     if "Lithology type" not in lithotypes.columns:
#         for cand in ["Lithology name", "Lithology", "Rock type"]:
#             if cand in lithotypes.columns:
#                 lithotypes = lithotypes.rename(columns={cand: "Lithology type"})
#                 break
#     if "Lithology type" not in lithotypes.columns and len(lithotypes.columns) > 0:
#         lithotypes = lithotypes.rename(columns={lithotypes.columns[0]: "Lithology type"})

#     # unify porosity column name
#     if "Initial porosity" not in lithotypes.columns:
#         for c in lithotypes.columns:
#             if "porosity" in c.lower():
#                 lithotypes = lithotypes.rename(columns={c: "Initial porosity"})
#                 break

#     return df_well, lithotypes, xl.sheet_names, sheet_default


# @st.cache_data(show_spinner=True)
# def run_paleo(df_well, lithotypes):
#     work_layers = df_well.iloc[1:].copy().reset_index(drop=True)

#     # FIX: drop rows with empty lithology
#     if "Lithology_type" in work_layers.columns:
#         work_layers["Lithology_type"] = work_layers["Lithology_type"].astype(str).str.strip()
#         work_layers = work_layers[work_layers["Lithology_type"].notna()]
#         work_layers = work_layers[~work_layers["Lithology_type"].isin(["", "nan", "NaN", "None"])].reset_index(drop=True)

#     res = run_decompaction_por_perm(
#         work_layers=work_layers,
#         lithotypes=lithotypes,
#         basement_age_special=None,
#         basement_depth_km=0.3,

#     )
#     return work_layers, res

# # ----------------------------
# # Sidebar: mode + inputs
# # ----------------------------
# MODE_PAL = "Paleogeometry reconstruction"
# MODE_THERM = "Thermal history reconstruction"
# MODE_PETRO = "Petroleum generation"
# MODE_FULL = "Full reconstruction (Paleo + Thermal + Petroleum)"

# with st.sidebar:
#     st.header("Run options")
#     mode = st.radio(
#         "Select module",
#         [MODE_PAL, MODE_THERM, MODE_PETRO, MODE_FULL],
#         index=0,
#     )

#     st.divider()
#     st.header("Inputs")
#     well_file = st.file_uploader("Well parameters (.csv)", type=["csv"])
#     # litho_xlsx = st.file_uploader("Lithotypes database (.xlsx)", type=["xlsx"])
#     assume_meters = st.checkbox("Depth columns are in meters (convert to km)", value=True)

#     st.divider()
#     run_btn = st.button("Run", type="primary")

# # ----------------------------
# # Main routing by mode
# # ----------------------------
# if not well_file:
#     st.info("Upload Well parameters CSV (left sidebar).")
#     st.stop()

# # Load well + bundled lithotypes ONCE (for all modes)
# df_well, lithotypes, sheet_names, sheet_used = load_inputs_paleo(well_file, assume_meters)

# # Common checks (needed for PALEO and FULL)
# def validate_paleo_inputs():
#     required = ["Lithology type", "Initial porosity"]
#     missing_cols = [c for c in required if c not in lithotypes.columns]
#     if missing_cols:
#         st.error("Lithotypes missing required columns: " + ", ".join(missing_cols))
#         st.write("Lithotypes columns (first 50):", list(lithotypes.columns)[:50])
#         st.stop()

#     if "Lithology_type" in df_well.columns and "Lithology type" in lithotypes.columns:
#         well_lith = sorted(set(df_well["Lithology_type"].dropna().map(norm_txt)))
#         db_lith = sorted(set(lithotypes["Lithology type"].dropna().map(norm_txt)))
#         missing = [x for x in well_lith if x not in db_lith]
#         if missing:
#             st.error("These lithologies from Well CSV are NOT found in Lithotypes:")
#             st.write(missing)
#             suggestions = {m: difflib.get_close_matches(m, db_lith, n=5, cutoff=0.5) for m in missing}
#             st.caption("Closest matches (suggestions):")
#             st.write(suggestions)
#             st.stop()

# # ----------------------------
# # Mode pages
# # ----------------------------
# if mode == MODE_PAL:
#     validate_paleo_inputs()

#     st.subheader("Preview: inputs")
#     c1, c2 = st.columns(2)
#     with c1:
#         st.caption("Well parameters")
#         st.dataframe(df_well.head(20), use_container_width=True)
#     with c2:
#         st.caption(f"Lithotypes (sheet: {sheet_used})")
#         preview_cols = [c for c in [
#             "Lithology type",
#             "Initial porosity",
#             "Athy factor k (depth)",
#             "Density",
#             "Specific surface",
#             "Permeability factor",
#             "Constant Value 2",
#             "Shear Modulus",
#         ] if c in lithotypes.columns]
#         st.dataframe(lithotypes[preview_cols].head(20) if preview_cols else lithotypes.head(20), use_container_width=True)

#     if run_btn:
#         work_layers, res = run_paleo(df_well, lithotypes)
#         st.success("Paleogeometry reconstruction completed.")

#         ages = [float(a) for a in res["porosity"].columns]
#         ages_sorted = sorted(ages)
#         age_pick = st.selectbox("Select age (Ma)", ages_sorted, index=0)
#         col = f"{float(age_pick)}"

#         tabs = st.tabs(["Porosity", "Permeability", "Bottom depth", "Vp/Vs", "Downloads"])

#         with tabs[0]:
#             depth = res["bottom_depth_km"][col].values
#             phi = res["porosity"][col].values
#             mask = depth > 0
#             depth, phi = depth[mask], phi[mask]

#             fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=100)
#             ax.plot(phi, depth, marker="o")
#             ax.invert_yaxis()
#             ax.set_xlabel("Porosity")
#             ax.set_ylabel("Depth (km)")
#             ax.grid(True, linestyle="--", linewidth=0.5)
#             st.pyplot(fig, use_container_width=False)

#             st.dataframe(pd.DataFrame({"depth_km": depth, "porosity": phi}), use_container_width=True)

#         with tabs[1]:
#             depth = res["bottom_depth_km"][col].values
#             perm = res["permeability"][col].values
#             mask = depth > 0
#             depth, perm = depth[mask], perm[mask]

#             fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=100)
#             ax.plot(perm, depth, marker="o")
#             ax.set_xscale("log")
#             ax.invert_yaxis()
#             ax.set_xlabel("Permeability (log)")
#             ax.set_ylabel("Depth (km)")
#             ax.grid(True, linestyle="--", linewidth=0.5)
#             st.pyplot(fig, use_container_width=False)

#         with tabs[2]:
#             st.dataframe(res["bottom_depth_km"], use_container_width=True)

#         with tabs[3]:
#             depth = res["bottom_depth_km"][col].values
#             vp = res["vp_km_s"][col].values
#             vs = res["vs_km_s"][col].values
#             mask = depth > 0
#             depth, vp, vs = depth[mask], vp[mask], vs[mask]

#             fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=100)
#             ax.plot(vp, depth, marker="o", label="Vp")
#             ax.plot(vs, depth, marker="o", label="Vs")
#             ax.invert_yaxis()
#             ax.set_xlabel("km/s")
#             ax.set_ylabel("Depth (km)")
#             ax.grid(True, linestyle="--", linewidth=0.5)
#             ax.legend()
#             st.pyplot(fig, use_container_width=False)

#         with tabs[4]:
#             def dl(name: str, df: pd.DataFrame):
#                 csv = df.to_csv(index=False).encode("utf-8")
#                 st.download_button(
#                     f"Download {name}.csv",
#                     data=csv,
#                     file_name=f"{name}.csv",
#                     mime="text/csv",
#                 )

#             dl("porosity", res["porosity"])
#             dl("permeability", res["permeability"])
#             dl("bottom_depth_km", res["bottom_depth_km"])
#             dl("decompaction_thickness_km", res["decompaction_thickness_km"])
#             dl("vp_km_s", res["vp_km_s"])
#             dl("vs_km_s", res["vs_km_s"])
#             dl("sedimentation_rate_mm_yr", res["sedimentation_rate_mm_yr"])

# elif mode == MODE_THERM:
#     st.info("Thermal history reconstruction: code will be added here.")
#     if run_btn:
#         st.warning("Not implemented yet.")

# elif mode == MODE_PETRO:
#     st.info("Petroleum generation: code will be added here.")
#     if run_btn:
#         st.warning("Not implemented yet.")

# elif mode == MODE_FULL:
#     validate_paleo_inputs()
#     st.info("Full reconstruction selected (Paleo + Thermal + Petroleum).")

#     if run_btn:
#         work_layers, res = run_paleo(df_well, lithotypes)
#         st.success("Paleogeometry completed (part of Full).")
#         st.warning("Thermal + Petroleum not implemented yet.")




# app.py
import io
import difflib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import zipfile


from pathlib import Path

LITHO_XLSX_PATH = Path(__file__).parent / "data" / "lithotypes.xlsx"

from bm_core import (
    normalize_inputs,
    ensure_depths_km,
    load_lithotypes_sheet,
    run_decompaction_por_perm,
)

plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 100


st.set_page_config(page_title="Basin Modeling Demo-Simulator", layout="wide")
st.title("Basin Simulation Demo")

# ----------------------------
# Helpers
# ----------------------------
def read_csv_smart(upload) -> pd.DataFrame:
    raw = upload.getvalue()
    try:
        return pd.read_csv(io.BytesIO(raw), sep=";", header=0)
    except Exception:
        return pd.read_csv(io.BytesIO(raw), sep=",", header=0)

def make_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    raw_cols = [str(c).strip() if not pd.isna(c) else "" for c in df.columns]
    seen = {}
    new_cols = []
    for i, c in enumerate(raw_cols):
        base = c if c else f"col_{i}"
        k = seen.get(base, 0)
        seen[base] = k + 1
        new_cols.append(base if k == 0 else f"{base}.{k}")
    df.columns = new_cols
    return df

def norm_txt(x) -> str:
    return str(x).replace("\xa0", " ").strip().lower()

def norm_colname(c: str) -> str:
    return str(c).strip().lower().replace(" ", "_")

HIDE_WELL_COLS = {
    "well",
    "paleobathymetry",
    "sea_level",
    "sea level",
    "pse",
    "kenetic",
    "kinetic",
    "toc",
    "hi",
}

def well_preview_df(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [c for c in df.columns if norm_colname(c) not in {norm_colname(x) for x in HIDE_WELL_COLS}]
    return df[keep_cols]

def read_table_optional(upload) -> pd.DataFrame:
    """Reads CSV/XLSX from st.file_uploader; returns empty df if upload is None."""
    if upload is None:
        return pd.DataFrame()
    name = (upload.name or "").lower()
    raw = upload.getvalue()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(raw))
    # csv: try ; then ,
    try:
        return pd.read_csv(io.BytesIO(raw), sep=";")
    except Exception:
        return pd.read_csv(io.BytesIO(raw), sep=",")


# ---- Paleo-style tables + evolution plots (reusable) ----
def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def plot_evolution(df_values: pd.DataFrame, title: str, ylabel: str):
    ages = [float(c) for c in df_values.columns]
    order = np.argsort(ages)
    ages_sorted = np.array(ages)[order]

    fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=100)
    cmap = plt.get_cmap("turbo")
    n = df_values.shape[0]

    for i in range(n):
        y = df_values.iloc[i, order].astype(float).values
        ax.plot(ages_sorted, y, marker="o", linewidth=1.5, color=cmap(i / max(n - 1, 1)))

    ax.set_title(title)
    ax.set_xlabel("Age (Ma)")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.5)

    # you asked: reverse Y for each property
    ax.invert_yaxis()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max(n - 1, 1)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Layer index (0=shallowest)")
    return fig

def render_paleo_results(work_layers: pd.DataFrame, res: dict, prefix: str = "paleo"):
    lith_col = pick_col(work_layers, ["Lithology_type", "Lithology type", "Lithology"])
    event_col = pick_col(work_layers, ["Event_name", "Event name", "Event", "Layer", "Name", "Layer_name"])

    meta = pd.DataFrame({
        "Lithology_type": work_layers[lith_col].astype(str) if lith_col else "",
        "Event_name": work_layers[event_col].astype(str) if event_col else [f"Layer {i+1}" for i in range(len(work_layers))],
    })

    PROPS = {
        "Burial depth":     {"key": "bottom_depth_km", "scale": 1.0, "ylabel": "Burial depth (km)"},
        "Layer thickness":  {"key": "decompaction_thickness_km", "scale": 1.0, "ylabel": "Thickness (km)"},
        "Pore volume":      {"key": "porosity", "scale": 100.0, "ylabel": "Porosity (%)"},
    }

    st.subheader("Reconstruction results")

    # keep bytes to build a global ZIP once at the bottom (optional)
    all_downloads = {}

    tabs = st.tabs(list(PROPS.keys()))
    for tab, (title, cfg) in zip(tabs, PROPS.items()):
        with tab:
            key = cfg["key"]
            if key not in res:
                st.warning(f"'{key}' is not available in results yet.")
                continue

            df_prop = res[key].copy() * cfg["scale"]

            # table
            table_df = pd.concat([meta.reset_index(drop=True), df_prop.reset_index(drop=True)], axis=1)
            st.caption("Calculation result table")
            st.dataframe(table_df, use_container_width=True)

            # plot
            st.caption("Evolution plot (value vs Age, one curve per layer)")
            fig = plot_evolution(df_prop, title=title, ylabel=cfg["ylabel"])
            st.pyplot(fig, use_container_width=False)

            # ---- per-property downloads (ONLY for this property) ----
            csv_bytes = table_df.to_csv(index=False).encode("utf-8")

            png_buf = io.BytesIO()
            fig.savefig(png_buf, format="png", dpi=150, bbox_inches="tight")
            png_bytes = png_buf.getvalue()

            c1, c2, c3 = st.columns(3)
            with c1:
                st.download_button(
                    f"Download {title} table (CSV)",
                    data=csv_bytes,
                    file_name=f"{key}.csv",
                    mime="text/csv",
                    key=f"{prefix}_csv_{key}",
                )
            with c2:
                st.download_button(
                    f"Download {title} plot (PNG)",
                    data=png_bytes,
                    file_name=f"{key}.png",
                    mime="image/png",
                    key=f"{prefix}_png_{key}",
                )
            with c3:
                zip_one = io.BytesIO()
                with zipfile.ZipFile(zip_one, "w", compression=zipfile.ZIP_DEFLATED) as z:
                    z.writestr(f"{key}.csv", csv_bytes)
                    z.writestr(f"{key}.png", png_bytes)
                st.download_button(
                    f"Download {title} (ZIP)",
                    data=zip_one.getvalue(),
                    file_name=f"{key}.zip",
                    mime="application/zip",
                    key=f"{prefix}_zip_{key}",
                )

            # store for optional "download all" below
            all_downloads[key] = {"csv": csv_bytes, "png": png_bytes}

    # ---- bottom of page: optional "download ALL properties" ----
    if all_downloads:
        st.divider()
        st.subheader("Download all properties (optional)")

        zip_all = io.BytesIO()
        with zipfile.ZipFile(zip_all, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for k, d in all_downloads.items():
                z.writestr(f"{k}.csv", d["csv"])
                z.writestr(f"{k}.png", d["png"])

        st.download_button(
            "Download ALL (tables + plots) as ZIP",
            data=zip_all.getvalue(),
            file_name=f"{prefix}_outputs.zip",
            mime="application/zip",
            key=f"{prefix}_zip_all",
        )


# ----------------------------
# Data loaders (Paleo)
# ----------------------------
@st.cache_data(show_spinner=False)
def load_inputs_paleo(well_upload, assume_meters: bool):
    df_well = read_csv_smart(well_upload)

    if "Lithology_type" not in df_well.columns and "Lithology type" in df_well.columns:
        df_well = df_well.rename(columns={"Lithology type": "Lithology_type"})

    df_well = normalize_inputs(df_well)
    df_well = ensure_depths_km(df_well, assume_meters=assume_meters)

    if not LITHO_XLSX_PATH.exists():
        raise FileNotFoundError(f"Bundled lithotypes file not found: {LITHO_XLSX_PATH}")

    xl = pd.ExcelFile(LITHO_XLSX_PATH)
    sheet_default = "Lithotypes" if "Lithotypes" in xl.sheet_names else xl.sheet_names[0]

    lithotypes = load_lithotypes_sheet(LITHO_XLSX_PATH, sheet_default)
    lithotypes = make_unique_columns(lithotypes)

    # unify lithology column name
    if "Lithology type" not in lithotypes.columns:
        for cand in ["Lithology name", "Lithology", "Rock type"]:
            if cand in lithotypes.columns:
                lithotypes = lithotypes.rename(columns={cand: "Lithology type"})
                break
    if "Lithology type" not in lithotypes.columns and len(lithotypes.columns) > 0:
        lithotypes = lithotypes.rename(columns={lithotypes.columns[0]: "Lithology type"})

    # unify porosity column name
    if "Initial porosity" not in lithotypes.columns:
        for c in lithotypes.columns:
            if "porosity" in c.lower():
                lithotypes = lithotypes.rename(columns={c: "Initial porosity"})
                break

    return df_well, lithotypes, xl.sheet_names, sheet_default


@st.cache_data(show_spinner=True)
def run_paleo(df_well, lithotypes):
    work_layers = df_well.iloc[1:].copy().reset_index(drop=True)

    # FIX: drop rows with empty lithology
    if "Lithology_type" in work_layers.columns:
        work_layers["Lithology_type"] = work_layers["Lithology_type"].astype(str).str.strip()
        work_layers = work_layers[work_layers["Lithology_type"].notna()]
        work_layers = work_layers[~work_layers["Lithology_type"].isin(["", "nan", "NaN", "None"])].reset_index(drop=True)

    res = run_decompaction_por_perm(
        work_layers=work_layers,
        lithotypes=lithotypes,
        basement_age_special=None,
        basement_depth_km=0.3,

    )
    return work_layers, res

# ----------------------------
# Sidebar: mode + inputs
# ----------------------------
MODE_PAL = "Paleogeometry reconstruction"
MODE_THERM = "Thermal history reconstruction"
MODE_PETRO = "Petroleum generation"
MODE_FULL = "Full reconstruction (Paleo + Thermal + Petroleum)"

with st.sidebar:
    st.header("Run options")
    mode = st.radio(
        "Select module",
        [MODE_PAL, MODE_THERM, MODE_PETRO, MODE_FULL],
        index=0,
    )

    st.divider()
    st.header("Inputs")
    well_file = st.file_uploader("Well parameters (.csv)", type=["csv"])
    # litho_xlsx = st.file_uploader("Lithotypes database (.xlsx)", type=["xlsx"])
    assume_meters = st.checkbox("Depth columns are in meters (convert to km)", value=True)

    st.divider()
    run_btn = st.button("Run", type="primary")

# ----------------------------
# Main routing by mode
# ----------------------------
if not well_file:
    st.info("Upload Well parameters CSV (left sidebar).")
    st.stop()

# Load well + bundled lithotypes ONCE (for all modes)
df_well, lithotypes, sheet_names, sheet_used = load_inputs_paleo(well_file, assume_meters)

# Common checks (needed for PALEO and FULL)
def validate_paleo_inputs():
    required = ["Lithology type", "Initial porosity"]
    missing_cols = [c for c in required if c not in lithotypes.columns]
    if missing_cols:
        st.error("Lithotypes missing required columns: " + ", ".join(missing_cols))
        st.write("Lithotypes columns (first 50):", list(lithotypes.columns)[:50])
        st.stop()

    if "Lithology_type" in df_well.columns and "Lithology type" in lithotypes.columns:
        well_lith = sorted(set(df_well["Lithology_type"].dropna().map(norm_txt)))
        db_lith = sorted(set(lithotypes["Lithology type"].dropna().map(norm_txt)))
        missing = [x for x in well_lith if x not in db_lith]
        if missing:
            st.error("These lithologies from Well CSV are NOT found in Lithotypes:")
            st.write(missing)
            suggestions = {m: difflib.get_close_matches(m, db_lith, n=5, cutoff=0.5) for m in missing}
            st.caption("Closest matches (suggestions):")
            st.write(suggestions)
            st.stop()

# ----------------------------
# Mode pages
# ----------------------------
if mode == MODE_PAL:
    validate_paleo_inputs()

    st.subheader("Preview: inputs")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Well parameters")
        st.dataframe(well_preview_df(df_well).head(20), use_container_width=True)

        # -------- Optional parameters (do not affect Paleo if empty) --------
        with st.expander("Optional parameters (Thermal/Petroleum) — app still runs if empty", expanded=False):

            st.markdown("These inputs are optional. Paleogeometry reconstruction will work even if you leave them empty.")

            # SWIT #1 and #2 (latitude)
            sw1, sw2 = st.columns(2)
            with sw1:
                st.markdown("**SWIT #1 (latitude)**")
                swit1_ns = st.selectbox("N/S", ["N", "S"], key="swit1_ns")
                swit1_deg = st.number_input("Degrees (0–90)", min_value=0.0, max_value=90.0, value=0.0, step=0.5, key="swit1_deg")
            with sw2:
                st.markdown("**SWIT #2 (latitude)**")
                swit2_ns = st.selectbox("N/S ", ["N", "S"], key="swit2_ns")
                swit2_deg = st.number_input("Degrees (0–90) ", min_value=0.0, max_value=90.0, value=0.0, step=0.5, key="swit2_deg")

            st.divider()

            # Upload tables
            pwd_file = st.file_uploader("PWD table (CSV/XLSX) — optional", type=["csv", "xlsx"], key="pwd_file")
            hf_file  = st.file_uploader("Heat flow table (CSV/XLSX) — optional", type=["csv", "xlsx"], key="hf_file")

            # Optional previews (does not affect run)
            if pwd_file is not None:
                try:
                    st.caption("PWD preview")
                    st.dataframe(read_table_optional(pwd_file).head(10), use_container_width=True)
                except Exception as e:
                    st.warning(f"PWD file could not be read: {e}")

            if hf_file is not None:
                try:
                    st.caption("Heat flow preview")
                    st.dataframe(read_table_optional(hf_file).head(10), use_container_width=True)
                except Exception as e:
                    st.warning(f"Heat flow file could not be read: {e}")

            st.divider()

            # Kinetics layer selection from Event_name
            event_col = pick_col(df_well, ["Event_name", "Event name", "Event", "Layer", "Name", "Layer_name"])
            event_options = []
            if event_col:
                event_options = df_well[event_col].dropna().astype(str).unique().tolist()

            kinetics_layer = st.selectbox(
                "Kinetics: choose layer (Event_name) — optional",
                options=event_options if event_options else ["(Event_name not found in CSV)"],
                key="kinetics_layer",
            )

            # HI & TOC
            toc = st.number_input("TOC (wt%) — optional", min_value=0.0, value=0.0, step=0.1, key="toc_val")
            hi  = st.number_input("HI — optional", min_value=0.0, value=0.0, step=1.0, key="hi_val")

    with c2:
        st.caption(f"Lithotypes (sheet: {sheet_used})")
        preview_cols = [c for c in [
            "Lithology type",
            "Initial porosity",
            "Athy factor k (depth)",
            "Density",
            "Specific surface",
            "Permeability factor",
            "Constant Value 2",
            "Shear Modulus",
        ] if c in lithotypes.columns]
        st.dataframe(lithotypes[preview_cols].head(20) if preview_cols else lithotypes.head(20), use_container_width=True)

    if run_btn:
        work_layers, res = run_paleo(df_well, lithotypes)
        render_paleo_results(work_layers, res, prefix="paleo")


        # --- helper: pick metadata columns from work_layers ---
        def pick_col(df, candidates):
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        lith_col = pick_col(work_layers, ["Lithology_type", "Lithology type", "Lithology"])
        event_col = pick_col(work_layers, ["Event_name", "Event name", "Event", "Layer", "Name", "Layer_name"])

        meta = pd.DataFrame({
            "Lithology_type": work_layers[lith_col].astype(str) if lith_col else "",
            "Event_name": work_layers[event_col].astype(str) if event_col else [f"Layer {i+1}" for i in range(len(work_layers))],
        })

        # --- properties you want to show in paleo ---
        # res keys you already have:
        #   bottom_depth_km, decompaction_thickness_km, porosity, permeability, vp_km_s, vs_km_s, ...
        PROPS = {
            "Burial depth": {"key": "bottom_depth_km", "scale": 1.0, "ylabel": "Burial depth (km)", "invert_y": True},
            "Layer thickness":       {"key": "decompaction_thickness_km", "scale": 1.0, "ylabel": "Thickness (km)", "invert_y": True},
            "Pore volume": {"key": "porosity", "scale": 100.0, "ylabel": "Porosity (%)", "invert_y": True},
            # add more later if you want:
            # "Permeability": {"key":"permeability", "scale": 1.0, "ylabel":"Permeability", "invert_y": False},
        }

        def plot_evolution(df_values: pd.DataFrame, title: str, ylabel: str, invert_y: bool):
            ages = [float(c) for c in df_values.columns]
            order = np.argsort(ages)  # left->right increasing age
            ages_sorted = np.array(ages)[order]

            fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=100)

            cmap = plt.get_cmap("turbo")
            n = df_values.shape[0]
            for i in range(n):
                y = df_values.iloc[i, order].astype(float).values
                color = cmap(i / max(n - 1, 1))
                ax.plot(ages_sorted, y, marker="o", linewidth=1.5, color=color)

            ax.set_title(title)
            ax.set_xlabel("Age (Ma)")
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", linewidth=0.5)

            if invert_y:
                ax.invert_yaxis()

            # colorbar = layer index
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max(n - 1, 1)))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label("Layer index (0=shallowest)")

            return fig

        # st.subheader("Paleogeometry reconstruction results")

        # tabs = st.tabs(list(PROPS.keys()))

        # for tab, (title, cfg) in zip(tabs, PROPS.items()):
        #     with tab:
        #         key = cfg["key"]
        #         scale = cfg["scale"]

        #         if key not in res:
        #             st.warning(f"'{key}' is not available in results yet.")
        #             continue

        #         df_prop = res[key].copy() * scale

        #         # ---- TABLE (meta + ages) ----
        #         table_df = pd.concat([meta.reset_index(drop=True), df_prop.reset_index(drop=True)], axis=1)
        #         st.caption("Calculation result table")
        #         st.dataframe(table_df, use_container_width=True)

        #         # Download table
        #         csv = table_df.to_csv(index=False).encode("utf-8")
        #         st.download_button(
        #             f"Download table: {title}.csv",
        #             data=csv,
        #             file_name=f"{cfg['key']}.csv",
        #             mime="text/csv",
        #         )

        #         # ---- EVOLUTION PLOT ----
        #         st.caption("Evolution plot (value vs Age, one curve per layer)")
        #         fig = plot_evolution(df_prop, title=title, ylabel=cfg["ylabel"], invert_y=cfg["invert_y"])
        #         st.pyplot(fig, use_container_width=False)


elif mode == MODE_THERM:
    st.info("Thermal history reconstruction: code will be added here.")
    if run_btn:
        st.warning("Not implemented yet.")

elif mode == MODE_PETRO:
    st.info("Petroleum generation: code will be added here.")
    if run_btn:
        st.warning("Not implemented yet.")

elif mode == MODE_FULL:
    validate_paleo_inputs()
    st.info("Full reconstruction selected (Paleo + Thermal + Petroleum).")

    if run_btn:
        work_layers, res = run_paleo(df_well, lithotypes)
        # st.success("Paleogeometry completed (part of Full).")
        render_paleo_results(work_layers, res, prefix="full_paleo")

        st.warning("Thermal + Petroleum not implemented yet.")



