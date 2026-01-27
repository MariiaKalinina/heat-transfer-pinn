# app.py
import io
import difflib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from bm_core import (
    normalize_inputs,
    ensure_depths_km,
    load_lithotypes_sheet,
    run_decompaction_por_perm,
)

plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 100

st.set_page_config(page_title="Basin Modeling Demo Simulator", layout="wide")
st.title("Demo simulator (Basin Modeling / Paleo reconstruction)")

with st.sidebar:
    st.header("Inputs")
    well_file = st.file_uploader("Well parameters (.csv)", type=["csv"])
    litho_xlsx = st.file_uploader("Lithotypes database (.xlsx)", type=["xlsx"])
    assume_meters = st.checkbox("Depth columns are in meters (convert to km)", value=True)

    st.divider()
    st.header("Run controls")
    basement_age = st.number_input("Special basement age (Ma) (optional)", value=260.0, step=1.0)
    basement_depth_km = st.number_input("Basement depth used in special case (km)", value=0.3, step=0.1)
    run_btn = st.button("Run simulation", type="primary")


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


@st.cache_data(show_spinner=False)
def load_inputs(well_upload, litho_upload, assume_meters: bool):
    df_well = read_csv_smart(well_upload)

    # важно: переименовать ДО normalize_inputs, чтобы маппинг литологий применился
    if "Lithology_type" not in df_well.columns and "Lithology type" in df_well.columns:
        df_well = df_well.rename(columns={"Lithology type": "Lithology_type"})

    df_well = normalize_inputs(df_well)
    df_well = ensure_depths_km(df_well, assume_meters=assume_meters)

    xl = pd.ExcelFile(litho_upload)
    sheet_default = "Lithotypes" if "Lithotypes" in xl.sheet_names else xl.sheet_names[0]
    lithotypes = load_lithotypes_sheet(litho_upload, sheet_default)
    lithotypes = make_unique_columns(lithotypes)

    # привести имя колонки литологии к ожидаемому
    if "Lithology type" not in lithotypes.columns:
        for cand in ["Lithology name", "Lithology", "Rock type"]:
            if cand in lithotypes.columns:
                lithotypes = lithotypes.rename(columns={cand: "Lithology type"})
                break
    if "Lithology type" not in lithotypes.columns and len(lithotypes.columns) > 0:
        lithotypes = lithotypes.rename(columns={lithotypes.columns[0]: "Lithology type"})

    # попытаться найти колонку пористости
    if "Initial porosity" not in lithotypes.columns:
        for c in lithotypes.columns:
            if "porosity" in c.lower():
                lithotypes = lithotypes.rename(columns={c: "Initial porosity"})
                break

    return df_well, lithotypes, xl.sheet_names, sheet_default

@st.cache_data(show_spinner=True)
def run_model(df_well: pd.DataFrame, lithotypes: pd.DataFrame, basement_age: float, basement_depth_km: float):
    # work_layers: как в ноутбуке — пропускаем первый слой (эрозия/вода) если он у вас есть
    work_layers = df_well.iloc[1:].copy().reset_index(drop=True)

    # --- FIX: убрать строки без литологии ---
    if "Lithology_type" in work_layers.columns:
        work_layers["Lithology_type"] = work_layers["Lithology_type"].astype(str).str.strip()
        work_layers = work_layers[work_layers["Lithology_type"].notna()]
        work_layers = work_layers[~work_layers["Lithology_type"].isin(["", "nan", "NaN", "None"])].reset_index(drop=True)

    res = run_decompaction_por_perm(
        work_layers=work_layers,
        lithotypes=lithotypes,
        basement_age_special=basement_age if basement_age else None,
        basement_depth_km=basement_depth_km,
    )
    return work_layers, res



# --- UI logic ---
if well_file and litho_xlsx:
    df_well, lithotypes, sheet_names, sheet_used = load_inputs(well_file, litho_xlsx, assume_meters)

    # 1) быстрые проверки обязательных колонок в lithotypes
    required = ["Lithology type", "Initial porosity"]
    missing_cols = [c for c in required if c not in lithotypes.columns]
    if missing_cols:
        st.error("В Lithotypes не найдены нужные колонки: " + ", ".join(missing_cols))
        st.write("Колонки в Lithotypes (первые 50):", list(lithotypes.columns)[:50])
        st.stop()

    # 2) диагностика: какие литологии из Well не находятся в базе
    if "Lithology_type" in df_well.columns and "Lithology type" in lithotypes.columns:
        well_lith = sorted(set(df_well["Lithology_type"].dropna().map(norm_txt)))
        db_lith = sorted(set(lithotypes["Lithology type"].dropna().map(norm_txt)))
        missing = [x for x in well_lith if x not in db_lith]
        if missing:
            st.error("Эти литологии из Well CSV НЕ найдены в Lithotypes:")
            st.write(missing)
            suggestions = {m: difflib.get_close_matches(m, db_lith, n=5, cutoff=0.5) for m in missing}
            st.caption("Ближайшие совпадения (подсказки):")
            st.write(suggestions)
            st.stop()

    st.subheader("Preview: inputs")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Well parameters")
        st.dataframe(df_well.head(20), use_container_width=True)
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
        work_layers, res = run_model(df_well, lithotypes, basement_age, basement_depth_km)
        st.success("Done. Results are below.")

        ages = [float(a) for a in res["porosity"].columns]
        ages_sorted = sorted(ages)
        age_pick = st.selectbox("Select age (Ma)", ages_sorted, index=0)
        col = f"{float(age_pick)}"

        tabs = st.tabs(["Porosity", "Permeability", "Bottom depth", "Vp/Vs", "Downloads"])

        with tabs[0]:
            depth = res["bottom_depth_km"][col].values
            phi = res["porosity"][col].values
            mask = depth > 0
            depth, phi = depth[mask], phi[mask]

            fig, ax = plt.subplots(figsize=(5, 7))
            ax.plot(phi, depth, marker="o")
            ax.invert_yaxis()
            ax.set_xlabel("Porosity (fraction)")
            ax.set_ylabel("Depth (km)")
            ax.grid(True, linestyle="--", linewidth=0.5)
            st.pyplot(fig, use_container_width=False)

            st.dataframe(pd.DataFrame({"depth_km": depth, "porosity": phi}), use_container_width=True)

        with tabs[1]:
            depth = res["bottom_depth_km"][col].values
            perm = res["permeability"][col].values
            mask = depth > 0
            depth, perm = depth[mask], perm[mask]

            fig, ax = plt.subplots(figsize=(5, 7))
            ax.plot(perm, depth, marker="o")
            ax.set_xscale("log")
            ax.invert_yaxis()
            ax.set_xlabel("Permeability (model units, log scale)")
            ax.set_ylabel("Depth (km)")
            ax.grid(True, linestyle="--", linewidth=0.5)
            st.pyplot(fig, use_container_width=False)

        with tabs[2]:
            st.dataframe(res["bottom_depth_km"], use_container_width=True)

        with tabs[3]:
            depth = res["bottom_depth_km"][col].values
            vp = res["vp_km_s"][col].values
            vs = res["vs_km_s"][col].values
            mask = depth > 0
            depth, vp, vs = depth[mask], vp[mask], vs[mask]

            fig, ax = plt.subplots(figsize=(6, 7))
            ax.plot(vp, depth, marker="o", label="Vp")
            ax.plot(vs, depth, marker="o", label="Vs")
            ax.invert_yaxis()
            ax.set_xlabel("km/s")
            ax.set_ylabel("Depth (km)")
            ax.grid(True, linestyle="--", linewidth=0.5)
            ax.legend()
            st.pyplot(fig, use_container_width=False)

        with tabs[4]:
            def dl(name: str, df: pd.DataFrame):
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    f"Download {name}.csv",
                    data=csv,
                    file_name=f"{name}.csv",
                    mime="text/csv",
                )

            dl("porosity", res["porosity"])
            dl("permeability", res["permeability"])
            dl("bottom_depth_km", res["bottom_depth_km"])
            dl("decompaction_thickness_km", res["decompaction_thickness_km"])
            dl("vp_km_s", res["vp_km_s"])
            dl("vs_km_s", res["vs_km_s"])
            dl("sedimentation_rate_mm_yr", res["sedimentation_rate_mm_yr"])

else:
    st.info("Загрузите Well parameters CSV и Lithotypes XLSX (в левом сайдбаре).")
