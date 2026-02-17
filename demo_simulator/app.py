import io
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import streamlit.components.v1 as components

import base64


import zipfile


from bm_core import (
    ensure_depths_km,
    load_lithotypes_sheet,
    normalize_inputs,
    run_decompaction_por_perm,
)

plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 100

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
LITHO_XLSX_PATH = DATA_DIR / "lithotypes.xlsx"
DEMO_WELL_PATH = DATA_DIR / "Well_parametrs.csv"


# ----------------------------
# Helpers
# ----------------------------
def read_csv_smart(upload) -> pd.DataFrame:
    raw = upload.getvalue()
    try:
        return pd.read_csv(io.BytesIO(raw), sep=";", header=0)
    except Exception:
        return pd.read_csv(io.BytesIO(raw), sep=",", header=0)

def load_porosity_table_csv(path: Path) -> pd.DataFrame | None:
    """Читает porosity_table.csv из data/ и приводит к виду Layer × Age."""
    if not path.exists():
        return None

    # Твой CSV, судя по скрину, может быть с запятыми/точками и странными разделителями.
    # Пробуем несколько вариантов.
    candidates = [
        dict(sep=";", decimal=","),
        dict(sep=";", decimal="."),
        dict(sep=",", decimal="."),
        dict(sep=",", decimal=","),
    ]

    last_err = None
    for kwargs in candidates:
        try:
            df = pd.read_csv(path, index_col=0, **kwargs)

            # Пытаемся сделать колонки числами (возраст) и отсортировать 260..0
            try:
                cols_num = pd.to_numeric(df.columns, errors="coerce")
                if cols_num.notna().all():
                    df.columns = cols_num.astype(int)
                    df = df.reindex(sorted(df.columns, reverse=True), axis=1)
            except Exception:
                pass

            df.index.name = "Layer"
            return df

        except Exception as e:
            last_err = e

    # если ни один вариант не сработал
    st.warning(f"Не удалось прочитать {path.name}: {last_err}")
    return None


def fade_in_css() -> str:
    return """
    <style>
      @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0px); }
      }
      .splash-title {
        font-size: 56px;
        font-weight: 800;
        letter-spacing: 0.2px;
        margin-top: 10vh;
        animation: fadeInUp 1.0s ease-out forwards;
      }
      .splash-sub {
        font-size: 18px;
        opacity: 0.75;
        margin-top: 14px;
        animation: fadeInUp 1.2s ease-out forwards;
      }
      section[data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem;
      }
    </style>
    """


def show_splash_then_continue():
    if st.session_state.get("splash_done", False):
        return

    st.markdown(fade_in_css(), unsafe_allow_html=True)
    st.markdown('<div class="splash-title">Basin Simulation Demo</div>', unsafe_allow_html=True)
    st.markdown('<div class="splash-sub">Демо интерфейс бассейного симулятора</div>', unsafe_allow_html=True)

    time.sleep(2.0)
    st.session_state["splash_done"] = True
    st.rerun()


def col_pick(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df is None:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None


def get_well_name(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "Скважина"
    c = col_pick(df, ["Well", "well", "WELL"])
    if c is None:
        return "Скважина"
    val = str(df[c].iloc[0])
    if not val or val.lower() == "nan":
        return "Скважина"
    return val


def get_layer_options(df_well: pd.DataFrame) -> list[str]:
    if df_well is None or df_well.empty:
        return []
    c_ev = col_pick(df_well, ["Event_name", "event_name", "Event name", "Event"])
    if c_ev is None:
        return []
    s = df_well.loc[1:, c_ev].astype(str)
    s = s[(s.str.strip() != "") & (s.str.lower() != "nan")].tolist()
    seen = set()
    out = []
    for x in s:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def filter_df_by_text(df: pd.DataFrame, query: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    q = (query or "").strip().lower()
    if not q:
        return df

    mask = np.zeros(len(df), dtype=bool)
    for c in df.columns:
        s = df[c].astype(str).str.lower()
        mask = mask | s.str.contains(q, na=False)
    return df.loc[mask].copy()


def safe_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        return float(x)
    except Exception:
        return None

def fill_missing_lithology(
    layers: pd.DataFrame,
    lithotypes: pd.DataFrame,
    default_lith: str = "Shales",
):
    """
    Заполняет пустые/NaN/'' значения в колонке литологии слоя.
    Возвращает (layers_fixed, changed_rows_idx, chosen_default).
    Гарантирует, что chosen_default НЕ 'nan' и существует в базе (если база корректна).
    """
    if layers is None or layers.empty:
        return layers, [], default_lith

    # 1) Найдём колонку литологии в layers
    lith_col = None
    for c in ["Lithology_type", "Lithology type", "lithology_type", "Lithology"]:
        if c in layers.columns:
            lith_col = c
            break
    if lith_col is None:
        return layers, [], default_lith

    # 2) Сформируем валидный список литологий из базы lithotypes
    base_col = None
    if lithotypes is not None and not lithotypes.empty:
        for c in ["Lithology type", "Lithology_type", "Lithology"]:
            if c in lithotypes.columns:
                base_col = c
                break

    valid = []
    if base_col is not None:
        s = lithotypes[base_col].copy()
        # приводим к строке, чистим пробелы
        s = s.astype(str).str.strip()
        # убираем мусор
        s = s[~s.str.lower().isin(["", "nan", "none", "null"])]
        # уникальные в порядке появления
        seen = set()
        for x in s.tolist():
            if x not in seen:
                valid.append(x)
                seen.add(x)

    # 3) Выбираем дефолт: либо default_lith (если он валидный и есть в базе),
    #    либо первый валидный из базы, либо (в худшем случае) просто "Shales"
    chosen = default_lith
    if valid:
        if chosen not in valid:
            chosen = valid[0]
    # если вдруг chosen всё ещё мусор (на всякий случай)
    if str(chosen).strip().lower() in ["", "nan", "none", "null"]:
        chosen = "Shales"

    # 4) Подставляем туда, где пусто/NaN
    s_layers = layers[lith_col].astype(str).str.strip()
    is_bad = s_layers.isna() | s_layers.str.lower().isin(["", "nan", "none", "null"])

    changed_idx = layers.index[is_bad].tolist()

    layers_fixed = layers.copy()
    if changed_idx:
        layers_fixed.loc[is_bad, lith_col] = chosen

    return layers_fixed, changed_idx, chosen

def tabs_highlight_css() -> str:
    return """
    <style>
    /* Подсветка активной вкладки st.tabs */
    div[role="tablist"] button[role="tab"][aria-selected="true"]{
        background: #e8f1ff !important;
        border-bottom: 3px solid #2a6df4 !important;
        font-weight: 700 !important;
    }
    div[role="tablist"] button[role="tab"]{
        padding-top: 10px !important;
        padding-bottom: 10px !important;
        border-radius: 10px 10px 0 0 !important;
    }
    </style>
    """

def tab_name_ru(prop_key: str) -> str:
    k = (prop_key or "").strip()
    k_low = k.lower()

    mapping = {
        "burial depth": "Глубина захоронения",
        "bottom_depth_km": "Глубина",
        "top_depth_km": "Глубина кровли",
        "decompaction_thickness_km": "Мощность",
        "thickness": "Мощность",
        "porosity": "Пористость",
        "pore volume (porosity %)": "Пористость",
        "pore pressure": "Поровое давление",
        "effective stress": "Эффективное напряжение",
        "density": "Плотность",
        "density_column_avg": "Плотность",
        "permeability": "Проницаемость",
        "sedimentation_rate_mm_yr": "Скорость осадконакопления",
        "vp_km_s": "Vp (км/с)",
        "vs_km_s": "Vs (км/с)",
    }

    # если уже русское — оставляем
    if any("А" <= ch <= "я" or ch in "ёЁ" for ch in k):
        return k

    if k in mapping:
        return mapping[k]
    if k_low in mapping:
        return mapping[k_low]

    # fallback: humanize snake_case
    nice = k.replace("_", " ").strip()
    # специальные кейсы
    if "km" in nice.lower():
        nice = nice.replace(" km", " (км)")
    return nice[:1].upper() + nice[1:]

    """
    Всегда возвращает русское имя вкладки.
    Если ключ неизвестен — возвращаем 'Результат' + ключ (но тоже по-русски можно подправить).
    """
    k = (prop_key or "").strip()

    mapping = {
        # частые варианты
        "Burial depth": "Глубина захоронения",
        "Burial_depth": "Глубина захоронения",
        "Depth": "Глубина",
        "Thickness": "Мощность",
        "Porosity": "Пористость",
        "Pore volume (Porosity %)": "Пористость",
        "Pore pressure": "Поровое давление",
        "Pore_pressure": "Поровое давление",
        "Effective stress": "Эффективное напряжение",
        "Overpressure": "Избыточное давление",
        # можно расширять
    }

    # если уже русское — оставляем
    if any("А" <= ch <= "я" or ch in "ёЁ" for ch in k):
        return k

    return mapping.get(k, f"Результат: {k}")




def porosity_table_layer_by_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводим df к виду: строки = Layer, колонки = Age (Ma).
    Поддержка 2 форматов:
      - index=Age, columns=Layer  -> транспонируем
      - index=Layer, columns=Age  -> оставляем
    """
    if df is None or df.empty:
        return df

    idx_is_age = False
    try:
        if df.index.name and "age" in str(df.index.name).lower():
            idx_is_age = True
        else:
            idx_is_age = pd.api.types.is_numeric_dtype(df.index)
    except Exception:
        idx_is_age = False

    out = df.T.copy() if idx_is_age else df.copy()

    # если колонки похожи на возраста — сортируем как в примере (260 ... 0)
    try:
        cols_num = pd.to_numeric(out.columns, errors="coerce")
        if cols_num.notna().all():
            out.columns = cols_num.astype(int)
            out = out.reindex(sorted(out.columns, reverse=True), axis=1)
    except Exception:
        pass

    out.index.name = "Layer"
    return out


def sanitize_filename(name: str) -> str:
    # простая “безопасная” транслитерация не нужна, достаточно убрать странные символы
    bad = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
    for b in bad:
        name = name.replace(b, "_")
    name = name.strip().replace("  ", " ")
    return name


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8")


def make_lineplot_png_bytes(df_prop: pd.DataFrame, title: str, y_label: str, plot_cfg: dict) -> bytes | None:
    """
    Генерит PNG для свойств, которые зависят от возраста (index=Age).
    Если не похоже на Age — возвращает None.
    """
    if df_prop is None or df_prop.empty:
        return None

    try:
        idx_is_age = False
        if df_prop.index.name and "age" in str(df_prop.index.name).lower():
            idx_is_age = True
        else:
            idx_is_age = pd.api.types.is_numeric_dtype(df_prop.index)
        if not idx_is_age:
            return None

        fig, ax = plt.subplots()
        for col in df_prop.columns:
            ax.plot(df_prop.index.values, df_prop[col].values, label=str(col))

        ax.set_title(title)
        ax.set_xlabel("Age (Ma)")
        ax.set_ylabel(y_label)

        if plot_cfg.get("show_grid", True):
            ax.grid(True, alpha=0.3)

        if plot_cfg.get("invert_y", True):
            ax.invert_yaxis()

        ax.legend(fontsize=8, ncol=int(plot_cfg.get("legend_cols", 2)))

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception:
        return None


def build_results_zip_bytes(results: dict, plot_cfg: dict) -> bytes:
    """
    Собирает ZIP:
      - CSV для каждого свойства
      - PNG для каждого свойства (если можем построить line-plot)
      - для пористости дополнительно:
          assets/porosity_left.png, assets/porosity_right.png (если существуют)
          CSV "porosity_table_layer_by_age.csv"
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # добавим все таблицы как CSV
        for prop, df in results.items():
            if isinstance(df, pd.DataFrame):
                csv_name = sanitize_filename(prop) + ".csv"
                zf.writestr(f"csv/{csv_name}", df_to_csv_bytes(df))

                # png если это временная эволюция
                cfg = dict(plot_cfg)
                if "poro" in prop.lower() or "porosity" in prop.lower():
                    cfg["invert_y"] = False
                png_bytes = make_lineplot_png_bytes(df, title=prop, y_label=prop, plot_cfg=cfg)
                if png_bytes is not None:
                    png_name = sanitize_filename(prop) + ".png"
                    zf.writestr(f"png/{png_name}", png_bytes)

        # если есть пористость — положим специальную таблицу Layer×Age
        poro_key = None
        for k in results.keys():
            if "poro" in k.lower() or "porosity" in k.lower():
                poro_key = k
                break
        if poro_key is not None and isinstance(results.get(poro_key), pd.DataFrame):
            table = porosity_table_layer_by_age(results[poro_key])
            zf.writestr("csv/porosity_table_layer_by_age.csv", df_to_csv_bytes(table))

        # добавим “презентационные” картинки пористости (если есть)
        left = DATA_DIR / "porosity_left.png"
        right = DATA_DIR / "porosity_right.png"

        if left.exists():
            zf.write(left, arcname="png/porosity_left.png")
        if right.exists():
            zf.write(right, arcname="png/porosity_right.png")

    buf.seek(0)
    return buf.getvalue()



# ----------------------------
# Plotting with settings
# ----------------------------
def plot_evolution(df_prop: pd.DataFrame, title: str, y_label: str, plot_cfg: dict):
    fig, ax = plt.subplots()
    for col in df_prop.columns:
        ax.plot(df_prop.index.values, df_prop[col].values, label=str(col))

    ax.set_title(title)
    ax.set_xlabel("Age (Ma)")
    ax.set_ylabel(y_label)

    if plot_cfg.get("show_grid", True):
        ax.grid(True, alpha=0.3)

    if plot_cfg.get("invert_y", True):
        ax.invert_yaxis()

    ax.legend(fontsize=8, ncol=int(plot_cfg.get("legend_cols", 2)))
    st.pyplot(fig)

def load_porosity_table_csv(path: Path) -> pd.DataFrame | None:
    """Читает porosity_table.csv, пропуская служебные строки типа 'Table 1'."""
    try:
        if path is None or not Path(path).exists():
            return None

        raw = Path(path).read_text(encoding="utf-8", errors="replace")
        lines = raw.splitlines()

        # пропускаем первые строки, пока не встретим настоящую шапку с числами (260,245,...)
        skip = 0
        for i, line in enumerate(lines):
            low = line.strip().lower()
            if not low:
                continue
            # хедер обычно содержит запятые и хотя бы одно число возраста
            if "," in line and any(ch.isdigit() for ch in line):
                skip = i
                break

        df = pd.read_csv(io.StringIO(raw), skiprows=skip, sep=",", index_col=0)

        # "None"/"none"/"" -> NaN, чтобы таблица была чистая и можно было выкинуть пустые строки
        df = df.replace(["None", "none", "NULL", "null", ""], np.nan)
        df = df.apply(pd.to_numeric, errors="ignore")  # числа останутся числами

        # удалить полностью пустые строки/колонки
        df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)

        # привести колонки к int и отсортировать 260 ... 0
        cols_num = pd.to_numeric(df.columns, errors="coerce")
        if cols_num.notna().all():
            df.columns = cols_num.astype(int)
            df = df.reindex(sorted(df.columns, reverse=True), axis=1)

        df.index.name = "Layer"

        # если есть дубли в названиях слоёв — делаем их уникальными (Shales -> Shales1, Shales2, ...)
        counts = {}
        new_idx = []
        for name in df.index.astype(str):
            if name not in counts:
                counts[name] = 0
                new_idx.append(name)
            else:
                counts[name] += 1
                new_idx.append(f"{name}{counts[name]}")
        df.index = new_idx
        df = df.where(pd.notna(df), "")

        return df
    except Exception:
        return None



def render_paleo_results(results: dict, plot_cfg: dict):
    # 1) Заголовок
    st.header("Результаты")

    if not isinstance(results, dict) or len(results) == 0:
        st.warning("Нет результатов для отображения.")
        return

    # 2) Кнопка ZIP (PNG+CSV)
    try:
        zip_bytes = build_results_zip_bytes(results, plot_cfg=plot_cfg)
        st.download_button(
            "Скачать результаты (PNG + CSV) — ZIP",
            data=zip_bytes,
            file_name="results_png_csv.zip",
            mime="application/zip",
            type="primary",
        )
    except Exception as e:
        st.warning(f"Не удалось сформировать ZIP: {e}")

    # 3) Подсветка активной вкладки
    st.markdown(tabs_highlight_css(), unsafe_allow_html=True)

    # 4) Гарантированно русские вкладки
    #    Сначала “Пористость”, затем “Поровое давление”, затем “Глубина…”, остальное в конце.
    #    Это удобно для презентации.
    def is_poro(k: str) -> bool:
        k2 = (k or "").lower()
        return ("poro" in k2) or ("porosity" in k2)

    def is_pressure(k: str) -> bool:
        k2 = (k or "").lower()
        return ("pressure" in k2) or ("pore pressure" in k2)

    def is_depth(k: str) -> bool:
        k2 = (k or "").lower()
        return ("burial" in k2) or ("depth" in k2)

    keys = list(results.keys())
    poro_keys = [k for k in keys if is_poro(k)]
    pressure_keys = [k for k in keys if (k not in poro_keys and is_pressure(k))]
    depth_keys = [k for k in keys if (k not in poro_keys and k not in pressure_keys and is_depth(k))]
    other_keys = [k for k in keys if k not in poro_keys + pressure_keys + depth_keys]

    ordered_props = poro_keys + pressure_keys + depth_keys + other_keys
    tab_labels = [tab_name_ru(k) for k in ordered_props]

    tabs = st.tabs(tab_labels)

    # пути к картинкам пористости (у тебя они уже должны быть в demo_simulator/assets/)
    # пути к картинкам пористости (у тебя они в папке data)
    PORO_IMG_LEFT = DATA_DIR / "porosity_left.png"
    PORO_IMG_RIGHT = DATA_DIR / "porosity_right.png"

    # если вдруг файлы названы иначе — покажем, что реально лежит в data
    def list_png_in_data():
        try:
            return [p.name for p in DATA_DIR.glob("*.png")]
        except Exception:
            return []
        
    # CSS (один раз, перед циклом вкладок)
    st.markdown("""
    <style>
    /* правая картинка ниже */
        .poro-right img{
        max-height: 720px;
        width: auto;
        display: block;
        margin: 0 auto;
        object-fit: contain;
    }
    /* левая картинка чуть больше (можно больше/меньше) */
    .poro-left img{
        max-height: 720px;
        width: auto;
        display: block;
        margin: 0 auto;
        object-fit: contain;
    }
    </style>
    """, unsafe_allow_html=True)



    for tab, prop_key, label in zip(tabs, ordered_props, tab_labels):
        with tab:
            df = results.get(prop_key)
            # --- ПОРИСТОСТЬ: 2 рисунка рядом + таблица ---
            if "poro" in prop_key.lower() or "porosity" in prop_key.lower():
                # st.subheader("Пористость")

                # 1) Две колонки для картинок
                c1, c2 = st.columns([1.35, 0.65])

                with c1:
                    if PORO_IMG_LEFT.exists():
                        import base64
                        b64 = base64.b64encode(PORO_IMG_LEFT.read_bytes()).decode("utf-8")
                        st.markdown(
                            f'<div class="poro-left"><img src="data:image/png;base64,{b64}"/></div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.warning("Не найден файл porosity_left.png рядом с app.py.")

                with c2:
                    if PORO_IMG_RIGHT.exists():
                        import base64
                        b64 = base64.b64encode(PORO_IMG_RIGHT.read_bytes()).decode("utf-8")
                        st.markdown(
                            f'<div class="poro-right"><img src="data:image/png;base64,{b64}"/></div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.warning("Не найден файл porosity_right.png рядом с app.py.")

                # 2) ВАЖНО: таблица ниже — уже НЕ внутри колонок
                #st.markdown("#### Расчётные значения (слой × геологическое время)")

                poro_csv = DATA_DIR / "porosity_table.csv"
                table = load_porosity_table_csv(poro_csv)

                if table is None:
                    st.warning(f"Не найден файл таблицы: {poro_csv.name}")
                else:
                    n = len(table)
                    height = min(800, 35 * (n + 1) + 20)  # ограничим сверху, чтобы не была огромной
                    st.dataframe(table, use_container_width=True, height=420)

                continue

                st.markdown("#### Расчётные значения (слой × геологическое время)")

                if isinstance(df, pd.DataFrame) and not df.empty:
                    table = porosity_table_layer_by_age(df)

                    # чтобы выглядело как твой пример: целые возраста в заголовках, Layer слева
                    table_disp = table.copy()
                    # округлим численные значения красиво
                    for c in table_disp.columns:
                        try:
                            table_disp[c] = pd.to_numeric(table_disp[c], errors="ignore")
                        except Exception:
                            pass

                    st.dataframe(table_disp, use_container_width=True, height=420)
                else:
                    st.warning("Нет табличных данных по пористости для отображения.")

                # важно: ничего “лишнего” на вкладке пористости не рисуем
                continue

            # --- ОСТАЛЬНЫЕ ВКЛАДКИ (презентационный минимум): таблица + при желании график ---
            st.subheader(label)

            if isinstance(df, pd.DataFrame) and not df.empty:
                st.dataframe(df, use_container_width=True)

                # Если это эволюция по возрасту — рисуем простой график (можно оставить)
                try:
                    idx_is_age = False
                    if df.index.name and "age" in str(df.index.name).lower():
                        idx_is_age = True
                    else:
                        idx_is_age = pd.api.types.is_numeric_dtype(df.index)

                    if idx_is_age:
                        plot_evolution(df, title=label, y_label=label, plot_cfg=plot_cfg)
                except Exception:
                    pass
            else:
                st.warning("Нет данных для отображения.")

def force_ru_uploader_text():
    components.html(
        """
        <script>
        const replaceUploaderText = () => {
          // заменяем фразу в зоне дропа
          const all = Array.from(document.querySelectorAll("*"));
          for (const el of all) {
            if (el.childNodes && el.childNodes.length === 1 && el.childNodes[0].nodeType === Node.TEXT_NODE) {
              const t = el.innerText?.trim();
              if (t === "Drag and drop file here") el.innerText = "Перетащите файл сюда";
              if (t === "Browse files") el.innerText = "Выбрать файл";
              if (t && t.includes("Limit") && t.includes("per file") && t.includes("CSV")) {
                el.innerText = "Ограничение 200MB на файл • CSV";
              }
            }
          }
        };

        // запускаем несколько раз, потому что Streamlit дорисовывает компонент асинхронно
        replaceUploaderText();
        setTimeout(replaceUploaderText, 200);
        setTimeout(replaceUploaderText, 600);
        setTimeout(replaceUploaderText, 1200);
        </script>
        """,
        height=0,
    )

def render_inputs_uploader():
    st.subheader("Входные данные")
    st.markdown("**Параметры скважины (.csv)**")

    upload = st.file_uploader(
        "",
        type=["csv"],
        help="Ограничение 200MB на файл • CSV",
        label_visibility="collapsed",
    )

    force_ru_uploader_text() 

    df = read_csv_smart(upload) if upload is not None else None
    in_meters = True
    return df, in_meters


# ----------------------------
# Русские подписи колонок (ТОЛЬКО для отображения)
# ----------------------------
RU_COLS = {
    "Event_name": "Название события",
    "Event_type": "Тип события",
    "Lithology_type": "Литотип",
    "Age (Ma)": "Возраст (млн лет)",
    "Depth top, m": "Глубина кровли, м",
    "Depth bottom, m": "Глубина подошвы, м",
    "Thickness, m": "Мощность, м",
    "D Thickness": "Δ Мощность",
    "D Thickness (m)": "Δ Мощность, м",
    "D Thickness (erosion), m": "Δ Мощность эрозии, м",
    "Sublayers": "Подслои",
    "N Sublayers": "Кол-во подслоев"

}

HIDE_COLS = [
    "Well", "well", "WELL",

    "Paleobathymetry, Ma",
    "Paleobathymetry",
    "Paleobathyometry, Ma",
    "Paleobathyometry",
    "paleobathymetry",
    "paleobathyometry",

    "Sea level, m",
    "Sea level",
    "SeaLevel",
    "sealevel",
    "sea_level",

    "Kinetic",
    "kinetic",
    "Kenetic",
    "kenetic",
    "Kinetics",

    "TOC initial, %",
    "TOC initial",
    "TOC_initial",
    "toc_initial",
    "TOC",
    "toc",

    "HI initial, %",
    "HI initial",
    "HI_initial",
    "hi_initial",
    "HI",
    "hi",
]


def build_preview_table(df_well: pd.DataFrame) -> pd.DataFrame:
    df_show = df_well.copy()
    df_show = df_show.drop(columns=[c for c in HIDE_COLS if c in df_show.columns], errors="ignore")
    df_show = df_show.rename(columns={k: v for k, v in RU_COLS.items() if k in df_show.columns})
    return df_show


# ----------------------------
# Streamlit config
# ----------------------------
st.set_page_config(page_title="Basin Simulation Demo", layout="wide")
show_splash_then_continue()
st.title("Basin Simulation Demo")

def ru_uploader_css() -> str:
    return """
    <style>
    /* --- file_uploader: прячем английские строки --- */
    div[data-testid="stFileUploaderDropzone"] small { display: none !important; } /* "Limit 200MB..." (если мешает) */

    /* Заголовок внутри зоны: "Drag and drop file here" */
    div[data-testid="stFileUploaderDropzone"] p {
        visibility: hidden !important;
        position: relative;
    }
    div[data-testid="stFileUploaderDropzone"] p::after {
        content: "Перетащите файл сюда" !important;
        visibility: visible !important;
        position: absolute;
        left: 0;
        top: 0;
        font-size: 16px;
        font-weight: 600;
        color: #111;
    }

    /* Кнопка: "Browse files" */
    div[data-testid="stFileUploaderDropzone"] button {
        visibility: hidden !important;
        position: relative;
    }
    div[data-testid="stFileUploaderDropzone"] button::after {
        content: "Выбрать файл" !important;
        visibility: visible !important;
        position: absolute;
        inset: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
    }
    </style>
    """
st.markdown(ru_uploader_css(), unsafe_allow_html=True)

def blue_theme_css() -> str:
    return """
    <style>
    /* Делает primary-кнопки (download_button и т.п.) синими */
    div.stDownloadButton > button,
    div.stButton > button[kind="primary"] {
        background: #2a6df4 !important;
        border: 1px solid #2a6df4 !important;
        color: white !important;
    }
    div.stDownloadButton > button:hover,
    div.stButton > button[kind="primary"]:hover {
        background: #1f57c8 !important;
        border: 1px solid #1f57c8 !important;
        color: white !important;
    }

    /* Убираем “красный” у активной вкладки и делаем синий */
    div[role="tablist"] button[role="tab"][aria-selected="true"]{
        background: #e8f1ff !important;
        border-bottom: 3px solid #2a6df4 !important;
        color: #2a6df4 !important;
        font-weight: 700 !important;
    }
    div[role="tablist"] button[role="tab"]{
        color: #1a1a1a !important;
        padding-top: 10px !important;
        padding-bottom: 10px !important;
        border-radius: 10px 10px 0 0 !important;
    }
    </style>
    """


st.markdown(blue_theme_css(), unsafe_allow_html=True)


def table_header_css() -> str:
    return """
    <style>
    /* заголовки колонок */
    div[data-testid="stDataFrame"] thead tr th,
    div[data-testid="stDataFrame"] thead tr th span {
        color: #000 !important;
        font-weight: 800 !important;
    }

    /* левый столбец (индекс/Layer) */
    div[data-testid="stDataFrame"] tbody tr th,
    div[data-testid="stDataFrame"] tbody tr th span {
        color: #000 !important;
        font-weight: 800 !important;
    }
    </style>
    """
st.markdown(table_header_css(), unsafe_allow_html=True)

def sidebar_css() -> str:
    return """
    <style>
    /* ширина сайдбара */
    section[data-testid="stSidebar"]{
        width: 430px !important;
    }
    section[data-testid="stSidebar"] > div{
        width: 430px !important;
    }

    /* увеличим шрифт */
    section[data-testid="stSidebar"] *{
        font-size: 18px !important;
    }

    /* радио/чекбоксы/кнопки крупнее */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p{
        font-size: 18px !important;
    }
    </style>
    """
st.markdown(sidebar_css(), unsafe_allow_html=True)


# ----------------------------
# Sidebar (RU) + navigation + status on Inputs
# ----------------------------
with st.sidebar:
    st.header("Опции запуска")

    page = st.radio(
        " ",
        [
            "Входные данные",
            "Модуль создания литологий",
            "Палеогеометрическая реконструкция",
            "Тепловая история",
            "Нефтегазогенерация",
            "Полная реконструкция",
        ],
        index=0,
        label_visibility="collapsed",
    )

    # Индикатор модели показываем ТОЛЬКО на странице входных данных
    if page == "Входные данные":
        st.divider()
        if "model_df_well" in st.session_state and st.session_state["model_df_well"] is not None:
            nm = get_well_name(st.session_state["model_df_well"])
            st.success(f"Модель загружена: {nm}")
        else:
            st.info("Модель не сохранена.")


# ----------------------------
# Load lithotypes
# ----------------------------
try:
    LITHO_SHEET_NAME = "Lithotypes"
    lithotypes = load_lithotypes_sheet(LITHO_XLSX_PATH, sheet_name=LITHO_SHEET_NAME)
except Exception as e:
    lithotypes = None
    st.error(f"Не удалось загрузить lithotypes.xlsx: {e}")


# ----------------------------
# Page: Inputs (fills the model)
# ----------------------------
if page == "Входные данные":
    df_well, in_meters = render_inputs_uploader()

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Входные данные: параметры скважины")

        if df_well is None:
            st.info("Загрузите CSV слева в боковой панели.")
        else:
            well_name = get_well_name(df_well)
            df_show = build_preview_table(df_well)
            st.text(f"Предпросмотр: параметры скважины «{well_name}»")
            #st.subheader(f"Предпросмотр: параметры скважины «{well_name}»")
            st.dataframe(df_show, use_container_width=True)

            # ---- Дополнительные параметры под таблицей ----
            st.markdown("### Граничные параметры (заполняются при необходимости)")

            with st.expander("SWIT (палеоширотные данные)", expanded=True):
                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    hemisphere = st.selectbox("Полушарие", ["", "Северное", "Южное"], index=0)
                with c2:
                    continent = st.text_input("Континент", value="")
                with c3:
                    latitude = st.number_input("Широта (°)", value=0.0, step=0.5)

                swit = {
                    "hemisphere": hemisphere or None,
                    "continent": continent.strip() or None,
                    "latitude": safe_float(latitude),
                }

            with st.expander("Палеобатиметрия и уровень моря", expanded=False):
                base = pd.DataFrame([{
                    "Палеобатиметрия, м": None,
                    "Уровень моря, м": None,
                }])

                paleo_tbl = st.data_editor(
                    base,
                    use_container_width=True,
                    num_rows="fixed",
                    key="paleo_table_inputs",
                )

                paleobathyometry_val = paleo_tbl.loc[0, "Палеобатиметрия, м"]
                sealevel_val = paleo_tbl.loc[0, "Уровень моря, м"]

            with st.expander("Тепловой поток", expanded=False):
                c1, c2 = st.columns([1.2, 1])
                with c1:
                    hf_file = st.file_uploader(
                        "Табличные данные теплового потока (CSV)",
                        type=["csv"],
                        key="hf_csv_inputs",
                    )
                with c2:
                    hf_single = st.number_input(
                        "Одно значение теплового потока",
                        value=0.0,
                        step=1.0,
                        key="hf_single_inputs",
                    )

                hf_table = None
                if hf_file is not None:
                    try:
                        hf_table = read_csv_smart(hf_file)
                    except Exception:
                        hf_table = None

                heatflow = {
                    "table": hf_table,
                    "single_value": safe_float(hf_single),
                }

            with st.expander("Кинетика (слои) и свойства материнской породы", expanded=True):
                layer_options = get_layer_options(df_well)
                layer_options_ui = [""] + layer_options

                if "kinetic_rows" not in st.session_state:
                    st.session_state["kinetic_rows"] = [{"layer": "", "toc": None, "hi": None}]

                def add_kin_row():
                    st.session_state["kinetic_rows"].append({"layer": "", "toc": None, "hi": None})

                st.button("Добавить слой", on_click=add_kin_row)

                new_rows = []
                for i, row in enumerate(st.session_state["kinetic_rows"]):
                    st.markdown(f"**Слой #{i+1}**")
                    a, b, c, d = st.columns([1.4, 1, 1, 0.6])

                    default_layer = row.get("layer", "")
                    if default_layer not in layer_options_ui:
                        default_layer = ""

                    layer = a.selectbox(
                        "Слой (из списка событий)",
                        options=layer_options_ui,
                        index=layer_options_ui.index(default_layer),
                        key=f"kin_layer_sel_{i}",
                    )

                    toc = b.number_input(
                        "TOC initial",
                        value=float(row.get("toc") or 0.0),
                        step=0.1,
                        key=f"kin_toc_{i}",
                    )
                    hi = c.number_input(
                        "HI initial",
                        value=float(row.get("hi") or 0.0),
                        step=1.0,
                        key=f"kin_hi_{i}",
                    )

                    remove = d.button("Удалить", key=f"kin_rm_{i}")
                    if not remove:
                        new_rows.append({
                            "layer": (layer or "").strip(),
                            "toc": safe_float(toc),
                            "hi": safe_float(hi),
                        })

                st.session_state["kinetic_rows"] = new_rows
                kinetic = st.session_state["kinetic_rows"]

            # сохраняем extra_inputs
            st.session_state["extra_inputs"] = {
                "swit": swit,
                "paleobathyometry": safe_float(paleobathyometry_val),
                "sealevel": safe_float(sealevel_val),
                "heatflow": heatflow,
                "kinetic": kinetic,
            }

            # --- UX: кнопка сохранения модели ---
            st.markdown("---")
            save_btn = st.button("Сохранить модель", type="primary")

            if save_btn:
                st.session_state["model_df_well"] = df_well.copy()
                st.session_state["model_in_meters"] = in_meters
                st.session_state["model_name"] = well_name
                st.success(f"Модель сохранена: {well_name}")

    with col2:
        st.subheader("База литотипов")

        if lithotypes is None or lithotypes.empty:
            st.warning("Таблица литотипов пуста или не загружена.")
        else:
            q = st.text_input("Литология", placeholder="например: Shalea, Quartz, ...")
            litho_f = filter_df_by_text(lithotypes, q)

            ALL_COLS = list(lithotypes.columns)

            def pick_existing(cols):
                return [c for c in cols if c in ALL_COLS]

            PRESETS = {
                # "Кратко (рекомендуется)": pick_existing([
                #     "Lithology type", "Density", "Porosity",
                #     "Compaction model", "Compressibility model",
                #     "Thermal conductivity", "Heat capacity",
                # ]),
                "Компакция": pick_existing([
                    "Lithology type", "Density", "Initial porosity",
                    "Compressibility MAX", "Compressibility MIN"
                ]),
                "Теплофизика": pick_existing([
                    "Lithology type", "Thermal conductivity",
                    "Thermal conductivity (vertical)", "Thermal conductivity (horizontal)",
                    "Anisotropy factor", "Thermal expansion", "Heat capacity",
                ]),
                "Радиогенные": pick_existing([
                    "Lithology type", "Radiogenic model", "Uranium", "Thorium", "Potassium",
                    "Gamma Ray", "HeatFlow Production",
                ]),
                "Показать все колонки": ALL_COLS,
            }

            # if len(PRESETS["Кратко (рекомендуется)"]) < 2:
            #     PRESETS["Кратко (рекомендуется)"] = ALL_COLS[:8]

            preset_name = st.selectbox("Модульные свойства", list(PRESETS.keys()), index=0)
            selected_cols = PRESETS[preset_name]

            with st.expander("Справка: доступные параметры"):
                st.caption("Все доступные параметры в базе литотипов:")
                st.code("\n".join(ALL_COLS))

            st.dataframe(litho_f[selected_cols], use_container_width=True, height=420)

    st.markdown("---")
    st.caption("Далее выберите модуль в боковом меню и нажмите «Запустить расчёт».")



# ----------------------------
# Page: Paleo reconstruction (run) - uses saved model only
# ----------------------------
elif page == "Палеогеометрическая реконструкция":
    st.subheader("Палеогеометрическая реконструкция")

    # берём модель
    df_well = st.session_state.get("model_df_well", None)
    in_meters = st.session_state.get("model_in_meters", True)

    if df_well is None or df_well.empty:
        st.warning("Сначала заполните и сохраните «Входные данные», затем запускайте расчёт.")
        st.stop()

    well_name = get_well_name(df_well)
    st.caption(f"Модель: «{well_name}» • Модуль: палеогеометрическая реконструкция")  

    with st.sidebar:
        st.subheader("Расчёт")
        st.caption(f"Модель: {well_name}")

        # st.markdown("### Настройки графиков")
        # invert_y = st.checkbox("Инвертировать ось Y (глубина вниз)", value=True)
        # show_grid = st.checkbox("Показывать сетку", value=True)
        # legend_cols = st.slider("Колонок в легенде", min_value=1, max_value=4, value=2)

        st.divider()
        allow_edit = st.checkbox("Разрешить редактирование слоёв (демо)", value=True)
        run_btn = st.button("Запустить расчёт", type="primary")

    plot_cfg = {"invert_y": True, "show_grid": True, "legend_cols": 2}

    meta = df_well.iloc[:1].copy()
    work_layers = df_well.iloc[1:].copy().reset_index(drop=True)


    if run_btn:
        try:
            df_join = pd.concat([meta, work_layers], ignore_index=True)
            df_norm = normalize_inputs(df_join)
            if in_meters:
                df_norm = ensure_depths_km(df_norm)
            layers_n = df_norm.iloc[1:].copy().reset_index(drop=True)

            # мягко фиксируем пустую литологию
            layers_n, changed_rows, used_default = fill_missing_lithology(
                layers_n, lithotypes, default_lith="Shales"
            )
            # if changed_rows:
            #     pretty_rows = [i + 1 for i in changed_rows]
            #     st.warning(
            #         f"В некоторых слоях не задан литотип. "
            #         f"Подставлен «{used_default}» для строк слоёв: {', '.join(map(str, pretty_rows))}."
            #     )

            results = run_decompaction_por_perm(layers_n, lithotypes)

            render_paleo_results(results, plot_cfg={"invert_y": True, "show_grid": False, "legend_cols": 2})


        except Exception as e:
            st.exception(e)


else:
    st.info("Этот модуль пока в разработке (интерфейс оставлен для макета/презентации).")

