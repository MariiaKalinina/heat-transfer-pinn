
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset
from .config import (
    NP_RNG, WELLS, N_SAMPLES, TIME_START, TIME_END, LITHO_NAMES, LITHO_TO_INT,
    TRAIN_FRAC, VAL_FRAC, TARGET_COL, DYN_COLS, STATIC_CONT_COLS, STATIC_CAT_COLS,
)

# Need to correct 
def heat_flux(beta, t):
    a = 125
    tau = 62.8
    lam = 3.5
    T1 = 1333
    N = 100
    T_sum = 0.0
    for n in range(1, N+1):
        Cn = beta/(n*np.pi) * np.sin(n*np.pi/beta) * np.exp(-(n**2)*t/tau)
        T_sum += Cn
    T_over_Tm = 0.8 * (1 + 2*T_sum)
    return T_over_Tm * 1e-3 * 60 * 697

def make_synthetic():
    rows, layer_rows = [], []
    t_myr = np.linspace(TIME_START, TIME_END, N_SAMPLES)
    for w in range(WELLS):
        litho_name = NP_RNG.choice(LITHO_NAMES)
        litho = LITHO_TO_INT[litho_name]
        porosity = NP_RNG.uniform(0.0, 0.7)
        k = NP_RNG.uniform(1.0, 6.0)
        swit = NP_RNG.uniform(-20.0, 2.0)
        age = NP_RNG.uniform(0.0, 260.0)

        n_layers = NP_RNG.integers(2, 7)
        rad_layers = NP_RNG.uniform(0.1, 4.0, n_layers)
        rad_heat_total = float(rad_layers.sum())
        rad_heat_mean  = float(rad_layers.mean())
        for li, rh in enumerate(rad_layers):
            layer_rows.append({"well_id": w, "layer_id": li, "radiogenic_heat": float(rh)})

        beta = NP_RNG.uniform(1.5, 3.0)
        q_base = heat_flux(beta, t_myr)

        q = (
            q_base
            + 0.05*(k - 3.5)
            - 0.06*(porosity - 0.2)
            + 0.005*(swit + 10.0)
            + 0.0008*(age - 130.0)
            + 0.02*(rad_heat_total - 8.0)
            + NP_RNG.normal(0, 0.05, len(t_myr))
        )

        dfw = pd.DataFrame({
            "well_id": w,
            "time_myr": t_myr,
            "q": q.astype(float),
            "lithology": litho,
            "lithology_name": litho_name,
            "thermal_conductivity": k,
            "porosity": porosity,
            "swit": swit,
            "age": age,
            "rad_heat_total": rad_heat_total,
            "rad_heat_mean": rad_heat_mean,
            "layers_count": n_layers,
        })
        rows.append(dfw)

    df = pd.concat(rows).reset_index(drop=True).sort_values(["well_id","time_myr"]).reset_index(drop=True)
    well_layers = pd.DataFrame(layer_rows).sort_values(["well_id", "layer_id"]).reset_index(drop=True)

    # normalized time feature
    df["time_myr_norm"] = (df["time_myr"] - df["time_myr"].min()) / (df["time_myr"].max() - df["time_myr"].min())
    return df, well_layers

def split_per_well(df):
    parts = []
    for wid, g in df.groupby("well_id", sort=False):
        g = g.sort_values("time_myr").copy()
        n = len(g)
        n_tr = int(n*TRAIN_FRAC)
        n_va = int(n*(TRAIN_FRAC+VAL_FRAC))
        g.loc[g.index[:n_tr], "split"] = "train"
        g.loc[g.index[n_tr:n_va], "split"] = "val"
        g.loc[g.index[n_va:], "split"] = "test"
        parts.append(g)
    return pd.concat(parts).reset_index(drop=True)

@dataclass
class NormStats:
    dyn_mean: pd.Series
    dyn_std: pd.Series
    stat_mean: pd.Series
    stat_std: pd.Series

def normalize(df_splits) -> tuple[pd.DataFrame, NormStats]:
    dyn_mean = df_splits[df_splits["split"]=="train"][DYN_COLS].mean()
    dyn_std  = df_splits[df_splits["split"]=="train"][DYN_COLS].std().replace(0,1.0).fillna(1.0)
    stat_mean = df_splits[df_splits["split"]=="train"][STATIC_CONT_COLS].mean()
    stat_std  = df_splits[df_splits["split"]=="train"][STATIC_CONT_COLS].std().replace(0,1.0).fillna(1.0)

    df_norm = df_splits.copy()
    df_norm[DYN_COLS] = (df_norm[DYN_COLS]-dyn_mean)/dyn_std
    df_norm[STATIC_CONT_COLS] = (df_norm[STATIC_CONT_COLS]-stat_mean)/stat_std
    return df_norm, NormStats(dyn_mean, dyn_std, stat_mean, stat_std)

class MultiWellTS(Dataset):
    def __init__(self, df_all, win, horizon, target_split: str,
                 dyn_cols=DYN_COLS, target_col=TARGET_COL):
        self.samples = []
        self.win, self.h = win, horizon
        assert target_split in {"train","val","test"}
        for w, g in df_all.groupby("well_id", sort=False):
            g = g.sort_values("time_myr").copy()
            Xdyn = g[dyn_cols].to_numpy(np.float32)
            y    = g[target_col].to_numpy(np.float32)
            tmyr = g["time_myr"].to_numpy(np.float32)
            scont = g[STATIC_CONT_COLS].iloc[0].to_numpy(np.float32)
            scat  = {c:int(g[c].iloc[0]) for c in STATIC_CAT_COLS}
            wid   = int(w)
            mask_target = (g["split"].values == target_split)
            for j in range(self.win, len(g)-self.h+1):
                if not mask_target[j]:
                    continue
                x_win = Xdyn[j-self.win:j]
                y_next = y[j:j+self.h]
                if np.isnan(x_win).any() or np.isnan(y_next).any():
                    continue
                t_val = tmyr[j]
                self.samples.append((x_win, y_next, scont, scat, wid, t_val))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        x, y, scont, scat, wid, t_val = self.samples[idx]
        return (torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(scont),
                torch.tensor([scat[c] for c in STATIC_CAT_COLS], dtype=torch.long),
                torch.tensor(wid, dtype=torch.long),
                torch.tensor(t_val, dtype=torch.float32))
    
### --- Delete the remain lines ---- 
if __name__ == "__main__":
    df, well_layers = make_synthetic()
    
    print("=== Synthetic Data Generated ===")
    print(f"DataFrame shape: {df.shape}")
    print(f"Wells: {df['well_id'].nunique()}")
    print("\nFirst 5 rows:")
    print(df.head())
<<<<<<< HEAD
    # print(df.columns)
=======
    print(df.columns)
>>>>>>> bc6f211 (all commits)
    plt.scatter(df["time_myr"], df["q"])
    plt.show()
    
    print(f"\nWell layers shape: {well_layers.shape}")
    print(well_layers.head())

