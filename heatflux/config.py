
import numpy as np

# Reproducibility
TORCH_SEED = 0
NP_RNG = np.random.default_rng(42)

# Data params
WELLS = 5
N_SAMPLES = 500
TIME_START, TIME_END = 0.0, 260.0  # Myr

LITHO_NAMES = ["Shale", "Chalk", "Limestone", "Anhydrite", "Quartzite", "Dolomite"]
LITHO_TO_INT = {name: i for i, name in enumerate(LITHO_NAMES)}

# Splits
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

# Windowing
WIN = 168
HORIZON = 1

# Columns
TARGET_COL = "q"
DYN_COLS = ["q", "time_myr_norm"]
STATIC_CONT_COLS = ["thermal_conductivity","porosity","swit","age","rad_heat_total"]
STATIC_CAT_COLS  = ["lithology"]
USE_WELL_ID_EMB = True
