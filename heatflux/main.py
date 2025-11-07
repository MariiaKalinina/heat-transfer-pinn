
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .config import (
    TORCH_SEED, WIN, HORIZON, TARGET_COL, DYN_COLS, STATIC_CONT_COLS, STATIC_CAT_COLS,
    USE_WELL_ID_EMB
)
from .data import make_synthetic, split_per_well, normalize, MultiWellTS
from .model import GRUWithStatics
from .model import LSTMWithStatics
from .train import train_model
from .evaluate import collect_predictions, metrics, per_well_mae
from .visualize import (
    plot_train_history, plot_test_time_series, plot_scatter_truth_vs_pred,
    plot_residual_histogram, plot_per_well_bar
)

def run():
    torch.manual_seed(TORCH_SEED)

    # Data
    df, well_layers = make_synthetic()
    df_splits = split_per_well(df)
    df_norm, norm = normalize(df_splits)

    num_wells = df_norm["well_id"].nunique()
    cat_card  = {c: int(df_norm[c].max())+1 for c in STATIC_CAT_COLS}
    F_DYN = len(DYN_COLS)

    # Datasets & loaders
    train_ds = MultiWellTS(df_norm, WIN, HORIZON, target_split="train")
    val_ds   = MultiWellTS(df_norm, WIN, HORIZON, target_split="val")
    test_ds  = MultiWellTS(df_norm, WIN, HORIZON, target_split="test")

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False)

    print(f"Windows per split -> train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
    assert len(train_ds)>0 and len(val_ds)>0 and len(test_ds)>0

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = GRUWithStatics(
        f_dyn=F_DYN, hidden=192, layers=1, dropout=0.1,
        cat_card=cat_card, cat_emb_dim=8, static_cont_dim=len(STATIC_CONT_COLS),
        use_well_emb=USE_WELL_ID_EMB, num_wells=num_wells, well_emb_dim=8, horizon=1
    ).to(device)

    model = LSTMWithStatics(
        f_dyn=F_DYN, hidden=192, layers=1, dropout=0.1,
        cat_card=cat_card, cat_emb_dim=8, static_cont_dim=len(STATIC_CONT_COLS),
        use_well_emb=USE_WELL_ID_EMB, num_wells=num_wells, well_emb_dim=8, horizon=1,
        bidirectional=False,   # set True to use BiLSTM (head already accounts for 2x hidden)
    ).to(device)


    # Train
    train_hist, val_hist = train_model(model, train_loader, val_loader, device=device,
                                       epochs=30, lr=1e-3, weight_decay=1e-2, patience=5)

    # Evaluate
    preds, truths, well_ids, tvals = collect_predictions(model, test_loader, device=device)
    m = metrics(preds, truths)
    print(f"Test MAE={m['MAE']:.3f} | RMSE={m['RMSE']:.3f}  (normalized units)")

    # Back to original q-units using training std of q
    q_std = float(norm.dyn_std['q'])
    print("Approx in original q-units:", f"MAE≈{m['MAE']*q_std:.3f}, RMSE≈{m['RMSE']*q_std:.3f}")

    # Visuals
    plot_train_history(train_hist, val_hist)
    plot_test_time_series(well_ids, truths, preds, tvals)
    plot_scatter_truth_vs_pred(truths, preds)
    plot_residual_histogram(truths, preds)
    per_well = per_well_mae(preds, truths, well_ids)
    plot_per_well_bar(per_well)

if __name__ == "__main__":
    run()
