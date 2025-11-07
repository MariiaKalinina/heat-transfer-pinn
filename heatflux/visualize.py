
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_train_history(train_hist, val_hist):
    plt.figure(figsize=(7,3))
    plt.plot(train_hist, label="train MAE")
    plt.plot(val_hist, label="val MAE")
    plt.title("Training history")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend(); plt.tight_layout(); plt.show()

def plot_test_time_series(well_ids, truths, preds, tvals):
    well_to_plot = int(pd.Series(well_ids).mode()[0])
    mask = (well_ids == well_to_plot)
    df_plot = pd.DataFrame({
        "time_myr": tvals[mask],
        "y_true": truths[mask],
        "y_pred": preds[mask],
    }).sort_values("time_myr")
    tail = df_plot.tail(1000)
    plt.figure(figsize=(10,3))
    plt.plot(tail["time_myr"], tail["y_true"], label="true")
    plt.plot(tail["time_myr"], tail["y_pred"], label="pred")
    plt.title(f"Test predictions vs truth (well {well_to_plot}, tail)")
    plt.xlabel("Time (Myr)")
    plt.ylabel("Normalized q")
    plt.legend(); plt.tight_layout(); plt.show()

def plot_scatter_truth_vs_pred(truths, preds):
    plt.figure(figsize=(4,4))
    plt.scatter(truths, preds, s=10, alpha=0.5)
    minv, maxv = float(min(truths.min(), preds.min())), float(max(truths.max(), preds.max()))
    plt.plot([minv, maxv], [minv, maxv])
    plt.title("Predicted vs True (test)")
    plt.xlabel("True"); plt.ylabel("Predicted")
    plt.tight_layout(); plt.show()

def plot_residual_histogram(truths, preds):
    res = preds - truths
    plt.figure(figsize=(7,3))
    plt.hist(res, bins=40)
    plt.title("Residuals (pred - true)")
    plt.xlabel("Residual"); plt.ylabel("Count")
    plt.tight_layout(); plt.show()

def plot_per_well_bar(per_well_df):
    plt.figure(figsize=(6,3))
    plt.bar(per_well_df["well_id"].astype(str), per_well_df["MAE"])
    plt.title("Per-well MAE (test)")
    plt.xlabel("well_id"); plt.ylabel("MAE (normalized)")
    plt.tight_layout(); plt.show()
