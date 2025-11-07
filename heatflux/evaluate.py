
import numpy as np
import torch
import pandas as pd

def collect_predictions(model, data_loader, device='cpu'):
    model.eval()
    preds, truths, well_ids, t_list = [], [], [], []
    with torch.no_grad():
        for xb, yb, scont, scat, wid, t_val in data_loader:
            xb, yb = xb.to(device), yb.to(device)
            scont, scat, wid = scont.to(device), scat.to(device), wid.to(device)
            p = model(xb, scont, scat, wid)
            preds.append(p.cpu().numpy())
            truths.append(yb.cpu().numpy())
            well_ids.append(wid.cpu().numpy())
            t_list.append(t_val.cpu().numpy())

    preds    = np.vstack(preds).squeeze(-1)
    truths   = np.vstack(truths).squeeze(-1)
    well_ids = np.concatenate(well_ids)
    tvals    = np.concatenate(t_list)
    return preds, truths, well_ids, tvals

def metrics(preds, truths):
    mae = float(np.mean(np.abs(preds-truths)))
    rmse = float(np.sqrt(np.mean((preds-truths)**2)))
    return {"MAE": mae, "RMSE": rmse}

def per_well_mae(preds, truths, well_ids):
    out = []
    for w in np.unique(well_ids):
        m = (well_ids==w)
        out.append((int(w), float(np.mean(np.abs(preds[m]-truths[m])))))
    df = pd.DataFrame(out, columns=["well_id","MAE"]).sort_values("well_id")
    return df
