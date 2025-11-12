
from typing import Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn

def train_model(model, train_loader, val_loader, device="cpu",
                epochs=30, lr=1e-3, weight_decay=1e-2, patience=5):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.L1Loss()

    train_hist, val_hist = [], []
    best_val = 1e9; best = None; noimp=0

    for epoch in range(1, epochs+1):
        model.train(); tr_losses=[]
        for xb, yb, scont, scat, wid, _t in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            scont, scat, wid = scont.to(device), scat.to(device), wid.to(device)
            opt.zero_grad()
            pred = model(xb, scont, scat, wid)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_losses.append(loss.item())

        model.eval(); va_losses=[]
        with torch.no_grad():
            for xb, yb, scont, scat, wid, _t in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                scont, scat, wid = scont.to(device), scat.to(device), wid.to(device)
                va_losses.append(loss_fn(model(xb, scont, scat, wid), yb).item())

        tr, va = float(np.mean(tr_losses)), float(np.mean(va_losses))
        print(f"Epoch {epoch:02d} | train MAE={tr:.4f} | val MAE={va:.4f}")
        train_hist.append(tr); val_hist.append(va)

        if va + 1e-6 < best_val:
            best_val = va; best = {k:v.cpu().clone() for k,v in model.state_dict().items()}; noimp=0
        else:
            noimp += 1
            if noimp >= patience:
                print("Early stopping."); break

    if best:
        model.load_state_dict({k:v.to(device) for k,v in best.items()})

    return train_hist, val_hist
