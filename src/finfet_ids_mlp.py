"""MLP surrogate for FinFET drain current (Ids) from BSIM-CMG simulation records.

Refactored from the July 2025 summer-program script in ../original/. The network,
loss, optimizer and scheduler follow the original; this version adds explicit
train/test split strategies, fixed seeds, held-out metrics and test-set plots.
"""
import argparse, json, os, random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURES = ["Nfin", "L", "Corner", "Temp", "Vds", "Vgs"]


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(6, 100), nn.BatchNorm1d(100), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(100, 50), nn.BatchNorm1d(50), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(50, 20), nn.ReLU(),
            nn.Linear(20, 1))

    def forward(self, x):
        return self.features(x).squeeze(-1)


def load_records(path):
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    corners, rows = {}, []
    for rec_id, r in enumerate(meta["Records"]):
        c = corners.setdefault(r["Corner"], len(corners))
        for vgs, ids in zip(r["Vgs"], r["Ids"]):
            rows.append((rec_id, r["Nfin"], r["L"], c, r["Temp"], r["Vds"], vgs, ids))
    a = np.array(rows, dtype=float)
    return meta, corners, a[:, 0].astype(int), a[:, 1:7], a[:, 7]


def split_indices(rec, X, how, seed):
    rng = np.random.default_rng(seed)
    n = len(X)
    if how == "ordered":            # original script: first 90 % of points in file order
        k = int(0.9 * n); return np.arange(k), np.arange(k, n)
    if how == "random-points":      # shuffled bias points
        p = rng.permutation(n); k = int(0.9 * n); return p[:k], p[k:]
    if how == "random-curves":      # whole Id-Vg curves held out
        ids = np.unique(rec); rng.shuffle(ids); test = set(ids[: max(1, len(ids) // 10)])
        m = np.array([r in test for r in rec]); return np.where(~m)[0], np.where(m)[0]
    if how == "holdout-temp":       # extrapolate to the highest temperature
        t = X[:, 3].max(); m = X[:, 3] == t; return np.where(~m)[0], np.where(m)[0]
    raise ValueError(how)


def train(X, y, tr, te, seed, epochs=2000, patience=50):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    sx = StandardScaler().fit(X[tr, :5]); sy = StandardScaler().fit(y[tr, None])
    def tx(A):
        A = A.copy(); A[:, :5] = sx.transform(A[:, :5]); return torch.tensor(A, dtype=torch.float32)
    Xtr, Xte = tx(X[tr]), tx(X[te])
    ytr = torch.tensor(sy.transform(y[tr, None]).ravel(), dtype=torch.float32)
    yte = torch.tensor(sy.transform(y[te, None]).ravel(), dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True, generator=g)
    net, lossf = MLP(), nn.SmoothL1Loss()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
    best, state, wait, hist = float("inf"), None, 0, []
    # NOTE: as in the original script, early stopping monitors the held-out set,
    # so reported test metrics are mildly optimistic.
    for ep in range(epochs):
        net.train(); tot = 0
        for xb, yb in dl:
            opt.zero_grad(); l = lossf(net(xb), yb); l.backward(); opt.step(); tot += l.item()
        net.eval()
        with torch.no_grad(): vl = lossf(net(Xte), yte).item()
        hist.append((tot / len(dl), vl)); sch.step()
        if vl < best: best, state, wait = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience: break
    net.load_state_dict(state); net.eval()
    pred = lambda A: sy.inverse_transform(net(tx(A)).detach().numpy()[:, None]).ravel()
    return net, pred, hist


def metrics(y, p):
    on = np.abs(y) > 1e-6
    return {"n": int(len(y)), "R2": float(r2_score(y, p)),
            "RMSE_uA": float(np.sqrt(np.mean((y - p) ** 2)) * 1e6),
            "MAPE_on_state_pct": float(np.mean(np.abs((p[on] - y[on]) / y[on])) * 100) if on.any() else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="random-curves",
                    choices=["ordered", "random-points", "random-curves", "holdout-temp"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    meta, corners, rec, X, y = load_records(a.data)
    tr, te = split_indices(rec, X, a.split, a.seed)
    net, pred, hist = train(X, y, tr, te, a.seed)
    res = {"split": a.split, "seed": a.seed, "epochs_run": len(hist),
           "train": metrics(y[tr], pred(X[tr])), "test": metrics(y[te], pred(X[te]))}
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    with open(os.path.join(a.out, f"metrics_{a.split}_seed{a.seed}.json"), "w") as f: json.dump(res, f, indent=2)
    print(json.dumps(res))
    if a.seed == 0:
        tag = a.split
        p = pred(X[te]); fig, ax = plt.subplots(figsize=(4.2, 4))
        ax.scatter(y[te] * 1e3, p * 1e3, s=10, alpha=.6); m = max(y[te].max(), p.max()) * 1e3
        ax.plot([0, m], [0, m], "k--", lw=1); ax.set_xlabel("BSIM-CMG I$_{ds}$ (mA)"); ax.set_ylabel("MLP I$_{ds}$ (mA)")
        ax.set_title(f"Held-out set ({tag}), R$^2$ = {res['test']['R2']:.4f}", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "figures", f"parity_{tag}.png"), dpi=200); plt.close(fig)
        # Id-Vg family for one held-out curve group
        rid = rec[te][np.argmax(y[te])]; same = (X[:, 0] == X[rec == rid][0, 0]) & (X[:, 3] == X[rec == rid][0, 3])
        fig, axs = plt.subplots(1, 2, figsize=(8, 3.4))
        for vds in sorted(set(X[same, 4])):
            if vds == 0: continue
            m_ = same & (X[:, 4] == vds); o = np.argsort(X[m_, 5])
            vg = np.linspace(0, X[m_, 5].max(), 120); Q = np.tile(X[m_][0], (120, 1)); Q[:, 5] = vg
            for ax in axs:
                l, = ax.plot(vg, pred(Q) * 1e3, lw=1.2)
                ax.plot(X[m_, 5][o], y[m_][o] * 1e3, "o", ms=3.5, color=l.get_color(), label=f"V$_{{ds}}$={vds:g} V")
        axs[1].set_yscale("log"); axs[1].set_ylim(1e-6, None)
        for ax in axs: ax.set_xlabel("V$_{gs}$ (V)"); ax.set_ylabel("I$_{ds}$ (mA)")
        axs[0].legend(fontsize=7); nf, t = X[rec == rid][0, 0], X[rec == rid][0, 3]
        fig.suptitle(f"N$_{{fin}}$={nf:g}, T={t:g} °C (markers: BSIM-CMG, lines: MLP)", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "figures", f"idvg_{tag}.png"), dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
