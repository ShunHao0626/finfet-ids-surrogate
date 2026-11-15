# FinFET drain-current surrogate (finfet-ids-surrogate)

**English** · [简体中文](README_CN.md)

A PyTorch study of how well a neural-network surrogate reproduces BSIM-CMG drain-current simulations for a 15 nm-class n-type FinFET. It uses seeded runs and held-out bias-point, curve, and temperature evaluations.

**Main finding.** The model follows the on-state current trend on a linear scale, yet its high R² conceals large relative errors and unreliable subthreshold predictions. The comparison across operating regimes is the central result.

Inputs are fin number, channel length, process corner, temperature, V<sub>ds</sub> and V<sub>gs</sub>; the output is I<sub>ds</sub>.

## Repository contents

| Folder | Contents |
| --- | --- |
| [`original/`](original/) | Original training script and three output figures |
| [`src/finfet_ids_mlp.py`](src/finfet_ids_mlp.py) | Reproducible training script with fixed seeds, four evaluation splits, and held-out metrics |
| [`results/`](results/) | Metrics for every split and seed, a summary, and held-out figures |
| [`data/`](data/README.md) | Data format; the simulation file itself is not redistributed |

## Model

- MLP: 6 → 100 → 50 → 20 → 1, ReLU, batch normalization and dropout (0.1) in the first two hidden layers
- Huber (smooth L1) loss on standardized I<sub>ds</sub>, Adam (lr 1e-3, weight decay 1e-5), cosine-annealing schedule, early stopping (patience 50)

## Results

Held-out metrics are means over three seeds (0, 1, 2). Early stopping monitors the same held-out partition, so these are model-selection metrics rather than independent final-test estimates. MAPE is computed for points with |I<sub>ds</sub>| > 1 µA.

| Split | What is held out | Evaluation points | R² (mean, range) | RMSE (µA) | MAPE above 1 µA |
| --- | --- | --- | --- | --- | --- |
| `ordered` | last 10 % of the file | 113 | 0.996 (0.996–0.998) | 32 | 54 % |
| `random-points` | random 10 % of bias points | 113 | 0.997 (0.996–0.998) | 23 | 78 % |
| `random-curves` | random 10 % of whole Id–Vg sweeps | 108 | 0.997 (0.995–0.999) | 22 | 75 % |
| `holdout-temp` | every sweep at 125 °C (temperature extrapolation) | 225 | 0.988 (0.982–0.991) | 46 | 87 % |

<p align="center">
<img src="results/figures/idvg_random-curves.png" width="720" alt="Held-out Id-Vg curves on linear and log scales">
</p>

**What this shows**

1. **On-state trends are captured on a linear scale.** R² stays above 0.99 for held-out sweeps, while MAPE above 1 µA remains 75 % for the random-curve split. The two metrics describe different aspects of fit.
2. **Subthreshold current remains unresolved.** On a log scale, predictions can differ by orders of magnitude or turn negative. The linear-current loss places little weight on near-zero values.
3. **Temperature extrapolation is harder.** Holding out all 125 °C sweeps lowers R² to 0.988 and raises RMSE to 46 µA.

A log-current or arcsinh target, or a physics-based output layer, is a concrete next experiment for subthreshold behavior.

## Reproduce

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
# put N15A-nmos1.json in data/ (see data/README.md)
for s in ordered random-points random-curves holdout-temp; do
  for seed in 0 1 2; do python src/finfet_ids_mlp.py --data data/N15A-nmos1.json --split $s --seed $seed; done
done
```

Runs on CPU in a few seconds each.
