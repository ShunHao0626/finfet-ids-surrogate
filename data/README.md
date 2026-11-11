# Data

The training data are BSIM-CMG HSPICE simulation records for an n-type FinFET (`N15A-nmos1.json`) provided during the 2025 summer program. They are not redistributed here. Place the file in this folder to rerun the analysis.

Expected structure:

```json
{
  "PDK": "N15A", "Node": 15, "Model": "BSIM-CMG", "Device": "nmos1", "Type": "NMOS", "Simulator": "hspice",
  "Records": [
    {"Nfin": 1, "L": 15, "Corner": "tt", "Temp": -40, "Vds": 0.2,
     "Vgs": [0.0, 0.1, "..."], "Ids": [0.0, 1.2e-9, "..."], "Cgg": ["..."]}
  ]
}
```

The file used here contains 125 Id–Vg sweeps (1,125 bias points): Nfin = 1, 2, 4, 8, 16; L = 15 nm; corner tt; T = −40, 0, 25, 85, 125 °C; Vds = 0–0.8 V; Vgs = 0–0.8 V in 0.1 V steps.
