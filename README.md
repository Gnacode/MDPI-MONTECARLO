# TSNFA Monte Carlo Simulation

Companion repository for the manuscript:

> **Restoring CFAR Validity for Single-Channel IoT Sensor Streams: A Monte Carlo Comparison of CFAR-Family and Sequential Detectors under Cortex-M0+ Constraints**
> S. Makovetskyi, O. Zhelanov, V. Kauk, and L. Thomsen. Submitted to *MDPI Sensors*, 2026.

The repository contains the Monte Carlo simulator used to produce the results of the manuscript, together with the algorithm flow figures. The archived, citable version of the simulation code and data accompanying the manuscript is on Zenodo (DOI: 10.5281/zenodo.20192887); this repository is the working mirror.

## Contents

| Path | Description |
|---|---|
| `simulator_v1_2.py` | The Monte Carlo simulator: signal model, mesh network model, TSNFA variants (median, mean/EMA, hybrid), and the classical comparators. |
| `comparators_classical.py` | Locked comparator implementations (CA-CFAR, OS-CFAR, Lipski adaptation, CUSUM, and the tier 2 CFAR family). **Required by the simulator.** |
| `figures/` | Algorithm flow diagrams (editable SVG) for the manuscript's Figures 4 to 8. |
| `requirements.txt` | Python dependencies. |
| `setup_venv.bat` / `setup_venv.sh` | One-shot virtual-environment setup for Windows and Linux/macOS. |

Simulation outputs (`results/`, `snapshots/`) are **not** committed: raw-waveform snapshots in particular are large. They regenerate locally from the simulator and are excluded by `.gitignore`.

## Setup

Python 3.10 or newer.

**Windows**

```bat
setup_venv.bat
```

**Linux / macOS**

```bash
./setup_venv.sh
```

Both create `.venv/` in the repository root and install `numpy`, `pandas`, `scipy`, and `matplotlib`. Activate later with `.venv\Scripts\activate.bat` (Windows) or `source .venv/bin/activate` (Linux/macOS).

## Running

The recommended TSNFA configuration of the manuscript (hybrid variant: mean trigger, gated 64-entry median floor, zeta = 2.5, three-frame confirmation):

```bash
python simulator_v1_2.py --zeta 2.5 --confirm 3
```

Key command-line options (see `python simulator_v1_2.py --help` for the full list):

- `--zeta` : threshold multiplier (2.5 recommended with confirmation; 6.0 for single-frame operation)
- `--confirm` : consecutive-frame confirmation requirement M (default 1 = no confirmation)
- Snapshot collection is off by default (`enable_snapshots = False` in the configuration); enabling it produces the large raw-waveform files excluded from the repository.

One simulated test configuration covers 24 h of operation per node; the full campaign of the manuscript (nine SNR levels at 10 nodes, checks at 50 nodes, five replicates each) is a batch of such runs.

## Reproducing the manuscript results

The headline tables of the manuscript (Tables 6 to 13) derive from the campaign defined in Section 3.3 of the paper: SNR levels 1.5, 2, 3, 4, 5, 6, 12, 18, 24 dB at 10 nodes, with 3, 12, 18 dB repeated at 50 nodes, plus one supplementary test configuration measuring GO-, SO-, TM-, and VI-CFAR at 10 nodes / 12 dB. Every detector processes the identical realisation at each Monte Carlo seed.

## Citation

If you use this code, please cite the manuscript above and the Zenodo archive:

```
Makovetskyi, S.; Thomsen, L. TSNFA Monte Carlo simulation: Code and data.
Zenodo, 2026. https://doi.org/10.5281/zenodo.20192887
```

## License

MIT License. © 2026 GNACODE Inc. See [LICENSE](LICENSE).
