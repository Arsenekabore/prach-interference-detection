# Machine Learning for PRACH Interference Detection in 5G NR Networks — code and results

Code and raw results accompanying the revised manuscript.

## Contents
- `prach_sim.py` — 3GPP-style PRACH Format 0 simulator (Zadoff–Chu preambles, cyclic shifts, CP, ETU/EPA fading with Doppler, CFO, 2 Rx antennas, intra-/inter-cell interferers, AWGN) and receiver front end (CP removal, FFT, correlation maps).
- `features.py` — input representations, the 14 XGBoost features and the conventional correlation-based detector.
- `experiments.py` — the benchmark: 10 independent runs × 7 detectors (Correlation, XGBoost, CNN, Conv1D, LSTM, Transformer, Conv1D-IQ ablation), leakage-free data generation, metrics per class/SNR/ISR, timing.
- `analysis.py` — statistics (mean ± std, 95 % CI, Friedman, Wilcoxon–Holm), LaTeX tables and figures.
- `PRACH_interference_campaign.ipynb` — Google Colab notebook that runs everything (GPU recommended).
- `results/` — raw outputs of the campaign reported in the paper (Tesla T4, TensorFlow 2.21, XGBoost 3.4.1).

## Reproduce
```python
import experiments, analysis
experiments.run_all(dict(out_dir="results"))   # ~3–5 h on a T4 GPU
analysis.full_report("results")
```
Seeds: run r uses 1000r+1 … 1000r+5 for the five data sets and r for model initialization.
