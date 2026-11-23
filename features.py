"""Input representations, engineered features and the conventional correlation-based detector."""
import numpy as np
import prach_sim as ps

# detection window of preamble v in the own-root delay profile: (-C_v + [0, N_CS)) mod L_RA
_WIN = np.stack([(np.arange(ps.N_CS) - ps.N_CS * v) % ps.L_RA for v in range(ps.N_PRE_PER_ROOT)])


def candidate_roots(cfg):
    return (cfg["u0"],) + tuple(cfg["neighbour_roots"])


def maps_from(Yf, cfg):
    """Correlation maps (B, R, 839) normalised by each sample's own-root median (noise-floor proxy)."""
    P = ps.correlation_maps(Yf, candidate_roots(cfg))
    floor = np.median(P[:, 0, :], axis=1)[:, None, None] + 1e-12
    return (P / floor).astype(np.float32)


def deep_inputs(Yf, cfg, frame=8):
    """Returns the three tensors used by the deep models:
       cmap  (B, R, 839, 1)  log-compressed correlation map  -> 2D CNN
       iq    (B, 839, 2*n_rx) per-sample standardised freq-domain IQ  -> Conv1D
       tok   (B, T, R*frame)  framed correlation map tokens  -> LSTM / Transformer"""
    M = np.log1p(maps_from(Yf, cfg))
    B, R, L = M.shape
    cmap = M[..., None]
    iq = np.concatenate([Yf.real, Yf.imag], axis=1).transpose(0, 2, 1)
    iq = (iq - iq.mean(axis=(1, 2), keepdims=True)) / (iq.std(axis=(1, 2), keepdims=True) + 1e-9)
    Lp = int(np.ceil(L / frame) * frame)
    Mp = np.pad(M, ((0, 0), (0, 0), (0, Lp - L)))
    tok = Mp.reshape(B, R, Lp // frame, frame).transpose(0, 2, 1, 3).reshape(B, Lp // frame, R * frame)
    return cmap.astype(np.float32), iq.astype(np.float32), tok.astype(np.float32)


def engineered_features(Yf, cfg):
    """Tabular features for XGBoost (names in FEATURE_NAMES)."""
    P = maps_from(Yf, cfg)                         # (B, R, L), own-root median = 1
    own = P[:, 0, :]
    wpk = np.sort(own[:, _WIN].max(axis=2), axis=1)[:, ::-1]   # sorted per-window peaks (B, 32)
    f = {}
    f["own_peak1"] = np.log1p(wpk[:, 0])
    f["own_peak2"] = np.log1p(wpk[:, 1])
    f["own_peak3"] = np.log1p(wpk[:, 2])
    f["own_peak2_over_peak1"] = wpk[:, 1] / (wpk[:, 0] + 1e-9)
    f["own_mean_over_median"] = own.mean(axis=1)
    f["own_p95_over_median"] = np.quantile(own, 0.95, axis=1)
    f["own_n_windows_gt8"] = (wpk > 8).sum(axis=1)
    f["own_n_windows_gt16"] = (wpk > 16).sum(axis=1)
    nb = P[:, 1:, :]
    nb_peak = nb.max(axis=2)
    nb_psr = nb_peak / (np.median(nb, axis=2) + 1e-9)
    f["nb_peak_max"] = np.log1p(nb_peak.max(axis=1))
    f["nb_psr_max"] = np.log1p(nb_psr.max(axis=1))
    f["nb_peak_over_own_peak"] = nb_peak.max(axis=1) / (wpk[:, 0] + 1e-9)
    f["nb_mean_over_own_median"] = nb.mean(axis=(1, 2))
    rx_pow = (np.abs(Yf) ** 2).mean(axis=(1, 2))
    raw_own = ps.correlation_maps(Yf, (cfg["u0"],))[:, 0, :]
    f["energy_over_own_floor"] = np.log(rx_pow / (np.median(raw_own, axis=1) / ps.L_RA * 1 + 1e-12))
    f["own_kurtosis"] = ((own - own.mean(1, keepdims=True)) ** 4).mean(1) / (own.var(1) ** 2 + 1e-12)
    return np.stack([f[k] for k in FEATURE_NAMES], axis=1).astype(np.float32)


FEATURE_NAMES = ["own_peak1", "own_peak2", "own_peak3", "own_peak2_over_peak1", "own_mean_over_median",
                 "own_p95_over_median", "own_n_windows_gt8", "own_n_windows_gt16", "nb_peak_max",
                 "nb_psr_max", "nb_peak_over_own_peak", "nb_mean_over_own_median",
                 "energy_over_own_floor", "own_kurtosis"]


# ------------------------------------------------------------ conventional detector
def baseline_stats(Yf, cfg):
    """Classical test statistics computed from matched-filter outputs (no learning):
       r_floor = second-largest own-root window peak / own-root noise floor
       r_main  = second-largest own-root window peak / largest own-root window peak
       t_inter = largest neighbour-root peak / neighbour-root noise floor"""
    P = ps.correlation_maps(Yf, candidate_roots(cfg))
    own = P[:, 0, :]
    wpk = np.sort(own[:, _WIN].max(axis=2), axis=1)
    r_floor = wpk[:, -2] / (np.median(own, axis=1) + 1e-12)
    r_main = wpk[:, -2] / (wpk[:, -1] + 1e-12)
    nb = P[:, 1:, :]
    t_inter = (nb.max(axis=2) / (np.median(nb, axis=2) + 1e-12)).max(axis=1)
    return r_floor, r_main, t_inter


def baseline_predict(stats, th):
    r_floor, r_main, t_inter = stats
    pred = np.zeros(len(r_floor), int)
    pred[(r_floor > th[0]) & (r_main > th[1])] = 1
    pred[t_inter > th[2]] = 2          # a neighbour-root peak takes precedence
    return pred


def baseline_fit(stats, y, n_grid=20):
    """Grid search of the three detection thresholds maximising macro-F1 on the validation set
    (the usual way thresholds of a correlation detector are calibrated)."""
    from sklearn.metrics import f1_score
    grids = [np.quantile(s, np.linspace(0.05, 0.995, n_grid)) for s in stats]
    best, bt = -1, None
    for a in grids[0]:
        for b in grids[1]:
            for c in grids[2]:
                sc = f1_score(y, baseline_predict(stats, (a, b, c)), average="macro")
                if sc > best:
                    best, bt = sc, (a, b, c)
    return bt
