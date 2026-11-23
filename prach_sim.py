"""
prach_sim.py -- 3GPP-style 5G NR PRACH Format 0 simulator for interference detection.

Signal chain (one sample = one PRACH occasion observed at the gNB):
  desired UE preamble  (own-cell root u0, random cyclic shift v_d, timing advance TA_d)
  + optional interferer (class 1: intra-cell = same root u0, other cyclic shift;
                         class 2: inter-cell = neighbour-cell root, random delay)
  each through an independent multipath fading channel (ETU or EPA delay profile,
  Jakes Doppler via sum-of-sinusoids, N_RX receive antennas), with its own CFO,
  + AWGN at the requested SNR (defined on the PRACH subcarriers).
Receiver: CP removal, unitary FFT, extraction of the 839 PRACH subcarriers,
frequency-domain correlation with candidate root sequences (power delay profiles).

Numerology (TS 38.211, Format 0): L_RA = 839, Delta f_RA = 1.25 kHz,
N_u = 24576 kappa*Tc (0.8 ms), N_CP = 3168 kappa*Tc (103.1 us).
The simulator works at fs = 1.28 MHz (N_FFT = 1024 = 24576/24, N_CP = 132 = 3168/24),
which is sufficient because the PRACH occupies 839 x 1.25 kHz = 1.05 MHz.
"""
import numpy as np

# ----------------------------------------------------------------- constants
L_RA = 839
SCS = 1.25e3
N_FFT = 1024
FS = N_FFT * SCS                 # 1.28 MHz
N_CP = 132                       # 3168 samples @ 30.72 MHz -> 132 @ 1.28 MHz
N_CS = 26                        # zeroCorrelationZoneConfig = 5 (unrestricted set)
N_PRE_PER_ROOT = L_RA // N_CS    # 32 preambles per root
PAD = 96                         # zero padding (75 us) for delayed arrivals

# 3GPP TS 36.104 Annex B delay profiles (delays in ns, relative powers in dB)
DELAY_PROFILES = {
    "ETU": ([0, 50, 120, 200, 230, 500, 1600, 2300, 5000],
            [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, -3.0, -5.0, -7.0]),
    "EPA": ([0, 30, 70, 90, 110, 190, 410],
            [0.0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8]),
}

DEFAULT_CFG = dict(
    u0=129,                          # own-cell physical root index
    neighbour_roots=(300, 501, 688), # roots of the neighbouring cells (known to the gNB)
    unseen_roots=(77, 412, 777),     # roots never seen in training (robustness test)
    profile="ETU", doppler_hz=70.0, n_rx=2, n_sin=16,
    cfo_desired_hz=270.0, cfo_interf_max_hz=270.0,
    ta_max_us=14.0,                  # desired and intra-cell UEs (inside the ZCZ)
    inter_delay_max_us=40.0,         # inter-cell interferer: unsynchronised arrival
)


def zc_root(u):
    n = np.arange(L_RA)
    return np.exp(-1j * np.pi * u * n * (n + 1) / L_RA)


def preamble_freq(u, v):
    """Frequency-domain preamble y_{u,v}(k) per TS 38.211 6.3.3.1 (unit average power)."""
    x = zc_root(u)
    x_uv = np.roll(x, -v * N_CS)                # x_u((n + C_v) mod L_RA), C_v = v*N_CS
    return np.fft.fft(x_uv) / np.sqrt(L_RA)


# PRACH subcarriers centred in the FFT grid
_SC = (np.arange(L_RA) - L_RA // 2) % N_FFT


def preamble_time(u, v):
    """Time-domain PRACH symbol with CP at fs = 1.28 MHz, zero padded."""
    grid = np.zeros(N_FFT, complex)
    grid[_SC] = preamble_freq(u, v)
    s = np.fft.ifft(grid) * np.sqrt(N_FFT)      # unitary -> unit power per used subcarrier
    s = np.concatenate([s[-N_CP:], s])
    return np.concatenate([s, np.zeros(PAD, complex)])


N_TOT = N_CP + N_FFT + PAD
_t = np.arange(N_TOT) / FS
_f = np.fft.fftfreq(N_TOT, 1 / FS)
_tc = np.linspace(0, _t[-1], 17)
_W_INTERP = np.zeros((len(_tc), N_TOT))
for _j in range(len(_tc)):
    _e = np.zeros(len(_tc)); _e[_j] = 1.0
    _W_INTERP[_j] = np.interp(_t, _tc, _e)


def _fading_channel(rng, B, profile, fd, n_rx, n_sin):
    """Time-varying tap gains, shape (B, n_rx, n_taps, N_TOT); total average power 1."""
    delays_ns, p_db = DELAY_PROFILES[profile]
    p = 10 ** (np.asarray(p_db) / 10)
    p = p / p.sum()
    L = len(p)
    alpha = rng.uniform(0, 2 * np.pi, (B, n_rx, L, n_sin, 1))
    phi = rng.uniform(0, 2 * np.pi, (B, n_rx, L, n_sin, 1))
    # sum-of-sinusoids evaluated on a coarse grid (the burst lasts ~1 ms << 1/fd) and
    # linearly interpolated to the sample grid
    gc = np.exp(1j * (2 * np.pi * fd * np.cos(alpha) * _tc + phi)).sum(axis=3) / np.sqrt(n_sin)
    g = gc @ _W_INTERP
    g *= np.sqrt(p)[None, None, :, None]
    return g, np.asarray(delays_ns) * 1e-9


def _propagate(rng, s, B, delay_s, cfo_hz, power, cfg):
    """Pass B copies of the time signal s through independent channels.
    delay_s, cfo_hz, power: arrays of shape (B,). Returns (B, n_rx, N_TOT)."""
    g, tap_d = _fading_channel(rng, B, cfg["profile"], cfg["doppler_hz"], cfg["n_rx"], cfg["n_sin"])
    S = np.fft.fft(s)
    tot_d = delay_s[:, None] + tap_d[None, :]                   # (B, L)
    ramps = np.exp(-2j * np.pi * _f[None, None, :] * tot_d[:, :, None])
    delayed = np.fft.ifft(S[None, None, :] * ramps, axis=-1)     # (B, L, N_TOT) fractional delays
    y = np.einsum("brln,bln->brn", g, delayed)
    y *= np.exp(2j * np.pi * cfo_hz[:, None, None] * _t[None, None, :])
    return y * np.sqrt(power)[:, None, None]


def receiver_freq(y):
    """CP removal + unitary FFT + PRACH subcarrier extraction. y: (B, n_rx, N_TOT) -> (B, n_rx, 839)."""
    Y = np.fft.fft(y[..., N_CP:N_CP + N_FFT], axis=-1) / np.sqrt(N_FFT)
    return Y[..., _SC]


_ROOT_FREQ = {}


def _root_conj(u):
    if u not in _ROOT_FREQ:
        _ROOT_FREQ[u] = np.conj(preamble_freq(u, 0))
    return _ROOT_FREQ[u]


def correlation_maps(Yf, roots):
    """Power delay profiles P_u[tau] = sum_a |IDFT(Y_a[k] conj(X_u[k]))|^2,
    returns (B, len(roots), 839)."""
    out = []
    for u in roots:
        r = np.fft.ifft(Yf * _root_conj(u)[None, None, :], axis=-1) * np.sqrt(L_RA)
        out.append((np.abs(r) ** 2).sum(axis=1))
    return np.stack(out, axis=1)


def generate(rng, n_per_cell, snr_db_list, isr_db_list, cfg=None, inter_roots=None, batch=128):
    """Balanced dataset: for each (class, SNR, ISR) cell, n_per_cell independent occasions.
    Class 0 (no interference) is generated for every ISR value too, so that classes are balanced
    and every (SNR, ISR) bin contains the three classes (ISR is irrelevant for class 0).
    Returns dict with Yf (B,n_rx,839) complex64, labels, snr, isr, metadata."""
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    inter_roots = tuple(inter_roots or cfg["neighbour_roots"])
    jobs = [(c, s, i) for c in (0, 1, 2) for s in snr_db_list for i in isr_db_list]
    Yf_all, lab, snr, isr, meta = [], [], [], [], []
    s_cache = {}

    def st(u, v):
        if (u, v) not in s_cache:
            s_cache[(u, v)] = preamble_time(u, v)
        return s_cache[(u, v)]

    for (c, s_db, i_db) in jobs:
        done = 0
        while done < n_per_cell:
            B = min(batch, n_per_cell - done)
            v_d = rng.integers(0, N_PRE_PER_ROOT, B)
            ta_d = rng.uniform(0, cfg["ta_max_us"], B) * 1e-6
            y = np.zeros((B, cfg["n_rx"], N_TOT), complex)
            # desired UEs: group by preamble index to reuse the base waveform
            for v in np.unique(v_d):
                m = v_d == v
                y[m] += _propagate(rng, st(cfg["u0"], v), m.sum(), ta_d[m],
                                   np.full(m.sum(), cfg["cfo_desired_hz"]), np.ones(m.sum()), cfg)
            u_i = np.full(B, -1)
            v_i = np.full(B, -1)
            if c > 0:
                if c == 1:   # intra-cell: same root, different cyclic shift, synchronised (TA in ZCZ)
                    u_i[:] = cfg["u0"]
                    v_i = (v_d + rng.integers(1, N_PRE_PER_ROOT, B)) % N_PRE_PER_ROOT
                    d_i = rng.uniform(0, cfg["ta_max_us"], B) * 1e-6
                else:        # inter-cell: neighbour root, any preamble, unsynchronised delay
                    u_i = rng.choice(inter_roots, B)
                    v_i = rng.integers(0, N_PRE_PER_ROOT, B)
                    d_i = rng.uniform(0, cfg["inter_delay_max_us"], B) * 1e-6
                cfo_i = rng.uniform(-cfg["cfo_interf_max_hz"], cfg["cfo_interf_max_hz"], B)
                p_i = np.full(B, 10 ** (i_db / 10))
                for key in set(zip(u_i, v_i)):
                    m = (u_i == key[0]) & (v_i == key[1])
                    y[m] += _propagate(rng, st(*key), m.sum(), d_i[m], cfo_i[m], p_i[m], cfg)
            # AWGN: unit-power preamble per subcarrier -> noise variance 1/SNR per sample
            sigma2 = 10 ** (-s_db / 10)
            y += np.sqrt(sigma2 / 2) * (rng.standard_normal(y.shape) + 1j * rng.standard_normal(y.shape))
            Yf_all.append(receiver_freq(y).astype(np.complex64))
            lab += [c] * B
            snr += [s_db] * B
            isr += [i_db if c > 0 else np.nan] * B
            meta.append(np.c_[v_d, ta_d * 1e6, u_i, v_i])
            done += B
    return dict(Yf=np.concatenate(Yf_all), y=np.array(lab), snr=np.array(snr, float),
                isr=np.array(isr, float), isr_bin=np.repeat([i for (_, _, i) in jobs], n_per_cell).astype(float),
                meta=np.concatenate(meta), cfg=cfg)
