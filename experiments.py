"""
experiments.py -- multi-run benchmark of PRACH interference detectors.

For every run r = 1..N_RUNS a fresh, independent dataset is generated (disjoint random
streams for train / validation / test / robustness sets) and every detector is retrained
with its own seed. Results are appended to CSV files after each (run, model) so that an
interrupted Colab session can be resumed.
"""
import os, json, time, platform
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix)
import prach_sim as ps
import features as ft

CLASSES = ["none", "intra", "inter"]
MODELS = ["Correlation", "XGBoost", "CNN", "Conv1D", "LSTM", "Transformer", "Conv1D-IQ"]

DEFAULT_EXP = dict(
    n_runs=10,
    snr_db=[-18, -15, -12, -9, -6, -3, 0, 5, 10],
    isr_db=[-9, -6, -3, 0, 3],
    n_train_cell=150, n_val_cell=40, n_test_cell=60, n_rob_cell=30,
    epochs=60, batch=64, lr=1e-3, patience=8,
    models=MODELS,
    out_dir="results",
)


# =============================================================== deep models (Keras)
def _keras():
    import tensorflow as tf
    from tensorflow.keras import layers as L, Model, Input
    return tf, L, Model, Input


def build_cnn(shape):
    tf, L, Model, Input = _keras()
    x0 = Input(shape)                                   # (R=4 roots, 839 delays, 1)
    x = x0
    for f, k in [(16, (2, 7)), (32, (2, 5)), (64, (2, 3))]:
        x = L.Conv2D(f, k, padding="same")(x)
        x = L.BatchNormalization()(x)
        x = L.ReLU()(x)
        x = L.MaxPooling2D((1, 2))(x)
    x = L.GlobalMaxPooling2D()(x)
    x = L.Dense(64, activation="relu")(x)
    x = L.Dropout(0.3)(x)
    return Model(x0, L.Dense(3, activation="softmax")(x), name="CNN")


def build_conv1d(shape, name="Conv1D"):
    tf, L, Model, Input = _keras()
    x0 = Input(shape)              # (839 delays, R roots) or, for the ablation, (839 subcarriers, 2*N_RX)
    x = x0
    for f, k, d in [(32, 9, 1), (64, 7, 1), (64, 5, 2), (64, 5, 4)]:
        x = L.Conv1D(f, k, padding="same", dilation_rate=d)(x)
        x = L.BatchNormalization()(x)
        x = L.ReLU()(x)
        if d == 1:
            x = L.MaxPooling1D(2)(x)
    x = L.Concatenate()([L.GlobalAveragePooling1D()(x), L.GlobalMaxPooling1D()(x)])
    x = L.Dense(64, activation="relu")(x)
    x = L.Dropout(0.3)(x)
    return Model(x0, L.Dense(3, activation="softmax")(x), name=name)


def build_lstm(shape):
    tf, L, Model, Input = _keras()
    x0 = Input(shape)                                   # (T tokens, R*frame)
    x = L.TimeDistributed(L.Dense(64, activation="relu"))(x0)   # token embedding
    x = L.Bidirectional(L.LSTM(64, return_sequences=True))(x)
    x = L.Bidirectional(L.LSTM(64, return_sequences=True))(x)
    x = L.Concatenate()([L.GlobalMaxPooling1D()(x), L.GlobalAveragePooling1D()(x)])  # temporal pooling
    x = L.Dense(64, activation="relu")(x)
    x = L.Dropout(0.3)(x)
    return Model(x0, L.Dense(3, activation="softmax")(x), name="LSTM")


def build_transformer(shape, d_model=64, heads=4, ff=128, blocks=2, drop=0.1):
    """Pre-LayerNorm Transformer encoder (more stable to train than the post-LN variant)."""
    tf, L, Model, Input = _keras()
    T = shape[0]
    pos = np.arange(T)[:, None]
    i = np.arange(d_model)[None, :]
    ang = pos / np.power(10000, (2 * (i // 2)) / d_model)
    pe = np.where(i % 2 == 0, np.sin(ang), np.cos(ang)).astype("float32")[None]
    x0 = Input(shape)
    x = L.Dense(d_model)(L.LayerNormalization()(x0)) + tf.constant(pe)
    for _ in range(blocks):
        h = L.LayerNormalization()(x)
        x = x + L.Dropout(drop)(L.MultiHeadAttention(heads, d_model // heads, dropout=drop)(h, h))
        h = L.LayerNormalization()(x)
        x = x + L.Dropout(drop)(L.Dense(d_model)(L.Dense(ff, activation="relu")(h)))
    x = L.LayerNormalization()(x)
    x = L.Concatenate()([L.GlobalMaxPooling1D()(x), L.GlobalAveragePooling1D()(x)])
    x = L.Dense(64, activation="relu")(x)
    x = L.Dropout(0.3)(x)
    return Model(x0, L.Dense(3, activation="softmax")(x), name="Transformer")


BUILDERS = {"CNN": (build_cnn, "cmap"), "Conv1D": (build_conv1d, "seq"),
            "Conv1D-IQ": (lambda sh: build_conv1d(sh, "Conv1D_IQ"), "iq"),
            "LSTM": (build_lstm, "tok"), "Transformer": (build_transformer, "tok")}


# =============================================================== helpers
def make_split(seed, n_cell, exp, inter_roots=None, cfg=None):
    return ps.generate(np.random.default_rng(seed), n_cell, exp["snr_db"], exp["isr_db"],
                       cfg=cfg, inter_roots=inter_roots)


def prepare(d, cfg):
    t = time.perf_counter()
    cmap, iq, tok = ft.deep_inputs(d["Yf"], cfg)
    t_deep = (time.perf_counter() - t) / len(d["y"])
    t = time.perf_counter()
    F = ft.engineered_features(d["Yf"], cfg)
    t_feat = (time.perf_counter() - t) / len(d["y"])
    seq = cmap[..., 0].transpose(0, 2, 1).copy()      # (B, 839, R): roots as channels
    return dict(cmap=cmap, seq=seq, iq=iq, tok=tok, feat=F, t_deep=t_deep, t_feat=t_feat)


def metrics(y, p):
    m = dict(accuracy=accuracy_score(y, p),
             macro_f1=f1_score(y, p, average="macro"),
             weighted_f1=f1_score(y, p, average="weighted"),
             macro_precision=precision_score(y, p, average="macro", zero_division=0),
             macro_recall=recall_score(y, p, average="macro", zero_division=0))
    pc = precision_score(y, p, average=None, labels=[0, 1, 2], zero_division=0)
    rc = recall_score(y, p, average=None, labels=[0, 1, 2], zero_division=0)
    fc = f1_score(y, p, average=None, labels=[0, 1, 2], zero_division=0)
    for k, c in enumerate(CLASSES):
        m[f"precision_{c}"], m[f"recall_{c}"], m[f"f1_{c}"] = pc[k], rc[k], fc[k]
    yb, pb = (y > 0).astype(int), (p > 0).astype(int)          # interference present vs absent
    m.update(bin_precision=precision_score(yb, pb, zero_division=0),
             bin_recall=recall_score(yb, pb, zero_division=0), bin_f1=f1_score(yb, pb, zero_division=0))
    return m


def per_condition(run, model, tag, d, p):
    rows = []
    for s in np.unique(d["snr"]):
        for i in np.unique(d["isr_bin"]):
            m = (d["snr"] == s) & (d["isr_bin"] == i)
            y, q = d["y"][m], p[m]
            rows.append(dict(run=run, model=model, test_set=tag, snr_db=s, isr_db=i, n=int(m.sum()),
                             n_correct=int((y == q).sum()),
                             **{f"n_{c}": int((y == k).sum()) for k, c in enumerate(CLASSES)},
                             **{f"correct_{c}": int(((y == k) & (q == k)).sum()) for k, c in enumerate(CLASSES)},
                             **{f"pred_{c}": int((q == k).sum()) for k, c in enumerate(CLASSES)}))
    return rows


def _append(path, rows):
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def hardware():
    info = dict(platform=platform.platform(), processor=platform.processor(), python=platform.python_version())
    try:
        import tensorflow as tf
        info["tensorflow"] = tf.__version__
        info["gpus"] = [g.name for g in tf.config.list_physical_devices("GPU")]
        try:
            info["gpu_details"] = [tf.config.experimental.get_device_details(g) for g in tf.config.list_physical_devices("GPU")]
        except Exception:
            pass
    except Exception:
        pass
    try:
        import xgboost
        info["xgboost"] = xgboost.__version__
    except Exception:
        pass
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
        info["cpu_count"] = psutil.cpu_count()
    except Exception:
        info["cpu_count"] = os.cpu_count()
    return info


# =============================================================== one run
def run_once(r, exp, done, log=print):
    out = exp["out_dir"]
    cfg = dict(ps.DEFAULT_CFG)
    base = 1000 * r
    t0 = time.perf_counter()
    tr = make_split(base + 1, exp["n_train_cell"], exp)
    va = make_split(base + 2, exp["n_val_cell"], exp)
    te = make_split(base + 3, exp["n_test_cell"], exp)
    rob_root = make_split(base + 4, exp["n_rob_cell"], exp, inter_roots=cfg["unseen_roots"])
    rob_epa = make_split(base + 5, exp["n_rob_cell"], exp, cfg=dict(profile="EPA", doppler_hz=5.0))
    log(f"[run {r}] data generated in {time.perf_counter()-t0:.0f}s "
        f"(train {len(tr['y'])}, val {len(va['y'])}, test {len(te['y'])})")
    if r == 1:
        rows = []
        for name, d in [("train", tr), ("validation", va), ("test", te),
                        ("test_unseen_roots", rob_root), ("test_EPA5", rob_epa)]:
            g = pd.DataFrame(dict(cls=[CLASSES[k] for k in d["y"]], snr=d["snr"], isr=d["isr_bin"]))
            for (c, s, i), n in g.groupby(["cls", "snr", "isr"]).size().items():
                rows.append(dict(split=name, cls=c, snr_db=s, isr_db=i, n=n))
        pd.DataFrame(rows).to_csv(os.path.join(out, "dataset_summary.csv"), index=False)
    X = {k: prepare(d, cfg) for k, d in [("tr", tr), ("va", va), ("te", te), ("rr", rob_root), ("re", rob_epa)]}
    sets = {"test": te, "test_unseen_roots": rob_root, "test_EPA5": rob_epa}
    keymap = {"test": "te", "test_unseen_roots": "rr", "test_EPA5": "re"}

    for name in exp["models"]:
        if (r, name) in done:
            continue
        log(f"[run {r}] {name} ...")
        extra = dict(thresholds="")
        if name == "Correlation":
            stats = {k: ft.baseline_stats(d["Yf"], cfg) for k, d in [("va", va)] + [(keymap[s], sets[s]) for s in sets]}
            t = time.perf_counter()
            th = ft.baseline_fit(stats["va"], va["y"])
            fit_s = time.perf_counter() - t
            pred = {}
            for s in sets:
                t = time.perf_counter()
                pred[s] = ft.baseline_predict(stats[keymap[s]], th)
                if s == "test":
                    inf = (time.perf_counter() - t) / len(te["y"])
            n_params, epochs, pre = 3, np.nan, X["te"]["t_deep"]
            extra = dict(thresholds=json.dumps([float(v) for v in th]))
        elif name == "XGBoost":
            import xgboost as xgb
            m = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1, subsample=0.8,
                                  colsample_bytree=0.8, reg_lambda=1.0, gamma=0.1, tree_method="hist",
                                  objective="multi:softprob", eval_metric="mlogloss",
                                  early_stopping_rounds=20, random_state=r, n_jobs=-1)
            t = time.perf_counter()
            m.fit(X["tr"]["feat"], tr["y"], eval_set=[(X["va"]["feat"], va["y"])], verbose=False)
            fit_s = time.perf_counter() - t
            pred = {}
            for s in sets:
                t = time.perf_counter()
                pred[s] = m.predict(X[keymap[s]]["feat"])
                if s == "test":
                    inf = (time.perf_counter() - t) / len(te["y"])
            epochs = int(m.best_iteration) + 1
            n_params = int(sum(len(tdf) for _, tdf in m.get_booster().trees_to_dataframe().groupby("Tree")))
            pre = X["te"]["t_feat"]
            if r == 1:
                imp = pd.DataFrame(dict(feature=ft.FEATURE_NAMES, gain=m.feature_importances_))
                imp.to_csv(os.path.join(out, "xgb_feature_importance.csv"), index=False)
        else:
            import tensorflow as tf
            tf.keras.utils.set_random_seed(r)
            builder, key = BUILDERS[name]
            model = builder(X["tr"][key].shape[1:])
            model.compile(tf.keras.optimizers.Adam(exp["lr"], clipnorm=1.0),
                          loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            if r == 1:
                with open(os.path.join(out, "model_summaries.txt"), "a") as fh:
                    model.summary(print_fn=lambda s, **k: fh.write(s + "\n"))
                    fh.write(f"Trainable parameters: {int(sum(np.prod(v.shape) for v in model.trainable_weights))}\n\n")
            cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=exp["patience"],
                                                   restore_best_weights=True)]
            t = time.perf_counter()
            h = model.fit(X["tr"][key], tr["y"], validation_data=(X["va"][key], va["y"]),
                          epochs=exp["epochs"], batch_size=exp["batch"], callbacks=cb, verbose=0)
            fit_s = time.perf_counter() - t
            if r == 1:
                pd.DataFrame(h.history).assign(epoch=lambda d: np.arange(1, len(d) + 1)).to_csv(
                    os.path.join(out, f"history_{name}.csv"), index=False)
            pred = {}
            model.predict(X["te"][key][:256], batch_size=256, verbose=0)     # warm-up
            for s in sets:
                t = time.perf_counter()
                pred[s] = model.predict(X[keymap[s]][key], batch_size=256, verbose=0).argmax(1)
                if s == "test":
                    inf = (time.perf_counter() - t) / len(te["y"])
            epochs = len(h.history["loss"])
            n_params = int(sum(np.prod(v.shape) for v in model.trainable_weights))
            pre = X["te"]["t_deep"]
            tf.keras.backend.clear_session()

        row = dict(run=r, model=name, fit_s=fit_s, infer_ms_per_sample=1e3 * inf,
                   preproc_ms_per_sample=1e3 * pre, n_params=n_params, epochs=epochs,
                   **metrics(te["y"], pred["test"]), **extra)
        for s in ["test_unseen_roots", "test_EPA5"]:
            mm = metrics(sets[s]["y"], pred[s])
            row[f"{s}_accuracy"], row[f"{s}_macro_f1"] = mm["accuracy"], mm["macro_f1"]
        _append(os.path.join(out, "runs.csv"), [row])
        pc = []
        for s in sets:
            pc += per_condition(r, name, s, sets[s], pred[s])
        _append(os.path.join(out, "per_condition.csv"), pc)
        cm = confusion_matrix(te["y"], pred["test"], labels=[0, 1, 2])
        _append(os.path.join(out, "confusion.csv"),
                [dict(run=r, model=name, true=CLASSES[i], pred=CLASSES[j], count=int(cm[i, j]))
                 for i in range(3) for j in range(3)])
        log(f"[run {r}] {name}: acc={row['accuracy']:.4f} macroF1={row['macro_f1']:.4f} "
            f"fit={fit_s:.1f}s inf={row['infer_ms_per_sample']:.3f}ms/sample")


def run_all(exp=None, log=print):
    exp = dict(DEFAULT_EXP, **(exp or {}))
    os.makedirs(exp["out_dir"], exist_ok=True)
    with open(os.path.join(exp["out_dir"], "config.json"), "w") as fh:
        json.dump(dict(experiment=exp, simulator=ps.DEFAULT_CFG, hardware=hardware(),
                       numerology=dict(L_RA=ps.L_RA, SCS=ps.SCS, N_FFT=ps.N_FFT, FS=ps.FS, N_CP=ps.N_CP,
                                       N_CS=ps.N_CS)), fh, indent=2, default=str)
    path = os.path.join(exp["out_dir"], "runs.csv")
    done = set()
    if os.path.exists(path):
        d = pd.read_csv(path)
        done = set(zip(d["run"], d["model"]))
    for r in range(1, exp["n_runs"] + 1):
        if all((r, m) in done for m in exp["models"]):
            continue
        run_once(r, exp, done, log)
