"""NDVI-change baseline and the compact dense autoencoder template."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_json, save_json


# --------------------------------------------------------------------------- data

def load_dataset(cfg: dict) -> pd.DataFrame:
    """Prepared records joined with the split assignment, buffer cells removed."""
    from .config import get_path

    proc = get_path(cfg, "processed_dir")
    data = pd.read_parquet(proc / "prepared.parquet")
    splits = pd.read_parquet(proc / "splits.parquet")
    df = data.merge(splits, on="cell_id", how="inner", validate="one_to_one")
    return df[~df["buffer_dropped"]].sort_values("cell_id").reset_index(drop=True)


# --------------------------------------------------------------------------- baseline

def baseline_score(df: pd.DataFrame) -> np.ndarray:
    """Larger NDVI decrease = more likely loss. One-sided, unlike reconstruction error."""
    return -df["d_ndvi"].to_numpy(float)


# --------------------------------------------------------------------------- scaling

@dataclass
class Scaler:
    columns: list[str]
    mean: list[float]
    std: list[float]

    @classmethod
    def fit(cls, df: pd.DataFrame, columns: list[str]) -> "Scaler":
        x = df[columns].to_numpy(float)
        std = x.std(axis=0)
        std[std == 0] = 1.0
        return cls(list(columns), x.mean(axis=0).tolist(), std.tolist())

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.columns].to_numpy(np.float32)
        return ((x - np.asarray(self.mean, np.float32)) / np.asarray(self.std, np.float32)).astype(np.float32)

    def save(self, path: Path) -> None:
        save_json(asdict(self), path)

    @classmethod
    def load(cls, path: Path) -> "Scaler":
        return cls(**load_json(path))


# --------------------------------------------------------------------------- autoencoder

def ae_params(cfg: dict, **overrides) -> dict:
    a = cfg["autoencoder"]
    p = {k: a[k] for k in ("hidden_units", "latent_divisor", "noise_std", "learning_rate", "batch_size", "max_epochs", "patience")}
    p.update(overrides)
    return p


def build_autoencoder(n_features: int, params: dict, seed: int):
    import keras
    from keras import layers

    keras.utils.set_random_seed(seed)
    latent = max(1, n_features // params["latent_divisor"])
    inputs = keras.Input(shape=(n_features,), name="features")
    # Input noise (training only) keeps the 1-feature model from learning an identity map:
    # it learns a shrinkage towards typical stable values, so unusual inputs reconstruct badly.
    h = layers.GaussianNoise(params["noise_std"], seed=seed)(inputs)
    h = layers.Dense(params["hidden_units"], activation="relu")(h)
    z = layers.Dense(latent, activation="linear", name="latent")(h)
    h = layers.Dense(params["hidden_units"], activation="relu")(z)
    outputs = layers.Dense(n_features, activation="linear")(h)
    model = keras.Model(inputs, outputs, name=f"ae_{n_features}in_{latent}lat")
    model.compile(optimizer=keras.optimizers.Adam(params["learning_rate"]), loss="mse")
    return model


def train_autoencoder(x_train: np.ndarray, x_val: np.ndarray, params: dict, seed: int):
    """Fit on stable cells only; early stopping on stable validation reconstruction loss."""
    import keras

    model = build_autoencoder(x_train.shape[1], params, seed)
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=params["patience"], restore_best_weights=True)
    t0 = time.perf_counter()
    hist = model.fit(
        x_train, x_train,
        validation_data=(x_val, x_val),
        epochs=params["max_epochs"],
        batch_size=params["batch_size"],
        shuffle=True,
        callbacks=[stop],
        verbose=0,
    )
    seconds = time.perf_counter() - t0
    val_loss = hist.history["val_loss"]
    info = {
        "train_seconds": seconds,
        "epochs_completed": len(val_loss),
        "best_epoch": int(np.argmin(val_loss)) + 1,
        "best_val_loss": float(np.min(val_loss)),
        "stopped_early": len(val_loss) < params["max_epochs"],
        "n_params": int(model.count_params()),
        "n_train": int(len(x_train)),
        "n_val_stable": int(len(x_val)),
    }
    return model, info


def reconstruction_error(model, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    recon = model.predict(x, batch_size=batch_size, verbose=0)
    return np.mean(np.square(recon - x), axis=1)
