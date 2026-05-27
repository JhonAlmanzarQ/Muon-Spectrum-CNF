import numpy as np
import torch


def logE_to_mm(df, stats):
    """
    Transform energy to the normalized log-energy space used by the model.
    """
    z = np.log(df["energy"].to_numpy(dtype=float))
    z_min = stats["logE"]["min"]
    z_range = stats["logE"]["range"]
    out = df.copy()
    out["logE_mm"] = (z - z_min) / (z_range + 1e-12)
    return out

def mm_to_energy(y_mm, stats):
    """
    Map normalized log-energy back to the physical energy scale.
    """
    z_min = stats["logE"]["min"]
    z_range = stats["logE"]["range"]
    return np.exp(y_mm * z_range + z_min)

def add_mu(df, eps=1e-6):
    """
    Add the angular target mu = 1 - cos(theta).
    """
    out = df.copy()
    mu = 1.0 - np.cos(np.deg2rad(out["theta"].to_numpy(dtype=float)))
    out["mu"] = np.clip(mu, eps, 1.0 - eps)
    return out

def mu_to_theta(mu):
    """
    Convert mu back to the zenith angle in degrees.
    """
    mu = np.clip(mu, 1e-6, 1.0 - 1e-6)
    return np.rad2deg(np.arccos(1.0 - mu))


def con_mm(df, stats, context_cols):
    """
    Normalize the context variables using training-set statistics.
    """
    out = df.copy()
    for c in context_cols:
        c_min = stats[c]["min"]
        c_range = stats[c]["range"]
        out[c + "_mm"] = (out[c] - c_min) / (c_range + 1e-12)
    return out

def energy_spectrum(E, bins_E):
    """
    Compute the binned energy spectrum.
    """
    counts, bins = np.histogram(E, bins=bins_E)
    centers = np.sqrt(bins[:-1] * bins[1:])
    spec = counts / 3600.0
    return centers, spec

def angle_spectrum(theta_deg, bins_theta):
    """
    Compute the binned angular spectrum.
    """
    counts, bins = np.histogram(theta_deg, bins=bins_theta)
    centers = 0.5 * (bins[:-1] + bins[1:])
    spec = counts / 3600.0
    return centers, spec

def abs_relative_error_per_bin(real_spec, model_spec, eps=1e-12):
    """
    Compute the absolute relative error for bins with positive reference values.
    """
    real_spec = np.asarray(real_spec, dtype=float)
    model_spec = np.asarray(model_spec, dtype=float)

    out = np.full_like(real_spec, np.nan, dtype=float)

    mask = (
        np.isfinite(real_spec) &
        np.isfinite(model_spec) &
        (real_spec > eps)
    )

    out[mask] = np.abs((model_spec[mask] - real_spec[mask]) / real_spec[mask])
    return out


def weighted_relative_error_per_bin(real_spec, model_spec, eps=1e-12):
    """
    Compute the per-bin signed error weighted by the total reference spectrum.
    """
    real_spec = np.asarray(real_spec, dtype=float)
    model_spec = np.asarray(model_spec, dtype=float)

    out = np.full_like(real_spec, np.nan, dtype=float)

    mask = (
        np.isfinite(real_spec) &
        np.isfinite(model_spec) &
        (real_spec > eps)
    )

    if not np.any(mask):
        return out

    total_real = np.sum(real_spec[mask])
    if total_real <= 0:
        return out

    out[mask] = (model_spec[mask] - real_spec[mask]) / total_real
    return out

def masked_mean(x):
    """
    Compute the mean ignoring NaN values.
    """
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.any(np.isfinite(x)) else float("nan")

def range_mean(values, centers, low, high):
    """
    Compute the mean over a selected interval of bin centers.
    """
    mask = (centers >= low) & (centers < high)
    return float(np.nanmean(values[mask])) if np.any(mask) else float("nan")

@torch.no_grad()
def sample_model_joint(flow, c, y_min, y_max, mu_min, mu_max, batch, sample):
    """
    Sample from the joint flow and retain only points inside the valid region
    of the target space.
    """
    flow.eval()

    out_cpu = []
    need = int(sample)

    while need > 0:
        y = flow.sample(batch, context=c)
        y = y.reshape(-1, 2)  

        y_logE = y[:, 0]
        y_mu   = y[:, 1]

        mask = (
            (y_logE >= y_min) & (y_logE <= y_max) &
            (y_mu   >= mu_min) & (y_mu   <= mu_max)
        )

        y = y[mask]

        # Shuffle accepted samples before taking the required subset.
        if y.size(0) > 1:
            perm = torch.randperm(y.size(0), device=y.device)
            y = y[perm]

        take = min(need, y.size(0))
        if take > 0:
            out_cpu.append(y[:take].detach().cpu())
            need -= take

    return torch.cat(out_cpu, dim=0).numpy()