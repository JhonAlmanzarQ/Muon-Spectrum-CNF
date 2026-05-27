import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import random

RNG = np.random.default_rng(42)

def load_sites(sites, files_map, data_base):
    """
    Load the CSV files associated with a list of sites and merge them
    into a single DataFrame. A 'site' column is added to preserve the
    origin of each event.
    """
    dfs = []
    
    for site in sites:
        path = data_base / files_map[site]
        df = pd.read_csv(path)
        df["site"] = site  # keep site label
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if len(dfs) else pd.DataFrame()


def downsample_per_site_joint(df, target_per_site, bins_energy, bins_angle, RNG, EPS_ANG=1e-6):
    """
    Downsample each site independently using a joint binning defined over
    log-energy and angle. This preserves the local shape of both variables
    while enforcing a common sample size per site.
    """
    dfs_out = []

    for site, df_site in df.groupby("site"):
        n_site = len(df_site)
        if n_site == 0:
            print(f"Site without data: {site}")
            continue

        # Sampling factor needed to reach the target size for this site.
        factor = target_per_site / n_site

        # Variables used to construct the joint bins.
        logE_site = np.log(df_site["energy"].to_numpy(dtype=float))
        theta = df_site["theta"].to_numpy(dtype=float)
        ang_site = np.clip(theta, EPS_ANG, 90.0 - EPS_ANG)

        e_ids = np.digitize(logE_site, bins_energy[1:-1], right=False)
        a_ids = np.digitize(ang_site, bins_angle[1:-1], right=False)

        # Joint stratum index
        joint_ids = e_ids * (len(bins_angle) - 1) + a_ids

        indices_keep = []

        for jid in np.unique(joint_ids):
            idx_bin = np.flatnonzero(joint_ids == jid)
            n_bin = len(idx_bin)
            if n_bin == 0:
                continue

            # Keep approximately the same proportion from each joint bin.
            n_keep = int(round(n_bin * factor))
            if n_keep == 0:
                continue

            if n_keep >= n_bin:
                chosen = idx_bin
            else:
                chosen = RNG.choice(idx_bin, size=n_keep, replace=False)

            indices_keep.append(chosen)

        # Fallback in case no joint bin contributes samples.
        if len(indices_keep) == 0:
            print(f"Fallback in {site}")
            all_idx = np.arange(n_site)
            if n_site >= target_per_site:
                indices_keep = RNG.choice(all_idx, size=target_per_site, replace=False)
            else:
                indices_keep = RNG.choice(all_idx, size=target_per_site, replace=True)
        else:
            indices_keep = np.concatenate(indices_keep)

            # Trim or complete the sample to match the exact target size.
            if len(indices_keep) > target_per_site:
                indices_keep = RNG.choice(indices_keep, size=target_per_site, replace=False)

            elif len(indices_keep) < target_per_site:
                all_idx = np.arange(n_site)
                remaining = np.setdiff1d(all_idx, indices_keep, assume_unique=False)

                need = target_per_site - len(indices_keep)
                if len(remaining) >= need:
                    extra = RNG.choice(remaining, size=need, replace=False)
                else:
                    extra = RNG.choice(all_idx, size=need, replace=True)

                indices_keep = np.concatenate([indices_keep, extra])

        df_down = df_site.iloc[indices_keep]
        dfs_out.append(df_down)

    return pd.concat(dfs_out, ignore_index=True)


def make_loader(tensor_ds, batch_size, shuffle, num_workers=2):
    # Build a PyTorch data loader
    return DataLoader(
        tensor_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers
    )