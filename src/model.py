import time
import torch
from nflows import transforms, distributions, flows

def build_flow(hidden_features, num_layers, num_bins, tail_bound, context_features=3):
    """
    Build a conditional normalizing flow for the joint target [logE_mm, mu].
    """
    layers = []

    for _ in range(num_layers):
        layers.append(
            transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=2,  # joint target: [logE_mm, mu]
                hidden_features=hidden_features,
                num_bins=num_bins,
                tails="linear", 
                tail_bound=float(tail_bound),
                context_features=context_features,  
                min_bin_width=1e-3,
                min_bin_height=1e-4,
                min_derivative=1e-3
            )
        )
        layers.append(transforms.ReversePermutation(features=2))

    # Compose all transforms into a single invertible mapping
    transform = transforms.CompositeTransform(layers)
    base_dist = distributions.StandardNormal(shape=[2])  

    return flows.Flow(transform, base_dist)


@torch.no_grad()
def val_flow(flow, loader, device):
    """
    Compute the mean validation negative log-likelihood.
    """
    flow.eval()
    total = 0.0
    n = 0

    for xb, cb in loader:
        xb = xb.to(device, non_blocking=True)
        cb = cb.to(device, non_blocking=True)

        loss = -flow.log_prob(xb, context=cb).mean()

        bs = xb.size(0)
        total += float(loss.item()) * bs
        n += bs

    return total / max(n, 1)


def train_flow(flow, train_loader, val_loader, device, epochs, lr=3e-4):
    """
    Train the flow by minimizing the negative log-likelihood.
    """
    flow = flow.to(device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)

    history = []
    t0 = time.time()

    for ep in range(1, epochs + 1):
        flow.train()
        running = 0.0
        n = 0

        for xb, cb in train_loader:
            xb = xb.to(device, non_blocking=True)
            cb = cb.to(device, non_blocking=True)

            # Negative log-likelihood to minimize
            loss = -flow.log_prob(xb, context=cb).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Gradient clipping helps stabilize training.
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            optimizer.step()

            bs = xb.size(0)
            running += float(loss.item()) * bs
            n += bs

        train_nll = running / max(n, 1)

        if val_loader is not None:
            # Preserve RNG states so validation does not affect reproducibility.
            cpu_rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

            val_nll = val_flow(flow, val_loader, device)

            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
        else:
            val_nll = float("nan")

        history.append({
            "epoch": ep,
            "train_nll": float(train_nll),
            "val_nll": float(val_nll),
        })

        if ep % 5 == 0 or ep == 1 or ep == epochs:
            print(f"[ep {ep:02d}/{epochs}] train NLL: {train_nll:.6f} | val NLL: {val_nll:.6f}")

    print(f"Done. Total time: {(time.time() - t0)/60:.2f} min")
    return flow, history