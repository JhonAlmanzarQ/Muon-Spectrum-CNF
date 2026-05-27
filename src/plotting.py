import numpy as np
import matplotlib.pyplot as plt

def save_energy_plot(city, bin_centers_E, spec_real_E, spec_mean_E, spec_std_E,
                     contrib_mean_E, contrib_std_E, save_path):
    """
    Save the energy spectrum comparison together with the weighted
    relative error contribution in a two-panel figure.
    """
    # Upper panel: spectra. Lower panel: error contribution.
    fig, (ax, axr) = plt.subplots(
        2, 1,
        figsize=(8, 8),
        dpi=130,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0}
    )

    # Reference and model spectra are shown as step curves to match histogram bins.
    h_real, = ax.step(bin_centers_E, spec_real_E, where="mid", label="ARTI")
    h_cnf, = ax.step(bin_centers_E, spec_mean_E, where="mid", linestyle="--", label="CNFs")

    # The shaded band summarizes the dispersion across evaluation seeds.
    h_band = ax.fill_between(
        bin_centers_E,
        np.clip(spec_mean_E - spec_std_E, 0, None),
        spec_mean_E + spec_std_E,
        step="mid",
        alpha=0.25,
        label="±1σ"
    )

    # Log-log scale is used because the energy spectrum spans several orders of magnitude.
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("part/(m²·s)")
    ax.set_title(f"Energy spectrum — {city.upper()}")
    ax.grid(True, which="both", ls="--", lw=0.5)

    # Lower panel: weighted relative error contribution by bin.
    h_err, = axr.step(bin_centers_E, contrib_mean_E, where="mid", label="Contribution")
    axr.fill_between(
        bin_centers_E,
        contrib_mean_E - contrib_std_E,
        contrib_mean_E + contrib_std_E,
        step="mid",
        alpha=0.25
    )

    axr.set_xscale("log")
    axr.axhline(0.0, linewidth=1.2)
    axr.set_ylim(-0.01, 0.01)
    axr.set_xlabel("Energy (GeV)")
    axr.set_ylabel("Weig. rel. err.")
    axr.grid(True, which="both", ls="--", lw=0.5, alpha=0.7)
    axr.tick_params(which="both", direction="in", top=True, right=True)

    ax.tick_params(labelbottom=False)

    # A single legend is placed in the upper panel for the full figure.
    ax.legend(
        handles=[h_real, h_cnf, h_band],
        labels=["ARTI", "CNFs", "±1σ"],
        loc="upper right",
        frameon=True
    )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_angle_plot(city, bin_centers_A, spec_real_A, spec_mean_A, spec_std_A,
                    contrib_mean_A, contrib_std_A, save_path):
    """
    Save the angular spectrum comparison together with the weighted
    relative error contribution in a two-panel figure.
    """
    fig, (ax, axr) = plt.subplots(
        2, 1,
        figsize=(8, 8),
        dpi=130,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0}
    )

    # Step curves are used to preserve the histogram-like representation.
    h_real, = ax.step(bin_centers_A, spec_real_A, where="mid", label="ARTI")
    h_cnf, = ax.step(bin_centers_A, spec_mean_A, where="mid", linestyle="--", label="CNFs")

    h_band = ax.fill_between(
        bin_centers_A,
        np.clip(spec_mean_A - spec_std_A, 0, None),
        spec_mean_A + spec_std_A,
        step="mid",
        alpha=0.25,
        label="±1σ"
    )

    ax.set_ylabel("part/(m²·s)")
    ax.set_title(f"Angular spectrum — {city.upper()}")
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.7)

    # In the angular plot, minor ticks help readability in the linear scale.
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="minor", length=3)
    ax.tick_params(which="major", length=6)

    # Lower panel: weighted relative error contribution by angular bin.
    h_err, = axr.step(bin_centers_A, contrib_mean_A, where="mid", label="Contribution") 
    axr.fill_between(
        bin_centers_A,
        contrib_mean_A - contrib_std_A,
        contrib_mean_A + contrib_std_A,
        step="mid",
        alpha=0.25
    )

    axr.axhline(0.0, linewidth=1.2)
    axr.set_ylim(-0.01, 0.01)
    axr.set_xlabel(r"Zenith angle ($\theta^\circ$)")
    axr.set_ylabel("Weig. rel. err.")
    axr.grid(True, which="both", ls="--", lw=0.5, alpha=0.7)
    axr.tick_params(which="both", direction="in", top=True, right=True)

    ax.tick_params(labelbottom=False)

    ax.legend(
        handles=[h_real, h_cnf, h_band],
        labels=["ARTI", "CNFs", "±1σ"],
        loc="upper right",
        frameon=True
    )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)