# =============================================================================
# Imports
# =============================================================================

# Core
import itertools
import numpy as np
import pandas as pd

# Stats
from scipy.stats import ks_2samp, wasserstein_distance
from statsmodels.stats.multitest import multipletests

# Plotting
import seaborn as sns

# Local libraries
from fig1_helpers import *
from settings.agent_settings import *
from settings.general_settings import *
from utils.general_utils import load_tracking_df

# =============================================================================
# User settings
# =============================================================================

# Paths
experiment_name = 'arena_locked'
path_name = 'figS1C_wall_interactions'
path_to_fig_folder = path_to_main_fig_folder.joinpath(path_name)
path_to_stats_txt = path_to_fig_folder.joinpath('stats.txt')

# Agents
agents = [Larva(), Juvie()]
agents_str = '_'.join(agent.name for agent in agents)

# Radius binning
start_radius = 0  # cm
dr = 1
r_bins = np.arange(start_radius, 6, dr)

# Orientation change settings
prop_class = OrientationChange()
bins = np.arange(-100, 101, 5)
bins_straight = np.arange(-turn_threshold, turn_threshold + 0.01, 5)

# =============================================================================
# Data loading & preparation
# =============================================================================

event_df = load_event_df(
    path_to_main_data_folder,
    'fig1_and_4',
    agents,
)

# Keep only control stimulus
stim_name = 'control'
event_df = event_df.xs(
    stim_name,
    level='stimulus_name',
    drop_level=False,
)

# Sanity check: arena coverage
if event_df['radius'].max() < 6:
    raise ValueError(
        "Data does not extend to arena border (6 cm). "
        f"Max radius: {event_df['radius'].max():.2f} cm"
    )


# =============================================================================
# Analysis helpers
# =============================================================================
def collect_radius_bins(agent_df, r_bins, dr, value_col):
    """Return dict: radius -> 1D numpy array"""
    out = {}
    for r in r_bins:
        r0, r1 = r, r + dr
        out[r] = (
            agent_df
            .query(f'{r0} <= radius < {r1}')[value_col]
            .dropna()
            .to_numpy()
        )
    return out


def pairwise_ks_within_agent(agent_name, agent_df, r_bins, dr, value_col):
    bin_data = collect_radius_bins(agent_df, r_bins, dr, value_col)

    rows = []
    for r1, r2 in itertools.combinations(r_bins, 2):
        x, y = bin_data[r1], bin_data[r2]
        if len(x) == 0 or len(y) == 0:
            continue

        stat, p = ks_2samp(x, y)
        rows.append({
            "agent": agent_name,
            "r1": r1,
            "r2": r2,
            "ks_stat": stat,
            "p_value": p,
        })

    df = pd.DataFrame(rows)

    if not df.empty:
        _, p_corr, _, _ = multipletests(df["p_value"], method="fdr_bh")
        df["p_value_corr"] = p_corr

    return df


def wasserstein_matrix(agent_df, r_bins, dr, value_col):
    bin_data = collect_radius_bins(agent_df, r_bins, dr, value_col)

    mat = np.full((len(r_bins), len(r_bins)), np.nan)

    for i, r1 in enumerate(r_bins):
        for j, r2 in enumerate(r_bins):
            x, y = bin_data[r1], bin_data[r2]
            if len(x) > 0 and len(y) > 0:
                mat[i, j] = wasserstein_distance(x, y)

    return pd.DataFrame(mat, index=r_bins, columns=r_bins)


# =============================================================================
# Figure creation
# =============================================================================
fig = create_figure(
    fig_width=fig_width_cm,
    fig_height=4 * small_grid_y,
)

grid_x = 8 / 6 * small_grid_x
pad = 0.4

ks_results = []
wd_matrices = {}

# Plot histograms + effect-size matrices
for j, agent in enumerate(agents[::-1]):
    agent_df = event_df.query(agent.query)

    # ---------------------------------------------------------
    # Orientation change histograms
    # ---------------------------------------------------------
    for k, r in enumerate(r_bins):
        r_start, r_end = r, r + dr

        l, b, w, h = (
            grid_x * (k + 1),
            small_grid_y * (j + 1),
            grid_x - pad,
            small_grid_y - pad,
        )
        ax = add_axes(fig, l, b, w, h)

        sns.histplot(
            data=agent_df.query(f'{r_start} <= radius < {r_end}').reset_index(),
            x=prop_class.prop_name,
            bins=bins,
            color=agent.color,
            alpha=0.7,
            linestyle='none',
            stat='density',
            ax=ax,
        )

        set_lims(ax, y=[0, 0.05])

        ax.axvspan(
            -turn_threshold,
            turn_threshold,
            color=COLOR_AGENT_MARKER,
            alpha=0.3,
        )

        set_spine_position(ax, spines='left')
        set_lims_and_bounds(ax, x=[-100, 100])
        hide_spines(ax, spines=['left', 'top', 'right'])
        set_ticks(ax, x_ticks=np.arange(-100, 101, 100), x_ticklabels=[], y_ticks=[])
        set_labels(ax, x='', y='')

        if j == 1:
            ax.set_title(rf'${r_start} \leq r < {r_end}$ cm')
        if j == 0:
            set_ticks(ax, x_ticks=np.arange(-100, 101, 100), x_ticklabels=[-100, 0, 100])
            show_spines(ax, spines='bottom')
        if k == 0:
            set_ticks(ax, y_ticks=[0, 0.025, 0.05], y_ticklabels=[0, 25, 50])
            show_spines(ax, spines='left')
        if k == 0 and j == 0:
            set_labels(ax, y='Prob. density (%)')
        if k == 2 and j == 0:
            set_labels(ax, x='Orientation change (deg)')

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------
    ks_results.append(
        pairwise_ks_within_agent(
            agent.name,
            agent_df,
            r_bins,
            dr,
            prop_class.prop_name,
        )
    )

    wd_matrices[agent.name] = wasserstein_matrix(
        agent_df,
        r_bins,
        dr,
        prop_class.prop_name,
    )

    # ---------------------------------------------------------
    # Effect-size matrix plot
    # ---------------------------------------------------------
    l, b, w, h = (
        grid_x * (len(r_bins) + 1) + pad,
        small_grid_y * (j + 1),
        2 * small_grid_x - pad/2,
        small_grid_y - pad/2,
    )
    ax = add_axes(fig, l, b, w, h)

    data = wd_matrices[agent.name].values
    mask = np.tril(np.ones_like(data, dtype=bool))

    hm = sns.heatmap(
        data, mask=mask,
        ax=ax,
        cmap=agent.cmap,
        square=True, cbar=True,
        vmin=0, vmax=6,
    )

    ax.set_aspect('equal')
    hide_all_spines_and_ticks(ax)
    centers = np.arange(len(r_bins)) + 0.5
    set_ticks(
        ax,
        y_ticks=centers, y_ticklabels=wd_matrices[agent.name].index,
    )
    set_labels(ax, y='Radius (cm)')
    show_spines(ax, spines='left')
    if j == 0:
        set_ticks(
            ax, x_ticks=centers, x_ticklabels=wd_matrices[agent.name].columns,
        )
        show_spines(ax, spines='bottom')
        set_labels(ax, x='Radius (cm)')

    # colorbar ticks
    cbar = hm.collections[0].colorbar
    cbar.set_ticks([0, 3, 6])
    cbar.set_ticklabels([0, 3, 6])

# =============================================================================
# Save outputs
# =============================================================================

ks_results = pd.concat(ks_results, ignore_index=True)

savefig(
    fig,
    path_to_fig_folder.joinpath('orientation_dist.pdf'),
)
