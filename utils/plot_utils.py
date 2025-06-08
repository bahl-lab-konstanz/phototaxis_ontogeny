# Standard library imports
import warnings

# Third party library imports
import imageio
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgba
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import blended_transform_factory
import seaborn as sns

# Local library imports
from settings.plot_settings import *
from utils.general_utils import *


# #############################################################################
# Figure and axes: sizes
# #############################################################################
def create_figure(fig_width=fig_width_cm, fig_height=fig_height_cm, **kwargs):
    """
    Create a matplotlib figure with specified width and height in centimeters.

    Args:
        fig_width (float): Width of the figure in cm.
        fig_height (float): Height of the figure in cm.
        **kwargs: Additional keyword arguments for `plt.figure`.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    # Convert cm to inches
    fig = plt.figure(figsize=(fig_width * cm, fig_height * cm), **kwargs)
    return fig


def add_axes(fig, l=pad_x_cm, b=pad_y_cm, w=ax_x_cm, h=ax_y_cm, **kwargs):
    """
    Add axes to a figure with specified dimensions in cm.

    Args:
        fig (matplotlib.figure.Figure): The figure to add axes to.
        l (float): Left position of the axes in cm.
        b (float): Bottom position of the axes in cm.
        w (float): Width of the axes in cm.
        h (float): Height of the axes in cm.
        **kwargs: Additional keyword arguments for `fig.add_axes`.

    Returns:
        matplotlib.axes.Axes: The created axes.
    """
    # Convert to figure coordinates
    fig_width, fig_height = fig.get_size_inches() / cm
    l, b, w, h = l / fig_width, b / fig_height, w / fig_width, h / fig_height
    return fig.add_axes((l, b, w, h), **kwargs)


def add_grid(axs, **kwargs):
    """
    Add a grid to the provided axes.

    Args:
        axs (matplotlib.axes.Axes or np.ndarray): Axes to add the grid to.
        **kwargs: Additional keyword arguments for `ax.grid`.
    """
    if not isinstance(axs, np.ndarray):
        axs = np.asarray([axs])

    for ax in axs.flatten():
        ax.grid(**kwargs)


# #############################################################################
# Saving figures
# #############################################################################
def savefig(fig, path: Path, dpi=300, transparent=None, close_fig=False, **kwargs):
    """
    Save a matplotlib figure to a file.

    Args:
        fig (matplotlib.figure.Figure): The figure to save.
        path (Path): Path to save the figure.
        dpi (int): Resolution of the saved figure in dots per inch.
        transparent (bool, optional): Whether to save with a transparent background.
        close_fig (bool): Whether to close the figure after saving.
        **kwargs: Additional keyword arguments for `fig.savefig`.
    """
    # Replace all but the last '.' with '_'
    path_str = str(path)
    last_dot_index = path_str.rfind('.')   # Find the last occurrence of '.' to preserve file extension
    if last_dot_index != -1:  # Ensure there's a dot
        # Replace all earlier dots with underscores
        path_str = path_str[:last_dot_index].replace('.', '_') + path_str[last_dot_index:]
    path = Path(path_str)

    # Check the path and its parents
    existing_part = path.parent
    while not existing_part.exists() and existing_part != existing_part.parent:
        existing_part = existing_part.parent
    existing_part_str = str(existing_part)
    non_existing_part_str = str(path.parent)[len(existing_part_str):]
    if len(non_existing_part_str):
        print(f"\tsavefig(): Creating folder {existing_part_str}\033[1m{non_existing_part_str}\033[0m")
        path.parent.mkdir(parents=True, exist_ok=True)

    # Set transparency
    if isinstance(transparent, type(None)) and path.suffix == '.png':
        transparent = False
    if isinstance(transparent, type(None)) and path.suffix == '.pdf':
        transparent = True

    fig.savefig(path, dpi=dpi, transparent=transparent, **kwargs)
    if close_fig:
        plt.close(fig)


def closefigs(figs):
    """
    Close one or more matplotlib figures.

    Args:
        figs (list or matplotlib.figure.Figure): Figures to close.
    """
    # Check if figs is iterable
    if not isinstance(figs, list):
        figs = [figs]

    for fig in figs:
        plt.close(fig)


def create_animated_gif(path_to_input_folder: Path, path_to_output_file: Path, input_glob='*.jpg', bounce=False, duration=0.1, dpi=60):
    """
    Create an animated GIF from a folder of images.

    Args:
        path_to_input_folder (Path): Path to the folder containing input images.
        path_to_output_file (Path): Path to save the output GIF.
        input_glob (str): Glob pattern to match input images.
        bounce (bool): Whether to create a bounce effect by reversing the frames.
        duration (float): Duration of each frame in seconds.
        dpi (int): Resolution of the GIF in dots per inch.
    """
    images = []
    filenames = sorted(path_to_input_folder.glob(input_glob))

    # Only plot every fifth image
    # filenames = filenames  # [::5]
    # duration = duration * 5

    for i, filename in enumerate(filenames):
        print(f"\tCreating gif: {i+1}/{len(filenames)}...", end='\r')
        # print(f"Creating gif: |{int(i+1/len(filenames) * 30) * '#':30s}|", end='\r')
        images.append(imageio.v3.imread(filename))

    if bounce:
        images += images[::-1]  # Add reversed images to create bounce effect

    imageio.mimsave(path_to_output_file, images, duration=duration, dpi=dpi, loop=True)
    print(f"\tCreating gif: {len(filenames)}/{len(filenames)}... done")


# #############################################################################
# Annotations
# #############################################################################
def add_stats(axs, x0: float, x1: float, y: float = 1, s: float | str = None):
    """
    Add statistical annotations to axes.

    Args:
        axs (matplotlib.axes.Axes or np.ndarray): Axes to annotate.
        x0 (float): Start x-coordinate in data space.
        x1 (float): End x-coordinate in data space.
        y (float): y-coordinate in axes space.
        s (str or float, optional): Annotation text.
    """
    if not isinstance(axs, np.ndarray):
        axs = np.asarray([axs])

    if isinstance(s, type(None)):
        # Do not plot anything when no text is given
        return

    for ax in axs.flatten():
        ax.text(
            # x-coordinates in dataspace, y-coordinates in axes space
            (x0 + x1) / 2, y, s,
            transform=ax.get_xaxis_transform(), ha='center', va='bottom',
            color=COLOR_ANNOT, fontsize=SMALL_SIZE,
        )
        if x0 != x1:
            ax.plot(
                # x-coordinates in dataspace, y-coordinates in axes space
                [x0, x1], [y, y],
                transform=ax.get_xaxis_transform(), clip_on=False,
                color=COLOR_ANNOT, lw=LW_ANNOT,
            )


def add_scalebar(ax, size: float, label: str, loc='lower right', color=COLOR_TEXT, pad=0, size_vertical=0):
    """
    Add a scale bar to the provided axis.

    Args:
        ax (matplotlib.axes.Axes): Axis to add the scale bar to.
        size (float): Size of the scale bar.
        label (str): Label for the scale bar.
        loc (str): Location of the scale bar.
        color (str): Color of the scale bar.
        pad (float): Padding around the scale bar.
        size_vertical (float): Vertical size of the scale bar.
    """
    scalebar = AnchoredSizeBar(ax.transData,
                               size, label, loc,
                               pad=pad, color=color,
                               frameon=False,
                               size_vertical=size_vertical)
    ax.add_artist(scalebar)


def add_scalebar_horizontal(ax, size: float, label: str, loc='lower right',
                            color=COLOR_TEXT, pad=0, thickness=0.2, outside=False):
    """Add a horizontal scale bar with thickness converted from figure to data coordinates."""
    fig = ax.get_figure()
    trans = ax.transData.inverted()

    # Convert thickness from figure coordinates to data coordinates
    thickness_data = trans.transform((0, thickness))[1] - trans.transform((0, 0))[1]

    if outside:
        # Adjust bbox_to_anchor to move the scalebar below the ax
        bbox_to_anchor = (1.02, -0.02)  # Move it slightly outside
        loc = 'upper left'  # Fix label positioning
    else:
        bbox_to_anchor = None  # Normal placement

    # Add scalebar
    scalebar = AnchoredSizeBar(ax.transData,
                               size, label, loc,
                               pad=pad, color=color,
                               frameon=False,
                               bbox_to_anchor=bbox_to_anchor, bbox_transform=ax.transAxes)

    ax.add_artist(scalebar)


def add_scalebar_vertical(ax, size: float, label: str, loc='lower right',
                          color=COLOR_TEXT, pad=0, thickness=0.2, outside=False):
    """Add a vertical scale bar. If `outside=True`, place it outside the axes."""
    trans = ax.transData.inverted()
    thickness_data = trans.transform((thickness, 0))[0] - trans.transform((0, 0))[0]

    if outside:
        # Adjust bbox_to_anchor to move the scalebar to the right of the ax
        bbox_to_anchor = (1.02, 0)  # Shift outside to the right
        loc = 'lower left'  # Align label properly
    else:
        bbox_to_anchor = None  # Normal placement

    scalebar = AnchoredSizeBar(ax.transData,
                               thickness_data, label, loc,
                               pad=pad, color=color, label_top=False,
                               frameon=False,
                               size_vertical=size,
                               bbox_to_anchor=bbox_to_anchor, bbox_transform=ax.transAxes)
    ax.add_artist(scalebar)


def set_axlines(axs, axvlines=None, axhlines=None, vlim=None, hlim=None, color=COLOR_ANNOT, lw=LW_ANNOT, zorder=-100, **kwargs):
    """
    Set vertical and horizontal lines on the provided axes.

    Args:
        axs (matplotlib.axes.Axes or np.ndarray): Axes to modify.
        axvlines (float or list, optional): x-coordinates for vertical lines.
        axhlines (float or list, optional): y-coordinates for horizontal lines.
        vlim (list, optional): y-limits for vertical lines.
        hlim (list, optional): x-limits for horizontal lines.
        color (str): Color of the lines.
        lw (float): Line width.
        zorder (int): Z-order for the lines.
        **kwargs: Additional keyword arguments for `ax.axvline` and `ax.axhline`.
    """
    # Make axs iterable
    if not isinstance(axs, np.ndarray):
        axs = np.asarray([axs])

    for ax in axs.flatten():
        if isinstance(axvlines, type(None)):
            pass
        elif isinstance(axvlines, float) or isinstance(axvlines, int):
            _add_axvline(ax, axvlines, lim=vlim, color=color, lw=lw, zorder=zorder, **kwargs)
        else:
            for x in axvlines:
                _add_axvline(ax, x, lim=vlim, color=color, lw=lw, zorder=zorder, **kwargs)

        if isinstance(axhlines, type(None)):
            pass
        elif isinstance(axhlines, float) or isinstance(axhlines, int):
            _add_axhline(ax, axhlines, lim=hlim, color=color, lw=lw, zorder=zorder, **kwargs)
        else:
            for y in axhlines:
                _add_axhline(ax, y, lim=hlim, color=color, lw=lw, zorder=zorder, **kwargs)


def _add_axvline(ax, x, lim, color, lw, zorder, **kwargs):
    if lim is None:
        ax.axvline(x, color=color, lw=lw, zorder=zorder, **kwargs)
    else:
        ax.plot([x, x], [lim[0], lim[1]], color=color, lw=lw, zorder=zorder, **kwargs)


def _add_axhline(ax, y, lim, color, lw, zorder, **kwargs):
    if lim is None:
        ax.axhline(y, color=color, lw=lw, zorder=zorder, **kwargs)
    else:
        ax.plot([lim[0], lim[1]], [y, y], color=color, lw=lw, zorder=zorder, **kwargs)


def set_legend(axs, loc='upper left', bbox_to_anchor=(1.1, 1.1), **kwargs):
    """
    Set the legend for the provided axes.

    Args:
        axs (matplotlib.axes.Axes or np.ndarray): Axes to add the legend to.
        loc (str): Location of the legend.
        bbox_to_anchor (tuple): Bounding box anchor for the legend.
        **kwargs: Additional keyword arguments for `ax.legend`.
    """
    # Make axs iterable
    if not isinstance(axs, np.ndarray):
        axs = np.asarray([axs])

    for ax in axs.flatten():
        ax.set_visible(True)  # make sure ax is visible
        ax.legend(loc=loc, bbox_to_anchor=bbox_to_anchor, **kwargs)


def get_legend_fig(agents):
    """
    Create a figure with a legend for the provided agents.

    Args:
        agents (list): List of agent objects with `label` and `color` attributes.

    Returns:
        matplotlib.figure.Figure: The created figure with the legend.
    """
    # Create palette_dit and palette_df for seaborn plots
    palette_dict = {}
    for agent in agents:
        palette_dict[agent.label] = agent.color
    palette_dict['Control'] = COLOR_ANNOT
    palette_df = pd.DataFrame({'group': palette_dict.keys(), 'x': np.nan, 'y': np.nan})

    fig = create_figure(ax_x_cm + pad_x_cm, 2*ax_y_cm + pad_y_cm)
    ax = add_axes(fig, 0, 0, 1, 1)
    hide_all_spines_and_ticks(ax)
    # Fit
    ax.plot([], [], color=COLOR_MODEL, label='Fit', linestyle='dashed')
    ax.plot([], [], color=COLOR_MODEL, label=r'$a + b\ \ln\left(x\right)$', linestyle='dashed')
    ax.plot([], [], marker='x', markersize=MARKER_SIZE_LARGE, linestyle='none', color=COLOR_MODEL, label='Fit')
    # Line
    for agent in agents:
        ax.plot([], [], color=agent.color, label=agent.label)
    # Line with alpha
    for agent in agents:
        ax.plot([], [], color=agent.color, label=agent.label_single, alpha=ALPHA)
    # Dot
    # for agent in agents:
    #     ax.scatter([], [], color=agent.color, label=agent.label_single, alpha=ALPHA)
    # ax.scatter([], [], color=COLOR_ANNOT, label='Control', alpha=ALPHA)
    # Stripplot: hollow markers
    strip = sns.stripplot(
        data=palette_df,
        x='x', y='y',
        hue='group', palette=palette_dict, alpha=1, size=MARKER_SIZE,
        marker=MARKER_HOLLOW,
        legend=True,
    )
    # Errorbar
    for agent in agents:
        ax.errorbar([], [], yerr=[], fmt='o', color=agent.color, label=agent.label)
    # Error shading
    for agent in agents:
        ax.fill_between([], [], color=agent.color, alpha=ALPHA, label='SEM')
    # Histogram
    for agent in agents:
        ax.hist([], color=agent.color, label=agent.label, alpha=2*ALPHA)
    # Set legend!
    set_legend(ax, loc='lower left', bbox_to_anchor=(0, 0), )
    return fig


# #############################################################################
# Axes
# #############################################################################
def hide_axes(axs):
    """
    Hide the provided axes.

    Args:
        axs (matplotlib.axes.Axes or np.ndarray): Axes to hide.
    """
    if not isinstance(axs, np.ndarray):
        axs = np.asarray([axs])

    for ax in axs.flatten():
        ax.set_visible(False)


def set_aspect(axs, aspect='equal'):
    """
    Set the aspect ratio for the provided axes.

    Args:
        axs (matplotlib.axes.Axes or np.ndarray): Axes to modify.
        aspect (str): Aspect ratio to set (e.g., 'equal').
    """
    # Make axs iterable
    if not isinstance(axs, np.ndarray):
        axs = np.asarray([axs])

    for ax in axs.flatten():
        ax.set_aspect(aspect)


# Limits and bounds ###########################################################
def set_lims_and_bounds(axs, x=None, y=None):
    """
    Set both limits and bounds for the provided axes.

    Args:
        axs (matplotlib.axes.Axes or np.ndarray): Axes to modify.
        x (list, optional): x-axis limits and bounds.
        y (list, optional): y-axis limits and bounds.
    """
    set_bounds(axs, x, y)
    set_lims(axs, x, y)


def set_bounds(axs, x=None, y=None):
    """
    Set bounds for the spines of the provided axes.

    Args:
        axs (matplotlib.axes.Axes or np.ndarray): Axes to modify.
        x (list, optional): Bounds for the x-axis spines.
        y (list, optional): Bounds for the y-axis spines.
    """
    # Make ax iterable
    if not isinstance(axs, np.ndarray):
        axs = np.asarray([axs])

    for ax in axs.flatten():
        if not isinstance(x, type(None)):
            if len(x) > 2:
                x = [np.min(x), np.max(x)]
            ax.spines[['top', 'bottom']].set_bounds(x)

        if not isinstance(y, type(None)):
            if len(y) > 2:
                y = [np.min(y), np.max(y)]
            ax.spines[['left', 'right']].set_bounds(y)


def set_lims(axs, x=None, y=None, adjust_lim=False):
    """
    Set axis limits for the provided axes.

    Args:
        axs (matplotlib.axes.Axes or np.ndarray): Axes to modify.
        x (list, optional): x-axis limits.
        y (list, optional): y-axis limits.
        adjust_lim (bool): Whether to adjust limits slightly for better visibility.
    """
    if not isinstance(axs, np.ndarray):
        axs = np.asarray([axs])

    for ax in axs.flatten():
        if not isinstance(x, type(None)):
            if adjust_lim:
                x = _adjust_lim(x)
            ax.set_xlim(x)
        if not isinstance(y, type(None)):
            if adjust_lim:
                y = _adjust_lim(y)
            ax.set_ylim(y)


def _adjust_lim(lim: list, factor=0.05):
    """
    Adjust axis limits to ensure markers are visible.

    Args:
        lim (list): Original limits.
        factor (float): Adjustment factor.

    Returns:
        list: Adjusted limits.
    """
    # Adjust axis limits to ensure markers are still visible
    new_lim = lim.copy()
    dlim = lim[1] - lim[0]
    new_lim[0] = lim[0] - dlim * factor
    new_lim[1] = lim[1] + dlim * factor
    return new_lim


def set_xlims_symmetric(axs,):
    """
    Set symmetric x-axis limits for the provided axes.

    Args:
        axs (matplotlib.axes.Axes or np.ndarray): Axes to modify.
    """
    if not isinstance(axs, np.ndarray):
        axs = np.asarray([axs])

    for ax in axs.flatten():
        x_lim = ax.get_xlim()
        x_lim = np.abs(x_lim)
        x_lim = [-np.max(x_lim), np.max(x_lim)]
        ax.set_xlim(x_lim)


# Spines, ticks and labels ####################################################
def hide_all_spines_and_ticks(ax):
    """
    Hide all spines and ticks for the provided axis.

    Args:
        ax (matplotlib.axes.Axes): Axis to modify.
    """
    hide_spines(ax, spines=['top', 'right', 'bottom', 'left'])
    set_ticks(ax, x_ticks='none', y_ticks='none')


def hide_spines(axs, spines=None, hide_ticks=False):
    """
    Hide specific spines for the provided axes.

    Args:
        axs (matplotlib.axes.Axes or np.ndarray): Axes to modify.
        spines (list, optional): List of spines to hide (e.g., ['top', 'right']).
        hide_ticks (bool): Whether to hide ticks for the hidden spines.
    """
    # Make ax iterable
    if not isinstance(axs, np.ndarray):
        axs = np.asarray([axs])
    if isinstance(spines, type(None)):
        spines = ['top', 'right']
    for ax in axs.flatten():
        for spine in spines:
            ax.spines[spine].set_visible(False)
            if hide_ticks and spine == 'bottom':
                ax.set_xticks([])
                ax.set_xlabel('')
            if hide_ticks and spine == 'left':
                ax.set_yticks([])
                ax.set_ylabel('')


def show_spines(axs, spines=None):
    """
    Show specific spines for the provided axes.

    Args:
        axs (matplotlib.axes.Axes or np.ndarray): Axes to modify.
        spines (list, optional): List of spines to show.
    """
    # Make ax iterable
    if not isinstance(axs, np.ndarray):
        axs = np.asarray([axs])

    if isinstance(spines, type(None)):
        spines = ['top', 'right', 'bottom', 'left']
    elif isinstance(spines, str):
        spines = [spines]

    for ax in axs.flatten():
        for spine in spines:
            ax.spines[spine].set_visible(True)


def set_spine_position(axs, spines=None, distance=2):
    """
    Set the position of specific spines for the provided axes.

    Args:
        axs (matplotlib.axes.Axes or np.ndarray): Axes to modify.
        spines (list, optional): List of spines to reposition.
        distance (float): Distance to move the spines outward.
    """
    # Make ax iterable
    if not isinstance(axs, np.ndarray):
        axs = np.asarray([axs])

    if spines is None:
        spines = ['left', 'bottom']

    if isinstance(spines, str):
        spines = [spines]
    for ax in axs.flatten():
        for spine in spines:
            ax.spines[spine].set_position(('outward', distance))


def align_ylabels(fig, axs):
    """
    Align y-axis labels for the provided figure and axes.

    Args:
        fig (matplotlib.figure.Figure): Figure containing the axes.
        axs (matplotlib.axes.Axes or np.ndarray): Axes to align.
    """
    fig.align_ylabels(axs)


def prepare_label(string: str, n_chars=10):
    """
    Prepare a label string by trimming or padding it to a fixed length.

    Args:
        string (str): Input string.
        n_chars (int): Desired length of the string.

    Returns:
        str: Prepared label string.
    """
    # Trim the string to a maximum of n_chars characters
    trimmed_string = string[:n_chars]
    # Fill with trailing spaces if it is shorter than n_chars characters
    filled_string = trimmed_string.ljust(n_chars)
    return filled_string


def set_labels(
        axs,  # Axes object or np.ndarray of Axes objects
        x=None, y=None,
):
    """
    Set axis labels for the provided axes.

    Args:
        axs (matplotlib.axes.Axes or np.ndarray): Axes to modify.
        x (str, optional): Label for the x-axis.
        y (str, optional): Label for the y-axis.
    """
    # Make ax iterable
    if not isinstance(axs, np.ndarray):
        axs = np.asarray([axs])

    for ax in axs.flatten():
        if not isinstance(x, type(None)):
            ax.set_xlabel(x)
        if not isinstance(y, type(None)):
            ax.set_ylabel(y)


def set_ticks(
        axs,  # Axes object or np.ndarray of Axes objects
        x_ticks=None, x_ticklabels=None, x_tickrotation=None, x_ticksize=None, x_step=None,
        y_ticks=None, y_ticklabels=None, y_tickrotation=None, y_ticksize=None, y_step=None,
):
    """
    Set ticks and tick labels for the provided axes.

    Args:
        axs (matplotlib.axes.Axes or np.ndarray): Axes to modify.
        x_ticks (list, optional): x-axis tick positions.
        x_ticklabels (list, optional): x-axis tick labels.
        x_tickrotation (float, optional): Rotation angle for x-axis tick labels.
        x_ticksize (float, optional): Size of x-axis ticks.
        x_step (float, optional): Step size for x-axis ticks.
        y_ticks (list, optional): y-axis tick positions.
        y_ticklabels (list, optional): y-axis tick labels.
        y_tickrotation (float, optional): Rotation angle for y-axis tick labels.
        y_ticksize (float, optional): Size of y-axis ticks.
        y_step (float, optional): Step size for y-axis ticks.
    """
    # Make ax iterable
    if not isinstance(axs, np.ndarray):
        axs = np.asarray([axs])

    for ax in axs.flatten():
        # x ticks
        if isinstance(x_ticks, type(None)):
            pass  # do nothing
        elif isinstance(x_ticks, str) and x_ticks == 'none':
            ax.set_xticks([])
        else:
            ax.set_xticks(x_ticks)
        if not isinstance(x_ticklabels, type(None)):
            ax.set_xticklabels(x_ticklabels, rotation=x_tickrotation, )
        if not isinstance(x_ticksize, type(None)):
            ax.tick_params(axis='x', size=x_ticksize)

        # y ticks
        if isinstance(y_ticks, type(None)):
            pass  # do nothing
        elif isinstance(y_ticks, str) and y_ticks == 'none':
            ax.set_yticks([])
        else:
            ax.set_yticks(y_ticks)
        if not isinstance(y_ticklabels, type(None)):
            ax.set_yticklabels(y_ticklabels, rotation=y_tickrotation, )
        if not isinstance(y_ticksize, type(None)):
            ax.tick_params(axis='y', size=y_ticksize)

        # Set x-tick steps
        if isinstance(x_step, type(None)):
            pass
        elif isinstance(x_step, float) or isinstance(x_step, int):
            # Floor the min and ceil the max to the nearest step
            x_min, x_max = ax.get_xlim()
            x_min = np.floor(x_min / x_step) * x_step
            x_max = np.ceil(x_max / x_step) * x_step
            # Update limits
            ax.set_xlim(x_min, x_max)
            # Set ticks
            ax.xaxis.set_major_locator(MultipleLocator(x_step))

        # Set y-tick steps
        if isinstance(y_step, type(None)):
            pass
        elif isinstance(y_step, float) or isinstance(y_step, int):
            # Floor the min and ceil the max to the nearest step
            y_min, y_max = ax.get_ylim()
            y_min = np.floor(y_min / y_step) * y_step
            y_max = np.ceil(y_max / y_step) * y_step
            # Update limits
            ax.set_ylim(y_min, y_max)
            # Set ticks
            ax.yaxis.set_major_locator(MultipleLocator(y_step))


# #############################################################################
# Colors and colormaps
# #############################################################################
def get_transparent_cmap(target_color: list or str, ncolors: int=256, alpha_range=(0, 1)):
    """
    Create a colormap that transitions to transparent.

    Args:
        target_color (list or str): Target color for the colormap.
        ncolors (int): Number of colors in the colormap.
        alpha_range (tuple): Range of alpha values.

    Returns:
        LinearSegmentedColormap: The created colormap.
    """
    if isinstance(target_color, str):
        target_color = to_rgba(target_color)

    cmap_name = str(target_color) + '_alpha'

    # Create color_array based on target_color
    color_array = np.tile(target_color, (ncolors, 1))
    color_array[:, -1] = np.linspace(alpha_range[0], alpha_range[1], ncolors)  # change alpha values
    map_object = LinearSegmentedColormap.from_list(name=cmap_name, colors=color_array)  # create a colormap object
    # plt.colormaps.unregister(cmap_name)  # unregister the original cmap
    # plt.colormaps.register(cmap=map_object)  # register this new colormap with matplotlib
    return map_object


def cmaps_to_transparent(cmap: object, ncolors: int=256, alpha_range=(0, 1)):
    color_array = cmap(np.linspace(0, 1, ncolors))
    # color_array[:, -1] = np.linspace(alpha_range[0], alpha_range[1], ncolors)  # change alpha values
    # Set alpha for lower 10% of values
    color_array[:int(ncolors * 0.1), -1] = np.linspace(alpha_range[0], alpha_range[1], int(ncolors * 0.1))  # change alpha values
    map_object = LinearSegmentedColormap.from_list(name=cmap.name + '_alpha', colors=color_array)
    return map_object


def add_colorbar(im, ax, label=None, tick_length=2, orientation='vertical', **kwargs):
    cbar = plt.colorbar(im, ax=ax, orientation=orientation, **kwargs)  # ticks=ticks, ticklabels=ticklabels,
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=tick_length)
    if isinstance(label, str):
        cbar.set_label(label)
    # if isinstance(ticks, type(None)):
    #     cbar.ax.set_xticklabels([])
    #     cbar.ax.tick_params(length=0)
    # if ticklabels:
    #     cbar.set_ticklabels(ticklabels)
    return cbar


def get_colorbar(cmap, ticks=None, ticklabels=None, orientation='horizontal', figsize=(3*cm, 3*cm), **kwargs):
    """Create a colorbar figure with specified colormap, ticks, and labels."""
    if isinstance(ticks, type(None)):
        vmin, vmax = 0, 1
    else:
        vmin, vmax = np.min(ticks), np.max(ticks)

    a = np.array([[vmin, vmax]])
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    ax.set_visible(False)
    img = ax.imshow(a, cmap=cmap)

    # Add colorbar with ticks
    cbar = add_colorbar(img, ax, orientation=orientation, **kwargs)
    if ticklabels:
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)

    # Add colorbar without ticks
    cbar = add_colorbar(img, ax, orientation=orientation, **kwargs)
    cbar.set_ticks([])

    return fig


# #############################################################################
# Standard figures
# #############################################################################
def plot_midline_length(df, n=11):
    """
    Plot the midline length of fish across experiments.

    Args:
        df (pd.DataFrame): DataFrame containing fish data with columns 'experiment_ID', 'fish_age', 'folder_name', and 'midline_length'.
        n (int, optional): Number of bins for plotting. Defaults to 11.

    Raises:
        Warning: If the maximum midline length exceeds 25 mm.

    Returns:
        None
    """
    ax_x_cm, ax_y_cm = 4, 4  # cm

    # Rename fish age of 26 to 27
    df.rename(index={26: 27}, level='fish_age', inplace=True)

    # Compute mean within experiments
    df_mean = df.reset_index().groupby(['experiment_ID', 'fish_age', 'folder_name'])['midline_length'].mean()

    if df_mean.max() > 2.5:
        warnings.warn(f"plot_midline_length(): Midline length is larger than 25 mm: {df_mean.max() * 10}")

    fig = create_figure(pad_x_cm + ax_x_cm, pad_y_cm + ax_y_cm)
    i, j = 0, 0
    l, b, w, h = (
        pad_x_cm + i * ax_x_cm,
        pad_y_cm + j * ax_y_cm,
        ax_x_cm - pad_x_cm,
        ax_y_cm - pad_y_cm
    )
    ax = add_axes(fig, l, b, w, h)
    sns.histplot(
        data=df_mean.reset_index(), x='midline_length', bins=np.linspace(0, 2.5, n),
        hue='fish_age', palette=AGE_PALETTE_DICT, edgecolor=None, alpha=0.7,
        ax=ax, legend=False,
    )

    hide_spines(ax)
    set_ticks(ax, x_ticks=np.linspace(0, 2.5, 6), x_ticklabels=[0, 5, 10, 15, 20, 25])
    set_labels(ax, x='Midline length (mm)', y='Nr. fish')
    set_bounds(ax, x=[0, 2.5])

    return fig, ax


def plot_orientation_change_dist(event_df, agents, turn_threshold):
    """
    Plot the distribution of orientation changes for agents.

    Args:
        event_df (pd.DataFrame): DataFrame containing event data with orientation changes.
        agents (list): List of agent objects to include in the plot.
        turn_threshold (float): Threshold for defining straight swims.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    from scipy.optimize import curve_fit
    from settings.prop_settings import OrientationChange, DoubleMaxwellCenterNormal
    _ax_x_cm, _ax_y_cm = 3.5, 4  # cm

    prop_class = OrientationChange()
    bins = np.arange(-100, 101, 5)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bins_straight = np.arange(-turn_threshold, turn_threshold + 0.01, 5)
    plot_bins = np.arange(-100, 101, 2)  # Plot fit at finer resolution
    fig = create_figure()

    for i, do_fit in enumerate([False, True]):
        j = 0  # Plot agents on same row instead
        l, b, w, h = (
            pad_x_cm + i * _ax_x_cm,
            pad_y_cm + j * _ax_y_cm,
            _ax_x_cm - pad_x_cm,
            _ax_y_cm - pad_y_cm
        )
        ax = add_axes(fig, l, b, w, h)
        for j, agent in enumerate(agents[::-1]):  # Plot agents from top to bottom
            # l, b, w, h = (
            #     pad_x_cm + i * _ax_x_cm,
            #     pad_y_cm + j * _ax_y_cm,
            #     _ax_x_cm - pad_x_cm,
            #     _ax_y_cm - pad_y_cm
            # )
            # ax = add_axes(fig, l, b, w, h)

            agent_df = event_df.query(agent.query)

            if do_fit:
                # Fit DoubleMaxwell-Normal distribution
                dist_class = DoubleMaxwellCenterNormal()
                y = agent_df[prop_class.prop_name].values
                hist, _ = np.histogram(y, bins=bins, density=True)
                std = np.std(y)
                fish_age = agent_df.index.unique('fish_age')[0]
                p0 = dist_class.guess_pdf(bin_centers, hist, std, fish_age)
                res = curve_fit(
                    f=prop_class.pdf,
                    xdata=bin_centers, ydata=hist,
                    p0=p0,
                )
                popt = res[0]
                popt_print = np.round(popt, 2)
                print(f"\tplot_orientation_change_dist(): {agent.name}: popt={repr(popt_print)} ({dist_class.par_names_pdf})")

                # PDF: all swims
                ax.hist(
                    agent_df[prop_class.prop_name].values,
                    bins=bins, density=do_fit,
                    color=agent.color, alpha=0.7,
                    label='Turns'
                )

                # Plot fit
                y_hat = prop_class.pdf(plot_bins, *popt)
                ax.plot(plot_bins, y_hat, color=COLOR_MODEL, linestyle='--', label='Fit')

                # Median turn angle
                if j == 0:
                    median_turn_angle = agent_df['turn_angle'].median()
                    y = prop_class.pdf(median_turn_angle, *popt)
                    ax.scatter(
                        median_turn_angle, y + 0.005, label='Median',
                        marker='v', color=COLOR_ANNOT, s=MARKER_SIZE*10, zorder=100,
                    )
                    # if j == 1:
                    #     # Add text for top row
                    #     ax.text(median_turn_angle, y + 0.01, 'Abs. turn\nangle', ha='left', va='bottom',)

                # Format
                set_labels(ax, x=f"{prop_class.label} ({prop_class.unit})", y='Prob. density')
                hide_spines(ax, ['left'])
                set_ticks(ax, y_ticks=[])
                set_lims(ax, y=[0, 0.05])  # Set y-limits to match the fit
                set_legend(ax)
            else:
                # Histogram: all swims
                sns.histplot(
                    data=agent_df.reset_index(), x=prop_class.prop_name, bins=bins,
                    color=agent.color, alpha=0.7, linestyle='none',
                    ax=ax,
                )

                # Median turn angle
                if j == 0:
                    median_turn_angle = agent_df['turn_angle'].median()
                    y = ax.get_ylim()[1] * 0.9
                    ax.scatter(
                        median_turn_angle, y + 0.005, label='Median',
                        marker='v', color=COLOR_ANNOT, s=MARKER_SIZE*10, zorder=100,
                    )

                # Format
                set_labels(ax, y='Nr. swims')

            # Make straight swims lighter
            ax.axvspan(-turn_threshold, turn_threshold, color=COLOR_AGENT_MARKER, alpha=0.3, label='Straight swims')

            # Format
            hide_spines(ax)
            set_spine_position(ax, spines='left')
            set_labels(ax, x=f'{prop_class.label} ({prop_class.unit})')
            set_ticks(ax, x_ticks=np.arange(-100, 101, 50))
            set_lims_and_bounds(ax, x=[-100, 100])
    return fig


# #############################################################################
# Illustrate stimuli
# #############################################################################
def add_stimulus_bar(axs, t_ns, b_left_ns, b_right_ns, max_value=300, axvline=True, zorder=-100, y_pos=0.95, height=0.05):
    """
    Add a stimulus bar to the provided axes, representing left and right stimulus brightness.

    Args:
        axs (matplotlib.axes.Axes or np.ndarray): Axes to add the stimulus bar to.
        t_ns (array-like): Time points in data coordinates.
        b_left_ns (array-like): Normalized brightness values for the left stimulus.
        b_right_ns (array-like): Normalized brightness values for the right stimulus.
        max_value (float): Maximum value for brightness normalization.
        axvline (bool): Whether to add vertical lines at the end of each segment.
        zorder (int): Z-order for the stimulus bar elements.
        y_pos (float): Center position of the stimulus bar in axes coordinates (0 = bottom, 1 = top).
        height (float): Half-height of the stimulus bar.
    """
    # Make axs iterable
    if not isinstance(axs, np.ndarray):
        axs = np.asarray([axs])

    # Define colormap from black to white
    cmap_grey = plt.get_cmap('Greys_r')

    # Normalise brightness values
    # max_value = np.max([np.max(b_left_ns), np.max(b_right_ns)])
    norm_b_left_ns = np.asarray(b_left_ns) / max_value
    norm_b_right_ns = np.asarray(b_right_ns) / max_value

    # Add last values to make the last rectangle visible
    t_ns = np.append(t_ns, t_ns[-1])
    norm_b_left_ns = np.append(norm_b_left_ns, norm_b_left_ns[-1])
    norm_b_right_ns = np.append(norm_b_right_ns, norm_b_right_ns[-1])

    for ax in axs.flatten():
        # Define a transform to have x in data-coordinates and y in axes-coordinates
        transform = blended_transform_factory(ax.transData, ax.transAxes)
        for t, t_start, t_end, left_start, right_start in zip(
                t_ns, t_ns, t_ns[1:], norm_b_left_ns, norm_b_right_ns,
        ):
            ax.add_patch(patches.Rectangle(
                (t_start, y_pos), (t_end - t_start), height,
                facecolor=cmap_grey(left_start), transform=transform, zorder=zorder)
            )
            # ax.text(t_start, 0.6, f'{left_start:.1f}', color='tab:blue')
            # ax.text(t_start, 0.4, f'{right_start:.1f}', color='tab:blue')
            ax.add_patch(patches.Rectangle(
                (t_start, y_pos), (t_end - t_start), -height,
                facecolor=cmap_grey(right_start), transform=transform, zorder=zorder)
            )
            if axvline:
                ax.axvline(t, color=COLOR_ANNOT, linestyle='--', linewidth=0.5, zorder=zorder)

        # Add line around bar
        ax.add_patch(patches.Rectangle(
            (0, y_pos - height), t_end, 2 * height,
            fill=False, edgecolor=COLOR_TEXT, lw=1, transform=transform, zorder=zorder)
        )
