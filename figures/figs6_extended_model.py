
# Third party library imports
import pandas as pd
from scipy.optimize import minimize

# Local library imports
from utils.general_utils import load_event_df, get_median_df, get_b_values
from settings.general_settings import path_to_main_fig_folder
from settings.agent_settings import *
from settings.prop_settings import *
from fig3_helpers import plot_time_series
from utils.models import FullModelFig2and3


# Paths
path_to_fig_folder = path_to_main_fig_folder.joinpath(f'figS6_extended_model')
path_to_fig_folder.mkdir(exist_ok=True)

# Agents
agents = [Larva(), Juvie()]
agents_str = '_and_'.join([agent.name for agent in agents])


# Models
model = FullModelFig2and3()
hdf5_file = path_to_fig_folder.joinpath(f'fit_df_{model.name}.hdf5')

# #############################################################################
# Load fig2 data
# #############################################################################
from fig2_helpers import *
stim_name = 'azimuth_left_dark_right_bright_virtual_yes'
path_to_fig2_data = path_to_main_fig_folder.joinpath(f'fig2_{stim_name}').joinpath(f'analysed_data_{stim_name}.hdf5')
fig2_df = pd.read_hdf(path_to_fig2_data, key='all_bout_data_pandas_event')

# Remove wall interactions ####################################################
fig2_df = fig2_df.loc[fig2_df['radius'] <= 5].copy()  # cm

# Compute azimuth and map to brightness
c_min, c_mid = 10, 300  # lux
fig2_df['azimuth_rad'] = np.arctan2(fig2_df['y_position'], fig2_df['x_position'])  # -pi to pi rad
fig2_df[col_name] = (c_mid - c_min) * (np.pi - np.abs(fig2_df['azimuth_rad'])) / np.pi + c_min  # c_min to c_mid lux

fig2_df[col_name] = map_azimuth_brightness(fig2_df, c_min=10, c_max=300)

# Set brightness bins, use bin centers as labels
fig2_df[bin_name] = pd.cut(fig2_df[col_name], bins=brightness_bins, labels=brightness_bin_centers, include_lowest=True)

# Get abs, to later find median orientation_change
fig2_df['estimated_orientation_change_abs'] = fig2_df['estimated_orientation_change'].abs()
# Set all orientation changes straight swims (below 10 degrees) to NaN
fig2_df['turn_angle'] = fig2_df['estimated_orientation_change_abs'].where(fig2_df['estimated_orientation_change_abs'] > 10)

median_fig2_df = get_median_df(fig2_df, bin_name)

# Prepare input data for dataset 1 (bin-based)
b_left1, b_right1 = brightness_bin_centers, brightness_bin_centers
dt1 = 1  # Placeholder, not used in this dataset

# #############################################################################
# Load fig3 data
# #############################################################################
from settings.stim_brightness_choice_simple import *
path_to_fig3_data = path_to_main_fig_folder.joinpath(f'fig3_{experiment_name}', 'models', 'median_df_bootstrapped.hdf5')
median_fig3_df = pd.read_hdf(path_to_fig3_data, key='median_df')

# Prepare input data for dataset 2 (time series)
dt2 = 1
ts_hat = np.arange(t_ns[0] + dt2 / 2, t_ns[-1] + dt2 / 2, dt2)
b_left, b_right = get_b_values(ts_hat, t_ns, b_left_ns, b_right_ns)  # shift by dt to correct for binning


# #############################################################################
# Prepare combined fitting strategy
# # Later we can move this to a helpers file
# #############################################################################
def combined_loss(
    params, model,
    # Dataset 1 (bin-based)
    x1, y1,
    # Dataset 2 (time series)
    b_left2, b_right2, dt2, y2,
    weight1=1.0, weight2=1.0
):
    # Model prediction for dataset1: bin-based (fig2)
    # This dataset contains only average brightness values, so we ignore the
    # other pathways.
    params1 = []
    for param, par_name in zip(params, model.par_names):
        if par_name == 'a' or par_name == 'bA':
            params1.append(param)
        else:
            params1.append(0)
    # Evaluate model
    dt1 = 1  # Placeholder, not used in this dataset
    y_pred1 = model.eval_cont(
        x1, x1, dt1,
        *params1
    )

    # Model prediction for dataset2: time series (fig3)
    y_pred2 = model.eval_cont(
        b_left2, b_right2, dt2,
        *params
    )

    # Compute losses
    mse1 = np.mean((y1 - y_pred1) ** 2)
    mse2 = np.mean((y2 - y_pred2) ** 2)
    # Weighted sum
    return weight1 * mse1 + weight2 * mse2


# #############################################################################
# Loop over properties
# #############################################################################
# Properties (define again to ensure correct order)
prop_classes = [
    PercentageTurns(),
    PercentageLeft(),
    TurnAngle(),
    TotalDuration(),
    Distance(),
]

fit_index = [
    'prop_name', 'dist_name', 'bin_name',
    'fish_age', 'fish_genotype',
    'experiment_ID',
    'model_name', 'par_name',
]
# After fitting, collect results in a list of dicts
mean_meta_fit_results = []

for prop_class in prop_classes:
    for agent in agents:
        # Get behavioural data for dataset 1 ##################################
        agent_df = median_fig2_df.query(agent.query)
        # Mean over individuals
        group = agent_df.groupby(bin_name, observed=True)
        y1 = group[prop_class.prop_name].mean()
        sem = group[prop_class.prop_name].sem()
        std = group[prop_class.prop_name].std()
        x1 = y1.index.to_numpy()

        # Get behavioural data for dataset 2 ##################################
        # Ensure data is same length as t_ns
        max_t = max(t_ns)
        agent_df = median_fig3_df.query(f'time < {max_t}').query(agent.query)
        # Mean over individuals
        group = agent_df.groupby('time')[prop_class.prop_name]
        y2 = group.mean()
        b_left2, b_right2 = get_b_values(ts_hat, t_ns, b_left_ns, b_right_ns)  # shift by dt to correct for binning

        # Run optimization ####################################################
        res = minimize(
            combined_loss,
            x0=model.x0,
            args=(
                model,
                x1, y1.to_numpy(),
                b_left2, b_right2, dt2, y2.to_numpy(),
                1.0, 1.0,   # weights for dataset 1 and 2
            ),
            # bounds=model.bounds_curve_fit,
        )
        best_params = res.x

        fish_age = agent_df.index.unique('fish_age')[0]
        fish_genotype = agent_df.index.unique('fish_genotype')[0]
        exp_ID = 0
        _index_dict = dict(zip(fit_index, [
            prop_class.prop_name, prop_class.dist_name, prop_class.prop_name,
            fish_age, fish_genotype, exp_ID, model.name, 'median',
        ]))
        _popt_dict = dict(zip(model.par_names, best_params))
        mean_meta_fit_results.append({**_index_dict, **_popt_dict})

# Create dataframe of fit results #############################################
mean_fit_df = pd.DataFrame(mean_meta_fit_results).set_index(fit_index).sort_index()
mean_fit_df.to_hdf(hdf5_file, key=f'{model.name}_meta_mean', mode='a')

fig = plot_median(
    median_fig2_df, mean_fit_df,
    agents, prop_classes,
    bin_name, model.name,
    label=label, ticks=brightness_bin_ticks, tick_labels=brightness_bin_tick_labels,
    ax_y_cm=3.5,
    direction='vertical',
)
savefig(fig, path_to_fig_folder.joinpath('fit_brightness.pdf'), close_fig=True)

fig = plot_time_series(
    median_fig3_df, agents, agents,  # Plot same agents for fit
    prop_classes, model,
    time_lim, time_ticks,
    t_ns, b_left_ns, b_right_ns,
    mean_fit_df,
    row_y_cm=3.5,
    ax_y_cm=1.7,
)
savefig(fig, path_to_fig_folder.joinpath('fit_timeseries.pdf'),  close_fig=True)


