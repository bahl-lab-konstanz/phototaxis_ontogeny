from fig3_helpers import fit_spatial_temporal_model_v2, plot_time_series, plot_params, plot_fit_errors


from utils.general_utils import load_event_df, get_n_fish, get_median_df_time, get_stats_two_groups
from utils.plot_utils import *
from settings.general_settings import rng, path_to_main_fig_folder, path_to_main_data_folder
from settings.agent_settings import *
from settings.prop_settings import *
from utils.models import *


# #############################################################################
# User settings
# #############################################################################
# Stimulus settings
from settings.stim_brightness_choice_simple import *

# Paths
path_to_fig_folder = path_to_main_fig_folder.joinpath(f'fig3_{experiment_name}', 'models')
path_to_fig_folder.mkdir(exist_ok=True)

# Agents
agents = [Larva(), Juvie()]
agents_str = '_and_'.join([agent.name for agent in agents])

# Properties
prop_classes = [
    PercentageTurns(),
    PercentageLeft(),
    TurnAngle(),
    TotalDuration(),
    Distance(),
]

# Models
models = [
    # Single pathways
    BlindModel(),
    ModelAFig3(),   # (Fig. 2) Eye-averaging pathway, trained on fig 3 data
    ModelADFig3(),  # (Fig. 2s) Eye-average derivative pathway, trained on fig 3 data
    ModelC(),       # Spatial contrast pathway
    ModelD(),       # Eye-specific derivative pathway (left has opposite sign of right)
    ModelDA(),      # Eye-specific derivative averaging pathway
    # Single constrained pathways
    ModelCSign(),   # Spatial contrast pathway: maintain direction of contrast
    ModelCAbs(),    # Spatial contrast pathway: detection of contrast presence
    # Combined pathways
    ModelA_CAbs(),
    ModelA_DA(),  # Eye-averaging pathway + Eye-specific derivative averaging pathway
    ModelDA_CAbs(),
    ModelA_D_CAbs(),
    ModelD_C(),     # (Fig. 3) Eye-specific derivative pathway + Spatial contrast pathway
    # Full model
    FullModel(),  # (Five-pathways model)
]

# #############################################################################
# Load and prepare data
# #############################################################################
# Load data
event_df = load_event_df(path_to_main_data_folder, experiment_name, agents)

# Get bootstrapped data #######################################################
if path_to_fig_folder.joinpath('median_df_bootstrapped.hdf5').exists():
    median_df = pd.read_hdf(path_to_fig_folder.joinpath('median_df_bootstrapped.hdf5'), key='median_df')
else:
    # Recompute median_df for bootstrapping
    do_bootstrap = 2/3
    bootstrapped_event_df_list = []
    for agent in agents:
        agent_df = event_df.query(agent.query)
        grouped = agent_df.groupby(['fish_age', 'fish_genotype', 'experiment_ID'])
        index = agent_df.index.names

        for exp_ID, (idx, _) in enumerate(grouped):
            # Randomly sample groups
            group_is = np.arange(grouped.ngroups)
            random_group_is = rng.choice(group_is, int(do_bootstrap * len(group_is)), replace=True)
            exp_df = agent_df[grouped.ngroup().isin(random_group_is)].reset_index()

            # Overwrite indices
            exp_df['experiment_ID'] = exp_ID
            exp_df['fish_or_agent_name'] = 0
            exp_df['experiment_repeat'] = 0
            exp_df['arena_index'] = 0
            exp_df['setup_index'] = 0
            exp_df['folder_name'] = 0
            exp_df.set_index(index, inplace=True)
            # Append to list
            bootstrapped_event_df_list.append(exp_df)
    bootstrapped_event_df = pd.concat(bootstrapped_event_df_list)
    median_df = get_median_df_time(bootstrapped_event_df, resampling_window)
    # Store bootstrapped data
    median_df.to_hdf(path_to_fig_folder.joinpath('median_df_bootstrapped.hdf5'), key='median_df')


# #####################################################################
# Loop over models
# #####################################################################
for model in models:
    # Define hdf5_file and check if it exists #################################
    # # Define hdf5 file based on key_base as model.name
    key_base = model.name
    hdf5_file = path_to_fig_folder.joinpath('fit_dfs', f'fit_df_{model.name}.hdf5')
    path_to_fig_folder.joinpath('fit_dfs').mkdir(parents=True, exist_ok=True)
    if hdf5_file.exists():
        # Fit already performed and stored
        continue

    # Fit model ###############################################################
    ind_fit_df, mean_over_ind_fit_df, mean_fit_df = fit_spatial_temporal_model_v2(
        median_df, agents,
        prop_classes, model,
        t_ns, b_left_ns, b_right_ns,
    )

    # Store model parameters ##################################################
    # Save dataframe of meta fit results for bootstrapped data
    ind_fit_df.to_hdf(hdf5_file, key=f'{key_base}_meta', mode='a')
    mean_over_ind_fit_df.to_hdf(hdf5_file, key=f'{key_base}_mean_over_ind', mode='a')
    mean_fit_df.to_hdf(hdf5_file, key=f'{key_base}_meta_mean', mode='a')

# Load model parameters ###################################################
ind_fit_df_list = []
mean_over_ind_fit_df_list = []
mean_fit_df_list = []
for model in models:
    # # Define hdf5 file based on key_base as model.name
    hdf5_file = path_to_fig_folder.joinpath('fit_dfs', f'fit_df_{model.name}.hdf5')
    path_to_fig_folder.joinpath('fit_dfs').mkdir(parents=True, exist_ok=True)
    # # Load model parameters
    ind_fit_df = pd.read_hdf(hdf5_file, key=f'{model.name}_meta')
    mean_over_ind_fit_df = pd.read_hdf(hdf5_file, key=f'{model.name}_mean_over_ind')
    mean_fit_df = pd.read_hdf(hdf5_file, key=f'{model.name}_meta_mean')

    ind_fit_df_list.append(ind_fit_df)
    mean_over_ind_fit_df_list.append(mean_over_ind_fit_df)
    mean_fit_df_list.append(mean_fit_df)
ind_fit_df = pd.concat(ind_fit_df_list)
mean_over_ind_fit_df = pd.concat(mean_over_ind_fit_df_list)
mean_fit_df = pd.concat(mean_fit_df_list)

# # Plot time-series data with fit for Proposed and Full model ################
plot_models = [FullModel()]
model_str = '_and_'.join([model.name for model in plot_models])
fig = plot_time_series(
    median_df, agents, agents,  # Plot same agents for fit
    prop_classes, plot_models,
    time_lim, time_ticks,
    t_ns, b_left_ns, b_right_ns,
    mean_fit_df,
)
savefig(fig, path_to_fig_folder.joinpath('time_series', f'{model_str}.pdf'), close_fig=True)


# Plot fit error ##############################################################
fig = plot_fit_errors(
    ind_fit_df, mean_over_ind_fit_df,
    agents, prop_classes, models,
    metric='MSE',
)
savefig(fig, path_to_fig_folder.joinpath(f'fit_errors_MSE.pdf'), close_fig=True)


for model in models:
    # # Plot time-series data with fit ########################################
    fig = plot_time_series(
        median_df, agents, agents,  # Plot same agents for fit
        prop_classes, model,
        time_lim, time_ticks,
        t_ns, b_left_ns, b_right_ns,
        mean_fit_df,
    )
    savefig(fig, path_to_fig_folder.joinpath('time_series', f'{model.name}.pdf'), close_fig=True)

    # Plot fitted parameters ##################################################
    fig = plot_params(
        ind_fit_df, mean_over_ind_fit_df,
        agents, prop_classes, model,
    )
    savefig(fig, path_to_fig_folder.joinpath('pars', f'{model.name}.pdf'), close_fig=True)


