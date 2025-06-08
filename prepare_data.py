"""Script to prepare event data
    - CombineData: combine data from all experiments, per age category
    - SanityCheck
    - AnalysisStart
"""


from utils.prepare_data_wrapper import prepare_data_wrapper


if __name__ == '__main__':
    # #########################################################################
    # User input
    # #########################################################################
    # Experiment settings
    from settings.stim_arena_locked import *

    # Fish/agent settings
    age_category = 'juvie'  # 'larva' or 'juvie' or 'agents' or 'simulations'
    agent_name = ''  # agent name or ''
    do_preprocess = False  # True or False

    # Your Slack name here
    slack_receiver = 'Max Capelle'

    prepare_data_wrapper(
        age_category=age_category, agent_name=agent_name,
        experiment_name=experiment_name,
        flip_dict=flip_dict, split_dict=split_dict, label_dict=label_dict,
        included_stim_names=stim_names,
        rolling_window=rolling_window, resampling_window=resampling_window,
        do_preprocess=do_preprocess,
        do_tracking=do_tracking, do_event=do_event,
        do_bootstrap=do_bootstrap, do_median_df=do_median_df, do_rolling_df=do_rolling_df,
        slack_receiver=slack_receiver
    )

