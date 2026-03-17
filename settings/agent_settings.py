"""Defines agent classes and model functions"""

# Imports
from .plot_settings import *
from utils.plot_utils import ListedColormap


# #############################################################################
# Super class
# #############################################################################
class Agent:
    def __init__(
            self,
            name='agent', label='All agents', label_single='Agent',
            folder_name=None,
            fish_age=None, fish_genotype=None,
            query="index==index",  # query all
            ref_agent=None, ref_agent_str='agent',
            color=COLOR_MODEL, markerfacecolor=COLOR_MODEL, color_accent=None,
            marker='o',
            cmap=CMAP_GREY, palette=ListedColormap([COLOR_MODEL]),
            vmin=0, vmax=2,
            n_par=None,
    ):
        self.name = name  # corresponds to folder name and fit_df/meta_fit_df genotype
        self.label = label
        self.label_single = label_single
        self.folder_name = folder_name
        self.fish_age = fish_age
        self.fish_genotype = fish_genotype
        self.n_par = n_par
        self.query = query.lower()  # Genotypes are always stored in lower case
        self.color = color  # line color
        self.markerfacecolor = markerfacecolor  # marker face color
        self.color_accent = color_accent
        self.marker = marker
        self.cmap = cmap
        self.palette = palette
        # Hexbin density limits
        self.vmin = vmin
        self.vmax = vmax

        self.ref_agent = ref_agent
        self.ref_agent_str = ref_agent_str

    def __repr__(self):
        return f"Agent({self.name})"


# #############################################################################
# Agent classes
# #############################################################################
class Larva(Agent):
    def __init__(self,):
        super().__init__(
            name='larva', label='Larvae', label_single='Larva',
            folder_name='fig1',
            fish_age=5, fish_genotype='wt-kn',
            query='fish_age <= 5 and fish_genotype == "wt-kn"',
            ref_agent_str='larva',
            color=color_larva, markerfacecolor=color_larva, color_accent=color_larva_accent,
            cmap=CMAP_LARVA, palette=palette_larva,
        )

        self.color_bright = color_larva_bright
        self.color_dark = color_larva_dark


class Juvie(Agent):
    def __init__(self, ):
        super().__init__(
            name='juvie', label='Juveniles', label_single='Juvenile',
            folder_name='fig1',
            fish_age=21, fish_genotype='wt-kn',
            query='fish_age >= 21 and fish_genotype == "wt-kn"',
            ref_agent_str='juvie_larva', ref_agent=Larva(),  # show larva as reference
            color=color_juvie, markerfacecolor=color_juvie, color_accent=color_juvie_accent,
            cmap=CMAP_JUVIE, palette=palette_juvie,
        )

        self.color_bright = color_juvie_bright
        self.color_dark = color_juvie_dark


class LarvaAgent(Agent):
    def __init__(self):
        super().__init__(
            name='model_ptAV_plST_aAV_tAV_sAV_05dpf', label='Larvae (agent)', label_single='Single larva (agent)',
            folder_name='fig4',
            fish_age=5, fish_genotype='model_ptAV_plST_aAV_tAV_sAV',
            query='fish_age <= 5 and fish_genotype == "model_ptAV_plST_aAV_tAV_sAV"',
            ref_agent_str='sim', ref_agent=Larva(),  # show larva as reference
            color=color_larva_agent, marker=MARKER_HOLLOW,
            markerfacecolor=COLOR_AGENT_MARKER, color_accent=color_larva_accent,
            cmap=CMAP_LARVA_AGENT, palette=palette_larva,
            n_par=12,
        )


class JuvieAgent(Agent):
    def __init__(self,):
        super().__init__(
            name='model_ptAV_plST_aAV_tAV_sAV_27dpf', label='Juveniles (agent)', label_single='Single juvenile (agent)',
            folder_name='fig4',
            fish_age=21, fish_genotype='model_ptAV_plST_aAV_tAV_sAV',
            query='fish_age >= 21 and fish_genotype == "model_ptAV_plST_aAV_tAV_sAV"',
            ref_agent_str='sim', ref_agent=Juvie(),  # show juvie as reference
            color=color_juvie_agent, marker=MARKER_HOLLOW,
            markerfacecolor=COLOR_AGENT_MARKER, color_accent=color_juvie_accent,
            cmap=CMAP_JUVIE_AGENT, palette=palette_juvie,
            n_par=12,
        )

