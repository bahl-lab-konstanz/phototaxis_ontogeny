
# Import packages #############################################################
import os
# # Third-party packages
import matplotlib
matplotlib.use('TkAgg')  # Use Tkinter backend for interactive plotting
import matplotlib.pyplot as plt
plt.ion()  # Enable interactive mode

# # Local library imports
from settings.general_settings import do_presentation

# #############################################################################
# Plot settings
# #############################################################################
# Presentation vs print settings ##############################################
if do_presentation:
    plt.style.use('dark_background')  # For presentations
    SMALL_SIZE = 6
    BIGGER_SIZE = 8
    ALPHA = 0.5
    CMAP_GREY = plt.get_cmap('Greys_r')  # cmaps.grayC_r   # white to black
    CMAP_DIFF = plt.get_cmap('vanimo')
    CMAP_LARVA = plt.get_cmap('Blues_r')
    CMAP_JUVIE = plt.get_cmap('Oranges_r')
    CMAP_LARVA_AGENT = CMAP_LARVA
    CMAP_JUVIE_AGENT = CMAP_JUVIE
    COLOR_TEXT = 'w'
    COLOR_ANNOT = 'w'
    COLOR_MODEL = 'w'
    COLOR_AGENT_MARKER = 'k'    # same as background color
else:
    SMALL_SIZE = 6
    BIGGER_SIZE = 8
    ALPHA = 0.3
    CMAP_GREY = plt.get_cmap('Greys')  # cmaps.grayC     # black to white
    CMAP_DIFF = plt.get_cmap('PiYG')
    CMAP_LARVA = plt.get_cmap('Blues')
    CMAP_JUVIE = plt.get_cmap('Oranges')
    CMAP_LARVA_AGENT = CMAP_LARVA
    CMAP_JUVIE_AGENT = plt.get_cmap('Reds')
    COLOR_TEXT = 'k'
    COLOR_ANNOT = 'k'
    COLOR_MODEL = 'k'
    COLOR_AGENT_MARKER = 'w'    # same as background color

# Colors ######################################################################
# # Larva
color_larva = 'tab:blue'
color_larva_agent = plt.get_cmap('tab20')(1)   # light blue
color_larva_accent = plt.get_cmap('tab20')(1)   # light blue
color_larva_bright = plt.get_cmap('tab20c')(2)  # light blue
color_larva_dark = plt.get_cmap('tab20c')(0)    # dark blue

# # Juveniles
color_juvie = 'tab:orange'
color_juvie_agent = 'salmon'  # plt.get_cmap('tab20c')(4)   # dark orange
color_juvie_accent = plt.get_cmap('tab20')(3)   # light orange from tab20
color_juvie_bright = plt.get_cmap('tab20c')(6)  # light orange
color_juvie_dark = plt.get_cmap('tab20c')(4)    # dark orange

# # Palettes
palette_larva = "mako"      # seaborn palette
palette_juvie = "rocket"    # seaborn palette
AGE_PALETTE_DICT = {
    5: 'tab:blue', '5': 'tab:blue',
    26: 'tab:orange', '26': 'tab:orange',
    27: 'tab:orange', '27': 'tab:orange',
}


# Figure sizes ################################################################
cm = 1/2.54  # centimeters in inches
legend_space = 2 * cm
fig_width_cm = 18.4                 # cm (Science Advances)
fig_height_cm = fig_width_cm / 2    # cm
fig_x_cm = 18  # cm
fig_y_cm = 9  # cm
pad_x_cm = 1.5  # cm, allows for y-ticks, y-label
pad_y_cm = 1.5  # cm, allows for x-ticks, x-label and title
ax_x_cm = 2.5   # cm, full axis width (incl. ticks and labels)
ax_y_cm = 4     # cm, full axis height (incl. ticks and labels)

# Extra large figures for supplementary columns or rows
fig_big_width = 29          # cm, extra wide for supplementary columns
fig_big_height = 18         # cm, extra tall for supplementary rows

# Ticks and labels ############################################################
# Define ticks
x_ticks = [-5, 0, 5]
radius_ticks = [0, 2.5, 5]
azimuth_ticks = [0, 180, 360]
brightness_ticks = [0, 150, 300]

x_ticklabels = [-5, 0, 5]
radius_ticklabels = ['0', 2.5, '5']
azimuth_ticklabels = azimuth_ticks
brightness_ticklabels = brightness_ticks

# Text, lines and markers #####################################################
LW_ANNOT = 0.5
MARKER_SIZE = 3
MARKER_SIZE_LARGE = 6
ANNOT_Y, ANNOT_Y_HIGH = 1, 1.15  # Height of annotation lines (in axes coordinates)
# if os.name == 'posix':
#     # Hollow circle for stripplot (works only on mac?)
#     MARKER_HOLLOW = r'$\circ$'
# else:
#     MARKER_HOLLOW = 'o'
MARKER_HOLLOW = r'$\circ$'

# # Text
plt.rcParams['pdf.fonttype'] = 42  #
plt.rcParams['ps.fonttype'] = 42
plt.rc('font', size=SMALL_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE, frameon=False)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE) # fontsize of the figure title
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
print("Using font:", plt.rcParams["font.sans-serif"])

# # Lines
plt.rcParams['lines.markersize'] = MARKER_SIZE
# # Axes
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['ytick.major.size'] = 4
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['xtick.minor.size'] = 0
plt.rcParams['ytick.minor.size'] = 0
# # Error bars
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['lines.markersize'] = 2

