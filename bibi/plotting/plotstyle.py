# Set default plotting parameters
import matplotlib
from matplotlib import pyplot
import matplotlib.font_manager as fm
from pylab import cm
from matplotlib.colors import TABLEAU_COLORS as tcolors

matplotlib.use('TkAgg')
#fm._rebuild()

font  = {'family' : 'serif', 'serif' : 'Times New Roman',\
	     'weight' : 'normal', 'style' : 'normal', 'size' : 8.0}
axes = {'labelweight' : 'normal', 'titlesize' : 10.0, 'labelsize' : 8.0,\
        'titlepad' : 8.0, 'labelpad' : 8.0, 'linewidth' : 1.0,\
        'formatter.use_mathtext' : True, 'formatter.min_exponent' : True,\
        'grid' : False, 'grid.axis' : 'both', 'xmargin' : 0.05, 'ymargin' : 0.05,\
        'autolimit_mode': 'data'}
lines  = {'linewidth' : 2.0, 'antialiased' : True, 'dashed_pattern' : (4.0, 2.0)}
tick  =  {'direction' : 'in', 'minor.visible': True,\
          'major.size' : 7.0, 'major.width' : 1.0, 'major.pad' : 4.0,\
          'minor.size' : 4.0, 'minor.width' : 1.0, 'minor.pad' : 4.0}
figure = {'figsize' : (3.5,2.8), 'dpi' : 300, 'autolayout' : False}
math = {'fontset' : 'stix', 'bf' : 'normal', 'it' : 'italic', 'rm' : 'sans'}

matplotlib.rc('font', **font)
matplotlib.rc('lines', **lines)
matplotlib.rc('axes', **axes)
matplotlib.rc('xtick', **tick)
matplotlib.rc('ytick', **tick)
matplotlib.rc('figure', **figure)
matplotlib.rc('mathtext', **math)
colors = {'black':'#000000', 'blue':'#377eb8', 'orange':'#ff7f00'\
        , 'green':'#4daf4a', 'red':'#e41a1c', 'purple':'#984ea3'\
        , 'magenta':'#FF00FF','pink':'#EE1289',  'yellow':'#ffff33'\
        , 'cyan':'#00FFFF', 'crimson':'#8C000F', 'gold':'#FFD700'\
        , 'lime':'#00FF00', 'white':'#ffffff'}

