# Set default plotting parameters
import matplotlib
from matplotlib import pyplot
import matplotlib.font_manager as fm 
from pylab import cm

matplotlib.use('TkAgg')
#fm._rebuild()

font  = {'family' : 'Times New Roman', 'serif' : 'Times New Roman',\
	 'weight' : 'normal', 'style' : 'normal', 'size' : 16.0}
axes = {'labelweight' : 'normal', 'titlesize' : 20.0, 'labelsize' : 16.0,\
        'titlepad' : 8.0, 'labelpad' : 8.0, 'linewidth' : 1.0,\
        'formatter.use_mathtext' : True, 'formatter.min_exponent' : True,\
        'grid' : False, 'grid.axis' : 'both', 'xmargin' : 0.05, 'ymargin' : 0.05,\
        'autolimit_mode': 'data'}
lines  = {'linewidth' : 2.0, 'antialiased' : True, 'dashed_pattern' : (4.0, 2.0)}
tick  =  {'direction' : 'in', 'minor.visible': True,\
          'major.size' : 7.0, 'major.width' : 1.0, 'major.pad' : 4.0,\
          'minor.size' : 4.0, 'minor.width' : 1.0, 'minor.pad' : 4.0}
figure = {'figsize' : (6.5,4.0), 'dpi' : 400, 'autolayout' : True}
math = {'fontset' : 'stix', 'bf' : 'normal', 'it' : 'italic', 'rm' : 'sans'}

matplotlib.rc('font', **font)
matplotlib.rc('lines', **lines)
matplotlib.rc('axes', **axes)
matplotlib.rc('xtick', **tick)
matplotlib.rc('ytick', **tick)
matplotlib.rc('figure', **figure)
matplotlib.rc('mathtext', **math)
colors = {'black':'#000000', 'red':'#e41a1c', 'blue':'#377eb8', 'green':'#4daf4a', 'purple':'#984ea3', 'orange':'#ff7f00', 'yellow':'#ffff33','pink':'#EE1289', 'cyan':'#00FFFF', 'magenta':'#FF00FF', 'crimson':'#8C000F', 'gold':'#FFD700', 'lime':'#00FF00', 'white':'#ffffff'}

