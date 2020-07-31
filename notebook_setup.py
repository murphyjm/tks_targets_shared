# Adapted from exoplanet

"""isort:skip_file"""

# Magic commands
get_ipython().magic('matplotlib inline')
get_ipython().magic('config InlineBackend.figure_format = "retina"') # For high quality figures
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

import os
import logging
import warnings

import matplotlib.pyplot as plt
from matplotlib import rcParams

###################################################
##### From exoplanet's notebook_setup.py file #####
###################################################

# Don't use the schmantzy progress bar
os.environ["EXOPLANET_NO_AUTO_PBAR"] = "true"

# Remove when Theano is updated
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Remove when arviz is updated
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger("theano.gof.compilelock")
logger.setLevel(logging.ERROR)
logger = logging.getLogger("exoplanet")
logger.setLevel(logging.DEBUG)

plt.style.use("default")

###################################################
###################################################
###################################################

rcParams["savefig.dpi"] = 100
rcParams["figure.dpi"] = 100
rcParams["font.size"] = 16

rcParams["font.family"] = "serif"
rcParams["font.serif"] = "Times New Roman"
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = [r'\usepackage{amsmath} \usepackage{bm} \usepackage{physics}']
