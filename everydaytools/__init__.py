# Set the module version consistent with pip freeze. Handle exception if didn't
# install with pip
import pkg_resources as pkg
try:
	__version__ = pkg.get_distribution("EverydayTools").version.lstrip('-').rstrip('-')
except:
	__version__ = "unknown_version"

# Add all submodules
from . import dataProcessing
from . import nicePlots
from . import batchTools
from . import fluidProperties # TODO needs to be improved in order to work
from . import igesToolbox
from . import OpenFOAMiO # TODO needs a complete overhaul...
from . import stlToolbox # TODO okay but very old and could be done a lot better
from . import tecPlotIO
from . import ReFRESCOiO # TODO very basic (only one file type); but rest can be done with standard TecPlot, usually. pymatt has a much better interface
from . import scripting
from . import vrmlToolbox
