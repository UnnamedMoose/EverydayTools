# Set the module version consistent with pip freeze. Handle exception if didn't
# install with pip
import pkg_resources as pkg
try:
	__version__ = pkg.get_distribution("everydaytools").version.lstrip('-').rstrip('-')
except:
	__version__ = "unknown_version"

# Add all submodules
from . import nicePlots
from . import dynamometryTools
from . import igesTools
from . import vrmlTools
from . import hexpressTools
from . import objTools
from . import geomTools
from . import tecPlotIO

from . import scripting  # TODO these two are duplicates...
from . import batchTools

from . import ReFRESCOiO # TODO very basic (only one file type); but rest can be done with standard TecPlot, usually. pymatt has a much better interface
from . import fluidProperties # TODO needs to be improved in order to work
# from . import OpenFOAMiO # TODO needs a complete overhaul...
from . import stlTools # TODO okay but very old and could be done a lot better
