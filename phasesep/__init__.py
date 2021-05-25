"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

__version__ = "0.3"

from pde.fields import *  # @UnusedWildImport
from pde.grids import *  # @UnusedWildImport
from pde.solvers import *  # @UnusedWildImport
from pde.storage import *  # @UnusedWildImport
from pde.trackers import *  # @UnusedWildImport
from pde.visualization import *  # @UnusedWildImport

from .cahn_hilliard_multiple import CahnHilliardMultiplePDE  # @UnusedWildImport
from .flory_huggins import FloryHugginsNComponents  # @UnusedWildImport
from .reactions import Reaction, Reactions
