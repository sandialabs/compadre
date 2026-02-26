from ._pycompadre import *
__doc__ = _pycompadre.__doc__
try:
    from ._build_info import Compadre_DEBUG
except ImportError:
    Compadre_DEBUG = "Debug status unknown"
__doc__ += '\nCompadre_DEBUG: ' + str(Compadre_DEBUG)
__version__ = _pycompadre.__version__
__all__ = _pycompadre.__dict__
