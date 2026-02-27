from ._pycompadre import *
__doc__ = _pycompadre.__doc__
try:
    from ._build_info import Compadre_DEBUG
except ImportError:
    Compadre_DEBUG = "Debug status unknown"
__doc__ += '\nCompadre_DEBUG: ' + str(Compadre_DEBUG)
try:
    from ._build_info import Compadre_BUILD_TYPE
except ImportError:
    Compadre_BUILD_TYPE = "Build type unknown"
__doc__ += '\nCompadre_BUILD_TYPE: ' + str(Compadre_BUILD_TYPE)
__version__ = _pycompadre.__version__
__all__ = _pycompadre.__dict__
