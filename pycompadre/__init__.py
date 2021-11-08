import glob
import os

def _bootstrap():
   global _bootstrap, __loader__, __file__
   import sys, pkg_resources, imp
   shared_object_files = glob.glob(str(os.path.dirname(os.path.abspath(__file__)) + '/pycompadre*'))
   assert len(shared_object_files)==1, "Failed to import pycompadre."
   __file__ = pkg_resources.resource_filename(__name__, os.path.basename(shared_object_files[0]))
   __loader__ = None
   del _bootstrap, __loader__
   imp.load_dynamic(__name__, __file__)
_bootstrap()
