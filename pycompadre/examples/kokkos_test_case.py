from unittest import TestCase
import pycompadre
import sys

class KokkosTestCase(TestCase):

    instances = 0

    @classmethod
    def setUpClass(cls):
        KokkosTestCase.instances += 1
        if KokkosTestCase.instances==1: 
            KokkosTestCase.shared_resource = pycompadre.KokkosParser(sys.argv)
