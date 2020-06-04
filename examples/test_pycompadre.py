from unittest import TestCase
import numpy as np
import math
import random
import pycompadre

kokkos_obj=pycompadre.KokkosParser()

# function used to generate sample data
def exact(coord,order,dimension):
    x = coord[0]
    y = coord[1] if (dimension>1) else 0
    z = coord[2] if (dimension>2) else 0
    if order==1:
        return 1 + x + y + z
    elif order==2:
        return 1 + x + y + z + x*x + x*y + x*z + y*y + y*z + z*z
    elif order==3:
        return 1 + x + y + z + x*x + x*y + x*z + y*y + y*z + z*z + x*x*x + x*x*y + x*x*z + x*y*y + x*y*z + x*z*z + y*y*y + y*y*z + y*z*z + z*z*z

# function used to get analytic gradient
def grad_exact(coord,component,order,dimension):
    x = coord[0]
    y = coord[1] if (dimension>1) else 0
    z = coord[2] if (dimension>2) else 0
    if (component==0):
        if order==1:
            return 1
        elif order==2:
            return 1 + 2*x + y + z
        elif order==3:
            return 1 + 2*x + y + z + 3*x*x + 2*x*y + 2*x*z + y*y + y*z + z*z
    elif (component==1):
        if order==1:
            return 1
        elif order==2:
            return 1 + x + 2*y + z
        elif order==3:
            return 1 + x + 2*y + z + x*x + x*2*y + x*z + 3*y*y + 2*y*z + z*z
    elif (component==2):
        if order==1:
            return 1
        elif order==2:
            return 1 + x + y + 2*z
        elif order==3:
            return 1 + x + y + 2*z + x*x + x*y + x*2*z + y*y + y*2*z + 3*z*z

def remap(polyOrder,dimension):

    minND = [[10,20,30],[10,20,100],[30,30,60]]
    ND = minND[dimension-1][polyOrder-1]

    random.seed(1234) # for consistent results

    dimensions = dimension

    # initialize 3rd order reconstruction using 2nd order basis in 3D (GMLS)
    gmls_obj=pycompadre.GMLS(polyOrder, dimensions, "QR", "STANDARD")
    gmls_obj.setWeightingPower(4)
    gmls_obj.setWeightingType("power")

    NT = 10 # Targets
    nx, ny, nz = (ND, ND, ND)

    xmax = 1
    if (dimension>1): ymax = 1
    if (dimension>2): zmax = 1
    xmin = -xmax
    if (dimension>1): ymin = -ymax
    if (dimension>2): zmin = -zmax

    dx = np.linspace(xmin, xmax, nx)
    if (dimension>1): dy = np.linspace(ymin, ymax, ny)
    if (dimension>2): dz = np.linspace(zmin, zmax, nz)

    N=1
    for i in range(dimension):
        N*=ND

    # target sites
    target_sites = []
    for i in range(NT):
        if (dimension==1):
            target_sites.append([random.uniform(xmin, xmax)])
        elif (dimension==2):
            target_sites.append([random.uniform(xmin, xmax), random.uniform(ymin, ymax)])
        elif (dimension==3):
            target_sites.append([random.uniform(xmin, xmax), random.uniform(ymin, ymax), random.uniform(zmin, zmax)])
    target_sites = np.array(target_sites, dtype='d')

    # source sites
    t_sites = list()
    for i in range(ND):
        if (dimension==1):
            t_sites.append([dx[i],])
        else:
            for j in range(ND):
                if (dimension==2):
                    t_sites.append([dx[i],dy[j]])
                else:
                    for k in range(ND):
                        t_sites.append([dx[i],dy[j],dz[k]])
    source_sites = np.array(t_sites, dtype=np.dtype('d'))

    # neighbor search
    epsilon_multiplier = 1.5
    gmls_helper = pycompadre.ParticleHelper(gmls_obj)
    gmls_helper.generateKDTree(source_sites)
    gmls_helper.generateNeighborListsFromKNNSearchAndSet(target_sites, polyOrder, dimensions, epsilon_multiplier)

    # set data in gmls object
    gmls_helper.setSourceSites(source_sites)

    # used in combination with polynomial coefficients
    epsilons = gmls_helper.getWindowSizes()

    gmls_obj.addTargets(pycompadre.TargetOperation.ScalarPointEvaluation)
    gmls_obj.addTargets(pycompadre.TargetOperation.PartialXOfScalarPointEvaluation)
    (dimensions>1) and gmls_obj.addTargets(pycompadre.TargetOperation.PartialYOfScalarPointEvaluation)
    (dimensions>2) and gmls_obj.addTargets(pycompadre.TargetOperation.PartialYOfScalarPointEvaluation)

    # generate stencil with number of batches set to 1, and keeping coefficients (not necessary)
    gmls_obj.generateAlphas(1, True)

    # create sample data at source sites
    data_vector = []
    for i in range(N):
        data_vector.append(exact(source_sites[i], polyOrder, dimension))
    data_vector = np.array(data_vector, dtype=np.dtype('d'))

    # apply stencil to sample data for all targets
    computed_answer = gmls_helper.applyStencil(data_vector, pycompadre.TargetOperation.ScalarPointEvaluation)

    l2_error = 0
    for i in range(NT):
        l2_error += np.power(abs(computed_answer[i] - exact(target_sites[i],polyOrder,dimension)),2)
    l2_error = math.sqrt(l2_error/float(NT))

    # get polynomial coefficients
    polynomial_coefficients = gmls_helper.getPolynomialCoefficients(data_vector)

    # alternative way to compute h1 semi norm
    # could remap using the gradient operator, but instead this uses calculated polynomial coefficients and applies
    # the action of the gradient target operation on the polynomial basis at the target sites
    # this serves as a test for getting accurate calculation and retrieval of polynomial coefficients using
    # the python interface
    h1_seminorm_error = 0
    for i in range(NT):
        h1_seminorm_error += np.power(abs(1./epsilons[i]*polynomial_coefficients[i,1] - grad_exact(target_sites[i], 0, polyOrder, dimension)),2)
        if (dimension>1): h1_seminorm_error += np.power(abs(1./epsilons[i]*polynomial_coefficients[i,2] - grad_exact(target_sites[i], 1, polyOrder, dimension)),2)
        if (dimension>2): h1_seminorm_error += np.power(abs(1./epsilons[i]*polynomial_coefficients[i,3] - grad_exact(target_sites[i], 2, polyOrder, dimension)),2)
    h1_seminorm_error = math.sqrt(h1_seminorm_error/float(NT))

    return l2_error, h1_seminorm_error

class TestPycompadre(TestCase):
    def test_1d_order1(self):
        l2,h1=remap(1,1)
        self.assertTrue(l2<1e-13 and h1<1e-13)
    def test_1d_order2(self):
        l2,h1=remap(2,1)
        self.assertTrue(l2<1e-13 and h1<1e-13)
    def test_1d_order3(self):
        l2,h1=remap(3,1)
        self.assertTrue(l2<1e-13 and h1<1e-13)

    def test_2d_order1(self):
        l2,h1=remap(1,2)
        self.assertTrue(l2<1e-13 and h1<1e-13)
    def test_2d_order2(self):
        l2,h1=remap(2,2)
        self.assertTrue(l2<1e-13 and h1<1e-13)
    def test_2d_order3(self):
        l2,h1=remap(3,2)
        self.assertTrue(l2<1e-13 and h1<1e-13)

    def test_3d_order1(self):
        l2,h1=remap(1,3)
        self.assertTrue(l2<1e-13 and h1<1e-13)
    def test_3d_order2(self):
        l2,h1=remap(2,3)
        self.assertTrue(l2<1e-13 and h1<1e-13)
    def test_3d_order3(self):
        l2,h1=remap(3,3)
        self.assertTrue(l2<1e-13 and h1<1e-13)
