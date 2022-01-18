from kokkos_test_case import KokkosTestCase
import numpy as np
import math
import random
import pycompadre
from functools import partialmethod

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

def remap(polyOrder,dimension,additional_sites=False,epsilon_multiplier=1.5,reconstruction_space=pycompadre.ReconstructionSpace.VectorOfScalarClonesTaylorPolynomial,sampling_functional=pycompadre.SamplingFunctional["VectorPointSample"]):

    minND = [[10,20,30],[10,20,100],[30,30,60]]
    ND = minND[dimension-1][polyOrder-1]

    random.seed(1234) # for consistent results

    dimensions = dimension

    # initialize 3rd order reconstruction using 2nd order basis in 3D (GMLS)
    gmls_obj=pycompadre.GMLS(reconstruction_space, sampling_functional, polyOrder, dimensions, "QR", "STANDARD")
    gmls_obj.setWeightingParameter(4)
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
    (dimensions>2) and gmls_obj.addTargets(pycompadre.TargetOperation.PartialZOfScalarPointEvaluation)

    # add additional evaluation sites (if specified)
    if additional_sites:
        additional_sites_indices = np.zeros(shape=(NT,5), dtype='i4')
        additional_site_coordinates = np.zeros(shape=(4*NT,dimension), dtype='f8')
        additional_sites_indices[:,0] = 4
        inds = np.arange(0, 4*NT)
        inds = np.reshape(inds, newshape=(-1,4))
        additional_sites_indices[:,1::] = inds
        h = np.linalg.norm(source_sites[0,:]-source_sites[1,:])
        for i in range(NT):
            for j in range(4):
                for k in range(dimension):
                    additional_site_coordinates[i*4+j,k] = target_sites[i,k] + random.uniform(-h, h)

        gmls_helper.setAdditionalEvaluationSitesData(additional_sites_indices, additional_site_coordinates)

    # generate stencil with number of batches set to 1, and keeping coefficients (not necessary)
    gmls_obj.generateAlphas(1, True)

    # create sample data at source sites
    data_vector = []
    for i in range(N):
        data_vector.append(exact(source_sites[i], polyOrder, dimension))
    # use rank 2 array and only insert into one column to test
    # whether layouts are being properly propagated into pycompadre
    new_data_vector = np.zeros(shape=(len(data_vector), 3), dtype='f8')
    new_data_vector[:,1] = np.array(data_vector, dtype=np.dtype('d'))

    # apply stencil to sample data for all targets
    computed_answer = gmls_helper.applyStencil(new_data_vector[:,1], pycompadre.TargetOperation.ScalarPointEvaluation)

    l2_error = 0
    for i in range(NT):
        l2_error += np.power(abs(computed_answer[i] - exact(target_sites[i],polyOrder,dimension)),2)
    l2_error = math.sqrt(l2_error/float(NT))

    additional_sites_l2_error = 0.0
    if additional_sites:
        nl = gmls_helper.getNeighborLists()
        # test min/max num neighbors computation works
        nl.computeMinNumNeighbors()
        nl.computeMaxNumNeighbors()
        n_min = nl.getMinNumNeighbors()
        n_max = nl.getMaxNumNeighbors()
        for i in range(NT):
            for j in range(4):
                computed_answer = 0.0
                for k in range(nl.getNumberOfNeighbors(i)):
                    computed_answer += gmls_obj.getSolutionSet().getAlpha(pycompadre.TargetOperation.ScalarPointEvaluation, 
                                                            i, 0, 0, k, 0, 0, j+1)*data_vector[nl.getNeighbor(i,k)]
                additional_sites_l2_error += \
                    np.power(abs(computed_answer - exact(additional_site_coordinates[i*4+j],polyOrder,dimension)),2)

    # retrieve additional sites neighbor lists just to make sure it works
    a_nl = gmls_helper.getAdditionalEvaluationIndices()

    # get polynomial coefficients
    polynomial_coefficients = gmls_helper.getPolynomialCoefficients(data_vector)

    # alternative way to compute h1 semi norm
    # could remap using the gradient operator, but instead this uses calculated polynomial coefficients and applies
    # the action of the gradient target operation on the polynomial basis at the target sites
    # this serves as a test for getting accurate calculation and retrieval of polynomial coefficients using
    # the python interface
    h1_seminorm_error = 0
    if reconstruction_space in (pycompadre.ReconstructionSpace.VectorOfScalarClonesTaylorPolynomial, pycompadre.ReconstructionSpace.ScalarTaylorPolynomial):
        for i in range(NT):
            h1_seminorm_error += np.power(abs(1./epsilons[i]*polynomial_coefficients[i,1] - grad_exact(target_sites[i], 0, polyOrder, dimension)),2)
            if (dimension>1): h1_seminorm_error += np.power(abs(1./epsilons[i]*polynomial_coefficients[i,2] - grad_exact(target_sites[i], 1, polyOrder, dimension)),2)
            if (dimension>2): h1_seminorm_error += np.power(abs(1./epsilons[i]*polynomial_coefficients[i,3] - grad_exact(target_sites[i], 2, polyOrder, dimension)),2)
    h1_seminorm_error = math.sqrt(h1_seminorm_error/float(NT))

    if additional_sites:
        return l2_error, h1_seminorm_error, additional_sites_l2_error
    return l2_error, h1_seminorm_error


class TestPyCOMPADRE(KokkosTestCase):

    # most tests are added from a dictionary below 

    def test_square_qr_bugfix(self):

        source_sites = np.array([2.0,3.0,5.0,6.0,7.0], dtype='f8')
        source_sites = np.reshape(source_sites, newshape=(source_sites.size,1))
        data = np.array([2.0,3.0,5.0,6.0,7.0], dtype='f8')

        polynomial_order = 1
        dim = 1

        gmls_obj=pycompadre.GMLS(polynomial_order, 1, "QR", "STANDARD")
        gmls_obj.addTargets(pycompadre.TargetOperation.ScalarPointEvaluation)
        gmls_obj.addTargets(pycompadre.TargetOperation.PartialXOfScalarPointEvaluation)

        gmls_helper = pycompadre.ParticleHelper(gmls_obj)
        gmls_helper.generateKDTree(source_sites)

        point = np.array([4.0], dtype='f8')
        target_site = np.reshape(point, newshape=(1,dim))

        gmls_helper.generateNeighborListsFromKNNSearchAndSet(target_site, polynomial_order, dim, 1.5)
        gmls_obj.generateAlphas(1, True)

        output = gmls_helper.applyStencilSingleTarget(data, pycompadre.TargetOperation.PartialXOfScalarPointEvaluation)

        del gmls_helper
        del gmls_obj

        self.assertAlmostEqual(output, 1.0, places=15)

# space / sampling combinations
space_sample_combos = {
        "stp_ps":(pycompadre.ReconstructionSpace.ScalarTaylorPolynomial, pycompadre.SamplingFunctional["PointSample"]), 
        "vsctp_vps":(pycompadre.ReconstructionSpace.VectorOfScalarClonesTaylorPolynomial, pycompadre.SamplingFunctional["VectorPointSample"]),
        #"brnst_vps":(pycompadre.ReconstructionSpace.BernsteinPolynomial, pycompadre.SamplingFunctional["VectorPointSample"])
        }

#############################

#
# Begin Most Generic Tests
#

# generic test that can handle all options
def base_test(self,polyOrder,dimension,additional_sites=False,epsilon_multiplier=1.5,reconstruction_space=pycompadre.ReconstructionSpace.VectorOfScalarClonesTaylorPolynomial,sampling_functional=pycompadre.SamplingFunctional["VectorPointSample"]):
    copy_kwargs = locals()
    del copy_kwargs['self']
    out = remap(**copy_kwargs)
    l2 = out[0]
    h1 = out[1]
    self.assertTrue(l2<1e-13 and h1<1e-13)
    if (len(out)>2):
        l2a = out[2]
        self.assertTrue(l2a<1e-13)

# combine order, dim, etc.... to form all possible functions to test
for polyorder in (1,2,3):
    for dim in (1,2,3):
        for space_sample_combo in space_sample_combos:
            for additional_sites in (False, True):
                # add these member functions to TestPyCOMPADRE
                setattr(TestPyCOMPADRE,"test_%dd_order%d_a%s_"%(polyorder,dim,str(additional_sites))+str(space_sample_combo),partialmethod(base_test, polyOrder=polyorder, dimension=dim, additional_sites=additional_sites, epsilon_multiplier=1.5, reconstruction_space=space_sample_combos[space_sample_combo][0], sampling_functional=space_sample_combos[space_sample_combo][1]))
#
# End Most Generic Tests
#

#############################

#
# Begin Square Matrix Tests
#
def test_square_qr(self, space_sample_combo):
    l2,h1=remap(1,1,False,epsilon_multiplier=1.01,reconstruction_space=space_sample_combo[0],sampling_functional=space_sample_combo[1])
    self.assertTrue(l2<1e-13 and h1<1e-13)

# add combos as tests from generic tests
#[setattr(TestPyCOMPADRE,"test_square_qr_"+str(key),partialmethod(test_square_qr, space_sample_combo=space_sample_combos[key])) for key in space_sample_combos]
#
# End Square Matrix Tests
#

#############################

#tc = TestPycompadre()
#tc.test_additional_sites()
#tc.test_3d_order1()

if __name__ == '__main__':
    import unittest
    unittest.main()
