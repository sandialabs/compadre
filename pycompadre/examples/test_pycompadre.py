from kokkos_test_case import KokkosTestCase
from unittest import TestCase, skipIf
import numpy as np
import math
import random
import pycompadre
import pickle

try:
    import sys
    if sys.version_info[0]==2:
        from functools import partial
        class partialmethod(partial):
            def __get__(self, instance, owner):
                if instance is None:
                    return self
                return partial(self.func, instance,
                               *(self.args or ()), **(self.keywords or {}))
    else:
        raise
except:
    from functools import partialmethod

def skip_additional_initializes_for_kokkos():
    # don't reinitialize with CUDA
    # get status
    st = pycompadre.Kokkos.status()
    if "KOKKOS_ENABLE_CUDA: yes" in st:
        return True
    return False
kokkos_skip_check = skipIf(skip_additional_initializes_for_kokkos(), "Can't reinitialize Kokkos with CUDA")

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

def remap(polyOrder,dimension,additional_sites=False,epsilon_multiplier=1.5,reconstruction_space=pycompadre.ReconstructionSpace.VectorOfScalarClonesTaylorPolynomial,sampling_functional=pycompadre.SamplingFunctionals["VectorPointSample"]):

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
    else:
        grad_x = gmls_helper.applyStencil(new_data_vector[:,1], pycompadre.TargetOperation.PartialXOfScalarPointEvaluation)
        for i in range(NT):
            h1_seminorm_error += np.power(grad_x[i] - grad_exact(target_sites[i], 0, polyOrder, dimension),2)
        if (dimension>1): 
            grad_y = gmls_helper.applyStencil(new_data_vector[:,1], pycompadre.TargetOperation.PartialYOfScalarPointEvaluation)
            for i in range(NT):
                h1_seminorm_error += np.power(grad_y[i] - grad_exact(target_sites[i], 1, polyOrder, dimension),2)
        if (dimension>2): 
            grad_z = gmls_helper.applyStencil(new_data_vector[:,1], pycompadre.TargetOperation.PartialZOfScalarPointEvaluation)
            for i in range(NT):
                h1_seminorm_error += np.power(grad_z[i] - grad_exact(target_sites[i], 2, polyOrder, dimension),2)
    h1_seminorm_error = math.sqrt(h1_seminorm_error/float(NT))

    if additional_sites:
        return l2_error, h1_seminorm_error, additional_sites_l2_error
    return l2_error, h1_seminorm_error

class TestKokkosParser(TestCase):

    @kokkos_skip_check
    def test_init_and_finalize_sysargv(self):
        pycompadre.Kokkos.initialize(sys.argv)
        pycompadre.Kokkos.finalize()

    @kokkos_skip_check
    def test_init_and_finalize_empty(self):
        pycompadre.Kokkos.initialize()
        pycompadre.Kokkos.finalize()

    @kokkos_skip_check
    def test_init_and_finalize_list(self):
        pycompadre.Kokkos.initialize(["--kokkos-num-threads=4"])
        pycompadre.Kokkos.finalize()

    @kokkos_skip_check
    def test_init_and_finalize_initarguments(self):
        kia = pycompadre.Kokkos.InitArguments()
        pycompadre.Kokkos.initialize(kia)
        pycompadre.Kokkos.finalize()

    @kokkos_skip_check
    def test_scope_guard_sysargv(self):
        pycompadre.KokkosParser(sys.argv)

    @kokkos_skip_check
    def test_scope_guard_empty(self):
        pycompadre.KokkosParser()

    @kokkos_skip_check
    def test_scope_guard_list(self):
        pycompadre.KokkosParser(["--kokkos-num-threads=4"])

    @kokkos_skip_check
    def test_scope_guard_initarguments(self):
        kia = pycompadre.Kokkos.InitArguments()
        pycompadre.KokkosParser(kia)


class TestPyCOMPADRE(KokkosTestCase):

    # most tests are added from a dictionary below 

    def test_kokkos_double_initialize(self):

        # setup class for KokkosTestCase already initialized Kokkos
        kp = pycompadre.KokkosParser(sys.argv)

    def test_1d_line_quadrature(self):

        # integrand
        f = lambda x: -3 + x - x**2
        # exact integral
        solution = -175.0/6.0

        q_order = 2
        q_dim   = 1
        qp = pycompadre.Quadrature(q_order, q_dim, "LINE")
        self.assertTrue(qp.validQuadrature())

        qpoints = qp.getSites()
        qweights = qp.getWeights()

        result = 0
        # domain is [-1,4]
        # scale from [0,1] to [-1,4]
        scale = 5
        shift = -1
        for i in range(len(qpoints)):
            result += f(scale*qpoints[i][0]+shift)*qweights[i]*scale

        self.assertAlmostEqual(result, solution, places=13)

    def test_2d_tri_quadrature(self):
        # integrand
        f = lambda x: -3 + x[1] + x[0] - x[0]**2 - x[1]**2 + 4*x[0]*x[1]
        # exact
        solution = -38 - 1.0/3

        q_order = 2
        q_dim   = 2
        qp = pycompadre.Quadrature(q_order, q_dim, "TRI")
        self.assertTrue(qp.validQuadrature())

        # reference
        qpoints = qp.getSites()
        qweights = qp.getWeights()

        # directions 
        v = np.zeros(shape=(2,3), dtype='f8')
        # domain is area covered by [-1,-1], [4,-1], [1,3]
        v[:,0] = [-1,-1]
        v[:,1] = [ 4,-1]
        v[:,2] = [ 1, 3]

        # physical pt from quadrature
        result = 0.0
        for i in range(len(qpoints)):
            new_dir = []
            new_dir.append(v[:,1]-v[:,0])
            new_dir.append(v[:,2]-v[:,0])

            new_q =  v[:,0].copy()
            for j in range(2):
                new_q += qpoints[i][j] * new_dir[j]

            # cross product gives stretch factor
            scaling = np.abs(np.cross(new_dir[0], new_dir[1]))
            result += f(new_q)*qweights[i]*scaling

        self.assertAlmostEqual(result, solution, places=13)

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

    def test_pickle_sampling_functional(self):
        # test pickling of SamplingFunctional
        sf = pycompadre.SamplingFunctionals["VectorPointSample"]
        state = sf.__getstate__()
        import pickle
        byte_sf = pickle.dumps(sf)
        new_sf = pickle.loads(byte_sf)
        new_state = new_sf.__getstate__()
        self.assertEqual(len(state), len(new_state))
        for i in range(len(state)):
            self.assertEqual(state[i], new_state[i])

    def test_pickle_gmls(self):

        source_sites = np.array([2.0,3.0,5.0,6.0,7.0], dtype='f8')
        source_sites = np.reshape(source_sites, newshape=(source_sites.size,1))
        data = np.array([2.0,3.0,5.0,6.0,7.0], dtype='f8')

        polynomial_order = 1
        dim = 1

        point = np.array([4.0], dtype='f8')
        target_site = np.reshape(point, newshape=(1,dim))
    
        # rvalue or std::unique_ptr passed to ParticleHelper (not obvious lifecycle of GMLS object)
        # in Clang, rvalue type argument works because it keeps track of which was created first
        #           and python keeps track of rvalue like object
        # in GCC,   rvalue type arguments do not work and cause increment/decrement issues
        # -----v NOT ok in GCC
        #gmls_helper = pycompadre.ParticleHelper(pycompadre.GMLS(polynomial_order, 1, "QR", "STANDARD"))
        gmls_obj = pycompadre.GMLS(polynomial_order, 1, "QR", "STANDARD")
        gmls_helper = pycompadre.ParticleHelper(gmls_obj)
        gmls_helper.generateKDTree(source_sites)
        gmls_helper.generateNeighborListsFromKNNSearchAndSet(target_site, polynomial_order, dim, 1.5)
        # print("1",gmls_helper.getGMLSObject().__getstate__())
        gmls_helper.getGMLSObject().__getstate__()

        # gets GMLS object from gmls_helper (python knows C++ owns it, but C++ treats it as
        # a python owned object at construction) so lifecycle of GMLS object is not tied to
        # the lifecycle of the GMLS helper
        gmls_obj = gmls_helper.getGMLSObject()
        # destroy helper 
        del gmls_helper
        # python kept a reference count on the argument to GMLS helper
        # which isn't obvious, but there is no move constructor for GMLS helper class
        # so the following is not destroyed, yet
        # print("2",gmls_obj.__getstate__())
        gmls_obj.__getstate__()

        # python knows that original GMLS object was tied to GMLS helper
        # so replacing the helper tells python the GMLS object is no longer needed
        gmls_obj2 = pycompadre.GMLS(polynomial_order, 1, "QR", "STANDARD")
        gmls_helper = pycompadre.ParticleHelper(gmls_obj2)
        # but python keeps this handle internally and now points at new rvalue like object
        # so the following still works
        # print("3",gmls_obj.__getstate__())
        gmls_obj2.__getstate__()

        # # In Clang, this will confuse python because it thinks the rvalue like GMLS object is
        # # no longer needed, so it will throw it away
        # gmls_helper = pycompadre.ParticleHelper(gmls_obj)
        # # so the gmls_obj actual gets destroyed in the previous call
        # gmls_obj = gmls_helper.getGMLSObject()
        # # see what happens to internal GMLS object
        # # object is destroyed and this segfaults
        # # we don't call v--- because it will segfault and unittest can't catch that
        # # print("4",gmls_obj.__getstate__())
        # gmls_obj.__getstate__()
        # # resetting GMLS helper after this block of code will cause deallocation 
        # ^--- This wouldn't be tested in GCC because we don't pass rvalue type
        #      arguments to constructors

        # GMLS object destroyed and then relied upon in gmls_helper
        gmls_obj=pycompadre.GMLS(polynomial_order, 1, "QR", "STANDARD")
        gmls_helper = pycompadre.ParticleHelper(gmls_obj)
        gmls_helper.generateKDTree(source_sites)
        gmls_helper.generateNeighborListsFromKNNSearchAndSet(target_site, polynomial_order, dim, 1.5)
        # the following v---- will segfault because gmls_obj is deleted
        # del gmls_obj
        # print(gmls_helper.getNeighborLists())
        # print(gmls_helper.getGMLSObject().__getstate__())
        gmls_obj2=gmls_helper.getGMLSObject()

        # delete python owned gmls_obj, non-owning gmls_obj2 (which points at owned gmls_obj)
        del gmls_obj, gmls_obj2, gmls_helper

        gmls_obj=pycompadre.GMLS(polynomial_order, 1, "QR", "STANDARD")
        gmls_obj.addTargets(pycompadre.TargetOperation.ScalarPointEvaluation)
        gmls_obj.addTargets(pycompadre.TargetOperation.PartialXOfScalarPointEvaluation)

        gmls_helper = pycompadre.ParticleHelper(gmls_obj)
        gmls_helper.generateKDTree(source_sites)

        point = np.array([4.0], dtype='f8')
        target_site = np.reshape(point, newshape=(1,dim))

        gmls_helper.generateNeighborListsFromKNNSearchAndSet(target_site, polynomial_order, dim, 1.5)
        gmls_obj.generateAlphas(1, True)

        # test pickling the gmls_obj but not gmls_helper
        byte_gmls = pickle.dumps(gmls_obj)
        new_gmls_obj = pickle.loads(byte_gmls)

        with open('test.p', 'wb') as fn:
            pickle.dump(gmls_obj, fn)
        with open('test.p', 'rb') as fn:
            new_gmls_obj = pickle.load(fn)

        del gmls_obj

        gmls_helper = pycompadre.ParticleHelper(new_gmls_obj)

        # explicitly we do not call generateKDTree (it must come from the older gmls_object)
        # gmls_helper.generateKDTree(source_sites)

        # explicitly we do not call generateNeighborL... so that target info must come from older gmls_object
        # gmls_helper.generateNeighborListsFromKNNSearchAndSet(target_site, polynomial_order, dim, 1.5)

        new_gmls_obj.generateAlphas(1, True)

        output = gmls_helper.applyStencilSingleTarget(data, pycompadre.TargetOperation.PartialXOfScalarPointEvaluation)

        byte_gmls_helper = pickle.dumps(gmls_helper)
        new_gmls_helper = pickle.loads(byte_gmls_helper)

        with open('test.p', 'wb') as fn:
            pickle.dump(gmls_helper, fn)
        del gmls_helper
        del new_gmls_obj

        self.assertAlmostEqual(output, 1.0, places=15)
        
        # test pickling of gmls_helper
        with open('test.p', 'rb') as fn:
            new_gmls_helper = pickle.load(fn)

        # should not contain a GMLS instance
        with self.assertRaises(RuntimeError):
            gmls_obj = new_gmls_helper.getGMLSObject()

    def test_pickling_additional_evaluation_sites(self):

        source_sites = np.array([2.0,3.0,5.0,6.0,7.0], dtype='f8')
        source_sites = np.reshape(source_sites, newshape=(source_sites.size,1))
        data = np.array([4.0,9.0,25.0,36.0,49.0], dtype='f8')

        polynomial_order = 2
        dim = 1

        point = np.array([4.0, 3.0], dtype='f8')
        target_site = np.reshape(point, newshape=(2,dim))

        extra_sites_coords = np.atleast_2d(np.linspace(0,4,5)).T
        extra_sites_idx    = np.zeros(shape=(len(point),len(extra_sites_coords)+1), dtype='i4')
        extra_sites_idx[0,0] = 0
        extra_sites_idx[0,1:] = np.arange(len(extra_sites_coords))
        extra_sites_idx[1,0] = len(extra_sites_coords)
        extra_sites_idx[1,1:] = np.arange(len(extra_sites_coords))

        gmls_obj = pycompadre.GMLS(polynomial_order, 1, "QR", "STANDARD")
        gmls_helper = pycompadre.ParticleHelper(gmls_obj)
        gmls_helper.generateKDTree(source_sites)
        gmls_helper.generateNeighborListsFromKNNSearchAndSet(target_site, polynomial_order, dim, 1.5)
        gmls_helper.setAdditionalEvaluationSitesData(extra_sites_idx, extra_sites_coords)

        gmls_obj.addTargets(pycompadre.TargetOperation.ScalarPointEvaluation)

        sol1 = [16.0, 0.0, 0.0, 0.0]
        sol2 = [ 9.0, 0.0, 1.0, 4.0]
        def check_answer(helper, i):
            output = helper.applyStencil(data, 
                                              pycompadre.TargetOperation.ScalarPointEvaluation,
                                              pycompadre.SamplingFunctionals['PointSample'],
                                              i)
            self.assertAlmostEqual(output[0], sol1[i], places=13)
            self.assertAlmostEqual(output[1], sol2[i], places=13)

        # throws error because alphas are not generated
        with self.assertRaises(RuntimeError):
            [check_answer(gmls_helper, i) for i in range(4)]

        self.assertEqual(gmls_obj.containsValidAlphas(), False)
        # generate alphas and run again
        gmls_obj.generateAlphas(1, True)
        self.assertEqual(gmls_obj.containsValidAlphas(), True)

        # now it works
        [check_answer(gmls_helper, i) for i in range(4)]

        # now pickle to a file
        with open('test.p', 'wb') as fn:
            pickle.dump(gmls_helper, fn)
        del gmls_helper

        with open('test.p', 'rb') as fn:
            new_gmls_helper = pickle.load(fn)
        # should throw an error because GMLS object is not set
        with self.assertRaises(RuntimeError):
            new_gmls_helper.getGMLSObject()

        # reuse solution computed from gmls_obj with gmls_helper 
        # loaded from pickle
        new_gmls_helper.setGMLSObject(gmls_obj)
        self.assertEqual(gmls_obj.containsValidAlphas(), True)
        [check_answer(new_gmls_helper, i) for i in range(4)]

    # end of inline tests

# space / sampling combinations
space_sample_combos = {
        "stp_ps":(pycompadre.ReconstructionSpace.ScalarTaylorPolynomial, pycompadre.SamplingFunctionals["PointSample"]), 
        "vsctp_vps":(pycompadre.ReconstructionSpace.VectorOfScalarClonesTaylorPolynomial, pycompadre.SamplingFunctionals["VectorPointSample"]),
        #"brnst_vps":(pycompadre.ReconstructionSpace.BernsteinPolynomial, pycompadre.SamplingFunctionals["VectorPointSample"])
        "brnst_vps":(pycompadre.ReconstructionSpace.BernsteinPolynomial, pycompadre.SamplingFunctionals["PointSample"])
        }


#############################

#
# Begin Most Generic Tests
#

# generic test that can handle all options
def base_test(self,polyOrder,dimension,additional_sites=False,epsilon_multiplier=1.5,reconstruction_space=pycompadre.ReconstructionSpace.VectorOfScalarClonesTaylorPolynomial,sampling_functional=pycompadre.SamplingFunctionals["VectorPointSample"]):
    copy_kwargs = locals()
    del copy_kwargs['self']
    out = remap(**copy_kwargs)
    l2 = out[0]
    h1 = out[1]
    #assert False, str(l2)
    #assert False, str(h1)
    tol = 1e-13*np.power(10,polyOrder)
    self.assertTrue(l2<tol)
    self.assertTrue(h1<tol)
    if (len(out)>2):
        l2a = out[2]
        self.assertTrue(l2a<tol)

# combine order, dim, etc.... to form all possible functions to test
for polyorder in (1,2,3):
    for dim in (1,2,3):
        for space_sample_combo in space_sample_combos:
            for additional_sites in (False, True):
                # add these member functions to TestPyCOMPADRE
                setattr(TestPyCOMPADRE,"test_%dd_order%d_a%s_"%(dim,polyorder,str(additional_sites))+str(space_sample_combo),partialmethod(base_test, polyOrder=polyorder, dimension=dim, additional_sites=additional_sites, epsilon_multiplier=1.5, reconstruction_space=space_sample_combos[space_sample_combo][0], sampling_functional=space_sample_combos[space_sample_combo][1]))
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
[setattr(TestPyCOMPADRE,"test_square_qr_"+str(key),partialmethod(test_square_qr, space_sample_combo=space_sample_combos[key])) for key in space_sample_combos]
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
