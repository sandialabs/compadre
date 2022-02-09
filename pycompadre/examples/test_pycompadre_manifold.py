from kokkos_test_case import KokkosTestCase
import pycompadre
import numpy as np

# get GMLS approximate at all x_pred, as well as reconstruction about attempt_center_about_coord
def approximate(input_dimensions, porder, wpower, wtype, epsilon_multiplier, attempt_center_about_coord):

    global xy, xy_pred, z
    gmls_obj=pycompadre.GMLS(porder, input_dimensions, "QR", "MANIFOLD")
    gmls_obj.setWeightingParameter(wpower)
    gmls_obj.setWeightingType(wtype)
    gmls_helper = pycompadre.ParticleHelper(gmls_obj)
    gmls_helper.generateKDTree(xy)
    gmls_obj.addTargets(pycompadre.TargetOperation.ScalarPointEvaluation)

    # one less dimension because it is a manifold problem
    gmls_helper.generateNeighborListsFromKNNSearchAndSet(xy_pred, porder, input_dimensions-1, epsilon_multiplier)
    gmls_obj.generateAlphas(1, False)
    
    # helper function for applying of alphas
    z_pred = gmls_helper.applyStencil(z, pycompadre.TargetOperation.ScalarPointEvaluation)

    # tests that setting and getting tangent bundle works
    gmls_helper.setTangentBundle(np.ones(shape=(xy_pred.shape[0], input_dimensions, input_dimensions), dtype='f8'))
    tb = gmls_helper.getTangentBundle()
    
    del gmls_obj
    del gmls_helper
    return z_pred

class TestPycompadreManifold(KokkosTestCase):

    def test_1d_order2(self):

        # initialize parameters
        polynomial_order = 5
        input_dimensions = 2
        epsilon_multiplier = 1.6
        weighting_type = 'power'
        weighting_power = 3

        # functions for y(x) and z(x,y(x))
        y_function = lambda x: np.sin(x)
        function = lambda x,y: x*x+y*y

        global xy_pred
        x_pred = np.linspace(0,4,200)
        y_pred = y_function(x_pred)
        xy_pred = np.vstack((x_pred,y_pred)).T

        last_norm_diff = 0
        for i in range(5):
            # set up data
            num_data_points = 10*pow(2,i)
            x = np.linspace(0,4,num_data_points)
            y = y_function(x)
            global xy, z
            xy = np.vstack((x,y)).T
            z = function(x,y)
            z_pred = approximate(input_dimensions, polynomial_order, weighting_power, weighting_type, epsilon_multiplier, epsilon_multiplier)
            diff = z_pred-function(x_pred,y_pred)
            norm_diff = np.linalg.norm(diff)
            self.assertTrue(i==0 or np.log(norm_diff/last_norm_diff)/np.log(0.5)>polynomial_order)
            last_norm_diff = norm_diff

#tc = TestPycompadreManifold()
#tc.test_1d_order2()

if __name__ == '__main__':
    import unittest
    unittest.main()
