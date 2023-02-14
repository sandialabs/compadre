from kokkos_test_case import KokkosTestCase
import pycompadre
import sys
import numpy as np
#np.set_printoptions(threshold=sys.maxsize)

class TestPycompadreTangentsOnManifold(KokkosTestCase):

    def test_sphere(self):

        # initialize parameters
        polynomial_order = 4
        input_dimensions = 3
        epsilon_multiplier = 1.9
        weighting_type = 'power'
        weighting_power = 3

        number_target_coords = 1000
        target_coords = np.ndarray(shape=(1000,3), dtype='f8')
        r = 1.0
        enough_pts_found = False
        N_count = 0
        a = 4.0*np.pi*r*r/float(number_target_coords)
        d = np.sqrt(a)
        M_theta = int(np.round(np.pi/d))
        d_theta = np.pi/float(M_theta)
        d_phi = a/d_theta
        for i in range(M_theta):
            theta = np.pi*(i + 0.5)/M_theta
            M_phi = int(np.ceil(2.0*np.pi*np.sin(theta)/d_phi))
            for j in range(M_phi):
                phi = 2.0*np.pi*j/M_phi;
                target_coords[N_count, 0] = theta
                target_coords[N_count, 1] = phi
                N_count+=1
                if N_count == number_target_coords:
                    enough_pts_found = True
                    break
            if enough_pts_found:
                break

        target_coords = target_coords[0:N_count,:]
        for i in range(N_count):
            theta = target_coords[i,0]
            phi = target_coords[i,1]
            target_coords[i,0] = r*np.sin(theta)*np.cos(phi);
            target_coords[i,1] = r*np.sin(theta)*np.sin(phi);
            target_coords[i,2] = r*np.cos(theta);

        # filter bottom half out
        target_coords = target_coords[target_coords[:,2]>=0.0, :]



        ## visualize top half of sphere sampled
        #import matplotlib.pyplot as plt
        #fig = plt.figure()
        #ax = fig.add_subplot(projection='3d')
        #ax.scatter(target_coords[:,0], target_coords[:,1], target_coords[:,2], marker='o')
        #plt.show()


        # copy coordinates as data
        data = np.copy(target_coords)
        source_coords = np.copy(target_coords)

        # ORIGINAL reconstruction that gets tangent vector reconstruction (projects off normal direction in data)
        gmls_obj_vec = pycompadre.GMLS(
                                       pycompadre.ReconstructionSpace.VectorOfScalarClonesTaylorPolynomial, 
                                       pycompadre.SamplingFunctionals["ManifoldVectorPointSample"],
                                       polynomial_order, input_dimensions,
                                       "QR", "MANIFOLD")
        gmls_obj_vec.setWeightingParameter(weighting_power)
        gmls_obj_vec.setWeightingType(weighting_type)
        gmls_helper_vec = pycompadre.ParticleHelper(gmls_obj_vec)
        gmls_helper_vec.generateKDTree(source_coords)
        gmls_obj_vec.addTargets(pycompadre.TargetOperation.VectorPointEvaluation)
        gmls_helper_vec.generateNeighborListsFromKNNSearchAndSet(target_coords, polynomial_order, input_dimensions-1, epsilon_multiplier)
        gmls_helper_vec.setReferenceOutwardNormalDirection(data)
        gmls_obj_vec.generateAlphas(1, False)
        tbb = np.reshape(gmls_helper_vec.getTangentBundle(), newshape=(-1,3,3))
        v_pred = gmls_helper_vec.applyStencil(data, pycompadre.TargetOperation.VectorPointEvaluation, pycompadre.SamplingFunctionals["ManifoldVectorPointSample"])
        # note that normal direction of data is projected off

        # AUGMENTED reconstruction that picks up normal as a scalar function of two principal directions
        gmls_obj=pycompadre.GMLS(polynomial_order, input_dimensions, "QR", "MANIFOLD")
        gmls_obj.setWeightingParameter(weighting_power)
        gmls_obj.setWeightingType(weighting_type)
        gmls_helper = pycompadre.ParticleHelper(gmls_obj)
        gmls_helper.generateKDTree(source_coords)
        gmls_obj.addTargets(pycompadre.TargetOperation.ScalarPointEvaluation)
        gmls_helper.generateNeighborListsFromKNNSearchAndSet(target_coords, polynomial_order, input_dimensions-1, epsilon_multiplier)
        # use tangent bundle from vector problem in scalar problem
        gmls_helper.setTangentBundle(tbb)
        gmls_helper.setReferenceOutwardNormalDirection(data)
        gmls_obj.generateAlphas(1, False)



        # TEST: get tangent bundle back for comparison (ensure no changes)
        tbc = np.reshape(gmls_helper.getTangentBundle(), newshape=(-1,3,3))
        assert np.linalg.norm(tbc-tbb)==0.0, "Error in setting / getting tangent directions"


        # can we reconstruct w/ normal direction cut off?
        x_data = np.zeros(shape=(data.shape[0],), dtype='f8')
        y_data = np.zeros(shape=(data.shape[0],), dtype='f8')
        n_data = np.zeros(shape=(data.shape[0],), dtype='f8')
        for i in range(data.shape[0]):
            x_data[i] = np.dot(data[i,:], tbb[i, 0, :])
            y_data[i] = np.dot(data[i,:], tbb[i, 1, :])
            n_data[i] = np.dot(data[i,:], tbb[i, 2, :])
        # reference solution of data with normal direction projected off
        ref_projected = np.multiply(x_data, tbc[:,0,:].T).T + np.multiply(y_data, tbc[:,2,:].T).T
        # reconstruction error of the vector problem in getting the tangent aspects correct
        #print("Norm in projected case: ", np.linalg.norm(ref_projected-v_pred))
        assert np.linalg.norm(ref_projected-v_pred) < 1e-1, "Failed to capture projected off solution accurately enough"

        # use scalar problem to pick up function time the normal direction
        n_pred = gmls_helper.applyStencil(n_data, pycompadre.TargetOperation.ScalarPointEvaluation)
        # add this quantity into the prediction
        v_pred += np.multiply(n_pred[:,None], tbc[:,2,:])
        # reconstruction error of original data in tangent + normal directions
        #print("Norm of original data: ", np.linalg.norm(data-v_pred))
        assert np.linalg.norm(data-v_pred) < 1e-4, "Failed to capture full solution accurately enough"

        # TODO: Add a test for data not tangent to manifold evaluated at point other than target site
        
        del gmls_obj_vec
        del gmls_helper_vec
        del gmls_obj
        del gmls_helper

if __name__ == '__main__':
    import unittest
    unittest.main()
