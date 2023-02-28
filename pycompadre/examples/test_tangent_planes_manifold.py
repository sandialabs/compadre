from kokkos_test_case import KokkosTestCase
import pycompadre
import sys
import numpy as np
#np.set_printoptions(threshold=sys.maxsize)

# Whether to test not just at the target sites, but also at sites h distance away
# from each target site
TEST_H_CLOSE_SITES = False
# Whether to use vector reconstruction for tangent portions and scalar
# reconstruction for normal portions or use scalar for everything
USE_VECTOR_PLUS_SCALAR_RECONSTRUCT = False

# https://github.com/MPAS-Dev/MPAS-Model/blob/master/src/operators/mpas_vector_operations.F
# Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
# and the University Corporation for Atmospheric Research (UCAR).

# Unless noted otherwise source code is licensed under the BSD license.
# Additional copyright and license information can be found in the LICENSE file
# distributed with this code, or at http://mpas-dev.github.com/license.html
def get_sphere_basis(lat, lon):
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    zonalUnitVector = np.zeros(shape=(3), dtype='f8')
    zonalUnitVector[0] = - sin_lon
    zonalUnitVector[1] =   cos_lon
    zonalUnitVector[2] =   0

    meridionalUnitVector = np.zeros(shape=(3), dtype='f8')
    meridionalUnitVector[0] = - sin_lat * cos_lon
    meridionalUnitVector[1] = - sin_lat * sin_lon
    meridionalUnitVector[2] =   cos_lat

    verticalUnitVector = np.zeros(shape=(3), dtype='f8')
    verticalUnitVector[0] = cos_lat * cos_lon
    verticalUnitVector[1] = cos_lat * sin_lon
    verticalUnitVector[2] = sin_lat

    return (zonalUnitVector, meridionalUnitVector, verticalUnitVector)

def atan4(y, x):
    result = 0.0
    PI = np.pi
    if ( x == 0.0 ):
        if ( y > 0.0 ) :
            result = 0.5 * PI
        elif ( y < 0.0 ):
            result = 1.5 * PI
        elif ( y == 0.0 ):
            result = 0.0
    elif ( y == 0 ):
        if ( x > 0.0 ):
            result = 0.0
        elif ( x < 0.0 ):
            result = PI
    else:
        theta = np.arctan2( abs(y), abs(x) )
        if ( x > 0.0 and y > 0.0 ):
            result = theta
        elif ( x < 0.0 and y > 0.0 ):
            result = PI - theta
        elif ( x < 0.0 and y < 0.0 ):
            result = PI + theta
        elif ( x > 0.0 and y < 0.0 ):
            result = 2.0 * PI - theta
    return result

def get_lat_lon(x):
    lat = np.arctan2(x[2], np.sqrt(x[0]*x[0] + x[1]*x[1]))
    lon = atan4(x[1], x[0])
    return (lat, lon)

class TestPycompadreTangentsOnManifold(KokkosTestCase):

    def test_sphere(self):

        # initialize parameters
        polynomial_order = 4
        input_dimensions = 3
        epsilon_multiplier = 1.9
        weighting_type = 'power'
        weighting_power = 3

        # increasing points by factor of 4 is roughly equivalent to refining mesh in 2D by factor of 2
        number_target_coords_list = [1000, 4000, 16000,]# 64000] # 1000, 4000, 16000, ...
        l2_errors_in_projected = np.zeros(shape=(len(number_target_coords_list)), dtype='f8')
        l2_errors_in_original  = np.zeros(shape=(len(number_target_coords_list)), dtype='f8')
        l2_errors_in_original_extra  = np.zeros(shape=(len(number_target_coords_list)), dtype='f8')
        for m, number_target_coords in enumerate(number_target_coords_list):
            target_coords = np.ndarray(shape=(number_target_coords,3), dtype='f8')
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
            N_count = target_coords.shape[0]


            # get reference tangent directions
            tba = np.zeros(shape=(target_coords.shape[0], 3, 3), dtype='f8')
            for i in range(target_coords.shape[0]):
                lat, lon = get_lat_lon(target_coords[i,:])
                xv, yv, nv = get_sphere_basis(lat, lon)
                tba[i,0,:] = xv
                tba[i,1,:] = yv
                tba[i,2,:] = nv


            # visualize top half of sphere sampled
            #import matplotlib.pyplot as plt
            #fig = plt.figure()
            #ax = fig.add_subplot(projection='3d')
            #ax.scatter(target_coords[:,0], target_coords[:,1], target_coords[:,2], marker='o')
            #plt.show()


            # copy coordinates as data
            data = np.copy(target_coords)
            source_coords = np.copy(target_coords)

            if TEST_H_CLOSE_SITES:
                # Test for data not tangent to manifold evaluated at point other than target site
                # using same data and same target sites, get the solution distance h away,
                # where h ~ sqrt(1/N_count). 
                # sample from h perturbation of target site in reference tangent directions
                # and then normalize to the sphere
                h = np.sqrt(1.0/target_coords.shape[0])
                h_close_target_coords = np.copy(target_coords)
                rand_dir_1 = np.random.rand(h_close_target_coords.shape[0])
                rand_dir_2 = np.random.rand(h_close_target_coords.shape[0])
                # perturb roughly by h * tangent directions (move them away from target site, 
                # but not too much and not in the normal direction which is off-manifold)
                h_close_target_coords += h*(np.multiply(rand_dir_1, tba[:,0,:].T).T + np.multiply(rand_dir_2, tba[:,1,:].T).T)
                # normal new data back onto the sphere (and also off of the tangent plane to the target site)
                h_close_target_coords = 1.0/np.linalg.norm(h_close_target_coords, axis=1)[:,None] * h_close_target_coords
                # set the index maps like [# extra eval sites (1), index of coordinate for that location from h_close_target_coords]
                extra_sites_idx = np.zeros(shape=(target_coords.shape[0],2), dtype='i4')
                extra_sites_idx[:,0] = 1
                extra_sites_idx[:,1] = range(h_close_target_coords.shape[0])


            # ORIGINAL reconstruction that gets tangent vector reconstruction (projects off normal direction in data)
            gmls_obj_vec = pycompadre.GMLS(
                                           pycompadre.ReconstructionSpace.VectorOfScalarClonesTaylorPolynomial, 
                                           pycompadre.SamplingFunctionals["ManifoldVectorPointSample"],
                                           polynomial_order, input_dimensions,
                                           "QR", "MANIFOLD", curvature_poly_order=polynomial_order)
            gmls_obj_vec.setWeightingParameter(weighting_power)
            gmls_obj_vec.setWeightingType(weighting_type)
            gmls_helper_vec = pycompadre.ParticleHelper(gmls_obj_vec)
            gmls_helper_vec.generateKDTree(source_coords)
            gmls_obj_vec.addTargets(pycompadre.TargetOperation.VectorPointEvaluation)
            gmls_helper_vec.generateNeighborListsFromKNNSearchAndSet(target_coords, polynomial_order, input_dimensions-1, epsilon_multiplier)
            #gmls_helper_vec.setTangentBundle(tba)
            gmls_helper_vec.setReferenceOutwardNormalDirection(data)
            if TEST_H_CLOSE_SITES:
                gmls_helper_vec.setAdditionalEvaluationSitesData(extra_sites_idx, h_close_target_coords)
            gmls_obj_vec.generateAlphas(1, False)
            tbb = np.reshape(gmls_helper_vec.getTangentBundle(), newshape=(-1,3,3))
            # note that normal direction of data is projected off
            v_pred = gmls_helper_vec.applyStencil(data, pycompadre.TargetOperation.VectorPointEvaluation, pycompadre.SamplingFunctionals["ManifoldVectorPointSample"])
            if TEST_H_CLOSE_SITES:
                v_pred_extra = gmls_helper_vec.applyStencil(data, pycompadre.TargetOperation.VectorPointEvaluation, pycompadre.SamplingFunctionals["ManifoldVectorPointSample"], evaluation_site_local_index=1)


            # AUGMENTED reconstruction that picks up normal as a scalar function of two principal directions
            gmls_obj=pycompadre.GMLS(polynomial_order, input_dimensions, "QR", "MANIFOLD", curvature_poly_order=polynomial_order)
            gmls_obj.setWeightingParameter(weighting_power)
            gmls_obj.setWeightingType(weighting_type)
            gmls_helper = pycompadre.ParticleHelper(gmls_obj)
            gmls_helper.generateKDTree(source_coords)
            gmls_obj.addTargets(pycompadre.TargetOperation.ScalarPointEvaluation)
            gmls_helper.generateNeighborListsFromKNNSearchAndSet(target_coords, polynomial_order, input_dimensions-1, epsilon_multiplier)
            # use tangent bundle from vector problem in scalar problem
            gmls_helper.setTangentBundle(tbb)
            gmls_helper.setReferenceOutwardNormalDirection(data)
            if TEST_H_CLOSE_SITES:
                gmls_helper.setAdditionalEvaluationSitesData(extra_sites_idx, h_close_target_coords)
            gmls_obj.generateAlphas(1, False)



            # TEST: get tangent bundle back for comparison (ensure no changes)
            tbc = np.reshape(gmls_helper.getTangentBundle(), newshape=(-1,3,3))
            assert np.linalg.norm(tbc-tbb)==0.0, "Error in setting / getting tangent directions"



            x_ref_data = np.zeros(shape=(data.shape[0],), dtype='f8')
            y_ref_data = np.zeros(shape=(data.shape[0],), dtype='f8')
            x_data = np.zeros(shape=(data.shape[0],), dtype='f8')
            y_data = np.zeros(shape=(data.shape[0],), dtype='f8')
            n_data = np.zeros(shape=(data.shape[0],), dtype='f8')
            for i in range(data.shape[0]):
                x_ref_data[i] = np.dot(data[i,:], tba[i, 0, :])
                y_ref_data[i] = np.dot(data[i,:], tba[i, 1, :])
                x_data[i] = np.dot(data[i,:], tbb[i, 0, :])
                y_data[i] = np.dot(data[i,:], tbb[i, 1, :])
                n_data[i] = np.dot(data[i,:], tbb[i, 2, :])


            if TEST_H_CLOSE_SITES:
                # keep copy of data for each target site, projected into differently
                # the reason is that when using the scalar reconstruction the data must
                # be projected down consistent with the tangent directions for that
                # target site. This varies from site to site, so must be done for each.
                x_data_full = np.zeros(shape=(data.shape[0],data.shape[0]), dtype='f8')
                y_data_full = np.zeros(shape=(data.shape[0],data.shape[0]), dtype='f8')
                n_data_full = np.zeros(shape=(data.shape[0],data.shape[0]), dtype='f8')
                # only project down data to tangent plane that will be used (neighbor data)
                nl = gmls_helper.getNeighborLists()
                for i in range(data.shape[0]):
                    for j in range(nl.getNumberOfNeighbors(i)):
                        x_data_full[i,nl.getNeighbor(i,j)] = np.dot(data[nl.getNeighbor(i,j),:], tbb[i, 0, :])
                        y_data_full[i,nl.getNeighbor(i,j)] = np.dot(data[nl.getNeighbor(i,j),:], tbb[i, 1, :])
                        n_data_full[i,nl.getNeighbor(i,j)] = np.dot(data[nl.getNeighbor(i,j),:], tbb[i, 2, :])


            # reference solution of data with normal direction projected off
            ref_projected = np.multiply(x_ref_data, tba[:,0,:].T).T + np.multiply(y_ref_data, tba[:,1,:].T).T
            # reconstruction error of the vector problem in getting the tangent aspects correct
            #print("Norm in projected case: ", np.linalg.norm(ref_projected-v_pred)/np.sqrt(N_count))
            l2_errors_in_projected[m] = np.linalg.norm(ref_projected-v_pred)/np.sqrt(N_count)
            


            # use scalar problem to pick up function time the normal direction
            if TEST_H_CLOSE_SITES:
                n_pred = np.zeros(shape=(data.shape[0],), dtype='f8')
                x_pred = np.zeros(shape=(data.shape[0],), dtype='f8')
                y_pred = np.zeros(shape=(data.shape[0],), dtype='f8')
                n_pred_extra = np.zeros(shape=(data.shape[0],), dtype='f8')
                x_pred_extra = np.zeros(shape=(data.shape[0],), dtype='f8')
                y_pred_extra = np.zeros(shape=(data.shape[0],), dtype='f8')
                for i in range(data.shape[0]):
                    t_n_pred_full = gmls_helper.applyStencil(n_data_full[i,:], pycompadre.TargetOperation.ScalarPointEvaluation)
                    t_x_pred_full = gmls_helper.applyStencil(x_data_full[i,:], pycompadre.TargetOperation.ScalarPointEvaluation)
                    t_y_pred_full = gmls_helper.applyStencil(y_data_full[i,:], pycompadre.TargetOperation.ScalarPointEvaluation)
                    t_n_pred_full_extra = gmls_helper.applyStencil(n_data_full[i,:], pycompadre.TargetOperation.ScalarPointEvaluation, evaluation_site_local_index=1)
                    t_x_pred_full_extra = gmls_helper.applyStencil(x_data_full[i,:], pycompadre.TargetOperation.ScalarPointEvaluation, evaluation_site_local_index=1)
                    t_y_pred_full_extra = gmls_helper.applyStencil(y_data_full[i,:], pycompadre.TargetOperation.ScalarPointEvaluation, evaluation_site_local_index=1)
                    n_pred[i] = t_n_pred_full[i]
                    x_pred[i] = t_x_pred_full[i]
                    y_pred[i] = t_y_pred_full[i]
                    n_pred_extra[i] = t_n_pred_full_extra[i]
                    x_pred_extra[i] = t_x_pred_full_extra[i]
                    y_pred_extra[i] = t_y_pred_full_extra[i]
            else:
                n_pred = gmls_helper.applyStencil(n_data, pycompadre.TargetOperation.ScalarPointEvaluation)
                x_pred = gmls_helper.applyStencil(x_data, pycompadre.TargetOperation.ScalarPointEvaluation)
                y_pred = gmls_helper.applyStencil(y_data, pycompadre.TargetOperation.ScalarPointEvaluation)



            if USE_VECTOR_PLUS_SCALAR_RECONSTRUCT:
                v_pred += np.multiply(n_pred[:,None], tbb[:,2,:])
            else:
                # add this quantity into the prediction
                v_pred  = np.multiply(n_pred[:,None], tbb[:,2,:])
                v_pred += np.multiply(x_pred[:,None], tbb[:,0,:])
                v_pred += np.multiply(y_pred[:,None], tbb[:,1,:])
            # still accurate ^-- because it keeps track of t1 and t2 direction
            # reconstruction error of original data in tangent + normal directions
            l2_errors_in_original[m] = np.linalg.norm(data-v_pred)/np.sqrt(N_count)


            if TEST_H_CLOSE_SITES:
                if USE_VECTOR_PLUS_SCALAR_RECONSTRUCT:
                    v_pred_extra += np.multiply(n_pred_extra[:,None], tbb[:,2,:])
                else:
                    v_pred_extra  = np.multiply(n_pred_extra[:,None], tbb[:,2,:])
                    v_pred_extra += np.multiply(x_pred_extra[:,None], tbb[:,0,:])
                    v_pred_extra += np.multiply(y_pred_extra[:,None], tbb[:,1,:])
                l2_errors_in_original_extra[m] = np.linalg.norm(h_close_target_coords-v_pred_extra)/np.sqrt(N_count)

            
            del gmls_obj_vec
            del gmls_helper_vec
            del gmls_obj
            del gmls_helper


        PRINT_NORMS = True if __name__=='__main__' else False
        PRINT_NORMS and print("\n\nNorm of projected data recovery:")
        PRINT_NORMS and print(l2_errors_in_projected)
        PRINT_NORMS and print("Rates of convergence of projected data:")
        l2_rates_projected = np.zeros(shape=(l2_errors_in_projected.shape[0]-1), dtype='f8')
        for i in range(1,l2_errors_in_projected.shape[0]):
            l2_rates_projected[i-1] = np.log(l2_errors_in_projected[i]/l2_errors_in_projected[i-1])/np.log(0.5)
        PRINT_NORMS and print(l2_rates_projected)
        assert np.all(l2_rates_projected>polynomial_order), \
                "Failed to capture projected off solution accurately enough"
        
        PRINT_NORMS and print("\n\nNorm of original data recovery:")
        PRINT_NORMS and print(l2_errors_in_original)
        PRINT_NORMS and print("Rates of convergence of original data:")
        l2_rates_original = np.zeros(shape=(l2_errors_in_original.shape[0]-1), dtype='f8')
        for i in range(1,l2_errors_in_original.shape[0]):
            l2_rates_original[i-1] = np.log(l2_errors_in_original[i]/l2_errors_in_original[i-1])/np.log(0.5)
        PRINT_NORMS and print(l2_rates_original)
        assert np.all(l2_rates_original>polynomial_order), \
            "Failed to capture original solution accurately enough"

        if TEST_H_CLOSE_SITES:
            PRINT_NORMS and print("\n\nNorm of original_extra data recovery:")
            PRINT_NORMS and print(l2_errors_in_original_extra)
            PRINT_NORMS and print("Rates of convergence of original_extra data:")
            l2_rates_original_extra = np.zeros(shape=(l2_errors_in_original_extra.shape[0]-1), dtype='f8')
            for i in range(1,l2_errors_in_original_extra.shape[0]):
                l2_rates_original_extra[i-1] = np.log(l2_errors_in_original_extra[i]/l2_errors_in_original_extra[i-1])/np.log(0.5)
            PRINT_NORMS and print(l2_rates_original_extra)
            assert np.all(l2_rates_original_extra>polynomial_order), \
                "Failed to capture original_extra solution accurately enough"

if __name__ == '__main__':
    import unittest
    unittest.main()
