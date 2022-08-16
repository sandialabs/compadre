from kokkos_test_case import KokkosTestCase
import pycompadre
from netCDF4 import Dataset
import numpy as np
import scipy.io
import scipy.sparse
import scipy.sparse.linalg

'''

Creation of variational mass matrix for use in Hodge star definition

Requires a .nc file that contains all of the fields referenced in this file.

'''
class TestSphereRemapCellIntegral(KokkosTestCase):

    def test_cell_integrated_remap(self):

        def run(level):

            # will need to have cell integral DOFs to point values (quadrature on cells)
            # then sum over quadrature to build up entries

            dataset = Dataset('../../../test_data/grids/sphere_{0}.nc'.format(str(level)), "r", format="NETCDF4")

            THREE = 3 # number of vertices in a triangle
            radius = 6371220.00
            dimensions = dataset.dimensions
            variables = dataset.variables

            p_order = 2
            # get cells and put averaged quantities on them
            q_order = 4

            DIM = 3
            assert DIM==3, "Only created for 3D problem with 2D manifolds (DIM==3)"
            Q_DIM   = 2 # local manifold dimension

            qp = pycompadre.Quadrature(q_order, Q_DIM, "TRI")

            # make a triangle from this midpoint, previous midpoint, and centroid
            qpoints = qp.getSites()
            qweights = qp.getWeights()

            nEOnC = dataset['nEdgesOnCell']
            max_nEOnC = np.max(nEOnC)
            #min_nEOnC = np.min(nEOnC)
            #print(max_nEOnC, min_nEOnC)
            #assert max_nEOnC==min_nEOnC, "Max and min edges on cells are not the same"

            # loop over quantities
            # get number of cells
            vOnC = dataset['verticesOnCell']
            vOnE = dataset['verticesOnEdge']
            eOnC = dataset['edgesOnCell']
            xC = dataset['xCell']
            yC = dataset['yCell']
            zC = dataset['zCell']
            xV = dataset['xVertex']
            yV = dataset['yVertex']
            zV = dataset['zVertex']
            nCells = dataset.dimensions['nCells'].size
            extra_data = np.zeros(shape=(nCells, DIM*max_nEOnC), dtype='f8')
            cell_points = np.zeros(shape=(nCells, DIM), dtype='f8')
            q_scaling = np.zeros(shape=(nCells, max_nEOnC, len(qpoints)), dtype='f8')

            # need to produce data for the remap (get cell integrated data)
            f = lambda x: (x[0]/radius)**2 - (x[1]/radius)**2 + (x[1]/radius)*(x[0]/radius) \
                           + 55.0 + (x[2]/radius)**2 - (x[0]/radius)*(x[2]/radius)

            v = np.zeros(shape=(DIM,3), dtype='f8') # 3 is from triangle nodes
            exact_in_field = np.zeros(shape=(nCells), dtype='f8')
            computed_total_area = 0.0
            ref_total_area = 0.0
            for cell in range(nCells):
                cell_points[cell,:] = np.asarray([xC[cell], yC[cell], zC[cell]], dtype='f8')

                # normalization to sphere
                cell_points[cell,:] = cell_points[cell,:] * radius / np.linalg.norm(cell_points[cell,:])

                cell_area = 0.0
                facet_cell_area = 0.0
                ref_total_area += dataset['areaCell'][cell]

                for local_edge in range(max_nEOnC):
                #for local_edge in range(dataset['nEdgesOnCell'][cell]):
                    if local_edge < dataset['nEdgesOnCell'][cell]:
                        local_v1 = (local_edge - 1) % dataset['nEdgesOnCell'][cell]
                        local_v2 = local_edge
                        v1 = vOnC[cell, local_v1] - 1
                        v2 = vOnC[cell, local_v2] - 1

                        # for debugging we can check that v1 or v2 are in the edge
                        e  = eOnC[cell, local_edge] - 1
                        e_v1 = vOnE[e, 0]-1
                        e_v2 = vOnE[e, 1]-1
                        assert (v1==e_v1 or v1==e_v2), "Cell %d, not found %d in [%d,%d]" % (cell, v1, e_v1, e_v2)
                        assert (v2==e_v1 or v2==e_v2), "Cell %d, not found %d in [%d,%d]" % (cell, v2, e_v1, e_v2)

                        vertex_num = v1
                        # write first vertex on edge to extra_data
                        extra_data[cell,local_edge*DIM:(local_edge+1)*DIM] = \
                            np.asarray([xV[vertex_num], yV[vertex_num], zV[vertex_num]], dtype='f8')
                        extra_data[cell,local_edge*DIM:(local_edge+1)*DIM] = extra_data[cell,local_edge*DIM:(local_edge+1)*DIM] * radius / np.linalg.norm(extra_data[cell,local_edge*DIM:(local_edge+1)*DIM])
                    else:
                        extra_data[cell,local_edge*DIM:(local_edge+1)*DIM] = \
                            np.asarray([np.NaN, np.NaN, np.NaN], dtype='f8')

                #for local_vertex in range(max_nEOnC):
                for local_vertex in range(dataset['nEdgesOnCell'][cell]):
                    v[:,0] = cell_points[cell,:].copy()
                    alt_local_vertex = (local_vertex-1) % dataset['nEdgesOnCell'][cell]
                    v[:,1] = extra_data[cell,alt_local_vertex*DIM:(alt_local_vertex+1)*DIM] - v[:,0]
                    v[:,2] = extra_data[cell,local_vertex*DIM:(local_vertex+1)*DIM] - v[:,0]
                    facet_cell_area += 0.5*np.linalg.norm(np.cross(v[:,1], v[:,2]))

                    result = 0.0
                    for i in range(len(qpoints)):

                        unscaled_transformed_qp = v[:,0].copy()
                        for j in range(DIM-1):
                            unscaled_transformed_qp += qpoints[i][j] * v[:,j+1]
                        transformed_qp_norm = np.linalg.norm(unscaled_transformed_qp)
                        scaled_transformed_qp = unscaled_transformed_qp.copy() * radius / transformed_qp_norm

                        # u_qp = midpoint + r_qp[1]*(v_1-midpoint) + r_qp[2]*(v_2-midpoint)
                        # s_qp = u_qp * radius / norm(u_qp) = radius * u_qp / norm(u_qp)
                        #
                        # so G(:,i) is \partial{s_qp}/ \partial{r_qp[i]}
                        # where r_qp is reference quadrature point (R^2 in 2D manifold in R^3)
                        #
                        # G(:,i) = radius * ( \partial{u_qp}/\partial{r_qp[i]} * (\sum_m u_qp[k]^2)^{-1/2}
                        #          + u_qp * \partial{(\sum_m u_qp[k]^2)^{-1/2}}/\partial{r_qp[i]} )
                        #
                        #        = radius * ( T(:,i)/norm(u_qp) + u_qp*(-1/2)*(\sum_m u_qp[k]^2)^{-3/2}
                        #                              *2*(\sum_k u_qp[k]*\partial{u_qp[k]}/\partial{r_qp[i]}) )
                        #
                        #        = radius * ( T(:,i)/norm(u_qp) + u_qp*(-1/2)*(\sum_m u_qp[k]^2)^{-3/2}
                        #                              *2*(\sum_k u_qp[k]*T(k,i)) )
                        #
                        qp_norm_sq = transformed_qp_norm**2
                        G = v.copy() / transformed_qp_norm
                        for k in range(DIM):
                            G[:,1] += unscaled_transformed_qp*(-0.5)*pow(qp_norm_sq,-1.5) \
                                      *2*(unscaled_transformed_qp[k]*v[k,1]);
                            G[:,2] += unscaled_transformed_qp*(-0.5)*pow(qp_norm_sq,-1.5) \
                                      *2*(unscaled_transformed_qp[k]*v[k,2]);

                        # cross product gives stretch factor
                        scaling = radius**2 * np.linalg.norm(np.cross(G[:,1], G[:,2]))
                        result += f(scaled_transformed_qp)*qweights[i]*scaling
                        q_scaling[cell, local_vertex, i] = qweights[i]*scaling
                        cell_area += qweights[i]*scaling

                    exact_in_field[cell] += result
                computed_total_area += cell_area

            print("AREA", computed_total_area, " vs ", ref_total_area)
            #print("SPHERE AREA:",str(4.0*3.141592653*(radius**2)))

            exact_out_field = np.zeros(shape=(nCells), dtype='f8')
            for cell in range(nCells):
                exact_out_field[cell] = f(cell_points[cell,:])

            # gmls object for mapping cell integrated DOFs to point values at cell centers
            # quadrature will be added as extra evaluation sites
            # related neighbor lists will also be cells that are neighbors of cells

            #gmls_obj=pycompadre.GMLS(pycompadre.ReconstructionSpace.ScalarTaylorPolynomial, 
            #gmls_obj=pycompadre.GMLS(pycompadre.ReconstructionSpace.VectorTaylorPolynomial, 
            gmls_obj=pycompadre.GMLS(pycompadre.ReconstructionSpace.ScalarTaylorPolynomial,
                                     pycompadre.SamplingFunctionals['ScalarFaceIntegralSample'], 
                                     pycompadre.SamplingFunctionals['PointSample'],
                                     p_order, 
                                     DIM, 
                                     "QR", 
                                     "MANIFOLD",
                                     "NO_CONSTRAINT",
                                     p_order)

            gmls_obj.setOrderOfQuadraturePoints(q_order)
            gmls_obj.setDimensionOfQuadraturePoints(Q_DIM)
            gmls_obj.setQuadratureType("TRI")
            gmls_helper = pycompadre.ParticleHelper(gmls_obj)
            gmls_helper.generateKDTree(cell_points)
            gmls_obj.addTargets(pycompadre.TargetOperation.ScalarPointEvaluation)
            gmls_obj.setTargetExtraData(extra_data)
            gmls_obj.setSourceExtraData(extra_data)

            gmls_helper.generateNeighborListsFromKNNSearchAndSet(cell_points, p_order, DIM-1, 2.2)

            # centroids are cell centroids
            # for now, the DOFs also exist at cell centroids
            cell_to_dof_nl = gmls_helper.getNeighborLists()
            dof_to_cell_nl = cell_to_dof_nl

            # loop 
            dof_to_dof_nl  = list()
            nDOFs = nCells
            for dof in range(nDOFs):
                dof_to_dof_nl.append(set())
                # loop cell neighbors of DOF
                for j in range(dof_to_cell_nl.getNumberOfNeighbors(dof)):
                    cell = dof_to_cell_nl.getNeighbor(dof, j)
                    # loop DOF neighbors of cell
                    for k in range(cell_to_dof_nl.getNumberOfNeighbors(cell)):
                        dof_to_dof_nl[dof].add(cell_to_dof_nl.getNeighbor(cell, k))

            #print(dof_to_dof_nl)

            # create a map from (cell(int), edge(int), quadrature(int), exterior(bool)) -> extra site number (int)
            extra_points_cell_map = np.zeros(shape=(cell_to_dof_nl.getNumberOfTargets(), max_nEOnC, len(qpoints), 1), dtype=np.int32)
            # create a tall skinny matrix of quadrature coordinates from which to select for extra_point_indices
            extra_points_coordinates = np.zeros(shape=(cell_to_dof_nl.getNumberOfTargets()*max_nEOnC*len(qpoints), DIM), dtype=np.float64)
            # refer to indices of quadrature coordinates in extra_points_coordinates
            # first column indicates # of extra evaluation sites for that cell
            extra_points_indices = np.zeros(shape=(cell_to_dof_nl.getNumberOfTargets(),max_nEOnC*len(qpoints)+1), dtype=np.int32)
            i = 0

            # loops cells
            for cell in range(nCells):

                # keep track of column for extra_points_indices
                local_entries = 0

                # loops edges
                for local_vertex in range(dataset['nEdgesOnCell'][cell]):

                    # needed for quadrature point transformation
                    v[:,0] = cell_points[cell,:].copy()
                    alt_local_vertex = (local_vertex-1) % dataset['nEdgesOnCell'][cell]
                    v[:,1] = extra_data[cell,alt_local_vertex*DIM:(alt_local_vertex+1)*DIM] - v[:,0]
                    v[:,2] = extra_data[cell,local_vertex*DIM:(local_vertex+1)*DIM] - v[:,0]

                    # loops quadrature
                    for q in range(len(qpoints)):

                        extra_points_cell_map[cell, local_vertex, q, 0] = local_entries

                        unscaled_transformed_qp = v[:,0].copy()
                        for j in range(DIM-1):
                            unscaled_transformed_qp += qpoints[q][j] * v[:,j+1]
                        transformed_qp_norm = np.linalg.norm(unscaled_transformed_qp)
                        scaled_transformed_qp = unscaled_transformed_qp.copy() * radius / transformed_qp_norm

                        extra_points_coordinates[i, :] = scaled_transformed_qp[:]
                        extra_points_indices[cell, local_entries+1] = i

                        i+=1 # keeps increasing over cell loops
                        local_entries+=1 # resets at cell loops

                extra_points_indices[cell, 0] = local_entries

            # TEST
            # pick a cell, then print all quadrature for that cell, also do it by subcell (side)

            cell = 5
            edge = 2

            cell_point = np.asarray([xC[cell], yC[cell], zC[cell]], dtype='f8')

            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            # plot cell center
            ax.scatter(cell_point[0], cell_point[1], cell_point[2],c='#000000')

            # get edge vertices
            for local_edge in range(dataset['nEdgesOnCell'][cell]):
                #if local_edge==edge:
                local_v1 = (local_edge - 1) % dataset['nEdgesOnCell'][cell]
                local_v2 = local_edge
                v1 = vOnC[cell, local_v1] - 1
                v2 = vOnC[cell, local_v2] - 1

                # for debugging we can check that v1 or v2 are in the edge
                e  = eOnC[cell, local_edge] - 1
                e_v1 = vOnE[e, 0]-1
                e_v2 = vOnE[e, 1]-1
                assert (v1==e_v1 or v1==e_v2), "Cell %d, not found %d in [%d,%d]" % (cell, v1, e_v1, e_v2)
                assert (v2==e_v1 or v2==e_v2), "Cell %d, not found %d in [%d,%d]" % (cell, v2, e_v1, e_v2)
                v1_coords = \
                    np.asarray([xV[v1], yV[v1], zV[v1]], dtype='f8')
                v2_coords = \
                    np.asarray([xV[v2], yV[v2], zV[v2]], dtype='f8')
                # plot edge vertices
                #print(v1_coords, v2_coords)
                ax.scatter(v1_coords[0], v1_coords[1], v1_coords[2],c='#ff0000')
                ax.scatter(v2_coords[0], v2_coords[1], v2_coords[2],c='#ff0000')

            # plot quadrature
            for k in range(extra_points_indices[cell, 0]):
                pt = extra_points_coordinates[extra_points_indices[cell, k+1],:]
                ax.scatter(pt[0], pt[1], pt[2],c='#00ff00')
                #print(extra_points_coordinates[extra_points_indices[cell, k+1],:])

            #plt.show()
            #exit()


            # set up additional evaluation sites
            gmls_helper.setAdditionalEvaluationSitesData(extra_points_indices, extra_points_coordinates)
            gmls_obj.generateAlphas(1, False)
            ss = gmls_obj.getSolutionSet()
            print("getSS")

            # loop over cells
            cell_max_neighbors = cell_to_dof_nl.getMaxNumNeighbors()
            I = np.zeros(shape=(nCells, max_nEOnC, cell_max_neighbors, cell_max_neighbors), dtype=np.int32)
            J = np.zeros(shape=(nCells, max_nEOnC, cell_max_neighbors, cell_max_neighbors), dtype=np.int32)
            K = np.zeros(shape=(nCells, max_nEOnC, cell_max_neighbors, cell_max_neighbors), dtype=np.float64)
            print("getIJK")
            for cell in range(nCells):
                # loop over DOFs that are neighbors of the cell i
                for local_vertex in range(dataset['nEdgesOnCell'][cell]):
                    for j in range(cell_to_dof_nl.getNumberOfNeighbors(cell)):
                        # loop over DOFs that are neighbors of the cell i
                        for k in range(cell_to_dof_nl.getNumberOfNeighbors(cell)):
                            print("%d of %d\n"%(cell, nCells))
                            # loop over quadrature on cell
                            contribution = 0
                            for q in range(len(qpoints)):
                                u = ss.getAlpha(lro=pycompadre.TargetOperation.ScalarPointEvaluation, target_index=cell, output_component_axis_1=0, output_component_axis_2=0, neighbor_index=j, input_component_axis_1=0, input_component_axis_2=0, additional_evaluation_site=extra_points_cell_map[cell,local_vertex,q,0]+1)
                                v = ss.getAlpha(lro=pycompadre.TargetOperation.ScalarPointEvaluation, target_index=cell, output_component_axis_1=0, output_component_axis_2=0, neighbor_index=k, input_component_axis_1=0, input_component_axis_2=0, additional_evaluation_site=extra_points_cell_map[cell,local_vertex,q,0]+1)
                                alt_u = ss.getAlpha(lro=pycompadre.TargetOperation.ScalarPointEvaluation, target_index=cell, output_component_axis_1=0, output_component_axis_2=0, neighbor_index=j, input_component_axis_1=0, input_component_axis_2=0, additional_evaluation_site=0)
                                alt_v = ss.getAlpha(lro=pycompadre.TargetOperation.ScalarPointEvaluation, target_index=cell, output_component_axis_1=0, output_component_axis_2=0, neighbor_index=k, input_component_axis_1=0, input_component_axis_2=0, additional_evaluation_site=0)
                                contribution += q_scaling[cell, local_vertex, q] * u * v
                            I[cell, local_vertex, j, k] = cell_to_dof_nl.getNeighbor(cell, j)
                            J[cell, local_vertex, j, k] = cell_to_dof_nl.getNeighbor(cell, k)
                            K[cell, local_vertex, j, k] = contribution
            print("assembly done")
            I = I.flatten()
            J = J.flatten()
            K = K.flatten()

            #mass_matrix = scipy.sparse.coo_matrix(I,J,K)
            mass_matrix = scipy.sparse.coo_matrix( (K, (I,J)), shape=(dof_to_cell_nl.getNumberOfTargets(), dof_to_cell_nl.getNumberOfTargets()), dtype=np.float64)#.toarray()
            mass_matrix.sum_duplicates()
            print('condition of M:', scipy.sparse.linalg.norm(mass_matrix))


            print(mass_matrix)
            scipy.io.mmwrite('matrix.mtx', mass_matrix)
            print('norm:',str(scipy.sparse.linalg.norm(mass_matrix)))


            #!!## NOTE: From here on, handles whether solution is correct
            #!!#out_field = gmls_helper.applyStencil(exact_in_field, pycompadre.TargetOperation.ScalarPointEvaluation, sampling_functional=pycompadre.SamplingFunctionals['ScalarFaceIntegralSample'], evaluation_site_local_index=0)

            #!!#print('exact out:',exact_out_field)
            #!!#print('computed out:',out_field)
            #!!#print('diff:',out_field - exact_out_field)

            #print('exact_out',exact_out_field, 'exact_in',exact_in_field, 'computed_out',out_field)
            #if direction==1:
            #    # swap out field with in field
            #    exact_out_field = exact_in_field
            #print('out_field-exact_out_field',out_field-exact_out_field)

            ## calculate error norm (l2)
            ##rel_l2_error = np.linalg.norm(out_field-exact_out_field, axis=0)/np.sqrt(out_field.shape[0])
            #rel_l2_error = np.linalg.norm(out_field-exact_out_field, axis=0)/np.linalg.norm(exact_out_field)
            #print('Error norm:', rel_l2_error)
            #return rel_l2_error

        rel_l2_errors = []
        for i in range(args.grids):
            rel_l2_errors.append(run(i))
        print('cell integrals :: rel_l2_errors', rel_l2_errors)

        rel_l2_rates = []
        for i in range(1,len(rel_l2_errors)):
            rel_l2_rates.append(np.log(rel_l2_errors[i]/rel_l2_errors[i-1])/np.log(0.5))
        print('cell integrals :: rel_l2_rates', rel_l2_rates)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='test remap on the sphere')
    parser.add_argument('-g','--grids', dest='grids', type=int, default=1, help='number of grids for refinement sequence')
    #parser.add_argument('-l','--list', nargs='*', help='{0: cell-integrated, 1: edge-integrated}', default=['0','1'], required=False)
    parser.add_argument('-l','--list', nargs='*', help='{0: cell-integrated, 1: edge-integrated}', default=['0',], required=False)
    args = parser.parse_args()

    # remove tests not requested
    if '0' not in args.list:
        TestSphereRemapCellIntegral = None
    if '1' not in args.list:
        TestSphereRemapEdgeIntegral = None

    import sys
    sys.argv = sys.argv[0:1]

    import unittest
    unittest.main()
