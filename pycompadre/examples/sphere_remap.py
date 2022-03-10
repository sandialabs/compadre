from kokkos_test_case import KokkosTestCase
import pycompadre
from netCDF4 import Dataset
import numpy as np

'''

Demonstration of remap of cell-integrated quantities to point values on the sphere
with radius != 1.0.

Requires a .nc file that contains all of the fields referenced in this file.

'''
class TestSphereRemap(KokkosTestCase):

    def test_cell_integrated_remap(self):

        # initialize parameters
        polynomial_order = 5
        input_dimensions = 3
        epsilon_multiplier = 1.6
        weighting_type = 'power'
        weighting_power = 3

        dataset = Dataset('../../../test_data/grids/hodge_0.nc', "r", format="NETCDF4")

        radius = 6371220.00
        dimensions = dataset.dimensions
        variables = dataset.variables

        p_order = 2
        # get cells and put averaged quantities on them
        q_order = 2
        q_dim   = 2 # local manifold dimension
        qp = pycompadre.Quadrature(q_order, q_dim, "TRI")

        # make a triangle from this midpoint, previous midpoint, and centroid
        qpoints = qp.getSites()
        qweights = qp.getWeights()

        nEOnC = dataset['nEdgesOnCell']
        max_nEOnC = np.max(nEOnC)
        min_nEOnC = np.min(nEOnC)
        assert max_nEOnC==min_nEOnC, "Max and min edges on cells are not the same"

        dim = 3

        # loop over quantities
        # get number of cells
        vOnC = dataset['verticesOnCell']
        xC = dataset['xCell']
        yC = dataset['yCell']
        zC = dataset['zCell']
        xV = dataset['xVertex']
        yV = dataset['yVertex']
        zV = dataset['zVertex']
        nCells = dataset.dimensions['nCells'].size
        extra_data = np.zeros(shape=(nCells, dim*max_nEOnC), dtype='f8')
        cell_points = np.zeros(shape=(nCells, dim), dtype='f8')

        # need to produce data for the remap (get cell integrated data)
        f = lambda x: (x[0]/radius)**2 - (x[1]/radius)**2 + (x[1]/radius)*(x[0]/radius) + 55.0 + (x[2]/radius)**2 - (x[0]/radius)*(x[2]/radius)

        # directions 
        v = np.zeros(shape=(dim,3), dtype='f8') # 3 is from triangle nodes
        in_field = np.zeros(shape=(nCells), dtype='f8')
        computed_total_area = 0.0
        ref_total_area = 0.0
        for cell in range(nCells):
            cell_points[cell,:] = np.asarray([xC[cell], yC[cell], zC[cell]], dtype='f8')
            cell_area = 0.0
            facet_cell_area = 0.0
            ref_total_area += dataset['areaCell'][cell]
            for local_vertex in range(max_nEOnC):
                vertex_num = vOnC[cell, local_vertex]-1
                extra_data[cell,local_vertex*dim:(local_vertex+1)*dim] = np.asarray([xV[vertex_num], yV[vertex_num], zV[vertex_num]], dtype='f8')

            for local_vertex in range(max_nEOnC):
                v[:,0] = cell_points[cell,:].copy()
                v[:,1] = extra_data[cell,local_vertex*dim:(local_vertex+1)*dim] - v[:,0]
                alt_local_vertex = (local_vertex+1) % max_nEOnC
                v[:,2] = extra_data[cell,alt_local_vertex*dim:(alt_local_vertex+1)*dim] - v[:,0]
                facet_cell_area += 0.5*np.linalg.norm(np.cross(v[:,1], v[:,2]))

                result = 0.0
                for i in range(len(qpoints)):

                    unscaled_transformed_qp = v[:,0].copy()
                    for j in range(dim-1):
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
                    for k in range(dim):
                        G[:,1] += unscaled_transformed_qp*(-0.5)*pow(qp_norm_sq,-1.5) \
                                  *2*(unscaled_transformed_qp[k]*v[k,1]);
                        G[:,2] += unscaled_transformed_qp*(-0.5)*pow(qp_norm_sq,-1.5) \
                                  *2*(unscaled_transformed_qp[k]*v[k,2]);

                    # cross product gives stretch factor
                    scaling = radius**2 * np.linalg.norm(np.cross(G[:,1], G[:,2]))
                    #print(scaling, np.linalg.norm(np.cross(v[:,1], v[:,2])))
                    #print(G)
                    result += f(scaled_transformed_qp)*qweights[i]*scaling
                    #print(f(scaled_transformed_qp),qweights[i],scaling)
                    cell_area += qweights[i]*scaling

                in_field[cell] += result
            #print(in_field[cell])
            #print(cell_area)
            #print(facet_cell_area, ref_cell_area)
            computed_total_area += cell_area

        print("AREA", computed_total_area, " vs ", ref_total_area)
        print("SPHERE AREA:",str(4.0*3.141592653*(radius**2)))
        #print(in_field)
        #print(cell_points)
        #gmls_obj=pycompadre.GMLS(pycompadre.ReconstructionSpace.VectorOfScalarClonesTaylorPolynomial, 
        gmls_obj=pycompadre.GMLS(pycompadre.ReconstructionSpace.ScalarTaylorPolynomial, 
                                 pycompadre.SamplingFunctionals['ScalarFaceIntegralSample'], 
                                 p_order, 
                                 dim, 
                                 "QR", 
                                 "MANIFOLD",
                                 "NO_CONSTRAINT",
                                 p_order)

        gmls_obj.setOrderOfQuadraturePoints(q_order)
        gmls_obj.setDimensionOfQuadraturePoints(q_dim)
        gmls_obj.setQuadratureType("TRI")
        gmls_helper = pycompadre.ParticleHelper(gmls_obj)
        gmls_helper.generateKDTree(cell_points)
        gmls_obj.addTargets(pycompadre.TargetOperation.ScalarPointEvaluation)
        gmls_obj.setTargetExtraData(extra_data)
        gmls_obj.setSourceExtraData(extra_data)

        gmls_helper.generateNeighborListsFromKNNSearchAndSet(cell_points, p_order, dim-1, 2.2)
        gmls_obj.generateAlphas(1, False)

        exact_out_field = np.zeros(shape=(nCells), dtype='f8')
        for cell in range(nCells):
            exact_out_field[cell] = f(cell_points[cell,:])
        out_field = gmls_helper.applyStencil(in_field, pycompadre.TargetOperation.ScalarPointEvaluation)
        print('exact out:',exact_out_field)
        print('computed out:',out_field)
        print('diff:',out_field - exact_out_field)

        # circle area should be 6.2886e-04 of whatever area of sphere is

        del gmls_helper, gmls_obj

    #def test_edge_integrated_remap(self):

    #    TWO = 2 # number of vertices on an edge

    #    # initialize parameters
    #    polynomial_order = 5
    #    input_dimensions = 3
    #    epsilon_multiplier = 1.6
    #    weighting_type = 'power'
    #    weighting_power = 3

    #    dataset = Dataset('../../../test_data/grids/hodge_0.nc', "r", format="NETCDF4")

    #    radius = 6371220.00
    #    dimensions = dataset.dimensions
    #    variables = dataset.variables

    #    p_order = 2
    #    # get cells and put averaged quantities on them
    #    q_order = 2
    #    q_dim   = 1 # local manifold dimension
    #    qp = pycompadre.Quadrature(q_order, q_dim, "LINE")

    #    # make a triangle from this midpoint, previous midpoint, and centroid
    #    qpoints = qp.getSites()
    #    qweights = qp.getWeights()

    #    dim = 3

    #    # loop over quantities
    #    # get number of cells
    #    vOnE = dataset['verticesOnEdge']
    #    xE = dataset['xEdge']
    #    yE = dataset['yEdge']
    #    zE = dataset['zEdge']
    #    xV = dataset['xVertex']
    #    yV = dataset['yVertex']
    #    zV = dataset['zVertex']
    #    latE = dataset['latEdge']
    #    lonE = dataset['lonEdge']
    #    nEdges = dataset.dimensions['nEdges'].size
    #    extra_data = np.zeros(shape=(nEdges, dim*TWO), dtype='f8')
    #    edge_points = np.zeros(shape=(nEdges, dim), dtype='f8')

    #    # need to produce data for the remap (get cell integrated data)
    #    f = lambda x: (x[0]/radius)**2 - (x[1]/radius)**2 + (x[1]/radius)*(x[0]/radius) + 55.0

    #    lat = 0.0
    #    lon = 0.0

    # https://github.com/MPAS-Dev/MPAS-Model/blob/master/src/operators/mpas_vector_operations.F
    # Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
    # and the University Corporation for Atmospheric Research (UCAR).
    #
    # Unless noted otherwise source code is licensed under the BSD license.
    # Additional copyright and license information can be found in the LICENSE file
    # distributed with this code, or at http://mpas-dev.github.com/license.html
    #
    #    sin_lat = np.sin(lat)
    #    cos_lat = np.cos(lat)
    #    sin_lon = np.sin(lon)
    #    cos_lon = np.cos(lon)

    #    zonalUnitVector = np.zeros(shape=(3), dtype='f8')
    #    zonalUnitVector[0] = - sin_lon
    #    zonalUnitVector[1] =   cos_lon
    #    zonalUnitVector[2] =   0

    #    meridionalUnitVector = np.zeros(shape=(3), dtype='f8')
    #    meridionalUnitVector[0] = - sin_lat * cos_lon
    #    meridionalUnitVector[1] = - sin_lat * sin_lon
    #    meridionalUnitVector[2] =   cos_lat

    #    verticalUnitVector = np.zeros(shape=(3), dtype='f8')
    #    verticalUnitVector[0] = cos_lat * cos_lon
    #    verticalUnitVector[1] = cos_lat * sin_lon
    #    verticalUnitVector[2] = sin_lat

    #    print(zonalUnitVector, meridionalUnitVector, verticalUnitVector)

    #    ## directions 
    #    #v = np.zeros(shape=(dim,3), dtype='f8') # 3 is from triangle nodes
    #    #in_field = np.zeros(shape=(nCells), dtype='f8')
    #    #total_area = 0.0
    #    #for cell in range(nCells):
    #    #    cell_points[cell,:] = np.asarray([xC[cell], yC[cell], zC[cell]], dtype='f8')
    #    #    cell_area = 0.0
    #    #    for local_vertex in range(max_nEOnC):
    #    #        vertex_num = vOnC[cell, local_vertex]-1
    #    #        extra_data[cell,local_vertex*dim:(local_vertex+1)*dim] = np.asarray([xV[vertex_num], yV[vertex_num], zV[vertex_num]], dtype='f8')

    #    #    for local_vertex in range(max_nEOnC):
    #    #        v[:,0] = cell_points[cell,:]
    #    #        v[:,1] = extra_data[cell,local_vertex*dim:(local_vertex+1)*dim] - v[:,0]
    #    #        alt_local_vertex = (local_vertex+1) % max_nEOnC
    #    #        v[:,2] = extra_data[cell,alt_local_vertex*dim:(alt_local_vertex+1)*dim] - v[:,0]

    #    #        result = 0.0
    #    #        for i in range(len(qpoints)):

    #    #            unscaled_transformed_qp =  v[:,0].copy()
    #    #            for j in range(dim-1):
    #    #                unscaled_transformed_qp += qpoints[i][j] * v[:,j+1]
    #    #            transformed_qp_norm = np.linalg.norm(unscaled_transformed_qp)
    #    #            scaled_transformed_qp = unscaled_transformed_qp.copy() * radius / transformed_qp_norm

    #    #            # u_qp = midpoint + r_qp[1]*(v_1-midpoint) + r_qp[2]*(v_2-midpoint)
    #    #            # s_qp = u_qp * radius / norm(u_qp) = radius * u_qp / norm(u_qp)
    #    #            #
    #    #            # so G(:,i) is \partial{s_qp}/ \partial{r_qp[i]}
    #    #            # where r_qp is reference quadrature point (R^2 in 2D manifold in R^3)
    #    #            #
    #    #            # G(:,i) = radius * ( \partial{u_qp}/\partial{r_qp[i]} * (\sum_m u_qp[k]^2)^{-1/2}
    #    #            #          + u_qp * \partial{(\sum_m u_qp[k]^2)^{-1/2}}/\partial{r_qp[i]} )
    #    #            #
    #    #            #        = radius * ( T(:,i)/norm(u_qp) + u_qp*(-1/2)*(\sum_m u_qp[k]^2)^{-3/2}
    #    #            #                              *2*(\sum_k u_qp[k]*\partial{u_qp[k]}/\partial{r_qp[i]}) )
    #    #            #
    #    #            #        = radius * ( T(:,i)/norm(u_qp) + u_qp*(-1/2)*(\sum_m u_qp[k]^2)^{-3/2}
    #    #            #                              *2*(\sum_k u_qp[k]*T(k,i)) )
    #    #            #
    #    #            qp_norm_sq = transformed_qp_norm**2
    #    #            G = v.copy() / transformed_qp_norm
    #    #            for k in range(dim):
    #    #                G[:,1] += unscaled_transformed_qp*(-0.5)*pow(qp_norm_sq,-1.5) \
    #    #                          *2*(unscaled_transformed_qp[k]*v[k,1]);
    #    #                G[:,2] += unscaled_transformed_qp*(-0.5)*pow(qp_norm_sq,-1.5) \
    #    #                          *2*(unscaled_transformed_qp[k]*v[k,2]);

    #    #            # cross product gives stretch factor
    #    #            scaling = radius * np.linalg.norm(np.cross(G[:,1], G[:,2]))
    #    #            #print(G)
    #    #            result += f(scaled_transformed_qp)*qweights[i]*scaling
    #    #            #print(f(scaled_transformed_qp),qweights[i],scaling)
    #    #            cell_area += qweights[i]*scaling

    #    #        in_field[cell] += result
    #    #    #print(in_field[cell])
    #    #    total_area += cell_area

    #    #print("AREA",np.sum(total_area)," vs ",str(4.0*3.141592653*(radius**2)))
    #    ##print(in_field)
    #    ##print(cell_points)
    #    ##gmls_obj=pycompadre.GMLS(pycompadre.ReconstructionSpace.VectorOfScalarClonesTaylorPolynomial, 
    #    #gmls_obj=pycompadre.GMLS(pycompadre.ReconstructionSpace.ScalarTaylorPolynomial, 
    #    #                         pycompadre.SamplingFunctionals['ScalarFaceIntegralSample'], 
    #    #                         p_order, 
    #    #                         dim, 
    #    #                         "QR", 
    #    #                         "MANIFOLD",
    #    #                         "NO_CONSTRAINT",
    #    #                         p_order)

    #    #gmls_obj.setOrderOfQuadraturePoints(q_order)
    #    #gmls_obj.setDimensionOfQuadraturePoints(q_dim)
    #    #gmls_obj.setQuadratureType("TRI")
    #    #gmls_helper = pycompadre.ParticleHelper(gmls_obj)
    #    #gmls_helper.generateKDTree(cell_points)
    #    #gmls_obj.addTargets(pycompadre.TargetOperation.ScalarPointEvaluation)
    #    #gmls_obj.setTargetExtraData(extra_data)
    #    #gmls_obj.setSourceExtraData(extra_data)

    #    #gmls_helper.generateNeighborListsFromKNNSearchAndSet(cell_points, p_order, dim-1, 2.2)
    #    #gmls_obj.generateAlphas(1, False)

    #    #exact_out_field = np.zeros(shape=(nCells), dtype='f8')
    #    #for cell in range(nCells):
    #    #    exact_out_field[cell] = f(cell_points[cell,:])
    #    #out_field = gmls_helper.applyStencil(in_field, pycompadre.TargetOperation.ScalarPointEvaluation)
    #    #print('exact out:',exact_out_field)
    #    #print('computed out:',out_field)
    #    #print('diff:',out_field - exact_out_field)

    #    #del gmls_helper, gmls_obj

if __name__ == '__main__':
    import unittest
    unittest.main()
