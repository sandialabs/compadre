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

        dataset = Dataset('../../../test_data/grids/hodge_0.nc', "r", format="NETCDF4")

        THREE = 3 # number of vertices in a triangle
        radius = 6371220.00
        dimensions = dataset.dimensions
        variables = dataset.variables

        p_order = 2
        # get cells and put averaged quantities on them
        q_order = 2

        DIM = 3
        assert DIM==3, "Only created for 3D problem with 2D manifolds (DIM==3)"
        Q_DIM   = 2 # local manifold dimension

        qp = pycompadre.Quadrature(q_order, Q_DIM, "TRI")

        # make a triangle from this midpoint, previous midpoint, and centroid
        qpoints = qp.getSites()
        qweights = qp.getWeights()

        nEOnC = dataset['nEdgesOnCell']
        max_nEOnC = np.max(nEOnC)
        min_nEOnC = np.min(nEOnC)
        assert max_nEOnC==min_nEOnC, "Max and min edges on cells are not the same"

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
        extra_data = np.zeros(shape=(nCells, DIM*max_nEOnC), dtype='f8')
        cell_points = np.zeros(shape=(nCells, DIM), dtype='f8')

        # need to produce data for the remap (get cell integrated data)
        f = lambda x: (x[0]/radius)**2 - (x[1]/radius)**2 + (x[1]/radius)*(x[0]/radius) \
                       + 55.0 + (x[2]/radius)**2 - (x[0]/radius)*(x[2]/radius)

        # directions 
        v = np.zeros(shape=(DIM,3), dtype='f8') # 3 is from triangle nodes
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
                extra_data[cell,local_vertex*DIM:(local_vertex+1)*DIM] = \
                    np.asarray([xV[vertex_num], yV[vertex_num], zV[vertex_num]], dtype='f8')

            for local_vertex in range(max_nEOnC):
                v[:,0] = cell_points[cell,:].copy()
                v[:,1] = extra_data[cell,local_vertex*DIM:(local_vertex+1)*DIM] - v[:,0]
                alt_local_vertex = (local_vertex+1) % max_nEOnC
                v[:,2] = extra_data[cell,alt_local_vertex*DIM:(alt_local_vertex+1)*DIM] - v[:,0]
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
                    cell_area += qweights[i]*scaling

                in_field[cell] += result
            computed_total_area += cell_area

        print("AREA", computed_total_area, " vs ", ref_total_area)
        print("SPHERE AREA:",str(4.0*3.141592653*(radius**2)))
        gmls_obj=pycompadre.GMLS(pycompadre.ReconstructionSpace.ScalarTaylorPolynomial, 
                                 pycompadre.SamplingFunctionals['ScalarFaceIntegralSample'], 
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
        gmls_obj.generateAlphas(1, False)

        exact_out_field = np.zeros(shape=(nCells), dtype='f8')
        for cell in range(nCells):
            exact_out_field[cell] = f(cell_points[cell,:])
        out_field = gmls_helper.applyStencil(in_field, pycompadre.TargetOperation.ScalarPointEvaluation)
        print('exact out:',exact_out_field)
        print('computed out:',out_field)
        print('diff:',out_field - exact_out_field)

        # calculate error norm (l2)
        print('Error norm:',np.linalg.norm(out_field-exact_out_field, axis=0)/np.sqrt(out_field.shape[0]))

        # clean up Kokkos dependent objects
        del gmls_helper, gmls_obj

    def test_edge_integrated_remap(self):

        dataset = Dataset('../../../test_data/grids/hodge_0.nc', "r", format="NETCDF4")

        TWO = 2 # number of vertices on an edge

        remap_type  = 1
        remap_types = (pycompadre.SamplingFunctionals['FaceNormalPointSample'], 
                      pycompadre.SamplingFunctionals['FaceNormalIntegralSample'])
        sampling_functional = remap_types[remap_type]

        radius = 6371220.00
        dimensions = dataset.dimensions
        variables = dataset.variables

        p_order = 3
        c_order = 3
        q_order = 3
        if remap_type==0:
            q_order = 1 # point sample can be though of as 1 pt quadrature

        DIM = 3
        assert DIM==3, "Only created for 3D problem with 2D manifolds (DIM==3)"
        Q_DIM   = 1 # local manifold dimension

        qp = pycompadre.Quadrature(q_order, Q_DIM, "LINE")

        # make a triangle from this midpoint, previous midpoint, and centroid
        qpoints = qp.getSites()
        qweights = qp.getWeights()


        # fields needed from nc file
        vOnE = dataset['verticesOnEdge']
        xE = dataset['xEdge']
        yE = dataset['yEdge']
        zE = dataset['zEdge']
        xV = dataset['xVertex']
        yV = dataset['yVertex']
        zV = dataset['zVertex']
        latE = dataset['latEdge']
        lonE = dataset['lonEdge']
        nEdges = dataset.dimensions['nEdges'].size

        # two vertex coordinates plus a normal vector and optional tangent vector (we don't use)
        extra_data = np.zeros(shape=(nEdges, DIM*TWO + 2*DIM), dtype='f8') 

        edge_points = np.zeros(shape=(nEdges, DIM), dtype='f8')

        # need to produce data for the remap (get edge normal integrated data)
        # NOTE: This must be tangent to the sphere!!!
        # NOTE: lon from 0 to 2*pi can exist in a neighborhood, creating a discontinuity
        f_lat = lambda lat: np.sin(lat)
        f_lon = lambda lon: np.cos(lon)
        f = lambda lat, lon, zonal, meridional: f_lat(lat)*zonal + f_lon(lon)*meridional

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

        # directions 
        T = np.zeros(shape=(nEdges,DIM,DIM), dtype='f8')
        v = np.zeros(shape=(DIM,TWO), dtype='f8') # 2 is from vertices on edge
        in_field = np.zeros(shape=(nEdges), dtype='f8')

        computed_total_length = 0.0
        ref_total_length = 0.0
        # get edges and put integral quantities on them
        for edge in range(nEdges):
            edge_points[edge,:] = np.asarray([xE[edge], yE[edge], zE[edge]], dtype='f8')
            edge_length = 0.0
            ref_total_length += dataset['dvEdge'][edge]
            for local_vertex in range(TWO):
                vertex_num = vOnE[edge, local_vertex] - 1
                extra_data[edge,local_vertex*DIM:(local_vertex+1)*DIM] = np.asarray([xV[vertex_num], yV[vertex_num], zV[vertex_num]], dtype='f8')

            v[:,0] = extra_data[edge,0:DIM]
            v[:,1] = extra_data[edge,DIM:2*DIM] - v[:,0]

            lat = dataset['latEdge'][edge]
            lon = dataset['lonEdge'][edge]
            zonalUnitVector, meridionalUnitVector, verticalUnitVector = get_sphere_basis(lat, lon)
            T[edge,0,:] = zonalUnitVector
            T[edge,1,:] = meridionalUnitVector
            T[edge,2,:] = verticalUnitVector

            # not constant over all quadrature, just midpoint
            angle = dataset['angleEdge'][edge] # angle at centroid eastwards

            # normal at the midpoint of edge (use for orientation)
            # but constant on a great circle so valid at all quadrature as well
            norm_vec = np.cos(angle)*zonalUnitVector + np.sin(angle)*meridionalUnitVector
            extra_data[edge,TWO*DIM:(TWO+1)*DIM] = norm_vec

            result = 0.0
            for i in range(len(qpoints)):
                unscaled_transformed_qp = v[:,0].copy()
                unscaled_transformed_qp += qpoints[i][0] * v[:,1]
                transformed_qp_norm = np.linalg.norm(unscaled_transformed_qp)
                scaled_transformed_qp = unscaled_transformed_qp.copy() * radius / transformed_qp_norm

                if remap_type!=0:
                    lat_qp, lon_qp = get_lat_lon(scaled_transformed_qp)
                    zonalUnitVector_qp, meridionalUnitVector_qp, verticalUnitVector_qp = get_sphere_basis(lat_qp, lon_qp)

                ## check whether get_lat_lon is working correctly
                ## testing that lat_qp is very close to lat
                #if (abs(lat)>1e-8):
                #    if (abs(lat_qp-lat)/abs(lat)>1e-2):
                #        print(lat_qp,"vs",lat)
                #else:
                #    print(lat_qp,"vs2.",lat)

                #if (abs(lon)>1e-8):
                #    if (abs(lon_qp-lon)/abs(lon)>1e-2):
                #        print(lon_qp,"vs",lon)
                #else:
                #    print(lon_qp,"vs2.",lon)

                # u_qp = v_1 + r_qp[1]*(v_2 - v_1)
                # s_qp = u_qp * radius / norm(u_qp) = radius * u_qp / norm(u_qp)
                #
                # so G(:,1) is \partial{s_qp}/ \partial{r_qp[1]}
                # where r_qp is reference quadrature point (R^1 in 1D manifold in R^3)
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
                scaling = radius * np.linalg.norm(G[:,1])

                if remap_type==0:
                    result += np.dot(f(lat, lon, zonalUnitVector, meridionalUnitVector), norm_vec)
                else:
                    # norm_vec is constant for all qp on a great circle
                    result += np.dot(f(lat_qp, lon_qp, zonalUnitVector_qp, meridionalUnitVector_qp), norm_vec)*qweights[i]*scaling

                edge_length += qweights[i]*scaling

            in_field[edge] += result
            computed_total_length += edge_length

        if remap_type!=0: # doesn't make sense to calculate arc length when not integrating
            print("LENGTH", computed_total_length, " vs ", ref_total_length)

        gmls_obj=pycompadre.GMLS(pycompadre.ReconstructionSpace.VectorTaylorPolynomial, 
                                 sampling_functional,
                                 pycompadre.SamplingFunctionals['PointSample'],
                                 p_order, 
                                 DIM, 
                                 "QR", 
                                 "MANIFOLD",
                                 "NO_CONSTRAINT",
                                 c_order)

        gmls_obj.setOrderOfQuadraturePoints(q_order)
        gmls_obj.setDimensionOfQuadraturePoints(Q_DIM)
        gmls_obj.setQuadratureType("LINE")

        gmls_obj.setWeightingParameter(2)
        gmls_obj.setWeightingType("power")

        gmls_helper = pycompadre.ParticleHelper(gmls_obj)
        gmls_helper.generateKDTree(edge_points)
        gmls_obj.addTargets(pycompadre.TargetOperation.VectorPointEvaluation)
        gmls_obj.setTargetExtraData(extra_data)
        gmls_obj.setSourceExtraData(extra_data)

        gmls_helper.generateNeighborListsFromKNNSearchAndSet(edge_points, max(p_order,c_order), DIM-1, 3.8)
        gmls_helper.setTangentBundle(T)
        gmls_obj.generateAlphas(1, True)

        # get exact solution
        exact_out_field = np.zeros(shape=(nEdges,3), dtype='f8')
        for edge in range(nEdges):
            lat = dataset['latEdge'][edge]
            lon = dataset['lonEdge'][edge]
            zonalUnitVector, meridionalUnitVector, verticalUnitVector = get_sphere_basis(lat, lon)
            exact_out_field[edge,:] = f(lat, lon, zonalUnitVector, meridionalUnitVector)

        # get computed solution
        out_field = gmls_helper.applyStencil(in_field, 
                                             pycompadre.TargetOperation.VectorPointEvaluation, 
                                             sampling_functional)

        print('computed out:',out_field)
        print('exact out:',exact_out_field)
        print('diff:',out_field - exact_out_field)
        #with np.printoptions(threshold=np.inf):
        #    print('diff:',out_field - exact_out_field)

        # calculate error norm (l2)
        print('Error norm:',np.linalg.norm(out_field-exact_out_field, axis=0)/np.sqrt(out_field.shape[0]))

        # clean up Kokkos dependent objects
        del gmls_helper, gmls_obj

if __name__ == '__main__':
    import unittest
    unittest.main()
