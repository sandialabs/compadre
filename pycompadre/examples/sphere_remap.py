from kokkos_test_case import KokkosTestCase
import pycompadre
from netCDF4 import Dataset
import numpy as np

do_plot = False

'''

Demonstration of remap of cell-integrated quantities to point values on the sphere
with radius != 1.0.

Requires a .nc file that contains all of the fields referenced in this file.

'''
class TestSphereRemapCellIntegral(KokkosTestCase):

    def test_cell_integrated_remap(self):

        def run(level, direction=0):

            dataset = Dataset('../../../test_data/grids/sphere_{0}.nc'.format(str(level)), "r", format="NETCDF4")

            THREE = 3 # number of vertices in a triangle
            radius = 6371220.00
            dimensions = dataset.dimensions
            variables = dataset.variables

            p_order = 4
            # get cells and put averaged quantities on them
            q_order = 5

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

            # need to produce data for the remap (get cell integrated data)
            f = lambda x: (x[0]/radius)**2 - (x[1]/radius)**2 + (x[1]/radius)*(x[0]/radius) \
                           + 55.0 + (x[2]/radius)**2 - (x[0]/radius)*(x[2]/radius)

            # directions 
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
                        cell_area += qweights[i]*scaling

                    exact_in_field[cell] += result
                computed_total_area += cell_area

            print("AREA", computed_total_area, " vs ", ref_total_area)
            print("SPHERE AREA:",str(4.0*3.141592653*(radius**2)))

            exact_out_field = np.zeros(shape=(nCells), dtype='f8')
            for cell in range(nCells):
                exact_out_field[cell] = f(cell_points[cell,:])

            if direction==0:
                gmls_obj=pycompadre.GMLS(pycompadre.ReconstructionSpace.ScalarTaylorPolynomial, 
                                         pycompadre.SamplingFunctionals['CellIntegralSample'], 
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
                out_field = gmls_helper.applyStencil(exact_in_field, pycompadre.TargetOperation.ScalarPointEvaluation)
            elif direction==1:
                gmls_obj=pycompadre.GMLS(pycompadre.ReconstructionSpace.ScalarTaylorPolynomial, 
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
                gmls_obj.addTargets(pycompadre.TargetOperation.CellIntegralEvaluation)
                gmls_obj.setTargetExtraData(extra_data)
                gmls_obj.setSourceExtraData(extra_data)

                gmls_helper.generateNeighborListsFromKNNSearchAndSet(cell_points, p_order, DIM-1, 2.2)
                gmls_obj.generateAlphas(1, False)
                # use exact_out_field instead of exact_in_field
                out_field = gmls_helper.applyStencil(exact_out_field, pycompadre.TargetOperation.CellIntegralEvaluation)

            print('exact out:',exact_out_field)
            print('computed out:',out_field)
            print('diff:',out_field - exact_out_field)

            print('exact_out',exact_out_field, 'exact_in',exact_in_field, 'computed_out',out_field)
            if direction==1:
                # swap out field with in field
                exact_out_field = exact_in_field
            print('out_field-exact_out_field',out_field-exact_out_field)

            # calculate error norm (l2)
            #rel_l2_error = np.linalg.norm(out_field-exact_out_field, axis=0)/np.sqrt(out_field.shape[0])
            rel_l2_error = np.linalg.norm(out_field-exact_out_field, axis=0)/np.linalg.norm(exact_out_field)
            print('Error norm:', rel_l2_error)
            return rel_l2_error

        rel_l2_errors = []
        for i in range(args.grids):
            rel_l2_errors.append(run(i))
        print('cell integrals :: rel_l2_errors', rel_l2_errors)

        rel_l2_rates = []
        for i in range(1,len(rel_l2_errors)):
            rel_l2_rates.append(np.log(rel_l2_errors[i]/rel_l2_errors[i-1])/np.log(0.5))
        print('cell integrals :: rel_l2_rates', rel_l2_rates)

class TestSphereRemapEdgeIntegral(KokkosTestCase):

    def test_edge_normal_integrated_remap_2d_manifold(self):

        def run(level, p_order):

            global input_operation, output_operation, basis_type

            dataset = Dataset('../../../test_data/grids/sphere_{0}.nc'.format(str(level)), "r", format="NETCDF4")

            TWO = 2 # number of vertices on an edge

            remap_input_type  = input_operation
            remap_output_type = output_operation

            remap_sampling_types = (pycompadre.SamplingFunctionals['ManifoldVectorPointSample'],
                                    pycompadre.SamplingFunctionals['FaceNormalAverageSample'], 
                                    pycompadre.SamplingFunctionals['FaceNormalIntegralSample'],
                                    pycompadre.SamplingFunctionals['EdgeTangentAverageSample'], 
                                    pycompadre.SamplingFunctionals['EdgeTangentIntegralSample'])
            remap_target_types = (pycompadre.TargetOperation.VectorPointEvaluation,
                                  pycompadre.TargetOperation.FaceNormalAverageEvaluation,
                                  pycompadre.TargetOperation.FaceNormalIntegralEvaluation,
                                  pycompadre.TargetOperation.EdgeTangentAverageEvaluation,
                                  pycompadre.TargetOperation.EdgeTangentIntegralEvaluation)

            sampling_functional = remap_sampling_types[remap_input_type]
            if remap_input_type > 0:
                data_sampling_functional = pycompadre.SamplingFunctionals['PointSample']
            else:
                data_sampling_functional = pycompadre.SamplingFunctionals['ManifoldVectorPointSample']

            target_operator = remap_target_types[remap_output_type]

            if basis_type==0:
                basis = pycompadre.ReconstructionSpace.VectorOfScalarClonesTaylorPolynomial
            else:
                basis = pycompadre.ReconstructionSpace.VectorTaylorPolynomial

            print("Performing: input type %d to %s"%(remap_input_type, target_operator))


            radius = 6371220.00
            dimensions = dataset.dimensions
            variables = dataset.variables

            c_order = 4
            q_order = 5

            DIM = 3
            assert DIM==3, "Only created for 3D problem with 2D manifolds (DIM==3)"
            Q_DIM   = 1 # local manifold dimension for quadrature

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


            #def check_var(var):
            #    for edge in range(nEdges):
            #        if (np.single(var[edge]) != np.double(var[edge])):
            #            print(var[edge],np.single(var[edge])-np.double(var[edge]))
            #check_var(lonE)

            # two vertex coordinates plus a normal vector and optional tangent vector (we don't use)
            extra_data = np.zeros(shape=(nEdges, DIM*TWO + 2*DIM), dtype='f8') 

            edge_points = np.zeros(shape=(nEdges, DIM), dtype='f8')
            err = np.zeros(shape=(nEdges, DIM), dtype='f8')

            # need to produce data for the remap (get edge normal integrated data)
            # NOTE: This must be tangent to the sphere!!!
            # NOTE: lon from 0 to 2*pi can exist in a neighborhood, creating a discontinuity
            f_zonal = lambda lat, lon: np.cos(lat)
            f_meridional = lambda lat, lon: lat**4-(np.pi/2.0)**4
            f = lambda lat, lon, zonal, meridional: f_zonal(lat,lon)*zonal + f_meridional(lat,lon)*meridional
            u_xyz = lambda x: [np.sin(x[0]), np.sin(x[1]), np.sin(x[2])]
            f_xyz = lambda x, zonal, meridional: (u_xyz(x)[0]*zonal[0]+u_xyz(x)[1]*zonal[1]+u_xyz(x)[2]*zonal[2])*zonal + (u_xyz(x)[0]*meridional[0]+u_xyz(x)[1]*meridional[1]+u_xyz(x)[2]*meridional[2])*meridional

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

            exact_edge_average_normal   = np.zeros(shape=(nEdges), dtype='f8')
            exact_edge_integral_normal  = np.zeros(shape=(nEdges), dtype='f8')
            exact_edge_average_tangent  = np.zeros(shape=(nEdges), dtype='f8')
            exact_edge_integral_tangent = np.zeros(shape=(nEdges), dtype='f8')

            rot_rad = np.pi/2.0
            cos_r = np.cos(rot_rad)
            sin_r = np.sin(rot_rad)
            #R = np.array([[1, 0, 0], [0, cos_r, -sin_r], [0, sin_r, cos_r]], dtype='f8')
            R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='f8')

            computed_total_length = 0.0
            ref_total_length = 0.0

            # logic for evaluation at extra target sites (quadrature)
            extra_points_coordinates = np.zeros(shape=(nEdges*len(qpoints), DIM), dtype=np.float64)
            extra_points_indices = np.zeros(shape=(nEdges,len(qpoints)+1), dtype=np.int32)
            quadrature_i = 0

            # get edges and put integral quantities on them
            for edge in range(nEdges):

                # keep track of column for extra_points_indices
                target_quadrature_local_entries = 0

                edge_points[edge,:] = np.asarray([xE[edge], yE[edge], zE[edge]], dtype='f8')
                edge_length = 0.0
                ref_total_length += dataset['dvEdge'][edge]
                for local_vertex in range(TWO):
                    vertex_num = vOnE[edge, local_vertex] - 1
                    extra_data[edge,local_vertex*DIM:(local_vertex+1)*DIM] = \
                        np.asarray([xV[vertex_num], yV[vertex_num], zV[vertex_num]], dtype='f8')
                    extra_data[edge,local_vertex*DIM:(local_vertex+1)*DIM] = radius * extra_data[edge,local_vertex*DIM:(local_vertex+1)*DIM] / np.linalg.norm(extra_data[edge,local_vertex*DIM:(local_vertex+1)*DIM])

                v[:,0] = extra_data[edge,0:DIM].copy()
                v[:,1] = extra_data[edge,DIM:2*DIM].copy() - v[:,0]

                a = extra_data[edge,0:DIM]
                b = extra_data[edge,DIM:2*DIM]

                #theta = np.arccos(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
                # arctan less sensitive near the poles
                theta = np.arctan(np.linalg.norm(np.cross(a,b)) / np.dot(a,b))
                arclength = theta * radius

                #lat = dataset['latEdge'][edge]
                #lon = dataset['lonEdge'][edge]
                lat, lon = get_lat_lon(edge_points[edge,:])
                zonalUnitVector, meridionalUnitVector, verticalUnitVector = get_sphere_basis(lat, lon)
                # can add assert that latEdge, lonEdge is same lat as calculated from midpoint
                #print(lat, dataset['latEdge'][edge], (lat-dataset['latEdge'][edge]) / abs(dataset['latEdge'][edge]))
                #print(lon, dataset['lonEdge'][edge], (lon-dataset['lonEdge'][edge]) / abs(dataset['lonEdge'][edge]))

                # is verticalUnitVector close to xyz?
                #print(radius*verticalUnitVector, edge_points[edge,:], np.linalg.norm(radius*verticalUnitVector-edge_points[edge,:])/radius)
                #print(np.linalg.norm(radius*verticalUnitVector-edge_points[edge,:])/radius)

                T[edge,0,:] = zonalUnitVector
                T[edge,1,:] = meridionalUnitVector
                T[edge,2,:] = verticalUnitVector

                xyz = verticalUnitVector * radius
                rot_xyz = np.matmul(R, xyz)

                ## not constant over all quadrature, just midpoint
                #angle = dataset['angleEdge'][edge] # angle at centroid eastwards

                ## normal at the midpoint of edge (use for orientation)
                ## but constant on a great circle so valid at all quadrature as well
                ## not sufficiently accurate near poles
                #alt_norm_vec = np.cos(angle)*zonalUnitVector + np.sin(angle)*meridionalUnitVector

                # replace with cross product of endpoint vertices
                norm_vec = np.cross(a,b) / np.linalg.norm(np.cross(a,b))
                #norm_vec *= np.sign(np.dot(norm_vec, alt_norm_vec))
                extra_data[edge,TWO*DIM:(TWO+1)*DIM] = norm_vec

                result_normal = 0.0
                result_tangent = 0.0
                for i in range(len(qpoints)):

                    unscaled_transformed_qp = v[:,0].copy()
                    unscaled_transformed_qp += qpoints[i][0] * v[:,1]
                    transformed_qp_norm = np.linalg.norm(unscaled_transformed_qp)
                    scaled_transformed_qp = unscaled_transformed_qp.copy() * radius / transformed_qp_norm

                    lat_qp, lon_qp = get_lat_lon(scaled_transformed_qp)
                    zonalUnitVector_qp, meridionalUnitVector_qp, verticalUnitVector_qp = get_sphere_basis(lat_qp, lon_qp)

                    scaling = arclength

                    ## u_qp = v_1 + r_qp[1]*(v_2 - v_1)
                    ## s_qp = u_qp * radius / norm(u_qp) = radius * u_qp / norm(u_qp)
                    ##
                    ## so G(:,1) is \partial{s_qp}/ \partial{r_qp[1]}
                    ## where r_qp is reference quadrature point (R^1 in 1D manifold in R^3)
                    ##
                    ## G(:,i) = radius * ( \partial{u_qp}/\partial{r_qp[i]} * (\sum_m u_qp[k]^2)^{-1/2}
                    ##          + u_qp * \partial{(\sum_m u_qp[k]^2)^{-1/2}}/\partial{r_qp[i]} )
                    ##
                    ##        = radius * ( T(:,i)/norm(u_qp) + u_qp*(-1/2)*(\sum_m u_qp[k]^2)^{-3/2}
                    ##                              *2*(\sum_k u_qp[k]*\partial{u_qp[k]}/\partial{r_qp[i]}) )
                    ##
                    ##        = radius * ( T(:,i)/norm(u_qp) + u_qp*(-1/2)*(\sum_m u_qp[k]^2)^{-3/2}
                    ##                              *2*(\sum_k u_qp[k]*T(k,i)) )
                    ##
                    #qp_norm_sq = transformed_qp_norm**2
                    #G = v.copy() / transformed_qp_norm
                    #for k in range(DIM):
                    #    G[:,1] += unscaled_transformed_qp*(-0.5)*pow(qp_norm_sq,-1.5) \
                    #              *2*(unscaled_transformed_qp[k]*v[k,1]);
                    #scaling = radius * np.linalg.norm(G[:,1])

                    # norm_vec is constant for all qp on a great circle
                    #result += np.dot(f(lat_qp, lon_qp, zonalUnitVector_qp, meridionalUnitVector_qp), norm_vec)*qweights[i]*scaling

                    # rotate the vector 90
                    xyz_qp = verticalUnitVector_qp.copy() * radius
                    rot_xyz_qp = np.matmul(R, xyz_qp)
                    d_rot = np.linalg.norm(rot_xyz_qp-rot_xyz)
                    assert d_rot < dataset['dcEdge'][edge], "Quadrature and midpoint too far apart"
                    alt_lat, alt_lon = get_lat_lon(rot_xyz_qp/np.linalg.norm(rot_xyz_qp))
                    alt_zonalUnitVector, alt_meridionalUnitVector, alt_verticalUnitVector = get_sphere_basis(alt_lat, alt_lon)
                    #ans_f = f(alt_lat, alt_lon, alt_zonalUnitVector, alt_meridionalUnitVector)
                    ans_f = f_xyz(rot_xyz_qp/np.linalg.norm(rot_xyz_qp), alt_zonalUnitVector, alt_meridionalUnitVector)
                    alt_f = np.linalg.solve(R, ans_f)
                    result_normal += np.dot(alt_f, norm_vec)*qweights[i]*scaling

                    cross_k_n_qp = np.cross(rot_xyz_qp/np.linalg.norm(rot_xyz_qp), norm_vec)
                    tang_vec = cross_k_n_qp / np.linalg.norm(cross_k_n_qp)
                    result_tangent += np.dot(alt_f, tang_vec)*qweights[i]*scaling

                    # check if tangent vector is completely inside of zonal and meridional
                    #print("dot",np.dot(alt_zonalUnitVector,tang_vec)+ np.dot(alt_meridionalUnitVector,tang_vec))
                    ans=np.linalg.norm(np.dot(alt_zonalUnitVector,tang_vec)*alt_zonalUnitVector+ np.dot(alt_meridionalUnitVector,tang_vec)*alt_meridionalUnitVector-tang_vec)
                    if abs(ans)>1e-14:
                        print("norm:",ans)

                    edge_length += qweights[i]*scaling

                    # logic for evaluation at extra target sites (quadrature)
                    extra_points_coordinates[quadrature_i, :] = scaled_transformed_qp[:]
                    extra_points_indices[edge, target_quadrature_local_entries+1] = quadrature_i
                    quadrature_i+=1 # keeps increasing over cell loops
                    target_quadrature_local_entries+=1 # resets after edge quadrature loops

                exact_edge_integral_normal[edge] += result_normal
                exact_edge_average_normal[edge] += result_normal / edge_length
                exact_edge_integral_tangent[edge] += result_tangent
                exact_edge_average_tangent[edge] += result_tangent / edge_length

                computed_total_length += edge_length
                err[edge] = abs(edge_length-arclength)/abs(edge_length)

                extra_points_indices[edge, 0] = target_quadrature_local_entries

            #print('arc norm:',np.linalg.norm(err))
            print("LENGTH", computed_total_length, " vs ", ref_total_length)

            # get exact solution
            exact_point = np.zeros(shape=(nEdges,3), dtype='f8')
            for edge in range(nEdges):
                lat = dataset['latEdge'][edge]
                lon = dataset['lonEdge'][edge]
                zonalUnitVector, meridionalUnitVector, verticalUnitVector = get_sphere_basis(lat, lon)
                #exact_out_field[edge,:] = f(lat, lon, zonalUnitVector, meridionalUnitVector)

                # rotate the vector 90
                xyz = verticalUnitVector * radius
                rot_xyz = np.matmul(R, xyz)

                alt_lat, alt_lon = get_lat_lon(rot_xyz/np.linalg.norm(rot_xyz))
                alt_zonalUnitVector, alt_meridionalUnitVector, alt_verticalUnitVector = get_sphere_basis(alt_lat, alt_lon)
                #ans_f = f(alt_lat, alt_lon, alt_zonalUnitVector, alt_meridionalUnitVector)
                ans_f = f_xyz(rot_xyz/np.linalg.norm(rot_xyz), alt_zonalUnitVector, alt_meridionalUnitVector)
                alt_f = np.linalg.solve(R, ans_f)
                exact_point[edge,:] = alt_f


            # logic for evaluation at extra target sites (quadrature)
            # get exact solution at quadrature #n
            extra_n = 0
            if remap_output_type==0 and extra_n!=0:
                for edge in range(nEdges):
                    rot_xyz = extra_points_coordinates[extra_points_indices[edge, extra_n],:]
                    alt_lat, alt_lon = get_lat_lon(rot_xyz/np.linalg.norm(rot_xyz))
                    alt_zonalUnitVector, alt_meridionalUnitVector, alt_verticalUnitVector = get_sphere_basis(alt_lat, alt_lon)
                    #ans_f = f(alt_lat, alt_lon, alt_zonalUnitVector, alt_meridionalUnitVector)
                    ans_f = f_xyz(rot_xyz/np.linalg.norm(rot_xyz), alt_zonalUnitVector, alt_meridionalUnitVector)
                    alt_f = np.linalg.solve(R, ans_f)
                    exact_point[edge,:] = alt_f

            gmls_obj=pycompadre.GMLS(basis,
                                     sampling_functional,
                                     data_sampling_functional,
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
            gmls_obj.addTargets(target_operator)
            gmls_obj.setTargetExtraData(extra_data)
            gmls_obj.setSourceExtraData(extra_data)

            # logic for evaluation at extra target sites (quadrature)
            # get extra sites
            if remap_output_type==0 and extra_n!=0:
                gmls_helper.setAdditionalEvaluationSitesData(extra_points_indices, extra_points_coordinates)

            gmls_helper.generateNeighborListsFromKNNSearchAndSet(edge_points, max(p_order,c_order), DIM-1, 3.8)
            gmls_helper.setTangentBundle(T)
            gmls_obj.generateAlphas(1, True)

            if remap_input_type==0:
                exact_in = exact_point
            elif remap_input_type==1:
                exact_in = exact_edge_average_normal
            elif remap_input_type==2:
                exact_in = exact_edge_integral_normal
            elif remap_input_type==3:
                exact_in = exact_edge_average_tangent
            elif remap_input_type==4:
                exact_in = exact_edge_integral_tangent

            # get computed solution
            out_field = gmls_helper.applyStencil(exact_in, 
                                                 target_operator,
                                                 sampling_functional)

            # logic for evaluation at extra target sites (quadrature)
            if remap_output_type==0 and extra_n!=0:
                # get value at nth quadrature point
                out_field = gmls_helper.applyStencil(exact_in, 
                                                     target_operator,
                                                     sampling_functional,
                                                     extra_n)

            ## remove extreme lats
            #for edge in range(nEdges):
            #    lat = dataset['latEdge'][edge]
            #    if abs(lat)>np.pi/3.:
            #        out_field[edge,:] = 0.0
            #        exact_out_field[edge,:] = 0.0

            ## remove equator
            #for edge in range(nEdges):
            #    lat = dataset['latEdge'][edge]
            #    if abs(lat)<np.pi/6.:
            #        out_field[edge,:] = 0.0
            #        exact_out_field[edge,:] = 0.0

            if remap_output_type==0:
                exact_out = exact_point
            elif remap_output_type==1:
                exact_out = exact_edge_average_normal
            elif remap_output_type==2:
                exact_out = exact_edge_integral_normal
            elif remap_output_type==3:
                exact_out = exact_edge_average_tangent
            elif remap_output_type==4:
                exact_out = exact_edge_integral_tangent

            print('computed out:',out_field)
            print('exact out:',exact_out)
            print('diff:',out_field - exact_out)
            #with np.printoptions(threshold=np.inf):
            #    print('diff:',out_field - exact_out_field)
            print(np.max(out_field-exact_out, axis=0))

            # calculate error norm (l2)
            rel_l2_error = np.linalg.norm(out_field-exact_out, axis=0)/np.sqrt(out_field.shape[0])

            
            global do_plot
            if do_plot:
                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=(12, 12))
                ax = fig.add_subplot(projection='3d')
                #err = np.linalg.norm(out_field-exact_out, axis=1)
                #ax.scatter(dataset['xEdge'], dataset['yEdge'], dataset['zEdge'], c=err)
                sc = ax.scatter(dataset['xEdge'], dataset['yEdge'], dataset['zEdge'], c=exact_out[:,1]-out_field[:,1])
                #sc = ax.scatter(dataset['xEdge'], dataset['yEdge'], dataset['zEdge'], c=exact_out-out_field)#[:,0])
                #sc = ax.scatter(dataset['xEdge'], dataset['yEdge'], dataset['zEdge'], c=exact_out_field[:,0])
                #sc = ax.scatter(dataset['xEdge'], dataset['yEdge'], dataset['zEdge'], c=in_field[:])
                plt.colorbar(sc)
                plt.show() 

            #with np.printoptions(threshold=np.inf):
            #    print(np.linalg.norm(out_field-exact_out_field, axis=1))
            print('Error norm:', rel_l2_error)
            print('Error norm (all comps.):', np.linalg.norm(rel_l2_error))
            return rel_l2_error

        p_order = 4

        rel_l2_errors = []
        for i in range(0,args.grids):
            rel_l2_errors.append(run(i, p_order))
        print('edge normal integrals :: rel_l2_errors', rel_l2_errors)

        rel_l2_rates = []
        for i in range(1,len(rel_l2_errors)):
            rel_l2_rates.append(np.log(rel_l2_errors[i]/rel_l2_errors[i-1])/np.log(0.5))
        print('edge normal integrals :: rel_l2_rates', rel_l2_rates)

        result=True
        for rate_block in rel_l2_rates:
            print(type(rate_block))
            if type(rate_block)==np.float64:
                if rate_block < 5:
                    result &= False
            else:
                for i in range(rate_block.shape[0]):
                    if rate_block[i] < p_order+1:
                        result &= False

        assert result, "Failed"

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='test remap on the sphere')
    parser.add_argument('-b','--basis', dest='basis_type', type=int, default=1, help='basis type (0-VectorOfScalarClonesTaylorPolynomial 1-VectorTaylorPolynomial')
    parser.add_argument('-io','--input-operation', dest='input_operation', type=int, default=1, help='input operation type (0-IntegralAverage 1-Integral 2-PointValues)')
    parser.add_argument('-oo','--output-operation', dest='output_operation', type=int, default=1, help='output operation type (0-IntegralAverage 1-Integral 2-PointValues )')
    parser.add_argument('-g','--grids', dest='grids', type=int, default=2, help='number of grids for refinement sequence')
    parser.add_argument('-l','--list', nargs='*', help='{0: cell-integrated, 1: edge-integrated}', default=['0','1'], required=False)
    args = parser.parse_args()

    global input_operation, output_operation, basis_type
    input_operation = args.input_operation
    output_operation = args.output_operation
    basis_type = args.basis_type

    # remove tests not requested
    if '0' not in args.list:
        TestSphereRemapCellIntegral = None
    if '1' not in args.list:
        TestSphereRemapEdgeIntegral = None

    import sys
    sys.argv = sys.argv[0:1]

    import unittest
    unittest.main()
