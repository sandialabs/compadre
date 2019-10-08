#include <Compadre_AdvectionDiffusion_Operator.hpp>

#include <Compadre_CoordsT.hpp>
#include <Compadre_ParticlesT.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_DOFManager.hpp>
#include <Compadre_FieldT.hpp>
#include <Compadre_NeighborhoodT.hpp>
#include "Compadre_nanoflannInformation.hpp"
#include "Compadre_nanoflannPointCloudT.hpp"
#include <Compadre_XyzVector.hpp>

#include <Compadre_GMLS.hpp>

#ifdef COMPADREHARNESS_USE_OPENMP
#include <omp.h>
#endif

/*
 Constructs the matrix operator.
 
 */
namespace Compadre {

typedef Compadre::CoordsT coords_type;
typedef Compadre::FieldT fields_type;
typedef Compadre::NeighborhoodT neighborhood_type;
typedef Compadre::NanoFlannInformation nanoflann_neighborhood_type;
typedef Compadre::XyzVector xyz_type;




Teuchos::RCP<crs_graph_type> AdvectionDiffusionPhysics::computeGraph(local_index_type field_one, local_index_type field_two) {
	if (field_two == -1) {
		field_two = field_one;
	}

	Teuchos::RCP<Teuchos::Time> ComputeGraphTime = Teuchos::TimeMonitor::getNewCounter ("Compute Graph Time");
	ComputeGraphTime->start();
	TEUCHOS_TEST_FOR_EXCEPT_MSG(this->_A_graph.is_null(), "Tpetra CrsGraph for Physics not yet specified.");

	const local_index_type nlocal = static_cast<local_index_type>(this->_coords->nLocal());
	const std::vector<Teuchos::RCP<fields_type> >& fields = this->_particles->getFieldManagerConst()->getVectorOfFields();
	const neighborhood_type * particles_particles_neighborhood = this->_particles->getNeighborhoodConst();
    const local_dof_map_view_type local_to_dof_map = _dof_data->getDOFMap();

	//#pragma omp parallel for
    for(local_index_type i = 0; i < nlocal; i++) {
        local_index_type num_neighbors = particles_particles_neighborhood->getNeighbors(i).size();
        for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
            local_index_type row = local_to_dof_map(i, field_one, k);

            Teuchos::Array<local_index_type> col_data(num_neighbors * fields[field_two]->nDim());
            Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);
            std::vector<std::pair<size_t, scalar_type> > neighbors = particles_particles_neighborhood->getNeighbors(i);

            for (local_index_type l = 0; l < num_neighbors; l++) {
                for (local_index_type n = 0; n < fields[field_two]->nDim(); ++n) {
                    col_data[l*fields[field_two]->nDim() + n] = local_to_dof_map(static_cast<local_index_type>(neighbors[l].first), field_two, n);
                }
            }
            //#pragma omp critical
            {
                this->_A_graph->insertLocalIndices(row, cols);
            }
        }
    }
	ComputeGraphTime->stop();
    return this->_A_graph;
}

void AdvectionDiffusionPhysics::computeMatrix(local_index_type field_one, local_index_type field_two, scalar_type time) {


    // we don't know which particles are in each cell
    // cell neighbor search should someday be bigger than the particles neighborhoods (so that we are sure it includes all interacting particles)
    // get neighbors of cells (the neighbors are particles)
    local_index_type maxLeaf = _parameters->get<Teuchos::ParameterList>("neighborhood").get<int>("max leaf");
    auto cell_particles_neighborhood = Teuchos::rcp_static_cast<neighborhood_type>(Teuchos::rcp(
            new nanoflann_neighborhood_type(_particles.getRawPtr(), _parameters, maxLeaf, _cells, false /* material coords */)));
    // for now, since cells = particles, neighbor lists should be the same, later symmetric neighbor search should be enforced
    auto particle_cells_neighborhood = Teuchos::rcp_static_cast<neighborhood_type>(Teuchos::rcp(
            new nanoflann_neighborhood_type(_cells, _parameters, maxLeaf, _particles.getRawPtr(), false /* material coords */)));
    auto particles_particles_neighborhood = Teuchos::rcp_static_cast<neighborhood_type>(Teuchos::rcp(
            new nanoflann_neighborhood_type(_particles.getRawPtr(), _parameters, maxLeaf, _particles.getRawPtr(), false /* material coords */)));

    auto neighbors_needed = GMLS::getNP(_parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"), 2);
    local_index_type extra_neighbors = _parameters->get<Teuchos::ParameterList>("remap").get<double>("neighbors needed multiplier") * neighbors_needed;

    TEUCHOS_TEST_FOR_EXCEPT_MSG(_particles->getCoordsConst()->getComm()->getRank()>0, "Only for serial.");
    particles_particles_neighborhood->constructAllNeighborList(0, extra_neighbors,
        _parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("size"),
        maxLeaf,
        false);
    cell_particles_neighborhood->constructAllNeighborList(0, extra_neighbors,
        _parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("size"),
        maxLeaf,
        false);
    particle_cells_neighborhood->constructAllNeighborList(0, extra_neighbors,
        _parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("size"),
        maxLeaf,
        false);

    Teuchos::RCP<Teuchos::Time> ComputeMatrixTime = Teuchos::TimeMonitor::getNewCounter ("Compute Matrix Time");
    ComputeMatrixTime->start();

    bool use_physical_coords = true; // can be set on the operator in the future

    TEUCHOS_TEST_FOR_EXCEPT_MSG(this->_A.is_null(), "Tpetra CrsMatrix for Physics not yet specified.");

    //Loop over all particles, convert to GMLS data types, solve problem, and insert into matrix:

    const local_index_type nlocal = static_cast<local_index_type>(this->_coords->nLocal());
    const std::vector<Teuchos::RCP<fields_type> >& fields = this->_particles->getFieldManagerConst()->getVectorOfFields();
    //const neighborhood_type * particles_particles_neighborhood = this->_particles->getNeighborhoodConst();
    const local_dof_map_view_type local_to_dof_map = _dof_data->getDOFMap();
    const host_view_local_index_type bc_id = this->_particles->getFlags()->getLocalView<host_view_local_index_type>();


    //****************
    //
    //  Copying data from particles (std::vector's, multivectors, etc....) to views used by local reconstruction class
    //
    //****************

    // generate the interpolation operator and call the coefficients needed (storing them)
    const coords_type* target_coords = this->_coords;
    const coords_type* source_coords = this->_coords;

    const std::vector<std::vector<std::pair<size_t, scalar_type> > >& particles_particles_all_neighbors = particles_particles_neighborhood->getAllNeighbors();
    const std::vector<std::vector<std::pair<size_t, scalar_type> > >& particle_cells_all_neighbors = particle_cells_neighborhood->getAllNeighbors();
    const std::vector<std::vector<std::pair<size_t, scalar_type> > >& cell_particles_all_neighbors = cell_particles_neighborhood->getAllNeighbors();

    size_t particles_particles_max_num_neighbors = 0;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()),
            KOKKOS_LAMBDA (const int i, size_t &myVal) {
        myVal = (particles_particles_all_neighbors[i].size() > myVal) ? particles_particles_all_neighbors[i].size() : myVal;
    }, Kokkos::Experimental::Max<size_t>(particles_particles_max_num_neighbors));

    size_t particle_cells_max_num_neighbors = 0;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()),
            KOKKOS_LAMBDA (const int i, size_t &myVal) {
        myVal = (particle_cells_all_neighbors[i].size() > myVal) ? particle_cells_all_neighbors[i].size() : myVal;
    }, Kokkos::Experimental::Max<size_t>(particle_cells_max_num_neighbors));

    Kokkos::View<int**> kokkos_neighbor_lists("neighbor lists", target_coords->nLocal(), particles_particles_max_num_neighbors+1);
    Kokkos::View<int**>::HostMirror kokkos_neighbor_lists_host = Kokkos::create_mirror_view(kokkos_neighbor_lists);

    // fill in the neighbor lists into a kokkos view. First entry is # of neighbors for that target
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
        const int num_i_neighbors = particles_particles_all_neighbors[i].size();
        for (int j=1; j<num_i_neighbors+1; ++j) {
            kokkos_neighbor_lists_host(i,j) = particles_particles_all_neighbors[i][j-1].first;
        }
        kokkos_neighbor_lists_host(i,0) = num_i_neighbors;
    });

    Kokkos::View<double**> kokkos_augmented_source_coordinates("source_coordinates", source_coords->nLocal(true /* include halo in count */), source_coords->nDim());
    Kokkos::View<double**>::HostMirror kokkos_augmented_source_coordinates_host = Kokkos::create_mirror_view(kokkos_augmented_source_coordinates);

    // fill in the source coords, adding regular with halo coordiantes into a kokkos view
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,source_coords->nLocal(true /* include halo in count*/)), KOKKOS_LAMBDA(const int i) {
        xyz_type coordinate = source_coords->getLocalCoords(i, true /*include halo*/, use_physical_coords);
        kokkos_augmented_source_coordinates_host(i,0) = coordinate.x;
        kokkos_augmented_source_coordinates_host(i,1) = coordinate.y;
        kokkos_augmented_source_coordinates_host(i,2) = coordinate.z;
    });

    Kokkos::View<double**> kokkos_target_coordinates("target_coordinates", target_coords->nLocal(), target_coords->nDim());
    Kokkos::View<double**>::HostMirror kokkos_target_coordinates_host = Kokkos::create_mirror_view(kokkos_target_coordinates);
    // fill in the target, adding regular coordiantes only into a kokkos view
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
        xyz_type coordinate = target_coords->getLocalCoords(i, false /*include halo*/, use_physical_coords);
        kokkos_target_coordinates_host(i,0) = coordinate.x;
        kokkos_target_coordinates_host(i,1) = coordinate.y;
        kokkos_target_coordinates_host(i,2) = coordinate.z;
    });

    auto epsilons = particles_particles_neighborhood->getHSupportSizes()->getLocalView<const host_view_type>();
    Kokkos::View<double*> kokkos_epsilons("target_coordinates", target_coords->nLocal(), target_coords->nDim());
    Kokkos::View<double*>::HostMirror kokkos_epsilons_host = Kokkos::create_mirror_view(kokkos_epsilons);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
        kokkos_epsilons_host(i) = epsilons(i,0);
    });

    // quantities contained on cells (mesh)
    auto quadrature_points = _cells->getFieldManager()->getFieldByName("quadrature_points")->getMultiVectorPtr()->getLocalView<host_view_type>();
    auto quadrature_weights = _cells->getFieldManager()->getFieldByName("quadrature_weights")->getMultiVectorPtr()->getLocalView<host_view_type>();
    auto quadrature_type = _cells->getFieldManager()->getFieldByName("interior")->getMultiVectorPtr()->getLocalView<host_view_type>();
    auto unit_normals = _cells->getFieldManager()->getFieldByName("unit_normal")->getMultiVectorPtr()->getLocalView<host_view_type>();




    // get all quadrature put together here as auxiliary evaluation sites
    auto weights_ndim = quadrature_weights.extent(1);
    Kokkos::View<double**> kokkos_quadrature_coordinates("all quadrature coordinates", _cells->getCoordsConst()->nLocal()*weights_ndim, 3);
    Kokkos::View<double**>::HostMirror kokkos_quadrature_coordinates_host = Kokkos::create_mirror_view(kokkos_quadrature_coordinates);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,_cells->getCoordsConst()->nLocal()), KOKKOS_LAMBDA(const int i) {
        for (int j=0; j<weights_ndim; ++j) {
            kokkos_quadrature_coordinates_host(i*weights_ndim + j, 0) = quadrature_weights(i,2*j+0);
            kokkos_quadrature_coordinates_host(i*weights_ndim + j, 1) = quadrature_weights(i,2*j+1);
        }
    });

    // get cell neighbors of particles, and add indices to these for quadrature
    // loop over particles, get cells in neighborhood and get their quadrature
    Kokkos::View<int**> kokkos_quadrature_neighbor_lists("quadrature neighbor lists", target_coords->nLocal(), weights_ndim*particle_cells_max_num_neighbors+1);
    Kokkos::View<int**>::HostMirror kokkos_quadrature_neighbor_lists_host = Kokkos::create_mirror_view(kokkos_quadrature_neighbor_lists);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
        kokkos_quadrature_neighbor_lists(i,0) = static_cast<int>(particle_cells_all_neighbors[i].size());
        for (int j=0; j<kokkos_quadrature_neighbor_lists(i,0); ++j) {
            auto cell_num = static_cast<int>(particle_cells_all_neighbors[i][j].first);
            for (int k=0; k<weights_ndim; ++k) {
                kokkos_quadrature_neighbor_lists_host(i, 1 + j*weights_ndim + k) = cell_num*weights_ndim + k;
            }
        }
    });

    //****************
    //
    // End of data copying
    //
    //****************

    // GMLS operator
    GMLS my_GMLS(_parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"),
                 2 /* dimension */,
                 "QR", "STANDARD", "NO_CONSTRAINT");
    my_GMLS.setProblemData(kokkos_neighbor_lists_host,
                    kokkos_augmented_source_coordinates_host,
                    kokkos_target_coordinates,
                    kokkos_epsilons_host);
    my_GMLS.setWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("weighting type"));
    my_GMLS.setWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("weighting power"));

    my_GMLS.addTargets(TargetOperation::ScalarPointEvaluation);
    my_GMLS.setAdditionalEvaluationSitesData(kokkos_quadrature_neighbor_lists_host, kokkos_quadrature_coordinates_host);
    my_GMLS.generateAlphas();

    // get maximum number of neighbors * fields[field_two]->nDim()
    //int team_scratch_size = host_scratch_vector_type::shmem_size(max_num_neighbors * fields[field_two]->nDim()); // values
    //team_scratch_size += host_scratch_local_index_type::shmem_size(max_num_neighbors * fields[field_two]->nDim()); // local column indices
    //const local_index_type host_scratch_team_level = 0; // not used in Kokkos currently

    //#pragma omp parallel for
//  for(local_index_type i = 0; i < nlocal; i++) {
//  Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,nlocal), KOKKOS_LAMBDA(const int i) {
    //Kokkos::parallel_for(host_team_policy(nlocal, Kokkos::AUTO).set_scratch_size(host_scratch_team_level,Kokkos::PerTeam(team_scratch_size)), [=](const host_member_type& teamMember) {



    // loop over cells
    //Kokkos::View<int*, Kokkos::HostSpace> row_seen("has row had dirichlet addition", target_coords->nLocal());
    host_vector_local_index_type row_seen("has row had dirichlet addition", target_coords->nLocal());
    Kokkos::deep_copy(row_seen,0);
    host_vector_local_index_type col_data("col data", 1);//particles_particles_max_num_neighbors*fields[field_two]->nDim());
    host_vector_scalar_type val_data("val data", 1);//particles_particles_max_num_neighbors*fields[field_two]->nDim());

    for (int i=0; i<_cells->getCoordsConst()->nLocal(); ++i) {
        // get all particle neighbors of cell i
        for (size_t j=0; j<cell_particles_all_neighbors[i].size(); ++j) {
            auto particle_j = cell_particles_all_neighbors[i][j].first;
            for (size_t k=0; k<cell_particles_all_neighbors[i].size(); ++k) {
                auto particle_k = cell_particles_all_neighbors[i][k].first;

                int k_relative_to_j = -1;
                int j_relative_to_k = -1;
                if (j==k) {
                    k_relative_to_j = 0;
                    j_relative_to_k = 0;
                } else {
                    // search 
                    for (size_t l=0; l<particles_particles_all_neighbors[particle_j].size(); ++l) {
                        if (particles_particles_all_neighbors[particle_j][l].first==particle_k) {
                            k_relative_to_j = l;
                            break;
                        }
                    }
                    for (size_t l=0; l<particles_particles_all_neighbors[particle_k].size(); ++l) {
                        if (particles_particles_all_neighbors[particle_k][l].first==particle_j) {
                            j_relative_to_k = l;
                            break;
                        }
                    }
                }

                int cell_i_for_particle_j = -1;
                int cell_i_for_particle_k = -1;
                for (int l=0; l<static_cast<int>(particle_cells_all_neighbors[particle_j].size()); ++l) {
                    if (particle_cells_all_neighbors[particle_j][l].first==i) {
                        cell_i_for_particle_j = l;    
                        break;
                    }
                }
                for (int l=0; l<static_cast<int>(particle_cells_all_neighbors[particle_k].size()); ++l) {
                    if (particle_cells_all_neighbors[particle_k][l].first==i) {
                        cell_i_for_particle_k = l;    
                        break;
                    }
                }


                // check distance between j and k
                auto diff = XYZ(kokkos_target_coordinates_host(particle_j,0)-kokkos_target_coordinates_host(particle_k,0),
                                kokkos_target_coordinates_host(particle_j,1)-kokkos_target_coordinates_host(particle_k,1),
                                kokkos_target_coordinates_host(particle_j,2)-kokkos_target_coordinates_host(particle_k,2));

                auto r = GMLS::EuclideanVectorLength(diff, 2);
                auto weight = GMLS::Wab(r, kokkos_epsilons_host(i), WeightingFunctionType::Power, _parameters->get<Teuchos::ParameterList>("remap").get<int>("weighting power"));

                if (weight>0 && k_relative_to_j>=0 && j_relative_to_k>=0 && cell_i_for_particle_j>=0 && cell_i_for_particle_k>=0) {


                    //Put the values of alpha in the proper place in the global matrix
                    //for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
                    local_index_type row = local_to_dof_map(particle_j, field_one, 0 /* component 0*/);
                    col_data(0) = local_to_dof_map(particle_k, field_two, 0 /* component */);
                    local_index_type corresponding_particle_id = row;
                    if (bc_id(corresponding_particle_id, 0) != 0) {
                        if (row_seen(corresponding_particle_id)==0) {
                            if (particle_j == particle_k) {
                                val_data(0) = 1.0;
                                row_seen(corresponding_particle_id) = 1;
                            } else {
                                val_data(0) = 0.0;
                            }
                        } else {
                            val_data(0) = 0;
                        }
                    } else {
                        // loop over quadrature
                        double contribution = 0;
                        for (int q=0; q<weights_ndim; ++q) {
                            if (quadrature_type(i,q)==1) { // interior
                                int extra_site_num_for_j = cell_i_for_particle_j*weights_ndim+q;
                                int extra_site_num_for_k = cell_i_for_particle_k*weights_ndim+q;;
                                contribution += quadrature_weights(i,q) * my_GMLS.getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, k, j_relative_to_k, extra_site_num_for_k) * my_GMLS.getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, j, k_relative_to_j, extra_site_num_for_j);
                            }
                        }
                        val_data(0) = contribution;
                        //if (particle_j==1) {
                        //   printf("i: %d, j: %lu, k: %lu, wt: %f, val: %f\n", i, particle_j, particle_k, weight, contribution);
                        //}
                    }
                    {
                        this->_A->sumIntoLocalValues(row, 1, val_data.data(), col_data.data());//, /*atomics*/false);
                    }
                }
            }
        }
    }

	TEUCHOS_ASSERT(!this->_A.is_null());
	ComputeMatrixTime->stop();

}

const std::vector<InteractingFields> AdvectionDiffusionPhysics::gatherFieldInteractions() {
	std::vector<InteractingFields> field_interactions;
	field_interactions.push_back(InteractingFields(op_needing_interaction::physics, _particles->getFieldManagerConst()->getIDOfFieldFromName("solution")));
	return field_interactions;
}
    
}
