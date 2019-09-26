#include <Compadre_GMLS_StaggeredAnalytic_PinnedLaplacian_Operator.hpp>

#include <Compadre_CoordsT.hpp>
#include <Compadre_ParticlesT.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_DOFManager.hpp>
#include <Compadre_FieldT.hpp>
#include <Compadre_NeighborhoodT.hpp>
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
typedef Compadre::XyzVector xyz_type;




Teuchos::RCP<crs_graph_type> GMLS_StaggeredAnalytic_LaplacianPhysics::computeGraph(local_index_type field_one, local_index_type field_two) {
	if (field_two == -1) {
		field_two = field_one;
	}

	Teuchos::RCP<Teuchos::Time> ComputeGraphTime = Teuchos::TimeMonitor::getNewCounter ("Compute Graph Time");
	ComputeGraphTime->start();
	TEUCHOS_TEST_FOR_EXCEPT_MSG(this->_A_graph.is_null(), "Tpetra CrsGraph for Physics not yet specified.");

	const local_index_type nlocal = static_cast<local_index_type>(this->_coords->nLocal());
	const std::vector<Teuchos::RCP<fields_type> >& fields = this->_particles->getFieldManagerConst()->getVectorOfFields();
	const neighborhood_type * neighborhood = this->_particles->getNeighborhoodConst();
    const local_dof_map_view_type local_to_dof_map = _dof_data->getDOFMap();

	//#pragma omp parallel for
	for(local_index_type i = 0; i < nlocal; i++) {
		local_index_type num_neighbors = neighborhood->getNeighbors(i).size();
		for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
			local_index_type row = local_to_dof_map(i, field_one, k);

			Teuchos::Array<local_index_type> col_data(num_neighbors * fields[field_two]->nDim());
			Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);
			std::vector<std::pair<size_t, scalar_type> > neighbors = neighborhood->getNeighbors(i);

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

void GMLS_StaggeredAnalytic_LaplacianPhysics::computeMatrix(local_index_type field_one, local_index_type field_two, scalar_type time) {
	Teuchos::RCP<Teuchos::Time> ComputeMatrixTime = Teuchos::TimeMonitor::getNewCounter ("Compute Matrix Time");
	ComputeMatrixTime->start();

	const local_index_type neighbors_needed = GMLS::getNP(Porder);

	bool use_physical_coords = true; // can be set on the operator in the future

	TEUCHOS_TEST_FOR_EXCEPT_MSG(this->_A.is_null(), "Tpetra CrsMatrix for Physics not yet specified.");

	//Loop over all particles, convert to GMLS data types, solve problem, and insert into matrix:

	const local_index_type nlocal = static_cast<local_index_type>(this->_coords->nLocal());
	const std::vector<Teuchos::RCP<fields_type> >& fields = this->_particles->getFieldManagerConst()->getVectorOfFields();
	const neighborhood_type * neighborhood = this->_particles->getNeighborhoodConst();
    const local_dof_map_view_type local_to_dof_map = _dof_data->getDOFMap();
	const host_view_type bc_id = this->_particles->getFlags()->getLocalView<host_view_type>();



	//****************
	//
	//  Copying data from particles (std::vector's, multivectors, etc....) to views used by local reconstruction class
	//
	//****************

	// generate the interpolation operator and call the coefficients needed (storing them)
	const coords_type* target_coords = this->_coords;
	const coords_type* source_coords = this->_coords;

	const std::vector<std::vector<std::pair<size_t, scalar_type> > >& all_neighbors = neighborhood->getAllNeighbors();

	size_t max_num_neighbors = 0;
	Kokkos::parallel_reduce(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()),
			KOKKOS_LAMBDA (const int i, size_t &myVal) {
		myVal = (all_neighbors[i].size() > myVal) ? all_neighbors[i].size() : myVal;
	}, Kokkos::Experimental::Max<size_t>(max_num_neighbors));

	Kokkos::View<int**> kokkos_neighbor_lists("neighbor lists", target_coords->nLocal(), max_num_neighbors+1);
	Kokkos::View<int**>::HostMirror kokkos_neighbor_lists_host = Kokkos::create_mirror_view(kokkos_neighbor_lists);

	// fill in the neighbor lists into a kokkos view. First entry is # of neighbors for that target
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
		const int num_i_neighbors = all_neighbors[i].size();
		for (int j=1; j<num_i_neighbors+1; ++j) {
			kokkos_neighbor_lists_host(i,j) = all_neighbors[i][j-1].first;
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

	auto epsilons = neighborhood->getHSupportSizes()->getLocalView<const host_view_type>();
	Kokkos::View<double*> kokkos_epsilons("target_coordinates", target_coords->nLocal(), target_coords->nDim());
	Kokkos::View<double*>::HostMirror kokkos_epsilons_host = Kokkos::create_mirror_view(kokkos_epsilons);
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
		kokkos_epsilons_host(i) = epsilons(i,0);
	});

	//****************
	//
	// End of data copying
	//
	//****************

	// GMLS operator

	GMLS my_GMLS (ReconstructionSpace::ScalarTaylorPolynomial,
                      StaggeredEdgeAnalyticGradientIntegralSample,
                      _parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"),
                      "QR", 0 /* manifold order */, 3 /* dimension */);
	my_GMLS.setProblemData(kokkos_neighbor_lists_host,
                               kokkos_augmented_source_coordinates_host,
                               kokkos_target_coordinates,
                               kokkos_epsilons_host);
	my_GMLS.setWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("weighting type"));
	my_GMLS.setWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("weighting power"));

	my_GMLS.addTargets(TargetOperation::DivergenceOfVectorPointEvaluation);
	my_GMLS.generateAlphas(); // just point evaluations

	Teuchos::RCP<Teuchos::Time> GMLSTime = Teuchos::TimeMonitor::getNewCounter ("GMLS");

	// get maximum number of neighbors * fields[field_two]->nDim()
	int team_scratch_size = host_scratch_vector_scalar_type::shmem_size(max_num_neighbors * fields[field_two]->nDim()); // values
	team_scratch_size += host_scratch_vector_local_index_type::shmem_size(max_num_neighbors * fields[field_two]->nDim()); // local column indices
	const local_index_type host_scratch_team_level = 0; // not used in Kokkos currently

	//#pragma omp parallel for
//	for(local_index_type i = 0; i < nlocal; i++) {
//	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,nlocal), KOKKOS_LAMBDA(const int i) {
	Kokkos::parallel_for(host_team_policy(nlocal, Kokkos::AUTO).set_scratch_size(host_scratch_team_level,Kokkos::PerTeam(team_scratch_size)), [=](const host_member_type& teamMember) {
		const int i = teamMember.league_rank();

		host_scratch_vector_local_index_type col_data(teamMember.team_scratch(host_scratch_team_level), max_num_neighbors*fields[field_two]->nDim());
		host_scratch_vector_scalar_type val_data(teamMember.team_scratch(host_scratch_team_level), max_num_neighbors*fields[field_two]->nDim());

		const std::vector<std::pair<size_t, scalar_type> > neighbors = neighborhood->getNeighbors(i);
		const local_index_type num_neighbors = neighbors.size();

		//Print error if there's not enough neighbors:
		TEUCHOS_TEST_FOR_EXCEPT_MSG(num_neighbors < neighbors_needed,
				"ERROR: Number of neighbors: " + std::to_string(num_neighbors) << " Neighbors needed: " << std::to_string(neighbors_needed) );

		//Put the values of alpha in the proper place in the global matrix
		for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
			local_index_type row = local_to_dof_map(i, field_one, k);

			for (local_index_type l = 0; l < num_neighbors; l++) {
				for (local_index_type n = 0; n < fields[field_two]->nDim(); ++n) {
					col_data(l*fields[field_two]->nDim() + n) = local_to_dof_map(static_cast<local_index_type>(neighbors[l].first), field_two, n);
					if (n==k) { // same field, same component
						// implicitly this is dof = particle#*ntotalfielddimension so this is just getting the particle number from dof
						// and checking its boundary condition
						local_index_type corresponding_particle_id = row/fields[field_one]->nDim();
						if (bc_id(corresponding_particle_id, 0) != 0) {
							if (i==static_cast<local_index_type>(neighbors[l].first)) {
								val_data(l*fields[field_two]->nDim() + n) = 1.0;
							} else {
								val_data(l*fields[field_two]->nDim() + n) = 0.0;
							}
						} else {
                                                    val_data(l*fields[field_two]->nDim() + n) = my_GMLS.getAlpha0TensorTo0Tensor(TargetOperation::DivergenceOfVectorPointEvaluation, i, l)*my_GMLS.getPreStencilWeight(StaggeredEdgeAnalyticGradientIntegralSample, i, l, false, 0, 0); // adding to the neighbour index
                                                    val_data(0) += my_GMLS.getAlpha0TensorTo0Tensor(TargetOperation::DivergenceOfVectorPointEvaluation, i, l)*my_GMLS.getPreStencilWeight(StaggeredEdgeAnalyticGradientIntegralSample, i, l, true, 0, 0); // adding to target index
                                                }
					} else {
                                            val_data(l*fields[field_two]->nDim() + n) = 0.0;
					}
				}
			}
			//#pragma omp critical
			{
				//this->_A->insertLocalValues(row, cols, values);
				this->_A->sumIntoLocalValues(row, num_neighbors * fields[field_two]->nDim(), val_data.data(), col_data.data());//, /*atomics*/false);
			}
		}
//	}
	});

	TEUCHOS_ASSERT(!this->_A.is_null());
//	this->_A->print(std::cout);
	ComputeMatrixTime->stop();

}

const std::vector<InteractingFields> GMLS_StaggeredAnalytic_LaplacianPhysics::gatherFieldInteractions() {
	std::vector<InteractingFields> field_interactions;
	field_interactions.push_back(InteractingFields(op_needing_interaction::physics, 0));
	return field_interactions;
}

}
