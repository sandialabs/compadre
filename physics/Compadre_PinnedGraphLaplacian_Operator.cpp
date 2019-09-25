#include <Compadre_PinnedGraphLaplacian_Operator.hpp>

#include <Compadre_CoordsT.hpp>
#include <Compadre_ParticlesT.hpp>
#include <Compadre_FieldT.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_DOFManager.hpp>
#include <Compadre_NeighborhoodT.hpp>

#ifdef COMPADREHARNESS_USE_OPENMP
#include <omp.h>
#endif

namespace Compadre {

typedef Compadre::CoordsT coords_type;
typedef Compadre::FieldT fields_type;
typedef Compadre::NeighborhoodT neighborhood_type;

Teuchos::RCP<crs_graph_type> PinnedGraphLaplacianPhysics::computeGraph(local_index_type field_one, local_index_type field_two) {
	Teuchos::RCP<Teuchos::Time> ComputeGraphTime = Teuchos::TimeMonitor::getNewCounter ("Compute Graph Time");
	ComputeGraphTime->start();
	TEUCHOS_TEST_FOR_EXCEPT_MSG(this->_A_graph.is_null(), "Tpetra CrsGraph for Physics not yet specified.");
	if (field_two == -1) {
		field_two = field_one;
	}

	const local_index_type nlocal = static_cast<local_index_type>(this->_coords->nLocal());

	const std::vector<Teuchos::RCP<fields_type> >& fields = this->_particles->getFieldManagerConst()->getVectorOfFields();
	const neighborhood_type * neighborhood = this->_particles->getNeighborhoodConst();
	const std::vector<std::vector<std::vector<local_index_type> > >& local_to_dof_map =
			_dof_data->getDOFMap();

	//#pragma omp parallel for
	for(local_index_type i = 0; i < nlocal; i++) {

		local_index_type num_neighbors = neighborhood->getNeighbors(i).size();
		for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
			local_index_type row = local_to_dof_map[i][field_one][k];

			Teuchos::Array<local_index_type> col_data(num_neighbors * fields[field_two]->nDim());
			Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);
			std::vector<std::pair<size_t, scalar_type> > neighbors = neighborhood->getNeighbors(i);

			for (local_index_type l = 0; l < num_neighbors; l++) {
				for (local_index_type n = 0; n < fields[field_two]->nDim(); ++n) {
					cols[l*fields[field_two]->nDim() + n] = local_to_dof_map[static_cast<local_index_type>(neighbors[l].first)][field_two][n];
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

void PinnedGraphLaplacianPhysics::computeMatrix(local_index_type field_one, local_index_type field_two, scalar_type time) {
	Teuchos::RCP<Teuchos::Time> ComputeMatrixTime = Teuchos::TimeMonitor::getNewCounter ("Compute Matrix Time");
	ComputeMatrixTime->start();
	TEUCHOS_TEST_FOR_EXCEPT_MSG(this->_A.is_null(), "Tpetra CrsMatrix for Physics not yet specified.");
	if (field_two == -1) {
		field_two = field_one;
	}

//			IDENTITY MATRIX
	// Fill the sparse matrix, one row at a time.
//			for (local_index_type row = 0; row < static_cast<local_index_type>(_coords->nLocal())*_particles->getTotalFieldDimensions(); ++row) {
//
//				Teuchos::Array<local_index_type> col_data(1);
//				Teuchos::Array<scalar_type> val_data(1);
//				Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);
//				Teuchos::ArrayView<scalar_type> values = Teuchos::ArrayView<scalar_type>(val_data);
//
//				if (_particles->getFlag(row / _particles->getTotalFieldDimensions()) != 0) {
//					// temporary pinning
//					values[0] = 1.0;
//					cols[0] = row;
//					//this->_A->sumIntoLocalValues(row, cols, values);
//					this->_A->insertLocalValues(row, cols, values);// important if a CrsGraph hasn't been constructed as a sparsity pattern
//				} else {
//					values[0] = 1.0;
//					cols[0] = row;
//					//this->_A->sumIntoLocalValues(row, cols, values);
//					this->_A->insertLocalValues(row, cols, values); // important if a CrsGraph hasn't been constructed as a sparsity pattern
//				}
//			}

//			GRAPH LAPLACIAN, no contribution on components and fields with no interactions

	const local_index_type nlocal = static_cast<local_index_type>(this->_coords->nLocal());
	const local_index_type ntotalfielddimensions = this->_particles->getFieldManagerConst()->getTotalFieldDimensions();
	const std::vector<Teuchos::RCP<fields_type> >& fields = this->_particles->getFieldManagerConst()->getVectorOfFields();
	const neighborhood_type * neighborhood = this->_particles->getNeighborhoodConst();
	const std::vector<std::vector<std::vector<local_index_type> > >& local_to_dof_map =
			_dof_data->getDOFMap();
	const host_view_type bc_id = this->_particles->getFlags()->getLocalView<host_view_type>();

	// get maximum number of neighbors * fields[field_two]->nDim()
	const local_index_type max_num_neighbors = neighborhood->getMaxNumNeighbors();
	int team_scratch_size = host_scratch_vector_scalar_type::shmem_size(max_num_neighbors * fields[field_two]->nDim()); // values
	team_scratch_size += host_scratch_vector_local_index_type::shmem_size(max_num_neighbors * fields[field_two]->nDim()); // local column indices
	const local_index_type host_scratch_team_level = 0; // not used in Kokkos currently

	//#pragma omp parallel for
	//for (local_index_type i = 0; i < nlocal; ++i) {  // i is particle, j is field, k is comp#
//	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,nlocal), KOKKOS_LAMBDA(const int i) {
	Kokkos::parallel_for(host_team_policy(nlocal, Kokkos::AUTO).set_scratch_size(host_scratch_team_level,Kokkos::PerTeam(team_scratch_size)), [=](const host_member_type& teamMember) {
		const int i = teamMember.league_rank();

		host_scratch_vector_local_index_type col_data(teamMember.team_scratch(host_scratch_team_level), max_num_neighbors*fields[field_two]->nDim());
		host_scratch_vector_scalar_type val_data(teamMember.team_scratch(host_scratch_team_level), max_num_neighbors*fields[field_two]->nDim());

		for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {

			local_index_type row = local_to_dof_map[i][field_one][k]; // get dof#
			//std::cout << "insert row : " << row << " for field " << j << " with dim " << k << " of " << this->_particles->getFieldByID(j)->nDim() << std::endl;
			//std::cout << "diagnostic: "<< this->_particles->getNeighborhood()->getNeighbors(i).size() << " " << this->_particles->getFieldByID(j)->nDim() << std::endl;

			const std::vector<std::pair<size_t, scalar_type> > neighbors = neighborhood->getNeighbors(i);

			//std::cout << "size(N) : " << neighbors.size() << std::endl;
			if (neighborhood->getNeighbors(i).size() == 0) std::cout << neighborhood->getNeighbors(i).size() << std::endl;
			for (size_t l = 0; l < neighbors.size(); l++) { // loop over neighbors
				// std::cout << l << " first : " << neighbors[l].first << " second : " << neighbors[l].second << std::endl;
				for (local_index_type n = 0; n < fields[field_two]->nDim(); ++n) { // loop over comp#'s
					//std::cout << n << " of " << this->_particles->getFieldByID(j)->nDim() << " is " << this->_particles->getDOFFromLID(static_cast<local_index_type>(neighbors[l].first),j,n) << std::endl;
						  // local offset, size_t to local_index_type conversion of neighbor number, j is for the same field, component number is n
					col_data(l*fields[field_two]->nDim() + n) = local_to_dof_map[static_cast<local_index_type>(neighbors[l].first)][field_two][n];
					//std::cout << "neighbor: " << static_cast<local_index_type>(neighbors[l].first) << std::endl;

					if (n==k) { // same field, same component
						// implicitly this is dof = particle#*ntotalfielddimension so this is just getting the particle number from dof
						// and checking its boundary condition
						local_index_type corresponding_particle_id = row/fields[field_one]->nDim();
						if (bc_id(corresponding_particle_id,0) != 0) {
							if (i==static_cast<local_index_type>(neighbors[l].first)) {
								val_data(l*fields[field_two]->nDim() + n) = 1.0;
							} else {
								val_data(l*fields[field_two]->nDim() + n) = 0.0;
							}
						} else {
							if (i==static_cast<local_index_type>(neighbors[l].first)) {
								val_data(l*fields[field_two]->nDim() + n) = -static_cast<double>(static_cast<int>(neighborhood->getNeighbors(i).size())-1);
							} else {
								val_data(l*fields[field_two]->nDim() + n) = 1.0;
							}
						}
					} else {
						val_data(l*fields[field_two]->nDim() + n) = 0.0;
					}

				}
			}
//						std::cout << "Fill data" << std::endl << "===========" << std::endl;
//						std::cout << row << std::endl;
//						for (int o=0; o<cols.size(); o++) std::cout << cols[o] << " ";
//						std::cout << std::endl;
//						for (int o=0; o<values.size(); o++) std::cout << values[o] << " ";
//						std::cout << std::endl;

//						auto gids = this->_particles->getColMap()->getMyGlobalIndices();
//						for (int o=0; o<gids.size(); o++) {
//							std::cout << gids[o] << std::endl;
//						}
//
//						auto gids = this->_particles->getRowMap()->getMyGlobalIndices();
//						for (int o=0; o<gids.size(); o++) {
//							std::cout << gids[o] << std::endl;
//						}
			//std::cout << this_->particles->getRowMap()->
//						std::cout << "right before insertion." << std::endl;

//						int num_found = 0;
//						for (int o=0; o<values.size(); o++) {
//
//						}
			//this->_A->insertLocalValues(row, cols, values);
			//#pragma omp critical
			{
				this->_A->sumIntoLocalValues(row, neighbors.size()*fields[field_two]->nDim(), val_data.data(), col_data.data());
			}
		}
	});


////			GRAPH LAPLACIAN, zeros on components and fields with no interactions
//			for (local_index_type i = 0; i < static_cast<local_index_type>(_coords->nLocal()); ++i) {
//				for (local_index_type j = 0; j < _particles->getVectorOfFields().size(); ++j) {
//					for (local_index_type k = 0; k < _particles->getFieldByID(j)->nDim(); ++k) {
//
//						local_index_type row = _particles->getDOFFromLID(i,j,k);
//
//						Teuchos::Array<local_index_type> col_data(_particles->getNeighbors(i).size() * _particles->getTotalFieldDimensions());
//						Teuchos::Array<scalar_type> val_data(_particles->getNeighbors(i).size() * _particles->getTotalFieldDimensions());
//						Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);
//						Teuchos::ArrayView<scalar_type> values = Teuchos::ArrayView<scalar_type>(val_data);
//						std::vector<std::pair<size_t, scalar_type> > neighbors = _particles->getNeighbors(i);
//						if (_particles->getNeighbors(i).size() == 0) std::cout << _particles->getNeighbors(i).size() << std::endl;
//						for (local_index_type l = 0; l < neighbors.size(); l++) {
//							for (local_index_type m = 0; m < _particles->getVectorOfFields().size(); ++m) {
//								for (local_index_type n = 0; n < _particles->getFieldByID(m)->nDim(); ++n) {
//									cols[l*_particles->getTotalFieldDimensions() + _particles->getFieldOffset(m) + n] = _particles->getDOFFromLID(static_cast<local_index_type>(neighbors[l].first),m,n);
//									if (m==j && n==k) { // same field, same field
//										if (_particles->getFlag(row / _particles->getTotalFieldDimensions()) != 0) {
//											if (i==static_cast<local_index_type>(neighbors[l].first)) {
//												values[l*_particles->getTotalFieldDimensions() + _particles->getFieldOffset(m) + n] = 1.0;
//											} else {
//												values[l*_particles->getTotalFieldDimensions() + _particles->getFieldOffset(m) + n] = 0.0;
//											}
//										} else {
//											if (i==static_cast<local_index_type>(neighbors[l].first)) {
//												values[l*_particles->getTotalFieldDimensions() + _particles->getFieldOffset(m) + n] = -static_cast<double>(static_cast<int>(_particles->getNeighbors(i).size())-1);
//											} else {
//												values[l*_particles->getTotalFieldDimensions() + _particles->getFieldOffset(m) + n] = 1.0;
//											}
//										}
//									} else {
//										values[l*_particles->getTotalFieldDimensions() + _particles->getFieldOffset(m) + n] = 0.0;
//									}
//
//								}
//							}
//						}
//						this->_A->insertLocalValues(row, cols, values);
//					}
//				}
//			}


	TEUCHOS_ASSERT(!this->_A.is_null());
	ComputeMatrixTime->stop();
	//_A->print(std::cout);
}

const std::vector<InteractingFields> PinnedGraphLaplacianPhysics::gatherFieldInteractions() {
	std::vector<InteractingFields> field_interactions;
	field_interactions.push_back(InteractingFields(op_needing_interaction::physics, 0));
	field_interactions.push_back(InteractingFields(op_needing_interaction::physics, 1));
	field_interactions.push_back(InteractingFields(op_needing_interaction::physics, 2));
	field_interactions.push_back(InteractingFields(op_needing_interaction::physics, 3));
	field_interactions.push_back(InteractingFields(op_needing_interaction::physics, 4));
	return field_interactions;
}

}
