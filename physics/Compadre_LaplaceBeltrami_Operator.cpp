#include <Compadre_LaplaceBeltrami_Operator.hpp>

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

#include <Compadre_AnalyticFunctions.hpp>

/*
 Constructs the matrix operator.
 
 */
namespace Compadre {

typedef Compadre::CoordsT coords_type;
typedef Compadre::FieldT fields_type;
typedef Compadre::NeighborhoodT neighborhood_type;
typedef Compadre::XyzVector xyz_type;

/*
 * 
 *
 *  Fixup idea: only one of the processors has the DOF 0 for LM, we must identify this a priori or just pick it
 *  then we need each processor to fill in their portion of this row, but they may not even have local (so they can't sum into local, because it isn't local)
 *  Normally, we use the halo to determine the column maps
 *
 */ 


Kokkos::View<size_t*, Kokkos::HostSpace> LaplaceBeltramiPhysics::getMaxEntriesPerRow(local_index_type field_one, local_index_type field_two) {

    auto comm = this->_particles->getCoordsConst()->getComm();

	const local_index_type nlocal = static_cast<local_index_type>(this->_coords->nLocal());
    Kokkos::View<size_t*, Kokkos::HostSpace> maxEntriesPerRow("max entries per row", nlocal);

    auto solution_field_id = _particles->getFieldManagerConst()->getIDOfFieldFromName("solution");
    auto lm_field_id = _particles->getFieldManagerConst()->getIDOfFieldFromName("lagrange multiplier");

	const neighborhood_type * neighborhood = this->_particles->getNeighborhoodConst();
	const std::vector<Teuchos::RCP<fields_type> >& fields = this->_particles->getFieldManagerConst()->getVectorOfFields();

    if (field_one == solution_field_id && field_two == solution_field_id) {
        for(local_index_type i = 0; i < nlocal; i++) {
            maxEntriesPerRow(i) = fields[field_one]->nDim()*fields[field_two]->nDim()*neighborhood->getNumNeighbors(i);
        }
    } else if (field_one == lm_field_id && field_two == solution_field_id) {
        auto row_map_entries = _row_map->getMyGlobalIndices();
        Kokkos::resize(maxEntriesPerRow, row_map_entries.extent(0));
        Kokkos::deep_copy(maxEntriesPerRow, 0);
        if (comm->getRank()==0) {
            maxEntriesPerRow(0) = nlocal*fields[field_two]->nDim();
        } else {
            maxEntriesPerRow(row_map_entries.extent(0)-1) = nlocal*fields[field_two]->nDim();
        }
        // write to last entry
    } else if (field_one == solution_field_id && field_two == lm_field_id) {
        for(local_index_type i = 0; i < nlocal; i++) {
            maxEntriesPerRow(i) = fields[field_two]->nDim();
        }
    } else {
        for(local_index_type i = 0; i < nlocal; i++) {
            maxEntriesPerRow(i) = fields[field_two]->nDim();
        }
    }

    return maxEntriesPerRow;
}

Teuchos::RCP<crs_graph_type> LaplaceBeltramiPhysics::computeGraph(local_index_type field_one, local_index_type field_two) {

	if (field_two == -1) {
		field_two = field_one;
	}

    auto solution_field_id = _particles->getFieldManagerConst()->getIDOfFieldFromName("solution");
    auto lm_field_id = _particles->getFieldManagerConst()->getIDOfFieldFromName("lagrange multiplier");

    TEUCHOS_TEST_FOR_EXCEPT_MSG(_row_map.is_null(), "Row map used for construction of CrsGraph before being initialized.");
    TEUCHOS_TEST_FOR_EXCEPT_MSG(_col_map.is_null(), "Column map used for construction of CrsGraph before being initialized.");

    if (field_one == lm_field_id && field_two == solution_field_id) {
        // row all DOFs for solution against Lagrange Multiplier

        // create a new graph from the existing one, but with an updated row map augmented by a single global dof
        // for the lagrange multiplier
        auto existing_row_map = _row_map;
        auto existing_col_map = _col_map;

        size_t max_entries_per_row;
        Teuchos::ArrayRCP<const size_t> empty_array;
        bool bound_same_for_all_local_rows = true;
        this->_A_graph->getNumEntriesPerLocalRowUpperBound(empty_array, max_entries_per_row, bound_same_for_all_local_rows);

        auto row_map_index_base = existing_row_map->getIndexBase();
        auto row_map_entries = existing_row_map->getMyGlobalIndices();

        auto comm = this->_particles->getCoordsConst()->getComm();
        const int offset_size = (comm->getRank() == 0) ? 0 : 1;
        Kokkos::View<global_index_type*> new_row_map_entries("", row_map_entries.extent(0)+offset_size);

        for (size_t i=0; i<row_map_entries.extent(0); ++i) {
            //new_row_map_entries(i+offset_size) = row_map_entries(i);
            new_row_map_entries(i) = row_map_entries(i);
        }

        {
            // exchange information between processors to get first index on processor 0
            global_index_type global_index_proc_0_element_0;
            if (comm->getRank()==0) {
                global_index_proc_0_element_0 = row_map_entries(0);
            }

            // broadcast from processor 0 to other ranks
            Teuchos::broadcast<local_index_type, global_index_type>(*comm, 0 /*processor broadcasting*/, 
                    1 /*size*/, &global_index_proc_0_element_0);

            if (comm->getRank() > 0) {
                // now the first entry in the map is to the shared gid for the LM
                new_row_map_entries(new_row_map_entries.extent(0)-1) = global_index_proc_0_element_0;
            }

        }

        auto new_row_map = Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
                										new_row_map_entries,
                										row_map_index_base,
                										this->_particles->getCoordsConst()->getComm()));

        this->setRowMap(new_row_map);

        auto entries = this->getMaxEntriesPerRow(field_one, field_two);
        auto dual_view_entries = Kokkos::DualView<size_t*>("dual view", entries.extent(0));
        auto host_view_entries = dual_view_entries.h_view;
        Kokkos::deep_copy(host_view_entries, entries);
        dual_view_entries.modify<Kokkos::DefaultHostExecutionSpace>();
        dual_view_entries.sync<Kokkos::DefaultExecutionSpace>();
        _A_graph = Teuchos::rcp(new crs_graph_type (_row_map, _col_map, dual_view_entries, Tpetra::StaticProfile));
    } else if (field_one == solution_field_id && field_two == lm_field_id) {

        // create a new graph from the existing one, but with an updated col map augmented by a single global dof
        // for the lagrange multiplier
        auto existing_row_map = _row_map;
        auto existing_col_map = _col_map;

        size_t max_entries_per_row;
        Teuchos::ArrayRCP<const size_t> empty_array;
        bool bound_same_for_all_local_rows = true;
        this->_A_graph->getNumEntriesPerLocalRowUpperBound(empty_array, max_entries_per_row, bound_same_for_all_local_rows);

        auto col_map_index_base = existing_col_map->getIndexBase();
        auto col_map_entries = existing_col_map->getMyGlobalIndices();
        auto row_map_entries = existing_row_map->getMyGlobalIndices();

        auto comm = this->_particles->getCoordsConst()->getComm();
        const int offset_size = (comm->getRank() == 0) ? 0 : 1;
        Kokkos::View<global_index_type*> new_col_map_entries("", col_map_entries.extent(0)+offset_size);

        for (size_t i=0; i<col_map_entries.extent(0); ++i) {
            //new_col_map_entries(i+offset_size) = col_map_entries(i);
            new_col_map_entries(i) = col_map_entries(i);
        }

        {
            // exchange information between processors to get first index on processor 0
            global_index_type global_index_proc_0_element_0;
            if (comm->getRank()==0) {
                global_index_proc_0_element_0 = col_map_entries(0);
            }

            // broadcast from processor 0 to other ranks
            Teuchos::broadcast<local_index_type, global_index_type>(*comm, 0 /*processor broadcasting*/, 
                    1 /*size*/, &global_index_proc_0_element_0);

            if (comm->getRank() > 0) {
                // now the first entry in the map is to the shared gid for the LM
                new_col_map_entries(new_col_map_entries.extent(0)-1) = global_index_proc_0_element_0;
            }

        }

        auto new_col_map = Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
                										new_col_map_entries,
                										col_map_index_base,
                										this->_particles->getCoordsConst()->getComm()));

        this->setColMap(new_col_map);

        auto entries = this->getMaxEntriesPerRow(field_one, field_two);
        auto dual_view_entries = Kokkos::DualView<size_t*>("dual view", entries.extent(0));
        auto host_view_entries = dual_view_entries.h_view;
        Kokkos::deep_copy(host_view_entries, entries);
        dual_view_entries.modify<Kokkos::DefaultHostExecutionSpace>();
        dual_view_entries.sync<Kokkos::DefaultExecutionSpace>();
        _A_graph = Teuchos::rcp(new crs_graph_type (_row_map, _col_map, dual_view_entries, Tpetra::StaticProfile));

    }
 

    if (_A_graph.is_null()) {
        auto entries = this->getMaxEntriesPerRow(field_one, field_two);
        auto dual_view_entries = Kokkos::DualView<size_t*>("dual view", entries.extent(0));
        auto host_view_entries = dual_view_entries.h_view;
        Kokkos::deep_copy(host_view_entries, entries);
        dual_view_entries.modify<Kokkos::DefaultHostExecutionSpace>();
        dual_view_entries.sync<Kokkos::DefaultExecutionSpace>();
        _A_graph = Teuchos::rcp(new crs_graph_type (_row_map, _col_map, dual_view_entries, Tpetra::StaticProfile));
    }


    if (!_A_graph->isFillActive()) _A_graph->resumeFill();


    // check that the solver is in 'blocked' mode
    TEUCHOS_TEST_FOR_EXCEPT_MSG(_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==false, 
            "Non-blocked matrix system incompatible with Lagrange multiplier based Laplace-Beltrami solve.");

	Teuchos::RCP<Teuchos::Time> ComputeGraphTime = Teuchos::TimeMonitor::getNewCounter ("Compute Graph Time");
	ComputeGraphTime->start();
	//TEUCHOS_TEST_FOR_EXCEPT_MSG(this->_A_graph.is_null(), "Tpetra CrsGraph for Physics not yet specified.");

	const local_index_type nlocal = static_cast<local_index_type>(this->_coords->nLocal());
	const std::vector<Teuchos::RCP<fields_type> >& fields = this->_particles->getFieldManagerConst()->getVectorOfFields();
	const neighborhood_type * neighborhood = this->_particles->getNeighborhoodConst();
    const local_dof_map_view_type local_to_dof_map = _dof_data->getDOFMap();


    if (field_one == solution_field_id && field_two == solution_field_id) {
		//#pragma omp parallel for
		for(local_index_type i = 0; i < nlocal; i++) {
			local_index_type num_neighbors = neighborhood->getNumNeighbors(i);
			for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
				local_index_type row = local_to_dof_map(i, field_one, k);

				Teuchos::Array<local_index_type> col_data(num_neighbors * fields[field_two]->nDim());
				Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);

				for (local_index_type l = 0; l < num_neighbors; l++) {
					for (local_index_type n = 0; n < fields[field_two]->nDim(); ++n) {
						cols[l*fields[field_two]->nDim() + n] = local_to_dof_map(neighborhood->getNeighbor(i,l), field_two, n);
					}
				}
				//#pragma omp critical
				{
					this->_A_graph->insertLocalIndices(row, cols);
				}
			}
		}
    } else if (field_one == lm_field_id && field_two == solution_field_id) {
        // row all DOFs for solution against Lagrange Multiplier
		Teuchos::Array<local_index_type> col_data(nlocal * fields[field_two]->nDim());
		Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);

		for (local_index_type l = 0; l < nlocal; l++) {
			for (local_index_type n = 0; n < fields[field_two]->nDim(); ++n) {
				cols[l*fields[field_two]->nDim() + n] = local_to_dof_map(l, field_two, n);
			}
		}
        // local index 0 is global index shared by all processors
        //local_index_type row = local_to_dof_map[0][field_one][0];
		//this->_A_graph->insertLocalIndices(0, cols);

        auto comm = this->_particles->getCoordsConst()->getComm();
        if (comm->getRank() == 0) {
            // local index 0 is the shared global id for the lagrange multiplier
            this->_A_graph->insertLocalIndices(0, cols);
        } else {
            // local index 0 is the shared global id for the lagrange multiplier
            this->_A_graph->insertLocalIndices(nlocal, cols);
        }

    } else if (field_one == solution_field_id && field_two == lm_field_id) {
        // col all DOFs for solution against Lagrange Multiplier
        auto comm = this->_particles->getCoordsConst()->getComm();
        //if (comm->getRank() == 0) {
		    for(local_index_type i = 0; i < nlocal; i++) {
		    	for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
		    		local_index_type row = local_to_dof_map(i, field_one, k);

		    		Teuchos::Array<local_index_type> col_data(1);
		    		Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);
                    if (comm->getRank() == 0) {
		    		    cols[0] = 0; // local index 0 is global shared index
                    } else {
		    		    cols[0] = nlocal; // local index 0 is global shared index
                    }
		    		this->_A_graph->insertLocalIndices(row, cols);
		    	}
		    }
        //}
    } else {
        // identity on all DOFs for Lagrange Multiplier (even DOFs not really used, since we only use first LM dof for now)
		for(local_index_type i = 0; i < nlocal; i++) {
			for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
				local_index_type row = local_to_dof_map(i, field_one, k);

				Teuchos::Array<local_index_type> col_data(fields[field_two]->nDim());
				Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);
				for (local_index_type n = 0; n < fields[field_two]->nDim(); ++n) {
					cols[n] = local_to_dof_map(i, field_two, n);
				}
				//#pragma omp critical
				{
					this->_A_graph->insertLocalIndices(row, cols);
				}
			}
		}
    }
	ComputeGraphTime->stop();
    return _A_graph;
}

void LaplaceBeltramiPhysics::computeMatrix(local_index_type field_one, local_index_type field_two, scalar_type time) {
	Teuchos::RCP<Teuchos::Time> ComputeMatrixTime = Teuchos::TimeMonitor::getNewCounter ("Compute Matrix Time");
	ComputeMatrixTime->start();

	bool use_physical_coords = true; // can be set on the operator in the future

	TEUCHOS_TEST_FOR_EXCEPT_MSG(this->_A.is_null(), "Tpetra CrsMatrix for Physics not yet specified.");

	//Loop over all particles, convert to GMLS data types, solve problem, and insert into matrix:

	const local_index_type nlocal = static_cast<local_index_type>(this->_coords->nLocal());
	const global_index_type nglobal = this->_coords->nGlobal();
	const std::vector<Teuchos::RCP<fields_type> >& fields = this->_particles->getFieldManagerConst()->getVectorOfFields();
	const neighborhood_type * neighborhood = this->_particles->getNeighborhoodConst();
    const local_dof_map_view_type local_to_dof_map = _dof_data->getDOFMap();
	const host_view_local_index_type bc_id = this->_particles->getFlags()->getLocalView<host_view_local_index_type>();

    auto solution_field_id = _particles->getFieldManagerConst()->getIDOfFieldFromName("solution");
    auto lm_field_id = _particles->getFieldManagerConst()->getIDOfFieldFromName("lagrange multiplier");

if (field_one == solution_field_id && field_two == solution_field_id) {

	//****************
	//
	//  Copying data from particles (std::vector's, multivectors, etc....) to views used by local reconstruction class
	//
	//****************
    
	const local_index_type neighbors_needed = GMLS::getNP(Porder, 2);

	// generate the interpolation operator and call the coefficients needed (storing them)
	const coords_type* target_coords = this->_coords;
	const coords_type* source_coords = this->_coords;

	size_t max_num_neighbors = neighborhood->computeMaxNumNeighbors(false /* local processor max*/);

	Kokkos::View<int**> kokkos_neighbor_lists("neighbor lists", target_coords->nLocal(), max_num_neighbors+1);
	Kokkos::View<int**>::HostMirror kokkos_neighbor_lists_host = Kokkos::create_mirror_view(kokkos_neighbor_lists);

	// fill in the neighbor lists into a kokkos view. First entry is # of neighbors for that target
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
		const int num_i_neighbors = neighborhood->getNumNeighbors(i);
		for (int j=1; j<num_i_neighbors+1; ++j) {
			kokkos_neighbor_lists_host(i,j) = neighborhood->getNeighbor(i,j-1);
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
	Kokkos::View<double*> kokkos_epsilons("epsilons", target_coords->nLocal());
	Kokkos::View<double*>::HostMirror kokkos_epsilons_host = Kokkos::create_mirror_view(kokkos_epsilons);
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
		kokkos_epsilons_host(i) = epsilons(i,0);
	});

	//****************
	//
	// End of data copying
	//
	//****************

	Teuchos::RCP<Teuchos::Time> GMLSTime = Teuchos::TimeMonitor::getNewCounter("GMLS");

	Teuchos::RCP<Compadre::FiveStripOnSphere> fsos = Teuchos::rcp(new Compadre::FiveStripOnSphere);

	//#pragma omp parallel for
//	for(local_index_type i = 0; i < nlocal; i++) {

//	if (_parameters->get<std::string>("solution type")=="lb solve") { // Traditional Laplace-Beltrami
//
//		// GMLS operator
//
//		GMLS my_GMLS (kokkos_neighbor_lists_host,
//				kokkos_augmented_source_coordinates_host,
//				kokkos_target_coordinates,
//				kokkos_epsilons_host,
//				_parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"),
//				_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("dense linear solver"),
//				_parameters->get<Teuchos::ParameterList>("remap").get<int>("curvature porder"));
//
//		my_GMLS.setWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("weighting type"));
//		my_GMLS.setWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("weighting power"));
//		my_GMLS.setCurvatureWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("curvature weighting type"));
//		my_GMLS.setCurvatureWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("curvature weighting power"));
//
//		my_GMLS.addTargets(TargetOperation::LaplacianOfScalarPointEvaluation);
//		my_GMLS.generateAlphas(); // just point evaluations
//
//		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,nlocal), KOKKOS_LAMBDA(const int i) {
//
//			const local_index_type num_neighbors = neighbors.size();
//
//			//Print error if there's not enough neighbors:
//			TEUCHOS_TEST_FOR_EXCEPT_MSG(num_neighbors < neighbors_needed,
//					"ERROR: Number of neighbors: " + std::to_string(num_neighbors) << " Neighbors needed: " << std::to_string(neighbors_needed) );
//
//			//Put the values of alpha in the proper place in the global matrix
//			for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
//				local_index_type row = local_to_dof_map[i][field_one][k];
//
//				Teuchos::Array<local_index_type> col_data(num_neighbors * fields[field_two]->nDim());
//				Teuchos::Array<scalar_type> val_data(num_neighbors * fields[field_two]->nDim());
//				Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);
//				Teuchos::ArrayView<scalar_type> values = Teuchos::ArrayView<scalar_type>(val_data);
//
//				for (local_index_type l = 0; l < num_neighbors; l++) {
//					for (local_index_type n = 0; n < fields[field_two]->nDim(); ++n) {
//						cols[l*fields[field_two]->nDim() + n] = local_to_dof_map[neighborhood->getNeighbor(i,l)][field_two][n];
//	//					local_index_type corresponding_particle_id = blocked_matrix ? row/fields[field_one]->nDim() : row/ntotalfielddimensions;
//	//					if (bc_id(corresponding_particle_id, 0) != 0) {
//	//						if (i==neighborhood->getNeighbor(i,l)) {
//	//							values[l*fields[field_two]->nDim() + n] = 1.0;
//	//							printf("put 1 at: %d,%d\n", row, cols[l*fields[field_two]->nDim() + n]);
//	//						}
//	//					}
//						if (n==k) { // same field, same component
//							// implicitly this is dof = particle#*ntotalfielddimension so this is just getting the particle number from dof
//							// and checking its boundary condition
//							if (bc_id(i, 0) != 0) {
//								if (i==neighborhood->getNeighbor(i,l)) {
//									values[l*fields[field_two]->nDim() + n] = 1.0;
//								} else {
//									values[l*fields[field_two]->nDim() + n] = 0.0;
//								}
//							} else {
//								if (i==neighborhood->getNeighbor(i,l)) {
//									values[l*fields[field_two]->nDim() + n] = my_GMLS.getAlpha0TensorTo0Tensor(TargetOperation::LaplacianOfScalarPointEvaluation, i, l);
//								} else {
//									values[l*fields[field_two]->nDim() + n] = my_GMLS.getAlpha0TensorTo0Tensor(TargetOperation::LaplacianOfScalarPointEvaluation, i, l);
//								}
//							}
//						} else {
//							values[l*fields[field_two]->nDim() + n] = 0.0;
//						}
//					}
//				}
//				//#pragma omp critical
//				{
//					//this->_A->insertLocalValues(row, cols, values);
//					this->_A->sumIntoLocalValues(row, cols, values);//, /*atomics*/false);
//				}
//			}
//	//	}
//		});
	if (_parameters->get<std::string>("solution type")=="lb solve") { // Staggered Laplace-Beltrami
		// GMLS operator
                GMLS my_GMLS(ReconstructionSpace::ScalarTaylorPolynomial,
                             StaggeredEdgeAnalyticGradientIntegralSample,
                             StaggeredEdgeAnalyticGradientIntegralSample,
                             _parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"),
                             target_coords->nDim(),
                             _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("dense solver type"),
                             _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("problem type"),
                             _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("constraint type"),
                             _parameters->get<Teuchos::ParameterList>("remap").get<int>("curvature porder"));

//		GMLS my_GMLS (ReconstructionSpace::VectorTaylorPolynomial,
//				StaggeredEdgeIntegralSample,
//				StaggeredEdgeAnalyticGradientIntegralSample,
//				_parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"),
//				_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("dense linear solver"),
//				_parameters->get<Teuchos::ParameterList>("remap").get<int>("curvature porder"));
		my_GMLS.setProblemData(
				kokkos_neighbor_lists_host,
				kokkos_augmented_source_coordinates_host,
				kokkos_target_coordinates,
				kokkos_epsilons_host);
		my_GMLS.setWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("weighting type"));
		my_GMLS.setWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("weighting power"));
		my_GMLS.setCurvatureWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("curvature weighting type"));
		my_GMLS.setCurvatureWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("curvature weighting power"));
		my_GMLS.setOrderOfQuadraturePoints(_parameters->get<Teuchos::ParameterList>("remap").get<int>("quadrature order"));
		my_GMLS.setDimensionOfQuadraturePoints(_parameters->get<Teuchos::ParameterList>("remap").get<int>("quadrature dimension"));
		my_GMLS.setQuadratureType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("quadrature type"));

		my_GMLS.addTargets(TargetOperation::ChainedStaggeredLaplacianOfScalarPointEvaluation);
//		my_GMLS.addTargets(TargetOperation::DivergenceOfVectorPointEvaluation);
		my_GMLS.generateAlphas(); // just point evaluations

		// get maximum number of neighbors * fields[field_two]->nDim()
		int team_scratch_size = host_scratch_vector_scalar_type::shmem_size(max_num_neighbors * fields[field_two]->nDim()); // values
		team_scratch_size += host_scratch_vector_local_index_type::shmem_size(max_num_neighbors * fields[field_two]->nDim()); // local column indices
		const local_index_type host_scratch_team_level = 0; // not used in Kokkos currently

//		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,nlocal), KOKKOS_LAMBDA(const int i) {
		Kokkos::parallel_for(host_team_policy(nlocal, Kokkos::AUTO).set_scratch_size(host_scratch_team_level,Kokkos::PerTeam(team_scratch_size)), [=](const host_member_type& teamMember) {
			const int i = teamMember.league_rank();

			host_scratch_vector_local_index_type col_data(teamMember.team_scratch(host_scratch_team_level), max_num_neighbors*fields[field_two]->nDim());
			host_scratch_vector_scalar_type val_data(teamMember.team_scratch(host_scratch_team_level), max_num_neighbors*fields[field_two]->nDim());

			scalar_type target_coeff = fsos->evalDiffusionCoefficient(target_coords->getLocalCoords(i, false));

			local_index_type num_neighbors = neighborhood->getNumNeighbors(i);


			//Print error if there's not enough neighbors:
			TEUCHOS_TEST_FOR_EXCEPT_MSG(num_neighbors < neighbors_needed,
					"ERROR: Number of neighbors: " + std::to_string(num_neighbors) << " Neighbors needed: " << std::to_string(neighbors_needed) );

			//Put the values of alpha in the proper place in the global matrix
			for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
				local_index_type row = local_to_dof_map(i, field_one, k);

				for (local_index_type l = 0; l < num_neighbors; l++) {

					scalar_type avg_coeff = 1;
					if (_physics_type==3) {
						scalar_type neighbor_coeff = fsos->evalDiffusionCoefficient(source_coords->getLocalCoords(neighborhood->getNeighbor(i,l), true));
						avg_coeff = 0.5*(target_coeff + neighbor_coeff);
					}

					for (local_index_type n = 0; n < fields[field_two]->nDim(); ++n) {
						col_data(l*fields[field_two]->nDim() + n) = local_to_dof_map(neighborhood->getNeighbor(i,l), field_two, n);
						if (n==k) { // same field, same component
							// implicitly this is dof = particle#*ntotalfielddimension so this is just getting the particle number from dof
							// and checking its boundary condition
							if (bc_id(i, 0) != 0) {
								if (i==neighborhood->getNeighbor(i,l)) {
									val_data(l*fields[field_two]->nDim() + n) = 1.0;
								} else {
									val_data(l*fields[field_two]->nDim() + n) = 0.0;
								}
							} else {
								val_data(l*fields[field_two]->nDim() + n) = avg_coeff * my_GMLS.getAlpha0TensorTo0Tensor(TargetOperation::ChainedStaggeredLaplacianOfScalarPointEvaluation, i, l) * my_GMLS.getPreStencilWeight(StaggeredEdgeAnalyticGradientIntegralSample, i, l, false, 0 /*output component*/, 0 /*input component*/);
								val_data(0*fields[field_two]->nDim() + n) += avg_coeff * my_GMLS.getAlpha0TensorTo0Tensor(TargetOperation::ChainedStaggeredLaplacianOfScalarPointEvaluation, i, l) * my_GMLS.getPreStencilWeight(StaggeredEdgeAnalyticGradientIntegralSample, i, l, true, 0 /*output component*/, 0 /*input component*/);
//								val_data(l*fields[field_two]->nDim() + n) = avg_coeff * my_GMLS.getAlpha0TensorTo0Tensor(TargetOperation::DivergenceOfVectorPointEvaluation, i, l) * my_GMLS.getPreStencilWeight(StaggeredEdgeAnalyticGradientIntegralSample, i, l, false, 0 /*output component*/, 0 /*input component*/);
//								val_data(0*fields[field_two]->nDim() + n) += avg_coeff * my_GMLS.getAlpha0TensorTo0Tensor(TargetOperation::DivergenceOfVectorPointEvaluation, i, l) * my_GMLS.getPreStencilWeight(StaggeredEdgeAnalyticGradientIntegralSample, i, l, true, 0 /*output component*/, 0 /*input component*/);
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
	} else if (_parameters->get<std::string>("solution type")!="lb solve" && (_physics_type==1 || _physics_type==3)) {
		// GMLS operator
                GMLS my_GMLS(ReconstructionSpace::ScalarTaylorPolynomial,
                             StaggeredEdgeAnalyticGradientIntegralSample,
                             StaggeredEdgeAnalyticGradientIntegralSample,
                             _parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"),
                             target_coords->nDim(),
                             _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("dense solver type"),
                             _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("problem type"),
                             _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("constraint type"),
                             _parameters->get<Teuchos::ParameterList>("remap").get<int>("curvature porder"));

//		GMLS my_GMLS (ReconstructionSpace::VectorTaylorPolynomial,
//				StaggeredEdgeIntegralSample,
//				StaggeredEdgeAnalyticGradientIntegralSample,
//				_parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"),
//				_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("dense linear solver"),
//				_parameters->get<Teuchos::ParameterList>("remap").get<int>("curvature porder"));
		my_GMLS.setProblemData(
				kokkos_neighbor_lists_host,
				kokkos_augmented_source_coordinates_host,
				kokkos_target_coordinates,
				kokkos_epsilons_host);
		my_GMLS.setWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("weighting type"));
		my_GMLS.setWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("weighting power"));
		my_GMLS.setCurvatureWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("curvature weighting type"));
		my_GMLS.setCurvatureWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("curvature weighting power"));
		my_GMLS.setOrderOfQuadraturePoints(_parameters->get<Teuchos::ParameterList>("remap").get<int>("quadrature order"));
		my_GMLS.setDimensionOfQuadraturePoints(_parameters->get<Teuchos::ParameterList>("remap").get<int>("quadrature dimension"));
		my_GMLS.setQuadratureType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("quadrature type"));

		my_GMLS.addTargets(TargetOperation::ChainedStaggeredLaplacianOfScalarPointEvaluation);
		//my_GMLS.addTargets(TargetOperation::DivergenceOfVectorPointEvaluation);
		my_GMLS.generateAlphas(); // just point evaluations

		// get maximum number of neighbors * fields[field_two]->nDim()
		int team_scratch_size = host_scratch_vector_scalar_type::shmem_size(max_num_neighbors * fields[field_two]->nDim()); // values
		team_scratch_size += host_scratch_vector_local_index_type::shmem_size(max_num_neighbors * fields[field_two]->nDim()); // local column indices
		const local_index_type host_scratch_team_level = 0; // not used in Kokkos currently

//		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,nlocal), KOKKOS_LAMBDA(const int i) {
		Kokkos::parallel_for(host_team_policy(nlocal, Kokkos::AUTO).set_scratch_size(host_scratch_team_level,Kokkos::PerTeam(team_scratch_size)), [=](const host_member_type& teamMember) {
			const int i = teamMember.league_rank();

			host_scratch_vector_local_index_type col_data(teamMember.team_scratch(host_scratch_team_level), max_num_neighbors*fields[field_two]->nDim());
			host_scratch_vector_scalar_type val_data(teamMember.team_scratch(host_scratch_team_level), max_num_neighbors*fields[field_two]->nDim());

			scalar_type target_coeff = fsos->evalDiffusionCoefficient(target_coords->getLocalCoords(i, false));

			local_index_type num_neighbors = neighborhood->getNumNeighbors(i);

			//Print error if there's not enough neighbors:
			TEUCHOS_TEST_FOR_EXCEPT_MSG(num_neighbors < neighbors_needed,
					"ERROR: Number of neighbors: " + std::to_string(num_neighbors) << " Neighbors needed: " << std::to_string(neighbors_needed) );

			//Put the values of alpha in the proper place in the global matrix
			for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
				local_index_type row = local_to_dof_map(i, field_one, k);

				for (local_index_type l = 0; l < num_neighbors; l++) {

					scalar_type avg_coeff = 1;
					if (_physics_type==3) {
						scalar_type neighbor_coeff = fsos->evalDiffusionCoefficient(source_coords->getLocalCoords(neighborhood->getNeighbor(i,l), true));
						avg_coeff = 0.5*(target_coeff + neighbor_coeff);
					}

					for (local_index_type n = 0; n < fields[field_two]->nDim(); ++n) {
						col_data(l*fields[field_two]->nDim() + n) = local_to_dof_map(neighborhood->getNeighbor(i,l), field_two, n);
						if (n==k) { // same field, same component
							// implicitly this is dof = particle#*ntotalfielddimension so this is just getting the particle number from dof
							// and checking its boundary condition
							if (bc_id(i, 0) != 0) {
								if (i==neighborhood->getNeighbor(i,l)) {
									val_data(l*fields[field_two]->nDim() + n) = 1.0;
								} else {
									val_data(l*fields[field_two]->nDim() + n) = 0.0;
								}
							} else {
								val_data(l*fields[field_two]->nDim() + n) = avg_coeff * my_GMLS.getAlpha0TensorTo0Tensor(TargetOperation::ChainedStaggeredLaplacianOfScalarPointEvaluation, i, l) * my_GMLS.getPreStencilWeight(StaggeredEdgeAnalyticGradientIntegralSample, i, l, false, 0 /*output component*/, 0 /*input component*/);
								val_data(0*fields[field_two]->nDim() + n) += avg_coeff * my_GMLS.getAlpha0TensorTo0Tensor(TargetOperation::ChainedStaggeredLaplacianOfScalarPointEvaluation, i, l) * my_GMLS.getPreStencilWeight(StaggeredEdgeAnalyticGradientIntegralSample, i, l, true, 0 /*output component*/, 0 /*input component*/);
//								val_data(l*fields[field_two]->nDim() + n) = avg_coeff * my_GMLS.getAlpha0TensorTo0Tensor(TargetOperation::DivergenceOfVectorPointEvaluation, i, l) * my_GMLS.getPreStencilWeight(StaggeredEdgeAnalyticGradientIntegralSample, i, l, false, 0 /*output component*/, 0 /*input component*/);
//								val_data(0*fields[field_two]->nDim() + n) += avg_coeff * my_GMLS.getAlpha0TensorTo0Tensor(TargetOperation::DivergenceOfVectorPointEvaluation, i, l) * my_GMLS.getPreStencilWeight(StaggeredEdgeAnalyticGradientIntegralSample, i, l, true, 0 /*output component*/, 0 /*input component*/);
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

	} else
		if (_physics_type==1 || _physics_type==3) { // Chained staggered grad and div

		// strategy is to register the solution as a 3 field
		// then convert everything around however it needs to be
		// then to only solve where 2/3 of rows are zeros

		// matrix operators

		Teuchos::RCP<crs_matrix_type> gradient = Teuchos::rcp<crs_matrix_type>(new crs_matrix_type(this->_A_graph));
		Teuchos::RCP<crs_matrix_type> staggered_divergence_operator = Teuchos::rcp<crs_matrix_type>(new crs_matrix_type(this->_A_graph));
		Teuchos::RCP<crs_matrix_type> kappa_tag = Teuchos::rcp<crs_matrix_type>(new crs_matrix_type(this->_A_graph));

		gradient->setAllToScalar(0.0);
		staggered_divergence_operator->setAllToScalar(0.0);
		kappa_tag->setAllToScalar(0.0);

		// GMLS operators

		GMLS my_GMLS_staggered_grad (ReconstructionSpace::ScalarTaylorPolynomial,
				StaggeredEdgeAnalyticGradientIntegralSample,
                                _parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"),
                                target_coords->nDim(),
                                _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("dense solver type"),
                                _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("problem type"),
                                _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("constraint type"),
                                _parameters->get<Teuchos::ParameterList>("remap").get<int>("curvature porder"));
		my_GMLS_staggered_grad.setProblemData(
				kokkos_neighbor_lists_host,
				kokkos_augmented_source_coordinates_host,
				kokkos_target_coordinates,
				kokkos_epsilons_host);
		my_GMLS_staggered_grad.setWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("weighting type"));
		my_GMLS_staggered_grad.setWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("weighting power"));
		my_GMLS_staggered_grad.setCurvatureWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("curvature weighting type"));
		my_GMLS_staggered_grad.setCurvatureWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("curvature weighting power"));
		my_GMLS_staggered_grad.setOrderOfQuadraturePoints(_parameters->get<Teuchos::ParameterList>("remap").get<int>("quadrature order"));
		my_GMLS_staggered_grad.setDimensionOfQuadraturePoints(_parameters->get<Teuchos::ParameterList>("remap").get<int>("quadrature dimension"));
		my_GMLS_staggered_grad.setQuadratureType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("quadrature type"));

		my_GMLS_staggered_grad.addTargets(TargetOperation::GradientOfScalarPointEvaluation);
		my_GMLS_staggered_grad.generateAlphas();

		GMLS my_GMLS_staggered_div (ReconstructionSpace::VectorTaylorPolynomial,
				StaggeredEdgeIntegralSample,
	                        _parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"),
                                target_coords->nDim(),
                                _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("dense solver type"),
                                _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("problem type"),
                                _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("constraint type"),
                                _parameters->get<Teuchos::ParameterList>("remap").get<int>("curvature porder"));
		my_GMLS_staggered_div.setProblemData(
				kokkos_neighbor_lists_host,
				kokkos_augmented_source_coordinates_host,
				kokkos_target_coordinates,
				kokkos_epsilons_host);
		my_GMLS_staggered_div.setWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("weighting type"));
		my_GMLS_staggered_div.setWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("weighting power"));
		my_GMLS_staggered_div.setCurvatureWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("curvature weighting type"));
		my_GMLS_staggered_div.setCurvatureWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("curvature weighting power"));
		my_GMLS_staggered_div.setOrderOfQuadraturePoints(_parameters->get<Teuchos::ParameterList>("remap").get<int>("quadrature order"));
		my_GMLS_staggered_div.setDimensionOfQuadraturePoints(_parameters->get<Teuchos::ParameterList>("remap").get<int>("quadrature dimension"));
		my_GMLS_staggered_div.setQuadratureType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("quadrature type"));

		my_GMLS_staggered_div.addTargets(TargetOperation::DivergenceOfVectorPointEvaluation);
		my_GMLS_staggered_div.generateAlphas();

		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,nlocal), KOKKOS_LAMBDA(const int i) {

			scalar_type target_coeff = fsos->evalDiffusionCoefficient(target_coords->getLocalCoords(i, false));

			local_index_type num_neighbors = neighborhood->getNumNeighbors(i);

			//Print error if there's not enough neighbors:
			TEUCHOS_TEST_FOR_EXCEPT_MSG(num_neighbors < neighbors_needed,
					"ERROR: Number of neighbors: " + std::to_string(num_neighbors) << " Neighbors needed: " << std::to_string(neighbors_needed) );

			// build gradient operator matrix

			for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
				local_index_type row = local_to_dof_map(i, field_one, k);

				Teuchos::Array<local_index_type> col_data(num_neighbors * fields[field_two]->nDim());
				Teuchos::Array<scalar_type> val_data(num_neighbors * fields[field_two]->nDim());
				Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);
				Teuchos::ArrayView<scalar_type> values = Teuchos::ArrayView<scalar_type>(val_data);

				for (local_index_type l = 0; l < num_neighbors; l++) {

					scalar_type avg_coeff = 1;
					if (_physics_type==3) {
						scalar_type neighbor_coeff = fsos->evalDiffusionCoefficient(source_coords->getLocalCoords(neighborhood->getNeighbor(i,l), true));
						avg_coeff = 0.5*(target_coeff + neighbor_coeff);
					}

					cols[l*fields[field_two]->nDim()] = local_to_dof_map(neighborhood->getNeighbor(i,l), field_two, 0);
					values[l*fields[field_two]->nDim()] = avg_coeff * my_GMLS_staggered_grad.getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, k, l) * my_GMLS_staggered_grad.getPreStencilWeight(StaggeredEdgeAnalyticGradientIntegralSample, i, l, false, 0 /*output component*/, 0 /*input component*/);
					values[0*fields[field_two]->nDim()] += avg_coeff * my_GMLS_staggered_grad.getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, k, l) * my_GMLS_staggered_grad.getPreStencilWeight(StaggeredEdgeAnalyticGradientIntegralSample, i, l, true, 0 /*output component*/, 0 /*input component*/);
				}
				{
					gradient->sumIntoLocalValues(row, cols, values);
				}
			}

			// build divergence operator matrix
			{
				local_index_type row = local_to_dof_map(i, field_one, 0);

				Teuchos::Array<local_index_type> col_data(num_neighbors * fields[field_two]->nDim());
				Teuchos::Array<scalar_type> val_data(num_neighbors * fields[field_two]->nDim());
				Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);
				Teuchos::ArrayView<scalar_type> values = Teuchos::ArrayView<scalar_type>(val_data);

				for (local_index_type l = 0; l < num_neighbors; l++) {
					for (local_index_type m = 0; m < fields[field_two]->nDim(); ++m) {
						cols[l*fields[field_two]->nDim() + m] = local_to_dof_map(neighborhood->getNeighbor(i,l), field_two, m);
						values[l*fields[field_two]->nDim() + m] = my_GMLS_staggered_div.getAlpha1TensorTo1Tensor(TargetOperation::DivergenceOfVectorPointEvaluation, i, 0, l, 0) * my_GMLS_staggered_div.getPreStencilWeight(StaggeredEdgeIntegralSample, i, l, false, 0 /*output component*/, m /*input component*/);
						values[0*fields[field_two]->nDim() + m] += my_GMLS_staggered_div.getAlpha1TensorTo1Tensor(TargetOperation::DivergenceOfVectorPointEvaluation, i, 0, l, 0) * my_GMLS_staggered_div.getPreStencilWeight(StaggeredEdgeIntegralSample, i, l, true, 0 /*output component*/, m /*input component*/);
					}
				}
				{
					staggered_divergence_operator->sumIntoLocalValues(row, cols, values);
				}
			}
		});

		gradient->fillComplete();
		staggered_divergence_operator->fillComplete();

		auto out = Teuchos::getFancyOStream(Teuchos::rcpFromRef(std::cout));

		crs_matrix_type t1(staggered_divergence_operator->getRowMap(), gradient->getColMap(), 2*neighbors_needed);
		Tpetra::MatrixMatrix::Multiply(*(staggered_divergence_operator.getRawPtr()), false, *(gradient.getRawPtr()), false, t1, true);

//		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,nlocal), KOKKOS_LAMBDA(const int i) {
		for (local_index_type i=0; i<nlocal; ++i) {
			local_index_type num_neighbors = neighborhood->getNumNeighbors(i);

			//Put the values of alpha in the proper place in the global matrix
			for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
				local_index_type row = local_to_dof_map(i, field_one, k);

				Teuchos::Array<local_index_type> col_data(max_num_neighbors * num_neighbors * fields[field_two]->nDim());
				Teuchos::Array<scalar_type> val_data(max_num_neighbors * num_neighbors * fields[field_two]->nDim());
				Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);
				Teuchos::ArrayView<scalar_type> values = Teuchos::ArrayView<scalar_type>(val_data);

				Teuchos::Array<local_index_type> const_col_data(max_num_neighbors * num_neighbors * fields[field_two]->nDim());
				Teuchos::Array<scalar_type> const_val_data(max_num_neighbors * num_neighbors * fields[field_two]->nDim());
				Teuchos::ArrayView<const local_index_type> const_cols = Teuchos::ArrayView<local_index_type>(const_col_data);
				Teuchos::ArrayView<const scalar_type> const_values = Teuchos::ArrayView<scalar_type>(const_val_data);

				t1.getLocalRowView(row, const_cols, const_values);

				if (bc_id(i, 0) != 0) {
					cols[0] = row;
					values[0] = 1.0;
					for (local_index_type l = 1; l < const_values.size(); l++) {
						cols[l] = 0;
						values[l] = 0;
					}
				} else if (k==0) {
					for (local_index_type l = 0; l < const_values.size(); l++) {
						cols[l] = const_cols[l];
						values[l] = const_values[l];
					}
				} else {
//					local_index_type last_row = local_to_dof_map[i][field_one][0];
//					t1.getLocalRowView(last_row, const_cols, const_values);
//
//					for (local_index_type l = 0; l < const_values.size(); l++) {
//						cols[l] = const_cols[l] + k;
//						values[l] = const_values[l];
//					}
					cols[0] = row;
					values[0] = 1.0;
					for (local_index_type l = 1; l < const_values.size(); l++) {
						cols[l] = 0;
						values[l] = 0;
					}
				}

				//#pragma omp critical
				{
					if (k==0) {
//						this->_A->sumIntoLocalValues(row, cols, values);
//						row = local_to_dof_map[i][0][0];
//						cols[0] = row;
//						this->_A->sumIntoLocalValues(row, cols, values);
//						row = local_to_dof_map[i][1][0];
//						cols[0] = row;
						this->_A->sumIntoLocalValues(row, cols, values);
					} else {
						this->_A->sumIntoLocalValues(row, cols, values);
					}
				}
			}


		};
		//);

//		_A->describe(*out, Teuchos::VERB_EXTREME);

	} else if (_physics_type==2 || _physics_type==4) { // Chained staggered grad and div

		// strategy is to register the solution as a 3 field
		// then convert everything around however it needs to be
		// then to only solve where 2/3 of rows are zeros

		// matrix operators

		Teuchos::RCP<crs_matrix_type> gradient = Teuchos::rcp<crs_matrix_type>(new crs_matrix_type(this->_A_graph));
		Teuchos::RCP<crs_matrix_type> staggered_divergence_operator = Teuchos::rcp<crs_matrix_type>(new crs_matrix_type(this->_A_graph));
		Teuchos::RCP<crs_matrix_type> kappa_tag = Teuchos::rcp<crs_matrix_type>(new crs_matrix_type(this->_A_graph));

		gradient->setAllToScalar(0.0);
		staggered_divergence_operator->setAllToScalar(0.0);
		kappa_tag->setAllToScalar(0.0);

		// GMLS operators

		GMLS my_GMLS_staggered_grad (ReconstructionSpace::ScalarTaylorPolynomial,
				PointSample,
			        _parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"),
                                target_coords->nDim(),
                                _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("dense solver type"),
                                _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("problem type"),
                                _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("constraint type"),
                                _parameters->get<Teuchos::ParameterList>("remap").get<int>("curvature porder"));
		my_GMLS_staggered_grad.setProblemData(
				kokkos_neighbor_lists_host,
				kokkos_augmented_source_coordinates_host,
				kokkos_target_coordinates,
				kokkos_epsilons_host);
		my_GMLS_staggered_grad.setWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("weighting type"));
		my_GMLS_staggered_grad.setWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("weighting power"));
		my_GMLS_staggered_grad.setCurvatureWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("curvature weighting type"));
		my_GMLS_staggered_grad.setCurvatureWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("curvature weighting power"));
		my_GMLS_staggered_grad.setOrderOfQuadraturePoints(_parameters->get<Teuchos::ParameterList>("remap").get<int>("quadrature order"));
		my_GMLS_staggered_grad.setDimensionOfQuadraturePoints(_parameters->get<Teuchos::ParameterList>("remap").get<int>("quadrature dimension"));
		my_GMLS_staggered_grad.setQuadratureType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("quadrature type"));

		my_GMLS_staggered_grad.addTargets(TargetOperation::GradientOfScalarPointEvaluation);
		my_GMLS_staggered_grad.generateAlphas();

		GMLS my_GMLS_staggered_div (ReconstructionSpace::VectorTaylorPolynomial,
				ManifoldVectorPointSample,
			        _parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"),
                                target_coords->nDim(),
                                _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("dense solver type"),
                                _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("problem type"),
                                _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("constraint type"),
                                _parameters->get<Teuchos::ParameterList>("remap").get<int>("curvature porder"));
		my_GMLS_staggered_div.setProblemData(
				kokkos_neighbor_lists_host,
				kokkos_augmented_source_coordinates_host,
				kokkos_target_coordinates,
				kokkos_epsilons_host);
		my_GMLS_staggered_div.setWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("weighting type"));
		my_GMLS_staggered_div.setWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("weighting power"));
		my_GMLS_staggered_div.setCurvatureWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("curvature weighting type"));
		my_GMLS_staggered_div.setCurvatureWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("curvature weighting power"));
		my_GMLS_staggered_div.setOrderOfQuadraturePoints(_parameters->get<Teuchos::ParameterList>("remap").get<int>("quadrature order"));
		my_GMLS_staggered_div.setDimensionOfQuadraturePoints(_parameters->get<Teuchos::ParameterList>("remap").get<int>("quadrature dimension"));
		my_GMLS_staggered_div.setQuadratureType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("quadrature type"));

		my_GMLS_staggered_div.addTargets(TargetOperation::DivergenceOfVectorPointEvaluation);
		my_GMLS_staggered_div.generateAlphas();

		Teuchos::RCP<Compadre::FiveStripOnSphere> fsos = Teuchos::rcp(new Compadre::FiveStripOnSphere);

		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,nlocal), KOKKOS_LAMBDA(const int i) {

			//scalar_type target_coeff = fsos->evalDiffusionCoefficient(target_coords->getLocalCoords(i, false));

			local_index_type num_neighbors = neighborhood->getNumNeighbors(i);

			//Print error if there's not enough neighbors:
			TEUCHOS_TEST_FOR_EXCEPT_MSG(num_neighbors < neighbors_needed,
					"ERROR: Number of neighbors: " + std::to_string(num_neighbors) << " Neighbors needed: " << std::to_string(neighbors_needed) );

			// build gradient operator matrix

			for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
				local_index_type row = local_to_dof_map(i, field_one, k);

				Teuchos::Array<local_index_type> col_data(num_neighbors * fields[field_two]->nDim());
				Teuchos::Array<scalar_type> val_data(num_neighbors * fields[field_two]->nDim());
				Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);
				Teuchos::ArrayView<scalar_type> values = Teuchos::ArrayView<scalar_type>(val_data);

				for (local_index_type l = 0; l < num_neighbors; l++) {

					scalar_type avg_coeff = 1;
//					if (_physics_type==4) {
//						scalar_type neighbor_coeff = fsos->evalDiffusionCoefficient(source_coords->getLocalCoords(neighborhood->getNeighbor(i,l), true));
//						avg_coeff = 0.5*(target_coeff + neighbor_coeff);
//					}

					cols[l*fields[field_two]->nDim()] = local_to_dof_map(neighborhood->getNeighbor(i,l), field_two, 0);
					values[l*fields[field_two]->nDim()] = avg_coeff * my_GMLS_staggered_grad.getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, k, l) * my_GMLS_staggered_grad.getPreStencilWeight(PointSample, i, l, false, 0 /*output component*/, 0 /*input component*/);
					values[0*fields[field_two]->nDim()] += avg_coeff * my_GMLS_staggered_grad.getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, k, l) * my_GMLS_staggered_grad.getPreStencilWeight(PointSample, i, l, true, 0 /*output component*/, 0 /*input component*/);
				}
				{
					gradient->sumIntoLocalValues(row, cols, values);
				}
			}

			// build divergence operator matrix
			{
				local_index_type row = local_to_dof_map(i, field_one, 0);

				Teuchos::Array<local_index_type> col_data(num_neighbors * fields[field_two]->nDim());
				Teuchos::Array<scalar_type> val_data(num_neighbors * fields[field_two]->nDim());
				Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);
				Teuchos::ArrayView<scalar_type> values = Teuchos::ArrayView<scalar_type>(val_data);

				for (local_index_type l = 0; l < num_neighbors; l++) {

					scalar_type avg_coeff = 1;
//					if (_physics_type==4) {
//						scalar_type neighbor_coeff = fsos->evalDiffusionCoefficient(source_coords->getLocalCoords(neighborhood->getNeighbor(i,l), true));
//						avg_coeff = 0.5*(target_coeff + neighbor_coeff);
//					}

					for (local_index_type n = 0; n < fields[field_two]->nDim(); ++n) {
						for (local_index_type m = 0; m < fields[field_two]->nDim(); ++m) {
							// get neighbor l, evaluate coordinate and get latitude from strip
							cols[l*fields[field_two]->nDim() + m] = local_to_dof_map(neighborhood->getNeighbor(i,l), field_two, m);
							values[l*fields[field_two]->nDim() + m] = avg_coeff * my_GMLS_staggered_div.getAlpha1TensorTo1Tensor(TargetOperation::DivergenceOfVectorPointEvaluation, i, 0, l, n) * my_GMLS_staggered_div.getPreStencilWeight(ManifoldVectorPointSample, i, l, false, n /*output component*/, m /*input component*/);
							values[0*fields[field_two]->nDim() + m] += avg_coeff * my_GMLS_staggered_div.getAlpha1TensorTo1Tensor(TargetOperation::DivergenceOfVectorPointEvaluation, i, 0, l, n) * my_GMLS_staggered_div.getPreStencilWeight(ManifoldVectorPointSample, i, l, true, n /*output component*/, m /*input component*/);
						}
					}
				}
				{
					staggered_divergence_operator->sumIntoLocalValues(row, cols, values);
				}
			}
		});

		gradient->fillComplete();
		staggered_divergence_operator->fillComplete();

		auto out = Teuchos::getFancyOStream(Teuchos::rcpFromRef(std::cout));

		crs_matrix_type t1(staggered_divergence_operator->getRowMap(), gradient->getColMap(), 2*neighbors_needed);
		Tpetra::MatrixMatrix::Multiply(*(staggered_divergence_operator.getRawPtr()), false, *(gradient.getRawPtr()), false, t1, true);

//		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,nlocal), KOKKOS_LAMBDA(const int i) {
		for (local_index_type i=0; i<nlocal; ++i) {
			local_index_type num_neighbors = neighborhood->getNumNeighbors(i);

			//Put the values of alpha in the proper place in the global matrix
			for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
				local_index_type row = local_to_dof_map(i, field_one, k);

				Teuchos::Array<local_index_type> col_data(max_num_neighbors * num_neighbors * fields[field_two]->nDim());
				Teuchos::Array<scalar_type> val_data(max_num_neighbors * num_neighbors * fields[field_two]->nDim());
				Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);
				Teuchos::ArrayView<scalar_type> values = Teuchos::ArrayView<scalar_type>(val_data);

				Teuchos::Array<local_index_type> const_col_data(max_num_neighbors * num_neighbors * fields[field_two]->nDim());
				Teuchos::Array<scalar_type> const_val_data(max_num_neighbors * num_neighbors * fields[field_two]->nDim());
				Teuchos::ArrayView<const local_index_type> const_cols = Teuchos::ArrayView<local_index_type>(const_col_data);
				Teuchos::ArrayView<const scalar_type> const_values = Teuchos::ArrayView<scalar_type>(const_val_data);

				t1.getLocalRowView(row, const_cols, const_values);

				if (bc_id(i, 0) != 0) {
					cols[0] = row;
					values[0] = 1.0;
					for (local_index_type l = 1; l < const_values.size(); l++) {
						cols[l] = 0;
						values[l] = 0;
					}
				} else if (k==0) {
					for (local_index_type l = 0; l < const_values.size(); l++) {
						cols[l] = const_cols[l];
						values[l] = const_values[l];
					}
				} else {
//					local_index_type last_row = local_to_dof_map[i][field_one][0];
//					t1.getLocalRowView(last_row, const_cols, const_values);
//
//					for (local_index_type l = 0; l < const_values.size(); l++) {
//						cols[l] = const_cols[l] + k;
//						values[l] = const_values[l];
//					}
					cols[0] = row;
					values[0] = 1.0;
					for (local_index_type l = 1; l < const_values.size(); l++) {
						cols[l] = 0;
						values[l] = 0;
					}
				}
				//#pragma omp critical
				{
					if (k==0) {
//						this->_A->sumIntoLocalValues(row, cols, values);
//						row = local_to_dof_map[i][0][0];
//						cols[0] = row;
//						this->_A->sumIntoLocalValues(row, cols, values);
//						row = local_to_dof_map[i][1][0];
//						cols[0] = row;
						this->_A->sumIntoLocalValues(row, cols, values);
					} else {
						this->_A->sumIntoLocalValues(row, cols, values);
					}
				}
			}


		};
		//);

//		_A->describe(*out, Teuchos::VERB_EXTREME);
	}

} else if (field_one == lm_field_id && field_two == solution_field_id) {
    // row all DOFs for solution against Lagrange Multiplier
	Teuchos::Array<local_index_type> col_data(nlocal * fields[field_two]->nDim());
	Teuchos::Array<scalar_type> val_data(nlocal * fields[field_two]->nDim());
	Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);
	Teuchos::ArrayView<scalar_type> vals = Teuchos::ArrayView<scalar_type>(val_data);

	for (local_index_type l = 0; l < nlocal; l++) {
		for (local_index_type n = 0; n < fields[field_two]->nDim(); ++n) {
			cols[l*fields[field_two]->nDim() + n] = local_to_dof_map(l, field_two, n);
			vals[l*fields[field_two]->nDim() + n] = 1;
		}
	}
    auto comm = this->_particles->getCoordsConst()->getComm();
    if (comm->getRank() == 0) {
        // local index 0 is the shared global id for the lagrange multiplier
        this->_A->sumIntoLocalValues(0, cols, vals);
    } else {
        // local index 0 is the shared global id for the lagrange multiplier
        this->_A->sumIntoLocalValues(nlocal, cols, vals);
    }
} else if (field_one == solution_field_id && field_two == lm_field_id) {

    // col all DOFs for solution against Lagrange Multiplier
    auto comm = this->_particles->getCoordsConst()->getComm();
    //if (comm->getRank() == 0) {
	    for(local_index_type i = 0; i < nlocal; i++) {
	    	for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
	    		local_index_type row = local_to_dof_map(i, field_one, k);

	    		Teuchos::Array<local_index_type> col_data(1);
	    		Teuchos::Array<scalar_type> val_data(1);
	    		Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);
	    		Teuchos::ArrayView<scalar_type> vals = Teuchos::ArrayView<scalar_type>(val_data);

                if (comm->getRank() == 0) {
	    		    cols[0] = 0; // local index 0 is global shared index
                } else {
                    cols[0] = nlocal;
                }
                vals[0] = 1;
	    	    this->_A->sumIntoLocalValues(row, cols, vals);
	    	}
	    }
    //}
} else {
    // identity on all DOFs for Lagrange Multiplier (even DOFs not really used, since we only use first LM dof for now)
    scalar_type eps_penalty = nglobal;
    //scalar_type eps_penalty = 1e-5;

    auto comm = this->_particles->getCoordsConst()->getComm();
	for(local_index_type i = 0; i < nlocal; i++) {
		for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
			local_index_type row = local_to_dof_map(i, field_one, k);

			Teuchos::Array<local_index_type> col_data(fields[field_two]->nDim());
			Teuchos::Array<scalar_type> val_data(fields[field_two]->nDim());
			Teuchos::ArrayView<local_index_type> cols = Teuchos::ArrayView<local_index_type>(col_data);
			Teuchos::ArrayView<scalar_type> vals = Teuchos::ArrayView<scalar_type>(val_data);
			for (local_index_type n = 0; n < fields[field_two]->nDim(); ++n) {
                if (i==0 && comm->getRank()==0) {
				    cols[n] = local_to_dof_map(i, field_two, n);
				    vals[n] = eps_penalty;
                } else {
				    cols[n] = local_to_dof_map(i, field_two, n);
				    vals[n] = 1;
                }
			}
			//#pragma omp critical
			{
		        this->_A->sumIntoLocalValues(row, cols, vals);
			}
		}
	}
}

//	this->_A->print(std::cout);
	ComputeMatrixTime->stop();

}

const std::vector<InteractingFields> LaplaceBeltramiPhysics::gatherFieldInteractions() {
	std::vector<InteractingFields> field_interactions;
	field_interactions.push_back(InteractingFields(op_needing_interaction::physics, _particles->getFieldManagerConst()->getIDOfFieldFromName("solution")));
	field_interactions.push_back(InteractingFields(op_needing_interaction::physics, _particles->getFieldManagerConst()->getIDOfFieldFromName("solution"), _particles->getFieldManagerConst()->getIDOfFieldFromName("lagrange multiplier")));
	field_interactions.push_back(InteractingFields(op_needing_interaction::physics, _particles->getFieldManagerConst()->getIDOfFieldFromName("lagrange multiplier"), _particles->getFieldManagerConst()->getIDOfFieldFromName("solution")));
	field_interactions.push_back(InteractingFields(op_needing_interaction::physics, _particles->getFieldManagerConst()->getIDOfFieldFromName("lagrange multiplier")));
	return field_interactions;
}
    
}
