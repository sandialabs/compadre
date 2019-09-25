#include "Compadre_SolverT.hpp"

#include "Compadre_ParameterManager.hpp"

#include <Teuchos_AbstractFactoryStd.hpp>

#ifdef TRILINOS_LINEAR_SOLVES

#include <Amesos2.hpp>
#include <Amesos2_Version.hpp>

#include <Teko_Utilities.hpp>
#include <Teko_StratimikosFactory.hpp>

#include <Thyra_TpetraLinearOp.hpp>
#include <Thyra_TpetraVectorSpace.hpp>
#include <Thyra_LinearOpWithSolveFactoryHelpers.hpp>
#include <Thyra_PreconditionerFactoryHelpers.hpp>
#include <Thyra_MueLuPreconditionerFactory.hpp>
#include <Thyra_Ifpack2PreconditionerFactory.hpp>


#include <Stratimikos_MueLuHelpers.hpp>

#include <set>

namespace Compadre {

void SolverT::amalgamateBlockMatrices(std::vector<std::vector<Teuchos::RCP<crs_matrix_type> > > A, 
        std::vector<Teuchos::RCP<mvec_type> > b) {
    // ASSUMPTIONS: assumes all columns have the same row map  
    // now being change to all row maps are a subset of the row map with the largest number of entries 

    // DOFs from blocks should already be disjoint from one another, i.e.
    // field one has global ID 0...N and field two has global IDs N+1...N+M

    Teuchos::RCP<map_type> new_row_map;
    Teuchos::RCP<map_type> new_range_map;

    std::vector<local_index_type> valid_j_for_i(A.size());

    { // creates new_row_map and new_range_map

        // for each row, do a merge over columns

        std::set<global_index_type> all_range_indices;
        std::set<global_index_type> all_row_indices;
        std::set<global_index_type> ghost_row_indices;

        for (size_t i=0; i<A.size(); ++i) {
            for (size_t j=0; j<A[0].size(); ++j) {
                if (!A[i][j].is_null()) {
                    auto range_indices = A[i][j]->getRangeMap()->getMyGlobalIndices();
                    auto row_indices = A[i][j]->getRowMap()->getMyGlobalIndices();

                    for (size_t k=0; k<range_indices.extent(0); ++k) {
                        all_range_indices.insert(range_indices(k));
                    }

                    all_row_indices.insert(row_indices(0));
                    bool increasing = true;
                    for (size_t k=1; k<row_indices.extent(0); ++k) {
                        increasing &= (row_indices(k) >=  row_indices(k-1));
                        if (increasing) {
                            all_row_indices.insert(row_indices(k));
                        } else {
                            ghost_row_indices.insert(row_indices(k));
                        }
                    }
                }
            }
        }
	    Kokkos::View<global_index_type*, Kokkos::HostSpace> new_row_local_indices("local row indices", all_row_indices.size() + ghost_row_indices.size());
	    //Kokkos::View<global_index_type*, Kokkos::HostSpace> new_range_local_indices("local range indices", all_range_indices.size());
	    Kokkos::View<global_index_type*, Kokkos::HostSpace> new_range_local_indices("local range indices", all_row_indices.size());
        auto iter = all_row_indices.begin();
        for (size_t i=0; i<all_row_indices.size(); ++i) {
            new_range_local_indices(i) = *iter;
            ++iter;
        }
        iter = all_row_indices.begin();
        for (size_t i=0; i<all_row_indices.size(); ++i) {
            new_row_local_indices(i) = *iter;
            ++iter;
        }
        iter = ghost_row_indices.begin();
        for (size_t i=0; i<ghost_row_indices.size(); ++i) {
            new_row_local_indices(all_row_indices.size() + i) = *iter;
            ++iter;
        }
        for (size_t i=0; i<A.size(); ++i) {
            for (size_t j=0; j<A[0].size(); ++j) {
                if (!A[i][j].is_null()) {
                    printf("i: %d, j: %d, r: %d, row_size: %lu\n", i, j, _comm->getRank(), A[i][j]->getRowMap()->getNodeNumElements());
                    printf("i: %d, j: %d, r: %d, range_size: %lu\n", i, j, _comm->getRank(), A[i][j]->getRangeMap()->getNodeNumElements());
                }
            }
        }

        //std::vector<global_index_type> local_ids_offsets_row(A.size());
        //std::vector<global_index_type> local_ids_offsets_range(A.size());

        //local_index_type local_row_count = 0;
        //local_index_type local_range_count = 0;

	    //// gets local and global offsets
        //for (size_t i=0; i<A.size(); ++i) {
	    //	local_ids_offsets_row[i] = local_row_count;
	    //	local_ids_offsets_range[i] = local_range_count;

        //    // not all columns necessarily have a valid matrix in them
        //    global_index_type max_row_count = 0;
        //    global_index_type max_range_count = 0;
        //    for (size_t j=0; j<A[0].size(); ++j) {
        //        if (!A[i][j].is_null()) {
        //            //max_row_count = std::max(max_row_count, A[i][j]->getRowMap()->getNodeNumElements());
        //            //max_range_count = std::max(max_row_count, A[i][j]->getRangeMap()->getNodeNumElements());
        //            //printf("i: %d, j: %d, r: %d, size: %lu\n", i, j, _comm->getRank(), A[i][j]->getRowMap()->getNodeNumElements());
	    //	        //local_row_count += A[i][j]->getRowMap()->getNodeNumElements();
	    //	        //local_range_count += A[i][j]->getRangeMap()->getNodeNumElements();
	    //            ////local_row_count += A[i][j]->getRowMap()->getMyGlobalIndices();
	    //            ////local_range_count += A[i][j]->getRangeMap()->getMyGlobalIndices();
        //            if (A[i][j]->getRowMap()->getNodeNumElements() > max_row_count) {
        //                max_row_count = A[i][j]->getRowMap()->getNodeNumElements();
        //                max_range_count = A[i][j]->getRangeMap()->getNodeNumElements();
        //                valid_j_for_i[i] = j;
        //                //printf("i: %d, j: %d, r: %d, row_size: %lu\n", i, j, _comm->getRank(), A[i][j]->getRowMap()->getNodeNumElements());
        //                //printf("i: %d, j: %d, r: %d, ran_size: %lu\n", i, j, _comm->getRank(), A[i][j]->getRangeMap()->getNodeNumElements());
        //            }
        //            //valid_j_for_i[i] = j;
        //            //break;
        //        }
        //    }
	    //    local_row_count += max_row_count;
	    //    local_range_count += max_range_count;
        //    //printf("i: %d, r: %d, lrow: %lu, lrange: %lu\n", i, _comm->getRank(), local_ids_offsets_row[i], local_ids_offsets_range[i]);
        //}

        ////printf("r: %d, lrow: %lu, lrange: %lu\n", _comm->getRank(), local_row_count, local_range_count);
	    //Kokkos::View<global_index_type*, Kokkos::HostSpace> new_row_local_indices("local row indices", local_row_count);
	    //Kokkos::View<global_index_type*, Kokkos::HostSpace> new_range_local_indices("local range indices", local_range_count);
        //// copy over indices
        //for (size_t i=0; i<A.size(); ++i) {

        //    auto row_indices = A[i][valid_j_for_i[i]]->getRowMap()->getMyGlobalIndices();
        //    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,row_indices.extent(0)), 
        //                KOKKOS_LAMBDA(const int j) {
        //        new_row_local_indices(local_ids_offsets_row[i] + j) = row_indices(j);
        //    });

        //    auto range_indices = A[i][valid_j_for_i[i]]->getRangeMap()->getMyGlobalIndices();
        //    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,range_indices.extent(0)), 
        //                KOKKOS_LAMBDA(const int j) {
        //        new_range_local_indices(local_ids_offsets_range[i] + j) = range_indices(j);
        //    });
        //    //for (size_t j=0; j<row_indices.extent(0); ++j) {
        //    //    if (i==0 && _comm->getRank()==0) {
        //    //        printf("i: %d, r: %d, j: %d, val: %d\n", i, _comm->getRank(), j, row_indices(j));
        //    //    }
        //    //}
        //}

        new_row_map = Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), 
                new_row_local_indices, 0 /*index base*/, _comm));
        new_range_map = Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), 
                new_range_local_indices, 0 /*index base*/, _comm));
    } 

    Teuchos::RCP<map_type> new_col_map;
    Teuchos::RCP<map_type> new_domain_map;
    std::vector<local_index_type> valid_i_for_j(A[0].size());
    { // creates new_col_map and new_domain_map

        std::set<global_index_type> all_domain_indices;
        std::set<global_index_type> all_col_indices;
        std::set<global_index_type> ghost_col_indices;

        for (size_t i=0; i<A.size(); ++i) {
            for (size_t j=0; j<A[0].size(); ++j) {
                if (!A[i][j].is_null()) {

                    auto domain_indices = A[i][j]->getDomainMap()->getMyGlobalIndices();
                    auto col_indices = A[i][j]->getColMap()->getMyGlobalIndices();

                    for (size_t k=0; k<domain_indices.extent(0); ++k) {
                        all_domain_indices.insert(domain_indices(k));
                    }

                    all_col_indices.insert(col_indices(0));
                    bool increasing = true;
                    for (size_t k=1; k<col_indices.extent(0); ++k) {
                        increasing &= (col_indices(k) >=  col_indices(k-1));
                        if (increasing) {
                            all_col_indices.insert(col_indices(k));
                        } else {
                            ghost_col_indices.insert(col_indices(k));
                        }
                    }
                }
            }
        }
	    Kokkos::View<global_index_type*, Kokkos::HostSpace> new_col_local_indices("local col indices", all_col_indices.size() + ghost_col_indices.size());
	    Kokkos::View<global_index_type*, Kokkos::HostSpace> new_domain_local_indices("local domain indices", all_col_indices.size());
        auto iter = all_col_indices.begin();
        for (size_t i=0; i<all_col_indices.size(); ++i) {
            new_domain_local_indices(i) = *iter;
            ++iter;
        }
        iter = all_col_indices.begin();
        for (size_t i=0; i<all_col_indices.size(); ++i) {
            new_col_local_indices(i) = *iter;
            ++iter;
        }
        iter = ghost_col_indices.begin();
        for (size_t i=0; i<ghost_col_indices.size(); ++i) {
            new_col_local_indices(all_col_indices.size() + i) = *iter;
            ++iter;
        }
        //printf("col s: %lu, dom s: %lu\n", all_col_indices.size(), all_domain_indices.size());

        //std::vector<global_index_type> local_ids_offsets_col(A[0].size());
        //std::vector<global_index_type> local_ids_offsets_domain(A[0].size());

        //local_index_type local_col_count = 0;
        //local_index_type local_domain_count = 0;

	    //// gets local and global offsets
        //for (size_t j=0; j<A[0].size(); ++j) {
	    //	local_ids_offsets_col[j] = local_col_count;
	    //	local_ids_offsets_domain[j] = local_domain_count;

        //    // not all columns necessarily have a valid matrix in them
        //    global_index_type max_col_count = 0;
        //    global_index_type max_domain_count = 0;
        //    for (size_t i=0; i<A.size(); ++i) {
        //        if (!A[i][j].is_null()) {
        //            //max_row_count = std::max(max_row_count, A[i][j]->getRowMap()->getNodeNumElements());
        //            //max_range_count = std::max(max_row_count, A[i][j]->getRangeMap()->getNodeNumElements());
        //            //printf("i: %d, j: %d, r: %d, size: %lu\n", i, j, _comm->getRank(), A[i][j]->getRowMap()->getNodeNumElements());
	    //	        //local_row_count += A[i][j]->getRowMap()->getNodeNumElements();
	    //	        //local_range_count += A[i][j]->getRangeMap()->getNodeNumElements();
	    //            ////local_row_count += A[i][j]->getRowMap()->getMyGlobalIndices();
	    //            ////local_range_count += A[i][j]->getRangeMap()->getMyGlobalIndices();
        //            if (A[i][j]->getColMap()->getNodeNumElements() > max_col_count) {
        //                max_col_count = A[i][j]->getColMap()->getNodeNumElements();
        //                max_domain_count = A[i][j]->getDomainMap()->getNodeNumElements();
        //                valid_i_for_j[j] = i;
        //                printf("i: %d, j: %d, r: %d, col_size: %lu\n", i, j, _comm->getRank(), A[i][j]->getColMap()->getNodeNumElements());
        //                printf("i: %d, j: %d, r: %d, dom_size: %lu\n", i, j, _comm->getRank(), A[i][j]->getDomainMap()->getNodeNumElements());
        //                //printf("s1 %lu s2 %lu\n", A[i][j]->getDomainMap()->getNodeNumElements(), A[i][j]->getDomainMap()->getMyGlobalIndices().extent(0));
        //            }
        //            //valid_j_for_i[i] = j;
        //            //break;
        //        }
        //    }
	    //    local_col_count += max_col_count;
	    //    local_domain_count += max_domain_count;
        //    printf("j: %d, r: %d, lcol: %lu, ldomain: %lu\n", j, _comm->getRank(), local_ids_offsets_col[j], local_ids_offsets_domain[j]);

        //    //// not all columns necessarily have a valid matrix in them
        //    //for (size_t i=0; i<A.size(); ++i) {
        //    //    if (!A[i][j].is_null()) {
	    //    //        local_col_count += A[i][j]->getColMap()->getNodeNumElements();
	    //    //        local_domain_count += A[i][j]->getDomainMap()->getNodeNumElements();
	    //    //        //local_col_count += A[i][j]->getColMap()->getMyGlobalIndices();
	    //    //        //local_domain_count += A[i][j]->getDomainMap()->getMyGlobalIndices();
        //    //        valid_i_for_j[j] = i;
        //    //        break;
        //    //    }
        //    //}
        //    //printf("j: %d, r: %d, lcol: %lu, ldomain: %lu\n", j, _comm->getRank(), local_col_count, local_domain_count);
        //}

	    //Kokkos::View<global_index_type*, Kokkos::HostSpace> alt_new_col_local_indices("local col indices", local_col_count);
	    //Kokkos::View<global_index_type*, Kokkos::HostSpace> alt_new_domain_local_indices("local domain indices", local_domain_count);
        //// copy over indices
        //for (size_t j=0; j<A[0].size(); ++j) {

        //    auto col_indices = A[valid_i_for_j[j]][j]->getColMap()->getMyGlobalIndices();
        //    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,col_indices.extent(0)), 
        //                KOKKOS_LAMBDA(const int i) {
        //        alt_new_col_local_indices(local_ids_offsets_col[j] + i) = col_indices(i);
        //    });
        //    auto domain_indices = A[valid_i_for_j[j]][j]->getDomainMap()->getMyGlobalIndices();
        //    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,domain_indices.extent(0)), 
        //                KOKKOS_LAMBDA(const int i) {
        //        alt_new_domain_local_indices(local_ids_offsets_domain[j] + i) = domain_indices(i);
        //    });

        //}
        //printf("col s: %lu, dom s: %lu\n", local_col_count, local_domain_count);

        //for (size_t k=0; k<alt_new_col_local_indices.extent(0); ++k) {
        //    printf("%d: %lu, %lu, diff: %lu\n", _comm->getRank(), alt_new_col_local_indices(k), new_col_local_indices(k), alt_new_col_local_indices(k)-new_col_local_indices(k));
        //}

        new_col_map = Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), 
                new_col_local_indices, 0 /*index base*/, _comm));
        new_domain_map = Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), 
                new_domain_local_indices, 0 /*index base*/, _comm));
    }

    // get size for maximum entries per row
    local_index_type max_entries_per_row_over_all_rows = 0;
    for (size_t i=0; i<A.size(); ++i) {
        local_index_type max_entries_this_row = 0;
        for (size_t j=0; j<A[i].size(); ++j) {
            if (!A[i][j].is_null()) {
                max_entries_this_row += A[i][j]->getCrsGraph()->getGlobalMaxNumRowEntries();
                printf("max added r:%i i:%lu j:%lu amt:%lu\n", _comm->getRank(), i, j, A[i][j]->getCrsGraph()->getGlobalMaxNumRowEntries());
            }
        }
        //max_entries_per_row_over_all_rows = std::max(max_entries_per_row_over_all_rows, max_entries_this_row);
        max_entries_per_row_over_all_rows += max_entries_this_row;
    }
    printf("max added %lu\n", max_entries_per_row_over_all_rows);

    // create giant graph using new maps
    auto A_single_block_graph = Teuchos::rcp(new crs_graph_type(new_row_map, new_col_map, max_entries_per_row_over_all_rows, Tpetra::StaticProfile));

    _row_exporters.resize(A.size()); // from each block row to global
    _range_exporters.resize(A.size()); // from each block row to global
    _col_exporters.resize(A.size()); // from each block col to global
    _domain_exporters.resize(A.size()); // from each block col to global

    for (size_t i=0; i<A.size(); ++i) {
        _row_exporters[i].resize(A[0].size()); // from each block row to global
        _range_exporters[i].resize(A[0].size()); // from each block row to global
        _col_exporters[i].resize(A[0].size()); // from each block row to global
        _domain_exporters[i].resize(A[0].size()); // from each block row to global
        for (size_t j=0; j<A[0].size(); ++j) {
            if (!A[i][j].is_null()) {
                _row_exporters[i][j] = Teuchos::rcp(new exporter_type(A[i][j]->getRowMap(), new_row_map));
                _range_exporters[i][j] = Teuchos::rcp(new exporter_type(A[i][j]->getRangeMap(), new_range_map));
            }
        }
    }
    for (size_t j=0; j<A[0].size(); ++j) {
        for (size_t i=0; i<A.size(); ++i) {
            if (!A[i][j].is_null()) {
                _col_exporters[i][j] = Teuchos::rcp(new exporter_type(A[i][j]->getColMap(), new_col_map));
                _domain_exporters[i][j] = Teuchos::rcp(new exporter_type(A[i][j]->getDomainMap(), new_domain_map));
            }
        }
    }

    //printf("%lu entries dom, %lu entrie col\n", new_domain_map->getGlobalNumElements(), new_col_map->getGlobalNumElements());

    for (size_t i=0; i<A.size(); ++i) {
        for (size_t j=0; j<A[0].size(); ++j) {
            if (!A[i][j].is_null()) {
                A_single_block_graph->doExport(*(A[i][j]->getCrsGraph()), *(_row_exporters[i][j]), Tpetra::INSERT);
            }
        }
    }
    A_single_block_graph->fillComplete(new_domain_map, new_range_map);
    //A_single_block_graph->fillComplete();

    _A_single_block = Teuchos::rcp(new crs_matrix_type(A_single_block_graph));
    for (size_t i=0; i<A.size(); ++i) {
        for (size_t j=0; j<A[0].size(); ++j) {
            if (!A[i][j].is_null()) {
                _A_single_block->doExport(*(A[i][j]), *(_row_exporters[i][j]), Tpetra::INSERT);
                printf("nnz i: %lu, j: %lu, %lu\n", i, j, A[i][j]->getCrsGraph()->getGlobalNumEntries());
            }
        }
    }
    //_A_single_block->fillComplete();
    _A_single_block->fillComplete(new_domain_map, new_range_map);
    //auto out = Teuchos::getFancyOStream(Teuchos::rcpFromRef(std::cout));
    //_A_single_block->describe(*out, Teuchos::VERB_EXTREME);
    //printf("nnz single: %lu\n", A_single_block_graph->getGlobalNumEntries());

    _x_single_block = Teuchos::rcp(new mvec_type(_A_single_block->getDomainMap(),1,true/*set to zero*/));
    _b_single_block = Teuchos::rcp(new mvec_type(_A_single_block->getRangeMap(),1,true/*set to zero*/));
    for (size_t i=0; i<A.size(); ++i) {
        //_b_single_block->doExport(*(b[i]), *(_range_exporters[i][valid_j_for_i[i]]), Tpetra::INSERT);
        _b_single_block->doExport(*(b[i]), *(_range_exporters[i][i]), Tpetra::INSERT);
    }

}

void SolverT::prepareMatrices() {

    // determines if blocks of matrices need amalgamated into one giant matrix
    // calls for the amalgamation if necessary
    // checks if an iterative approach is used, and sets up Thyra objects if it is
        
    TEUCHOS_TEST_FOR_EXCEPT_MSG( _parameter_filename.empty() && _parameters->get<std::string>("type")!="direct", 
            "Parameters not set before solve() called." );

    // direct solvers only work on a single block system
    // if not blocked, then a single matrix must be constructed from blocks
    _consolidate_blocks = _A_tpetra.size()>1 
        && (_parameters->get<std::string>("type")=="direct" || !(_parameters->get<bool>("blocked")));
   
    if (_consolidate_blocks) {
	    this->amalgamateBlockMatrices(_A_tpetra, _b_tpetra);
    }

    if (_parameters->get<std::string>("type")!="direct") {
        _A_thyra = Thyra::defaultBlockedLinearOp<scalar_type>();
        if (_consolidate_blocks) {
            _A_thyra->beginBlockFill(1,1);
            auto range_space = Thyra::tpetraVectorSpace<scalar_type>( _A_single_block->getRangeMap() );
            auto domain_space = Thyra::tpetraVectorSpace<scalar_type>( _A_single_block->getDomainMap() );
            auto crs_as_operator = 
                Teuchos::rcp_implicit_cast<Tpetra::Operator<scalar_type, local_index_type, global_index_type> >(_A_single_block);
            auto tpetra_operator = Thyra::tpetraLinearOp<scalar_type>(range_space,domain_space,crs_as_operator);
            _A_thyra->setBlock(0, 0, tpetra_operator);
            _A_thyra->endBlockFill();
            _b_thyra.resize(1);
            Teuchos::RCP<const Thyra::TpetraVectorSpace<scalar_type,local_index_type,global_index_type,node_type> > range 
                = Thyra::tpetraVectorSpace<scalar_type>(_A_single_block->getRangeMap());
            _b_thyra[0] = Thyra::tpetraVector(range, Teuchos::rcp_static_cast<vec_scalar_type>(_b_single_block));
        } else {
            _A_thyra->beginBlockFill(_A_tpetra.size(), _A_tpetra[0].size());
            for (size_t i=0; i<_A_tpetra.size(); i++) {
                for (size_t j=0; j<_A_tpetra[0].size(); j++) {
                    if (!(_A_tpetra[i][j].is_null())) {
                        auto range_space = Thyra::tpetraVectorSpace<scalar_type>( _A_tpetra[i][j]->getRangeMap() );
                        auto domain_space = Thyra::tpetraVectorSpace<scalar_type>( _A_tpetra[i][j]->getDomainMap() );
                        auto crs_as_operator = 
                            Teuchos::rcp_implicit_cast<Tpetra::Operator<scalar_type, local_index_type, global_index_type> >(_A_tpetra[i][j]);
                        auto tpetra_operator = Thyra::tpetraLinearOp<scalar_type>(range_space,domain_space,crs_as_operator);
                        _A_thyra->setBlock(i, j, tpetra_operator);
                    }
                }
            }
            _A_thyra->endBlockFill();
            _b_thyra.resize(_b_tpetra.size());
            for (size_t i=0; i<_b_tpetra.size(); i++) {
                Teuchos::RCP<const Thyra::TpetraVectorSpace<scalar_type,local_index_type,global_index_type,node_type> > range 
                    = Thyra::tpetraVectorSpace<scalar_type>(_A_tpetra[i][i]->getRangeMap());
                _b_thyra[i] = Thyra::tpetraVector(range, Teuchos::rcp_static_cast<vec_scalar_type>(_b_tpetra[i]));
            }
        }
    }
}

SolverT::SolverT( const Teuchos::RCP<const Teuchos::Comm<int> > comm, std::vector<std::vector<Teuchos::RCP<crs_matrix_type> > > A, std::vector<Teuchos::RCP<mvec_type> > b) :
        _comm(comm), _consolidate_blocks(false)
{

    _A_tpetra = std::vector<std::vector<Teuchos::RCP<crs_matrix_type> > > (A.size(), std::vector<Teuchos::RCP<crs_matrix_type> >(A[0].size(), Teuchos::null));
    for (size_t i=0; i<A.size(); i++) {
        _A_tpetra.resize(A[i].size());
        for (size_t j=0; j<A[0].size(); j++) {
            if (!(A[i][j].is_null())) {
                _A_tpetra[i][j] = A[i][j];
            }
        }
    }

    _b_tpetra.resize(b.size());
    for (size_t i=0; i<b.size(); i++) {
        _b_tpetra[i] = b[i];
    }
}

void SolverT::solve() {


    TEUCHOS_TEST_FOR_EXCEPT_MSG( _parameter_filename.empty() && _parameters->get<std::string>("type")!="direct", 
            "Parameters not set before solve() called." );

    const bool setToZero = true;
    _x_tpetra.resize(_A_tpetra[0].size());
    for (size_t i=0; i<_A_tpetra[0].size(); i++) {
        _x_tpetra[i] = Teuchos::rcp(new mvec_type(_A_tpetra[i][i]->getDomainMap(),1,setToZero));
    }

    //// ASSUMES SQUARE (col and row #blocks have same cardinality)
    //_x_tpetra.resize(_b_tpetra.size());
    ////std::vector<Teko::MultiVector> x_teko(_b_tpetra.size()); // if this has to be size 1 for nonblocked, then it must be for b also

    //for (size_t i=0; i<_b_tpetra.size(); i++) {
    //    TEUCHOS_TEST_FOR_EXCEPT_MSG(_A_tpetra[i][i]->getDomainMap()==Teuchos::null, "Domain map null.");
    //    _x_tpetra[i] = Teuchos::rcp(new mvec_type(_A_tpetra[i][i]->getDomainMap(),1,setToZero));
    //    //Teuchos::RCP<const Thyra::TpetraVectorSpace<scalar_type,local_index_type,global_index_type,node_type> > domain = Thyra::tpetraVectorSpace<scalar_type>(_A_tpetra[i][i]->getDomainMap());
    //    //x_teko[i] = Thyra::tpetraVector(domain, Teuchos::rcp_static_cast<vec_scalar_type>(_x_tpetra[i]));
    //    //TEUCHOS_TEST_FOR_EXCEPT_MSG(x_teko[i].is_null(), "Cast failed.");
    //}


    // import _x_tpetra to _x_single_block

    // COPIED, DELETE LATER, but for now all solve directly
    //{
    //    //_x_single_block = Teuchos::rcp(new mvec_type(_A_single_block->getDomainMap(),1,setToZero));
    //    // Ensure that you do not configure for a direct solve and then call an iterative solver
    //    // Constructor from Factory
    //    Teuchos::RCP<Amesos2::Solver<crs_matrix_type,mvec_type> > solver;
    //    try{
    //        solver = Amesos2::create<crs_matrix_type,mvec_type>("KLU", _A_single_block, _x_single_block, _b_single_block);
    //    } catch (std::invalid_argument e){
    //        std::cout << e.what() << std::endl;
    //    }
    //    solver->solve();
    //    auto out = Teuchos::getFancyOStream(Teuchos::rcpFromRef(std::cout));
    //    solver->printTiming(*out, Teuchos::VERB_LOW);
    //    //if (_parameters->get<bool>("blocked")) {
    //        for (size_t i=0; i<_x_tpetra.size(); i++) {
    //            _x_tpetra[i]->doImport(*_x_single_block, *(_domain_exporters[i][i]), Tpetra::REPLACE);
    //            //_x_tpetra[i]->describe(*out, Teuchos::VERB_EXTREME);
    //        }
   ////auto out = Teuchos::getFancyOStream(Teuchos::rcpFromRef(std::cout));
   ////_x_single_block->describe(*out, Teuchos::VERB_EXTREME);
    //    //} else {
    //    //    _x_tpetra[i]->doExport(*_x_single_block, *(_row_exporters[i]), Tpetra::REPLACE);
    //    //}
    //}


    //// BEGINNING OF COMMENTING OUT
    ////// export from _x_single_block to _x_tpetra


   // //_x_tpetra.resize(_b_tpetra.size());
   // std::vector<Teko::MultiVector> x_teko(_b_thyra.size()); // if this has to be size 1 for nonblocked, then it must be for b also
   // //std::vector<Teko::MultiVector> x_teko(_b_tpetra.size()); // if this has to be size 1 for nonblocked, then it must be for b also

   // // TEMPORARILY we overwrite "blocked" to put it into one matrix (like in unblocked, but not using _A_tpetra)
   // // solve the matrix
   // // then unpack back into blocks
   // // once it works, make that the "unblocked" behavior in cases of a.) direct solve b.) unblocked is requested

   // const bool setToZero = true;
   // //if (_parameters->get<bool>("blocked")) {
   //     for (size_t i=0; i<_b_thyra.size(); i++) {
   //         Teuchos::RCP<const Thyra::TpetraVectorSpace<scalar_type,local_index_type,global_index_type,node_type> > domain = Thyra::tpetraVectorSpace<scalar_type>(_A_single_block->getDomainMap());
   //         x_teko[0] = Thyra::tpetraVector(domain, Teuchos::rcp_static_cast<vec_scalar_type>(_x_single_block));
   //         //_x_tpetra[i] = Teuchos::rcp(new mvec_type(_A_tpetra[i][i]->getDomainMap(),1,setToZero));
   //         //TEUCHOS_TEST_FOR_EXCEPT_MSG(_A_tpetra[i][i]->getDomainMap()==Teuchos::null, "Domain map null.");

   //         //Teuchos::RCP<const Thyra::TpetraVectorSpace<scalar_type,local_index_type,global_index_type,node_type> > domain = Thyra::tpetraVectorSpace<scalar_type>(_A_tpetra[i][i]->getDomainMap());
   //         //x_teko[i] = Thyra::tpetraVector(domain, Teuchos::rcp_static_cast<vec_scalar_type>(_x_tpetra[i]));

   //         //TEUCHOS_TEST_FOR_EXCEPT_MSG(x_teko[i].is_null(), "Cast failed.");
   //     }
   // //} else {
   // //    _x_tpetra[0] = Teuchos::rcp(new mvec_type(_A_tpetra[0][0]->getDomainMap(),1,setToZero));
   // //    Teuchos::RCP<const Thyra::TpetraVectorSpace<scalar_type,local_index_type,global_index_type,node_type> > domain = Thyra::tpetraVectorSpace<scalar_type>(_A_tpetra[0][0]->getDomainMap());
   // //    x_teko[0] = Thyra::tpetraVector(domain, Teuchos::rcp_static_cast<vec_scalar_type>(_x_tpetra[0]));

   // //    TEUCHOS_TEST_FOR_EXCEPT_MSG(x_teko[0].is_null(), "Cast failed.");
   // //}

   // //Teko::MultiVector Thyra_x = Teko::buildBlockedMultiVector(x_teko);

   // // Diagnostics:
   ////auto out = Teuchos::getFancyOStream(Teuchos::rcpFromRef(std::cout));
   //// Thyra_x->describe(*out, Teuchos::VERB_EXTREME);
   //// Thyra_b->describe(*out, Teuchos::VERB_EXTREME);
   ////_A_thyra->describe(*out, Teuchos::VERB_EXTREME);



    if (_parameters->get<std::string>("type")=="direct") {

        // Constructor from Factory
        Teuchos::RCP<Amesos2::Solver<crs_matrix_type,mvec_type> > solver;
        try{
            // either _consolidate_blocks is true or the blocked matrices are 1x1
            if (_consolidate_blocks) {
                solver = Amesos2::create<crs_matrix_type,mvec_type>("KLU", _A_single_block, _x_single_block, _b_single_block);
            } else { // _A_tpetra.size()==1
                solver = Amesos2::create<crs_matrix_type,mvec_type>("KLU", _A_tpetra[0][0], _x_tpetra[0], _b_tpetra[0]);
            }
        } catch (std::invalid_argument e){
            std::cout << e.what() << std::endl;
        }
        solver->solve();
        auto out = Teuchos::getFancyOStream(Teuchos::rcpFromRef(std::cout));
        solver->printTiming(*out, Teuchos::VERB_LOW);

    } else if (_parameters->get<std::string>("type")=="iterative") {

        // _b_thyra is sized for _consolidate_blocks or not
        std::vector<Teko::MultiVector> x_teko(_b_thyra.size()); 
        if (_consolidate_blocks) {
            Teuchos::RCP<const Thyra::TpetraVectorSpace<scalar_type,local_index_type,global_index_type,node_type> > domain 
                = Thyra::tpetraVectorSpace<scalar_type>(_A_single_block->getDomainMap());
            x_teko[0] = Thyra::tpetraVector(domain, Teuchos::rcp_static_cast<vec_scalar_type>(_x_single_block));
            TEUCHOS_TEST_FOR_EXCEPT_MSG(x_teko[0].is_null(), "Cast failed.");
        } else {
            for (size_t i=0; i<_b_thyra.size(); i++) {
                Teuchos::RCP<const Thyra::TpetraVectorSpace<scalar_type,local_index_type,global_index_type,node_type> > domain 
                    = Thyra::tpetraVectorSpace<scalar_type>(_A_tpetra[i][i]->getDomainMap());
                x_teko[i] = Thyra::tpetraVector(domain, Teuchos::rcp_static_cast<vec_scalar_type>(_x_tpetra[i]));
                TEUCHOS_TEST_FOR_EXCEPT_MSG(x_teko[i].is_null(), "Cast failed.");
            }
        }

        Thyra::SolveStatus<scalar_type> status;
        Teko::MultiVector Thyra_b = Teko::buildBlockedMultiVector(_b_thyra);
        Teko::MultiVector Thyra_x = Teko::buildBlockedMultiVector(x_teko);

        Teuchos::RCP<Teuchos::ParameterList> precList;
        // read in xml file from "file" into ParameterList
        if (_parameter_filename=="" || _parameter_filename=="default_solver") {
            if (_comm->getRank()==0) std::cout << "WARNING!: No preconditioner parameter file specified. Proceeding with default IFPACK2 parameters." << std::endl;
            precList = Teuchos::rcp(new Teuchos::ParameterList());
        } else {
            size_t pos = _parameter_filename.rfind('.', _parameter_filename.length());
            std::string extension = _parameter_filename.substr(pos+1, _parameter_filename.length() - pos);
            transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            if (extension == "xml") {
                precList = Compadre::ParameterManager::readInXML(_parameter_filename);
            } else if (extension == "yaml" || extension == "yml") {
                precList = Compadre::ParameterManager::readInYAML(_parameter_filename);
            } else { // _parameter_filename != ""
                std::ostringstream msg;
                msg << "Invalid parameter list file-type specified: " << extension.c_str() << std::endl;
                TEUCHOS_TEST_FOR_EXCEPT_MSG(true, msg.str());
            }
        }

        Teuchos::RCP<Stratimikos::DefaultLinearSolverBuilder> linearSolverBuilder = Teuchos::rcp(new Stratimikos::DefaultLinearSolverBuilder);

        typedef Thyra::PreconditionerFactoryBase<scalar_type> PreconditionerBase;
        typedef Thyra::Ifpack2PreconditionerFactory<crs_matrix_type> Ifpack2Impl;

        linearSolverBuilder->setPreconditioningStrategyFactory(Teuchos::abstractFactoryStd<PreconditionerBase, Ifpack2Impl>(), "Ifpack2");
        Stratimikos::enableMueLu<local_index_type,global_index_type,node_type>(*linearSolverBuilder,"MueLu");

        Teuchos::RCP<Teko::RequestHandler> requestHandler_;
        Teko::addTekoToStratimikosBuilder(*linearSolverBuilder,requestHandler_);

        linearSolverBuilder->setParameterList(precList);

        Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<scalar_type> > lowsFactory
            = linearSolverBuilder->createLinearSolveStrategy("");

        Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::VerboseObjectBase::getDefaultOStream();
        lowsFactory->setOStream(out);
        lowsFactory->setVerbLevel(Teuchos::VERB_HIGH);

        Teuchos::RCP<Thyra::LinearOpWithSolveBase<scalar_type> > th_invA
            = Thyra::linearOpWithSolve(*lowsFactory, Teuchos::rcp_dynamic_cast<const thyra_linearop_type>(_A_thyra));

        Thyra::assign(Thyra_x.ptr(), 0.0);
        status
            = Thyra::solve<scalar_type>(*th_invA, Thyra::NOTRANS, *Thyra_b, Thyra_x.ptr());

        *out << "\nSolve status:\n" << status;

//      // DIAGNOSTIC for multiply b by A and storing in x
//      Thyra::apply<scalar_type>(*Teuchos::rcp_dynamic_cast<const thyra_linearop_type>(_A_thyra), Thyra::NOTRANS, *Thyra_b, Thyra_x.ptr());

    }

    if (_consolidate_blocks) {
        // move solution from _x_single_block to _x_tpetra
        for (size_t i=0; i<_x_tpetra.size(); i++) {
            _x_tpetra[i]->doImport(*_x_single_block, *(_domain_exporters[i][i]), Tpetra::REPLACE);
            //_x_tpetra[i]->describe(*out, Teuchos::VERB_EXTREME);
        }
    }

//  TODO: fix this up for blocked and unblocked case
//
//    if (_parameters->get<bool>("save residual over solution")) {
//        // writes residual over top of the solution variable
//        Teuchos::RCP<mvec_type> Ax = Teuchos::rcp(new mvec_type(*x, Teuchos::View)); // shallow copy, so it writes over the solution
//        _A_thyra->apply(*x,*Ax);
//        Ax->update(-1.0, *_b, 1.0);
//        if (_comm->getRank() == 0) std::cout << "WARNING: Over-writing solution variables with the residual." << std::endl;
//    }
}

Teuchos::RCP<mvec_type> SolverT::getSolution(local_index_type idx) const {

    Teuchos::RCP<mvec_type> return_ptr;

    if (_parameters->get<bool>("blocked")) {
        return_ptr = _x_tpetra[idx];
    } else {
        return_ptr = _x_tpetra[0];
    }

    return return_ptr;

}

}

#endif
