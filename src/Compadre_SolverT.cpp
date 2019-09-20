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


namespace Compadre {

    // PACKING AND UNPACKING WILL BE ALL DONE WITH BLOCKED IN THE FUTURE
    // _x must be correct size in order to unpack
    // goal is to write something that takes blocked thyra matrix and blocked thyra vectors
    // and returns a single thyra matrix and vectors that are not blocked
    //
    // next step is to optionally solve on these
    //
    // next step is to export from unblocked thyra vec to blocked thyra vec
    // then unpacking from _x to particles will work as is currently implemented
    //
    // need to solve blocked system with multiple fields being solved for as a good test case

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

        std::vector<global_index_type> local_ids_offsets_row(A.size());
        std::vector<global_index_type> local_ids_offsets_range(A.size());

        local_index_type local_row_count = 0;
        local_index_type local_range_count = 0;

	    // gets local and global offsets
        for (size_t i=0; i<A.size(); ++i) {
	    	local_ids_offsets_row[i] = local_row_count;
	    	local_ids_offsets_range[i] = local_range_count;

            // not all columns necessarily have a valid matrix in them
            global_index_type max_row_count = 0;
            global_index_type max_range_count = 0;
            for (size_t j=0; j<A[0].size(); ++j) {
                if (!A[i][j].is_null()) {
                    //max_row_count = std::max(max_row_count, A[i][j]->getRowMap()->getNodeNumElements());
                    //max_range_count = std::max(max_row_count, A[i][j]->getRangeMap()->getNodeNumElements());
                    //printf("i: %d, j: %d, r: %d, size: %lu\n", i, j, _comm->getRank(), A[i][j]->getRowMap()->getNodeNumElements());
	    	        //local_row_count += A[i][j]->getRowMap()->getNodeNumElements();
	    	        //local_range_count += A[i][j]->getRangeMap()->getNodeNumElements();
	                ////local_row_count += A[i][j]->getRowMap()->getMyGlobalIndices();
	                ////local_range_count += A[i][j]->getRangeMap()->getMyGlobalIndices();
                    if (A[i][j]->getRowMap()->getNodeNumElements() > max_row_count) {
                        max_row_count = A[i][j]->getRowMap()->getNodeNumElements();
                        max_range_count = A[i][j]->getRangeMap()->getNodeNumElements();
                        valid_j_for_i[i] = j;
                        printf("i: %d, j: %d, r: %d, row_size: %lu\n", i, j, _comm->getRank(), A[i][j]->getRowMap()->getNodeNumElements());
                        printf("i: %d, j: %d, r: %d, ran_size: %lu\n", i, j, _comm->getRank(), A[i][j]->getRangeMap()->getNodeNumElements());
                    }
                    //valid_j_for_i[i] = j;
                    //break;
                }
            }
	        local_row_count += max_row_count;
	        local_range_count += max_range_count;
            printf("i: %d, r: %d, lrow: %lu, lrange: %lu\n", i, _comm->getRank(), local_ids_offsets_row[i], local_ids_offsets_range[i]);
        }

        printf("r: %d, lrow: %lu, lrange: %lu\n", _comm->getRank(), local_row_count, local_range_count);
	    Kokkos::View<global_index_type*, Kokkos::HostSpace> new_row_local_indices("local row indices", local_row_count);
	    Kokkos::View<global_index_type*, Kokkos::HostSpace> new_range_local_indices("local range indices", local_range_count);
        // copy over indices
        for (size_t i=0; i<A.size(); ++i) {

            auto row_indices = A[i][valid_j_for_i[i]]->getRowMap()->getMyGlobalIndices();
            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,row_indices.extent(0)), 
                        KOKKOS_LAMBDA(const int j) {
                new_row_local_indices(local_ids_offsets_row[i] + j) = row_indices(j);
            });

            auto range_indices = A[i][valid_j_for_i[i]]->getRangeMap()->getMyGlobalIndices();
            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,range_indices.extent(0)), 
                        KOKKOS_LAMBDA(const int j) {
                new_range_local_indices(local_ids_offsets_range[i] + j) = range_indices(j);
            });
            //for (size_t j=0; j<row_indices.extent(0); ++j) {
            //    if (i==0 && _comm->getRank()==0) {
            //        printf("i: %d, r: %d, j: %d, val: %d\n", i, _comm->getRank(), j, row_indices(j));
            //    }
            //}
        }

        new_row_map = Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), 
                new_row_local_indices, 0 /*index base*/, _comm));
        new_range_map = Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), 
                new_range_local_indices, 0 /*index base*/, _comm));
    } 

    Teuchos::RCP<map_type> new_col_map;
    Teuchos::RCP<map_type> new_domain_map;
    std::vector<local_index_type> valid_i_for_j(A[0].size());
    { // creates new_col_map and new_domain_map

        std::vector<global_index_type> local_ids_offsets_col(A[0].size());
        std::vector<global_index_type> local_ids_offsets_domain(A[0].size());

        local_index_type local_col_count = 0;
        local_index_type local_domain_count = 0;

	    // gets local and global offsets
        for (size_t j=0; j<A[0].size(); ++j) {
	    	local_ids_offsets_col[j] = local_col_count;
	    	local_ids_offsets_domain[j] = local_domain_count;

            // not all columns necessarily have a valid matrix in them
            global_index_type max_col_count = 0;
            global_index_type max_domain_count = 0;
            for (size_t i=0; i<A.size(); ++i) {
                if (!A[i][j].is_null()) {
                    //max_row_count = std::max(max_row_count, A[i][j]->getRowMap()->getNodeNumElements());
                    //max_range_count = std::max(max_row_count, A[i][j]->getRangeMap()->getNodeNumElements());
                    //printf("i: %d, j: %d, r: %d, size: %lu\n", i, j, _comm->getRank(), A[i][j]->getRowMap()->getNodeNumElements());
	    	        //local_row_count += A[i][j]->getRowMap()->getNodeNumElements();
	    	        //local_range_count += A[i][j]->getRangeMap()->getNodeNumElements();
	                ////local_row_count += A[i][j]->getRowMap()->getMyGlobalIndices();
	                ////local_range_count += A[i][j]->getRangeMap()->getMyGlobalIndices();
                    if (A[i][j]->getColMap()->getNodeNumElements() > max_col_count) {
                        max_col_count = A[i][j]->getColMap()->getNodeNumElements();
                        max_domain_count = A[i][j]->getDomainMap()->getNodeNumElements();
                        valid_i_for_j[j] = i;
                        //printf("i: %d, j: %d, r: %d, row_size: %lu\n", i, j, _comm->getRank(), A[i][j]->getRowMap()->getNodeNumElements());
                        //printf("i: %d, j: %d, r: %d, ran_size: %lu\n", i, j, _comm->getRank(), A[i][j]->getRangeMap()->getNodeNumElements());
                    }
                    //valid_j_for_i[i] = j;
                    //break;
                }
            }
	        local_col_count += max_col_count;
	        local_domain_count += max_domain_count;
            //printf("i: %d, r: %d, lrow: %lu, lrange: %lu\n", i, _comm->getRank(), local_ids_offsets_row[i], local_ids_offsets_range[i]);

            //// not all columns necessarily have a valid matrix in them
            //for (size_t i=0; i<A.size(); ++i) {
            //    if (!A[i][j].is_null()) {
	        //        local_col_count += A[i][j]->getColMap()->getNodeNumElements();
	        //        local_domain_count += A[i][j]->getDomainMap()->getNodeNumElements();
	        //        //local_col_count += A[i][j]->getColMap()->getMyGlobalIndices();
	        //        //local_domain_count += A[i][j]->getDomainMap()->getMyGlobalIndices();
            //        valid_i_for_j[j] = i;
            //        break;
            //    }
            //}
            //printf("j: %d, r: %d, lcol: %lu, ldomain: %lu\n", j, _comm->getRank(), local_col_count, local_domain_count);
        }

	    Kokkos::View<global_index_type*, Kokkos::HostSpace> new_col_local_indices("local col indices", local_col_count);
	    Kokkos::View<global_index_type*, Kokkos::HostSpace> new_domain_local_indices("local domain indices", local_domain_count);
        // copy over indices
        for (size_t j=0; j<A[0].size(); ++j) {

            auto col_indices = A[valid_i_for_j[j]][j]->getColMap()->getMyGlobalIndices();
            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,col_indices.extent(0)), 
                        KOKKOS_LAMBDA(const int i) {
                new_col_local_indices(local_ids_offsets_col[j] + i) = col_indices(i);
            });
            auto domain_indices = A[valid_i_for_j[j]][j]->getDomainMap()->getMyGlobalIndices();
            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,domain_indices.extent(0)), 
                        KOKKOS_LAMBDA(const int i) {
                new_domain_local_indices(local_ids_offsets_domain[j] + i) = domain_indices(i);
            });

        }

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
        max_entries_per_row_over_all_rows = std::max(max_entries_per_row_over_all_rows, max_entries_this_row);
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
                A_single_block_graph->doExport(*(A[i][j]->getCrsGraph()), *(_range_exporters[i][j]), Tpetra::REPLACE);
            }
        }
    }
    A_single_block_graph->fillComplete();

    _A_single_block = Teuchos::rcp(new crs_matrix_type(A_single_block_graph));
    for (size_t i=0; i<A.size(); ++i) {
        for (size_t j=0; j<A[0].size(); ++j) {
            if (!A[i][j].is_null()) {
                _A_single_block->doExport(*(A[i][j]), *(_range_exporters[i][j]), Tpetra::REPLACE);
                printf("nnz i: %lu, j: %lu, %lu\n", i, j, A[i][j]->getCrsGraph()->getGlobalNumEntries());
            }
        }
    }
    _A_single_block->fillComplete();
    auto out = Teuchos::getFancyOStream(Teuchos::rcpFromRef(std::cout));
    _A_single_block->describe(*out, Teuchos::VERB_EXTREME);
    //printf("nnz single: %lu\n", A_single_block_graph->getGlobalNumEntries());

    _x_single_block = Teuchos::rcp(new mvec_type(_A_single_block->getDomainMap(),1,true/*set to zero*/));
    _b_single_block = Teuchos::rcp(new mvec_type(_A_single_block->getRangeMap(),1,true/*set to zero*/));
    for (size_t i=0; i<A.size(); ++i) {
        _b_single_block->doExport(*(b[i]), *(_range_exporters[i][valid_j_for_i[i]]), Tpetra::INSERT);
    }

}

SolverT::SolverT( const Teuchos::RCP<const Teuchos::Comm<int> > comm, std::vector<std::vector<Teuchos::RCP<crs_matrix_type> > > A, std::vector<Teuchos::RCP<mvec_type> > b) :
        _comm(comm)
{

    // _A_tpetra used for direct solve and _A for Stratimikos, so focus on converting tpetra matrices,
    // and vectors, and then just cast first block with thyra (don't write for thyra)

	this->amalgamateBlockMatrices(A, b);
 
    _A_tpetra = std::vector<std::vector<Teuchos::RCP<crs_matrix_type> > > (A.size(), std::vector<Teuchos::RCP<crs_matrix_type> >(A[0].size(), Teuchos::null));
    //_A_thyra = Thyra::defaultBlockedLinearOp<scalar_type>();

    // the following should only happen when really blocked, so comment out and replace
    //_A_thyra->beginBlockFill(A.size(), A[0].size());
    for (size_t i=0; i<A.size(); i++) {
        _A_tpetra.resize(A[i].size());
        for (size_t j=0; j<A[0].size(); j++) {
            if (!(A[i][j].is_null())) {

                //auto range_space = Thyra::tpetraVectorSpace<scalar_type>( A[i][j]->getRangeMap() );
                //auto domain_space = Thyra::tpetraVectorSpace<scalar_type>( A[i][j]->getDomainMap() );
                //auto crs_as_operator = 
                //    Teuchos::rcp_implicit_cast<Tpetra::Operator<scalar_type, local_index_type, global_index_type> >(A[i][j]);
                //auto tpetra_operator = Thyra::tpetraLinearOp<scalar_type>(range_space,domain_space,crs_as_operator);

                ////if (i!=j) {
                ////    auto out = Teuchos::getFancyOStream(Teuchos::rcpFromRef(std::cout));
                ////    //tpetra_operator->describe(*out, Teuchos::VERB_EXTREME);
                ////    auto t_mvec = mvec_type(A[i][j]->getDomainMap(),1);
                ////    auto y_mvec = mvec_type(A[i][j]->getRangeMap(),1);
                ////    t_mvec.putScalar(1.0);
                ////    A[i][j]->apply(t_mvec, y_mvec);
                ////    A[i][j]->getRangeMap()->describe(*out, Teuchos::VERB_EXTREME);
                ////    A[i][j]->getDomainMap()->describe(*out, Teuchos::VERB_EXTREME);
                ////    t_mvec.describe(*out, Teuchos::VERB_EXTREME);
                ////    y_mvec.describe(*out, Teuchos::VERB_EXTREME);
                ////    A[i][j]->describe(*out, Teuchos::VERB_EXTREME);
                ////}

    //          //  _A_thyra->setBlock(i, j, tpetra_operator);
                _A_tpetra[i][j] = A[i][j];
            }
        }
    }
    //_A_thyra->endBlockFill();

    //// create _A_single_block from _A_tpetra





    //
    ////// TEMP CODE, _A_tpetra to Thyra for one block
    ////_A_thyra = Thyra::defaultBlockedLinearOp<scalar_type>();
    ////_A_thyra->beginBlockFill(1, 1);
    ////auto range_space = Thyra::tpetraVectorSpace<scalar_type>( _A_single_block->getRangeMap() );
    ////auto domain_space = Thyra::tpetraVectorSpace<scalar_type>( _A_single_block->getDomainMap() );
    ////auto crs_as_operator = 
    ////    Teuchos::rcp_implicit_cast<Tpetra::Operator<scalar_type, local_index_type, global_index_type> >(_A_single_block);
    ////auto tpetra_operator = Thyra::tpetraLinearOp<scalar_type>(range_space,domain_space,crs_as_operator);
    ////_A_thyra->setBlock(0, 0, tpetra_operator);
    ////_A_thyra->endBlockFill();

    //// this works for unblocked now, with size 1, but blocked needs changed to create a new b
    //_b_thyra.resize(b.size());
    _b_tpetra.resize(b.size());
    for (size_t i=0; i<b.size(); i++) {
    //    Teuchos::RCP<const Thyra::TpetraVectorSpace<scalar_type,local_index_type,global_index_type,node_type> > range = Thyra::tpetraVectorSpace<scalar_type>(_A_tpetra[i][i]->getRangeMap());
    //    _b_thyra[i] = Thyra::tpetraVector(range, Teuchos::rcp_static_cast<vec_scalar_type>(b[i]));
        _b_tpetra[i] = b[i];
    }

    const bool setToZero = true;
    _x_tpetra.resize(A[0].size());
    for (size_t i=0; i<A[0].size(); i++) {
        _x_tpetra[i] = Teuchos::rcp(new mvec_type(_A_tpetra[i][i]->getDomainMap(),1,setToZero));
    }
    //// create _b_single_block from _b_tpetra
    



  
}

void SolverT::solve() {
    TEUCHOS_TEST_FOR_EXCEPT_MSG( _parameter_filename.empty() && _parameters->get<std::string>("type")!="direct", "Parameters not set before solve() called." );

    //const bool setToZero = true;

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
    {
        //_x_single_block = Teuchos::rcp(new mvec_type(_A_single_block->getDomainMap(),1,setToZero));
        // Ensure that you do not configure for a direct solve and then call an iterative solver
        // Constructor from Factory
        Teuchos::RCP<Amesos2::Solver<crs_matrix_type,mvec_type> > solver;
        try{
            solver = Amesos2::create<crs_matrix_type,mvec_type>("KLU", _A_single_block, _x_single_block, _b_single_block);
        } catch (std::invalid_argument e){
            std::cout << e.what() << std::endl;
        }
        solver->solve();
        auto out = Teuchos::getFancyOStream(Teuchos::rcpFromRef(std::cout));
        solver->printTiming(*out, Teuchos::VERB_LOW);
        //if (_parameters->get<bool>("blocked")) {
            for (size_t i=0; i<_x_tpetra.size(); i++) {
                _x_tpetra[i]->doImport(*_x_single_block, *(_domain_exporters[i][i]), Tpetra::REPLACE);
                //_x_tpetra[i]->describe(*out, Teuchos::VERB_EXTREME);
            }
   //auto out = Teuchos::getFancyOStream(Teuchos::rcpFromRef(std::cout));
   //_x_single_block->describe(*out, Teuchos::VERB_EXTREME);
        //} else {
        //    _x_tpetra[i]->doExport(*_x_single_block, *(_row_exporters[i]), Tpetra::REPLACE);
        //}
    }

    ////// export from _x_single_block to _x_tpetra


    //_x_tpetra.resize(_b_tpetra.size());
    //std::vector<Teko::MultiVector> x_teko(_b_tpetra.size()); // if this has to be size 1 for nonblocked, then it must be for b also

    //// TEMPORARILY we overwrite "blocked" to put it into one matrix (like in unblocked, but not using _A_tpetra)
    //// solve the matrix
    //// then unpack back into blocks
    //// once it works, make that the "unblocked" behavior in cases of a.) direct solve b.) unblocked is requested

    //const bool setToZero = true;
    //if (_parameters->get<bool>("blocked")) {
    //    for (size_t i=0; i<_b_thyra.size(); i++) {
    //        _x_tpetra[i] = Teuchos::rcp(new mvec_type(_A_tpetra[i][i]->getDomainMap(),1,setToZero));
    //        TEUCHOS_TEST_FOR_EXCEPT_MSG(_A_tpetra[i][i]->getDomainMap()==Teuchos::null, "Domain map null.");

    //        Teuchos::RCP<const Thyra::TpetraVectorSpace<scalar_type,local_index_type,global_index_type,node_type> > domain = Thyra::tpetraVectorSpace<scalar_type>(_A_tpetra[i][i]->getDomainMap());
    //        x_teko[i] = Thyra::tpetraVector(domain, Teuchos::rcp_static_cast<vec_scalar_type>(_x_tpetra[i]));

    //        TEUCHOS_TEST_FOR_EXCEPT_MSG(x_teko[i].is_null(), "Cast failed.");
    //    }
    //} else {
    //    _x_tpetra[0] = Teuchos::rcp(new mvec_type(_A_tpetra[0][0]->getDomainMap(),1,setToZero));
    //    Teuchos::RCP<const Thyra::TpetraVectorSpace<scalar_type,local_index_type,global_index_type,node_type> > domain = Thyra::tpetraVectorSpace<scalar_type>(_A_tpetra[0][0]->getDomainMap());
    //    x_teko[0] = Thyra::tpetraVector(domain, Teuchos::rcp_static_cast<vec_scalar_type>(_x_tpetra[0]));

    //    TEUCHOS_TEST_FOR_EXCEPT_MSG(x_teko[0].is_null(), "Cast failed.");
    //}

    //Teko::MultiVector Thyra_b = Teko::buildBlockedMultiVector(_b_thyra);
    //Teko::MultiVector Thyra_x = Teko::buildBlockedMultiVector(x_teko);

    //// Diagnostics:
   //// auto out = Teuchos::getFancyOStream(Teuchos::rcpFromRef(std::cout));
   //// Thyra_x->describe(*out, Teuchos::VERB_EXTREME);
   //// Thyra_b->describe(*out, Teuchos::VERB_EXTREME);
   //// _A_thyra->describe(*out, Teuchos::VERB_EXTREME);

    //Thyra::SolveStatus<scalar_type> status;


    //if (_parameters->get<std::string>("type")=="direct") {
    //    TEUCHOS_TEST_FOR_EXCEPT_MSG(_parameters->get<bool>("blocked"), "Blocked matrix used is incompatible with direct solves.");
    //    // Ensure that you do not configure for a direct solve and then call an iterative solver

    //    // Constructor from Factory
    //    Teuchos::RCP<Amesos2::Solver<crs_matrix_type,mvec_type> > solver;
    //    try{
    //        solver = Amesos2::create<crs_matrix_type,mvec_type>("KLU", _A_tpetra[0][0], _x_tpetra[0], _b_tpetra[0]);
    //    } catch (std::invalid_argument e){
    //        std::cout << e.what() << std::endl;
    //    }

    //    solver->solve();
    //    auto out = Teuchos::getFancyOStream(Teuchos::rcpFromRef(std::cout));
    //    solver->printTiming(*out, Teuchos::VERB_LOW);

    //} else if (_parameters->get<std::string>("type")=="iterative") {

    //    Teuchos::RCP<Teuchos::ParameterList> precList;
    //    // read in xml file from "file" into ParameterList
    //    if (_parameter_filename=="" || _parameter_filename=="default_solver") {
    //        if (_comm->getRank()==0) std::cout << "WARNING!: No preconditioner parameter file specified. Proceeding with default IFPACK2 parameters." << std::endl;
    //        precList = Teuchos::rcp(new Teuchos::ParameterList());
    //    } else {
    //        size_t pos = _parameter_filename.rfind('.', _parameter_filename.length());
    //        std::string extension = _parameter_filename.substr(pos+1, _parameter_filename.length() - pos);
    //        transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    //        if (extension == "xml") {
    //            precList = Compadre::ParameterManager::readInXML(_parameter_filename);
    //        } else if (extension == "yaml" || extension == "yml") {
    //            precList = Compadre::ParameterManager::readInYAML(_parameter_filename);
    //        } else { // _parameter_filename != ""
    //            std::ostringstream msg;
    //            msg << "Invalid parameter list file-type specified: " << extension.c_str() << std::endl;
    //            TEUCHOS_TEST_FOR_EXCEPT_MSG(true, msg.str());
    //        }
    //    }

    //    Teuchos::RCP<Stratimikos::DefaultLinearSolverBuilder> linearSolverBuilder = Teuchos::rcp(new Stratimikos::DefaultLinearSolverBuilder);

    //    typedef Thyra::PreconditionerFactoryBase<scalar_type> PreconditionerBase;
    //    typedef Thyra::Ifpack2PreconditionerFactory<crs_matrix_type> Ifpack2Impl;

    //    linearSolverBuilder->setPreconditioningStrategyFactory(Teuchos::abstractFactoryStd<PreconditionerBase, Ifpack2Impl>(), "Ifpack2");
    //    Stratimikos::enableMueLu<local_index_type,global_index_type,node_type>(*linearSolverBuilder,"MueLu");

    //    Teuchos::RCP<Teko::RequestHandler> requestHandler_;
    //    Teko::addTekoToStratimikosBuilder(*linearSolverBuilder,requestHandler_);

    //    linearSolverBuilder->setParameterList(precList);

    //    Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<scalar_type> > lowsFactory
    //        = linearSolverBuilder->createLinearSolveStrategy("");

    //    Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::VerboseObjectBase::getDefaultOStream();
    //    lowsFactory->setOStream(out);
    //    lowsFactory->setVerbLevel(Teuchos::VERB_HIGH);

    //    Teuchos::RCP<Thyra::LinearOpWithSolveBase<scalar_type> > th_invA
    //        = Thyra::linearOpWithSolve(*lowsFactory, Teuchos::rcp_dynamic_cast<const thyra_linearop_type>(_A_thyra));

    //    Thyra::assign(Thyra_x.ptr(), 0.0);
    //    status
    //        = Thyra::solve<scalar_type>(*th_invA, Thyra::NOTRANS, *Thyra_b, Thyra_x.ptr());

    //    *out << "\nSolve status:\n" << status;

//        // DIAGNOSTIC for multiply b by A and storing in x
//        Thyra::apply<scalar_type>(*Teuchos::rcp_dynamic_cast<const thyra_linearop_type>(_A_thyra), Thyra::NOTRANS, *Thyra_b, Thyra_x.ptr());

        //needs "blocked" case temporarily of unpacking x vector

    //}

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
