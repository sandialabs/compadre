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

Teuchos::RCP<crs_matrix_type> amalgamateBlockMatrices(const Teuchos::RCP<const Teuchos::Comm<int> > comm, 
        std::vector<std::vector<Teuchos::RCP<crs_matrix_type> > > A) {

    Teuchos::RCP<map_type> new_row_map;
    Teuchos::RCP<map_type> new_range_map;

    { // creates new_row_map and new_range_map
        std::vector<global_index_type> global_ids_offsets_row(A.size());
        std::vector<global_index_type> global_ids_offsets_range(A.size());

        local_index_type local_row_count = 0;
        local_index_type local_range_count = 0;
        global_index_type global_row_count = 0;
        global_index_type global_range_count = 0;

	    // gets local and global offsets
        for (size_t i=0; i<A.size(); ++i) {
	    	global_ids_offsets_row[i] = global_row_count;
	    	global_ids_offsets_range[i] = global_range_count;

	    	local_row_count += A[i][0]->getRowMap()->getNodeNumElements();
	    	local_range_count += A[i][0]->getRangeMap()->getNodeNumElements();

	    	global_row_count += A[i][0]->getRowMap()->getGlobalNumElements();
	    	global_range_count += A[i][0]->getRangeMap()->getGlobalNumElements();
        }

	    Kokkos::View<global_index_type*, Kokkos::HostSpace> new_row_local_indices("local row indices", local_row_count);
	    Kokkos::View<global_index_type*, Kokkos::HostSpace> new_range_local_indices("local range indices", local_range_count);
        // copy over indices
        for (size_t i=0; i<A.size(); ++i) {

            auto row_indices = A[i][0]->getRowMap()->getMyGlobalIndices();
            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,row_indices.extent(0)), 
                        KOKKOS_LAMBDA(const int j) {
                new_row_local_indices(global_ids_offsets_row[i] + j) = row_indices(j);
            });
            auto range_indices = A[i][0]->getRangeMap()->getMyGlobalIndices();
            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,range_indices.extent(0)), 
                        KOKKOS_LAMBDA(const int j) {
                new_range_local_indices(global_ids_offsets_range[i] + j) = range_indices(j);
            });
        }

        new_row_map = Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), 
                new_row_local_indices, 0 /*index base*/, comm));
        new_range_map = Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), 
                new_range_local_indices, 0 /*index base*/, comm));
    } 

    Teuchos::RCP<map_type> new_col_map;
    Teuchos::RCP<map_type> new_domain_map;
    { // creates new_col_map and new_domain_map
        std::vector<global_index_type> global_ids_offsets_col(A.size());
        std::vector<global_index_type> global_ids_offsets_domain(A.size());

        local_index_type local_col_count = 0;
        local_index_type local_domain_count = 0;
        global_index_type global_col_count = 0;
        global_index_type global_domain_count = 0;

	    // gets local and global offsets
        for (size_t i=0; i<A.size(); ++i) {
	    	global_ids_offsets_col[i] = global_col_count;
	    	global_ids_offsets_domain[i] = global_domain_count;

	    	local_col_count += A[i][0]->getColMap()->getNodeNumElements();
	    	local_domain_count += A[i][0]->getDomainMap()->getNodeNumElements();

	    	global_col_count += A[i][0]->getColMap()->getGlobalNumElements();
	    	global_domain_count += A[i][0]->getDomainMap()->getGlobalNumElements();
        }

	    Kokkos::View<global_index_type*, Kokkos::HostSpace> new_col_local_indices("local col indices", local_col_count);
	    Kokkos::View<global_index_type*, Kokkos::HostSpace> new_domain_local_indices("local domain indices", local_domain_count);
        // copy over indices
        for (size_t i=0; i<A.size(); ++i) {

            auto col_indices = A[i][0]->getColMap()->getMyGlobalIndices();
            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,col_indices.extent(0)), 
                        KOKKOS_LAMBDA(const int j) {
                new_col_local_indices(global_ids_offsets_col[i] + j) = col_indices(j);
            });
            auto domain_indices = A[i][0]->getDomainMap()->getMyGlobalIndices();
            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,domain_indices.extent(0)), 
                        KOKKOS_LAMBDA(const int j) {
                new_domain_local_indices(global_ids_offsets_domain[i] + j) = domain_indices(j);
            });
        }

        new_col_map = Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), 
                new_col_local_indices, 0 /*index base*/, comm));
        new_domain_map = Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), 
                new_domain_local_indices, 0 /*index base*/, comm));
    }

    // get size for maximum entries per row
    local_index_type max_entries_per_row_over_all_rows = 0;
    for (size_t i=0; i<A.size(); ++i) {
        local_index_type max_entries_this_row = 0;
        for (size_t j=0; j<A[0].size(); ++j) {
            max_entries_this_row += A[i][j]->getGraph()->getGlobalMaxNumRowEntries();
        }
        max_entries_per_row_over_all_rows += max_entries_this_row;
    }

    // create giant graph using new maps
    auto A_single_block_graph = Teuchos::rcp(new crs_graph_type(new_row_map, new_col_map, max_entries_per_row_over_all_rows, Tpetra::StaticProfile));

    // 
    for (size_t i=0; i<A.size(); ++i) {
        for (size_t j=0; j<A[0].size(); ++j) {
//            
//            A[i][j]->getGraph()
//                getNodeRowPtrs  (       )
//
//
//            A[i][j]    getLocalValuesView  (       )
//            void    setAllValues (const typename local_matrix_type::row_map_type &ptr, const typename local_graph_type::entries_type::non_const_type &ind, const typename local_matrix_type::values_type &val)
//                Set the local matrix using three (compressed sparse row) arrays. More...
//                 
//                void    setAllValues (const Teuchos::ArrayRCP< size_t > &ptr, const Teuchos::ArrayRCP< LocalOrdinal > &ind, const Teuchos::ArrayRCP< Scalar > &val)
//                    Set the local matrix using three (compressed sparse row) arrays. More...
//                     
//                    void    getAllValues (Teuchos::ArrayRCP< const size_t > &rowPointers, Teuchos::ArrayRCP< const LocalOrdinal > &columnIndices, Teuchos::ArrayRCP< const Scalar > &values) const 
//            Teuchos::RCP<crs_graph_type> tmp_graph = Teuchos::null;
//            A[i][j]->getGraph()->importAndFilleComplete(tmp_graph, );
//
//            A_single_block_graph->doExport(*(A[i][j]->getGraph()), this_exporter, Tpetra::INSERT);
        }
    }

    // create giant matrix using new graph 
    auto A_single_block = Teuchos::rcp(new crs_matrix_type(A_single_block_graph));
                

    //A_single_block_graph->setDomainRangeMaps(new_range_map, new_domain_map);
    //// export from little graphs to big graph
    //A_single_block_graph->doExport( *oGraph, *exporter, Tpetra::INSERT );


    return A_single_block;

}

SolverT::SolverT( const Teuchos::RCP<const Teuchos::Comm<int> > comm, std::vector<std::vector<Teuchos::RCP<crs_matrix_type> > > A, std::vector<Teuchos::RCP<mvec_type> > b) :
        _comm(comm)
{

    // _A_tpetra used for direct solve and _A for Stratimikos, so focus on converting tpetra matrices,
    // and vectors, and then just cast first block with thyra (don't write for thyra)

	//amalgamateBlockMatrices(comm, A);
 
    _A_tpetra = std::vector<std::vector<Teuchos::RCP<crs_matrix_type> > > (A.size(), std::vector<Teuchos::RCP<crs_matrix_type> >(A[0].size(), Teuchos::null));
    _A_thyra = Thyra::defaultBlockedLinearOp<scalar_type>();

    // the following should only happen when really blocked, so comment out and replace
    _A_thyra->beginBlockFill(A.size(), A[0].size());
    for (size_t i=0; i<A.size(); i++) {
        _A_tpetra.resize(A[i].size());
        for (size_t j=0; j<A[0].size(); j++) {
            if (!(A[i][j].is_null())) {

                auto range_space = Thyra::tpetraVectorSpace<scalar_type>( A[i][j]->getRangeMap() );
                auto domain_space = Thyra::tpetraVectorSpace<scalar_type>( A[i][j]->getDomainMap() );
                auto crs_as_operator = 
                    Teuchos::rcp_implicit_cast<Tpetra::Operator<scalar_type, local_index_type, global_index_type> >(A[i][j]);
                auto tpetra_operator = Thyra::tpetraLinearOp<scalar_type>(range_space,domain_space,crs_as_operator);

                //if (i!=j) {
                //    auto out = Teuchos::getFancyOStream(Teuchos::rcpFromRef(std::cout));
                //    //tpetra_operator->describe(*out, Teuchos::VERB_EXTREME);
                //    auto t_mvec = mvec_type(A[i][j]->getDomainMap(),1);
                //    auto y_mvec = mvec_type(A[i][j]->getRangeMap(),1);
                //    t_mvec.putScalar(1.0);
                //    A[i][j]->apply(t_mvec, y_mvec);
                //    A[i][j]->getRangeMap()->describe(*out, Teuchos::VERB_EXTREME);
                //    A[i][j]->getDomainMap()->describe(*out, Teuchos::VERB_EXTREME);
                //    t_mvec.describe(*out, Teuchos::VERB_EXTREME);
                //    y_mvec.describe(*out, Teuchos::VERB_EXTREME);
                //    A[i][j]->describe(*out, Teuchos::VERB_EXTREME);
                //}

                _A_thyra->setBlock(i, j, tpetra_operator);
                _A_tpetra[i][j] = A[i][j];
            }
        }
    }
    _A_thyra->endBlockFill();

    // create _A_single_block from _A_tpetra





    
    //// TEMP CODE, _A_tpetra to Thyra for one block
    //_A_thyra = Thyra::defaultBlockedLinearOp<scalar_type>();
    //_A_thyra->beginBlockFill(1, 1);
    //auto range_space = Thyra::tpetraVectorSpace<scalar_type>( _A_single_block->getRangeMap() );
    //auto domain_space = Thyra::tpetraVectorSpace<scalar_type>( _A_single_block->getDomainMap() );
    //auto crs_as_operator = 
    //    Teuchos::rcp_implicit_cast<Tpetra::Operator<scalar_type, local_index_type, global_index_type> >(_A_single_block);
    //auto tpetra_operator = Thyra::tpetraLinearOp<scalar_type>(range_space,domain_space,crs_as_operator);
    //_A_thyra->setBlock(0, 0, tpetra_operator);
    //_A_thyra->endBlockFill();

    // this works for unblocked now, with size 1, but blocked needs changed to create a new b
    _b_thyra.resize(b.size());
    _b_tpetra.resize(b.size());
    for (size_t i=0; i<b.size(); i++) {
        Teuchos::RCP<const Thyra::TpetraVectorSpace<scalar_type,local_index_type,global_index_type,node_type> > range = Thyra::tpetraVectorSpace<scalar_type>(_A_tpetra[i][i]->getRangeMap());
        _b_thyra[i] = Thyra::tpetraVector(range, Teuchos::rcp_static_cast<vec_scalar_type>(b[i]));
        _b_tpetra[i] = b[i];
    }

    // create _b_single_block from _b_tpetra
    



  
}

void SolverT::solve() {
    TEUCHOS_TEST_FOR_EXCEPT_MSG( _parameter_filename.empty() && _parameters->get<std::string>("type")!="direct", "Parameters not set before solve() called." );

    //const bool setToZero = true;

    //// ASSUMES SQUARE (col and row #blocks have same cardinality)
    //_x_tpetra.resize(_b_tpetra.size());
    ////std::vector<Teko::MultiVector> x_teko(_b_tpetra.size()); // if this has to be size 1 for nonblocked, then it must be for b also

    //for (size_t i=0; i<_b_thyra.size(); i++) {
    //    TEUCHOS_TEST_FOR_EXCEPT_MSG(_A_tpetra[i][i]->getDomainMap()==Teuchos::null, "Domain map null.");
    //    _x_tpetra[i] = Teuchos::rcp(new mvec_type(_A_tpetra[i][i]->getDomainMap(),1,setToZero));
    //    //Teuchos::RCP<const Thyra::TpetraVectorSpace<scalar_type,local_index_type,global_index_type,node_type> > domain = Thyra::tpetraVectorSpace<scalar_type>(_A_tpetra[i][i]->getDomainMap());
    //    //x_teko[i] = Thyra::tpetraVector(domain, Teuchos::rcp_static_cast<vec_scalar_type>(_x_tpetra[i]));
    //    //TEUCHOS_TEST_FOR_EXCEPT_MSG(x_teko[i].is_null(), "Cast failed.");
    //}


    //// import _x_tpetra to _x_single_block

    //// COPIED, DELETE LATER, but for now all solve directly
    //{
    //    _x_single_block = Teuchos::rcp(new mvec_type(_A_single_block->getDomainMap(),1,setToZero));
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
    //}

    //// export from _x_single_block to _x_tpetra


    _x_tpetra.resize(_b_tpetra.size());
    std::vector<Teko::MultiVector> x_teko(_b_tpetra.size()); // if this has to be size 1 for nonblocked, then it must be for b also

    // TEMPORARILY we overwrite "blocked" to put it into one matrix (like in unblocked, but not using _A_tpetra)
    // solve the matrix
    // then unpack back into blocks
    // once it works, make that the "unblocked" behavior in cases of a.) direct solve b.) unblocked is requested

    const bool setToZero = true;
    if (_parameters->get<bool>("blocked")) {
        for (size_t i=0; i<_b_thyra.size(); i++) {
            _x_tpetra[i] = Teuchos::rcp(new mvec_type(_A_tpetra[i][i]->getDomainMap(),1,setToZero));
            TEUCHOS_TEST_FOR_EXCEPT_MSG(_A_tpetra[i][i]->getDomainMap()==Teuchos::null, "Domain map null.");

            Teuchos::RCP<const Thyra::TpetraVectorSpace<scalar_type,local_index_type,global_index_type,node_type> > domain = Thyra::tpetraVectorSpace<scalar_type>(_A_tpetra[i][i]->getDomainMap());
            x_teko[i] = Thyra::tpetraVector(domain, Teuchos::rcp_static_cast<vec_scalar_type>(_x_tpetra[i]));

            TEUCHOS_TEST_FOR_EXCEPT_MSG(x_teko[i].is_null(), "Cast failed.");
        }
    } else {
        _x_tpetra[0] = Teuchos::rcp(new mvec_type(_A_tpetra[0][0]->getDomainMap(),1,setToZero));
        Teuchos::RCP<const Thyra::TpetraVectorSpace<scalar_type,local_index_type,global_index_type,node_type> > domain = Thyra::tpetraVectorSpace<scalar_type>(_A_tpetra[0][0]->getDomainMap());
        x_teko[0] = Thyra::tpetraVector(domain, Teuchos::rcp_static_cast<vec_scalar_type>(_x_tpetra[0]));

        TEUCHOS_TEST_FOR_EXCEPT_MSG(x_teko[0].is_null(), "Cast failed.");
    }

    Teko::MultiVector Thyra_b = Teko::buildBlockedMultiVector(_b_thyra);
    Teko::MultiVector Thyra_x = Teko::buildBlockedMultiVector(x_teko);

    // Diagnostics:
   // auto out = Teuchos::getFancyOStream(Teuchos::rcpFromRef(std::cout));
   // Thyra_x->describe(*out, Teuchos::VERB_EXTREME);
   // Thyra_b->describe(*out, Teuchos::VERB_EXTREME);
   // _A_thyra->describe(*out, Teuchos::VERB_EXTREME);

    Thyra::SolveStatus<scalar_type> status;


    if (_parameters->get<std::string>("type")=="direct") {
        TEUCHOS_TEST_FOR_EXCEPT_MSG(_parameters->get<bool>("blocked"), "Blocked matrix used is incompatible with direct solves.");
        // Ensure that you do not configure for a direct solve and then call an iterative solver

        // Constructor from Factory
        Teuchos::RCP<Amesos2::Solver<crs_matrix_type,mvec_type> > solver;
        try{
            solver = Amesos2::create<crs_matrix_type,mvec_type>("KLU", _A_tpetra[0][0], _x_tpetra[0], _b_tpetra[0]);
        } catch (std::invalid_argument e){
            std::cout << e.what() << std::endl;
        }

        solver->solve();
        auto out = Teuchos::getFancyOStream(Teuchos::rcpFromRef(std::cout));
        solver->printTiming(*out, Teuchos::VERB_LOW);

    } else if (_parameters->get<std::string>("type")=="iterative") {

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

//        // DIAGNOSTIC for multiply b by A and storing in x
//        Thyra::apply<scalar_type>(*Teuchos::rcp_dynamic_cast<const thyra_linearop_type>(_A_thyra), Thyra::NOTRANS, *Thyra_b, Thyra_x.ptr());

        //needs "blocked" case temporarily of unpacking x vector

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
