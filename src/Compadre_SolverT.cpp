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

    // This function can handle row map != range map (row dof shared between many processors, 
    // but owned by only one). However, even though the matrix is correct, it can't be handled
    // properly by Amesos2, Ifpack2, or MueLu with more than one level of coarsening.
    
    // There is an assumption on row and column maps provided from the blocks, namely that 
    // locally owned DOFs are ordered first, then non-owned DOFs appear afterwards.

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
        TEUCHOS_TEST_FOR_EXCEPT_MSG(ghost_row_indices.size()>0, "Row map != Range map will not work with most solvers.");

	    Kokkos::View<global_index_type*, Kokkos::HostSpace> new_row_local_indices("local row indices", all_row_indices.size() + ghost_row_indices.size());
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
            }
        }
        //max_entries_per_row_over_all_rows = std::max(max_entries_per_row_over_all_rows, max_entries_this_row);
        max_entries_per_row_over_all_rows += max_entries_this_row;
    }

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

    for (size_t i=0; i<A.size(); ++i) {
        for (size_t j=0; j<A[0].size(); ++j) {
            if (!A[i][j].is_null()) {
                A_single_block_graph->doExport(*(A[i][j]->getCrsGraph()), *(_row_exporters[i][j]), Tpetra::INSERT);
            }
        }
    }
    A_single_block_graph->fillComplete(new_domain_map, new_range_map);

    _A_single_block = Teuchos::rcp(new crs_matrix_type(A_single_block_graph));
    for (size_t i=0; i<A.size(); ++i) {
        for (size_t j=0; j<A[0].size(); ++j) {
            if (!A[i][j].is_null()) {
                _A_single_block->doExport(*(A[i][j]), *(_row_exporters[i][j]), Tpetra::INSERT);
            }
        }
    }
    _A_single_block->fillComplete(new_domain_map, new_range_map);

    _x_single_block = Teuchos::rcp(new mvec_type(_A_single_block->getDomainMap(),1,true/*set to zero*/));
    _b_single_block = Teuchos::rcp(new mvec_type(_A_single_block->getRangeMap(),1,true/*set to zero*/));
    for (size_t i=0; i<A.size(); ++i) {
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

    if (_parameters->get<std::string>("type")=="direct") {

        //// DIAGNOSTICS
        //// extra mat-vec then print sum
        //_A_tpetra[0][0]->apply (*_b_tpetra[0], *_x_tpetra[0], Teuchos::NO_TRANS);
        //auto x_view = _x_tpetra[0]->getLocalView<Compadre::host_view_type>();
        //double sum = 0;
        //for (size_t j=0; j<x_view.extent(0); ++j) {
        //    sum += x_view(j,0);
        //}
        //printf("SUM IS: %.16f\n", sum);


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
        //_A_tpetra[0][0]->describe(*out, Teuchos::VERB_EXTREME);
        //_A_tpetra[1][0]->describe(*out, Teuchos::VERB_EXTREME);
        //_A_single_block->describe(*out, Teuchos::VERB_EXTREME);
        //solver->printTiming(*out, Teuchos::VERB_LOW);
        //_x_single_block->describe(*out, Teuchos::VERB_EXTREME);
        //_b_single_block->describe(*out, Teuchos::VERB_EXTREME);
        //_b_tpetra[0]->describe(*out, Teuchos::VERB_EXTREME);

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

void SolverT::residual() {

    // can only be called after solve() has been called, because solve() sets up the linear operators
    // solution on particles is imported back onto solution vector prior to this call, allowing
    // the exact solution to be put on the particles

    if (_consolidate_blocks) {
        // move x in blocks into single block, if needed
        for (size_t i=0; i<_A_tpetra[0].size(); ++i) {
            _x_single_block->doExport(*(_x_tpetra[i]), *(_domain_exporters[i][i]), Tpetra::INSERT);
        }
    }

    if (_parameters->get<std::string>("type")=="direct") {
        if (_consolidate_blocks) {
            _A_single_block->apply(*_x_single_block, *_b_single_block, Teuchos::NO_TRANS, 1.0, -1.0);
        } else {
            _A_tpetra[0][0]->apply(*_x_tpetra[0], *_b_tpetra[0], Teuchos::NO_TRANS, 1.0, -1.0);
        }
    } else {
        std::vector<Teko::MultiVector> x_teko;
        if (_consolidate_blocks) {
            x_teko.resize(1);
            Teuchos::RCP<const Thyra::TpetraVectorSpace<scalar_type,local_index_type,global_index_type,node_type> > domain 
                = Thyra::tpetraVectorSpace<scalar_type>(_A_single_block->getDomainMap());
            x_teko[0] = Thyra::tpetraVector(domain, Teuchos::rcp_static_cast<vec_scalar_type>(_x_single_block));
            TEUCHOS_TEST_FOR_EXCEPT_MSG(x_teko[0].is_null(), "Cast failed.");
        } else {
            x_teko.resize(_b_thyra.size()); 
            for (size_t i=0; i<_b_thyra.size(); i++) {
                Teuchos::RCP<const Thyra::TpetraVectorSpace<scalar_type,local_index_type,global_index_type,node_type> > domain 
                    = Thyra::tpetraVectorSpace<scalar_type>(_A_tpetra[i][i]->getDomainMap());
                x_teko[i] = Thyra::tpetraVector(domain, Teuchos::rcp_static_cast<vec_scalar_type>(_x_tpetra[i]));
                TEUCHOS_TEST_FOR_EXCEPT_MSG(x_teko[i].is_null(), "Cast failed.");
            }
        }

        // wrap with Thyra
        Teko::MultiVector Thyra_b = Teko::buildBlockedMultiVector(_b_thyra);
        Teko::MultiVector Thyra_x = Teko::buildBlockedMultiVector(x_teko);

        _A_thyra->apply(Thyra::NOTRANS, *Thyra_x, Thyra_b.ptr(), 1.0, -1.0);
    }

    if (_consolidate_blocks) {
        // move solution from _x_single_block to _x_tpetra
        for (size_t i=0; i<_x_tpetra.size(); i++) {
            _x_tpetra[i]->doImport(*_x_single_block, *(_domain_exporters[i][i]), Tpetra::REPLACE);
        }
    }

    if (_comm->getRank() == 0) std::cout << "WARNING: Over-writing solution variables with the residual." << std::endl;
}

Teuchos::RCP<mvec_type> SolverT::getSolution(local_index_type idx) const {

    Teuchos::RCP<mvec_type> return_ptr;
    return_ptr = _x_tpetra[idx];

    return return_ptr;

}

void SolverT::setSolution(local_index_type idx, Teuchos::RCP<mvec_type> vec) {

    auto _x_tpetra_view = _x_tpetra[idx]->getLocalView<host_view_scalar_type>();
    auto vec_view = vec->getLocalView<host_view_scalar_type>();

    TEUCHOS_TEST_FOR_EXCEPT_MSG(_x_tpetra_view.extent(0)!=vec_view.extent(0), "Number of rows differs between vectors.");
    TEUCHOS_TEST_FOR_EXCEPT_MSG(_x_tpetra_view.extent(1)!=vec_view.extent(1), "Number of cols differs between vectors.");

    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,vec_view.extent(0)), KOKKOS_LAMBDA(const int i) {
        for (int j=0; j<vec_view.extent(1); ++j) {
            _x_tpetra_view(i,j) = vec_view(i,j);
        } 
    });
}

}

#endif
