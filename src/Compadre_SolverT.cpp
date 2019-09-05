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


SolverT::SolverT( const Teuchos::RCP<const Teuchos::Comm<int> > comm, std::vector<std::vector<Teuchos::RCP<crs_matrix_type> > > A, std::vector<Teuchos::RCP<mvec_type> > b) :
        _comm(comm)
{
    _A_tpetra = std::vector<std::vector<Teuchos::RCP<crs_matrix_type> > > (A.size(), std::vector<Teuchos::RCP<crs_matrix_type> >(A[0].size(), Teuchos::null));
    _A = Thyra::defaultBlockedLinearOp<scalar_type>();
    _A->beginBlockFill(A.size(), A[0].size());
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

                _A->setBlock(i, j, tpetra_operator);
                _A_tpetra[i][j] = A[i][j];
            }
        }
    }
    _A->endBlockFill();

    _b.resize(b.size());
    _b_tpetra.resize(b.size());
    for (size_t i=0; i<b.size(); i++) {
        Teuchos::RCP<const Thyra::TpetraVectorSpace<scalar_type,local_index_type,global_index_type,node_type> > range = Thyra::tpetraVectorSpace<scalar_type>(_A_tpetra[i][i]->getRangeMap());
        _b[i] = Thyra::tpetraVector(range, Teuchos::rcp_static_cast<vec_scalar_type>(b[i]));
        _b_tpetra[i] = b[i];
    }
}

void SolverT::solve() {
    TEUCHOS_TEST_FOR_EXCEPT_MSG( _parameter_filename.empty() && _parameters->get<std::string>("type")!="direct", "Parameters not set before solve() called." );

    // ASSUMES SQUARE (col and row #blocks have same cardinality)
    _x.resize(_b.size());
    std::vector<Teko::MultiVector> x(_b.size()); // if this has to be size 1 for nonblocked, then it must be for b also

    const bool setToZero = true;
    if (_parameters->get<bool>("blocked")) {
        for (size_t i=0; i<_b.size(); i++) {
            _x[i] = Teuchos::rcp(new mvec_type(_A_tpetra[i][i]->getDomainMap(),1,setToZero));
            TEUCHOS_TEST_FOR_EXCEPT_MSG(_A_tpetra[i][i]->getDomainMap()==Teuchos::null, "Domain map null.");

            Teuchos::RCP<const Thyra::TpetraVectorSpace<scalar_type,local_index_type,global_index_type,node_type> > domain = Thyra::tpetraVectorSpace<scalar_type>(_A_tpetra[i][i]->getDomainMap());
            x[i] = Thyra::tpetraVector(domain, Teuchos::rcp_static_cast<vec_scalar_type>(_x[i]));

            TEUCHOS_TEST_FOR_EXCEPT_MSG(x[i].is_null(), "Cast failed.");
        }
    } else {
        _x[0] = Teuchos::rcp(new mvec_type(_A_tpetra[0][0]->getDomainMap(),1,setToZero));
        Teuchos::RCP<const Thyra::TpetraVectorSpace<scalar_type,local_index_type,global_index_type,node_type> > domain = Thyra::tpetraVectorSpace<scalar_type>(_A_tpetra[0][0]->getDomainMap());
        x[0] = Thyra::tpetraVector(domain, Teuchos::rcp_static_cast<vec_scalar_type>(_x[0]));

        TEUCHOS_TEST_FOR_EXCEPT_MSG(x[0].is_null(), "Cast failed.");
    }

    Teko::MultiVector Thyra_b = Teko::buildBlockedMultiVector(_b);
    Teko::MultiVector Thyra_x = Teko::buildBlockedMultiVector(x);

    // Diagnostics:
   // auto out = Teuchos::getFancyOStream(Teuchos::rcpFromRef(std::cout));
   // Thyra_x->describe(*out, Teuchos::VERB_EXTREME);
   // Thyra_b->describe(*out, Teuchos::VERB_EXTREME);
   // _A->describe(*out, Teuchos::VERB_EXTREME);

    Thyra::SolveStatus<scalar_type> status;


    if (_parameters->get<std::string>("type")=="direct") {
        TEUCHOS_TEST_FOR_EXCEPT_MSG(_parameters->get<bool>("blocked"), "Blocked matrix used is incompatible with direct solves.");
        // Ensure that you do not configure for a direct solve and then call an iterative solver

        // Constructor from Factory
        Teuchos::RCP<Amesos2::Solver<crs_matrix_type,mvec_type> > solver;
        try{
            solver = Amesos2::create<crs_matrix_type,mvec_type>("KLU", _A_tpetra[0][0], _x[0], _b_tpetra[0]);
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
            = Thyra::linearOpWithSolve(*lowsFactory, Teuchos::rcp_dynamic_cast<const thyra_linearop_type>(_A));

        Thyra::assign(Thyra_x.ptr(), 0.0);
        status
            = Thyra::solve<scalar_type>(*th_invA, Thyra::NOTRANS, *Thyra_b, Thyra_x.ptr());

        *out << "\nSolve status:\n" << status;

//        // DIAGNOSTIC for multiply b by A and storing in x
//        Thyra::apply<scalar_type>(*Teuchos::rcp_dynamic_cast<const thyra_linearop_type>(_A), Thyra::NOTRANS, *Thyra_b, Thyra_x.ptr());

    }

//  TODO: fix this up for blocked and unblocked case
//
//    if (_parameters->get<bool>("save residual over solution")) {
//        // writes residual over top of the solution variable
//        Teuchos::RCP<mvec_type> Ax = Teuchos::rcp(new mvec_type(*x, Teuchos::View)); // shallow copy, so it writes over the solution
//        _A->apply(*x,*Ax);
//        Ax->update(-1.0, *_b, 1.0);
//        if (_comm->getRank() == 0) std::cout << "WARNING: Over-writing solution variables with the residual." << std::endl;
//    }
}

Teuchos::RCP<mvec_type> SolverT::getSolution(local_index_type idx) const {

    Teuchos::RCP<mvec_type> return_ptr;

    if (_parameters->get<bool>("blocked")) {
        return_ptr = _x[idx];
    } else {
        return_ptr = _x[0];
    }

    return return_ptr;

}

}

#endif
