#include "Compadre_ProblemT.hpp"

#include "Compadre_CoordsT.hpp"
#include "Compadre_ParticlesT.hpp"
#include "Compadre_FieldT.hpp"
#include "Compadre_FieldManager.hpp"
#include "Compadre_DOFManager.hpp"
#include "Compadre_NeighborhoodT.hpp"
#include "Compadre_PhysicsT.hpp"
#include "Compadre_SourcesT.hpp"
#include "Compadre_BoundaryConditionsT.hpp"
#include "Compadre_ParameterManager.hpp"

#include "Compadre_SolverT.hpp"


//#include "Compadre_AnalyticFunctions.hpp"
#ifdef TRILINOS_LINEAR_SOLVES

namespace Compadre {



ProblemT::ProblemT(    Teuchos::RCP<ProblemT::particle_type> particles,
            Teuchos::RCP<Teuchos::ParameterList> parameters,
            Teuchos::RCP<ProblemT::physics_type> physics,
            Teuchos::RCP<ProblemT::source_type> sources,
            Teuchos::RCP<ProblemT::bc_type> bcs) :
            _particles(particles),
            _parameters(parameters),
            _coords(particles->getCoordsConst())
{
    _comm = particles->getCoordsConst()->getComm();
    this->setPhysics(physics);
    this->setSources(sources);
    this->setBCS(bcs);
    if (!_OP.is_null()) _OP->setParameters(*_parameters);
    if (!_RHS.is_null()) _RHS->setParameters(*_parameters);
    if (!_BCS.is_null()) _BCS->setParameters(*_parameters);
}


ProblemT::ProblemT(    Teuchos::RCP<ProblemT::particle_type> particles,
            Teuchos::RCP<ProblemT::parameter_manager_type> parameter_manager,
            Teuchos::RCP<ProblemT::physics_type> physics,
            Teuchos::RCP<ProblemT::source_type> sources,
            Teuchos::RCP<ProblemT::bc_type> bcs) :
                ProblemT(particles, parameter_manager->getList(), physics, sources, bcs)
{}


ProblemT::ProblemT(    Teuchos::RCP<ProblemT::particle_type> particles,
                    Teuchos::RCP<ProblemT::physics_type> physics,
                    Teuchos::RCP<ProblemT::source_type> sources,
                    Teuchos::RCP<ProblemT::bc_type> bcs) :
                        ProblemT(particles, particles->getParameters(), physics, sources, bcs)
{}

void ProblemT::setPhysics(Teuchos::RCP<ProblemT::physics_type> physics) {
    if (!physics.is_null()) {
        _OP = physics;
        this->registerFieldInteractions(_OP->gatherFieldInteractions());
    }
}
void ProblemT::setSources(Teuchos::RCP<ProblemT::source_type> sources) {
    if (!sources.is_null()) {
        _RHS = sources;
        this->registerFieldInteractions(_RHS->gatherFieldInteractions());
    }
}
void ProblemT::setBCS(Teuchos::RCP<ProblemT::bc_type> bcs) {
    if (!bcs.is_null()) {
        _BCS = bcs;
        this->registerFieldInteractions(_BCS->gatherFieldInteractions());
    }
}

void ProblemT::registerFieldInteractions(const std::vector<InteractingFields>& ops_to_register) {
    for (InteractingFields op : ops_to_register) {
        bool similar_field_interaction = false;
        for (InteractingFields op_already_in : _field_interactions) {
            if (op==op_already_in) {
                op_already_in += op; // add to the operations that need this interaction
                similar_field_interaction = true;
                break;
            }
        }
        if (!similar_field_interaction) { // this needs added to the set because it hasn't been registered yet
            _field_interactions.push_back(op);
        }
    }
}

Teuchos::RCP<const crs_matrix_type> ProblemT::getA(local_index_type row_block, local_index_type col_block) const {
    Teuchos::RCP<const crs_matrix_type> return_ptr = Teuchos::null;
    if (row_block==-1) return_ptr =  _A[0][0].getConst(); // default
    if (col_block==-1 && row_block>-1) return_ptr =  _A[row_block][row_block].getConst();
    else return_ptr =  _A[row_block][col_block].getConst();
    return return_ptr;
}

Teuchos::RCP<const mvec_type> ProblemT::getb(local_index_type row_block) const {
    if (row_block==-1) return _b[0].getConst();
    else return _b[row_block].getConst();
}

void ProblemT::initialize(scalar_type initial_simulation_time) {

    // do a check on whether any field in the interaction list
    // has type "Global" rather than "Banded", then it is not supported
    // for the nonblocked style (because nonblocked assumes all
    // field are of the "Banded" style)
    // Putting this in a single matrix implies having a row DOF that is
    // shared between multiple processors, which breaks Amesos2, Ifpack2, etc...
    for (InteractingFields interaction:_field_interactions) {
        auto src_sparsity_type = _particles->getFieldManager()->getFieldByID(interaction.src_fieldnum)->getFieldSparsityType();
        TEUCHOS_TEST_FOR_EXCEPT_MSG(_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==false && (src_sparsity_type == FieldSparsityType::Global), "Non-blocked matrix system incompatible with global FieldSparsityType.");
    }


    // first get the max number of blocks that will be used
    local_index_type max_blocks = 0;
    for (InteractingFields interaction:_field_interactions) {
        if (interaction.src_fieldnum > max_blocks) {
            max_blocks = interaction.src_fieldnum;
        } else if (interaction.trg_fieldnum > max_blocks) {
            max_blocks = interaction.trg_fieldnum;
        }
    }
    max_blocks = max_blocks + 1; // counting begins at 0

    // maps from field number to actual block of matrix
    _field_to_block_row_map = std::vector<local_index_type>(max_blocks, -1);
    _field_to_block_col_map = std::vector<local_index_type>(max_blocks, -1);

    local_index_type row_count = 0;
    local_index_type col_count = 0;
    for (local_index_type i=0; i<max_blocks; i++) {
        bool row_found = false;
        bool col_found = false;
        for (InteractingFields interaction:_field_interactions) {
            if (interaction.src_fieldnum == i) {
                _field_to_block_row_map[i] = row_count;
                row_found = true;
                row_count++;
            }
            if (interaction.trg_fieldnum == i) {
                _field_to_block_col_map[i] = col_count;
                col_found = true;
                col_count++;
            }
            if (row_found && col_found) break;
        }
    }

    // maps from block of matrix to field number
    _block_row_to_field_map = std::vector<local_index_type>(row_count);
    _block_col_to_field_map = std::vector<local_index_type>(col_count);
    row_count = 0;
    col_count = 0;
    for (local_index_type i=0; i<max_blocks; i++) {
        if (_field_to_block_row_map[i] > -1) {
            _block_row_to_field_map[row_count] = i;
            row_count++;
        }
        if (_field_to_block_col_map[i] > -1) {
            _block_col_to_field_map[col_count] = i;
            col_count++;
        }
    }

    // now that we know actual # of blocks we can build a DOF data with the DOF manager
    // regenerate DOF information, because we have a local ordering specific to our field interactions
    _problem_dof_data = _particles->getDOFManager()->generateDOFMap(_block_row_to_field_map);

    // now that we know actual # of blocks, we can set up the graphs, and row and column maps
    row_map.resize(row_count);
    col_map.resize(col_count);
    _b.resize(row_count);
    _A = std::vector<std::vector<Teuchos::RCP<crs_matrix_type> > > (row_count, std::vector<Teuchos::RCP<crs_matrix_type> > (col_count, Teuchos::null));
    A_graph = std::vector<std::vector<Teuchos::RCP<crs_graph_type> > > (row_count, std::vector<Teuchos::RCP<crs_graph_type> > (col_count, Teuchos::null));
    row_map = std::vector<Teuchos::RCP<const map_type> > (row_count, Teuchos::null);
    col_map = std::vector<Teuchos::RCP<const map_type> > (col_count, Teuchos::null);

    for (InteractingFields field_interaction : _field_interactions) {
        buildMaps(field_interaction.src_fieldnum, field_interaction.trg_fieldnum);
    }

    TEUCHOS_TEST_FOR_EXCEPT_MSG(_OP.is_null(), "Physics not set before initialize() called.");
    _OP->setParameters(*_parameters);
    _OP->setDOFData(_problem_dof_data);

    // generate all initial data needed for graphs and assembly
    _OP->initialize();

    // first is graph construction
    for (InteractingFields field_interaction : _field_interactions) {
        local_index_type field_one = field_interaction.src_fieldnum;
        local_index_type field_two = field_interaction.trg_fieldnum;

        local_index_type row_block = _field_to_block_row_map[field_one];
        local_index_type col_block = _field_to_block_col_map[field_two];

        // build the graph

        this->buildGraphs(field_one, field_two);

        // complete each block after graph computed
        A_graph[row_block][col_block]->fillComplete(_problem_dof_data->getRowMap(col_block), _problem_dof_data->getRowMap(row_block));
    }

    for (InteractingFields field_interaction : _field_interactions) {
        local_index_type field_one = field_interaction.src_fieldnum;
        local_index_type field_two = field_interaction.trg_fieldnum;

        local_index_type row_block = _field_to_block_row_map[field_one];
        local_index_type col_block = _field_to_block_col_map[field_two];

        assembleOperator(field_one, field_two);
        assembleBCS(field_one, field_two);
        assembleRHS(field_one, field_two);
        if (_A[row_block][col_block]->isFillActive()) {
            _A[row_block][col_block]->fillComplete(_problem_dof_data->getRowMap(col_block), _problem_dof_data->getRowMap(row_block));
        }
    }
}

void ProblemT::solve() {

    // build solver
    if (_solver.is_null()) buildSolver();

    // run solver
    _solver->solve();

//  // Check residual from exact solution
//    Compadre::GMLSLaplacianExactSolution fn;
//    _particles->getFieldByID(0)->localInitFromScalarFunction(&fn);
//
//    // still need to get residual
//    Teuchos::RCP<mvec_type> Ax = Teuchos::rcp(new mvec_type(*_solver->getSolution(), Teuchos::View)); // shallow copy, so it writes over the solution
//    // get dual view of an empty vector
//    // loop over resetting values
//    Teuchos::RCP<mvec_type> exact_solution = Teuchos::rcp(new mvec_type(*_solver->getSolution(), Teuchos::Copy)); // need a copy with the same structure
//    host_view_type vals = exact_solution->getLocalView<host_view_type>();
//    const Teuchos::RCP<Compadre::FieldT> field = _particles->getFieldByID(0);
//    Kokkos::parallel_for(vals.extent(0), KOKKOS_LAMBDA(const int i) {
//        vals(i,0) = field->getLocalScalarVal(i);
//    });
//
//    _A->apply(*exact_solution,*Ax);
//
//    Ax->update(-1.0, *b, 1.0);
//    if (_comm->getRank() == 0) std::cout << "WARNING: Over-writing solution variables with the exact solution residual." << std::endl;

    for (InteractingFields field_interaction : _field_interactions) {
        local_index_type field_one = field_interaction.src_fieldnum;

        local_index_type row_block = _field_to_block_row_map[field_one];

        _particles->getFieldManager()->updateFieldsFromVector(_solver->getSolution(row_block), field_one, _problem_dof_data);
    }
}

void ProblemT::residual() {

    TEUCHOS_TEST_FOR_EXCEPT_MSG(_solver.is_null(), "Can only be called after solve().");

    for (InteractingFields field_interaction : _field_interactions) {
        local_index_type field_one = field_interaction.src_fieldnum;

        local_index_type row_block = _field_to_block_row_map[field_one];

        _particles->getFieldManager()->updateVectorFromFields(_solver->getSolution(row_block), field_one, _problem_dof_data);
    }

    // generate residual, stored in solution
    _solver->residual();

    for (InteractingFields field_interaction : _field_interactions) {
        local_index_type field_one = field_interaction.src_fieldnum;

        local_index_type row_block = _field_to_block_row_map[field_one];

        _particles->getFieldManager()->updateFieldsFromVector(_solver->getSolution(row_block), field_one, _problem_dof_data);
    }
}

void ProblemT::buildMaps(local_index_type field_one, local_index_type field_two) {
    if (field_two<0) field_two = field_one;
    local_index_type row_block = _field_to_block_row_map[field_one];
    local_index_type col_block = _field_to_block_col_map[field_two];

    if (row_map[row_block].is_null()) {
        row_map[row_block] = _problem_dof_data->getRowMap(row_block);
    }
    if (col_map[col_block].is_null()) {
        col_map[col_block] = _problem_dof_data->getColMap(col_block);
    }
}

void ProblemT::buildGraphs(local_index_type field_one, local_index_type field_two) {
    if (field_two<0) field_two = field_one;
    local_index_type row_block = _field_to_block_row_map[field_one];
    local_index_type col_block = _field_to_block_col_map[field_two];

    // let the physics operator determine max num neighbors
    auto max_num_neighbors = _OP->getMaxNumNeighbors();

    if (row_map[row_block].is_null()) {
        buildMaps(field_one, field_two);
    }
    if (A_graph[row_block][col_block].is_null()) {
        local_index_type field_one_dim = _particles->getFieldManagerConst()->getFieldByID(field_one)->nDim();
        local_index_type field_two_dim = _particles->getFieldManagerConst()->getFieldByID(field_two)->nDim();
        A_graph[row_block][col_block] = Teuchos::rcp(new crs_graph_type (row_map[row_block], col_map[col_block], max_num_neighbors*field_one_dim*field_two_dim, Tpetra::StaticProfile));
    }
    // _OP sets the initial graph and may change it
    _OP->setGraph(A_graph[row_block][col_block]);
    _OP->setRowMap(row_map[row_block]);
    _OP->setColMap(col_map[col_block]);
    // the resulting, potentially changed graph, is now returned
    A_graph[row_block][col_block] = _OP->computeGraph(field_one, field_two);
}

void ProblemT::buildSolver() {
    // build solver
    TEUCHOS_TEST_FOR_EXCEPT_MSG(_parameters.is_null(), "Parameters not set before buildSolver() called.");
    _solver = Teuchos::rcp_static_cast<solver_type>(Teuchos::rcp(new Compadre::SolverT(_comm, _A, _b)));
    try {
        _solver->setParameters(_parameters->get<Teuchos::ParameterList>("solver"));
    } catch (Teuchos::Exceptions::InvalidParameter & exception) {
        TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "ParameterList: \"solver\" not found. This should have been set in the input file.");
    }
}

void ProblemT::assembleOperator(local_index_type field_one, local_index_type field_two, scalar_type simulation_time, scalar_type delta_time) {
    if (field_two<0) field_two = field_one;
    local_index_type row_block = _field_to_block_row_map[field_one];
    local_index_type col_block = _field_to_block_col_map[field_two];

    TEUCHOS_TEST_FOR_EXCEPT_MSG(_OP.is_null(), "Physics not set before assembleOperator() called.");
    _OP->setParameters(*_parameters);

    if (_A[row_block][col_block].is_null()) {
        _A[row_block][col_block] = Teuchos::rcp(new crs_matrix_type (A_graph[row_block][col_block]));
//            local_index_type field_one_dim = _particles->getFieldManagerConst()->getFieldByID(field_one)->nDim();
//            local_index_type field_two_dim = _particles->getFieldManagerConst()->getFieldByID(field_two)->nDim();
//            _A[row_block][col_block] = Teuchos::rcp(new crs_matrix_type (row_map[row_block], col_map[col_block], _particles->getNeighborhood()->getMaxNumNeighbors()*field_one_dim*field_two_dim));
    }
    _OP->setMatrix(_A[row_block][col_block]);
    _OP->computeMatrix(field_one, field_two);
}

void ProblemT::assembleRHS(local_index_type field_one, local_index_type field_two, scalar_type simulation_time, scalar_type delta_time) {
    if (field_two<0) field_two = field_one;
    local_index_type row_block = _field_to_block_row_map[field_one];

    TEUCHOS_TEST_FOR_EXCEPT_MSG(_RHS.is_null(), "Sources not set before assembleRHS() called.");

    _RHS->setParameters(*_parameters);

    if (_b[row_block].is_null()) {
        bool setToZero = true;
        _b[row_block] = Teuchos::rcp(new mvec_type(row_map[row_block], 1, setToZero));
    }
    _RHS->setMultiVector(_b[row_block].getRawPtr());
    _RHS->setDOFData(_problem_dof_data);
    _RHS->evaluateRHS(field_one, field_two, 0.0);
}

void ProblemT::assembleBCS(local_index_type field_one, local_index_type field_two, scalar_type simulation_time, scalar_type delta_time) {
    if (field_two<0) field_two = field_one;
    local_index_type row_block = _field_to_block_row_map[field_one];

    TEUCHOS_TEST_FOR_EXCEPT_MSG(_BCS.is_null(), "Boundary conditions not set before assembleBCS() called.");

    _BCS->setParameters(*_parameters);

    // fill out this row block of b
    if (_b[row_block].is_null()) {
        bool setToZero = true;
        _b[row_block] = Teuchos::rcp(new mvec_type(row_map[row_block], 1, setToZero));
    }
    _BCS->setMultiVector(_b[row_block].getRawPtr());
    _BCS->setDOFData(_problem_dof_data);
    _BCS->flagBoundaries();
    _BCS->applyBoundaries(field_one, field_two, 0.0);

}

}
#endif
