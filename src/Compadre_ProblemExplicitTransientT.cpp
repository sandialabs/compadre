#include "Compadre_ProblemExplicitTransientT.hpp"

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
#include "Compadre_FileIO.hpp"

#include "Compadre_SolverT.hpp"

#include <Compadre_AnalyticFunctions.hpp>
#include <Compadre_XyzVector.hpp>
#include <Compadre_GMLS.hpp> // for getNP()

namespace Compadre {



ProblemExplicitTransientT::ProblemExplicitTransientT(    Teuchos::RCP<ProblemExplicitTransientT::particle_type> particles,
            Teuchos::RCP<Teuchos::ParameterList> parameters,
            Teuchos::RCP<ProblemExplicitTransientT::physics_type> physics,
            Teuchos::RCP<ProblemExplicitTransientT::source_type> sources,
            Teuchos::RCP<ProblemExplicitTransientT::bc_type> bcs) :
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


ProblemExplicitTransientT::ProblemExplicitTransientT(    Teuchos::RCP<ProblemExplicitTransientT::particle_type> particles,
            Teuchos::RCP<ProblemExplicitTransientT::parameter_manager_type> parameter_manager,
            Teuchos::RCP<ProblemExplicitTransientT::physics_type> physics,
            Teuchos::RCP<ProblemExplicitTransientT::source_type> sources,
            Teuchos::RCP<ProblemExplicitTransientT::bc_type> bcs) :
                ProblemExplicitTransientT(particles, parameter_manager->getList(), physics, sources, bcs)
{}


ProblemExplicitTransientT::ProblemExplicitTransientT(    Teuchos::RCP<ProblemExplicitTransientT::particle_type> particles,
                    Teuchos::RCP<ProblemExplicitTransientT::physics_type> physics,
                    Teuchos::RCP<ProblemExplicitTransientT::source_type> sources,
                    Teuchos::RCP<ProblemExplicitTransientT::bc_type> bcs) :
                        ProblemExplicitTransientT(particles, particles->getParameters(), physics, sources, bcs)
{}

void ProblemExplicitTransientT::setPhysics(Teuchos::RCP<ProblemExplicitTransientT::physics_type> physics) {
    if (!physics.is_null()) {
        _OP = physics;
        this->registerFieldInteractions(_OP->gatherFieldInteractions());
    }
}
void ProblemExplicitTransientT::setSources(Teuchos::RCP<ProblemExplicitTransientT::source_type> sources) {
    if (!sources.is_null()) {
        _RHS = sources;
        this->registerFieldInteractions(_RHS->gatherFieldInteractions());
    }
}
void ProblemExplicitTransientT::setBCS(Teuchos::RCP<ProblemExplicitTransientT::bc_type> bcs) {
    if (!bcs.is_null()) {
        _BCS = bcs;
        this->registerFieldInteractions(_BCS->gatherFieldInteractions());
    }
}

void ProblemExplicitTransientT::registerFieldInteractions(const std::vector<InteractingFields>& ops_to_register) {
    for (InteractingFields fi : ops_to_register) {
        bool similar_field_interaction = false;
        for (InteractingFields& op_already_in : _field_interactions) {
            if (fi==op_already_in) {
                op_already_in += fi; // add to the operations that need this interaction
                similar_field_interaction = true;
                break;
            }
        }
        if (!similar_field_interaction) { // this needs added to the set because it hasn't been registered yet
            _field_interactions.push_back(fi);
        }
    }
    for (InteractingFields fi : _field_interactions) {
        // sort ops_needing list for each field interaction
        std::sort(fi.ops_needing.begin(), fi.ops_needing.end());
    }
    //for (InteractingFields& fi : _field_interactions) {
    //    std::sort(fi.ops_needing.begin(), fi.ops_needing.end());
    //    printf("fieldnum: (%d,%d), ops_needing size: %d\n", fi.src_fieldnum, fi.trg_fieldnum, fi.ops_needing.size());
    //    for (auto op : fi.ops_needing) {
    //        printf("fieldnum: (%d,%d), ops_needing: %d\n", fi.src_fieldnum, fi.trg_fieldnum, op);
    //    }
    //}
}

Teuchos::RCP<const crs_matrix_type> ProblemExplicitTransientT::getA(local_index_type row_block, local_index_type col_block) const {
    Teuchos::RCP<const crs_matrix_type> return_ptr = Teuchos::null;
//    if (row_block==-1) return_ptr =  _A[0][0].getConst(); // default
//    if (col_block==-1 && row_block>-1) return_ptr =  _A[row_block][row_block].getConst();
//    else return_ptr =  _A[row_block][col_block].getConst();
    return return_ptr;
}

Teuchos::RCP<const mvec_type> ProblemExplicitTransientT::getb(local_index_type row_block) const {
    if (row_block==-1) return _b[0].getConst();
    else return _b[row_block].getConst();
}

void ProblemExplicitTransientT::initialize() {
    // TODO: break function into an initialize and a reinitialize function that is selective

    // first get the max number of blocks that will be used
    local_index_type max_blocks = 0;
    for (InteractingFields interaction:_field_interactions) {
        if (interaction.src_fieldnum > max_blocks) {
            max_blocks = interaction.src_fieldnum;
        } 
        if (interaction.trg_fieldnum > max_blocks) {
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
    //_A = std::vector<std::vector<Teuchos::RCP<crs_matrix_type> > > (row_count, std::vector<Teuchos::RCP<crs_matrix_type> > (col_count, Teuchos::null));
    //A_graph = std::vector<std::vector<Teuchos::RCP<crs_graph_type> > > (row_count, std::vector<Teuchos::RCP<crs_graph_type> > (col_count, Teuchos::null));
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

    //for (InteractingFields field_interaction : _field_interactions) {
    //    local_index_type field_one = field_interaction.src_fieldnum;
    //    local_index_type field_two = field_interaction.trg_fieldnum;

    //    local_index_type row_block = _field_to_block_row_map[field_one];
    //    local_index_type col_block = _field_to_block_col_map[field_two];

    //    assembleOperator(field_one, field_two);
    //    assembleBCS(field_one, field_two);
    //    assembleRHS(field_one, field_two);
    //    if (_A[row_block][col_block]->isFillActive()) {
    //        _A[row_block][col_block]->fillComplete(_problem_dof_data->getRowMap(col_block), _problem_dof_data->getRowMap(row_block));
    //    }
    //}

//    // first is graph construction
//    for (InteractingFields field_interaction : _field_interactions) {
//        local_index_type field_one = field_interaction.src_fieldnum;
//        local_index_type field_two = field_interaction.trg_fieldnum;
//
//        local_index_type row_block = (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) ? _field_to_block_row_map[field_one] : 0;
//        local_index_type col_block = (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) ? _field_to_block_col_map[field_two] : 0;
//
//        // build the graph
//
//        this->buildGraphs(field_one, field_two);
//
//        // complete each block after graph computed
//        if (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) {
//            A_graph[row_block][col_block]->fillComplete();
//        }
//    }
//    if (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==false) {
//        A_graph[0][0]->fillComplete();
//    }
//
//
//    for (InteractingFields field_interaction : _field_interactions) {
//        local_index_type field_one = field_interaction.src_fieldnum;
//        local_index_type field_two = field_interaction.trg_fieldnum;
//
//        local_index_type row_block = (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) ? _field_to_block_row_map[field_one] : 0;
//        local_index_type col_block = (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) ? _field_to_block_col_map[field_two] : 0;
//
//        assembleBCS(field_one, field_two);
//        assembleRHS(field_one, field_two);
//        assembleOperatorToVector(field_one, field_two);
////        if (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) {
////            _A[row_block][col_block]->fillComplete();
////        }
//    }
//    if (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==false) {
//        _A[0][0]->fillComplete();
//    }

}

void ProblemExplicitTransientT::solve(Teuchos::ParameterList& parameters_time, scalar_type t_0, scalar_type t_end, scalar_type dt, std::string filename) {

    int order = parameters_time.get<int>("order");

    // details for writing to files every nth timestep
    Compadre::FileManager fm;
    std::string base, extension;

    if (_parameters->get<Teuchos::ParameterList>("io").get<int>("write every") >= 0) {
        // parse by removing extension of filename

        // get extension
        size_t pos = filename.rfind('.', filename.length());

        // verify there was an extension extension
        TEUCHOS_TEST_FOR_EXCEPT_MSG(pos <= 0, "Filename specified to solve(...) is missing an extension.");

        base = filename.substr(0, pos);
        extension = filename.substr(pos+1, filename.length() - pos);
    }

    auto stepper_type = parameters_time.get<std::string>("type");
    std::transform(stepper_type.begin(), stepper_type.end(), stepper_type.begin(), ::tolower);
    if (stepper_type == "rk") {
        this->solveRK(parameters_time, t_0, t_end, dt, filename);
    } else if (stepper_type == "newmark") {
        this->solveNewmark(parameters_time, t_0, t_end, dt, filename);
    } else {
        TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Filename specified to solve(...) is missing an extension.");
    }
}

void ProblemExplicitTransientT::solveRK(Teuchos::ParameterList& parameters_time, scalar_type t_0, scalar_type t_end, scalar_type dt, std::string filename) {

//    Compadre::ShallowWaterTestCases sw_function = Compadre::ShallowWaterTestCases(_parameters->get<int>("shallow water test case"), 0 /*alpha*/);
    int rk_order = parameters_time.get<int>("order");

    // details for writing to files every nth timestep
    Compadre::FileManager fm;
    std::string base, extension;

    if (_parameters->get<Teuchos::ParameterList>("io").get<int>("write every") >= 0) {
        // parse by removing extension of filename

        // get extension
        size_t pos = filename.rfind('.', filename.length());

        // verify there was an extension extension
        TEUCHOS_TEST_FOR_EXCEPT_MSG(pos <= 0, "Filename specified to solve(...) is missing an extension.");

        base = filename.substr(0, pos);
        extension = filename.substr(pos+1, filename.length() - pos);
    }

    // dt given is a stable timestep, so any timestep adjustment must be equal to or smaller than this

    Teuchos::RCP<Teuchos::Time> RKSetupTime = Teuchos::TimeMonitor::getNewCounter ("RK Setup Time");
    Teuchos::RCP<Teuchos::Time> RKAdvanceTime = Teuchos::TimeMonitor::getNewCounter ("RK Advance Time");
    RKSetupTime->start();
    // integrates using RK method

    std::vector<scalar_type> c(rk_order);
    std::vector<scalar_type> b(rk_order);
    std::vector<std::vector<scalar_type> > a(rk_order, std::vector<scalar_type>(rk_order));

    if (rk_order == 1) {
        c[0] = 0;
        b[0] = 1.;
    } else if (rk_order == 2) {
        c[0] = 0; c[1] = 0.5;
        b[0] = 0; b[1] = 1.;
        a[1][0] = 0.5;
    } else if (rk_order == 3) {
        c[0] = 0; c[1] = 1./3.; c[2] = 2./3.;
        b[0] = 0.25; b[1] = 0; b[2] = 0.75;
        a[1][0] = 1./3.;
        a[2][1] = 2./3.;
    } else if (rk_order == 4) {
        c[0] = 0; c[1] = 0.5; c[2] = 0.5; c[3] = 1.0;
        b[0] = 1./6.; b[1] = 1./3.; b[2] = 1./3.; b[3] = 1./6.;
        a[1][0] = 0.5;
        a[2][1] = 0.5;
        a[3][2] = 1.0;
    }

//    local_index_type num_time_steps = std::ceil((t_end-t_0)/dt);
//    dt = (t_end-t_0)/num_time_steps;
    if (_particles->getCoordsConst()->getComm()->getRank()==0)
        printf("solving beginning at t=%f and ending at te=%f with dt=%.16f\n", t_0, t_end, dt);
//    printf(" with num time steps = %d\n", num_time_steps);

    std::vector<Teuchos::RCP<mvec_type> > reference_solution(_particles->getFieldManagerConst()->getVectorOfFields().size(), Teuchos::null);
    std::vector<Teuchos::RCP<mvec_type> > updated_solution(_particles->getFieldManagerConst()->getVectorOfFields().size(), Teuchos::null);
    std::vector<Teuchos::RCP<mvec_type> > rk_offsets(_particles->getFieldManagerConst()->getVectorOfFields().size(), Teuchos::null);


    for (InteractingFields field_interaction : _field_interactions) {

        local_index_type field_one = field_interaction.src_fieldnum;

        local_index_type row_block = (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) ? _field_to_block_row_map[field_one] : 0;

        // this allows us to overwrite the fields in _particles for each RK step and use
        // existing tools to distribute halo data
        // get all the vectors set up
        if (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) {
            if (reference_solution[row_block].is_null()) {
                bool setToZero = true;
                reference_solution[row_block] = Teuchos::rcp(new mvec_type(row_map[row_block], 1, setToZero));
                updated_solution[row_block] = Teuchos::rcp(new mvec_type(row_map[row_block], 1, setToZero));
                rk_offsets[row_block] = Teuchos::rcp(new mvec_type(row_map[row_block], 1, setToZero));
                // get solution from _particles and put it in reference_solution
                _particles->getFieldManagerConst()->updateVectorFromFields(reference_solution[row_block], field_one, _problem_dof_data);

                // copy this to updated_solution so that it starts in the right place
                host_view_type updated_data = updated_solution[row_block]->getLocalView<host_view_type>();
                host_view_type reference_data = reference_solution[row_block]->getLocalView<host_view_type>();
                host_view_type rk_offsets_data = rk_offsets[row_block]->getLocalView<host_view_type>();
                for (size_t j=0; j<updated_data.extent(0); j++) {
                    updated_data(j,0) = reference_data(j,0);
                    rk_offsets_data(j,0) = reference_data(j,0);
                }
            }
        } else {
            bool setToZero = true;
            reference_solution[0] = Teuchos::rcp(new mvec_type(row_map[0], 1, setToZero));
            updated_solution[0] = Teuchos::rcp(new mvec_type(row_map[0], 1, setToZero));
            rk_offsets[0] = Teuchos::rcp(new mvec_type(row_map[0], 1, setToZero));
            // get solution from _particles and put it in reference_solution
            _particles->getFieldManagerConst()->updateVectorFromFields(reference_solution[0], -1, _problem_dof_data);

            // copy this to updated_solution so that it starts in the right place
            host_view_type updated_data = updated_solution[0]->getLocalView<host_view_type>();
            host_view_type reference_data = reference_solution[0]->getLocalView<host_view_type>();
            host_view_type rk_offsets_data = rk_offsets[0]->getLocalView<host_view_type>();
            for (size_t j=0; j<updated_data.extent(0); j++) {
                updated_data(j,0) = reference_data(j,0);
                rk_offsets_data(j,0) = reference_data(j,0);
            }
            break;
        }
    }

    RKSetupTime->stop();

    // reference_solution now exists in multiple mvec_types
    // we add RK offsets to this and then move onto _particles fields, which propagates
    // the offset data onto halos, making it available on other processors
    scalar_type t = t_0;

    local_index_type timestep_count = 0; // keeps track of how many timesteps computed
    local_index_type write_count = 0; // keeps track of how many timesteps written

    while ( t < t_end && std::abs(t-t_end) >1e-15 ) {
        scalar_type internal_dt = std::min(dt, t_end-t);
//        dt = (dt > (t_end-t)) ? (t_end-t) : dt;

        for (local_index_type j=0; j<rk_order; ++j) {

//            // TODO: temporary
//            for (InteractingFields field_interaction : _field_interactions) {
//                local_index_type field_one = field_interaction.src_fieldnum;
//                local_index_type field_two = field_interaction.trg_fieldnum;
//
//                local_index_type row_block = (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) ? _field_to_block_row_map[field_one] : 0;
//                local_index_type col_block = (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) ? _field_to_block_col_map[field_two] : 0;
//
//                // update _particles with reference data plus RK addition
//                if (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) {
//
//                    // block reference solution plus offset
//                    host_view_type rk_offsets_data = rk_offsets[row_block]->getLocalView<host_view_type>();
////                    host_view_type reference_data = reference_solution[row_block]->getLocalView<host_view_type>();
////                    host_view_type updated_data = updated_solution[row_block]->getLocalView<host_view_type>();
////                    host_view_type step_data = _b[row_block]->getLocalView<host_view_type>();
//
//                    // beginning of first stage
//                    if (field_one == _particles->getFieldManager()->getIDOfFieldFromName("velocity")) {
//
//                        // TODO remove this
//                        // test by replacing solution with analytical at each updated point
//                        _particles->getFieldManager()->getFieldByName("velocity")->
//                                localInitFromVectorFunction(&sw_function);
//                        const mvec_type* velocity_ptr = _particles->getFieldManager()->getFieldByName("velocity")->
//                                getMultiVectorPtrConst();
//                        host_view_type velocity_data = velocity_ptr->getLocalView<host_view_type>();
//
//                        const std::vector<std::vector<std::vector<local_index_type> > >& local_to_dof_map =
//                                this->_particles->getDOFManagerConst()->getDOFMap();
//
//                        for (local_index_type k=0; k<_particles->getCoordsConst()->nLocal(); ++k) {
//                            for (local_index_type l=0; l<3; ++l) {
//                                local_index_type vdof = local_to_dof_map[k][field_one][l];
////                                reference_data(vdof,0) = velocity_data(k,l);
//                                rk_offsets_data(vdof,0) = velocity_data(k,l);
////                                updated_data(vdof,0) = velocity_data(k,l);
//                            }
//                        }
//                    }
//                }
//            }

            //if (_particles->getCoordsConst()->isLagrangian()) {
            //    // TODO remove this
            //    // test by replacing solution with analytical at each updated point
            //    _particles->getFieldManager()->getFieldByName("velocity")->
            //            localInitFromVectorFunction(&sw_function);

//            //    // TODO: here is where we advance the coordinates based on field values for displacement
//
//            //    _particles->getFieldManager()->getFieldByName("reference velocity")->
//            //            localInitFromVectorFunction(&sw_function);
//            //    _particles->getCoords()->
////            //            deltaUpdatePhysicalCoordsFromVectorByTimestep(c[j]*internal_dt,
//            //                    deltaUpdatePhysicalCoordsFromVectorByTimestep(internal_dt,
//            //                    _particles->getFieldManagerConst()->getFieldByName("velocity")->getMultiVectorPtrConst());


            //    _particles->getCoords()->updatePhysicalCoordinatesFromFieldData(_particles->getFieldManager()->getFieldByName("velocity")->getMultiVectorPtrConst(), c[j]*internal_dt);

            //}

            // build the rk_offsets vector and put it in _particles
            for (InteractingFields field_interaction : _field_interactions) {
                local_index_type field_one = field_interaction.src_fieldnum;

                local_index_type row_block = (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) ? _field_to_block_row_map[field_one] : 0;

                // update _particles with reference data plus RK addition
                if (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) {

                    _particles->getFieldManager()->updateFieldsFromVector(rk_offsets[row_block], field_one, _problem_dof_data);

                } else {

                    _particles->getFieldManager()->updateFieldsFromVector(rk_offsets[0], field_one, _problem_dof_data);

                }
            }

            // refresh fields' halo data
            _particles->getFieldManager()->updateFieldsHaloData();


            if (_particles->getCoordsConst()->isLagrangian()) {
//
                _particles->updatePhysicalCoordinatesFromField("displacement");
//
            }

            RKAdvanceTime->start();

            //// print ordering of field interactions with their operations
            //for (InteractingFields& fi : _field_interactions) {
            //    printf("fieldnum: %d, ops_needing size: %d\n", fi.src_fieldnum, fi.ops_needing.size());
            //    for (auto op : fi.ops_needing) {
            //        printf("fieldnum: %d, ops_needing: %d\n", fi.src_fieldnum, op);
            //    }
            //}
            
            // continuation of finding this stages answer
            // a different for loop than the last section because all rk_offsets for all blocks
            // must be calculated before computing any field's contributions
            for (InteractingFields field_interaction : _field_interactions) {

                local_index_type field_one = field_interaction.src_fieldnum;
                local_index_type field_two = field_interaction.trg_fieldnum;

                for (op_needing_interaction op : field_interaction.ops_needing) {
                    // (all store into _b), so _b holds stage solution
                    if (op == op_needing_interaction::bc) {
                        assembleBCS(field_one, field_two, t + c[j]*internal_dt, internal_dt);
                    }
                    else if (op == op_needing_interaction::source) {
                        assembleRHS(field_one, field_two, t + c[j]*internal_dt, internal_dt);
                    }
                    else if (op == op_needing_interaction::physics) {
                        assembleOperatorToVector(field_one, field_two, t + c[j]*internal_dt, internal_dt);
                    }
                }
            }

            if (_particles->getCoordsConst()->isLagrangian()) {
////                // undoing moving the particles
////                _particles->getCoords()->
//////                        deltaUpdatePhysicalCoordsFromVectorByTimestep(-c[j]*internal_dt,
////                                deltaUpdatePhysicalCoordsFromVectorByTimestep(-internal_dt,
////                                            _particles->getFieldManagerConst()->getFieldByName("velocity")->getMultiVectorPtrConst());
//
                _particles->updatePhysicalCoordinatesFromField("displacement", -1);
//
////                _particles->getCoords()->updatePhysicalCoordinatesFromFieldData(_particles->getFieldManager()->getFieldByName("reference velocity")->getMultiVectorPtrConst(), -c[j]*internal_dt);
            }
//
//            if (_particles->getCoordsConst()->isLagrangian()) {
//                // TODO remove this
//                // test by replacing solution with analytical at each updated point
//                _particles->getFieldManager()->getFieldByName("velocity")->
//                        localInitFromVectorFunction(&sw_function);
//
////                // TODO: here is where we advance the coordinates based on field values for displacement
////
////                _particles->getFieldManager()->getFieldByName("reference velocity")->
////                        localInitFromVectorFunction(&sw_function);
////                _particles->getCoords()->
//////                        deltaUpdatePhysicalCoordsFromVectorByTimestep(c[j]*internal_dt,
////                                deltaUpdatePhysicalCoordsFromVectorByTimestep(internal_dt,
////                                _particles->getFieldManagerConst()->getFieldByName("velocity")->getMultiVectorPtrConst());
//
//
//                _particles->getCoords()->updatePhysicalCoordinatesFromFieldData(_particles->getFieldManager()->getFieldByName("velocity")->getMultiVectorPtrConst(), -c[j]*internal_dt);
//
//            }

            // Use this stages solution to update the update solution and also to update the rk_offsets vector
            // step contribution k_i is in _b
            for (InteractingFields field_interaction : _field_interactions) {

                local_index_type field_one = field_interaction.src_fieldnum;

                local_index_type row_block = (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) ? _field_to_block_row_map[field_one] : 0;

                // update _particles with reference data plus RK addition
                if (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) {

                    // block reference solution plus offset
                    host_view_type rk_offsets_data = rk_offsets[row_block]->getLocalView<host_view_type>();
                    host_view_type reference_data = reference_solution[row_block]->getLocalView<host_view_type>();
                    host_view_type updated_data = updated_solution[row_block]->getLocalView<host_view_type>();
                    host_view_type step_data = _b[row_block]->getLocalView<host_view_type>();

                    // deal with Dirichlet boundary conditions
	                host_view_local_index_type bc_id = _particles->getFlags()->getLocalView<host_view_local_index_type>();
                    for (size_t k=0; k<rk_offsets_data.extent(0); ++k) {
                        if (bc_id(k,0)==0) {
                            rk_offsets_data(k,0) = reference_data(k,0);
                            if (j<(rk_order-1)) {
                                rk_offsets_data(k,0) += internal_dt * a[j+1][j] * step_data(k,0); // previous steps contribution
                            }
                            updated_data(k,0) += internal_dt * b[j] * step_data(k,0);
                        } else {
                            updated_data(k,0) = step_data(k,0);
                        }
                        step_data(k,0) = 0;
                    }
                } else {
                    // reference solution plus offset
                    host_view_type rk_offsets_data = rk_offsets[0]->getLocalView<host_view_type>();
                    host_view_type reference_data = reference_solution[0]->getLocalView<host_view_type>();
                    host_view_type updated_data = updated_solution[0]->getLocalView<host_view_type>();
                    host_view_type step_data = _b[0]->getLocalView<host_view_type>();

                    for (size_t k=0; k<rk_offsets_data.extent(0); ++k) {
                        rk_offsets_data(k,0) = reference_data(k,0);
                        if (j<(rk_order-1))
                            rk_offsets_data(k,0) += internal_dt * a[j+1][j] * step_data(k,0); // previous steps contribution

                        updated_data(k,0) += internal_dt * b[j] * step_data(k,0);
                        step_data(k,0) = 0;
                    }
                    break;
                }
            }
            RKAdvanceTime->stop();
        }

        // update the reference solution for beginning the next time step
        for (InteractingFields field_interaction : _field_interactions) {

            local_index_type field_one = field_interaction.src_fieldnum;

            local_index_type row_block = (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) ? _field_to_block_row_map[field_one] : 0;

//            if (_particles->getCoordsConst()->getComm()->getRank()==0)
//                printf("field one: %d at timestep %d\n", field_one, timestep_count);
            if (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) {
                // copy this to updated_solution so that it starts in the right place
                host_view_type updated_data = updated_solution[row_block]->getLocalView<host_view_type>();
                host_view_type reference_data = reference_solution[row_block]->getLocalView<host_view_type>();
                host_view_type rk_offsets_data = rk_offsets[row_block]->getLocalView<host_view_type>();
                for (size_t k=0; k<updated_data.extent(0); ++k) {
                    reference_data(k,0) = updated_data(k,0);
                    rk_offsets_data(k,0) = updated_data(k,0);
                }

//                    if (field_one == _particles->getFieldManager()->getIDOfFieldFromName("velocity")) {
//
//                        // TODO remove this
//                        // test by replacing solution with analytical at each updated point
//                        _particles->getFieldManager()->getFieldByName("velocity")->
//                                localInitFromVectorFunction(&sw_function);
//                        const mvec_type* velocity_ptr = _particles->getFieldManager()->getFieldByName("velocity")->
//                                getMultiVectorPtrConst();
//                        host_view_type velocity_data = velocity_ptr->getLocalView<host_view_type>();
//
//                        const std::vector<std::vector<std::vector<local_index_type> > >& local_to_dof_map =
//                                this->_particles->getDOFManagerConst()->getDOFMap();
//
//                        for (local_index_type k=0; k<_particles->getCoordsConst()->nLocal(); ++k) {
//                            for (local_index_type l=0; l<3; ++l) {
//                                local_index_type vdof = local_to_dof_map[k][field_one][l];
//                                reference_data(vdof,0) = velocity_data(k,l);
//                                rk_offsets_data(vdof,0) = velocity_data(k,l);
//                                updated_data(vdof,0) = velocity_data(k,l);
//                            }
//                        }
//                    }
//                    if (field_one == _particles->getFieldManager()->getIDOfFieldFromName("height")) {
//
//                        // TODO remove this
//                        // test by replacing solution with analytical at each updated point
//                        _particles->getFieldManager()->getFieldByName("height")->
//                                localInitFromScalarFunction(&sw_function);
//                        _particles->getFieldManager()->getFieldByName("height")->
//                                scale(1./sw_function.getGravity());
//                        const mvec_type* height_ptr = _particles->getFieldManager()->getFieldByName("height")->
//                                getMultiVectorPtrConst();
//                        host_view_type height_data = height_ptr->getLocalView<host_view_type>();
//
//                        const std::vector<std::vector<std::vector<local_index_type> > >& local_to_dof_map =
//                                this->_particles->getDOFManagerConst()->getDOFMap();
//
//                        for (local_index_type k=0; k<_particles->getCoordsConst()->nLocal(); ++k) {
//                            for (local_index_type l=0; l<1; ++l) {
//                                local_index_type vdof = local_to_dof_map[k][field_one][l];
//                                reference_data(vdof,0) = height_data(k,l);
//                                rk_offsets_data(vdof,0) = height_data(k,l);
//                                updated_data(vdof,0) = height_data(k,l);
//                            }
//                        }
//                    }
            } else {
                // copy this to updated_solution so that it starts in the right place
                host_view_type updated_data = updated_solution[0]->getLocalView<host_view_type>();
                host_view_type reference_data = reference_solution[0]->getLocalView<host_view_type>();
                host_view_type rk_offsets_data = rk_offsets[0]->getLocalView<host_view_type>();
                for (size_t k=0; k<updated_data.extent(0); ++k) {
                    reference_data(k,0) = updated_data(k,0);
                    rk_offsets_data(k,0) = updated_data(k,0);
                }
                break;
            }
        }

        t += internal_dt; // advance the time
        timestep_count++;

        bool displacement_data_moved_to_fields = false;
        if (_parameters->get<Teuchos::ParameterList>("io").get<int>("write every") > 0) {

            // check if this is a timestep to write at
            if (timestep_count % _parameters->get<Teuchos::ParameterList>("io").get<int>("write every") == 0) {
                write_count++;

                // add in the write_count
                filename = _parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + base + "_" + std::to_string(write_count) + "." + extension;

                fm.setWriter(filename, _particles);
                if (_comm->getRank()==0)
                    std::cout << "Wrote " << filename << " to disk for simulation time t=" << t << "." << std::endl;

                // requires mapping solution back in order to write particles to disk

                // get solution back from vector onto particles (only at last time step)
                for (InteractingFields field_interaction : _field_interactions) {

                    local_index_type field_one = field_interaction.src_fieldnum;

                    local_index_type row_block = (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) ? _field_to_block_row_map[field_one] : 0;

                    if (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) {
                        _particles->getFieldManager()->updateFieldsFromVector(reference_solution[row_block], field_one, _problem_dof_data);
                    } else {
                        _particles->getFieldManager()->updateFieldsFromVector(reference_solution[0], field_one, _problem_dof_data);
                    }
                }

                fm.write();
                displacement_data_moved_to_fields = true;
            }
        }

        if (!displacement_data_moved_to_fields && _particles->getCoordsConst()->isLagrangian()) {
            // need to update displacements from vector to fields, but this has not
            // already been performed by the writer

            for (InteractingFields field_interaction : _field_interactions) {

                local_index_type field_one = field_interaction.src_fieldnum;

                // only update displacements from the rk solution
                if (field_one == _particles->getFieldManager()->getIDOfFieldFromName("displacement")) {

                    local_index_type row_block = (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) ? _field_to_block_row_map[field_one] : 0;

                    if (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) {
                        _particles->getFieldManager()->updateFieldsFromVector(reference_solution[row_block], field_one, _problem_dof_data);
                    } else {
                        _particles->getFieldManager()->updateFieldsFromVector(reference_solution[0], field_one, _problem_dof_data);
                    }

                    displacement_data_moved_to_fields = true;
                }
            }
        }


//        // TODO: just for diagnostics
//        _particles->getFieldManager()->getFieldByName("reference velocity")->
//                localInitFromVectorFunction(&sw_function);
////             _particles->getFieldManager()->createField(1, "reference height", "m");
//        _particles->getFieldManager()->getFieldByName("reference height")->
//                localInitFromScalarFunction(&sw_function);
//        _particles->getFieldManager()->getFieldByName("reference height")->
//                scale(1./sw_function.getGravity());

        // refresh fields' halo data
        _particles->getFieldManager()->updateFieldsHaloData();

        // increment the physical coordinates by the displacement field
        if (_particles->getCoordsConst()->isLagrangian()) {
            TEUCHOS_TEST_FOR_EXCEPT_MSG(displacement_data_moved_to_fields==false,
                    "Displacement data not updated in particles before up of coordinates.");

            //_particles->getCoords()->updatePhysicalCoordinatesFromFieldData(_particles->getFieldManager()->getFieldByName("reference velocity")->getMultiVectorPtrConst(), internal_dt);

//            double scaling = 1./100.;
//            _particles->getFieldManager()->getFieldByName("velocity")->getMultiVectorPtr()->scale(scaling);
//
//            for (int mm=0; mm<100; ++mm) {
//                _particles->updatePhysicalCoordinatesFromField("displacement");
////                _particles->getCoords()->snapToSphere(sw_function.getRadius());
//            }

            _particles->updatePhysicalCoordinatesFromField("displacement");
//            _particles->getCoords()->snapToSphere(sw_function.getRadius());
//            const mvec_type* velocity_ptr = _particles->getFieldManager()->getFieldByName("velocity")->getMultiVectorPtrConst();
//            host_view_type velocity_data = velocity_ptr->getLocalView<host_view_type>();
//            host_view_type coords_data = _particles->getCoords()->getPts(false, true)->getLocalView<host_view_type>();
//
//            Compadre::CangaSphereTransform sphere_transform(false);
//
//            for (int m=0; m<_particles->getCoordsConst()->nLocal(); ++m) {
//                Compadre::XyzVector xIn = _particles->getCoords()->getLocalCoords(m, false, true);
//
//                const scalar_type lat = xIn.latitude(); // phi
//                const scalar_type lon = xIn.longitude(); // lambda
//                const scalar_type alpha = 0;
//
//                Teuchos::SerialDenseMatrix<int,double> Q(3,2);
//                Teuchos::SerialDenseMatrix<int,double> NormalQ(2,2);
//                Teuchos::SerialDenseMatrix<int,double> solution(2,1);
//                Teuchos::SerialDenseMatrix<int,double> rhs(3,1);
//                Teuchos::SerialDenseMatrix<int,double> NormalRhs(2,1);
//
//                Q(0,0) = -std::sin(lon); Q(0,1) = - std::sin(lat) * std::cos(lon);
//                Q(1,0) = std::cos(lon);  Q(1,1) = - std::sin(lat) * std::sin(lon);
//                Q(2,0) = 0;              Q(2,1) = std::cos(lat);
//
//                rhs(0,0) = velocity_data(m, 0);
//                rhs(1,0) = velocity_data(m, 1);
//                rhs(2,0) = velocity_data(m, 2);
//
//                NormalQ.multiply((Teuchos::ETransp)1,(Teuchos::ETransp)0,1,Q,Q,0);
//                NormalRhs.multiply((Teuchos::ETransp)1,(Teuchos::ETransp)0,1,Q,rhs,0);
//
//                // invert the matrix
//                Teuchos::SerialDenseSolver<int,double> solver;
//                solver.setMatrix( Teuchos::rcp( &NormalQ , false ) );
//                solver.setVectors( Teuchos::rcp( &solution, false ), Teuchos::rcp( &NormalRhs, false ) );
//                solver.solve();
//
////
////                const scalar_type U = _u0*(std::cos(lat)*std::cos(_alpha) +  std::cos(lon)*std::sin(lat)*std::sin(_alpha));
////                const scalar_type V = -_u0*std::sin(lon)*std::sin(_alpha);
////
////                return xyz_type( -U * std::sin(lon) - V * std::sin(lat) * std::cos(lon),
////                                    U * std::cos(lon) - V * std::sin(lat) * std::sin(lon),
////                                    V * std::cos(lat) )
//
//                double u_velocity = solution(0,0); // velocity with respect to lat and lon
//                double v_velocity = solution(1,0);
//
//                printf("u velocity %f\n", u_velocity);
//                printf("v velocity %f\n", v_velocity);
//
//                const scalar_type new_lat = lat + internal_dt*u_velocity; // phi
//                const scalar_type new_lon = lon + internal_dt*v_velocity; // lambda
//
//                Compadre::XyzVector lat_lon_in(new_lat, new_lon, 0);
//                Compadre::XyzVector transformed_lat_lon = sphere_transform.evalVector(lat_lon_in);
//
//                scalar_type coord_norm = std::sqrt(transformed_lat_lon.x*transformed_lat_lon.x + transformed_lat_lon.y*transformed_lat_lon.y + transformed_lat_lon.z*transformed_lat_lon.z);
//
//                transformed_lat_lon.x *= sw_function.getRadius()*coord_norm;
//                transformed_lat_lon.y *= sw_function.getRadius()*coord_norm;
//                transformed_lat_lon.z *= sw_function.getRadius()*coord_norm;
//
//                coords_data(m, 0) = transformed_lat_lon.x;
//                coords_data(m, 1) = transformed_lat_lon.y;
//                coords_data(m, 2) = transformed_lat_lon.z;
//
//            }


            // TODO: just for diagnos// scale displacements to zero
                        for (InteractingFields field_interaction : _field_interactions) {
            
                            local_index_type field_one = field_interaction.src_fieldnum;
            
                            // only update displacements from the rk solution
                            if (field_one == _particles->getFieldManager()->getIDOfFieldFromName("displacement")) {
            
                                local_index_type row_block = (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) ? _field_to_block_row_map[field_one] : 0;
            
                                if (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) {
                                    reference_solution[row_block]->scale(0);
                                    rk_offsets[row_block]->scale(0);
                                    updated_solution[row_block]->scale(0);
                                    _particles->getFieldManager()->updateFieldsFromVector(reference_solution[row_block], field_one, _problem_dof_data);
                                } else {
                                    host_view_type reference_data = reference_solution[0]->getLocalView<host_view_type>();
                                    host_view_type rk_offsets_data = rk_offsets[0]->getLocalView<host_view_type>();
                                    host_view_type updated_data = updated_solution[0]->getLocalView<host_view_type>();

                                    const local_dof_map_view_type local_to_dof_map =
                                            this->_particles->getDOFManagerConst()->getDOFMap();

                                    for (local_index_type k=0; k<_particles->getCoordsConst()->nLocal(); ++k) {
                                        for (local_index_type l=0; l<3; ++l) {
                                            local_index_type vdof = local_to_dof_map(k, field_one, l);
                                            reference_data(vdof,0) = 0;
                                            rk_offsets_data(vdof,0) = 0;
                                            updated_data(vdof,0) = 0;
                                        }
                                    }
                                    _particles->getFieldManager()->updateFieldsFromVector(reference_solution[0], field_one, _problem_dof_data);
                                }
            
                                displacement_data_moved_to_fields = true;
                            }
                        }
//            _particles->getFieldManager()->getFieldByName("reference velocity")->
//                    localInitFromVectorFunction(&sw_function);
////             _particles->getFieldManager()->createField(1, "reference height", "m");
//            _particles->getFieldManager()->getFieldByName("reference height")->
//                    localInitFromScalarFunction(&sw_function);
//            _particles->getFieldManager()->getFieldByName("reference height")->
//                    scale(1./sw_function.getGravity());



            // zero out displacements
            //_particles->getFieldManager()->getFieldByName("displacement")->scale(0);


//            // scale displacements to zero
//            for (InteractingFields field_interaction : _field_interactions) {
//
//                local_index_type field_one = field_interaction.src_fieldnum;
//                local_index_type field_two = field_interaction.trg_fieldnum;
//
//                // only update displacements from the rk solution
//                if (field_one == _particles->getFieldManager()->getIDOfFieldFromName("displacement")) {
//
//                    local_index_type row_block = (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) ? _field_to_block_row_map[field_one] : 0;
//                    local_index_type col_block = (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) ? _field_to_block_col_map[field_two] : 0;
//
//                    if (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) {
//                        reference_solution[row_block]->scale(0);
//                        rk_offsets[row_block]->scale(0);
//                        updated_solution[row_block]->scale(0);
//                        _particles->getFieldManager()->updateFieldsFromVector(reference_solution[row_block], field_one);
//                    } else {
//                        // TODO: would need to scale just this fields portions by zero
//                    }
//
//                    displacement_data_moved_to_fields = true;
//                }
//            }

            // don't find new neighborhoods, keep old ones from original configuration
            //// find new neighborhoods
            //{
            //    // get old search size
            //    scalar_type h_min = _particles->getNeighborhood()->computeMinHSupportSize(true /*global min*/);

            //    // destroy old neighborhood and create new one
            //    _particles->createNeighborhood();
            //    _particles->getNeighborhood()->setAllHSupportSizes(h_min);

            //    if (_particles->getCoordsConst()->getComm()->getRank()==0)
            //        printf("hmin: %f\n", h_min);
            //    // determine number of neighbors needed and construct list
            //    local_index_type neighbors_needed = GMLS::getNP(_parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"), 2);
            //    _particles->getNeighborhood()->constructAllNeighborLists(_particles->getCoordsConst()->getHaloSize(), 
            //            _parameters->get<Teuchos::ParameterList>("neighborhood").get<std::string>("search type"), 
            //            neighbors_needed,
            //            _parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("cutoff multiplier"),
            //            _parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("size"),
            //            _parameters->get<Teuchos::ParameterList>("neighborhood").get<bool>("uniform radii"),
            //            _parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("radii post search scaling"));
            //}

        }
    }

    // refresh fields' halo data
    _particles->getFieldManager()->updateFieldsHaloData();

    // get solution back from vector onto particles (only at last time step)
    for (InteractingFields field_interaction : _field_interactions) {

        local_index_type field_one = field_interaction.src_fieldnum;

        local_index_type row_block = (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) ? _field_to_block_row_map[field_one] : 0;

        if (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) {
            _particles->getFieldManager()->updateFieldsFromVector(reference_solution[row_block], field_one, _problem_dof_data);
        } else {
            _particles->getFieldManager()->updateFieldsFromVector(reference_solution[0], field_one, _problem_dof_data);
        }
    }

    // user selected to only output as last time step
    if (_parameters->get<Teuchos::ParameterList>("io").get<int>("write every") == 0) {
        // only write last timestep data

        filename = _parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + base + "." + extension;
        if (_particles->getCoordsConst()->getComm()->getRank()==0)
            printf("filename: %s\n", filename.c_str());

        fm.setWriter(filename, _particles);

        fm.write();
    }


    if (_particles->getCoordsConst()->getComm()->getRank()==0)
        printf("final time: %.16f\n", t);
//        if (_particles->getCoordsConst()->isLagrangian()) {
//            printf("Lagrangian!");
//            // temporary
//            // retrieve the displacement from the solution and put it in the particles
//            for (InteractingFields field_interaction : _field_interactions) {
//
//                local_index_type field_one = field_interaction.src_fieldnum;
//                local_index_type field_two = field_interaction.trg_fieldnum;
//
//                local_index_type row_block = (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) ? _field_to_block_row_map[field_one] : 0;
//                local_index_type col_block = (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) ? _field_to_block_col_map[field_two] : 0;
//
//                if (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) {
//                    _particles->getFieldManager()->updateFieldsFromVector(reference_solution[row_block], field_one);
//                } else {
//                    _particles->getFieldManager()->updateFieldsFromVector(reference_solution[0]);
//                    break;
//                }
//            }
//
//            // update the coordinates by the displacements calculated
//            _particles->updatePhysicalCoordinatesFromField("displacement");
////            _particles->getCoords()->snapToSphere();
//
//            // zero out displacements
//            _particles->getFieldManager()->getFieldByName("displacement")]->scale(0);
//
//            printf("time: %.16f\n", t);
//        }

//    // get new neighborhoods before going to next timestep
//    {
//        // get old search size
//        scalar_type h_min = _particles->getNeighborhood()->getMinimumHSupportSize();
//
//        // destroy old neighborhood and create new one
//        _particles->createNeighborhood();
//        _particles->getNeighborhood()->setAllHSupportSizes(h_min);
//
//        // determine number of neighbors needed and construct list
//        local_index_type neighbors_needed = GMLS::getNP(_parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"), 2);
//        local_index_type extra_neighbors = _parameters->get<Teuchos::ParameterList>("remap").get<double>("neighbors needed multiplier") * neighbors_needed;
//        _particles->getNeighborhood()->constructAllNeighborList(_particles->getCoordsConst()->getHaloSize(), extra_neighbors);
//    }


//    for (InteractingFields field_interaction : _field_interactions) {
//
//        local_index_type field_one = field_interaction.src_fieldnum;
//        local_index_type field_two = field_interaction.trg_fieldnum;
//
//        local_index_type row_block = (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) ? _field_to_block_row_map[field_one] : 0;
//        local_index_type col_block = (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) ? _field_to_block_col_map[field_two] : 0;
//
//        if (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) {
//            // copy this to updated_solution so that it starts in the right place
//            host_view_type reference_data = reference_solution[row_block]->getLocalView<host_view_type>();
//            for (local_index_type k=0; k<reference_data.extent(0); ++k) {
//                printf("field #%d: index #%d: val:%.16f\n", field_one, k, reference_data(k,0));
//            }
//        } else {
//            // copy this to updated_solution so that it starts in the right place
//            host_view_type reference_data = reference_solution[0]->getLocalView<host_view_type>();
//            for (local_index_type k=0; k<reference_data.extent(0); ++k) {
//                printf("field #%d: index #%d: val:%.16f\n", field_one, k, reference_data(k,0));
//            }
//            break;
//        }
//    }
}

void ProblemExplicitTransientT::solveNewmark(Teuchos::ParameterList& parameters_time, scalar_type t_0, scalar_type t_end, scalar_type dt, std::string filename) {

    // only support explicit methods, i.e. beta=0
    scalar_type gamma = parameters_time.get<double>("gamma");
    scalar_type beta = parameters_time.get<double>("beta");
    TEUCHOS_TEST_FOR_EXCEPT_MSG(beta != 0, "Currently only support for beta==0");

    // check that any field having an interaction as source (is a member of the fields being time-integrated) 
    // also has a given *_velocity as a field
    //
    // e.g. if "v" is a field that is registered for interaction (a solution variable)
    //      then "v_velocity" should be a field for the particle set (not registered for interaction)
    std::vector<local_index_type> displacement_to_velocity_map(_particles->getFieldManagerConst()->getVectorOfFields().size());
    for (InteractingFields field_interaction : _field_interactions) {
        auto this_src_field_name = _particles->getFieldManagerConst()->getFieldByID(field_interaction.src_fieldnum)->getName();
        // this_src_field_name holds * in *_velocity field name we are looking for
        try {
            displacement_to_velocity_map[field_interaction.src_fieldnum] = 
                _particles->getFieldManagerConst()->getIDOfFieldFromName(this_src_field_name+"_velocity");
        } catch (...) {
            std::string msg = "Field " + this_src_field_name + " involved in field interactions, but field " 
                + this_src_field_name + "_velocity not found.";
            TEUCHOS_TEST_FOR_EXCEPT_MSG(true, msg.c_str());
        }
    }

    // location *_acceleration for any registered field interaction on * (for populating particles at end of simulation)
    std::vector<local_index_type> displacement_to_acceleration_map(_particles->getFieldManagerConst()->getVectorOfFields().size(), -1);
    for (InteractingFields field_interaction : _field_interactions) {
        auto this_src_field_name = _particles->getFieldManagerConst()->getFieldByID(field_interaction.src_fieldnum)->getName();
        // this_src_field_name holds * in *_velocity field name we are looking for
        try {
            displacement_to_acceleration_map[field_interaction.src_fieldnum] = 
                _particles->getFieldManagerConst()->getIDOfFieldFromName(this_src_field_name+"_acceleration");
        } catch (...) {
            std::string msg = "Field " + this_src_field_name + " involved in field interactions, but field " 
                + this_src_field_name + "_acceleration not found.";
            TEUCHOS_TEST_FOR_EXCEPT_MSG(true, msg.c_str());
        }
    }

    // details for writing to files every nth timestep
    Compadre::FileManager fm;
    std::string base, extension;

    if (_parameters->get<Teuchos::ParameterList>("io").get<int>("write every") >= 0) {
        // parse by removing extension of filename

        // get extension
        size_t pos = filename.rfind('.', filename.length());

        // verify there was an extension extension
        TEUCHOS_TEST_FOR_EXCEPT_MSG(pos <= 0, "Filename specified to solve(...) is missing an extension.");

        base = filename.substr(0, pos);
        extension = filename.substr(pos+1, filename.length() - pos);
    }

    std::vector<Teuchos::RCP<mvec_type> > displacement_t_solution(_particles->getFieldManagerConst()->getVectorOfFields().size(), Teuchos::null);
    std::vector<Teuchos::RCP<mvec_type> > velocity_t_solution(_particles->getFieldManagerConst()->getVectorOfFields().size(), Teuchos::null);
    std::vector<Teuchos::RCP<mvec_type> > acceleration_tm1_solution(_particles->getFieldManagerConst()->getVectorOfFields().size(), Teuchos::null);

    // allocate displacement, velocity, and acceleration multivectors and copy initial conditions to these vectors
    for (InteractingFields field_interaction : _field_interactions) {

        local_index_type field_one = field_interaction.src_fieldnum;
        local_index_type row_block = (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) ? _field_to_block_row_map[field_one] : 0;

        // this allows us to overwrite the fields in _particles for each RK step and use
        // existing tools to distribute halo data
        // get all the vectors set up
        if (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) {
            if (displacement_t_solution[row_block].is_null()) {
                auto field_one_velocity = displacement_to_velocity_map[field_one];
                bool setToZero = true;
                displacement_t_solution[row_block] = Teuchos::rcp(new mvec_type(row_map[row_block], 1, setToZero));
                velocity_t_solution[row_block] = Teuchos::rcp(new mvec_type(row_map[row_block], 1, setToZero));
                acceleration_tm1_solution[row_block] = Teuchos::rcp(new mvec_type(row_map[row_block], 1, setToZero));

                // get solution from _particles and put it in displacements
                _particles->getFieldManagerConst()->updateVectorFromFields(displacement_t_solution[row_block], field_one, _problem_dof_data);

                // get velocity from _particles (use mapping we created above)
                _particles->getFieldManagerConst()->updateVectorFromFields(velocity_t_solution[row_block], field_one_velocity, _problem_dof_data);
            }
        } else {
            // unblocked system is incompatible with the displacement/velocity design
            std::string msg = "In the solver parameter list, blocked must be set to true.";
            TEUCHOS_TEST_FOR_EXCEPT_MSG(true, msg.c_str());
        }
        _particles->getCoordsConst()->getComm()->barrier();
    }


    // TODO: check parameter list for if operator is lumped
    bool LUMPED=true;
    
    // size of first timestep that will be taken
    scalar_type internal_dt = std::min(dt, t_end-t_0);

    // solve acceleration at t0 (NO DAMPING!)
    for (InteractingFields field_interaction : _field_interactions) {

        local_index_type field_one = field_interaction.src_fieldnum;
        local_index_type field_two = field_interaction.trg_fieldnum;
        auto field_one_velocity = displacement_to_velocity_map[field_one];

        for (op_needing_interaction op : field_interaction.ops_needing) {
            // (all store into _b), so _b holds stage solution
            if (op == op_needing_interaction::bc) {
                assembleBCS(field_one, field_two, t_0, internal_dt, -1.0 /* indicates t_0 condition */);
            }
            else if (op == op_needing_interaction::source) {
                assembleRHS(field_one, field_two, t_0, internal_dt, -1.0 /* indicates t_0 condition */);
            }
            else if (op == op_needing_interaction::physics) {
                if (LUMPED) {
                    assembleOperatorToVector(field_one, field_two, t_0, internal_dt, -1.0 /* indicates t_0 condition */);
                } else {
                    // assembleOperator to a matrix
                }
            }
        }
        _particles->getCoordsConst()->getComm()->barrier();
    }

    // a solve could go here to support implicit aspects
    if (!LUMPED) {
        // solve would go here, solution in _x needs extracted
    }

    // extract acceleration solution at time t0 from particles
    for (InteractingFields field_interaction : _field_interactions) {
        local_index_type field_one = field_interaction.src_fieldnum;
        local_index_type row_block = _field_to_block_row_map[field_one];

        host_view_type solution_data = _b[row_block]->getLocalView<host_view_type>();
        host_view_type acceleration_data = acceleration_tm1_solution[row_block]->getLocalView<host_view_type>();
        // diagnostic only
        //host_view_type velocity_data = velocity_t_solution[row_block]->getLocalView<host_view_type>();
        //host_view_type displacement_data = displacement_t_solution[row_block]->getLocalView<host_view_type>();

        for (size_t k=0; k<acceleration_data.extent(0); ++k) {
            acceleration_data(k,0) = solution_data(k,0);
            // diagnostic only
            //printf("acceleration data(%d): %f\n", k, acceleration_data(k,0));
            //printf("velocity data(%d): %f\n", k, velocity_data(k,0));
            //printf("displacement data(%d): %f\n", k, displacement_data(k,0));
        }
        _particles->getCoordsConst()->getComm()->barrier();
    }

    
    scalar_type old_t = t_0;
    scalar_type t = t_0 + internal_dt;
    scalar_type old_dt = 0;

    int timestep_count = 0;
    int write_count = 0;
    while ( t<(t_end+1e-15) && internal_dt>1e-15) {
        //printf("t: %f, dt: %f\n", t, internal_dt);

        // move displacement to t-dt to t
        for (InteractingFields field_interaction : _field_interactions) {
            local_index_type field_one = field_interaction.src_fieldnum;
            local_index_type row_block = _field_to_block_row_map[field_one];

            host_view_type displacement_data = displacement_t_solution[row_block]->getLocalView<host_view_type>();
            host_view_type velocity_data = velocity_t_solution[row_block]->getLocalView<host_view_type>();
            host_view_type acceleration_data = acceleration_tm1_solution[row_block]->getLocalView<host_view_type>();

            // time integrate solution
            host_view_local_index_type bc_id = _particles->getFlags()->getLocalView<host_view_local_index_type>();
            for (size_t k=0; k<acceleration_data.extent(0); ++k) {
                displacement_data(k,0) += internal_dt*velocity_data(k,0) + 0.5*(1-2.0*beta)*internal_dt*internal_dt*acceleration_data(k,0);
            }
            _particles->getCoordsConst()->getComm()->barrier();
        }

        // move the updated displacements and old velocity/acceleration back onto the particles
        for (InteractingFields field_interaction : _field_interactions) {
            local_index_type field_one = field_interaction.src_fieldnum;
            local_index_type row_block = _field_to_block_row_map[field_one];
            auto field_one_velocity = displacement_to_velocity_map[field_one];
            _particles->getFieldManager()->updateFieldsFromVector(displacement_t_solution[row_block], field_one, _problem_dof_data);
            _particles->getFieldManager()->updateFieldsFromVector(velocity_t_solution[row_block], field_one_velocity, _problem_dof_data);
            auto field_one_acceleration = displacement_to_acceleration_map[field_one];
            if (field_one_acceleration > -1) {
                _particles->getFieldManager()->updateFieldsFromVector(acceleration_tm1_solution[row_block], field_one_acceleration, _problem_dof_data);
            }
            _particles->getCoordsConst()->getComm()->barrier();
        }

        // solve acceleration (NO DAMPING!)
        for (InteractingFields field_interaction : _field_interactions) {

            local_index_type field_one = field_interaction.src_fieldnum;
            local_index_type field_two = field_interaction.trg_fieldnum;

            for (op_needing_interaction op : field_interaction.ops_needing) {
                // (all store into _b), so _b holds solution
                if (op == op_needing_interaction::bc) {
                    assembleBCS(field_one, field_two, t, internal_dt, old_dt);
                }
                if (op == op_needing_interaction::source) {
                    assembleRHS(field_one, field_two, t, internal_dt, old_dt);
                }
                else if (op == op_needing_interaction::physics) {
                    if (LUMPED) {
                        assembleOperatorToVector(field_one, field_two, t, internal_dt, old_dt);
                    } else {
                        // assembleOperator to a matrix
                    }
                }
            }
            //// TODO: REMOVE THIS!
            //local_index_type row_block = _field_to_block_row_map[field_one];
            //host_view_type solution_data = _b[row_block]->getLocalView<host_view_type>();
            //for (size_t k=0; k<solution_data.extent(0); ++k) {
            //    solution_data(k,0) = 1.0;
            //}
            _particles->getCoordsConst()->getComm()->barrier();
        }

        // a solve could go here to support implicit aspects
        if (!LUMPED) {
            // solve would go here, solution in _x extracted later
        } 

        for (InteractingFields field_interaction : _field_interactions) {

            local_index_type field_one = field_interaction.src_fieldnum;
            local_index_type row_block = _field_to_block_row_map[field_one];

            host_view_type solution_data;
            if (!LUMPED) {
                // solution in _x needs extracted
            } else {
                // extract solution from _b
                solution_data = _b[row_block]->getLocalView<host_view_type>();
            }

            // update _particles
            host_view_type velocity_data = velocity_t_solution[row_block]->getLocalView<host_view_type>();
            host_view_type acceleration_data = acceleration_tm1_solution[row_block]->getLocalView<host_view_type>();

            // time integrate solution
            host_view_local_index_type bc_id = _particles->getFlags()->getLocalView<host_view_local_index_type>();
            for (size_t k=0; k<acceleration_data.extent(0); ++k) {
                if (bc_id(k,0)==0) {
                    //printf("nb:vel before: %f, a_before: %f, a_new: %f\n", velocity_data(k,0), acceleration_data(k,0), solution_data(k,0));
                    velocity_data(k,0) += (1-gamma)*internal_dt*acceleration_data(k,0) + gamma*internal_dt*solution_data(k,0);
                    acceleration_data(k,0) = solution_data(k,0);
                } else {
                    //printf("ib:vel before: %f, a_before: %f, a_new: %f\n", velocity_data(k,0), acceleration_data(k,0), solution_data(k,0));
                    velocity_data(k,0) += (1-gamma)*internal_dt*acceleration_data(k,0) + gamma*internal_dt*solution_data(k,0);
                    acceleration_data(k,0) = solution_data(k,0);
                }
            }
            _particles->getCoordsConst()->getComm()->barrier();
        }
        
        old_t = t;
        old_dt = internal_dt;
        internal_dt = std::min(dt, t_end-t);
        t += internal_dt; // advance the time
        timestep_count++;

        if (_parameters->get<Teuchos::ParameterList>("io").get<int>("write every") > 0) {

            // check if this is a timestep to write at
            if (timestep_count % _parameters->get<Teuchos::ParameterList>("io").get<int>("write every") == 0) {
                write_count++;

                // add in the write_count
                filename = _parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + base + "_" + std::to_string(write_count) + "." + extension;

                fm.setWriter(filename, _particles);
                if (_comm->getRank()==0)
                    std::cout << "Wrote " << filename << " to disk for simulation time t=" << t << "." << std::endl;

                // requires mapping solution back in order to write particles to disk
                for (InteractingFields field_interaction : _field_interactions) {
                    local_index_type field_one = field_interaction.src_fieldnum;
                    local_index_type row_block = _field_to_block_row_map[field_one];
                    auto field_one_velocity = displacement_to_velocity_map[field_one];
                    _particles->getFieldManager()->updateFieldsFromVector(displacement_t_solution[row_block], field_one, _problem_dof_data);
                    _particles->getFieldManager()->updateFieldsFromVector(velocity_t_solution[row_block], field_one_velocity, _problem_dof_data);
                    auto field_one_acceleration = displacement_to_acceleration_map[field_one];
                    if (field_one_acceleration > -1) {
                        _particles->getFieldManager()->updateFieldsFromVector(acceleration_tm1_solution[row_block], field_one_acceleration, _problem_dof_data);
                    }
                    _particles->getCoordsConst()->getComm()->barrier();
                }

                fm.write();
            }
        }
    }


    // get solution back from vector onto particles (only at last time step)
    for (InteractingFields field_interaction : _field_interactions) {
        local_index_type field_one = field_interaction.src_fieldnum;
        local_index_type row_block = _field_to_block_row_map[field_one];
        auto field_one_velocity = displacement_to_velocity_map[field_one];
        _particles->getFieldManager()->updateFieldsFromVector(displacement_t_solution[row_block], field_one, _problem_dof_data);
        _particles->getFieldManager()->updateFieldsFromVector(velocity_t_solution[row_block], field_one_velocity, _problem_dof_data);
        auto field_one_acceleration = displacement_to_acceleration_map[field_one];
        if (field_one_acceleration > -1) {
            _particles->getFieldManager()->updateFieldsFromVector(acceleration_tm1_solution[row_block], field_one_acceleration, _problem_dof_data);
        }
        _particles->getCoordsConst()->getComm()->barrier();
    }

    // refresh fields' halo data
    _particles->getFieldManager()->updateFieldsHaloData();

    // user selected to only output as last time step
    if (_parameters->get<Teuchos::ParameterList>("io").get<int>("write every") == 0) {
        // only write last timestep data

        filename = _parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + base + "." + extension;
        if (_particles->getCoordsConst()->getComm()->getRank()==0)
            printf("filename: %s\n", filename.c_str());

        fm.setWriter(filename, _particles);

        fm.write();
    }

    if (_comm->getRank()==0) {
        printf("%d timestep%s taken.\n", timestep_count, (timestep_count==1) ? "" : "s");
        printf("final time: %.16f\n", t);
    }
}

void ProblemExplicitTransientT::buildMaps(local_index_type field_one, local_index_type field_two) {
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

void ProblemExplicitTransientT::assembleOperatorToVector(local_index_type field_one, local_index_type field_two, scalar_type simulation_time, scalar_type delta_time, scalar_type previous_delta_time) {
    if (field_two<0) field_two = field_one;
    local_index_type row_block = _field_to_block_row_map[field_one];

    TEUCHOS_TEST_FOR_EXCEPT_MSG(_OP.is_null(), "Physics not set before assembleOperator() called.");
    _OP->setParameters(*_parameters);

    if (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")==true) {
        if (_b[row_block].is_null()) {
            bool setToZero = true;
            _b[row_block] = Teuchos::rcp(new mvec_type(row_map[row_block], 1, setToZero));
        }
        _OP->setMultiVector(_b[row_block].getRawPtr());
    } else {
        if (_b[0].is_null()) {
            bool setToZero = true;
            _b[0] = Teuchos::rcp(new mvec_type(row_map[0], 1, setToZero));
        }
            _OP->setMultiVector(_b[0].getRawPtr());
    }
    _OP->computeVector(field_one, field_two, simulation_time, delta_time, previous_delta_time);
}

void ProblemExplicitTransientT::assembleRHS(local_index_type field_one, local_index_type field_two, scalar_type simulation_time, scalar_type delta_time, scalar_type previous_delta_time) {
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
    _RHS->evaluateRHS(field_one, field_two, simulation_time, delta_time, previous_delta_time);
}

void ProblemExplicitTransientT::assembleBCS(local_index_type field_one, local_index_type field_two, scalar_type simulation_time, scalar_type delta_time, scalar_type previous_delta_time) {
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
    _BCS->applyBoundaries(field_one, field_two, simulation_time, delta_time, previous_delta_time);

}

}
