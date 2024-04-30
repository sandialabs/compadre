#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_Core.hpp>

#include <Kokkos_Core.hpp>
#include <Compadre_ProblemT.hpp>
#include <Compadre_ParticlesT.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_DOFManager.hpp>
#include <Compadre_EuclideanCoordsT.hpp>
#include <Compadre_NeighborhoodT.hpp>
#include <Compadre_FieldT.hpp>
#include <Compadre_XyzVector.hpp>
#include <Compadre_AnalyticFunctions.hpp>
#include <Compadre_FileIO.hpp>
#include <Compadre_ParameterManager.hpp>
#include <Compadre_RemapManager.hpp>

#include <Compadre_GMLS.hpp> // for getNP()

#include <Compadre_ReactionDiffusion_Sources.hpp>
#include <Compadre_ReactionDiffusion_BoundaryConditions.hpp>
#include <Compadre_ReactionDiffusion_Operator.hpp>

#define STACK_TRACE(call) try { call; } catch (const std::exception& e ) { TEUCHOS_TRACE(e); }

typedef int LO;
typedef long GO;
typedef double ST;
typedef Compadre::XyzVector xyz_type;
typedef Compadre::EuclideanCoordsT CT;

using namespace Compadre;

int main (int argc, char* args[]) {
#ifdef TRILINOS_LINEAR_SOLVES

    Kokkos::initialize(argc, args);

    {
    Teuchos::RCP<Compadre::ParameterManager> parameter_manager;
    if (argc > 1)
        parameter_manager = Teuchos::rcp(new Compadre::ParameterManager(argc, args));
    else {
        parameter_manager = Teuchos::rcp(new Compadre::ParameterManager());
        std::cout << "WARNING: No parameter list given. Default parameters used." << std::endl;
    }
    if (parameter_manager->helpRequested()) return 0;
    if (parameter_manager->parseError()) return -1;


    Teuchos::RCP<Teuchos::ParameterList> parameters = parameter_manager->getList();

    Teuchos::GlobalMPISession mpi(&argc, &args);
    Teuchos::oblackholestream bstream;
    Teuchos::RCP<const Teuchos::Comm<int> > comm = Teuchos::DefaultComm<int>::getComm();
    
    
    Teuchos::RCP<Teuchos::Time> FirstReadTime = Teuchos::TimeMonitor::getNewCounter ("1st Read Time");
    Teuchos::RCP<Teuchos::Time> AssemblyTime = Teuchos::TimeMonitor::getNewCounter ("Assembly Time");
    Teuchos::RCP<Teuchos::Time> SolvingTime = Teuchos::TimeMonitor::getNewCounter ("Solving Time");
    Teuchos::RCP<Teuchos::Time> WriteTime = Teuchos::TimeMonitor::getNewCounter ("Write Time");
    Teuchos::RCP<Teuchos::Time> SecondReadTime = Teuchos::TimeMonitor::getNewCounter ("2nd Read Time");

    // this proceeds setting up the problem so that the parameters will be propagated down into the physics and bcs
    //try {
    //    parameters->get<std::string>("solution type");
    //} catch (Teuchos::Exceptions::InvalidParameter & exception) {
    //    parameters->set<std::string>("solution type","sine");
    //}
    try {
        parameters->get<Teuchos::ParameterList>("io").get<bool>("plot quadrature");
    } catch (Teuchos::Exceptions::InvalidParameter & exception) {
        parameters->get<Teuchos::ParameterList>("io").set<bool>("plot quadrature", false);
    }
    bool plot_quadrature = parameters->get<Teuchos::ParameterList>("io").get<bool>("plot quadrature");
    local_index_type input_dim = parameters->get<Teuchos::ParameterList>("io").get<local_index_type>("input dimensions");

    bool l2_op = parameters->get<Teuchos::ParameterList>("physics").get<std::string>("operator")=="l2";
    bool rd_op = parameters->get<Teuchos::ParameterList>("physics").get<std::string>("operator")=="rd";
    bool le_op = parameters->get<Teuchos::ParameterList>("physics").get<std::string>("operator")=="le";
    bool vl_op = parameters->get<Teuchos::ParameterList>("physics").get<std::string>("operator")=="vl";
    bool st_op = parameters->get<Teuchos::ParameterList>("physics").get<std::string>("operator")=="st";
    bool mix_le_op = parameters->get<Teuchos::ParameterList>("physics").get<std::string>("operator")=="mix_le";

    auto basis_type_str = parameters->get<Teuchos::ParameterList>("remap").get<std::string>("basis type");
    transform(basis_type_str.begin(), basis_type_str.end(), basis_type_str.begin(), ::tolower);
    auto velocity_basis_type_divfree = (basis_type_str == "divfree");
    auto velocity_basis_type_vector = (basis_type_str == "vector") || velocity_basis_type_divfree;

    bool use_vector_gmls = velocity_basis_type_vector;
    bool use_vector_grad_gmls = velocity_basis_type_vector;

    // treatment of null space in pressure 
    bool use_pinning = false, use_lm = false;
    if (st_op || mix_le_op) {
        if (parameters->get<Teuchos::ParameterList>("solver").get<std::string>("pressure null space")=="pinning") {
            use_pinning = true;
            use_lm = false;
        } else if (parameters->get<Teuchos::ParameterList>("solver").get<std::string>("pressure null space")=="lm") {
            use_pinning = false;
            use_lm = true;
        } else {
            use_pinning = false;
            use_lm = false;
        }
    }

    bool use_sip = false, use_vms = false;
    if (parameters->get<Teuchos::ParameterList>("physics").get<std::string>("dg stabilization")=="sip") {
        use_sip = true;
        use_vms = false;
    } else if (parameters->get<Teuchos::ParameterList>("physics").get<std::string>("dg stabilization")=="vms") {
        use_sip = false;
        use_vms = true;
    } else {
        use_sip = false;
        use_vms = false;
    }

    bool use_neighbor_weighting = false, use_side_weighting = false;
    if (parameters->get<Teuchos::ParameterList>("physics").get<std::string>("penalty weighting")=="neighbor") {
        use_neighbor_weighting = true;
        use_side_weighting = false;
    } else if (parameters->get<Teuchos::ParameterList>("physics").get<std::string>("penalty weighting")=="side") {
        use_neighbor_weighting = false;
        use_side_weighting = true;
    } else {
        //if (!use_vms) compadre_assert_release(false && "Illegal choice for \"penalty weighting\".");
    }

    {
        const std::string cells_testfilename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file prefix") + "/" + parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file");
        std::string particles_testfilename;
        try {
            particles_testfilename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file prefix") + "/" + parameters->get<Teuchos::ParameterList>("io").get<std::string>("particles input file");
        } catch (...) {
            particles_testfilename = cells_testfilename;
        }

        //
        // GMLS Test
        //
        double h_size = 1./12.;

        TEUCHOS_TEST_FOR_EXCEPT_MSG(input_dim<2, "Only supported for A-D problem in 2 or 3D.");
              
        Teuchos::RCP<Compadre::ParticlesT> particles;
        Teuchos::RCP<Compadre::ParticlesT> cells =
                Teuchos::rcp( new Compadre::ParticlesT(parameters, comm, input_dim));
        const CT * coords = (CT*)(cells->getCoordsConst());
        
        //Read in data file
        {
            FirstReadTime->start();
            Compadre::FileManager fm;
            fm.setReader(cells_testfilename, cells);
            fm.read();
            FirstReadTime->stop();
        }

        cells->zoltan2Initialize();

        if (particles_testfilename != cells_testfilename) {
            //Read in data file
            FirstReadTime->start();
            Compadre::FileManager fm;
            particles = Teuchos::rcp( new Compadre::ParticlesT(parameters, comm, input_dim));
            fm.setReader(particles_testfilename, particles);
            fm.read();
            FirstReadTime->stop();

            particles->zoltan2Initialize();
        } else {
            // particles are cell midpoints
            particles = cells;
        }

        ST halo_size;
        std::string velocity_name;
        std::string pressure_name;
        {
            if (parameters->get<Teuchos::ParameterList>("halo").get<bool>("dynamic")) {
                halo_size = h_size * parameters->get<Teuchos::ParameterList>("halo").get<double>("multiplier");
            } else {
                halo_size = parameters->get<Teuchos::ParameterList>("halo").get<double>("size");
            }
            cells->buildHalo(halo_size);
            if (particles!=cells) particles->buildHalo(halo_size);
            if (st_op || mix_le_op) { 
                velocity_name = "solution_velocity";
                pressure_name = "solution_pressure";
                particles->getFieldManager()->createField(input_dim, velocity_name, "m/s");
                particles->getFieldManager()->createField(1, pressure_name, "m/s");
				if (use_lm) particles->getFieldManager()->createField(1, "lagrange multiplier", "NA");
            } else if (le_op || vl_op) {
                velocity_name = "solution";
                particles->getFieldManager()->createField(input_dim, velocity_name, "m/s");
            } else {
                velocity_name = "solution";
                particles->getFieldManager()->createField(1, velocity_name, "m/s");
            }
            particles->createDOFManager();



            ////Set the radius for the neighbor list:
             //ST h_support;
             //if (parameters->get<Teuchos::ParameterList>("neighborhood").get<bool>("dynamic radius")) {
             //    h_support = h_size;
             //} else {
             //    h_support = parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("size");
             //}
             //cells->createNeighborhood();
             //cells->getNeighborhood()->setAllHSupportSizes(h_support);

             //LO neighbors_needed = Compadre::GMLS::getNP(Porder, input_dim /*dimension*/);

            //LO extra_neighbors = parameters->get<Teuchos::ParameterList>("remap").get<double>("neighbors needed multiplier") * neighbors_needed;
            //cells->getNeighborhood()->constructAllNeighborList(cells->getCoordsConst()->getHaloSize(), extra_neighbors);

        }

        Teuchos::RCP<Compadre::AnalyticFunction> velocity_function;
        Teuchos::RCP<Compadre::AnalyticFunction> pressure_function;
        if (parameters->get<Teuchos::ParameterList>("physics").get<std::string>("pressure solution")=="polynomial_1") {
            velocity_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::FirstOrderBasis(input_dim /*dimension*/)));
        } else if (parameters->get<Teuchos::ParameterList>("physics").get<std::string>("solution")=="polynomial_2") {
            velocity_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::SecondOrderBasis(input_dim /*dimension*/)));
        } else if (parameters->get<Teuchos::ParameterList>("physics").get<std::string>("solution")=="polynomial_3") {
            velocity_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::ThirdOrderBasis(input_dim /*dimension*/)));
        } else if (parameters->get<Teuchos::ParameterList>("physics").get<std::string>("solution")=="divfree_sinecos") {
            velocity_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::DivFreeSineCos(input_dim /*dimension*/)));
        } else if (parameters->get<Teuchos::ParameterList>("physics").get<std::string>("solution")=="divfree_polynomial_2") {
            velocity_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::DivFreeSecondOrderBasis(input_dim /*dimension*/)));
        } else if (parameters->get<Teuchos::ParameterList>("physics").get<std::string>("solution")=="sine") {
            velocity_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::SineProducts(input_dim /*dimension*/)));
        } else if (parameters->get<Teuchos::ParameterList>("physics").get<std::string>("solution")=="timoshenko") {
            velocity_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::Timoshenko(
                            parameters->get<Teuchos::ParameterList>("physics").get<double>("shear"),
                            parameters->get<Teuchos::ParameterList>("physics").get<double>("lambda"), 
                            input_dim /*dimension*/)));
        }
        if (st_op) { // mix_le_op uses pressure from divergence of velocity
            if (parameters->get<Teuchos::ParameterList>("physics").get<std::string>("pressure solution")=="polynomial_1") {
                pressure_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::FirstOrderBasis(input_dim /*dimension*/)));
            } else if (parameters->get<Teuchos::ParameterList>("physics").get<std::string>("pressure solution")=="polynomial_2") {
                pressure_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::SecondOrderBasis(input_dim /*dimension*/)));
            } else if (parameters->get<Teuchos::ParameterList>("physics").get<std::string>("pressure solution")=="polynomial_3") {
                pressure_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::ThirdOrderBasis(input_dim /*dimension*/)));
            } else if (parameters->get<Teuchos::ParameterList>("physics").get<std::string>("pressure solution")=="zero") {
                pressure_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::ConstantEachDimension(0, 0, 0)));
            } else if (parameters->get<Teuchos::ParameterList>("physics").get<std::string>("pressure solution")=="sine") {
                pressure_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::SineProducts(input_dim /*dimension*/)));
            }
        }

//        {
//            WriteTime->start();
//            std::string output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file");
//            std::string writetest_output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "writetest" + parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file");
//            fm.setWriter(output_filename, cells);
//            fm.write();
//            WriteTime->stop();
//        }
        //Teuchos::RCP<Compadre::ParticlesT> particles =
        //        Teuchos::rcp( new Compadre::ParticlesT(parameters, comm));
        //auto particle_coords = Teuchos::rcpFromRef<CT>(*(CT*)cells->getCoords());
        //particles->setCoords(particle_coords);

        //cells->getFieldManager()->createField(1, "solution", "m/s");
        //particles->createDOFManager();
        //particles->getDOFManager()->generateDOFMap();
        //{
        //    if (parameters->get<Teuchos::ParameterList>("halo").get<bool>("dynamic")) {
        //        halo_size = h_size * parameters->get<Teuchos::ParameterList>("halo").get<double>("multiplier");
        //    } else {
        //        halo_size = parameters->get<Teuchos::ParameterList>("halo").get<double>("size");
        //    }
         //    particles->buildHalo(halo_size);
         //    particles->createDOFManager();



        //    ////Set the radius for the neighbor list:
         //    //ST h_support;
         //    //if (parameters->get<Teuchos::ParameterList>("neighborhood").get<bool>("dynamic radius")) {
         //    //    h_support = h_size;
         //    //} else {
         //    //    h_support = parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("size");
         //    //}
         //    //particles->createNeighborhood();
         //    //particles->getNeighborhood()->setAllHSupportSizes(h_support);

         //    //LO neighbors_needed = Compadre::GMLS::getNP(Porder, input_dim /*dimension*/);

        //    //LO extra_neighbors = parameters->get<Teuchos::ParameterList>("remap").get<double>("neighbors needed multiplier") * neighbors_needed;
        //    //particles->getNeighborhood()->constructAllNeighborList(particles->getCoordsConst()->getHaloSize(), extra_neighbors);

        //}

        // Iterative solver for the problem
        Teuchos::RCP<Compadre::ProblemT> problem = Teuchos::rcp( new Compadre::ProblemT(particles));

        // construct physics, sources, and boundary conditions
        Teuchos::RCP<Compadre::ReactionDiffusionPhysics> physics =
          Teuchos::rcp( new Compadre::ReactionDiffusionPhysics(particles));
        Teuchos::RCP<Compadre::ReactionDiffusionSources> source =
            Teuchos::rcp( new Compadre::ReactionDiffusionSources(particles));
        Teuchos::RCP<Compadre::ReactionDiffusionBoundaryConditions> bcs =
            Teuchos::rcp( new Compadre::ReactionDiffusionBoundaryConditions(particles));


        physics->_velocity_basis_type_divfree = velocity_basis_type_divfree;
        physics->_velocity_basis_type_vector = velocity_basis_type_vector;
        physics->_use_vector_gmls = use_vector_gmls;
        physics->_use_vector_grad_gmls = use_vector_grad_gmls;

        // treatment of null space in pressure 
        physics->_use_pinning = use_pinning;
        physics->_use_lm = use_lm;

        physics->_use_sip = use_sip;
        physics->_use_vms = use_vms;

        physics->_use_neighbor_weighting = use_neighbor_weighting;
        physics->_use_side_weighting = use_side_weighting;

        physics->_velocity_name = velocity_name;
        physics->_pressure_name = pressure_name;

        physics->_velocity_function = velocity_function.getRawPtr();
        physics->_pressure_function = pressure_function.getRawPtr();

        physics->_l2_op = l2_op;
        physics->_rd_op = rd_op;
        physics->_le_op = le_op;
        physics->_vl_op = vl_op;
        physics->_st_op = st_op;
        physics->_mix_le_op = mix_le_op;

        physics->_velocity_field_id = particles->getFieldManagerConst()->getIDOfFieldFromName(velocity_name);
        if (st_op || mix_le_op) {
            physics->_pressure_field_id = particles->getFieldManagerConst()->getIDOfFieldFromName(pressure_name);
            if (use_lm) physics->_lagrange_field_id = particles->getFieldManagerConst()->getIDOfFieldFromName("lagrange multiplier");
        }

        // set physics, sources, and boundary conditions in the problem
        // set reaction and diffusion for physics
        double reaction_coeff, diffusion_coeff, shear_coeff, lambda_coeff;
        if (rd_op || st_op) {
            reaction_coeff = parameters->get<Teuchos::ParameterList>("physics").get<double>("reaction");
            diffusion_coeff = parameters->get<Teuchos::ParameterList>("physics").get<double>("diffusion");
            physics->setReaction(reaction_coeff);
            physics->setDiffusion(diffusion_coeff);
        } else if (le_op || vl_op || mix_le_op) {
            shear_coeff = parameters->get<Teuchos::ParameterList>("physics").get<double>("shear");
            lambda_coeff = parameters->get<Teuchos::ParameterList>("physics").get<double>("lambda");
            physics->setShear(shear_coeff);
            physics->setLambda(lambda_coeff);
        }

        physics->setCells(cells);
        problem->setPhysics(physics);
        source->setPhysics(physics);
        bcs->setPhysics(physics);

        problem->setSources(source);
        problem->setBCS(bcs);

        //bcs->flagBoundaries();

        // assembly
        AssemblyTime->start();
        problem->initialize();
        AssemblyTime->stop();

        //solving
        SolvingTime->start();
        if (input_dim==3) printf("Solve started.\n");
        problem->solve();

        //// test that gradient of vector does the same thing as gradient of scalar (repeated for each component)
        //{
        //auto neighborhood = physics->_cell_particles_neighborhood;
        //auto quadrature_weights = cells->getFieldManager()->getFieldByName("quadrature_weights")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
        //auto _vel_gmls = physics->_vel_gmls;
        //for(int i=0; i<coords->nLocal(); i++){
        //    LO num_neighbors = neighborhood->getNumNeighbors(i);
        //    for(int j_comp_out=0; j_comp_out<2; j_comp_out++){
        //        auto j_comp_in = j_comp_out;
        //        for(int k=0; k<2; k++){
        //            for (LO qn = 0; qn < quadrature_weights.extent(1); qn++) {
        //                // loop over particles neighbor to the cell
        //                for (LO l = 0; l < num_neighbors; l++) {
        //                    auto grad_v_k = _vel_gmls->getSolutionSetHost()->getAlpha1TensorTo2Tensor(TargetOperation::GradientOfVectorPointEvaluation, i, j_comp_out, k, l, j_comp_in, qn+1);
        //                    auto old_grad_v_k = _vel_gmls->getSolutionSetHost()->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, k, l, qn+1);
        //                    auto grad_diff = abs(grad_v_k - old_grad_v_k);
        //                    if (grad_diff>1e-14) printf("%d, Different %.16f vs. %.16f\n", j_comp_out, grad_v_k, old_grad_v_k);
        //                    //if (grad_diff<1e-14 && abs(grad_v_k)>0) printf("%d, %d, %d, Different %.16f vs. %.16f\n", j_comp_out, k, l, grad_v_k, old_grad_v_k);
        //                }
        //            }
        //        }
        //    }

        //}
        //}
        SolvingTime->stop();
        particles->getFieldManager()->updateFieldsHaloData();


        if (input_dim==3) printf("Exact solution started.\n");
        double avg_pressure_computed = 0.0;
        // post process solution ( modal DOF -> cell centered value )
        {
            if (st_op || mix_le_op) { 
                cells->getFieldManager()->createField(input_dim, "processed_"+velocity_name, "m/s");
                cells->getFieldManager()->createField(1, "processed_"+pressure_name, "m/s");
                cells->getFieldManager()->createField(input_dim, "diff_"+velocity_name, "m/s");
                cells->getFieldManager()->createField(1, "diff_"+pressure_name, "m/s");
            } else if (le_op || vl_op) { 
                cells->getFieldManager()->createField(input_dim, "processed_"+velocity_name, "m/s");
                cells->getFieldManager()->createField(input_dim, "diff_"+velocity_name, "m/s");
            } else {
                cells->getFieldManager()->createField(1, "processed_"+velocity_name, "m/s");
                cells->getFieldManager()->createField(1, "diff_"+velocity_name, "m/s");
            }
        }
        auto velocity_processed_view = cells->getFieldManager()->getFieldByName("processed_"+velocity_name)->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
        decltype(velocity_processed_view) pressure_processed_view;
        if (st_op || mix_le_op) {
            pressure_processed_view = cells->getFieldManager()->getFieldByName("processed_"+pressure_name)->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
        }
        {
            auto vel_gmls = physics->_vel_gmls;
            auto pressure_gmls = physics->_pressure_gmls;

            auto velocity_dof_view = particles->getFieldManager()->getFieldByName(velocity_name)->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
            auto velocity_halo_dof_view = particles->getFieldManager()->getFieldByName(velocity_name)->getHaloMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);

            decltype(velocity_dof_view) pressure_dof_view;
            decltype(velocity_halo_dof_view) pressure_halo_dof_view;
            if (st_op || mix_le_op) {
                pressure_dof_view = particles->getFieldManager()->getFieldByName(pressure_name)->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
                pressure_halo_dof_view = particles->getFieldManager()->getFieldByName(pressure_name)->getHaloMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
            }

            auto neighborhood = physics->_cell_particles_neighborhood;
            auto halo_neighborhood = physics->_halo_cell_particles_neighborhood;

            auto nlocal_cells = coords->nLocal();
            auto nlocal_particles = particles->getCoords()->nLocal();
            for(int j=0; j<nlocal_cells; j++){
                LO num_neighbors = neighborhood->getNumNeighbors(j);
                // loop over particles neighbor to the cell
                for (LO l = 0; l < num_neighbors; l++) {
                    for (LO k_out = 0; k_out < velocity_processed_view.extent(1); ++k_out) {
                    for (LO k_in = 0; k_in < velocity_processed_view.extent(1); ++k_in) {
                        if (k_out!=k_in && !velocity_basis_type_vector) continue;
                        auto particle_l = neighborhood->getNeighbor(j,l);
                        double v;
                        if (use_vector_gmls) v = vel_gmls->getSolutionSetHost()->getAlpha1TensorTo1Tensor(TargetOperation::VectorPointEvaluation, j, k_out, l, k_in, 0);
                        else v = vel_gmls->getSolutionSetHost()->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, j, l, 0);
                        auto dof_val = (particle_l<nlocal_particles) ? velocity_dof_view(particle_l,k_in) : velocity_halo_dof_view(particle_l-nlocal_particles,k_in);
                        velocity_processed_view(j,k_out) += dof_val * v;
                    }
                    }
                    if (st_op || mix_le_op) {
                        for (LO k = 0; k < pressure_processed_view.extent(1); ++k) {
                            auto particle_l = neighborhood->getNeighbor(j,l);
                            auto v = pressure_gmls->getSolutionSetHost()->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, j, l, 0);
                            auto dof_val = (particle_l<nlocal_particles) ? pressure_dof_view(particle_l,k) : pressure_halo_dof_view(particle_l-nlocal_particles,k);
                            // assumes pressure has same basis as velocity
                            pressure_processed_view(j,k) += dof_val * v;
                        }
                    }
                }
                if (st_op || mix_le_op) avg_pressure_computed += pressure_processed_view(j,0);
            }
            if (st_op || mix_le_op) {
                avg_pressure_computed /= nlocal_cells;
                for(int j=0; j<nlocal_cells; j++){
                    for (LO k = 0; k < pressure_processed_view.extent(1); ++k) {
                        pressure_processed_view(j,k) -= avg_pressure_computed;
                    }
                }
            }
        }

        //// DIAGNOSTIC:: serial only checking of basis function near (1,1)
        //{
        //    // to visualize basis function
        //    // first pick a DOF
        //    // then get all cells in that DOFs neighborhood
        //    // then loop over a cell 
        //    // then loop over quadrature point
        //    auto gmls = physics->_vel_gmls;
        //    auto quadrature_points = physics->_cells->getFieldManager()->getFieldByName("quadrature_points")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
        //    auto quadrature_weights = physics->_cells->getFieldManager()->getFieldByName("quadrature_weights")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
        //    auto quadrature_type = cells->getFieldManager()->getFieldByName("interior")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
        //    auto dof_view = cells->getFieldManager()->getFieldByName("solution")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
        //    auto neighborhood = physics->_cell_particles_neighborhood;
        //    auto nlocal = coords->nLocal();

        //    int DOF = 0;
        //    for(int p=0; p<nlocal; p++){
        //        xyz_type xyz = coords->getLocalCoords(p);
        //        if (((xyz.x-1)*(xyz.x-1) + (xyz.y-1)*(xyz.y-1)) < 1e-2) DOF = p;
        //    }
        //    LO num_neighbors = neighborhood->getNumNeighbors(DOF);
        //    for (LO l = 0; l < num_neighbors; l++) {
        //        auto neighbor_l = neighborhood->getNeighbor(DOF,l);
        //        for (LO q = 0; q < quadrature_weights.extent(1); q++) {
        //            if (((int)quadrature_type(neighbor_l,q))==1) {
        //                std::cout << quadrature_points(neighbor_l, 2*q+0) << " " << quadrature_points(neighbor_l, 2*q+1) << " ";
        //                int DOF_to_l = -1;
        //                for (LO z = 0; z < neighborhood->getNumNeighbors(neighbor_l); ++z) {
        //                    if (neighborhood->getNeighbor(neighbor_l,z)==DOF) {
        //                        DOF_to_l = z;
        //                        break;
        //                    }
        //                }
        //                auto v = gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, neighbor_l, DOF_to_l, q+1);
        //                std::cout << v << std::endl;
        //            }
        //        }
        //    }
        //}
   
        ////// DIAGNOSTIC:: get solution at quadrature pts
        //if (comm->getSize()==1) {
        //    auto quadrature_points = cells->getFieldManager()->getFieldByName("quadrature_points")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
        //    auto quadrature_weights = cells->getFieldManager()->getFieldByName("quadrature_weights")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
        //    auto quadrature_type = cells->getFieldManager()->getFieldByName("interior")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
        //    auto neighborhood = physics->_cell_particles_neighborhood;
        //    auto halo_neighborhood = physics->_halo_cell_particles_neighborhood;
        //    auto dof_view = cells->getFieldManager()->getFieldByName("solution")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
        //    auto halo_dof_view = cells->getFieldManager()->getFieldByName("solution")->getHaloMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
        //    auto gmls = physics->_vel_gmls;
        //    Teuchos::RCP<Compadre::ParticlesT> particles_new =
        //        Teuchos::rcp( new Compadre::ParticlesT(parameters, comm, parameters->get<Teuchos::ParameterList>("io").get<local_index_type>("input dimensions")));
        //    CT* new_coords = (CT*)particles_new->getCoords();
        //    std::vector<Compadre::XyzVector> verts_to_insert;
        //    // put real quadrature points here
        //    for( int j =0; j<coords->nLocal(); j++){
        //        for (int i=0; i<quadrature_weights.extent(1); ++i) {
        //            verts_to_insert.push_back(Compadre::XyzVector(quadrature_points(j,2*i+0), quadrature_points(j,2*i+1),0));
        //        }
        //    }
        //    //verts_to_insert.push_back(Compadre::XyzVector(0,0,0));
        //    //verts_to_insert.push_back(Compadre::XyzVector(0,1,0));
        //    //verts_to_insert.push_back(Compadre::XyzVector(1,1,0));
        //    new_coords->insertCoords(verts_to_insert);
        //    particles_new->resetWithSameCoords(); // must be called because particles doesn't know about coordinate insertions


        //    auto nlocal = coords->nLocal();
        //    particles_new->getFieldManager()->createField(1, "cell", "m/s");
        //    auto cell_id = particles_new->getFieldManager()->getFieldByName("cell")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
        //    int count = 0;
        //    for( int j =0; j<coords->nLocal(); j++){
        //        for (int i=0; i<quadrature_weights.extent(1); ++i) {
        //            double quad_val = 0;
        //            // needs reconstruction at this quadrature point
        //            LO num_neighbors = neighborhood->getNumNeighbors(j);
        //            // loop over particles neighbor to the cell
        //            for (LO l = 0; l < num_neighbors; l++) {
        //                auto particle_l = neighborhood->getNeighbor(j,l);
        //                auto dof_val = (particle_l<nlocal) ? dof_view(particle_l,0) : halo_dof_view(particle_l-nlocal,0);
        //                auto v = gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, j, l, i+1);
        //                quad_val += dof_val * v;
        //            }
        //            
        //            cell_id(count,0) = quad_val;
        //            count++;
        //        }
        //    }
        //    {
        //        Compadre::FileManager fm3;
        //        std::string output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "new_particles.nc";
        //        fm3.setWriter(output_filename, particles_new);
        //        fm3.write();
        //    }
        //}

        //// DIAGNOSTIC:: serial only checking of quadrature points
        //if (comm->getSize()==1) {
        //    auto quadrature_points = cells->getFieldManager()->getFieldByName("quadrature_points")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
        //    auto quadrature_weights = cells->getFieldManager()->getFieldByName("quadrature_weights")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
        //    auto quadrature_type = cells->getFieldManager()->getFieldByName("interior")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
        //    Teuchos::RCP<Compadre::ParticlesT> particles_new =
        //        Teuchos::rcp( new Compadre::ParticlesT(parameters, comm, parameters->get<Teuchos::ParameterList>("io").get<local_index_type>("input dimensions")));
        //    CT* new_coords = (CT*)particles_new->getCoords();
        //    std::vector<Compadre::XyzVector> verts_to_insert;
        //    // put real quadrature points here
        //    for( int j =0; j<coords->nLocal(); j++){
        //        for (int i=0; i<quadrature_weights.extent(1); ++i) {
        //            if (quadrature_type(j,i)==1) { // interior
        //                verts_to_insert.push_back(Compadre::XyzVector(quadrature_points(j,2*i+0), quadrature_points(j,2*i+1),0));
        //            }
        //        }
        //    }
        //    //verts_to_insert.push_back(Compadre::XyzVector(0,0,0));
        //    //verts_to_insert.push_back(Compadre::XyzVector(0,1,0));
        //    //verts_to_insert.push_back(Compadre::XyzVector(1,1,0));
        //    new_coords->insertCoords(verts_to_insert);
        //    particles_new->resetWithSameCoords(); // must be called because particles doesn't know about coordinate insertions


        //    particles_new->getFieldManager()->createField(1, "cell", "m/s");
        //    auto cell_id = particles_new->getFieldManager()->getFieldByName("cell")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
        //    int count = 0;
        //    for( int j =0; j<coords->nLocal(); j++){
        //        for (int i=0; i<quadrature_weights.extent(1); ++i) {
        //            if (quadrature_type(j,i)==1) { // interior
        //                cell_id(count,0) = j;
        //                count++;
        //            }
        //        }
        //    }
        //    {
        //        Compadre::FileManager fm3;
        //        std::string output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "new_particles.nc";
        //        fm3.setWriter(output_filename, particles_new);
        //        fm3.write();
        //    }
        //}

        // get the quadrature points for the first element
        // and insert these into another particle set, then print that particle set


        //Teuchos::RCP<Compadre::RemapManager> rm = Teuchos::rcp(new Compadre::RemapManager(parameters, particles.getRawPtr(), cells.getRawPtr(), halo_size));
        //Compadre::RemapObject ro("solution", "another_solution", TargetOperation::ScalarPointEvaluation);
        //rm->add(ro);
        //STACK_TRACE(rm->execute());
//
//            // write matrix and rhs to files
////                Tpetra::MatrixMarket::Writer<Compadre::mvec_type>::writeDenseFile("b"+std::to_string(i), *problem->getb(), "rhs", "description");
////                Tpetra::MatrixMarket::Writer<Compadre::crs_matrix_type>::writeSparseFile("A"+std::to_string(i), problem->getA(), "A", "description");
//        }
//
//
//        Teuchos::RCP< Compadre::FieldT > PField = particles->getFieldManagerConst()->getFieldByID(0);
//
        // check solution
        double velocity_norm = 0.0;
        //double pressure_norm = 0.0;
        double velocity_exact = 0.0;
        double pressure_exact = 0.0;
        double avg_pressure_exact = 0.0;

        //auto penalty = (parameters->get<Teuchos::ParameterList>("remap").get<int>("porder")+1)*parameters->get<Teuchos::ParameterList>("physics").get<double>("penalty")/physics->_cell_particles_neighborhood->computeMaxHSupportSize(true /* global processor max */);
        //// jump errors
        //double jump_error = 0.0;
        //if (penalty > 0) 
        //{
        //    auto gmls = physics->_vel_gmls;
        //    cells->getFieldManager()->createField(1, "processed_solution", "m/s");
        //    auto processed_view = cells->getFieldManager()->getFieldByName("processed_solution")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
        //    auto dof_view = cells->getFieldManager()->getFieldByName("solution")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
        //    auto halo_dof_view = cells->getFieldManager()->getFieldByName("solution")->getHaloMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
        //    auto neighborhood = physics->_cell_particles_neighborhood;
        //    auto halo_neighborhood = physics->_halo_cell_particles_neighborhood;

        //    auto quadrature_points = physics->_cells->getFieldManager()->getFieldByName("quadrature_points")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
        //    auto quadrature_weights = physics->_cells->getFieldManager()->getFieldByName("quadrature_weights")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
        //    auto quadrature_type = physics->_cells->getFieldManager()->getFieldByName("interior")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
        //    auto unit_normals = physics->_cells->getFieldManager()->getFieldByName("unit_normal")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
        //    auto adjacent_elements = physics->_cells->getFieldManager()->getFieldByName("adjacent_elements")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);

        //    auto halo_quadrature_points = physics->_cells->getFieldManager()->getFieldByName("quadrature_points")->getHaloMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
        //    auto halo_quadrature_weights = physics->_cells->getFieldManager()->getFieldByName("quadrature_weights")->getHaloMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
        //    auto halo_quadrature_type = physics->_cells->getFieldManager()->getFieldByName("interior")->getHaloMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
        //    auto halo_unit_normals = physics->_cells->getFieldManager()->getFieldByName("unit_normal")->getHaloMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
        //    auto halo_adjacent_elements = physics->_cells->getFieldManager()->getFieldByName("adjacent_elements")->getHaloMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);

        //    auto nlocal = coords->nLocal();
        //    // loop over cells
        //    for(int j=0; j<nlocal; j++){
        //        LO num_neighbors = neighborhood->getNumNeighbors(j);
        //        // loop over particles neighbor to the cell
        //        for (LO q = 0; q < quadrature_weights.extent(1); q++) {
        //            if (quadrature_type(j,q)==0 || quadrature_type(j,q)==2) {
        //                double val_at_quad = 0.0;
        //                for (LO l = 0; l < num_neighbors; l++) {
        //                    auto particle_l = neighborhood->getNeighbor(j,l);
        //                    auto dof_val = (particle_l<nlocal) ? dof_view(particle_l,0) : halo_dof_view(particle_l-nlocal,q+1);
        //                    auto v = gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, j, l, q+1);
        //                    val += dof_val * v;
        //                }
        //            }
        //        }
        //    }
        //}
 
        if (st_op || mix_le_op) { 
            cells->getFieldManager()->createField(input_dim, "exact_"+velocity_name, "m/s");
            cells->getFieldManager()->createField(1, "exact_"+pressure_name, "m/s");
        } else if (le_op || vl_op) { 
            cells->getFieldManager()->createField(input_dim, "exact_"+velocity_name, "m/s");
        } else {
            cells->getFieldManager()->createField(1, "exact_"+velocity_name, "m/s");
        }

        auto velocity_exact_view = cells->getFieldManager()->getFieldByName("exact_"+velocity_name)->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
        auto velocity_diff_view = cells->getFieldManager()->getFieldByName("diff_"+velocity_name)->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
        decltype(velocity_exact_view) pressure_exact_view;
        if (st_op || mix_le_op) pressure_exact_view = cells->getFieldManager()->getFieldByName("exact_"+pressure_name)->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
        decltype(velocity_exact_view) pressure_diff_view;
        if (st_op || mix_le_op) pressure_diff_view = cells->getFieldManager()->getFieldByName("diff_"+pressure_name)->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);

        // store exact velocity as a field
        if (le_op || vl_op || st_op || mix_le_op) { 
            for( int j =0; j<coords->nLocal(); j++){
                for( int k =0; k<velocity_exact_view.extent(1); k++){
                    xyz_type xyz = coords->getLocalCoords(j);
                    auto exacts = velocity_function->evalVector(xyz);
                    velocity_exact_view(j,k) = exacts[k];
                    velocity_diff_view(j,k) = velocity_exact_view(j,k)-velocity_processed_view(j,k);
                }
            }
        } else {
            for( int j =0; j<coords->nLocal(); j++){
                xyz_type xyz = coords->getLocalCoords(j);
                velocity_exact = velocity_function->evalScalar(xyz);
                velocity_exact_view(j,0) = velocity_exact;
                velocity_diff_view(j,0) = velocity_exact_view(j,0)-velocity_processed_view(j,0);
            }
        }
        // store exact pressure as a field
        if (mix_le_op) {
            double t_lambda_coeff = (lambda_coeff==std::numeric_limits<scalar_type>::infinity()) ? 0.0 : lambda_coeff;
            for( int j =0; j<coords->nLocal(); j++){
                xyz_type xyz = coords->getLocalCoords(j);
                // pressure is div of velocity
                auto velocity_jacobian = velocity_function->evalJacobian(xyz);
                pressure_exact = 1*(t_lambda_coeff+shear_coeff)*(velocity_jacobian[0][0] + velocity_jacobian[1][1]);
                avg_pressure_exact += pressure_exact;
            }

            avg_pressure_exact /= (double)(coords->nLocal());

            for( int j =0; j<coords->nLocal(); j++){
                xyz_type xyz = coords->getLocalCoords(j);
                // pressure is div of velocity
                auto velocity_jacobian = velocity_function->evalJacobian(xyz);
                pressure_exact = 1*(t_lambda_coeff+shear_coeff)*(velocity_jacobian[0][0] + velocity_jacobian[1][1]);
                pressure_exact_view(j,0) = pressure_exact-avg_pressure_exact;
                pressure_diff_view(j,0) = pressure_exact_view(j,0) - pressure_processed_view(j,0);
            }
        } else if (st_op) {
            for( int j =0; j<coords->nLocal(); j++){
                xyz_type xyz = coords->getLocalCoords(j);
                pressure_exact = pressure_function->evalScalar(xyz);
                avg_pressure_exact += pressure_exact;
            }

            avg_pressure_exact /= (double)(coords->nLocal());

            for( int j =0; j<coords->nLocal(); j++){
                xyz_type xyz = coords->getLocalCoords(j);
                pressure_exact = pressure_function->evalScalar(xyz);
                pressure_exact_view(j,0) = pressure_exact-avg_pressure_exact;
                pressure_diff_view(j,0) = pressure_exact_view(j,0) - pressure_processed_view(j,0);
            }
        }

        Teuchos::RCP<Compadre::ParticlesT> particles_new;
        //// DIAGNOSTIC:: get solution at quadrature pts
        if (plot_quadrature) {
            auto quadrature_points = cells->getFieldManager()->getFieldByName("quadrature_points")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
            auto quadrature_weights = cells->getFieldManager()->getFieldByName("quadrature_weights")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
            auto quadrature_type = cells->getFieldManager()->getFieldByName("interior")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
            auto neighborhood = physics->_cell_particles_neighborhood;
            auto halo_neighborhood = physics->_halo_cell_particles_neighborhood;
            auto dof_view = particles->getFieldManager()->getFieldByName(velocity_name)->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
            auto halo_dof_view = particles->getFieldManager()->getFieldByName(velocity_name)->getHaloMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
            auto gmls = physics->_vel_gmls;
            particles_new =
                Teuchos::rcp( new Compadre::ParticlesT(parameters, comm, input_dim));
            CT* new_coords = (CT*)particles_new->getCoords();
            std::vector<Compadre::XyzVector> verts_to_insert;
            // put real quadrature points here
            for( int j =0; j<coords->nLocal(); j++){
                for (int i=0; i<quadrature_weights.extent(1); ++i) {
                    if (input_dim==2) {
                        verts_to_insert.push_back(Compadre::XyzVector(quadrature_points(j,input_dim*i+0), quadrature_points(j,input_dim*i+1), 0));
                    } else {
                        verts_to_insert.push_back(Compadre::XyzVector(quadrature_points(j,input_dim*i+0), quadrature_points(j,input_dim*i+1), quadrature_points(j,input_dim*i+2)));
                    }
                }
            }
            new_coords->insertCoords(verts_to_insert);
            particles_new->resetWithSameCoords(); // must be called because particles doesn't know about coordinate insertions

            particles_new->getFieldManager()->createField(input_dim, "l2", "m/s");
            particles_new->getFieldManager()->createField(input_dim, "h1", "m/s");
            particles_new->getFieldManager()->createField(input_dim, "jump", "m/s");
        }

        if (l2_op) {
            for( int j =0; j<coords->nLocal(); j++){
                xyz_type xyz = coords->getLocalCoords(j);
                const ST val = cells->getFieldManagerConst()->getFieldByName("processed_"+velocity_name)->getLocalScalarVal(j);
                //const ST val = dof_view(j,0);
                //exact = 1;//+xyz[0]+xyz[1];//function->evalScalar(xyz);
                velocity_exact = velocity_function->evalScalar(xyz);
                velocity_norm += (velocity_exact - val)*(velocity_exact-val);
            }
            velocity_norm /= (double)(coords->nGlobalMax());
            printf("L2: %.5e\n", sqrt(velocity_norm));
        } else {
            // get error from l2, h1, and jump

            double velocity_jump_error = 0.0;
            double velocity_l2_error = 0.0;
            double velocity_h1_error = 0.0;

            //double pressure_jump_error = 0.0;
            double pressure_l2_error = 0.0;
            //double pressure_h1_error = 0.0;

            auto base_penalty = (parameters->get<Teuchos::ParameterList>("remap").get<int>("porder")+1)*parameters->get<Teuchos::ParameterList>("physics").get<double>("penalty");
            if (use_neighbor_weighting) base_penalty /= (input_dim==2) ? physics->_cell_particles_max_h : 1.772453850905516*physics->_cell_particles_max_h; // either h or sqrt(pi)*h

            host_view_type l2_view, h1_view, jump_view;
            if (plot_quadrature) {
                l2_view = particles_new->getFieldManager()->getFieldByName("l2")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
                h1_view = particles_new->getFieldManager()->getFieldByName("h1")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
                jump_view = particles_new->getFieldManager()->getFieldByName("jump")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
            }

            {

                auto quadrature_points = cells->getFieldManager()->getFieldByName("quadrature_points")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
                auto quadrature_weights = cells->getFieldManager()->getFieldByName("quadrature_weights")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
                auto quadrature_type = cells->getFieldManager()->getFieldByName("interior")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
                auto neighborhood = physics->_cell_particles_neighborhood;
                auto halo_neighborhood = physics->_halo_cell_particles_neighborhood;

                auto velocity_dof_view = particles->getFieldManager()->getFieldByName(velocity_name)->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
                auto velocity_halo_dof_view = particles->getFieldManager()->getFieldByName(velocity_name)->getHaloMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);

                decltype(velocity_dof_view) pressure_dof_view;
                decltype(velocity_halo_dof_view) pressure_halo_dof_view;
                if (st_op || mix_le_op) {
                    pressure_dof_view = particles->getFieldManager()->getFieldByName(pressure_name)->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
                    pressure_halo_dof_view = particles->getFieldManager()->getFieldByName(pressure_name)->getHaloMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
                }

                auto adjacent_elements = cells->getFieldManager()->getFieldByName("adjacent_elements")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
                auto vel_gmls = physics->_vel_gmls;
                auto pressure_gmls = physics->_pressure_gmls;
                auto num_edges = adjacent_elements.extent(1);
                auto num_interior_quadrature = 0;
                for (int q=0; q<quadrature_weights.extent(1); ++q) {
                    if (quadrature_type(0,q)!=1) { // some type of edge quadrature
                        num_interior_quadrature = q;
                        break;
                    }
                }
                auto num_exterior_quadrature_per_edge = (quadrature_weights.extent(1) - num_interior_quadrature)/num_edges;
                auto nlocal_cells = coords->nLocal();
                auto nlocal_particles = particles->getCoords()->nLocal();
                for( int j =0; j<coords->nLocal(); j++){
                    auto num_sides = adjacent_elements.extent(1);
                    std::vector<double> side_lengths(num_sides);
                    {
                        for (int current_side_num = 0; current_side_num < num_sides; ++current_side_num) {
                           for (int q=0; q<quadrature_weights.extent(1); ++q) {
                                auto q_type = quadrature_type(j,q);
                                if (q_type!=1) { // side
                                    auto q_wt = quadrature_weights(j,q);
                                    int side_num = (q - num_interior_quadrature)/num_exterior_quadrature_per_edge;
                                    if (side_num==current_side_num) {
                                        side_lengths[current_side_num] += q_wt;
                                    }
                                }
                            }
                        }
                    }
                    double l2_error_on_cell = 0.0;
                    double h1_error_on_cell = 0.0;
                    double jump_error_on_cell = 0.0;
                    int count = quadrature_weights.extent(1) * j;
                    for (int m_out=0; m_out<velocity_dof_view.extent(1); ++m_out) {
                        for (int i=0; i<quadrature_weights.extent(1); ++i) {
                            if (quadrature_type(j,i)==1) { // interior
                                double l2_val = 0.0;
                                XYZ h1_val;
                                // needs reconstruction at this quadrature point
                                LO num_neighbors = neighborhood->getNumNeighbors(j);
                                // loop over particles neighbor to the cell
                                for (int m_in=0; m_in<velocity_dof_view.extent(1); ++m_in) {
                                    if (m_out!=m_in && !velocity_basis_type_vector) continue;
                                    for (LO l = 0; l < num_neighbors; l++) {
                                        auto particle_l = neighborhood->getNeighbor(j,l);
                                        auto dof_val = (particle_l<nlocal_particles) ? velocity_dof_view(particle_l,m_in) : velocity_halo_dof_view(particle_l-nlocal_particles,m_in);
                                        double v, v_x, v_y;
                                        XYZ v_grad;
                                        if (use_vector_gmls) v = vel_gmls->getSolutionSetHost()->getAlpha1TensorTo1Tensor(TargetOperation::VectorPointEvaluation, j, m_out, l, m_in, i+1);
                                        else v = vel_gmls->getSolutionSetHost()->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, j, l, i+1);
                                        if (use_vector_grad_gmls) {
                                            for (int d=0; d<input_dim; ++d) {
                                                v_grad[d] = vel_gmls->getSolutionSetHost()->getAlpha1TensorTo2Tensor(TargetOperation::GradientOfVectorPointEvaluation, j, m_out, d, l, m_in, i+1);
                                            }
                                        } else {
                                            for (int d=0; d<input_dim; ++d) {
                                                v_grad[d] = vel_gmls->getSolutionSetHost()->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, j, d, l, i+1);
                                            }
                                        }
                                        l2_val += dof_val * v;
                                        for (int d=0; d<input_dim; ++d) {
                                            h1_val[d] += dof_val * v_grad[d];
                                        }
                                    }
                                }
                                auto xyz = Compadre::XyzVector(quadrature_points(j,input_dim*i+0), quadrature_points(j,input_dim*i+1), (input_dim==3) ? quadrature_points(j,input_dim*i+2): 0);
                                if (rd_op) {
                                    double l2_exact = velocity_function->evalScalar(xyz);
                                    xyz_type h1_exact = velocity_function->evalScalarDerivative(xyz);
                                    l2_error_on_cell += quadrature_weights(j,i) * (l2_val - l2_exact) * (l2_val - l2_exact);
                                    for (int d=0; d<input_dim; ++d) {
                                        h1_error_on_cell += quadrature_weights(j,i) * (h1_val[d] - h1_exact[d]) * (h1_val[d] - h1_exact[d]);
                                    }
                                } else if (le_op || vl_op || mix_le_op) {
                                    double l2_exact = velocity_function->evalVector(xyz)[m_out];
                                    xyz_type h1_exact = velocity_function->evalJacobian(xyz)[m_out];
                                    // still needs customized to LE stress tensor
                                    l2_error_on_cell += quadrature_weights(j,i) * (l2_val - l2_exact) * (l2_val - l2_exact);
                                    for (int d=0; d<input_dim; ++d) {
                                        h1_error_on_cell += quadrature_weights(j,i) * (h1_val[d] - h1_exact[d]) * (h1_val[d] - h1_exact[d]);
                                    }
                                    if (plot_quadrature) {
                                        // only for velocity
                                        l2_view(count+i,m_out) += quadrature_weights(j,i) * (l2_val - l2_exact) * (l2_val - l2_exact);
                                        for (int d=0; d<input_dim; ++d) {
                                            h1_view(count+i,m_out) += quadrature_weights(j,i) * (h1_val[d] - h1_exact[d]) * (h1_val[d] - h1_exact[d]);
                                        }
                                    }
                                } else if (st_op) {
                                    double l2_exact = velocity_function->evalVector(xyz)[m_out];
                                    xyz_type h1_exact = velocity_function->evalJacobian(xyz)[m_out];
                                    // still needs customized to LE stress tensor
                                    l2_error_on_cell += quadrature_weights(j,i) * (l2_val - l2_exact) * (l2_val - l2_exact);
                                    for (int d=0; d<input_dim; ++d) {
                                        h1_error_on_cell += quadrature_weights(j,i) * (h1_val[d] - h1_exact[d]) * (h1_val[d] - h1_exact[d]);
                                    }
                                }
                            } else if (quadrature_type(j,i)==2) { // exterior edge
                                int current_side_num = (i - num_interior_quadrature)/num_exterior_quadrature_per_edge;
                                double penalty = (use_side_weighting) ? base_penalty/side_lengths[current_side_num] : base_penalty;

                                double u_val = 0.0;
                                // needs reconstruction at this quadrature point
                                LO num_neighbors = neighborhood->getNumNeighbors(j);
                                // loop over particles neighbor to the cell
                                for (int m_in=0; m_in<velocity_dof_view.extent(1); ++m_in) {
                                    if (m_out!=m_in && !velocity_basis_type_vector) continue;
                                    for (LO l = 0; l < num_neighbors; l++) {
                                        auto particle_l = neighborhood->getNeighbor(j,l);
                                        auto dof_val = (particle_l<nlocal_particles) ? velocity_dof_view(particle_l,m_in) : velocity_halo_dof_view(particle_l-nlocal_particles,m_in);
                                        double v;
                                        if (use_vector_gmls) v = vel_gmls->getSolutionSetHost()->getAlpha1TensorTo1Tensor(TargetOperation::VectorPointEvaluation, j, m_out, l, m_in, i+1);
                                        else v = vel_gmls->getSolutionSetHost()->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, j, l, i+1);
                                        u_val += dof_val * v;
                                    }
                                }
                                auto xyz = Compadre::XyzVector(quadrature_points(j,input_dim*i+0), quadrature_points(j,input_dim*i+1), (input_dim==3) ? quadrature_points(j,input_dim*i+2): 0);
                                double u_exact = 0;
                                if (rd_op) {
                                    u_exact = velocity_function->evalScalar(xyz);
                                } else {
                                    u_exact = velocity_function->evalVector(xyz)[m_out];
                                }


				// VMS "jump" error norm (instead of sipg penalty)
                                if (use_vms){
                                    // VMSDG tau (penalty-like) parameter
                                    // in this script, i is the cubature point number (instead of q or qn)
                                    // in this script, input_dim (instead of _ndim_requested)
                                    //const int current_edge_num_in_current_cell = (i - num_interior_quadrature)/num_exterior_quadrature_per_edge; //should be same as "current_side_num"
                                    Teuchos :: SerialDenseMatrix <local_index_type, scalar_type> tau_edge(input_dim, input_dim);
                                    Teuchos :: SerialDenseSolver <local_index_type, scalar_type> vmsdg_solver;
				    tau_edge.putScalar(0.0);

				    for (int j1 = 0; j1 < input_dim; ++j1) {
				        for (int j2 = 0; j2 < input_dim; ++j2) {
				            tau_edge(j1, j2) = physics->_tau(j, current_side_num, j1, j2); //tau^(1)_s from 'current' elem
				        }
				    }
                  
                                    vmsdg_solver.setMatrix(Teuchos::rcp(&tau_edge, false));
                                    auto info = vmsdg_solver.invert();

				    double tau_times_jump_u = 0.0;
				    for (int m_temp=0; m_temp<velocity_dof_view.extent(1); ++m_temp){
                                        double temp_u_val = 0.0;
                                        // needs reconstruction at this quadrature point // replace m_out with m_temp in this code block, m_temp is index for dot product with tau_edge
                                        // LO num_neighbors = neighborhood->getNumNeighbors(j); //num_neighbors already defined in this scope
                                        // loop over particles neighbor to the cell
                                        for (int m_in=0; m_in<velocity_dof_view.extent(1); ++m_in) {
                                            if (m_temp!=m_in && !velocity_basis_type_vector) continue;
                                            for (LO l = 0; l < num_neighbors; l++) {
                                                auto particle_l = neighborhood->getNeighbor(j,l);
                                                auto dof_val = (particle_l<nlocal_particles) ? velocity_dof_view(particle_l,m_in) : velocity_halo_dof_view(particle_l-nlocal_particles,m_in);
                                                double v;
                                                if (use_vector_gmls) v = vel_gmls->getSolutionSetHost()->getAlpha1TensorTo1Tensor(TargetOperation::VectorPointEvaluation, j, m_temp, l, m_in, i+1);
                                                else v = vel_gmls->getSolutionSetHost()->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, j, l, i+1);
                                                temp_u_val += dof_val * v;
                                            }
				        }
                                        double temp_u_exact = 0.0;
                                        temp_u_exact = velocity_function->evalVector(xyz)[m_temp]; // NOTE: reconstruction uses index m_temp because contraction with tau_edge
                                        tau_times_jump_u += tau_edge(m_out, m_temp) * (temp_u_val - temp_u_exact);
				    }

				    jump_error_on_cell += quadrature_weights(j,i) * (u_val - u_exact) * tau_times_jump_u;
				} else { // sipg by default
                                    jump_error_on_cell += penalty * quadrature_weights(j,i) * (u_val - u_exact) * (u_val - u_exact);
				}
				if (plot_quadrature) {
                                    jump_view(count+i,m_out) += penalty * quadrature_weights(j,i) * (u_val - u_exact) * (u_val - u_exact);
                                }
                            } else if (quadrature_type(j,i)==0) { // interior edge
                                int current_side_num = (i - num_interior_quadrature)/num_exterior_quadrature_per_edge;
                                double penalty = (use_side_weighting) ? base_penalty/side_lengths[current_side_num] : base_penalty;
                                double u_val = 0.0;
                                double other_u_val = 0.0;
                                // needs reconstruction at this quadrature point
                                LO num_neighbors = neighborhood->getNumNeighbors(j);
                                // loop over particles neighbor to the cell
                                for (int m_in=0; m_in<velocity_dof_view.extent(1); ++m_in) {
                                    if (m_out!=m_in && !velocity_basis_type_vector) continue;
                                    for (LO l = 0; l < num_neighbors; l++) {
                                        auto particle_l = neighborhood->getNeighbor(j,l);
                                        auto dof_val = (particle_l<nlocal_particles) ? velocity_dof_view(particle_l,m_in) : velocity_halo_dof_view(particle_l-nlocal_particles,m_in);
                                        double v;
                                        if (use_vector_gmls) v = vel_gmls->getSolutionSetHost()->getAlpha1TensorTo1Tensor(TargetOperation::VectorPointEvaluation, j, m_out, l, m_in, i+1);
                                        else v = vel_gmls->getSolutionSetHost()->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, j, l, i+1);
                                        u_val += dof_val * v;
                                    }
                                }
                                int current_edge_num = (i - num_interior_quadrature)/num_exterior_quadrature_per_edge;
                                int adj_j = (int)(adjacent_elements(j, current_edge_num));

                                int side_j_to_adjacent_cell = -1;
                                for (int z=0; z<num_edges; ++z) {
                                    if ((int)(adjacent_elements(adj_j,z))==j) {
                                        side_j_to_adjacent_cell = z;
                                        break;
                                    }
                                }
                                //int adj_i = num_interior_quadrature + side_j_to_adjacent_cell*num_exterior_quadrature_per_edge + (num_exterior_quadrature_per_edge - ((i-num_interior_quadrature)%num_exterior_quadrature_per_edge) - 1);
                                int adj_i = -1;
                                if (input_dim==2) {
                                    adj_i = num_interior_quadrature + side_j_to_adjacent_cell*num_exterior_quadrature_per_edge + (num_exterior_quadrature_per_edge - ((i-num_interior_quadrature)%num_exterior_quadrature_per_edge) - 1);
                                } else {
                                    for (int alt_i=0; alt_i<num_exterior_quadrature_per_edge; ++alt_i) {
                                        int potential_adj_i = num_interior_quadrature + side_j_to_adjacent_cell*num_exterior_quadrature_per_edge + alt_i;
                                        bool all_same = true;
                                        for (int d=0; d<3; ++d) {
                                            double diff_d = quadrature_points(adj_j, input_dim*potential_adj_i+d) - quadrature_points(j, input_dim*i+d);
                                            if (diff_d*diff_d > 1e-14) {
                                                all_same = false;
                                                break;
                                            }
                                        }
                                        if (all_same) {
                                            adj_i = potential_adj_i;
                                            break;
                                        }        
                                    }
                                }

                                // needs reconstruction at this quadrature point
                                num_neighbors = neighborhood->getNumNeighbors(adj_j);
                                // loop over particles neighbor to the cell
                                for (int m_in=0; m_in<velocity_dof_view.extent(1); ++m_in) {
                                    if (m_out!=m_in && !velocity_basis_type_vector) continue;
                                    for (LO l = 0; l < num_neighbors; l++) {
                                        auto particle_l = neighborhood->getNeighbor(adj_j,l);
                                        auto dof_val = (particle_l<nlocal_particles) ? velocity_dof_view(particle_l,m_in) : velocity_halo_dof_view(particle_l-nlocal_particles,m_in);
                                        double v;
                                        if (use_vector_gmls) v = vel_gmls->getSolutionSetHost()->getAlpha1TensorTo1Tensor(TargetOperation::VectorPointEvaluation, adj_j, m_out, l, m_in, adj_i+1);
                                        else v = vel_gmls->getSolutionSetHost()->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, adj_j, l, adj_i+1);
                                        other_u_val += dof_val * v;
                                    }
                                }
                                jump_error_on_cell += penalty * quadrature_weights(j,i) * (u_val - other_u_val) * (u_val - other_u_val);
                                if (plot_quadrature) {
                                    jump_view(count+i,m_out) += penalty * quadrature_weights(j,i) * (u_val - other_u_val) * (u_val - other_u_val);
                                }
                            }
                        }
                    }
                    velocity_l2_error += l2_error_on_cell;
                    velocity_h1_error += h1_error_on_cell;
                    velocity_jump_error += jump_error_on_cell;
                }
                if (st_op || mix_le_op) {
                    for( int j =0; j<coords->nLocal(); j++){
                        double l2_error_on_cell = 0.0;
                        int count = quadrature_weights.extent(1) * j;
                        for (int m=0; m<pressure_dof_view.extent(1); ++m) {
                            for (int i=0; i<quadrature_weights.extent(1); ++i) {
                                if (quadrature_type(j,i)==1) { // interior
                                    double l2_val = 0.0;
                                    // needs reconstruction at this quadrature point
                                    LO num_neighbors = neighborhood->getNumNeighbors(j);
                                    // loop over particles neighbor to the cell
                                    for (LO l = 0; l < num_neighbors; l++) {
                                        auto particle_l = neighborhood->getNeighbor(j,l);
                                        auto dof_val = (particle_l<nlocal_particles) ? pressure_dof_view(particle_l,m) : pressure_halo_dof_view(particle_l-nlocal_particles,m);
                                        auto v = pressure_gmls->getSolutionSetHost()->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, j, l, i+1);
                                        l2_val += dof_val * v;
                                    }
                                    auto xyz = Compadre::XyzVector(quadrature_points(j,input_dim*i+0), quadrature_points(j,input_dim*i+1), (input_dim==3) ? quadrature_points(j,input_dim*i+2): 0);
                                    if (st_op) {
                                        l2_val -= avg_pressure_computed;
                                        double l2_exact = pressure_function->evalScalar(xyz)-avg_pressure_exact;
                                        l2_error_on_cell += quadrature_weights(j,i) * (l2_val - l2_exact) * (l2_val - l2_exact);
                                    } else if (mix_le_op) {
                                        l2_val -= avg_pressure_computed;
                                        auto velocity_jacobian = velocity_function->evalJacobian(xyz);
                                        double t_lambda_coeff = (lambda_coeff==std::numeric_limits<scalar_type>::infinity()) ? 0.0 : lambda_coeff;
                                        double l2_exact = -1*(t_lambda_coeff + shear_coeff)*(velocity_jacobian[0][0] + velocity_jacobian[1][1])-avg_pressure_exact;
                                        l2_error_on_cell += quadrature_weights(j,i) * (l2_val - l2_exact) * (l2_val - l2_exact);
                                    }
                                }
                            }
                        }
                        pressure_l2_error += l2_error_on_cell;
                    }
                }
            }
            printf("L2: %.5e\n", sqrt(velocity_l2_error));
            printf("H1: %.5e\n", sqrt(velocity_h1_error));
            printf("Ju: %.5e\n", sqrt(velocity_jump_error));
            if (st_op || mix_le_op) {
                printf("Pressure L2: %.5e\n", sqrt(pressure_l2_error));
            }
            double velocity_l2_coeff=0, velocity_h1_coeff=0, pressure_l2_coeff=0;
            if (rd_op) {
                velocity_l2_coeff = reaction_coeff;
                velocity_h1_coeff = diffusion_coeff;
            } else if (le_op || vl_op || mix_le_op) {
                velocity_l2_coeff = shear_coeff;
                if (mix_le_op) {
                    pressure_l2_coeff = 0; // double check SIPG norm
                }
            } else if (st_op) {
                velocity_h1_coeff = diffusion_coeff;
                pressure_l2_coeff = 0; // double check SIPG norm
            }
            // SIPG norm
            velocity_norm = velocity_jump_error + velocity_l2_coeff*velocity_l2_error + velocity_h1_coeff*velocity_h1_error + pressure_l2_coeff*pressure_l2_error;
        }

        //// DIAGNOSTIC:: get solution at quadrature pts
        if (plot_quadrature) {
            {
                Compadre::FileManager fm3;
                std::string output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "errors_at_quadrature.nc";
                fm3.setWriter(output_filename, particles_new);
                fm3.write();
            }
        }

        // get matrix out from problem_T

        ST global_norm;
        Teuchos::Ptr<ST> global_norm_ptr(&global_norm);
        Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_SUM, velocity_norm, global_norm_ptr);
        global_norm = sqrt(global_norm);
        if (comm->getRank()==0) std::cout << "Global Norm: " << global_norm << "\n";

        {
            WriteTime->start();
            Compadre::FileManager fm;
            std::string output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file");
            fm.setWriter(output_filename, cells);
            fm.write();
            WriteTime->stop();
        }
        //{
        //    WriteTime->start();
        //    std::string output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file");
        //    fm.setWriter(output_filename, cells);
        //    fm.write();
        //    WriteTime->stop();
        //}
//        errors[i] = global_norm;
//
//        WriteTime->start();
//        std::string output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file");
//        std::string writetest_output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "writetest" + std::to_string(i) /* loop # */ + parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file");
//        fm.setWriter(output_filename, particles);
//        fm.write();
//        WriteTime->stop();
//        {
//            //
//            // VTK File Reader Test and Parallel VTK File Re-Reader
//            //
//
//            // read solution back in which should also reset flags
//
//            SecondReadTime->start();
//            Teuchos::RCP<Compadre::ParticlesT> test_particles =
//                    Teuchos::rcp( new Compadre::ParticlesT(parameters, comm));
//
//            fm.setReader(output_filename, test_particles);
//            fm.read();
//
//            fm.setWriter(writetest_output_filename, test_particles);
//            fm.write();
//            SecondReadTime->stop();
//
//            test_particles->getFieldManagerConst()->listFields(std::cout);
//
//        }
//
//        if (parameters->get<Teuchos::ParameterList>("solver").get<std::string>("type")=="direct")
//            Teuchos::TimeMonitor::summarize();
//
//        TEUCHOS_TEST_FOR_EXCEPT_MSG(errors[i]!=errors[i], "NaN found in error norm.");
//        if (parameters->get<std::string>("solution type")=="sine") {
//             if (i>0) TEUCHOS_TEST_FOR_EXCEPT_MSG(errors[i-1]/errors[i] < 3.5, "Second order not achieved for sine solution (should be 4).");
//        } else {
//            TEUCHOS_TEST_FOR_EXCEPT_MSG(errors[i] > 1e-13, "Second order solution not recovered exactly.");
//        }
//        if (comm->getRank()==0) parameters->print();
    }
    Teuchos::TimeMonitor::summarize();
    }
    Kokkos::finalize();
    #endif
    return 0;
}


