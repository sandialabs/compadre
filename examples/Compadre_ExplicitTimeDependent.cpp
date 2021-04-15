#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_Core.hpp>
#include <Kokkos_Core.hpp>
#include <Compadre_GlobalConstants.hpp>
#include <Compadre_ProblemExplicitTransientT.hpp>
#include <Compadre_ParticlesT.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_DOFManager.hpp>
#include <Compadre_EuclideanCoordsT.hpp>
#include <Compadre_NeighborhoodT.hpp>
#include <Compadre_FieldT.hpp>
#include <Compadre_XyzVector.hpp>
#include <Compadre_AnalyticFunctions.hpp>
#include <Compadre_FileIO.hpp>
#include <Compadre_ParameterManager.hpp>

#include <Compadre_ExplicitTimeDependent_Operator.hpp>
#include <Compadre_ExplicitTimeDependent_Sources.hpp>
#include <Compadre_ExplicitTimeDependent_BoundaryConditions.hpp>

#include "Compadre_GlobalConstants.hpp"
static const Compadre::GlobalConstants consts;

#include <Compadre_GMLS.hpp> // for getNP()

#include <iostream>

typedef int LO;
typedef long GO;
typedef double ST;
typedef Compadre::XyzVector xyz_type;


int main (int argc, char* args[]) {

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
    
    {
    Kokkos::initialize(argc, args);

    Teuchos::RCP<Teuchos::Time> FirstReadTime = Teuchos::TimeMonitor::getNewCounter ("1st Read Time");
    Teuchos::RCP<Teuchos::Time> AssemblyTime = Teuchos::TimeMonitor::getNewCounter ("Assembly Time");
    Teuchos::RCP<Teuchos::Time> SolvingTime = Teuchos::TimeMonitor::getNewCounter ("Solving Time");
    Teuchos::RCP<Teuchos::Time> WriteTime = Teuchos::TimeMonitor::getNewCounter ("Write Time");
    Teuchos::RCP<Teuchos::Time> NormTime = Teuchos::TimeMonitor::getNewCounter ("Norm calculation");

    auto solution_type = parameters->get<std::string>("solution type");
    auto simulation_type = parameters->get<std::string>("simulation type");
    auto timestepper_type = parameters->get<Teuchos::ParameterList>("time").get<std::string>("type");

    {

        typedef Compadre::EuclideanCoordsT CT;
        Teuchos::RCP<Compadre::ParticlesT> particles =
                Teuchos::rcp( new Compadre::ParticlesT(parameters, comm));

        //Read in data file.
        FirstReadTime->start();
        Compadre::FileManager fm;
        std::string testfilename(
                parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file prefix") 
                + parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file"));
        fm.setReader(testfilename, particles);
        fm.getReader()->setCoordinateLayout(
                parameters->get<Teuchos::ParameterList>("io").get<std::string>("particles number dimension name"),
                parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinates layout"),
                parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate 0 name"),
                parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate 1 name"),
                parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate 2 name"));
        fm.getReader()->setCoordinateUnitStyle(
                parameters->get<Teuchos::ParameterList>("io").get<std::string>("lat lon units"));
        fm.getReader()->setKeepOriginalCoordinates(
                parameters->get<Teuchos::ParameterList>("io").get<bool>("keep original lat lon"));
        fm.read();
        FirstReadTime->stop();

        ST halo_size;
        particles->getFieldManagerConst()->listFields(std::cout);


        // resize to a sphere of Earth's radius
        Teuchos::RCP<Compadre::AnalyticFunction> vel_source_function, h_source_function;
        Teuchos::RCP<Compadre::AnalyticFunction> vel_boundary_function, h_boundary_function;

        Teuchos::RCP<Compadre::AnalyticFunction> zero_function = 
            Teuchos::rcp<Compadre::AnalyticFunction>(new Compadre::ConstantEachDimension(0,0,0));
        ST t0 = parameters->get<Teuchos::ParameterList>("time").get<double>("t0");
        if (solution_type=="polynomial time" && simulation_type=="eulerian") {
            // exact solution is t^5/5 + 2*t - (t0^5/5 + 2*t0), rhs is: t^4 + 2
            Teuchos::RCP<Compadre::AnalyticFunction> t = Teuchos::rcp<Compadre::AnalyticFunction>(new Compadre::LinearInT());
            if (timestepper_type=="rk") {
                vel_source_function = pow(t,4) + 2;//zero_function;
                vel_boundary_function = .2*pow(t,5) + 2*t - (pow(t0,5)/5.0 + 2.0*t0);// (t/zero_function;
                // exact solution is t^3/3 - 10*t - (t0^3/3 - 10*t0), rhs is: t^2 - 10
                h_boundary_function = pow(t,3)/3 - 10*t - (pow(t0,3)/3.0 - 10.0*t0);//zero_function;
                h_source_function = t*t - 10;//zero_function;
            } else if (timestepper_type=="newmark") {
                // exact solution is d: t^2/2 - 10*t, v: t - 10, a: 1
                h_boundary_function = pow(t,2)/2 - 10*t;
                h_source_function = t - 10;
                vel_source_function = 1 + zero_function;
                vel_boundary_function = t - 10;
            }
        }
        else if (solution_type=="nonpolynomial time" && simulation_type=="eulerian") {
            // exact solution is -cos(t)+cos(t0)
            Teuchos::RCP<Compadre::AnalyticFunction> cos_t = Teuchos::rcp<Compadre::AnalyticFunction>(new Compadre::CosT());
            Teuchos::RCP<Compadre::AnalyticFunction> sin_t = Teuchos::rcp<Compadre::AnalyticFunction>(new Compadre::SinT());
            if (timestepper_type=="rk") {
                vel_source_function = sin_t;
                vel_boundary_function = -1*cos_t + std::cos(t0);//zero_function;
                // exact solution is sin(t)-sin(t0)
                h_boundary_function = sin_t - std::sin(t0);
                h_source_function = cos_t;
            } else if (timestepper_type=="newmark") {
                // exact solution is sin(t)-sin(t0)
                h_boundary_function = sin_t - std::sin(t0);
                h_source_function = cos_t;
                vel_source_function = -1*sin_t;
                vel_boundary_function = cos_t;
            }
        } else {
            TEUCHOS_ASSERT(false); // invalid choice
        }


        {
            const ST h_size = .0833333;

            particles->zoltan2Initialize(); // also applies the repartition to coords, flags, and fields
            if (parameters->get<Teuchos::ParameterList>("halo").get<bool>("dynamic")) {
                halo_size = h_size * parameters->get<Teuchos::ParameterList>("halo").get<double>("multiplier");
            } else {
                halo_size = parameters->get<Teuchos::ParameterList>("halo").get<double>("size");
            }
            particles->buildHalo(halo_size);


            if (timestepper_type=="rk") {
                particles->getFieldManager()->createField(1, "velocity", "m/s");
                particles->getFieldManager()->createField(1, "height", "m");


                particles->getFieldManager()->getFieldByName("velocity")->
                        localInitFromScalarFunction(vel_boundary_function.getRawPtr(), parameters->get<Teuchos::ParameterList>("time").get<double>("t0"));
                        //localInitFromScalarFunction(zero_function.getRawPtr(), parameters->get<Teuchos::ParameterList>("time").get<double>("t0"));
                particles->getFieldManager()->getFieldByName("height")->
                        localInitFromScalarFunction(h_boundary_function.getRawPtr(), parameters->get<Teuchos::ParameterList>("time").get<double>("t0"));
                        //localInitFromScalarFunction(zero_function.getRawPtr(), parameters->get<Teuchos::ParameterList>("time").get<double>("t0"));
            } else if (timestepper_type=="newmark") {
                particles->getFieldManager()->createField(1, "d", "m/s");
                particles->getFieldManager()->createField(1, "d_velocity", "m/s");
                particles->getFieldManager()->getFieldByName("d")->
                        localInitFromScalarFunction(h_boundary_function.getRawPtr(), parameters->get<Teuchos::ParameterList>("time").get<double>("t0"));
                        //localInitFromScalarFunction(zero_function.getRawPtr(), parameters->get<Teuchos::ParameterList>("time").get<double>("t0"));
                particles->getFieldManager()->getFieldByName("d_velocity")->
                        localInitFromScalarFunction(vel_boundary_function.getRawPtr(), parameters->get<Teuchos::ParameterList>("time").get<double>("t0"));
                        //localInitFromScalarFunction(zero_function.getRawPtr(), parameters->get<Teuchos::ParameterList>("time").get<double>("t0"));

            }

            Compadre::ConstantEachDimension proc_func = Compadre::ConstantEachDimension(comm->getRank(),1,1);
            particles->getFieldManager()->createField(1, "processor", "none");
            particles->getFieldManager()->getFieldByName("processor")->
                    localInitFromScalarFunction(&proc_func);


            particles->getFieldManager()->createField(3, "displacement", "m/s");
            particles->getFieldManager()->getFieldByName("displacement")->
                    localInitFromVectorFunction(zero_function.getRawPtr());

            particles->getFieldManager()->updateFieldsHaloData();


            particles->createDOFManager();
            particles->getDOFManager()->generateDOFMap();

            const ST h_support = parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("size");

            particles->createNeighborhood();
            particles->getNeighborhood()->setAllHSupportSizes(h_support);

            LO neighbors_needed = Compadre::GMLS::getNP(parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"), 2);
            particles->getNeighborhood()->constructAllNeighborLists(particles->getCoordsConst()->getHaloSize(),
                    parameters->get<Teuchos::ParameterList>("neighborhood").get<std::string>("search type"),
                    neighbors_needed,
                    parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("cutoff multiplier"),
                    parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("size"),
                    parameters->get<Teuchos::ParameterList>("neighborhood").get<bool>("uniform radii"),
                    parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("radii post search scaling"));

            double h_min = particles->getNeighborhood()->computeMinHSupportSize(true /*global min*/);

            Teuchos::RCP<Compadre::ProblemExplicitTransientT> problem =
                Teuchos::rcp( new Compadre::ProblemExplicitTransientT(particles));

            // construct physics, sources, and boundary conditions
            Teuchos::RCP<Compadre::ExplicitTimeDependentPhysics> physics =
                Teuchos::rcp( new Compadre::ExplicitTimeDependentPhysics(particles));
            Teuchos::RCP<Compadre::ExplicitTimeDependentSources> source =
                Teuchos::rcp( new Compadre::ExplicitTimeDependentSources(particles));
            source->setParameters(*parameters);
            Teuchos::RCP<Compadre::ExplicitTimeDependentBoundaryConditions> bcs =
                Teuchos::rcp( new Compadre::ExplicitTimeDependentBoundaryConditions(particles));
            bcs->setParameters(*parameters);


            // set physics, sources, and boundary conditions in the problem
            physics->vel_source_function = vel_source_function;
            physics->h_source_function = h_source_function;
            physics->vel_boundary_function = vel_boundary_function;
            physics->h_boundary_function = h_boundary_function;
            problem->setPhysics(physics);
            source->setPhysics(physics);
            bcs->setPhysics(physics);
            problem->setSources(source);
            problem->setBCS(bcs);

            // assembly
            AssemblyTime->start();
            problem->initialize();
            AssemblyTime->stop();

            // get stable timestep
            double stable_timestep_size;
            stable_timestep_size = parameters->get<Teuchos::ParameterList>("time").get<double>("dt");
            //} else {
            //    stable_timestep_size = (parameters->get<Teuchos::ParameterList>("time").get<double>("dt") > 0) ?
            //        parameters->get<Teuchos::ParameterList>("time").get<double>("dt") : physics->getStableTimeStep(h_min, 5960)*parameters->get<Teuchos::ParameterList>("time").get<double>("dt multiplier");
            //}

//            double stable_timestep_size = physics->getStableTimeStep(h_min, (2.94e+4)/sw_function.getGravity());
            TEUCHOS_TEST_FOR_EXCEPT_MSG(stable_timestep_size==0, "Stable timestep not found or dt specified to 0.");

            //solving
            SolvingTime->start();
            problem->solve(parameters->get<Teuchos::ParameterList>("time"), parameters->get<Teuchos::ParameterList>("time").get<double>("t0"), parameters->get<Teuchos::ParameterList>("time").get<double>("t_end"), stable_timestep_size, parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file"));
            SolvingTime->stop();

//            WriteTime->start();
//            {
//                std::string output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file");
//                Compadre::FileManager fm;
//                fm.setWriter(output_filename, particles);
//                fm.write();
//            }
//            WriteTime->stop();

            // point by point check to see how far off from correct value
            // check solution

            NormTime->start();
            ST velocity_error_weighted_l2_norm = 0;
            ST exact_velocity_weighted_l2_norm = 0;

            ST height_error_weighted_l2_norm = 0;
            ST exact_height_weighted_l2_norm = 0;

            CT* coords = (CT*)particles->getCoords();

            Compadre::host_view_type initial_coords = 
                particles->getCoords()->getPts(false /*halo*/, false /*use_physical_coords*/)->getLocalView<Compadre::host_view_type>();

            Compadre::host_view_type velocity_field, height_field;
            if (timestepper_type=="rk") {
                velocity_field = 
                    particles->getFieldManager()->getFieldByName("velocity")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();
                height_field = 
                    particles->getFieldManager()->getFieldByName("height")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();
            } else if (timestepper_type=="newmark") {
                velocity_field = 
                    particles->getFieldManager()->getFieldByName("d_velocity")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();
                height_field = 
                    particles->getFieldManager()->getFieldByName("d")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();
            }
            //auto integral_vel_function = Compadre::CosT(parameters->get<Teuchos::ParameterList>("time").get<double>("t_end"),-1)+1;
            //auto integral_h_function = Compadre::CosT(parameters->get<Teuchos::ParameterList>("time").get<double>("t_end"),-1)+1;
            //auto integral_vel_function = 1+Compadre::CosT(parameters->get<Teuchos::ParameterList>("time").get<double>("t_end"),-1);
            //auto integral_h_function = 1+Compadre::CosT(parameters->get<Teuchos::ParameterList>("time").get<double>("t_end"),-1);
            //auto v1 = Compadre::CosT();//(parameters->get<Teuchos::ParameterList>("time").get<double>("t_end"),-1);
            //auto v2 = Compadre::CosT();//(parameters->get<Teuchos::ParameterList>("time").get<double>("t_end"),-1);
            auto integral_vel_function = vel_boundary_function;//-1*v1+1;//Compadre::CosT(parameters->get<Teuchos::ParameterList>("time").get<double>("t_end"),-1);
            auto integral_h_function = h_boundary_function;//-1*v2+1;//Compadre::CosT(parameters->get<Teuchos::ParameterList>("time").get<double>("t_end"),-1);

            for(int j =0; j<coords->nLocal(); j++){

                xyz_type xyz = coords->getLocalCoords(j, true /* use_halo */, true);
                xyz_type exact = integral_vel_function->evalScalar(xyz,0,parameters->get<Teuchos::ParameterList>("time").get<double>("t_end"));
                ST exact_height = integral_h_function->evalScalar(xyz,0,parameters->get<Teuchos::ParameterList>("time").get<double>("t_end"));
                auto t_end = parameters->get<Teuchos::ParameterList>("time").get<double>("t_end");

                velocity_error_weighted_l2_norm += (velocity_field(j,0) - exact.x)*(velocity_field(j,0) - exact.x);
                height_error_weighted_l2_norm += (height_field(j,0) - exact_height)*(height_field(j,0) - exact_height);
                if (solution_type=="polynomial time" && simulation_type=="eulerian") {
                    double this_diff = exact[0] - velocity_field(j,0);
                    if (std::abs(this_diff)>1e-8) {
                        printf("v: %.16g vs %.16g: %.16f\n", exact[0], velocity_field(j,0), exact[0] - velocity_field(j,0));
                    }
                    this_diff = exact_height - height_field(j,0);
                    if (std::abs(this_diff)>1e-15) {
                        printf("h: %.16g vs %.16g: %.16f\n", exact_height, height_field(j,0), exact_height - height_field(j,0));
                    }

                }

//                physical_coordinate_weighted_l2_norm += (disp_field(j,0) - (initial_coords(j,0)*std::cos(2*PI*end_time) + initial_coords(j,1)*std::sin(2*PI*end_time)))*(disp_field(j,0) - (initial_coords(j,0)*std::cos(2*PI*end_time) + initial_coords(j,1)*std::sin(2*PI*end_time)))*grid_area_field(j,0);
//                physical_coordinate_weighted_l2_norm += (disp_field(j,1) - (initial_coords(j,1)*std::cos(2*PI*end_time) - initial_coords(j,0)*std::sin(2*PI*end_time)))*(disp_field(j,1) - (initial_coords(j,1)*std::cos(2*PI*end_time) - initial_coords(j,0)*std::sin(2*PI*end_time)))*grid_area_field(j,0);
//                physical_coordinate_weighted_l2_norm += (disp_field(j,2) - initial_coords(j,2))*(disp_field(j,2) - initial_coords(j,2))*grid_area_field(j,0);

                exact_velocity_weighted_l2_norm += exact.x*exact.x;
                //for(int k=0; k<coords->nLocal(); j++){
                //    exact_velocity_weighted_l2_norm += exact[k]*exact[k];
                //}
                exact_height_weighted_l2_norm += exact_height*exact_height;

            }

            {
                // velocity error communication
                ST global_norm_velocity_error;
                ST global_norm_exact_velocity;
                Teuchos::Ptr<ST> global_norm_ptr_velocity_error(&global_norm_velocity_error);
                Teuchos::Ptr<ST> global_norm_ptr_exact_velocity(&global_norm_exact_velocity);

                Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_SUM, velocity_error_weighted_l2_norm, global_norm_ptr_velocity_error);
                Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_SUM, exact_velocity_weighted_l2_norm, global_norm_ptr_exact_velocity);

                if (comm->getRank()==0) std::cout << "Global relative VELOCITY error: " << std::sqrt(global_norm_velocity_error) /  std::sqrt(global_norm_exact_velocity)  << "\n";
                if (solution_type=="polynomial time" && simulation_type=="eulerian") {
                    TEUCHOS_TEST_FOR_EXCEPT_MSG(std::abs(global_norm_velocity_error)>1e-6, 
                            "Global relative VELOCITY error should be less than 1e-6.");
                }
            }
            {
                // height error communication
                ST global_norm_height_error;
                ST global_norm_exact_height;
                Teuchos::Ptr<ST> global_norm_ptr_height_error(&global_norm_height_error);
                Teuchos::Ptr<ST> global_norm_ptr_exact_height(&global_norm_exact_height);

                Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_SUM, height_error_weighted_l2_norm, global_norm_ptr_height_error);
                Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_SUM, exact_height_weighted_l2_norm, global_norm_ptr_exact_height);

                if (comm->getRank()==0) std::cout << "Global relative HEIGHT error: " << std::sqrt(global_norm_height_error) /  std::sqrt(global_norm_exact_height)  << "\n\n\n\n";
                if (solution_type=="polynomial time" && simulation_type=="eulerian") {
                    TEUCHOS_TEST_FOR_EXCEPT_MSG(std::abs(global_norm_height_error)>1e-15, 
                            "Global relative HEIGHT error should be less than 1e-15.");
                }
            }

            NormTime->stop();
        }
    }

    if (comm->getRank()==0) parameters->print();
    Teuchos::TimeMonitor::summarize();
    Kokkos::finalize();
    }
return 0;
}

