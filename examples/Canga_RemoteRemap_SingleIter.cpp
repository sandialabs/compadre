#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_Core.hpp>

#include <Kokkos_Core.hpp>

#include <Compadre_GMLS.hpp> // for getNP()
#include <Compadre_GlobalConstants.hpp>
#include <Compadre_ProblemT.hpp>
#include <Compadre_ParticlesT.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_DOFManager.hpp>
#include <Compadre_EuclideanCoordsT.hpp>
#include <Compadre_SphericalCoordsT.hpp>
#include <Compadre_NeighborhoodT.hpp>
#include <Compadre_FieldT.hpp>
#include <Compadre_XyzVector.hpp>
#include <Compadre_AnalyticFunctions.hpp>
#include <Compadre_FileIO.hpp>
#include <Compadre_ParameterManager.hpp>
#include <Compadre_RemapManager.hpp>

#include "Compadre_GlobalConstants.hpp"
static const Compadre::GlobalConstants consts;

#include <iostream>

#define STACK_TRACE(call) try { call; } catch (const std::exception& e ) { TEUCHOS_TRACE(e); }

typedef int LO;
typedef long GO;
typedef double ST;
typedef Compadre::XyzVector xyz_type;

static const ST PI = consts.Pi();

using namespace Compadre;

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

    bool post_process_grad = false;
    try {
        post_process_grad = parameters->get<bool>("post process grad");
    } catch (...) {
    }

    std::string nc_type;
    try {
        nc_type = parameters->get<Teuchos::ParameterList>("io").get<std::string>("nc type");
    } catch (...) {
        nc_type = "";
    }

    Teuchos::GlobalMPISession mpi(&argc, &args);
    Teuchos::oblackholestream bstream;
    Teuchos::RCP<const Teuchos::Comm<int> > comm = Teuchos::DefaultComm<int>::getComm();

    Kokkos::initialize(argc, args);
    
    const int procRank = comm->getRank();
    const int nProcs = comm->getSize();

    Teuchos::RCP<Teuchos::Time> SphericalParticleTime = Teuchos::TimeMonitor::getNewCounter ("Spherical Particle Time");
    Teuchos::RCP<Teuchos::Time> NeighborSearchTime = Teuchos::TimeMonitor::getNewCounter ("Laplace Beltrami - Neighbor Search Time");
    Teuchos::RCP<Teuchos::Time> FirstReadTime = Teuchos::TimeMonitor::getNewCounter ("1st Read Time");
    Teuchos::RCP<Teuchos::Time> SecondReadTime = Teuchos::TimeMonitor::getNewCounter ("2nd Read Time");
    Teuchos::RCP<Teuchos::Time> AssemblyTime = Teuchos::TimeMonitor::getNewCounter ("Assembly Time");
    Teuchos::RCP<Teuchos::Time> SolvingTime = Teuchos::TimeMonitor::getNewCounter ("Solving Time");
    Teuchos::RCP<Teuchos::Time> WriteTime = Teuchos::TimeMonitor::getNewCounter ("Write Time");
    Teuchos::RCP<Teuchos::Time> NormTime = Teuchos::TimeMonitor::getNewCounter ("Norm calculation");
    Teuchos::RCP<Teuchos::Time> RemapTime = Teuchos::TimeMonitor::getNewCounter ("Remap calculation");

    {

        // SOURCE PARTICLES
        typedef Compadre::EuclideanCoordsT CT;
        Teuchos::RCP<Compadre::ParticlesT> particles =
                Teuchos::rcp( new Compadre::ParticlesT(parameters, comm));
         CT* coords = (CT*)particles->getCoords();

        //READ IN SOURCE DATA FILE
        {
            FirstReadTime->start();
            Compadre::FileManager fm;
            std::string testfilename(parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file prefix") + parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file"));
            fm.setReader(testfilename, particles, nc_type);
            //fm.getReader()->setCoordinateLayout(parameters->get<Teuchos::ParameterList>("io").get<std::string>("particles number dimension name"),
            //                                    parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinates layout"),
            //                                    parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate 0 name"),
            //                                    parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate 1 name"),
            //                                    parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate 2 name"));
            //fm.getReader()->setCoordinateUnitStyle(parameters->get<Teuchos::ParameterList>("io").get<std::string>("lat lon units"));
            //fm.getReader()->setKeepOriginalCoordinates(parameters->get<Teuchos::ParameterList>("io").get<bool>("keep original lat lon"));
            fm.read();
            FirstReadTime->stop();
        }

        auto x_field = particles->getCoordsConst()->getPts()->getLocalView<Compadre::host_view_type>();
        //for (int i=0; i<x_field.extent(0); ++i) {
        //    printf("s:x %d: %f, ", i, x_field(i,0));
        //}


        ST halo_size;
        // BUILD HALO INFO UP
        {
            const ST h_size = .0833333;

            particles->zoltan2Initialize(); // also applies the repartition to coords, flags, and fields
            if (parameters->get<Teuchos::ParameterList>("halo").get<bool>("dynamic")) {
                halo_size = h_size * parameters->get<Teuchos::ParameterList>("halo").get<double>("multiplier");
            } else {
                halo_size = parameters->get<Teuchos::ParameterList>("halo").get<double>("size");
            }
            particles->buildHalo(halo_size);
        }

        // GET OR CREATE FIELDS FROM SOURCE FILE
        // ALREADY EXIST, SO REALLY JUST A CHECK FOR EXISTENCE OF FIELD
        //particles->getFieldManager()->getFieldByName("solution")

        // OUTPUT PARTICLES
        Teuchos::RCP<Compadre::ParticlesT> new_particles =
            Teuchos::rcp( new Compadre::ParticlesT(parameters, comm));
        CT* new_coords = (CT*)new_particles->getCoords();
        const ST h_support = parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("size");

        // READ IN OUTPUT PARTICLE DATA
        {
            SecondReadTime->start();
            Compadre::FileManager fm2;
            std::string testfilename(parameters->get<Teuchos::ParameterList>("io").get<std::string>("target file prefix") + parameters->get<Teuchos::ParameterList>("io").get<std::string>("target file"));
            fm2.setReader(testfilename, new_particles, nc_type);
            //fm2.getReader()->setCoordinateLayout(parameters->get<Teuchos::ParameterList>("io").get<std::string>("particles number dimension name"),
            //                                    parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinates layout"),
            //                                    parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate 0 name"),
            //                                    parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate 1 name"),
            //                                    parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate 2 name"));
            //fm2.getReader()->setCoordinateUnitStyle(parameters->get<Teuchos::ParameterList>("io").get<std::string>("lat lon units"));
            //fm2.getReader()->setKeepOriginalCoordinates(parameters->get<Teuchos::ParameterList>("io").get<bool>("keep original lat lon"));
            fm2.read();
            SecondReadTime->stop();
        }

        x_field = new_particles->getCoordsConst()->getPts()->getLocalView<Compadre::host_view_type>();
        //for (int i=0; i<x_field.extent(0); ++i) {
        //    printf("t:x %d: %f, ", i, x_field(i,0));
        //}

        // REMAP
        RemapTime->start();
        {
            Teuchos::RCP<Compadre::RemapManager> rm = Teuchos::rcp(new Compadre::RemapManager(parameters, particles.getRawPtr(), new_particles.getRawPtr(), halo_size));
            new_particles->buildHalo(halo_size);
            new_particles->getFieldManager()->updateFieldsHaloData();

            // The following two parameters are used for mass conservation
            // _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("source weighting field name")
            // _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("target weighting field name")

            // _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("dense linear solver") determines whether problem is solved on a manifold

            Compadre::RemapObject r0("TotalPrecipWater", "TotalPrecipWater", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
            Compadre::OptimizationObject opt_obj_0 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, true /*bounds preservation*/, 0, 1.0e+15);
            r0.setOptimizationObject(opt_obj_0);
            rm->add(r0);

            Compadre::RemapObject r1("CloudFraction", "CloudFraction", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
            Compadre::OptimizationObject opt_obj_1 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, true /*bounds preservation*/, 0, 1);
            r1.setOptimizationObject(opt_obj_1);
            rm->add(r1);

            Compadre::RemapObject r2("Topography", "Topography", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
            Compadre::OptimizationObject opt_obj_2 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, true /*bounds preservation*/, -1.0e+15, 1.0e+15);
            r2.setOptimizationObject(opt_obj_2);
            rm->add(r2);

            STACK_TRACE(rm->execute());
        }
        RemapTime->stop();

        // WRITE TO NEW FILE HERE
        {
            Compadre::FileManager fm3;
            std::string output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file");
            // NEED OPTION TO WRITE COORDINATES OUT WITH ORIGINAL COORDINATES (if kept), and with variable names
            fm3.setWriter(output_filename, new_particles);
            fm3.write();

            output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("alt output file prefix") + parameters->get<Teuchos::ParameterList>("io").get<std::string>("alt output file");
            // NEED OPTION TO WRITE COORDINATES OUT WITH ORIGINAL COORDINATES (if kept), and with variable names
            fm3.setWriter(output_filename, new_particles);
            fm3.write();
        }

        //ST physical_coordinate_weighted_l2_norm = 0;
        //ST exact_coordinate_weighted_l2_norm = 0;
        //Compadre::host_view_type solution_field;

        //bool use_grid_area_for_L2 = false;
        //Compadre::host_view_type grid_area_field;
        //try {
        //    grid_area_field = particles->getFieldManager()->getFieldByName("grid_area")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();
        //    use_grid_area_for_L2 = true;
        //} catch (...) {
        //}
        //// if a weighted L2 is desire, use_grid_area_for_L2 can be checked before-hand

        //Compadre::host_view_type exact_solution_field = new_particles->getFieldManager()->getFieldByName("exact_solution")->getMultiVectorPtr()->getLocalView<Compadre::host_view_type>();

        //double exact = 0;
        //GO num_solved_for = 0;
        //for(int j=0; j<coords->nLocal(); j++){
        //    if     (particles->getFlag(j)==0) {
        //        xyz_type xyz = coords->getLocalCoords(j);
        //        if (parameters->get<std::string>("solution type")=="point") {
        //            if (parameters->get<LO>("physics number")<3) {
        //                exact = function->evalScalar(xyz) * (1.0 / (5 * (5 + 1)));
        //            } else {
        //                exact = function->evalScalar(xyz);
        //            }
        //        } else if (parameters->get<std::string>("solution type")=="vector") {
        //            exact = function->evalScalarDerivative(xyz).x;
        //        } else if (parameters->get<std::string>("solution type")=="laplace" || parameters->get<std::string>("solution type")=="staggered_laplace") {
        //            exact = -function->evalScalar(xyz);
        //        } else if (parameters->get<std::string>("solution type")=="grad" || parameters->get<std::string>("solution type")=="staggered_grad") {
        //            exact = function->evalVector(xyz).x;
        //        } else if (parameters->get<std::string>("solution type")=="div_grad" || parameters->get<std::string>("solution type")=="staggered_div_grad") {
        //            exact = -function->evalScalar(xyz);
        //        } else if (parameters->get<std::string>("solution type")=="div" || parameters->get<std::string>("solution type")=="staggered_div") {
        //            exact = -function->evalScalar(xyz);
        //        } else if (parameters->get<std::string>("solution type")=="five_strip") {
        //            if (post_process_grad) {
        //                xyz_type this_xyz = new_coords->getLocalCoords(j);
        //                exact = function->evalVector(this_xyz).x;
        //            } else {
        //                exact = function->evalScalar(xyz);
        //            }
        //        } else { // lb solve
        //            if (parameters->get<LO>("physics number")<3) {
        //                exact = function->evalScalar(xyz) * (1.0 / (5 * (5 + 1)));
        //            } else {
        //                exact = function->evalScalar(xyz);
        //            }
        //        }

    //    //        exact = function->evalScalar(xyz) * (1.0 / (5 * (5 + 1)));
        //        physical_coordinate_weighted_l2_norm += (solution_field(j,0) - exact)*(solution_field(j,0) - exact);//*grid_area_field(j,0);
    //    //        if (parameters->get<std::string>("solution type")=="div_grad" ||  parameters->get<std::string>("solution type")=="staggered_div_grad")
    //    //            physical_coordinate_weighted_l2_norm += (solution_field2(j,0) - exact)*(solution_field2(j,0) - exact);

        //        if (parameters->get<std::string>("solution type")=="grad" || parameters->get<std::string>("solution type")=="staggered_grad" || (parameters->get<std::string>("solution type")=="five_strip" && post_process_grad)) {
        //            physical_coordinate_weighted_l2_norm += (solution_field(j,1) - function->evalVector(xyz).y)*(solution_field(j,1) - function->evalVector(xyz).y);
        //            physical_coordinate_weighted_l2_norm += (solution_field(j,2) - function->evalVector(xyz).z)*(solution_field(j,2) - function->evalVector(xyz).z);
        //        } else if (parameters->get<std::string>("solution type")=="vector") {
        //            physical_coordinate_weighted_l2_norm += (solution_field(j,1) - function->evalScalarDerivative(xyz).y)*(solution_field(j,1) - function->evalScalarDerivative(xyz).y);
        //            physical_coordinate_weighted_l2_norm += (solution_field(j,2) - function->evalScalarDerivative(xyz).z)*(solution_field(j,2) - function->evalScalarDerivative(xyz).z);
        //        }

    //    //        physical_coordinate_weighted_l2_norm += (solution_field(j,0) - exact)*(solution_field(j,0) - exact)*grid_area_field(j,0);
        //        exact_coordinate_weighted_l2_norm += exact*exact;//*grid_area_field(j,0);
    //    //        exact_coordinate_weighted_l2_norm += exact*exact*grid_area_field(j,0);
        //        num_solved_for++;
        //    }
        //}

        //// get global # of dofs solved for
        //GO global_num_solved_for = 0;
        //Teuchos::Ptr<GO> global_num_solved_for_ptr(&global_num_solved_for);
        //Teuchos::reduceAll<int, GO>(*comm, Teuchos::REDUCE_SUM, num_solved_for, global_num_solved_for_ptr);


        //physical_coordinate_weighted_l2_norm /= global_num_solved_for;
        ////            physical_coordinate_weighted_l2_norm = sqrt(physical_coordinate_weighted_l2_norm) /((ST)coords->nGlobalMax());

        //ST global_norm_physical_coord = 0;
        //ST global_norm_exact_coord = 0;

        //Teuchos::Ptr<ST> global_norm_ptr_physical_coord(&global_norm_physical_coord);
        //Teuchos::Ptr<ST> global_norm_ptr_exact_coord(&global_norm_exact_coord);

        //Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_SUM, physical_coordinate_weighted_l2_norm, global_norm_ptr_physical_coord);
        //Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_SUM, exact_coordinate_weighted_l2_norm, global_norm_ptr_exact_coord);

        //global_norm_physical_coord = std::sqrt(global_norm_physical_coord);
        //if (comm->getRank()==0)
        //    printf("\nGlobal Norm: %.16f\n\n\n\n\n", global_norm_physical_coord);

//        //global_norm_exact_coord = std::max(global_norm_exact_coord, 1e-15);
//        //if (comm->getRank()==0) std::cout << "Global Norm Physical Coordinates: " << std::sqrt(global_norm_physical_coord) << " " << std::sqrt(global_norm_physical_coord) /  std::sqrt(global_norm_exact_coord)  << "\n\n\n\n\n";
//        //if (comm->getRank()==0) std::cout << "Global Norm Physical Coordinates: " << std::sqrt(global_norm_physical_coord) /  std::sqrt(global_norm_exact_coord)  << "\n\n\n\n\n";

        //double fail_threshold = 0;
        //try {
        //    fail_threshold = parameters->get<double>("fail threshold");
        //} catch (...) {
        //}

        //std::ostringstream msg;
        //msg << "Error [" << global_norm_physical_coord << "] larger than acceptable [" << fail_threshold << "].\n";
        //if (fail_threshold > 0) TEUCHOS_TEST_FOR_EXCEPT_MSG(global_norm_physical_coord > fail_threshold, msg.str());

//        //TEUCHOS_TEST_FOR_EXCEPT_MSG(global_norm_physical_coord > 1e-15, "Physical coordinate error too large (should be exact).");
        //NormTime->stop();

    }

    if (comm->getRank()==0) parameters->print();
    Teuchos::TimeMonitor::summarize();
    Kokkos::finalize();
    return 0;
}


