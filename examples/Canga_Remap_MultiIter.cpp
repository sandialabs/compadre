#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_Core.hpp>
#include <Kokkos_Core.hpp>
#include <Compadre_GlobalConstants.hpp>
#include <Compadre_ParticlesT.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_RemapManager.hpp>
#include <Compadre_EuclideanCoordsT.hpp>
#include <Compadre_SphericalCoordsT.hpp>
#include <Compadre_FieldT.hpp>
#include <Compadre_XyzVector.hpp>
#include <Compadre_AnalyticFunctions.hpp>
#include <Compadre_FileIO.hpp>
#include <Compadre_ParameterManager.hpp>

#include <iostream>

typedef int LO;
typedef long GO;
typedef double ST;
typedef Compadre::EuclideanCoordsT CT;
typedef Compadre::XyzVector xyz_type;

using namespace Compadre;

int main (int argc, char* args[]) {

    Teuchos::GlobalMPISession mpi(&argc, &args);
    Teuchos::RCP<const Teuchos::Comm<int> > global_comm = Teuchos::DefaultComm<int>::getComm();

    Kokkos::initialize(argc, args);

    Teuchos::RCP<Teuchos::Time> ReadTime = Teuchos::TimeMonitor::getNewCounter ("Read Time");
    Teuchos::RCP<Teuchos::Time> WriteTime = Teuchos::TimeMonitor::getNewCounter ("Write Time");

    //********* BASIC PARAMETER SETUP FOR ANY PROBLEM
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
    parameters->print();
    // There is a balance between "neighbors needed multiplier" and "cutoff multiplier"
    // Their product must increase somewhat with "porder", but "neighbors needed multiplier"
    // requires more of an increment than "cutoff multiplier" to achieve the required threshold (the other staying the same)
    //*********
 
    local_index_type initial_step_to_compute = parameters->get<local_index_type>("initial step");
    local_index_type save_every = parameters->get<local_index_type>("save every");

    Teuchos::RCP<const Teuchos::Comm<int> > comm = global_comm;

	Teuchos::RCP<Compadre::AnalyticFunction> function;
	function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::SineProducts(3 /*dimension*/)));

    bool preserve_local_bounds  = parameters->get<Teuchos::ParameterList>("remap").get<bool>("preserve local bounds");

	{

        {
            //
            // Remote Data Test (Two differing distributions of particles)
            //

             Teuchos::RCP<Compadre::ParticlesT> particles_left =
                 Teuchos::rcp( new Compadre::ParticlesT(parameters, comm));

             Teuchos::RCP<Compadre::ParticlesT> particles_right =
                 Teuchos::rcp( new Compadre::ParticlesT(parameters, comm));

            //Read in data file.
            ReadTime->start();
            {
                Compadre::FileManager fm;
                std::string testfilename;
                if (initial_step_to_compute==1) {
                    testfilename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file prefix") + parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file 1");
                } else {
                    testfilename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "backward_" + std::to_string(initial_step_to_compute-1) + ".nc";
                }
                fm.setReader(testfilename, particles_left);
                fm.getReader()->setCoordinateLayout(parameters->get<Teuchos::ParameterList>("io").get<std::string>("particles number dimension name"),
                                                    parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinates layout"),
                                                    parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate 0 name"),
                                                    parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate 1 name"),
                                                    parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate 2 name"));
                fm.getReader()->setCoordinateUnitStyle(parameters->get<Teuchos::ParameterList>("io").get<std::string>("lat lon units"));
                fm.getReader()->setKeepOriginalCoordinates(parameters->get<Teuchos::ParameterList>("io").get<bool>("keep original lat lon"));
                fm.read();
            }
            {
                Compadre::FileManager fm;
                std::string testfilename;
                if (initial_step_to_compute==1) {
                    testfilename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file prefix") + parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file 2");
                } else {
                    testfilename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "backward_" + std::to_string(initial_step_to_compute-1) + ".nc";
                }
                fm.setReader(testfilename, particles_right);
                fm.getReader()->setCoordinateLayout(parameters->get<Teuchos::ParameterList>("io").get<std::string>("particles number dimension name"),
                                                    parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinates layout"),
                                                    parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate 0 name"),
                                                    parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate 1 name"),
                                                    parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate 2 name"));
                fm.getReader()->setCoordinateUnitStyle(parameters->get<Teuchos::ParameterList>("io").get<std::string>("lat lon units"));
                fm.getReader()->setKeepOriginalCoordinates(parameters->get<Teuchos::ParameterList>("io").get<bool>("keep original lat lon"));
                fm.read();
            }
            ReadTime->stop();
                
            if (initial_step_to_compute==1) {
                particles_left->getFieldManager()->createField(9,"tangent_bundle","na");
                particles_left->getFieldManager()->createField(1,"exact CloudFraction","m^2/1a");
                particles_left->getFieldManager()->createField(1,"exact TotalPrecipWater","m^2/1a");
                particles_left->getFieldManager()->createField(1,"exact Topography","m^2/1a");
                particles_left->getFieldManager()->createField(1,"exact AnalyticalFun1","m^2/1a");
                particles_left->getFieldManager()->createField(1,"exact AnalyticalFun2","m^2/1a");
                Kokkos::deep_copy(particles_left->getFieldManager()->getFieldByName("exact CloudFraction")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll),
                    particles_left->getFieldManager()->getFieldByName("CloudFraction")->getMultiVectorPtrConst()->getLocalViewHost(Tpetra::Access::ReadOnly));
                Kokkos::deep_copy(particles_left->getFieldManager()->getFieldByName("exact TotalPrecipWater")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll),
                    particles_left->getFieldManager()->getFieldByName("TotalPrecipWater")->getMultiVectorPtrConst()->getLocalViewHost(Tpetra::Access::ReadOnly));
                Kokkos::deep_copy(particles_left->getFieldManager()->getFieldByName("exact Topography")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll),
                    particles_left->getFieldManager()->getFieldByName("Topography")->getMultiVectorPtrConst()->getLocalViewHost(Tpetra::Access::ReadOnly));
                Kokkos::deep_copy(particles_left->getFieldManager()->getFieldByName("exact AnalyticalFun1")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll),
                    particles_left->getFieldManager()->getFieldByName("AnalyticalFun1")->getMultiVectorPtrConst()->getLocalViewHost(Tpetra::Access::ReadOnly));
                Kokkos::deep_copy(particles_left->getFieldManager()->getFieldByName("exact AnalyticalFun2")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll),
                    particles_left->getFieldManager()->getFieldByName("AnalyticalFun2")->getMultiVectorPtrConst()->getLocalViewHost(Tpetra::Access::ReadOnly));
                {
			        const CT * coords = (CT*)(particles_left->getCoordsConst());
                    auto tangent_view = particles_left->getFieldManager()->getFieldByName("tangent_bundle")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
                    auto lat_view = particles_left->getFieldManager()->getFieldByName("lat")->getMultiVectorPtrConst()->getLocalViewHost(Tpetra::Access::ReadOnly);
                    auto lon_view = particles_left->getFieldManager()->getFieldByName("lon")->getMultiVectorPtrConst()->getLocalViewHost(Tpetra::Access::ReadOnly);
                    for (int i=0; i<tangent_view.extent(0); ++i) {
				        xyz_type xyz = coords->getLocalCoords(i);
                        tangent_view(i,0) = -sin(lat_view(i,0))*cos(lon_view(i,0));
                        tangent_view(i,1) = -sin(lat_view(i,0))*sin(lon_view(i,0));
                        tangent_view(i,2) = cos(lat_view(i,0));
                        tangent_view(i,3) = -cos(lat_view(i,0))*sin(lon_view(i,0));
                        tangent_view(i,4) = cos(lat_view(i,0))*cos(lon_view(i,0));
                        tangent_view(i,5) = 0;
                        tangent_view(i,6) = xyz[0];
                        tangent_view(i,7) = xyz[1];
                        tangent_view(i,8) = xyz[2];
                    }
                }

                particles_right->getFieldManager()->createField(9,"tangent_bundle","na");
                particles_right->getFieldManager()->createField(1,"exact CloudFraction","m^2/1a");
                particles_right->getFieldManager()->createField(1,"exact TotalPrecipWater","m^2/1a");
                particles_right->getFieldManager()->createField(1,"exact Topography","m^2/1a");
                particles_right->getFieldManager()->createField(1,"exact AnalyticalFun1","m^2/1a");
                particles_right->getFieldManager()->createField(1,"exact AnalyticalFun2","m^2/1a");
                Kokkos::deep_copy(particles_right->getFieldManager()->getFieldByName("exact CloudFraction")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll),
                    particles_right->getFieldManager()->getFieldByName("CloudFraction")->getMultiVectorPtrConst()->getLocalViewHost(Tpetra::Access::ReadOnly));
                Kokkos::deep_copy(particles_right->getFieldManager()->getFieldByName("exact TotalPrecipWater")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll),
                    particles_right->getFieldManager()->getFieldByName("TotalPrecipWater")->getMultiVectorPtrConst()->getLocalViewHost(Tpetra::Access::ReadOnly));
                Kokkos::deep_copy(particles_right->getFieldManager()->getFieldByName("exact Topography")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll),
                    particles_right->getFieldManager()->getFieldByName("Topography")->getMultiVectorPtrConst()->getLocalViewHost(Tpetra::Access::ReadOnly));
                Kokkos::deep_copy(particles_right->getFieldManager()->getFieldByName("exact AnalyticalFun1")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll),
                    particles_right->getFieldManager()->getFieldByName("AnalyticalFun1")->getMultiVectorPtrConst()->getLocalViewHost(Tpetra::Access::ReadOnly));
                Kokkos::deep_copy(particles_right->getFieldManager()->getFieldByName("exact AnalyticalFun2")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll),
                    particles_right->getFieldManager()->getFieldByName("AnalyticalFun2")->getMultiVectorPtrConst()->getLocalViewHost(Tpetra::Access::ReadOnly));
                {
			        const CT * coords = (CT*)(particles_right->getCoordsConst());
                    auto tangent_view = particles_right->getFieldManager()->getFieldByName("tangent_bundle")->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
                    auto lat_view = particles_right->getFieldManager()->getFieldByName("lat")->getMultiVectorPtrConst()->getLocalViewHost(Tpetra::Access::ReadOnly);
                    auto lon_view = particles_right->getFieldManager()->getFieldByName("lon")->getMultiVectorPtrConst()->getLocalViewHost(Tpetra::Access::ReadOnly);
                    for (int i=0; i<tangent_view.extent(0); ++i) {
				        xyz_type xyz = coords->getLocalCoords(i);
                        tangent_view(i,0) = -sin(lat_view(i,0))*cos(lon_view(i,0));
                        tangent_view(i,1) = -sin(lat_view(i,0))*sin(lon_view(i,0));
                        tangent_view(i,2) = cos(lat_view(i,0));
                        tangent_view(i,3) = -cos(lat_view(i,0))*sin(lon_view(i,0));
                        tangent_view(i,4) = cos(lat_view(i,0))*cos(lon_view(i,0));
                        tangent_view(i,5) = 0;
                        tangent_view(i,6) = xyz[0];
                        tangent_view(i,7) = xyz[1];
                        tangent_view(i,8) = xyz[2];
                    }
                }
            }
            printf("Exact solutions copied.\n");
            // else exact data already is created in the particle set

             // calculate h_size for mesh
             // build halo data
            ST halo_size = parameters->get<Teuchos::ParameterList>("halo").get<double>("size") *
                    ( parameters->get<Teuchos::ParameterList>("halo").get<double>("multiplier")
                            + (ST)(parameters->get<Teuchos::ParameterList>("remap").get<int>("porder")));

            particles_left->zoltan2Initialize();
            particles_left->buildHalo(halo_size);
            particles_left->getFieldManager()->updateFieldsHaloData();

            particles_right->zoltan2Initialize();
            particles_right->buildHalo(halo_size);
            particles_right->getFieldManager()->updateFieldsHaloData();



            //if (parameters->get<Teuchos::ParameterList>("remap").get<bool>("cell averaged")) {
            //    Compadre::RemapObject r0("TotalPrecipWater", "TotalPrecipWater", TargetOperation::CellAverageEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, CellAverageSample, PointSample);
            //    //Compadre::RemapObject r0("TotalPrecipWater", "TotalPrecipWater", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, CellAverageSample, PointSample);
            //    //Compadre::RemapObject r0("TotalPrecipWater", "TotalPrecipWater", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
            //    r0.setSourceExtraData(extra_data_name);
            //    r0.setTargetExtraData(extra_data_name);
            //    Compadre::OptimizationObject opt_obj_0 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, preserve_local_bounds, 0, 1.0e+15);
            //    r0.setOptimizationObject(opt_obj_0);
            //    rm_right_to_left->add(r0);

            //    Compadre::RemapObject r1("CloudFraction", "CloudFraction", TargetOperation::CellAverageEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, CellAverageSample, PointSample);
            //    //Compadre::RemapObject r1("CloudFraction", "CloudFraction", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, CellAverageSample, PointSample);
            //    //Compadre::RemapObject r1("CloudFraction", "CloudFraction", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
            //    r1.setSourceExtraData(extra_data_name);
            //    r1.setTargetExtraData(extra_data_name);
            //    Compadre::OptimizationObject opt_obj_1 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, preserve_local_bounds, 0, 1);
            //    r1.setOptimizationObject(opt_obj_1);
            //    rm_right_to_left->add(r1);

            //    Compadre::RemapObject r2("Topography", "Topography", TargetOperation::CellAverageEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, CellAverageSample, PointSample);
            //    //Compadre::RemapObject r2("Topography", "Topography", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, CellAverageSample, PointSample);
            //    //Compadre::RemapObject r2("Topography", "Topography", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
            //    r2.setSourceExtraData(extra_data_name);
            //    r2.setTargetExtraData(extra_data_name);
            //    Compadre::OptimizationObject opt_obj_2 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, preserve_local_bounds, -10994.0, 8848.0);
            //    r2.setOptimizationObject(opt_obj_2);
            //    rm_right_to_left->add(r2);

            //    Compadre::RemapObject r3("AnalyticalFun1", "AnalyticalFun1", TargetOperation::CellAverageEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, CellAverageSample, PointSample);
            //    r3.setSourceExtraData(extra_data_name);
            //    r3.setTargetExtraData(extra_data_name);
            //    Compadre::OptimizationObject opt_obj_3 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, preserve_local_bounds);
            //    r3.setOptimizationObject(opt_obj_3);
            //    rm_right_to_left->add(r3);

            //    Compadre::RemapObject r4("AnalyticalFun2", "AnalyticalFun2", TargetOperation::CellAverageEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, CellAverageSample, PointSample);
            //    r4.setSourceExtraData(extra_data_name);
            //    r4.setTargetExtraData(extra_data_name);
            //    Compadre::OptimizationObject opt_obj_4 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, preserve_local_bounds);
            //    r4.setOptimizationObject(opt_obj_4);
            //    rm_right_to_left->add(r4);

            //} else {
            //    Compadre::RemapObject r0("TotalPrecipWater", "TotalPrecipWater", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
            //    Compadre::OptimizationObject opt_obj_0 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, preserve_local_bounds, 0, 1.0e+15);
            //    r0.setOptimizationObject(opt_obj_0);
            //    rm_right_to_left->add(r0);

            //    Compadre::RemapObject r1("CloudFraction", "CloudFraction", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
            //    Compadre::OptimizationObject opt_obj_1 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, preserve_local_bounds, 0, 1);
            //    r1.setOptimizationObject(opt_obj_1);
            //    rm_right_to_left->add(r1);

            //    Compadre::RemapObject r2("Topography", "Topography", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
            //    Compadre::OptimizationObject opt_obj_2 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, preserve_local_bounds, -10994.0, 8848.0);
            //    r2.setOptimizationObject(opt_obj_2);
            //    rm_right_to_left->add(r2);

            //    Compadre::RemapObject r3("AnalyticalFun1", "AnalyticalFun1", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
            //    Compadre::OptimizationObject opt_obj_3 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, preserve_local_bounds);
            //    r3.setOptimizationObject(opt_obj_3);
            //    rm_right_to_left->add(r3);

            //    Compadre::RemapObject r4("AnalyticalFun2", "AnalyticalFun2", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
            //    Compadre::OptimizationObject opt_obj_4 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, preserve_local_bounds);
            //    r4.setOptimizationObject(opt_obj_4);
            //    rm_right_to_left->add(r4);

            //}
			Teuchos::RCP<Compadre::RemapManager> rm_left_to_right = Teuchos::rcp(new Compadre::RemapManager(parameters, particles_left.getRawPtr(), particles_right.getRawPtr(), halo_size));
			Teuchos::RCP<Compadre::RemapManager> rm_right_to_left = Teuchos::rcp(new Compadre::RemapManager(parameters, particles_right.getRawPtr(), particles_left.getRawPtr(), halo_size));
            std::string extra_data_name = parameters->get<Teuchos::ParameterList>("remap").get<std::string>("extra data field name");

            if (parameters->get<Teuchos::ParameterList>("remap").get<bool>("cell averaged")) {
                Compadre::RemapObject r0("TotalPrecipWater", "TotalPrecipWater", TargetOperation::CellAverageEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, CellAverageSample, PointSample);
                //Compadre::RemapObject r0("TotalPrecipWater", "TotalPrecipWater", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, CellAverageSample, PointSample);
                //Compadre::RemapObject r0("TotalPrecipWater", "TotalPrecipWater", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
                r0.setSourceExtraData(extra_data_name);
                r0.setTargetExtraData(extra_data_name);
                //r0.setNormalDirections("tangent_bundle");
                Compadre::OptimizationObject opt_obj_0 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, preserve_local_bounds, 0);
                r0.setOptimizationObject(opt_obj_0);
                rm_left_to_right->add(r0);
                rm_right_to_left->add(r0);

                Compadre::RemapObject r1("CloudFraction", "CloudFraction", TargetOperation::CellAverageEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, CellAverageSample, PointSample);
                //Compadre::RemapObject r1("CloudFraction", "CloudFraction", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, CellAverageSample, PointSample);
                //Compadre::RemapObject r1("CloudFraction", "CloudFraction", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
                r1.setSourceExtraData(extra_data_name);
                r1.setTargetExtraData(extra_data_name);
                //r1.setNormalDirections("tangent_bundle");
                Compadre::OptimizationObject opt_obj_1 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, preserve_local_bounds, 0, 1);
                r1.setOptimizationObject(opt_obj_1);
                rm_left_to_right->add(r1);
                rm_right_to_left->add(r1);

                Compadre::RemapObject r2("Topography", "Topography", TargetOperation::CellAverageEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, CellAverageSample, PointSample);
                //Compadre::RemapObject r2("Topography", "Topography", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, CellAverageSample, PointSample);
                //Compadre::RemapObject r2("Topography", "Topography", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
                r2.setSourceExtraData(extra_data_name);
                r2.setTargetExtraData(extra_data_name);
                //r2.setNormalDirections("tangent_bundle");
                Compadre::OptimizationObject opt_obj_2 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, preserve_local_bounds, -10994.0, 8848.0);
                r2.setOptimizationObject(opt_obj_2);
                rm_left_to_right->add(r2);
                rm_right_to_left->add(r2);

                Compadre::RemapObject r3("AnalyticalFun1", "AnalyticalFun1", TargetOperation::CellAverageEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, CellAverageSample, PointSample);
                r3.setSourceExtraData(extra_data_name);
                r3.setTargetExtraData(extra_data_name);
                //r3.setNormalDirections("tangent_bundle");
                Compadre::OptimizationObject opt_obj_3 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, preserve_local_bounds);
                r3.setOptimizationObject(opt_obj_3);
                rm_left_to_right->add(r3);
                rm_right_to_left->add(r3);

                Compadre::RemapObject r4("AnalyticalFun2", "AnalyticalFun2", TargetOperation::CellAverageEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, CellAverageSample, PointSample);
                r4.setSourceExtraData(extra_data_name);
                r4.setTargetExtraData(extra_data_name);
                //r4.setNormalDirections("tangent_bundle");
                Compadre::OptimizationObject opt_obj_4 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, preserve_local_bounds);
                r4.setOptimizationObject(opt_obj_4);
                rm_left_to_right->add(r4);
                rm_right_to_left->add(r4);

            } else {
                Compadre::RemapObject r0("TotalPrecipWater", "TotalPrecipWater", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
                //r0.setNormalDirections("tangent_bundle");
                Compadre::OptimizationObject opt_obj_0 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, preserve_local_bounds, 0);
                r0.setOptimizationObject(opt_obj_0);
                rm_left_to_right->add(r0);
                rm_right_to_left->add(r0);

                Compadre::RemapObject r1("CloudFraction", "CloudFraction", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
                //r1.setNormalDirections("tangent_bundle");
                Compadre::OptimizationObject opt_obj_1 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, preserve_local_bounds, 0, 1);
                r1.setOptimizationObject(opt_obj_1);
                rm_left_to_right->add(r1);
                rm_right_to_left->add(r1);

                Compadre::RemapObject r2("Topography", "Topography", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
                //r2.setNormalDirections("tangent_bundle");
                Compadre::OptimizationObject opt_obj_2 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, preserve_local_bounds, -10994.0, 8848.0);
                r2.setOptimizationObject(opt_obj_2);
                rm_left_to_right->add(r2);
                rm_right_to_left->add(r2);

                Compadre::RemapObject r3("AnalyticalFun1", "AnalyticalFun1", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
                //r3.setNormalDirections("tangent_bundle");
                Compadre::OptimizationObject opt_obj_3 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, preserve_local_bounds);
                r3.setOptimizationObject(opt_obj_3);
                rm_left_to_right->add(r3);
                rm_right_to_left->add(r3);

                Compadre::RemapObject r4("AnalyticalFun2", "AnalyticalFun2", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
                //r4.setNormalDirections("tangent_bundle");
                Compadre::OptimizationObject opt_obj_4 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, preserve_local_bounds);
                r4.setOptimizationObject(opt_obj_4);
                rm_left_to_right->add(r4);
                rm_right_to_left->add(r4);
            }

            local_index_type last_step_to_compute = parameters->get<local_index_type>("final step");
            for (local_index_type i=initial_step_to_compute; i<= last_step_to_compute; ++i) {
                printf("step %d ", i);

                rm_left_to_right->execute(true /* keep neighborhoods */, true /* keep GMLS */, true /* reuse neighborhoods */, true /* reuse GMLS */);
                particles_right->getFieldManager()->updateFieldsHaloData();
                particles_left->getFieldManager()->updateFieldsHaloData();
                Kokkos::fence();
                if (i%save_every==0 || i==1) {
                    WriteTime->start();
                    Compadre::FileManager fm2;
                    std::string output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "forward_" + std::to_string(i) + ".g";
                    fm2.setWriter(output_filename, particles_right);
                    fm2.write();
                    WriteTime->stop();
                }
                Kokkos::fence();

                rm_right_to_left->execute(true /* keep neighborhoods */, true /* keep GMLS */, true /* reuse neighborhoods */, true /* reuse GMLS */);
                particles_left->getFieldManager()->updateFieldsHaloData();
                particles_right->getFieldManager()->updateFieldsHaloData();
                Kokkos::fence();
                if (i%save_every==0 || i==1) {
                    WriteTime->start();
                    Compadre::FileManager fm2;
                    std::string output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "backward_" + std::to_string(i) + ".g";
                    fm2.setWriter(output_filename, particles_left);
                    fm2.write();
                    WriteTime->stop();
                }
                Kokkos::fence();

            }
        }
    }

    Teuchos::TimeMonitor::summarize();
    Kokkos::finalize();
return 0;
}
