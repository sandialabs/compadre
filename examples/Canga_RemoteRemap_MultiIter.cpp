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
#include <Compadre_RemoteDataManager.hpp>

#include <iostream>

typedef int LO;
typedef long GO;
typedef double ST;
typedef Compadre::XyzVector xyz_type;

using namespace Compadre;

int main (int argc, char* args[]) {

    Teuchos::GlobalMPISession mpi(&argc, &args);
    Teuchos::oblackholestream bstream;
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

    const LO my_coloring = parameters->get<LO>("my coloring");
    const LO peer_coloring = parameters->get<LO>("peer coloring");

    Teuchos::RCP<const Teuchos::Comm<int> > comm = global_comm->split(my_coloring, global_comm->getRank());

    {

        {
            //
            // Remote Data Test (Two differing distributions of particles)
            //

            typedef Compadre::EuclideanCoordsT CT;
             Teuchos::RCP<Compadre::ParticlesT> particles =
                 Teuchos::rcp( new Compadre::ParticlesT(parameters, comm));
             CT* coords = (CT*)particles->getCoords();

            //Read in data file.
            ReadTime->start();
            Compadre::FileManager fm;
            std::string testfilename;
            if (initial_step_to_compute==1) {
                testfilename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file prefix") + parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file");
            } else {
                testfilename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "backward_" + std::to_string(initial_step_to_compute-1) + ".pvtp";
            }
            fm.setReader(testfilename, particles);
            fm.getReader()->setCoordinateLayout(parameters->get<Teuchos::ParameterList>("io").get<std::string>("particles number dimension name"),
                                                parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinates layout"),
                                                parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate 0 name"),
                                                parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate 1 name"),
                                                parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate 2 name"));
            fm.getReader()->setCoordinateUnitStyle(parameters->get<Teuchos::ParameterList>("io").get<std::string>("lat lon units"));
            fm.getReader()->setKeepOriginalCoordinates(parameters->get<Teuchos::ParameterList>("io").get<bool>("keep original lat lon"));
            fm.read();
            ReadTime->stop();











    //// temporary data for testing purposes
    //if (my_coloring == 25) {
    //    particles->getFieldManager()->createField(3,"extra_data_name","m^2/1a");
    //} else {
    //    particles->getFieldManager()->createField(5,"extra_data_name","m^2/1a");
    //}
    //auto vals = particles->getFieldManager()->getFieldByName("extra_data_name")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();
    //if (my_coloring == 25) {
    //    for (size_t i=0; i<vals.extent(0); ++i) {
    //        for (size_t j=0; j<vals.extent(1); ++j) {
    //            vals(i,j) = j;
    //        }
    //    }
    //} else {
    //    for (size_t i=0; i<vals.extent(0); ++i) {
    //        for (size_t j=0; j<vals.extent(1); ++j) {
    //            vals(i,j) = 10+j;
    //        }
    //    }
    //}
    

    


            // initially,
            // my_coloring==25 already has CloudFraction, TotalPrecipWater, and Topography
            // my_coloring==33 has none
            // nice to get exact versions of these and store them as exact
            if (my_coloring == 25) {
                if (initial_step_to_compute==1) {
                    particles->getFieldManager()->createField(1,"exact CloudFraction","m^2/1a");
                    particles->getFieldManager()->createField(1,"exact TotalPrecipWater","m^2/1a");
                    particles->getFieldManager()->createField(1,"exact Topography","m^2/1a");
                    Kokkos::deep_copy(particles->getFieldManager()->getFieldByName("exact CloudFraction")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>(),
                        particles->getFieldManager()->getFieldByName("CloudFraction")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>());
                    Kokkos::deep_copy(particles->getFieldManager()->getFieldByName("exact TotalPrecipWater")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>(),
                        particles->getFieldManager()->getFieldByName("TotalPrecipWater")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>());
                    Kokkos::deep_copy(particles->getFieldManager()->getFieldByName("exact Topography")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>(),
                        particles->getFieldManager()->getFieldByName("Topography")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>());
                }
                // else exact data already is created in the particle set
            }

             // calculate h_size for mesh
             // build halo data
            ST halo_size = parameters->get<Teuchos::ParameterList>("halo").get<double>("size") *
                    ( parameters->get<Teuchos::ParameterList>("halo").get<double>("multiplier")
                            + (ST)(parameters->get<Teuchos::ParameterList>("remap").get<int>("porder")));
            particles->zoltan2Initialize();
            particles->buildHalo(halo_size);
            particles->getFieldManager()->updateFieldsHaloData();

            // remote
            Teuchos::RCP<Compadre::RemoteDataManager> remoteDataManager =
                    Teuchos::rcp(new Compadre::RemoteDataManager(global_comm, comm, coords, my_coloring, peer_coloring));
             Teuchos::RCP<Compadre::ParticlesT> peer_processors_particles =
                 Teuchos::rcp( new Compadre::ParticlesT(parameters, comm));
             remoteDataManager->putRemoteCoordinatesInParticleSet(peer_processors_particles.getRawPtr());

             if (parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm")!="NONE") {
                 std::string target_weights_name = parameters->get<Teuchos::ParameterList>("remap").get<std::string>("target weighting field name");
                 std::string source_weights_name = parameters->get<Teuchos::ParameterList>("remap").get<std::string>("source weighting field name");
                 TEUCHOS_TEST_FOR_EXCEPT_MSG(target_weights_name=="", "\"target weighting field name\" not specified in parameter list.");
                 TEUCHOS_TEST_FOR_EXCEPT_MSG(source_weights_name=="", "\"source weighting field name\" not specified in parameter list.");
                 remoteDataManager->putRemoteWeightsInParticleSet(particles.getRawPtr(), peer_processors_particles.getRawPtr(), target_weights_name);
             }

            if (parameters->get<Teuchos::ParameterList>("remap").get<bool>("cell averaged")) {
                // adds in cell data for the peers (target sites), allowing calculation of tau
                 std::string extra_data_name = parameters->get<Teuchos::ParameterList>("remap").get<std::string>("extra data field name");
                 remoteDataManager->putExtraRemapDataInParticleSet(particles.getRawPtr(), peer_processors_particles.getRawPtr(), extra_data_name);
            }

            local_index_type last_step_to_compute = parameters->get<local_index_type>("final step");

            for (local_index_type i=initial_step_to_compute; i<= last_step_to_compute; ++i) {

                // first the forward pass from 25->33
                std::vector<Compadre::RemapObject> remap_vec;
                if (my_coloring == 25) {
                    // no field remapped to here, only sent
                } else if (my_coloring == 33) {
                    if (parameters->get<Teuchos::ParameterList>("remap").get<bool>("cell averaged")) {
                         std::string extra_data_name = parameters->get<Teuchos::ParameterList>("remap").get<std::string>("extra data field name");
                        Compadre::RemapObject r0("TotalPrecipWater", "TotalPrecipWater", TargetOperation::ScalarFaceAverageEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, ScalarFaceAverageSample, PointSample);
                        //Compadre::RemapObject r0("TotalPrecipWater", "TotalPrecipWater", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
                        r0.setSourceExtraData(extra_data_name);
                        Compadre::OptimizationObject opt_obj_0 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, true /*bounds preservation*/, 0, 1.0e+15);
                        r0.setOptimizationObject(opt_obj_0);
                        remap_vec.push_back(r0);

                        Compadre::RemapObject r1("CloudFraction", "CloudFraction", TargetOperation::ScalarFaceAverageEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, ScalarFaceAverageSample, PointSample);
                        //Compadre::RemapObject r1("CloudFraction", "CloudFraction", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
                        r1.setSourceExtraData(extra_data_name);
                        Compadre::OptimizationObject opt_obj_1 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, true /*bounds preservation*/, 0, 1);
                        r1.setOptimizationObject(opt_obj_1);
                        remap_vec.push_back(r1);

                        Compadre::RemapObject r2("Topography", "Topography", TargetOperation::ScalarFaceAverageEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, ScalarFaceAverageSample, PointSample);
                        //Compadre::RemapObject r2("Topography", "Topography", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
                        r2.setSourceExtraData(extra_data_name);
                        Compadre::OptimizationObject opt_obj_2 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, true /*bounds preservation*/, -1.0e+15, 1.0e+15);
                        r2.setOptimizationObject(opt_obj_2);
                        remap_vec.push_back(r2);
                    } else {
                        Compadre::RemapObject r0("TotalPrecipWater", "TotalPrecipWater", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
                        Compadre::OptimizationObject opt_obj_0 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, true /*bounds preservation*/, 0, 1.0e+15);
                        r0.setOptimizationObject(opt_obj_0);
                        remap_vec.push_back(r0);

                        Compadre::RemapObject r1("CloudFraction", "CloudFraction", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
                        Compadre::OptimizationObject opt_obj_1 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, true /*bounds preservation*/, 0, 1);
                        r1.setOptimizationObject(opt_obj_1);
                        remap_vec.push_back(r1);

                        Compadre::RemapObject r2("Topography", "Topography", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
                        Compadre::OptimizationObject opt_obj_2 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, true /*bounds preservation*/, -1.0e+15, 1.0e+15);
                        r2.setOptimizationObject(opt_obj_2);
                        remap_vec.push_back(r2);
                    }
                }

                remoteDataManager->remapData(remap_vec, parameters, particles.getRawPtr(), peer_processors_particles.getRawPtr(), halo_size, false /* use physical coords*/, true /* keep neighborhoods*/);
                // data is now on 33
                if (my_coloring == 33) {
                    WriteTime->start();
                    std::string output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "forward_" + std::to_string(i) + ".pvtp";
                    Compadre::FileManager fm2;
                    fm2.setWriter(output_filename, particles);
                    fm2.write();
                    output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "forward_" + std::to_string(i) + ".pvtp.g";
                    fm2.setWriter(output_filename, particles);
                    fm2.write();
                    WriteTime->stop();
                }

                // now the backwards pass from 33->25
                remap_vec = std::vector<Compadre::RemapObject>();
                if (my_coloring == 33) {
                    // no field remapped to here, only sent
                } else if (my_coloring == 25) {
                    if (parameters->get<Teuchos::ParameterList>("remap").get<bool>("cell averaged")) {
                         std::string extra_data_name = parameters->get<Teuchos::ParameterList>("remap").get<std::string>("extra data field name");
                        Compadre::RemapObject r0("TotalPrecipWater", "TotalPrecipWater", TargetOperation::ScalarFaceAverageEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, ScalarFaceAverageSample, PointSample);
                        //Compadre::RemapObject r0("TotalPrecipWater", "TotalPrecipWater", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
                        r0.setSourceExtraData(extra_data_name);
                        Compadre::OptimizationObject opt_obj_0 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, true /*bounds preservation*/, 0, 1.0e+15);
                        r0.setOptimizationObject(opt_obj_0);
                        remap_vec.push_back(r0);

                        Compadre::RemapObject r1("CloudFraction", "CloudFraction", TargetOperation::ScalarFaceAverageEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, ScalarFaceAverageSample, PointSample);
                        //Compadre::RemapObject r1("CloudFraction", "CloudFraction", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
                        r1.setSourceExtraData(extra_data_name);
                        Compadre::OptimizationObject opt_obj_1 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, true /*bounds preservation*/, 0, 1);
                        r1.setOptimizationObject(opt_obj_1);
                        remap_vec.push_back(r1);

                        Compadre::RemapObject r2("Topography", "Topography", TargetOperation::ScalarFaceAverageEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, ScalarFaceAverageSample, PointSample);
                        //Compadre::RemapObject r2("Topography", "Topography", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
                        r2.setSourceExtraData(extra_data_name);
                        Compadre::OptimizationObject opt_obj_2 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, true /*bounds preservation*/, -1.0e+15, 1.0e+15);
                        r2.setOptimizationObject(opt_obj_2);
                        remap_vec.push_back(r2);
                    } else {
                        Compadre::RemapObject r0("TotalPrecipWater", "TotalPrecipWater", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
                        Compadre::OptimizationObject opt_obj_0 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, true /*bounds preservation*/, 0, 1.0e+15);
                        r0.setOptimizationObject(opt_obj_0);
                        remap_vec.push_back(r0);

                        Compadre::RemapObject r1("CloudFraction", "CloudFraction", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
                        Compadre::OptimizationObject opt_obj_1 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, true /*bounds preservation*/, 0, 1);
                        r1.setOptimizationObject(opt_obj_1);
                        remap_vec.push_back(r1);

                        Compadre::RemapObject r2("Topography", "Topography", TargetOperation::ScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);
                        Compadre::OptimizationObject opt_obj_2 = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, true /*bounds preservation*/, -1.0e+15, 1.0e+15);
                        r2.setOptimizationObject(opt_obj_2);
                        remap_vec.push_back(r2);
                    }
                }

                remoteDataManager->remapData(remap_vec, parameters, particles.getRawPtr(), peer_processors_particles.getRawPtr(), halo_size, false /* use physical coords*/, true /* keep neighborhoods*/);
                // data is now on 25
                if (my_coloring == 25) {
                    WriteTime->start();
                    std::string output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "backward_" + std::to_string(i) + ".pvtp";
                    Compadre::FileManager fm2;
                    fm2.setWriter(output_filename, particles);
                    fm2.write();
                    output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "backward_" + std::to_string(i) + ".pvtp.g";
                    fm2.setWriter(output_filename, particles);
                    fm2.write();
                    WriteTime->stop();
                }
            }
        }
    }

    Teuchos::TimeMonitor::summarize();
    Kokkos::finalize();
return 0;
}
