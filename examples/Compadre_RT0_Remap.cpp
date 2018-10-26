#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_Core.hpp>

#include <Kokkos_Core.hpp>

#include <GMLS.hpp> // for getNP()
#include <Compadre_GlobalConstants.hpp>
#include <Compadre_ProblemT.hpp>
#include <Compadre_ParticlesT.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_DOFManager.hpp>
#include <Compadre_EuclideanCoordsT.hpp>
#include <Compadre_SphericalCoordsT.hpp>
#ifdef COMPADRE_USE_NANOFLANN
#include <Compadre_nanoflannInformation.hpp>
#endif
#include <Compadre_FieldT.hpp>
#include <Compadre_XyzVector.hpp>
#include <Compadre_AnalyticFunctions.hpp>
#include <Compadre_FileIO.hpp>
#include <Compadre_ParameterManager.hpp>
#include <Compadre_RemapManager.hpp>

#include "Compadre_GlobalConstants.hpp"
static const Compadre::GlobalConstants consts;

#include <iostream>

typedef int LO;
typedef long GO;
typedef double ST;
typedef Compadre::XyzVector xyz_type;

static const ST PI = consts.Pi();

using namespace Compadre;

int main (int argc, char* args[]) {
#ifdef TRILINOS_LINEAR_SOLVES

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

	Kokkos::initialize(argc, args);
	
	const int procRank = comm->getRank();
	const int nProcs = comm->getSize();

	Teuchos::RCP<Teuchos::Time> SphericalParticleTime = Teuchos::TimeMonitor::getNewCounter ("Spherical Particle Time");
	Teuchos::RCP<Teuchos::Time> FirstReadTime = Teuchos::TimeMonitor::getNewCounter ("1st Read Time");
	Teuchos::RCP<Teuchos::Time> SecondReadTime = Teuchos::TimeMonitor::getNewCounter ("2nd Read Time");
	Teuchos::RCP<Teuchos::Time> AssemblyTime = Teuchos::TimeMonitor::getNewCounter ("Assembly Time");
	Teuchos::RCP<Teuchos::Time> SolvingTime = Teuchos::TimeMonitor::getNewCounter ("Solving Time");
	Teuchos::RCP<Teuchos::Time> WriteTime = Teuchos::TimeMonitor::getNewCounter ("Write Time");
	Teuchos::RCP<Teuchos::Time> NormTime = Teuchos::TimeMonitor::getNewCounter ("Norm calculation");

	{
		//
		// VTK File Reader Test and Parallel VTK File Reader
		//

		typedef Compadre::EuclideanCoordsT CT;
		Teuchos::RCP<Compadre::ParticlesT> particles =
				Teuchos::rcp( new Compadre::ParticlesT(parameters, comm));
	 	CT* coords = (CT*)particles->getCoords();

		//Read in data file.
		FirstReadTime->start();
		Compadre::FileManager fm;
		std::string testfilename(parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file prefix") + parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file"));
		fm.setReader(testfilename, particles);
		fm.read();
		FirstReadTime->stop();

		ST halo_size;
		{
			const ST h_size = .0833333;

			particles->zoltan2Initialize(); // also applies the repartition to coords, flags, and fields
			if (parameters->get<Teuchos::ParameterList>("halo").get<bool>("dynamic")) {
				halo_size = h_size * parameters->get<Teuchos::ParameterList>("halo").get<double>("multiplier");
			} else {
				halo_size = parameters->get<Teuchos::ParameterList>("halo").get<double>("size");
			}
			particles->buildHalo(halo_size);
	 		particles->getFieldManager()->listFields(std::cout);
			particles->getFieldManager()->updateFieldsHaloData();

			//
			//
			// REMAP TO TEST OPERATORS
			//
			//
		 	Teuchos::RCP<Compadre::ParticlesT> new_particles =
		 		Teuchos::rcp( new Compadre::ParticlesT(parameters, comm));
		 	CT* new_coords = (CT*)new_particles->getCoords();

			const ST h_support = parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("size");

			//Remap
			Teuchos::RCP<Compadre::RemapManager> rm = Teuchos::rcp(new Compadre::RemapManager(parameters, particles.getRawPtr(), new_particles.getRawPtr(), halo_size));

			new_particles->getFieldManager()->createField(1, "computedSphereHarmonic", "m/s");
			new_particles->getFieldManager()->createField(1, "exact_solution", "m/s");

			Compadre::RemapObject ro("u", "computedSphereHarmonic", ReconstructionOperator::TargetOperation::LaplacianOfScalarPointEvaluation);
			rm->add(ro);
			rm->execute();

		 	// point by point check to see how far off from correct value
			// check solution

			NormTime->start();

			Teuchos::RCP<Compadre::AnalyticFunction> function;
			function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::SphereTestVelocity()));

			ST physical_coordinate_weighted_l2_norm = 0;
		 	ST exact_coordinate_weighted_l2_norm = 0;

			Compadre::host_view_type solution_field;
			solution_field = new_particles->getFieldManager()->getFieldByName("computedGradSphereHarmonic")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();
			Compadre::host_view_type exact_solution_field = new_particles->getFieldManager()->getFieldByName("exact_solution")->getMultiVectorPtr()->getLocalView<Compadre::host_view_type>();

			for(int j=0; j<coords->nLocal(); j++){
				xyz_type xyz = coords->getLocalCoords(j);
				xyz_type vector_exact = function->evalScalarDerivative(xyz);

				exact_solution_field(j,0) = vector_exact.x;
				exact_solution_field(j,1) = vector_exact.y;
				exact_solution_field(j,2) = vector_exact.z;
			}

			WriteTime->start();
			{
				std::string output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file prefix") + parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file");
				Compadre::FileManager fm;
				fm.setWriter(output_filename, new_particles);
				fm.write();

			}
			WriteTime->stop();


			double exact = 0;
			GO num_solved_for = 0;
			for(int j=0; j<coords->nLocal(); j++){
				if 	(particles->getFlag(j)==0) {
					xyz_type xyz = coords->getLocalCoords(j);
					exact = function->evalScalarDerivative(xyz).x;
					physical_coordinate_weighted_l2_norm += (solution_field(j,0) - exact)*(solution_field(j,0) - exact);//*grid_area_field(j,0);
					physical_coordinate_weighted_l2_norm += (solution_field(j,1) - function->evalScalarDerivative(xyz).y)*(solution_field(j,1) - function->evalScalarDerivative(xyz).y);
					physical_coordinate_weighted_l2_norm += (solution_field(j,2) - function->evalScalarDerivative(xyz).z)*(solution_field(j,2) - function->evalScalarDerivative(xyz).z);

					exact_coordinate_weighted_l2_norm += exact*exact;//*grid_area_field(j,0);
					num_solved_for++;
				}
			}

			// get global # of dofs solved for
			GO global_num_solved_for = 0;
			Teuchos::Ptr<GO> global_num_solved_for_ptr(&global_num_solved_for);
			Teuchos::reduceAll<int, GO>(*comm, Teuchos::REDUCE_SUM, num_solved_for, global_num_solved_for_ptr);


			physical_coordinate_weighted_l2_norm /= global_num_solved_for;

			ST global_norm_physical_coord = 0;
			ST global_norm_exact_coord = 0;

			Teuchos::Ptr<ST> global_norm_ptr_physical_coord(&global_norm_physical_coord);
			Teuchos::Ptr<ST> global_norm_ptr_exact_coord(&global_norm_exact_coord);

			Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_SUM, physical_coordinate_weighted_l2_norm, global_norm_ptr_physical_coord);
			Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_SUM, exact_coordinate_weighted_l2_norm, global_norm_ptr_exact_coord);

			global_norm_physical_coord = std::sqrt(global_norm_physical_coord);
			if (comm->getRank()==0)
				printf("\nGlobal Norm: %.16f\n\n\n\n\n", global_norm_physical_coord);

//			global_norm_exact_coord = std::max(global_norm_exact_coord, 1e-15);
//			if (comm->getRank()==0) std::cout << "Global Norm Physical Coordinates: " << std::sqrt(global_norm_physical_coord) << " " << std::sqrt(global_norm_physical_coord) /  std::sqrt(global_norm_exact_coord)  << "\n\n\n\n\n";
//			if (comm->getRank()==0) std::cout << "Global Norm Physical Coordinates: " << std::sqrt(global_norm_physical_coord) /  std::sqrt(global_norm_exact_coord)  << "\n\n\n\n\n";

			double fail_threshold = 0;
			try {
				fail_threshold = parameters->get<double>("fail threshold");
			} catch (...) {
			}

			std::ostringstream msg;
			msg << "Error [" << global_norm_physical_coord << "] larger than acceptable [" << fail_threshold << "].\n";
			if (fail_threshold > 0) TEUCHOS_TEST_FOR_EXCEPT_MSG(global_norm_physical_coord > fail_threshold, msg.str());

//			TEUCHOS_TEST_FOR_EXCEPT_MSG(global_norm_physical_coord > 1e-15, "Physical coordinate error too large (should be exact).");
			NormTime->stop();

		}
	}

	if (comm->getRank()==0) parameters->print();
	Teuchos::TimeMonitor::summarize();
	Kokkos::finalize();
#endif
return 0;
}


