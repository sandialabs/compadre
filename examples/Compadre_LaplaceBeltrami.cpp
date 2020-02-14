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

#include <Compadre_LaplaceBeltrami_Operator.hpp>
#include <Compadre_LaplaceBeltrami_Sources.hpp>
#include <Compadre_LaplaceBeltrami_BoundaryConditions.hpp>

#include "Compadre_GlobalConstants.hpp"
static const Compadre::GlobalConstants consts;

#include <iostream>

#define STACK_TRACE(call) try { call; } catch (const std::exception& e ) { TEUCHOS_TRACE(e); }

typedef int LO;
typedef long GO;
typedef double ST;
typedef Compadre::XyzVector xyz_type;

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

	Teuchos::RCP<Teuchos::Time> SphericalParticleTime = Teuchos::TimeMonitor::getNewCounter ("Spherical Particle Time");
	Teuchos::RCP<Teuchos::Time> NeighborSearchTime = Teuchos::TimeMonitor::getNewCounter ("Laplace Beltrami - Neighbor Search Time");
	Teuchos::RCP<Teuchos::Time> FirstReadTime = Teuchos::TimeMonitor::getNewCounter ("1st Read Time");
	Teuchos::RCP<Teuchos::Time> SecondReadTime = Teuchos::TimeMonitor::getNewCounter ("2nd Read Time");
	Teuchos::RCP<Teuchos::Time> AssemblyTime = Teuchos::TimeMonitor::getNewCounter ("Assembly Time");
	Teuchos::RCP<Teuchos::Time> SolvingTime = Teuchos::TimeMonitor::getNewCounter ("Solving Time");
	Teuchos::RCP<Teuchos::Time> WriteTime = Teuchos::TimeMonitor::getNewCounter ("Write Time");
	Teuchos::RCP<Teuchos::Time> NormTime = Teuchos::TimeMonitor::getNewCounter ("Norm calculation");

	{


		typedef Compadre::EuclideanCoordsT CT;
		Teuchos::RCP<Compadre::ParticlesT> particles =
				Teuchos::rcp( new Compadre::ParticlesT(parameters, comm));
	 	CT* coords = (CT*)particles->getCoords();

		//Read in data file.
		FirstReadTime->start();
		Compadre::FileManager fm;
		std::string testfilename(parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file prefix") + parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file"));
		fm.setReader(testfilename, particles, nc_type);
        fm.getReader()->setCoordinateLayout(parameters->get<Teuchos::ParameterList>("io").get<std::string>("particles number dimension name"),
                                            parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinates layout"),
                                            parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate 0 name"),
                                            parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate 1 name"),
                                            parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate 2 name"));
        fm.getReader()->setCoordinateUnitStyle(parameters->get<Teuchos::ParameterList>("io").get<std::string>("lat lon units"));
        fm.getReader()->setKeepOriginalCoordinates(parameters->get<Teuchos::ParameterList>("io").get<bool>("keep original lat lon"));

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


			if (parameters->get<std::string>("solution type") == "lb solve") {
				particles->getFieldManager()->createField(1, "solution", "m/s");
				particles->getFieldManager()->createField(1, "lagrange multiplier", "NA");
				Compadre::ConstantEachDimension function1 = Compadre::ConstantEachDimension(1,1,1);
				particles->getFieldManager()->getFieldByName("solution")->
						localInitFromVectorFunction(&function1);
				particles->getFieldManager()->getFieldByName("lagrange multiplier")->
						localInitFromVectorFunction(&function1);

			} else if (parameters->get<std::string>("solution type") == "five_strip") {
				particles->getFieldManager()->createField(1, "solution", "m/s");
				particles->getFieldManager()->createField(1, "lagrange multiplier", "NA");
				Compadre::ConstantEachDimension function1 = Compadre::ConstantEachDimension(1,1,1);
				particles->getFieldManager()->getFieldByName("solution")->
						localInitFromVectorFunction(&function1);
				particles->getFieldManager()->getFieldByName("lagrange multiplier")->
						localInitFromVectorFunction(&function1);
//				Compadre::FiveStripOnSphere function1 = Compadre::FiveStripOnSphere();
//				particles->getFieldManager()->getFieldByName("solution")->
//					localInitFromScalarFunction(&function1);

			} else {
				if (parameters->get<int>("physics number") < 3) {
					particles->getFieldManager()->createField(1, "scaledSphereHarmonic", "m/s");
					Compadre::SphereHarmonic function2 = Compadre::SphereHarmonic(4,5);
					particles->getFieldManager()->getFieldByName("scaledSphereHarmonic")->
							localInitFromScalarFunction(&function2);
					particles->getFieldManager()->getFieldByName("scaledSphereHarmonic")->
							scale(1.0 / (5 * (5 + 1)));
				} else if (parameters->get<int>("physics number") == 3) {
					particles->getFieldManager()->createField(1, "scaledSphereHarmonic", "m/s");
					Compadre::FiveStripOnSphere function2 = Compadre::FiveStripOnSphere();
					particles->getFieldManager()->getFieldByName("scaledSphereHarmonic")->
							localInitFromScalarFunction(&function2);
				} else if (parameters->get<int>("physics number") == 10) {
					particles->getFieldManager()->createField(1, "scaledSphereHarmonic", "m/s");
					Compadre::CylinderSinLonCosZ function2 = Compadre::CylinderSinLonCosZ();
//					Compadre::ConstantEachDimension function2 = Compadre::ConstantEachDimension(1,1,1);
					particles->getFieldManager()->getFieldByName("scaledSphereHarmonic")->
							localInitFromScalarFunction(&function2);
				}

				if (parameters->get<std::string>("solution type") == "div" || parameters->get<std::string>("solution type") == "staggered_div" || parameters->get<std::string>("solution type") == "vector") {
					particles->getFieldManager()->createField(3, "sourceGradientSphereHarmonic", "m/s");
					Compadre::SphereHarmonic function3 = Compadre::SphereHarmonic(4,5);
					particles->getFieldManager()->getFieldByName("sourceGradientSphereHarmonic")->
							localInitFromScalarFunctionGradient(&function3);
				}
			}
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


			if (parameters->get<std::string>("solution type") == "lb solve" || parameters->get<std::string>("solution type") == "five_strip") {

				NeighborSearchTime->start();
				particles->createDOFManager();
				particles->getDOFManager()->generateDOFMap();

				particles->createNeighborhood();
				particles->getNeighborhood()->setAllHSupportSizes(h_support);
				LO neighbors_needed = Compadre::GMLS::getNP(std::max(parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"),
                           parameters->get<Teuchos::ParameterList>("remap").get<int>("curvature porder")), 2);
 		        particles->getNeighborhood()->constructAllNeighborLists(particles->getCoordsConst()->getHaloSize(),
                    parameters->get<Teuchos::ParameterList>("neighborhood").get<std::string>("search type"),
                    true /*dry run for sizes*/,
                    neighbors_needed,
                    parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("cutoff multiplier"),
                    parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("size"),
                    parameters->get<Teuchos::ParameterList>("neighborhood").get<bool>("uniform radii"),
                    parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("radii post search scaling"));
				NeighborSearchTime->stop();

			}

			if ((parameters->get<std::string>("solution type") != "lb solve" && parameters->get<std::string>("solution type") != "five_strip") || post_process_grad) {

				//Read in data file.
				SecondReadTime->start();
				Compadre::FileManager fm2;
				std::string testfilename(parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file prefix") + parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file"));
				fm2.setReader(testfilename, new_particles, nc_type);
                fm2.getReader()->setCoordinateLayout(parameters->get<Teuchos::ParameterList>("io").get<std::string>("particles number dimension name"),
                                                    parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinates layout"),
                                                    parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate 0 name"),
                                                    parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate 1 name"),
                                                    parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate 2 name"));
                fm2.getReader()->setCoordinateUnitStyle(parameters->get<Teuchos::ParameterList>("io").get<std::string>("lat lon units"));
                fm2.getReader()->setKeepOriginalCoordinates(parameters->get<Teuchos::ParameterList>("io").get<bool>("keep original lat lon"));
				fm2.read();
				SecondReadTime->stop();

			}

			//Remap or Solve
			Teuchos::RCP<Compadre::RemapManager> rm = Teuchos::rcp(new Compadre::RemapManager(parameters, particles.getRawPtr(), new_particles.getRawPtr(), halo_size));
//			Compadre::RemapObject ro1("scaledSphereHarmonic", "computedSphereHarmonic", TargetOperation::PointEvaluation, StaggeredEdgeAnalyticGradientIntegralSample);
//			Compadre::RemapObject ro2("scaledSphereHarmonic", "computedSphereHarmonic", TargetOperation::PointEvaluation, StaggeredEdgeIntegralSample);
//			Compadre::RemapObject ro3("scaledSphereHarmonic", "computedSphereHarmonic", TargetOperation::PointEvaluation, PointSample);
//			Compadre::RemapObject ro4("scaledSphereHarmonic", "computedSphereHarmonic", TargetOperation::PointEvaluation, StaggeredEdgeIntegralSamplestaggered_div_gradompadre::RemapObject ro5("scaledSphereHarmonic", "computedSphereHarmonic", TargetOperation::PointEvaluation, PointSample);
//
//			rm->add(ro1);
//			rm->add(ro2);
//			rm->add(ro3);
//			rm->add(ro4);
//			rm->add(ro5);
//
//			std::cout << rm->queueToString() << std::endl;
//
//			rm->clear();
			if (parameters->get<std::string>("solution type")=="point") {
				new_particles->getFieldManager()->createField(1, "computedSphereHarmonic", "m/s");
				new_particles->getFieldManager()->createField(1, "exact_solution", "m/s");

				Compadre::RemapObject ro("scaledSphereHarmonic", "computedSphereHarmonic", TargetOperation::ScalarPointEvaluation);
				rm->add(ro);
				STACK_TRACE(rm->execute());
			} else if (parameters->get<std::string>("solution type")=="vector") {
				new_particles->getFieldManager()->createField(3, "computedGradSphereHarmonic", "m/s");
				new_particles->getFieldManager()->createField(3, "exact_solution", "m/s");

				//Compadre::RemapObject ro("sourceGradientSphereHarmonic", "computedGradSphereHarmonic", TargetOperation::VectorPointEvaluation, ReconstructionSpace::VectorTaylorPolynomial, ManifoldVectorPointSample);
				Compadre::RemapObject ro("sourceGradientSphereHarmonic", "computedGradSphereHarmonic", TargetOperation::VectorPointEvaluation, ReconstructionSpace::VectorOfScalarClonesTaylorPolynomial, ManifoldVectorPointSample);
				rm->add(ro);
				STACK_TRACE(rm->execute());
			} else if (parameters->get<std::string>("solution type")=="laplace") {
				new_particles->getFieldManager()->createField(1, "computedSphereHarmonic", "m/s");
				new_particles->getFieldManager()->createField(1, "exact_solution", "m/s");

				Compadre::RemapObject ro("scaledSphereHarmonic", "computedSphereHarmonic", TargetOperation::LaplacianOfScalarPointEvaluation);
				rm->add(ro);
				STACK_TRACE(rm->execute());
			} else if (parameters->get<std::string>("solution type")=="staggered_laplace") {
				new_particles->getFieldManager()->createField(1, "computedSphereHarmonic", "m/s");
				new_particles->getFieldManager()->createField(1, "exact_solution", "m/s");

				Compadre::RemapObject ro("scaledSphereHarmonic", "computedSphereHarmonic", TargetOperation::ChainedStaggeredLaplacianOfScalarPointEvaluation, ReconstructionSpace::VectorTaylorPolynomial, StaggeredEdgeIntegralSample, StaggeredEdgeAnalyticGradientIntegralSample);
//				Compadre::RemapObject ro("scaledSphereHarmonic", "computedSphereHarmonic", TargetOperation::ChainedStaggeredLaplacianOfScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, StaggeredEdgeAnalyticGradientIntegralSample, StaggeredEdgeAnalyticGradientIntegralSample);
				rm->add(ro);
				STACK_TRACE(rm->execute());
			} else if (parameters->get<std::string>("solution type")=="staggered_div_grad") {
				new_particles->getFieldManager()->createField(3, "computedGradSphereHarmonic", "m/s");
//				new_particles->getFieldManager()->createField(1, "computedLaplacianSphereHarmonic", "m/s");
//				new_particles->getFieldManager()->createField(1, "computedDivGradSphereHarmonic", "m/s");

				new_particles->getFieldManager()->createField(3, "exact_solution", "m/s");
				new_particles->getFieldManager()->createField(3, "solution", "m/s");

				Compadre::RemapObject ro("scaledSphereHarmonic", "computedGradSphereHarmonic", TargetOperation::GradientOfScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, StaggeredEdgeAnalyticGradientIntegralSample);
//				Compadre::RemapObject ro1("sourceGradientSphereHarmonic", "computedLaplacianSphereHarmonic", DivergenceOfVectorPointEvaluation, StaggeredEdgeIntegralSample);

				rm->add(ro);
//				rm->add(ro1);

				STACK_TRACE(rm->execute());

//				Compadre::SphereHarmonic function3 = Compadre::SphereHarmonic(4,5);
//				new_particles->getFieldManager()->getFieldByName("computedGradSphereHarmonic")->
//						localInitFromScalarFunctionGradient(&function3);

			} else if (parameters->get<std::string>("solution type")=="grad") {
//				new_particles->getFieldManager()->createField(3, "computedGradSphereHarmonic", "m/s");

//				new_particles->getFieldManager()->createField(1, "computedGradSphereHarmonicX", "m/s");
//				new_particles->getFieldManager()->createField(1, "computedGradSphereHarmonicY", "m/s");
//				new_particles->getFieldManager()->createField(1, "computedGradSphereHarmonicZ", "m/s");
				new_particles->getFieldManager()->createField(3, "computedGradSphereHarmonic", "m/s");

				new_particles->getFieldManager()->createField(3, "solution", "m/s");
				new_particles->getFieldManager()->createField(3, "exact_solution", "m/s");

//				Compadre::RemapObject ro("scaledSphereHarmonic", "computedGradSphereHarmonicX", TargetOperation::PartialXOfScalarPointEvaluation);
//				Compadre::RemapObject ro2("scaledSphereHarmonic", "computedGradSphereHarmonicY", TargetOperation::PartialYOfScalarPointEvaluation);
//				Compadre::RemapObject ro3("scaledSphereHarmonic", "computedGradSphereHarmonicZ", TargetOperation::PartialZOfScalarPointEvaluation);
				Compadre::RemapObject ro("scaledSphereHarmonic", "computedGradSphereHarmonic", TargetOperation::GradientOfScalarPointEvaluation);

//				Compadre::RemapObject ro("scaledSphereHarmonic", "computedGradSphereHarmonicX", TargetOperation::PartialXOfScalarPointEvaluation, StaggeredEdgeAnalyticGradientIntegralSample);
//				Compadre::RemapObject ro2("scaledSphereHarmonic", "computedGradSphereHarmonicY", TargetOperation::PartialYOfScalarPointEvaluation, StaggeredEdgeAnalyticGradientIntegralSample);
//				Compadre::RemapObject ro3("scaledSphereHarmonic", "computedGradSphereHarmonicZ", TargetOperation::PartialZOfScalarPointEvaluation, StaggeredEdgeAnalyticGradientIntegralSample);

//				Compadre::RemapObject ro("scaledSphereHarmonic", "computedGradSphereHarmonicX", TargetOperation::PartialXOfScalarPointEvaluation, StaggeredEdgeIntegralSample);
//				Compadre::RemapObject ro2("scaledSphereHarmonic", "computedGradSphereHarmonicY", TargetOperation::PartialYOfScalarPointEvaluation, StaggeredEdgeIntegralSample);
//				Compadre::RemapObject ro3("scaledSphereHarmonic", "computedGradSphereHarmonicZ", TargetOperation::PartialZOfScalarPointEvaluation, StaggeredEdgeIntegralSample);

				rm->add(ro);
//				rm->add(ro2);
//				rm->add(ro3);

//				std::cout << rm->queueToString() << std::endl;

				STACK_TRACE(rm->execute());
			} else if (parameters->get<std::string>("solution type")=="staggered_grad") {
//				new_particles->getFieldManager()->createField(3, "computedGradSphereHarmonic", "m/s");

				new_particles->getFieldManager()->createField(3, "computedGradSphereHarmonic", "m/s");
				new_particles->getFieldManager()->createField(3, "solution", "m/s");
				new_particles->getFieldManager()->createField(3, "exact_solution", "m/s");

//				Compadre::RemapObject ro("scaledSphereHarmonic", "computedGradSphereHarmonic", TargetOperation::GradientOfScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, StaggeredEdgeAnalyticGradientIntegralSample);
				Compadre::RemapObject ro("scaledSphereHarmonic", "computedGradSphereHarmonic", TargetOperation::GradientOfScalarPointEvaluation, ReconstructionSpace::VectorTaylorPolynomial, StaggeredEdgeIntegralSample, StaggeredEdgeAnalyticGradientIntegralSample);

				if (parameters->get<int>("physics number")==3) {
					particles->getFieldManager()->createField(1, "kappa", "m/s");
					Compadre::FiveStripOnSphere function3 = Compadre::FiveStripOnSphere();
					Compadre::host_view_type kappa_field =  particles->getFieldManager()->getFieldByName("kappa")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();
					for(int j=0; j<coords->nLocal(); j++){
						xyz_type xyz = coords->getLocalCoords(j);
						kappa_field(j,0) = function3.evalDiffusionCoefficient(xyz);
					}
					particles->getFieldManager()->updateFieldsHaloData();
					ro.setOperatorCoefficients("kappa");
				}

				rm->add(ro);
				STACK_TRACE(rm->execute());

			} else if (parameters->get<std::string>("solution type")=="div_grad") {
				new_particles->getFieldManager()->createField(3, "computedGradSphereHarmonic", "m/s");
//				new_particles->getFieldManager()->createField(1, "computedLaplacianSphereHarmonic", "m/s");
//				new_particles->getFieldManager()->createField(1, "computedDivGradSphereHarmonic", "m/s");

				new_particles->getFieldManager()->createField(3, "exact_solution", "m/s");
				new_particles->getFieldManager()->createField(3, "solution", "m/s");

				Compadre::RemapObject ro("scaledSphereHarmonic", "computedGradSphereHarmonic", TargetOperation::GradientOfScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);

//				Compadre::RemapObject ro("scaledSphereHarmonic", "computedGradSphereHarmonic", TargetOperation::GradientOfScalarPointEvaluation, StaggeredEdgeAnalyticGradientIntegralSample);
//				Compadre::RemapObject ro1("sourceGradientSphereHarmonic", "computedLaplacianSphereHarmonic", DivergenceOfVectorPointEvaluation, StaggeredEdgeIntegralSample);

				rm->add(ro);
//				rm->add(ro1);

				STACK_TRACE(rm->execute());

//				Compadre::SphereHarmonic function3 = Compadre::SphereHarmonic(4,5);
//				new_particles->getFieldManager()->getFieldByName("computedGradSphereHarmonic")->
//										localInitFromScalarFunctionGradient(&function3);
			} else if (parameters->get<std::string>("solution type")=="div") {
				new_particles->getFieldManager()->createField(1, "exact_solution", "m/s");

				new_particles->buildHalo(halo_size);
				new_particles->getFieldManager()->updateFieldsHaloData();

				new_particles->getFieldManager()->createField(1, "computedDivGradSphereHarmonic", "m/s");

				Compadre::RemapObject ro("sourceGradientSphereHarmonic", "computedDivGradSphereHarmonic", TargetOperation::DivergenceOfVectorPointEvaluation, ReconstructionSpace::VectorOfScalarClonesTaylorPolynomial, ManifoldVectorPointSample);
				rm->add(ro);
				STACK_TRACE(rm->execute());
			} else if (parameters->get<std::string>("solution type")=="staggered_div") {
				new_particles->getFieldManager()->createField(1, "exact_solution", "m/s");

				new_particles->buildHalo(halo_size);
				new_particles->getFieldManager()->updateFieldsHaloData();

				new_particles->getFieldManager()->createField(1, "computedDivGradSphereHarmonic", "m/s");

				// this is a good test that with no integration error in the sampling, we can perform the correct divergence on the reconstructed polynomial that is calculated via integrating polynomials
//				Compadre::RemapObject ro("scaledSphereHarmonic","computedDivGradSphereHarmonic", TargetOperation::DivergenceOfVectorPointEvaluation, ReconstructionSpace::VectorTaylorPolynomial, StaggeredEdgeIntegralSample, StaggeredEdgeAnalyticGradientIntegralSample);

				// this tests the integration on the samples (likely second order at best given each edge only contains two values) combined with divergence on the reconstructed polynomial calculated via integrating polynomials
				Compadre::RemapObject ro("sourceGradientSphereHarmonic", "computedDivGradSphereHarmonic", TargetOperation::DivergenceOfVectorPointEvaluation, ReconstructionSpace::VectorTaylorPolynomial, StaggeredEdgeIntegralSample);
				rm->add(ro);
				STACK_TRACE(rm->execute());
			} else { // lb solve or five strip

				particles->getFieldManager()->createField(1, "exact_solution", "m/s");

				Teuchos::RCP<Compadre::ProblemT> problem =
					Teuchos::rcp( new Compadre::ProblemT(particles));

				// construct physics, sources, and boundary conditions
				Teuchos::RCP<Compadre::LaplaceBeltramiPhysics> physics =
					Teuchos::rcp( new Compadre::LaplaceBeltramiPhysics(particles, std::max(parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"), parameters->get<Teuchos::ParameterList>("remap").get<int>("curvature porder"))));
				Teuchos::RCP<Compadre::LaplaceBeltramiSources> source =
					Teuchos::rcp( new Compadre::LaplaceBeltramiSources(particles));
				Teuchos::RCP<Compadre::LaplaceBeltramiBoundaryConditions> bcs =
					Teuchos::rcp( new Compadre::LaplaceBeltramiBoundaryConditions(particles));

				physics->setPhysicsType(parameters->get<LO>("physics number")); // chained staggered grad-div
				bcs->setPhysicsType(parameters->get<LO>("physics number"));
	//			for (int j=0; j<coords->nLocal(); j++) {
	//				if (j%2==0)
	//					particles->setFlag(j,1);// make one dirichlet point
	//			}

				// set physics, sources, and boundary conditions in the problem
				problem->setPhysics(physics);
				problem->setSources(source);
				problem->setBCS(bcs);

				// assembly
				AssemblyTime->start();
				STACK_TRACE(problem->initialize());
				AssemblyTime->stop();

				//solving
				SolvingTime->start();
				STACK_TRACE(problem->solve());
				SolvingTime->stop();

				particles->getFieldManager()->updateFieldsHaloData();
			}




			// perform divergence on already calculated gradient
			if (parameters->get<std::string>("solution type")=="div_grad") {

				new_particles->buildHalo(halo_size);
				new_particles->getFieldManager()->updateFieldsHaloData();

				Teuchos::RCP<Compadre::RemapManager> rm2 = Teuchos::rcp(new Compadre::RemapManager(parameters, new_particles.getRawPtr(), new_particles.getRawPtr(), halo_size));
				new_particles->getFieldManager()->createField(1, "computedDivGradSphereHarmonic", "m/s");
//				new_particles->getFieldManager()->createField(3, "computedDivGradSphereHarmonic", "m/s");

				Compadre::RemapObject ro("computedGradSphereHarmonic", "computedDivGradSphereHarmonic", TargetOperation::DivergenceOfVectorPointEvaluation, ReconstructionSpace::VectorOfScalarClonesTaylorPolynomial, ManifoldVectorPointSample);
//				Compadre::RemapObject ro("computedGradSphereHarmonic", "computedDivGradSphereHarmonic", TargetOperation::StaggeredDivergenceOfVectorPointEvaluation, StaggeredEdgeIntegralSample);
//				Compadre::RemapObject ro("computedGradSphereHarmonic", "computedDivGradSphereHarmonic", TargetOperation::VectorPointEvaluation, StaggeredEdgeIntegralSample);

				rm2->add(ro);
				STACK_TRACE(rm2->execute());

//				Compadre::SphereHarmonic function3 = Compadre::SphereHarmonic(4,5);
//				new_particles->getFieldManager()->getFieldByName("computedDivGradSphereHarmonic")->
//										localInitFromScalarFunction(&function3);
//				new_particles->getFieldManager()->getFieldByName("computedDivGradSphereHarmonic")->scale(-1.0);

			} else if (parameters->get<std::string>("solution type")=="staggered_div_grad") {

				new_particles->buildHalo(halo_size);
				new_particles->getFieldManager()->updateFieldsHaloData();

				Teuchos::RCP<Compadre::RemapManager> rm2 = Teuchos::rcp(new Compadre::RemapManager(parameters, new_particles.getRawPtr(), new_particles.getRawPtr(), halo_size));
				new_particles->getFieldManager()->createField(1, "computedDivGradSphereHarmonic", "m/s");
				Compadre::RemapObject ro("computedGradSphereHarmonic", "computedDivGradSphereHarmonic", TargetOperation::DivergenceOfVectorPointEvaluation, ReconstructionSpace::VectorTaylorPolynomial, StaggeredEdgeIntegralSample);

				rm2->add(ro);
				STACK_TRACE(rm2->execute());

//				Compadre::SphereHarmonic function3 = Compadre::SphereHarmonic(4,5);
//				new_particles->getFieldManager()->getFieldByName("computedDivGradSphereHarmonic")->
//										localInitFromScalarFunction(&function3);
//				new_particles->getFieldManager()->getFieldByName("computedDivGradSphereHarmonic")->scale(-1.0);
			} else if (post_process_grad && (parameters->get<std::string>("solution type") == "lb solve" || parameters->get<std::string>("solution type") == "five_strip")) {
				new_particles->buildHalo(halo_size);
				new_particles->getFieldManager()->updateFieldsHaloData();

                // DEBUG
                //printf("DEBUG MODE: solution computed set to exact.\n");
				//Compadre::FiveStripOnSphere function2 = Compadre::FiveStripOnSphere();
				//particles->getFieldManager()->getFieldByName("solution")->
				//						localInitFromScalarFunction(&function2);
				//particles->getFieldManager()->updateFieldsHaloData();


				new_particles->getFieldManager()->createField(3, "computedKappaGrad", "m/s");
				Compadre::RemapObject ro("solution", "computedKappaGrad", TargetOperation::GradientOfScalarPointEvaluation, ReconstructionSpace::ScalarTaylorPolynomial, PointSample);

				rm->add(ro);
				STACK_TRACE(rm->execute());

				if (parameters->get<int>("physics number")==3) {
					new_particles->getFieldManager()->createField(1, "kappa", "m/s");
					new_particles->getFieldManager()->createField(1, "lon", "m/s");
					Compadre::FiveStripOnSphere function3 = Compadre::FiveStripOnSphere();

				    new_particles->getFieldManager()->getFieldByName("lon")->
						localInitFromScalarFunction(&function3);


					Compadre::host_view_type kappa_field =  new_particles->getFieldManager()->getFieldByName("kappa")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();
					Compadre::host_view_type lon_field =  new_particles->getFieldManager()->getFieldByName("lon")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();
					Compadre::host_view_type computed_kappa_grad_field =  new_particles->getFieldManager()->getFieldByName("computedKappaGrad")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();

					for(int j=0; j<coords->nLocal(); j++){
						xyz_type xyz = new_coords->getLocalCoords(j);
						kappa_field(j,0) = function3.evalDiffusionCoefficient(xyz);
					    for(int k=0; k<3; k++){
						    computed_kappa_grad_field(j,k) *= kappa_field(j,0);
                        }
					}
					new_particles->getFieldManager()->updateFieldsHaloData();
				}

				Compadre::FiveStripOnSphere function3 = Compadre::FiveStripOnSphere();
				new_particles->getFieldManager()->createField(3, "exactKappaGrad", "m/s");
				new_particles->getFieldManager()->getFieldByName("exactKappaGrad")->
						localInitFromVectorFunction(&function3);
			}

		 	// point by point check to see how far off from correct value
			// check solution

			NormTime->start();

			Teuchos::RCP<Compadre::AnalyticFunction> function;
//			function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::ConstantEachDimension(0,0,0)));
			if (parameters->get<int>("physics number") < 3) {
				if (parameters->get<std::string>("solution type")=="point" || parameters->get<std::string>("solution type")=="laplace" || parameters->get<std::string>("solution type")=="vector") {
					function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::SphereHarmonic(4,5)));
				} else if (parameters->get<std::string>("solution type")=="grad" || parameters->get<std::string>("solution type")=="staggered_grad") {
					function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::SphereTestVelocity()));
				} else if (parameters->get<std::string>("solution type")=="div_grad" || parameters->get<std::string>("solution type")=="staggered_div_grad") {
					function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::SphereHarmonic(4,5)));
				} else if (parameters->get<std::string>("solution type")=="div" || parameters->get<std::string>("solution type")=="staggered_div") {
					function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::SphereHarmonic(4,5)));
				} else { // lb solve
					function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::SphereHarmonic(4,5)));
				}
			} else if (parameters->get<int>("physics number") == 3){
				if (parameters->get<std::string>("solution type")=="lb solve")
					function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::ConstantEachDimension(0,0,0)));
				else
					function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::FiveStripOnSphere));
			} else if (parameters->get<int>("physics number") == 10){
				if (parameters->get<std::string>("solution type")=="lb solve" || parameters->get<std::string>("solution type")=="point" || parameters->get<std::string>("solution type")=="vector")
					function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::CylinderSinLonCosZ));
				else
					function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::CylinderSinLonCosZRHS));
			}


//			WriteTime->start();
//			{
//				std::string output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file");
//				Compadre::FileManager fm;
//
//				if (!post_process_grad && (parameters->get<std::string>("solution type")=="lb solve" || parameters->get<std::string>("solution type") == "five_strip")) {
//					fm.setWriter(output_filename, particles);
//				} else {
//					fm.setWriter(output_filename, new_particles);
//				}
//				fm.write();
//
//			}
//			WriteTime->stop();

			ST physical_coordinate_weighted_l2_norm = 0;
		 	ST exact_coordinate_weighted_l2_norm = 0;

//			Compadre::host_view_type solution_field = particles->getFieldManager()->getFieldByName("solution")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();
			Compadre::host_view_type solution_field;
			Compadre::host_view_type solution_field2;
			if (parameters->get<std::string>("solution type")=="point" || parameters->get<std::string>("solution type")=="laplace" || parameters->get<std::string>("solution type")=="staggered_laplace") {
				solution_field = new_particles->getFieldManager()->getFieldByName("computedSphereHarmonic")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();
				Compadre::host_view_type exact_solution_field = new_particles->getFieldManager()->getFieldByName("exact_solution")->getMultiVectorPtr()->getLocalView<Compadre::host_view_type>();
//				printf("direct applicatino:\n");
				for(int j=0; j<coords->nLocal(); j++){
					xyz_type xyz = coords->getLocalCoords(j);
					double exact_val = function->evalScalar(xyz);
					exact_solution_field(j,0) = exact_val;
//					printf("%.16f\n", solution_field(j,0));
				}
//				auto out = Teuchos::getFancyOStream(Teuchos::rcpFromRef(std::cout));
//				new_particles->getFieldManager()->getFieldByName("computedSphereHarmonic")->getMultiVectorPtrConst()->describe(*out, Teuchos::VERB_EXTREME);
			} else if (parameters->get<std::string>("solution type")=="div_grad" || parameters->get<std::string>("solution type")=="staggered_div_grad") {
				Compadre::host_view_type solution_field1 = new_particles->getFieldManager()->getFieldByName("computedGradSphereHarmonic")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();
				solution_field = new_particles->getFieldManager()->getFieldByName("computedDivGradSphereHarmonic")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();
//				solution_field2 = new_particles->getFieldManager()->getFieldByName("computedLaplacianSphereHarmonic")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();

				Compadre::host_view_type exact_solution_field = new_particles->getFieldManager()->getFieldByName("exact_solution")->getMultiVectorPtr()->getLocalView<Compadre::host_view_type>();

				for(int j=0; j<coords->nLocal(); j++){
					xyz_type xyz = coords->getLocalCoords(j);
					double exact_val = function->evalScalar(xyz);
					exact_solution_field(j,0) = -exact_val;
				}
			} else if (parameters->get<std::string>("solution type")=="grad") {
				Compadre::host_view_type solution_field_grad = new_particles->getFieldManager()->getFieldByName("computedGradSphereHarmonic")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();
		 		solution_field = new_particles->getFieldManager()->getFieldByName("solution")->getMultiVectorPtr()->getLocalView<Compadre::host_view_type>();
		 		Compadre::host_view_type exact_solution_field = new_particles->getFieldManager()->getFieldByName("exact_solution")->getMultiVectorPtr()->getLocalView<Compadre::host_view_type>();

				for(int j=0; j<coords->nLocal(); j++){
					xyz_type xyz = coords->getLocalCoords(j);
					xyz_type velocity_exact = function->evalVector(xyz);

					solution_field(j,0) = (solution_field_grad(j,1)*xyz.z - xyz.y*solution_field_grad(j,2));
					solution_field(j,1) = (-(solution_field_grad(j,0)*xyz.z - xyz.x*solution_field_grad(j,2)));
					solution_field(j,2) = (solution_field_grad(j,0)*xyz.y - xyz.x*solution_field_grad(j,1));

					exact_solution_field(j,0) = velocity_exact.x;
					exact_solution_field(j,1) = velocity_exact.y;
					exact_solution_field(j,2) = velocity_exact.z;
				}
//				Compadre::host_view_type solution_field1 = new_particles->getFieldManager()->getFieldByName("computedGradSphereHarmonicX")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();
//				Compadre::host_view_type solution_field2 = new_particles->getFieldManager()->getFieldByName("computedGradSphereHarmonicY")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();
//				Compadre::host_view_type solution_field3 = new_particles->getFieldManager()->getFieldByName("computedGradSphereHarmonicZ")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();
//
//		 		solution_field = new_particles->getFieldManager()->getFieldByName("solution")->getMultiVectorPtr()->getLocalView<Compadre::host_view_type>();
//		 		Compadre::host_view_type exact_solution_field = new_particles->getFieldManager()->getFieldByName("exact_solution")->getMultiVectorPtr()->getLocalView<Compadre::host_view_type>();
//
//				for(int j=0; j<coords->nLocal(); j++){
//					xyz_type xyz = coords->getLocalCoords(j);
//					xyz_type velocity_exact = function->evalVector(xyz);
//
//					solution_field(j,0) = (solution_field2(j,0)*xyz.z - xyz.y*solution_field3(j,0));
//					solution_field(j,1) = (-(solution_field1(j,0)*xyz.z - xyz.x*solution_field3(j,0)));
//					solution_field(j,2) = (solution_field1(j,0)*xyz.y - xyz.x*solution_field2(j,0));
//
//					exact_solution_field(j,0) = velocity_exact.x;
//					exact_solution_field(j,1) = velocity_exact.y;
//					exact_solution_field(j,2) = velocity_exact.z;
//				}
			} else if (parameters->get<std::string>("solution type")=="staggered_grad") {
				Compadre::host_view_type solution_field_grad = new_particles->getFieldManager()->getFieldByName("computedGradSphereHarmonic")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();
		 		solution_field = new_particles->getFieldManager()->getFieldByName("solution")->getMultiVectorPtr()->getLocalView<Compadre::host_view_type>();
		 		Compadre::host_view_type exact_solution_field = new_particles->getFieldManager()->getFieldByName("exact_solution")->getMultiVectorPtr()->getLocalView<Compadre::host_view_type>();

				for(int j=0; j<coords->nLocal(); j++){
					xyz_type xyz = coords->getLocalCoords(j);
					xyz_type velocity_exact = function->evalVector(xyz);

					solution_field(j,0) = (solution_field_grad(j,1)*xyz.z - xyz.y*solution_field_grad(j,2));
					solution_field(j,1) = (-(solution_field_grad(j,0)*xyz.z - xyz.x*solution_field_grad(j,2)));
					solution_field(j,2) = (solution_field_grad(j,0)*xyz.y - xyz.x*solution_field_grad(j,1));

					exact_solution_field(j,0) = velocity_exact.x;
					exact_solution_field(j,1) = velocity_exact.y;
					exact_solution_field(j,2) = velocity_exact.z;
				}
			} else if (parameters->get<std::string>("solution type")=="div" || parameters->get<std::string>("solution type")=="staggered_div") {
				solution_field = new_particles->getFieldManager()->getFieldByName("computedDivGradSphereHarmonic")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();

				Compadre::host_view_type exact_solution_field = new_particles->getFieldManager()->getFieldByName("exact_solution")->getMultiVectorPtr()->getLocalView<Compadre::host_view_type>();

				for(int j=0; j<coords->nLocal(); j++){
					xyz_type xyz = coords->getLocalCoords(j);
					double exact_val = function->evalScalar(xyz);
					exact_solution_field(j,0) = -exact_val;
				}
			} else if (parameters->get<std::string>("solution type")=="vector") {
		 		solution_field = new_particles->getFieldManager()->getFieldByName("computedGradSphereHarmonic")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();
		 		Compadre::host_view_type exact_solution_field = new_particles->getFieldManager()->getFieldByName("exact_solution")->getMultiVectorPtr()->getLocalView<Compadre::host_view_type>();

				for(int j=0; j<coords->nLocal(); j++){
					xyz_type xyz = coords->getLocalCoords(j);
					xyz_type vector_exact = function->evalScalarDerivative(xyz);

					exact_solution_field(j,0) = vector_exact.x;
					exact_solution_field(j,1) = vector_exact.y;
					exact_solution_field(j,2) = vector_exact.z;
				}
			} else if (parameters->get<std::string>("solution type")=="five_strip") {
                if (post_process_grad) {
				    solution_field = new_particles->getFieldManager()->getFieldByName("computedKappaGrad")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();
                } else {
				    solution_field = particles->getFieldManager()->getFieldByName("solution")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();
                }
			} else {
				solution_field = particles->getFieldManager()->getFieldByName("solution")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();
//				printf("A operation:\n");
//				for(int j=0; j<coords->nLocal(); j++){
//					printf("%.16f\n", solution_field(j,0));
//				}
				Compadre::host_view_type exact_solution_field = particles->getFieldManager()->getFieldByName("exact_solution")->getMultiVectorPtr()->getLocalView<Compadre::host_view_type>();
				for(int j=0; j<coords->nLocal(); j++){
					xyz_type xyz = coords->getLocalCoords(j);
					double exact_val = function->evalScalar(xyz);
					if (parameters->get<LO>("physics number")<3) {
						exact_solution_field(j,0) = -exact_val;
					} else {
						exact_solution_field(j,0) = exact_val;
					}
				}
			}

			WriteTime->start();
			{
				std::string output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file prefix") + parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file");
				Compadre::FileManager fm;

				if (!post_process_grad && (parameters->get<std::string>("solution type")=="lb solve" || parameters->get<std::string>("solution type") == "five_strip")) {
					fm.setWriter(output_filename, particles);
				} else {
					fm.setWriter(output_filename, new_particles);
				}
				fm.write();

			}
			WriteTime->stop();

			//bool use_grid_area_for_L2 = false;
			Compadre::host_view_type grid_area_field;
		    try {
		        grid_area_field = particles->getFieldManager()->getFieldByName("grid_area")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();
		    //    use_grid_area_for_L2 = true;
		    } catch (...) {
		    }
		    // if a weighted L2 is desire, use_grid_area_for_L2 can be checked before-hand


			double exact = 0;
			GO num_solved_for = 0;
			for(int j=0; j<coords->nLocal(); j++){
				if 	(particles->getFlag(j)==0) {
					xyz_type xyz = coords->getLocalCoords(j);
					if (parameters->get<std::string>("solution type")=="point") {
						if (parameters->get<LO>("physics number")<3) {
							exact = function->evalScalar(xyz) * (1.0 / (5 * (5 + 1)));
						} else {
							exact = function->evalScalar(xyz);
						}
					} else if (parameters->get<std::string>("solution type")=="vector") {
						exact = function->evalScalarDerivative(xyz).x;
					} else if (parameters->get<std::string>("solution type")=="laplace" || parameters->get<std::string>("solution type")=="staggered_laplace") {
						exact = -function->evalScalar(xyz);
					} else if (parameters->get<std::string>("solution type")=="grad" || parameters->get<std::string>("solution type")=="staggered_grad") {
						exact = function->evalVector(xyz).x;
					} else if (parameters->get<std::string>("solution type")=="div_grad" || parameters->get<std::string>("solution type")=="staggered_div_grad") {
						exact = -function->evalScalar(xyz);
					} else if (parameters->get<std::string>("solution type")=="div" || parameters->get<std::string>("solution type")=="staggered_div") {
						exact = -function->evalScalar(xyz);
					} else if (parameters->get<std::string>("solution type")=="five_strip") {
                        if (post_process_grad) {
					        xyz_type this_xyz = new_coords->getLocalCoords(j);
                            exact = function->evalVector(this_xyz).x;
                        } else {
							exact = function->evalScalar(xyz);
                        }
					} else { // lb solve
						if (parameters->get<LO>("physics number")<3) {
							exact = function->evalScalar(xyz) * (1.0 / (5 * (5 + 1)));
						} else {
							exact = function->evalScalar(xyz);
						}
					}

	//				exact = function->evalScalar(xyz) * (1.0 / (5 * (5 + 1)));
					physical_coordinate_weighted_l2_norm += (solution_field(j,0) - exact)*(solution_field(j,0) - exact);//*grid_area_field(j,0);
	//				if (parameters->get<std::string>("solution type")=="div_grad" ||  parameters->get<std::string>("solution type")=="staggered_div_grad")
	//					physical_coordinate_weighted_l2_norm += (solution_field2(j,0) - exact)*(solution_field2(j,0) - exact);

					if (parameters->get<std::string>("solution type")=="grad" || parameters->get<std::string>("solution type")=="staggered_grad" || (parameters->get<std::string>("solution type")=="five_strip" && post_process_grad)) {
						physical_coordinate_weighted_l2_norm += (solution_field(j,1) - function->evalVector(xyz).y)*(solution_field(j,1) - function->evalVector(xyz).y);
						physical_coordinate_weighted_l2_norm += (solution_field(j,2) - function->evalVector(xyz).z)*(solution_field(j,2) - function->evalVector(xyz).z);
					} else if (parameters->get<std::string>("solution type")=="vector") {
						physical_coordinate_weighted_l2_norm += (solution_field(j,1) - function->evalScalarDerivative(xyz).y)*(solution_field(j,1) - function->evalScalarDerivative(xyz).y);
						physical_coordinate_weighted_l2_norm += (solution_field(j,2) - function->evalScalarDerivative(xyz).z)*(solution_field(j,2) - function->evalScalarDerivative(xyz).z);
					}

	//				physical_coordinate_weighted_l2_norm += (solution_field(j,0) - exact)*(solution_field(j,0) - exact)*grid_area_field(j,0);
					exact_coordinate_weighted_l2_norm += exact*exact;//*grid_area_field(j,0);
	//				exact_coordinate_weighted_l2_norm += exact*exact*grid_area_field(j,0);
					num_solved_for++;
				}
			}

			// get global # of dofs solved for
			GO global_num_solved_for = 0;
			Teuchos::Ptr<GO> global_num_solved_for_ptr(&global_num_solved_for);
			Teuchos::reduceAll<int, GO>(*comm, Teuchos::REDUCE_SUM, num_solved_for, global_num_solved_for_ptr);


			physical_coordinate_weighted_l2_norm /= global_num_solved_for;
			//			physical_coordinate_weighted_l2_norm = sqrt(physical_coordinate_weighted_l2_norm) /((ST)coords->nGlobalMax());

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


