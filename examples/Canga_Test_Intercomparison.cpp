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

class FieldOne : public Compadre::AnalyticFunction {

	public :

		FieldOne() {}

		virtual ST evalScalar(const xyz_type& xIn, const local_index_type input_comp=0) const {
			const ST lat = xIn.latitude(); // phi
			const ST lon = xIn.longitude(); // lambda

			return 2 + std::cos(lat)*std::cos(lat)*std::cos(2*lon);
		}
};

class FieldTwo : public Compadre::AnalyticFunction {

	public :

		FieldTwo() {}

		virtual ST evalScalar(const xyz_type& xIn, const local_index_type input_comp=0) const {
			const ST lat = xIn.latitude(); // phi
			const ST lon = xIn.longitude(); // lambda

			return 2 + std::pow(std::sin(2*lat),16)*std::cos(16*lon);
		}
};

class FieldThree : public Compadre::AnalyticFunction {

	public :

		FieldThree() {}

		virtual ST evalScalar(const xyz_type& xIn, const local_index_type input_comp=0) const {
			const ST lat = xIn.latitude(); // theta
			const ST lon = xIn.longitude(); // lambda

			const LO d = 5;
			const LO t = 6;

			// may be reverse of the correct rotation, be sure to check

			// https://gis.stackexchange.com/questions/10808/manually-transforming-rotated-lat-lon-to-regular-lat-lon
			const ST r0 = 3,
					lambda0 = 0,
					theta0 = 0.6,
					temp_lambda = -lambda0,
					temp_theta = -(90+theta0);

			const ST
					lambda_prime = atan2(sin(lon), tan(lat)*sin(temp_theta) + cos(lon)*cos(temp_theta)) - temp_lambda,
					theta_prime = asin(cos(temp_theta)*sin(lat) - cos(lon)*sin(temp_theta)*cos(lat));

			const ST rho_prime = r0 * std::cos(theta_prime),
					sech_rho_prime = 1./cosh(rho_prime),
					Vt = 1.5*std::sqrt(3)*sech_rho_prime*sech_rho_prime*tanh(rho_prime),
					omega_prime = (rho_prime == 0) ? 0 : Vt / rho_prime;

			return 1 - tanh((rho_prime / d) * std::sin(lambda_prime - omega_prime*t));
		}
};

int main (int argc, char* args[]) {

	Teuchos::GlobalMPISession mpi(&argc, &args);
	Teuchos::oblackholestream bstream;
	Teuchos::RCP<const Teuchos::Comm<int> > global_comm = Teuchos::DefaultComm<int>::getComm();

	Kokkos::initialize(argc, args);

	Teuchos::RCP<Teuchos::Time> ParameterTime = Teuchos::TimeMonitor::getNewCounter ("Parameter Initialization");
	Teuchos::RCP<Teuchos::Time> MiscTime = Teuchos::TimeMonitor::getNewCounter ("Miscellaneous");
	Teuchos::RCP<Teuchos::Time> NormTime = Teuchos::TimeMonitor::getNewCounter ("Norm calculation");
	Teuchos::RCP<Teuchos::Time> CoordinateInsertionTime = Teuchos::TimeMonitor::getNewCounter ("Coordinate Insertion Time");
	Teuchos::RCP<Teuchos::Time> ParticleInsertionTime = Teuchos::TimeMonitor::getNewCounter ("Particle Insertion Time");
	Teuchos::RCP<Teuchos::Time> ReadTime = Teuchos::TimeMonitor::getNewCounter ("Read Time");

	//********* BASIC PARAMETER SETUP FOR ANY PROBLEM
	ParameterTime->start();
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
	ParameterTime->stop();
	// There is a balance between "neighbors needed multiplier" and "cutoff multiplier"
	// Their product must increase somewhat with "porder", but "neighbors needed multiplier"
	// requires more of an increment than "cutoff multiplier" to achieve the required threshold (the other staying the same)
	//*********

	const LO my_coloring = parameters->get<LO>("my coloring");
	const LO peer_coloring = parameters->get<LO>("peer coloring");

	Teuchos::RCP<const Teuchos::Comm<int> > comm = global_comm->split(my_coloring, global_comm->getRank());



	{

		MiscTime->start();
		{
			//
			// Remote Data Test (Two differing distributions of particles)
			//

			typedef Compadre::EuclideanCoordsT CT;
		 	Teuchos::RCP<Compadre::ParticlesT> particles =
		 		Teuchos::rcp( new Compadre::ParticlesT(parameters, comm));
		 	CT* coords = (CT*)particles->getCoords();

		 	// Read in appropriate file here between two codes
		 	// equiangular cubed-sphere,
		 	// quasi-uniform centroidal Voronoi,
			// regionally refined cubed-sphere
			// regionally-refined centroidal Voronoi

			// fields
			// 0 - smooth
			// 1 - higher frequency
			// 2 - vortex
			// 3 - constant field (may require either SVD or enriching manifold reconstruction)

			// regridding
			// n - 2 way passes (could be 0, viewed as 1 way pass)

			// metric
			// 0 - conservation, global conservation of fields
			// 1 - locality
			// 2 - accuracy
			// 3 - global extrema
			// 4 - local extrema (later)
			// 5 - gradients (later)



			//Read in data file.
			ReadTime->start();
			Compadre::FileManager fm;
			std::string testfilename(parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file prefix") + parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file"));
			fm.setReader(testfilename, particles, parameters->get<Teuchos::ParameterList>("io").get<std::string>("nc type"));
            fm.getReader()->setCoordinateLayout(parameters->get<Teuchos::ParameterList>("io").get<std::string>("particles number dimension name"),
                                                parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinates layout"),
                                                parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate 0 name"),
                                                parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate 1 name"),
                                                parameters->get<Teuchos::ParameterList>("io").get<std::string>("coordinate 2 name"));
            fm.getReader()->setCoordinateUnitStyle(parameters->get<Teuchos::ParameterList>("io").get<std::string>("lat lon units"));
            fm.getReader()->setKeepOriginalCoordinates(parameters->get<Teuchos::ParameterList>("io").get<bool>("keep original lat lon"));
			fm.read();
			ReadTime->stop();


			// define some field on the source particles
			// define a field specific to the process running
			if (my_coloring == 25) {
				// ROLE PLAYED HERE BY TWO WAY PASSES
				// ALWAYS DEFINES FIELD TO PASS
				particles->getFieldManager()->createField(1,"source","m^2/1a");

				if (parameters->get<LO>("field")==0) {
					FieldOne source;
					particles->getFieldManager()->getFieldByName("source")->
							localInitFromScalarFunction(&source);
				} else if (parameters->get<LO>("field")==1) {
					FieldTwo source;
					particles->getFieldManager()->getFieldByName("source")->
							localInitFromScalarFunction(&source);
				} else if (parameters->get<LO>("field")==2) {
					FieldThree source;
					particles->getFieldManager()->getFieldByName("source")->
							localInitFromScalarFunction(&source);
				} else if (parameters->get<LO>("field")==3) {
					Compadre::ConstantEachDimension source(1,1,1);
					particles->getFieldManager()->getFieldByName("source")->
							localInitFromScalarFunction(&source);
				} else if (parameters->get<LO>("field")==4) {
					Compadre::DiscontinuousOnSphere source;
					particles->getFieldManager()->getFieldByName("source")->
							localInitFromScalarFunction(&source);
				}
			} else if (my_coloring == 33) {
				// no fields defined here, they come from remap
				if (parameters->get<LO>("field")==4) {
					particles->getFieldManager()->createField(1,"exact target","m^2/1a");
					Compadre::DiscontinuousOnSphere exact_target;
					particles->getFieldManager()->getFieldByName("exact target")->
							localInitFromScalarFunction(&exact_target);
				}
			}

		 	// calculate h_size for mesh

		 	// build halo data
			ST halo_size = parameters->get<Teuchos::ParameterList>("halo").get<double>("size") *
					( parameters->get<Teuchos::ParameterList>("halo").get<double>("multiplier")
							+ (ST)(parameters->get<Teuchos::ParameterList>("remap").get<int>("porder")));
			particles->zoltan2Initialize();
			particles->buildHalo(halo_size);
			particles->getFieldManager()->updateFieldsHaloData();

			// diagnostic
			std::string output_filename1 = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "pre_swap" + std::to_string(my_coloring) + ".nc";
			fm.setWriter(output_filename1, particles);
			fm.write();


//			std::vector<int> flags_to_transfer(2);
//			flags_to_transfer[0] = 1;
//			flags_to_transfer[1] = 2;
//			// should add in a filter in the remoteDataManager
//			Compadre::host_view_type flags = particles->getFlags()->getLocalView<Compadre::host_view_type>();
//			Teuchos::RCP<Compadre::RemoteDataManager> remoteDataManager =
//					Teuchos::rcp(new Compadre::RemoteDataManager(global_comm, comm, coords, my_coloring, peer_coloring, flags, flags_to_transfer));


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

			std::string output_filename2 = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "before_swap" + std::to_string(my_coloring) + ".nc";
			fm.setWriter(output_filename2, peer_processors_particles);
			fm.write();

		 	if (parameters->get<LO>("two way passes")==0) {
				std::vector<Compadre::RemapObject> remap_vec;
				if (my_coloring == 25) {
					// no field remapped to here, only sent
				} else if (my_coloring == 33) {
					Compadre::RemapObject r1("source", "target");
                    Compadre::OptimizationObject opt_obj = Compadre::OptimizationObject(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("optimization algorithm"), true /*single linear bound*/, true /*bounds preservation*/, parameters->get<Teuchos::ParameterList>("remap").get<double>("global lower bound")/*global lower bound*/, parameters->get<Teuchos::ParameterList>("remap").get<double>("global upper bound")/*global upper bound*/);
					r1.setOptimizationObject(opt_obj);
					remap_vec.push_back(r1);
				}
				remoteDataManager->remapData(remap_vec, parameters, particles.getRawPtr(), peer_processors_particles.getRawPtr(), halo_size);
		 	} else {
				for (LO i=0; i<parameters->get<LO>("two way passes"); ++i) {
					std::vector<Compadre::RemapObject> remap_vec_to;
					if (my_coloring == 33) {
						if (i==0) {
							Compadre::RemapObject r1("source", "target");
							remap_vec_to.push_back(r1);
						} else {
							Compadre::RemapObject r1("target", "target");
							remap_vec_to.push_back(r1);
						}
					} else if (my_coloring == 25) {
						// no field remapped to here, only sent
					}
					remoteDataManager->remapData(remap_vec_to, parameters, particles.getRawPtr(), peer_processors_particles.getRawPtr(), halo_size);

					std::vector<Compadre::RemapObject> remap_vec_from;
					if (my_coloring == 33) {
						// no field remapped to here, only sent

					} else if (my_coloring == 25) {
						Compadre::RemapObject r1("target", "target");
						remap_vec_to.push_back(r1);
					}
					remoteDataManager->remapData(remap_vec_to, parameters, particles.getRawPtr(), peer_processors_particles.getRawPtr(), halo_size);
				}
			}

			std::string output_filename = parameters->get<Teuchos::ParameterList>("io").get<std::string>("output file prefix") + "after_swap" + std::to_string(my_coloring) + ".nc";
			fm.setWriter(output_filename, particles);
			fm.write();

//			output_filename = "after_swap" + std::to_string(my_coloring) + ".nc";
//			fm.setWriter(output_filename, particles);
//			fm.write();

			if (parameters->get<LO>("two way passes")==0 && my_coloring == 25) {

				Compadre::host_view_type source_grid_area_field = particles->getFieldManager()->getFieldByName("grid_area")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();

				Teuchos::RCP<Compadre::AnalyticFunction> source_function;

				if (parameters->get<LO>("field")==0) {
					source_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new FieldOne));
				} else if (parameters->get<LO>("field")==1) {
					source_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new FieldTwo));
				} else if (parameters->get<LO>("field")==2) {
					source_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new FieldThree));
				} else if (parameters->get<LO>("field")==3) {
					source_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::ConstantEachDimension(1,1,1)));
				} else if (parameters->get<LO>("field")==4) {
					source_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::DiscontinuousOnSphere));
				}

				if (parameters->get<LO>("metric")==0) { // global conservation
					double exact_source_integral = 0;

					for( int j=0; j<coords->nLocal(); j++){
						xyz_type xyz = coords->getLocalCoords(j);
						ST exact = source_function->evalScalar(xyz);

						exact_source_integral += source_grid_area_field(j,0)*exact;
					}

					ST global_exact_source_integral = 0;

					Teuchos::Ptr<ST> global_exact_source_integral_ptr(&global_exact_source_integral);

					Teuchos::reduceAll<int, ST>(*global_comm, Teuchos::REDUCE_SUM, exact_source_integral, global_exact_source_integral_ptr);
				}
			}

			// hardest case is when it is a one way pass and
			if ((parameters->get<LO>("two way passes")==0 && my_coloring == 33) ||
					(parameters->get<LO>("two way passes")>0 && my_coloring == 25)) { // one way ends up with data on proc 33, two way ends up with data on proc 25
				// compare errors against

				Compadre::host_view_type target_grid_area_field = particles->getFieldManager()->getFieldByName("grid_area")->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();

				Teuchos::RCP<Compadre::AnalyticFunction> target_function;
				// here we have the true solution on the target mesh, or back on the source mesh

				if (parameters->get<LO>("field")==0) {
					target_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new FieldOne));
				} else if (parameters->get<LO>("field")==1) {
					target_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new FieldTwo));
				} else if (parameters->get<LO>("field")==2) {
					target_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new FieldThree));
				} else if (parameters->get<LO>("field")==3) {
					target_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::ConstantEachDimension(1,1,1)));
				} else if (parameters->get<LO>("field")==4) {
					target_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::DiscontinuousOnSphere));
				}


				// metric
				// 0 - conservation, global conservation of fields
				// 1 - locality
				// 2 - accuracy
				// 3 - global extrema
				// 4 - local extrema (later)
				// 5 - gradients (later)
				if (parameters->get<LO>("metric")==0) { // global conservation
					// makes most sense if 0 passes (1-way)
					// requires integral on source mesh and target
					// get appropriate "true solution" for the peer processes

					double computed_target_integral = 0;
					double exact_source_integral = 0;
					double exact_target_integral = 0;

					for( int j=0; j<coords->nLocal(); j++){
						xyz_type xyz = coords->getLocalCoords(j);
						ST exact = target_function->evalScalar(xyz);

						ST val = particles->getFieldManager()->getFieldByName("target")->getLocalScalarVal(j);

						computed_target_integral += target_grid_area_field(j,0)*val;
						exact_target_integral += target_grid_area_field(j,0)*exact;
					}

					ST global_computed_target_integral = 0;
					ST global_exact_source_integral = 0;
					ST global_exact_target_integral = 0;

					Teuchos::Ptr<ST> global_computed_target_integral_ptr(&global_computed_target_integral);
					Teuchos::Ptr<ST> global_exact_source_integral_ptr(&global_exact_source_integral);
					Teuchos::Ptr<ST> global_exact_target_integral_ptr(&global_exact_target_integral);

					Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_SUM, computed_target_integral, global_computed_target_integral_ptr);
					Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_SUM, exact_target_integral, global_exact_target_integral_ptr);

					if (parameters->get<LO>("two way passes")>0) {
						global_exact_source_integral = global_exact_target_integral;
					} else {
						// communicate with other processor to get source integral information
						Teuchos::reduceAll<int, ST>(*global_comm, Teuchos::REDUCE_SUM, exact_source_integral, global_exact_source_integral_ptr);
					}

					if (comm->getRank()==0) {
						std::cout << "\nGlobal Conservation Error: " << (global_computed_target_integral - global_exact_source_integral) /  global_exact_target_integral  << "\n";
					}

				} else if (parameters->get<LO>("metric")==1) { // locality
					// requires sum over source to "where it went", i.e. many transfers


				} else if (parameters->get<LO>("metric")==2) { // accuracy
					// do l1, l2, linfty
					// check solution
					double l1_norm = 0, l2_norm = 0, linf_norm = 0;
					double l1_exact_norm = 0, l2_exact_norm = 0, linf_exact_norm = 0;

					for( int j=0; j<coords->nLocal(); j++){
						xyz_type xyz = coords->getLocalCoords(j);
						ST exact = target_function->evalScalar(xyz);
//						printf("std::abs(exact): %f\n", std::abs(exact));
						ST val = particles->getFieldManager()->getFieldByName("target")->getLocalScalarVal(j);

						l1_norm += target_grid_area_field(j,0)*std::abs(exact - val);
						l2_norm += target_grid_area_field(j,0)*std::abs(exact - val)*std::abs(exact - val);

						l1_exact_norm += target_grid_area_field(j,0)*std::abs(exact);
						l2_exact_norm += target_grid_area_field(j,0)*std::abs(exact)*std::abs(exact);

						linf_norm = std::max(linf_norm, std::abs(exact-val));
						linf_exact_norm = std::max(linf_exact_norm, std::abs(exact));
					}

					ST global_l1_norm = 0;
					ST global_l1_exact_norm = 0;
					ST global_l2_norm = 0;
					ST global_l2_exact_norm = 0;
					ST global_linf_norm = 0;
					ST global_linf_exact_norm = 0;

					Teuchos::Ptr<ST> global_l1_norm_ptr(&global_l1_norm);
					Teuchos::Ptr<ST> global_l1_exact_norm_ptr(&global_l1_exact_norm);
					Teuchos::Ptr<ST> global_l2_norm_ptr(&global_l2_norm);
					Teuchos::Ptr<ST> global_l2_exact_norm_ptr(&global_l2_exact_norm);
					Teuchos::Ptr<ST> global_linf_norm_ptr(&global_linf_norm);
					Teuchos::Ptr<ST> global_linf_exact_norm_ptr(&global_linf_exact_norm);

					Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_SUM, l1_norm, global_l1_norm_ptr);
					Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_SUM, l1_exact_norm, global_l1_exact_norm_ptr);
					Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_SUM, l2_norm, global_l2_norm_ptr);
					Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_SUM, l2_exact_norm, global_l2_exact_norm_ptr);
					Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_MAX, linf_norm, global_linf_norm_ptr);
					Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_MAX, linf_exact_norm, global_linf_exact_norm_ptr);

					if (comm->getRank()==0) {
						std::cout << "\nL1 Error: " << global_l1_norm /  global_l1_exact_norm  << "\n";
						std::cout << "L2 Error: " << std::sqrt(global_l2_norm) /  std::sqrt(global_l2_exact_norm)  << "\n";
						std::cout << "LInf Error: " << global_linf_norm /  global_linf_exact_norm << "\n";
					}
				} else if (parameters->get<LO>("metric")==3) { // global extrema
					// only requires information on target
					double min_norm = std::numeric_limits<ST>::max(), max_norm = 0;
					double min_exact_norm = std::numeric_limits<ST>::max(), max_exact_norm = 0;

					for( int j=0; j<coords->nLocal(); j++){
						xyz_type xyz = coords->getLocalCoords(j);
						ST exact = target_function->evalScalar(xyz);
						ST val = particles->getFieldManager()->getFieldByName("target")->getLocalScalarVal(j);

						min_norm = std::min(min_norm, val);
						max_norm = std::max(max_norm, val);

						min_exact_norm = std::min(min_norm, exact);
						max_exact_norm = std::max(max_norm, exact);
					}

					ST global_min_norm = std::numeric_limits<ST>::max();
					ST global_max_norm = 0;
					ST global_min_exact_norm = std::numeric_limits<ST>::max();
					ST global_max_exact_norm = 0;

					Teuchos::Ptr<ST> global_min_norm_ptr(&global_min_norm);
					Teuchos::Ptr<ST> global_min_exact_norm_ptr(&global_min_exact_norm);
					Teuchos::Ptr<ST> global_max_norm_ptr(&global_max_norm);
					Teuchos::Ptr<ST> global_max_exact_norm_ptr(&global_max_exact_norm);

					Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_MIN, min_norm, global_min_norm_ptr);
					Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_MIN, min_exact_norm, global_min_exact_norm_ptr);
					Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_MAX, max_norm, global_max_norm_ptr);
					Teuchos::reduceAll<int, ST>(*comm, Teuchos::REDUCE_MAX, max_exact_norm, global_max_exact_norm_ptr);

					if (comm->getRank()==0) {
						std::cout << "\nMin Error: " <<  (global_min_exact_norm - global_min_norm) / global_max_exact_norm << "\n";
						std::cout << "Max Error: " << (global_max_norm - global_max_exact_norm) / global_max_exact_norm  << "\n";
					}

				} else if (parameters->get<LO>("metric")==4) { // local extrema
					// for later
					TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Not supported.");

				} else if (parameters->get<LO>("metric")==5) { // gradients
					// for later
					TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Not supported.");

				}

			}

		}
		MiscTime->stop();
	}


	Teuchos::TimeMonitor::summarize();
	Kokkos::finalize();
return 0;
}

