#include <Compadre_LagrangianShallowWater_Operator.hpp>

#include <Compadre_CoordsT.hpp>
#include <Compadre_ParticlesT.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_DOFManager.hpp>
#include <Compadre_FieldT.hpp>
#include <Compadre_AnalyticFunctions.hpp>
#include <Compadre_NeighborhoodT.hpp>
#include <Compadre_XyzVector.hpp>

#include <Compadre_GMLS.hpp>

#ifdef COMPADREHARNESS_USE_OPENMP
#include <omp.h>
#endif

#include <Compadre_GlobalConstants.hpp>

/*
 Constructs the matrix operator.
 
 */
namespace Compadre {

static const GlobalConstants consts;
//static const scalar_type PI = consts.Pi();

typedef Compadre::CoordsT coords_type;
typedef Compadre::FieldT fields_type;
typedef Compadre::NeighborhoodT neighborhood_type;
typedef Compadre::XyzVector xyz_type;


void LagrangianShallowWaterPhysics::computeVector(local_index_type field_one, local_index_type field_two, scalar_type time) {
	Teuchos::RCP<Teuchos::Time> ComputeMatrixTime = Teuchos::TimeMonitor::getNewCounter ("Compute Matrix Time");
	ComputeMatrixTime->start();

//	/*
//	 * Example for velocity = cos(t), and height = -sin(t)
//	 */
//
//	host_view_type b_data = _b->getLocalView<host_view_type>();
//	// get field_one and check which one it is
//	int my_field = field_one;
//	host_view_type other_data = _particles->getFieldManager()->getFieldByID(field_two)->getMultiVectorPtr()->getLocalView<host_view_type>();
//
//	if (field_two != field_one) {
//		if (_particles->getFieldManager()->getIDOfFieldFromName("velocity") == my_field) {
//			// if velocity, set to height
//			for (int i=0; i<b_data.dimension_0(); ++i) {
//				b_data(i,0) = other_data(i,0);
//			}
//		} else {
//			// if height, set to -velocity
//			for (int i=0; i<b_data.dimension_0(); ++i) {
//				b_data(i,0) = -other_data(i,0);
//			}
//		}
//	}

//	/*
//	 *  Rigid body rotation test
//	 */
//
//	double Omega = 2*PI;
//
//	host_view_type b_data = _b->getLocalView<host_view_type>();
//	// get field_one and check which one it is
//	int my_field = field_one;
//	host_view_type other_data = _particles->getFieldManager()->getFieldByID(field_two)->getMultiVectorPtr()->getLocalView<host_view_type>();
//
//	if (field_two == field_one) {
//		if (_particles->getFieldManager()->getIDOfFieldFromName("displacement") == my_field) {
//
//			const std::vector<std::vector<std::vector<local_index_type> > >& local_to_dof_map =
//					this->_particles->getDOFManagerConst()->getDOFMap();
//
//			// if velocity, set to height
//			for (int i=0; i<other_data.dimension_0(); ++i) {
//				local_index_type dof_x = local_to_dof_map[i][field_one][0];
//				b_data(dof_x,0) =  other_data(i,1) * Omega;
//				local_index_type dof_y = local_to_dof_map[i][field_one][1];
//				b_data(dof_y,0) = -other_data(i,0) * Omega;
//				local_index_type dof_z = local_to_dof_map[i][field_one][2];
//				b_data(dof_z,0) = 0;
//			}
//		}
//	}

	bool EULERIAN = true;
	if (_particles->getCoordsConst()->isLagrangian()) {
		EULERIAN = false;
	}


	bool include_halo = true;

	bool use_physical_coords = true; // can be set on the operator in the future

	TEUCHOS_TEST_FOR_EXCEPT_MSG(_b==NULL, "Tpetra multivector for Physics not yet specified.");

	const local_index_type nlocal = static_cast<local_index_type>(this->_coords->nLocal());
    const local_dof_map_view_type local_to_dof_map = _dof_data->getDOFMap();
	const host_view_local_index_type bc_id = this->_particles->getFlags()->getLocalView<host_view_local_index_type>();
	const neighborhood_type * neighborhood = this->_particles->getNeighborhoodConst();
	const std::vector<Teuchos::RCP<fields_type> >& fields = this->_particles->getFieldManagerConst()->getVectorOfFields();

	//****************
	//
	//  Copying data from particles (std::vector's, multivectors, etc....) to views used by local reconstruction class
	//
	//****************

	// generate the interpolation operator and call the coefficients needed (storing them)
	const coords_type* target_coords = this->_coords;
	const coords_type* source_coords = this->_coords;

	size_t max_num_neighbors = neighborhood->computeMaxNumNeighbors(false /* use local processor max*/);

	Kokkos::View<int**> kokkos_neighbor_lists("neighbor lists", target_coords->nLocal(), max_num_neighbors+1);
	Kokkos::View<int**>::HostMirror kokkos_neighbor_lists_host = Kokkos::create_mirror_view(kokkos_neighbor_lists);

	// fill in the neighbor lists into a kokkos view. First entry is # of neighbors for that target
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
		const int num_i_neighbors = neighborhood->getNumNeighbors(i);
		for (int j=1; j<num_i_neighbors+1; ++j) {
			kokkos_neighbor_lists_host(i,j) = neighborhood->getNeighbor(i,j-1);
		}
		kokkos_neighbor_lists_host(i,0) = num_i_neighbors;
	});

	Kokkos::View<double**> kokkos_augmented_source_coordinates("source_coordinates", source_coords->nLocal(true /* include halo in count */), source_coords->nDim());
	Kokkos::View<double**>::HostMirror kokkos_augmented_source_coordinates_host = Kokkos::create_mirror_view(kokkos_augmented_source_coordinates);

	// fill in the source coords, adding regular with halo coordinates into a kokkos view
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,source_coords->nLocal(true /* include halo in count*/)), KOKKOS_LAMBDA(const int i) {
		xyz_type coordinate = source_coords->getLocalCoords(i, true /*include halo*/, use_physical_coords);
		kokkos_augmented_source_coordinates_host(i,0) = coordinate.x;
		kokkos_augmented_source_coordinates_host(i,1) = coordinate.y;
		kokkos_augmented_source_coordinates_host(i,2) = coordinate.z;
	});

	Kokkos::View<double**> kokkos_target_coordinates("target_coordinates", target_coords->nLocal(), target_coords->nDim());
	Kokkos::View<double**>::HostMirror kokkos_target_coordinates_host = Kokkos::create_mirror_view(kokkos_target_coordinates);
	// fill in the target, adding regular coordinates only into a kokkos view
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
		xyz_type coordinate = target_coords->getLocalCoords(i, false /*include halo*/, use_physical_coords);
		kokkos_target_coordinates_host(i,0) = coordinate.x;
		kokkos_target_coordinates_host(i,1) = coordinate.y;
		kokkos_target_coordinates_host(i,2) = coordinate.z;
	});

	auto epsilons = neighborhood->getHSupportSizes()->getLocalView<const host_view_type>();
	Kokkos::View<double*> kokkos_epsilons("epsilons", target_coords->nLocal());
	Kokkos::View<double*>::HostMirror kokkos_epsilons_host = Kokkos::create_mirror_view(kokkos_epsilons);
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
		kokkos_epsilons_host(i) = epsilons(i,0);
	});

	//****************
	//
	// End of data copying
	//
	//****************

	// GMLS operator

        GMLS my_scalar_GMLS(_parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"),
                            target_coords->nDim(),
                            _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("dense sovler type"),
                            _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("problem type"),
                            _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("constraint type"),
                            _parameters->get<Teuchos::ParameterList>("remap").get<int>("curvature porder"));
	my_scalar_GMLS.setProblemData(
			kokkos_neighbor_lists_host,
			kokkos_augmented_source_coordinates_host,
			kokkos_target_coordinates,
			kokkos_epsilons_host);
	my_scalar_GMLS.setWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("weighting type"));
	my_scalar_GMLS.setWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("weighting power"));
	my_scalar_GMLS.setCurvatureWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("curvature weighting type"));
	my_scalar_GMLS.setCurvatureWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("curvature weighting power"));

        GMLS my_vector_GMLS(ReconstructionSpace::VectorOfScalarClonesTaylorPolynomial,
                            ManifoldVectorPointSample,
                            _parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"),
                            target_coords->nDim(),
                            _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("dense sovler type"),
                            _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("problem type"),
                            _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("constraint type"),
                            _parameters->get<Teuchos::ParameterList>("remap").get<int>("curvature porder"));

	my_vector_GMLS.setProblemData(
			kokkos_neighbor_lists_host,
			kokkos_augmented_source_coordinates_host,
			kokkos_target_coordinates,
			kokkos_epsilons_host);
	my_vector_GMLS.setWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("weighting type"));
	my_vector_GMLS.setWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("weighting power"));
	my_vector_GMLS.setCurvatureWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("curvature weighting type"));
	my_vector_GMLS.setCurvatureWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("curvature weighting power"));


	bool use_velocity_interpolation = true;//true;
	bool use_staggered_gradient_fix = false;
	int mountain_source_data = 0; // 0-GMLS, 1-GMLS-Staggered
	bool use_staggered_divergence_fix = false;


	if (_particles->getFieldManager()->getIDOfFieldFromName("velocity") == field_one) {

		GMLS my_GMLS_staggered_grad (ReconstructionSpace::ScalarTaylorPolynomial,
                            StaggeredEdgeAnalyticGradientIntegralSample,
                            _parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"),
                            target_coords->nDim(),
                            _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("dense sovler type"),
                            _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("problem type"),
                            _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("constraint type"),
                            _parameters->get<Teuchos::ParameterList>("remap").get<int>("curvature porder"));

		my_GMLS_staggered_grad.setProblemData(
			kokkos_neighbor_lists_host,
			kokkos_augmented_source_coordinates_host,
			kokkos_target_coordinates,
			kokkos_epsilons_host);
		my_GMLS_staggered_grad.setWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("weighting type"));
		my_GMLS_staggered_grad.setWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("weighting power"));
		my_GMLS_staggered_grad.setCurvatureWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("curvature weighting type"));
		my_GMLS_staggered_grad.setCurvatureWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("curvature weighting power"));

		// add scalar sample targets
		my_scalar_GMLS.addTargets(TargetOperation::GradientOfScalarPointEvaluation);
//		my_scalar_GMLS.addTargets(TargetOperation::LaplacianOfScalarPointEvaluation);

		// add vector sample targets
		my_vector_GMLS.addTargets(TargetOperation::VectorPointEvaluation);
		my_vector_GMLS.addTargets(TargetOperation::VectorLaplacianPointEvaluation);

		if (use_staggered_gradient_fix) {
			my_GMLS_staggered_grad.addTargets(TargetOperation::GradientOfScalarPointEvaluation);
			my_GMLS_staggered_grad.generateAlphas();
		}
		my_scalar_GMLS.generateAlphas();
		my_vector_GMLS.generateAlphas();

        auto tangent_directions = my_scalar_GMLS.getTangentDirections();


		host_view_type b_data = _b->getLocalView<host_view_type>();
		host_view_type h_data = _particles->getFieldManager()->getFieldByName("height")->getMultiVectorPtrConst()->getLocalView<host_view_type>();
		host_view_type h_halo_data = _particles->getFieldManager()->getFieldByName("height")->getHaloMultiVectorPtrConst()->getLocalView<host_view_type>();
		host_view_type v_data = _particles->getFieldManager()->getFieldByName("velocity")->getMultiVectorPtrConst()->getLocalView<host_view_type>();
		host_view_type v_halo_data = _particles->getFieldManager()->getFieldByName("velocity")->getHaloMultiVectorPtrConst()->getLocalView<host_view_type>();

		Teuchos::RCP<Compadre::ShallowWaterTestCases> swtc = Teuchos::rcp(new Compadre::ShallowWaterTestCases(_parameters->get<int>("shallow water test case"), 0 /*alpha*/));
		swtc->setMountain();

		Teuchos::RCP<Compadre::CoriolisForce> cf = Teuchos::rcp(new Compadre::CoriolisForce(swtc->getOmega()));

        auto global_min_h = _particles->getNeighborhood()->computeMinHSupportSize(true /*global min*/);
		const double artificial_viscosity_coeff = _parameters->get<Teuchos::ParameterList>("physics").get<double>("artificial viscosity") * global_min_h * global_min_h;

		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,nlocal), KOKKOS_LAMBDA(const int i) {
			const local_index_type num_neighbors = neighborhood->getNumNeighbors(i);

			xyz_type xyz = _particles->getCoordsConst()->getLocalCoords(i, include_halo, use_physical_coords);
			xyz_type normal_direction = xyz / xyz.magnitude();
//			xyz_type normal_direction = xyz / swtc->getRadius();

			double coriolis_force = cf->evalScalar(xyz);

//						std::vector<std::vector<double> > P(3, std::vector<double>(3,0));
//						{
//							xyz_type xyz = _particles->getCoordsConst()->getLocalCoords(i, include_halo, use_physical_coords);
//							double r = 1;
//							double scaling = 1. / (r*r);
//
//							P[0][0] = r*r - xyz.x*xyz.x;
//							P[1][1] = r*r - xyz.y*xyz.y;
//							P[2][2] = r*r - xyz.z*xyz.z;
//
//							P[0][1] = -xyz.x*xyz.y;
//							P[0][2] = -xyz.x*xyz.z;
//							P[1][0] = -xyz.x*xyz.y;
//							P[2][0] = -xyz.x*xyz.z;
//
//							P[1][2] = -xyz.y*xyz.z;
//							P[2][1] = -xyz.y*xyz.z;
//
//							for (int j=0; j<3; ++j) {
//								for (int k=0; k<3; ++k) {
//									P[j][k] = P[j][k]*scaling;
//								}
//							}
//						}


            // local view of tangent directions
            scratch_matrix_right_type td
                    (tangent_directions.data() + i*3*3, 3, 3);


			if (bc_id(i, 0) == 0) {

				double reconstructed_h_grad_0_local = 0;
				double reconstructed_h_grad_1_local = 0;
				double reconstructed_h_grad_0_ambient = 0;
				double reconstructed_h_grad_1_ambient = 0;
				double reconstructed_h_grad_2_ambient = 0;
				double reconstructed_v0_local = 0;
				double reconstructed_v1_local = 0;
				double reconstructed_v0_ambient = 0;
				double reconstructed_v1_ambient = 0;
				double reconstructed_v2_ambient = 0;

				const local_index_type my_index = neighborhood->getNeighbor(i,0);
				double h_data_mine_0 = (my_index < nlocal) ? h_data(my_index, 0) : h_halo_data(my_index-nlocal, 0);
				double h_value_mine = swtc->evalScalar(xyz);
				double v_data_mine_0 = (my_index < nlocal) ? v_data(my_index, 0) : v_halo_data(my_index-nlocal, 0);
				double v_data_mine_1 = (my_index < nlocal) ? v_data(my_index, 1) : v_halo_data(my_index-nlocal, 1);
				double v_data_mine_2 = (my_index < nlocal) ? v_data(my_index, 2) : v_halo_data(my_index-nlocal, 2);

				for (local_index_type l = 0; l < num_neighbors; l++) {
					const local_index_type neighbor_index = neighborhood->getNeighbor(i,l);

					double h_data_0 = (neighbor_index < nlocal) ? h_data(neighbor_index, 0) : h_halo_data(neighbor_index-nlocal, 0);
					double v_data_0 = (neighbor_index < nlocal) ? v_data(neighbor_index, 0) : v_halo_data(neighbor_index-nlocal, 0);
					double v_data_1 = (neighbor_index < nlocal) ? v_data(neighbor_index, 1) : v_halo_data(neighbor_index-nlocal, 1);
					double v_data_2 = (neighbor_index < nlocal) ? v_data(neighbor_index, 2) : v_halo_data(neighbor_index-nlocal, 2);

					if (use_staggered_gradient_fix) {
						reconstructed_h_grad_0_local += h_data_0 * my_GMLS_staggered_grad.getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 0, l);
						reconstructed_h_grad_1_local += h_data_0 * my_GMLS_staggered_grad.getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 1, l);

						reconstructed_h_grad_0_local -= h_data_mine_0 * my_GMLS_staggered_grad.getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 0, l);
						reconstructed_h_grad_1_local -= h_data_mine_0 * my_GMLS_staggered_grad.getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 1, l);
					} else {
						reconstructed_h_grad_0_local += h_data_0 * my_scalar_GMLS.getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 0, l);
						reconstructed_h_grad_1_local += h_data_0 * my_scalar_GMLS.getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 1, l);
					}

					if (use_velocity_interpolation) {
						double contracted_data[2] = {0,0};
						for (local_index_type dim=0; dim<2; ++dim) {
							contracted_data[dim] += v_data_0 * my_vector_GMLS.getPreStencilWeight(ManifoldVectorPointSample, i, l, false /* for neighbor*/, dim, 0);
							contracted_data[dim] += v_data_1 * my_vector_GMLS.getPreStencilWeight(ManifoldVectorPointSample, i, l, false /* for neighbor*/, dim, 1);
							contracted_data[dim] += v_data_2 * my_vector_GMLS.getPreStencilWeight(ManifoldVectorPointSample, i, l, false /* for neighbor*/, dim, 2);
						}
						for (local_index_type dim=0; dim<2; ++dim) {
							reconstructed_v0_local += contracted_data[dim] * my_vector_GMLS.getAlpha1TensorTo1Tensor(TargetOperation::VectorPointEvaluation, i, 0, l, dim);
							reconstructed_v1_local += contracted_data[dim] * my_vector_GMLS.getAlpha1TensorTo1Tensor(TargetOperation::VectorPointEvaluation, i, 1, l, dim);
						}
					}


					if (_parameters->get<int>("shallow water test case") != 2) {
						xyz_type neighbor_xyz = _particles->getCoordsConst()->getLocalCoords(neighbor_index, include_halo, use_physical_coords);
						double h_value = swtc->evalScalar(neighbor_xyz);
						if (mountain_source_data==1 && use_staggered_gradient_fix) {
							reconstructed_h_grad_0_local += h_value * my_GMLS_staggered_grad.getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 0, l);
							reconstructed_h_grad_1_local += h_value * my_GMLS_staggered_grad.getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 1, l);

							reconstructed_h_grad_0_local -= h_value_mine * my_GMLS_staggered_grad.getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 0, l);
							reconstructed_h_grad_1_local -= h_value_mine * my_GMLS_staggered_grad.getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 1, l);
						} else if (mountain_source_data==0) {
							reconstructed_h_grad_0_local += h_value * my_scalar_GMLS.getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 0, l);
							reconstructed_h_grad_1_local += h_value * my_scalar_GMLS.getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 1, l);
						}
					}
				}
                

                // convert to ambient from local
                reconstructed_h_grad_0_ambient = td(0,0)*reconstructed_h_grad_0_local + td(1,0)*reconstructed_h_grad_1_local;
                reconstructed_h_grad_1_ambient = td(0,1)*reconstructed_h_grad_0_local + td(1,1)*reconstructed_h_grad_1_local;
                reconstructed_h_grad_2_ambient = td(0,2)*reconstructed_h_grad_0_local + td(1,2)*reconstructed_h_grad_1_local;

				if (use_velocity_interpolation) {
                    reconstructed_v0_ambient = td(0,0)*reconstructed_v0_local + td(1,0)*reconstructed_v1_local;
                    reconstructed_v1_ambient = td(0,1)*reconstructed_v0_local + td(1,1)*reconstructed_v1_local;
                    reconstructed_v2_ambient = td(0,2)*reconstructed_v0_local + td(1,2)*reconstructed_v1_local;
                } else {
					reconstructed_v0_ambient = v_data_mine_0;
					reconstructed_v1_ambient = v_data_mine_1;
					reconstructed_v2_ambient = v_data_mine_2;
                }


				double coriolis_1=0;
				double coriolis_2=0;
				double coriolis_3=0;

				// Eulerian
				double reconstructed_v0_grad0 = 0;
				double reconstructed_v0_grad1 = 0;
				double reconstructed_v0_grad2 = 0;
				double reconstructed_v1_grad0 = 0;
				double reconstructed_v1_grad1 = 0;
				double reconstructed_v1_grad2 = 0;
				double reconstructed_v2_grad0 = 0;
				double reconstructed_v2_grad1 = 0;
				double reconstructed_v2_grad2 = 0;

				if (EULERIAN) {
					for (local_index_type l = 0; l < num_neighbors; l++) {
						const local_index_type neighbor_index = neighborhood->getNeighbor(i,l);

						double v_data_0 = (neighbor_index < nlocal) ? v_data(neighbor_index, 0) : v_halo_data(neighbor_index-nlocal, 0);
						double v_data_1 = (neighbor_index < nlocal) ? v_data(neighbor_index, 1) : v_halo_data(neighbor_index-nlocal, 1);
						double v_data_2 = (neighbor_index < nlocal) ? v_data(neighbor_index, 2) : v_halo_data(neighbor_index-nlocal, 2);

						{
							reconstructed_v0_grad0 += v_data_0 * my_scalar_GMLS.getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 0, l);
							reconstructed_v0_grad1 += v_data_0 * my_scalar_GMLS.getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 1, l);
							reconstructed_v0_grad2 += v_data_0 * my_scalar_GMLS.getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 2, l);
						}
						{
							reconstructed_v1_grad0 += v_data_1 * my_scalar_GMLS.getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 0, l);
							reconstructed_v1_grad1 += v_data_1 * my_scalar_GMLS.getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 1, l);
							reconstructed_v1_grad2 += v_data_1 * my_scalar_GMLS.getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 2, l);
						}
						{
							reconstructed_v2_grad0 += v_data_2 * my_scalar_GMLS.getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 0, l);
							reconstructed_v2_grad1 += v_data_2 * my_scalar_GMLS.getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 1, l);
							reconstructed_v2_grad2 += v_data_2 * my_scalar_GMLS.getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 2, l);
						}
					}
				}

				// remove normal contribution of forces
				double force[3] = {-coriolis_force * (normal_direction.y*reconstructed_v2_ambient - reconstructed_v1_ambient*normal_direction.z) - _gravity*reconstructed_h_grad_0_ambient,
						coriolis_force * (normal_direction.x*reconstructed_v2_ambient - reconstructed_v0_ambient*normal_direction.z) - _gravity*reconstructed_h_grad_1_ambient,
						-coriolis_force * (normal_direction.x*reconstructed_v1_ambient - reconstructed_v0_ambient*normal_direction.y) - _gravity*reconstructed_h_grad_2_ambient};

//				double dot_product = force[0]*normal_direction.x + force[1]*normal_direction.y + force[2]*normal_direction.z;
//				force[0] -= dot_product * normal_direction.x;
//				force[1] -= dot_product * normal_direction.y;
//				force[2] -= dot_product * normal_direction.z;


				// remove normal contribution of forces
				double advective_force[3] = {-(reconstructed_v0_ambient*reconstructed_v0_grad0 + reconstructed_v1_ambient*reconstructed_v0_grad1 + reconstructed_v2_ambient*reconstructed_v0_grad2),
						-(reconstructed_v0_ambient*reconstructed_v1_grad0 + reconstructed_v1_ambient*reconstructed_v1_grad1 + reconstructed_v2_ambient*reconstructed_v1_grad2),
						-(reconstructed_v0_ambient*reconstructed_v2_grad0 + reconstructed_v1_ambient*reconstructed_v2_grad1 + reconstructed_v2_ambient*reconstructed_v2_grad2)};

//				double advective_dot_product = advective_force[0]*normal_direction.x + advective_force[1]*normal_direction.y + advective_force[2]*normal_direction.z;
//				advective_force[0] -= advective_dot_product * normal_direction.x;
//				advective_force[1] -= advective_dot_product * normal_direction.y;
//				advective_force[2] -= advective_dot_product * normal_direction.z;


				double artificial_diffusion_force[3] = {0, 0, 0};

				if (artificial_viscosity_coeff > 0) {

					double reconstructed_laplace_u_0_local = 0;
					double reconstructed_laplace_u_1_local = 0;

					for (local_index_type l = 0; l < num_neighbors; l++) {
						const local_index_type neighbor_index = neighborhood->getNeighbor(i,l);

						double v_data_0 = (neighbor_index < nlocal) ? v_data(neighbor_index, 0) : v_halo_data(neighbor_index-nlocal, 0);
						double v_data_1 = (neighbor_index < nlocal) ? v_data(neighbor_index, 1) : v_halo_data(neighbor_index-nlocal, 1);
						double v_data_2 = (neighbor_index < nlocal) ? v_data(neighbor_index, 2) : v_halo_data(neighbor_index-nlocal, 2);

						double contracted_data[2] = {0,0};
						for (local_index_type dim=0; dim<2; ++dim) {
							contracted_data[dim] += v_data_0 * my_vector_GMLS.getPreStencilWeight(ManifoldVectorPointSample, i, l, false /* for neighbor*/, dim, 0);
							contracted_data[dim] += v_data_1 * my_vector_GMLS.getPreStencilWeight(ManifoldVectorPointSample, i, l, false /* for neighbor*/, dim, 1);
							contracted_data[dim] += v_data_2 * my_vector_GMLS.getPreStencilWeight(ManifoldVectorPointSample, i, l, false /* for neighbor*/, dim, 2);
						}
						for (local_index_type dim=0; dim<2; ++dim) {
							reconstructed_laplace_u_0_local += contracted_data[dim] * my_vector_GMLS.getAlpha1TensorTo1Tensor(TargetOperation::VectorLaplacianPointEvaluation, i, 0, l, dim);
							reconstructed_laplace_u_1_local += contracted_data[dim] * my_vector_GMLS.getAlpha1TensorTo1Tensor(TargetOperation::VectorLaplacianPointEvaluation, i, 1, l, dim);
						}
					}
                    double reconstructed_laplace_u_0_ambient = td(0,0)*reconstructed_v0_local + td(1,0)*reconstructed_v1_local;
                    double reconstructed_laplace_u_1_ambient = td(0,1)*reconstructed_v0_local + td(1,1)*reconstructed_v1_local;
                    double reconstructed_laplace_u_2_ambient = td(0,2)*reconstructed_v0_local + td(1,2)*reconstructed_v1_local;

					artificial_diffusion_force[0] = artificial_viscosity_coeff * reconstructed_laplace_u_0_ambient;
					artificial_diffusion_force[1] = artificial_viscosity_coeff * reconstructed_laplace_u_1_ambient;
					artificial_diffusion_force[2] = artificial_viscosity_coeff * reconstructed_laplace_u_2_ambient;

//					double artificial_diffusion_dot_product = artificial_diffusion_force[0]*normal_direction.x + artificial_diffusion_force[1]*normal_direction.y + artificial_diffusion_force[2]*normal_direction.z;
//					artificial_diffusion_force[0] -= artificial_diffusion_dot_product * normal_direction.x;
//					artificial_diffusion_force[1] -= artificial_diffusion_dot_product * normal_direction.y;
//					artificial_diffusion_force[2] -= artificial_diffusion_dot_product * normal_direction.z;
				}


				for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
					local_index_type row = local_to_dof_map(i, field_one, k);

					b_data(row,0) = 0;

//					double coriolis_constant = 2*swtc->getOmega()*xyz.z / (swtc->getRadius()*swtc->getRadius());
//					if (k==0) {
//						b_data(row, 0) = coriolis_constant*(xyz.z*reconstructed_v1 - xyz.y*reconstructed_v2) - _gravity*reconstructed_h_grad_0;
//					} else if (k==1) {
//						b_data(row, 0) = -coriolis_constant*(xyz.z*reconstructed_v0 - xyz.x*reconstructed_v2) - _gravity*reconstructed_h_grad_1;
//					} else if (k==2) {
//						b_data(row, 0) = coriolis_constant*(xyz.y*reconstructed_v0 - xyz.x*reconstructed_v1) - _gravity*reconstructed_h_grad_2;
//					}

					if (k==0) {
						b_data(row, 0) -= 1./ (swtc->getRadius()*swtc->getRadius()) * xyz.x * (reconstructed_v0_ambient*reconstructed_v0_ambient + reconstructed_v1_ambient*reconstructed_v1_ambient + reconstructed_v2_ambient*reconstructed_v2_ambient);
					} else if (k==1) {
						b_data(row, 0) -= 1./ (swtc->getRadius()*swtc->getRadius()) * xyz.y * (reconstructed_v0_ambient*reconstructed_v0_ambient + reconstructed_v1_ambient*reconstructed_v1_ambient + reconstructed_v2_ambient*reconstructed_v2_ambient);
					} else if (k==2) {
						b_data(row, 0) -= 1./ (swtc->getRadius()*swtc->getRadius()) * xyz.z * (reconstructed_v0_ambient*reconstructed_v0_ambient + reconstructed_v1_ambient*reconstructed_v1_ambient + reconstructed_v2_ambient*reconstructed_v2_ambient);
					}

					if (k==0) {
						b_data(row, 0) += force[0];//-coriolis_force * (normal_direction.y*reconstructed_v2 - reconstructed_v1*normal_direction.z) - _gravity*reconstructed_h_grad_0;
						coriolis_1 += -coriolis_force * (normal_direction.y*reconstructed_v2_ambient - reconstructed_v1_ambient*normal_direction.z);
					} else if (k==1) {
						b_data(row, 0) += force[1];//coriolis_force * (normal_direction.x*reconstructed_v2 - reconstructed_v0*normal_direction.z) - _gravity*reconstructed_h_grad_1;
						coriolis_2 += coriolis_force * (normal_direction.x*reconstructed_v2_ambient - reconstructed_v0_ambient*normal_direction.z);

					} else if (k==2) {
						b_data(row, 0) += force[2];//-coriolis_force * (normal_direction.x*reconstructed_v1 - reconstructed_v0*normal_direction.y) - _gravity*reconstructed_h_grad_2;
						coriolis_3 += -coriolis_force * (normal_direction.x*reconstructed_v1_ambient - reconstructed_v0_ambient*normal_direction.y);
					}


					if (EULERIAN) {
						if (k==0) {
							b_data(row, 0) += advective_force[0];
//							b_data(row, 0) += -(reconstructed_v0*reconstructed_v0_grad0 + reconstructed_v1*reconstructed_v1_grad0 + reconstructed_v2*reconstructed_v2_grad0);
						} else if (k==1) {
							b_data(row, 0) += advective_force[1];
//							b_data(row, 0) += -(reconstructed_v0*reconstructed_v0_grad1 + reconstructed_v1*reconstructed_v1_grad1 + reconstructed_v2*reconstructed_v2_grad1);
						} else if (k==2) {
							b_data(row, 0) += advective_force[2];
//							b_data(row, 0) += -(reconstructed_v0*reconstructed_v0_grad2 + reconstructed_v1*reconstructed_v1_grad2 + reconstructed_v2*reconstructed_v2_grad2);
						}
					}

					if (artificial_viscosity_coeff > 0) {
						// artificial viscosity
						if (k==0) {
							b_data(row,0) -= artificial_diffusion_force[0];
						} else if (k==1) {
							b_data(row,0) -= artificial_diffusion_force[1];
						} else if (k==2) {
							b_data(row,0) -= artificial_diffusion_force[2];
						}
					}
				}
			}
		});

	} else if (_particles->getFieldManager()->getIDOfFieldFromName("height") == field_one) {

		GMLS my_GMLS_staggered_div (ReconstructionSpace::VectorTaylorPolynomial,
				StaggeredEdgeIntegralSample,
                                _parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"),
                                target_coords->nDim(),
                                _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("dense sovler type"),
                                _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("problem type"),
                                _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("constraint type"),
                                _parameters->get<Teuchos::ParameterList>("remap").get<int>("curvature porder"));
		my_GMLS_staggered_div.setProblemData(
			kokkos_neighbor_lists_host,
			kokkos_augmented_source_coordinates_host,
			kokkos_target_coordinates,
			kokkos_epsilons_host);
		my_GMLS_staggered_div.setWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("weighting type"));
		my_GMLS_staggered_div.setWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("weighting power"));
		my_GMLS_staggered_div.setCurvatureWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("curvature weighting type"));
		my_GMLS_staggered_div.setCurvatureWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("curvature weighting power"));
		my_GMLS_staggered_div.setOrderOfQuadraturePoints(_parameters->get<Teuchos::ParameterList>("remap").get<int>("quadrature order"));
		my_GMLS_staggered_div.setDimensionOfQuadraturePoints(_parameters->get<Teuchos::ParameterList>("remap").get<int>("quadrature dimension"));
		my_GMLS_staggered_div.setQuadratureType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("quadrature type"));

		my_scalar_GMLS.addTargets(TargetOperation::ScalarPointEvaluation);
		// for Eulerian
		my_scalar_GMLS.addTargets(TargetOperation::GradientOfScalarPointEvaluation);
		if (use_staggered_divergence_fix) {
			my_GMLS_staggered_div.addTargets(TargetOperation::DivergenceOfVectorPointEvaluation);
			my_GMLS_staggered_div.generateAlphas();
		} else {
			my_vector_GMLS.addTargets(TargetOperation::DivergenceOfVectorPointEvaluation);
			my_vector_GMLS.generateAlphas();
		}
		my_scalar_GMLS.generateAlphas();

        auto tangent_directions = my_scalar_GMLS.getTangentDirections();


		host_view_type b_data = _b->getLocalView<host_view_type>();
		host_view_type h_data = _particles->getFieldManager()->getFieldByName("height")->getMultiVectorPtrConst()->getLocalView<host_view_type>();
		host_view_type h_halo_data = _particles->getFieldManager()->getFieldByName("height")->getHaloMultiVectorPtrConst()->getLocalView<host_view_type>();
		host_view_type v_data = _particles->getFieldManager()->getFieldByName("velocity")->getMultiVectorPtrConst()->getLocalView<host_view_type>();
		host_view_type v_halo_data = _particles->getFieldManager()->getFieldByName("velocity")->getHaloMultiVectorPtrConst()->getLocalView<host_view_type>();


		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,nlocal), KOKKOS_LAMBDA(const int i) {

			const local_index_type num_neighbors = neighborhood->getNumNeighbors(i);

//			std::vector<std::vector<double> > P(3, std::vector<double>(3,0));
//			{
//				xyz_type xyz = _particles->getCoordsConst()->getLocalCoords(i, include_halo, use_physical_coords);
//				double r = 1;
//				double scaling = 1. / (r*r);
//
//				P[0][0] = r*r - xyz.x*xyz.x;
//				P[1][1] = r*r - xyz.y*xyz.y;
//				P[2][2] = r*r - xyz.z*xyz.z;
//
//				P[0][1] = -xyz.x*xyz.y;
//				P[0][2] = -xyz.x*xyz.z;
//				P[1][0] = -xyz.x*xyz.y;
//				P[2][0] = -xyz.x*xyz.z;
//
//				P[1][2] = -xyz.y*xyz.z;
//				P[2][1] = -xyz.y*xyz.z;
//
//				for (int j=0; j<3; ++j) {
//					for (int k=0; k<3; ++k) {
//						P[j][k] = P[j][k]*scaling;
//					}
//				}
//			}

//			if (bc_id(i, 0) == 0) {

                // local view of tangent directions
                scratch_matrix_right_type td
                        (tangent_directions.data() + i*3*3, 3, 3);

				local_index_type row = local_to_dof_map(i, field_one, 0);

				double reconstructed_h = 0;
				double reconstructed_div_v = 0;

                // Reconstruction needed for Eulerian case only
				double reconstructed_v0_local = 0;
				double reconstructed_v1_local = 0;
				double reconstructed_v0_ambient = 0;
				double reconstructed_v1_ambient = 0;
				double reconstructed_v2_ambient = 0;
				double reconstructed_h_grad0_local = 0;
				double reconstructed_h_grad1_local = 0;
				double reconstructed_h_grad0_ambient = 0;
				double reconstructed_h_grad1_ambient = 0;
				double reconstructed_h_grad2_ambient = 0;

				const local_index_type my_index = neighborhood->getNeighbor(i,0);
				double v_data_mine_0 = (my_index < nlocal) ? v_data(my_index, 0) : v_halo_data(my_index-nlocal, 0);
				double v_data_mine_1 = (my_index < nlocal) ? v_data(my_index, 1) : v_halo_data(my_index-nlocal, 1);
				double v_data_mine_2 = (my_index < nlocal) ? v_data(my_index, 2) : v_halo_data(my_index-nlocal, 2);

				if (EULERIAN) {
				    if (use_velocity_interpolation) {
				    	for (local_index_type l = 0; l < num_neighbors; l++) {
				    		const local_index_type neighbor_index = neighborhood->getNeighbor(i,l);
				    		double v_data_0 = (neighbor_index < nlocal) ? v_data(neighbor_index, 0) : v_halo_data(neighbor_index-nlocal, 0);
				    		double v_data_1 = (neighbor_index < nlocal) ? v_data(neighbor_index, 1) : v_halo_data(neighbor_index-nlocal, 1);
				    		double v_data_2 = (neighbor_index < nlocal) ? v_data(neighbor_index, 2) : v_halo_data(neighbor_index-nlocal, 2);

				    		double contracted_data[2] = {0,0};
				    		for (local_index_type dim=0; dim<2; ++dim) {
				    			contracted_data[dim] += v_data_0 * my_vector_GMLS.getPreStencilWeight(ManifoldVectorPointSample, i, l, false /* for neighbor*/, dim, 0);
				    			contracted_data[dim] += v_data_1 * my_vector_GMLS.getPreStencilWeight(ManifoldVectorPointSample, i, l, false /* for neighbor*/, dim, 1);
				    			contracted_data[dim] += v_data_2 * my_vector_GMLS.getPreStencilWeight(ManifoldVectorPointSample, i, l, false /* for neighbor*/, dim, 2);
				    		}
				    		for (local_index_type dim=0; dim<2; ++dim) {
				    			reconstructed_v0_local += contracted_data[dim] * my_vector_GMLS.getAlpha1TensorTo1Tensor(TargetOperation::VectorPointEvaluation, i, 0, l, dim);
				    			reconstructed_v1_local += contracted_data[dim] * my_vector_GMLS.getAlpha1TensorTo1Tensor(TargetOperation::VectorPointEvaluation, i, 1, l, dim);
				    		}
				    	}

                        // local view of tangent directions
                        scratch_matrix_right_type td
                                (tangent_directions.data() + i*3*3, 3, 3);

                        // convert to ambient from local
                        reconstructed_v0_ambient = td(0,0)*reconstructed_v0_local + td(1,0)*reconstructed_v1_local;
                        reconstructed_v1_ambient = td(0,1)*reconstructed_v0_local + td(1,1)*reconstructed_v1_local;
                        reconstructed_v2_ambient = td(0,2)*reconstructed_v0_local + td(1,2)*reconstructed_v1_local;

				    } else {
				    	reconstructed_v0_ambient = v_data_mine_0;
				    	reconstructed_v1_ambient = v_data_mine_1;
				    	reconstructed_v2_ambient = v_data_mine_2;
				    }
                }


				for (local_index_type l = 0; l < num_neighbors; l++) {
					const local_index_type neighbor_index = neighborhood->getNeighbor(i,l);

					double h_data_0 = (neighbor_index < nlocal) ? h_data(neighbor_index, 0) : h_halo_data(neighbor_index-nlocal, 0);

					double v_data_0 = (neighbor_index < nlocal) ? v_data(neighbor_index, 0) : v_halo_data(neighbor_index-nlocal, 0);
					double v_data_1 = (neighbor_index < nlocal) ? v_data(neighbor_index, 1) : v_halo_data(neighbor_index-nlocal, 1);
					double v_data_2 = (neighbor_index < nlocal) ? v_data(neighbor_index, 2) : v_halo_data(neighbor_index-nlocal, 2);

					reconstructed_h += h_data_0 * my_scalar_GMLS.getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, l);


					if (use_staggered_divergence_fix) {

						double contracted_data = 0;
						contracted_data += v_data_0 * my_GMLS_staggered_div.getPreStencilWeight(StaggeredEdgeIntegralSample, i, l, false /* for neighbor*/, 0, 0);
						contracted_data += v_data_1 * my_GMLS_staggered_div.getPreStencilWeight(StaggeredEdgeIntegralSample, i, l, false /* for neighbor*/, 0, 1);
						contracted_data += v_data_2 * my_GMLS_staggered_div.getPreStencilWeight(StaggeredEdgeIntegralSample, i, l, false /* for neighbor*/, 0, 2);
						contracted_data += v_data_mine_0 * my_GMLS_staggered_div.getPreStencilWeight(StaggeredEdgeIntegralSample, i, l, true /* for neighbor*/, 0, 0);
						contracted_data += v_data_mine_1 * my_GMLS_staggered_div.getPreStencilWeight(StaggeredEdgeIntegralSample, i, l, true /* for neighbor*/, 0, 1);
						contracted_data += v_data_mine_2 * my_GMLS_staggered_div.getPreStencilWeight(StaggeredEdgeIntegralSample, i, l, true /* for neighbor*/, 0, 2);
						reconstructed_div_v += contracted_data * my_GMLS_staggered_div.getAlpha0TensorTo0Tensor(TargetOperation::DivergenceOfVectorPointEvaluation, i, l);

					} else {
						double contracted_data[2] = {0,0};
						for (local_index_type dim=0; dim<2; ++dim) {
							contracted_data[dim] += v_data_0 * my_vector_GMLS.getPreStencilWeight(ManifoldVectorPointSample, i, l, false /* for neighbor*/, dim, 0);
							contracted_data[dim] += v_data_1 * my_vector_GMLS.getPreStencilWeight(ManifoldVectorPointSample, i, l, false /* for neighbor*/, dim, 1);
							contracted_data[dim] += v_data_2 * my_vector_GMLS.getPreStencilWeight(ManifoldVectorPointSample, i, l, false /* for neighbor*/, dim, 2);
						}
						for (local_index_type dim=0; dim<2; ++dim) {
							reconstructed_div_v += contracted_data[dim] * my_vector_GMLS.getAlpha1TensorTo1Tensor(TargetOperation::DivergenceOfVectorPointEvaluation, i, 0, l, dim);
						}
					}

					if (EULERIAN) {
						reconstructed_h_grad0_local += h_data_0 * my_scalar_GMLS.getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 0, l);
						reconstructed_h_grad1_local += h_data_0 * my_scalar_GMLS.getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 1, l);
					}
				}

                // convert to ambient from local
                reconstructed_h_grad0_ambient = td(0,0)*reconstructed_h_grad0_local + td(1,0)*reconstructed_h_grad1_local;
                reconstructed_h_grad1_ambient = td(0,1)*reconstructed_h_grad0_local + td(1,1)*reconstructed_h_grad1_local;
                reconstructed_h_grad2_ambient = td(0,2)*reconstructed_h_grad0_local + td(1,2)*reconstructed_h_grad1_local;


//				printf("%f %f\n", reconstructed_div_v, reconstructed_h);
				b_data(row, 0) = - reconstructed_h * reconstructed_div_v;

				// Eulerian
				if (EULERIAN) {
//					double transformed_h_grad0 = P[0][0]*reconstructed_h_grad0 + P[0][1]*reconstructed_h_grad1 + P[0][2]*reconstructed_h_grad2;
//					double transformed_h_grad1 = P[1][0]*reconstructed_h_grad0 + P[1][1]*reconstructed_h_grad1 + P[1][2]*reconstructed_h_grad2;
//					double transformed_h_grad2 = P[2][0]*reconstructed_h_grad0 + P[2][1]*reconstructed_h_grad1 + P[2][2]*reconstructed_h_grad2;
//					b_data(row, 0) -= (reconstructed_v0*transformed_h_grad0 + reconstructed_v1*transformed_h_grad1 + reconstructed_v2*transformed_h_grad2);
					b_data(row, 0) -= (reconstructed_v0_ambient*reconstructed_h_grad0_ambient + reconstructed_v1_ambient*reconstructed_h_grad1_ambient + reconstructed_v2_ambient*reconstructed_h_grad2_ambient);
				}
//			}

		});

	} else if (_particles->getFieldManager()->getIDOfFieldFromName("displacement") == field_one) {

		if (use_velocity_interpolation) {
			my_vector_GMLS.addTargets(TargetOperation::VectorPointEvaluation);
			my_vector_GMLS.generateAlphas();
		}
        auto tangent_directions = my_vector_GMLS.getTangentDirections();

		host_view_type b_data = _b->getLocalView<host_view_type>();
		host_view_type v_data = _particles->getFieldManager()->getFieldByName("velocity")->getMultiVectorPtrConst()->getLocalView<host_view_type>();
		host_view_type v_halo_data = _particles->getFieldManager()->getFieldByName("velocity")->getHaloMultiVectorPtrConst()->getLocalView<host_view_type>();

		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,nlocal), KOKKOS_LAMBDA(const int i) {

			//xyz_type xyz = _particles->getCoordsConst()->getLocalCoords(i, include_halo, use_physical_coords);
			//xyz_type normal_direction = xyz / xyz.magnitude();

			const local_index_type num_neighbors = neighborhood->getNumNeighbors(i);

			if (bc_id(i, 0) == 0) {

				double reconstructed_v0_local = 0;
				double reconstructed_v1_local = 0;
				double reconstructed_v0_ambient = 0;
				double reconstructed_v1_ambient = 0;
				double reconstructed_v2_ambient = 0;

				if (use_velocity_interpolation) {
					for (local_index_type l = 0; l < num_neighbors; l++) {
						const local_index_type neighbor_index = neighborhood->getNeighbor(i,l);
						double v_data_0 = (neighbor_index < nlocal) ? v_data(neighbor_index, 0) : v_halo_data(neighbor_index-nlocal, 0);
						double v_data_1 = (neighbor_index < nlocal) ? v_data(neighbor_index, 1) : v_halo_data(neighbor_index-nlocal, 1);
						double v_data_2 = (neighbor_index < nlocal) ? v_data(neighbor_index, 2) : v_halo_data(neighbor_index-nlocal, 2);

						double contracted_data[2] = {0,0};
						for (local_index_type dim=0; dim<2; ++dim) {
							contracted_data[dim] += v_data_0 * my_vector_GMLS.getPreStencilWeight(ManifoldVectorPointSample, i, l, false /* for neighbor*/, dim, 0);
							contracted_data[dim] += v_data_1 * my_vector_GMLS.getPreStencilWeight(ManifoldVectorPointSample, i, l, false /* for neighbor*/, dim, 1);
							contracted_data[dim] += v_data_2 * my_vector_GMLS.getPreStencilWeight(ManifoldVectorPointSample, i, l, false /* for neighbor*/, dim, 2);
						}
						for (local_index_type dim=0; dim<2; ++dim) {
							reconstructed_v0_local += contracted_data[dim] * my_vector_GMLS.getAlpha1TensorTo1Tensor(TargetOperation::VectorPointEvaluation, i, 0, l, dim);
							reconstructed_v1_local += contracted_data[dim] * my_vector_GMLS.getAlpha1TensorTo1Tensor(TargetOperation::VectorPointEvaluation, i, 1, l, dim);
						}
					}

                    // local view of tangent directions
                    scratch_matrix_right_type td
                            (tangent_directions.data() + i*3*3, 3, 3);

                    // convert to ambient from local
                    reconstructed_v0_ambient = td(0,0)*reconstructed_v0_local + td(1,0)*reconstructed_v1_local;
                    reconstructed_v1_ambient = td(0,1)*reconstructed_v0_local + td(1,1)*reconstructed_v1_local;
                    reconstructed_v2_ambient = td(0,2)*reconstructed_v0_local + td(1,2)*reconstructed_v1_local;

				} else {
					const local_index_type my_index = neighborhood->getNeighbor(i,0);
					double v_data_mine_0 = (my_index < nlocal) ? v_data(my_index, 0) : v_halo_data(my_index, 0);
					double v_data_mine_1 = (my_index < nlocal) ? v_data(my_index, 1) : v_halo_data(my_index, 1);
					double v_data_mine_2 = (my_index < nlocal) ? v_data(my_index, 2) : v_halo_data(my_index, 2);
					reconstructed_v0_ambient = v_data_mine_0;
					reconstructed_v1_ambient = v_data_mine_1;
					reconstructed_v2_ambient = v_data_mine_2;
				}
//				const local_index_type my_index = neighborhood->getNeighbor(i,0);
//				reconstructed_v0 = (my_index < nlocal) ? v_data(my_index, 0) : v_halo_data(my_index-nlocal, 0);
//				reconstructed_v1 = (my_index < nlocal) ? v_data(my_index, 1) : v_halo_data(my_index-nlocal, 1);
//				reconstructed_v2 = (my_index < nlocal) ? v_data(my_index, 2) : v_halo_data(my_index-nlocal, 2);

//				printf ("%f %f %f\n", reconstructed_v0, reconstructed_v1, reconstructed_v2);

//				for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
//					local_index_type row = local_to_dof_map[i][field_one][k];
//					if (k==0) {
//						b_data(row, 0) = reconstructed_v0;
//					} else if (k==1) {
//						b_data(row, 0) = reconstructed_v1;
//					} else if (k==2) {
//						b_data(row, 0) = reconstructed_v2;
//					}
//				}



				// remove normal contribution of forces
				double force[3] = {reconstructed_v0_ambient,
									reconstructed_v1_ambient,
									reconstructed_v2_ambient};

//				double dot_product = force[0]*normal_direction.x + force[1]*normal_direction.y + force[2]*normal_direction.z;
//				force[0] -= dot_product * normal_direction.x;
//				force[1] -= dot_product * normal_direction.y;
//				force[2] -= dot_product * normal_direction.z;

				for (local_index_type k = 0; k < fields[field_one]->nDim(); ++k) {
					local_index_type row = local_to_dof_map(i, field_one, k);
					if (k==0) {
						b_data(row, 0) = force[0];
					} else if (k==1) {
						b_data(row, 0) = force[1];
					} else if (k==2) {
						b_data(row, 0) = force[2];
					}
				}

			}

		});

	}

	ComputeMatrixTime->stop();

}

const std::vector<InteractingFields> LagrangianShallowWaterPhysics::gatherFieldInteractions() {
	std::vector<InteractingFields> field_interactions;
//	field_interactions.push_back(InteractingFields(op_needing_interaction::physics, _particles->getFieldManagerConst()->getIDOfFieldFromName("displacement"), _particles->getFieldManagerConst()->getIDOfFieldFromName("displacement")));
	field_interactions.push_back(InteractingFields(op_needing_interaction::physics, _particles->getFieldManagerConst()->getIDOfFieldFromName("height")));
	field_interactions.push_back(InteractingFields(op_needing_interaction::physics, _particles->getFieldManagerConst()->getIDOfFieldFromName("velocity")));
	if (_particles->getCoordsConst()->isLagrangian())
		field_interactions.push_back(InteractingFields(op_needing_interaction::physics, _particles->getFieldManagerConst()->getIDOfFieldFromName("displacement")));

	//	printf("op: %d %d\n", _particles->getFieldManagerConst()->getIDOfFieldFromName("velocity"), _particles->getFieldManagerConst()->getIDOfFieldFromName("height"));
	return field_interactions;
}
    
}
