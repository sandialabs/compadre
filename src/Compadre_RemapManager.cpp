#include "Compadre_RemapManager.hpp"

#include "Compadre_XyzVector.hpp"
#include "Compadre_CoordsT.hpp"
#include "Compadre_ParticlesT.hpp"
#include "Compadre_FieldManager.hpp"
#include "Compadre_FieldT.hpp"

#include "Compadre_OBFET.hpp"

#ifdef COMPADRE_USE_NANOFLANN
#include "Compadre_nanoflannInformation.hpp"
#include "Compadre_nanoflannPointCloudT.hpp"
#endif

#include <Compadre_GMLS.hpp>
#include <Compadre_Remap.hpp>



namespace Compadre {

typedef Compadre::NanoFlannInformation nanoflann_neighbors_type;
typedef Compadre::NeighborhoodT neighbors_type;

typedef Compadre::CoordsT coords_type;
typedef Compadre::XyzVector xyz_type;

void RemapManager::execute(bool keep_neighborhoods, bool use_physical_coords) {

	// currently this neighborhood search is always in physical coords for Lagrangian simulation, but it may need to be in Lagrangian

	if (_queue.size()>0) {

	// TODO: this function must some day organize by the operation requested in the objects in the queue
	// For now, they all have the same operation, so we can simply reused neighborhoods and coefficients

//	if (_neighborhoodInfo.is_null() || _GMLS.is_null()) {
		std::string method =_parameters->get<Teuchos::ParameterList>("neighborhood").get<std::string>("method");
		transform(method.begin(), method.end(), method.begin(), ::tolower);


		if (method=="nanoflann") {
#ifdef COMPADRE_USE_NANOFLANN
			local_index_type maxLeaf = _parameters->get<Teuchos::ParameterList>("neighborhood").get<int>("max leaf");
			_neighborhoodInfo = Teuchos::rcp_static_cast<neighbors_type>(Teuchos::rcp(
					new nanoflann_neighbors_type(_src_particles, _parameters, maxLeaf, _trg_particles, use_physical_coords)));
#else
			TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Nanoflann selected as search method but not built against Nanoflann.");
#endif
		}
		else {
			TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Selected a search method that is not nanoflann.");
		}


		local_index_type neighbors_needed;

		std::string solver_type_to_lower = _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("dense linear solver");
		transform(solver_type_to_lower.begin(), solver_type_to_lower.end(), solver_type_to_lower.begin(), ::tolower);

		if (solver_type_to_lower == "manifold") {
			int max_order = std::max(_parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"), _parameters->get<Teuchos::ParameterList>("remap").get<int>("curvature porder"));
			neighbors_needed = GMLS::getNP(max_order, 2);
		} else {
			neighbors_needed = GMLS::getNP(_parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"), 3);
		}

		local_index_type extra_neighbors = _parameters->get<Teuchos::ParameterList>("remap").get<double>("neighbors needed multiplier") * neighbors_needed;

		_neighborhoodInfo->constructAllNeighborList(_max_radius, extra_neighbors,
				_parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("size"),
				_parameters->get<Teuchos::ParameterList>("neighborhood").get<int>("max leaf"),
				use_physical_coords);

//		// test to see if distances between points as measured by neighborhood search matches the distance between points using use_physical_coords
//		for (local_index_type i=0; i<_trg_particles->getCoordsConst()->nLocal(); i++) {
//			const std::vector<std::pair<size_t, scalar_type> > neighbors = _neighborhoodInfo->getNeighbors(i);
//			const local_index_type num_neighbors = neighbors.size();
//			std::cout << std::setprecision(16) << _trg_particles->getCoordsConst()->getLocalCoords(i,false,use_physical_coords).x  << ", " << _trg_particles->getCoordsConst()->getLocalCoords(i,false,use_physical_coords).y << ", " << _trg_particles->getCoordsConst()->getLocalCoords(i,false,use_physical_coords).z << std::endl;
//			for (local_index_type l = 0; l < num_neighbors; l++) {
//				std::cout << i << ": " << neighbors[l].first << " ( " << _src_particles->getCoordsConst()->getLocalCoords(neighbors[l].first,true,use_physical_coords).x << ", " << _src_particles->getCoordsConst()->getLocalCoords(neighbors[l].first,true,use_physical_coords).y << ", " << _src_particles->getCoordsConst()->getLocalCoords(neighbors[l].first,true,use_physical_coords).z << " ) " << neighbors[l].second << std::endl;
//			}
//		}

		//****************
		//
		//  Copying data from particles (std::vector's, multivectors, etc....) to views used by local reconstruction class
		//
		//****************

		const std::vector<std::vector<std::pair<size_t, scalar_type> > >& all_neighbors = _neighborhoodInfo->getAllNeighbors();

		size_t max_num_neighbors = 0;
		Kokkos::parallel_reduce(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,_trg_particles->getCoordsConst()->nLocal()),
				KOKKOS_LAMBDA (const int i, size_t &myVal) {
			myVal = (all_neighbors[i].size() > myVal) ? all_neighbors[i].size() : myVal;
		}, Kokkos::Experimental::Max<size_t>(max_num_neighbors));

		// generate the interpolation operator and call the coefficients needed (storing them)
		const coords_type* target_coords = _trg_particles->getCoordsConst();
		const coords_type* source_coords = _src_particles->getCoordsConst();

		Kokkos::View<int**> kokkos_neighbor_lists("neighbor lists", target_coords->nLocal(), max_num_neighbors+1);
		Kokkos::View<int**>::HostMirror kokkos_neighbor_lists_host = Kokkos::create_mirror_view(kokkos_neighbor_lists);

		// fill in the neighbor lists into a kokkos view. First entry is # of neighbors for that target
		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
			const int num_i_neighbors = all_neighbors[i].size();
			for (int j=1; j<num_i_neighbors+1; ++j) {
				kokkos_neighbor_lists_host(i,j) = all_neighbors[i][j-1].first;
			}
			kokkos_neighbor_lists_host(i,0) = num_i_neighbors;
		});

		Kokkos::View<double**> kokkos_augmented_source_coordinates("source_coordinates", source_coords->nLocal(true /* include halo in count */), source_coords->nDim());
		Kokkos::View<double**>::HostMirror kokkos_augmented_source_coordinates_host = Kokkos::create_mirror_view(kokkos_augmented_source_coordinates);

		// fill in the source coords, adding regular with halo coordiantes into a kokkos view
		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,source_coords->nLocal(true /* include halo in count*/)), KOKKOS_LAMBDA(const int i) {
			xyz_type coordinate = source_coords->getLocalCoords(i, true /*include halo*/, use_physical_coords);
			kokkos_augmented_source_coordinates_host(i,0) = coordinate.x;
			kokkos_augmented_source_coordinates_host(i,1) = coordinate.y;
			kokkos_augmented_source_coordinates_host(i,2) = coordinate.z;
		});

		Kokkos::View<double**> kokkos_target_coordinates("target_coordinates", target_coords->nLocal(), target_coords->nDim());
		Kokkos::View<double**>::HostMirror kokkos_target_coordinates_host = Kokkos::create_mirror_view(kokkos_target_coordinates);
		// fill in the target, adding regular coordiantes only into a kokkos view
		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
			xyz_type coordinate = target_coords->getLocalCoords(i, false /*include halo*/, use_physical_coords);
			kokkos_target_coordinates_host(i,0) = coordinate.x;
			kokkos_target_coordinates_host(i,1) = coordinate.y;
			kokkos_target_coordinates_host(i,2) = coordinate.z;
		});

		auto epsilons = _neighborhoodInfo->getHSupportSizes()->getLocalView<const host_view_type>();
		Kokkos::View<double*> kokkos_epsilons("target_coordinates", target_coords->nLocal(), target_coords->nDim());
		Kokkos::View<double*>::HostMirror kokkos_epsilons_host = Kokkos::create_mirror_view(kokkos_epsilons);
		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
			kokkos_epsilons_host(i) = epsilons(i,0);
		});

		//****************
		//
		//  End of data copying
		//
		//****************


	// loop over all of the fields and use the stored coefficients
	for (local_index_type i=0; i<_queue.size(); i++) {

		// if first time through or sampling strategy changes
		// conditions for resetting GMLS object
		bool first_in_queue = (i==0);
		bool reconstruction_space_changed, polynomial_sampling_changed, data_sampling_changed, operator_coefficients_field_changed;
		if (!first_in_queue) {
			reconstruction_space_changed = _queue[i]._reconstruction_space != _queue[i-1]._reconstruction_space;
			polynomial_sampling_changed  = _queue[i]._polynomial_sampling_functional != _queue[i-1]._polynomial_sampling_functional;
			data_sampling_changed  = _queue[i]._data_sampling_functional != _queue[i-1]._data_sampling_functional;
			operator_coefficients_field_changed = _queue[i]._operator_coefficients_fieldname != _queue[i-1]._operator_coefficients_fieldname;
		}
		if (first_in_queue || reconstruction_space_changed || polynomial_sampling_changed || data_sampling_changed || operator_coefficients_field_changed) {
			_GMLS = Teuchos::rcp(new GMLS(_queue[i]._reconstruction_space,
					_queue[i]._polynomial_sampling_functional,
					_queue[i]._data_sampling_functional,
					_parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"),
					_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("dense linear solver"),
					_parameters->get<Teuchos::ParameterList>("remap").get<int>("curvature porder")));

			_GMLS->setProblemData(kokkos_neighbor_lists_host,
					kokkos_augmented_source_coordinates_host,
					kokkos_target_coordinates,
					kokkos_epsilons_host);

			_GMLS->setWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("weighting type"));
			_GMLS->setWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("weighting power"));
			_GMLS->setCurvatureWeightingType(_parameters->get<Teuchos::ParameterList>("remap").get<std::string>("curvature weighting type"));
			_GMLS->setCurvatureWeightingPower(_parameters->get<Teuchos::ParameterList>("remap").get<int>("curvature weighting power"));
			_GMLS->setNumberOfQuadraturePoints(_parameters->get<Teuchos::ParameterList>("remap").get<int>("quadrature points"));

			// are they using operator coefficients
			if (_queue[i].getOperatorCoefficients().length() > 0) {

				local_index_type operator_coefficients_fieldnum = _src_particles->getFieldManagerConst()->getIDOfFieldFromName(_queue[i]._operator_coefficients_fieldname);
				{ // check if field exists that was pointed to for operator coefficients
					std::ostringstream msg;
					msg << "Field " << _queue[i]._operator_coefficients_fieldname << " to be used for operator coefficients, but not found in source particles." << std::endl;
					TEUCHOS_TEST_FOR_EXCEPT_MSG(operator_coefficients_fieldnum < 0, msg.str());
				}

				// view into data on processor and halo
				Compadre::host_view_type operator_coefficients_field =  _src_particles->getFieldManagerConst()->getFieldByID(operator_coefficients_fieldnum)->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();
				Compadre::host_view_type operator_coefficients_field_halo =  _src_particles->getFieldManagerConst()->getFieldByID(operator_coefficients_fieldnum)->getHaloMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();

				// make view to be used in GMLS object containing on processor and halo data
				Kokkos::View<double**> kokkos_augmented_operator_coefficients("operator coefficients", source_coords->nLocal(true /* include halo in count */), operator_coefficients_field.dimension_1());
				Kokkos::View<double**>::HostMirror kokkos_augmented_operator_coefficients_host = Kokkos::create_mirror_view(kokkos_augmented_operator_coefficients);

				const local_index_type nlocal = source_coords->nLocal(false /* do not include halo in count*/);
				const local_index_type field_dim = _src_particles->getFieldManagerConst()->getFieldByID(operator_coefficients_fieldnum)->nDim();

				// fill in the operator coefficients, adding regular with halo values into a kokkos view
				Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,source_coords->nLocal(true /* include halo in count*/)), KOKKOS_LAMBDA(const int j) {
					const local_index_type offset_indx = j - nlocal;
					for (local_index_type k=0; k<field_dim; ++k) {
						if (j < nlocal) { // is local
							kokkos_augmented_operator_coefficients_host(j,k) = operator_coefficients_field(j,k);
						} else {
							kokkos_augmented_operator_coefficients_host(j,k) = operator_coefficients_field_halo(offset_indx,k);
						}
					}
				});
				//_GMLS->setOperatorCoefficients(kokkos_augmented_operator_coefficients_host);
			}

		    std::vector<TargetOperation> lro;
		    for (local_index_type j=i; j<_queue.size(); j++) {
		    	// only add targets with same sampling strategy and reconstruction space
		    	if ((_queue[j]._reconstruction_space == _queue[i]._reconstruction_space) && (_queue[j]._polynomial_sampling_functional == _queue[i]._polynomial_sampling_functional) && (_queue[j]._data_sampling_functional == _queue[i]._data_sampling_functional))
		    		lro.push_back(_queue[j]._target_operation);
		    }

	    	_GMLS->addTargets(lro);
			_GMLS->generateAlphas(); // all operations requested

			// check that src_coords == trg_coords
			TEUCHOS_TEST_FOR_EXCEPT_MSG((_queue[i]._data_sampling_functional!=SamplingFunctional::PointSample && _queue[i]._data_sampling_functional!=SamplingFunctional::VectorPointSample) && _src_particles->getCoordsConst()->nLocal()!=_trg_particles->getCoordsConst()->nLocal(), "Sampling method requires target coordinates be the same as source coordinates.");
		}


		// if a string field was given, and this string doesn't exist, then register that field first

		// get target and source field numbers
		local_index_type source_field_num;
		if (_queue[i].src_fieldnum != -1) {
			source_field_num = _queue[i].src_fieldnum;
		} else {
			source_field_num = _src_particles->getFieldManagerConst()->getIDOfFieldFromName(_queue[i].src_fieldname);
		}

		if (source_field_num >= 0) { // a field to remap
			// get field dimension for source from _src_particles using info in the object in the queue
			const local_index_type source_field_dim = _src_particles->getFieldManagerConst()->
					getFieldByID(source_field_num)->nDim();

            printf("field we are working on is: %s\n", _queue[i].trg_fieldname.c_str());

			local_index_type target_field_num;
			if (_queue[i].trg_fieldnum >= 0) target_field_num = _queue[i].trg_fieldnum;
			else {
				try {
					target_field_num = _trg_particles->getFieldManagerConst()->getIDOfFieldFromName(_queue[i].trg_fieldname);
				} catch (std::logic_error & exception) {
					// check for point evaluation on n_dimensions
					// otherwise make compatible dimensions under operation

					int output_dim;

					// special case of point remap
					if (_queue[i]._target_operation==TargetOperation::ScalarPointEvaluation || _queue[i]._target_operation==TargetOperation::VectorPointEvaluation) {
						output_dim = source_field_dim;
					} else {
						output_dim = _GMLS->getOutputDimensionOfOperation(_queue[i]._target_operation, true /*always ambient representation*/);
					}
                    printf("output dim: %d\n", output_dim);

					// field wasn't registered in target, so we are collecting its information from source particles
					_trg_particles->getFieldManager()->createField(
							output_dim,
							_queue[i].trg_fieldname,
							_src_particles->getFieldManagerConst()->getFieldByID(source_field_num)->getUnits()
							);
					if (_trg_particles->getCoordsConst()->getComm()->getRank()==0) {
						std::cout << "The field \"" << _queue[i].trg_fieldname << "\" was requested to be calculated from the source field \""
								<< _src_particles->getFieldManagerConst()->getFieldByID(source_field_num)->getName() << "\". However, \""
								<< _queue[i].trg_fieldname << "\" was never registered in the target particles. This has been resolved by"
								<< " registering the field \"" << _queue[i].trg_fieldname << "\" in the target particle set with the "
								<< "units and dimensions of the operator on the field \""
								<< _src_particles->getFieldManagerConst()->getFieldByID(source_field_num)->getName() << "\" from the source particles.";
					}
					target_field_num = _trg_particles->getFieldManagerConst()->getIDOfFieldFromName(_queue[i].trg_fieldname);
				}
			}

			// get field dimension for target from _trg_particles using info in the object in the queue
			const local_index_type target_field_dim = _trg_particles->getFieldManagerConst()->
					getFieldByID(target_field_num)->nDim();

			//// check for the case of point remap from n dim to ndim
			//bool component_by_component_pointwise = false;
			//if (_queue[i]._target_operation==TargetOperation::ScalarPointEvaluation) {
			//	TEUCHOS_TEST_FOR_EXCEPT_MSG(target_field_dim != source_field_dim, "Dimensions between target field and source field do not match under pointwise remap component by component.");
			//	component_by_component_pointwise = true;
			//} else {
				// check reasonableness of dimensions in and out of the operation for remap against the source and target fields
				std::ostringstream msg;
				printf("field num: %d\n", _queue[i].trg_fieldnum);
				msg << "Dimensions between target field (" << target_field_dim << ") and operation on source field (" << _GMLS->getOutputDimensionOfOperation(_queue[i]._target_operation, true /*always ambient representation*/) << ") does not match for the field." << std::endl;
				TEUCHOS_TEST_FOR_EXCEPT_MSG(target_field_dim != _GMLS->getOutputDimensionOfOperation(_queue[i]._target_operation, true), msg.str());
				if (_GMLS->getInputDimensionOfSampling(_queue[i]._data_sampling_functional)==_GMLS->getOutputDimensionOfSampling(_queue[i]._data_sampling_functional)) {
					printf("%d, %d, %d, %d\n\n\n", _queue[i]._target_operation, _queue[i]._data_sampling_functional, _GMLS->getInputDimensionOfSampling(_queue[i]._data_sampling_functional), _GMLS->getOutputDimensionOfSampling(_queue[i]._data_sampling_functional));
					TEUCHOS_TEST_FOR_EXCEPT_MSG(source_field_dim != _GMLS->getInputDimensionOfOperation(_queue[i]._target_operation), "Dimensions between source field and operation input dimension does not match.");
				} else {
					TEUCHOS_TEST_FOR_EXCEPT_MSG(source_field_dim != _GMLS->getInputDimensionOfSampling(_queue[i]._data_sampling_functional), "Dimensions between source field and sampling strategy input dimension does not match.");
				}
			//}

			host_view_type target_values = _trg_particles->getFieldManager()->
					getFieldByID(target_field_num)->getMultiVectorPtr()->getLocalView<host_view_type>();
			const local_index_type num_target_values = target_values.dimension_0();

			const host_view_type source_values_holder = _src_particles->getFieldManagerConst()->
					getFieldByID(source_field_num)->getMultiVectorPtrConst()->getLocalView<const host_view_type>();
			const host_view_type source_halo_values_holder = _src_particles->getFieldManagerConst()->
					getFieldByID(source_field_num)->getHaloMultiVectorPtrConst()->getLocalView<const host_view_type>();

			const local_index_type num_local_particles = source_values_holder.dimension_0();



		    // fill in the source values, adding regular with halo values into a kokkos view
		    Kokkos::View<double**, Kokkos::HostSpace> kokkos_source_values("source values", source_coords->nLocal(true /* include halo in count */), source_field_dim);

            // copy data local owned by this process
		    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,source_coords->nLocal(false /* include halo in count*/)), KOKKOS_LAMBDA(const int j) {
                for (local_index_type k=0; k<source_field_dim; ++k) {
		    	    kokkos_source_values(j,k) = source_values_holder(j,k);
                }
		    });
            // copy halo data 
		    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(source_coords->nLocal(false /* not includinghalo in count*/), source_coords->nLocal(true /*include halo in count */)), KOKKOS_LAMBDA(const int j) {
                for (local_index_type k=0; k<source_field_dim; ++k) {
		    	    kokkos_source_values(j,k) = source_halo_values_holder(j-num_local_particles,k);
                }
            });




            // create a remap class object to handle applying data transforms and GMLS operator
            // as well as the transformation back from local charts to ambient space (if on a manifold)
            Remap remap_agent(_GMLS.getRawPtr());

            auto output_vector = remap_agent.applyAlphasToDataAllComponentsAllTargetSites<double**, Kokkos::HostSpace>
                (kokkos_source_values, _queue[i]._target_operation, _queue[i]._data_sampling_functional);
            
            // copy remapped solution back into field
			Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, num_target_values), KOKKOS_LAMBDA(const int j) {
				for (local_index_type k=0; k<target_field_dim; k++) {
                    target_values(j,k) = output_vector(j,k);
                }
            });


			// optional OBFET
			if (_queue[i]._obfet) {
				Compadre::OBFET obfet_object(_src_particles, _trg_particles, source_field_num, target_field_num, _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("source weighting field name"), _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("target weighting field name"), _neighborhoodInfo.getRawPtr());
				obfet_object.solveAndUpdate();
			}

		} else { // coordinate remap

			// get field dimension for source from _src_particles using info in the object in the queue
			const local_index_type source_field_dim = _src_particles->getCoordsConst()->nDim();

			// get field dimension for target from _trg_particles using info in the object in the queue
			const local_index_type target_field_dim = source_field_dim;

			host_view_type target_values;
			// these are the values that are to be written over
			if (source_field_num == -20) { // traditional case of getting Lagrange coordinates using physical coordinates
				target_values = _trg_particles->getCoordsConst()->getPts(false/*halo*/,false /*use_physical_coordinates*/)->getLocalView<host_view_type>();
			} else if (source_field_num == -21) {
				target_values = _trg_particles->getCoordsConst()->getPts(false/*halo*/,true /*use_physical_coordinates*/)->getLocalView<host_view_type>();
			}

			const local_index_type num_target_values = target_values.dimension_0();

			host_view_type source_values_holder;
			host_view_type source_halo_values_holder;
			// these are the values that will be combined through coefficients to overwrite target_values
			if (source_field_num == -20) { // traditional case of getting Lagrange coordinates using physical coordinates
				source_values_holder = _src_particles->getCoordsConst()->getPts(false/*halo*/, false /*use_physical_coordinates*/)->getLocalView<host_view_type>();
				source_halo_values_holder = _src_particles->getCoordsConst()->getPts(true/*halo*/, false /*use_physical_coordinates*/)->getLocalView<host_view_type>();
			} else if (source_field_num == -21) {
				source_values_holder = _src_particles->getCoordsConst()->getPts(false/*halo*/, true /*use_physical_coordinates*/)->getLocalView<host_view_type>();
				source_halo_values_holder = _src_particles->getCoordsConst()->getPts(true/*halo*/, true /*use_physical_coordinates*/)->getLocalView<host_view_type>();
			}

			const local_index_type num_local_particles = source_values_holder.dimension_0();





		    // fill in the source values, adding regular with halo values into a kokkos view
		    Kokkos::View<double**, Kokkos::HostSpace> kokkos_source_values("source values", source_coords->nLocal(true /* include halo in count */), source_field_dim);

            // copy data local owned by this process
		    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,source_coords->nLocal(false /* include halo in count*/)), KOKKOS_LAMBDA(const int j) {
                for (local_index_type k=0; k<source_field_dim; ++k) {
		    	    kokkos_source_values(j,k) = source_values_holder(j,k);
                }
		    });
            // copy halo data 
		    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(source_coords->nLocal(false /* not includinghalo in count*/), source_coords->nLocal(true /*include halo in count */)), KOKKOS_LAMBDA(const int j) {
                for (local_index_type k=0; k<source_field_dim; ++k) {
		    	    kokkos_source_values(j,k) = source_halo_values_holder(j-num_local_particles,k);
                }
            });




            // create a remap class object to handle applying data transforms and GMLS operator
            // as well as the transformation back from local charts to ambient space (if on a manifold)
            Remap remap_agent(_GMLS.getRawPtr());

            auto output_vector = remap_agent.applyAlphasToDataAllComponentsAllTargetSites<double**, Kokkos::HostSpace>
                (kokkos_source_values, _queue[i]._target_operation, _queue[i]._data_sampling_functional);
            
            // copy remapped coordinates back into target particles coordinates
			Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, num_target_values), KOKKOS_LAMBDA(const int j) {
				for (local_index_type k=0; k<target_field_dim; k++) {
                    target_values(j,k) = output_vector(j,k);
                }
            });

		}
	}
	} // _queue.size>0

	// clear the queue
	this->clear();
	if (!keep_neighborhoods) {
		_neighborhoodInfo = Teuchos::null;
		_GMLS = Teuchos::null;
	}
}

void RemapManager::add(RemapObject obj) {
	// This function sorts through insert by checking the sampling strategy type
	// and inserting in the correct position
    

    // The only special case is when a user provides no sampling type, so they get a point sample
    // but they want to remap a vector field
    
    // get output dimension of target
	// get target and source field numbers
	local_index_type source_field_num;
	if (obj.src_fieldnum != -1) { // use the field id provided
		source_field_num = obj.src_fieldnum;
	} else {
		source_field_num = _src_particles->getFieldManagerConst()->getIDOfFieldFromName(obj.src_fieldname);
	}

    // user asked to remap a vector, but provided point sample as the target operation
    // or they asked to remap coordinates and provided point sample as the target operation
    if ((source_field_num >= 0 && obj._target_operation == TargetOperation::ScalarPointEvaluation && _src_particles->getFieldManagerConst()->getFieldByID(source_field_num)->nDim() > 1)
            || (source_field_num < 0 && obj._target_operation == TargetOperation::ScalarPointEvaluation && _src_particles->getCoordsConst()->nDim() > 1)) {
        printf("changed %s from scalar to vector.\n", obj.src_fieldname.c_str());
        printf("changed %d from scalar to vector.\n", obj.src_fieldnum);
        obj._target_operation = TargetOperation::VectorPointEvaluation;
        obj._reconstruction_space = ReconstructionSpace::VectorOfScalarClonesTaylorPolynomial;
        //obj._reconstruction_space = ReconstructionSpace::VectorTaylorPolynomial;
        obj._polynomial_sampling_functional = SamplingFunctional::VectorPointSample;
        obj._data_sampling_functional = obj._polynomial_sampling_functional;
    } 
//else { // coordinates
//        if (obj._target_operation == TargetOperation::ScalarPointEvaluation && _src_particles->getCoordsConst()->nDim() > 1) {//->nDim() > 1) {
//            obj._target_operation = TargetOperation::VectorPointEvaluation;
//            obj._reconstruction_space = ReconstructionSpace::VectorOfScalarClonesTaylorPolynomial;
//            obj._polynomial_sampling_functional = SamplingFunctional::VectorPointSample;
//            obj._data_sampling_functional = obj._polynomial_sampling_functional;
//        }
//    }


	// if empty, then add the remap object
	if (_queue.size()==0) {
		_queue.push_back(obj);
		return;
	}
	// loop through remap objects until we find one with the same samplings and reconstruction space, then add to the end of it
	bool obj_inserted = false;
	local_index_type index_with_last_match = -1;

	for (local_index_type i=0; i<_queue.size(); ++i) {
		if ((_queue[i]._reconstruction_space == obj._reconstruction_space) && (_queue[i]._polynomial_sampling_functional == obj._polynomial_sampling_functional) && (_queue[i]._data_sampling_functional == obj._data_sampling_functional)) {
			index_with_last_match = i;
		} else if (index_with_last_match > -1) { // first time they differ in strategies after finding the sampling / space match
			// insert at this index
			_queue.insert(_queue.begin()+i, obj); // end of previously found location
			obj_inserted = true;
			break;
		}
	}
	if (!obj_inserted) _queue.push_back(obj); // also takes care of case where you match up to the last item in the queue, but never found one that differed
}

}
