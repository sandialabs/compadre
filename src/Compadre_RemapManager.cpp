#include "Compadre_RemapManager.hpp"

#include "Compadre_XyzVector.hpp"
#include "Compadre_CoordsT.hpp"
#include "Compadre_ParticlesT.hpp"
#include "Compadre_FieldManager.hpp"
#include "Compadre_FieldT.hpp"

#include "Compadre_OptimizationManager.hpp"

#include "Compadre_nanoflannInformation.hpp"
#include "Compadre_nanoflannPointCloudT.hpp"

#include <Compadre_GMLS.hpp>
#include <Compadre_Evaluator.hpp>



namespace Compadre {

typedef Compadre::NanoFlannInformation nanoflann_neighbors_type;
typedef Compadre::NeighborhoodT neighbors_type;

typedef Compadre::CoordsT coords_type;
typedef Compadre::XyzVector xyz_type;

void RemapManager::execute(bool keep_neighborhoods, bool use_physical_coords) {

    // currently this neighborhood search is always in physical coords for Lagrangian simulation, but it may need to be in Lagrangian
    
    if (_queue.size()>0) {
    
        std::string method =_parameters->get<Teuchos::ParameterList>("neighborhood").get<std::string>("method");
        transform(method.begin(), method.end(), method.begin(), ::tolower);
    
    
        if (method=="nanoflann") {
            local_index_type maxLeaf = _parameters->get<Teuchos::ParameterList>("neighborhood").get<int>("max leaf");
            _neighborhoodInfo = Teuchos::rcp_static_cast<neighbors_type>(Teuchos::rcp(
                    new nanoflann_neighbors_type(_src_particles, _parameters, maxLeaf, _trg_particles, use_physical_coords)));
        }
        else {
            TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Selected a search method that is not nanoflann.");
        }
    
    
        local_index_type neighbors_needed;
    
        std::string solver_type_to_lower = _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("dense solver type");
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
        bool gmls_problem_compatible_with_previous_object = false;
        for (size_t i=0; i<_queue.size(); i++) {
    
            // if first time through or sampling strategy changes
            // conditions for resetting GMLS object
            bool first_in_queue = (i==0);
            if (!first_in_queue) {
                gmls_problem_compatible_with_previous_object = this->isCompatible(_queue[i], _queue[i-1]);
            }

            if (!gmls_problem_compatible_with_previous_object) {

                _GMLS = Teuchos::rcp(new GMLS(_queue[i]._reconstruction_space,
                        _queue[i]._polynomial_sampling_functional,
                        _queue[i]._data_sampling_functional,
                        _parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"),
                        _parameters->get<Teuchos::ParameterList>("remap").get<int>("dimensions"),
                        _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("dense solver type"),
                        _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("problem type"),
                        _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("boundary type"),
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

                if (_queue[i]._reference_normal_directions_fieldname != "") {
                    auto reference_normal_directions = 
                        _trg_particles->getFieldManager()->getFieldByName(_queue[i]._reference_normal_directions_fieldname)->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();
                    _GMLS->setReferenceOutwardNormalDirection(reference_normal_directions, true /*use_to_orient_surface*/);
                }

                if (_queue[i]._extra_data_fieldname != "") {

                    auto extra_data_local = 
                        _src_particles->getFieldManagerConst()->getFieldByName(_queue[i]._extra_data_fieldname)->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();
                    auto extra_data_halo = 
                        _src_particles->getFieldManagerConst()->getFieldByName(_queue[i]._extra_data_fieldname)->getHaloMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();
                    auto combined_extra_data = host_view_type("combined extra data", extra_data_local.extent(0) + extra_data_halo.extent(0), extra_data_local.extent(1));
    
                    // fill in combined data
                    auto nlocal = source_coords->nLocal(false); // locally owned #
                    auto ncols = combined_extra_data.extent(1);
                    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,source_coords->nLocal(true /* include halo in count*/)), KOKKOS_LAMBDA(const int j) {
                        for (size_t k=0; k<ncols; ++k) {
                            if (j < nlocal) {
                                combined_extra_data(j,k) = extra_data_local(j,k);
                            } else {
                                combined_extra_data(j,k) = extra_data_halo(j-nlocal,k);
                            }
                        }
                    });

                    _GMLS->setExtraData(combined_extra_data);
                }
    
    
                std::vector<TargetOperation> lro;
                for (size_t j=i; j<_queue.size(); j++) {
                    // only add targets with all other remap object properties matching
                    gmls_problem_compatible_with_previous_object = this->isCompatible(_queue[i], _queue[j]);
                    if (gmls_problem_compatible_with_previous_object) {
                        lro.push_back(_queue[j]._target_operation);
                    }
                }
    
                _GMLS->addTargets(lro);
                _GMLS->generateAlphas(); // all operations requested
    
            }
    
    
    
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
    
                local_index_type target_field_num;
                if (_queue[i].trg_fieldnum >= 0) target_field_num = _queue[i].trg_fieldnum;
                else {
                    // if a string field was given, and this string doesn't exist, then register that field first
                    try {
                        target_field_num = _trg_particles->getFieldManagerConst()->getIDOfFieldFromName(_queue[i].trg_fieldname);
                    } catch (std::logic_error & exception) {
                        // check for point evaluation on n_dimensions
                        // otherwise make compatible dimensions under operation
    
                        int output_dim;
    
                        output_dim = _GMLS->getOutputDimensionOfOperation(_queue[i]._target_operation, true /*always ambient representation*/);
    
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
    
                // check reasonableness of dimensions in and out of the operation for remap against the source and target fields
                std::ostringstream msg;
                msg << "Dimensions between target field (" << target_field_dim << ") and operation on source field (" << _GMLS->getOutputDimensionOfOperation(_queue[i]._target_operation, true /*always ambient representation*/) << ") does not match for the field." << std::endl;
                TEUCHOS_TEST_FOR_EXCEPT_MSG(target_field_dim != _GMLS->getOutputDimensionOfOperation(_queue[i]._target_operation, true), msg.str());
                if (_GMLS->getInputDimensionOfSampling(_queue[i]._data_sampling_functional)==_GMLS->getOutputDimensionOfSampling(_queue[i]._data_sampling_functional)) {
                    TEUCHOS_TEST_FOR_EXCEPT_MSG(source_field_dim != _GMLS->getInputDimensionOfOperation(_queue[i]._target_operation), "Dimensions between source field and operation input dimension does not match.");
                } else {
                    TEUCHOS_TEST_FOR_EXCEPT_MSG(source_field_dim != _GMLS->getInputDimensionOfSampling(_queue[i]._data_sampling_functional), "Dimensions between source field and sampling strategy input dimension does not match.");
                }
                
    
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
                Kokkos::fence();
    
    
    
                // create an Evaluator class object to handle applying data transforms and GMLS operator
                // as well as the transformation back from local charts to ambient space (if on a manifold)
                Evaluator remap_agent(_GMLS.getRawPtr());
    
                auto output_vector = remap_agent.applyAlphasToDataAllComponentsAllTargetSites<double**, Kokkos::HostSpace>
                    //(source_values_holder, _queue[i]._target_operation, _queue[i]._data_sampling_functional);
                    (kokkos_source_values, _queue[i]._target_operation, _queue[i]._data_sampling_functional);
                
                // copy remapped solution back into field
                Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, num_target_values), KOKKOS_LAMBDA(const int j) {
                    for (local_index_type k=0; k<target_field_dim; k++) {
                        target_values(j,k) = output_vector(j,k);
                    }
                });
    
    
                // optional OBFET
                if (_queue[i]._optimization_object._optimization_algorithm!=OptimizationAlgorithm::NONE) {
                    Compadre::OptimizationManager optimization_manager(_src_particles, _trg_particles, source_field_num, target_field_num, _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("source weighting field name"), _parameters->get<Teuchos::ParameterList>("remap").get<std::string>("target weighting field name"), _neighborhoodInfo.getRawPtr(), _queue[i]._optimization_object);
                    optimization_manager.solveAndUpdate();
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
    
    
    
    
                // create an Evaluator class object to handle applying data transforms and GMLS operator
                // as well as the transformation back from local charts to ambient space (if on a manifold)
                Evaluator remap_agent(_GMLS.getRawPtr());
    
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
    // and inserting in the correct position.
    

    // The only special case is when a user provides no sampling type, so they get a point sample
    // but they want to remap a vector field.
    
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
        obj._target_operation = TargetOperation::VectorPointEvaluation;
        //obj._reconstruction_space = ReconstructionSpace::VectorTaylorPolynomial;
        obj._reconstruction_space = ReconstructionSpace::VectorOfScalarClonesTaylorPolynomial;
        obj._polynomial_sampling_functional = VectorPointSample;
        obj._data_sampling_functional = obj._polynomial_sampling_functional;
    } 


    // if empty, then add the remap object
    if (_queue.size()==0) {
        _queue.push_back(obj);
        return;
    }

    // Loop through remap objects until we find one with the same samplings and reconstruction space, then add to the end of it.
    // We do not sort for additional properties, such as normal directions field which means that more reconstructions may be 
    // being called than are expected.
    //
    // As an example, consider three remap objects with the same sampling functional and space,
    // and three target operations. If the second one names a normal directions field, then looping over the list of
    // remap objects in the queue will indicate that a new problem setup is needed between the first and second 
    // remap object, as well as between the second and third. The results is three separate problems solved, whereas
    // if a normal directions field had not been given, all three target operations would have been calculated on
    // the same problem. This is an efficiency issue, but not an issue of correctness.
 
    bool obj_inserted = false;
    local_index_type index_with_last_match = -1;

    for (size_t i=0; i<_queue.size(); ++i) {
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

bool RemapManager::isCompatible(const RemapObject obj_1, const RemapObject obj_2) const {

    bool are_same = true;

    // check various properties for ability to reuse a GMLS problem setup
    are_same &= obj_2._reconstruction_space == obj_1._reconstruction_space;
    are_same &= obj_2._polynomial_sampling_functional == obj_1._polynomial_sampling_functional;
    are_same &= obj_2._data_sampling_functional == obj_1._data_sampling_functional;
    are_same &= obj_2._operator_coefficients_fieldname == obj_1._operator_coefficients_fieldname;
    are_same &= obj_2._reference_normal_directions_fieldname == obj_1._reference_normal_directions_fieldname;
    are_same &= obj_2._extra_data_fieldname == obj_1._extra_data_fieldname;

    return are_same;

}

} // Compadre
