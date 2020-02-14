#include "Compadre_OptimizationManager.hpp"

#include "Compadre_FieldManager.hpp"
#include "Compadre_FieldT.hpp"
#include "Compadre_CoordsT.hpp"
#include "Compadre_NeighborhoodT.hpp"
#include "Compadre_ParticlesT.hpp"
#include "Compadre_XyzVector.hpp"

#include "TPL/Cobra_SLBQP.hpp" // OBFET

#ifdef COMPADREHARNESS_USE_COMPOSE
  #include "cedr_mpi.hpp"
  #include "cedr_util.hpp"
  #include "cedr_caas.hpp"
#endif

namespace Compadre {

//void display_vector(const std::vector<double> &v, bool column=false) {
//    if (column) std::copy(v.begin(), v.end(),
//                        std::ostream_iterator<double>(std::cout, "\n"));
//    else std::copy(v.begin(), v.end(),
//                   std::ostream_iterator<double>(std::cout, " "));
//}

void OptimizationManager::solveAndUpdate() {
if (_optimization_object._optimization_algorithm != OptimizationAlgorithm::NONE) {

    // current algorithms are written for a single linear bound and with bounds preservation
    TEUCHOS_TEST_FOR_EXCEPT_MSG(_optimization_object._single_linear_bound_constraint==false, "Only algorithms with single linear bound constraint == true are supported.");
    TEUCHOS_TEST_FOR_EXCEPT_MSG(_optimization_object._bounds_preservation==false, "Only algorithms with bounds preservation are supported.");

    // verify weighting field name specified
    TEUCHOS_TEST_FOR_EXCEPT_MSG(_source_weighting_name.length()==0, "Source weighting field name not specified.");
    TEUCHOS_TEST_FOR_EXCEPT_MSG(_target_weighting_name.length()==0, "Target weighting field name not specified.");

    // weights
    Compadre::host_view_type source_grid_weighting_field = _source_particles->getFieldManagerConst()->getFieldByName(_source_weighting_name)->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();
    Compadre::host_view_type target_grid_weighting_field = _target_particles->getFieldManager()->getFieldByName(_target_weighting_name)->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();
    // target values
    Compadre::host_view_type target_solution_data = _target_particles->getFieldManager()->getFieldByID(_target_field_num)->getMultiVectorPtr()->getLocalView<Compadre::host_view_type>();
    // for maxs/mins
    Compadre::host_view_type source_solution_data = _source_particles->getFieldManagerConst()->getFieldByID(_source_field_num)->getMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();
    Compadre::host_view_type source_halo_solution_data = _source_particles->getFieldManagerConst()->getFieldByID(_source_field_num)->getHaloMultiVectorPtrConst()->getLocalView<Compadre::host_view_type>();


    std::vector<scalar_type> weights(target_grid_weighting_field.extent(0));
    for (local_index_type j=0; j<_target_particles->getCoordsConst()->nLocal(); ++j) {
        weights[j] = target_grid_weighting_field(j,0);
    }

    const local_index_type source_nlocal = _source_particles->getCoordsConst()->nLocal();
    const local_index_type target_nlocal = _target_particles->getCoordsConst()->nLocal();

    // loop over field's dimensions
    for (size_t i=0; i<target_solution_data.extent(1); ++i) {

        scalar_type local_conserved_quantity = 0, global_conserved_quantity = 0;
        for (local_index_type j=0; j<_source_particles->getCoordsConst()->nLocal(); ++j) {
            local_conserved_quantity += source_grid_weighting_field(j,0)*source_solution_data(j,i);
        }
        Teuchos::Ptr<scalar_type> global_conserved_quantity_ptr(&global_conserved_quantity);
        Teuchos::reduceAll<local_index_type, scalar_type>(*(_source_particles->getCoordsConst()->getComm()), Teuchos::REDUCE_SUM, local_conserved_quantity, global_conserved_quantity_ptr);

        std::vector<scalar_type> target_values(target_solution_data.extent(0));
        std::vector<scalar_type> updated_target_values(target_solution_data.extent(0),0);
        std::vector<scalar_type> source_mins(target_solution_data.extent(0), 1e+15);//std::numeric_limits<scalar_type>::max());
        std::vector<scalar_type> source_maxs(target_solution_data.extent(0), -1e+15);//std::numeric_limits<scalar_type>::lowest());

        bool use_global_lower_bound = false;
        bool use_global_upper_bound = false;

        if (std::numeric_limits<scalar_type>::lowest() != _optimization_object._global_lower_bound) {
            use_global_lower_bound = true;
#ifdef COMPADREHARNESS_DEBUG
            std::cout << "Global minimum enforced of " << _optimization_object._global_lower_bound << "." << std::endl;
#endif
        }
        if (std::numeric_limits<scalar_type>::max() != _optimization_object._global_upper_bound) {
            use_global_upper_bound = true;
#ifdef COMPADREHARNESS_DEBUG
            std::cout << "Global maximum enforced of " << _optimization_object._global_upper_bound << "." << std::endl;
#endif
        }

        for (local_index_type j=0; j<target_nlocal; ++j) {

            target_values[j] = target_solution_data(j,i);

            // loop over neighbors to get max and min for each component of the field
            const local_index_type num_neighbors = _neighborhood_info->getNumNeighbors(j);

            for (local_index_type k=0; k<num_neighbors; ++k) {
                scalar_type neighbor_value = (_neighborhood_info->getNeighbor(j,k) < (size_t)source_nlocal) ? source_solution_data(_neighborhood_info->getNeighbor(j,k), i) : source_halo_solution_data(_neighborhood_info->getNeighbor(j,k)-source_nlocal, i);

                source_mins[j] = (neighbor_value < source_mins[j]) ? neighbor_value : source_mins[j];
                source_maxs[j] = (neighbor_value > source_maxs[j]) ? neighbor_value : source_maxs[j];
            }

            if (use_global_lower_bound) {
                source_mins[j] = std::max(source_mins[j], _optimization_object._global_lower_bound);
            }
            if (use_global_upper_bound) {
                source_maxs[j] = std::min(source_maxs[j], _optimization_object._global_upper_bound);
            }
        }


        if (_optimization_object._optimization_algorithm==OptimizationAlgorithm::OBFET) {

            // written for serial
            TEUCHOS_TEST_FOR_EXCEPT_MSG(_source_particles->getCoordsConst()->getComm()->getSize()>1, "OBFET only written for serial.");

            // fill std::vec with these and then call Cobra_
            // Set up optimization inputs/outputs.
            Cobra::AlgoStatus<scalar_type> astatus;
            Cobra::StdVectorInterface svi;

            // Run optimization.
            Cobra::slbqp<scalar_type, std::vector<scalar_type> > (astatus, updated_target_values, target_values, source_mins, source_maxs, weights, global_conserved_quantity, &svi);

            // Display algorithm status.
            std::cout.precision(15);
            std::cout << std::scientific;
            std::cout << "\nSLBQP Algorithm Termination Report:\n";
            std::cout << "Converged? ... " << astatus.converged << "\n";
            std::cout << "Iterations ... " << astatus.iterations << "\n";
            std::cout << "Bracketing ... " << astatus.bracketing_iterations << "\n";
            std::cout << "Secant     ... " << astatus.secant_iterations << "\n";
            std::cout << "Lambda     ... " << astatus.lambda << "\n";
            std::cout << "Residual   ... " << astatus.residual << "\n";

            TEUCHOS_TEST_FOR_EXCEPT_MSG(astatus.converged==0, "OBFET failed to converge.");

        } else if (_optimization_object._optimization_algorithm==OptimizationAlgorithm::CAAS) {
#ifdef COMPADREHARNESS_USE_COMPOSE
            MPI_Comm comm = *(Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(this->_target_particles->getCoordsConst()->getComm(), true /*throw on fail*/)->getRawMpiComm());
            auto parallel = cedr::mpi::make_parallel(comm);
            srand(0);
            
            typedef cedr::caas::CAAS<Kokkos::DefaultHostExecutionSpace> CAAST;
            CAAST caas(parallel, this->_target_particles->getCoordsConst()->nLocal());
            caas.declare_tracer(cedr::ProblemType::conserve |
                                      cedr::ProblemType::shapepreserve, 0);
            caas.end_tracer_declarations();
            auto target_num_local = this->_target_particles->getCoordsConst()->nLocal();
            auto target_num_global = this->_target_particles->getCoordsConst()->nGlobal();
            for (int j=0; j<target_num_local; ++j) {
                caas.set_rhom(j, 0, weights[j]);
                caas.set_Qm(j, 0, target_values[j]*weights[j], source_mins[j]*weights[j], source_maxs[j]*weights[j], global_conserved_quantity / target_num_global);
            }
            caas.run();
            for (int j=0; j<target_num_local; ++j) {
                updated_target_values[j] = caas.get_Qm(j, 0)/weights[j];
            }

#ifdef COMPADREHARNESS_DEBUG
            // diagnostic
            scalar_type local_quantity = 0, global_quantity = 0;
            for (local_index_type j=0; j<_target_particles->getCoordsConst()->nLocal(); ++j) {
                local_quantity += weights[j]*updated_target_values[j];
            }
            Teuchos::Ptr<scalar_type> global_quantity_ptr(&global_quantity);
            Teuchos::reduceAll<local_index_type, scalar_type>(*(_target_particles->getCoordsConst()->getComm()), Teuchos::REDUCE_SUM, local_quantity, global_quantity_ptr);

            scalar_type residual = global_quantity - global_conserved_quantity;
            local_index_type out_of_bounds = 0;
            for (int j=0; j<target_num_local; ++j) {
                if (updated_target_values[j] > source_maxs[j]+std::abs(residual) || updated_target_values[j] < source_mins[j]-std::abs(residual)) {
                    out_of_bounds++;
                }
            }

            // Display algorithm status.
            std::cout.precision(15);
            std::cout << std::scientific;
            if (_target_particles->getCoordsConst()->getComm()->getRank() == 0) {
                std::cout << "\nCAAS Algorithm Termination Report:\n";
                std::cout << "Residual        ... " << residual << "\n";
            }
            TEUCHOS_TEST_FOR_EXCEPT_MSG(out_of_bounds!=0, "At least one constructed value out of bounds after bounds preservation enforced.");
#endif // DEBUG
#else
            TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Compose package called but not built. Enable with '-D CompadreHarness_USE_Compose:BOOL=ON'.");
#endif // COMPADREHARNESS_USE_COMPOSE
        }
        // replace data on target with this
        for (local_index_type j=0; j<_target_particles->getCoordsConst()->nLocal(); ++j) {
            target_solution_data(j,i) = updated_target_values[j];
        }
    }

}
}

}
