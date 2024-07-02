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

    // all algorithms != NONE use single linear bound (conservation)
    TEUCHOS_TEST_FOR_EXCEPT_MSG(_optimization_object._single_linear_bound_constraint==false, "Only algorithms with single linear bound constraint == true are supported.");

    // verify weighting field name specified
    TEUCHOS_TEST_FOR_EXCEPT_MSG(_source_weighting_name.length()==0, "Source weighting field name not specified.");
    TEUCHOS_TEST_FOR_EXCEPT_MSG(_target_weighting_name.length()==0, "Target weighting field name not specified.");

	OptimizationTime->start();

    // weights
    auto source_grid_weighting_field = _source_particles->getFieldManagerConst()->getFieldByName(_source_weighting_name)->getMultiVectorPtrConst()->getLocalViewHost(Tpetra::Access::ReadOnly);
    auto target_grid_weighting_field = _target_particles->getFieldManager()->getFieldByName(_target_weighting_name)->getMultiVectorPtrConst()->getLocalViewHost(Tpetra::Access::ReadOnly);
    // target values
    auto target_solution_data = _target_particles->getFieldManager()->getFieldByID(_target_field_num)->getMultiVectorPtr()->getLocalViewHost(Tpetra::Access::OverwriteAll);
    // for maxs/mins
    auto source_solution_data = _source_particles->getFieldManagerConst()->getFieldByID(_source_field_num)->getMultiVectorPtrConst()->getLocalViewHost(Tpetra::Access::ReadOnly);
    auto source_halo_solution_data = _source_particles->getFieldManagerConst()->getFieldByID(_source_field_num)->getHaloMultiVectorPtrConst()->getLocalViewHost(Tpetra::Access::ReadOnly);


    std::vector<scalar_type> weights(target_grid_weighting_field.extent(0));
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,_target_particles->getCoordsConst()->nLocal()), [&](const int j) {
        weights[j] = target_grid_weighting_field(j,0);
    });
    Kokkos::fence();

    const local_index_type source_nlocal = _source_particles->getCoordsConst()->nLocal();
    const local_index_type target_nlocal = _target_particles->getCoordsConst()->nLocal();

    // loop over field's dimensions
    for (size_t i=0; i<target_solution_data.extent(1); ++i) {

        scalar_type local_conserved_quantity = 0, global_conserved_quantity = 0;
        Kokkos::parallel_reduce(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,_source_particles->getCoordsConst()->nLocal()), [&](const int j, double& t_local_conserved_quantity) {
            t_local_conserved_quantity += source_grid_weighting_field(j,0)*source_solution_data(j,i);
        }, local_conserved_quantity);
        Kokkos::fence();
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

        if (_optimization_object._local_bounds_preservation) {

            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_nlocal), [&](const int j) {

                target_values[j] = target_solution_data(j,i);

                // loop over neighbors to get max and min for each component of the field
                const local_index_type num_neighbors = _neighborhood_info->getNumNeighbors(j);

                for (local_index_type k=0; k<num_neighbors; ++k) {
                    scalar_type neighbor_value = (_neighborhood_info->getNeighbor(j,k) < (size_t)source_nlocal) ? source_solution_data(_neighborhood_info->getNeighbor(j,k), i) : source_halo_solution_data(_neighborhood_info->getNeighbor(j,k)-source_nlocal, i);

                    source_mins[j] = (neighbor_value < source_mins[j]) ? neighbor_value : source_mins[j];
                    source_maxs[j] = (neighbor_value > source_maxs[j]) ? neighbor_value : source_maxs[j];
                }

                // local bounds take into account global bounds as well
                if (use_global_lower_bound) {
                    source_mins[j] = std::max(source_mins[j], _optimization_object._global_lower_bound);
                }
                if (use_global_upper_bound) {
                    source_maxs[j] = std::min(source_maxs[j], _optimization_object._global_upper_bound);
                }
            });
            Kokkos::fence();

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
                Kokkos::fence();
                Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_num_local), [&](const int j) {
                    updated_target_values[j] = caas.get_Qm(j, 0)/weights[j];
                });
                Kokkos::fence();

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
        } else { // algorithm != NONE && NOT _optimization_object._local_bounds_preservation, so global_bounds_preservation

            // can handle a lower bound, an upper bound, or no bounds, but not both lower and upper bounds
            //TEUCHOS_TEST_FOR_EXCEPT_MSG(use_global_lower_bound && use_global_upper_bound, "Simultaneously supporting lower and upper global bounds without local bounds preservation is not currently supported.");
            //TEUCHOS_TEST_FOR_EXCEPT_MSG(use_global_upper_bound, "Not tested yet.");

            // get sum of target weights
            scalar_type local_target_weight_sum = 0, global_target_weight_sum = 0;
            Kokkos::parallel_reduce(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,_target_particles->getCoordsConst()->nLocal()), [&](const int j, double& t_local_target_weight_sum) {
                t_local_target_weight_sum += target_grid_weighting_field(j,0);
            }, local_target_weight_sum);
            Kokkos::fence();
            Teuchos::Ptr<scalar_type> global_target_weight_sum_ptr(&global_target_weight_sum);
            Teuchos::reduceAll<local_index_type, scalar_type>(*(_target_particles->getCoordsConst()->getComm()), Teuchos::REDUCE_SUM, local_target_weight_sum, global_target_weight_sum_ptr);

            //// not using NONE, so user wants conservation
            //// not using local_bounds_preservation, so cut to their global min/max and then scale
            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_nlocal), [&](const int j) {
                updated_target_values[j] = target_solution_data(j,i);
                //// cut to user's lower bound
                //if (use_global_lower_bound) {
                //    updated_target_values[j] = std::max(updated_target_values[j], _optimization_object._global_lower_bound);
                //}
                //// cut to user's upper bound
                //if (use_global_upper_bound) {
                //    updated_target_values[j] = std::min(updated_target_values[j], _optimization_object._global_upper_bound);
                //}
            });
            Kokkos::fence();
            
            // get current value of weighted target field (not yet conserved)
            scalar_type local_target_conserved_quantity = 0, global_target_conserved_quantity = 0;
            Kokkos::parallel_reduce(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,_target_particles->getCoordsConst()->nLocal()), [&](const int j, double& t_local_target_conserved_quantity) {
                t_local_target_conserved_quantity += target_grid_weighting_field(j,0)*updated_target_values[j];
            }, local_target_conserved_quantity);
            Kokkos::fence();
            Teuchos::Ptr<scalar_type> global_target_conserved_quantity_ptr(&global_target_conserved_quantity);
            Teuchos::reduceAll<local_index_type, scalar_type>(*(_target_particles->getCoordsConst()->getComm()), Teuchos::REDUCE_SUM, local_target_conserved_quantity, global_target_conserved_quantity_ptr);


            double scaling_factor = 0.0;
            double shift = 0.0;
            //if (use_global_lower_bound && _optimization_object._global_bounds_preservation)  {
            //    scaling_factor = (global_conserved_quantity - _optimization_object._global_lower_bound*global_target_weight_sum) 
            //                        / (global_target_conserved_quantity - _optimization_object._global_lower_bound*global_target_weight_sum);
            //    shift = _optimization_object._global_lower_bound;
            //    // replace data on target with this
            //    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,_target_particles->getCoordsConst()->nLocal()), [&](const int j) {
            //        updated_target_values[j] = scaling_factor*(updated_target_values[j]-shift)+shift;
            //    });
            //    Kokkos::fence();
            //} else if (use_global_upper_bound && _optimization_object._global_bounds_preservation) {
            //    scaling_factor = (-global_conserved_quantity + _optimization_object._global_upper_bound*global_target_weight_sum) 
            //                        / (-global_target_conserved_quantity + _optimization_object._global_upper_bound*global_target_weight_sum);
            //    shift = -_optimization_object._global_upper_bound;
            //    // replace data on target with this
            //    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,_target_particles->getCoordsConst()->nLocal()), [&](const int j) {
            //        updated_target_values[j] = scaling_factor*(updated_target_values[j]-shift)+shift;
            //    });
            //    Kokkos::fence();
            //} else {
                if (abs(global_conserved_quantity) > 1) {
                    scaling_factor = global_conserved_quantity / global_target_conserved_quantity;
                    // replace data on target with this
                    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,_target_particles->getCoordsConst()->nLocal()), [&](const int j) {
                        updated_target_values[j] = scaling_factor*(updated_target_values[j]-shift)+shift;
                    });
                    Kokkos::fence();
                } else {
                    shift = (global_target_conserved_quantity - global_conserved_quantity) / global_target_weight_sum;
                    // replace data on target with this
                    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,_target_particles->getCoordsConst()->nLocal()), [&](const int j) {
                        updated_target_values[j] = updated_target_values[j]-shift;
                    });
                    Kokkos::fence();
                }
            //}
            //// replace data on target with this
            //Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,_target_particles->getCoordsConst()->nLocal()), [&](const int j) {
            //    updated_target_values[j] = scaling_factor*(updated_target_values[j]-shift)+shift;
            //});
            //Kokkos::fence();
        }

        // replace data on target with this
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,_target_particles->getCoordsConst()->nLocal()), [&](const int j) {
            target_solution_data(j,i) = updated_target_values[j];
        });
        Kokkos::fence();
    }

    OptimizationTime->stop();

} // algorithm != NONE
}

}
