#include "Compadre_OBFET.hpp"

#include "Compadre_FieldManager.hpp"
#include "Compadre_FieldT.hpp"
#include "Compadre_CoordsT.hpp"
#include "Compadre_NeighborhoodT.hpp"
#include "Compadre_ParticlesT.hpp"
#include "Compadre_XyzVector.hpp"

#include "TPL/Cobra_SLBQP.hpp" // OBFET

namespace Compadre {

//void display_vector(const std::vector<double> &v, bool column=false) {
//    if (column) std::copy(v.begin(), v.end(),
//                        std::ostream_iterator<double>(std::cout, "\n"));
//    else std::copy(v.begin(), v.end(),
//                   std::ostream_iterator<double>(std::cout, " "));
//}

void OBFET::solveAndUpdate() {
	// written for serial
	TEUCHOS_TEST_FOR_EXCEPT_MSG(_source_particles->getCoordsConst()->getComm()->getSize()>1, "OBFET only written for serial.");

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


	std::vector<scalar_type> weights(target_grid_weighting_field.dimension_0());
	for (local_index_type j=0; j<_target_particles->getCoordsConst()->nLocal(); ++j) {
		weights[j] = target_grid_weighting_field(j,0);
	}

	const local_index_type source_nlocal = _source_particles->getCoordsConst()->nLocal();
	const local_index_type target_nlocal = _target_particles->getCoordsConst()->nLocal();

	// loop over field's dimensions
	for (local_index_type i=0; i<target_solution_data.dimension_1(); ++i) {

		scalar_type local_conserved_quantity = 0, global_conserved_quantity = 0;
		for (local_index_type j=0; j<_source_particles->getCoordsConst()->nLocal(); ++j) {
			local_conserved_quantity += source_grid_weighting_field(j,0)*source_solution_data(j,i);
		}
		Teuchos::Ptr<scalar_type> global_conserved_quantity_ptr(&global_conserved_quantity);
		Teuchos::reduceAll<local_index_type, scalar_type>(*(_source_particles->getCoordsConst()->getComm()), Teuchos::REDUCE_SUM, local_conserved_quantity, global_conserved_quantity_ptr);

		std::vector<scalar_type> target_values(target_solution_data.dimension_0());
		std::vector<scalar_type> updated_target_values(target_solution_data.dimension_0(),0);
		std::vector<scalar_type> source_mins(target_solution_data.dimension_0(), 1e+15);//std::numeric_limits<scalar_type>::max());
		std::vector<scalar_type> source_maxs(target_solution_data.dimension_0(), -1e+15);//std::numeric_limits<scalar_type>::lowest());

		for (local_index_type j=0; j<target_nlocal; ++j) {

			target_values[j] = target_solution_data(j,i);

			// loop over neighbors to get max and min for each component of the field
			const std::vector<std::pair<size_t, scalar_type> > neighbors = _neighborhood_info->getNeighbors(j);
			const local_index_type num_neighbors = neighbors.size();

			for (local_index_type k=0; k<num_neighbors; ++k) {
				scalar_type neighbor_value = (neighbors[k].first < source_nlocal) ? source_solution_data(neighbors[k].first, i) : source_halo_solution_data(neighbors[k].first-source_nlocal, i);

				source_mins[j] = (neighbor_value < source_mins[j]) ? neighbor_value : source_mins[j];
				source_maxs[j] = (neighbor_value > source_maxs[j]) ? neighbor_value : source_maxs[j];
			}
		}

		// fill std::vec with these and then call Cobra_
		// Set up optimization inputs/outputs.
		Cobra::AlgoStatus<scalar_type> astatus;
		Cobra::StdVectorInterface svi;

		// Run optimization.
		Cobra::slbqp<scalar_type, std::vector<scalar_type> > (astatus, updated_target_values, target_values, source_mins, source_maxs, weights, global_conserved_quantity, &svi);

		// Display algorithm status.
		std::cout.precision(15);
		std::cout << std::scientific;
		std::cout << "\nOBFET Algorithm Termination Report:\n";
		std::cout << "Converged? ... " << astatus.converged << "\n";
		std::cout << "Iterations ... " << astatus.iterations << "\n";
		std::cout << "Bracketing ... " << astatus.bracketing_iterations << "\n";
		std::cout << "Secant     ... " << astatus.secant_iterations << "\n";
		std::cout << "Lambda     ... " << astatus.lambda << "\n";
		std::cout << "Residual   ... " << astatus.residual << "\n";

		// replace data on target with this
		for (local_index_type j=0; j<_target_particles->getCoordsConst()->nLocal(); ++j) {
			target_solution_data(j,i) = updated_target_values[j];
		}
	}

}

}
