#include "GMLS.hpp"

#include <float.h> // for DBL_MAX

#define ASSERT_WITH_MESSAGE(condition, message) do { \
if (!(condition)) { printf((message)); } \
assert ((condition)); } while(false)

#ifdef COMPADRE_USE_KOKKOSCORE

KOKKOS_INLINE_FUNCTION
double GMLS::Wab(const double r, const double h, const ReconstructionOperator::WeightingFunctionType& weighting_type, const int power) const {

    if (weighting_type == ReconstructionOperator::WeightingFunctionType::Power) {
    	return std::pow(1.0-std::abs(r/(3*h)), power) * double(1.0-std::abs(r/(3*h))>0.0);
    } else { // Gaussian
    	// 2.5066282746310002416124 = sqrt(2*pi)
    	double h_over_3 = h/3.0;
    	return 1./( h_over_3 * 2.5066282746310002416124 ) * std::exp(-.5*r*r/(h_over_3*h_over_3));
    }

}

KOKKOS_INLINE_FUNCTION
double GMLS::factorial(const int n) const {

    double f = 1.0;
    for(int i = 1; i <= n; i++){
        f*=i;
    }
    return f;

}

KOKKOS_INLINE_FUNCTION
void GMLS::calcWij(double* delta, const int target_index, int neighbor_index, const double alpha, const int dimension, const int poly_order, bool specific_order_only, scratch_matrix_type* V, scratch_matrix_type* T, const ReconstructionOperator::SamplingFunctional polynomial_sampling_functional, scratch_vector_type* target_manifold_gradient, scratch_matrix_type* quadrature_manifold_gradients) const {
/*
 * This class is under two levels of hierarchical parallelism, so we
 * do not put in any finer grain parallelism in this function
 */
	const int my_num_neighbors = this->getNNeighbors(target_index);

	int component = 0;
	if (neighbor_index >= my_num_neighbors) {
		component = neighbor_index / my_num_neighbors;
		neighbor_index = neighbor_index % my_num_neighbors;
	}

	XYZ relative_coord;
	if (neighbor_index > -1)
		for (int i=0; i<dimension; ++i) {
			// calculates (alpha*target+(1-alpha)*neighbor)-1*target = (alpha-1)*target + (1-alpha)*neighbor
			relative_coord[i] = (alpha-1)*getTargetCoordinate(target_index, i, V);
			relative_coord[i] += (1-alpha)*getNeighborCoordinate(target_index, neighbor_index, i, V);
		}
	else {
		for (int i=0; i<3; ++i) relative_coord[i] = 0;
	}

	if (polynomial_sampling_functional == ReconstructionOperator::SamplingFunctional::PointSample) {

//		double operator_coefficient = 1;
//		// use neighbor scalar value
//		if (_operator_coefficients.dimension_0() > 0) { // user set this vector
//			operator_coefficient = _operator_coefficients(neighbor_index,0);
//		}

		double cutoff_p = _epsilons(target_index);
		int alphax, alphay, alphaz;
		double alphaf;
		int i = 0;
		const int start_index = specific_order_only ? poly_order : 0; // only compute specified order if requested
		for (int n = start_index; n <= poly_order; n++){
			if (dimension == 3) {
				for (alphaz = 0; alphaz <= n; alphaz++){

					int s = n - alphaz;
					for (alphay = 0; alphay <= s; alphay++){
						alphax = s - alphay;
						alphaf = factorial(alphax)*factorial(alphay)*factorial(alphaz);
						*(delta+i) = std::pow(relative_coord.x/cutoff_p,alphax)
									*std::pow(relative_coord.y/cutoff_p,alphay)
									*std::pow(relative_coord.z/cutoff_p,alphaz)/alphaf;
						i++;
					}
				}
			} else if (dimension == 2) {
				for (alphay = 0; alphay <= n; alphay++){
					alphax = n - alphay;
					alphaf = factorial(alphax)*factorial(alphay);
					*(delta+i) = std::pow(relative_coord.x/cutoff_p,alphax)
								*std::pow(relative_coord.y/cutoff_p,alphay)/alphaf;
					i++;
				}
			} else { // dimension == 1
					alphax = n;
					alphaf = factorial(alphax);
					*(delta+i) = std::pow(relative_coord.x/cutoff_p,alphax)/alphaf;
					i++;
			}
		}
	} else if (polynomial_sampling_functional == ReconstructionOperator::SamplingFunctional::ManifoldVectorSample || polynomial_sampling_functional == ReconstructionOperator::SamplingFunctional::ManifoldGradientVectorSample) {

//		double operator_coefficient = 1;
//		// use neighbor scalar value
//		if (_operator_coefficients.dimension_0() > 0) { // user set this vector
//			operator_coefficient = _operator_coefficients(neighbor_index,0);
//		}

		const int dimension_offset = this->getNP(_poly_order, dimension);
		double cutoff_p = _epsilons(target_index);
		int alphax, alphay, alphaz;
		double alphaf;
		int i = 0;

		for (int d=0; d<dimension; ++d) {
			if (d==component) {
				const int start_index = specific_order_only ? poly_order : 0; // only compute specified order if requested
				for (int n = start_index; n <= poly_order; n++){
					if (dimension == 3) {
						for (alphaz = 0; alphaz <= n; alphaz++){

							int s = n - alphaz;
							for (alphay = 0; alphay <= s; alphay++){
								alphax = s - alphay;
								alphaf = factorial(alphax)*factorial(alphay)*factorial(alphaz);
								*(delta+component*dimension_offset+i) = std::pow(relative_coord.x/cutoff_p,alphax)
											*std::pow(relative_coord.y/cutoff_p,alphay)
											*std::pow(relative_coord.z/cutoff_p,alphaz)/alphaf;
								i++;
							}
						}
					} else if (dimension == 2) {
						for (alphay = 0; alphay <= n; alphay++){
							alphax = n - alphay;
							alphaf = factorial(alphax)*factorial(alphay);
							*(delta+component*dimension_offset+i) = std::pow(relative_coord.x/cutoff_p,alphax)
										*std::pow(relative_coord.y/cutoff_p,alphay)/alphaf;
							i++;
						}
					} else { // dimension == 1
							alphax = n;
							alphaf = factorial(alphax);
							*(delta+component*dimension_offset+i) = std::pow(relative_coord.x/cutoff_p,alphax)/alphaf;
							i++;
					}
				}
			} else {
				for (int n=0; n<dimension_offset; ++n) {
					*(delta+d*dimension_offset+n) = 0;
				}
			}
		}
	} else if (polynomial_sampling_functional == ReconstructionOperator::SamplingFunctional::StaggeredEdgeAnalyticGradientIntegralSample) {

		double operator_coefficient = 1;
//		// use average scalar value
//		if (_operator_coefficients.dimension_0() > 0) { // user set this vector
//			operator_coefficient = 0.5*(_operator_coefficients(target_index,0) + _operator_coefficients(neighbor_index,0));
//			std::cout << operator_coefficient << std::endl;
//		}

		{

			double cutoff_p = _epsilons(target_index);
			int alphax, alphay, alphaz;
			double alphaf;
			int i = 0;
			const int start_index = specific_order_only ? poly_order : 0; // only compute specified order if requested
			for (int n = start_index; n <= poly_order; n++){
				if (dimension == 3) {
					for (alphaz = 0; alphaz <= n; alphaz++){

						int s = n - alphaz;
						for (alphay = 0; alphay <= s; alphay++){
							alphax = s - alphay;
							alphaf = factorial(alphax)*factorial(alphay)*factorial(alphaz);
							*(delta+i) = -operator_coefficient*std::pow(relative_coord.x/cutoff_p,alphax)
										*std::pow(relative_coord.y/cutoff_p,alphay)
										*std::pow(relative_coord.z/cutoff_p,alphaz)/alphaf;
							i++;
						}
					}
				} else if (dimension == 2) {
					for (alphay = 0; alphay <= n; alphay++){
						alphax = n - alphay;
						alphaf = factorial(alphax)*factorial(alphay);
						*(delta+i) = -operator_coefficient*std::pow(relative_coord.x/cutoff_p,alphax)
									*std::pow(relative_coord.y/cutoff_p,alphay)/alphaf;
						i++;
					}
				} else { // dimension == 1
						alphax = n;
						alphaf = factorial(alphax);
						*(delta+i) = -operator_coefficient*std::pow(relative_coord.x/cutoff_p,alphax)/alphaf;
						i++;
				}
			}
		}
		{
			relative_coord.x = 0;
			relative_coord.y = 0;
			relative_coord.z = 0;

			double cutoff_p = _epsilons(target_index);
			int alphax, alphay, alphaz;
			double alphaf;
			int i = 0;
			const int start_index = specific_order_only ? poly_order : 0; // only compute specified order if requested
			for (int n = start_index; n <= poly_order; n++){
				if (dimension == 3) {
					for (alphaz = 0; alphaz <= n; alphaz++){

						int s = n - alphaz;
						for (alphay = 0; alphay <= s; alphay++){
							alphax = s - alphay;
							alphaf = factorial(alphax)*factorial(alphay)*factorial(alphaz);
							*(delta+i) += operator_coefficient*std::pow(relative_coord.x/cutoff_p,alphax)
										*std::pow(relative_coord.y/cutoff_p,alphay)
										*std::pow(relative_coord.z/cutoff_p,alphaz)/alphaf;
							i++;
						}
					}
				} else if (dimension == 2) {
					for (alphay = 0; alphay <= n; alphay++){
						alphax = n - alphay;
						alphaf = factorial(alphax)*factorial(alphay);
						*(delta+i) += operator_coefficient*std::pow(relative_coord.x/cutoff_p,alphax)
									*std::pow(relative_coord.y/cutoff_p,alphay)/alphaf;
						i++;
					}
				} else { // dimension == 1
						alphax = n;
						alphaf = factorial(alphax);
						*(delta+i) += operator_coefficient*std::pow(relative_coord.x/cutoff_p,alphax)/alphaf;
						i++;
				}
			}
		}
	} else if (polynomial_sampling_functional == ReconstructionOperator::SamplingFunctional::StaggeredEdgeIntegralSample) {

		double operator_coefficient = 1;
//		// use average scalar value
//		if (_operator_coefficients.dimension_0() > 0) { // user set this vector
//			operator_coefficient = 0.5*(_operator_coefficients(target_index,0) + _operator_coefficients(neighbor_index,0));
//		}

		double cutoff_p = _epsilons(target_index);

		int alphax, alphay;
		double alphaf;
		const int start_index = specific_order_only ? poly_order : 0; // only compute specified order if requested

		for (int quadrature = 0; quadrature<_number_of_quadrature_points; ++quadrature) {

			int i = 0;

			XYZ tangent_quadrature_coord_2d;
			XYZ quadrature_coord_2d;
			for (int j=0; j<dimension; ++j) {
				// calculates (alpha*target+(1-alpha)*neighbor)-1*target = (alpha-1)*target + (1-alpha)*neighbor
				quadrature_coord_2d[j] = (_parameterized_quadrature_sites[quadrature]-1)*getTargetCoordinate(target_index, j, V);
				quadrature_coord_2d[j] += (1-_parameterized_quadrature_sites[quadrature])*getNeighborCoordinate(target_index, neighbor_index, j, V);
				tangent_quadrature_coord_2d[j] = getTargetCoordinate(target_index, j, V);
				tangent_quadrature_coord_2d[j] += -getNeighborCoordinate(target_index, neighbor_index, j, V);
			}
			for (int j=0; j<_basis_multiplier; ++j) {
				for (int n = start_index; n <= poly_order; n++){
					for (alphay = 0; alphay <= n; alphay++){
						alphax = n - alphay;
						alphaf = factorial(alphax)*factorial(alphay);

						// local evaluation of vector [0,p] or [p,0]
						double v0, v1;
						v0 = (j==0) ? std::pow(quadrature_coord_2d.x/cutoff_p,alphax)
							*std::pow(quadrature_coord_2d.y/cutoff_p,alphay)/alphaf : 0;
						v1 = (j==0) ? 0 : std::pow(quadrature_coord_2d.x/cutoff_p,alphax)
							*std::pow(quadrature_coord_2d.y/cutoff_p,alphay)/alphaf;

						double dot_product = tangent_quadrature_coord_2d[0]*v0 + tangent_quadrature_coord_2d[1]*v1;

						// multiply by quadrature weight
						if (quadrature==0) {
							*(delta+i) = operator_coefficient * dot_product * _quadrature_weights[quadrature];
						} else {
							*(delta+i) += operator_coefficient * dot_product * _quadrature_weights[quadrature];
						}
						i++;
					}
				}
			}
		}
	}
}


KOKKOS_INLINE_FUNCTION
void GMLS::calcGradientWij(double* delta, const int target_index, const int neighbor_index, const double alpha, const int partial_direction, const int dimension, const int poly_order, bool specific_order_only, scratch_matrix_type* V, const ReconstructionOperator::SamplingFunctional polynomial_sampling_functional) const {
/*
 * This class is under two levels of hierarchical parallelism, so we
 * do not put in any finer grain parallelism in this function
 */
	// alpha corresponds to the linear combination of target_index and neighbor_index coordinates
	// coordinate to evaluate = alpha*(target_index's coordinate) + (1-alpha)*(neighbor_index's coordinate)

	// partial_direction - 0=x, 1=y, 2=z
	XYZ relative_coord;
	if (neighbor_index > -1)
		for (int i=0; i<dimension; ++i) {
			// calculates (alpha*target+(1-alpha)*neighbor)-1*target = (alpha-1)*target + (1-alpha)*neighbor
			relative_coord[i] = (alpha-1)*getTargetCoordinate(target_index, i, V);
			relative_coord[i] += (1-alpha)*getNeighborCoordinate(target_index, neighbor_index, i, V);
		}
	else {
		for (int i=0; i<3; ++i) relative_coord[i] = 0;
	}

	double cutoff_p = _epsilons(target_index);

	int alphax, alphay, alphaz;
	double alphaf;
	int i = 0;
	const int start_index = specific_order_only ? poly_order : 0; // only compute specified order if requested

	const int alphax_start = (partial_direction == 0) ? 1 : 0;
	const int alphay_start = (partial_direction == 1) ? 1 : 0;
	const int alphaz_start = (partial_direction == 2) ? 1 : 0;

	for (int n = start_index; n <= poly_order; n++){
		if (dimension == 3) {
			for (alphaz = 0; alphaz <= n; alphaz++){
				int s = n - alphaz;
				for (alphay = 0; alphay <= s; alphay++){
					alphax = s - alphay;
					bool zero_out = (partial_direction==0 && alphax==0)
									|| (partial_direction==1 && alphay==0)
									|| (partial_direction==2 && alphaz==0);
					bool ones = (partial_direction==0 && alphax==1)
									|| (partial_direction==1 && alphay==1)
									|| (partial_direction==2 && alphaz==1);

					if (zero_out) {
						*(delta+i) = 0;
						i++;
					} else {
						alphaf = factorial(alphax-alphax_start)*factorial(alphay-alphay_start)*factorial(alphaz-alphaz_start);
						if (ones) {
							if (partial_direction==0) {
								*(delta+i) = 1./cutoff_p
											*std::pow(relative_coord.y/cutoff_p, alphay)
											*std::pow(relative_coord.z/cutoff_p, alphaz)/alphaf;
								i++;
							} else if (partial_direction==1) {
								*(delta+i) = 1./cutoff_p * std::pow(relative_coord.x/cutoff_p, alphax)
											*std::pow(relative_coord.z/cutoff_p, alphaz)/alphaf;
								i++;
							} else {
								*(delta+i) = 1./cutoff_p * std::pow(relative_coord.x/cutoff_p, alphax)
											*std::pow(relative_coord.y/cutoff_p, alphay)/alphaf;
								i++;
							}
						} else {
							*(delta+i) = 1./cutoff_p * std::pow(relative_coord.x/cutoff_p, alphax)
										*std::pow(relative_coord.y/cutoff_p, alphay)
										*std::pow(relative_coord.z/cutoff_p, alphaz)/alphaf;
							i++;
						}
					}
				}
			}
		} else if (dimension == 2) {
			for (alphay = 0; alphay <= n; alphay++){
				alphax = n - alphay;

				bool zero_out = (partial_direction==0 && alphax==0)
								|| (partial_direction==1 && alphay==0);
				bool ones = (partial_direction==0 && alphax==1)
								|| (partial_direction==1 && alphay==1);

				if (zero_out) {
					*(delta+i) = 0;
					i++;
				} else {
					alphaf = factorial(alphax-alphax_start)*factorial(alphay-alphay_start);
					if (ones) {
						if (partial_direction==0) {
							*(delta+i) = 1./cutoff_p
										*std::pow(relative_coord.y/cutoff_p, alphay)/alphaf;
							i++;
						} else if (partial_direction==1) {
							*(delta+i) = 1./cutoff_p * std::pow(relative_coord.x/cutoff_p, alphax)/alphaf;
							i++;
						}
					} else {
						*(delta+i) = 1./cutoff_p * std::pow(relative_coord.x/cutoff_p, alphax)
									*std::pow(relative_coord.y/cutoff_p, alphay)/alphaf;
						i++;
					}
				}
			}
		} else { // dimension == 1
				alphax = n;

				bool zero_out = (partial_direction==0 && alphax==0);
				bool ones = (partial_direction==0 && alphax==1);

				if (zero_out) {
					*(delta+i) = 0;
					i++;
				} else {
					alphaf = factorial(alphax-alphax_start);
					if (ones) {
						if (partial_direction==0) {
							*(delta+i) = 1./cutoff_p/alphaf;
							i++;
						}
					} else {
						*(delta+i) = 1./cutoff_p * std::pow(relative_coord.x/cutoff_p, alphax)/alphaf;
						i++;
					}
				}
		}
	}
}


KOKKOS_INLINE_FUNCTION
void GMLS::createWeightsAndP(const member_type& teamMember, scratch_vector_type delta, scratch_matrix_type P, scratch_vector_type w, const int dimension, int polynomial_order, bool weight_p, scratch_matrix_type* V, scratch_matrix_type* T, const ReconstructionOperator::SamplingFunctional polynomial_sampling_functional, scratch_vector_type* target_manifold_gradient, scratch_matrix_type* quadrature_manifold_gradients) const {
    /*
     * Creates sqrt(W)*P
     */
	const int target_index = teamMember.league_rank();
//	printf("specific order: %d\n", specific_order);
//	{
//		const int storage_size = (specific_order > 0) ? this->getNP(specific_order, dimension)-this->getNP(specific_order-1, dimension) : this->getNP(_poly_order, dimension);
//		printf("storage size: %d\n", storage_size);
//	}
//	printf("weight_p: %d\n", weight_p);

	teamMember.team_barrier();
	Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,this->getNNeighbors(target_index)),
			[=] (const int i) {

		const int my_num_neighbors = this->getNNeighbors(target_index);

		for (int d=0; d<_sampling_multiplier; ++d) {
			// in 2d case would use distance between SVD coordinates

			// ignores V when calculating weights from a point, i.e. uses actual point values
			double r;

			// coefficient muliplied by relative distance (allows for mid-edge weighting if applicable)
			double alpha_weight = 1;
			if (_polynomial_sampling_functional==ReconstructionOperator::SamplingFunctional::StaggeredEdgeIntegralSample
					|| _polynomial_sampling_functional==ReconstructionOperator::SamplingFunctional::StaggeredEdgeAnalyticGradientIntegralSample) {
				alpha_weight = 0.5;
			}

			// get Euchlidean distance of scaled relative coordinate from the origin
			if (V==NULL) {
				r = this->EuclideanVectorLength(this->getRelativeCoord(target_index, i, dimension) * alpha_weight, dimension);
			} else {
				r = this->EuclideanVectorLength(this->getRelativeCoord(target_index, i, dimension, V) * alpha_weight, dimension);
			}

			// generate weight vector from distances and window sizes
			w(i+my_num_neighbors*d) = this->Wab(r, _epsilons(target_index), _weighting_type, _weighting_power);

			this->calcWij(delta.data(), target_index, i + d*my_num_neighbors, 0 /*alpha*/, dimension, polynomial_order, false /*bool on only specific order*/, V, T, polynomial_sampling_functional, target_manifold_gradient, quadrature_manifold_gradients);

			int storage_size = this->getNP(polynomial_order, dimension);
			storage_size *= _basis_multiplier;

			if (weight_p) {
				for (int j = 0; j < storage_size; ++j) {
					// stores as transposed P on gpu
					P(ORDER_INDICES(i+my_num_neighbors*d,j)) = delta[j] * std::sqrt(w(i+my_num_neighbors*d));
				}

			} else {

				for (int j = 0; j < storage_size; ++j) {
					// stores as transposed P on gpu
					P(ORDER_INDICES(i+my_num_neighbors*d,j)) = delta[j];
				}
			}
		}
    });
	teamMember.team_barrier();
//	Kokkos::single(Kokkos::PerTeam(teamMember), [=] () {
//		for (int k=0; k<this->getNNeighbors(target_index); k++) {
//			for (int l=0; l<_NP; l++) {
//				printf("%i %i %0.16f\n", k, l, P(k,l) );// << " " << l << " " << R(k,l) << std::endl;
//			}
//		}
//	});
}

KOKKOS_INLINE_FUNCTION
void GMLS::createWeightsAndPForCurvature(const member_type& teamMember, scratch_vector_type delta, scratch_matrix_type P, scratch_vector_type w, const int dimension, bool only_specific_order, scratch_matrix_type* V) const {
    /*
     * This function has two purposes
     * 1.) Used to calculate specifically for 1st order polynomials, from which we can reconstruct a tangent plane
     * 2.) Used to calculate a polynomial of _manifold_poly_order, which we use to calculate curvature of the manifold
     */

	const int target_index = teamMember.league_rank();

	teamMember.team_barrier();
	Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,this->getNNeighbors(target_index)),
			[=] (const int i) {

		const int my_num_neighbors = this->getNNeighbors(target_index);

		// ignores V when calculating weights from a point, i.e. uses actual point values
		double r;

		// get Euclidean distance of scaled relative coordinate from the origin
		if (V==NULL) {
			r = this->EuclideanVectorLength(this->getRelativeCoord(target_index, i, dimension), dimension);
		} else {
			r = this->EuclideanVectorLength(this->getRelativeCoord(target_index, i, dimension, V), dimension);
		}

		// generate weight vector from distances and window sizes
		if (only_specific_order) {
			w(i) = this->Wab(r, _epsilons(target_index), _manifold_weighting_type, _manifold_weighting_power);
			this->calcWij(delta.data(), target_index, i, 0 /*alpha*/, dimension, 1, true /*bool on only specific order*/);
		} else {
			w(i) = this->Wab(r, _epsilons(target_index), _manifold_weighting_type, _manifold_weighting_power);
			this->calcWij(delta.data(), target_index, i, 0 /*alpha*/, dimension, _manifold_poly_order, false /*bool on only specific order*/, V);
		}

		int storage_size = only_specific_order ? this->getNP(1, dimension)-this->getNP(0, dimension) : this->getNP(_manifold_poly_order, dimension);

		for (int j = 0; j < storage_size; ++j) {
			// stores as transposed P on gpu
			P(ORDER_INDICES(i,j)) = delta[j] * std::sqrt(w(i));
		}

    });
	teamMember.team_barrier();
}

KOKKOS_INLINE_FUNCTION
void GMLS::computeTargetFunctionals(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_matrix_type P_target_row, const int basis_multiplier_component) const {

	const int target_index = teamMember.league_rank();

	const int target_NP = this->getNP(_poly_order, _dimensions);
	for (int i=0; i<_operations.size(); ++i) {
		if (_operations(i) == ReconstructionOperator::TargetOperation::ScalarPointEvaluation) {
			Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
				this->calcWij(P_target_row.data()+_lro_total_offsets[i]*target_NP, target_index, -1 /* target is neighbor */, 1 /*alpha*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, NULL /*&T*/, ReconstructionOperator::SamplingFunctional::PointSample);
			});
			teamMember.team_barrier();
		} else if (_operations(i) == ReconstructionOperator::TargetOperation::VectorPointEvaluation) {
			ASSERT_WITH_MESSAGE(false, "Functionality for vectors not yet available. Currently performed using ScalarPointEvaluation for each component.");
		} else if (_operations(i) == ReconstructionOperator::TargetOperation::LaplacianOfScalarPointEvaluation) {
			Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
				const int offset = _lro_total_offsets[i]*target_NP;
				for (int j=0; j<target_NP; ++j) {
					P_target_row(offset + j, basis_multiplier_component) = 0;
				}
				switch (_dimensions) {
				case 3:
					P_target_row(offset + 4, basis_multiplier_component) = std::pow(_epsilons(target_index), -2);
					P_target_row(offset + 6, basis_multiplier_component) = std::pow(_epsilons(target_index), -2);
					P_target_row(offset + 9, basis_multiplier_component) = std::pow(_epsilons(target_index), -2);
					break;
				case 2:
					P_target_row(offset + 3, basis_multiplier_component) = std::pow(_epsilons(target_index), -2);
					P_target_row(offset + 5, basis_multiplier_component) = std::pow(_epsilons(target_index), -2);
					break;
				default:
					P_target_row(offset + 2, basis_multiplier_component) = std::pow(_epsilons(target_index), -2);
				}
			});
			teamMember.team_barrier();
		} else if (_operations(i) == ReconstructionOperator::TargetOperation::GradientOfScalarPointEvaluation) {
			Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
				int offset = (_lro_total_offsets[i]+0)*target_NP;
				for (int j=0; j<target_NP; ++j) {
					P_target_row(offset + j, basis_multiplier_component) = 0;
				}
				P_target_row(offset + 1, basis_multiplier_component) = std::pow(_epsilons(target_index), -1);
//				this->calcGradientWij(P_target_row.data()+offset, target_index, -1 /* target is neighbor */, 1 /*alpha*/, 0 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionOperator::SamplingFunctional::PointSample);

				if (_dimensions>1) {
					offset = (_lro_total_offsets[i]+1)*target_NP;
					for (int j=0; j<target_NP; ++j) {
						P_target_row(offset + j, basis_multiplier_component) = 0;
					}
					P_target_row(offset + 2, basis_multiplier_component) = std::pow(_epsilons(target_index), -1);
//					this->calcGradientWij(P_target_row.data()+offset, target_index, -1 /* target is neighbor */, 1 /*alpha*/, 1 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionOperator::SamplingFunctional::PointSample);
				}

				if (_dimensions>2) {
					offset = (_lro_total_offsets[i]+2)*target_NP;
					for (int j=0; j<target_NP; ++j) {
						P_target_row(offset + j, basis_multiplier_component) = 0;
					}
					P_target_row(offset + 3, basis_multiplier_component) = std::pow(_epsilons(target_index), -1);
//					this->calcGradientWij(P_target_row.data()+offset, target_index, -1 /* target is neighbor */, 1 /*alpha*/, 2 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionOperator::SamplingFunctional::PointSample);
				}
			});
			teamMember.team_barrier();
		} else if (_operations(i) == ReconstructionOperator::TargetOperation::PartialXOfScalarPointEvaluation) {
			Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
				int offset = _lro_total_offsets[i]*target_NP;
				for (int j=0; j<target_NP; ++j) {
					P_target_row(offset + j, basis_multiplier_component) = 0;
				}
				P_target_row(offset + 1, basis_multiplier_component) = std::pow(_epsilons(target_index), -1);
//				this->calcGradientWij(P_target_row.data()+offset, target_index, -1 /* target is neighbor */, 1 /*alpha*/, 0 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionOperator::SamplingFunctional::PointSample);
				});
			teamMember.team_barrier();
		} else if (_operations(i) == ReconstructionOperator::TargetOperation::PartialYOfScalarPointEvaluation) {
			if (_dimensions>1) {
				Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
					int offset = _lro_total_offsets[i]*target_NP;
					for (int j=0; j<target_NP; ++j) {
						P_target_row(offset + j, basis_multiplier_component) = 0;
					}
					P_target_row(offset + 2, basis_multiplier_component) = std::pow(_epsilons(target_index), -1);
//					this->calcGradientWij(P_target_row.data()+offset, target_index, -1 /* target is neighbor */, 1 /*alpha*/, 1 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionOperator::SamplingFunctional::PointSample);
				});
				teamMember.team_barrier();
			}
		} else if (_operations(i) == ReconstructionOperator::TargetOperation::PartialZOfScalarPointEvaluation) {
			if (_dimensions>2) {
				Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
					int offset = _lro_total_offsets[i]*target_NP;
					for (int j=0; j<target_NP; ++j) {
						P_target_row(offset + j, basis_multiplier_component) = 0;
					}
					P_target_row(offset + 3, basis_multiplier_component) = std::pow(_epsilons(target_index), -1);
//					this->calcGradientWij(P_target_row.data()+offset, target_index, -1 /* target is neighbor */, 1 /*alpha*/, 2 /*partial_direction*/, _dimensions, _poly_order, false /*specific order only*/, NULL /*&V*/, ReconstructionOperator::SamplingFunctional::PointSample);
				});
			}
			teamMember.team_barrier();
		} else if (_operations(i) == ReconstructionOperator::TargetOperation::DivergenceOfVectorPointEvaluation) {
			Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
				int offset = (_lro_total_offsets[i]+0)*target_NP;
				for (int j=0; j<target_NP; ++j) {
					P_target_row(offset + j, basis_multiplier_component) = 0;
				}

				P_target_row(offset + 1, basis_multiplier_component) = std::pow(_epsilons(target_index), -1);

				if (_dimensions>1) {
					offset = (_lro_total_offsets[i]+1)*target_NP;
					for (int j=0; j<target_NP; ++j) {
						P_target_row(offset + j, basis_multiplier_component) = 0;
					}
					P_target_row(offset + 2, basis_multiplier_component) = std::pow(_epsilons(target_index), -1);
				}

				if (_dimensions>2) {
					offset = (_lro_total_offsets[i]+2)*target_NP;
					for (int j=0; j<target_NP; ++j) {
						P_target_row(offset + j, basis_multiplier_component) = 0;
					}
					P_target_row(offset + 3, basis_multiplier_component) = std::pow(_epsilons(target_index), -1);
				}
			});
			teamMember.team_barrier();
		} else if (_operations(i) == ReconstructionOperator::TargetOperation::CurlOfVectorPointEvaluation) {
			// comments based on taking curl of vector [u_{0},u_{1},u_{2}]^T
			// with as e.g., u_{1,z} being the partial derivative with respect to z of
			// u_{1}
			if (_dimensions==3) {
				Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
					// output component 0
					// u_{2,y} - u_{1,z}
					{
						int offset = (_lro_total_offsets[i]+0)*target_NP;
						for (int j=0; j<target_NP; ++j) {
							P_target_row(offset + j, basis_multiplier_component) = 0;
						}
						// role of input 0 on component 0 of curl
						// (no contribution)

						offset = (_lro_total_offsets[i]+1*_lro_output_tile_size[i]+0)*target_NP;
						for (int j=0; j<target_NP; ++j) {
							P_target_row(offset + j, basis_multiplier_component) = 0;
						}
						// role of input 1 on component 0 of curl
						// -u_{1,z}
						P_target_row(offset + 3, basis_multiplier_component) = -std::pow(_epsilons(target_index), -1);

						offset = (_lro_total_offsets[i]+2*_lro_output_tile_size[i]+0)*target_NP;
						for (int j=0; j<target_NP; ++j) {
							P_target_row(offset + j, basis_multiplier_component) = 0;
						}
						// role of input 2 on component 0 of curl
						// u_{2,y}
						P_target_row(offset + 2, basis_multiplier_component) = std::pow(_epsilons(target_index), -1);
					}

					// output component 1
					// -u_{2,x} + u_{0,z}
					{
						int offset = (_lro_total_offsets[i]+0*_lro_output_tile_size[i]+1)*target_NP;
						for (int j=0; j<target_NP; ++j) {
							P_target_row(offset + j, basis_multiplier_component) = 0;
						}
						// role of input 0 on component 1 of curl
						// u_{0,z}
						P_target_row(offset + 3, basis_multiplier_component) = std::pow(_epsilons(target_index), -1);

						offset = (_lro_total_offsets[i]+1*_lro_output_tile_size[i]+1)*target_NP;
						for (int j=0; j<target_NP; ++j) {
							P_target_row(offset + j, basis_multiplier_component) = 0;
						}
						// role of input 1 on component 1 of curl
						// (no contribution)

						offset = (_lro_total_offsets[i]+2*_lro_output_tile_size[i]+1)*target_NP;
						for (int j=0; j<target_NP; ++j) {
							P_target_row(offset + j, basis_multiplier_component) = 0;
						}
						// role of input 2 on component 1 of curl
						// -u_{2,x}
						P_target_row(offset + 1, basis_multiplier_component) = -std::pow(_epsilons(target_index), -1);
					}

					// output component 2
					// u_{1,x} - u_{0,y}
					{
						int offset = (_lro_total_offsets[i]+0*_lro_output_tile_size[i]+2)*target_NP;
						for (int j=0; j<target_NP; ++j) {
							P_target_row(offset + j, basis_multiplier_component) = 0;
						}
						// role of input 0 on component 1 of curl
						// -u_{0,y}
						P_target_row(offset + 2, basis_multiplier_component) = -std::pow(_epsilons(target_index), -1);

						offset = (_lro_total_offsets[i]+1*_lro_output_tile_size[i]+2)*target_NP;
						for (int j=0; j<target_NP; ++j) {
							P_target_row(offset + j, basis_multiplier_component) = 0;
						}
						// role of input 1 on component 1 of curl
						// u_{1,x}
						P_target_row(offset + 1, basis_multiplier_component) = std::pow(_epsilons(target_index), -1);

						offset = (_lro_total_offsets[i]+2*_lro_output_tile_size[i]+2)*target_NP;
						for (int j=0; j<target_NP; ++j) {
							P_target_row(offset + j, basis_multiplier_component) = 0;
						}
						// role of input 2 on component 1 of curl
						// (no contribution)
					}
				});
			} else if (_dimensions==2) {
				Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
					// output component 0
					// u_{1,y}
					{
						int offset = (_lro_total_offsets[i]+0)*target_NP;
						for (int j=0; j<target_NP; ++j) {
							P_target_row(offset + j, basis_multiplier_component) = 0;
						}
						// role of input 0 on component 0 of curl
						// (no contribution)

						offset = (_lro_total_offsets[i]+1*_lro_output_tile_size[i]+0)*target_NP;
						for (int j=0; j<target_NP; ++j) {
							P_target_row(offset + j, basis_multiplier_component) = 0;
						}
						// role of input 1 on component 0 of curl
						// -u_{1,z}
						P_target_row(offset + 2, basis_multiplier_component) = std::pow(_epsilons(target_index), -1);
					}

					// output component 1
					// -u_{0,x}
					{
						int offset = (_lro_total_offsets[i]+0*_lro_output_tile_size[i]+1)*target_NP;
						for (int j=0; j<target_NP; ++j) {
							P_target_row(offset + j, basis_multiplier_component) = 0;
						}
						// role of input 0 on component 1 of curl
						// u_{0,z}
						P_target_row(offset + 1, basis_multiplier_component) = -std::pow(_epsilons(target_index), -1);

						offset = (_lro_total_offsets[i]+1*_lro_output_tile_size[i]+1)*target_NP;
						for (int j=0; j<target_NP; ++j) {
							P_target_row(offset + j, basis_multiplier_component) = 0;
						}
						// role of input 1 on component 1 of curl
						// (no contribution)
					}
				});
			}
			teamMember.team_barrier();
		} else {
			ASSERT_WITH_MESSAGE(false, "Functionality not yet available.");
		}
	}
}

KOKKOS_INLINE_FUNCTION
void GMLS::computeManifoldFunctionals(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_matrix_type P_target_row, scratch_matrix_type* V, const int neighbor_index, const double alpha, const int basis_multiplier_component) const {

	const int target_index = teamMember.league_rank();

	const int manifold_NP = this->getNP(_manifold_poly_order, _dimensions-1);
	for (int i=0; i<_manifold_support_operations.size(); ++i) {
		if (_manifold_support_operations(i) == ReconstructionOperator::TargetOperation::ScalarPointEvaluation) {
			Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
				int offset = i*manifold_NP;
				this->calcWij(t1.data(), target_index, neighbor_index, alpha, _dimensions-1, _manifold_poly_order, false /*bool on only specific order*/, V, NULL /*&T*/, ReconstructionOperator::SamplingFunctional::PointSample);
				for (int j=0; j<manifold_NP; ++j) {
					P_target_row(offset + j, basis_multiplier_component) = t1(j);
				}
			});
			teamMember.team_barrier();
		} else if (_manifold_support_operations(i) == ReconstructionOperator::TargetOperation::GradientOfScalarPointEvaluation) {
			Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
				int offset = i*manifold_NP;
				this->calcGradientWij(t1.data(), target_index, neighbor_index, alpha, 0 /*partial_direction*/, _dimensions-1, _manifold_poly_order, false /*specific order only*/, V, ReconstructionOperator::SamplingFunctional::PointSample);
				for (int j=0; j<manifold_NP; ++j) {
					P_target_row(offset + j, basis_multiplier_component) = t1(j);
				}
				if (_dimensions>2) { // _dimensions-1 > 1
					offset = (i+1)*manifold_NP;
					this->calcGradientWij(t1.data(), target_index, neighbor_index, alpha, 1 /*partial_direction*/, _dimensions-1, _manifold_poly_order, false /*specific order only*/, V, ReconstructionOperator::SamplingFunctional::PointSample);
					for (int j=0; j<manifold_NP; ++j) {
						P_target_row(offset + j, basis_multiplier_component) = t1(j);
					}
				}
			});
			teamMember.team_barrier();
		} else {
			ASSERT_WITH_MESSAGE(false, "Functionality not yet available.");
		}
	}
}

KOKKOS_INLINE_FUNCTION
void GMLS::computeTargetFunctionalsOnManifold(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_matrix_type P_target_row, Kokkos::View<ReconstructionOperator::TargetOperation*> operations, scratch_matrix_type V, scratch_matrix_type T, scratch_matrix_type G_inv, scratch_vector_type manifold_coefficients, scratch_vector_type manifold_gradients, const int basis_multiplier_component) const {

	// only designed for 2D manifold embedded in 3D space
	const int target_index = teamMember.league_rank();
	const int target_NP = this->getNP(_poly_order, _dimensions-1);

	for (int i=0; i<_operations.size(); ++i) {
		if (_dimensions>2) {
			if (_operations(i) == ReconstructionOperator::TargetOperation::ScalarPointEvaluation) {
				Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
//					this->calcWij(P_target_row.data()+_lro_total_offsets[i]*target_NP, target_index, -1 /* target is neighbor */, 1 /*alpha*/, _dimensions-1, _poly_order, false /*bool on only specific order*/, &V, ReconstructionOperator::SamplingFunctional::PointSample);
					this->calcWij(t1.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, _dimensions-1, _poly_order, false /*bool on only specific order*/, &V, &T, ReconstructionOperator::SamplingFunctional::PointSample);
					int offset = _lro_total_offsets[i]*target_NP;
					for (int j=0; j<target_NP; ++j) {
						P_target_row(offset + j, basis_multiplier_component) = t1(j);
					}
				});
				teamMember.team_barrier();
			} else if (_operations(i) == ReconstructionOperator::TargetOperation::VectorPointEvaluation) {

				Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {

					// output component 0
					this->calcWij(t1.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, _dimensions-1, _poly_order, false /*bool on only specific order*/, &V, &T, ReconstructionOperator::SamplingFunctional::PointSample);
					int offset = (_lro_total_offsets[i]+0*_lro_output_tile_size[i]+0)*_basis_multiplier*target_NP;
					for (int j=0; j<target_NP; ++j) {
						P_target_row(offset + j, basis_multiplier_component) = t1(j)*T(0,0);
						P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
					}
					offset = (_lro_total_offsets[i]+1*_lro_output_tile_size[i]+0)*_basis_multiplier*target_NP;
					for (int j=0; j<target_NP; ++j) {
						P_target_row(offset + j, basis_multiplier_component) = 0;
						P_target_row(offset + target_NP + j, basis_multiplier_component) = t1(j)*T(0,1);
					}
					offset = (_lro_total_offsets[i]+2*_lro_output_tile_size[i]+0)*_basis_multiplier*target_NP;
					for (int j=0; j<target_NP; ++j) {
						P_target_row(offset + j, basis_multiplier_component) = 0;
						P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
					}

					// output component 1
					offset = (_lro_total_offsets[i]+0*_lro_output_tile_size[i]+1)*_basis_multiplier*target_NP;
					for (int j=0; j<target_NP; ++j) {
						P_target_row(offset + j, basis_multiplier_component) = t1(j)*T(1,0);
						P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
					}
					offset = (_lro_total_offsets[i]+1*_lro_output_tile_size[i]+1)*_basis_multiplier*target_NP;
					for (int j=0; j<target_NP; ++j) {
						P_target_row(offset + j, basis_multiplier_component) = 0;
						P_target_row(offset + target_NP + j, basis_multiplier_component) = t1(j)*T(1,1);
					}
					offset = (_lro_total_offsets[i]+2*_lro_output_tile_size[i]+1)*_basis_multiplier*target_NP;
					for (int j=0; j<target_NP; ++j) {
						P_target_row(offset + j, basis_multiplier_component) = 0;
						P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
					}

					// output component 2
					offset = (_lro_total_offsets[i]+0*_lro_output_tile_size[i]+2)*_basis_multiplier*target_NP;
					for (int j=0; j<target_NP; ++j) {
						P_target_row(offset + j, basis_multiplier_component) = t1(j)*T(2,0);
						P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
					}
					offset = (_lro_total_offsets[i]+1*_lro_output_tile_size[i]+2)*_basis_multiplier*target_NP;
					for (int j=0; j<target_NP; ++j) {
						P_target_row(offset + j, basis_multiplier_component) = 0;
						P_target_row(offset + target_NP + j, basis_multiplier_component) = t1(j)*T(2,1);
					}
					offset = (_lro_total_offsets[i]+2*_lro_output_tile_size[i]+2)*_basis_multiplier*target_NP;
					for (int j=0; j<target_NP; ++j) {
						P_target_row(offset + j, basis_multiplier_component) = 0;
						P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
					}
				});
				teamMember.team_barrier();
			} else if (_operations(i) == ReconstructionOperator::TargetOperation::LaplacianOfScalarPointEvaluation) {
				Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {

					double h = _epsilons(target_index);
					double a1, a2, a3, a4, a5;
					if (_manifold_poly_order > 0) {
						a1 = manifold_coefficients(1);
						a2 = manifold_coefficients(2);
					}
					if (_manifold_poly_order > 1) {
						a3 = manifold_coefficients(3);
						a4 = manifold_coefficients(4);
						a5 = manifold_coefficients(5);
					}
					double den = (h*h + a1*a1 + a2*a2);

					// Gaussian Curvature sanity check
//					double a1 = manifold_coefficients(1);
//					double a2 = manifold_coefficients(2);
//					double a3 = 0.5*manifold_coefficients(3);
//					double a4 = 1.0*manifold_coefficients(4);
//					double a5 = 0.5*manifold_coefficients(5);
//					double den = (h*h + a1*a1 + a2*a2);
//					double K_curvature = ( - a4*a4 + 4*a3*a5) / den2 / den2;
//					std::cout << "Gaussian curvature is: " << K_curvature << std::endl;


					const int offset = _lro_total_offsets[i]*target_NP;
					for (int j=0; j<target_NP; ++j) {
						P_target_row(offset + j, basis_multiplier_component) = 0;
					}
					// scaled
					if (_poly_order > 0 && _manifold_poly_order > 1) {
						P_target_row(offset + 1, basis_multiplier_component) = (-a1*((h*h+a2*a2)*a3 - 2*a1*a2*a4 + (h*h+a1*a1)*a5))/den/den/(h*h);
						P_target_row(offset + 2, basis_multiplier_component) = (-a2*((h*h+a2*a2)*a3 - 2*a1*a2*a4 + (h*h+a1*a1)*a5))/den/den/(h*h);
					}
					if (_poly_order > 1 && _manifold_poly_order > 0) {
						P_target_row(offset + 3, basis_multiplier_component) = (h*h+a2*a2)/den/(h*h);
						P_target_row(offset + 4, basis_multiplier_component) = -2*a1*a2/den/(h*h);
						P_target_row(offset + 5, basis_multiplier_component) = (h*h+a1*a1)/den/(h*h);
					}

				});
				teamMember.team_barrier();
			} else if (_operations(i) == ReconstructionOperator::TargetOperation::ChainedStaggeredLaplacianOfScalarPointEvaluation) {
				if (_reconstruction_space == ReconstructionOperator::ReconstructionSpace::VectorTaylorPolynomial) {
					Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {

						double h = _epsilons(target_index);
						double a1, a2, a3, a4, a5;
						if (_manifold_poly_order > 0) {
							a1 = manifold_coefficients(1);
							a2 = manifold_coefficients(2);
						}
						if (_manifold_poly_order > 1) {
							a3 = manifold_coefficients(3);
							a4 = manifold_coefficients(4);
							a5 = manifold_coefficients(5);
						}
						double den = (h*h + a1*a1 + a2*a2);

						double c0a = -a1*((h*h+a2*a2)*a3 - 2*a1*a2*a4 + (h*h+a1*a1)*a5)/den/den/h;
						double c1a = (h*h+a2*a2)/den/h;
						double c2a = -a1*a2/den/h;

						double c0b = -a2*((h*h+a2*a2)*a3 - 2*a1*a2*a4 + (h*h+a1*a1)*a5)/den/den/h;
						double c1b = -a1*a2/den/h;
						double c2b = (h*h+a1*a1)/den/h;

						// 1st input component
						int offset = (_lro_total_offsets[i]+0*_lro_output_tile_size[i]+0)*_basis_multiplier*target_NP;
						for (int j=0; j<target_NP; ++j) {
							P_target_row(offset + j, basis_multiplier_component) = 0;
							P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
						}
						P_target_row(offset + 0, basis_multiplier_component) = c0a;
						P_target_row(offset + 1, basis_multiplier_component) = c1a;
						P_target_row(offset + 2, basis_multiplier_component) = c2a;
						P_target_row(offset + target_NP + 0, basis_multiplier_component) = c0b;
						P_target_row(offset + target_NP + 1, basis_multiplier_component) = c1b;
						P_target_row(offset + target_NP + 2, basis_multiplier_component) = c2b;
					});
					teamMember.team_barrier();
				} else if (_reconstruction_space == ReconstructionOperator::ReconstructionSpace::ScalarTaylorPolynomial) {
					Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {

						double h = _epsilons(target_index);
						double a1, a2, a3, a4, a5;
						if (_manifold_poly_order > 0) {
							a1 = manifold_coefficients(1);
							a2 = manifold_coefficients(2);
						}
						if (_manifold_poly_order > 1) {
							a3 = manifold_coefficients(3);
							a4 = manifold_coefficients(4);
							a5 = manifold_coefficients(5);
						}
						double den = (h*h + a1*a1 + a2*a2);

						const int offset = _lro_total_offsets[i]*target_NP;
						for (int j=0; j<target_NP; ++j) {
							P_target_row(offset + j, basis_multiplier_component) = 0;
						}

						// verified
						if (_poly_order > 0 && _manifold_poly_order > 1) {
							P_target_row(offset + 1, basis_multiplier_component) = (-a1*((h*h+a2*a2)*a3 - 2*a1*a2*a4 + (h*h+a1*a1)*a5))/den/den/(h*h);
							P_target_row(offset + 2, basis_multiplier_component) = (-a2*((h*h+a2*a2)*a3 - 2*a1*a2*a4 + (h*h+a1*a1)*a5))/den/den/(h*h);
						}
						if (_poly_order > 1 && _manifold_poly_order > 0) {
							P_target_row(offset + 3, basis_multiplier_component) = (h*h+a2*a2)/den/(h*h);
							P_target_row(offset + 4, basis_multiplier_component) = -2*a1*a2/den/(h*h);
							P_target_row(offset + 5, basis_multiplier_component) = (h*h+a1*a1)/den/(h*h);
						}

					});
					teamMember.team_barrier();
				}
			} else if (_operations(i) == ReconstructionOperator::TargetOperation::LaplacianOfVectorPointEvaluation) {
				Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {

					double epsilons_target_index = _epsilons(target_index);

					double h = _epsilons(target_index);
					double a1, a2, a3, a4, a5;
					if (_manifold_poly_order > 0) {
						a1 = manifold_coefficients(1);
						a2 = manifold_coefficients(2);
					}
					if (_manifold_poly_order > 1) {
						a3 = manifold_coefficients(3);
						a4 = manifold_coefficients(4);
						a5 = manifold_coefficients(5);
					}
					double den = (h*h + a1*a1 + a2*a2);

					for (int j=0; j<target_NP; ++j) {
						t1(j) = 0;
					}

					// 1/Sqrt[Det[G[r, s]]])*Div[Sqrt[Det[G[r, s]]]*Inv[G]*P
					if (_poly_order > 0 && _manifold_poly_order > 1) {
						t1(1) = (-a1*((h*h+a2*a2)*a3 - 2*a1*a2*a4+(h*h+a1*a1)*a5))/den/den/(h*h);
						t1(2) = (-a2*((h*h+a2*a2)*a3 - 2*a1*a2*a4+(h*h+a1*a1)*a5))/den/den/(h*h);
					}
					if (_poly_order > 1 && _manifold_poly_order > 0) {
						t1(3) = (h*h+a2*a2)/den/(h*h);
						t1(4) = -2*a1*a2/den/(h*h);
						t1(5) = (h*h+a1*a1)/den/(h*h);
					}

					// output component 0
					int offset = (_lro_total_offsets[i]+0*_lro_output_tile_size[i]+0)*_basis_multiplier*target_NP;
					for (int j=0; j<target_NP; ++j) {
						P_target_row(offset + j, basis_multiplier_component) = t1(j)*T(0,0);
						P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
					}
					offset = (_lro_total_offsets[i]+1*_lro_output_tile_size[i]+0)*_basis_multiplier*target_NP;
					for (int j=0; j<target_NP; ++j) {
						P_target_row(offset + j, basis_multiplier_component) = 0;
						P_target_row(offset + target_NP + j, basis_multiplier_component) = t1(j)*T(0,1);
					}
					offset = (_lro_total_offsets[i]+2*_lro_output_tile_size[i]+0)*_basis_multiplier*target_NP;
					for (int j=0; j<target_NP; ++j) {
						P_target_row(offset + j, basis_multiplier_component) = 0;
						P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
					}

					// output component 1
					offset = (_lro_total_offsets[i]+0*_lro_output_tile_size[i]+1)*_basis_multiplier*target_NP;
					for (int j=0; j<target_NP; ++j) {
						P_target_row(offset + j, basis_multiplier_component) = t1(j)*T(1,0);
						P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
					}
					offset = (_lro_total_offsets[i]+1*_lro_output_tile_size[i]+1)*_basis_multiplier*target_NP;
					for (int j=0; j<target_NP; ++j) {
						P_target_row(offset + j, basis_multiplier_component) = 0;
						P_target_row(offset + target_NP + j, basis_multiplier_component) = t1(j)*T(1,1);
					}
					offset = (_lro_total_offsets[i]+2*_lro_output_tile_size[i]+1)*_basis_multiplier*target_NP;
					for (int j=0; j<target_NP; ++j) {
						P_target_row(offset + j, basis_multiplier_component) = 0;
						P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
					}

					// output component 2
					offset = (_lro_total_offsets[i]+0*_lro_output_tile_size[i]+2)*_basis_multiplier*target_NP;
					for (int j=0; j<target_NP; ++j) {
						P_target_row(offset + j, basis_multiplier_component) = t1(j)*T(2,0);
						P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
					}
					offset = (_lro_total_offsets[i]+1*_lro_output_tile_size[i]+2)*_basis_multiplier*target_NP;
					for (int j=0; j<target_NP; ++j) {
						P_target_row(offset + j, basis_multiplier_component) = 0;
						P_target_row(offset + target_NP + j, basis_multiplier_component) = t1(j)*T(2,1);
					}
					offset = (_lro_total_offsets[i]+2*_lro_output_tile_size[i]+2)*_basis_multiplier*target_NP;
					for (int j=0; j<target_NP; ++j) {
						P_target_row(offset + j, basis_multiplier_component) = 0;
						P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
					}

				});
				teamMember.team_barrier();
			} else if (_operations(i) == ReconstructionOperator::TargetOperation::GradientOfScalarPointEvaluation) {
				if (_polynomial_sampling_functional==ReconstructionOperator::SamplingFunctional::StaggeredEdgeIntegralSample) {
					Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
						this->calcWij(t1.data(), target_index, -1 /* target is neighbor */, 1 /*alpha*/, _dimensions-1, _poly_order, false /*bool on only specific order*/, &V, &T, ReconstructionOperator::SamplingFunctional::PointSample);
					});
					teamMember.team_barrier();
					int offset;
					Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
						for (int j=0; j<target_NP; ++j) {
							double t1a = G_inv(0,0);
							double t2a = G_inv(0,1);
							double t3a = G_inv(0,0)*manifold_gradients(0) + G_inv(0,1)*manifold_gradients(1);

							double t1b = G_inv(0,1);
							double t2b = G_inv(1,1);
							double t3b = G_inv(0,1)*manifold_gradients(0) + G_inv(1,1)*manifold_gradients(1);

							double v1c = V(0,0)*t1a + V(0,1)*t2a + V(0,2)*t3a;
							double v2c = V(1,0)*t1a + V(1,1)*t2a + V(1,2)*t3a;
							double v3c = V(2,0)*t1a + V(2,1)*t2a + V(2,2)*t3a;

							double v1d = V(0,0)*t1b + V(0,1)*t2b + V(0,2)*t3b;
							double v2d = V(1,0)*t1b + V(1,1)*t2b + V(1,2)*t3b;
							double v3d = V(2,0)*t1b + V(2,1)*t2b + V(2,2)*t3b;

							offset = (_lro_total_offsets[i]+0)*_basis_multiplier*target_NP;
							P_target_row(offset + j, basis_multiplier_component) = v1c*t1(j);
							P_target_row(offset + target_NP + j, basis_multiplier_component) = v1d*t1(j);

							offset = (_lro_total_offsets[i]+1)*_basis_multiplier*target_NP;
							P_target_row(offset + j, basis_multiplier_component) = v2c*t1(j);
							P_target_row(offset + target_NP + j, basis_multiplier_component) = v2d*t1(j);

							offset = (_lro_total_offsets[i]+2)*_basis_multiplier*target_NP;
							P_target_row(offset + j, basis_multiplier_component) = v3c*t1(j);
							P_target_row(offset + target_NP + j, basis_multiplier_component) = v3d*t1(j);
						}
					});
					teamMember.team_barrier();
				} else {
					Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {

						double h = _epsilons(target_index);
						double a1 = manifold_coefficients(1);
						double a2 = manifold_coefficients(2);

						double q1 = (h*h + a2*a2)/(h*h*h + h*a1*a1 + h*a2*a2);
						double q2 = -(a1*a2)/(h*h*h + h*a1*a1 + h*a2*a2);
						double q3 = (h*h + a1*a1)/(h*h*h + h*a1*a1 + h*a2*a2);

						double t1a = q1*1 + q2*0;
						double t2a = q1*0 + q2*1;
						double t3a = q1*manifold_gradients(0) + q2*manifold_gradients(1);

						double t1b = q2*1 + q3*0;
						double t2b = q2*0 + q3*1;
						double t3b = q2*manifold_gradients(0) + q3*manifold_gradients(1);

						// scaled
						int offset = (_lro_total_offsets[i]+0)*target_NP;
						for (int j=0; j<target_NP; ++j) {
							P_target_row(offset + j, basis_multiplier_component) = 0;
						}
						if (_poly_order > 0 && _manifold_poly_order > 0) {
							P_target_row(offset + 1, basis_multiplier_component) = t1a*V(0,0) + t2a*V(0,1) + t3a*V(0,2);
							P_target_row(offset + 2, basis_multiplier_component) = t1b*V(0,0) + t2b*V(0,1) + t3b*V(0,2);
						}

						offset = (_lro_total_offsets[i]+1)*target_NP;
						for (int j=0; j<target_NP; ++j) {
							P_target_row(offset + j, basis_multiplier_component) = 0;
						}
						if (_poly_order > 0 && _manifold_poly_order > 0) {
							P_target_row(offset + 1, basis_multiplier_component) = t1a*V(1,0) + t2a*V(1,1) + t3a*V(1,2);
							P_target_row(offset + 2, basis_multiplier_component) = t1b*V(1,0) + t2b*V(1,1) + t3b*V(1,2);
						}

						offset = (_lro_total_offsets[i]+2)*target_NP;
						for (int j=0; j<target_NP; ++j) {
							P_target_row(offset + j, basis_multiplier_component) = 0;
						}
						if (_poly_order > 0 && _manifold_poly_order > 0) {
							P_target_row(offset + 1, basis_multiplier_component) = t1a*V(2,0) + t2a*V(2,1) + t3a*V(2,2);
							P_target_row(offset + 2, basis_multiplier_component) = t1b*V(2,0) + t2b*V(2,1) + t3b*V(2,2);
						}

					});
					teamMember.team_barrier();
				}
			} else if (_operations(i) == ReconstructionOperator::TargetOperation::PartialXOfScalarPointEvaluation) {
				Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
					int offset = _lro_total_offsets[i]*target_NP;
					for (int j=0; j<target_NP; ++j) {
						P_target_row(offset + j, basis_multiplier_component) = 0;
						t1(j) = (j == 1) ? std::pow(_epsilons(target_index), -1) : 0;
						t2(j) = (j == 2) ? std::pow(_epsilons(target_index), -1) : 0;
					}
					for (int j=0; j<target_NP; ++j) {
						double v1 =	t1(j)*G_inv(0,0) + t2(j)*G_inv(1,0);
						double v2 =	t1(j)*G_inv(0,1) + t2(j)*G_inv(1,1);

						double t1 = v1*1 + v2*0;
						double t2 = v1*0 + v2*1;
						double t3 = v1*manifold_gradients(0) + v2*manifold_gradients(1);

						P_target_row(offset + j, basis_multiplier_component) = t1*V(0,0) + t2*V(0,1) + t3*V(0,2);
					}
				});
				teamMember.team_barrier();
			} else if (_operations(i) == ReconstructionOperator::TargetOperation::PartialYOfScalarPointEvaluation) {
				Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
					int offset = _lro_total_offsets[i]*target_NP;
					for (int j=0; j<target_NP; ++j) {
						P_target_row(offset + j, basis_multiplier_component) = 0;
						t1(j) = (j == 1) ? std::pow(_epsilons(target_index), -1) : 0;
						t2(j) = (j == 2) ? std::pow(_epsilons(target_index), -1) : 0;
					}

					for (int j=0; j<target_NP; ++j) {
						double v1 =	t1(j)*G_inv(0,0) + t2(j)*G_inv(1,0);
						double v2 =	t1(j)*G_inv(0,1) + t2(j)*G_inv(1,1);

						double t1 = v1*1 + v2*0;
						double t2 = v1*0 + v2*1;
						double t3 = v1*manifold_gradients(0) + v2*manifold_gradients(1);

						P_target_row(offset + j, basis_multiplier_component) = t1*V(1,0) + t2*V(1,1) + t3*V(1,2);
					}
				});
				teamMember.team_barrier();
			} else if (_operations(i) == ReconstructionOperator::TargetOperation::PartialZOfScalarPointEvaluation) {
				Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
					int offset = _lro_total_offsets[i]*target_NP;
					for (int j=0; j<target_NP; ++j) {
						P_target_row(offset + j, basis_multiplier_component) = 0;
						t1(j) = (j == 1) ? std::pow(_epsilons(target_index), -1) : 0;
						t2(j) = (j == 2) ? std::pow(_epsilons(target_index), -1) : 0;
					}

					for (int j=0; j<target_NP; ++j) {
						double v1 =	t1(j)*G_inv(0,0) + t2(j)*G_inv(1,0);
						double v2 =	t1(j)*G_inv(0,1) + t2(j)*G_inv(1,1);

						double t1 = v1*1 + v2*0;
						double t2 = v1*0 + v2*1;
						double t3 = v1*manifold_gradients(0) + v2*manifold_gradients(1);

						P_target_row(offset + j, basis_multiplier_component) = t1*V(2,0) + t2*V(2,1) + t3*V(2,2);
					}
				});
				teamMember.team_barrier();
			} else if (_operations(i) == ReconstructionOperator::TargetOperation::DivergenceOfVectorPointEvaluation) {
				if (_data_sampling_functional==ReconstructionOperator::SamplingFunctional::StaggeredEdgeIntegralSample) {
					ASSERT_WITH_MESSAGE(false, "Functionality not yet available.");
				} else if (_data_sampling_functional==ReconstructionOperator::SamplingFunctional::ManifoldGradientVectorSample) {
					Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {

						double h = _epsilons(target_index);
						double a1, a2, a3, a4, a5;
						if (_manifold_poly_order > 0) {
							a1 = manifold_coefficients(1);
							a2 = manifold_coefficients(2);
						}
						if (_manifold_poly_order > 1) {
							a3 = manifold_coefficients(3);
							a4 = manifold_coefficients(4);
							a5 = manifold_coefficients(5);
						}
						double den = (h*h + a1*a1 + a2*a2);

						// 1/Sqrt[Det[G[r, s]]])*Div[Sqrt[Det[G[r, s]]]*P
						// i.e. P recovers G^{-1}*grad of scalar
						double c0a = (a1*a3+a2*a4)/(h*den);
						double c1a = 1./h;
						double c2a = 0;

						double c0b = (a1*a4+a2*a5)/(h*den);
						double c1b = 0;
						double c2b = 1./h;

						// 1st input component
						int offset = (_lro_total_offsets[i]+0*_lro_output_tile_size[i]+0)*_basis_multiplier*target_NP;
						for (int j=0; j<target_NP; ++j) {
							P_target_row(offset + j, basis_multiplier_component) = 0;
							P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
						}
						P_target_row(offset + 0, basis_multiplier_component) = c0a;
						P_target_row(offset + 1, basis_multiplier_component) = c1a;
						P_target_row(offset + 2, basis_multiplier_component) = c2a;

						// 2nd input component
						offset = (_lro_total_offsets[i]+1*_lro_output_tile_size[i]+0)*_basis_multiplier*target_NP;
						for (int j=0; j<target_NP; ++j) {
							P_target_row(offset + j, basis_multiplier_component) = 0;
							P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
						}
						P_target_row(offset + target_NP + 0, basis_multiplier_component) = c0b;
						P_target_row(offset + target_NP + 1, basis_multiplier_component) = c1b;
						P_target_row(offset + target_NP + 2, basis_multiplier_component) = c2b;

						// 3rd input component (no contribution)
						offset = (_lro_total_offsets[i]+2*_lro_output_tile_size[i]+0)*_basis_multiplier*target_NP;
						for (int j=0; j<target_NP; ++j) {
							P_target_row(offset + j, basis_multiplier_component) = 0;
							P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
						}
					});
					teamMember.team_barrier();
				}
			} else if (_operations(i) == ReconstructionOperator::TargetOperation::DivergenceOfScalarPointEvaluation) {
				if (_data_sampling_functional==ReconstructionOperator::SamplingFunctional::StaggeredEdgeAnalyticGradientIntegralSample) {
					Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {

						double h = _epsilons(target_index);
						double a1, a2, a3, a4, a5;
						if (_manifold_poly_order > 0) {
							a1 = manifold_coefficients(1);
							a2 = manifold_coefficients(2);
						}
						if (_manifold_poly_order > 1) {
							a3 = manifold_coefficients(3);
							a4 = manifold_coefficients(4);
							a5 = manifold_coefficients(5);
						}
						double den = (h*h + a1*a1 + a2*a2);

						// 1/Sqrt[Det[G[r, s]]])*Div[Sqrt[Det[G[r, s]]]*Inv[G].P
						// i.e. P recovers grad of scalar
						double c0a = -a1*((h*h+a2*a2)*a3 - 2*a1*a2*a4 + (h*h+a1*a1)*a5)/den/den/h;
						double c1a = (h*h+a2*a2)/den/h;
						double c2a = -a1*a2/den/h;

						double c0b = -a2*((h*h+a2*a2)*a3 - 2*a1*a2*a4 + (h*h+a1*a1)*a5)/den/den/h;
						double c1b = -a1*a2/den/h;
						double c2b = (h*h+a1*a1)/den/h;

						// 1st input component
						int offset = (_lro_total_offsets[i]+0*_lro_output_tile_size[i]+0)*_basis_multiplier*target_NP;
						for (int j=0; j<target_NP; ++j) {
							P_target_row(offset + j, basis_multiplier_component) = 0;
							P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
						}
						P_target_row(offset + 0, basis_multiplier_component) = c0a;
						P_target_row(offset + 1, basis_multiplier_component) = c1a;
						P_target_row(offset + 2, basis_multiplier_component) = c2a;
						P_target_row(offset + target_NP + 0, basis_multiplier_component) = c0b;
						P_target_row(offset + target_NP + 1, basis_multiplier_component) = c1b;
						P_target_row(offset + target_NP + 2, basis_multiplier_component) = c2b;

					});
					teamMember.team_barrier();
				} else if (_data_sampling_functional==ReconstructionOperator::SamplingFunctional::StaggeredEdgeIntegralSample) {
					Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {

						double h = _epsilons(target_index);
						double a1, a2, a3, a4, a5;
						if (_manifold_poly_order > 0) {
							a1 = manifold_coefficients(1);
							a2 = manifold_coefficients(2);
						}
						if (_manifold_poly_order > 1) {
							a3 = manifold_coefficients(3);
							a4 = manifold_coefficients(4);
							a5 = manifold_coefficients(5);
						}
						double den = (h*h + a1*a1 + a2*a2);

						// 1/Sqrt[Det[G[r, s]]])*Div[Sqrt[Det[G[r, s]]]*.P
						// i.e. P recovers G^{-1} * grad of scalar
						double c0a = (a1*a3+a2*a4)/(h*den);
						double c1a = 1./h;
						double c2a = 0;

						double c0b = (a1*a4+a2*a5)/(h*den);
						double c1b = 0;
						double c2b = 1./h;

						// 1st input component
						int offset = (_lro_total_offsets[i]+0*_lro_output_tile_size[i]+0)*_basis_multiplier*target_NP;
						for (int j=0; j<target_NP; ++j) {
							P_target_row(offset + j, basis_multiplier_component) = 0;
							P_target_row(offset + target_NP + j, basis_multiplier_component) = 0;
						}
						P_target_row(offset + 0, basis_multiplier_component) = c0a;
						P_target_row(offset + 1, basis_multiplier_component) = c1a;
						P_target_row(offset + 2, basis_multiplier_component) = c2a;
						P_target_row(offset + target_NP + 0, basis_multiplier_component) = c0b;
						P_target_row(offset + target_NP + 1, basis_multiplier_component) = c1b;
						P_target_row(offset + target_NP + 2, basis_multiplier_component) = c2b;

					});
					teamMember.team_barrier();
				}
			}
			else {
				ASSERT_WITH_MESSAGE(false, "Functionality not yet available.");
			}
		} else if (_dimensions==2) { // 1D manifold in 2D problem
			if (_operations(i) == ReconstructionOperator::TargetOperation::GradientOfScalarPointEvaluation) {
				Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
					int offset = (_lro_total_offsets[i]+0)*target_NP;
					for (int j=0; j<target_NP; ++j) {
						P_target_row(offset + j, basis_multiplier_component) = 0;
						t1(j) = (j == 1) ? std::pow(_epsilons(target_index), -1) : 0;
					}
					for (int j=0; j<target_NP; ++j) {
						double v1 =	t1(j)*G_inv(0,0);

						double t1 = v1*1;
						double t2 = v1*manifold_gradients(0);

						P_target_row(offset + j, basis_multiplier_component) = t1*V(0,0) + t2*V(0,1);
					}

					offset = (_lro_total_offsets[i]+1)*target_NP;
					for (int j=0; j<target_NP; ++j) {
						P_target_row(offset + j, basis_multiplier_component) = 0;
						t1(j) = (j == 1) ? std::pow(_epsilons(target_index), -1) : 0;
					}

					for (int j=0; j<target_NP; ++j) {
						double v1 =	t1(j)*G_inv(0,0);

						double t1 = v1*1;
						double t2 = v1*manifold_gradients(0);

						P_target_row(offset + j, basis_multiplier_component) = t1*V(1,0) + t2*V(1,1);
					}
				});
				teamMember.team_barrier();
			} else if (_operations(i) == ReconstructionOperator::TargetOperation::PartialXOfScalarPointEvaluation) {
				Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
					int offset = _lro_total_offsets[i]*target_NP;
					for (int j=0; j<target_NP; ++j) {
						P_target_row(offset + j, basis_multiplier_component) = 0;
						t1(j) = (j == 1) ? std::pow(_epsilons(target_index), -1) : 0;
					}
					for (int j=0; j<target_NP; ++j) {
						double v1 =	t1(j)*G_inv(0,0);

						double t1 = v1*1;
						double t2 = v1*manifold_gradients(0);

						P_target_row(offset + j, basis_multiplier_component) = t1*V(0,0) + t2*V(0,1);
					}
				});
				teamMember.team_barrier();
			} else if (_operations(i) == ReconstructionOperator::TargetOperation::PartialYOfScalarPointEvaluation) {
				Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
					int offset = _lro_total_offsets[i]*target_NP;
					for (int j=0; j<target_NP; ++j) {
						P_target_row(offset + j, basis_multiplier_component) = 0;
						t1(j) = (j == 1) ? std::pow(_epsilons(target_index), -1) : 0;
					}

					for (int j=0; j<target_NP; ++j) {
						double v1 =	t1(j)*G_inv(0,0);

						double t1 = v1*1;
						double t2 = v1*manifold_gradients(0);

						P_target_row(offset + j, basis_multiplier_component) = t1*V(1,0) + t2*V(1,1);
					}
				});
				teamMember.team_barrier();
			} else {
				ASSERT_WITH_MESSAGE(false, "Functionality not yet available.");
			}
		}
	}
}

void GMLS::printNeighbors(const int target_index) const {
    int i;
    fprintf(stdout, "%i neighbors: ", target_index);
    
    for(i = 0; i < this->getNNeighbors(target_index); i++){
    	fprintf(stdout, ", %i ", this->getNeighborIndex(target_index, i));
	}
    fprintf(stdout, "\n");
}

void GMLS::printNeighborData(const int target_index, const int dimensions) const {
    
    fprintf(stdout, "my location: ");
    for(int i = 0; i < dimensions; i++){
        fprintf(stdout, " %f ", this->getTargetCoordinate(target_index, i));
    }
    fprintf(stdout, "\n");
    
    for(int i = 0; i < this->getNNeighbors(target_index); i++){
      fprintf(stdout, "%d: %d, ", i, this->getNeighborIndex(target_index,i));
		for(int j = 0; j < dimensions; j++){
		  fprintf(stdout, "%f ", this->getNeighborCoordinate(target_index, i, j));
		}
		fprintf(stdout, "\n");
    }
}

KOKKOS_INLINE_FUNCTION
void GMLS::operator()(const member_type& teamMember) const {

	const int target_index = teamMember.league_rank();
	if (_data_sampling_functional == ReconstructionOperator::SamplingFunctional::StaggeredEdgeAnalyticGradientIntegralSample) {

		double operator_coefficient = 1;
		if (_operator_coefficients.dimension_0() > 0) {
			operator_coefficient = _operator_coefficients(this->getNeighborIndex(target_index,0),0);
		}
		_prestencil_weights(target_index, 0) = -operator_coefficient;
		_prestencil_weights(target_index, 1) =  operator_coefficient;

		for (int i=1; i<this->getNNeighbors(target_index); ++i) {

			if (_operator_coefficients.dimension_0() > 0) {
				operator_coefficient = 0.5*(_operator_coefficients(this->getNeighborIndex(target_index,0),0) + _operator_coefficients(this->getNeighborIndex(target_index,i),0));
			}
			_prestencil_weights(target_index, 2*i  ) = -operator_coefficient;
			_prestencil_weights(target_index, 2*i+1) =  operator_coefficient;

		}
	}

	const int max_neighbors_or_basis = ((_neighbor_lists.dimension_1()-1)*_sampling_multiplier > _NP*_basis_multiplier)
			? (_neighbor_lists.dimension_1()-1)*_sampling_multiplier : _NP*_basis_multiplier;

	if (_dense_solver_type == ReconstructionOperator::DenseSolverType::QR) {

		// team_scratch all have a copy each per team
		// the threads in a team work on these local copies that only the team sees
		// thread_scratch has a copy per thread

		scratch_matrix_type PsqrtW(teamMember.team_scratch(_scratch_team_level_b), max_neighbors_or_basis, max_neighbors_or_basis); // full, only _neighbor_lists.dimension_1()-1x_NP needed, but must be transposed
		scratch_matrix_type Q(teamMember.team_scratch(_scratch_team_level_b), _neighbor_lists.dimension_1()-1, _neighbor_lists.dimension_1()-1); // Q triangular
		scratch_vector_type t1(teamMember.team_scratch(_scratch_team_level_a), _neighbor_lists.dimension_1()-1);
		scratch_vector_type t2(teamMember.team_scratch(_scratch_team_level_a), _neighbor_lists.dimension_1()-1);
#ifdef COMPADRE_USE_LAPACK
		// find optimal blocksize when using LAPACK in order to allocate workspace needed
		int ipsec = 1, unused = -1, bnp = _basis_multiplier*_NP, lwork = -1, info = 0; double wkopt = 0;
		int mnob = max_neighbors_or_basis;
		dormqr_( (char *)"L", (char *)"T", &mnob, &mnob, 
                	&bnp, (double *)NULL, &mnob, (double *)NULL, 
                	(double *)NULL, &mnob, &wkopt, &lwork, &info);
		int lapack_opt_blocksize = (int)wkopt;
		scratch_vector_type t3(teamMember.team_scratch(_scratch_team_level_b), lapack_opt_blocksize);
#else
		scratch_vector_type t3(teamMember.team_scratch(_scratch_team_level_b), (_neighbor_lists.dimension_1()-1));
#endif
		scratch_vector_type w(teamMember.team_scratch(_scratch_team_level_b), _neighbor_lists.dimension_1()-1);

		// delta is a temporary variable that preallocates space on which solutions of size _NP will
		// be computed. This allows the gpu to allocate the memory in advance
		scratch_vector_type delta(teamMember.thread_scratch(_scratch_thread_level_b), _basis_multiplier*_NP);

		// creates the matrix sqrt(W)*P
		this->createWeightsAndP(teamMember, delta, PsqrtW, w, _dimensions, _poly_order, true /*weight_p*/, NULL /*&V*/, NULL /*&T*/, _polynomial_sampling_functional);

		// GivensQR is faster than Householder on GPU and CPU
		GMLS_LinearAlgebra::GivensQR(teamMember, t1, t2, t3, Q, PsqrtW, _NP, this->getNNeighbors(target_index));

		teamMember.team_barrier();

		scratch_matrix_type P_target_row(teamMember.team_scratch(_scratch_team_level_b), _NP*_total_alpha_values, _sampling_multiplier);
		this->computeTargetFunctionals(teamMember, t1, t2, P_target_row);

		this->applyQR(teamMember, t1, t2, Q, PsqrtW, w, P_target_row, _NP); 

	} else if (_dense_solver_type == ReconstructionOperator::DenseSolverType::LU) {

		// M_data, M_inv, and weight_P all have a copy each per team
		// the threads in a team work on these local copies that only the team sees
		scratch_matrix_type PsqrtW(teamMember.team_scratch(_scratch_team_level_b), max_neighbors_or_basis, max_neighbors_or_basis); // full
		scratch_matrix_type M_data(teamMember.team_scratch(_scratch_team_level_b), _NP, _NP); // lower triangular
		scratch_matrix_type M_inv(teamMember.team_scratch(_scratch_team_level_b), _NP, _NP); // lower triangular
		scratch_matrix_type L(teamMember.team_scratch(_scratch_team_level_b), _NP, _NP); // L
		scratch_vector_type w(teamMember.team_scratch(_scratch_team_level_b), _neighbor_lists.dimension_1()-1);
		scratch_vector_type t1(teamMember.team_scratch(_scratch_team_level_a), _neighbor_lists.dimension_1()-1);
		scratch_vector_type t2(teamMember.team_scratch(_scratch_team_level_a), _neighbor_lists.dimension_1()-1);

		// delta is a temporary variable that preallocates space on which solutions of size _NP will
		// be computed. This allows the gpu to allocate the memory in advance
		scratch_vector_type delta(teamMember.thread_scratch(_scratch_thread_level_b), _basis_multiplier*_NP);

		scratch_matrix_type P_target_row(teamMember.team_scratch(_scratch_team_level_b), _NP*_total_alpha_values, _sampling_multiplier);
		scratch_matrix_type b_data(teamMember.team_scratch(_scratch_team_level_a), _NP, _sampling_multiplier);

		// creates the matrix sqrt(W)*P
		this->createWeightsAndP(teamMember, delta, PsqrtW, w, _dimensions, _poly_order, true /*weight_p*/, NULL /*&V*/, NULL /*&T*/, _polynomial_sampling_functional);

		// creates the matrix P^T*W*P
		GMLS_LinearAlgebra::createM(teamMember, M_data, PsqrtW, _NP, this->getNNeighbors(target_index));

		// inverts M against the identity
		GMLS_LinearAlgebra::invertM(teamMember, delta, M_inv, L, M_data, _NP);

		this->computeTargetFunctionals(teamMember, t1, t2, P_target_row);

		this->applyMInverse(teamMember, b_data, M_inv, PsqrtW, w, P_target_row, _NP);

	} else if (_dense_solver_type == ReconstructionOperator::DenseSolverType::MANIFOLD) {

		// team_scratch all have a copy each per team
		// the threads in a team work on these local copies that only the team sees
		// thread_scratch has a copy per thread
		const int manifold_NP = this->getNP(_manifold_poly_order, _dimensions-1);
		const int target_NP = this->getNP(_poly_order, _dimensions-1);

		const int max_manifold_NP = (manifold_NP > target_NP) ? manifold_NP : target_NP;
		const int max_NP = (max_manifold_NP > _NP) ? max_manifold_NP : _NP;
		const int max_P_row_size = ((_dimensions-1)*manifold_NP > max_NP*_total_alpha_values*_basis_multiplier) ? (_dimensions-1)*manifold_NP : max_NP*_total_alpha_values*_basis_multiplier;

		const int max_neighbors_or_basis_manifold = ((_neighbor_lists.dimension_1()-1)*_sampling_multiplier > max_NP*_basis_multiplier)
				? (_neighbor_lists.dimension_1()-1)*_sampling_multiplier : max_NP*_basis_multiplier;

		scratch_matrix_type PsqrtW(teamMember.team_scratch(_scratch_team_level_b), max_neighbors_or_basis_manifold, max_neighbors_or_basis_manifold); // must be square so that transpose is possible

		scratch_matrix_type Q(teamMember.team_scratch(_scratch_team_level_b), (_neighbor_lists.dimension_1()-1)*_sampling_multiplier, (_neighbor_lists.dimension_1()-1)*_sampling_multiplier);
		scratch_matrix_type V(teamMember.team_scratch(_scratch_team_level_b), _dimensions, _dimensions);
		scratch_matrix_type T(teamMember.team_scratch(_scratch_team_level_b), _dimensions, _dimensions-1);

		scratch_matrix_type G(teamMember.team_scratch(_scratch_team_level_b), _dimensions-1, _dimensions-1);
		scratch_matrix_type G_inv(teamMember.team_scratch(_scratch_team_level_b), _dimensions-1, _dimensions-1);
		scratch_matrix_type PTP(teamMember.team_scratch(_scratch_team_level_b), _dimensions, _dimensions);

		scratch_vector_type t1(teamMember.team_scratch(_scratch_team_level_b), (_neighbor_lists.dimension_1()-1)*((_sampling_multiplier>_basis_multiplier) ? _sampling_multiplier : _basis_multiplier));
		scratch_vector_type t2(teamMember.team_scratch(_scratch_team_level_b), (_neighbor_lists.dimension_1()-1)*((_sampling_multiplier>_basis_multiplier) ? _sampling_multiplier : _basis_multiplier));

#ifdef COMPADRE_USE_LAPACK
		// find optimal blocksize when using LAPACK in order to allocate workspace needed
		int ipsec = 1, unused = -1, bnp = _basis_multiplier*max_NP, lwork = -1, info = 0; double wkopt = 0;
		int mnob = max_neighbors_or_basis_manifold;
		dormqr_( (char *)"L", (char *)"T", &mnob, &mnob, 
                	&bnp, (double *)NULL, &mnob, (double *)NULL, 
                	(double *)NULL, &mnob, &wkopt, &lwork, &info);
		int lapack_opt_blocksize = (int)wkopt;
		scratch_vector_type t3(teamMember.team_scratch(_scratch_team_level_b), lapack_opt_blocksize);
#else
		scratch_vector_type t3(teamMember.team_scratch(_scratch_team_level_b), (_neighbor_lists.dimension_1()-1));
#endif

		scratch_vector_type manifold_gradient(teamMember.team_scratch(_scratch_team_level_b), (_dimensions-1)*(_neighbor_lists.dimension_1()-1));
		scratch_vector_type manifold_coeffs(teamMember.team_scratch(_scratch_team_level_b), manifold_NP);

		scratch_vector_type w(teamMember.team_scratch(_scratch_team_level_b), (_neighbor_lists.dimension_1()-1)*_sampling_multiplier);
		scratch_matrix_type P_target_row(teamMember.team_scratch(_scratch_team_level_b), max_P_row_size, _sampling_multiplier);

		scratch_vector_type manifold_gradient_coeffs(teamMember.team_scratch(_scratch_team_level_b), (_dimensions-1));

		// delta is a temporary variable that preallocates space on which solutions of size _NP will
		// be computed. This allows the gpu to allocate the memory in advance
		scratch_vector_type delta(teamMember.thread_scratch(_scratch_thread_level_b), max_NP*_basis_multiplier);

		scratch_vector_type S(teamMember.team_scratch(_scratch_team_level_b), max_manifold_NP*_basis_multiplier);
		scratch_matrix_type b_data(teamMember.team_scratch(_scratch_team_level_b), max_manifold_NP*_basis_multiplier, _sampling_multiplier);
		scratch_matrix_type Vsvd(teamMember.team_scratch(_scratch_team_level_b), max_manifold_NP*_basis_multiplier, max_manifold_NP*_basis_multiplier);

		scratch_matrix_type quadrature_manifold_gradients(teamMember.team_scratch(_scratch_team_level_b), (_neighbor_lists.dimension_1()-1)*(_dimensions-1), _number_of_quadrature_points); // for integral sampling

//		double* u_raw = Q.data();
//		const size_t N = Vsvd.extent_0();
//		std::cout << u_raw << std::endl;
//		double* v_raw = Vsvd.data();
//		const size_t N = Vsvd.extent_0();
//		std::cout << v_raw << std::endl;
//		printf("%f\n", Vsvd(0,0));

		//
		//  DETERMINE TANGENT PLANE TO MANIFOLD
		//

		// getting x y and z from which to derive a manifold
		this->createWeightsAndPForCurvature(teamMember, delta, PsqrtW, w, _dimensions, true /* only specific order */);

		bool use_SVD_for_manifold = false;

		if (use_SVD_for_manifold) {

			GMLS_LinearAlgebra::GolubReinschSVD(teamMember, t1, t2, PsqrtW, Q, S, V, _dimensions /* custom # of columns*/, this->getNNeighbors(target_index));
			// from these 3 SVD values, find the two largest and rearrange the V matrix to use
			// these as the first two columns
			// get max val

			// rearrange so that largest singular value is placed in last column of matrix V

			double min_val = DBL_MAX;
			int min_ind;
			for (int i=0; i<_dimensions; ++i) {
				if (PsqrtW(i,i) < min_val) {
					min_val = S(i,i);
					min_ind = i;
				}
			}
			// check if normal vector is not last column of matrix
			if (min_ind!=_dimensions-1) {
				// min_ind is not last, but is the normal vector column
				for (int i=0; i<_dimensions; ++i) {
					// copy column min_ind to the last column and the last column to min_ind
					double tmp = V(i,min_ind);
					V(i,min_ind) = V(i,_dimensions-1);
					V(i,_dimensions-1) = tmp;
				}
			}

			// orthogonalize V (not necessary since SVD already does this)
			// GMLS_LinearAlgebra::orthogonalizeVectorBasis(teamMember, V);

		} else {

			GMLS_LinearAlgebra::createM(teamMember, PTP, PsqrtW, _dimensions /* # of columns */, this->getNNeighbors(target_index));
			GMLS_LinearAlgebra::largestTwoEigenvectorsThreeByThreeSymmetric(teamMember, V, PTP, _dimensions);

		}

//		Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
//			// DIAGNOSTIC for normal and tangent vectors on the sphere of radius 1
//			// overwrite vectors based on x,y,z of target
//
//			double unit_scaling = 1./std::sqrt(
//					this->getTargetCoordinate(target_index, 0)*this->getTargetCoordinate(target_index, 0)
//					+ this->getTargetCoordinate(target_index, 1)*this->getTargetCoordinate(target_index, 1)
//					+ this->getTargetCoordinate(target_index, 2)*this->getTargetCoordinate(target_index, 2));
//
//			V(0,2) = this->getTargetCoordinate(target_index, 0)*unit_scaling;
//			V(1,2) = this->getTargetCoordinate(target_index, 1)*unit_scaling;
//			V(2,2) = this->getTargetCoordinate(target_index, 2)*unit_scaling;
//
//			double den = 1./std::sqrt(1 - this->getTargetCoordinate(target_index, 2)*unit_scaling*this->getTargetCoordinate(target_index, 2)*unit_scaling);
//			V(0,1) = -this->getTargetCoordinate(target_index, 1)*unit_scaling/den;
//			V(1,1) = this->getTargetCoordinate(target_index, 0)*unit_scaling/den;
//			V(2,1) = 0;
//
//			V(0,0) = -this->getTargetCoordinate(target_index, 0)*unit_scaling*this->getTargetCoordinate(target_index, 2)*unit_scaling/den;
//			V(1,0) = -this->getTargetCoordinate(target_index, 1)*unit_scaling*this->getTargetCoordinate(target_index, 2)*unit_scaling/den;
//			V(2,0) = (1-this->getTargetCoordinate(target_index, 2)*unit_scaling*this->getTargetCoordinate(target_index, 2)*unit_scaling)/den;
//
//			// DIAGNOSTIC for normal and tangent vectors on the cylinder of radius 1
//			// overwrite vectors based on x,y,z of target
//
//			double unit_scaling = 1./std::sqrt(
//					this->getTargetCoordinate(target_index, 0)*this->getTargetCoordinate(target_index, 0)
//					+ this->getTargetCoordinate(target_index, 1)*this->getTargetCoordinate(target_index, 1));
//
//			V(0,2) = this->getTargetCoordinate(target_index, 0)*unit_scaling;
//			V(1,2) = this->getTargetCoordinate(target_index, 1)*unit_scaling;
//			V(2,2) = 0;
//
//			V(0,1) = -this->getTargetCoordinate(target_index, 1)*unit_scaling;
//			V(1,1) = this->getTargetCoordinate(target_index, 0)*unit_scaling;
//			V(2,1) = 0;
//
//			V(0,0) = 0;
//			V(1,0) = 0;
//			V(2,0) = 1;
//
//			double norm = std::sqrt(V(0,0)*V(0,0) + V(1,0)*V(1,0) + V(2,0)*V(2,0));
//			V(0,0) /= norm;
//			V(1,0) /= norm;
//			V(2,0) /= norm;
//
//			norm = std::sqrt(V(0,1)*V(0,1) + V(1,1)*V(1,1) + V(2,1)*V(2,1));
//			V(0,1) /= norm;
//			V(1,1) /= norm;
//			V(2,1) /= norm;
//
//			// DIAGNOSTIC that eigenvectors are orthonormal
//			printf("dot1: %.16f\n", V(0,2)*V(0,1)+V(1,2)*V(1,1)+V(2,2)*V(2,1));
//			printf("dot2: %.16f\n", V(0,2)*V(0,0)+V(1,2)*V(1,0)+V(2,2)*V(2,0));
//			printf("dot3: %.16f\n", V(0,1)*V(0,2)+V(1,1)*V(1,2)+V(2,1)*V(2,2));
//			printf("dot1a: %.16f\n", V(0,0)*V(0,0)+V(1,0)*V(1,0)+V(2,0)*V(2,0));
//			printf("dot2a: %.16f\n", V(0,1)*V(0,1)+V(1,1)*V(1,1)+V(2,1)*V(2,1));
//			printf("dot3a: %.16f\n", V(0,2)*V(0,2)+V(1,2)*V(1,2)+V(2,2)*V(2,2));
//		});
		teamMember.team_barrier();

		//
		//  RECONSTRUCT ON THE TANGENT PLANE USING LOCAL COORDINATES
		//

		// creates the matrix sqrt(W)*P
		this->createWeightsAndPForCurvature(teamMember, delta, PsqrtW, w, _dimensions-1, false /* only specific order */, &V);

		GMLS_LinearAlgebra::GivensQR(teamMember, t1, t2, t3, Q, PsqrtW, manifold_NP, this->getNNeighbors(target_index));
		teamMember.team_barrier();

		// gives GMLS coefficients for gradient using basis defined on reduced space
		GMLS_LinearAlgebra::upperTriangularBackSolve(teamMember, t1, t2, Q, PsqrtW, w, manifold_NP, this->getNNeighbors(target_index)); // stores solution in Q
		teamMember.team_barrier();

		//
		//  GET TARGET COEFFICIENTS RELATED TO GRADIENT TERMS
		//
		// reconstruct grad_xi1 and grad_xi2, not used for manifold_coeffs
		this->computeManifoldFunctionals(teamMember, t1, t2, P_target_row, &V, -1 /*target as neighbor index*/, 1 /*alpha*/);

		Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
			for (int j=0; j<manifold_NP; ++j) { // set to zero
				manifold_coeffs(j) = 0;
			}
		});

		double grad_xi1 = 0, grad_xi2 = 0;
		for (int i=0; i<this->getNNeighbors(target_index); ++i) {
			for (int k=0; k<_dimensions-1; ++k) {
				double alpha_ij = 0;
				Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember,
						manifold_NP), [=] (const int l, double &talpha_ij) {
					talpha_ij += P_target_row(manifold_NP*k+l,0)*Q(i,l);
				}, alpha_ij);
				Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
					manifold_gradient(i*(_dimensions-1) + k) = alpha_ij; // stored staggered, grad_xi1, grad_xi2, grad_xi1, grad_xi2, ....
				});
			}
			teamMember.team_barrier();

			XYZ rel_coord = getRelativeCoord(target_index, i, _dimensions, &V);
			double normal_coordinate = rel_coord[_dimensions-1];

			// apply coefficients to sample data
			grad_xi1 += manifold_gradient(i*(_dimensions-1)) * normal_coordinate;
			if (_dimensions>2) grad_xi2 += manifold_gradient(i*(_dimensions-1)+1) * normal_coordinate;

			Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
				// coefficients without a target premultiplied
				for (int j=0; j<manifold_NP; ++j) {
					manifold_coeffs(j) += Q(i,j) * normal_coordinate;
//					manifold_coeffs(j) += b_data(j,0) * normal_coordinate;
				}
			});
		}

		// Constructs high order orthonormal tangent space T and inverse of metric tensor
		Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {

			manifold_gradient_coeffs(0) = grad_xi1;
			if (_dimensions>2) manifold_gradient_coeffs(1) = grad_xi2;

			// Construct T (high order approximation of orthonormal tangent vectors)
			for (int i=0; i<_dimensions-1; ++i) {
				// build
				for (int j=0; j<_dimensions; ++j) {
					T(j,i) = manifold_gradient_coeffs(i)*V(j,_dimensions-1);
					T(j,i) += V(j,i);
				}
			}

			// calculate norm
			double norm = 0;
			for (int j=0; j<_dimensions; ++j) {
				norm += T(j,0)*T(j,0);
			}

			// normalize first vector
			norm = std::sqrt(norm);
			for (int j=0; j<_dimensions; ++j) {
				T(j,0) /= norm;
			}

			// orthogonalize next vector
			if (_dimensions-1 == 2) { // 2d manifold
				double dot_product = T(0,0)*T(0,1) + T(1,0)*T(1,1) + T(2,0)*T(2,1);
				for (int j=0; j<_dimensions; ++j) {
					T(j,1) -= dot_product*T(j,0);
				}
				// normalize second vector
				norm = 0;
				for (int j=0; j<_dimensions; ++j) {
					norm += T(j,1)*T(j,1);
				}
				norm = std::sqrt(norm);
				for (int j=0; j<_dimensions; ++j) {
					T(j,1) /= norm;
				}
			}

			// need to get 2x2 matrix of metric tensor
			G(0,0) = 1 + grad_xi1*grad_xi1;

			if (_dimensions>2) {
				G(0,1) = grad_xi1*grad_xi2;
				G(1,0) = grad_xi2*grad_xi1;
				G(1,1) = 1 + grad_xi2*grad_xi2;
			}

			double G_determinant;
			if (_dimensions==2) {
				G_determinant = G(0,0);
				ASSERT_WITH_MESSAGE(G_determinant!=0, "Determinant is zero.");
				G_inv(0,0) = 1/G_determinant;
			} else {
				G_determinant = G(0,0)*G(1,1) - G(0,1)*G(1,0); //std::sqrt(G_inv(0,0)*G_inv(1,1) - G_inv(0,1)*G_inv(1,0));
				ASSERT_WITH_MESSAGE(G_determinant!=0, "Determinant is zero.");
				{
					// inverse of 2x2
					G_inv(0,0) = G(1,1)/G_determinant;
					G_inv(1,1) = G(0,0)/G_determinant;
					G_inv(0,1) = -G(0,1)/G_determinant;
					G_inv(1,0) = -G(1,0)/G_determinant;
				}
			}

		});
		teamMember.team_barrier();

		// generate prestencils that require information about the change of basis to the manifold
		if (_data_sampling_functional == ReconstructionOperator::SamplingFunctional::ManifoldVectorSample) {
			const int neighbor_offset = _neighbor_lists.dimension_1()-1;
			Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,this->getNNeighbors(target_index)),
						[=] (const int m) {
				for (int j=0; j<_dimensions; ++j) {
					for (int k=0; k<_dimensions-1; ++k) {
						_prestencil_weights(target_index, k*_dimensions*2*neighbor_offset + j*2*neighbor_offset + 2*m  ) =  T(j,k);
						_prestencil_weights(target_index, k*_dimensions*2*neighbor_offset + j*2*neighbor_offset + 2*m+1) =  0;
					}
					_prestencil_weights(target_index, (_dimensions-1)*_dimensions*2*neighbor_offset + j*2*neighbor_offset + 2*m  ) =  0;
					_prestencil_weights(target_index, (_dimensions-1)*_dimensions*2*neighbor_offset + j*2*neighbor_offset + 2*m+1) =  0;
				}
			});
		} else if (_data_sampling_functional == ReconstructionOperator::SamplingFunctional::ManifoldGradientVectorSample) {
			const int neighbor_offset = _neighbor_lists.dimension_1()-1;
			Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,this->getNNeighbors(target_index)),
						[=] (const int m) {
				for (int j=0; j<_dimensions; ++j) {
					for (int k=0; k<_dimensions-1; ++k) {
						_prestencil_weights(target_index, k*_dimensions*2*neighbor_offset + j*2*neighbor_offset + 2*m  ) =  V(j,k);
						_prestencil_weights(target_index, k*_dimensions*2*neighbor_offset + j*2*neighbor_offset + 2*m+1) =  0;
					}
					_prestencil_weights(target_index, (_dimensions-1)*_dimensions*2*neighbor_offset + j*2*neighbor_offset + 2*m  ) =  0;
					_prestencil_weights(target_index, (_dimensions-1)*_dimensions*2*neighbor_offset + j*2*neighbor_offset + 2*m+1) =  0;
				}
			});
		} else if (_data_sampling_functional == ReconstructionOperator::SamplingFunctional::StaggeredEdgeIntegralSample) {
			const int neighbor_offset = _neighbor_lists.dimension_1()-1;
			Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,this->getNNeighbors(target_index)), [=] (const int m) {
				for (int quadrature = 0; quadrature<_number_of_quadrature_points; ++quadrature) {
					XYZ tangent_quadrature_coord_2d;
					for (int j=0; j<_dimensions-1; ++j) {
						tangent_quadrature_coord_2d[j] = getTargetCoordinate(target_index, j, &V);
						tangent_quadrature_coord_2d[j] -= getNeighborCoordinate(target_index, m, j, &V);
					}
					double tangent_vector[3];
					tangent_vector[0] = tangent_quadrature_coord_2d[0]*V(0,0) + tangent_quadrature_coord_2d[1]*V(0,1);
					tangent_vector[1] = tangent_quadrature_coord_2d[0]*V(1,0) + tangent_quadrature_coord_2d[1]*V(1,1);
					tangent_vector[2] = tangent_quadrature_coord_2d[0]*V(2,0) + tangent_quadrature_coord_2d[1]*V(2,1);

					for (int j=0; j<_dimensions; ++j) {
						_prestencil_weights(target_index, j*2*neighbor_offset + 2*m  ) +=  (1-_parameterized_quadrature_sites[quadrature])*tangent_vector[j]*_quadrature_weights[quadrature];
						_prestencil_weights(target_index, j*2*neighbor_offset + 2*m+1) +=  _parameterized_quadrature_sites[quadrature]*tangent_vector[j]*_quadrature_weights[quadrature];
					}
				}
			});
		}
		//
		//  END OF MANIFOLD METRIC CALCULATIONS
		//

		for (int n=0; n<_sampling_multiplier; ++n) {
			this->computeTargetFunctionalsOnManifold(teamMember, t1, t2, P_target_row, _operations, V, T, G_inv, manifold_coeffs, manifold_gradient_coeffs, n);
		}

		if (_nontrivial_nullspace) {
			// PsqrtW contains valid entries if _poly_order == _manifold_poly_order
			// However, if they differ, then they must be recomputed
			this->createWeightsAndP(teamMember, delta, PsqrtW, w, _dimensions-1, _poly_order, true /* weight with W*/, &V, &T, _polynomial_sampling_functional, &manifold_gradient_coeffs, &quadrature_manifold_gradients);

			GMLS_LinearAlgebra::GolubReinschSVD(teamMember, t1, t2, PsqrtW, Q, S, Vsvd, target_NP*_basis_multiplier /* custom # of columns*/, this->getNNeighbors(target_index)*_sampling_multiplier /* custom # of rows*/);

			// threshold for dropping singular values
			double S0 = S(0);
			double eps = 1e-14;
			int maxVal = (target_NP*_basis_multiplier > _sampling_multiplier*this->getNNeighbors(target_index)) ? target_NP*_basis_multiplier : this->getNNeighbors(target_index)*_sampling_multiplier;
			double abs_threshold = maxVal*eps*S0;

			this->applySVD(teamMember, b_data, t1, Q, S, Vsvd, w, P_target_row, target_NP, abs_threshold);

		} else {
			// PsqrtW contains valid entries if _poly_order == _manifold_poly_order
			// However, if they differ, then they must be recomputed
			this->createWeightsAndP(teamMember, delta, PsqrtW, w, _dimensions-1, _poly_order, true /* weight with W*/, &V, &T, _polynomial_sampling_functional);

			GMLS_LinearAlgebra::GivensQR(teamMember, t1, t2, t3, Q, PsqrtW, _basis_multiplier*target_NP, _sampling_multiplier*this->getNNeighbors(target_index) /* custom # of rows*/);
			teamMember.team_barrier();

			this->applyQR(teamMember, t1, t2, Q, PsqrtW, w, P_target_row, target_NP); 

		}
	} else { // SVD

		// team_scratch all have a copy each per team
		// the threads in a team work on these local copies that only the team sees
		// thread_scratch has a copy per thread

		scratch_matrix_type PsqrtW(teamMember.team_scratch(_scratch_team_level_b), max_neighbors_or_basis, max_neighbors_or_basis); // larger than needed to allow for transpose
		scratch_matrix_type U(teamMember.team_scratch(_scratch_team_level_b), _neighbor_lists.dimension_1()-1, _neighbor_lists.dimension_1()-1);
		scratch_matrix_type V(teamMember.team_scratch(_scratch_team_level_b), _NP, _NP);
		scratch_vector_type S(teamMember.team_scratch(_scratch_team_level_b), _NP);
		scratch_matrix_type b_data(teamMember.team_scratch(_scratch_team_level_a), _NP, _sampling_multiplier);
		scratch_vector_type t1(teamMember.team_scratch(_scratch_team_level_a), _neighbor_lists.dimension_1()-1);
		scratch_vector_type t2(teamMember.team_scratch(_scratch_team_level_a), _neighbor_lists.dimension_1()-1);
		scratch_vector_type w(teamMember.team_scratch(_scratch_team_level_b), _neighbor_lists.dimension_1()-1);
		scratch_matrix_type P_target_row(teamMember.team_scratch(_scratch_team_level_b), _NP*_total_alpha_values, _sampling_multiplier);

		// delta is a temporary variable that preallocates space on which solutions of size _NP will
		// be computed. This allows the gpu to allocate the memory in advance
		scratch_vector_type delta(teamMember.thread_scratch(_scratch_thread_level_b), _basis_multiplier*_NP);

		// creates the matrix sqrt(W)*P
		this->createWeightsAndP(teamMember, delta, PsqrtW, w, _dimensions, _poly_order, true /* weight with W */, NULL /*&V*/, NULL /*&T*/, _polynomial_sampling_functional);

#if defined(USE_CUSTOM_SVD)
		GMLS_LinearAlgebra::GolubReinschSVD(teamMember, t1, t2, PsqrtW, U, S, V, _NP, this->getNNeighbors(target_index));
#else
		GMLS_LinearAlgebra::computeSVD(teamMember, U, S, V, PsqrtW, _NP, this->getNNeighbors(target_index)); // V is V'
#endif
		teamMember.team_barrier();

		this->computeTargetFunctionals(teamMember, t1, t2, P_target_row);

		// absolute threshold for dropping singular values
		double S0 = S(0);
		double eps = 1e-18;
		int maxVal = (_NP > this->getNNeighbors(target_index)) ? _NP : this->getNNeighbors(target_index);
		double abs_threshold = maxVal*eps*S0;

		this->applySVD(teamMember, b_data, t1, U, S, V, w, P_target_row, _NP, abs_threshold);

	}
	teamMember.team_barrier();
}

void GMLS::generateAlphas() {
	// copy over operations
    _operations = Kokkos::View<ReconstructionOperator::TargetOperation*> ("operations", _lro.size());
    _host_operations = Kokkos::create_mirror_view(_operations);
    for (int i=0; i<_lro.size(); ++i) _host_operations(i) = _lro[i];
	Kokkos::deep_copy(_operations, _host_operations);

	_alphas = Kokkos::View<double**, layout_type>("coefficients", _neighbor_lists.dimension_0(), _total_alpha_values*(_neighbor_lists.dimension_1()-1));

	if (ReconstructionOperator::SamplingNontrivialNullspace[_polynomial_sampling_functional]==1) {
		_nontrivial_nullspace = true;
	}

	if (_data_sampling_functional != ReconstructionOperator::SamplingFunctional::PointSample) {
		_prestencil_weights = Kokkos::View<double**, layout_type>("prestencil weights", _neighbor_lists.dimension_0(), std::pow(_dimensions,ReconstructionOperator::SamplingOutputTensorRank[_data_sampling_functional]+1)*2*(_neighbor_lists.dimension_1()-1), 0);
	}

	if (ReconstructionOperator::SamplingOutputTensorRank[_data_sampling_functional] > 0) {
		if (_dense_solver_type == ReconstructionOperator::DenseSolverType::MANIFOLD) {
			_sampling_multiplier = _dimensions-1;
		} else {
			_sampling_multiplier = _dimensions;
		}
	} else {
		_sampling_multiplier = 1;
	}

	if (ReconstructionOperator::ReconstructionSpaceRank[_reconstruction_space] > 0) {
		if (_dense_solver_type == ReconstructionOperator::DenseSolverType::MANIFOLD) {
			_basis_multiplier = _dimensions-1;
		} else {
			_basis_multiplier = _dimensions;
		}
	} else {
		_basis_multiplier = 1;
	}

	if (_polynomial_sampling_functional == ReconstructionOperator::SamplingFunctional::StaggeredEdgeAnalyticGradientIntegralSample) {
		// if the reconstruction is being made with a gradient of a basis, then we want that basis to be one order higher so that
		// the gradient is consistent with the convergence order expected.
		_poly_order += 1;
	}

	int team_scratch_size_a = 0;
	int team_scratch_size_b = scratch_vector_type::shmem_size((_neighbor_lists.dimension_1()-1)*_sampling_multiplier); // weights W;
	int thread_scratch_size_a = 0;
	int thread_scratch_size_b = 0;

	const int max_neighbors_or_basis = ((_neighbor_lists.dimension_1()-1)*_sampling_multiplier > _NP*_basis_multiplier)
		? (_neighbor_lists.dimension_1()-1)*_sampling_multiplier : _NP*_basis_multiplier;

	if (_dense_solver_type == ReconstructionOperator::DenseSolverType::LU) {
		team_scratch_size_b += scratch_matrix_type::shmem_size(max_neighbors_or_basis, max_neighbors_or_basis); // P matrix will not be used by QR or SVD
		team_scratch_size_b += scratch_matrix_type::shmem_size(_NP, _NP); // M_data
		team_scratch_size_b += scratch_matrix_type::shmem_size(_NP, _NP); // M_inv
		team_scratch_size_b += scratch_matrix_type::shmem_size(_NP, _NP); // L
		team_scratch_size_a += scratch_matrix_type::shmem_size(_NP, _sampling_multiplier); // b_data, used for eachM_data team
		team_scratch_size_a += scratch_vector_type::shmem_size(_neighbor_lists.dimension_1()-1); // t1
		team_scratch_size_a += scratch_vector_type::shmem_size(_neighbor_lists.dimension_1()-1); // t2

		team_scratch_size_b += scratch_matrix_type::shmem_size(_NP*_total_alpha_values, _sampling_multiplier); // row of P matrix, one for each operator

		thread_scratch_size_b += scratch_vector_type::shmem_size(_basis_multiplier*_NP); // delta, used for each thread
	} else if (_dense_solver_type == ReconstructionOperator::DenseSolverType::SVD){
		team_scratch_size_b += scratch_matrix_type::shmem_size(max_neighbors_or_basis, max_neighbors_or_basis); // P matrix made square in case or transpose needed
		team_scratch_size_b += scratch_matrix_type::shmem_size(_neighbor_lists.dimension_1()-1, _neighbor_lists.dimension_1()-1); // U
		team_scratch_size_b += scratch_matrix_type::shmem_size(_NP, _NP); // Vt
		team_scratch_size_b += scratch_vector_type::shmem_size(_NP); // S
		team_scratch_size_a += scratch_matrix_type::shmem_size(_NP, _sampling_multiplier); // b_data, used for eachM_data team
		team_scratch_size_a += scratch_vector_type::shmem_size(_neighbor_lists.dimension_1()-1); // t1 work vector for qr
		team_scratch_size_a += scratch_vector_type::shmem_size(_neighbor_lists.dimension_1()-1); // t2 work vector for qr

		team_scratch_size_b += scratch_matrix_type::shmem_size(_NP*_total_alpha_values, _sampling_multiplier); // row of P matrix, one for each operator

		thread_scratch_size_b += scratch_vector_type::shmem_size(_basis_multiplier*_NP); // delta, used for each thread
	} else if (_dense_solver_type == ReconstructionOperator::DenseSolverType::MANIFOLD) {
		// designed for ND-1 manifold embedded in ND problem
		const int manifold_NP = this->getNP(_manifold_poly_order, _dimensions-1);
		const int target_NP = this->getNP(_poly_order, _dimensions-1);

		const int max_manifold_NP = (manifold_NP > target_NP) ? manifold_NP : target_NP;
		const int max_NP = (max_manifold_NP > _NP) ? max_manifold_NP : _NP;
		const int max_P_row_size = ((_dimensions-1)*manifold_NP > max_NP*_total_alpha_values*_basis_multiplier) ? (_dimensions-1)*manifold_NP : max_NP*_total_alpha_values*_basis_multiplier;

		const int max_neighbors_or_basis_manifold = ((_neighbor_lists.dimension_1()-1)*_sampling_multiplier > max_NP*_basis_multiplier)
						? (_neighbor_lists.dimension_1()-1)*_sampling_multiplier : max_NP*_basis_multiplier;

		team_scratch_size_b += scratch_matrix_type::shmem_size(max_neighbors_or_basis_manifold, max_neighbors_or_basis_manifold); // P matrix must be square so that transpose is possible

		team_scratch_size_b += scratch_matrix_type::shmem_size((_neighbor_lists.dimension_1()-1)*_sampling_multiplier, (_neighbor_lists.dimension_1()-1)*_sampling_multiplier); // U
		team_scratch_size_b += scratch_matrix_type::shmem_size(_dimensions, _dimensions); // Vt (low-order tangent approximations)
		team_scratch_size_b += scratch_matrix_type::shmem_size(_dimensions, _dimensions-1); // T (high-order tangent approximations)
		team_scratch_size_b += scratch_matrix_type::shmem_size(_dimensions-1, _dimensions-1); // G
		team_scratch_size_b += scratch_matrix_type::shmem_size(_dimensions-1, _dimensions-1); // G inverse
		team_scratch_size_b += scratch_matrix_type::shmem_size(_dimensions, _dimensions); // PTP matrix
		team_scratch_size_b += scratch_vector_type::shmem_size( (_dimensions-1)*(_neighbor_lists.dimension_1()-1) ); // manifold_gradient
		team_scratch_size_b += scratch_vector_type::shmem_size( (_dimensions-1)*manifold_NP ); // manifold_coeffs
		team_scratch_size_b += scratch_vector_type::shmem_size((_neighbor_lists.dimension_1()-1)*std::max(_sampling_multiplier,_basis_multiplier)); // t1 work vector for qr
		team_scratch_size_b += scratch_vector_type::shmem_size((_neighbor_lists.dimension_1()-1)*std::max(_sampling_multiplier,_basis_multiplier)); // t2 work vector for qr

#ifdef COMPADRE_USE_LAPACK
		// find optimal blocksize when using LAPACK in order to allocate workspace needed
		int ipsec = 1, unused = -1, bnp = _basis_multiplier*max_NP, lwork = -1, info = 0; double wkopt = 0;
		int mnob = max_neighbors_or_basis_manifold;
		dormqr_( (char *)"L", (char *)"T", &mnob, &mnob, 
                	&bnp, (double *)NULL, &mnob, (double *)NULL, 
                	(double *)NULL, &mnob, &wkopt, &lwork, &info);
		int lapack_opt_blocksize = (int)wkopt;
		team_scratch_size_b += scratch_vector_type::shmem_size(lapack_opt_blocksize); // t3 work vector for qr
#else
		team_scratch_size_b += scratch_vector_type::shmem_size(_neighbor_lists.dimension_1()-1); // t3 work vector for qr
#endif

		team_scratch_size_b += scratch_vector_type::shmem_size( _dimensions-1 ); // manifold_gradient_coeffs

		team_scratch_size_b += scratch_matrix_type::shmem_size(max_P_row_size, _sampling_multiplier); // row of P matrix, one for each operator
		team_scratch_size_b += scratch_matrix_type::shmem_size((_neighbor_lists.dimension_1()-1)*(_dimensions-1), _number_of_quadrature_points); // manifold info for integral point sampling

		thread_scratch_size_b += scratch_vector_type::shmem_size(max_NP*_basis_multiplier); // delta, used for each thread

		team_scratch_size_b += scratch_vector_type::shmem_size(max_manifold_NP*_basis_multiplier); // S
		team_scratch_size_b += scratch_matrix_type::shmem_size(max_manifold_NP*_basis_multiplier, _sampling_multiplier); // b_data, used for eachM_data team
		team_scratch_size_b += scratch_matrix_type::shmem_size(max_manifold_NP*_basis_multiplier, max_manifold_NP*_basis_multiplier); // Vsvd

		// generate 1d quadrature points
		this->generate1DQuadrature();

	} else  { // QR
		team_scratch_size_b += scratch_matrix_type::shmem_size(max_neighbors_or_basis, max_neighbors_or_basis); // P matrix, only _neighbor_lists.dimension_1()-1 x _NP needed, but may need to be transposed
		//team_scratch_size_b += scratch_matrix_type::shmem_size(_neighbor_lists.dimension_1()-1, _NP); // R matrix
		team_scratch_size_b += scratch_matrix_type::shmem_size(_neighbor_lists.dimension_1()-1, _neighbor_lists.dimension_1()-1); // Q
		team_scratch_size_a += scratch_vector_type::shmem_size(_neighbor_lists.dimension_1()-1); // t1 work vector for qr
		team_scratch_size_a += scratch_vector_type::shmem_size(_neighbor_lists.dimension_1()-1); // t2 work vector for qr
#ifdef COMPADRE_USE_LAPACK
		// find optimal blocksize when using LAPACK in order to allocate workspace needed
		int ipsec = 1, unused = -1, bnp = _basis_multiplier*_NP, lwork = -1, info = 0; double wkopt = 0;
		int mnob = max_neighbors_or_basis;
		dormqr_( (char *)"L", (char *)"T", &mnob, &mnob, 
                	&bnp, (double *)NULL, &mnob, (double *)NULL, 
                	(double *)NULL, &mnob, &wkopt, &lwork, &info);
		int lapack_opt_blocksize = (int)wkopt;
		team_scratch_size_b += scratch_vector_type::shmem_size(lapack_opt_blocksize); // t3 work vector for qr
#else
		team_scratch_size_b += scratch_vector_type::shmem_size(_neighbor_lists.dimension_1()-1); // t3 work vector for qr
#endif

		team_scratch_size_b += scratch_matrix_type::shmem_size(_NP*_total_alpha_values, _sampling_multiplier); // row of P matrix, one for each operator

		thread_scratch_size_b += scratch_vector_type::shmem_size(_basis_multiplier*_NP); // delta, used for each thread
	}

	Kokkos::fence();

#ifdef KOKKOS_ENABLE_CUDA
	int threads_per_team = 32;
	if (_basis_multiplier*_NP > 96) threads_per_team += 32;
	Kokkos::parallel_for(
			team_policy(_target_coordinates.dimension_0(), threads_per_team)
			.set_scratch_size(_scratch_team_level_a, Kokkos::PerTeam(team_scratch_size_a))
			.set_scratch_size(_scratch_team_level_b, Kokkos::PerTeam(team_scratch_size_b))
			.set_scratch_size(_scratch_thread_level_a, Kokkos::PerThread(thread_scratch_size_a))
			.set_scratch_size(_scratch_thread_level_b, Kokkos::PerThread(thread_scratch_size_b)),
			*this
	, "generateAlphas");
#else
	if (_scratch_team_level_b != _scratch_team_level_a) {
		Kokkos::parallel_for(
				team_policy(_target_coordinates.dimension_0(), Kokkos::AUTO)
				.set_scratch_size(_scratch_team_level_a, Kokkos::PerTeam(team_scratch_size_a))
				.set_scratch_size(_scratch_team_level_b, Kokkos::PerTeam(team_scratch_size_b))
				.set_scratch_size(_scratch_thread_level_a, Kokkos::PerThread(thread_scratch_size_a))
				.set_scratch_size(_scratch_thread_level_b, Kokkos::PerThread(thread_scratch_size_b)),
				*this
		, "generateAlphas");
	} else {
		Kokkos::parallel_for(
				team_policy(_target_coordinates.dimension_0(), Kokkos::AUTO)
				.set_scratch_size(_scratch_team_level_a, Kokkos::PerTeam(team_scratch_size_a+team_scratch_size_b))
				.set_scratch_size(_scratch_thread_level_a, Kokkos::PerThread(thread_scratch_size_a+thread_scratch_size_b)),
				*this
		, "generateAlphas");
	}
#endif

	Kokkos::fence();
	// copy computed alphas back to the host
	_host_alphas = Kokkos::create_mirror_view(_alphas);
	if (_data_sampling_functional != ReconstructionOperator::SamplingFunctional::PointSample)
		_host_prestencil_weights = Kokkos::create_mirror_view(_prestencil_weights);
	Kokkos::deep_copy(_host_alphas, _alphas);
	Kokkos::fence();
}

void GMLS::generate1DQuadrature() {

	_quadrature_weights = Kokkos::View<double*, layout_type>("1d quadrature weights", _number_of_quadrature_points);
	_parameterized_quadrature_sites = Kokkos::View<double*, layout_type>("1d quadrature sites", _number_of_quadrature_points);

	Kokkos::View<double*, layout_type>::HostMirror quadrature_weights = create_mirror(_quadrature_weights);
	Kokkos::View<double*, layout_type>::HostMirror parameterized_quadrature_sites = create_mirror(_parameterized_quadrature_sites);

	switch (_number_of_quadrature_points) {
	case 1:
		quadrature_weights[0] = 1.0;
		parameterized_quadrature_sites[0] = 0.5;
		break;
	case 2:
		quadrature_weights[0] = 0.5;
		quadrature_weights[1] = 0.5;
		parameterized_quadrature_sites[0] = 0.5*(1-std::sqrt(1./3.));
		parameterized_quadrature_sites[1] = 0.5*(1+std::sqrt(1./3.));
		break;
	case 3:
		quadrature_weights[0] = 2.5/9.;
		quadrature_weights[1] = 2.5/9;
		quadrature_weights[2] = 4./9.;
		parameterized_quadrature_sites[0] = 0.5*(1-std::sqrt(3./5.));
		parameterized_quadrature_sites[1] = 0.5*(1+std::sqrt(3./5.));
		parameterized_quadrature_sites[2] = 0.5;
		break;
	case 4:
		quadrature_weights[0] = (18+std::sqrt(30))/72.;
		quadrature_weights[1] = (18+std::sqrt(30))/72.;
		quadrature_weights[2] = (18-std::sqrt(30))/72.;
		quadrature_weights[3] = (18-std::sqrt(30))/72.;
		parameterized_quadrature_sites[0] = 0.5*(1-std::sqrt(3./7.-2./7.*std::sqrt(6./5.)));
		parameterized_quadrature_sites[1] = 0.5*(1+std::sqrt(3./7.-2./7.*std::sqrt(6./5.)));
		parameterized_quadrature_sites[2] = 0.5*(1-std::sqrt(3./7.+2./7.*std::sqrt(6./5.)));
		parameterized_quadrature_sites[3] = 0.5*(1+std::sqrt(3./7.+2./7.*std::sqrt(6./5.)));
		break;
	case 5:
		quadrature_weights[0] = 128./450.;
		quadrature_weights[1] = (322+13*std::sqrt(70))/1800.;
		quadrature_weights[2] = (322+13*std::sqrt(70))/1800.;
		quadrature_weights[3] = (322-13*std::sqrt(70))/1800.;
		quadrature_weights[4] = (322-13*std::sqrt(70))/1800.;
		parameterized_quadrature_sites[0] = 0.5;
		parameterized_quadrature_sites[1] = 0.5*(1+1./3.*std::sqrt(5-2*std::sqrt(10./7.)));
		parameterized_quadrature_sites[2] = 0.5*(1-1./3.*std::sqrt(5-2*std::sqrt(10./7.)));
		parameterized_quadrature_sites[3] = 0.5*(1+1./3.*std::sqrt(5+2*std::sqrt(10./7.)));
		parameterized_quadrature_sites[4] = 0.5*(1-1./3.*std::sqrt(5+2*std::sqrt(10./7.)));
		break;
	default:
		ASSERT_WITH_MESSAGE(false, "Unsupported number of quadrature points.");
		quadrature_weights[0] = 0.5;
		quadrature_weights[1] = 0.5;
		parameterized_quadrature_sites[0] = 0.5*(1-std::sqrt(1./3.));
		parameterized_quadrature_sites[1] = 0.5*(1+std::sqrt(1./3.));
		for (int i=2; i<_number_of_quadrature_points; ++i) {
			quadrature_weights[i] = 0;
			parameterized_quadrature_sites[i] = 0;
		}
	}

    Kokkos::deep_copy(_quadrature_weights, quadrature_weights);
    Kokkos::deep_copy(_parameterized_quadrature_sites, parameterized_quadrature_sites);
}

KOKKOS_INLINE_FUNCTION
void GMLS::applySVD(const member_type& teamMember, scratch_matrix_type b_data, scratch_vector_type t1, scratch_matrix_type U, scratch_vector_type S, scratch_matrix_type V, scratch_vector_type w, scratch_matrix_type P_target_row, const int target_NP, const double abs_threshold) const {

	const int target_index = teamMember.league_rank();

	// t1 takes on the role of S inverse
	for (int i=0; i<target_NP*_basis_multiplier; i++) {
		t1(i) = ( std::abs(S(i)) > abs_threshold ) ? 1./S(i) : 0;
	}

	if (std::is_same<scratch_matrix_type::array_layout, Kokkos::LayoutRight>::value) {
		// CPU
		for (int i=0; i<this->getNNeighbors(target_index); ++i) {

			for (int m=0; m<_sampling_multiplier; ++m) {
				for (int j=0; j<target_NP*_basis_multiplier; ++j) {
					double  bdataj = 0;
					Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember,
							target_NP*_basis_multiplier), [&] (const int k, double &tbdataj) {
		#if defined(USE_CUSTOM_SVD)
						tbdataj += V(j,k)*t1(k)*U(i + m*this->getNNeighbors(target_index),k);
		#else
						tbdataj += V(k,j)*t1(k)*U(i + m*this->getNNeighbors(target_index),k);
		#endif
					}, bdataj);
					Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
						b_data(j,m) = bdataj*std::sqrt(w(i + m*this->getNNeighbors(target_index)));
					});
					teamMember.team_barrier();
				}
			}
			teamMember.team_barrier();


			for (int j=0; j<_operations.size(); ++j) {
				for (int k=0; k<_lro_output_tile_size[j]; ++k) {
					for (int m=0; m<_lro_input_tile_size[j]; ++m) {

						const int column = (_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k)*_neighbor_lists(target_index,0) + i;

						double alpha_ij = 0;
						Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember,
								_basis_multiplier*target_NP), [=] (const int l, double &talpha_ij) {
							if (_sampling_multiplier>1 && m<_sampling_multiplier) {
								talpha_ij += P_target_row(_basis_multiplier*target_NP*(_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k) + l, 0)*b_data(l,m);
							} else if (_sampling_multiplier == 1) {
								talpha_ij += P_target_row(_basis_multiplier*target_NP*(_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k) + l, 0)*b_data(l,0);
							} else {
								talpha_ij += 0;
							}
						}, alpha_ij);
						Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
							// not optimal for LayoutLeft
							_alphas(target_index, column) = alpha_ij;
						});
						teamMember.team_barrier();
					}
				}
			}

			teamMember.team_barrier();
		}
	} else {
		// GPU
		for (int i=0; i<this->getNNeighbors(target_index); ++i) {
			for (int m=0; m<_sampling_multiplier; ++m) {
				Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,target_NP*_basis_multiplier), [=] (const int j) {
					double  bdataj = 0;
					for (int k=0; k<target_NP*_basis_multiplier; ++k) {
	#if defined(USE_CUSTOM_SVD)
						bdataj += V(j,k)*t1(k)*U(i + m*this->getNNeighbors(target_index),k);
	#else
						bdataj += V(k,j)*t1(k)*U(i + m*this->getNNeighbors(target_index),k);
	#endif
					}
					bdataj *= std::sqrt(w(i + m*this->getNNeighbors(target_index)));
					b_data(j,m) = bdataj;
				});
			}

			for (int j=0; j<_operations.size(); ++j) {
				for (int k=0; k<_lro_output_tile_size[j]; ++k) {
					for (int m=0; m<_lro_input_tile_size[j]; ++m) {

						const int column = (_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k)*_neighbor_lists(target_index,0) + i;

						double alpha_ij = 0;
						Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember,
								_basis_multiplier*target_NP), [=] (const int l, double &talpha_ij) {
							if (_sampling_multiplier>1 && m<_sampling_multiplier) {
								talpha_ij += P_target_row(_basis_multiplier*target_NP*(_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k) + l, 0)*b_data(l,m);
							} else if (_sampling_multiplier == 1) {
								talpha_ij += P_target_row(_basis_multiplier*target_NP*(_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k) + l, 0)*b_data(l,0);
							} else {
								talpha_ij += 0;
							}
						}, alpha_ij);
						Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
							// not optimal for LayoutLeft
							_alphas(target_index, column) = alpha_ij;
						});
						teamMember.team_barrier();
					}
				}
			}

			teamMember.team_barrier();
		}
	}
}

KOKKOS_INLINE_FUNCTION
void GMLS::applyQR(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_matrix_type Q, scratch_matrix_type R, scratch_vector_type w, scratch_matrix_type P_target_row, const int target_NP) const {

	const int target_index = teamMember.league_rank();

	GMLS_LinearAlgebra::upperTriangularBackSolve(teamMember, t1, t2, Q, R, w, _basis_multiplier*target_NP, _sampling_multiplier*this->getNNeighbors(target_index)); // stores solution in Q

	if (std::is_same<scratch_matrix_type::array_layout, Kokkos::LayoutRight>::value) {
		// CPU
		for (int i=0; i<this->getNNeighbors(target_index); ++i) {
			for (int j=0; j<_operations.size(); ++j) {
				for (int k=0; k<_lro_output_tile_size[j]; ++k) {
					for (int m=0; m<_lro_input_tile_size[j]; ++m) {
						double alpha_ij = 0;
						Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember,
							_basis_multiplier*target_NP), [=] (const int l, double &talpha_ij) {
							if (_sampling_multiplier>1 && m<_sampling_multiplier) {
								talpha_ij += P_target_row(_basis_multiplier*target_NP*(_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k) + l, 0)*Q(i + m*this->getNNeighbors(target_index),l);
							} else if (_sampling_multiplier == 1) {
								talpha_ij += P_target_row(_basis_multiplier*target_NP*(_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k) + l, 0)*Q(i,l);
							} else {
								talpha_ij += 0;
							}
						}, alpha_ij);
						Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
							_alphas(target_index, (_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k)*_neighbor_lists(target_index,0) + i) = alpha_ij;
						});
					}
				}
			}
		}
	} else {
		// GPU
		Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,
			this->getNNeighbors(target_index)), [=] (const int i) {
			for (int j=0; j<_operations.size(); ++j) {
				for (int k=0; k<_lro_output_tile_size[j]; ++k) {
					for (int m=0; m<_lro_input_tile_size[j]; ++m) {
						double alpha_ij = 0;
						if (_sampling_multiplier>1 && m<_sampling_multiplier) {
							for (int l=0; l<_basis_multiplier*target_NP; ++l) {
								alpha_ij += P_target_row(_basis_multiplier*target_NP*(_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k) + l, 0)*Q(i + m*this->getNNeighbors(target_index),l);
							}
						} else if (_sampling_multiplier == 1) {
							for (int l=0; l<_basis_multiplier*target_NP; ++l) {
								alpha_ij += P_target_row(_basis_multiplier*target_NP*(_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k) + l, 0)*Q(i,l);
							}
						} 
						_alphas(target_index, (_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k)*_neighbor_lists(target_index,0) + i) = alpha_ij;
					}
				}
			}
		});
	}
}

KOKKOS_INLINE_FUNCTION
void GMLS::applyMInverse(const member_type& teamMember, scratch_matrix_type b_data, scratch_matrix_type MInv, scratch_matrix_type PsqrtW, scratch_vector_type w, scratch_matrix_type P_target_row, const int target_NP) const {

	const int target_index = teamMember.league_rank();

	if (std::is_same<scratch_matrix_type::array_layout, Kokkos::LayoutRight>::value) {
		// CPU
		for (int i=0; i<this->getNNeighbors(target_index); ++i) {
			for (int m=0; m<_sampling_multiplier; ++m) {
				Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,
					target_NP*_basis_multiplier), [=] (const int j) {
					double  bdataj = 0;
					for (int k=0; k<_NP; ++k) {
						bdataj += MInv(j,k)*PsqrtW(i,k);
					}
					bdataj *= std::sqrt(w(i + m*this->getNNeighbors(target_index)));
					b_data(j,m) = bdataj;
				});
			}
			teamMember.team_barrier();

			for (int j=0; j<_operations.size(); ++j) {
				for (int k=0; k<_lro_output_tile_size[j]; ++k) {
					for (int m=0; m<_lro_input_tile_size[j]; ++m) {
						double alpha_ij = 0;
						Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember,
							_basis_multiplier*target_NP), [=] (const int l, double &talpha_ij) {
							if (_sampling_multiplier>1 && m<_sampling_multiplier) {
								talpha_ij += P_target_row(_basis_multiplier*target_NP*(_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k) + l, 0)*b_data(l,m);
							} else if (_sampling_multiplier == 1) {
								talpha_ij += P_target_row(_basis_multiplier*target_NP*(_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k) + l, 0)*b_data(l,0);
							} 
						}, alpha_ij);
						Kokkos::single(Kokkos::PerTeam(teamMember), [&] () {
							_alphas(target_index, (_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k)*_neighbor_lists(target_index,0) + i) = alpha_ij;
						});
					}
				}
			}
		}
	} else {
		// GPU
		Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,
			this->getNNeighbors(target_index)), [=] (const int i) {
			for (int m=0; m<_sampling_multiplier; ++m) {
				for (int j=0; j<target_NP*_basis_multiplier; ++j) {
					double  bdataj = 0;
					for (int k=0; k<_NP; ++k) {
						bdataj += MInv(j,k)*PsqrtW(i,k);
					}
					bdataj *= std::sqrt(w(i + m*this->getNNeighbors(target_index)));
					b_data(j,m) = bdataj;
				}
			}

			for (int j=0; j<_operations.size(); ++j) {
				for (int k=0; k<_lro_output_tile_size[j]; ++k) {
					for (int m=0; m<_lro_input_tile_size[j]; ++m) {
						double alpha_ij = 0;
						if (_sampling_multiplier>1 && m<_sampling_multiplier) {
							for (int l=0; l<_basis_multiplier*target_NP; ++l) {
								alpha_ij += P_target_row(_basis_multiplier*target_NP*(_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k) + l, 0)*b_data(l,m);
							}
						} else if (_sampling_multiplier == 1) {
							for (int l=0; l<_basis_multiplier*target_NP; ++l) {
								alpha_ij += P_target_row(_basis_multiplier*target_NP*(_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k) + l, 0)*b_data(l,0);
							}
						} 
						_alphas(target_index, (_lro_total_offsets[j] + m*_lro_output_tile_size[j] + k)*_neighbor_lists(target_index,0) + i) = alpha_ij;
					}
				}
			}
		});
	}
}

#endif
