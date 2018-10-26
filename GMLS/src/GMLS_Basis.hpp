#ifndef _GMLS_BASIS_HPP_
#define _GMLS_BASIS_HPP_

#include "GMLS.hpp"

#ifdef COMPADRE_USE_KOKKOSCORE

namespace Compadre {

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
void GMLS::calcWij(double* delta, const int target_index, int neighbor_index, const double alpha, const int dimension, const int poly_order, bool specific_order_only, scratch_matrix_type* V, scratch_matrix_type* T, const ReconstructionOperator::SamplingFunctional polynomial_sampling_functional) const {
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
void GMLS::createWeightsAndP(const member_type& teamMember, scratch_vector_type delta, scratch_matrix_type P, scratch_vector_type w, const int dimension, int polynomial_order, bool weight_p, scratch_matrix_type* V, scratch_matrix_type* T, const ReconstructionOperator::SamplingFunctional polynomial_sampling_functional) const {
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
        double * p_data = P.data();
	const int my_num_neighbors = this->getNNeighbors(target_index);

	teamMember.team_barrier();
	Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember,this->getNNeighbors(target_index)),
			[=] (const int i) {

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

			this->calcWij(delta.data(), target_index, i + d*my_num_neighbors, 0 /*alpha*/, dimension, polynomial_order, false /*bool on only specific order*/, V, T, polynomial_sampling_functional);

			int storage_size = this->getNP(polynomial_order, dimension);
			storage_size *= _basis_multiplier;

			if (weight_p) {
				for (int j = 0; j < storage_size; ++j) {
                        		// stores layout left for CUDA or LAPACK calls later
					*(p_data + j*P.dimension_0() + i+my_num_neighbors*d) = delta[j] * std::sqrt(w(i+my_num_neighbors*d));
				}

			} else {
				for (int j = 0; j < storage_size; ++j) {
                        		// stores layout left for CUDA or LAPACK calls later
					*(p_data + j*P.dimension_0() + i+my_num_neighbors*d) = delta[j];
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
	double * p_data = P.data();

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
                        // stores layout left for CUDA or LAPACK calls later
			*(p_data + j*P.dimension_0() + i) = delta[j] * std::sqrt(w(i));
		}

	});
	teamMember.team_barrier();
}

}; // Compadre
#endif
#endif
