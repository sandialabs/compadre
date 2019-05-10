#ifndef _COMPADRE_GMLS_BASIS_HPP_
#define _COMPADRE_GMLS_BASIS_HPP_

#include "Compadre_GMLS.hpp"

namespace Compadre {

KOKKOS_INLINE_FUNCTION
double GMLS::Wab(const double r, const double h, const WeightingFunctionType& weighting_type, const int power) const {

    if (weighting_type == WeightingFunctionType::Power) {
        return std::pow(1.0-std::abs(r/(3*h)), power) * double(1.0-std::abs(r/(3*h))>0.0);
    } else { // Gaussian
        // 2.5066282746310002416124 = sqrt(2*pi)
        double h_over_3 = h/3.0;
        return 1./( h_over_3 * 2.5066282746310002416124 ) * std::exp(-.5*r*r/(h_over_3*h_over_3));
    }

}

KOKKOS_INLINE_FUNCTION
void GMLS::calcPij(double* delta, const int target_index, int neighbor_index, const double alpha, const int dimension, const int poly_order, bool specific_order_only, scratch_matrix_right_type* V, const ReconstructionSpace reconstruction_space, const SamplingFunctional polynomial_sampling_functional, const int additional_evaluation_local_index) const {
/*
 * This class is under two levels of hierarchical parallelism, so we
 * do not put in any finer grain parallelism in this function
 */
    const int my_num_neighbors = this->getNNeighbors(target_index);
    
    // store precalculated factorials for speedup
    const double factorial[15] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200};

    int component = 0;
    if (neighbor_index >= my_num_neighbors) {
        component = neighbor_index / my_num_neighbors;
        neighbor_index = neighbor_index % my_num_neighbors;
    }

    XYZ relative_coord;
    if (neighbor_index > -1) {
        for (int i=0; i<dimension; ++i) {
            // calculates (alpha*target+(1-alpha)*neighbor)-1*target = (alpha-1)*target + (1-alpha)*neighbor
            relative_coord[i] = (alpha-1)*getTargetCoordinate(target_index, i, V);
            relative_coord[i] += (1-alpha)*getNeighborCoordinate(target_index, neighbor_index, i, V);
        }
    } else if (additional_evaluation_local_index > 0) {
        for (int i=0; i<dimension; ++i) {
            relative_coord[i] = getTargetAuxiliaryCoordinate(target_index, additional_evaluation_local_index, i, V);
            relative_coord[i] -= getTargetCoordinate(target_index, i, V);
        }
    } else {
        for (int i=0; i<3; ++i) relative_coord[i] = 0;
    }

    // basis ActualReconstructionSpaceRank is 0 (evaluated like a scalar) and sampling functional is traditional
    if ((polynomial_sampling_functional == SamplingFunctional::PointSample ||
            polynomial_sampling_functional == SamplingFunctional::VectorPointSample ||
            polynomial_sampling_functional == SamplingFunctional::ManifoldVectorPointSample) &&
            (reconstruction_space == ScalarTaylorPolynomial || reconstruction_space == VectorOfScalarClonesTaylorPolynomial)) {

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
                        alphaf = factorial[alphax]*factorial[alphay]*factorial[alphaz];
                        *(delta+i) = std::pow(relative_coord.x/cutoff_p,alphax)
                                    *std::pow(relative_coord.y/cutoff_p,alphay)
                                    *std::pow(relative_coord.z/cutoff_p,alphaz)/alphaf;
                        i++;
                    }
                }
            } else if (dimension == 2) {
                for (alphay = 0; alphay <= n; alphay++){
                    alphax = n - alphay;
                    alphaf = factorial[alphax]*factorial[alphay];
                    *(delta+i) = std::pow(relative_coord.x/cutoff_p,alphax)
                                *std::pow(relative_coord.y/cutoff_p,alphay)/alphaf;
                    i++;
                }
            } else { // dimension == 1
                    alphax = n;
                    alphaf = factorial[alphax];
                    *(delta+i) = std::pow(relative_coord.x/cutoff_p,alphax)/alphaf;
                    i++;
            }
        }

    // basis ActualReconstructionSpaceRank is 1 (is a true vector basis) and sampling functional is traditional
    } else if ((polynomial_sampling_functional == SamplingFunctional::VectorPointSample ||
                polynomial_sampling_functional == SamplingFunctional::ManifoldVectorPointSample) &&
                    (reconstruction_space == VectorTaylorPolynomial)) {

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
                                alphaf = factorial[alphax]*factorial[alphay]*factorial[alphaz];
                                *(delta+component*dimension_offset+i) = std::pow(relative_coord.x/cutoff_p,alphax)
                                            *std::pow(relative_coord.y/cutoff_p,alphay)
                                            *std::pow(relative_coord.z/cutoff_p,alphaz)/alphaf;
                                i++;
                            }
                        }
                    } else if (dimension == 2) {
                        for (alphay = 0; alphay <= n; alphay++){
                            alphax = n - alphay;
                            alphaf = factorial[alphax]*factorial[alphay];
                            *(delta+component*dimension_offset+i) = std::pow(relative_coord.x/cutoff_p,alphax)
                                        *std::pow(relative_coord.y/cutoff_p,alphay)/alphaf;
                            i++;
                        }
                    } else { // dimension == 1
                            alphax = n;
                            alphaf = factorial[alphax];
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


    // basis is actually scalar with staggered sampling functional
    } else if ((polynomial_sampling_functional == SamplingFunctional::StaggeredEdgeAnalyticGradientIntegralSample) &&
            (reconstruction_space == ScalarTaylorPolynomial)) {

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
                            alphaf = factorial[alphax]*factorial[alphay]*factorial[alphaz];
                            *(delta+i) = -std::pow(relative_coord.x/cutoff_p,alphax)
                                        *std::pow(relative_coord.y/cutoff_p,alphay)
                                        *std::pow(relative_coord.z/cutoff_p,alphaz)/alphaf;
                            i++;
                        }
                    }
                } else if (dimension == 2) {
                    for (alphay = 0; alphay <= n; alphay++){
                        alphax = n - alphay;
                        alphaf = factorial[alphax]*factorial[alphay];
                        *(delta+i) = -std::pow(relative_coord.x/cutoff_p,alphax)
                                    *std::pow(relative_coord.y/cutoff_p,alphay)/alphaf;
                        i++;
                    }
                } else { // dimension == 1
                        alphax = n;
                        alphaf = factorial[alphax];
                        *(delta+i) = -std::pow(relative_coord.x/cutoff_p,alphax)/alphaf;
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
                            alphaf = factorial[alphax]*factorial[alphay]*factorial[alphaz];
                            *(delta+i) += std::pow(relative_coord.x/cutoff_p,alphax)
                                        *std::pow(relative_coord.y/cutoff_p,alphay)
                                        *std::pow(relative_coord.z/cutoff_p,alphaz)/alphaf;
                            i++;
                        }
                    }
                } else if (dimension == 2) {
                    for (alphay = 0; alphay <= n; alphay++){
                        alphax = n - alphay;
                        alphaf = factorial[alphax]*factorial[alphay];
                        *(delta+i) += std::pow(relative_coord.x/cutoff_p,alphax)
                                    *std::pow(relative_coord.y/cutoff_p,alphay)/alphaf;
                        i++;
                    }
                } else { // dimension == 1
                        alphax = n;
                        alphaf = factorial[alphax];
                        *(delta+i) += std::pow(relative_coord.x/cutoff_p,alphax)/alphaf;
                        i++;
                }
            }
        }
    } else if (polynomial_sampling_functional == SamplingFunctional::StaggeredEdgeIntegralSample) {

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
                        alphaf = factorial[alphax]*factorial[alphay];

                        // local evaluation of vector [0,p] or [p,0]
                        double v0, v1;
                        v0 = (j==0) ? std::pow(quadrature_coord_2d.x/cutoff_p,alphax)
                            *std::pow(quadrature_coord_2d.y/cutoff_p,alphay)/alphaf : 0;
                        v1 = (j==0) ? 0 : std::pow(quadrature_coord_2d.x/cutoff_p,alphax)
                            *std::pow(quadrature_coord_2d.y/cutoff_p,alphay)/alphaf;

                        double dot_product = tangent_quadrature_coord_2d[0]*v0 + tangent_quadrature_coord_2d[1]*v1;

                        // multiply by quadrature weight
                        if (quadrature==0) {
                            *(delta+i) = dot_product * _quadrature_weights[quadrature];
                        } else {
                            *(delta+i) += dot_product * _quadrature_weights[quadrature];
                        }
                        i++;
                    }
                }
            }
        }
    } else if (polynomial_sampling_functional == SamplingFunctional::FaceNormalIntegralSample) {

        double cutoff_p = _epsilons(target_index);

        compadre_kernel_assert_debug(_dimensions==2 && "Only written for 2D");
        compadre_kernel_assert_debug(_extra_data.extent(0)>0 && "Extra data used but not set.");

        int neighbor_index_in_source = getNeighborIndex(target_index, neighbor_index);
        //printf("%d, %d, %f, %f\n", target_index, neighbor_index_in_source, _extra_data(neighbor_index_in_source,0), _extra_data(neighbor_index_in_source,1));

        /*
         * requires quadrature points defined on an edge, not a target/source edge (spoke)
         *
         * _extra_data will contain the endpoints (2 for 2D, 3 for 3D) and then the unit normals
         * (e0_x, e0_y, e1_x, e1_y, n_x, n_y)
         */

        XYZ endpoints_difference;
        for (int j=0; j<dimension; ++j) {
            endpoints_difference[j] = _extra_data(neighbor_index_in_source, j) - _extra_data(neighbor_index_in_source, j+2);
        }
        double magnitude = EuclideanVectorLength(endpoints_difference, 2);
        
        int alphax, alphay;
        double alphaf;
        const int start_index = specific_order_only ? poly_order : 0; // only compute specified order if requested

        for (int quadrature = 0; quadrature<_number_of_quadrature_points; ++quadrature) {

            int i = 0;

            XYZ normal_quadrature_coord_2d;
            XYZ quadrature_coord_2d;
            for (int j=0; j<dimension; ++j) {
                
                // coord site
                quadrature_coord_2d[j] = _parameterized_quadrature_sites[quadrature]*_extra_data(neighbor_index_in_source, j);
                quadrature_coord_2d[j] += (1-_parameterized_quadrature_sites[quadrature])*_extra_data(neighbor_index_in_source, j+2);
                quadrature_coord_2d[j] -= getTargetCoordinate(target_index, j);

                // normal direction
                normal_quadrature_coord_2d[j] = _extra_data(neighbor_index_in_source, 4 + j);

            }
            for (int j=0; j<_basis_multiplier; ++j) {
                for (int n = start_index; n <= poly_order; n++){
                    for (alphay = 0; alphay <= n; alphay++){
                        alphax = n - alphay;
                        alphaf = factorial[alphax]*factorial[alphay];

                        // local evaluation of vector [0,p] or [p,0]
                        double v0, v1;
                        v0 = (j==0) ? std::pow(quadrature_coord_2d.x/cutoff_p,alphax)
                            *std::pow(quadrature_coord_2d.y/cutoff_p,alphay)/alphaf : 0;
                        v1 = (j==0) ? 0 : std::pow(quadrature_coord_2d.x/cutoff_p,alphax)
                            *std::pow(quadrature_coord_2d.y/cutoff_p,alphay)/alphaf;

                        double dot_product = normal_quadrature_coord_2d[0]*v0 + normal_quadrature_coord_2d[1]*v1;

                        // multiply by quadrature weight
                        if (quadrature==0) {
                            *(delta+i) = dot_product * _quadrature_weights[quadrature] * magnitude;
                        } else {
                            *(delta+i) += dot_product * _quadrature_weights[quadrature] * magnitude;
                        }
                        i++;
                    }
                }
            }
        }
    } else {
        compadre_kernel_assert_release((false) && "Sampling and basis space combination not defined.");
    }
}


KOKKOS_INLINE_FUNCTION
void GMLS::calcGradientPij(double* delta, const int target_index, const int neighbor_index, const double alpha, const int partial_direction, const int dimension, const int poly_order, bool specific_order_only, scratch_matrix_right_type* V, const ReconstructionSpace reconstruction_space, const SamplingFunctional polynomial_sampling_functional, const int additional_evaluation_index) const {
/*
 * This class is under two levels of hierarchical parallelism, so we
 * do not put in any finer grain parallelism in this function
 */
    // store precalculated factorials for speedup
    const double factorial[15] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200};

    // alpha corresponds to the linear combination of target_index and neighbor_index coordinates
    // coordinate to evaluate = alpha*(target_index's coordinate) + (1-alpha)*(neighbor_index's coordinate)

    // partial_direction - 0=x, 1=y, 2=z
    XYZ relative_coord;
    if (neighbor_index > -1) {
        for (int i=0; i<dimension; ++i) {
            // calculates (alpha*target+(1-alpha)*neighbor)-1*target = (alpha-1)*target + (1-alpha)*neighbor
            relative_coord[i] = (alpha-1)*getTargetCoordinate(target_index, i, V);
            relative_coord[i] += (1-alpha)*getNeighborCoordinate(target_index, neighbor_index, i, V);
        }
    } else if (additional_evaluation_index > 0) {
        for (int i=0; i<dimension; ++i) {
            relative_coord[i] = getTargetAuxiliaryCoordinate(target_index, additional_evaluation_index, i, V);
            relative_coord[i] -= getTargetCoordinate(target_index, i, V);
        }
    } else {
        for (int i=0; i<3; ++i) relative_coord[i] = 0;
    }

    double cutoff_p = _epsilons(target_index);

    if ((polynomial_sampling_functional == SamplingFunctional::PointSample ||
            polynomial_sampling_functional == SamplingFunctional::VectorPointSample ||
            polynomial_sampling_functional == SamplingFunctional::ManifoldVectorPointSample) &&
            (reconstruction_space == ScalarTaylorPolynomial || reconstruction_space == VectorOfScalarClonesTaylorPolynomial)) {

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

                        int x_pow = (partial_direction == 0) ? alphax-1 : alphax;
                        int y_pow = (partial_direction == 1) ? alphay-1 : alphay;
                        int z_pow = (partial_direction == 2) ? alphaz-1 : alphaz;

                        if (x_pow<0 || y_pow<0 || z_pow<0) {
                            *(delta+i) = 0;
                        } else {
                            alphaf = factorial[x_pow]*factorial[y_pow]*factorial[z_pow];
                            *(delta+i) = 1./cutoff_p 
                                        *std::pow(relative_coord.x/cutoff_p,x_pow)
                                        *std::pow(relative_coord.y/cutoff_p,y_pow)
                                        *std::pow(relative_coord.z/cutoff_p,z_pow)/alphaf;
                        }
                        i++;
                    }
                }
            } else if (dimension == 2) {
                for (alphay = 0; alphay <= n; alphay++){
                    alphax = n - alphay;

                    int x_pow = (partial_direction == 0) ? alphax-1 : alphax;
                    int y_pow = (partial_direction == 1) ? alphay-1 : alphay;

                    if (x_pow<0 || y_pow<0) {
                        *(delta+i) = 0;
                    } else {
                        alphaf = factorial[x_pow]*factorial[y_pow];
                        *(delta+i) = 1./cutoff_p 
                                    *std::pow(relative_coord.x/cutoff_p,x_pow)
                                    *std::pow(relative_coord.y/cutoff_p,y_pow)/alphaf;
                    }
                    i++;
                }
            } else { // dimension == 1
                    alphax = n;

                    int x_pow = (partial_direction == 0) ? alphax-1 : alphax;
                    if (x_pow<0) {
                        *(delta+i) = 0;
                    } else {
                        alphaf = factorial[x_pow];
                        *(delta+i) = 1./cutoff_p 
                                    *std::pow(relative_coord.x/cutoff_p,x_pow)/alphaf;
                    }
                    i++;
            }
        }
    } else {
        compadre_kernel_assert_release((false) && "Sampling and basis space combination not defined.");
    }
}


KOKKOS_INLINE_FUNCTION
void GMLS::createWeightsAndP(const member_type& teamMember, scratch_vector_type delta, scratch_matrix_right_type P, scratch_vector_type w, const int dimension, int polynomial_order, bool weight_p, scratch_matrix_right_type* V, const ReconstructionSpace reconstruction_space, const SamplingFunctional polynomial_sampling_functional) const {
    /*
     * Creates sqrt(W)*P
     */
    const int target_index = teamMember.league_rank();
//    printf("specific order: %d\n", specific_order);
//    {
//        const int storage_size = (specific_order > 0) ? this->getNP(specific_order, dimension)-this->getNP(specific_order-1, dimension) : this->getNP(_poly_order, dimension);
//        printf("storage size: %d\n", storage_size);
//    }
//    printf("weight_p: %d\n", weight_p);
    // P is stored layout left, because that is what CUDA and LAPACK expect, and storing it
    // this way prevents copying data later
    auto alt_P = scratch_matrix_left_type(P.data(), P.extent(0), P.extent(1));
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
            if (_polynomial_sampling_functional==SamplingFunctional::StaggeredEdgeIntegralSample
                    || _polynomial_sampling_functional==SamplingFunctional::StaggeredEdgeAnalyticGradientIntegralSample) {
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

            this->calcPij(delta.data(), target_index, i + d*my_num_neighbors, 0 /*alpha*/, dimension, polynomial_order, false /*bool on only specific order*/, V, reconstruction_space, polynomial_sampling_functional);

            int storage_size = this->getNP(polynomial_order, dimension);
            storage_size *= _basis_multiplier;

            if (weight_p) {
                for (int j = 0; j < storage_size; ++j) {
                    // stores layout left for CUDA or LAPACK calls later
                    // no need to convert offsets to global indices because the sum will never be large
                    alt_P(i+my_num_neighbors*d, j) = delta[j] * std::sqrt(w(i+my_num_neighbors*d));
                }

            } else {
                for (int j = 0; j < storage_size; ++j) {
                    // stores layout left for CUDA or LAPACK calls later
                    // no need to convert offsets to global indices because the sum will never be large
                    alt_P(i+my_num_neighbors*d, j) = delta[j];
                }
            }
        }
  });

    teamMember.team_barrier();
//    Kokkos::single(Kokkos::PerTeam(teamMember), [=] () {
//        for (int k=0; k<this->getNNeighbors(target_index); k++) {
//            for (int l=0; l<_NP; l++) {
//                printf("%i %i %0.16f\n", k, l, P(k,l) );// << " " << l << " " << R(k,l) << std::endl;
//            }
//        }
//    });
}

KOKKOS_INLINE_FUNCTION
void GMLS::createWeightsAndPForCurvature(const member_type& teamMember, scratch_vector_type delta, scratch_matrix_right_type P, scratch_vector_type w, const int dimension, bool only_specific_order, scratch_matrix_right_type* V) const {
/*
 * This function has two purposes
 * 1.) Used to calculate specifically for 1st order polynomials, from which we can reconstruct a tangent plane
 * 2.) Used to calculate a polynomial of _curvature_poly_order, which we use to calculate curvature of the manifold
 */

    const int target_index = teamMember.league_rank();

    // P is stored layout left, because that is what CUDA and LAPACK expect, and storing it
    // this way prevents copying data later
    auto alt_P = scratch_matrix_left_type(P.data(), P.extent(0), P.extent(1));

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
            w(i) = this->Wab(r, _epsilons(target_index), _curvature_weighting_type, _curvature_weighting_power);
            this->calcPij(delta.data(), target_index, i, 0 /*alpha*/, dimension, 1, true /*bool on only specific order*/);
        } else {
            w(i) = this->Wab(r, _epsilons(target_index), _curvature_weighting_type, _curvature_weighting_power);
            this->calcPij(delta.data(), target_index, i, 0 /*alpha*/, dimension, _curvature_poly_order, false /*bool on only specific order*/, V);
        }

        int storage_size = only_specific_order ? this->getNP(1, dimension)-this->getNP(0, dimension) : this->getNP(_curvature_poly_order, dimension);

        for (int j = 0; j < storage_size; ++j) {
            // stores layout left for CUDA or LAPACK calls later
            alt_P(i, j) = delta[j] * std::sqrt(w(i));
        }

    });
    teamMember.team_barrier();
}

}; // Compadre
#endif
