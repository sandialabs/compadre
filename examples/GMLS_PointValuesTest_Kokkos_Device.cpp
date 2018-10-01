#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <stdlib.h> 
#include <cstdio>
#include <random>

#include "GMLS_Config.h"

#ifdef COMPADRE_USE_KOKKOSCORE
#include "GMLS.hpp"
#include <Kokkos_Timer.hpp>
#include <Kokkos_Core.hpp>

typedef std::vector<double> stl_vector_type;
#endif

KOKKOS_INLINE_FUNCTION
double device_max(double d1, double d2) {
	return (d1 > d2) ? d1 : d2;
}

double trueSolution(double x, double y, double z, int order, int dimension) {
	double ans = 0;
	for (int i=0; i<order+1; i++) {
		for (int j=0; j<order+1; j++) {
			for (int k=0; k<order+1; k++) {
				if (i+j+k <= order) {
					ans += std::pow(x,i)*std::pow(y,j)*std::pow(z,k);
				}
			}
		}
	}
	return ans;
}

double trueLaplacian(double x, double y, double z, int order, int dimension) {
	double ans = 0;
	for (int i=0; i<order+1; i++) {
		for (int j=0; j<order+1; j++) {
			for (int k=0; k<order+1; k++) {
				if (i+j+k <= order) {
					ans += device_max(0,i-1)*device_max(0,i)*
							std::pow(x,device_max(0,i-2))*std::pow(y,j)*std::pow(z,k);
				}
			}
		}
	}
	if (dimension>1) {
		for (int i=0; i<order+1; i++) {
			for (int j=0; j<order+1; j++) {
				for (int k=0; k<order+1; k++) {
					if (i+j+k <= order) {
						ans += 	device_max(0,j-1)*device_max(0,j)*
								std::pow(x,i)*std::pow(y,device_max(0,j-2))*std::pow(z,k);
					}
				}
			}
		}
	}
	if (dimension>2) {
		for (int i=0; i<order+1; i++) {
			for (int j=0; j<order+1; j++) {
				for (int k=0; k<order+1; k++) {
					if (i+j+k <= order) {
						ans += 	device_max(0,k-1)*device_max(0,k)*
								std::pow(x,i)*std::pow(y,j)*std::pow(z,device_max(0,k-2));
					}
				}
			}
		}
	}
//	std::cout << ans << std::endl;
	return ans;
}

std::vector<double> trueGradient(double x, double y, double z, int order, int dimension) {
	std::vector<double> ans(3,0);

	for (int i=0; i<order+1; i++) {
		for (int j=0; j<order+1; j++) {
			for (int k=0; k<order+1; k++) {
				if (i+j+k <= order) {
					ans[0] += device_max(0,i)*
							std::pow(x,device_max(0,i-1))*std::pow(y,j)*std::pow(z,k);
				}
			}
		}
	}
	if (dimension>1) {
		for (int i=0; i<order+1; i++) {
			for (int j=0; j<order+1; j++) {
				for (int k=0; k<order+1; k++) {
					if (i+j+k <= order) {
						ans[1] += device_max(0,j)*
								std::pow(x,i)*std::pow(y,device_max(0,j-1))*std::pow(z,k);
					}
				}
			}
		}
	}
	if (dimension>2) {
		for (int i=0; i<order+1; i++) {
			for (int j=0; j<order+1; j++) {
				for (int k=0; k<order+1; k++) {
					if (i+j+k <= order) {
						ans[2] += device_max(0,k)*
								std::pow(x,i)*std::pow(y,j)*std::pow(z,device_max(0,k-1));
					}
				}
			}
		}
	}
//	std::cout << ans << std::endl;
	return ans;
}

double trueDivergence(double x, double y, double z, int order, int dimension) {
	double ans = 0;
	for (int i=0; i<order+1; i++) {
		for (int j=0; j<order+1; j++) {
			for (int k=0; k<order+1; k++) {
				if (i+j+k <= order) {
					ans += device_max(0,i)*
							std::pow(x,device_max(0,i-1))*std::pow(y,j)*std::pow(z,k);
				}
			}
		}
	}
	if (dimension>1) {
		for (int i=0; i<order+1; i++) {
			for (int j=0; j<order+1; j++) {
				for (int k=0; k<order+1; k++) {
					if (i+j+k <= order) {
						ans += device_max(0,j)*
								std::pow(x,i)*std::pow(y,device_max(0,j-1))*std::pow(z,k);
					}
				}
			}
		}
	}
	if (dimension>2) {
		for (int i=0; i<order+1; i++) {
			for (int j=0; j<order+1; j++) {
				for (int k=0; k<order+1; k++) {
					if (i+j+k <= order) {
						ans += device_max(0,k)*
								std::pow(x,i)*std::pow(y,j)*std::pow(z,device_max(0,k-1));
					}
				}
			}
		}
	}
//	std::cout << ans << std::endl;
	return ans;
}

double divergenceTestSamples(double x, double y, double z, int component, int dimension) {
	// solution can be captured exactly by at least 2rd order
	switch (component) {
	case 0:
		return x*x + y*y - z*z;
	case 1:
		return 2*x +3*y + 4*z;
	default:
		return -21*x*y + 3*z*x*y + 4*z;
	}
}

double divergenceTestSolution(double x, double y, double z, int dimension) {
	switch (dimension) {
	case 1:
		// returns divergence of divergenceTestSamples
		return 2*x;
	case 2:
		return 2*x + 3;
	default:
		return 2*x + 3 + 3*x*y + 4;
	}
}

double curlTestSolution(double x, double y, double z, int component, int dimension) {
	if (dimension==3) {
		// returns curl of divergenceTestSamples
		switch (component) {
		case 0:
			// u3,y- u2,z
			return (-21*x + 3*z*x) - 4;
		case 1:
			// -u3,x + u1,z
			return (21*y - 3*z*y) - 2*z;
		default:
			// u2,x - u1,y
			return 2 - 2*y;
		}
	} else if (dimension==2) {
		switch (component) {
		case 0:
			// u2,y
			return 3;
		default:
			// -u1,x
			return -2*x;
		}
	} else {
		return 0;
	}
}



int main (int argc, char* args[])
{
#if defined(COMPADRE_USE_KOKKOSCORE)

{
	int solver_type = 0;
	if (argc >= 5) {
		int arg5toi = atoi(args[4]);
		if (arg5toi > 0) {
			solver_type = arg5toi;
		}
	}

	int dimension = 3;
	if (argc >= 4) {
		int arg4toi = atoi(args[3]);
		if (arg4toi > 0) {
			dimension = arg4toi;
		}
	}

	bool use_arbitrary_order_divergence = true;

	const double failure_tolerance = 1e-9;

    const int order = atoi(args[1]); // 9

    const int offset = 15;
	std::mt19937 rng(50);
	const int min_neighbors = 1*GMLS::getNP(order);
	const int max_neighbors = 1*GMLS::getNP(order)*1.15;
	std::cout << min_neighbors << " " << max_neighbors << std::endl;
	std::uniform_int_distribution<int> gen_num_neighbors(min_neighbors, max_neighbors); // uniform, unbiased


	Kokkos::initialize(argc, args);
	Kokkos::Timer timer;


    const int number_target_coords = atoi(args[2]); // 200
    const int N = 40000;
    std::uniform_int_distribution<int> gen_neighbor_number(offset, N); // 0 to 10 are junk (part of test)


    stl_vector_type tau;

	Kokkos::View<int**, Kokkos::DefaultExecutionSpace>	neighbor_lists_device("neighbor lists", number_target_coords, max_neighbors+1); // first column is # of neighbors
	Kokkos::View<int**>::HostMirror neighbor_lists = Kokkos::create_mirror_view(neighbor_lists_device);

	Kokkos::View<double**, Kokkos::DefaultExecutionSpace> source_coords_device("neighbor coordinates", N, dimension);
	Kokkos::View<double**>::HostMirror source_coords = Kokkos::create_mirror_view(source_coords_device);

	Kokkos::View<double*, Kokkos::DefaultExecutionSpace> epsilon_device("h supports", number_target_coords);
	Kokkos::View<double*>::HostMirror epsilon = Kokkos::create_mirror_view(epsilon_device);


	for (int i=0; i<number_target_coords; i++) {
		epsilon(i) = 0.5;
	}

//    // fake coordinates not to be used
	for(int i = 0; i < offset; i++){
		for(int j = 0; j < dimension; j++){
			source_coords(i,j) = 0.1;
		}
	}

    // filling others with random coordinates
    for(int i = offset; i < N; i++){ //ignore first ten entries
    	double randx = (2.0*(double)rand() / (double) RAND_MAX - 1.0)*epsilon(0)/2.0;
    	double randy = (2.0*(double)rand() / (double) RAND_MAX - 1.0)*epsilon(0)/2.0;
    	double randz = (2.0*(double)rand() / (double) RAND_MAX - 1.0)*epsilon(0)/2.0;
        source_coords(i,0) = randx;
        if (dimension>1) source_coords(i,1) = randy;
        if (dimension>2) source_coords(i,2) = randz;
    }

    const double target_epsilon = 0.1;
    // fill target coords

    Kokkos::View<double**, Kokkos::DefaultExecutionSpace> target_coords_device ("target coordinates", number_target_coords, dimension);
    Kokkos::View<double**>::HostMirror target_coords = Kokkos::create_mirror_view(target_coords_device);

    for(int i = 0; i < number_target_coords; i++){ //ignore first ten entries
    	double randx = (2.0*(double)rand() / (double) RAND_MAX - 1.0)*target_epsilon/2.0;
    	double randy = (2.0*(double)rand() / (double) RAND_MAX - 1.0)*target_epsilon/2.0;
    	double randz = (2.0*(double)rand() / (double) RAND_MAX - 1.0)*target_epsilon/2.0;
    	target_coords(i,0) = randx;
    	if (dimension>1) target_coords(i,1) = randy;
    	if (dimension>2) target_coords(i,2) = randz;
    }

    // randomly fill neighbor lists
    for (int i=0; i<number_target_coords; i++) {
//    	int r = gen_num_neighbors(rng);
//    	assert(r<source_coords.dimension_0()-offset);
    	int r = max_neighbors;
    	neighbor_lists(i,0) = r; // number of neighbors is random between max and min

		for(int j=0; j<r; j++){
			neighbor_lists(i,j+1) = offset + j + 1;
//			bool placed = false;
//			while (!placed) {
//				int ind = gen_neighbor_number(rng);
//				bool located = false;
//				for (int k=1; k<j+1; k++) {
//					if (neighbor_lists(i,k) == ind) {
//						located = true;
//						break;
//					}
//				}
//				if (!located) {
//					neighbor_lists(i,j+1) = ind;
//					placed = true;
//				} // neighbors can be from  10,...,N-1
//			}
		}
    }

    timer.reset();

    // Copy data back to device
    Kokkos::deep_copy(neighbor_lists_device, neighbor_lists);
    Kokkos::deep_copy(source_coords_device, source_coords);
    Kokkos::deep_copy(target_coords_device, target_coords);
    Kokkos::deep_copy(epsilon_device, epsilon);

    std::string solver_name;
    if (solver_type == 0) { // SVD
        solver_name = "SVD";
    } else if (solver_type == 1) { // QR
        solver_name = "QR";
    } else if (solver_type == 2) { // LU
        solver_name = "LU";
    }

    GMLS my_GMLS(order, solver_name.c_str(), 2 /*manifield order*/, dimension);
    my_GMLS.setProblemData(neighbor_lists_device, source_coords_device, target_coords_device, epsilon_device);

    std::vector<ReconstructionOperator::TargetOperation> lro(5);
    lro[0] = ReconstructionOperator::ScalarPointEvaluation;
    lro[1] = ReconstructionOperator::LaplacianOfScalarPointEvaluation;
    lro[2] = ReconstructionOperator::GradientOfScalarPointEvaluation;
    lro[3] = ReconstructionOperator::DivergenceOfVectorPointEvaluation;
    lro[4] = ReconstructionOperator::CurlOfVectorPointEvaluation;
    my_GMLS.addTargets(lro, dimension);
    my_GMLS.generateAlphas();

    double instantiation_time = timer.seconds();
    std::cout << "Took " << instantiation_time << "s to complete instantiation." << std::endl;

    bool all_passed = true;

    for (int i=0; i<number_target_coords; i++) {

		double GMLS_value = 0.0;
		for (int j = 0; j< neighbor_lists(i,0); j++){
			double xval = source_coords(neighbor_lists(i,j+1),0);
			double yval = (dimension>1) ? source_coords(neighbor_lists(i,j+1),1) : 0;
			double zval = (dimension>2) ? source_coords(neighbor_lists(i,j+1),2) : 0;
			GMLS_value += my_GMLS.getAlpha0TensorTo0Tensor(ReconstructionOperator::ScalarPointEvaluation, i, j)*trueSolution(xval, yval, zval, order, dimension);
		}

		double GMLS_Laplacian = 0.0;
		for (int j = 0; j< neighbor_lists(i,0); j++){
			double xval = source_coords(neighbor_lists(i,j+1),0);
			double yval = (dimension>1) ? source_coords(neighbor_lists(i,j+1),1) : 0;
			double zval = (dimension>2) ? source_coords(neighbor_lists(i,j+1),2) : 0;
			GMLS_Laplacian += my_GMLS.getAlpha0TensorTo0Tensor(ReconstructionOperator::LaplacianOfScalarPointEvaluation, i, j)*trueSolution(xval, yval, zval, order, dimension);
		}

		double GMLS_GradX = 0.0;
		for (int j = 0; j< neighbor_lists(i,0); j++){
			double xval = source_coords(neighbor_lists(i,j+1),0);
			double yval = (dimension>1) ? source_coords(neighbor_lists(i,j+1),1) : 0;
			double zval = (dimension>2) ? source_coords(neighbor_lists(i,j+1),2) : 0;
			GMLS_GradX += my_GMLS.getAlpha0TensorTo1Tensor(ReconstructionOperator::GradientOfScalarPointEvaluation, i, 0, j)*trueSolution(xval, yval, zval, order, dimension);
		}

		double GMLS_GradY = 0.0;
		if (dimension>1) {
			for (int j = 0; j< neighbor_lists(i,0); j++){
				double xval = source_coords(neighbor_lists(i,j+1),0);
				double yval = source_coords(neighbor_lists(i,j+1),1);
				double zval = (dimension>2) ? source_coords(neighbor_lists(i,j+1),2) : 0;
				GMLS_GradY += my_GMLS.getAlpha0TensorTo1Tensor(ReconstructionOperator::GradientOfScalarPointEvaluation, i, 1, j)*trueSolution(xval, yval, zval, order, dimension);
			}
		}

		double GMLS_GradZ = 0.0;
		if (dimension>2) {
			for (int j = 0; j< neighbor_lists(i,0); j++){
				double xval = source_coords(neighbor_lists(i,j+1),0);
				double yval = source_coords(neighbor_lists(i,j+1),1);
				double zval = source_coords(neighbor_lists(i,j+1),2);
				GMLS_GradZ += my_GMLS.getAlpha0TensorTo1Tensor(ReconstructionOperator::GradientOfScalarPointEvaluation, i, 2, j)*trueSolution(xval, yval, zval, order, dimension);
			}
		}

		double GMLS_Divergence = 0.0;
		for (int j = 0; j< neighbor_lists(i,0); j++){
			double xval = source_coords(neighbor_lists(i,j+1),0);
			double yval = (dimension>1) ? source_coords(neighbor_lists(i,j+1),1) : 0;
			double zval = (dimension>2) ? source_coords(neighbor_lists(i,j+1),2) : 0;
			// TODO: use different functions for the vector components
			if (use_arbitrary_order_divergence) {
				GMLS_Divergence += my_GMLS.getAlpha1TensorTo0Tensor(ReconstructionOperator::DivergenceOfVectorPointEvaluation, i, j, 0)*trueSolution(xval, yval, zval, order, dimension);
				if (dimension>1) GMLS_Divergence += my_GMLS.getAlpha1TensorTo0Tensor(ReconstructionOperator::DivergenceOfVectorPointEvaluation, i, j, 1)*trueSolution(xval, yval, zval, order, dimension);
				if (dimension>2) GMLS_Divergence += my_GMLS.getAlpha1TensorTo0Tensor(ReconstructionOperator::DivergenceOfVectorPointEvaluation, i, j, 2)*trueSolution(xval, yval, zval, order, dimension);
			} else {
				for (int k=0; k<dimension; ++k) {
					GMLS_Divergence += my_GMLS.getAlpha1TensorTo0Tensor(ReconstructionOperator::DivergenceOfVectorPointEvaluation, i, j, k)*divergenceTestSamples(xval, yval, zval, k, dimension);
				}
			}
		}

		double GMLS_CurlX = 0.0;
		double GMLS_CurlY = 0.0;
		double GMLS_CurlZ = 0.0;

		if (dimension>1) {
			for (int j = 0; j< neighbor_lists(i,0); j++){
				double xval = source_coords(neighbor_lists(i,j+1),0);
				double yval = source_coords(neighbor_lists(i,j+1),1);
				double zval = (dimension>2) ? source_coords(neighbor_lists(i,j+1),2) : 0;
				for (int k=0; k<dimension; ++k) {
					GMLS_CurlX += my_GMLS.getAlpha1TensorTo1Tensor(ReconstructionOperator::CurlOfVectorPointEvaluation, i, 0, j, k)*divergenceTestSamples(xval, yval, zval, k, dimension);
				}
			}

			for (int j = 0; j< neighbor_lists(i,0); j++){
				double xval = source_coords(neighbor_lists(i,j+1),0);
				double yval = source_coords(neighbor_lists(i,j+1),1);
				double zval = (dimension>2) ? source_coords(neighbor_lists(i,j+1),2) : 0;
				for (int k=0; k<dimension; ++k) {
					GMLS_CurlY += my_GMLS.getAlpha1TensorTo1Tensor(ReconstructionOperator::CurlOfVectorPointEvaluation, i, 1, j, k)*divergenceTestSamples(xval, yval, zval, k, dimension);
				}
			}
		}

		if (dimension>2) {
			for (int j = 0; j< neighbor_lists(i,0); j++){
				double xval = source_coords(neighbor_lists(i,j+1),0);
				double yval = source_coords(neighbor_lists(i,j+1),1);
				double zval = source_coords(neighbor_lists(i,j+1),2);
				for (int k=0; k<dimension; ++k) {
					GMLS_CurlZ += my_GMLS.getAlpha1TensorTo1Tensor(ReconstructionOperator::CurlOfVectorPointEvaluation, i, 2, j, k)*divergenceTestSamples(xval, yval, zval, k, dimension);
				}
			}
		}

		double xval = target_coords(i,0);
		double yval = (dimension>1) ? target_coords(i,1) : 0;
		double zval = (dimension>2) ? target_coords(i,2) : 0;

		double actual_value = trueSolution(xval, yval, zval, order, dimension);
		double actual_Laplacian = trueLaplacian(xval, yval, zval, order, dimension);
		std::vector<double> actual_Gradient = trueGradient(xval, yval, zval, order, dimension);
		double actual_Divergence;
		if (use_arbitrary_order_divergence) actual_Divergence = trueDivergence(xval, yval, zval, order, dimension);
		else actual_Divergence = divergenceTestSolution(xval, yval, zval, dimension);

		double actual_CurlX = 0;
		double actual_CurlY = 0;
		double actual_CurlZ = 0;
		if (dimension>1) {
			actual_CurlX = curlTestSolution(xval, yval, zval, 0, dimension);
			actual_CurlY = curlTestSolution(xval, yval, zval, 1, dimension);
		}
		if (dimension>2) {
			actual_CurlZ = curlTestSolution(xval, yval, zval, 2, dimension);
		}

//		fprintf(stdout, "Reconstructed value: %f \n", GMLS_value);
//		fprintf(stdout, "Actual value: %f \n", actual_value);
//		fprintf(stdout, "Reconstructed Laplacian: %f \n", GMLS_Laplacian);
//		fprintf(stdout, "Actual Laplacian: %f \n", actual_Laplacian);

		if(GMLS_value!=GMLS_value || std::abs(actual_value - GMLS_value) > failure_tolerance) {
			all_passed = false;
			std::cout << "Failed Actual by: " << std::abs(actual_value - GMLS_value) << std::endl;
		}

	    if(std::abs(actual_Laplacian - GMLS_Laplacian) > failure_tolerance) {
			all_passed = false;
			std::cout << "Failed Laplacian by: " << std::abs(actual_Laplacian - GMLS_Laplacian) << std::endl;
	    }

	    if(std::abs(actual_Gradient[0] - GMLS_GradX) > failure_tolerance) {
			all_passed = false;
			std::cout << "Failed GradX by: " << std::abs(actual_Gradient[0] - GMLS_GradX) << std::endl;
	    }

	    if (dimension>1) {
			if(std::abs(actual_Gradient[1] - GMLS_GradY) > failure_tolerance) {
				all_passed = false;
				std::cout << "Failed GradY by: " << std::abs(actual_Gradient[1] - GMLS_GradY) << std::endl;
			}
	    }

	    if (dimension>2) {
			if(std::abs(actual_Gradient[2] - GMLS_GradZ) > failure_tolerance) {
				all_passed = false;
				std::cout << "Failed GradZ by: " << std::abs(actual_Gradient[2] - GMLS_GradZ) << std::endl;
			}
	    }

	    if(std::abs(actual_Divergence - GMLS_Divergence) > failure_tolerance) {
			all_passed = false;
			std::cout << "Failed Divergence by: " << std::abs(actual_Divergence - GMLS_Divergence) << std::endl;
	    }

	    double tmp_diff = 0;
	    if (dimension>1)
	    	tmp_diff += std::abs(actual_CurlX - GMLS_CurlX) + std::abs(actual_CurlY - GMLS_CurlY);
	    if (dimension>2)
	    	tmp_diff += std::abs(actual_CurlZ - GMLS_CurlZ);
	    if(std::abs(tmp_diff) > failure_tolerance) {
			all_passed = false;
			std::cout << "Failed Curl by: " << std::abs(tmp_diff) << std::endl;
	    }
    }

    if(all_passed)
    	fprintf(stdout, "Passed test \n");
    else {
    	fprintf(stdout, "Failed test \n");
    	return -1;
    }
}

    Kokkos::finalize();
    return 0;
#else // Kokkos
    return -1;
#endif
};
