#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <stdlib.h> 
#include <cstdio>
#include <random>

#include "../src/GMLS_Config.h"

#if defined(COMPADRE_USE_BOOST) && defined(COMPADRE_USE_KOKKOS)
#include <Kokkos_Timer.hpp>
#include <Kokkos_Core.hpp>

#include "../src/GMLS_particle_boost.hpp"
#include "../src/GMLS_boost.hpp"

typedef std::vector<double> stl_vector_type;
typedef boost::numeric::ublas::vector<double> boost_vector_type;
#endif

double trueSolution(double x, double y, double z, int order) {
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

double trueLaplacian(double x, double y, double z, int order) {
	double ans = 0;
	for (int i=0; i<order+1; i++) {
		for (int j=0; j<order+1; j++) {
			for (int k=0; k<order+1; k++) {
				if (i+j+k <= order) {
					ans += std::max(0,i-1)*std::max(0,i)*
							std::pow(x,std::max(0,i-2))*std::pow(y,j)*std::pow(z,k);
				}
			}
		}
	}
	for (int i=0; i<order+1; i++) {
		for (int j=0; j<order+1; j++) {
			for (int k=0; k<order+1; k++) {
				if (i+j+k <= order) {
					ans += 	std::max(0,j-1)*std::max(0,j)*
							std::pow(x,i)*std::pow(y,std::max(0,j-2))*std::pow(z,k);
				}
			}
		}
	}
	for (int i=0; i<order+1; i++) {
		for (int j=0; j<order+1; j++) {
			for (int k=0; k<order+1; k++) {
				if (i+j+k <= order) {
					ans += 	std::max(0,k-1)*std::max(0,k)*
							std::pow(x,i)*std::pow(y,j)*std::pow(z,std::max(0,k-2));
				}
			}
		}
	}
//	std::cout << ans << std::endl;
	return ans;
}

int main (int argc, char* args[])
{
#if defined(COMPADRE_USE_BOOST) && defined(COMPADRE_USE_KOKKOS)

//	const double failure_tolerance = 1e-9;

    const int order = atoi(args[1]); // 9

    const int offset = 15;
	std::mt19937 rng(50);
	const int min_neighbors = 1*GMLS_T_BOOST::getNP(order);
	const int max_neighbors = 1*GMLS_T_BOOST::getNP(order)*1.15;
	std::cout << min_neighbors << " " << max_neighbors << std::endl;
	std::uniform_int_distribution<int> gen_num_neighbors(min_neighbors, max_neighbors); // uniform, unbiased


	Kokkos::initialize(argc, args);
	Kokkos::Timer timer;


    const int number_target_coords = atoi(args[2]); // 200
    const int N = 40000;
    std::uniform_int_distribution<int> gen_neighbor_number(offset, N); // 0 to 10 are junk (part of test)


	Kokkos::View<int**, Kokkos::HostSpace>	neighbor_lists("neighbor lists", number_target_coords, max_neighbors+1); // first column is # of neighbors
	Kokkos::View<double**, Kokkos::HostSpace> source_coords("neighbor coordinates", N, 3);
	Kokkos::View<double*, Kokkos::HostSpace> epsilon("h supports", number_target_coords);

	for (int i=0; i<number_target_coords; i++) {
		epsilon(i) = 0.5;
	}

//    // fake coordinates not to be used
	for(int i = 0; i < offset; i++){
		source_coords(i,0) = 0.1;
		source_coords(i,1) = 0.1;
		source_coords(i,2) = 0.1;
	}

    // filling others with random coordinates
    for(int i = offset; i < N; i++){ //ignore first ten entries
    	double randx = (2.0*(double)rand() / (double) RAND_MAX - 1.0)*epsilon(0)/2.0;
    	double randy = (2.0*(double)rand() / (double) RAND_MAX - 1.0)*epsilon(0)/2.0;
    	double randz = (2.0*(double)rand() / (double) RAND_MAX - 1.0)*epsilon(0)/2.0;
        source_coords(i,0) = randx;
        source_coords(i,1) = randy;
        source_coords(i,2) = randz;
    }

    const double target_epsilon = 0.1;
    // fill target coords
    Kokkos::View<double**, Kokkos::HostSpace> target_coords ("target coordinates", number_target_coords, 3);
    for(int i = 0; i < number_target_coords; i++){ //ignore first ten entries
    	double randx = (2.0*(double)rand() / (double) RAND_MAX - 1.0)*target_epsilon/2.0;
    	double randy = (2.0*(double)rand() / (double) RAND_MAX - 1.0)*target_epsilon/2.0;
    	double randz = (2.0*(double)rand() / (double) RAND_MAX - 1.0)*target_epsilon/2.0;
    	target_coords(i,0) = randx;
    	target_coords(i,1) = randy;
    	target_coords(i,2) = randz;
    }






//    my_GMLS.generateAlphas();
//    my_GMLS.generateLaplacianAlphas();


//    fprintf(stdout, "Number of neighbors: %d \n", my_GMLS.getNNeighbors(0));
//    my_GMLS.printNeighborData(0);
    
    bool all_passed = true;

    double total_timer_time = 0;

    timer.reset();
    Kokkos::parallel_for(number_target_coords, KOKKOS_LAMBDA(const int i) {
//    for (int i=0; i<number_target_coords; i++) {


        boost_vector_type my_coord(3,0);
        my_coord[0] = target_coords(i,0);
        my_coord[1] = target_coords(i,1);
        my_coord[2] = target_coords(i,2);

        GMLS_ParticleT_BOOST my_particle(my_coord, 0);

        std::vector<GMLS_ParticleT_BOOST> my_neighbors;
			// randomly fill neighbor lists
//			for (int k=0;k<number_target_coords; k++) {
		//    	int r = gen_num_neighbors(rng);
		//    	assert(r<source_coords.dimension_0()-offset);
				int r = max_neighbors;
		//    	neighbor_lists(i,0) = r; // number of neighbors is random between max and min

				for(int j=0; j<r; j++){
					neighbor_lists(i,j+1) = offset + j + 1;


					boost_vector_type neighbor_coord(3,0);
					neighbor_coord[0] = source_coords(offset + j +1, 0);
					neighbor_coord[1] = source_coords(offset + j +1, 1);
					neighbor_coord[2] = source_coords(offset + j +1, 2);

					my_neighbors.push_back(GMLS_ParticleT_BOOST(neighbor_coord, j));
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
//			}


		GMLS_T_BOOST my_GMLS(my_neighbors, my_particle, 0, 0.5, order);

	    stl_vector_type tau;
		tau = my_GMLS.get_alphaij();

//		double GMLS_value = 0.0;
//		for (int j = 0; j< tau.size(); j++){
////		  GMLS_value += tau[j]*(sin(source_coords(neighbor_lists(i,j+1),0)) + 2.0*cos(source_coords(neighbor_lists(i,j+1),1)));
//		  GMLS_value += tau[j]*trueSolution(source_coords(neighbor_lists(i,j+1),0), source_coords(neighbor_lists(i,j+1),1), source_coords(neighbor_lists(i,j+1),2), order);
//		}

////		timer.reset();
		tau = my_GMLS.get_Laplacian_alphaij();
////		instantiation_time = timer.seconds();
////		std::cout << "Took " << instantiation_time << "s to complete Laplacian." << std::endl;
//		double GMLS_Laplacian = 0.0;
//		for (int j = 0; j< my_GMLS.getNNeighbors(i); j++){
////		  GMLS_Laplacian += tau[j]*(sin(source_coords(neighbor_lists(i,j+1),0)) + 2.0*cos(source_coords(neighbor_lists(i,j+1),1)));
////		  GMLS_Laplacian += tau[j]*(sin(source_coords(neighbor_lists(i,j+1),0)) + 2.0*cos(source_coords(neighbor_lists(i,j+1),1)));
//		  GMLS_Laplacian += tau[j]*trueLaplacian(source_coords(neighbor_lists(i,j+1),0), source_coords(neighbor_lists(i,j+1),1), source_coords(neighbor_lists(i,j+1),2), order);
////		  std::cout << tau[j] << std::endl;
//		}

//		double actual_value =  0;//trueSolution(target_coords(i,0), target_coords(i,1), target_coords(i,2), order);
//		double actual_Laplacian = trueLaplacian(target_coords(i,0), target_coords(i,1), target_coords(i,2), order);

//		double actual_value =  sin(target_coords(i,0)) + 2.0*cos(target_coords(i,1));
//		double actual_Laplacian = -sin(target_coords(i,0)) -2.0*cos(target_coords(i,1));

//		fprintf(stdout, "Reconstructed value: %f \n", GMLS_value);
//		fprintf(stdout, "Actual value: %f \n", actual_value);
//		fprintf(stdout, "Reconstructed Laplacian: %f \n", GMLS_Laplacian);
//		fprintf(stdout, "Actual Laplacian: %f \n", actual_Laplacian);

//		if(std::abs(actual_value - GMLS_value) > failure_tolerance) {
//			all_passed = false;
//			std::cout << "Failed by: " << std::abs(actual_value - GMLS_value) << std::endl;
//		}

//	    if(std::abs(actual_Laplacian - GMLS_Laplacian) > failure_tolerance) {
//	    		all_passed = false;
//	    }
    });
    total_timer_time += timer.seconds();

    std::cout << "Took " << total_timer_time << "s to complete instantiation." << std::endl;

    Kokkos::finalize();



    if(all_passed)
      fprintf(stdout, "Passed test \n");
    else {
    	fprintf(stdout, "Failed test \n");
    	return -1;
    }


    return 0;
#else // Kokkos or Boost not defined
    return -1;
#endif
};
