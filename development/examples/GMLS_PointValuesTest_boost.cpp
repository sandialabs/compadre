#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <stdlib.h> 
#include <cstdio>

#include "../src/GMLS_Config.h"

#ifdef COMPADRE_USE_BOOST

#include "../src/GMLS_particle_boost.hpp"
#include "../src/GMLS_boost.hpp"

typedef std::vector<double> stl_vector_type;
typedef boost::numeric::ublas::vector<double> boost_vector_type;
#endif

int main (int argc, const char * argv[])
{
#ifdef COMPADRE_USE_BOOST
    int i;
    double randx;
    double randy;
    double randz;
    double epsilon = 0.5;

    boost_vector_type my_coord(3,0);
    my_coord[0] = 0.1;
    my_coord[1] = 0.1;
    my_coord[2] = 0.1;

    boost_vector_type neighbor_coord(3,0);
    int N = 40;

    int order = 4;

    stl_vector_type tau;
    std::vector<GMLS_ParticleT_BOOST> my_neighbors;
    my_neighbors.push_back(GMLS_ParticleT_BOOST(my_coord, 0));

    int curID = 1;
    for(i = 1; i < N; i++){
        randx = (2.0*(double)rand() / (double) RAND_MAX - 1.0)*epsilon/2.0;
        randy = (2.0*(double)rand() / (double) RAND_MAX - 1.0)*epsilon/2.0;
        randz = (2.0*(double)rand() / (double) RAND_MAX - 1.0)*epsilon/2.0;
        
        neighbor_coord[0] = randx;
        neighbor_coord[1] = randy;
        neighbor_coord[2] = randz;

        my_neighbors.push_back(GMLS_ParticleT_BOOST(neighbor_coord, curID));

        curID+=1;
    }


    
    GMLS_T_BOOST my_GMLS (my_neighbors, my_neighbors[0], 0, epsilon, order);

    fprintf(stdout, "Number of neighbors: %d \n", my_GMLS.get_Nneighbors());
    //    my_GMLS.print_neighbor_data();
    
    tau = my_GMLS.get_alphaij();
    double GMLS_value = 0.0; 
    for(i = 0; i< N; i++){
      GMLS_value += tau[i]*(sin(my_neighbors[i].get_coordinates()[0]) + 2.0*cos(my_neighbors[i].get_coordinates()[1]));
    }

    tau = my_GMLS.get_Laplacian_alphaij();
    double GMLS_Laplacian = 0.0; 
    for(i = 0; i< N; i++){
      GMLS_Laplacian += tau[i]*(sin(my_neighbors[i].get_coordinates()[0]) + 2.0*cos(my_neighbors[i].get_coordinates()[1]));
    }



    double actual_value =  sin(0.1) + 2.0*cos(0.1);
    double actual_Laplacian = -sin(0.1) -2.0*cos(0.1); 
    fprintf(stdout, "Reconstructed value: %f \n", GMLS_value);
    fprintf(stdout, "Actual value: %f \n", actual_value);
    fprintf(stdout, "Reconstructed Laplacian: %f \n", GMLS_Laplacian);
    fprintf(stdout, "Actual Laplacian: %f \n", actual_Laplacian);

    if(std::abs(actual_value -GMLS_value) < 0.001)
      fprintf(stdout, "Passed value test \n");
    else {
    	fprintf(stdout, "Failed value test \n");
    	return -1;
    }

    if(std::abs(actual_Laplacian -GMLS_Laplacian) < 0.001)
      fprintf(stdout, "Passed Laplacian test \n");
    else {
    	fprintf(stdout, "Failed Laplacian test \n");
    	return -1;
    }

    return 0;
#else // boost not defined
    return -1;
#endif
};
