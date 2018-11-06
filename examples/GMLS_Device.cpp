#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <stdlib.h> 
#include <cstdio>
#include <random>

#include <Compadre_Config.h>
#include <Compadre_GMLS.hpp>
#include "GMLS_Tutorial.hpp"

#ifdef COMPADRE_USE_MPI
#include <mpi.h>
#endif

#include <Kokkos_Timer.hpp>
#include <Kokkos_Core.hpp>

using namespace Compadre;

int main (int argc, char* args[])
{


//! [Parse Command Line Arguments]

// initializes MPI (if available) with command line arguments given
#ifdef COMPADRE_USE_MPI
MPI_Init(&argc, &args);
#endif

// initializes Kokkos with command line arguments given
Kokkos::initialize(argc, args);

// check if 5 arguments are given from the command line, the first being the program name
//  solver_type used for factorization in solving each GMLS problem:
//      0 - SVD used for factorization in solving each GMLS problem
//      1 - QR  used for factorization in solving each GMLS problem
int solver_type = 1; // QR by default
if (argc >= 5) {
    int arg5toi = atoi(args[4]);
    if (arg5toi > 0) {
        solver_type = arg5toi;
    }
}

// check if 4 arguments are given from the command line
//  dimension for the coordinates and the solution
int dimension = 3; // dimension 3 by default
if (argc >= 4) {
    int arg4toi = atoi(args[3]);
    if (arg4toi > 0) {
        dimension = arg4toi;
    }
}

// check if 3 arguments are given from the command line
//  set the number of target sites where we will reconstruct the target functionals at
int number_target_coords = 200; // 200 target sites by default
if (argc >= 3) {
    int arg3toi = atoi(args[2]);
    if (arg3toi > 0) {
        number_target_coords = arg3toi;
    }
}

// check if 2 arguments are given from the command line
//  set the number of target sites where we will reconstruct the target functionals at
int order = 3; // 3rd degree polynomial basis by default
if (argc >= 2) {
    int arg2toi = atoi(args[1]);
    if (arg2toi > 0) {
        order = arg2toi;
    }
}

// the functions we will be seeking to reconstruct are in the span of the basis
// of the reconstruction space we choose for GMLS, so the error should be very small
const double failure_tolerance = 1e-8;

// seed the random generator
std::mt19937 rng(50);

// minimum neighbors for unisolvency is the same as the size of the polynomial basis 
const int min_neighbors = Compadre::GMLS::getNP(order, dimension);

// maximum neighbors calculated to be the minimum number of neighbors needed for unisolvency 
// enlarged by some multiplier (generally 10% to 40%)
const int max_neighbors = Compadre::GMLS::getNP(order, dimension)*1.33;

// uniform distribution ranging from min_neighbors to max_neighbors allowing for each
// target site to have a different number of neighbors
std::uniform_int_distribution<int> gen_num_neighbors(min_neighbors, max_neighbors);

//! [Parse Command Line Arguments]
Kokkos::Timer timer;
Kokkos::Profiling::pushRegion("Setup");
//! [Setting Up The Point Cloud]

// number of source coordinate sites that is reasonable
const int number_source_coords = max_neighbors*number_target_coords;

// these are indices into rows of the source coordinates 2d array
std::uniform_int_distribution<int> gen_neighbor_number(0, number_source_coords-1); 

// each row is a neighbor list for a target site, with the first column of each row containing
// the number of neighbors for that rows corresponding target site
Kokkos::View<int**, Kokkos::DefaultExecutionSpace> neighbor_lists_device("neighbor lists", 
        number_target_coords, max_neighbors+1); // first column is # of neighbors
Kokkos::View<int**>::HostMirror neighbor_lists = Kokkos::create_mirror_view(neighbor_lists_device);

// coordinates of source sites
Kokkos::View<double**, Kokkos::DefaultExecutionSpace> source_coords_device("source coordinates", 
        number_source_coords, dimension);
Kokkos::View<double**>::HostMirror source_coords = Kokkos::create_mirror_view(source_coords_device);

// each target site has a window size
Kokkos::View<double*, Kokkos::DefaultExecutionSpace> epsilon_device("h supports", number_target_coords);
Kokkos::View<double*>::HostMirror epsilon = Kokkos::create_mirror_view(epsilon_device);

// coordinates of target sites
Kokkos::View<double**, Kokkos::DefaultExecutionSpace> target_coords_device ("target coordinates", number_target_coords, dimension);
Kokkos::View<double**>::HostMirror target_coords = Kokkos::create_mirror_view(target_coords_device);

// set a window size for each target site that is 0.5
for (int i=0; i<number_target_coords; i++) {
    epsilon(i) = 0.5;
}

// filling source sites with random coordinates
for(int i=0; i<number_source_coords; i++){

    // rand() produces random number between 0 and RAND_MAX
    // these values are all between [0.0,0.25]
    double randx = (2.0*(double)rand() / (double) RAND_MAX - 1.0)*epsilon(0)/2.0;
    double randy = (2.0*(double)rand() / (double) RAND_MAX - 1.0)*epsilon(0)/2.0;
    double randz = (2.0*(double)rand() / (double) RAND_MAX - 1.0)*epsilon(0)/2.0;

    source_coords(i,0) = randx;
    if (dimension>1) source_coords(i,1) = randy;
    if (dimension>2) source_coords(i,2) = randz;
}

// target_epsilon ensures that all targets are in a tighter ball than the source sites
const double target_epsilon = 0.1;

// fill target coords
for(int i=0; i<number_target_coords; i++){ //ignore first ten entries

    // rand() produces random number between 0 and RAND_MAX
    // these values are all between [0.0,0.05]
    double randx = (2.0*(double)rand() / (double) RAND_MAX - 1.0)*target_epsilon/2.0;
    double randy = (2.0*(double)rand() / (double) RAND_MAX - 1.0)*target_epsilon/2.0;
    double randz = (2.0*(double)rand() / (double) RAND_MAX - 1.0)*target_epsilon/2.0;

    target_coords(i,0) = randx;
    if (dimension>1) target_coords(i,1) = randy;
    if (dimension>2) target_coords(i,2) = randz;
}

// each row of neighbor lists is a neighbor list for the target site corresponding to that row
for (int i=0; i<number_target_coords; i++) {

    // number of neighbors is random between minumum and maximum number of neighbors
    const int num_neighbors_for_target_site_i = gen_num_neighbors(rng);

    // the number of neighbors is stored in column zero of the neighbor lists 2D array
    neighbor_lists(i,0) = num_neighbors_for_target_site_i;

    // loop over each neighbor index and fill with a value
    for(int j=0; j<num_neighbors_for_target_site_i; j++){
        // uses all source sites in order, but with overlap between adjacent target sites 
        // neighbor lists

        // these rows would normally be filled by the result of a neighbor search using
        // a kdtree
        neighbor_lists(i,j+1) = i*max_neighbors + j;
    }
}


//! [Setting Up The Point Cloud]


Kokkos::Profiling::popRegion();
timer.reset();

//! [Setting Up The GMLS Object]


// Copy data back to device (they were filled on the host)
// We could have filled Kokkos Views with memory space on the host
// and used these instead, and then the copying of data to the device
// would be performed in the GMLS class
Kokkos::deep_copy(neighbor_lists_device, neighbor_lists);
Kokkos::deep_copy(source_coords_device, source_coords);
Kokkos::deep_copy(target_coords_device, target_coords);
Kokkos::deep_copy(epsilon_device, epsilon);

// solver name for passing into the GMLS class
std::string solver_name;
if (solver_type == 0) { // SVD
    solver_name = "SVD";
} else if (solver_type == 1) { // QR
    solver_name = "QR";
}

// initialize and instance of the GMLS class 
GMLS my_GMLS(order, solver_name.c_str(), 2 /*manifield order*/, dimension);

// pass in neighbor lists, source coordinates, target coordinates, and window sizes
//
// neighbor lists have the format:
//      dimensions: (# number of target sites) X (# maximum number of neighbors for any given target + 1)
//                  the first column contains the number of neighbors for that rows corresponding target index
//
// source coordinates have the format:
//      dimensions: (# number of source sites) X (dimension)
//                  entries in the neighbor lists (integers) correspond to rows of this 2D array
//
// target coordinates have the format:
//      dimensions: (# number of target sites) X (dimension)
//                  # of target sites is same as # of rows of neighbor lists
//
my_GMLS.setProblemData(neighbor_lists_device, source_coords_device, target_coords_device, epsilon_device);

// create a vector of target operations
std::vector<TargetOperation> lro(5);
lro[0] = ScalarPointEvaluation;
lro[1] = LaplacianOfScalarPointEvaluation;
lro[2] = GradientOfScalarPointEvaluation;
lro[3] = DivergenceOfVectorPointEvaluation;
lro[4] = CurlOfVectorPointEvaluation;

// and then pass them to the GMLS class
my_GMLS.addTargets(lro);

// generate the alphas that to be combined with data for each target operation requested in lro
my_GMLS.generateAlphas();


//! [Setting Up The GMLS Object]

double instantiation_time = timer.seconds();
std::cout << "Took " << instantiation_time << "s to complete instantiation." << std::endl;

Kokkos::Profiling::pushRegion("Creating Data");

//! [Creating The Data]


// need Kokkos View storing true solution
Kokkos::View<double*, Kokkos::DefaultExecutionSpace> sampling_data_device("samples of true solution", 
        source_coords_device.dimension_0());

Kokkos::View<double**, Kokkos::DefaultExecutionSpace> gradient_sampling_data_device("samples of true gradient", 
        source_coords_device.dimension_0(), dimension);

Kokkos::View<double**, Kokkos::DefaultExecutionSpace> divergence_sampling_data_device
        ("samples of true solution for divergence test", source_coords_device.dimension_0(), dimension);

Kokkos::parallel_for("Sampling Manufactured Solutions", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>
        (0,source_coords.dimension_0()), KOKKOS_LAMBDA(const int i) {

    // coordinates of source site i
    double xval = source_coords_device(i,0);
    double yval = (dimension>1) ? source_coords_device(i,1) : 0;
    double zval = (dimension>2) ? source_coords_device(i,2) : 0;

    // data for targets with scalar input
    sampling_data_device(i) = trueSolution(xval, yval, zval, order, dimension);

    // data for targets with vector input (divergence)
    double true_grad[3] = {0,0,0};
    trueGradient(true_grad, xval, yval,zval, order, dimension);

    for (int j=0; j<dimension; ++j) {
        gradient_sampling_data_device(i,j) = true_grad[j];

        // data for target with vector input (curl)
        divergence_sampling_data_device(i,j) = divergenceTestSamples(xval, yval, zval, j, dimension);
    }

});


//! [Creating The Data]
Kokkos::fence();
Kokkos::Profiling::popRegion();
Kokkos::Profiling::pushRegion("Apply Alphas to Data");
//! [Apply GMLS Alphas To Data]


// remap is done here under a particular linear operator
// it is important to note that if you expect to use the data as a 1D view, then you should use double*
// however, if you know that the target operation will result in a 2D view (vector or matrix output),
// then you should template with double** as this is something that can not be infered from the input data
// or the target operator at compile time

auto output_value = my_GMLS.applyAlphasToDataAllComponentsAllTargetSites<double*, Kokkos::HostSpace>
        (sampling_data_device, ScalarPointEvaluation);

auto output_laplacian = my_GMLS.applyAlphasToDataAllComponentsAllTargetSites<double*, Kokkos::HostSpace>
        (sampling_data_device, LaplacianOfScalarPointEvaluation);

auto output_gradient = my_GMLS.applyAlphasToDataAllComponentsAllTargetSites<double**, Kokkos::HostSpace>
        (sampling_data_device, GradientOfScalarPointEvaluation);

auto output_divergence = my_GMLS.applyAlphasToDataAllComponentsAllTargetSites<double*, Kokkos::HostSpace>
        (gradient_sampling_data_device, DivergenceOfVectorPointEvaluation);

auto output_curl = my_GMLS.applyAlphasToDataAllComponentsAllTargetSites<double**, Kokkos::HostSpace>
        (divergence_sampling_data_device, CurlOfVectorPointEvaluation);


//! [Apply GMLS Alphas To Data]

Kokkos::Profiling::popRegion();

//! [Check That Solution Is Correct]


// becomes false if the computed solution not within the failure_threshold of the actual solution
bool all_passed = true;

// loop through the target sites
for (int i=0; i<number_target_coords; i++) {

    // load value from output
    double GMLS_value = output_value(i);

    // load laplacian from output
    double GMLS_Laplacian = output_laplacian(i);

    // load partial x from gradient
    double GMLS_GradX = output_gradient(i,0);

    // load partial y from gradient
    double GMLS_GradY = (dimension>1) ? output_gradient(i,1) : 0;

    // load partial z from gradient
    double GMLS_GradZ = (dimension>2) ? output_gradient(i,2) : 0;

    // load divergence from output
    double GMLS_Divergence = output_divergence(i);

    // load curl from output
    double GMLS_CurlX = (dimension>1) ? output_curl(i,0) : 0;
    double GMLS_CurlY = (dimension>1) ? output_curl(i,1) : 0;
    double GMLS_CurlZ = (dimension>2) ? output_curl(i,2) : 0;

    // times the Comparison in Kokkos
    Kokkos::Profiling::pushRegion("Comparison");

    // target site i's coordinate
    double xval = target_coords(i,0);
    double yval = (dimension>1) ? target_coords(i,1) : 0;
    double zval = (dimension>2) ? target_coords(i,2) : 0;

    // evaluation of various exact solutions
    double actual_value = trueSolution(xval, yval, zval, order, dimension);
    double actual_Laplacian = trueLaplacian(xval, yval, zval, order, dimension);

    double actual_Gradient[3] = {0,0,0}; // initialized for 3, but only filled up to dimension
    trueGradient(actual_Gradient, xval, yval, zval, order, dimension);

    double actual_Divergence;
    actual_Divergence = trueLaplacian(xval, yval, zval, order, dimension);

    double actual_Curl[3] = {0,0,0}; // initialized for 3, but only filled up to dimension 
    // (and not at all for dimimension = 1)
    double actual_CurlX = 0;
    double actual_CurlY = 0;
    double actual_CurlZ = 0;
    if (dimension>1) {
        actual_Curl[0] = curlTestSolution(xval, yval, zval, 0, dimension);
        actual_Curl[1] = curlTestSolution(xval, yval, zval, 1, dimension);
        if (dimension>2) {
            actual_Curl[2] = curlTestSolution(xval, yval, zval, 2, dimension);
        }
    }

    // check actual function value
    if(GMLS_value!=GMLS_value || std::abs(actual_value - GMLS_value) > failure_tolerance) {
        all_passed = false;
        std::cout << i << " Failed Actual by: " << std::abs(actual_value - GMLS_value) << std::endl;
    }

    // check Laplacian
    if(std::abs(actual_Laplacian - GMLS_Laplacian) > failure_tolerance) {
        all_passed = false;
        std::cout << i << " Failed Laplacian by: " << std::abs(actual_Laplacian - GMLS_Laplacian) << std::endl;
    }

    // check gradient
    if(std::abs(actual_Gradient[0] - GMLS_GradX) > failure_tolerance) {
        all_passed = false;
        std::cout << i << " Failed GradX by: " << std::abs(actual_Gradient[0] - GMLS_GradX) << std::endl;
        if (dimension>1) {
            if(std::abs(actual_Gradient[1] - GMLS_GradY) > failure_tolerance) {
                all_passed = false;
                std::cout << i << " Failed GradY by: " << std::abs(actual_Gradient[1] - GMLS_GradY) << std::endl;
            }
        }
        if (dimension>2) {
            if(std::abs(actual_Gradient[2] - GMLS_GradZ) > failure_tolerance) {
                all_passed = false;
                std::cout << i << " Failed GradZ by: " << std::abs(actual_Gradient[2] - GMLS_GradZ) << std::endl;
            }
        }
    }

    // check divergence
    if(std::abs(actual_Divergence - GMLS_Divergence) > failure_tolerance) {
        all_passed = false;
        std::cout << i << " Failed Divergence by: " << std::abs(actual_Divergence - GMLS_Divergence) << std::endl;
    }

    // check curl
    double tmp_diff = 0;
    if (dimension>1)
        tmp_diff += std::abs(actual_Curl[0] - GMLS_CurlX) + std::abs(actual_Curl[1] - GMLS_CurlY);
    if (dimension>2)
        tmp_diff += std::abs(actual_Curl[2] - GMLS_CurlZ);
    if(std::abs(tmp_diff) > failure_tolerance) {
        all_passed = false;
        std::cout << i << " Failed Curl by: " << std::abs(tmp_diff) << std::endl;
    }

    // ends timer
    Kokkos::Profiling::popRegion();
}

// finalize Kokkos and MPI (if available)
Kokkos::finalize();
#ifdef COMPADRE_USE_MPI
MPI_Finalize();
#endif

// output to user that test passed or failed
if(all_passed) {
    fprintf(stdout, "Passed test \n");
    return 0;
} else {
    fprintf(stdout, "Failed test \n");
    return -1;
}

//! [Check That Solution Is Correct]
};
