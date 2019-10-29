#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <stdlib.h> 
#include <cstdio>
#include <random>

#include <Compadre_Config.h>
#include <Compadre_GMLS.hpp>
#include <Compadre_Evaluator.hpp>
#include <Compadre_PointCloudSearch.hpp>
#include <Compadre_KokkosParser.hpp>

#include "GMLS_Tutorial.hpp"

#ifdef COMPADRE_USE_MPI
#include <mpi.h>
#endif

#include <Kokkos_Timer.hpp>
#include <Kokkos_Core.hpp>

using namespace Compadre;

//! [Parse Command Line Arguments]

// called from command line
int main (int argc, char* args[]) {

// initializes MPI (if available) with command line arguments given
#ifdef COMPADRE_USE_MPI
MPI_Init(&argc, &args);
#endif

// initializes Kokkos with command line arguments given
auto kp = KokkosParser(argc, args, true);

// becomes false if the computed solution not within the failure_threshold of the actual solution
bool all_passed = true;

// code block to reduce scope for all Kokkos View allocations
// otherwise, Views may be deallocating when we call Kokkos finalize() later
{

    // check if 8 arguments are given from the command line, the first being the program name
    int number_of_batches = 1; // 1 batch by default
    if (argc >= 8) {
        int arg8toi = atoi(args[7]);
        if (arg8toi > 0) {
            number_of_batches = arg8toi;
        }
    }

    // check if 7 arguments are given from the command line, the first being the program name
    //  constraint_type used in solving each GMLS problem:
    //      0 - No constraints used in solving each GMLS problem
    //      1 - Neumann Gradient Scalar used in solving each GMLS problem
    int constraint_type = 1; // Neumann Gradient Scalar by default 
    if (argc >= 7) {
        int arg7toi = atoi(args[6]);
        if (arg7toi > 0) {
            constraint_type = arg7toi;
        }
    }

    // check if 6 arguments are given from the command line, the first being the program name
    // problem_type used in solving each GMLS problem:
    //      0 - Standard GMLS problem
    //      1 - Manifold GMLS problem
    int problem_type = 0; // Standard by default
    if (argc >= 6) {
        int arg6toi = atoi(args[5]);
        if (arg6toi > 0) {
            problem_type = arg6toi;
        }
    }
    
    // check if 5 arguments are given from the command line, the first being the program name
    //  solver_type used for factorization in solving each GMLS problem:
    //      0 - SVD used for factorization in solving each GMLS problem
    //      1 - QR  used for factorization in solving each GMLS problem
    //      2 - LU  used for factorization in solving each GMLS problem
    int solver_type = 2; // LU by default
    if (argc >= 5) {
        int arg5toi = atoi(args[4]);
        if (arg5toi >= 0) {
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
    int number_target_coords = 10;
    if (argc >= 3) {
        int arg3toi = atoi(args[2]);
        if (arg3toi > 0) {
            number_target_coords = arg3toi;
        }
    }
    
    // check if 2 arguments are given from the command line
    //  set the number of target sites where we will reconstruct the target functionals at
    int order = 5; // 3rd degree polynomial basis by default
    if (argc >= 2) {
        int arg2toi = atoi(args[1]);
        if (arg2toi > 0) {
            order = arg2toi;
        }
    }
    
    // the functions we will be seeking to reconstruct are in the span of the basis
    // of the reconstruction space we choose for GMLS, so the error should be very small
    const double failure_tolerance = 1e-9;
    
    // Laplacian is a second order differential operator, which we expect to be slightly less accurate
    const double laplacian_failure_tolerance = 1e-9;
    
    // minimum neighbors for unisolvency is the same as the size of the polynomial basis 
    const int min_neighbors = Compadre::GMLS::getNP(order, dimension);
    
    //! [Parse Command Line Arguments]
    Kokkos::Timer timer;
    Kokkos::Profiling::pushRegion("Setup Point Data");
    //! [Setting Up The Point Cloud]
    
    // approximate spherical spacing of source sites
    double angle_spacing = 0.025;
    int n_zero_to_pi = round(M_PI/angle_spacing);
    int n_zero_to_2_pi = 2*n_zero_to_pi;
    // double h_spacing = 0.05;
    // int n_neg1_to_1 = 2*(1/h_spacing) + 1; // always odd
    
    // number of source coordinate sites
    const int number_source_coords = n_zero_to_pi*n_zero_to_2_pi;

    // coordinates of source sites
    Kokkos::View<double**, Kokkos::DefaultExecutionSpace> source_coords_device("source coordinates", 
            number_source_coords, 3);
    Kokkos::View<double**>::HostMirror source_coords = Kokkos::create_mirror_view(source_coords_device);
    
    // coordinates of target sites
    Kokkos::View<double**, Kokkos::DefaultExecutionSpace> target_coords_device ("target coordinates", number_target_coords, 3);
    Kokkos::View<double**>::HostMirror target_coords = Kokkos::create_mirror_view(target_coords_device);

    // tangent bundle for each target sites
    Kokkos::View<double***, Kokkos::DefaultExecutionSpace> tangent_bundles_device ("tangent bundles", number_target_coords, 3, 3);
    Kokkos::View<double***>::HostMirror tangent_bundles = Kokkos::create_mirror_view(tangent_bundles_device);

    // fill source coordinates with a uniform grid
    int source_index = 1;
    double theta, phi;
    double r = 1.0;
    source_coords(0, 0) = 0.0;
    source_coords(0, 1) = 0.0;
    source_coords(0, 2) = 1.0;
    for (int i_theta=1; i_theta < n_zero_to_pi; i_theta++) {
        for (int i_phi=1; i_phi < n_zero_to_2_pi; i_phi++) {
            theta = i_theta*angle_spacing;
            phi = i_phi*angle_spacing;
            source_coords(source_index, 0) = r*sin(theta)*cos(phi);
            source_coords(source_index, 1) = r*sin(theta)*sin(phi);
            source_coords(source_index, 2) = r*cos(theta);
            // std::cout << source_coords(source_index, 0) << " " << source_coords(source_index, 1) << " " << source_coords(source_index, 2) << std::endl;
            source_index++;
        }
    }
    
    // fill target coords somewhere inside of [-0.5,0.5]x[-0.5,0.5]x[-0.5,0.5]
    for(int i=0; i<number_target_coords; i++){
        double rand_theta =  ((double)rand() / (double) RAND_MAX)*M_PI;
        double rand_phi =  ((double)rand() / (double) RAND_MAX)*2.0*M_PI;

        target_coords(i, 0) = r*sin(rand_theta)*cos(rand_phi);
        target_coords(i, 1) = r*sin(rand_theta)*sin(rand_phi);
        target_coords(i, 2) = r*cos(rand_theta);
        // Set tangent bundles
        tangent_bundles(i, 0, 0) = 0.0;
        tangent_bundles(i, 0, 1) = 0.0;
        tangent_bundles(i, 0, 2) = 0.0;
        tangent_bundles(i, 1, 0) = 0.0;
        tangent_bundles(i, 1, 1) = 0.0;
        tangent_bundles(i, 1, 2) = 0.0;
        tangent_bundles(i, 2, 0) = r*sin(rand_theta)*cos(rand_phi);
        tangent_bundles(i, 2, 1) = r*sin(rand_theta)*sin(rand_phi);
        tangent_bundles(i, 2, 2) = r*cos(rand_theta);
    }
    
    
    //! [Setting Up The Point Cloud]
    
    Kokkos::Profiling::popRegion();
    Kokkos::Profiling::pushRegion("Creating Data");
    
    //! [Creating The Data]
    
    
    // source coordinates need copied to device before using to construct sampling data
    Kokkos::deep_copy(source_coords_device, source_coords);
    
    // target coordinates copied next, because it is a convenient time to send them to device
    Kokkos::deep_copy(target_coords_device, target_coords);
    
    // tangent bundles copied next, because it is a convenient time to send them to device
    Kokkos::deep_copy(tangent_bundles_device, tangent_bundles);

    //! [Creating The Data]
    
    Kokkos::Profiling::popRegion();
    Kokkos::Profiling::pushRegion("Neighbor Search");
    
    //! [Performing Neighbor Search]
    
    
    // Point cloud construction for neighbor search
    // CreatePointCloudSearch constructs an object of type PointCloudSearch, but deduces the templates for you
    auto point_cloud_search(CreatePointCloudSearch(source_coords));

    // each row is a neighbor list for a target site, with the first column of each row containing
    // the number of neighbors for that rows corresponding target site
    double epsilon_multiplier = 2.5;
    int estimated_upper_bound_number_neighbors = 
        point_cloud_search.getEstimatedNumberNeighborsUpperBound(min_neighbors, dimension, epsilon_multiplier);

    Kokkos::View<int**, Kokkos::DefaultExecutionSpace> neighbor_lists_device("neighbor lists", 
            number_target_coords, estimated_upper_bound_number_neighbors); // first column is # of neighbors
    Kokkos::View<int**>::HostMirror neighbor_lists = Kokkos::create_mirror_view(neighbor_lists_device);
    
    // each target site has a window size
    Kokkos::View<double*, Kokkos::DefaultExecutionSpace> epsilon_device("h supports", number_target_coords);
    Kokkos::View<double*>::HostMirror epsilon = Kokkos::create_mirror_view(epsilon_device);
    
    // query the point cloud to generate the neighbor lists using a kdtree to produce the n nearest neighbor
    // to each target site, adding (epsilon_multiplier-1)*100% to whatever the distance away the further neighbor used is from
    // each target to the view for epsilon
    point_cloud_search.generateNeighborListsFromKNNSearch(target_coords, neighbor_lists, epsilon, min_neighbors, dimension, 
            epsilon_multiplier);
    
    
    //! [Performing Neighbor Search]
    
    Kokkos::Profiling::popRegion();
    Kokkos::fence(); // let call to build neighbor lists complete before copying back to device
    timer.reset();
    
    //! [Setting Up The GMLS Object]
    
    
    // Copy data back to device (they were filled on the host)
    // We could have filled Kokkos Views with memory space on the host
    // and used these instead, and then the copying of data to the device
    // would be performed in the GMLS class
    Kokkos::deep_copy(neighbor_lists_device, neighbor_lists);
    Kokkos::deep_copy(epsilon_device, epsilon);
    
    // solver name for passing into the GMLS class
    std::string solver_name;
    if (solver_type == 0) { // SVD
        solver_name = "SVD";
    } else if (solver_type == 1) { // QR
        solver_name = "QR";
    } else if (solver_type == 2) { // LU
        solver_name = "LU";
    }

    // problem name for passing into the GMLS class
    std::string problem_name;
    if (problem_type == 0) { // Standard
        problem_name = "STANDARD";
    } else if (problem_type == 1) { // Manifold
        problem_name = "MANIFOLD";
    }

    // boundary name for passing into the GMLS class
    std::string constraint_name;
    if (constraint_type == 0) { // No constraints
        constraint_name = "NO_CONSTRAINT";
    } else if (constraint_type == 1) { // Neumann Gradient Scalar
        constraint_name = "NEUMANN_GRAD_SCALAR";
    }
    
    // initialize an instance of the GMLS class 
    GMLS my_GMLS(VectorOfScalarClonesTaylorPolynomial, VectorPointSample,
                 order, dimension,
                 solver_name.c_str(), problem_name.c_str(), constraint_name.c_str(),
                 2 /*manifold order*/);

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
    my_GMLS.setTangentBundle(tangent_bundles_device);

    // create a vector of target operations
    TargetOperation lro;
    lro = ScalarPointEvaluation;
    
    // and then pass them to the GMLS class
    my_GMLS.addTargets(lro);
    
    // sets the weighting kernel function from WeightingFunctionType
    my_GMLS.setWeightingType(WeightingFunctionType::Power);
    
    // power to use in that weighting kernel function
    my_GMLS.setWeightingPower(2);
    
    // generate the alphas that to be combined with data for each target operation requested in lro
    my_GMLS.generateAlphas(number_of_batches);
    
    //! [Setting Up The GMLS Object]
    
    double instantiation_time = timer.seconds();
    std::cout << "Took " << instantiation_time << "s to complete normal vectors generation." << std::endl;
    Kokkos::fence(); // let generateNormalVectors finish up before using alphas

    // auto output_normal = my_GMLS.getComputedNormalDirections();

    // for (int i=0; i<output_normal.extent(0); i++) {
    //     std::cout << "0 " << output_normal(i, 0) << " " << output_normal(i, 1) << " " << output_normal(i, 2) << std::endl;
    //     std::cout << "1 " << target_coords(i, 0) << " " << target_coords(i, 1) << " " << target_coords(i, 2) << std::endl;
    // }

} // end of code block to reduce scope, causing Kokkos View de-allocations
// otherwise, Views may be deallocating when we call Kokkos finalize() later

// finalize Kokkos and MPI (if available)
kp.finalize();
#ifdef COMPADRE_USE_MPI
MPI_Finalize();
#endif

} // main


//! [Finalize Program]
