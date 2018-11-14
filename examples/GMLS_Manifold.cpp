#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <stdlib.h> 
#include <cstdio>
#include <random>

#include <Compadre_Config.h>
#include <Compadre_GMLS.hpp>
#include <Compadre_Remap.hpp>
#include <Compadre_PointCloudSearch.hpp>

#include "GMLS_Manifold.hpp"

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
Kokkos::initialize(argc, args);

// code block to reduce scope for all Kokkos View allocations
// otherwise, Views may be deallocating when we call Kokkos::finalize() later
{
    // check if 6 arguments are given from the command line, the first being the program name
    //  N_pts_on_sphere used to determine spatial resolution
    int N_pts_on_sphere = 1000; // QR by default
    if (argc >= 6) {
        int arg6toi = atoi(args[5]);
        if (arg6toi > 0) {
            N_pts_on_sphere = arg6toi;
        }
    }
    
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
    
    // minimum neighbors for unisolvency is the same as the size of the polynomial basis 
    // dimension has one subtracted because it is a D-1 manifold represented in D dimensions
    const int min_neighbors = Compadre::GMLS::getNP(order, dimension-1);
    
    // maximum neighbors calculated to be the minimum number of neighbors needed for unisolvency 
    // enlarged by some multiplier (generally 10% to 40%, but calculated by GMLS class)
    const int max_neighbors = Compadre::GMLS::getNN(order, dimension-1);
    
    //! [Parse Command Line Arguments]
    Kokkos::Timer timer;
    Kokkos::Profiling::pushRegion("Setup Point Data");
    //! [Setting Up The Point Cloud]
    

    // coordinates of source sites, bigger than needed then resized later
    Kokkos::View<double**, Kokkos::DefaultExecutionSpace> source_coords_device("source coordinates", 
            1.25*N_pts_on_sphere, 3);
    Kokkos::View<double**>::HostMirror source_coords = Kokkos::create_mirror_view(source_coords_device);

    double r = 1.0;

    // set number of source coordinates from what was calculated
    int number_source_coords;

    {
        // fill source coordinates from surface of a sphere with quasiuniform points
        // algorithm described at https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
        int N_count = 0;
        double a = 4*PI*r*r/N_pts_on_sphere;
        double d = std::sqrt(a);
        int M_theta = std::round(PI/d);
        double d_theta = PI/M_theta;
        double d_phi = a/d_theta;
        for (int i=0; i<M_theta; ++i) {
            double theta = PI*(i + 0.5)/M_theta;
            int M_phi = std::round(2*PI*std::sin(theta)/d_phi);
            for (int j=0; j<M_phi; ++j) {
                double phi = 2*PI*j/M_phi;
                source_coords(N_count, 0) = theta;
                source_coords(N_count, 1) = phi;
                N_count++;
            }
        }

        for (int i=0; i<N_count; ++i) {
            double theta = source_coords(i,0);
            double phi = source_coords(i,1);
            source_coords(i,0) = r*std::sin(theta)*std::cos(phi);
            source_coords(i,1) = r*std::sin(theta)*std::sin(phi);
            source_coords(i,2) = r*cos(theta);
            //printf("%f %f %f\n", source_coords(i,0), source_coords(i,1), source_coords(i,2));
        }
        number_source_coords = N_count;
    }

    // resize source_coords to the size actually needed
    Kokkos::resize(source_coords, number_source_coords, 3);
    Kokkos::resize(source_coords_device, number_source_coords, 3);


    // each row is a neighbor list for a target site, with the first column of each row containing
    // the number of neighbors for that rows corresponding target site
    Kokkos::View<int**, Kokkos::DefaultExecutionSpace> neighbor_lists_device("neighbor lists", 
            number_target_coords, max_neighbors+1); // first column is # of neighbors
    Kokkos::View<int**>::HostMirror neighbor_lists = Kokkos::create_mirror_view(neighbor_lists_device);
    
    // each target site has a window size
    Kokkos::View<double*, Kokkos::DefaultExecutionSpace> epsilon_device("h supports", number_target_coords);
    Kokkos::View<double*>::HostMirror epsilon = Kokkos::create_mirror_view(epsilon_device);
    
    // coordinates of target sites
    Kokkos::View<double**, Kokkos::DefaultExecutionSpace> target_coords_device ("target coordinates", 
            number_target_coords, 3);
    Kokkos::View<double**>::HostMirror target_coords = Kokkos::create_mirror_view(target_coords_device);
    
    {
        bool enough_pts_found = false;
        // fill target coordinates from surface of a sphere with quasiuniform points
        // stop when enough points are found
        int N_count = 0;
        double a = 4.0*PI*r*r/((double)(5.0*number_target_coords)); // 5.0 is in case number_target_coords is set to something very small (like 1)
        double d = std::sqrt(a);
        int M_theta = std::round(PI/d);
        double d_theta = PI/((double)M_theta);
        double d_phi = a/d_theta;
        for (int i=0; i<M_theta; ++i) {
            double theta = PI*(i + 0.5)/M_theta;
            int M_phi = std::round(2*PI*std::sin(theta)/d_phi);
            for (int j=0; j<M_phi; ++j) {
                double phi = 2*PI*j/M_phi;
                target_coords(N_count, 0) = theta;
                target_coords(N_count, 1) = phi;
                N_count++;
                if (N_count == number_target_coords) {
                    enough_pts_found = true;
                    break;
                }
            }
            if (enough_pts_found) break;
        }

        for (int i=0; i<N_count; ++i) {
            double theta = target_coords(i,0);
            double phi = target_coords(i,1);
            target_coords(i,0) = r*std::sin(theta)*std::cos(phi);
            target_coords(i,1) = r*std::sin(theta)*std::sin(phi);
            target_coords(i,2) = r*cos(theta);
            //printf("%f %f %f\n", target_coords(i,0), target_coords(i,1), target_coords(i,2));
        }
    }

    
    //! [Setting Up The Point Cloud]
    
    Kokkos::Profiling::popRegion();
    Kokkos::fence();
    Kokkos::Profiling::pushRegion("Creating Data");
    
    //! [Creating The Data]
    
    
    // source coordinates need copied to device before using to construct sampling data
    Kokkos::deep_copy(source_coords_device, source_coords);
    
    // target coordinates copied next, because it is a convenient time to send them to device
    Kokkos::deep_copy(target_coords_device, target_coords);

    // ensure that source coordinates are sent to device before evaluating sampling data based on them
    Kokkos::fence(); 

    
    // need Kokkos View storing true solution (for samples)
    Kokkos::View<double*, Kokkos::DefaultExecutionSpace> sampling_data_device("samples of true solution", 
            source_coords_device.dimension_0());

    // need Kokkos View storing true vector solution (for samples)
    Kokkos::View<double**, Kokkos::DefaultExecutionSpace> sampling_vector_data_device("samples of vector true solution", 
            source_coords_device.dimension_0(), 3);
    
    Kokkos::parallel_for("Sampling Manufactured Solutions", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>
            (0,source_coords.dimension_0()), KOKKOS_LAMBDA(const int i) {
    
        // coordinates of source site i
        double xval = source_coords_device(i,0);
        double yval = (dimension>1) ? source_coords_device(i,1) : 0;
        double zval = (dimension>2) ? source_coords_device(i,2) : 0;
    
        // data for targets with scalar input
        sampling_data_device(i) = sphere_harmonic54(xval, yval, zval);

        for (int j=0; j<3; ++j) {
            double gradient[3] = {0,0,0};
            gradient_sphereHarmonic54_ambient(gradient, xval, yval, zval);
            sampling_vector_data_device(i,j) = gradient[j];
        }
    });
    
    
    //! [Creating The Data]
    
    Kokkos::Profiling::popRegion();
    Kokkos::Profiling::pushRegion("Neighbor Search");
    
    //! [Performing Neighbor Search]
    
    
    // Point cloud construction for neighbor search
    // CreatePointCloudSearch constructs an object of type PointCloudSearch, but deduces the templates for you
    auto point_cloud_search(CreatePointCloudSearch(source_coords, target_coords));
    
    // query the point cloud to generate the neighbor lists using a kdtree to produce the n nearest neighbor
    // to each target site, adding 20% to whatever the distance away the further neighbor used is from
    // each target to the view for epsilon
    point_cloud_search.generateNeighborListsFromKNNSearch(neighbor_lists, epsilon, max_neighbors, dimension, 
            1.2 /* epsilon multiplier */);
    

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
    solver_name = "MANIFOLD";
    
    // initialize an instance of the GMLS class for problems with a scalar basis and traditional point sampling as the sampling functional
    GMLS my_GMLS_scalar(order, solver_name.c_str(), order /*manifold order*/, dimension);
    
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
    my_GMLS_scalar.setProblemData(neighbor_lists_device, source_coords_device, target_coords_device, epsilon_device);
    
    // create a vector of target operations
    std::vector<TargetOperation> lro_scalar(3);
    lro_scalar[0] = ScalarPointEvaluation;
    lro_scalar[1] = LaplacianOfScalarPointEvaluation;
    lro_scalar[2] = GradientOfScalarPointEvaluation;

    // and then pass them to the GMLS class
    my_GMLS_scalar.addTargets(lro_scalar);

    // sets the weighting kernel function from WeightingFunctionType for curvature
    my_GMLS_scalar.setCurvatureWeightingType(WeightingFunctionType::Power);
    
    // power to use in the weighting kernel function for curvature coefficients
    my_GMLS_scalar.setCurvatureWeightingPower(2);
    
    // sets the weighting kernel function from WeightingFunctionType
    my_GMLS_scalar.setWeightingType(WeightingFunctionType::Power);
    
    // power to use in that weighting kernel function
    my_GMLS_scalar.setWeightingPower(2);
    
    // generate the alphas that to be combined with data for each target operation requested in lro
    my_GMLS_scalar.generateAlphas();
    
    // initialize another instance of the GMLS class for problems with a vector basis on a manifold and point 
    // evaluation of that vector as the sampling functional
    GMLS my_GMLS_vector(ReconstructionSpace::VectorTaylorPolynomial,SamplingFunctional::ManifoldVectorSample, order, solver_name.c_str(), order /*manifold order*/, dimension);
    my_GMLS_vector.setProblemData(neighbor_lists_device, source_coords_device, target_coords_device, epsilon_device);
    std::vector<TargetOperation> lro_vector(2);
    lro_vector[0] = VectorPointEvaluation;
    lro_vector[1] = DivergenceOfVectorPointEvaluation;
    my_GMLS_vector.addTargets(lro_vector);
    my_GMLS_vector.setCurvatureWeightingType(WeightingFunctionType::Power);
    my_GMLS_vector.setCurvatureWeightingPower(2);
    my_GMLS_vector.setWeightingType(WeightingFunctionType::Power);
    my_GMLS_vector.setWeightingPower(2);
    my_GMLS_vector.generateAlphas();


    //! [Setting Up The GMLS Object]
    
    double instantiation_time = timer.seconds();
    std::cout << "Took " << instantiation_time << "s to complete alphas generation." << std::endl;
    Kokkos::fence(); // let generateAlphas finish up before using alphas
    Kokkos::Profiling::pushRegion("Apply Alphas to Data");
    
    //! [Apply GMLS Alphas To Data]
    
    
    // remap is done here under a particular linear operator
    // it is important to note that if you expect to use the data as a 1D view, then you should use double*
    // however, if you know that the target operation will result in a 2D view (vector or matrix output),
    // then you should template with double** as this is something that can not be infered from the input data
    // or the target operator at compile time
    
    Remap scalar_remap_manager(&my_GMLS_scalar);
    Remap vector_remap_manager(&my_GMLS_vector);
    
    auto output_value = scalar_remap_manager.applyAlphasToDataAllComponentsAllTargetSites<double*, Kokkos::HostSpace>
            (sampling_data_device, ScalarPointEvaluation);
    
    auto output_laplacian = scalar_remap_manager.applyAlphasToDataAllComponentsAllTargetSites<double*, Kokkos::HostSpace>
            (sampling_data_device, LaplacianOfScalarPointEvaluation);

    auto output_gradient = scalar_remap_manager.applyAlphasToDataAllComponentsAllTargetSites<double**, Kokkos::HostSpace>
            (sampling_data_device, GradientOfScalarPointEvaluation);

    auto output_vector = vector_remap_manager.applyAlphasToDataAllComponentsAllTargetSites<double**, Kokkos::HostSpace>
            (sampling_vector_data_device, VectorPointEvaluation, ManifoldVectorSample);
    
    auto output_divergence = vector_remap_manager.applyAlphasToDataAllComponentsAllTargetSites<double*, Kokkos::HostSpace>
            (sampling_vector_data_device, DivergenceOfVectorPointEvaluation, ManifoldVectorSample);

//Kokkos::Profiling::pushRegion("Vector Remap");
//auto output_vector = Kokkos::View<double**, Kokkos::HostSpace>("vector remap values done manually", target_coords.dimension_0(), 3);
//{
//    auto sampling_vector_data_host = Kokkos::create_mirror(sampling_vector_data_device);
//    Kokkos::deep_copy(sampling_vector_data_host, sampling_vector_data_device);
//    Kokkos::fence();
//
//    //my_GMLS_vector.applyDataTransformGlobalOrLocal(sampling_vector_data_host, CoordinatesType::Ambient, CoordinatesType::Local);
//
//    Kokkos::fence();
//    //data now on host
//    
//    // remap is manual
//    Kokkos::parallel_for("Sampling Manufactured Solutions For Vector", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>
//            (0,target_coords.dimension_0()), KOKKOS_LAMBDA(const int i) {
//
//        double contracted_sum[2];
//        for (int l=0; l<neighbor_lists(i,0); l++) {
//            for (int m=0; m<2; m++) {
//                contracted_sum[m] = 0;
//                for (int n=0; n<3; n++) {
//                    contracted_sum[m] += my_GMLS_vector.getPreStencilWeight(SamplingFunctional::ManifoldVectorSample, i, l, false /* for neighbor*/, m, n) * sampling_vector_data_host(neighbor_lists(i,l+1), n);
//                }
//            }
//
//            double new_value[2] = {0,0};
//            for (int m=0; m<2; m++) {
//                // apply gmls coefficients to equal rank data
//                for (int k=0; k<2; k++) { // only 2 used
//                    double alphas_l = my_GMLS_vector.getAlpha1TensorTo1Tensor(TargetOperation::VectorPointEvaluation, i, k, l, m);
//                    assert((alphas_l==alphas_l) && "NaN in coefficients from GMLS");
//                    new_value[k] += alphas_l * contracted_sum[m];
//                }
//            }
//
//            double transformed_new_value[3];
//            for (int m=0; m<3; m++) {
//                transformed_new_value[m] = 0;
//                for (int n=0; n<2; n++) {
//                    transformed_new_value[m] += my_GMLS_vector.getPreStencilWeight(SamplingFunctional::ManifoldVectorSample, i, l, false /* for neighbor*/, n, m) * new_value[n];
//                }
//            }
//            for (int k=0; k<3; k++) { 
//                double old_value = output_vector(i,k);
//                output_vector(i,k) = transformed_new_value[k] + old_value;
//            }
//
//
//        }
//    });
//    Kokkos::fence();
//}
//Kokkos::Profiling::popRegion();


    Kokkos::fence(); // let application of alphas to data finish before using results

    //// move gradient data to device so that it can be transformed into velocity
    //auto output_gradient_device_mirror = Kokkos::create_mirror(Kokkos::DefaultExecutionSpace::memory_space(), output_gradient);
    //Kokkos::deep_copy(output_gradient_device_mirror, output_gradient);
    //Kokkos::parallel_for("Create Velocity From Surface Gradient", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>
    //        (0,target_coords.dimension_0()), KOKKOS_LAMBDA(const int i) {
    //
    //    // coordinates of target site i
    //    double xval = target_coords_device(i,0);
    //    double yval = (dimension>1) ? target_coords_device(i,1) : 0;
    //    double zval = (dimension>2) ? target_coords_device(i,2) : 0;

    //    double gradx = output_gradient_device_mirror(i,0);
    //    double grady = output_gradient_device_mirror(i,1);
    //    double gradz = output_gradient_device_mirror(i,2);
    //
    //    // overwrites gradient with velocity
    //    output_gradient_device_mirror(i,0) = (grady*zval - yval*gradz);
    //    output_gradient_device_mirror(i,1) = (-(gradx*zval - xval*gradz));
    //    output_gradient_device_mirror(i,2) = (gradx*yval - xval*grady);
    //
    //});
    //Kokkos::deep_copy(output_gradient, output_gradient_device_mirror);


    //! [Apply GMLS Alphas To Data]
    
    Kokkos::fence(); // let application of alphas to data finish before using results
    Kokkos::Profiling::popRegion();
    // times the Comparison in Kokkos
    Kokkos::Profiling::pushRegion("Comparison");
    
    //! [Check That Solutions Are Correct]
    
    double values_error = 0;
    double values_norm = 0;
    double laplacian_error = 0;
    double laplacian_norm = 0;
    double gradient_ambient_error = 0;
    double gradient_ambient_norm = 0;
    double vector_ambient_error = 0;
    double vector_ambient_norm = 0;
    double divergence_ambient_error = 0;
    double divergence_ambient_norm = 0;
    
    // loop through the target sites
    for (int i=0; i<number_target_coords; i++) {
    
        // load value from output
        double GMLS_value = output_value(i);
    
        // load laplacian from output
        double GMLS_Laplacian = output_laplacian(i);
    
        // target site i's coordinate
        double xval = target_coords(i,0);
        double yval = (dimension>1) ? target_coords(i,1) : 0;
        double zval = (dimension>2) ? target_coords(i,2) : 0;
    
        // evaluation of various exact solutions
        double actual_value = sphere_harmonic54(xval, yval, zval);
        double actual_Laplacian = laplace_beltrami_sphere_harmonic54(xval, yval, zval);
        double actual_Gradient_ambient[3] = {0,0,0}; // initialized for 3, but only filled up to dimension
        gradient_sphereHarmonic54_ambient(actual_Gradient_ambient, xval, yval, zval);
        //velocity_sphereHarmonic54_ambient(actual_Gradient_ambient, xval, yval, zval);

        values_error += (GMLS_value - actual_value)*(GMLS_value - actual_value);
        values_norm  += actual_value*actual_value;
    
        laplacian_error += (GMLS_Laplacian - actual_Laplacian)*(GMLS_Laplacian - actual_Laplacian);
        laplacian_norm += actual_Laplacian*actual_Laplacian;

        for (int j=0; j<dimension; ++j) {
            gradient_ambient_error += (output_gradient(i,j) - actual_Gradient_ambient[j])*(output_gradient(i,j) - actual_Gradient_ambient[j]);
            gradient_ambient_norm += actual_Gradient_ambient[j]*actual_Gradient_ambient[j];
        }

        for (int j=0; j<dimension; ++j) {
            vector_ambient_error += (output_vector(i,j) - actual_Gradient_ambient[j])*(output_vector(i,j) - actual_Gradient_ambient[j]);
            vector_ambient_norm += actual_Gradient_ambient[j]*actual_Gradient_ambient[j];
        }

        divergence_ambient_error += (output_divergence(i) - actual_Laplacian)*(output_divergence(i) - actual_Laplacian);
        divergence_ambient_norm += actual_Laplacian*actual_Laplacian;

    }
    
    values_error /= number_target_coords;
    values_error = std::sqrt(values_error);
    values_norm /= number_target_coords;
    values_norm = std::sqrt(values_norm);

    laplacian_error /= number_target_coords;
    laplacian_error = std::sqrt(laplacian_error);
    laplacian_norm /= number_target_coords;
    laplacian_norm = std::sqrt(laplacian_norm);

    gradient_ambient_error /= number_target_coords;
    gradient_ambient_error = std::sqrt(gradient_ambient_error);
    gradient_ambient_norm /= number_target_coords;
    gradient_ambient_norm = std::sqrt(gradient_ambient_norm);
   
    vector_ambient_error /= number_target_coords;
    vector_ambient_error = std::sqrt(vector_ambient_error);
    vector_ambient_norm /= number_target_coords;
    vector_ambient_norm = std::sqrt(vector_ambient_norm);

    divergence_ambient_error /= number_target_coords;
    divergence_ambient_error = std::sqrt(divergence_ambient_error);
    divergence_ambient_norm /= number_target_coords;
    divergence_ambient_norm = std::sqrt(divergence_ambient_norm);

    printf("Point Value Error: %g\n", values_error / values_norm);  
    printf("Laplace-Beltrami Error: %g\n", laplacian_error / laplacian_norm);  
    printf("Surface Gradient (Ambient) Error: %g\n", gradient_ambient_error / gradient_ambient_norm);  
    printf("Surface Vector Error: %g\n", vector_ambient_error / vector_ambient_norm);  
    printf("Surface Divergence Error: %g\n", divergence_ambient_error / divergence_ambient_norm);  
    //! [Check That Solutions Are Correct] 
    // popRegion hidden from tutorial
    // stop timing comparison loop
    Kokkos::Profiling::popRegion();
    //! [Finalize Program]


} // end of code block to reduce scope, causing Kokkos View de-allocations
// otherwise, Views may be deallocating when we call Kokkos::finalize() later

// finalize Kokkos and MPI (if available)
Kokkos::finalize();
#ifdef COMPADRE_USE_MPI
MPI_Finalize();
#endif

return 0;

} // main


//! [Finalize Program]
