#include <Compadre_Config.h>
#include <Compadre_PointCloudSearch.hpp>
#include <Compadre_KokkosParser.hpp>

#ifdef COMPADRE_USE_MPI
#include <mpi.h>
#endif

using namespace Compadre;

int main (int argc, char* args[]) {

// initializes MPI (if available) with command line arguments given
#ifdef COMPADRE_USE_MPI
MPI_Init(&argc, &args);
#endif

// initializes Kokkos with command line arguments given
auto kp = KokkosParser(argc, args, true);

// becomes false if the computed solution not within the failure_threshold of the actual solution
bool all_passed = true;

{
    Kokkos::Timer timer;

    // check if 4 arguments are given from the command line
    //  multiplier times h spacing for search
    double h_multiplier = 3.0; // dimension 3 by default
    if (argc >= 4) {
        double arg4tof = atof(args[3]);
        if (arg4tof > 0) {
            h_multiplier = arg4tof;
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
    //  set the number of dimensions for points and the search
    int dimension = 3; // 3D by default
    if (argc >= 2) {
        int arg2toi = atoi(args[1]);
        if (arg2toi > 0) {
            dimension = arg2toi;
        }
    }
    
    //// minimum neighbors for unisolvency is the same as the size of the polynomial basis 
    //const int min_neighbors = Compadre::GMLS::getNP(order, dimension);
    
    // approximate spacing of source sites
    double h_spacing = 1./std::pow(number_target_coords, 0.50 + 0.2*(dimension==2));
    int n_neg1_to_1 = 2*(1/h_spacing) + 1; // always odd
    
    // number of source coordinate sites that will fill a box of [-1,1]x[-1,1]x[-1,1] with a spacing approximately h
    int number_source_coords = (n_neg1_to_1+1)*(n_neg1_to_1+1);
    if (dimension==3) number_source_coords *= (n_neg1_to_1+1);
    printf("target coords: %d\n", number_target_coords);
    printf("source coords: %d\n", number_source_coords);
    
    // coordinates of source sites
    Kokkos::View<double**, Kokkos::DefaultHostExecutionSpace> source_coords("source coordinates", 
            number_source_coords, 3);
    
    // coordinates of target sites
    Kokkos::View<double**, Kokkos::DefaultHostExecutionSpace> target_coords("target coordinates", number_target_coords, 3);

    // fill source coordinates with a uniform grid
    int source_index = 0;
    double this_coord[3] = {0,0,0};
    for (int i=-n_neg1_to_1/2; i<n_neg1_to_1/2+1; ++i) {
        this_coord[0] = i*h_spacing;
        for (int j=-n_neg1_to_1/2; j<n_neg1_to_1/2+1; ++j) {
            this_coord[1] = j*h_spacing;
            for (int k=-n_neg1_to_1/2; k<n_neg1_to_1/2+1; ++k) {
                this_coord[2] = k*h_spacing;
                if (dimension==3) {
                    source_coords(source_index,0) = this_coord[0]; 
                    source_coords(source_index,1) = this_coord[1]; 
                    source_coords(source_index,2) = this_coord[2]; 
                    source_index++;
                }
            }
            if (dimension==2) {
                source_coords(source_index,0) = this_coord[0]; 
                source_coords(source_index,1) = this_coord[1]; 
                source_coords(source_index,2) = 0;
                source_index++;
            }
        }
        if (dimension==1) {
            source_coords(source_index,0) = this_coord[0]; 
            source_coords(source_index,1) = 0;
            source_coords(source_index,2) = 0;
            source_index++;
        }
    }
    
    // fill target coords somewhere inside of [-0.5,0.5]x[-0.5,0.5]x[-0.5,0.5]
    for(int i=0; i<number_target_coords; i++){
    
        // first, we get a uniformly random distributed direction
        double rand_dir[3] = {0,0,0};
    
        for (int j=0; j<dimension; ++j) {
            // rand_dir[j] is in [-0.5, 0.5]
            rand_dir[j] = ((double)rand() / (double) RAND_MAX) - 0.5;
        }
    
        // then we get a uniformly random radius
        for (int j=0; j<dimension; ++j) {
            target_coords(i,j) = rand_dir[j];
        }
    
    }

    timer.reset();
    // Point cloud construction for neighbor search
    // CreatePointCloudSearch constructs an object of type PointCloudSearch, but deduces the templates for you
    auto point_cloud_search(CreatePointCloudSearch(source_coords, dimension));

    Kokkos::View<int**, Kokkos::DefaultHostExecutionSpace> neighbor_lists("neighbor lists", 
            number_target_coords, 2); // first column is # of neighbors
    
    // each target site has a window size
    Kokkos::View<double*, Kokkos::DefaultHostExecutionSpace> epsilon("h supports", number_target_coords);
    
    // query the point cloud to generate the neighbor lists using a radius search

    // start with a dry-run, but without enough room to store the results
    auto max_num_neighbors = point_cloud_search.generateNeighborListsFromRadiusSearch(true /* dry run */,
        target_coords, neighbor_lists, epsilon, h_multiplier*h_spacing);
    printf("max neighbors: %lu\n", max_num_neighbors);

    // resize neighbor lists to be large enough to hold the results
    neighbor_lists = Kokkos::View<int**, Kokkos::DefaultHostExecutionSpace>("neighbor lists", 
        number_target_coords, 1+max_num_neighbors); // first column is # of neighbors

    // search again, now that we know that there is enough room to store the results
    point_cloud_search.generateNeighborListsFromRadiusSearch(false /* dry run */,
        target_coords, neighbor_lists, epsilon, h_multiplier*h_spacing);

    double radius_search_time = timer.seconds();
    printf("nanoflann search time: %f s\n", radius_search_time);

    // convert point cloud search to vector of maps
    timer.reset();
    std::vector<std::map<int, double> > point_cloud_neighbor_list(number_target_coords, std::map<int, double>());
    Kokkos::parallel_for("point search conversion", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>
            (0,number_target_coords), [&](const int i) {
        auto &point_cloud_neighbor_list_i = const_cast<std::map<int, double>& >(point_cloud_neighbor_list[i]);
        for (int j=1; j<=neighbor_lists(i,0); ++j) {
            point_cloud_neighbor_list_i[neighbor_lists(i,j)] = 1.0;
        }
    });
    double point_cloud_convert_time = timer.seconds();
    printf("point cloud convert time: %f s\n", point_cloud_convert_time);
    Kokkos::fence();

    // loop over targets, finding all of their neighbors by brute force
    timer.reset();
    std::vector<std::map<int, double> > brute_force_neighbor_list(number_target_coords, std::map<int, double>());
    Kokkos::parallel_for("brute force search", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>
            (0,number_target_coords), [&](const int i) {
        auto &brute_force_neighbor_list_i = const_cast<std::map<int, double>& >(brute_force_neighbor_list[i]);
        for (int j=0; j<number_source_coords; ++j) {
            double dist = 0;
            for (int k=0; k<dimension; ++k) {
                dist += (target_coords(i,k)-source_coords(j,k))*(target_coords(i,k)-source_coords(j,k));
            }
            dist = std::sqrt(dist);

            if (dist<h_multiplier*h_spacing) {
                brute_force_neighbor_list_i[j]=dist;
            }
        }
    });
    double brute_force_search_time = timer.seconds();
    printf("brute force search time: %f s\n", brute_force_search_time);
    Kokkos::fence();

    timer.reset();
    // check that all neighbors in brute force list are in original search list
    int num_passed = 0;
    Kokkos::parallel_reduce("brute force search", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>
            (0,number_target_coords), KOKKOS_LAMBDA(const int i, int& t_num_passed) {
        for (auto it=brute_force_neighbor_list[i].begin(); it!=brute_force_neighbor_list[i].end(); ++it) {
            if (point_cloud_neighbor_list[i].count(it->first)==1) t_num_passed++;
        }
    }, Kokkos::Sum<int>(num_passed));
    Kokkos::fence();
    compadre_kernel_assert_release(num_passed != number_target_coords 
            && "Neighbor not found in point cloud list\n");

    // check that all neighbors in original list are in brute force list
    Kokkos::parallel_reduce("original in brute force search", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>
            (0,number_target_coords), KOKKOS_LAMBDA(const int i, int& t_num_passed) {
        for (int j=1; j<=neighbor_lists(i,0); ++j) {
            if (brute_force_neighbor_list[i].count(neighbor_lists(i,j))==1) t_num_passed++;
        }
    }, Kokkos::Sum<int>(num_passed));
    Kokkos::fence();
    compadre_kernel_assert_release(num_passed != number_target_coords 
            && "Neighbor not found in brute force list\n");

    double check_time = timer.seconds();
    printf("check time: %f s\n", check_time);

}

// finalize Kokkos and MPI (if available)
kp.finalize();
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

} // main
