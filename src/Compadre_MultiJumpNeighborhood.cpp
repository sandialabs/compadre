#include "Compadre_MultiJumpNeighborhood.hpp"
#include "Compadre_XyzVector.hpp"
#include "Compadre_CoordsT.hpp"

namespace Compadre {

MultiJumpNeighborhood::MultiJumpNeighborhood(const NeighborhoodT* neighborhood_1, const NeighborhoodT* neighborhood_2) 
        : NeighborhoodT(neighborhood_1->getSourceCoordinates()->nDim()), _neighborhood_1(neighborhood_1), _neighborhood_2((neighborhood_2==NULL)?neighborhood_1:NULL) 
            {}

void MultiJumpNeighborhood::constructNeighborOfNeighborLists(const scalar_type max_search_size, bool use_physical_coords) {

    // it is a safe upper bound to use twice the previous search size as the maximum new radii
    scalar_type max_radius = 2.0 * _neighborhood_1->computeMaxHSupportSize(false /* local processor max */);
    // check that max_halo_size is not violated by distance of the double jump, if max_search_size != 0.0
    const local_index_type comm_size = _neighborhood_1->getSourceCoordinates()->getComm()->getSize();
    if (max_search_size != 0.0) {
        // this constraint comes from the halo search
		TEUCHOS_TEST_FOR_EXCEPT_MSG((comm_size > 1) && (max_search_size < max_radius), "Neighbor of neighbor search results in a search radius exceeding the halo size.");
    }

    // neighborhood_1 gives neighbors of target sites from source sites. source sites may include halo information, which means
    // that we may then require a neighbor list of a halo site, which is generally not available.
    // now that we know that twice the MaxHSupportSize is visible in halo information, we are assured that a neighbor search
    // can be performed for halo data with radius of MaxHSupportSize and that this neighbor list will contain all needed
    // entries for filling out the stencil
    //
    // TODO: [insert code here for this secondary search]
    TEUCHOS_TEST_FOR_EXCEPT_MSG((comm_size > 1), "Secondary search not yet added.");
 

    // perform neighbor search using existing neighborhoods for a double hop
    // i can see j and j see k, then i sees k
 
    // this function will generate the data for a valid call to get max num neighbors,
    // num neighbors for i, and index for neighbor j of i
   
    // do loops to get max num neighbors for allocation
    // loop over targets from neighborhood_1, which are sources for neighborhood_2
    TEUCHOS_TEST_FOR_EXCEPT_MSG(_neighborhood_1->getSourceCoordinates() != _neighborhood_1->getTargetCoordinates(), "Source and target coordinates must be the same for MultiJumpNeighborhood.");
    _local_max_num_neighbors = 0;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,_neighborhood_1->getTargetCoordinates()->nLocal(false /* no halo */)), 
            KOKKOS_LAMBDA(const int i, local_index_type& t_num_neighbors) {

        std::set<local_index_type> i_neighbors;

        for (local_index_type j = 0; j < _neighborhood_1->getNumNeighbors(i); j++) {
            local_index_type num_neighbors_j = _neighborhood_1->getNumNeighbors(_neighborhood_1->getNeighbor(i,j));
            for (local_index_type k = 0; k < num_neighbors_j; k++) {
                i_neighbors.insert(_neighborhood_1->getNeighbor(_neighborhood_1->getNeighbor(i,j),k));
            }
        }

        t_num_neighbors = std::max(t_num_neighbors, static_cast<int>(i_neighbors.size()));

    // set max num neighbors from calculation
    }, Kokkos::Max<local_index_type>(_local_max_num_neighbors));

    // allocate new neighbors lists
    _neighbor_lists = host_view_local_index_type("neighbor of neighbor lists", 
            _neighborhood_1->getTargetCoordinates()->nLocal(false /* no halo */), 1+_local_max_num_neighbors);

    // do loops again and add entries into list, also get maximum distance between double jumps
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,_neighborhood_1->getTargetCoordinates()->nLocal(false /* no halo */)), 
            KOKKOS_LAMBDA(const int i) {//, scalar_type& t_max_radius) {

        std::set<local_index_type> i_neighbors;
        scalar_type i_max_radius = 0.0;
        auto i_coords = _neighborhood_1->getSourceCoordinates()->getLocalCoords(i, false /* no halo */, use_physical_coords);

        for (local_index_type j = 0; j < _neighborhood_1->getNumNeighbors(i); j++) {
            local_index_type num_neighbors_j = _neighborhood_1->getNumNeighbors(_neighborhood_1->getNeighbor(i,j));
            for (local_index_type k = 0; k < num_neighbors_j; k++) {
                auto neighbor_k = _neighborhood_1->getNeighbor(_neighborhood_1->getNeighbor(i,j),k);
                i_neighbors.insert(neighbor_k);

                //// get distance from k to i
                //auto k_coords = _neighborhood_1->getSourceCoordinates()->getLocalCoords(neighbor_k, true /* use halo */, use_physical_coords);
                //
                //scalar_type distance_i_to_k = euclideanDistance(i_coords, k_coords);
                //i_max_radius = std::max(i_max_radius, distance_i_to_k);
            }
        }

        _neighbor_lists(i, 0) = i_neighbors.size();
        local_index_type count = 1; // offset by one to leave room for # of neighbors in first column
        for (auto this_ind : i_neighbors) {
            // copy data to neighbor lists
            _neighbor_lists(i, count) = this_ind;
            count++;
        }

        //t_max_radius = std::max(t_max_radius, i_max_radius);

    });//, Kokkos::Max<scalar_type>(max_radius));
    Kokkos::fence();
}

}
