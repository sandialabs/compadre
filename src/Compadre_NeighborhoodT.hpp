#ifndef _COMPADRE_NEIGHBORHOODT_HPP_
#define _COMPADRE_NEIGHBORHOODT_HPP_

#include "CompadreHarness_Config.h"
#include "CompadreHarness_Typedefs.hpp"
#include <Compadre_PointCloudSearch.hpp>


namespace Compadre {

class CoordsT;
class ParticlesT;

class NeighborhoodT {

    protected:

        typedef Compadre::CoordsT coords_type;
        typedef Compadre::ParticlesT particles_type;

        const coords_type* _source_coords;
        const coords_type* _target_coords;
        const particles_type* _source_particles;
        const particles_type* _target_particles;

        Teuchos::RCP<mvec_type> _h_support_size;
        host_view_local_index_type _neighbor_lists;
        host_view_scalar_type _kokkos_augmented_source_coordinates_host;

        typedef decltype(Compadre::PointCloudSearch<host_view_scalar_type>(_kokkos_augmented_source_coordinates_host))::tree_type tree_type;

        std::shared_ptr<tree_type> _kd_tree;
        std::shared_ptr<Compadre::PointCloudSearch<host_view_scalar_type> > _point_cloud_search;

        Teuchos::RCP<Teuchos::Time> NeighborSearchTime;
        Teuchos::RCP<Teuchos::Time> MaxNeighborComputeTime;

        local_index_type _n_dim;
        local_index_type _local_max_num_neighbors;
        local_index_type _storage_multiplier; 

    public:

        NeighborhoodT(const particles_type* source_particles, const particles_type* target_particles = NULL, 
                const bool use_physical_coords = true, const local_index_type max_leaf = 10);

        virtual ~NeighborhoodT() {};

        const coords_type* getSourceCoordinates() const { return _source_coords; }

        const coords_type* getTargetCoordinates() const { return _target_coords; }

        void setHSupportSize(const local_index_type idx, const scalar_type val);

        double getHSupportSize(const local_index_type idx) const;

        void setAllHSupportSizes(const scalar_type val);

        Teuchos::RCP<mvec_type> getHSupportSizes() const { return _h_support_size; }

        host_view_scalar_type getSourceCoordinatesView() const { return _kokkos_augmented_source_coordinates_host; }

        // multiplier for estimated storage of neighbor lists
        void setStorageMultiplier(const local_index_type multiplier) { _storage_multiplier = multiplier; }

        local_index_type getNumNeighbors(const local_index_type idx) const {
            return _neighbor_lists(idx, 0);
        }

        local_index_type getNeighbor(const local_index_type idx, const local_index_type local_neighbor_number) const {
            return _neighbor_lists(idx, local_neighbor_number+1);
        }

        local_index_type computeMaxNumNeighbors(const bool global) const;

        scalar_type computeMinHSupportSize(const bool global) const;

        scalar_type computeMaxHSupportSize(const bool global) const;

        void constructAllNeighborLists(const scalar_type halo_max_search_size, std::string search_type, 
            bool do_dry_run_for_sizes, const local_index_type knn_needed, const scalar_type cutoff_multiplier, 
            const scalar_type radius = 0.0, const bool equal_radii = false, 
            const scalar_type post_search_radii_scaling = 1.0, bool use_physical_coords = true);

};



}

#endif
