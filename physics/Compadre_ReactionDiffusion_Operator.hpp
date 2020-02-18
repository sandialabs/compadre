#ifndef _COMPADRE_REACTIONDIFFUSIONPHYSICS_HPP_
#define _COMPADRE_REACTIONDIFFUSIONPHYSICS_HPP_

#include <Compadre_PhysicsT.hpp>
#include <Compadre_XyzVector.hpp>

/*
 Constructs the matrix operator.
 
 */
namespace Compadre {

class GMLS;
class ParticlesT;
class NeighborhoodT;

class ReactionDiffusionPhysics : public PhysicsT {

	protected:

		typedef Compadre::ParticlesT particle_type;
        typedef Compadre::XyzVector xyz_type;
        typedef Compadre::NeighborhoodT neighborhood_type;
		local_index_type Porder;

    public:

        scalar_type _reaction;  // used for reaction-diffusion
        scalar_type _diffusion; // used for reaction-diffusion
        scalar_type _shear;     // used for linear elasticity
        scalar_type _lambda;    // used for linear elasticity

        
        particle_type* _cells;


		Teuchos::RCP<neighborhood_type> _particles_particles_neighborhood;
		Teuchos::RCP<neighborhood_type> _particle_cells_neighborhood;
		Teuchos::RCP<neighborhood_type> _cell_particles_neighborhood;
		Teuchos::RCP<neighborhood_type> _halo_cell_particles_neighborhood;

		Teuchos::RCP<neighborhood_type> _particles_triple_hop_neighborhood;

		size_t _particles_particles_max_num_neighbors;
		size_t _cell_particles_max_num_neighbors;
		size_t _particle_cells_max_num_neighbors;
        size_t _weights_ndim;
		scalar_type _cell_particles_max_h;

        Teuchos::RCP<GMLS> _gmls;
        Teuchos::RCP<GMLS> _halo_gmls;

        Kokkos::View<int**>::HostMirror _kokkos_neighbor_lists_host;
        Kokkos::View<double**>::HostMirror _kokkos_augmented_source_coordinates_host;
        Kokkos::View<double**>::HostMirror _kokkos_target_coordinates_host;
        Kokkos::View<double*>::HostMirror _kokkos_epsilons_host;
        Kokkos::View<double**>::HostMirror _kokkos_quadrature_coordinates_host;
        Kokkos::View<int**>::HostMirror _kokkos_quadrature_neighbor_lists_host;

        Teuchos::RCP<mvec_local_index_type> _ids;
        Teuchos::RCP<mvec_local_index_type> _halo_ids;
        host_view_local_index_type _id_view;
        host_view_local_index_type _halo_id_view;

        host_view_local_index_type _halo_big_to_small;
        host_view_local_index_type _halo_small_to_big;


		ReactionDiffusionPhysics(	Teuchos::RCP<particle_type> particles, local_index_type t_Porder,
								Teuchos::RCP<crs_graph_type> A_graph = Teuchos::null,
								Teuchos::RCP<crs_matrix_type> A = Teuchos::null) :
									PhysicsT(particles, A_graph, A), Porder(t_Porder) {

            _particles_particles_max_num_neighbors = 0;
            _cell_particles_max_num_neighbors = 0;
            _particle_cells_max_num_neighbors = 0;
            _weights_ndim = 0;
            _cell_particles_max_h = 0;
                                
        } 
		
		virtual ~ReactionDiffusionPhysics() {}
		
		virtual Teuchos::RCP<crs_graph_type> computeGraph(local_index_type field_one, local_index_type field_two = -1);

		virtual void computeMatrix(local_index_type field_one, local_index_type field_two = -1, scalar_type time = 0.0);

		virtual const std::vector<InteractingFields> gatherFieldInteractions();

        void setReaction(const scalar_type reaction) {
            _reaction = reaction;
        }
        
        void setDiffusion(const scalar_type diffusion) {
            _diffusion = diffusion;
        }

        void setShear(const scalar_type shear) {
            _shear = shear;
        }
        
        void setLambda(const scalar_type lambda) {
            _lambda = lambda;
        }

        void setCells(Teuchos::RCP<particle_type> cells) { _cells = cells.getRawPtr(); };

        virtual void initialize();
        
        virtual local_index_type getMaxNumNeighbors();

        Teuchos::RCP<GMLS> getGMLSInstance() { return _gmls; }

};

}
#endif
