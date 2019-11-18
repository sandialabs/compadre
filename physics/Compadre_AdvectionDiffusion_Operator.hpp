#ifndef _COMPADRE_ADVECTIONDIFFUSIONPHYSICS_HPP_
#define _COMPADRE_ADVECTIONDIFFUSIONPHYSICS_HPP_

#include <Compadre_PhysicsT.hpp>
#include <Compadre_XyzVector.hpp>

/*
 Constructs the matrix operator.
 
 */
namespace Compadre {

class GMLS;
class ParticlesT;
class NeighborhoodT;

class AdvectionDiffusionPhysics : public PhysicsT {

	protected:

		typedef Compadre::ParticlesT particle_type;
        typedef Compadre::XyzVector xyz_type;
        typedef Compadre::NeighborhoodT neighborhood_type;
		local_index_type Porder;

        xyz_type _advection_field;
        scalar_type _diffusion;

    public:
        
        particle_type* _cells;

		Teuchos::RCP<neighborhood_type> _particles_particles_neighborhood;
		Teuchos::RCP<neighborhood_type> _cell_particles_neighborhood;
		Teuchos::RCP<neighborhood_type> _particle_cells_neighborhood;

		size_t _particles_particles_max_num_neighbors;
		size_t _cell_particles_max_num_neighbors;
		size_t _particle_cells_max_num_neighbors;
        size_t _weights_ndim;

        Teuchos::RCP<GMLS> _gmls;

        Kokkos::View<int**>::HostMirror _kokkos_neighbor_lists_host;
        Kokkos::View<double**>::HostMirror _kokkos_augmented_source_coordinates_host;
        Kokkos::View<double**>::HostMirror _kokkos_target_coordinates_host;
        Kokkos::View<double*>::HostMirror _kokkos_epsilons_host;
        Kokkos::View<double**>::HostMirror _kokkos_quadrature_coordinates_host;
        Kokkos::View<int**>::HostMirror _kokkos_quadrature_neighbor_lists_host;


		AdvectionDiffusionPhysics(	Teuchos::RCP<particle_type> particles, local_index_type t_Porder,
								Teuchos::RCP<crs_graph_type> A_graph = Teuchos::null,
								Teuchos::RCP<crs_matrix_type> A = Teuchos::null) :
									PhysicsT(particles, A_graph, A), Porder(t_Porder) {

            _particles_particles_max_num_neighbors = 0;
            _cell_particles_max_num_neighbors = 0;
            _particle_cells_max_num_neighbors = 0;
            _weights_ndim = 0;
                                
        } 
		
		virtual ~AdvectionDiffusionPhysics() {}
		
		virtual Teuchos::RCP<crs_graph_type> computeGraph(local_index_type field_one, local_index_type field_two = -1);

		virtual void computeMatrix(local_index_type field_one, local_index_type field_two = -1, scalar_type time = 0.0);

		virtual const std::vector<InteractingFields> gatherFieldInteractions();

        void setAdvectionField(const xyz_type advection_field) {
            for (local_index_type i=0; i<3; ++i) {
                _advection_field[i] = advection_field[i];
            }
        }
        
        xyz_type getAdvectionField() const { return _advection_field; }

        void setDiffusion(const scalar_type diffusion) {
            _diffusion = diffusion;
        }

        scalar_type getDiffusion() const { return _diffusion; }

        void setCells(Teuchos::RCP<particle_type> cells) { _cells = cells.getRawPtr(); };

        void generateData();

        Teuchos::RCP<GMLS> getGMLSInstance() { return _gmls; }

};

}
#endif
