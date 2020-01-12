#ifndef _COMPADRE_GMLS_STOKESPHYSICS_HPP_
#define _COMPADRE_GMLS_STOKESPHYSICS_HPP_

#include <Compadre_PhysicsT.hpp>

/*
Construct the matrix operator.
*/

namespace Compadre {

class GMLS;
class ParticlesT;
class NeighborhoodT;

class GMLS_StokesPhysics : public PhysicsT {
    protected:
        typedef Compadre::ParticlesT particle_type;
        local_index_type Porder;
    public:
        Teuchos::RCP<GMLS> _velocity_all_GMLS;
        Teuchos::RCP<GMLS> _pressure_neumann_GMLS;
        Teuchos::RCP<GMLS> _pressure_all_GMLS;

        Kokkos::View<int*>::HostMirror _internal_filtered_flags;
        Kokkos::View<int*>::HostMirror _boundary_filtered_flags;

        GMLS_StokesPhysics(Teuchos::RCP<particle_type> particles, local_index_type t_Porder,
                           Teuchos::RCP<crs_graph_type> A_graph = Teuchos::null,
                           Teuchos::RCP<crs_matrix_type> A = Teuchos::null) :
            PhysicsT(particles, A_graph, A), Porder(t_Porder) {}

        virtual ~GMLS_StokesPhysics() {}

        virtual Teuchos::RCP<crs_graph_type> computeGraph(local_index_type field_one, local_index_type field_two = -1);

        virtual void computeMatrix(local_index_type field_one, local_index_type field_two = -1, scalar_type time = 0.0);

        virtual const std::vector<InteractingFields> gatherFieldInteractions();

        virtual void initialize();

        Kokkos::View<size_t*, Kokkos::HostSpace> getMaxEntriesPerRow(local_index_type field_one, local_index_type field_two);
};

}

#endif
