#ifndef _COMPADRE_REMOTEDATAMANAGER_HPP_
#define _COMPADRE_REMOTEDATAMANAGER_HPP_

#include "CompadreHarness_Config.h"
#include "CompadreHarness_Typedefs.hpp"
#include "Compadre_RemapManager.hpp"

namespace Compadre {

class ParticlesT;
class CoordsT;
class RemapObject;

class RemoteDataManager {
	/*
	 * This class establishes a connection with another group of processes on the same global mpi communicator.
	 * It is used for transferring information between two particle sets having different data distribution.
	 * Unique determination of lower and upper processes are based on coloring provided in initialization.
	 */
	protected:

		typedef Compadre::ParticlesT particles_type;
		typedef Compadre::CoordsT coords_type;

		Teuchos::RCP<const Teuchos::Comm<local_index_type> > _global_comm;
		Teuchos::RCP<const Teuchos::Comm<local_index_type> > _local_comm;
		const coords_type* _our_coords;

		Teuchos::RCP<const Teuchos::Comm<local_index_type> > _lower_root_plus_upper_all_comm;
		Teuchos::RCP<const Teuchos::Comm<local_index_type> > _upper_root_plus_lower_all_comm;

		Teuchos::RCP<map_type> _lower_map_for_lower_data;
		Teuchos::RCP<map_type> _lower_map_for_upper_data;
		Teuchos::RCP<map_type> _upper_map_for_lower_data;
		Teuchos::RCP<map_type> _upper_map_for_upper_data;

		Teuchos::RCP<importer_type> _lower_importer;
		Teuchos::RCP<importer_type> _upper_importer;

		const local_index_type _ndim;

		bool _amLower;

        Teuchos::RCP<Compadre::RemapManager> _rm;

		Teuchos::RCP<Teuchos::Time> RemoteDataMapConstructionTime;
		Teuchos::RCP<Teuchos::Time> RemoteDataCoordinatesTime;
		Teuchos::RCP<Teuchos::Time> RemoteDataRemapTime;


	public:

		RemoteDataManager(Teuchos::RCP<const Teuchos::Comm<local_index_type> > global_comm,
				Teuchos::RCP<const Teuchos::Comm<local_index_type> > local_comm,
				const coords_type* our_coords,
				const local_index_type my_program_coloring,
				const local_index_type peer_program_coloring,
				const bool use_physical_coords = true,
				const scalar_type bounding_box_relative_tolerance = 1e-2);

		RemoteDataManager(Teuchos::RCP<const Teuchos::Comm<local_index_type> > global_comm,
				Teuchos::RCP<const Teuchos::Comm<local_index_type> > local_comm,
				const coords_type* our_coords,
				const local_index_type my_program_coloring,
				const local_index_type peer_program_coloring,
				host_view_local_index_type flags,
				const std::vector<local_index_type> flags_to_transfer,
				const bool use_physical_coords = true,
				const scalar_type bounding_box_relative_tolerance = 1e-2);

		~RemoteDataManager() {};

		void putRemoteCoordinatesInParticleSet(particles_type* particles_to_overwrite, const bool use_physical_coords = true);

		void putRemoteWeightsInParticleSet(const particles_type* source_particles, particles_type* particles_to_overwrite, std::string weighting_field_name);

		// remap_vector contains name of source field name for peer process and target name for what we would like to call it
		void remapData(std::vector<RemapObject> remap_vector,
				Teuchos::RCP<Teuchos::ParameterList> parameters,
				particles_type* source_particles,
				particles_type* particles_to_overwrite,
				double max_halo_size,
				bool use_physical_coords = true,
				bool reuse_remap_solution = false);

};

}

#endif
