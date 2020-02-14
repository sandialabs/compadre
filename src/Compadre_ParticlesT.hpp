#ifndef _COMPADRE_PARTICLE_HPP_
#define _COMPADRE_PARTICLE_HPP_

#include "CompadreHarness_Config.h"
#include "CompadreHarness_Typedefs.hpp"

namespace Compadre {

class XyzVector;
class CoordsT;
class FieldT;
class NeighborhoodT;
class FieldManager;
class DOFManager;
class RemapManager;

/** Orchestrates the relationships between coordinates, fields, neighborhood searches, and DOF managers
 *
 *  Keeps coordination when resizing coordinates so that all existing fields and boundary condition ids are
 *  kept consistent. This includes when applying permutations to the coordinate maps such as a Zoltan2 permission.
 *
 *  Is responsible for creating neighborhoods and DOF managers, however, it does not manager them
 *  (the corresponding classes contain the tools to manage themselves)
 */

class ParticlesT {

	protected:

		typedef Compadre::XyzVector xyz_type;
		typedef Compadre::CoordsT coords_type;
		typedef Compadre::FieldT field_type;
		typedef Compadre::NeighborhoodT neighbors_type;
		typedef Compadre::FieldManager fieldmanager_type;
		typedef Compadre::DOFManager dofmanager_type;
		typedef Compadre::RemapManager remap_manager_type;
		typedef Compadre::ParticlesT particles_type;

		Teuchos::RCP<const Teuchos::Comm<local_index_type> > _comm;
		Teuchos::RCP<Teuchos::ParameterList> _parameters;

		Teuchos::RCP<coords_type> _coords;
		std::vector<Teuchos::RCP<field_type> > _fields;
		std::map<std::string,local_index_type> _field_map;
		const local_index_type _nDim;

		local_index_type total_field_offsets;
		std::vector<local_index_type> field_offsets;

		Teuchos::RCP<mvec_local_index_type> _flag;

		Teuchos::RCP<neighbors_type> _neighborhoodInfo;
		Teuchos::RCP<fieldmanager_type> _fieldManager;
		Teuchos::RCP<dofmanager_type> _DOFManager;

	public:
		//! to be called if a set of coordinates is NOT already initialized
		ParticlesT(Teuchos::RCP<Teuchos::ParameterList> parameters, const Teuchos::RCP<const Teuchos::Comm<int> >& _comm, const int nDim = 3);

		//! to be called if a set of coordinates is already initialized
		ParticlesT(Teuchos::RCP<Teuchos::ParameterList> parameters,	Teuchos::RCP<coords_type> coords);

		virtual ~ParticlesT() {};

		/** @name Coordination
		 *  Coordination calls that resize all fields, flags, and IDs to coordinate changes
		 */
		///@{
		//! creates new maps, vectors, and COPIES existing data
		void insertParticles(const std::vector<xyz_type>& new_pts_vector, const scalar_type rebuilt_halo_size = 0.0, bool repartition = false, bool inserting_physical_coords = true, bool repartition_with_physical_coords = true, bool verify_coords_on_processor = true);

		//! creates new maps, vectors, and COPIES existing data
		void removeParticles(const std::vector<local_index_type>& coord_ids, const scalar_type rebuilt_halo_size = 0.0, bool repartition = false, bool repartition_using_physical_coords = true);

		//! creates new maps, vectors, and COPIES existing data from other particle set
		void mergeWith(const particles_type* other_particles);

		//! creates new maps, vectors, and RESETS all existing data
		void setCoords(Teuchos::RCP<Compadre::CoordsT> coords);

		//! RESETS all existing data to conform to a change in coordinates that was not managed by ParticlesT
		void resetWithSameCoords();

		//! RESETS Lagrangian coordinates with physical coordinates in a Lagrangian simulation
		void snapLagrangianCoordsToPhysicalCoords();

		//! RESETS Physical coordinates with lagrangian coordinates in a Lagrangian simulation
		void snapPhysicalCoordsToLagrangianCoords();

		//! RESETS Physical coordinates by adding the displacement to the physical coordinates to get new positions for Lagrangian simulations
		void updatePhysicalCoordinatesFromField(std::string field_name, const double multiplier = 1);

		//! creates new maps, vectors, and RESETS all existing data
		void resize(const global_index_type nn, bool local_resize = false);

		//! creates new maps, vectors, and RESETS all existing data
		void resize(host_view_global_index_type gids);

		//! permutes data in coords, flags, and fields
		void zoltan2Initialize(bool use_physical_coords = true);
		///@}


		/** @name Access */
		///@{
		coords_type * getCoords() { return _coords.getRawPtr(); }

		const coords_type * getCoordsConst() const { return _coords.getRawPtr(); }

		Teuchos::RCP<const map_type> getCoordMap() const;

		// not const because ParameterLists must remain writeable (used/unused)
		Teuchos::RCP<Teuchos::ParameterList> getParameters() { return _parameters; }
		///@}


		/** @name Halo functionality */
		///@{
		void buildHalo(scalar_type h, bool use_physical_coords = true);

		void buildHalo(bool use_physical_coords = true);
		///@}


		/** @name Neighborhood access,
		further neighborhood controls found as member functions of getNeighborhood() */
		///@{
		const neighbors_type * getNeighborhoodConst() const;

		neighbors_type * getNeighborhood();

		void createNeighborhood();
		///@}


		/** @name Flag functions */
		///@{
		void setFlag(const local_index_type idx, const local_index_type val);

		local_index_type getFlag(const local_index_type idx) const;

		Teuchos::RCP<mvec_local_index_type> getFlags() const { return _flag; }
		///@}


		/** @name Field Manager */
		///@{
		const fieldmanager_type * getFieldManagerConst() const { return _fieldManager.getRawPtr(); }

		fieldmanager_type * getFieldManager() { return _fieldManager.getRawPtr(); }
		///@}

		/** @name DOF Manager */
		///@{
		const dofmanager_type * getDOFManagerConst() const {
			TEUCHOS_TEST_FOR_EXCEPT_MSG(_DOFManager.is_null(), "getDOFManagerConst() called before createDOFManager.");
			return _DOFManager.getRawPtr();
		}

		dofmanager_type * getDOFManager() {
			TEUCHOS_TEST_FOR_EXCEPT_MSG(_DOFManager.is_null(), "getDOFManager() called before createDOFManager.");
			return _DOFManager.getRawPtr();
		}

		void createDOFManager();
		///@}
};


}

#endif
