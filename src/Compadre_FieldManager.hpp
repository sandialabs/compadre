#ifndef _COMPADRE_FIELDMANAGER_HPP_
#define _COMPADRE_FIELDMANAGER_HPP_

#include "CompadreHarness_Config.h"
#include "CompadreHarness_Typedefs.hpp"

namespace Compadre {

/*
*  This class wraps coordinates, fields, dof manager and manages them
*/
class CoordsT;
class ParticlesT;
class FieldT;
class XyzVector;
class DOFData;

class FieldManager {

	/*! \brief FieldManager coordinates all fields (creating, modifying, outputting) and owns the DOF manager
	 *
	 *  FieldManage keeps coordination between all fields.
	 */

	protected:

		typedef Compadre::CoordsT coords_type;
		typedef Compadre::FieldT field_type;
		typedef Compadre::ParticlesT particles_type;
		typedef Compadre::XyzVector xyz_type;
		typedef Compadre::DOFData dof_data_type;

		Teuchos::RCP<Teuchos::ParameterList> _parameters;
		const particles_type* _particles;

		std::vector<Teuchos::RCP<field_type> > _fields;
		std::map<const std::string, const local_index_type> _field_map;

		local_index_type total_field_offsets;
		local_index_type max_num_field_dimensions;
		std::vector<local_index_type> field_offsets;


		// for managing what quantities are updated
		bool total_field_dimension_updated = false;
		bool field_offsets_updated = false;
		bool max_num_field_dim_updated = false;

		Teuchos::RCP<Teuchos::Time> RemapInsertionTime = Teuchos::TimeMonitor::getNewCounter ("Remap Insertion Time");

	public:

		FieldManager( const particles_type* particles, Teuchos::RCP<Teuchos::ParameterList> parameters );

		virtual ~FieldManager() {};


		// Meta-field functions (operations on the whole set of fields)

		void resetAll(const coords_type * coords);

		void applyZoltan2PartitionToAll();

		void insertParticlesToAll(const coords_type * coords, const std::vector<xyz_type>& new_pts_vector, const particles_type* other_particles);

		void removeParticlesToAll(const coords_type * coords, const std::vector<local_index_type>& coord_ids);

		void remapInsertions(const std::vector<xyz_type>& new_pts_vector);


		// Update Field Information

		void updateMaxNumFieldDimensions();

		void updateTotalFieldDimensions();

		void updateFieldOffsets();


		// Retrieve Field Information (if updated)

		local_index_type getMaxNumFieldDimensions() const;

		local_index_type getTotalFieldDimensions() const;

		local_index_type getFieldOffset(local_index_type idx) const;


		// Modify Fields / Data

		Teuchos::RCP<field_type> createField(const local_index_type nDim, const std::string name = "noName", const std::string units = "null", const FieldSparsityType fst = FieldSparsityType::Banded);

		void updateFieldsHaloData(local_index_type field_num = -1);

		void updateFieldsFromVector(Teuchos::RCP<mvec_type> source, local_index_type field_num = -1, Teuchos::RCP<const dof_data_type> vector_dof_data = Teuchos::null);

		void updateVectorFromFields(Teuchos::RCP<mvec_type> target, local_index_type field_num = -1, Teuchos::RCP<const dof_data_type> vector_dof_data = Teuchos::null) const;

		const std::vector<Teuchos::RCP<field_type> >& getVectorOfFields () const;
		
		const local_index_type getIDOfFieldFromName (const std::string& name) const;

		Teuchos::RCP<field_type> getFieldByID (const local_index_type field_num) const;

		Teuchos::RCP<field_type> getFieldByName (const std::string& name) const;


		// Output Field Info

		void listFields(std::ostream& os) const;

		void printAllFields(std::ostream& os) const;

};


}

#endif
