#ifndef _COMPADRE_DOFMANAGER_HPP_
#define _COMPADRE_DOFMANAGER_HPP_

#include "CompadreHarness_Config.h"
#include "CompadreHarness_Typedefs.hpp"
#include "Compadre_DOFData.hpp"

namespace Compadre {

class FieldT;
class FieldManager;
class ParticlesT;

class DOFManager {

	protected:

		typedef Compadre::FieldT field_type;
		typedef Compadre::FieldManager fieldmanager_type;
		typedef Compadre::ParticlesT particles_type;
		typedef Compadre::DOFData dof_data_type;

		Teuchos::RCP<Teuchos::ParameterList> _parameters;
		const fieldmanager_type* _field_manager;
		const std::vector<Teuchos::RCP<field_type> >& _fields;
		const particles_type* _particles;

		Teuchos::RCP<const dof_data_type> _particle_dof_data; // dof data for all fields on particles

	public:

		DOFManager(const particles_type* particles, Teuchos::RCP<Teuchos::ParameterList> parameters);

		virtual ~DOFManager() {};

		// dof manager
		// row map and column map produced through dof manager

		// Wrappers for the DOFData

		Teuchos::RCP<const map_type> getRowMap(local_index_type idx = -1) const {
			return this->_particle_dof_data->getRowMap(idx);
		}

		Teuchos::RCP<const map_type> getColMap(local_index_type idx = -1) const {
			return this->_particle_dof_data->getColMap(idx);
		}

		local_index_type getDOFFromLID(const local_index_type particle_num, const local_index_type field_num, const local_index_type component_num) const {
			return this->_particle_dof_data->getDOFFromLID(particle_num, field_num, component_num);
		}

		const std::vector<std::vector<std::vector<local_index_type> > >& getDOFMap() const {
			return this->_particle_dof_data->getDOFMap();
		}

		Teuchos::RCP<const dof_data_type> generateDOFMap(std::vector<local_index_type> field_numbers = std::vector<local_index_type>() );

		Teuchos::RCP<const dof_data_type> getParticlesDOFData() const {
			TEUCHOS_TEST_FOR_EXCEPT_MSG(&_particle_dof_data==NULL, "generateDOFMap() never called on the particle set");
			return _particle_dof_data;
		}

};

}

#endif
