#ifndef _COMPADRE_SOURCES_HPP_
#define _COMPADRE_SOURCES_HPP_

#include "CompadreHarness_Config.h"
#include "CompadreHarness_Typedefs.hpp"

namespace Compadre {

class CoordsT;
class ParticlesT;
class FieldT;
class DOFData;
class InteractingFields;

class SourcesT {

	protected:

		typedef Compadre::ParticlesT particle_type;
		typedef Compadre::CoordsT coords_type;
		typedef Compadre::DOFData dof_data_type;

		Teuchos::RCP<Teuchos::ParameterList> _parameters;
		Teuchos::RCP<const Teuchos::Comm<local_index_type> > _comm;
		const coords_type* _coords;
		Teuchos::RCP<particle_type> _particles;
		mvec_type* _b;
		Teuchos::RCP<const dof_data_type> _dof_data;

		device_view_type rhs;
		scalar_type time;

	public:

		SourcesT(	Teuchos::RCP<particle_type> particles,
					mvec_type* b = NULL);

		virtual ~SourcesT() {};
		
		void setParameters(Teuchos::ParameterList& parameters) { _parameters = Teuchos::rcp(&parameters, false); }

		void setMultiVector(mvec_type* b) {
			_b = b;
		}

		void setDOFData(Teuchos::RCP<const dof_data_type> dof_data) {
			_dof_data = dof_data;
		}

		virtual void evaluateRHS(local_index_type field_one, local_index_type field_two = -1, scalar_type time = 0.0, scalar_type current_timestep_size = 0.0, scalar_type previous_timestep_size = -1.0) = 0;

		virtual std::vector<InteractingFields> gatherFieldInteractions() = 0;
};

}

#endif
