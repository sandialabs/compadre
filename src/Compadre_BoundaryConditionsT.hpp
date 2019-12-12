#ifndef _COMPADRE_BOUNDARYCONDITIONS_HPP_
#define _COMPADRE_BOUNDARYCONDITIONS_HPP_

#include "CompadreHarness_Config.h"
#include "CompadreHarness_Typedefs.hpp"

namespace Compadre {

class CoordsT;
class ParticlesT;
class DOFData;

class BoundaryConditionsT {

	protected:

		typedef Compadre::CoordsT coords_type;
		typedef Compadre::ParticlesT particle_type;
		typedef Compadre::DOFData dof_data_type;

		Teuchos::RCP<Teuchos::ParameterList> _parameters;
		Teuchos::RCP<const Teuchos::Comm<local_index_type> > _comm;
		const coords_type* _coords;
		Teuchos::RCP<particle_type> _particles;
		mvec_type* _b;
		Teuchos::RCP<const dof_data_type> _dof_data;

	public:

		BoundaryConditionsT(	Teuchos::RCP<particle_type> particles,
								mvec_type* b = NULL);

		virtual ~BoundaryConditionsT() {}
		
		void setParameters(Teuchos::ParameterList& parameters) { _parameters = Teuchos::rcp(&parameters, false); }

		void setMultiVector(mvec_type* b) {
			_b = b;
		}

		void setDOFData(Teuchos::RCP<const dof_data_type> dof_data) {
			_dof_data = dof_data;
		}

		virtual void flagBoundaries() = 0;

		virtual void applyBoundaries(local_index_type field_one, local_index_type field_two = -1, scalar_type time = 0.0) = 0;

		virtual std::vector<InteractingFields> gatherFieldInteractions() = 0;
};

}

#endif
