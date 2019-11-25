#ifndef _COMPADRE_PHYSICS_HPP_
#define _COMPADRE_PHYSICS_HPP_

#include "CompadreHarness_Config.h"
#include "CompadreHarness_Typedefs.hpp"

namespace Compadre {

class CoordsT;
class ParticlesT;
class DOFData;

class PhysicsT {

	protected:

		typedef Compadre::CoordsT coords_type;
		typedef Compadre::ParticlesT particle_type;
		typedef Compadre::DOFData dof_data_type;

		Teuchos::RCP<Teuchos::ParameterList> _parameters;
		Teuchos::RCP<const Teuchos::Comm<local_index_type> > _comm;
		const coords_type* _coords;
		Teuchos::RCP<particle_type> _particles;
		Teuchos::RCP<const map_type> _row_map;
		Teuchos::RCP<const map_type> _col_map;
		Teuchos::RCP<crs_graph_type> _A_graph;
		Teuchos::RCP<crs_matrix_type> _A;
		Teuchos::RCP<mvec_type> _b;
		Teuchos::RCP<const dof_data_type> _dof_data;

	public:

		PhysicsT(	Teuchos::RCP<particle_type> particles,
					Teuchos::RCP<crs_graph_type> A_graph = Teuchos::null,
					Teuchos::RCP<crs_matrix_type> A = Teuchos::null);

		virtual ~PhysicsT() {};

		void setParameters(Teuchos::ParameterList& parameters) { _parameters = Teuchos::rcp(&parameters, false); }

		void setRowMap(Teuchos::RCP<const map_type> row_map) {
			_row_map = row_map;
		}

		void setColMap(Teuchos::RCP<const map_type> col_map) {
			_col_map = col_map;
		}

		void setGraph(Teuchos::RCP<crs_graph_type> A_graph) {
			_A_graph = A_graph;
		}
		
		void setMatrix(Teuchos::RCP<crs_matrix_type> A) {
			_A = A;
		}

		void setMultiVector(Teuchos::RCP<mvec_type> b) {
			_b = b;
		}

		void setDOFData(Teuchos::RCP<const dof_data_type> dof_data) {
			_dof_data = dof_data;
		}

		virtual Teuchos::RCP<crs_graph_type> computeGraph(local_index_type field_one, local_index_type field_two = -1) { return Teuchos::null; };

		virtual void computeMatrix(local_index_type field_one, local_index_type field_two = -1, scalar_type time = 0.0) {};

		virtual void computeVector(local_index_type field_one, local_index_type field_two = -1, scalar_type time = 0.0) {};

		virtual const std::vector<InteractingFields> gatherFieldInteractions() = 0;

		virtual double getStableTimeStep(double mesh_size = 0) {
			return 0;
		}

        // any problem data that needs generated once before assembly
        virtual void initialize() {}

        // default comes from particles class 
        virtual local_index_type getMaxNumNeighbors();

};

}

#endif
