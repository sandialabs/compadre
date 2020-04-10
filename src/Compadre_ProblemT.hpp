#ifndef _COMPADRE_PROBLEM_HPP_
#define _COMPADRE_PROBLEM_HPP_

#include "CompadreHarness_Config.h"
#include "CompadreHarness_Typedefs.hpp"



namespace Compadre {

class CoordsT;
class SolverT;
class ParticlesT;
class PhysicsT;
class SourcesT;
class BoundaryConditionsT;
class ParameterManager;
class DOFData;


/*
* This class should apply boundary conditions, assemble the matrix, and solve the system
*/
class ProblemT {

	protected:

		typedef Compadre::CoordsT coords_type;
		typedef Compadre::SolverT solver_type;

		typedef Compadre::ParticlesT particle_type;

		typedef Compadre::PhysicsT physics_type;
		typedef Compadre::SourcesT source_type;
		typedef Compadre::BoundaryConditionsT bc_type;

		typedef Compadre::ParameterManager parameter_manager_type;
		typedef Compadre::DOFData dof_data_type;

		Teuchos::RCP<const Teuchos::Comm<local_index_type> > _comm;
		Teuchos::RCP<particle_type> _particles;
		Teuchos::RCP<solver_type> _solver;

		Teuchos::RCP<physics_type> _OP;
		Teuchos::RCP<source_type> _RHS;
		Teuchos::RCP<bc_type> _BCS;

		Teuchos::RCP<Teuchos::ParameterList> _parameters;

		const coords_type* _coords;

		std::vector<Teuchos::RCP<const map_type> > row_map;
		std::vector<Teuchos::RCP<const map_type> > col_map;

		std::vector<std::vector<Teuchos::RCP<crs_graph_type> > > A_graph;
		std::vector<std::vector<Teuchos::RCP<crs_matrix_type> > > _A;

		std::vector<Teuchos::RCP<mvec_type> > _b;

		std::vector<InteractingFields> _field_interactions; // each physics, bcs, rhs, etc... registers their interactions

		// maps field numbers to matrix block numbers
		std::vector<local_index_type> _field_to_block_row_map;
		std::vector<local_index_type> _field_to_block_col_map;
		std::vector<local_index_type> _block_row_to_field_map;
		std::vector<local_index_type> _block_col_to_field_map;

		Teuchos::RCP<const DOFData> _problem_dof_data;

	public:

		ProblemT(	Teuchos::RCP<particle_type> particles,
					Teuchos::RCP<Teuchos::ParameterList> parameters,
					Teuchos::RCP<physics_type> physics = Teuchos::null,
					Teuchos::RCP<source_type> sources = Teuchos::null,
					Teuchos::RCP<bc_type> bcs = Teuchos::null);

		ProblemT(	Teuchos::RCP<particle_type> particles,
					Teuchos::RCP<parameter_manager_type> parameter_manager,
					Teuchos::RCP<physics_type> physics = Teuchos::null,
					Teuchos::RCP<source_type> sources = Teuchos::null,
					Teuchos::RCP<bc_type> bcs = Teuchos::null);

		ProblemT(	Teuchos::RCP<particle_type> particles,
					Teuchos::RCP<physics_type> physics = Teuchos::null,
					Teuchos::RCP<source_type> sources = Teuchos::null,
					Teuchos::RCP<bc_type> bcs = Teuchos::null);

		virtual ~ProblemT() {};

		void setPhysics(Teuchos::RCP<physics_type> physics);
		void setSources(Teuchos::RCP<source_type> sources);
		void setBCS(Teuchos::RCP<bc_type> bcs);

		Teuchos::RCP<physics_type> getPhysics() const { return _OP; }
		Teuchos::RCP<source_type> getSources() const { return _RHS; }
		Teuchos::RCP<bc_type> getBCS() const { return _BCS; }

		Teuchos::RCP<const crs_matrix_type> getA(local_index_type row_block = -1, local_index_type col_block = -1) const;
		Teuchos::RCP<const mvec_type> getb(local_index_type row_block = -1) const;

		void initialize(scalar_type initial_simulation_time = 0);
		void solve();
		void residual();

	private:

		void buildMaps(local_index_type field_one = -1, local_index_type field_two = -1);
		void buildGraphs(local_index_type field_one, local_index_type field_two);
		void buildSolver();
		void assembleOperator(local_index_type field_one = 0, local_index_type field_two = -1, scalar_type simulation_time = 0, scalar_type delta_time = 0);
		void assembleRHS(local_index_type field_one = 0, local_index_type field_two = -1, scalar_type simulation_time = 0, scalar_type delta_time = 0);
		void assembleBCS(local_index_type field_one = 0, local_index_type field_two = -1, scalar_type simulation_time = 0, scalar_type delta_time = 0);
		void registerFieldInteractions(const std::vector<InteractingFields>& ops_to_register);
};

}

#endif
