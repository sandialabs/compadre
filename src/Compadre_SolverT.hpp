#ifndef _COMPADRE_SOLVER_HPP_
#define _COMPADRE_SOLVER_HPP_

#include "CompadreHarness_Config.h"
#include "CompadreHarness_Typedefs.hpp"

#ifdef TRILINOS_LINEAR_SOLVES

namespace Tpetra {
template <typename scalar_type, typename  local_index_type, typename global_index_type, typename node_type>
class Operator;
}

namespace Ifpack2 {
template <typename scalar_type, typename  local_index_type, typename global_index_type, typename node_type>
class Preconditioner;
}

namespace Compadre {

class SolverT {

	protected:

		Teuchos::RCP<Teuchos::ParameterList> _parameters;
		std::string _parameter_filename;
		const Teuchos::RCP<const Teuchos::Comm<int> > _comm;

		Teuchos::RCP<thyra_block_op_type> _A_thyra;
        std::vector<std::vector<Teuchos::RCP<crs_matrix_type> > > _A_tpetra;
        Teuchos::RCP<crs_matrix_type> _A_single_block;

		std::vector<Teuchos::RCP<thyra_mvec_type> > _b_thyra;
		std::vector<Teuchos::RCP<mvec_type> > _b_tpetra;
        Teuchos::RCP<mvec_type> _b_single_block;

		std::vector<Teuchos::RCP<thyra_mvec_type> > _x_thyra;
		std::vector<Teuchos::RCP<mvec_type> > _x_tpetra;
        Teuchos::RCP<mvec_type> _x_single_block;

        std::vector<std::vector<Teuchos::RCP<exporter_type> > > _row_exporters;
        std::vector<std::vector<Teuchos::RCP<exporter_type> > > _range_exporters;
        std::vector<std::vector<Teuchos::RCP<exporter_type> > > _col_exporters;
        std::vector<std::vector<Teuchos::RCP<exporter_type> > > _domain_exporters;

        bool _consolidate_blocks;

	public:

		SolverT( const Teuchos::RCP<const Teuchos::Comm<int> > comm, std::vector<std::vector<Teuchos::RCP<crs_matrix_type> > > A,
				std::vector<Teuchos::RCP<mvec_type> > b);

		virtual ~SolverT() {};

		void setParameters(Teuchos::ParameterList& parameters) {
			_parameters = Teuchos::rcp(&parameters, false);
			_parameter_filename = _parameters->get<std::string>("file");
            this->prepareMatrices();
		}

		Teuchos::RCP<mvec_type> getSolution(local_index_type idx = -1) const;

		void setSolution(local_index_type idx, Teuchos::RCP<mvec_type> vec);

		void solve();

		void residual();

    protected:

        void amalgamateBlockMatrices(std::vector<std::vector<Teuchos::RCP<crs_matrix_type> > > A, 
                                        std::vector<Teuchos::RCP<mvec_type> > b);

        // called by setParameters, requires knowledge of solution method
        void prepareMatrices();


};

}

#endif
#endif
