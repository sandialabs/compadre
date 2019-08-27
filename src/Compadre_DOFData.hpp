#ifndef _COMPADRE_DOFDATA_HPP_
#define _COMPADRE_DOFDATA_HPP_

#include "CompadreHarness_Config.h"
#include "CompadreHarness_Typedefs.hpp"

namespace Compadre {

class DOFData {

	protected:

		Teuchos::RCP<Teuchos::ParameterList> _parameters;
		std::vector<Teuchos::RCP<map_type> > _dof_row_map; // contains only dofs local to this processor
		std::vector<Teuchos::RCP<map_type> > _dof_col_map; // contains only dofs local to this processor and in the halo
		std::vector<std::vector<std::vector<local_index_type> > > _particle_field_component_to_dof_map;

	public:

		DOFData(Teuchos::RCP<Teuchos::ParameterList> parameters, const int num_fields=-1);

		virtual ~DOFData() {};

		// dof data is a result of the dof manager
		// row map and column map produced through dof manager

		Teuchos::RCP<const map_type> getRowMap(local_index_type idx = -1) const;

		Teuchos::RCP<const map_type> getColMap(local_index_type idx = -1) const;

		local_index_type getDOFFromLID(const local_index_type particle_num, const local_index_type field_num, const local_index_type component_num) const;

		const std::vector<std::vector<std::vector<local_index_type> > >& getDOFMap() const;

		local_index_type getNumberOfFields() const {
			return _dof_row_map.size();
		}

		void setRowMap(Teuchos::RCP<map_type> map, local_index_type idx=-1);

		void setColMap(Teuchos::RCP<map_type> map, local_index_type idx=-1);

		void setDOFMap(std::vector<std::vector<std::vector<local_index_type> > > particle_field_component_to_dof_map);

};

}

#endif
