#include "Compadre_DOFData.hpp"

namespace Compadre {

DOFData::DOFData (Teuchos::RCP<Teuchos::ParameterList> parameters, const int num_fields) : _parameters(parameters) {
	TEUCHOS_TEST_FOR_EXCEPT_MSG(num_fields < -1 || num_fields == 0, "Invalid number of fields provided.");
	if (num_fields < 0) {
		_dof_row_map = std::vector<Teuchos::RCP<map_type> >(1);
		_dof_col_map = std::vector<Teuchos::RCP<map_type> >(1);
	} else {
		_dof_row_map = std::vector<Teuchos::RCP<map_type> >(num_fields);
		_dof_col_map = std::vector<Teuchos::RCP<map_type> >(num_fields);
	}
}

// dof data
// row map and column map produced through dof manager

Teuchos::RCP<const map_type> DOFData::getRowMap(local_index_type idx) const { // of all DOFS
	if (idx==-1)
		return _dof_row_map[0].getConst();
	return _dof_row_map[idx].getConst();
}

Teuchos::RCP<const map_type> DOFData::getColMap(local_index_type idx) const { // of all DOFS
	if (idx==-1)
		return _dof_col_map[0].getConst();
	return _dof_col_map[idx].getConst();
}

local_index_type DOFData::getDOFFromLID(const local_index_type particle_num,
		const local_index_type field_num, const local_index_type component_num) const {
	// access to LOCAL dof# from particle x field x component
	// includes dofs for halo particles
	TEUCHOS_TEST_FOR_EXCEPT_MSG(&_particle_field_component_to_dof_map==NULL, "generateDOFMap() never called");
	return _particle_field_component_to_dof_map[particle_num][field_num][component_num];
}

const std::vector<std::vector<std::vector<local_index_type> > >& DOFData::getDOFMap() const {
	return _particle_field_component_to_dof_map;
}

void DOFData::setRowMap(Teuchos::RCP<map_type> map, local_index_type idx) { // of all DOFS
	TEUCHOS_TEST_FOR_EXCEPT_MSG((size_t)idx >= _dof_row_map.size() && idx!=-1, "Invalid index requested.");
	if (idx==-1)
		_dof_row_map[0] = map;
	else _dof_row_map[idx] = map;
}

void DOFData::setColMap(Teuchos::RCP<map_type> map, local_index_type idx) { // of all DOFS
	TEUCHOS_TEST_FOR_EXCEPT_MSG((size_t)idx >= _dof_col_map.size() && idx!=-1, "Invalid index requested.");
	if (idx==-1)
		_dof_col_map[0] = map;
	else _dof_col_map[idx] = map;
}

void DOFData::setDOFMap(std::vector<std::vector<std::vector<local_index_type> > > particle_field_component_to_dof_map) {
	_particle_field_component_to_dof_map = particle_field_component_to_dof_map;
}

}
