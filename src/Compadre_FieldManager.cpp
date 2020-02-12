#include "Compadre_FieldManager.hpp"

#include "Compadre_FieldT.hpp"
#include "Compadre_CoordsT.hpp"
#include "Compadre_ParticlesT.hpp"
#include "Compadre_EuclideanCoordsT.hpp"
#include "Compadre_SphericalCoordsT.hpp"
#include "Compadre_XyzVector.hpp"
#include "Compadre_DOFManager.hpp"


namespace Compadre {

FieldManager::FieldManager ( const particles_type* particles, Teuchos::RCP<Teuchos::ParameterList> parameters ) :
			_parameters(parameters), _particles(particles), total_field_offsets(0), max_num_field_dimensions(0)
{}


// Meta-field functions (operations on the whole set of fields)

void FieldManager::resetAll(const coords_type * coords) {
	// resize the field data to match coordinates
	for (size_t i=0; i<_fields.size(); i++) {
		_fields[i]->resetCoords(coords);
		_fields[i]->resize();
	}
}

void FieldManager::applyZoltan2PartitionToAll() {
	// CAUTION: Be sure that _coords owned by each field has not changed
	// otherwise, a call to resetAll(...) may be in order
	for (size_t i=0; i<_fields.size(); i++) {
		_fields[i]->applyZoltan2Partition();
	}
}

void FieldManager::insertParticlesToAll(const coords_type * coords, const std::vector<xyz_type>& new_pts_vector, const particles_type* other_particles) {//host_view_type& inserted_field_values) {
	for (size_t i=0; i<_fields.size(); i++) {
		const host_view_type& field_vals = other_particles->getFieldManagerConst()->getFieldByID(i)->getMultiVectorPtrConst()->getLocalView<host_view_type>();
		_fields[i]->insertParticles(coords, new_pts_vector, field_vals);
	}
}

void FieldManager::removeParticlesToAll(const coords_type * coords, const std::vector<local_index_type>& coord_ids) {
	for (size_t i=0; i<_fields.size(); i++) {
		_fields[i]->removeParticles(coords, coord_ids);
	}
}



// Update Field Information

void FieldManager::updateMaxNumFieldDimensions() {
	max_num_field_dim_updated = true;

	max_num_field_dimensions = _fields[0]->nDim();
	for (size_t i=1; i<_fields.size(); i++) {
		max_num_field_dimensions = std::max(max_num_field_dimensions, _fields[i]->nDim());
	}
}

void FieldManager::updateTotalFieldDimensions() {
	total_field_dimension_updated = true;

	total_field_offsets = _fields[0]->nDim();
	for (size_t i=1; i<_fields.size(); i++) {
		total_field_offsets += (_fields[i])->nDim();
	}
}

void FieldManager::updateFieldOffsets() {
	field_offsets_updated = true;

	field_offsets.resize(_fields.size());
	field_offsets[0] = 0;

	for (size_t i=1; i<_fields.size(); i++) {
		field_offsets[i] = field_offsets[i-1] + (_fields[i-1])->nDim();
	}
}



// Retrieve Field Information (if updated)

local_index_type FieldManager::getMaxNumFieldDimensions() const {
	TEUCHOS_TEST_FOR_EXCEPT_MSG(!max_num_field_dim_updated, "getMaxNumFieldDimensions() called before updateMaxNumFieldDimensions()");
	return max_num_field_dimensions;
}

local_index_type FieldManager::getTotalFieldDimensions() const {
	TEUCHOS_TEST_FOR_EXCEPT_MSG(!total_field_dimension_updated, "getTotalFieldDimensions() called before updateTotalFieldDimensions()");
	return total_field_offsets;
}

local_index_type FieldManager::getFieldOffset(local_index_type idx) const {
	TEUCHOS_TEST_FOR_EXCEPT_MSG(!field_offsets_updated, "getFieldOffset() called before updateFieldOffsets()");
	return field_offsets[idx];
}


// Modify Fields / Data

Teuchos::RCP<FieldManager::field_type> FieldManager::createField(const local_index_type nDim, const std::string name, const std::string units, const FieldSparsityType fst) {
	total_field_dimension_updated = false;
	field_offsets_updated = false;
	max_num_field_dim_updated = false;

	_field_map.insert(std::pair<const std::string, const local_index_type>(name, _fields.size())); // size() is the correct index, since not yet appended
	_fields.push_back(Teuchos::rcp(new FieldManager::field_type(_particles->getCoordsConst(), nDim, name, units, fst)));
	// since an insertion of a field offsets every other field, this should only be marked as modified
	// and later regenerated all at once

	this->updateTotalFieldDimensions();
	this->updateFieldOffsets();
	this->updateMaxNumFieldDimensions();
	return _fields.back();
}


void FieldManager::updateFieldsHaloData(local_index_type field_num) {
	if (field_num==-1) {
		for (size_t i=0; i<_fields.size(); i++) {
			_fields[i]->updateHalo();
			_fields[i]->updateHaloData();
		}
	} else {
		_fields[field_num]->updateHalo();
		_fields[field_num]->updateHaloData();
	}
}

void FieldManager::updateFieldsFromVector(Teuchos::RCP<mvec_type> source, local_index_type field_num, Teuchos::RCP<const dof_data_type> vector_dof_data) {

	// first copy data out of source into field values, then perform an import from data to halo_data on that field

	typedef Kokkos::View<const global_index_type*> const_gid_view_type;

	// copy data to build column map
	Teuchos::RCP<const map_type> particle_map = _particles->getCoordsConst()->getMapConst(false /* no halo */);
	const_gid_view_type particle_gids_locally_owned = particle_map->getMyGlobalIndices();

	TEUCHOS_TEST_FOR_EXCEPT_MSG(field_num==-1, "field_num must be specified for updateFieldsFromVector.");

	// Determine if the vector given to fill the field is larger than the # of entries in the field, in
	// which case we assume it is a global vector of all fields, and we find the appropriate offset.
	// Otherwise if they are the same size, we assume the vector is only for this field.

	// blocked update (vector used to update only contains this field)

	host_view_type field_vals = _fields[field_num]->getLocalVectorVals()->getLocalView<host_view_type>(); // just this fields current info

	host_view_type source_data = source->getLocalView<host_view_type>(); // this field's new info

	// copy data from source to field_vals
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,particle_gids_locally_owned.extent(0)), KOKKOS_LAMBDA(const int i) {
		for (local_index_type j=0; j<_fields[field_num]->nDim(); j++) {
			field_vals(i,j) = source_data(i*this->_fields[field_num]->nDim() + j,0);
		}
	});

	this->updateFieldsHaloData(field_num);

}

void FieldManager::updateVectorFromFields(Teuchos::RCP<mvec_type> target, local_index_type field_num, Teuchos::RCP<const dof_data_type> vector_dof_data) const {

	// first copy data out of field values into target
	// halo_data does not get transferred

	typedef Kokkos::View<const global_index_type*> const_gid_view_type;

	// copy data to build column map
	Teuchos::RCP<const map_type> particle_map = _particles->getCoordsConst()->getMapConst(false /* no halo */);
	const_gid_view_type particle_gids_locally_owned = particle_map->getMyGlobalIndices();

	TEUCHOS_TEST_FOR_EXCEPT_MSG(field_num==-1, "field_num must be specified for updateVectorFromFields.");

	// blocked update (vector used to update only contains this field)

	host_view_type field_vals = _fields[field_num]->getLocalVectorVals()->getLocalView<host_view_type>(); // just this fields current info

	host_view_type target_data = target->getLocalView<host_view_type>(); // this field's new info

	// copy data from source to field_vals
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,particle_gids_locally_owned.extent(0)), KOKKOS_LAMBDA(const int i) {
		for (local_index_type j=0; j<_fields[field_num]->nDim(); j++) {
			target_data(i*this->_fields[field_num]->nDim() + j,0) = field_vals(i,j);
		}
	});
}

const std::vector<Teuchos::RCP<FieldManager::field_type> >& FieldManager::getVectorOfFields() const {
	return _fields;
}

local_index_type FieldManager::getIDOfFieldFromName(const std::string& name) const {;
	std::map<const std::string, const local_index_type>::const_iterator it = _field_map.find(name);
	if (it==_field_map.end()) {
		TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Field " + name + " not found. Possibilities listed. ");
		this->listFields(std::cout);
	}
	return it->second;
}

Teuchos::RCP<FieldManager::field_type> FieldManager::getFieldByID(const local_index_type field_num) const {
	return _fields[field_num];
}

Teuchos::RCP<FieldManager::field_type> FieldManager::getFieldByName(const std::string& name) const {
	return _fields[this->getIDOfFieldFromName(name)];
}


// Output Field Info

void FieldManager::listFields(std::ostream& os) const {
	if (_particles->getCoordsConst()->getComm()->getRank()==0) {
		os << "Fields registered:\n";
		os << "==================\n";
		for (size_t i=0; i<_fields.size(); i++) os << (_fields[i])->getName() << std::endl;
	}
}

void FieldManager::printAllFields(std::ostream& os) const {
	for (size_t i=0; i<_fields.size(); i++) {
		os << "Proc#: " << _particles->getCoordsConst()->getComm()->getRank() << std::endl;
		os << (_fields[i])->getName() << std::endl;
		(_fields[i])->print(os);
	}
}

}
