#include "Compadre_FieldT.hpp"

#include "Compadre_CoordsT.hpp"
#include "Compadre_AnalyticFunctions.hpp"

namespace Compadre {

FieldT::FieldT(const coords_type* coords, const int nDim, const std::string name, const std::string units, const FieldSparsityType fst)
				: _name(name), _units(units), _nDim(nDim), _coords(coords), _field_sparsity(fst) {
	   const bool setToZero = true;
	   _vals = Teuchos::rcp(new mvec_type(coords->getMapConst(), nDim, setToZero));
	   _halo_vals = Teuchos::rcp(new mvec_type(coords->getMapConst(true /*halo*/), nDim, setToZero));
}

void FieldT::resize() {
	// anything initialized off of a constructor of this class should be resized/reinitialized here
	const bool setToZero = true;
   _vals = Teuchos::rcp(new mvec_type(_coords->getMapConst(), nDim(), setToZero));
}

void FieldT::resetCoords(const coords_type* coords) {
	_coords = coords;
}

void FieldT::applyZoltan2Partition() {
	_coords->applyZoltan2Partition(this->_vals);
	TEUCHOS_TEST_FOR_EXCEPT_MSG((size_t)(_coords->nLocal())!=this->_vals->getLocalLength(), "After applyZoltan2Partition(), size of field differs from coordinates.");
}

void FieldT::insertParticles(const coords_type * coords, const std::vector<xyz_type>& new_pts_vector, const host_view_type& inserted_field_values) {
	TEUCHOS_TEST_FOR_EXCEPT_MSG(inserted_field_values.extent(0)!=new_pts_vector.size(), "Mismatch between field data generated in FieldManager and the number of points inserted.");
	TEUCHOS_TEST_FOR_EXCEPT_MSG(inserted_field_values.extent(1)!=(size_t)_nDim, "Mismatch between field data dimension in FieldManager and the field being inserted into.");
	const local_index_type num_added_coords = (local_index_type)new_pts_vector.size();

	const bool setToZero = false;
	Teuchos::RCP<mvec_type> new_vals = Teuchos::rcp(new mvec_type(coords->getMapConst(), _nDim, setToZero));
	host_view_type new_vals_data = new_vals->getLocalView<host_view_type>();
	host_view_type old_vals_data = _vals->getLocalView<host_view_type>();
	const local_index_type old_vals_data_num = (local_index_type)(old_vals_data.extent(0));
	// copy old data
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,old_vals_data_num), KOKKOS_LAMBDA(const int i) {
		for (local_index_type j=0; j<_nDim; j++) {
			new_vals_data(i,j) = old_vals_data(i,j);
		}
	});
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,num_added_coords), KOKKOS_LAMBDA(const int i) {
		for (local_index_type j=0; j<_nDim; j++) {
			new_vals_data(old_vals_data_num + i,j) = inserted_field_values(i,j);
//			if (inserted_field_values(i,j)!=0) {
//				std::cout << i << " by " << j << ": " << inserted_field_values(i,j);
//				std::cout << std::endl;
//			}
		}
	});
	_vals = new_vals;
	_coords = coords;

	_halo_vals = Teuchos::null;
}

void FieldT::removeParticles(const coords_type * coords, const std::vector<local_index_type>& coord_ids /* already ordered */) {
	const local_index_type num_remove_coords = (local_index_type)coord_ids.size();
	Kokkos::View <const local_index_type*,Kokkos::HostSpace,Kokkos::MemoryTraits<Kokkos::Unmanaged> >
		coord_ids_kok (&coord_ids[0] , num_remove_coords);

	const bool setToZero = false;
	Teuchos::RCP<mvec_type> new_vals = Teuchos::rcp(new mvec_type(coords->getMapConst(), _nDim, setToZero));
	host_view_type new_vals_data = new_vals->getLocalView<host_view_type>();
	host_view_type old_vals_data = _vals->getLocalView<host_view_type>();
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,old_vals_data.extent(0)), KOKKOS_LAMBDA(const int i) {
		bool deleted = false;
		local_index_type offset = num_remove_coords;
		for (local_index_type j=0; j<num_remove_coords; j++) {
			if (i == coord_ids_kok[j]) {
				deleted = true;
				break;
			} else if (i < coord_ids_kok[j]) {
				offset = j;
				break;
			}
		}
		if (!deleted) {
			for (local_index_type j=0; j<_nDim; j++) {
				new_vals_data(i-offset,j) = old_vals_data(i,j);
			}
		}
	});
	_vals = new_vals;
	_coords = coords;
//	std::cout << "old halo val size: " << _halo_vals->getLocalLength() << " " << _halo_vals->getGlobalLength() << std::endl;
	_halo_vals = Teuchos::null;
}

void FieldT::updateHalo() {
	const bool setToZero = true;
	_halo_vals = Teuchos::rcp(new mvec_type(_coords->getMapConst(true /*halo*/), _nDim, setToZero));
//	std::cout << "new halo size: " << _coords->getMapConst(true /*halo*/)->getNodeNumElements () << " " << _halo_vals->getGlobalLength() << std::endl;
//	if (_coords->getMapConst(true).getRawPtr()==_halo_vals->getMap().getRawPtr()) std::cout << "halo maps are the same." << std::endl;
}

void FieldT::localInitFromScalarFunction(function_type* fn, bool use_physical_coords) {
	host_view_type fieldView = this->_vals->getLocalView<host_view_type>();
	host_view_type coordsView = _coords->getPts(false /*halo*/, use_physical_coords)->getLocalView<host_view_type>();
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,this->_coords->nLocalMax()),
		evaluateScalar(fieldView, coordsView, fn));
}

void FieldT::localInitFromScalarFunctionGradient(function_type* fn, bool use_physical_coords) {
	host_view_type fieldView = this->_vals->getLocalView<host_view_type>();
	host_view_type coordsView = _coords->getPts(false /*halo*/, use_physical_coords)->getLocalView<host_view_type>();
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,this->_coords->nLocalMax()),
		evaluateVector(fieldView, coordsView, fn, 1));
}

void FieldT::localInitFromVectorFunction(function_type* fn, bool use_physical_coords) {
	host_view_type fieldView = this->_vals->getLocalView<host_view_type>();
	host_view_type coordsView = _coords->getPts(false /*halo*/, use_physical_coords)->getLocalView<host_view_type>();
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,this->_coords->nLocalMax()),
		evaluateVector(fieldView, coordsView, fn));
}

device_view_type FieldT::getDeviceView() {
    return this->_vals->getLocalView<device_view_type>();
}

scalar_type FieldT::getLocalScalarVal(const local_index_type idx, const local_index_type component, bool for_halo) const {
	device_view_type fieldView = for_halo ? this->_halo_vals->getLocalView<device_view_type>() : this->_vals->getLocalView<device_view_type>();
	return fieldView(idx,component);
}

std::vector<scalar_type> FieldT::getLocalVectorVal(const local_index_type idx, bool for_halo) const {
	device_view_type fieldView = for_halo ? this->_halo_vals->getLocalView<device_view_type>() : this->_vals->getLocalView<device_view_type>();
	std::vector<scalar_type> return_vals(_nDim);
	for (local_index_type i=0; i<_nDim; i++) return_vals[i] = fieldView(idx,i);
	return return_vals;
}

void FieldT::updateMultiVector(const scalar_type& alpha, const Teuchos::RCP<FieldT> fieldA, const scalar_type& beta) {
    this->_vals->update(alpha, *(fieldA->_vals), beta);
}

Teuchos::RCP<mvec_type> FieldT::getMultiVectorPtr() {
    return this->_vals;
}

const mvec_type* FieldT::getMultiVectorPtrConst() const {
    return this->_vals.getRawPtr();
}

Teuchos::RCP<mvec_type> FieldT::getHaloMultiVectorPtr() {
    return this->_halo_vals;
}

const mvec_type* FieldT::getHaloMultiVectorPtrConst() const {
    return this->_halo_vals.getRawPtr();
}

std::vector<scalar_type> FieldT::normInf() const {
    std::vector<scalar_type> result(this->_vals->getNumVectors());
    this->_vals->normInf(result);
    return result;
}

std::vector<scalar_type> FieldT::norm2() const {
    std::vector<scalar_type> result(this->_vals->getNumVectors());
    this->_vals->norm2(result);
    return result;
}

void FieldT::updateMultiVector(const scalar_type& alpha, const Teuchos::RCP<FieldT> fieldA, 
                               const scalar_type& beta, const Teuchos::RCP<FieldT> fieldB, const scalar_type& gamma) {
    this->_vals->update(alpha, *(fieldA->_vals), beta, *(fieldB->_vals), gamma);
}

const Teuchos::RCP<mvec_type>& FieldT::getLocalVectorVals(bool for_halo) const {
	return for_halo ? this->_halo_vals : this->_vals;
}

void FieldT::updateHaloData() {
	Teuchos::RCP<const importer_type> halo_importer = _coords->getHaloImporterConst();
//	halo_importer->print(std::cout);
//	std::cout << _vals->getLocalLength() << " " << _halo_vals->getLocalLength() << std::endl;
//	_vals->print(std::cout);
//	_halo_vals->print(std::cout);
	// move data into halos
	_halo_vals->doImport(*_vals, *halo_importer, Tpetra::CombineMode::INSERT);
}

void FieldT::syncMemory() {
	_vals->sync<device_view_type>();
	_vals->sync<host_view_type>();
}

void FieldT::print(std::ostream& os) const {
	auto out = Teuchos::getFancyOStream(Teuchos::rcpFromRef(os));
	_vals->describe(*out, Teuchos::VERB_EXTREME);
}

void FieldT::scale(const scalar_type a) {
    _vals->scale(a);
}

void FieldT::setLocalScalarVal(const local_index_type idx, const local_index_type component, const double value) {
	host_view_type fieldView = this->_vals->getLocalView<host_view_type>();
	fieldView(idx,component) = value;
}

}
