#include "Compadre_ParticlesT.hpp"

#include "Compadre_FieldT.hpp"
#include "Compadre_CoordsT.hpp"
#include "Compadre_EuclideanCoordsT.hpp"
#include "Compadre_SphericalCoordsT.hpp"
#include "Compadre_FieldManager.hpp"
#include "Compadre_NeighborhoodT.hpp"
#include "Compadre_DOFManager.hpp"
#include "Compadre_XyzVector.hpp"
#include "Compadre_RemapManager.hpp"

namespace Compadre {

ParticlesT::ParticlesT (Teuchos::RCP<Teuchos::ParameterList> parameters,
		const Teuchos::RCP<const Teuchos::Comm<int> >& comm, const int nDim) :
			_comm(comm), _parameters(parameters), _nDim(nDim),
			total_field_offsets(0)
{
	std::string coords_type_name =_parameters->get<Teuchos::ParameterList>("coordinates").get<std::string>("type");
	transform(coords_type_name.begin(), coords_type_name.end(), coords_type_name.begin(), ::tolower);
	if (coords_type_name == "spherical") {
		_coords = Teuchos::rcp_static_cast<Compadre::CoordsT>(Teuchos::rcp(new Compadre::SphericalCoordsT(0,_comm, nDim)));
	} else {
		_coords = Teuchos::rcp_static_cast<Compadre::CoordsT>(Teuchos::rcp(new Compadre::EuclideanCoordsT(0,_comm, nDim)));
	}
	if (_parameters->get<std::string>("simulation type")=="lagrangian") _coords->setLagrangian();
	_coords->setUnits(_parameters->get<Teuchos::ParameterList>("coordinates").get<std::string>("units"));

	const bool setToZero = true;
	_flag = Teuchos::rcp(new mvec_local_index_type(_coords->getMapConst(), 1, setToZero));
	_fieldManager = Teuchos::rcp(new Compadre::FieldManager(this, _parameters));
}


ParticlesT::ParticlesT (Teuchos::RCP<Teuchos::ParameterList> parameters,
		Teuchos::RCP<coords_type> coords) :
			_comm(coords->getComm()), _parameters(parameters), _nDim(coords->nDim()),
			total_field_offsets(0)
{
	_coords = coords;
	if (_parameters->get<std::string>("simulation type")=="lagrangian") _coords->setLagrangian();
	_coords->setUnits(_parameters->get<Teuchos::ParameterList>("coordinates").get<std::string>("units"));

	const bool setToZero = true;
	_flag = Teuchos::rcp(new mvec_local_index_type(_coords->getMapConst(), 1, setToZero));
	_fieldManager = Teuchos::rcp(new Compadre::FieldManager(this, _parameters));
}



// Coordination calls that resize all fields, flags, and IDs to coordinate changes

void ParticlesT::insertParticles(const std::vector<xyz_type>& new_pts_vector, const scalar_type rebuilt_halo_size,
		bool repartition, bool inserting_physical_coords, bool repartition_using_physical_coords, bool verify_coords_on_processor) {
	TEUCHOS_TEST_FOR_EXCEPT_MSG(_parameters->get<Teuchos::ParameterList>("io").get<bool>("preserve gids"), "Preservation of global IDs is incompatible with inserting particles.\n");
	/*
	 * A few important details on this function.
	 *
	 * 1. Determine if all point insertions are local to this processor.
	 *
	 * 2. Create a set of coordinates from the points being inserted that then initialize a particle set.
	 *
	 * 3. Call RemapManager from the original particles set to the new (temporary) particle set
	 *
	 * 4. Information from the new (temporary) particle set is then merged with the existing field data.
	 *
	 */

	// 	1.
	// verification will fail if the processor boundary boxes from a previous
	// partition were done in a difference coordinate frame
	if (verify_coords_on_processor)
		_coords->verifyCoordsOnProcessor(new_pts_vector, inserting_physical_coords);
	// bounding boxes are determined by partition which is either taken with respect to material or physical coordinates
	// in the future, this will limit the types of coords that can be inserted, material won't be able to be inserted into physical, etc..

	// create coordinates temporarily (how to get the derived type?)
	std::string coords_type_name =_parameters->get<Teuchos::ParameterList>("coordinates").get<std::string>("type");
	transform(coords_type_name.begin(), coords_type_name.end(), coords_type_name.begin(), ::tolower);
	Teuchos::RCP<coords_type> temp_coords;
	if (coords_type_name == "spherical") {
		temp_coords = Teuchos::rcp_static_cast<Compadre::CoordsT>(Teuchos::rcp(new Compadre::SphericalCoordsT(0,_comm)));
	} else {
		temp_coords = Teuchos::rcp_static_cast<Compadre::CoordsT>(Teuchos::rcp(new Compadre::EuclideanCoordsT(0,_comm)));
	}
	// start by filling both sets of coordinates
	// could be further optimized by a factor of 2x by not filling both since one will be written over
	if (_parameters->get<std::string>("simulation type")=="lagrangian") temp_coords->setLagrangian();
	temp_coords->insertCoords(new_pts_vector);

	// create particles from coordinates temporarily
	Teuchos::RCP<particles_type> temp_particles = Teuchos::rcp(new particles_type(this->getParameters(), temp_coords));

	// create a remap object and use this one we are in as the source and the temp one as the target
	Teuchos::RCP<remap_manager_type> remapManager = Teuchos::rcp(new remap_manager_type(_parameters, this, temp_particles.getRawPtr(), _coords->getHaloSize()));

	if (_coords->isLagrangian()) {
		// insertion requires interpolating either Lagrangian or physical coordinates
		if (inserting_physical_coords) { // expected choice, insert into physical, need to get Lagrangian coords
			Compadre::RemapObject remapObject(-20);// -20 is special flag for remap to indicate Lagrangian interpolated using physical values
			remapManager->add(remapObject);
		} else {
			Compadre::RemapObject remapObject(-21);// -21 is special flag for remap to indicate physical interpolated using Lagrangian values
			remapManager->add(remapObject);
		}
	}

	// re-register all fields from the current particles
	const std::vector<Teuchos::RCP<field_type> >& existing_fields = this->getFieldManagerConst()->getVectorOfFields();
	for (size_t i=0; i<existing_fields.size(); i++) {
		temp_particles->getFieldManager()->createField(existing_fields[i]->nDim(), existing_fields[i]->getName(), existing_fields[i]->getUnits());
		Compadre::RemapObject remapObject(i);
		remapManager->add(remapObject);
	}


	remapManager->execute(false /*keep neighborhoods*/, false /*keep GMLS*/, false /*reuse neighborhoods*/, false /*reuse GMLS*/, inserting_physical_coords);

	// call this->mergeWith(temp particles) to combine data into one particle set
	this->mergeWith(temp_particles.getRawPtr());

	/*
	 *  NOTE: halo rebuilding must be performed. If a repartition is needed for load balancing issues or change or processor
	 *  bounding boxes, then a repartition is needed, which should precede the halo rebuild.
	 */

	// Either physical or material coordinates were provided for insertion,
	// and their respective complement was determined through remap
	// A repartition and computation of the halo must now be performed,
	// but it does not have to be done w.r.t. the same coordinates as
	// were inserted

	if (repartition) this->zoltan2Initialize(repartition_using_physical_coords);

	if (rebuilt_halo_size==0) this->buildHalo(_coords->getHaloSize(), repartition_using_physical_coords);
	else this->buildHalo(rebuilt_halo_size, repartition_using_physical_coords);
	_fieldManager->updateFieldsHaloData();

}

void ParticlesT::removeParticles(const std::vector<local_index_type>& coord_ids, const scalar_type rebuilt_halo_size, bool repartition, bool repartition_using_physical_coords) {
	TEUCHOS_TEST_FOR_EXCEPT_MSG(_parameters->get<Teuchos::ParameterList>("io").get<bool>("preserve gids"), "Preservation of global IDs is incompatible with removing particles.\n");

	std::vector<local_index_type> coord_ids_ordered = coord_ids;
	std::sort (coord_ids_ordered.begin(), coord_ids_ordered.end());

	const local_index_type num_remove_coords = (local_index_type)coord_ids.size();
	Kokkos::View <const local_index_type*,Kokkos::HostSpace,Kokkos::MemoryTraits<Kokkos::Unmanaged> >
		coord_ids_kok (&coord_ids_ordered[0] , num_remove_coords);

	// _coords comes back with a new map and correctly sized
	_coords->removeCoords(coord_ids_ordered);

	const bool setToZero = false;
	Teuchos::RCP<mvec_local_index_type> new_flag = Teuchos::rcp(new mvec_local_index_type(_coords->getMapConst(), 1, setToZero));
	host_view_local_index_type new_bc_id = new_flag->getLocalView<host_view_local_index_type>();
	host_view_local_index_type old_bc_id = _flag->getLocalView<host_view_local_index_type>();
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,old_bc_id.extent(0)), KOKKOS_LAMBDA(const int i) {
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
		if (!deleted) new_bc_id(i-offset,0) = old_bc_id(i,0);
	});
	_flag = new_flag; // assigns new flags and deallocates old tpetra vector

	// need sophisticated copy of data here
	_fieldManager->removeParticlesToAll(_coords.getRawPtr(), coord_ids_ordered); // do the same as this function, but for fields


	/*
	 *  NOTE: halo update should really be done after a zoltan2 repartition if processor boundaries changed or
	 *  load on processors is now unbalanced. We allow the user to decide this, but by default assume
	 *  processor boundaries will not change and there will not be a load imbalance (so no repartition)
	 */

	if (repartition) this->zoltan2Initialize(repartition_using_physical_coords);

	// extra machinery that must be called to support halos, etc...
	if (rebuilt_halo_size==0) this->buildHalo(_coords->getHaloSize(), repartition_using_physical_coords);
	else this->buildHalo(rebuilt_halo_size, repartition_using_physical_coords);
	_fieldManager->updateFieldsHaloData();
}

void ParticlesT::mergeWith(const particles_type* other_particles) {
	TEUCHOS_TEST_FOR_EXCEPT_MSG(_parameters->get<Teuchos::ParameterList>("io").get<bool>("preserve gids"), "Preservation of global IDs is incompatible with merging particles.\n");

	// we could copy data using Kokkos arrays, but converting to std::vector isn't that expensive and
	// allows us to reuse the existing interface to CoordsT for inserting points
	host_view_type other_pts_vals;
	host_view_type other_pts_physical_vals;
	local_index_type other_pts_num;
	local_index_type other_pts_physical_num=0;
	if (other_particles->getCoordsConst()->isLagrangian()) {
		other_pts_vals = other_particles->getCoordsConst()->getPts(false /*halo*/, false)->getLocalView<host_view_type>();
		other_pts_physical_vals = other_particles->getCoordsConst()->getPts()->getLocalView<host_view_type>();
		other_pts_physical_num = other_pts_physical_vals.extent(0);
	} else {
		other_pts_vals = other_particles->getCoordsConst()->getPts()->getLocalView<host_view_type>();
		other_pts_physical_vals = other_particles->getCoordsConst()->getPts(false /*halo*/, false)->getLocalView<host_view_type>();
	}
	other_pts_num = other_pts_vals.extent(0);

	std::vector<xyz_type> copied_pts(other_pts_num);
	std::vector<xyz_type> copied_pts_physical;

	for (local_index_type i=0; i<other_pts_num; i++) {
		copied_pts[i] = xyz_type(other_pts_vals(i,0), (_nDim>1)*other_pts_vals(i,1), (_nDim>2)*other_pts_vals(i,2));
	}

	if (other_particles->getCoordsConst()->isLagrangian()) {
		copied_pts_physical.resize(other_pts_physical_num);
		for (local_index_type i=0; i<other_pts_physical_num; i++) {
			copied_pts_physical[i] = xyz_type(other_pts_physical_vals(i,0), (_nDim>1)*other_pts_physical_vals(i,1), (_nDim>2)*other_pts_physical_vals(i,2));
		}
		_coords->insertCoords(copied_pts, copied_pts_physical);
	} else {
		_coords->insertCoords(copied_pts);
	}


    // flags
	const bool setToZero = false;
	Teuchos::RCP<mvec_local_index_type> new_flag = Teuchos::rcp(new mvec_local_index_type(_coords->getMapConst(), 1, setToZero));
	host_view_local_index_type new_bc_id = new_flag->getLocalView<host_view_local_index_type>();
	host_view_local_index_type old_bc_id = _flag->getLocalView<host_view_local_index_type>();
	host_view_local_index_type other_bc_id = other_particles->getFlags()->getLocalView<host_view_local_index_type>();
	// copy over old boundary ids
	const local_index_type old_bc_id_num = (local_index_type)(old_bc_id.extent(0));
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,old_bc_id_num), KOKKOS_LAMBDA(const int i) {
		new_bc_id(i,0) = old_bc_id(i,0);
	});
	// NOTE, FOR NOW WE ONLY USE 0 AS BC_ID FOR NEW PARTICLES
	// TODO: Should this be a parameter or a more complicated struct passed to this function?
	// copy over old boundary ids
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,other_pts_num), KOKKOS_LAMBDA(const int i) {
		new_bc_id(old_bc_id_num + i,0) = other_bc_id(i,0);
	});
	_flag = new_flag; // assigns new flags and deallocates old tpetra vector


	_fieldManager->insertParticlesToAll(_coords.getRawPtr(), copied_pts, other_particles); // do the same as this function, but for fields
}

void ParticlesT::setCoords(Teuchos::RCP<CoordsT> coords) {
	_coords = coords;

	if (_parameters->get<std::string>("simulation type")=="lagrangian") _coords->setLagrangian();
	_coords->setUnits(_parameters->get<Teuchos::ParameterList>("coordinates").get<std::string>("units"));

	const bool setToZero = true;
	_flag = Teuchos::rcp(new mvec_local_index_type(_coords->getMapConst(), 1, setToZero));
	_fieldManager->resetAll(coords.getRawPtr());
}

void ParticlesT::resetWithSameCoords() {
	const bool setToZero = true;
	_flag = Teuchos::rcp(new mvec_local_index_type(_coords->getMapConst(), 1, setToZero));
	_fieldManager->resetAll(_coords.getRawPtr());
}

void ParticlesT::snapLagrangianCoordsToPhysicalCoords() {
	// wrapper for calling resetLagrangianCoords on the coordinates object
	_coords->snapLagrangianCoordsToPhysicalCoords();
}

void ParticlesT::snapPhysicalCoordsToLagrangianCoords() {
	// wrapper for calling resetLagrangianCoords on the coordinates object
	_coords->snapPhysicalCoordsToLagrangianCoords();
}

void ParticlesT::updatePhysicalCoordinatesFromField(std::string field_name, const double multiplier) {
	const mvec_type* field_mvec = this->getFieldManagerConst()->getFieldByName(field_name)->getMultiVectorPtrConst();
	// wrapper for calling resetLagrangianCoords on the coordinates object
	_coords->deltaUpdatePhysicalCoordsFromVectorByTimestep(multiplier, field_mvec);
}

void ParticlesT::resize(const global_index_type nn, bool local_resize) {
	// first resize coordinates, then reinitialize/resize all things depending on coordinates
	// next, resize data in each field

	// anything initialized off of a constructor of this class should be resized/reinitialized here
	if (local_resize) {
		_coords->localResize(nn);
	} else {
		_coords->globalResize(nn);
	}

	const bool setToZero = true;
	_flag = Teuchos::rcp(new mvec_local_index_type(_coords->getMapConst(), 1, setToZero));
//			auto out = Teuchos::getFancyOStream(Teuchos::rcpFromRef(std::cout));
//			flag->describe(*out, Teuchos::VERB_EXTREME);

	_fieldManager->resetAll(_coords.getRawPtr());
}

void ParticlesT::resize(host_view_global_index_type gids) {
	// first resize coordinates, then reinitialize/resize all things depending on coordinates
	// next, resize data in each field

	// anything initialized off of a constructor of this class should be resized/reinitialized here
	_coords->localResize(gids);

	const bool setToZero = true;
	_flag = Teuchos::rcp(new mvec_local_index_type(_coords->getMapConst(), 1, setToZero));
//			auto out = Teuchos::getFancyOStream(Teuchos::rcpFromRef(std::cout));
//			flag->describe(*out, Teuchos::VERB_EXTREME);

	_fieldManager->resetAll(_coords.getRawPtr());
}

void ParticlesT::zoltan2Initialize(bool use_physical_coords) {
	// if box data not found in VTKs
	_coords->zoltan2Init(use_physical_coords);
	_coords->zoltan2Partition();
	_coords->applyZoltan2Partition(this->_flag);
	_fieldManager->applyZoltan2PartitionToAll();
	TEUCHOS_TEST_FOR_EXCEPT_MSG((size_t)(_coords->nLocal())!=this->_flag->getLocalLength(), "After applyZoltan2Partition(), size of flag differs from coordinates.");
}



// Coordinate access

Teuchos::RCP<const map_type> ParticlesT::getCoordMap() const {
    return _coords->getMapConst();
}



// Halo functionality

void ParticlesT::buildHalo(scalar_type h, bool use_physical_coords) {
	// calls buildHalo on coordinates, but make uses this halo connectivity information for fields also
	_coords->buildHalo(h, use_physical_coords);
	// fields all rely on the halo building for offprocessor information
	for (size_t i=0; i<_fields.size(); i++) {
		_fields[i]->updateHalo();
	}
	_fieldManager->updateFieldsHaloData();
}

void ParticlesT::buildHalo(bool use_physical_coords) {
	scalar_type halo_size = _parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("halo: size");
	std::cout << "Warning: ParticlesT::buildHalo() called without halo size. Default halo size of "
			<< halo_size << " used." << std::endl;
	this->buildHalo(halo_size, use_physical_coords);
}


// Neighborhood access, further neighborhood controls found as member functions of getNeighborhood()

const ParticlesT::neighbors_type * ParticlesT::getNeighborhoodConst() const {
	TEUCHOS_TEST_FOR_EXCEPT_MSG(_neighborhoodInfo.is_null(), "getNeighborhood() called before createNeighborhood() in ParticlesT.");
	return _neighborhoodInfo.getRawPtr();
}

ParticlesT::neighbors_type * ParticlesT::getNeighborhood() {
	TEUCHOS_TEST_FOR_EXCEPT_MSG(_neighborhoodInfo.is_null(), "getNeighborhood() called before createNeighborhood() in ParticlesT.");
	return _neighborhoodInfo.getRawPtr();
}

void ParticlesT::createNeighborhood() {
	TEUCHOS_TEST_FOR_EXCEPT_MSG(_parameters.is_null(),"Either ParticlesT was created without a ParameterList or setParameters was never called on ParticlesT.");

	std::string method =_parameters->get<Teuchos::ParameterList>("neighborhood").get<std::string>("method");
	transform(method.begin(), method.end(), method.begin(), ::tolower);


	if (method=="nanoflann") {
		local_index_type maxLeaf = _parameters->get<Teuchos::ParameterList>("neighborhood").get<int>("max leaf");
		_neighborhoodInfo = Teuchos::rcp_static_cast<ParticlesT::neighbors_type>(Teuchos::rcp(
				new ParticlesT::neighbors_type(this, this, true /*use physical coords*/, maxLeaf)));
	}
	else {
		TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Selected a search method that is not nanoflann.");
	}

}



// Flag functions

void ParticlesT::setFlag(const local_index_type idx, const local_index_type val) {
	host_view_local_index_type flagView = _flag->getLocalView<host_view_local_index_type>();
	flagView(idx,0) = val;
}

local_index_type ParticlesT::getFlag(const local_index_type idx) const {
	host_view_local_index_type flagView = _flag->getLocalView<host_view_local_index_type>();
	return flagView(idx,0);
}



// DOF Manager

void ParticlesT::createDOFManager() {
	// package all field relevant field, component, particle info up into an object and pass this into the DOFManager class

	// make sure to update all offsets, etc.. here
	_DOFManager = Teuchos::rcp(new Compadre::DOFManager(this, _parameters));
}

}
