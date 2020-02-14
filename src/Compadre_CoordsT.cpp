#include "Compadre_CoordsT.hpp"

#include "Compadre_XyzVector.hpp"
#include "Compadre_AnalyticFunctions.hpp"
#include <limits>
#include <Compadre_PointCloudSearch.hpp>

namespace Compadre {

CoordsT::CoordsT( const global_index_type nn, const Teuchos::RCP<const Teuchos::Comm<int> >& _comm, const int nDim) :
		_nDim(nDim), comm(_comm), _halo_size(-1)
{
	map = Teuchos::rcp(new map_type(nn, 0, comm, Tpetra::GloballyDistributed));
	const bool setToZero = false;
	pts = Teuchos::rcp(new mvec_type(map, nDim, setToZero));
	setLocalNFromMap();
	_units = "";

	// no effect on Eulerian simulations, but defaults to physical neighbors
	// for Lagrangian simulations
	_partitioned_using_physical = true;
}

local_index_type CoordsT::nDim() const {return _nDim;}
local_index_type CoordsT::nLocalMax(bool for_halo) const {return for_halo ? this->halo_pts->getLocalLength() : this->pts->getLocalLength();}
local_index_type CoordsT::nLocal(bool include_halo) const {
	return include_halo ? _nLocal + _nHalo : _nLocal;}
local_index_type CoordsT::nHalo() const {return _nHalo;}
void CoordsT::setLocalN(const local_index_type newN, bool for_halo) { if (for_halo) _nHalo = newN; else _nLocal = newN;}
void CoordsT::setLocalNFromMap(bool for_halo) { if (for_halo) _nHalo = halo_map->getNodeNumElements(); else _nLocal = map->getNodeNumElements();}
global_index_type CoordsT::nGlobalMax() const {return this->pts->getGlobalLength();}
global_index_type CoordsT::nGlobal() const {
// 	global_index_type sum = 0;
// 	Teuchos::Ptr<global_index_type> sumPtr(&sum);
// 	Teuchos::reduceAll<int, global_index_type>(*comm, Teuchos::REDUCE_SUM, _nLocal, sumPtr);
// 	return sum;
    return map->getGlobalNumElements();
};
global_index_type CoordsT::getMinGlobalIndex() const {return map->getMinGlobalIndex();}
global_index_type CoordsT::getMaxGlobalIndex() const {return map->getMaxGlobalIndex();}

void CoordsT::globalResize(const global_index_type nn) {
	// this should be called only by the ParticlesT class that owns this CoordsT

	// anything initialized off of a constructor of this class should be resized/reinitialized here
// 	_nMaxGlobal = nn;
	map = Teuchos::rcp(new map_type(nn, 0, comm, Tpetra::GloballyDistributed));
	const bool setToZero = false;
	pts = Teuchos::rcp(new mvec_type(map, nDim(), setToZero));
	if (_is_lagrangian) pts_physical = Teuchos::rcp(new mvec_type(map, nDim(), setToZero));
	setLocalNFromMap();
	updateAllHostViews();
}

void CoordsT::localResize(const global_index_type nn) {
	// this should be called only by the ParticlesT class that owns this CoordsT

	// anything initialized off of a constructor of this class should be resized/reinitialized here
	map = Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
									 nn,
									 0,
									 comm));
	const bool setToZero = false;
	pts = Teuchos::rcp(new mvec_type(map, nDim(), setToZero));
	if (_is_lagrangian) pts_physical = Teuchos::rcp(new mvec_type(map, nDim(), setToZero));
	setLocalNFromMap();
	updateAllHostViews();
// 	_nMaxGlobal = map->getGlobalNumElements();
}

void CoordsT::localResize(host_view_global_index_type gids) {
	// this should be called only by the ParticlesT class that owns this CoordsT

    auto gids_subview = Kokkos::subview(gids,Kokkos::ALL(),0);
	// anything initialized off of a constructor of this class should be resized/reinitialized here
	map = Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
									 gids_subview,
									 0,
									 comm));
	const bool setToZero = false;
	pts = Teuchos::rcp(new mvec_type(map, nDim(), setToZero));
	if (_is_lagrangian) pts_physical = Teuchos::rcp(new mvec_type(map, nDim(), setToZero));
	setLocalNFromMap();
	updateAllHostViews();
// 	_nMaxGlobal = map->getGlobalNumElements();
}


//
// FIXME: Required for Lagrangian particles
//
// 		void deepCopyFromSourceCoords(const CoordsT<scalar_type, local_index_type, global_index_type>& srcCoords) {
// 			Tpetra::deep_copy(*(this->pts), *srcCoords.pts);
// 		}

bool CoordsT::globalIndexIsLocal(const global_index_type idx) const {return map->isNodeGlobalElement(idx);}

std::pair<int, local_index_type> CoordsT::getLocalIdFromGlobalId(const global_index_type idx) const {
	std::pair<int, local_index_type> result;
	result.first = -1;
	result.second = -1;
	if (globalIndexIsLocal(idx)) { result.first = comm->getRank();
		result.second = map->getLocalElement(idx);
	}
	else {
		const std::vector<global_index_type> lookup(1, idx);
		std::vector<int> proc(1);
		std::vector<local_index_type> localIds(1);

		Teuchos::ArrayView<const global_index_type> lookupView(lookup);
		Teuchos::ArrayView<int> procView(proc);
		Teuchos::ArrayView<local_index_type> localIdView(localIds);
		if (map->getRemoteIndexList(lookupView, procView, localIdView) == Tpetra::AllIDsPresent) {
			result.first = procView.front();
			result.second = localIdView.front();
		}
	}
	return result;
}

void CoordsT::insertCoords(const std::vector<xyz_type>& new_pts_vector, const std::vector<xyz_type>& new_pts_physical_vector) {

	/*
	 * If a Lagrangian simulation, then the user must insert coordinates for both the Lagrangian and physical coordinates
	 * If one is known, the other can be calculated using some interpolation algorithm such as GMLS
	 *
	 * A similar function is in ParticlesT for dealing inserting field data
	 * into field when new coordinates are added.
	 */

	TEUCHOS_TEST_FOR_EXCEPT_MSG(_is_lagrangian && new_pts_physical_vector.size()==0, "Both sets of coordinates (Lagrangian and physical) must be made available for a Lagrangian simulation.");
	TEUCHOS_TEST_FOR_EXCEPT_MSG(_is_lagrangian && (new_pts_vector.size()!=new_pts_physical_vector.size()), "Both sets of coordinates (Lagrangian and physical) must have same cardinality in Lagrangian simulation.");

	const local_index_type num_added_coords = (local_index_type)new_pts_vector.size();

	Teuchos::RCP<map_type> new_map = Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
													_nLocal+num_added_coords,
													0,
													comm));

	const bool setToZero = false;
	Teuchos::RCP<mvec_type> new_pts = Teuchos::rcp(new mvec_type(new_map, _nDim, setToZero));
	Teuchos::RCP<mvec_type> new_pts_physical;
	if (_is_lagrangian) new_pts_physical = Teuchos::rcp(new mvec_type(new_map, _nDim, setToZero));

	host_view_type new_pts_vals = new_pts->getLocalView<host_view_type>();
	host_view_type old_pts_vals = pts->getLocalView<host_view_type>();

	host_view_type new_pts_physical_vals;
	host_view_type old_pts_physical_vals;
	if (_is_lagrangian) {
		new_pts_physical_vals = new_pts_physical->getLocalView<host_view_type>();
		old_pts_physical_vals = pts_physical->getLocalView<host_view_type>();
	}

	// copy old data
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,_nLocal), KOKKOS_LAMBDA(const int i) {
		for (local_index_type j=0; j<_nDim; j++) {
			new_pts_vals(i,j) = old_pts_vals(i,j);
		}
	});
	if (_is_lagrangian) {
		// copy old data
		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,_nLocal), KOKKOS_LAMBDA(const int i) {
			for (local_index_type j=0; j<_nDim; j++) {
				new_pts_physical_vals(i,j) = old_pts_physical_vals(i,j);
			}
		});
	}

	// enter new data
	// writes all data to either physical or lagrange coordinates, depending on parameters
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,num_added_coords), KOKKOS_LAMBDA(const int i) {
		new_pts_vals(_nLocal+i,0) = new_pts_vector[i].x;
		if (_nDim>1) new_pts_vals(_nLocal+i,1) = new_pts_vector[i].y;
		if (_nDim>2) new_pts_vals(_nLocal+i,2) = new_pts_vector[i].z;
	});

	if (_is_lagrangian) {
		// this is just the other set of points not selected just above this conditional statement
		// writes all data to either physical or lagrange coordinates, depending on parameters
		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,num_added_coords), KOKKOS_LAMBDA(const int i) {
			new_pts_physical_vals(_nLocal+i,0) = new_pts_physical_vector[i].x;
			if (_nDim>1) new_pts_physical_vals(_nLocal+i,1) = new_pts_physical_vector[i].y;
			if (_nDim>2) new_pts_physical_vals(_nLocal+i,2) = new_pts_physical_vector[i].z;
		});
	}
	// if Lagrangian simulation, and for some reason inserted into Lagrangian coordinates, then we need to get values for
	// the physical coordinates using GMLS coefficients from the Lagrangian coordinates and apply to the physical coordinates

	// if Lagrangian simulation, and inserted into physical coordinates, then we need to get values for
	// the Lagrangian coordinates using GMLS coefficients from the physical coordinates and apply to the Lagrangian coordinates


	map = new_map;

	pts = new_pts;
	if (_is_lagrangian) pts_physical = new_pts_physical;

	halo_map = Teuchos::null;
	halo_pts = Teuchos::null;
	halo_pts_physical = Teuchos::null;
	halo_importer = Teuchos::null;

	setLocalNFromMap();
	updateAllHostViews();
}

void CoordsT::insertCoords(const std::vector<xyz_type>& new_pts_vector) {
	this->insertCoords(new_pts_vector, new_pts_vector);
}

void CoordsT::removeCoords(const std::vector<local_index_type>& coord_ids) {
	const local_index_type num_remove_coords = (local_index_type)coord_ids.size();
	Kokkos::View <const local_index_type*,Kokkos::HostSpace,Kokkos::MemoryTraits<Kokkos::Unmanaged> >
		coord_ids_kok (&coord_ids[0] , num_remove_coords);

//	std::cout << "old size: " << _nLocal << " new size: " << _nLocal-num_remove_coords << std::endl;
	Teuchos::RCP<map_type> new_map = Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
													_nLocal-num_remove_coords,
													0,
													comm));

	const bool setToZero = false;
	Teuchos::RCP<mvec_type> new_pts = Teuchos::rcp(new mvec_type(new_map, _nDim, setToZero));
	host_view_type new_pts_vals = new_pts->getLocalView<host_view_type>();
	host_view_type old_pts_vals = pts->getLocalView<host_view_type>();
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,old_pts_vals.extent(0)), KOKKOS_LAMBDA(const int i) {
		bool deleted = false;
		local_index_type offset = num_remove_coords;
		for (local_index_type j=0; j<num_remove_coords; j++) {
			if (i == coord_ids_kok[j]) {
				deleted = true;
				break;
			}
			else if (i < coord_ids_kok[j]) {
				offset = j;
				break;
			}
		}
		if (!deleted) {
			for (local_index_type j=0; j<_nDim; j++) {
				new_pts_vals(i-offset,j) = old_pts_vals(i,j);
			}
		}
	});

	Teuchos::RCP<mvec_type> new_pts_physical;
	if (_is_lagrangian) {
		const bool setToZero = false;
		new_pts_physical = Teuchos::rcp(new mvec_type(new_map, _nDim, setToZero));
		host_view_type new_pts_physical_vals = new_pts_physical->getLocalView<host_view_type>();
		host_view_type old_pts_physical_vals = pts_physical->getLocalView<host_view_type>();
		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,old_pts_physical_vals.extent(0)), KOKKOS_LAMBDA(const int i) {
			bool deleted = false;
			local_index_type offset = num_remove_coords;
			for (local_index_type j=0; j<num_remove_coords; j++) {
				if (i == coord_ids_kok[j]) {
					deleted = true;
					break;
				}
				else if (i < coord_ids_kok[j]) {
					offset = j;
					break;
				}
			}
			if (!deleted) {
				for (local_index_type j=0; j<_nDim; j++) {
					new_pts_physical_vals(i-offset,j) = old_pts_physical_vals(i,j);
				}
			}
		});
	}

	map = new_map;
	pts = new_pts;
	if (_is_lagrangian) pts_physical = new_pts_physical;

	halo_map = Teuchos::null;
	halo_pts = Teuchos::null;
	halo_importer = Teuchos::null;
	halo_pts_physical = Teuchos::null;


//	z2problem = Teuchos::null;
//	z2adapter = Teuchos::null;

	setLocalNFromMap();
	updateAllHostViews();
}

void CoordsT::snapLagrangianCoordsToPhysicalCoords() {
	// snaps Lagrangian coordinates to physical coordinates
	if (_is_lagrangian) {
		host_view_type lagrangian_pts = pts->getLocalView<host_view_type>();
		host_view_type physical_pts = pts_physical->getLocalView<host_view_type>();

		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,lagrangian_pts.extent(0)), KOKKOS_LAMBDA(const int i) {
			for (local_index_type j=0; j<_nDim; j++) {
				lagrangian_pts(i,j) = physical_pts(i,j);
			}
		});

		this->updateHaloPts();
	}
}

void CoordsT::snapPhysicalCoordsToLagrangianCoords() {
	// snaps physical coordinates back to Lagrangian coordinates
	if (_is_lagrangian) {
		host_view_type lagrangian_pts = pts->getLocalView<host_view_type>();
		host_view_type physical_pts = pts_physical->getLocalView<host_view_type>();

		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,lagrangian_pts.extent(0)), KOKKOS_LAMBDA(const int i) {
			for (local_index_type j=0; j<_nDim; j++) {
				physical_pts(i,j) = lagrangian_pts(i,j);
			}
		});

		this->updateHaloPts();
	}
}

//void CoordsT::setCoords(const mvec_type* update_vector, bool use_physical_coords) {
//	/*
//	 * Updates Lagrangian coords with physical points if update_vector is not specified
//	 */
//	if (_is_lagrangian && use_physical_coords) {
//		host_view_type lagrangian_pts = pts->getLocalView<host_view_type>();
//		host_view_type physical_pts = pts_physical->getLocalView<host_view_type>();
//
//		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,lagrangian_pts.extent(0)), KOKKOS_LAMBDA(const int i) {
//			for (local_index_type j=0; j<_nDim; j++) {
//				lagrangian_pts(i,j) = physical_pts(i,j);
//			}
//		});
//
//
//	} else {
//
//	}
//	this->updateHaloPts();
//}

void CoordsT::replaceLocalCoords(const local_index_type idx, const scalar_type x, const scalar_type y,
						const scalar_type z, bool use_physical_coords) {
	if (_is_lagrangian && use_physical_coords) {
		pts_physical_view(idx, 0) = x;
		if (_nDim>1) pts_physical_view(idx, 1) = y;
		if (_nDim>2) pts_physical_view(idx, 2) = z;
	} else {
		pts_view(idx, 0) = x;
		if (_nDim>1) pts_view(idx, 1) = y;
		if (_nDim>2) pts_view(idx, 2) = z;
	}
}

void CoordsT::replaceLocalCoords(const local_index_type idx, const CoordsT::xyz_type xyz, bool use_physical_coords) {
	if (_is_lagrangian && use_physical_coords) {
		pts_physical_view(idx, 0) = xyz.x;
		if (_nDim>1) pts_physical_view(idx, 1) = xyz.y;
		if (_nDim>2) pts_physical_view(idx, 2) = xyz.z;
	} else {
		pts_view(idx, 0) = xyz.x;
		if (_nDim>1) pts_view(idx, 1) = xyz.y;
		if (_nDim>2) pts_view(idx, 2) = xyz.z;
	}
}

void CoordsT::replaceGlobalCoords(const global_index_type idx, const scalar_type x, const scalar_type y,
						 const scalar_type z, bool use_physical_coords) {
	mvec_type* pts_vec = (_is_lagrangian && use_physical_coords) ? pts_physical.getRawPtr() : pts.getRawPtr();
	pts_vec->replaceGlobalValue(idx, 0, x);
	if (_nDim>1) pts_vec->replaceGlobalValue(idx, 1, y);
	if (_nDim>2) pts_vec->replaceGlobalValue(idx, 2, z);
}

void CoordsT::replaceGlobalCoords(const global_index_type idx, const CoordsT::xyz_type xyz, bool use_physical_coords) {
	mvec_type* pts_vec = (_is_lagrangian && use_physical_coords) ? pts_physical.getRawPtr() : pts.getRawPtr();
	pts_vec->replaceGlobalValue(idx, 0, xyz.x);
	if (_nDim>1) pts_vec->replaceGlobalValue(idx, 1, xyz.y);
	if (_nDim>2) pts_vec->replaceGlobalValue(idx, 2, xyz.z);
}

CoordsT::xyz_type CoordsT::getLocalCoords(const local_index_type idx1, bool use_halo, bool use_physical_coords) const {
    if (_nDim == 3) {
	    if (_is_lagrangian && use_physical_coords) {
	    	const local_index_type nlocal_minus_halo = pts_physical_view.extent(0);
	    	if (idx1 < nlocal_minus_halo) {
	    		return CoordsT::xyz_type(pts_physical_view(idx1, 0), pts_physical_view(idx1, 1), pts_physical_view(idx1, 2));
	    	} else {
	    		return CoordsT::xyz_type(halo_pts_physical_view(idx1-nlocal_minus_halo, 0), halo_pts_physical_view(idx1-nlocal_minus_halo, 1), halo_pts_physical_view(idx1-nlocal_minus_halo, 2));
	    	}
	    } else {
	    	const local_index_type nlocal_minus_halo = pts_view.extent(0);
	    	if (idx1 < nlocal_minus_halo) {
	    		return CoordsT::xyz_type(pts_view(idx1, 0), pts_view(idx1, 1), pts_view(idx1, 2));
	    	} else {
	    		return CoordsT::xyz_type(halo_pts_view(idx1-nlocal_minus_halo, 0), halo_pts_view(idx1-nlocal_minus_halo, 1), halo_pts_view(idx1-nlocal_minus_halo, 2));
	    	}
	    }
    } else if (_nDim == 2) {
	    if (_is_lagrangian && use_physical_coords) {
	    	const local_index_type nlocal_minus_halo = pts_physical_view.extent(0);
	    	if (idx1 < nlocal_minus_halo) {
	    		return CoordsT::xyz_type(pts_physical_view(idx1, 0), pts_physical_view(idx1, 1), 0);
	    	} else {
	    		return CoordsT::xyz_type(halo_pts_physical_view(idx1-nlocal_minus_halo, 0), halo_pts_physical_view(idx1-nlocal_minus_halo, 1), 0);
	    	}
	    } else {
	    	const local_index_type nlocal_minus_halo = pts_view.extent(0);
	    	if (idx1 < nlocal_minus_halo) {
	    		return CoordsT::xyz_type(pts_view(idx1, 0), pts_view(idx1, 1), 0);
	    	} else {
	    		return CoordsT::xyz_type(halo_pts_view(idx1-nlocal_minus_halo, 0), halo_pts_view(idx1-nlocal_minus_halo, 1), 0);
	    	}
	    }
    } else {
	    if (_is_lagrangian && use_physical_coords) {
	    	const local_index_type nlocal_minus_halo = pts_physical_view.extent(0);
	    	if (idx1 < nlocal_minus_halo) {
	    		return CoordsT::xyz_type(pts_physical_view(idx1, 0), 0, 0);
	    	} else {
	    		return CoordsT::xyz_type(halo_pts_physical_view(idx1-nlocal_minus_halo, 0), 0, 0);
	    	}
	    } else {
	    	const local_index_type nlocal_minus_halo = pts_view.extent(0);
	    	if (idx1 < nlocal_minus_halo) {
	    		return CoordsT::xyz_type(pts_view(idx1, 0), 0, 0);
	    	} else {
	    		return CoordsT::xyz_type(halo_pts_view(idx1-nlocal_minus_halo, 0), 0, 0);
	    	}
	    }
    }
}

const Teuchos::RCP<mvec_type>& CoordsT::getPts(const bool halo, bool use_physical_coords) const {
	if (_is_lagrangian && use_physical_coords) {
		return halo ? halo_pts_physical : pts_physical;
	} else {
		return halo ? halo_pts : pts;
	}
}

CoordsT::xyz_type CoordsT::getGlobalCoords(const global_index_type idx, bool use_physical_coords) const {
	const std::pair<int, local_index_type> remoteIds = getLocalIdFromGlobalId(idx);
	std::vector<scalar_type> cVec(_nDim);
	Teuchos::ArrayView<scalar_type> cView(cVec);
	if (comm->getRank() == remoteIds.first) {
		const CoordsT::xyz_type pt = getLocalCoords(remoteIds.second, use_physical_coords);
		pt.convertToStdVector(cVec);
	}
	Teuchos::broadcast(*comm, remoteIds.first, cView);
	return CoordsT::xyz_type(cVec);
}

CoordsT::xyz_type CoordsT::getCoordsPointToPoint(const global_index_type idx, const int destRank, bool use_physical_coords) const {
	CoordsT::xyz_type return_vec;
	const std::pair<int, local_index_type> remoteIds = getLocalIdFromGlobalId(idx);
	if (globalIndexIsLocal(idx)) {
		return_vec = getLocalCoords(remoteIds.second, false /*halo*/, use_physical_coords);
	}
	else {
		scalar_type arry[_nDim];
		if (comm->getRank() == remoteIds.first) {
			const CoordsT::xyz_type pt = getLocalCoords(remoteIds.second, use_physical_coords);
			pt.convertToArray(arry);
			Teuchos::send<int, scalar_type>(*comm, _nDim, arry, destRank);
		}
		else if (comm->getRank() == destRank) {
			Teuchos::receive<int, scalar_type>(*comm, remoteIds.first, _nDim, arry);
			return_vec = CoordsT::xyz_type(arry);
		}
	}
	return return_vec;
}

Teuchos::RCP<const Teuchos::Comm<local_index_type> > CoordsT::getComm() const { return comm; }
Teuchos::RCP<const map_type> CoordsT::getMapConst(bool for_halo) const { return for_halo ? Teuchos::rcp_const_cast<const map_type>(this->halo_map) : Teuchos::rcp_const_cast<const map_type>(this->map); }
Teuchos::RCP<const importer_type> CoordsT::getHaloImporterConst() const { return Teuchos::rcp_const_cast<const importer_type>(halo_importer); }

// TODO: Not currently thread-safe
CoordsT::xyz_type CoordsT::localCrossProduct(const local_index_type idx1, const CoordsT::xyz_type& queryPt, bool use_physical_coords) const {
    TEUCHOS_ASSERT(_nDim==3);
	mvec_type* pts_vec = (_is_lagrangian && use_physical_coords) ? pts_physical.getRawPtr() : pts.getRawPtr();
	host_view_type ptsView = pts_vec->getLocalView<host_view_type>();
	return CoordsT::xyz_type( ptsView(idx1, 1) * queryPt.z -
					   ptsView(idx1, 2) * queryPt.y,
					   ptsView(idx1, 2) * queryPt.x -
					   ptsView(idx1, 0) * queryPt.z,
					   ptsView(idx1, 0) * queryPt.y -
					   ptsView(idx1, 1) * queryPt.x );
}

// TODO: Not currently thread-safe
scalar_type CoordsT::localDotProduct( const local_index_type idx1, const CoordsT::xyz_type& queryPt, bool use_physical_coords) const {
    TEUCHOS_ASSERT(_nDim==3);
	mvec_type* pts_vec = (_is_lagrangian && use_physical_coords) ? pts_physical.getRawPtr() : pts.getRawPtr();
	host_view_type ptsView = pts_vec->getLocalView<host_view_type>();
	return ptsView(idx1, 0) * queryPt.x +
		   ptsView(idx1, 1) * queryPt.y +
		   ptsView(idx1, 2) * queryPt.z;
}

void CoordsT::syncMemory() {
	pts->sync<device_view_type>();
	pts->sync<host_view_type>();
	if (_is_lagrangian) {
		pts_physical->sync<device_view_type>();
		pts_physical->sync<host_view_type>();
	}
}

bool CoordsT::hostNeedsUpdate() const {return pts->need_sync<host_execution_space>(); }
bool CoordsT::deviceNeedsUpdate() const {return pts->need_sync<device_view_type>();}

void CoordsT::zoltan2Init(bool use_physical_coords) {
	Teuchos::ParameterList params("zoltan2 params");
	params.set("algorithm", "multijagged");
	params.set("mj_keep_part_boxes", true);
	params.set("rectilinear", true);
	params.set("num_global_parts", comm->getSize());
	//params.set("mj_enable_rcb", true);

	Teuchos::RCP<const mvec_type> input = Teuchos::rcp_const_cast<const mvec_type>((_is_lagrangian && use_physical_coords) ? this->pts_physical : this->pts);
	//Tpetra::MatrixMarket::Writer<mvec_type>::writeDenseFile("mj_input.txt", *(this->pts.getRawPtr()), "matrixname", "description");

	z2adapter_scalar = Teuchos::rcp(new z2_scalar_adapter_type(input));
	z2problem = Teuchos::rcp(new z2_problem_type(z2adapter_scalar.getRawPtr(), &params, comm));
	_partitioned_using_physical = use_physical_coords;
}

void CoordsT::zoltan2Partition() { //const Teuchos::RCP<const Teuchos::Comm<int> > comm) {
	// NOTE: This is known to hang for 7 or more cores on a single node
	z2problem->solve();

	Teuchos::RCP<const mvec_type> input_pts = Teuchos::rcp_const_cast<const mvec_type>(this->pts);
	Teuchos::RCP<const mvec_type> input_pts_physical;
	if (_is_lagrangian) input_pts_physical = Teuchos::rcp_const_cast<const mvec_type>(this->pts_physical);
	z2adapter_scalar->applyPartitioningSolution(*input_pts, this->pts, z2problem->getSolution());
	if (_is_lagrangian) z2adapter_scalar->applyPartitioningSolution(*input_pts_physical, this->pts_physical, z2problem->getSolution());

	map = this->pts->getMap();
	this->setLocalNFromMap();
	updateAllHostViews();

	std::vector<z2_box_type> boxes = z2problem->getSolution().getPartBoxesView();
	for (int i = 0; i < comm->getSize(); ++i){
		comm->barrier();
		if (comm->getRank() == i) {
			std::cout << "proc " << i << " : " << std::endl << "\t";
			boxes[i].print();
		}
	}
}

void CoordsT::applyZoltan2Partition(Teuchos::RCP<mvec_type>& vec) const {
	Teuchos::RCP<const mvec_type> input = Teuchos::rcp_const_cast<const mvec_type>(vec);
	z2adapter_scalar->applyPartitioningSolution(*input, vec, z2problem->getSolution());
}
void CoordsT::applyZoltan2Partition(Teuchos::RCP<mvec_local_index_type>& vec) const {
	auto z2adapter_local_index = Teuchos::rcp(new z2_local_index_adapter_type(vec));
	Teuchos::RCP<const mvec_local_index_type> input = Teuchos::rcp_const_cast<const mvec_local_index_type>(vec);
	z2adapter_local_index->applyPartitioningSolution(*input, vec, z2problem->getSolution());
}

bool CoordsT::verifyCoordsOnProcessor(const std::vector<xyz_type>& new_pts_vector, bool use_physical_coords, bool throw_exception) const {
	// check if use_physical_coords matches how the partitioning of the coordinates was performed
	// if it doesn't, then it isn't logical to compare the coordinates to the processor partitions
	TEUCHOS_TEST_FOR_EXCEPT_MSG(_is_lagrangian && use_physical_coords && (_partitioned_using_physical != use_physical_coords), "Coordinates being verified are in the physical frame, but processor partitioning was performed in Lagrangian frame.");
	TEUCHOS_TEST_FOR_EXCEPT_MSG(_is_lagrangian && (!use_physical_coords) && (_partitioned_using_physical != use_physical_coords), "Coordinates being verified are in the Lagrangian frame, but processor partitioning was performed in physical frame.");

	const local_index_type new_pts_vector_size = new_pts_vector.size();
	const z2_box_type box = z2problem->getSolution().getPartBoxesView()[comm->getRank()];
    int number_off_processor = 0; // should remain zero
	Kokkos::parallel_reduce(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,new_pts_vector_size), KOKKOS_LAMBDA(const int i, int& t_number_off_processor) {
		scalar_type coordinates[_nDim];
		new_pts_vector[i].convertToArray(coordinates);
        t_number_off_processor = std::max(t_number_off_processor, static_cast<int>(!box.pointInBox(_nDim, coordinates)));
		TEUCHOS_TEST_FOR_EXCEPT_MSG((throw_exception && !box.pointInBox(_nDim, coordinates)),"On processor# " + std::to_string(comm->getRank())
		+ ", Point inserted (" + std::to_string(coordinates[0]) + "," + std::to_string(coordinates[1]) +  ","
		+ std::to_string(coordinates[2]) + ") not within bounding box of this processor.");
	}, Kokkos::Max<int>(number_off_processor));
    return (number_off_processor == 0) ? true : false;
}

// boxes are related to either physical or material coordinates
// user should verify that this makes sense with the coordinates they are comparing against these boxes
std::vector<scalar_type> CoordsT::boundingBoxMinOnProcessor(const local_index_type processor_num) const {
	const local_index_type proc_num_to_query = (processor_num > -1) ? processor_num : comm->getRank();
	const z2_box_type box = z2problem->getSolution().getPartBoxesView()[proc_num_to_query];
	const scalar_type* mins = box.getlmins();
	std::vector<scalar_type> min_bound(mins, mins+_nDim);
	return min_bound;
}

std::vector<scalar_type> CoordsT::boundingBoxMaxOnProcessor(const local_index_type processor_num) const {
	const local_index_type proc_num_to_query = (processor_num > -1) ? processor_num : comm->getRank();
	const z2_box_type box = z2problem->getSolution().getPartBoxesView()[proc_num_to_query];
	const scalar_type* maxs = box.getlmaxs();
	std::vector<scalar_type> max_bound(maxs, maxs+_nDim);
	return max_bound;
}

void CoordsT::writeToMatlab(std::ostream& fs, const std::string coordsName, const int procRank, bool use_physical_coords) const {
	mvec_type* pt_vec = (_is_lagrangian && use_physical_coords) ? pts_physical.getRawPtr() : pts.getRawPtr();
	host_view_type ptsHostView = pt_vec->getLocalView<host_view_type>();
	fs << coordsName << "Xyz" << procRank << " = [";
	for (local_index_type i = 0; i < _nLocal - 1; ++i){
		fs << ptsHostView(i,0) << ", " << ptsHostView(i,1) << ", " << ptsHostView(i,2) << "; ..." << std::endl;
	}
	fs << ptsHostView(_nLocal-1, 0) << ", " << ptsHostView(_nLocal-1,1) << ", " << ptsHostView(_nLocal-1,2) << "];" << std::endl;
}

void CoordsT::writeHaloToMatlab(std::ostream& fs, const std::string coordsName, const int procRank, bool use_physical_coords) const {
	mvec_type* halo_pt_vec = (_is_lagrangian && use_physical_coords) ? halo_pts_physical.getRawPtr() : halo_pts.getRawPtr();
	host_view_type haloPtsHostView = halo_pt_vec->getLocalView<host_view_type>();
	if (halo_pt_vec->getLocalLength() > 0) {
		fs << coordsName << "Xyz" << procRank << " = [";
		for (size_t i = 0; i < halo_pt_vec->getLocalLength() - 1; ++i){
			fs << haloPtsHostView(i,0) << ", " << haloPtsHostView(i,1) << ", " << haloPtsHostView(i,2) << "; ..." << std::endl;
		}
		fs << haloPtsHostView(halo_pt_vec->getLocalLength()-1, 0) << ", " << haloPtsHostView(halo_pt_vec->getLocalLength()-1,1) << ", " << haloPtsHostView(halo_pt_vec->getLocalLength()-1,2) << "];" << std::endl;
	}
}

void CoordsT::print(std::ostream& os, bool use_physical_coords) const {
	auto out = Teuchos::getFancyOStream(Teuchos::rcpFromRef(os));
	mvec_type* pt_vec = (_is_lagrangian && use_physical_coords) ? pts_physical.getRawPtr() : pts.getRawPtr();
	mvec_type* halo_pt_vec = (_is_lagrangian && use_physical_coords) ? halo_pts_physical.getRawPtr() : halo_pts.getRawPtr();
	if (!pts.is_null()) {
		std::cout << "Locally owned points on processor " << comm->getRank() << std::endl;
		pt_vec->describe(*out, Teuchos::VERB_EXTREME);
	}
	if (!halo_pts.is_null()) {
		if (halo_map->getGlobalNumElements()>0) {
			std::cout << "Halo points on processor " << comm->getRank() << std::endl;
			halo_pt_vec->describe(*out, Teuchos::VERB_EXTREME);
		}
	}
}

void CoordsT::buildHalo(scalar_type h_size, bool use_physical_coords) {
	TEUCHOS_TEST_FOR_EXCEPT_MSG(_is_lagrangian && use_physical_coords && (_partitioned_using_physical != use_physical_coords), "Halo is being built in the physical frame, but processor partitioning was performed in material frame.");
	TEUCHOS_TEST_FOR_EXCEPT_MSG(_is_lagrangian && (!use_physical_coords) && (_partitioned_using_physical != use_physical_coords), "Halo is being built in the material frame, but processor partitioning was performed in physical frame.");
	TEUCHOS_TEST_FOR_EXCEPT_MSG(z2problem.is_null(), "Zoltan2 object does not exist. Needs to be called.");

	typedef Kokkos::View<const global_index_type*> const_gid_view_type;
	typedef Kokkos::View<global_index_type*> gid_view_type;
	typedef typename gid_view_type::HostMirror host_gid_view_type;
	typedef Kokkos::View<local_index_type> count_type;

	Teuchos::RCP<Teuchos::Time> haloGraphTime = Teuchos::TimeMonitor::getNewCounter ("Halo - Get Graph");
	haloGraphTime->start();
	// build trivial map in the case that there is no communication
	if (comm->getSize() == 1) {
		halo_map = Teuchos::rcp(new map_type(0, 0, comm, Tpetra::GloballyDistributed));
		const bool setToZero = false;
		halo_pts = Teuchos::rcp(new mvec_type(halo_map, _nDim, setToZero));
		if (_is_lagrangian) halo_pts_physical = Teuchos::rcp(new mvec_type(halo_map, _nDim, setToZero));
		halo_importer = Teuchos::rcp(new importer_type(map, halo_map));
		setLocalN(this->nLocalMax(true /*for halo*/), true /*for halo*/);
		haloGraphTime->stop();
		return;
	}

	// get bounding boxes from Zoltan2
	std::vector<z2_box_type> boxes = z2problem->getSolution().getPartBoxesView();

    auto source_sites = host_view_type("source sites", comm->getSize(), _nDim);
    auto target_sites = host_view_type("target sites", 1, _nDim);
    auto neighbor_lists = host_view_local_index_type("neighbor lists", 
            1, 1+(_nDim+2)*(_nDim+2)*(_nDim+2)-1 /* 2 layer max halo in Nd */);
    auto epsilons = host_vector_scalar_type("epsilons", 1);
    
    std::set<local_index_type> proc_neighbors;
    auto comm_size = comm->getSize();
    auto comm_rank = comm->getRank();
    if (_nDim==3) {
        for (local_index_type i=0; i<2; ++i) {
            for (local_index_type j=0; j<2; ++j) {
                for (local_index_type k=0; k<2; ++k) {
                    for (local_index_type a=0; a<2; ++a) {
                        for (local_index_type b=0; b<2; ++b) {
                            for (local_index_type c=0; c<2; ++c) {
                                for (local_index_type l=0; l<comm_size; ++l) {
                                    scalar_type* mins = boxes[l].getlmins();
                                    scalar_type* maxs = boxes[l].getlmaxs();
                                    source_sites(l,0) = (i==0) ? mins[0] : maxs[0];
                                    source_sites(l,1) = (j==0) ? mins[1] : maxs[1];
                                    source_sites(l,2) = (k==0) ? mins[2] : maxs[2];
                                }
                                scalar_type* mins = boxes[comm_rank].getlmins();
                                scalar_type* maxs = boxes[comm_rank].getlmaxs();
                                target_sites(0,0) = (a==0) ? mins[0] : maxs[0];
                                target_sites(0,1) = (b==0) ? mins[1] : maxs[1];
                                target_sites(0,2) = (c==0) ? mins[2] : maxs[2];
                                // do 3d search here
                                auto point_cloud_search(CreatePointCloudSearch(source_sites, _nDim));
                                point_cloud_search.generateNeighborListsFromRadiusSearch(false /*not dry run*/, target_sites, 
                                        neighbor_lists, epsilons, h_size);
                                for (local_index_type m=1; m<=neighbor_lists(0,0); ++m) {
                                    if (neighbor_lists(0,m)!=comm_rank)
                                        proc_neighbors.insert(neighbor_lists(0,m));   
                                } 
                            }
                        }
                    }
                }
            }
        }
    } else if (_nDim==2) {
        for (local_index_type i=0; i<2; ++i) {
            for (local_index_type j=0; j<2; ++j) {
                for (local_index_type a=0; a<2; ++a) {
                    for (local_index_type b=0; b<2; ++b) {
                        for (local_index_type l=0; l<comm_size; ++l) {
                            scalar_type* mins = boxes[l].getlmins();
                            scalar_type* maxs = boxes[l].getlmaxs();
                            source_sites(l,0) = (i==0) ? mins[0] : maxs[0];
                            source_sites(l,1) = (j==0) ? mins[1] : maxs[1];
                        }
                        scalar_type* mins = boxes[comm_rank].getlmins();
                        scalar_type* maxs = boxes[comm_rank].getlmaxs();
                        target_sites(0,0) = (a==0) ? mins[0] : maxs[0];
                        target_sites(0,1) = (b==0) ? mins[1] : maxs[1];
                        // do 2d search here
                        auto point_cloud_search(CreatePointCloudSearch(source_sites, _nDim));
                        point_cloud_search.generateNeighborListsFromRadiusSearch(false /*not dry run*/, target_sites, 
                                neighbor_lists, epsilons, h_size);
                        for (local_index_type m=1; m<=neighbor_lists(0,0); ++m) {
                            if (neighbor_lists(0,m)!=comm_rank)
                                proc_neighbors.insert(neighbor_lists(0,m));   
                        } 
                    }
                }
            }
        }
    } else if (_nDim==1) {
        for (local_index_type i=0; i<2; ++i) {
            for (local_index_type a=0; a<2; ++a) {
                for (local_index_type l=0; l<comm_size; ++l) {
                    scalar_type* mins = boxes[l].getlmins();
                    scalar_type* maxs = boxes[l].getlmaxs();
                    source_sites(l,0) = (i==0) ? mins[0] : maxs[0];
                }
                scalar_type* mins = boxes[comm_rank].getlmins();
                scalar_type* maxs = boxes[comm_rank].getlmaxs();
                target_sites(0,0) = (a==0) ? mins[0] : maxs[0];
                // do 1d search here
                auto point_cloud_search(CreatePointCloudSearch(source_sites, _nDim));
                point_cloud_search.generateNeighborListsFromRadiusSearch(false /*not dry run*/, target_sites, 
                        neighbor_lists, epsilons, h_size);
                for (local_index_type m=1; m<=neighbor_lists(0,0); ++m) {
                    if (neighbor_lists(0,m)!=comm_rank)
                        proc_neighbors.insert(neighbor_lists(0,m));   
                } 
            }
        }
    } else {
        TEUCHOS_ASSERT(false);
    }

	auto num_sending_processors = proc_neighbors.size();

    //// DIAGNOSTIC:: with 4 processors should see the 3 others
    //if (comm->getRank()==2) {
    //    printf("send to %d procs\n", num_sending_processors);
	//    for (auto proc : proc_neighbors) {
    //        printf("proc %d\n", proc);
    //    }
    //}
	haloGraphTime->stop();

	Teuchos::RCP<Teuchos::Time> haloSearchTime = Teuchos::TimeMonitor::getNewCounter ("Halo - Search");
	// decide which coordinates need to be sent to another processor's halo using bounding boxes
	std::vector<Teuchos::RCP<host_gid_view_type> > coords_to_send(num_sending_processors);
	const_gid_view_type gids = map->getMyGlobalIndices();
	// only loop over processors we border
	haloSearchTime->start();

	host_view_type ptsView = (_is_lagrangian && use_physical_coords)
			? pts_physical->getLocalView<host_view_type>() : pts->getLocalView<host_view_type>();

    auto p_count = 0;
	for (auto peer_processor_num : proc_neighbors) {

		// enlarge the bounding box by h
		z2_box_type enlarged_box(boxes[peer_processor_num]);
		scalar_type* mins = boxes[peer_processor_num].getlmins();
		scalar_type* maxs = boxes[peer_processor_num].getlmaxs();
		for (local_index_type j=0; j<_nDim; j++) {
			enlarged_box.updateMinMax(mins[j]-h_size,0,j);
			enlarged_box.updateMinMax(maxs[j]+h_size,1,j);
		}
		local_index_type ptsLocalLength = (_is_lagrangian && use_physical_coords)
				? pts_physical->getLocalLength() : pts->getLocalLength();
		{
			Teuchos::RCP<Teuchos::Time> haloCoordTime =
				Teuchos::TimeMonitor::getNewCounter ("Halo - Coords");
			haloCoordTime->start();
			count_type count_gids_found("GID_Count");
			gid_view_type local_to_global_id_found("gids", ptsLocalLength);

			struct SearchBoxFunctor {
				z2_box_type box;
				gid_view_type gids_found;
				host_view_type pts_view;
				const_gid_view_type gids;
				count_type count;
				global_index_type max_gid;
                local_index_type ndim;

				SearchBoxFunctor(z2_box_type box_, gid_view_type gids_found_,
						host_view_type pts_view_, const_gid_view_type gids_,
						count_type count_)
						: box(box_),gids_found(gids_found_),
						  pts_view(pts_view_),gids(gids_),count(count_){
					max_gid = std::numeric_limits<global_index_type>::max();
                    ndim = static_cast<int>(pts_view_.extent(1));
				}

				void operator()(const int i) const {
					scalar_type coordinates[3] = {0, 0, 0};
                    coordinates[0] = pts_view(i,0);
                    coordinates[1] = (ndim>1) ? pts_view(i,1) : 0;
                    coordinates[2] = (ndim>2) ? pts_view(i,2) : 0;
					if (box.pointInBox(ndim, coordinates)) {
						//Kokkos::atomic_add(&count(), 1);
						Kokkos::atomic_fetch_add(&count(), 1);
						gids_found(i) = gids(i);
					} else {
						gids_found(i) = max_gid;
					}
				}
			};

			Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,ptsLocalLength),
				SearchBoxFunctor(enlarged_box, local_to_global_id_found, ptsView,
						 gids, count_gids_found));
			haloCoordTime->stop();

			Teuchos::RCP<Teuchos::Time> haloCoordTime2 = Teuchos::TimeMonitor::getNewCounter ("Halo - Coords2");
			haloCoordTime2->start();
			struct VectorMergeFunctor {
				gid_view_type gids_to_send;
				gid_view_type gids_found;
				count_type count;
				global_index_type max_gid;

				VectorMergeFunctor(gid_view_type gids_to_send_,
						gid_view_type gids_found_)
						: gids_to_send(gids_to_send_), gids_found(gids_found_),
						  count(count_type("")){
					max_gid = std::numeric_limits<global_index_type>::max();
				}

				void operator()(const int i) const {
					if (gids_found(i) != max_gid) {
						const int idx = Kokkos::atomic_fetch_add(&count(), 1);
						gids_to_send(idx) = gids_found(i);
					}
				}
			};
			gid_view_type gids_to_send("gids to send", count_gids_found());
			Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,ptsLocalLength), VectorMergeFunctor(gids_to_send, local_to_global_id_found));

			Teuchos::RCP<host_gid_view_type> h_gids_to_send = Teuchos::rcp(new host_gid_view_type);
			*h_gids_to_send = Kokkos::create_mirror_view(gids_to_send);
			Kokkos::deep_copy(*h_gids_to_send, gids_to_send);


			coords_to_send[p_count] = h_gids_to_send;
            p_count++;
			haloCoordTime2->stop();
		}
	}
	haloSearchTime->stop();

	std::vector<local_index_type> processor_i_destined(num_sending_processors);
	for (size_t i=0; i<coords_to_send.size(); i++) {
		processor_i_destined[i] = coords_to_send[i]->size();
	}

	Teuchos::RCP<Teuchos::Time> haloCommunicateTime = Teuchos::TimeMonitor::getNewCounter ("Halo - Communicate");
	haloCommunicateTime->start();

	// collect the global indices of coordinates to be sent to each processor on that processor
	std::vector<global_index_type> indices_i_need;
	{
		// put out a receive for all processors of a single integer
		// go through my neighbor list
		// non-blocking receive
		Teuchos::Array<Teuchos::RCP<Teuchos::CommRequest<local_index_type> > > requests(2*num_sending_processors);

		Teuchos::ArrayRCP<local_index_type> recv_counts(num_sending_processors);
        auto p_count = 0;
	    for (auto comm_adj_p : proc_neighbors) {
			Teuchos::ArrayRCP<local_index_type> single_recv_value(&recv_counts[p_count], 0, 1, false);
			requests[p_count] = Teuchos::ireceive<local_index_type,local_index_type>(*comm, single_recv_value, comm_adj_p);
            p_count++;
		}

		// put out a send for all processors to send to of a single integer
		// go through my neighbor list
		// non-blocking sends
		Teuchos::ArrayRCP<local_index_type> send_counts(&processor_i_destined[0], 0, processor_i_destined.size(), false);
        p_count = 0;
	    for (auto comm_adj_p : proc_neighbors) {
			Teuchos::ArrayRCP<local_index_type> single_send_value(&send_counts[p_count], 0, 1, false);
			requests[num_sending_processors+(p_count)] = Teuchos::isend<local_index_type,local_index_type>(*comm, single_send_value, comm_adj_p);
            p_count++;
		}
		Teuchos::waitAll<local_index_type>(*comm, requests);

		local_index_type sum=0;
		std::vector<local_index_type> sending_processor_offsets(num_sending_processors);
		for (local_index_type i=0; i<num_sending_processors; i++) {
			sending_processor_offsets[i] += sum;
			sum += recv_counts[i];
		}
		indices_i_need.resize(sum);


		// put out a receive for all processors of a N integers
		// go through my neighbor list
		Teuchos::ArrayRCP<global_index_type> indices_received(&indices_i_need[0], 0, sum, false);
        p_count = 0;
	    for (auto comm_adj_p : proc_neighbors) {
			Teuchos::ArrayRCP<global_index_type> single_indices_recv;
			if (recv_counts[p_count] > 0) {
				single_indices_recv = Teuchos::ArrayRCP<global_index_type>(&indices_received[sending_processor_offsets[p_count]], 0, recv_counts[p_count], false);
			} else {
				single_indices_recv = Teuchos::ArrayRCP<global_index_type>(NULL, 0, 0, false);
			}
			requests[p_count] = Teuchos::ireceive<local_index_type,global_index_type>(*comm, single_indices_recv, comm_adj_p);
            p_count++;
		}

		// put out a send for all processors of N integers
		// go through my neighbor list
        p_count = 0;
	    for (auto comm_adj_p : proc_neighbors) {
			Teuchos::ArrayRCP<global_index_type> indices_to_send;
			if (coords_to_send[p_count]->extent(0) > 0) {
				indices_to_send = Teuchos::ArrayRCP<global_index_type>(&(*coords_to_send[p_count])(0), 0, coords_to_send[p_count]->extent(0), false);
			} else {
				indices_to_send = Teuchos::ArrayRCP<global_index_type>(NULL, 0, 0, false);
			}
			requests[num_sending_processors+(p_count)] = Teuchos::isend<local_index_type,global_index_type>(*comm, indices_to_send, comm_adj_p);
            p_count++;
		}
		Teuchos::waitAll<local_index_type>(*comm, requests);
	}
	haloCommunicateTime->stop();

	Teuchos::RCP<Teuchos::Time> haloMapTime = Teuchos::TimeMonitor::getNewCounter ("Halo - Map");
	haloMapTime->start();

	halo_map = Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
						 &indices_i_need[0],
						 indices_i_need.size(),
						 0,
						 comm));

	const bool setToZero = false;
	halo_pts = Teuchos::rcp(new mvec_type(halo_map, _nDim, setToZero ));
	if (_is_lagrangian) halo_pts_physical = Teuchos::rcp(new mvec_type(halo_map, _nDim, setToZero ));
	halo_importer = Teuchos::rcp(new importer_type(map, halo_map));
	haloMapTime->stop();

	this->setLocalN(this->nLocalMax(true /*for halo*/), true /*for halo*/);
	this->updateHaloPts();
	_halo_size = h_size;
}

scalar_type CoordsT::getHaloSize() const {
	// returns size of last computed halo
	return _halo_size;
}

// updates halo data for material and physical coordinates for Lagrangian frame
// only updates material coordinates (only set of coordinates) in Eulerian
void CoordsT::updateHaloPts() {
	Teuchos::RCP<Teuchos::Time> haloUpdateTime = Teuchos::TimeMonitor::getNewCounter ("Halo - Update");
	haloUpdateTime->start();

	if (!halo_pts.is_null()) {
		halo_pts->doImport(*pts, *halo_importer, Tpetra::CombineMode::INSERT);
		if (_is_lagrangian) halo_pts_physical->doImport(*pts_physical, *halo_importer, Tpetra::CombineMode::INSERT);
	}

	updateAllHostViews();
	haloUpdateTime->stop();
}

void CoordsT::setLagrangian(bool val) {
	if (val) {
		if (!_is_lagrangian) {
			_is_lagrangian = true;
			// insert code here to copy from pts to physical_pts since this is the first time the coordinates became Lagrangian

			host_view_type lagrangian_coords = pts->getLocalView<host_view_type>();

			pts_physical = Teuchos::rcp(new mvec_type(map, _nDim, false /*set to zero*/));
			host_view_type physical_coords = pts_physical->getLocalView<host_view_type>();

			Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,lagrangian_coords.extent(0)), KOKKOS_LAMBDA(const int i) {
				lagrangian_coords(i,0) = physical_coords(i,0);
				lagrangian_coords(i,1) = physical_coords(i,1);
				lagrangian_coords(i,2) = physical_coords(i,2);
			});

			this->updateHaloPts();
		}
		updateAllHostViews();
	}
}

void CoordsT::deltaUpdatePhysicalCoordsFromVector(const mvec_type* update_vector) {
	// to_vec should generally be physical coords (pts_physical)
	mvec_type* pts_evaluate_to_vec = (_is_lagrangian) ? pts_physical.getRawPtr() : pts.getRawPtr();
	host_view_type updateVals = update_vector->getLocalView<host_view_type>();
	host_view_type ptsViewTo = pts_evaluate_to_vec->getLocalView<host_view_type>();
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,ptsViewTo.extent(0)),  KOKKOS_LAMBDA(const int i) {
		for (int j=0; j<_nDim; j++) ptsViewTo(i,j) += updateVals(i,j);
	});

	this->updateHaloPts();
}

void CoordsT::deltaUpdatePhysicalCoordsFromVectorByTimestep(const double dt, const mvec_type* update_vector) {
	// to_vec should generally be physical coords (pts_physical)
	mvec_type* pts_evaluate_to_vec = (_is_lagrangian) ? pts_physical.getRawPtr() : pts.getRawPtr();
	host_view_type updateVals = update_vector->getLocalView<host_view_type>();
	host_view_type ptsViewTo = pts_evaluate_to_vec->getLocalView<host_view_type>();
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,ptsViewTo.extent(0)),  KOKKOS_LAMBDA(const int i) {
		for (int j=0; j<_nDim; j++) ptsViewTo(i,j) += dt*updateVals(i,j);
	});

	this->updateHaloPts();
}

void CoordsT::deltaUpdatePhysicalCoordsFromVectorFunction(function_type* fn, bool evaluate_using_lagrangian_coordinates) {
	// from_vec should generally be Lagrangian coords (pts)
	mvec_type* pts_evaluate_from_vec = (_is_lagrangian && !evaluate_using_lagrangian_coordinates) ? pts_physical.getRawPtr() : pts.getRawPtr();
	// to_vec should generally be physical coords (pts_physical)
	mvec_type* pts_evaluate_to_vec = (_is_lagrangian) ? pts_physical.getRawPtr() : pts.getRawPtr();
	host_view_type ptsViewFrom = pts_evaluate_from_vec->getLocalView<host_view_type>();
	host_view_type ptsViewTo = pts_evaluate_to_vec->getLocalView<host_view_type>();
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,ptsViewTo.extent(0)),
		incrementByEvaluateVector(ptsViewTo, ptsViewFrom, fn));

	this->updateHaloPts();
}

void CoordsT::overwriteCoordinates(const mvec_type* data, bool use_physical_coords) {
	// to_vec should generally be physical coords (pts_physical)
	mvec_type* pts_evaluate_to_vec = (_is_lagrangian && use_physical_coords) ? pts_physical.getRawPtr() : pts.getRawPtr();
	host_view_type ptsViewTo = pts_evaluate_to_vec->getLocalView<host_view_type>();
	host_view_type fieldData = data->getLocalView<host_view_type>();
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,ptsViewTo.extent(0)), KOKKOS_LAMBDA(const int i) {
		ptsViewTo(i,0) = fieldData(i,0);
		ptsViewTo(i,1) = fieldData(i,1);
		ptsViewTo(i,2) = fieldData(i,2);
	});
	this->updateHaloPts();
}

void CoordsT::snapToSphere(const double radius, bool use_physical_coords) {
	// scales coordinates to a unit vector

	// to_vec should generally be physical coords (pts_physical)
	mvec_type* pts_evaluate_to_vec = (_is_lagrangian && use_physical_coords) ? pts_physical.getRawPtr() : pts.getRawPtr();
	host_view_type ptsViewTo = pts_evaluate_to_vec->getLocalView<host_view_type>();
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,ptsViewTo.extent(0)),  KOKKOS_LAMBDA(const int i) {
		double this_norm = 0;
		for (int j=0; j<_nDim; j++) this_norm += ptsViewTo(i,j)*ptsViewTo(i,j);
		this_norm = std::sqrt(this_norm);
		for (int j=0; j<_nDim; j++) ptsViewTo(i,j) = radius*ptsViewTo(i,j)/this_norm;
	});

	this->updateHaloPts();
}

void CoordsT::transformLagrangianCoordsToPhysicalCoordsByVectorFunction(function_type* fn) {
	// from_vec should generally be Lagrangian coords (pts)
	mvec_type* pts_evaluate_from_vec = pts.getRawPtr();
	// to_vec should generally be physical coords (pts_physical)
	mvec_type* pts_evaluate_to_vec = (_is_lagrangian) ? pts_physical.getRawPtr() : pts.getRawPtr();
	host_view_type ptsViewFrom = pts_evaluate_from_vec->getLocalView<host_view_type>();
	host_view_type ptsViewTo = pts_evaluate_to_vec->getLocalView<host_view_type>();
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,ptsViewTo.extent(0)),
		evaluateVector(ptsViewTo, ptsViewFrom, fn));

	this->updateHaloPts();
}

void CoordsT::updateAllHostViews() {
	if (!pts.is_null()) pts_view = pts->getLocalView<host_view_type>();
	if (!pts_physical.is_null()) pts_physical_view = pts_physical->getLocalView<host_view_type>();
	if (!halo_pts.is_null()) halo_pts_view = halo_pts->getLocalView<host_view_type>();
	if (!halo_pts_physical.is_null()) halo_pts_physical_view = halo_pts_physical->getLocalView<host_view_type>();
}

}

std::ostream& operator << (std::ostream& os, const Compadre::CoordsT& coords) {
	os << "CoordsT basic info \n";
	os << "\t max coords allowed in Kokkos::View = " << coords.nLocalMax() << std::endl;
	os << "\t number of coords currently in Kokkos::View = " << coords.nLocal() << std::endl;
// 	os << "\t needs memory sync? : " << ( coords.needSync() ? "yes" : "no" ) << std::endl;
	return os;
}
