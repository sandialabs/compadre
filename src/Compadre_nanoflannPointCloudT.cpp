#include "Compadre_nanoflannPointCloudT.hpp"

#include "Compadre_CoordsT.hpp"
#include "Compadre_XyzVector.hpp"

namespace Compadre {

///// Converts a local point's xyz coordinates to an array of scalars.
//void LocalPointCloudT::convertToArray(scalar_type* arry, const local_index_type idx) {
//	const LocalPointCloudT::xyz_type pt = coords->getLocalCoords(idx,  true /* w/ halo */);
//	arry[0] = pt.x;
//	arry[1] = pt.y;
//	arry[2] = pt.z;
//}
//
//inline local_index_type LocalPointCloudT::kdtree_get_point_count() const {return coords->nLocal(true /* w/ halo */);}
//
///// Returns the coordinate value of a pt.
//inline scalar_type LocalPointCloudT::kdtree_get_pt(const local_index_type idx, int dim) const {
//	const LocalPointCloudT::xyz_type pt = coords->getLocalCoords(idx, true /* w/ halo */);
//	if (dim == 0)
//		return pt.x;
//	else if (dim == 1)
//		return pt.y;
//	else
//		return pt.z;
//}
//
//inline scalar_type LocalPointCloudT::kdtree_distance(const scalar_type* queryPt, const local_index_type idx,
//		   global_index_type sz) const {
//	const LocalPointCloudT::xyz_type queryVec(queryPt);
//	const scalar_type dist = coords->distance(idx, queryVec);
//	return dist * dist;
//}



///// Converts a local point's xyz coordinates to an array of scalars.
//void GlobalPointCloudT::convertToArray(scalar_type* arry, const global_index_type idx) {
//	const GlobalPointCloudT::xyz_type pt = coords->getGlobalCoords(idx);
//	arry[0] = pt.x;
//	arry[1] = pt.y;
//	arry[2] = pt.z;
//}
//
//inline global_index_type GlobalPointCloudT::kdtree_get_point_count() const {return coords->nGlobal();}
//
///// Returns the coordinate value of a pt.
//inline scalar_type GlobalPointCloudT::kdtree_get_pt(const global_index_type idx, int dim) const {
//	const GlobalPointCloudT::xyz_type pt = coords->getGlobalCoords(idx);
//	if (dim == 0)
//		return pt.x;
//	else if (dim == 1)
//		return pt.y;
//	else
//		return pt.z;
//}
//
//inline scalar_type GlobalPointCloudT::kdtree_distance(const scalar_type* queryPt, const global_index_type idx,
//		   global_index_type sz) const {
//	const GlobalPointCloudT::xyz_type queryVec(queryPt);
//	const scalar_type dist = coords->globalDistance(idx, queryVec);
//	return dist * dist;
//}

}

