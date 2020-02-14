#ifndef _COMPADRE_NANOFLANN_POINTCLOUD_HPP_
#define	_COMPADRE_NANOFLANN_POINTCLOUD_HPP_

#include "CompadreHarness_Config.h"
#include "CompadreHarness_Typedefs.hpp"

#include "Compadre_CoordsT.hpp"
#include "Compadre_XyzVector.hpp"

namespace Compadre {

/**
	@brief Provides an interface for SphericalCoordsT and EuclideanCoordsT into the Nanoflann.hpp TPL.
	See the [Nanoflann documentation](http://nanoflann-reference.mrpt.org/svn/classnanoflann_1_1KDTreeSingleIndexAdaptor.html) for more details.

	Note: This class interfaces to local coordinates only; the resulting kd-tree will only encompass coordinates on this processor.	
	
	This class implements the methods required by the Nanoflann library to interface to its kd-tree construction and query code.

	Note: Since Nanoflann can not link against objects corresponding to to this file's cpp, the definitions of this class must remain fully
	defined within the header file.
*/

//class CoordsT;
//class XyzVector;

class LocalPointCloudT {

	protected:

		typedef XyzVector xyz_type;
		typedef CoordsT coords_type;
	
		const coords_type* coords;
		bool _use_physical_coords;
		bool _use_precomputed_processor_boundaries;
		host_view_type _pts_view;
		host_view_type _halo_pts_view;
		local_index_type _nlocal_minus_halo;
	
	public:

		LocalPointCloudT(const coords_type* cIn, bool use_physical_coords = true, bool use_precomputed_processor_boundaries = true) :
			coords(cIn), _use_physical_coords(use_physical_coords), _use_precomputed_processor_boundaries(use_precomputed_processor_boundaries) {
			// this class is only called at the construction of a Neighborhood, so we can trust that the coordinates will not
			// change between its construction and use in a search.
			// we get a local view so that we have easier access to the coordinates that we will need for neighbor searches
			_pts_view = cIn->getPts(false,use_physical_coords)->getLocalView<host_view_type>();
			_halo_pts_view = cIn->getPts(true,use_physical_coords)->getLocalView<host_view_type>();
			_nlocal_minus_halo = _pts_view.extent(0);
		};
	
		~LocalPointCloudT() {};
	
		bool include_halo = true;

//		/// Converts a local point's xyz coordinates to an array of scalars.
//		void convertToArray(scalar_type* arry, const local_index_type idx);
//
//		local_index_type kdtree_get_point_count() const;
//
//		/// Returns the coordinate value of a pt.
//		scalar_type kdtree_get_pt(const local_index_type idx, int dim) const;
//
//		/// Returns the squared distance between a query point and a local point.
//		scalar_type kdtree_distance(const scalar_type* queryPt, const local_index_type idx,
//										   global_index_type sz = 3) const;

		/// Bounding box query method required by Nanoflann.
		template <class BBOX> bool kdtree_get_bbox(BBOX& bb) const {
			/*
			 * Returning 'true' indicates that we took care of the bounding box.
			 * 'false' uses nanoflann to loop over all coords on the processor
			 * (local + halo)
			 */
			if (_use_precomputed_processor_boundaries) {
				std::vector<scalar_type> min_vals = coords->boundingBoxMinOnProcessor();
				std::vector<scalar_type> max_vals = coords->boundingBoxMaxOnProcessor();
				scalar_type halo_offset = coords->getHaloSize();
				for (local_index_type i=0; i<coords->nDim(); i++) {
					bb[i].low = min_vals[i] - halo_offset;
					bb[i].high = max_vals[i] + halo_offset;
				}
			}
			return _use_precomputed_processor_boundaries;
		}

		/// Converts a local point's xyz coordinates to an array of scalars.
		void convertToArray(scalar_type* arry, const local_index_type idx) {
			if (idx>=_nlocal_minus_halo) {
				arry[0] = _halo_pts_view(idx - _nlocal_minus_halo,0);
				arry[1] = _halo_pts_view(idx - _nlocal_minus_halo,1);
				arry[2] = _halo_pts_view(idx - _nlocal_minus_halo,2);
			} else {
				arry[0] = _pts_view(idx,0);
				arry[1] = _pts_view(idx,1);
				arry[2] = _pts_view(idx,2);
			}
		}

		inline local_index_type kdtree_get_point_count() const {return _pts_view.extent(0) + _halo_pts_view.extent(0);}

		/// Returns the coordinate value of a pt.
		inline scalar_type kdtree_get_pt(const local_index_type idx, int dim) const {
			if (idx>=_nlocal_minus_halo) {
				switch (dim) {
				case 2:
					return _halo_pts_view(idx - _nlocal_minus_halo,2);
				case 1:
					return _halo_pts_view(idx - _nlocal_minus_halo,1);
				default:
					return _halo_pts_view(idx - _nlocal_minus_halo,0);
				}
			} else {
				switch (dim) {
				case 2:
					return _pts_view(idx,2);
				case 1:
					return _pts_view(idx,1);
				default:
					return _pts_view(idx,0);
				}
			}
		}

		inline scalar_type kdtree_distance(const scalar_type* queryPt, const local_index_type idx,
				   global_index_type sz) const {
			const LocalPointCloudT::xyz_type queryVec1(queryPt);
			LocalPointCloudT::xyz_type queryVec2;
			if (idx>=_nlocal_minus_halo) {
				queryVec2 = LocalPointCloudT::xyz_type(
						_halo_pts_view(idx - _nlocal_minus_halo,0),
						_halo_pts_view(idx - _nlocal_minus_halo,1),
						_halo_pts_view(idx - _nlocal_minus_halo,2));
			} else {
				queryVec2 = LocalPointCloudT::xyz_type(
						_pts_view(idx,0),
						_pts_view(idx,1),
						_pts_view(idx,2));
			}

			const scalar_type dist = coords->distance(queryVec1, queryVec2);
			return dist;
		}

};	

class GlobalPointCloudT {
	protected:
		typedef XyzVector xyz_type;
		typedef CoordsT coords_type;

		coords_type* coords;

	public:
		GlobalPointCloudT(coords_type* cIn) :
			coords(cIn) {};

		~GlobalPointCloudT() {};
	
//		/// Converts a local point's xyz coordinates to an array of scalars.
//		void convertToArray(scalar_type* arry, const global_index_type idx);
//
//		global_index_type kdtree_get_point_count() const;
//
//		/// Returns the coordinate value of a pt.
//		scalar_type kdtree_get_pt(const global_index_type idx, int dim) const;
//
//		/// Returns the squared distance between a query point and a local point.
//		scalar_type kdtree_distance(const scalar_type* queryPt, const global_index_type idx,
//										   global_index_type sz = 3) const;
	
		/// Bounding box query method required by Nanoflann.  I'm not really sure what this does (or doesn't).
		template <class BBOX> bool kdtree_get_bbox(BBOX&) const {return false;}

		/// Converts a local point's xyz coordinates to an array of scalars.
		void convertToArray(scalar_type* arry, const global_index_type idx) {
			const GlobalPointCloudT::xyz_type pt = coords->getGlobalCoords(idx);
			arry[0] = pt.x;
			arry[1] = pt.y;
			arry[2] = pt.z;
		}

		inline global_index_type kdtree_get_point_count() const {return coords->nGlobal();}

		/// Returns the coordinate value of a pt.
		inline scalar_type kdtree_get_pt(const global_index_type idx, int dim) const {
			const GlobalPointCloudT::xyz_type pt = coords->getGlobalCoords(idx);
			if (dim == 0)
				return pt.x;
			else if (dim == 1)
				return pt.y;
			else
				return pt.z;
		}

		inline scalar_type kdtree_distance(const scalar_type* queryPt, const global_index_type idx,
				   global_index_type sz) const {
			const GlobalPointCloudT::xyz_type queryVec(queryPt);
			const scalar_type dist = coords->globalDistance(idx, queryVec);
			return dist;
		}
	
};	
	
}

#endif

