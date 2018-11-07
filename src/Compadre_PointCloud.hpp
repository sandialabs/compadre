#include <tpl/nanoflann.hpp>

namespace Compadre {

template <typename view_type>
class PointCloud {

    protected:

        view_type _pts_view;
    
    public:

        PointCloud(view_type pts_view) : _pts_view(pts_view) {};
    
        ~PointCloud() {};
    
        //! Bounding box query method required by Nanoflann.
        template <class BBOX> bool kdtree_get_bbox(BBOX& bb) const {return false;}

        //! Returns the number of source sites
        inline int kdtree_get_point_count() const {return _pts_view.dimension_0();}

        //! Returns the coordinate value of a point
        inline double kdtree_get_pt(const int idx, int dim) const {return _pts_view(idx,dim);}

        //! Returns the distance between a point and a source site, given its index
        inline double kdtree_distance(const double* queryPt, const int idx, long long sz) const {

            double distance = 0;
            for (int i=0; i<3; ++i) {
                distance += (_pts_view(idx,i)-queryPt[i])*(_pts_view(idx,i)-queryPt[i]);
            }
            return std::sqrt(distance);

        }

}; // PointCloud
}; // Compadre
