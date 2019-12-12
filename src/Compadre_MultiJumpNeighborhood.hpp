#ifndef _COMPADRE_MULTIJUMPNEIGHBORHOOD_HPP_
#define _COMPADRE_MULTIJUMPNEIGHBORHOOD_HPP_

#include "CompadreHarness_Config.h"
#include "CompadreHarness_Typedefs.hpp"
#include "Compadre_NeighborhoodT.hpp"
#include <Compadre_PointCloudSearch.hpp>


namespace Compadre {

/** Generates neighbor lists from existing neighbor lists
 *
 * This class is designed to take in neighbor lists (possible more than one 
 * search), and generate a neighbors of neighbors type list.
 *
 * After constructing the lists, it is meant to be used as a NeighborhoodT
 * type class for accessing number of neighbors and neighbor indices.
*/
class MultiJumpNeighborhood : public NeighborhoodT {

    protected:

        const NeighborhoodT* _neighborhood_1;
        const NeighborhoodT* _neighborhood_2;

    public:

        MultiJumpNeighborhood(const NeighborhoodT* neighborhood_1, const NeighborhoodT* neighborhood_2 = NULL);

        virtual ~MultiJumpNeighborhood() {};

        void constructNeighborOfNeighborLists(const scalar_type max_search_size = 0.0, bool use_physical_coords = true);

};



}

#endif
