#ifndef _COMPADRE_FUNCTORS_HPP_
#define _COMPADRE_FUNCTORS_HPP_

namespace Compadre {

struct DefaultTag{
    DefaultTag() {};
    // intentionally empty
};

struct ConvertLayoutLeftToRight {

    //! Converts from layout left to right
    KOKKOS_INLINE_FUNCTION
    void operator() (const DefaultTag&, const member_type& teamMember) const {
        
        // Quang's code goes here
        int i = teamMember.league_rank();
        printf("CALLED!\n");

    }

}; 

}; // Compadre

#endif
