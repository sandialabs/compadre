#ifndef _COMPADRE_MISC_HPP_
#define _COMPADRE_MISC_HPP_

namespace Compadre {

struct DefaultTag{
    // intentionally empty
};

struct ConvertLayoutLeftToRight {

    //! Converts from layout left to right
    KOKKOS_INLINE_FUNCTION
    void operator() (const DefaultTag&, const member_type& teamMember) const {
        
        // Quang's code goes here
        printf("CALLED!\n");

    }

}; 

}; // Compadre

#endif
