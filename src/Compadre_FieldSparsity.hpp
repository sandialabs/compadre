#ifndef _COMPADRE_FIELDSPARSITY_HPP_
#define _COMPADRE_FIELDSPARSITY_HPP_

//! Whether DOF interactions in a field are global or locally supported
enum FieldSparsityType {
    //! Locally supported DOF interaction leading to a banded matrix
    Banded,
    //! Globally supported DOF interaction leading to completely filled row or column of a matrix
    Global
};

#endif
