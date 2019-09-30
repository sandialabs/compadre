#ifndef _COMPADRE_FUNCTORS_HPP_
#define _COMPADRE_FUNCTORS_HPP_

#include "Compadre_Operators.hpp"
#include "Compadre_ParallelManager.hpp"

namespace Compadre {

struct DefaultTag{
    DefaultTag() {};
    // intentionally empty
};

struct ConvertLayoutRightToLeft {
    int _rows, _cols;
    double* _permanent_mat_ptr;
    ParallelManager _pm;

    // Constructor
    ConvertLayoutRightToLeft(const ParallelManager &pm, int rows, int cols, double* mat_ptr):
        _pm(pm), _rows(rows), _cols(cols), _permanent_mat_ptr(mat_ptr) {};

    KOKKOS_INLINE_FUNCTION
    void operator() (const member_type& teamMember) const {
        std::cout << "IN FUNCTOR! rows " << _rows << " cols " << _cols << std::endl;
        // Create team member
        const int local_index = teamMember.league_rank();
        // Create a view for the right matrix type
        scratch_matrix_left_type permanent_mat_view(_permanent_mat_ptr + TO_GLOBAL(local_index)*TO_GLOBAL(_rows)*TO_GLOBAL(_cols), _rows, _cols);
        // scratch_matrix_left_type permanent_mat_view(_permanent_mat_ptr, _rows, _cols);

        // Create a matrix data living on the scratch memory
        scratch_matrix_left_type local_mat_view(teamMember.team_scratch(_pm.getTeamScratchLevel(1)), _cols, _rows);

        // Create 1D array view of the memory
        scratch_vector_type local_mat_view_flat(local_mat_view.data(), _cols*_rows);
        scratch_vector_type permanent_mat_view_flat(permanent_mat_view.data(), _rows*_cols);

        // Copy and transpose the matrix from permanent memory into scratch memory
        Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, _cols), [=] (const int i) {
            for (int j=0; j<_rows; j++) {
                // Transpose the matrix
                local_mat_view(i, j) = permanent_mat_view(j, i);
                // std::cout << "i " << i << " j " << j << " local_mat_view " << local_mat_view(i, j) << std::endl;
                // std::cout << "i " << i << " j " << j << " permanent_mat_view " << permanent_mat_view(j, i) << std::endl;
            }
        });
        teamMember.team_barrier();

        // Now copy the flat 1D memory over
        Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, _cols*_rows), [=] (const int i) {
            permanent_mat_view_flat(i) = local_mat_view_flat(i);
            // std::cout << " i " << i << " permanent_mat_view_flat " << permanent_mat_view_flat(i) << std::endl; 
        });
        teamMember.team_barrier();
    }
};

struct ConvertLayoutLeftToRight {
    int _rows, _cols;
    double* _permanent_mat_ptr;
    ParallelManager _pm;

    // Constructor
    ConvertLayoutLeftToRight(const ParallelManager &pm, int rows, int cols, double* mat_ptr):
        _pm(pm), _rows(rows), _cols(cols), _permanent_mat_ptr(mat_ptr) {};

    KOKKOS_INLINE_FUNCTION
    void operator() (const member_type& teamMember) const {
        // Create team member
        const int local_index = teamMember.league_rank();
        // Create a view for the right matrix type
        scratch_matrix_left_type permanent_mat_view(_permanent_mat_ptr + TO_GLOBAL(local_index)*TO_GLOBAL(_rows)*TO_GLOBAL(_cols), _rows, _cols);

        // Create a matrix data living on the scratch memory
        scratch_matrix_right_type local_mat_view(teamMember.team_scratch(_pm.getTeamScratchLevel(1)), _rows, _cols);

        // Create 1D array view of the memory
        scratch_vector_type local_mat_view_flat(local_mat_view.data(), _rows*_cols);
        scratch_vector_type permanent_mat_view_flat(permanent_mat_view.data(), _rows*_cols);

        // Copy and transpose the matrix from permanent memory into scratch memory
        Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, _rows), [=] (const int i) {
            for (int j=0; j<_cols; j++) {
                // Transpose the matrix
                local_mat_view(i, j) = permanent_mat_view(i, j);
            }
        });
        teamMember.team_barrier();

        // Now copy the flat 1D memory over
        Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, _rows*_cols), [=] (const int i) {
            permanent_mat_view_flat(i) = local_mat_view_flat(i);
        });
        teamMember.team_barrier();
    }
};

}; // Compadre

#endif
