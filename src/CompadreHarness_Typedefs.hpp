#ifndef _COMPADRE_HARNESS_TYPEDEFS_HPP_
#define _COMPADRE_HARNESS_TYPEDEFS_HPP_

#include <Teuchos_RCP.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_ParameterList.hpp>

#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_CrsMatrix.hpp>

#ifdef TRILINOS_LINEAR_SOLVES
#include <Thyra_DefaultBlockedLinearOp.hpp>
#endif

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <Zoltan2_XpetraMultiVectorAdapter.hpp>
#include <Zoltan2_PartitioningSolution.hpp>
#include <Zoltan2_PartitioningProblem.hpp>
#include <Zoltan2_CoordinatePartitioningGraph.hpp>

//#include <MatrixMarket_Tpetra.hpp>

#include <string>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <utility>
#include <string>

namespace Compadre {

    typedef int local_index_type;
    typedef long long global_index_type;
    typedef double scalar_type;

    typedef Tpetra::Map<local_index_type, global_index_type> map_type;

    typedef Tpetra::MultiVector<scalar_type, local_index_type, global_index_type> mvec_scalar_type;
    typedef Tpetra::MultiVector<local_index_type, local_index_type, global_index_type> mvec_local_index_type;
    typedef Tpetra::MultiVector<global_index_type, local_index_type, global_index_type> mvec_global_index_type;

    typedef mvec_scalar_type mvec_type; // used almost exclusively

    typedef Tpetra::Vector<scalar_type, local_index_type, global_index_type> vec_scalar_type;
    typedef Tpetra::Import<local_index_type, global_index_type> importer_type;
    typedef Tpetra::Export<local_index_type, global_index_type> exporter_type;

    typedef typename mvec_scalar_type::dual_view_type dual_view_scalar_type;
    typedef typename mvec_local_index_type::dual_view_type dual_view_local_index_type;
    typedef typename mvec_global_index_type::dual_view_type dual_view_global_index_type;
    typedef dual_view_scalar_type dual_view_type;

    typedef typename mvec_scalar_type::node_type scalar_node_type;
    typedef typename mvec_local_index_type::node_type local_index_node_type;
    typedef typename mvec_global_index_type::node_type global_index_node_type;
    typedef scalar_node_type node_type;

    typedef typename dual_view_scalar_type::t_dev device_view_scalar_type;
    typedef typename dual_view_local_index_type::t_dev device_view_local_index_type;
    typedef typename dual_view_global_index_type::t_dev device_view_global_index_type;
    typedef device_view_scalar_type device_view_type;

    typedef typename dual_view_scalar_type::t_host host_view_scalar_type;
    typedef typename dual_view_local_index_type::t_host host_view_local_index_type;
    typedef typename dual_view_global_index_type::t_host host_view_global_index_type;
    typedef host_view_scalar_type host_view_type;

    typedef typename dual_view_type::host_mirror_space host_execution_space;

    typedef Tpetra::Operator<scalar_type, local_index_type,
                                global_index_type, node_type> op_type;

    typedef Kokkos::View<local_index_type***, Kokkos::DefaultHostExecutionSpace> local_dof_map_view_type;

#ifdef TRILINOS_LINEAR_SOLVES
    typedef Thyra::MultiVectorBase<scalar_type> thyra_mvec_type;
    typedef Thyra::DefaultBlockedLinearOp<scalar_type> thyra_block_op_type;
    typedef Thyra::LinearOpBase<scalar_type> thyra_linearop_type;
#endif

    typedef Kokkos::Random_XorShift64_Pool<> pool_type;
    typedef typename pool_type::generator_type generator_type;

    typedef Kokkos::TeamPolicy<Kokkos::DefaultHostExecutionSpace>               host_team_policy;
    typedef Kokkos::TeamPolicy<Kokkos::DefaultHostExecutionSpace>::member_type  host_member_type;

    typedef Kokkos::View<scalar_type*, Kokkos::DefaultHostExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > host_scratch_vector_scalar_type;
    typedef Kokkos::View<local_index_type*, Kokkos::DefaultHostExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > host_scratch_vector_local_index_type;
    typedef Kokkos::View<global_index_type*, Kokkos::DefaultHostExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> > host_scratch_vector_global_index_type;

    typedef Kokkos::View<scalar_type*, Kokkos::DefaultHostExecutionSpace > host_vector_scalar_type;
    typedef Kokkos::View<local_index_type*, Kokkos::DefaultHostExecutionSpace > host_vector_local_index_type;
    typedef Kokkos::View<global_index_type*, Kokkos::DefaultHostExecutionSpace > host_vector_global_index_type;

    typedef Tpetra::CrsGraph<local_index_type, global_index_type> crs_graph_type;
    typedef Tpetra::CrsMatrix<scalar_type, local_index_type, global_index_type> crs_matrix_type;

    typedef Zoltan2::XpetraMultiVectorAdapter<mvec_type> z2_scalar_adapter_type;
    typedef Zoltan2::XpetraMultiVectorAdapter<mvec_local_index_type> z2_local_index_adapter_type;
    typedef Zoltan2::PartitioningProblem<z2_scalar_adapter_type> z2_problem_type;
    typedef typename z2_scalar_adapter_type::part_t z2_partition_type;
    typedef std::remove_reference<decltype(std::declval<z2_problem_type>().getSolution().getPartBoxesView()[0])>::type z2_box_type;

    enum op_needing_interaction { bc, source, physics };

    //! Whether DOF interactions in a field are global or locally supported
    enum FieldSparsityType {
        //! Locally supported DOF interaction leading to a banded matrix
        Banded,
        //! Globally supported DOF interaction leading to completely filled row or column of a matrix
        Global
    };

    struct InteractingFields {
        public:
            std::string src_fieldname;
            std::string trg_fieldname;
            local_index_type src_fieldnum;
            local_index_type trg_fieldnum;
            std::vector<op_needing_interaction> ops_needing;

            InteractingFields(op_needing_interaction op, const local_index_type source_num, const local_index_type target_num = -1) {
                src_fieldnum = source_num;
                if (target_num < 0) trg_fieldnum = source_num;
                else trg_fieldnum = target_num;
                ops_needing.push_back(op);
            }

            bool operator== (const InteractingFields& other) const
            {
                if (this->src_fieldnum == other.src_fieldnum &&  this->trg_fieldnum == other.trg_fieldnum)
                    return true;
                return false;
            }

            void operator+= (const InteractingFields& other)  {
                // performs a merge of the ops_needing from both objects
                for (op_needing_interaction an_op:other.ops_needing) {
                    // check if an_op already exists, and if not push it
                    bool found = false;
                    for (op_needing_interaction my_op:this->ops_needing) {
                        if (my_op==an_op) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) this->ops_needing.push_back(an_op);
                }
            }

    };

}

#endif
