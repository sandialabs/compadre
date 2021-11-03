#include <Kokkos_Core.hpp>

int main (int argc, char* args[]) {
    Kokkos::initialize(argc, args);
    {
    //    someHostFunction();
    typedef Kokkos::DefaultHostExecutionSpace host_execution_space;
    typedef typename Kokkos::TeamPolicy<host_execution_space> host_team_policy;
    typedef Kokkos::View<double*, host_execution_space, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
        host_scratch_vector_type;
    typedef typename host_team_policy::member_type  host_member_type;
    
    int team_scratch_size = 0;
    team_scratch_size += host_scratch_vector_type::shmem_size(100);
    
    std::vector<double> a(3, 1.0);
    size_t num_target_sites = 20;
    
    size_t min_num_neighbors = 0;
    Kokkos::parallel_reduce("knn search", host_team_policy(num_target_sites, Kokkos::AUTO)
    .set_scratch_size(0 /*shared memory level*/, Kokkos::PerTeam(team_scratch_size)),
    [&](const host_member_type& teamMember, size_t& t_min_num_neighbors) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, 12), [&](const int j) { 
            a.clear();
        });
        teamMember.team_barrier();
        a.clear();
        t_min_num_neighbors = a.size();
    }, Kokkos::Min<size_t>(min_num_neighbors) );
    }
    Kokkos::finalize();

return 1;
}
