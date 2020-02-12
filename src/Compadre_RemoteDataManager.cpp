#include "Compadre_RemoteDataManager.hpp"

#include "Compadre_XyzVector.hpp"
#include "Compadre_CoordsT.hpp"
#include "Compadre_ParticlesT.hpp"
#include "Compadre_RemapManager.hpp"

// temporary for diagnostics
#include <Compadre_FieldManager.hpp>
#include <Compadre_FieldT.hpp>
#include <Compadre_AnalyticFunctions.hpp>

namespace Compadre {

RemoteDataManager::RemoteDataManager(Teuchos::RCP<const Teuchos::Comm<local_index_type> > global_comm,
		Teuchos::RCP<const Teuchos::Comm<local_index_type> > local_comm,
		const coords_type* our_coords,
		const local_index_type my_program_coloring,
		const local_index_type peer_program_coloring,
		const bool use_physical_coords,
		const scalar_type bounding_box_relative_tolerance) :
		RemoteDataManager::RemoteDataManager(global_comm, local_comm, our_coords, my_program_coloring, peer_program_coloring,
			host_view_local_index_type("",0,0), std::vector<int>(0), use_physical_coords, bounding_box_relative_tolerance) {}

RemoteDataManager::RemoteDataManager(Teuchos::RCP<const Teuchos::Comm<local_index_type> > global_comm,
		Teuchos::RCP<const Teuchos::Comm<local_index_type> > local_comm,
		const coords_type* our_coords,
		const local_index_type my_program_coloring,
		const local_index_type peer_program_coloring,
		host_view_local_index_type flags,
		const std::vector<local_index_type> flags_for_transfer,
		const bool use_physical_coords,
		const scalar_type bounding_box_relative_tolerance)
	: _global_comm(global_comm), _local_comm(local_comm), _our_coords(our_coords), _ndim(our_coords->nDim()) {

	/*
	 *
	 * !!! The assumption is made that the processor bounding boxes are determined using the same
	 * coordinate system as the points being compared (otherwise neighbor won't be found)
	 *
	 */

	RemoteDataMapConstructionTime = Teuchos::TimeMonitor::getNewCounter ("Remote Data Map Construction Time");
	RemoteDataMapConstructionTime->start();

	// determine if this group of processes are lower or upper
	_amLower = my_program_coloring < peer_program_coloring;

	local_index_type peer_root;

	//  there is a coloring and key for each of the two comms
	local_index_type lower_root_plus_upper_all_color;
	local_index_type upper_root_plus_lower_all_color;
	local_index_type lower_root_plus_upper_all_key;
	local_index_type upper_root_plus_lower_all_key;

    if (_amLower) {
    	peer_root = local_comm->getSize();
    	if (local_comm->getRank()==0) {
    		lower_root_plus_upper_all_color = 0;
    		lower_root_plus_upper_all_key = 0;
    	} else {
    		lower_root_plus_upper_all_color = -1;
    		lower_root_plus_upper_all_key = -1;
    	}
    	upper_root_plus_lower_all_color = 0; // offset by peer root
    	upper_root_plus_lower_all_key = local_comm->getRank() + 1; // offset by peer root
    } else {
    	peer_root = 0;
    	if (local_comm->getRank()==0) {
    		upper_root_plus_lower_all_color = 0;
    		upper_root_plus_lower_all_key = 0;
    	} else {
    		upper_root_plus_lower_all_color = -1;
    		upper_root_plus_lower_all_key = -1;
    	}
    	lower_root_plus_upper_all_color = 0; // offset by peer root
    	lower_root_plus_upper_all_key = local_comm->getRank() + 1; // offset by peer root
    }

    _lower_root_plus_upper_all_comm = global_comm->split(lower_root_plus_upper_all_color, lower_root_plus_upper_all_key);
    _upper_root_plus_lower_all_comm = global_comm->split(upper_root_plus_lower_all_color, upper_root_plus_lower_all_key);

    // all processors already know our own bounding boxes so need need to send to local root
    // because of communicators already built, we already know peer processes # of processors
    // but only at the root
    local_index_type num_peer_processors;
    if (_amLower && local_comm->getRank()==0) {
    	num_peer_processors = _lower_root_plus_upper_all_comm->getSize() - 1;
    } else if (local_comm->getRank()==0) {
    	num_peer_processors = _upper_root_plus_lower_all_comm->getSize() - 1;
    }

    // we now share that information with the rest of the processors on our local communicator
    if (_amLower) {
    	Teuchos::broadcast<local_index_type, local_index_type>(*local_comm, 0, 1, &num_peer_processors);
    } else {
    	Teuchos::broadcast<local_index_type, local_index_type>(*local_comm, 0, 1, &num_peer_processors);
    }

    std::vector<scalar_type> lower_processes_bounding_box_mins;
    std::vector<scalar_type> lower_processes_bounding_box_maxs;
    std::vector<scalar_type> upper_processes_bounding_box_mins;
    std::vector<scalar_type> upper_processes_bounding_box_maxs;

    // to broadcast the bounding boxes from our processes root to all of the peer processes, we need that information
    // at our root with the same variable name that the peer processors expect to be broadcast to
    if (_amLower) {
    	lower_processes_bounding_box_mins.resize(_ndim*local_comm->getSize());
    	lower_processes_bounding_box_maxs.resize(_ndim*local_comm->getSize());
    	for (local_index_type i=0; i<local_comm->getSize(); i++) {
            const std::vector<scalar_type> our_processes_bounding_box_mins = our_coords->boundingBoxMinOnProcessor(i);
            const std::vector<scalar_type> our_processes_bounding_box_maxs = our_coords->boundingBoxMaxOnProcessor(i);
            for (local_index_type j=0; j<_ndim; j++) {
				lower_processes_bounding_box_mins[i*_ndim + j] = our_processes_bounding_box_mins[j]-bounding_box_relative_tolerance*(our_processes_bounding_box_maxs[j]-our_processes_bounding_box_mins[j]);
				lower_processes_bounding_box_maxs[i*_ndim + j] = our_processes_bounding_box_maxs[j]+bounding_box_relative_tolerance*(our_processes_bounding_box_maxs[j]-our_processes_bounding_box_mins[j]);
            }
    	}
//    	for (local_index_type i=0; i<local_comm->getSize()*_ndim; i++) {
//    		std::cout << _amLower << " " << lower_processes_bounding_box_mins[i] << std::endl;
//    		std::cout << _amLower << " " << lower_processes_bounding_box_maxs[i] << std::endl;
//    	}
    } else {
    	upper_processes_bounding_box_mins.resize(_ndim*local_comm->getSize());
    	upper_processes_bounding_box_maxs.resize(_ndim*local_comm->getSize());
    	for (local_index_type i=0; i<local_comm->getSize(); i++) {
            const std::vector<scalar_type> our_processes_bounding_box_mins = our_coords->boundingBoxMinOnProcessor(i);
            const std::vector<scalar_type> our_processes_bounding_box_maxs = our_coords->boundingBoxMaxOnProcessor(i);
            for (local_index_type j=0; j<_ndim; j++) {
				upper_processes_bounding_box_mins[i*_ndim + j] = our_processes_bounding_box_mins[j]-bounding_box_relative_tolerance*(our_processes_bounding_box_maxs[j]-our_processes_bounding_box_mins[j]);
				upper_processes_bounding_box_maxs[i*_ndim + j] = our_processes_bounding_box_maxs[j]+bounding_box_relative_tolerance*(our_processes_bounding_box_maxs[j]-our_processes_bounding_box_mins[j]);
            }
    	}
//    	for (local_index_type i=0; i<local_comm->getSize()*_ndim; i++) {
//			std::cout << _amLower << " " << upper_processes_bounding_box_mins[i] << std::endl;
//			std::cout << _amLower << " " << upper_processes_bounding_box_maxs[i] << std::endl;
//    	}
    }

    // now we broadcast from the root of both our processors as well as the peer processors
    // after this set of calls, all bounding boxes exist on all processors
    if (_amLower) {
    	if (local_comm->getRank()==0) {
    		Teuchos::broadcast<local_index_type, scalar_type>(*_lower_root_plus_upper_all_comm, 0, (local_index_type)(_ndim*local_comm->getSize()), &lower_processes_bounding_box_mins[0]);
    		Teuchos::broadcast<local_index_type, scalar_type>(*_lower_root_plus_upper_all_comm, 0, (local_index_type)(_ndim*local_comm->getSize()), &lower_processes_bounding_box_maxs[0]);
    	}
	} else {
		lower_processes_bounding_box_mins = std::vector<scalar_type>(num_peer_processors*_ndim);
		lower_processes_bounding_box_maxs = std::vector<scalar_type>(num_peer_processors*_ndim);
    	Teuchos::broadcast<local_index_type, scalar_type>(*_lower_root_plus_upper_all_comm, 0, _ndim*num_peer_processors, &lower_processes_bounding_box_mins[0]);
        Teuchos::broadcast<local_index_type, scalar_type>(*_lower_root_plus_upper_all_comm, 0, _ndim*num_peer_processors, &lower_processes_bounding_box_maxs[0]);
    }
    if (_amLower) {
    	upper_processes_bounding_box_mins = std::vector<scalar_type>(num_peer_processors*_ndim);
    	upper_processes_bounding_box_maxs = std::vector<scalar_type>(num_peer_processors*_ndim);
    	Teuchos::broadcast<local_index_type, scalar_type>(*_upper_root_plus_lower_all_comm, 0, _ndim*num_peer_processors, &upper_processes_bounding_box_mins[0]);
   	    Teuchos::broadcast<local_index_type, scalar_type>(*_upper_root_plus_lower_all_comm, 0, _ndim*num_peer_processors, &upper_processes_bounding_box_maxs[0]);
    } else {
    	if (local_comm->getRank()==0) {
    		Teuchos::broadcast<local_index_type, scalar_type>(*_upper_root_plus_lower_all_comm, 0, (local_index_type)(_ndim*local_comm->getSize()), &upper_processes_bounding_box_mins[0]);
    		Teuchos::broadcast<local_index_type, scalar_type>(*_upper_root_plus_lower_all_comm, 0, (local_index_type)(_ndim*local_comm->getSize()), &upper_processes_bounding_box_maxs[0]);
    	}
	}

//	if (_amLower) {
//		std::cout << global_comm->getRank() << " " << _amLower << ":min " << upper_processes_bounding_box_mins[0] << std::endl;
//		std::cout << global_comm->getRank() << " " << _amLower << ":min " << upper_processes_bounding_box_mins[1] << std::endl;
//		std::cout << global_comm->getRank() << " " << _amLower << ":min " << upper_processes_bounding_box_mins[2] << std::endl;
//		std::cout << global_comm->getRank() << " " << _amLower << ":min " << upper_processes_bounding_box_mins[3] << std::endl;
//		std::cout << global_comm->getRank() << " " << _amLower << ":min " << upper_processes_bounding_box_mins[4] << std::endl;
//		std::cout << global_comm->getRank() << " " << _amLower << ":min " << upper_processes_bounding_box_mins[5] << std::endl;
//		std::cout << global_comm->getRank() << " " << _amLower << ":min " << upper_processes_bounding_box_mins[6] << std::endl;
//		std::cout << global_comm->getRank() << " " << _amLower << ":min " << upper_processes_bounding_box_mins[7] << std::endl;
//		std::cout << global_comm->getRank() << " " << _amLower << ":min " << upper_processes_bounding_box_mins[8] << std::endl;
//		std::cout << global_comm->getRank() << " " << _amLower << ":max " << upper_processes_bounding_box_maxs[0] << std::endl;
//		std::cout << global_comm->getRank() << " " << _amLower << ":max " << upper_processes_bounding_box_maxs[1] << std::endl;
//		std::cout << global_comm->getRank() << " " << _amLower << ":max " << upper_processes_bounding_box_maxs[2] << std::endl;
//		std::cout << global_comm->getRank() << " " << _amLower << ":max " << upper_processes_bounding_box_maxs[3] << std::endl;
//		std::cout << global_comm->getRank() << " " << _amLower << ":max " << upper_processes_bounding_box_maxs[4] << std::endl;
//		std::cout << global_comm->getRank() << " " << _amLower << ":max " << upper_processes_bounding_box_maxs[5] << std::endl;
//		std::cout << global_comm->getRank() << " " << _amLower << ":max " << upper_processes_bounding_box_maxs[6] << std::endl;
//		std::cout << global_comm->getRank() << " " << _amLower << ":max " << upper_processes_bounding_box_maxs[7] << std::endl;
//		std::cout << global_comm->getRank() << " " << _amLower << ":max " << upper_processes_bounding_box_maxs[8] << std::endl;
//	}
//	else {
//		std::cout << global_comm->getRank() << " " << _amLower << ":min " << lower_processes_bounding_box_mins[0] << std::endl;
//		std::cout << global_comm->getRank() << " " << _amLower << ":min " << lower_processes_bounding_box_mins[1] << std::endl;
//		std::cout << global_comm->getRank() << " " << _amLower << ":min " << lower_processes_bounding_box_mins[2] << std::endl;
//		std::cout << global_comm->getRank() << " " << _amLower << ":min " << lower_processes_bounding_box_mins[3] << std::endl;
//		std::cout << global_comm->getRank() << " " << _amLower << ":min " << lower_processes_bounding_box_mins[4] << std::endl;
//		std::cout << global_comm->getRank() << " " << _amLower << ":min " << lower_processes_bounding_box_mins[5] << std::endl;
//		std::cout << global_comm->getRank() << " " << _amLower << ":max " << lower_processes_bounding_box_maxs[0] << std::endl;
//		std::cout << global_comm->getRank() << " " << _amLower << ":max " << lower_processes_bounding_box_maxs[1] << std::endl;
//		std::cout << global_comm->getRank() << " " << _amLower << ":max " << lower_processes_bounding_box_maxs[2] << std::endl;
//		std::cout << global_comm->getRank() << " " << _amLower << ":max " << lower_processes_bounding_box_maxs[3] << std::endl;
//		std::cout << global_comm->getRank() << " " << _amLower << ":max " << lower_processes_bounding_box_maxs[4] << std::endl;
//		std::cout << global_comm->getRank() << " " << _amLower << ":max " << lower_processes_bounding_box_maxs[5] << std::endl;
//	}


    // first just do processor intersection search
    // each processor finds all other processor boundary boxes that it intersects
    // it then uses these intersections to get a list of peer processors it must communicate
    // this follows closely to the halo search algorithm
    std::vector<local_index_type> peer_processors_i_overlap;
    if (_amLower) {
    	z2_box_type my_processor_box(0, _ndim, &lower_processes_bounding_box_mins[_ndim*local_comm->getRank()], &lower_processes_bounding_box_maxs[_ndim*local_comm->getRank()]);
    	for (local_index_type i=0; i<num_peer_processors; i++) {
    		if (my_processor_box.boxesOverlap(_ndim, &upper_processes_bounding_box_mins[_ndim*i], &upper_processes_bounding_box_maxs[_ndim*i]))
    				peer_processors_i_overlap.push_back(i);
    	}
    } else {
    	z2_box_type my_processor_box(0, _ndim, &upper_processes_bounding_box_mins[_ndim*local_comm->getRank()], &upper_processes_bounding_box_maxs[_ndim*local_comm->getRank()]);
    	for (local_index_type i=0; i<num_peer_processors; i++) {
    		if (my_processor_box.boxesOverlap(_ndim, &lower_processes_bounding_box_mins[_ndim*i], &lower_processes_bounding_box_maxs[_ndim*i]))
    				peer_processors_i_overlap.push_back(i);
    	}
    }

//    std::cout << global_comm->getRank() << ": " << peer_processors_i_overlap.size() << std::endl;
//    typedef Kokkos::View<local_index_type*> lid_view_type;
	typedef Kokkos::View<const global_index_type*> const_gid_view_type;
	typedef Kokkos::View<global_index_type*> gid_view_type;
	typedef typename gid_view_type::HostMirror host_gid_view_type;
	typedef Kokkos::View<local_index_type> count_type;

	host_view_type ptsView = our_coords->getPts(false /*halo*/, use_physical_coords)->getLocalView<host_view_type>();
	std::vector<Teuchos::RCP<host_gid_view_type> > coords_to_send(peer_processors_i_overlap.size());

	const_gid_view_type gids = our_coords->getMapConst()->getMyGlobalIndices();

//	std::cout << global_comm->getRank() << " overlaps " << peer_processors_i_overlap[0] << std::endl;
//	if (peer_processors_i_overlap.size()>1) std::cout << global_comm->getRank() << " overlaps " << peer_processors_i_overlap[1] << std::endl;

	// loop over all processors that could possibly provide the information to
	// fill the target sites we have on this processor.
	// 1.) we start each entry with 0, which indicates that it needs to be found
	// we loop over peer processors, and when we find the one that contains our
	// point, we put a 1 in that entry
	// 2.) 1 indicates that this is a point that we need to do something about at
	// the end of the loop (needed information from this particular peer processor)
	// 3.) at the end of the loop, we put 2's in these entries, to indicate that
	// this site has already been accounted for by some previous peer processor
	// search, and it will be skipped for all remaining peer processor searches

	// this has the effect of 1) ensuring that a target site is only claimed by one
	// peer processor

	gid_view_type local_to_global_id_found("found", our_coords->nLocal());
    for (size_t i=0; i<peer_processors_i_overlap.size(); i++) {

			count_type count_gids_found("GID_Count");


    		scalar_type * peer_box_mins;
    		scalar_type * peer_box_maxs;

    		if (_amLower) {
    			peer_box_mins = &upper_processes_bounding_box_mins[_ndim*peer_processors_i_overlap[i]];
				peer_box_maxs = &upper_processes_bounding_box_maxs[_ndim*peer_processors_i_overlap[i]];
    		} else {
    			peer_box_mins = &lower_processes_bounding_box_mins[_ndim*peer_processors_i_overlap[i]];
				peer_box_maxs = &lower_processes_bounding_box_maxs[_ndim*peer_processors_i_overlap[i]];
    		}

    		struct SearchBoxFunctorUnconstrained {
    			gid_view_type gids_found;
    			host_view_type pts_view;
    			const_gid_view_type gids;
    			count_type count;
    			scalar_type* peer_box_mins;
    			scalar_type* peer_box_maxs;

    			SearchBoxFunctorUnconstrained(scalar_type* peer_box_mins_, scalar_type* peer_box_maxs_, gid_view_type gids_found_,
    					host_view_type pts_view_, const_gid_view_type gids_,
    					count_type count_)
    					: gids_found(gids_found_),
    					  pts_view(pts_view_),gids(gids_),count(count_),peer_box_mins(peer_box_mins_),peer_box_maxs(peer_box_maxs_){}

    			void operator()(const int j) const {
    				z2_box_type peer_processor_box(0, 3, &peer_box_mins[0], &peer_box_maxs[0]);
    				scalar_type coordinates[3] = {pts_view(j,0), pts_view(j,1), pts_view(j,2)};
    				if (peer_processor_box.pointInBox(3, coordinates)) {
    					//Kokkos::atomic_add(&count(), 1);
    					if (gids_found(j)<1) {
							Kokkos::atomic_fetch_add(&count(), 1);
							gids_found(j) = 1;
    					}
    				}
    			}
    		};

    		struct SearchBoxFunctorConstrained {
    			gid_view_type gids_found;
    			host_view_type pts_view;
    			const_gid_view_type gids;
    			count_type count;
    			scalar_type* peer_box_mins;
    			scalar_type* peer_box_maxs;
    			host_view_local_index_type flags;
    			const std::vector<int>& flags_to_transfer;

    			SearchBoxFunctorConstrained(scalar_type* peer_box_mins_, scalar_type* peer_box_maxs_, gid_view_type gids_found_,
    					host_view_type pts_view_, const_gid_view_type gids_, count_type count_,
						host_view_local_index_type flags_, const std::vector<int>& flags_to_transfer_)
    					: gids_found(gids_found_), pts_view(pts_view_), gids(gids_), count(count_)
    			, peer_box_mins(peer_box_mins_), peer_box_maxs(peer_box_maxs_), flags(flags_), flags_to_transfer(flags_to_transfer_) {}

    			void operator()(const int j) const {
    				bool this_index_used = false;
    				for (size_t i=0; i<flags_to_transfer.size(); ++i) {
    					if (flags(j,0) == flags_to_transfer[i]) this_index_used = true;
    					break;
    				}

    				if (this_index_used) {
						z2_box_type peer_processor_box(0, 3, &peer_box_mins[0], &peer_box_maxs[0]);
						scalar_type coordinates[3] = {pts_view(j,0), pts_view(j,1), pts_view(j,2)};
						if (peer_processor_box.pointInBox(3, coordinates)) {
							//Kokkos::atomic_add(&count(), 1);
							if (gids_found(j)<1) {
								Kokkos::atomic_fetch_add(&count(), 1);
								gids_found(j) = 1;
							}
						}
    				}
    			}
    		};

//    		std::cout << global_comm->getRank() << " min: " << peer_box_mins[0] << " " << peer_box_mins[1] << " " << peer_box_mins[2] << std::endl;
//    		std::cout << global_comm->getRank() << " max: " << peer_box_maxs[0] << " " << peer_box_maxs[1] << " " << peer_box_maxs[2] << std::endl;

    		if (flags_for_transfer.size() > 0) {
    			Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,our_coords->nLocal()),
    				SearchBoxFunctorConstrained(&peer_box_mins[0], &peer_box_maxs[0],
    						local_to_global_id_found, ptsView, gids, count_gids_found, flags, flags_for_transfer));

    		} else {
    			Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,our_coords->nLocal()),
    				SearchBoxFunctorUnconstrained(&peer_box_mins[0], &peer_box_maxs[0],
    						local_to_global_id_found, ptsView, gids, count_gids_found));
    		}

//    		for (int m=0; m<our_coords->nLocal(); ++m) {
//    			if (global_comm->getRank()==2) {
//    				std::cout << m << " " << local_to_global_id_found(m) << " " << ptsView(m,0) << " " << ptsView(m,1) << " " << ptsView(m,2) << std::endl;
//    			}
//    		}
//
//    		if (_amLower) {
//    			Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,our_coords->nLocal()),
//    				SearchBoxFunctor(&peer_box_mins[0], &peer_box_maxs[0],
//    						local_to_global_id_found, ptsView, gids, count_gids_found));
//    		} else {
//				Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,our_coords->nLocal()),
//					SearchBoxFunctor(&peer_box_mins[0], &peer_box_maxs[0],
//							local_to_global_id_found, ptsView, gids, count_gids_found));
//    		}

			struct VectorMergeFunctor {
				gid_view_type gids_to_send;
				const_gid_view_type gids;
				gid_view_type gids_found;
				count_type count;

				VectorMergeFunctor(gid_view_type gids_to_send_, const_gid_view_type gids_,
						gid_view_type gids_found_)
						: gids_to_send(gids_to_send_), gids(gids_), gids_found(gids_found_),
						  count(count_type("")){}

				void operator()(const int j) const {
					if (gids_found(j)==1) {
						const int idx = Kokkos::atomic_fetch_add(&count(), 1);
						gids_to_send(idx) = gids(j);
						gids_found(j) = 2;
					}
				}
			};
			gid_view_type gids_to_send("gids to send", count_gids_found());
//			int my_count = 0;
//			for (int j=0; j<our_coords->nLocal(); ++j) {
//				if (local_to_global_id_found(j)>-1) {
//					gids_to_send(my_count) = local_to_global_id_found(j);
//					my_count++;
//				}
//			}
			Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,our_coords->nLocal()), VectorMergeFunctor(gids_to_send, gids, local_to_global_id_found));

			Teuchos::RCP<host_gid_view_type> h_gids_to_send = Teuchos::rcp(new host_gid_view_type);
			*h_gids_to_send = Kokkos::create_mirror_view(gids_to_send);
			Kokkos::deep_copy(*h_gids_to_send, gids_to_send);

			coords_to_send[i] = h_gids_to_send;

//			for (int j=0; j<coords_to_send[i]->size(); j++) {
//				if (!_amLower && global_comm->getRank()==3 && peer_processors_i_overlap[i]==0)
//					std::cout << global_comm->getRank() << " sends: " << (*coords_to_send[i])(j) << " to " <<  peer_processors_i_overlap[i] << std::endl;
//			}
//			if (global_comm->getRank()==3) std::cout << global_comm->getRank() << " sends: " << coords_to_send[i]->size() << std::endl;
//			std::cout << global_comm->getRank() << " sends: " << coords_to_send[i]->size() << std::endl;
    }

	std::vector<local_index_type> peer_processor_i_destined(peer_processors_i_overlap.size());
	for (size_t i=0; i<coords_to_send.size(); i++) {
		peer_processor_i_destined[i] = coords_to_send[i]->size();
	}


	// collect the global indices of coordinates to be sent to each processor on that processor
	std::vector<global_index_type> indices_i_need;
	{
		// put out a receive for all processors of a single integer
		// go through my neighbor list
		// non-blocking receive
		Teuchos::Array<Teuchos::RCP<Teuchos::CommRequest<local_index_type> > > requests(2*peer_processors_i_overlap.size());

		Teuchos::ArrayRCP<local_index_type> recv_counts(peer_processors_i_overlap.size());
		for (size_t i = 0; i<peer_processors_i_overlap.size(); i++) {
			Teuchos::ArrayRCP<local_index_type> single_recv_value(&recv_counts[i], 0, 1, false);
			// + peer_root offsets to get the processor number for global communicator
			requests[i] = Teuchos::ireceive<local_index_type,local_index_type>(*global_comm, single_recv_value, peer_processors_i_overlap[i] + peer_root);
		}

		// put out a send for all processors to send to of a single integer
		// go through my neighbor list
		// non-blocking sends
		Teuchos::ArrayRCP<local_index_type> send_counts(&peer_processor_i_destined[0], 0, peer_processor_i_destined.size(), false);
		for (size_t i = 0; i<peer_processor_i_destined.size(); i++) {
			Teuchos::ArrayRCP<local_index_type> single_send_value(&send_counts[i], 0, 1, false);
			requests[peer_processor_i_destined.size()+i] = Teuchos::isend<local_index_type,local_index_type>(*global_comm, single_send_value, peer_processors_i_overlap[i] + peer_root);
		}
		Teuchos::waitAll<local_index_type>(*global_comm, requests);

		local_index_type sum=0;
		std::vector<local_index_type> sending_processor_offsets(peer_processors_i_overlap.size());
		for (size_t i=0; i<peer_processors_i_overlap.size(); i++) {
			sending_processor_offsets[i] += sum;
			sum += recv_counts[i];
		}
		indices_i_need.resize(sum);


		// there is no implied connect between that processors that because on has nothing to receive, then they have nothing to send
		// therefore, even 0 must be sent

		// put out a receive for all processors of a N integers
		// go through my neighbor list
		Teuchos::ArrayRCP<global_index_type> indices_received(&indices_i_need[0], 0, sum, false);
		for (size_t i = 0; i<peer_processors_i_overlap.size(); i++) {
			Teuchos::ArrayRCP<global_index_type> single_indices_recv;
			if (recv_counts[i] > 0) {
				single_indices_recv = Teuchos::ArrayRCP<global_index_type>(&indices_received[sending_processor_offsets[i]], 0, recv_counts[i], false);
				requests[i] = Teuchos::ireceive<local_index_type,global_index_type>(*global_comm, single_indices_recv, peer_processors_i_overlap[i] + peer_root);
			} else {
				single_indices_recv = Teuchos::ArrayRCP<global_index_type>(NULL, 0, 0, false);
				requests[i] = Teuchos::ireceive<local_index_type,global_index_type>(*global_comm, single_indices_recv, peer_processors_i_overlap[i] + peer_root);
			}
		}

		// put out a send for all processors of N integers
		// go through my neighbor list
		for (size_t i = 0; i<peer_processors_i_overlap.size(); i++) {
			Teuchos::ArrayRCP<global_index_type> indices_to_send;
                        if (coords_to_send[i]->size() > 0) {
				indices_to_send = Teuchos::ArrayRCP<global_index_type>(&(*coords_to_send[i])(0), 0, coords_to_send[i]->extent(0), false);
			} else {
				indices_to_send = Teuchos::ArrayRCP<global_index_type>(NULL, 0, 0, false);
			}
			requests[peer_processors_i_overlap.size()+i] = Teuchos::isend<local_index_type,global_index_type>(*global_comm, indices_to_send, peer_processors_i_overlap[i] + peer_root);
		}
		Teuchos::waitAll<local_index_type>(*global_comm, requests);
	}

	// now to figure out offsets
	// each set of processes (lower & upper), both have their own global ids that are now stored in indices_i_need
	// we need to set up the importers and exporters such that we can move data
	std::vector<global_index_type> indices_i_have(gids.extent(0));
	for (size_t i=0; i<gids.extent(0); i++) {
		indices_i_have[i] = gids(i);
//		std::cout << global_comm->getRank() << " have " << indices_i_have.size() << " " << gids(i) << std::endl;
	}


//	std::cout << global_comm->getRank() << " has " << indices_i_need.size() << std::endl;
//	for (local_index_type i=0; i<indices_i_need.size(); i++) {
//		std::cout << global_comm->getRank() << " " << indices_i_need[i] << std::endl;
//	}

	if (_amLower) {
		// elements I need from upper
		_lower_map_for_lower_data = Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
																 &indices_i_have[0],
																 gids.extent(0),
																 0,
																 global_comm));

		// elements I provide for lower
		_lower_map_for_upper_data = Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
																 &indices_i_need[0],
																 indices_i_need.size(),
																 0,
																 global_comm));

		// null map
		_upper_map_for_lower_data = Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
																nullptr,
																0,
																0,
																global_comm));

		// null map
		_upper_map_for_upper_data = Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
																nullptr,
																0,
																0,
																global_comm));

	} else {
		// null map
		_lower_map_for_lower_data = Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
																nullptr,
																0,
																0,
																global_comm));

		// null map
		_lower_map_for_upper_data = Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
																nullptr,
																0,
																0,
																global_comm));

		// elements I need from upper
		_upper_map_for_lower_data = Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
																&indices_i_need[0],
																indices_i_need.size(),
																0,
																global_comm));

		// elements I provide for lower
		_upper_map_for_upper_data = Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
																&indices_i_have[0],
																gids.extent(0),
																0,
																global_comm));

	}

	// imports data from the upper processes to the lower processes
	_lower_importer = Teuchos::rcp(new importer_type(_upper_map_for_lower_data, _lower_map_for_lower_data));

	// imports data from the lower processes to the upper processes
	_upper_importer = Teuchos::rcp(new importer_type(_lower_map_for_upper_data, _upper_map_for_upper_data));


//	// TEMPORARY for testing
//	Teuchos::RCP<mvec_type> lower_processors_view_of_lower_processors_data;
//	Teuchos::RCP<mvec_type> lower_processors_view_of_upper_processors_data;
//	Teuchos::RCP<mvec_type> upper_processors_view_of_lower_processors_data;
//	Teuchos::RCP<mvec_type> upper_processors_view_of_upper_processors_data;
//	lower_processors_view_of_lower_processors_data = Teuchos::rcp(new mvec_type(_lower_map_for_lower_data, 1 /*_ndim*/, true /* set to zero*/ ));
//	lower_processors_view_of_upper_processors_data = Teuchos::rcp(new mvec_type(_lower_map_for_upper_data, 1 /*_ndim*/, true /* set to zero*/ ));
//	upper_processors_view_of_lower_processors_data = Teuchos::rcp(new mvec_type(_upper_map_for_lower_data, 1 /*_ndim*/, true /* set to zero*/ ));
//	upper_processors_view_of_upper_processors_data = Teuchos::rcp(new mvec_type(_upper_map_for_upper_data, 1 /*_ndim*/, true /* set to zero*/ ));
//
//	host_view_type lower_pt_vals;
//	host_view_type upper_pt_vals;
//	if (_amLower) {
//		upper_pt_vals = lower_processors_view_of_upper_processors_data->getLocalView<host_view_type>();
//		for (int i=0; i<upper_pt_vals.extent(0); i++) {
//			upper_pt_vals(i,0) = 50;
//		}
//	} else {
//		lower_pt_vals = upper_processors_view_of_lower_processors_data->getLocalView<host_view_type>();
//		for (int i=0; i<lower_pt_vals.extent(0); i++) {
//			lower_pt_vals(i,0) = -1;
//		}
//	}
//
//	lower_processors_view_of_lower_processors_data->doImport(*upper_processors_view_of_lower_processors_data, *_lower_importer, Tpetra::CombineMode::REPLACE);
//	upper_processors_view_of_upper_processors_data->doImport(*lower_processors_view_of_upper_processors_data, *_upper_importer, Tpetra::CombineMode::REPLACE);
//
//	if (_amLower) {
//		lower_pt_vals = lower_processors_view_of_lower_processors_data->getLocalView<host_view_type>();
//		for (int i=0; i<lower_pt_vals.extent(0); i++) {
//			std::cout << _amLower << " " << global_comm->getRank() << " " << gids(i) << " " << lower_pt_vals(i,0) << std::endl;
//		}
//	} else {
//		upper_pt_vals = upper_processors_view_of_upper_processors_data->getLocalView<host_view_type>();
//		for (int i=0; i<upper_pt_vals.extent(0); i++) {
//			std::cout << _amLower << " " << global_comm->getRank() << " " << gids(i) << " " << upper_pt_vals(i,0) << std::endl;
//		}
//	}
	RemoteDataMapConstructionTime->stop();

}

void RemoteDataManager::putRemoteCoordinatesInParticleSet(particles_type* particles_to_overwrite, const bool use_physical_coords) {

	RemoteDataCoordinatesTime = Teuchos::TimeMonitor::getNewCounter ("Remote Data Coordinates Time");
	RemoteDataCoordinatesTime->start();

	// coordinates that are used must be the same as those used for processor boxes

	// initialize the particle set to have the correct number of coordinates on each processor
	if (_amLower) {
		particles_to_overwrite->resize(_lower_map_for_upper_data->getNodeNumElements(),true);
	} else {
		particles_to_overwrite->resize(_upper_map_for_lower_data->getNodeNumElements(),true);
	}

	Teuchos::RCP<mvec_type>	lower_processors_view_of_lower_processors_data = Teuchos::rcp(new mvec_type(_lower_map_for_lower_data, _ndim, true /* set to zero*/ ));
	Teuchos::RCP<mvec_type>	lower_processors_view_of_upper_processors_data = Teuchos::rcp(new mvec_type(_lower_map_for_upper_data, _ndim, true /* set to zero*/ ));
	Teuchos::RCP<mvec_type>	upper_processors_view_of_lower_processors_data = Teuchos::rcp(new mvec_type(_upper_map_for_lower_data, _ndim, true /* set to zero*/ ));
	Teuchos::RCP<mvec_type>	upper_processors_view_of_upper_processors_data = Teuchos::rcp(new mvec_type(_upper_map_for_upper_data, _ndim, true /* set to zero*/ ));

	host_view_type original_pt_vals = _our_coords->getPts(false /*halo*/, use_physical_coords)->getLocalView<host_view_type>();
	host_view_type duplicated_pt_vals;
	if (_amLower) {
		duplicated_pt_vals = lower_processors_view_of_lower_processors_data->getLocalView<host_view_type>();
		for (size_t i=0; i<duplicated_pt_vals.extent(0); i++) {
			for (local_index_type j=0; j<_ndim; j++) {
				duplicated_pt_vals(i,j) = original_pt_vals(i,j);
			}
		}
	} else {
		duplicated_pt_vals = upper_processors_view_of_upper_processors_data->getLocalView<host_view_type>();
		for (size_t i=0; i<duplicated_pt_vals.extent(0); i++) {
			for (local_index_type j=0; j<_ndim; j++) {
				duplicated_pt_vals(i,j) = original_pt_vals(i,j);
			}
		}
	}

	// swap coordinates
	upper_processors_view_of_lower_processors_data->doExport(*lower_processors_view_of_lower_processors_data, *_lower_importer, Tpetra::CombineMode::REPLACE);
	lower_processors_view_of_upper_processors_data->doExport(*upper_processors_view_of_upper_processors_data, *_upper_importer, Tpetra::CombineMode::REPLACE);


	coords_type* coords_of_particles_to_overwrite = particles_to_overwrite->getCoords();
	// now it is time to move this data from the vector used for transfer to the vector for in the particle set
	host_view_type coordinates_to_move_into_particles;
	if (_amLower) {
		coordinates_to_move_into_particles = lower_processors_view_of_upper_processors_data->getLocalView<host_view_type>();
	} else {
		coordinates_to_move_into_particles = upper_processors_view_of_lower_processors_data->getLocalView<host_view_type>();
	}

	for (size_t i=0; i<coordinates_to_move_into_particles.extent(0); i++) {
		coords_of_particles_to_overwrite->replaceLocalCoords(i,
															coordinates_to_move_into_particles(i,0),
															coordinates_to_move_into_particles(i,1),
															coordinates_to_move_into_particles(i,2));
	}

//	// end of this function
//
//	// beginning of diagnostic
//
//	// remap, essentially
//
//	particles_to_overwrite->getFieldManager()->createField(1,"processor","unit");
//
//
//
//	Compadre::ConstantEachDimension constant_choice(particles_to_overwrite->getCoordsConst()->getComm()->getRank(),0,0);
//	particles_to_overwrite->getFieldManager()->getFieldByName("processor")->
//						localInitFromVectorFunction(&constant_choice);
//
//
//	host_view_type rank_data_of_who_received_to_send = particles_to_overwrite->getFieldManager()->getFieldByName("processor")->getMultiVectorPtr()->getLocalView<host_view_type>();
//	// copy data into vector
//	host_view_type duplicated_field_vals;
//	if (_amLower) {
//		duplicated_field_vals = lower_processors_view_of_upper_processors_data->getLocalView<host_view_type>();
//		for (local_index_type i=0; i<duplicated_field_vals.extent(0); i++) {
//			for (local_index_type j=0; j<1; j++) {
//				duplicated_field_vals(i,j) = rank_data_of_who_received_to_send(i,j);
//			}
//		}
//	} else {
//		duplicated_field_vals = upper_processors_view_of_lower_processors_data->getLocalView<host_view_type>();
//		for (local_index_type i=0; i<duplicated_field_vals.extent(0); i++) {
//			for (local_index_type j=0; j<1; j++) {
//				duplicated_field_vals(i,j) = rank_data_of_who_received_to_send(i,j);
//			}
//		}
//	}
//
//
//
//
//
//	// swap data
//
//
//
//	lower_processors_view_of_lower_processors_data->doImport(*upper_processors_view_of_lower_processors_data, *_lower_importer, Tpetra::CombineMode::REPLACE);
//	upper_processors_view_of_upper_processors_data->doImport(*lower_processors_view_of_upper_processors_data, *_upper_importer, Tpetra::CombineMode::REPLACE);
//
//	// now we have data where we want it, how do we print it?
//	// temporarily make a particle set, create a field, and initialize values, then print
//	// initialize the particle set to have the correct number of coordinates on each processor
//	if (_amLower) {
//		particles_to_overwrite->resize(_lower_map_for_lower_data->getNodeNumElements(),true);
//	} else {
//		particles_to_overwrite->resize(_upper_map_for_upper_data->getNodeNumElements(),true);
//	}
//
//	// now it is time to move this data from the vector used for transfer to the vector for in the particle set
//	if (_amLower) {
//		coordinates_to_move_into_particles = _our_coords->getPts()->getLocalView<host_view_type>();
//	} else {
//		coordinates_to_move_into_particles = _our_coords->getPts()->getLocalView<host_view_type>();
//	}
//
//	for (local_index_type i=0; i<coordinates_to_move_into_particles.extent(0); i++) {
//		coords_of_particles_to_overwrite->replaceLocalCoords(i,
//															coordinates_to_move_into_particles(i,0),
//															coordinates_to_move_into_particles(i,1),
//															coordinates_to_move_into_particles(i,2));
//	}
//
//	rank_data_of_who_received_to_send = particles_to_overwrite->getFieldManager()->getFieldByName("processor")->getMultiVectorPtr()->getLocalView<host_view_type>();
//	// copy data into vector
//	if (_amLower) {
//		duplicated_field_vals = lower_processors_view_of_lower_processors_data->getLocalView<host_view_type>();
//		for (local_index_type i=0; i<duplicated_field_vals.extent(0); i++) {
//			for (local_index_type j=0; j<1; j++) {
//				rank_data_of_who_received_to_send(i,j) = duplicated_field_vals(i,j);
//			}
//		}
//	} else {
//		duplicated_field_vals = upper_processors_view_of_upper_processors_data->getLocalView<host_view_type>();
//		for (local_index_type i=0; i<duplicated_field_vals.extent(0); i++) {
//			for (local_index_type j=0; j<1; j++) {
//				rank_data_of_who_received_to_send(i,j) = duplicated_field_vals(i,j);
//			}
//		}
//	}

	RemoteDataCoordinatesTime->stop();

}

void RemoteDataManager::putRemoteWeightsInParticleSet(const particles_type* source_particles, particles_type* particles_to_overwrite, std::string weighting_field_name) {
	// Used in OBFET

	Teuchos::RCP<mvec_type>	lower_processors_view_of_lower_processors_data = Teuchos::rcp(new mvec_type(_lower_map_for_lower_data, 1, true /* set to zero*/ ));
	Teuchos::RCP<mvec_type>	lower_processors_view_of_upper_processors_data = Teuchos::rcp(new mvec_type(_lower_map_for_upper_data, 1, true /* set to zero*/ ));
	Teuchos::RCP<mvec_type>	upper_processors_view_of_lower_processors_data = Teuchos::rcp(new mvec_type(_upper_map_for_lower_data, 1, true /* set to zero*/ ));
	Teuchos::RCP<mvec_type>	upper_processors_view_of_upper_processors_data = Teuchos::rcp(new mvec_type(_upper_map_for_upper_data, 1, true /* set to zero*/ ));

	host_view_type original_weighting_vals = source_particles->getFieldManagerConst()->getFieldByName(weighting_field_name)->getMultiVectorPtrConst()->getLocalView<host_view_type>();
	host_view_type duplicated_weighting_vals;

	if (_amLower) {
		duplicated_weighting_vals = lower_processors_view_of_lower_processors_data->getLocalView<host_view_type>();
		for (size_t i=0; i<duplicated_weighting_vals.extent(0); i++) {
			duplicated_weighting_vals(i,0) = original_weighting_vals(i,0);
		}
	} else {
		duplicated_weighting_vals = upper_processors_view_of_upper_processors_data->getLocalView<host_view_type>();
		for (size_t i=0; i<duplicated_weighting_vals.extent(0); i++) {
			duplicated_weighting_vals(i,0) = original_weighting_vals(i,0);
		}
	}

	// swap weighting vals
	upper_processors_view_of_lower_processors_data->doExport(*lower_processors_view_of_lower_processors_data, *_lower_importer, Tpetra::CombineMode::REPLACE);
	lower_processors_view_of_upper_processors_data->doExport(*upper_processors_view_of_upper_processors_data, *_upper_importer, Tpetra::CombineMode::REPLACE);


	// now it is time to move this data from the vector used for transfer to the vector for in the particle set
	host_view_type weights_to_move_into_particles;
	if (_amLower) {
		weights_to_move_into_particles = lower_processors_view_of_upper_processors_data->getLocalView<host_view_type>();
	} else {
		weights_to_move_into_particles = upper_processors_view_of_lower_processors_data->getLocalView<host_view_type>();
	}

	// get access to the field
	host_view_type weights_inside_particles;
	// check if field is already registered (verify it is not)
	try {
		weights_inside_particles = particles_to_overwrite->getFieldManager()->getFieldByName(weighting_field_name)->getMultiVectorPtr()->getLocalView<host_view_type>();
	} catch (...) {
		// register it
		particles_to_overwrite->getFieldManager()->createField(1, weighting_field_name, "na");
		weights_inside_particles = particles_to_overwrite->getFieldManager()->getFieldByName(weighting_field_name)->getMultiVectorPtr()->getLocalView<host_view_type>();
	}

	for (size_t i=0; i<weights_to_move_into_particles.extent(0); i++) {
		weights_inside_particles(i,0) = weights_to_move_into_particles(i,0);
	}
    // no halo field updating is intentional
}

void RemoteDataManager::putExtraRemapDataInParticleSet(const particles_type* source_particles, particles_type* particles_to_overwrite, std::string extra_data_for_remap_field_name) {

    _extra_data_for_remap_field_name = extra_data_for_remap_field_name;

	host_view_type original_extra_data_vals = source_particles->getFieldManagerConst()->getFieldByName(extra_data_for_remap_field_name)->getMultiVectorPtrConst()->getLocalView<host_view_type>();
    local_index_type original_num_components = static_cast<local_index_type>(original_extra_data_vals.extent(1));
    local_index_type peer_num_components = 0;

    // need exchange of number of points
	// send and receive # of components from peer
    if (_amLower) {
    	if (_local_comm->getRank()==0) {
    		Teuchos::broadcast<local_index_type, local_index_type>(*_lower_root_plus_upper_all_comm, 0, 1, &original_num_components);
    	}
	} else {
    	Teuchos::broadcast<local_index_type, local_index_type>(*_lower_root_plus_upper_all_comm, 0, 1, &peer_num_components);
    }
    if (_amLower) {
    	Teuchos::broadcast<local_index_type, local_index_type>(*_upper_root_plus_lower_all_comm, 0, 1, &peer_num_components);
    } else {
    	if (_local_comm->getRank()==0) {
    		Teuchos::broadcast<local_index_type, local_index_type>(*_upper_root_plus_lower_all_comm, 0, 1, &original_num_components);
    	}
	}

	Teuchos::RCP<mvec_type>	lower_processors_view_of_lower_processors_data;
	Teuchos::RCP<mvec_type>	upper_processors_view_of_upper_processors_data;
    if (_amLower) {
	    lower_processors_view_of_lower_processors_data = Teuchos::rcp(new mvec_type(_lower_map_for_lower_data, original_num_components, true /* set to zero*/ ));
	    upper_processors_view_of_upper_processors_data = Teuchos::rcp(new mvec_type(_upper_map_for_upper_data, peer_num_components, true /* set to zero*/ ));
    } else {
	    lower_processors_view_of_lower_processors_data = Teuchos::rcp(new mvec_type(_lower_map_for_lower_data, peer_num_components, true /* set to zero*/ ));
	    upper_processors_view_of_upper_processors_data = Teuchos::rcp(new mvec_type(_upper_map_for_upper_data, original_num_components, true /* set to zero*/ ));
    }

	host_view_type original_extra_vals = source_particles->getFieldManagerConst()->getFieldByName(extra_data_for_remap_field_name)->getMultiVectorPtrConst()->getLocalView<host_view_type>();
	host_view_type duplicated_extra_vals;

	if (_amLower) {
		duplicated_extra_vals = lower_processors_view_of_lower_processors_data->getLocalView<host_view_type>();
		for (size_t i=0; i<duplicated_extra_vals.extent(0); i++) {
	        for (size_t j=0; j<duplicated_extra_vals.extent(1); j++) {
			    duplicated_extra_vals(i,j) = original_extra_vals(i,j);
            }
		}
	} else {
		duplicated_extra_vals = upper_processors_view_of_upper_processors_data->getLocalView<host_view_type>();
		for (size_t i=0; i<duplicated_extra_vals.extent(0); i++) {
	        for (size_t j=0; j<duplicated_extra_vals.extent(1); j++) {
			    duplicated_extra_vals(i,j) = original_extra_vals(i,j);
            }
		}
	}

	Teuchos::RCP<mvec_type>	lower_processors_view_of_upper_processors_data;
	Teuchos::RCP<mvec_type>	upper_processors_view_of_lower_processors_data;
    if (_amLower) {
		lower_processors_view_of_upper_processors_data = Teuchos::rcp(new mvec_type(_lower_map_for_upper_data, peer_num_components, true /* set to zero*/ ));
	    upper_processors_view_of_lower_processors_data = Teuchos::rcp(new mvec_type(_upper_map_for_lower_data, original_num_components, true /* set to zero*/ ));
    } else {
		lower_processors_view_of_upper_processors_data = Teuchos::rcp(new mvec_type(_lower_map_for_upper_data, original_num_components, true /* set to zero*/ ));
	    upper_processors_view_of_lower_processors_data = Teuchos::rcp(new mvec_type(_upper_map_for_lower_data, peer_num_components, true /* set to zero*/ ));
    }

	// swap extra data for remap
	lower_processors_view_of_upper_processors_data->doExport(*upper_processors_view_of_upper_processors_data, *_upper_importer, Tpetra::CombineMode::REPLACE);
	upper_processors_view_of_lower_processors_data->doExport(*lower_processors_view_of_lower_processors_data, *_lower_importer, Tpetra::CombineMode::REPLACE);


	// now it is time to move this data from the vector used for transfer to the vector for in the particle set
	host_view_type extra_remap_data_to_move_into_particles;
	if (_amLower) {
		extra_remap_data_to_move_into_particles = lower_processors_view_of_upper_processors_data->getLocalView<host_view_type>();
	} else {
		extra_remap_data_to_move_into_particles = upper_processors_view_of_lower_processors_data->getLocalView<host_view_type>();
	}

	// get access to the field
	host_view_type extra_remap_data_inside_particles;
	try {
		extra_remap_data_inside_particles = particles_to_overwrite->getFieldManager()->getFieldByName(extra_data_for_remap_field_name)->getMultiVectorPtr()->getLocalView<host_view_type>();
	} catch (...) {
		// register it
		particles_to_overwrite->getFieldManager()->createField(peer_num_components, extra_data_for_remap_field_name, "na");
		extra_remap_data_inside_particles = particles_to_overwrite->getFieldManager()->getFieldByName(extra_data_for_remap_field_name)->getMultiVectorPtr()->getLocalView<host_view_type>();
	}

	for (size_t i=0; i<extra_remap_data_to_move_into_particles.extent(0); i++) {
	    for (size_t j=0; j<extra_remap_data_to_move_into_particles.extent(1); j++) {
		    extra_remap_data_inside_particles(i,j) = extra_remap_data_to_move_into_particles(i,j);
        }
	}
    // no halo field updating is intentional
}

void RemoteDataManager::remapData(std::vector<RemapObject> remap_vector,
		Teuchos::RCP<Teuchos::ParameterList> parameters,
		particles_type* source_particles,
		particles_type* particles_to_overwrite,
		double max_halo_size,
		bool use_physical_coords,
        bool reuse_remap_solution) {

	RemoteDataRemapTime = Teuchos::TimeMonitor::getNewCounter ("Remote Data Remap Time");
	RemoteDataRemapTime->start();

    _global_comm->barrier();

	//***************
	//
	// Establish # of fields and field names needed by peer program
	//
	//***************

	// remap_vector contains all source and target pairs for fields
	// source is the peer process, and target is what we will call the field upon receiving the data
	local_index_type my_num_fields_for_swap = remap_vector.size();
	local_index_type peer_num_fields_for_swap = 0;

	// make a vector of our source field name sizes
	std::vector<local_index_type> my_field_name_sizes(remap_vector.size());
	for (size_t i=0; i<remap_vector.size(); ++i) {
		my_field_name_sizes[i] = remap_vector[i].src_fieldname.size();
	}

	// send and receive # of fields from peer
    if (_amLower) {
    	if (_local_comm->getRank()==0) {
    		Teuchos::broadcast<local_index_type, local_index_type>(*_lower_root_plus_upper_all_comm, 0, 1, &my_num_fields_for_swap);
    	}
	} else {
    	Teuchos::broadcast<local_index_type, local_index_type>(*_lower_root_plus_upper_all_comm, 0, 1, &peer_num_fields_for_swap);
    }
    if (_amLower) {
    	Teuchos::broadcast<local_index_type, local_index_type>(*_upper_root_plus_lower_all_comm, 0, 1, &peer_num_fields_for_swap);
    } else {
    	if (_local_comm->getRank()==0) {
    		Teuchos::broadcast<local_index_type, local_index_type>(*_upper_root_plus_lower_all_comm, 0, 1, &my_num_fields_for_swap);
    	}
	}

    // send and receive size of each field name
    std::vector<local_index_type> peer_field_name_sizes(peer_num_fields_for_swap);
    if (_amLower) {
    	if (_local_comm->getRank()==0) {
    		Teuchos::broadcast<local_index_type, local_index_type>(*_lower_root_plus_upper_all_comm, 0, my_num_fields_for_swap, &my_field_name_sizes[0]);
    	}
	} else {
    	Teuchos::broadcast<local_index_type, local_index_type>(*_lower_root_plus_upper_all_comm, 0, peer_num_fields_for_swap, &peer_field_name_sizes[0]);
    }
    if (_amLower) {
    	Teuchos::broadcast<local_index_type, local_index_type>(*_upper_root_plus_lower_all_comm, 0, peer_num_fields_for_swap, &peer_field_name_sizes[0]);
    } else {
    	if (_local_comm->getRank()==0) {
    		Teuchos::broadcast<local_index_type, local_index_type>(*_upper_root_plus_lower_all_comm, 0, my_num_fields_for_swap, &my_field_name_sizes[0]);
    	}
	}

    // create placeholder for receiving peer field names
    std::vector<std::string> peer_field_names(peer_num_fields_for_swap);
    std::vector<int> peer_target_operation(peer_num_fields_for_swap,0);
    std::vector<int> peer_reconstruction_space(peer_num_fields_for_swap,0);

    auto num_bytes_sampling_functional = sizeof(SamplingFunctional);
    std::vector<SamplingFunctional> peer_polynomial_sampling_functional(peer_num_fields_for_swap, PointSample);
    std::vector<SamplingFunctional> peer_data_sampling_functional(peer_num_fields_for_swap, PointSample);

	for (local_index_type i=0; i<std::max((local_index_type)(peer_field_names.size()), my_num_fields_for_swap); ++i) {

		if ((size_t)i < peer_field_names.size())
			peer_field_names[i] = std::string(peer_field_name_sizes[i], 'a'); // placeholder

		// send and receive each field name
		if (_amLower) {
			if (i < my_num_fields_for_swap && _local_comm->getRank()==0) {
				Teuchos::broadcast<local_index_type, char>(*_lower_root_plus_upper_all_comm, 0, my_field_name_sizes[i], &remap_vector[i].src_fieldname[0]);
				int target_operation = (int)(remap_vector[i]._target_operation);
				int reconstruction_space = (int)(remap_vector[i]._reconstruction_space);
                SamplingFunctional polynomial_sampling_functional = remap_vector[i]._polynomial_sampling_functional;
                SamplingFunctional data_sampling_functional = remap_vector[i]._data_sampling_functional;
				Teuchos::broadcast<local_index_type, int>(*_lower_root_plus_upper_all_comm, 0, 1, &target_operation);
				Teuchos::broadcast<local_index_type, int>(*_lower_root_plus_upper_all_comm, 0, 1, &reconstruction_space);


				Teuchos::broadcast<local_index_type, char>(*_lower_root_plus_upper_all_comm, 0, num_bytes_sampling_functional, 
                        reinterpret_cast<char*>(&polynomial_sampling_functional));
				Teuchos::broadcast<local_index_type, char>(*_lower_root_plus_upper_all_comm, 0, num_bytes_sampling_functional, 
                        reinterpret_cast<char*>(&data_sampling_functional));
			}
		} else {
			if ((size_t)i < peer_field_names.size()) {
				Teuchos::broadcast<local_index_type, char>(*_lower_root_plus_upper_all_comm, 0, peer_field_name_sizes[i], &peer_field_names[i][0]);
				Teuchos::broadcast<local_index_type, int>(*_lower_root_plus_upper_all_comm, 0, 1, &peer_target_operation[i]);
				Teuchos::broadcast<local_index_type, int>(*_lower_root_plus_upper_all_comm, 0, 1, &peer_reconstruction_space[i]);


				Teuchos::broadcast<local_index_type, char>(*_lower_root_plus_upper_all_comm, 0, num_bytes_sampling_functional, 
                        reinterpret_cast<char*>(&peer_polynomial_sampling_functional[i]));
				Teuchos::broadcast<local_index_type, char>(*_lower_root_plus_upper_all_comm, 0, num_bytes_sampling_functional, 
                        reinterpret_cast<char*>(&peer_data_sampling_functional[i]));
			}
		}
		if (_amLower) {
			if ((size_t)i < peer_field_names.size()) {
				Teuchos::broadcast<local_index_type, char>(*_upper_root_plus_lower_all_comm, 0, peer_field_name_sizes[i], &peer_field_names[i][0]);
				Teuchos::broadcast<local_index_type, int>(*_upper_root_plus_lower_all_comm, 0, 1, &peer_target_operation[i]);
				Teuchos::broadcast<local_index_type, int>(*_upper_root_plus_lower_all_comm, 0, 1, &peer_reconstruction_space[i]);

				Teuchos::broadcast<local_index_type, char>(*_upper_root_plus_lower_all_comm, 0, num_bytes_sampling_functional, 
                        reinterpret_cast<char*>(&peer_polynomial_sampling_functional[i]));
				Teuchos::broadcast<local_index_type, char>(*_upper_root_plus_lower_all_comm, 0, num_bytes_sampling_functional, 
                        reinterpret_cast<char*>(&peer_data_sampling_functional[i]));
			}
		} else {
			if (i < my_num_fields_for_swap && _local_comm->getRank()==0) {
				Teuchos::broadcast<local_index_type, char>(*_upper_root_plus_lower_all_comm, 0, my_field_name_sizes[i], &remap_vector[i].src_fieldname[0]);
				int target_operation = (int)(remap_vector[i]._target_operation);
				int reconstruction_space = (int)(remap_vector[i]._reconstruction_space);
                SamplingFunctional polynomial_sampling_functional = remap_vector[i]._polynomial_sampling_functional;
                SamplingFunctional data_sampling_functional = remap_vector[i]._data_sampling_functional;
				Teuchos::broadcast<local_index_type, int>(*_upper_root_plus_lower_all_comm, 0, 1, &target_operation);
				Teuchos::broadcast<local_index_type, int>(*_upper_root_plus_lower_all_comm, 0, 1, &reconstruction_space);

				Teuchos::broadcast<local_index_type, char>(*_upper_root_plus_lower_all_comm, 0, num_bytes_sampling_functional, 
                        reinterpret_cast<char*>(&polynomial_sampling_functional));
				Teuchos::broadcast<local_index_type, char>(*_upper_root_plus_lower_all_comm, 0, num_bytes_sampling_functional, 
                        reinterpret_cast<char*>(&data_sampling_functional));
			}
		}
	}



    std::vector<std::string> requested_named_source_extra_data_field;
    { 
        // only for when source extra data needs set, this is how one code tells the other the field it needs 
        // to set as the source extra data

        // extra data for sources (sampling functionals), this is the name of the source field that the peer should have
	    std::vector<local_index_type> named_source_extra_data_field_sizes(remap_vector.size());
	    for (size_t i=0; i<remap_vector.size(); ++i) {
	    	named_source_extra_data_field_sizes[i] = remap_vector[i]._source_extra_data_fieldname.size();
	    }

        // send and receive size of each source extra data field requested by peer
        std::vector<local_index_type> requested_named_source_extra_data_field_sizes(peer_num_fields_for_swap);
        if (_amLower) {
        	if (_local_comm->getRank()==0) {
        		Teuchos::broadcast<local_index_type, local_index_type>(*_lower_root_plus_upper_all_comm, 0, my_num_fields_for_swap, &named_source_extra_data_field_sizes[0]);
        	}
	    } else {
        	Teuchos::broadcast<local_index_type, local_index_type>(*_lower_root_plus_upper_all_comm, 0, peer_num_fields_for_swap, &requested_named_source_extra_data_field_sizes[0]);
        }
        if (_amLower) {
        	Teuchos::broadcast<local_index_type, local_index_type>(*_upper_root_plus_lower_all_comm, 0, peer_num_fields_for_swap, &requested_named_source_extra_data_field_sizes[0]);
        } else {
        	if (_local_comm->getRank()==0) {
        		Teuchos::broadcast<local_index_type, local_index_type>(*_upper_root_plus_lower_all_comm, 0, my_num_fields_for_swap, &named_source_extra_data_field_sizes[0]);
        	}
	    }

        // create placeholder for receiving requested source extra data field names
        requested_named_source_extra_data_field.resize(peer_num_fields_for_swap);
	    for (local_index_type i=0; i<std::max((local_index_type)(peer_field_names.size()), my_num_fields_for_swap); ++i) {

	    	if ((size_t)i < peer_field_names.size())
	    		requested_named_source_extra_data_field[i] = std::string(requested_named_source_extra_data_field_sizes[i], 'a'); // placeholder

	    	// send and receive each field name
	    	if (_amLower) {
	    		if (i < my_num_fields_for_swap && _local_comm->getRank()==0) {
	    			Teuchos::broadcast<local_index_type, char>(*_lower_root_plus_upper_all_comm, 0, named_source_extra_data_field_sizes[i], &remap_vector[i]._source_extra_data_fieldname[0]);
	    		}
	    	} else {
	    		if ((size_t)i < peer_field_names.size()) {
	    			Teuchos::broadcast<local_index_type, char>(*_lower_root_plus_upper_all_comm, 0, requested_named_source_extra_data_field_sizes[i], &requested_named_source_extra_data_field[i][0]);
	    		}
	    	}
	    	if (_amLower) {
	    		if ((size_t)i < peer_field_names.size()) {
	    			Teuchos::broadcast<local_index_type, char>(*_upper_root_plus_lower_all_comm, 0, requested_named_source_extra_data_field_sizes[i], &requested_named_source_extra_data_field[i][0]);
	    		}
	    	} else {
	    		if (i < my_num_fields_for_swap && _local_comm->getRank()==0) {
	    			Teuchos::broadcast<local_index_type, char>(*_upper_root_plus_lower_all_comm, 0, named_source_extra_data_field_sizes[i], &remap_vector[i]._source_extra_data_fieldname[0]);
	    		}
	    	}
	    }
    }

	// exchange information about using Optimization in reconstruction
    std::vector<int> peer_optimization_algorithm(peer_num_fields_for_swap,0);
    std::vector<int> peer_single_linear_bound_constraint(peer_num_fields_for_swap,0);
    std::vector<int> peer_bounds_preservation(peer_num_fields_for_swap,0);
    std::vector<double> peer_global_lower_bound(peer_num_fields_for_swap,0);
    std::vector<double> peer_global_upper_bound(peer_num_fields_for_swap,0);
	for (local_index_type i=0; i<std::max((local_index_type)(peer_optimization_algorithm.size()), my_num_fields_for_swap); ++i) {

		// send and receive whether to use obfet 0-no, 1-yes
		if (_amLower) {
			if (i < my_num_fields_for_swap && _local_comm->getRank()==0) {
				int optimization_algorithm = (int)(remap_vector[i]._optimization_object._optimization_algorithm);
				int single_linear_bound_constraint = (int)(remap_vector[i]._optimization_object._single_linear_bound_constraint);
				int bounds_preservation = (int)(remap_vector[i]._optimization_object._bounds_preservation);
				double global_lower_bound = (double)(remap_vector[i]._optimization_object._global_lower_bound);
				double global_upper_bound = (double)(remap_vector[i]._optimization_object._global_upper_bound);
				Teuchos::broadcast<local_index_type, int>(*_lower_root_plus_upper_all_comm, 0, 1, &optimization_algorithm);
				Teuchos::broadcast<local_index_type, int>(*_lower_root_plus_upper_all_comm, 0, 1, &single_linear_bound_constraint);
				Teuchos::broadcast<local_index_type, int>(*_lower_root_plus_upper_all_comm, 0, 1, &bounds_preservation);
				Teuchos::broadcast<local_index_type, double>(*_lower_root_plus_upper_all_comm, 0, 1, &global_lower_bound);
				Teuchos::broadcast<local_index_type, double>(*_lower_root_plus_upper_all_comm, 0, 1, &global_upper_bound);
			}
		} else {
			if ((size_t)i < peer_field_names.size()) {
				Teuchos::broadcast<local_index_type, int>(*_lower_root_plus_upper_all_comm, 0, 1, &peer_optimization_algorithm[i]);
				Teuchos::broadcast<local_index_type, int>(*_lower_root_plus_upper_all_comm, 0, 1, &peer_single_linear_bound_constraint[i]);
				Teuchos::broadcast<local_index_type, int>(*_lower_root_plus_upper_all_comm, 0, 1, &peer_bounds_preservation[i]);
				Teuchos::broadcast<local_index_type, double>(*_lower_root_plus_upper_all_comm, 0, 1, &peer_global_lower_bound[i]);
				Teuchos::broadcast<local_index_type, double>(*_lower_root_plus_upper_all_comm, 0, 1, &peer_global_upper_bound[i]);
			}
		}
		if (_amLower) {
			if ((size_t)i < peer_field_names.size()) {
				Teuchos::broadcast<local_index_type, int>(*_upper_root_plus_lower_all_comm, 0, 1, &peer_optimization_algorithm[i]);
				Teuchos::broadcast<local_index_type, int>(*_upper_root_plus_lower_all_comm, 0, 1, &peer_single_linear_bound_constraint[i]);
				Teuchos::broadcast<local_index_type, int>(*_upper_root_plus_lower_all_comm, 0, 1, &peer_bounds_preservation[i]);
				Teuchos::broadcast<local_index_type, double>(*_upper_root_plus_lower_all_comm, 0, 1, &peer_global_lower_bound[i]);
				Teuchos::broadcast<local_index_type, double>(*_upper_root_plus_lower_all_comm, 0, 1, &peer_global_upper_bound[i]);
			}
		} else {
			if (i < my_num_fields_for_swap && _local_comm->getRank()==0) {
				//int optimization_algorithm = (int)(remap_vector[i]._optimization_object._optimization_algorithm);
				int optimization_algorithm = (int)(remap_vector[i]._optimization_object._optimization_algorithm);
				int single_linear_bound_constraint = (int)(remap_vector[i]._optimization_object._single_linear_bound_constraint);
				int bounds_preservation = (int)(remap_vector[i]._optimization_object._bounds_preservation);
				double global_lower_bound = (double)(remap_vector[i]._optimization_object._global_lower_bound);
				double global_upper_bound = (double)(remap_vector[i]._optimization_object._global_upper_bound);
				//Teuchos::broadcast<local_index_type, int>(*_upper_root_plus_lower_all_comm, 0, 1, &optimization_algorithm);
				Teuchos::broadcast<local_index_type, int>(*_upper_root_plus_lower_all_comm, 0, 1, &optimization_algorithm);
				Teuchos::broadcast<local_index_type, int>(*_upper_root_plus_lower_all_comm, 0, 1, &single_linear_bound_constraint);
				Teuchos::broadcast<local_index_type, int>(*_upper_root_plus_lower_all_comm, 0, 1, &bounds_preservation);
				Teuchos::broadcast<local_index_type, double>(*_upper_root_plus_lower_all_comm, 0, 1, &global_lower_bound);
				Teuchos::broadcast<local_index_type, double>(*_upper_root_plus_lower_all_comm, 0, 1, &global_upper_bound);
			}
		}
	}

#ifdef COMPADREHARNESS_DEBUG
	// diagnostic
    std::cout << "num peer fields: " << peer_num_fields_for_swap << std::endl;
    for (int i=0; i<peer_num_fields_for_swap; ++i) {
    	std::cout << "size for " << i << " is " << peer_field_name_sizes[i] << " and name is: " << peer_field_names[i] << std::endl;
    }
#endif

	//***************
	//
	// Construct a remap object and reconstruct data onto temporary particles
	//
	//***************
    
	// create a remap manager
    if (peer_num_fields_for_swap > 0) {
        Teuchos::RCP<Compadre::RemapManager> rm;
        if (!reuse_remap_solution) {
            _rm = Teuchos::null;
        }
        if (_rm.is_null()) {
	        rm = Teuchos::rcp(new Compadre::RemapManager(parameters, source_particles, particles_to_overwrite, max_halo_size));
            if (reuse_remap_solution) {
                _rm = rm;
            }
            for (local_index_type i=0; i<peer_num_fields_for_swap; ++i) {
                auto ro = RemapObject(peer_field_names[i], peer_field_names[i], 
                        static_cast<TargetOperation>(peer_target_operation[i]), 
                        static_cast<ReconstructionSpace>(peer_reconstruction_space[i]), 
                        peer_polynomial_sampling_functional[i], peer_data_sampling_functional[i]);
                if (peer_optimization_algorithm[i] > 0) {
                    OptimizationObject optimization_object((OptimizationAlgorithm)peer_optimization_algorithm[i],(bool)peer_single_linear_bound_constraint[i],(bool)peer_bounds_preservation[i],peer_global_lower_bound[i],peer_global_upper_bound[i]);
            	    ro.setOptimizationObject(optimization_object);
                }
                if (_extra_data_for_remap_field_name != "") {
                    ro.setTargetExtraData(_extra_data_for_remap_field_name);
                }
                if (requested_named_source_extra_data_field[i]!="") {
                    ro.setSourceExtraData(requested_named_source_extra_data_field[i]);
                }
            	rm->add(ro);
            }
        } else {
            // reusing old remap solution
            rm = _rm;
        }
        rm->execute(true /* keep neighborhoods */, true /* keep GMLS */, reuse_remap_solution /* reuse neighborhoods */, reuse_remap_solution /* reuse GMLS */, use_physical_coords);
    }

    // send and receive size of each field name
    std::vector<local_index_type> each_peer_field_units_name_sizes(peer_num_fields_for_swap);
    std::vector<local_index_type> received_field_units_name_sizes(my_num_fields_for_swap);
    for (local_index_type i=0; i<peer_num_fields_for_swap; ++i) {
    	each_peer_field_units_name_sizes[i] = particles_to_overwrite->getFieldManager()->getFieldByName(peer_field_names[i])->getUnits().size();
    }
    if (_amLower) {
    	if (_local_comm->getRank()==0) {
    		Teuchos::broadcast<local_index_type, local_index_type>(*_lower_root_plus_upper_all_comm, 0, peer_num_fields_for_swap, &each_peer_field_units_name_sizes[0]);
    	}
	} else {
    	Teuchos::broadcast<local_index_type, local_index_type>(*_lower_root_plus_upper_all_comm, 0, my_num_fields_for_swap, &received_field_units_name_sizes[0]);
    }
    if (_amLower) {
    	Teuchos::broadcast<local_index_type, local_index_type>(*_upper_root_plus_lower_all_comm, 0, my_num_fields_for_swap, &received_field_units_name_sizes[0]);
    } else {
    	if (_local_comm->getRank()==0) {
    		Teuchos::broadcast<local_index_type, local_index_type>(*_upper_root_plus_lower_all_comm, 0, peer_num_fields_for_swap, &each_peer_field_units_name_sizes[0]);
    	}
	}

    // create placeholder for receiving peer field names
    std::vector<std::string> received_field_unit_names(my_num_fields_for_swap);
	for (local_index_type i=0; i<std::max((local_index_type)(peer_field_names.size()), my_num_fields_for_swap); ++i) {

		if (i < my_num_fields_for_swap)
			received_field_unit_names[i] = std::string(received_field_units_name_sizes[i], 'a'); // placeholder

		// send and receive each field name
		if (_amLower) {
			if (i < peer_num_fields_for_swap && _local_comm->getRank()==0) {
				Teuchos::broadcast<local_index_type, char>(*_lower_root_plus_upper_all_comm, 0, each_peer_field_units_name_sizes[i], &(particles_to_overwrite->getFieldManager()->getFieldByName(peer_field_names[i])->getUnits()[0]));
			}
		} else {
			if (i < my_num_fields_for_swap) {
				Teuchos::broadcast<local_index_type, char>(*_lower_root_plus_upper_all_comm, 0, received_field_units_name_sizes[i], &received_field_unit_names[i][0]);
			}
		}
		if (_amLower) {
			if (i < my_num_fields_for_swap) {
				Teuchos::broadcast<local_index_type, char>(*_upper_root_plus_lower_all_comm, 0, received_field_units_name_sizes[i], &received_field_unit_names[i][0]);
			}
		} else {
			if (i < peer_num_fields_for_swap && _local_comm->getRank()==0) {
				Teuchos::broadcast<local_index_type, char>(*_upper_root_plus_lower_all_comm, 0, each_peer_field_units_name_sizes[i], &(particles_to_overwrite->getFieldManager()->getFieldByName(peer_field_names[i])->getUnits()[0]));
			}
		}
	}

#ifdef COMPADREHARNESS_DEBUG
	// diagnostic
    for (int i=0; i<my_num_fields_for_swap; ++i) {
    	std::cout << "name is: " << remap_vector[i].trg_fieldname << ", unit is: " << received_field_unit_names[i] << std::endl;
    }
#endif


    // from reconstructed fields, we get the field dimensions and transfer these back to the peer program
    std::vector<local_index_type> each_peer_field_dim(peer_num_fields_for_swap);
    std::vector<local_index_type> received_field_dim(my_num_fields_for_swap);
    for (local_index_type i=0; i<peer_num_fields_for_swap; ++i) {
    	each_peer_field_dim[i] = particles_to_overwrite->getFieldManager()->getFieldByName(peer_field_names[i])->nDim();
    }

	// send and receive dimension # for fields (reverse of previous vector transfer)
    if (_amLower) {
    	if (_local_comm->getRank()==0) {
    		Teuchos::broadcast<local_index_type, local_index_type>(*_lower_root_plus_upper_all_comm, 0, peer_num_fields_for_swap, &each_peer_field_dim[0]);
    	}
	} else {
    	Teuchos::broadcast<local_index_type, local_index_type>(*_lower_root_plus_upper_all_comm, 0, my_num_fields_for_swap, &received_field_dim[0]);
    }
    if (_amLower) {
    	Teuchos::broadcast<local_index_type, local_index_type>(*_upper_root_plus_lower_all_comm, 0, my_num_fields_for_swap, &received_field_dim[0]);
    } else {
    	if (_local_comm->getRank()==0) {
    		Teuchos::broadcast<local_index_type, local_index_type>(*_upper_root_plus_lower_all_comm, 0, peer_num_fields_for_swap, &each_peer_field_dim[0]);
    	}
	}

#ifdef COMPADREHARNESS_DEBUG
	// diagnostic
    for (int i=0; i<my_num_fields_for_swap; ++i) {
    	std::cout << "name is: " << remap_vector[i].trg_fieldname << ", dim is: " << received_field_dim[i] << std::endl;
    }
#endif

    // get maximum size between total field dimensions from our program and peer program
    local_index_type my_total_field_dimensions = 0;
    for (local_index_type i=0; i<my_num_fields_for_swap; ++i) {
    	my_total_field_dimensions += received_field_dim[i];
    }

    local_index_type max_peer_my_total_field_dimensions = 0;
	Teuchos::Ptr<local_index_type> maxPtr(&max_peer_my_total_field_dimensions);
	Teuchos::reduceAll<local_index_type, local_index_type>(*_global_comm, Teuchos::REDUCE_MAX, my_total_field_dimensions, maxPtr);


	//*************
	//
	//  Construct Tpetra vector to hold information for swap
	//
	//*************

    // create a tpetra vector of a size that can hold all field data for both programs
	Teuchos::RCP<mvec_type>	lower_processors_view_of_lower_processors_data = Teuchos::rcp(new mvec_type(_lower_map_for_lower_data, max_peer_my_total_field_dimensions, true /* set to zero*/ ));
	Teuchos::RCP<mvec_type>	lower_processors_view_of_upper_processors_data = Teuchos::rcp(new mvec_type(_lower_map_for_upper_data, max_peer_my_total_field_dimensions, true /* set to zero*/ ));
	Teuchos::RCP<mvec_type>	upper_processors_view_of_lower_processors_data = Teuchos::rcp(new mvec_type(_upper_map_for_lower_data, max_peer_my_total_field_dimensions, true /* set to zero*/ ));
	Teuchos::RCP<mvec_type>	upper_processors_view_of_upper_processors_data = Teuchos::rcp(new mvec_type(_upper_map_for_upper_data, max_peer_my_total_field_dimensions, true /* set to zero*/ ));


    // load data into the tpetra vector for each field requested
	local_index_type offset = 0;
	for (local_index_type i=0; i<peer_num_fields_for_swap; ++i) {

		host_view_type this_peer_fields_values = particles_to_overwrite->getFieldManager()->getFieldByName(peer_field_names[i])->getMultiVectorPtr()->getLocalView<host_view_type>();

		// copy data into vector
		host_view_type field_vals_to_send;
		if (_amLower) {
			field_vals_to_send = lower_processors_view_of_upper_processors_data->getLocalView<host_view_type>();
		} else {
			field_vals_to_send = upper_processors_view_of_lower_processors_data->getLocalView<host_view_type>();
		}
		for (size_t j=0; j<field_vals_to_send.extent(0); j++) {
			for (local_index_type k=0; k<each_peer_field_dim[i]; k++) {
				field_vals_to_send(j,k+offset) = this_peer_fields_values(j,k);
			}
		}

		offset += each_peer_field_dim[i];
	}

    // swap the tpetra vector data
	lower_processors_view_of_lower_processors_data->doImport(*upper_processors_view_of_lower_processors_data, *_lower_importer, Tpetra::CombineMode::REPLACE);
	upper_processors_view_of_upper_processors_data->doImport(*lower_processors_view_of_upper_processors_data, *_upper_importer, Tpetra::CombineMode::REPLACE);

    // unload data into the particles
	offset = 0;
	for (local_index_type i=0; i<my_num_fields_for_swap; ++i) {

		// register the field if not already registered
		local_index_type target_field_num;
		try {
			target_field_num = source_particles->getFieldManagerConst()->getIDOfFieldFromName(remap_vector[i].trg_fieldname);

			// check that dimensions match if the field already exists
			TEUCHOS_TEST_FOR_EXCEPT_MSG(source_particles->getFieldManagerConst()->getFieldByID(target_field_num)->nDim() != received_field_dim[i],
					"Field dimensions do not match between existing field and peer program field requested.");
		} catch (std::logic_error & exception) {
			// field wasn't registered in target, so we are collecting its information from source particles
			source_particles->getFieldManager()->createField(
					received_field_dim[i],
					remap_vector[i].trg_fieldname,
					received_field_unit_names[i]
					);
			if (source_particles->getCoordsConst()->getComm()->getRank()==0) {
				std::cout << "The field \"" << remap_vector[i].trg_fieldname << "\" was requested to be constructed from its peer program. "
						<< "We've registered the field \"" << remap_vector[i].trg_fieldname << "\" in the target particle set with the "
						<< "units and dimensions of the field \""
						<< remap_vector[i].src_fieldname << "\" from the source particles.\n\n";
			}
			target_field_num = source_particles->getFieldManagerConst()->getIDOfFieldFromName(remap_vector[i].trg_fieldname);
		}


		host_view_type my_fields_values = source_particles->getFieldManager()->getFieldByID(target_field_num)->getMultiVectorPtr()->getLocalView<host_view_type>();

		// copy data from vector into particles
		host_view_type field_vals_sent_from_peer;
		if (_amLower) {
			field_vals_sent_from_peer = lower_processors_view_of_lower_processors_data->getLocalView<host_view_type>();
		} else {
			field_vals_sent_from_peer = upper_processors_view_of_upper_processors_data->getLocalView<host_view_type>();
		}
		for (size_t j=0; j<field_vals_sent_from_peer.extent(0); j++) {
			for (local_index_type k=0; k<received_field_dim[i]; k++) {
				my_fields_values(j,k) = field_vals_sent_from_peer(j,k+offset);
			}
		}

		offset += received_field_dim[i];

	}

	// refresh the halo information now that we've updated the fields with the data from the peer program
	source_particles->getFieldManager()->updateFieldsHaloData();

	RemoteDataRemapTime->stop();
    _global_comm->barrier();
}

}
