#include "Compadre_NeighborhoodT.hpp"

#include "Compadre_XyzVector.hpp"
#include "Compadre_CoordsT.hpp"
#include "Compadre_ParticlesT.hpp"

#ifdef COMPADREHARNESS_USE_OPENMP
#include <omp.h>
#endif

namespace Compadre {

typedef Compadre::XyzVector xyz_type;
typedef Compadre::CoordsT coords_type;
typedef Compadre::CoordsT particles_type;

NeighborhoodT::NeighborhoodT( const particles_type* source_particles, Teuchos::RCP<Teuchos::ParameterList> parameters, const particles_type* target_particles) :
		_source_coords(source_particles->getCoordsConst()), _source_particles(source_particles), _nDim(_source_coords->nDim()), _max_num_neighbors(0), _parameters(parameters)
{
	if (target_particles) {
		_target_particles = target_particles;
		_target_coords = target_particles->getCoordsConst();
	} else {
		_target_particles = source_particles;
		_target_coords = source_particles->getCoordsConst();
	}

	const bool setToZero = true;
	h_support_size = Teuchos::rcp(new mvec_type(_target_coords->getMapConst(), _nDim, setToZero));
	neighbor_list.resize(h_support_size->getLocalLength());

	NeighborSearchTime = Teuchos::TimeMonitor::getNewCounter ("Neighbor Search Time");
	MaxNeighborComputeTime = Teuchos::TimeMonitor::getNewCounter ("Max Neighbor Compute Time");
}

void NeighborhoodT::setHSupportSize(const local_index_type idx, const scalar_type val) {
	host_view_type hsupportView = h_support_size->getLocalView<host_view_type>();
	hsupportView(idx,0) = val;
}

double NeighborhoodT::getHSupportSize(const local_index_type idx) const {
	host_view_type hsupportView = h_support_size->getLocalView<host_view_type>();
	return hsupportView(idx,0);
}

void NeighborhoodT::setAllHSupportSizes(const scalar_type val) {
	host_view_type hsupportView = h_support_size->getLocalView<host_view_type>();
	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,hsupportView.dimension_0()), KOKKOS_LAMBDA(const int j) {
		hsupportView(j,0) = val;
	});
}

const local_index_type NeighborhoodT::getMaxNumNeighbors() const {
	TEUCHOS_TEST_FOR_EXCEPT_MSG(_max_num_neighbors==0, "getMaxNumNeighbors() called before computeMaxNumNeighbors() in Neighborhood.");
	return _max_num_neighbors;
}

local_index_type NeighborhoodT::computeMaxNumNeighbors() {
	MaxNeighborComputeTime->start();

	local_index_type max_neighbors = 0;
	local_index_type my_max_neighbors = 0;

	//#pragma parallel
	{
		//local_index_type my_max_neighbors = max_neighbors;
		//#pragma omp for nowait
		for (size_t i=0; i<neighbor_list.size(); i++) {
			my_max_neighbors = std::max(my_max_neighbors, static_cast<local_index_type>(neighbor_list[i].size()));
		}
		//#pragma omp critical
		{
			if(my_max_neighbors > max_neighbors) max_neighbors = my_max_neighbors;
		}
	}

	_max_num_neighbors = max_neighbors;
	TEUCHOS_TEST_FOR_EXCEPT_MSG(neighbor_list.size()>0 && _max_num_neighbors==0, "No neighbors in neighbor list. Need to call constructNeighborList()");
	MaxNeighborComputeTime->stop();
	return _max_num_neighbors;
}

const scalar_type NeighborhoodT::getMinimumHSupportSize() const {

	host_view_type hsupportView = h_support_size->getLocalView<host_view_type>();

	// get min from this vector of h_support_size for this processor
	scalar_type local_processor_min_radius = 1e+15;
	for (local_index_type i=0; i<_target_coords->nLocal(); i++) {
		local_processor_min_radius = hsupportView(i,0)<local_processor_min_radius ? hsupportView(i,0) : local_processor_min_radius;
	}

	// collect max radius from all processors
	scalar_type global_processor_min_radius;
	Teuchos::Ptr<scalar_type> global_processor_min_radius_ptr(&global_processor_min_radius);
	Teuchos::reduceAll<local_index_type, scalar_type>(*(_target_coords->getComm()), Teuchos::REDUCE_MIN,
			local_processor_min_radius, global_processor_min_radius_ptr);

	return global_processor_min_radius;
}

void NeighborhoodT::constructAllNeighborList(const scalar_type max_radius, const local_index_type desired_num_neighbors,
				const scalar_type initial_radius, const local_index_type maxLeaf, bool use_physical_coords) {

	host_view_type ptsView = _target_coords->getPts(false/*halo*/, use_physical_coords)->getLocalView<host_view_type>();
	host_view_type hsupportView = h_support_size->getLocalView<host_view_type>();

	const local_index_type comm_size = _source_coords->getComm()->getSize();

	NeighborSearchTime->start();

	bool dynamic = _parameters->get<Teuchos::ParameterList>("neighborhood").get<bool>("dynamic radius");
	bool spatially_varying = _parameters->get<Teuchos::ParameterList>("neighborhood").get<bool>("spatially varying radius");

	TEUCHOS_TEST_FOR_EXCEPT_MSG(comm_size>1 && _source_coords->getHaloSize()<initial_radius && initial_radius!=0, "Initial search radius larger than halo size.");

	// can support spatially varying, but must be set apriori or through a different interface, not yet implemented
	TEUCHOS_TEST_FOR_EXCEPT_MSG(!dynamic && spatially_varying, "Not implemented. Requires additional interface specifying varying radii.");

	if (initial_radius != 0.0) { // user wishes to initially set all radius for searches to this number
		TEUCHOS_TEST_FOR_EXCEPT_MSG(comm_size>1 && _source_coords->getHaloSize()<initial_radius, "Initial search radius larger than halo size.");
		this->setAllHSupportSizes(initial_radius);
	} else if (!spatially_varying) {
		// get a uniform support size for all neighborhoods
		// h supports are already set before calling this function

		// get max from this vector of h_support_size for this processor
		scalar_type local_processor_max_radius = 0;
		for (local_index_type i=0; i<_target_coords->nLocal(); i++) {
			local_processor_max_radius = hsupportView(i,0)>local_processor_max_radius ? hsupportView(i,0) : local_processor_max_radius;
		}

		// collect max radius from all processors
		scalar_type global_processor_max_radius;
		Teuchos::Ptr<scalar_type> global_processor_max_radius_ptr(&global_processor_max_radius);
		Teuchos::reduceAll<local_index_type, scalar_type>(*(_target_coords->getComm()), Teuchos::REDUCE_MAX,
				local_processor_max_radius, global_processor_max_radius_ptr);

		// set the search radius consistently across all processors
		this->setAllHSupportSizes(global_processor_max_radius);
		if (comm_size>1 && _source_coords->getHaloSize()<hsupportView(0,0)) {
			std::ostringstream msg;
			msg << "hsupport " << hsupportView(0,0) << " larger than halo size " << _source_coords->getHaloSize() << "." << std::endl;
			TEUCHOS_TEST_FOR_EXCEPT_MSG(_source_coords->getHaloSize()<hsupportView(0,0), msg.str());
		}
	} else {
		// since spatially varying, we have to check each
		if (comm_size>1) {
			for (local_index_type i=0, nt=_target_coords->nLocal(); i<nt; i++) {
				if (_source_coords->getHaloSize()<hsupportView(i,0)) {
					std::ostringstream msg;
					msg << "hsupport " << hsupportView(0,0) << " larger than halo size " << _source_coords->getHaloSize() << "." << std::endl;
					TEUCHOS_TEST_FOR_EXCEPT_MSG(_source_coords->getHaloSize()<hsupportView(i,0), msg.str());
				}
			}
		}
	}

	const scalar_type increase_multiplier = _parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("multiplier");
	const scalar_type cutoff_multiplier = _parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("cutoff multiplier");

	// Used to determined when all neighborhood searches are complete
	Kokkos::View<local_index_type*> search_complete("search complete", (const local_index_type)(_target_coords->nLocal()));

	// Used for NOT spatially varying neighborhood searches as a way of collecting the max cutoff
	Kokkos::View<scalar_type*> cutoffs_collection("cutoffs collection", (const local_index_type)(_target_coords->nLocal()));

	bool dynamically_updating_search_size = true; // set to true to ensure that it begins
	while (dynamically_updating_search_size) {

		dynamically_updating_search_size = dynamic; // if !dynamic, loop will not repeat
		if (desired_num_neighbors == 0) {
			dynamically_updating_search_size = false; // if user doesn't specify needed # of neighbors, than any amount with given initial_radius will do
		}

		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,_target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
//		for (local_index_type i=0; i<_target_coords->nLocal(); i++) {
			search_complete(i) = 0;
			cutoffs_collection(i) = 0;
			scalar_type cutoff_as_new_support = 0; // used to determine update size for the case where the cutoff size is larger than the search radius

			scalar_type coordinate[_nDim];
			for (local_index_type j=0; j<_target_coords->nDim(); j++) coordinate[j] = ptsView(i,j);

			this->constructSingleNeighborList(&coordinate[0], hsupportView(i,0), neighbor_list[i], use_physical_coords); // neighbor_list[i]

			if (desired_num_neighbors != 0) {
				// we prune the neighbor list once we have a sufficient number of neighbors
//				if (search_complete(i)==0) {
				if (neighbor_list[i].size() >= desired_num_neighbors) {
					// farthest coordinate from point in question is atleast "cutoff multiplier" times as far away as the radius
					// for the cutoff point of what is required
					/*
					 *  # needed = 3
					 *  5 neighbors found (n1,n2,n3,n4,n5)
					 *
					 *  Take distance(n3)*cutoff_multiplier and use this as a cutoff
					 *
					 *
					 */

					/*
					 *  start at neighbor n4 and work our way up to n5,
					 *      finding the first entry that is bigger than cutoff away from neighborhood center
					 *       AND that is not exactly equal to the n# before it
					 *
					 *  Note the neighbor we stop at, and add eps to its distance from neighborhood center and use this as ideal radius
					 */

					// if !dynamic, we may find enough neighbors but by creating this cutoff, we decide we do not
					// have enough neighbors and therefor consider it a failed search. This is ok, as it prevents
					// the case where we believe we have enough neighbors, but do not, because they are on the
					// boundary of support.
					const scalar_type cutoff_radius = cutoff_multiplier * neighbor_list[i][desired_num_neighbors-1].second;

					/*
					 * If cutoff_radius > hsupport, there could be particles between cutoff_radius and hsupport
					 * that we could miss by setting the support to cutoff_radius, so we increase hsupport and try again
					 *
					 * Here, * is the point from which all other points determine there distance: ri = distance(*,xi)
					 * | is the cutoff
					 * {} is the search radius used
					 * x3 in this case determines "|", but setting the support to "|" lops off from x4 onward
					 *
					 *    <-    x7      x6   | x4  {   x2 * x1 x3 }   |x5         ->
					 *
					 * which would not know about x4 and miss it, although the support would be from | to | suggesting that we didn't
					 * miss it (inconsistent).
					 *
					 * If cutoff_radius <= hsupport, we can draw the line for support at cutoff_radius, and not fear that
					 * we missed any parts between the desired_num_neighbors and the last one within the cutoff
					 *
					 *   <-    x7      x6   { x4  |   x2 * x1 x3 |   }x5         ->
					 *
					 */

					if (cutoff_radius <= hsupportView(i,0)) {
						// find cutoff point
						local_index_type neighbor_cutoff_num = -2;
						// We enter this for loop knowing that at least one index will satisfy the inside condition.
						// Choice here whether to determine lopping point from front or back
						for (local_index_type k=neighbor_list[i].size()-1; k>-1; --k) {
							// we know there are enough neighbors before cutoff_radius, and we lop as soon as we get below the cutoff_radius
							if (neighbor_list[i][k].second <= cutoff_radius) {
								// since cutoff < hsupport
								neighbor_cutoff_num = k;
								break;
							}
						}

						if (spatially_varying) {

							neighbor_list[i].resize(neighbor_cutoff_num+1); // +1 because neighbor_cutoff_num is an index, not a size
							hsupportView(i,0) = cutoff_radius;
						} else {
							cutoffs_collection(i) = cutoff_radius;
						}

						search_complete(i) = 1;

					} else {
						// we have found enough neighbors, but the cutoff region is larger than the search region
						// which means that we would potentially be inconsistent if we set our support to this
						// but ignored the neighbors with nonzero support
						// (between the search radius and the cutoff radius, which we do not know about)

						// one way is to just call this a failed search and let the the search radius enlarge in the normal way
						// (times the neighbor search search multiplier)
						// the alternative is to just set the support to the cutoff, but both require labeling this a failed search

						search_complete(i) = 0;
						cutoff_as_new_support = cutoff_radius;

						// When enlarging the search radius for the next loop, we use the cutoff as the new support size for
						// the spatially varying case. For the non spatially varying case, we would have to have a global communication
						// for every loop of the while loop, which we opt not to do and instead let enlarge the search radius
						// by multiplying the current search radius by the multiplier.
					}
				}

				if (search_complete(i)==0) {
					const scalar_type h_support_i = spatially_varying ? hsupportView(i,0) : hsupportView(0,0);
					neighbor_list[i].resize(0);

					// check for exceeding halo size (potentially missing existing particles on other processors)
					const scalar_type new_h_support_i = std::max(increase_multiplier*h_support_i, cutoff_as_new_support);
					if (comm_size>1 && (new_h_support_i > max_radius)) {
						std::ostringstream msg;
						msg << "Size of search required (>" << new_h_support_i << ") to get enough points exceeds halo search size (=" << max_radius << "." << std::endl;
						TEUCHOS_TEST_FOR_EXCEPT_MSG(true, msg.str());
						// TODO: In the future, should should become a cycle with ParticlesT to continue extending the halo until we have the desired number of points
					}

					// for the spatially varying case, update this entry individually
					if (spatially_varying) hsupportView(i,0) = new_h_support_i;
					// if !spatially_varying, hsupport is set globally later

				}
			}
		});
//		}

		// if desired_num_neighbors==0, while loop will end anyways
		if (desired_num_neighbors != 0) {

			// are searches complete on this processor?
			local_index_type local_searches_complete = 1;
			// are searches complete on this processor?
			for (local_index_type i=0, nt=_target_coords->nLocal(); i<nt; i++) {
				local_searches_complete = search_complete(i) < local_searches_complete ? search_complete(i) : local_searches_complete;
			}

			// are searches complete on all processors?
			local_index_type global_searches_complete;
			Teuchos::Ptr<local_index_type> global_searches_complete_ptr(&global_searches_complete);
			Teuchos::reduceAll<local_index_type, local_index_type>(*(_target_coords->getComm()), Teuchos::REDUCE_MIN,
					local_searches_complete, global_searches_complete_ptr);

			// if !dynamic, while loop will end, so we need all neighbors now
			TEUCHOS_TEST_FOR_EXCEPT_MSG(!dynamic && !global_searches_complete, "Insufficient number of neighbors found for specified support radii.");

			// spatially_varying neighborhoods already have been set, but we must adjust all NON spatially_varying neighborhoods at once
			if (!spatially_varying && !(global_searches_complete > 0)) {
				// Since at least one point on one processor needs a larger radius,
				// we will increase the radius for all neighborhoods on all processors
				// consistently.

				// We can use local index 0 as a value from which to multiply consistently on any processor
				// This may become inconsistent since processors with no target coordinates have no entry 0
				// but will be remedied when the search is complete over all processors
				if (_target_coords->nLocal() > 0) {
					const scalar_type old_h_support = hsupportView(0,0);
					this->setAllHSupportSizes(increase_multiplier*old_h_support);
				}

				// If we wanted to use the independently calculated max of cutoff region or enlargement of the search
				// radius by a multiplier, we would have to do a reduceAll here and use it in lieu of what we use above
				// (which is the old support * the multiplier)
			}

			// if NOT spatially_varying and finished searching, we need to get a consensus on the support radius
			if (!spatially_varying && (global_searches_complete > 0)) {

				// collect max cutoff radius from this processors
				scalar_type local_max_cutoff = 0;
				for (local_index_type i=0, nt=_target_coords->nLocal(); i<nt; i++) {
					local_max_cutoff = cutoffs_collection(i)>local_max_cutoff ? cutoffs_collection(i) : local_max_cutoff;
				}

				// collect max cutoff radius from all processors
				scalar_type global_processor_max_cutoff;
				Teuchos::Ptr<scalar_type> global_processor_max_cutoff_ptr(&global_processor_max_cutoff);
				Teuchos::reduceAll<local_index_type, scalar_type>(*(_target_coords->getComm()), Teuchos::REDUCE_MAX,
						local_max_cutoff, global_processor_max_cutoff_ptr);

				this->setAllHSupportSizes(global_processor_max_cutoff);

				// prune the already computed list of neighbors to remove anything after cutoff
				// there may or may not be some neighbors after the cutoff
				Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,_target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
//				for (local_index_type i=0; i<_target_coords->nLocal(); ++i) {
					local_index_type neighbor_cutoff_num = desired_num_neighbors-1;
					for (local_index_type k=desired_num_neighbors, vecsize=neighbor_list[i].size(); k<vecsize; k++) {
						if (neighbor_list[i][k].second > global_processor_max_cutoff) {
							neighbor_cutoff_num = k;
							break; // breaks the for loop, search is done
						}
					}
					neighbor_list[i].resize(neighbor_cutoff_num+1);
				});
//				}
			}

			// check condition to end the loop
			if (global_searches_complete > 0) {
				dynamically_updating_search_size = false;
			}
		}
	}

	NeighborSearchTime->stop();
	this->computeMaxNumNeighbors();
}

std::vector<std::pair<size_t, scalar_type> > NeighborhoodT::constructSingleNeighborList
	(const scalar_type* coordinate,	const scalar_type radius, bool use_physical_coords) const {
	std::vector<std::pair<size_t, scalar_type> > generated_neighbor_list;
	this->constructSingleNeighborList(coordinate, radius, generated_neighbor_list, use_physical_coords);
	return generated_neighbor_list;
}

}
