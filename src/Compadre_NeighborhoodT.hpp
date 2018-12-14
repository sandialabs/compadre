#ifndef _COMPADRE_NEIGHBORHOODT_HPP_
#define _COMPADRE_NEIGHBORHOODT_HPP_

#include "CompadreHarness_Config.h"
#include "CompadreHarness_Typedefs.hpp"


namespace Compadre {

class CoordsT;
class ParticlesT;

/*
 *  TODO: Characteristic length determination
 */

class NeighborhoodT {

	protected:

		typedef Compadre::CoordsT coords_type;
		typedef Compadre::ParticlesT particles_type;

		const coords_type* _source_coords;
		const coords_type* _target_coords;
		const particles_type* _source_particles;
		const particles_type* _target_particles;

		local_index_type _nDim;
		local_index_type _max_num_neighbors;

		Teuchos::RCP<mvec_type> h_support_size;

		// size_t required by nanoflann
		std::vector<std::vector<std::pair<size_t, scalar_type> > > neighbor_list;

		Teuchos::RCP<Teuchos::ParameterList> _parameters;

		Teuchos::RCP<Teuchos::Time> NeighborSearchTime;
		Teuchos::RCP<Teuchos::Time> MaxNeighborComputeTime;

		local_index_type computeMaxNumNeighbors();


	public:

		static constexpr local_index_type DEFAULT_DESIRED_NUM_NEIGHBORS = 0;
		static constexpr scalar_type DEFAULT_RADIUS = 0.0;
		static constexpr local_index_type DEFAULT_MAXLEAF = 10;

		NeighborhoodT(const particles_type* source_particles, Teuchos::RCP<Teuchos::ParameterList> parameters, const particles_type* target_particles = NULL);

		virtual ~NeighborhoodT() {};

		const coords_type* getSourceCoordinates() const { return _source_coords; }

		const coords_type* getTargetCoordinates() const { return _target_coords; }

		void setHSupportSize(const local_index_type idx, const scalar_type val);

		double getHSupportSize(const local_index_type idx) const;

		void setAllHSupportSizes(const scalar_type val);

		Teuchos::RCP<mvec_type> getHSupportSizes() const { return h_support_size; }

		const std::vector<std::vector<std::pair<size_t, scalar_type> > >& getAllNeighbors() const {
			return neighbor_list;
		}

		const std::vector<std::pair<size_t, scalar_type> >& getNeighbors(const local_index_type idx) const {
			return neighbor_list.at(idx);
		}

		const local_index_type getMaxNumNeighbors() const;

		const scalar_type getMinimumHSupportSize() const;

		void constructAllNeighborList(const scalar_type max_radius, const local_index_type desired_num_neighbors = NeighborhoodT::DEFAULT_DESIRED_NUM_NEIGHBORS,
			const scalar_type initial_radius = NeighborhoodT::DEFAULT_RADIUS, const local_index_type maxLeaf = NeighborhoodT::DEFAULT_MAXLEAF, bool use_physical_coords = true);

		virtual std::vector<std::pair<size_t, scalar_type> > constructSingleNeighborList (const scalar_type* coordinate,
				const scalar_type radius, bool use_physical_coords = true) const;

		virtual void constructSingleNeighborList(const scalar_type* coordinate,
				const scalar_type radius, std::vector<std::pair<size_t, scalar_type> >& neighbors, bool use_physical_coords = true) const = 0;

};



}

#endif
