#ifndef _COMPADRE_COORDS_TPETRA_HPP_
#define _COMPADRE_COORDS_TPETRA_HPP_

#include "CompadreHarness_Config.h"
#include "CompadreHarness_Typedefs.hpp"

namespace Compadre {

// forward declaration of ParticlesT used to friend its class
class ParticlesT;
class LagCoordsT;
class LagrangianParticlesT;
class AnalyticFunction;
class XyzVector;

/** Manages coordinate data structures, and halo building
 *
 * This class is designed to store coordinate data, build halo information, and call for repartitioning.
 * However, if fields are defined on each particle, then ParticlesT should be used to call the halo building
 * and repartitioning to keep fields, flags, and coordinates in sync.
*
* This class currently handles repartition management, which
* any function using a processor bounding box requires and
* but which really should be managed at the ParticlesT level.
*
* If partitioning is called through ParticlesT member function,
* then the repartition management logic is already taken care of
* by that class. If it is called directly on the CoordsT class,
* then this is not the case and one should take care that they
* call neighborhood searches in a consistent way with how they
* partitioned the data (physical coordinates or material coordinates
* in a Lagrangian simulation, for instance)
*/

class CoordsT {

	protected: 

		typedef Compadre::ParticlesT particles_type;
		typedef AnalyticFunction function_type;
		typedef Compadre::XyzVector xyz_type;

		local_index_type _nMaxLocal;
		local_index_type _nMaxHalo;
		local_index_type _nLocal;
		local_index_type _nHalo;
		local_index_type _nDim;
// 		global_index_type _nGlobal;
// 		global_index_type _nMaxGlobal;
		std::string _units;
		bool _is_lagrangian = false;

		// partitioning can be done with respect to physical or
		// material coordinates, but not both
		// in Eulerian simulations, the physical coordinates
		// coincide and this is irrelevant, but in Lagrangian
		// simulations, as the points move with a velocity,
		// for instance, this option is meaningful
		bool _partitioned_using_physical = true;

		Teuchos::RCP<const Teuchos::Comm<local_index_type> > comm;
		Teuchos::RCP<const map_type> map;
		Teuchos::RCP<mvec_type> pts;
		Teuchos::RCP<z2_problem_type> z2problem;
		Teuchos::RCP<z2_adapter_type> z2adapter;
		Teuchos::RCP<map_type> halo_map;
		Teuchos::RCP<mvec_type> halo_pts;
		Teuchos::RCP<importer_type> halo_importer;

		// _physical designation used for Lagrangian simulations to denote the physical location of coordinates
		// Lagrangian coordinates are stores in pts and halo_pts
		Teuchos::RCP<mvec_type> pts_physical;
		Teuchos::RCP<mvec_type> halo_pts_physical;

		host_view_type pts_view;
		host_view_type pts_physical_view;
		host_view_type halo_pts_view;
		host_view_type halo_pts_physical_view;

		double _halo_size;

	public:	

		CoordsT( const global_index_type nn, const Teuchos::RCP<const Teuchos::Comm<int> >& _comm, const int nDim);
		virtual ~CoordsT() {};
		
		friend class LagCoordsT;

		local_index_type nDim() const;
		local_index_type nLocalMax(bool for_halo = false) const;
		local_index_type nLocal(bool include_halo = false) const;
		local_index_type nHalo() const;
		void setLocalN(const local_index_type newN, bool for_halo = false);
		void setLocalNFromMap(bool for_halo = false);
		global_index_type nGlobalMax() const;
		global_index_type nGlobal() const;
		global_index_type getMinGlobalIndex() const;
		global_index_type getMaxGlobalIndex() const;

// Modifiers

		// Coordinate data modifiers

		void globalResize(const global_index_type nn);

		void localResize(const global_index_type nn);
		
		void insertCoords(const std::vector<xyz_type>& new_pts_vector, const std::vector<xyz_type>& new_pts_physical_vector);

		// if in a lagrangian simulation, this inserts the same values in both sets of coordinates
		void insertCoords(const std::vector<xyz_type>& new_pts_vector);

		void removeCoords(const std::vector<local_index_type>& coord_ids);

		void zoltan2Init(bool use_physical_coords = true);

		void zoltan2Partition();

		void applyZoltan2Partition(Teuchos::RCP<mvec_type>& vec) const;

		void buildHalo(scalar_type h, bool use_physical_coords = true);

		void updateHaloPts();

		void snapLagrangianCoordsToPhysicalCoords();

		void snapPhysicalCoordsToLagrangianCoords();

//		void setCoords(const mvec_type* update_vector = NULL, bool use_physical_coords = true);

		// physical coordinates when in the Lagrangian frame by default
		void deltaUpdatePhysicalCoordsFromVector(const mvec_type* update_vector);

		void deltaUpdatePhysicalCoordsFromVectorByTimestep(const double dt, const mvec_type* update_vector);

		// physical coordinates when in the Lagrangian frame by default
		void deltaUpdatePhysicalCoordsFromVectorFunction(function_type* fn, bool evaluate_using_lagrangian_coordinates = true);

		// provide a vector of data that overwrites either the lagrangian or physical coordinates
		void overwriteCoordinates(const mvec_type* data, bool use_physical_coords = true);

		// snap all points to the unit sphere
		void snapToSphere(const double radius, bool use_physical_coords = true);

		// fn(Lagrangian coords) becomes the new physical coordinates
		void transformLagrangianCoordsToPhysicalCoordsByVectorFunction(function_type* fn);

		// ensures that views match the current data in the vector
		void updateAllHostViews();

		// Class modifiers

		void setLagrangian(bool val = true);

		void setUnits(std::string name) { _units = name; }

		void setPhysicalCoordinatesForPartitioning (bool value) { _partitioned_using_physical = value; }

		// Individual modifiers
		void replaceLocalCoords(const local_index_type idx, const scalar_type x, const scalar_type y,
								const scalar_type z = 0.0, bool use_physical_coords = true);

		void replaceLocalCoords(const local_index_type idx, const xyz_type xyz, bool use_physical_coords = true);

		void replaceGlobalCoords(const global_index_type idx, const scalar_type x, const scalar_type y,
								 const scalar_type z = 0.0, bool use_physical_coords = true);

		void replaceGlobalCoords(const global_index_type idx, const xyz_type xyz, bool use_physical_coords = true);

		// Derived class modifier

		virtual void initRandom(const scalar_type s, const local_index_type i, bool use_physical_coords = true) = 0;

// Accessors

		//  Global functions

		bool globalIndexIsLocal(const global_index_type idx) const;

		std::pair<int, local_index_type> getLocalIdFromGlobalId(const global_index_type idx) const;

		xyz_type getGlobalCoords(const global_index_type idx, bool use_physical_coords = true) const;

		xyz_type getCoordsPointToPoint(const global_index_type idx, const int destRank, bool use_physical_coords = true) const;

		//  Local functions
		
		xyz_type getLocalCoords(const local_index_type idx1, bool use_halo = false, bool use_physical_coords = true) const;

		const Teuchos::RCP<mvec_type>& getPts(const bool halo = false, bool use_physical_coords = true) const;
		
		Teuchos::RCP<const Teuchos::Comm<local_index_type> > getComm() const;
		
		Teuchos::RCP<const map_type> getMapConst(bool for_halo = false) const;

		Teuchos::RCP<const importer_type> getHaloImporterConst() const;

		const scalar_type getHaloSize() const;
		
		const bool isLagrangian() const { return _is_lagrangian; }

		const std::string getUnits() const { return _units; }

		// returns whether partitioning was done w.r.t. material or physical coordinates
		// (true = physical)
		bool getUsingPhysicalCoordinatesForPartitioning () const {
			return _partitioned_using_physical;
		}


// Coordinate Metrics

	// Virtual (depends on coordinate type)
 		virtual scalar_type distance(const local_index_type idx1, const xyz_type queryPt, bool use_physical_coords = true) const = 0;
 		virtual scalar_type distance(const xyz_type queryPt1, const xyz_type queryPt2) const = 0;
 		virtual scalar_type globalDistance(const global_index_type idx1, const xyz_type queryPt, bool use_physical_coords = true) const = 0;
 		virtual xyz_type midpoint(const local_index_type idx1, const xyz_type queryPt, bool use_physical_coords = true) const = 0;
		virtual xyz_type centroid(const std::vector<xyz_type> vecs) const = 0;
		virtual scalar_type triArea(const xyz_type vecA, const xyz_type vecB, const xyz_type vecC) const = 0;

	// General to any coordinate type
		xyz_type localCrossProduct(const local_index_type idx1, const xyz_type& queryPt, bool use_physical_coords = true) const;
		scalar_type localDotProduct( const local_index_type idx1, const xyz_type& queryPt, bool use_physical_coords = true) const;


// Memory
		void syncMemory();
		bool hostNeedsUpdate() const;
		bool deviceNeedsUpdate() const;

// Utilities

		void verifyCoordsOnProcessor(const std::vector<xyz_type>& new_pts_vector, bool use_physical_coords = true) const;

		const std::vector<scalar_type> boundingBoxMinOnProcessor(const local_index_type processor_num = -1) const;

		const std::vector<scalar_type> boundingBoxMaxOnProcessor(const local_index_type processor_num = -1) const;

		void writeToMatlab(std::ostream& fs, const std::string coordsName = "", const int procRank = 0, bool use_physical_coords = true) const;

		void writeHaloToMatlab(std::ostream& fs, const std::string coordsName = "", const int procRank = 0, bool use_physical_coords = true) const;

		void print(std::ostream& os, bool use_physical_coords = true) const;


};

}

std::ostream& operator << (std::ostream& os, const Compadre::CoordsT& coords);

#endif
