#ifndef _COMPADRE_LAG_COORDS_HPP_
#define _COMPADRE_LAG_COORDS_HPP_

#include "CompadreHarness_Config.h"
#include "CompadreHarness_Typedefs.hpp"

namespace Compadre {

//forward declarations
class XyzVector;
class CoordsT;
class ParticlesT;
class LagrangianParticlesT;


class LagCoordsT {
    protected:
        typedef XyzVector xyz_type;
        
        Teuchos::RCP<CoordsT> _physCoords;
        Teuchos::RCP<mvec_type> lagPts;
        Teuchos::RCP<mvec_type> lagHaloPts;
        
    public:
        LagCoordsT(Teuchos::RCP<CoordsT>& physCoords);
        LagCoordsT(CoordsT* physCoords);
        
        void reassociate(const Teuchos::RCP<CoordsT>& physCoords);
        void resetLagCoords();
        const Teuchos::RCP<mvec_type>& getLagPts() const;
        
        local_index_type nDim() const;
        local_index_type nLocalMax(bool for_halo = false) const;
        local_index_type nLocal(bool include_halo = false) const;
        local_index_type nHalo() const;

        global_index_type nGlobalMax() const;
        global_index_type nGlobal() const;
        global_index_type getMinGlobalIndex() const;
        global_index_type getMaxGlobalIndex() const;
        
        bool globalIndexIsLocal(const global_index_type idx) const;
        
        std::pair<int, local_index_type> getLocalIdFromGlobalId(const global_index_type idx) const;
        
        void replaceLocalCoords(const local_index_type idx, const scalar_type x, const scalar_type y,
            const scalar_type z = 0.0);
        void replaceLocalCoords(const local_index_type idx, const xyz_type& xyz);
        
        void replaceGlobalCoords(const global_index_type idx, const scalar_type x, const scalar_type y, 
            const scalar_type z = 0.0);
        void replaceGlobalCoords(const global_index_type idx, const xyz_type& xyz);
    
        xyz_type getLocalCoords(const local_index_type idx, bool use_halo = false) const;
        
        xyz_type getGlobalCoords(const global_index_type idx) const;
        
        xyz_type getCoordsPointToPoint(const global_index_type idx, const int destRank) const;
        
        Teuchos::RCP<const Teuchos::Comm<local_index_type> > getComm() const;
        Teuchos::RCP<const map_type> getMapConst(bool for_halo = false) const;
        Teuchos::RCP<const importer_type> getHaloImporterConst() const;
        
        xyz_type localCrossProduct(const local_index_type idx1, const xyz_type& queryPt) const;
        scalar_type localDotProduct(const local_index_type idx1, const xyz_type& queryPt) const;
        
        void syncMemory();
        bool hostNeedsUpdate() const;
        bool deviceNeedsUpdate() const;
        
        void writeToMatlab(std::ostream& fs, const std::string coordsName = "", const int procRank = 0) const;
        void writeHaloToMatlab(std::ostream& fs, const std::string coordsName="", const int procRank = 0 ) const;
        void print(std::ostream& fs) const;
        
        void buildHalo();
        void updateHaloPts();
        
        virtual scalar_type distance(const local_index_type idx1, const xyz_type& queryPt) const = 0;
 		virtual scalar_type globalDistance(const global_index_type idx1, const xyz_type& queryPt) const = 0;
 		virtual xyz_type midpoint(const local_index_type idx1, const xyz_type &queryPt) const = 0;
		virtual xyz_type centroid(const std::vector<xyz_type> vecs) const = 0;
		virtual scalar_type triArea(const xyz_type& vecA, const xyz_type& vecB, const xyz_type& vecC) const = 0;
};

std::ostream& operator << (std::ostream& os, const Compadre::LagCoordsT& lagCoords);


}

#endif