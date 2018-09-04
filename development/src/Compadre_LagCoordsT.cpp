#include "Compadre_LagCoordsT.hpp"
#include "Compadre_CoordsT.hpp"
#include "Compadre_XyzVector.hpp"

namespace Compadre {

LagCoordsT::LagCoordsT(Teuchos::RCP<CoordsT>& physCoords) {
    _physCoords = physCoords;
    lagPts = Teuchos::RCP<mvec_type>(new mvec_type(*(_physCoords->pts), Teuchos::Copy)); 
}

LagCoordsT::LagCoordsT(CoordsT* physCoords) {
    _physCoords = Teuchos::rcp(physCoords);
    lagPts = Teuchos::RCP<mvec_type>(new mvec_type(*(_physCoords->pts), Teuchos::Copy));
}

void LagCoordsT::reassociate(const Teuchos::RCP<CoordsT>& physCoords) {_physCoords = Teuchos::RCP<CoordsT>(physCoords);} 
   
void LagCoordsT::resetLagCoords() {
    lagPts = Teuchos::RCP<mvec_type>(new mvec_type(*(_physCoords->pts), Teuchos::Copy));
}

const Teuchos::RCP<mvec_type>& LagCoordsT::getLagPts() const {return lagPts;}


local_index_type LagCoordsT::nDim() const {return _physCoords->nDim();}
local_index_type LagCoordsT::nLocalMax(bool for_halo) const {return _physCoords->nLocalMax(for_halo);}
local_index_type LagCoordsT::nLocal(bool include_halo) const {return _physCoords->nLocal(include_halo);}
local_index_type LagCoordsT::nHalo() const {return _physCoords->nHalo();}

global_index_type LagCoordsT::nGlobalMax() const {return _physCoords->nGlobalMax();}
global_index_type LagCoordsT::nGlobal() const {return _physCoords->nGlobal();}
global_index_type LagCoordsT::getMinGlobalIndex() const {return _physCoords->getMinGlobalIndex();}
global_index_type LagCoordsT::getMaxGlobalIndex() const {return _physCoords->getMaxGlobalIndex();}

bool LagCoordsT::globalIndexIsLocal(const global_index_type idx) const {return _physCoords->globalIndexIsLocal(idx);}

std::pair<int, local_index_type> LagCoordsT::getLocalIdFromGlobalId(const global_index_type idx) const {
    return _physCoords->getLocalIdFromGlobalId(idx);
}

void LagCoordsT::replaceLocalCoords(const local_index_type idx, const scalar_type x, const scalar_type y, 
    const scalar_type z) {
    lagPts->replaceLocalValue(idx,0,x);
    lagPts->replaceLocalValue(idx,1,y);
    lagPts->replaceLocalValue(idx,2,z);
    lagPts->modify<host_view_type>();
}

void LagCoordsT::replaceLocalCoords(const local_index_type idx, const xyz_type& vec) {
    lagPts->replaceLocalValue(idx, 0, vec.x);
    lagPts->replaceLocalValue(idx, 1, vec.y);
    lagPts->replaceLocalValue(idx, 2, vec.z);
    lagPts->modify<host_view_type>();
}

void LagCoordsT::replaceGlobalCoords(const global_index_type idx, const scalar_type x, const scalar_type y, 
    const scalar_type z) {
    lagPts->replaceGlobalValue(idx,0,x);
    lagPts->replaceGlobalValue(idx,1,y);
    lagPts->replaceGlobalValue(idx,2,z);
    lagPts->modify<host_view_type>();
}

void LagCoordsT::replaceGlobalCoords(const global_index_type idx, const xyz_type& vec) {
    lagPts->replaceGlobalValue(idx, 0, vec.x);
    lagPts->replaceGlobalValue(idx, 1, vec.y);
    lagPts->replaceGlobalValue(idx, 2, vec.z);
    lagPts->modify<host_view_type>();
}

LagCoordsT::xyz_type LagCoordsT::getLocalCoords(const local_index_type idx, bool use_halo) const {
    host_view_type ptsView = (use_halo && idx >= _physCoords->_nLocal) ? 
        lagHaloPts->getLocalView<host_view_type>() : lagPts->getLocalView<host_view_type>();
    const local_index_type idx2 = (use_halo && idx >= _physCoords->_nLocal) ? idx - _physCoords->_nLocal : idx;
    return xyz_type(ptsView(idx2,0), ptsView(idx2,1), ptsView(idx2,2));
}

LagCoordsT::xyz_type LagCoordsT::getGlobalCoords(const global_index_type idx) const {
    const std::pair<int, local_index_type> remoteIds = _physCoords->getLocalIdFromGlobalId(idx);
    std::vector<scalar_type> cVec(3);
    Teuchos::ArrayView<scalar_type> cView(cVec);
    if (_physCoords->comm->getRank() == remoteIds.first) {
        const xyz_type vec = getLocalCoords(remoteIds.second);
        vec.convertToStdVector(cVec);
    }
    Teuchos::broadcast(*(_physCoords->comm), remoteIds.first, cView);
    return xyz_type(cVec);
}

LagCoordsT::xyz_type LagCoordsT::getCoordsPointToPoint(const global_index_type idx, const int destRank) const {
    xyz_type result;
    const std::pair<int, local_index_type> remoteIds = _physCoords->getLocalIdFromGlobalId(idx);
    const int sendRank = remoteIds.first;
    if (_physCoords->globalIndexIsLocal(idx)) 
        result = getLocalCoords(remoteIds.second);
    else {
        scalar_type arry[3];
        if (_physCoords->comm->getRank() == remoteIds.first) {
            const xyz_type vec = getLocalCoords(remoteIds.second);
            vec.convertToArray(arry);
            Teuchos::send<int,scalar_type>(*(_physCoords->comm), 3, arry, destRank);
        }
        else if (_physCoords->comm->getRank() == destRank) {
            const int srcRank = Teuchos::receive<int, scalar_type>(*(_physCoords->comm), remoteIds.first, 3, arry);
            result = xyz_type(arry);
        }
    }
    return result;
}

Teuchos::RCP<const Teuchos::Comm<local_index_type> > LagCoordsT::getComm() const {return _physCoords->comm;}
Teuchos::RCP<const map_type> LagCoordsT::getMapConst(bool for_halo) const {
    return for_halo ? Teuchos::rcp_const_cast<const map_type>(_physCoords->halo_map) : 
        Teuchos::rcp_const_cast<const map_type>(_physCoords->map);
}
Teuchos::RCP<const importer_type> LagCoordsT::getHaloImporterConst() const {
    return Teuchos::rcp_const_cast<const importer_type>(_physCoords->halo_importer);
}

LagCoordsT::xyz_type LagCoordsT::localCrossProduct(const local_index_type idx, const xyz_type& queryPt) const {
    xyz_type indVec = getLocalCoords(idx, true);
    return indVec.crossProduct(queryPt);
}

scalar_type LagCoordsT::localDotProduct(const local_index_type idx, const xyz_type& queryPt) const {
    xyz_type indVec = getLocalCoords(idx, true);
    return indVec.dotProduct(queryPt);
}

void LagCoordsT::syncMemory() {
    lagPts->sync<device_view_type>();
    lagPts->sync<host_view_type>();
}

bool LagCoordsT::hostNeedsUpdate() const {return lagPts->need_sync<host_execution_space>();}
bool LagCoordsT::deviceNeedsUpdate() const {return lagPts->need_sync<device_view_type>();}

void LagCoordsT::buildHalo() {
    TEUCHOS_TEST_FOR_EXCEPT_MSG(_physCoords->z2problem.is_null(), "Run Zoltan2 on physical coordinates first.");
    const bool setToZero = false;
    lagHaloPts = Teuchos::rcp(new mvec_type(_physCoords->halo_map, _physCoords->_nDim, setToZero));
    updateHaloPts();
}

void LagCoordsT::updateHaloPts() {
    lagHaloPts->doImport(*lagPts, *(_physCoords->halo_importer), Tpetra::CombineMode::INSERT);
}

std::ostream& operator << (std::ostream& os, const LagCoordsT& lagCoords) {
    os << "LagCoordsT basic info \n";
    os << "\tmax coords allowed on this node = " << lagCoords.nLocalMax() << std::endl;
    os << "\tnumber of coords currently on this node = " << lagCoords.nLocal() << std::endl;
    return os;
}

void LagCoordsT::writeToMatlab(std::ostream& fs, const std::string coordsName, const int procRank) const {
    device_view_type ptsHostView = lagPts->getLocalView<host_view_type>();
    fs << coordsName << "Xyz" << procRank << " = [";
    for (local_index_type i = 0; i < nLocal() - 1; ++i) 
        fs << ptsHostView(i, 0) << ", " << ptsHostView(i, 1) << ", " << ptsHostView(i, 2) << "; ..." << std::endl;
    fs << ptsHostView(_physCoords->_nLocal-1, 0) << ", " << ptsHostView(_physCoords->_nLocal-1, 1) << ", " << 
         ptsHostView(_physCoords->_nLocal-1, 2) << "]; " << std::endl;
}

void LagCoordsT::writeHaloToMatlab(std::ostream& fs, const std::string coordsName, const int procRank) const {
    device_view_type haloPtsView = lagHaloPts->getLocalView<host_view_type>();
    if (lagHaloPts->getLocalLength() > 0 ) {
        fs << coordsName << "Xyz" << procRank << " = ["; 
        for (local_index_type i = 0; i < lagHaloPts->getLocalLength() - 1; ++i)
            fs << haloPtsView(i, 0) << ", " << haloPtsView(i, 1) << ", " << haloPtsView(i, 2) << "; ..." << std::endl;
        fs << haloPtsView(lagHaloPts->getLocalLength() - 1, 0) << ", " << 
        haloPtsView(lagHaloPts->getLocalLength() - 1, 1) << ", " << 
        haloPtsView(lagHaloPts->getLocalLength() - 1, 2) << "]; " << std::endl;
    }
}

void LagCoordsT::print(std::ostream& os) const {
    auto out = Teuchos::getFancyOStream(Teuchos::rcpFromRef(os));
    if (!lagPts.is_null()) {
        std::cout << "Locally owned points on processor " << _physCoords->comm->getRank() << std::endl;
        lagPts->describe(*out, Teuchos::VERB_EXTREME);
    }
    if (!lagHaloPts.is_null() ) {
        std::cout << "Halo points on processor " << _physCoords->comm->getRank() << std::endl;
        lagHaloPts->describe(*out, Teuchos::VERB_EXTREME);
    }
}

}
