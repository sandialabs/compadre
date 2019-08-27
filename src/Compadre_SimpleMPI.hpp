#ifndef _COMPADRE_SIMPLE_MPI_
#define _COMPADRE_SIMPLE_MPI_

// Replicated data algorithm
#include "CompadreHarness_Typedefs.hpp"
#include <mpi.h>

namespace Compadre {

class SimpleMPI {
    protected:
        std::vector<local_index_type> _procStartIndex;
        std::vector<local_index_type> _procEndIndex;
        std::vector<local_index_type> _procMsgSize;
        
        local_index_type _nTotal;
        int _nProcs;

    public:
        SimpleMPI(const local_index_type nTotal, const local_index_type nProcs) :
            _procStartIndex(nProcs, -1), _procEndIndex(nProcs, -1), _procMsgSize(nProcs,0),
            _nTotal(nTotal), _nProcs(nProcs) {
            if (nTotal > 0)
                loadBalance(nTotal, nProcs);
        }
        
        int nProcs() const {return _nProcs;}
        local_index_type nTotal() const {return _nTotal;}
        
        local_index_type procStartIndex(const int rank) const {return _procStartIndex[rank];}
        local_index_type procEndIndex(const int rank) const {return _procEndIndex[rank];}
        local_index_type procMsgSize(const int rank) const {return _procMsgSize[rank];}
        
        void loadBalance(const local_index_type nTotal, const int nProcs) {
            _nTotal = nTotal;
            const local_index_type chunk_size = int(nTotal/nProcs);
            for (int i = 0; i < nProcs; ++i) {
                _procStartIndex[i] = i * chunk_size;
                _procEndIndex[i] = (i + 1) * chunk_size - 1;
            }
            _procEndIndex[_nProcs-1] = nTotal - 1;
            
            for (int i = 0; i < nProcs; ++i)
                _procMsgSize[i] = _procEndIndex[i] - _procStartIndex[i] + 1;
        }

};

std::ostream& operator << (std::ostream& os, const Compadre::SimpleMPI& mpi){
    os << "SimpleMPI: " << mpi.nTotal() << " items stored on each of " << mpi.nProcs() << " mpi ranks.\n";
    for (int i = 0; i < mpi.nProcs(); ++i) {
        os << "\tproc[" << i << "] has indices " << mpi.procStartIndex(i) << " through " << mpi.procEndIndex(i) <<
            "; msg. size = " << mpi.procMsgSize(i) << std::endl;
    }
    return os;
}

}
#endif
