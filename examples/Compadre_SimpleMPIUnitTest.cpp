#include "Compadre_SimpleMPI.hpp"
#include <iostream>
#include "Compadre_Config.h"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_RCP.hpp"
#include "Tpetra_Core.hpp"
#include "Teuchos_oblackholestream.hpp"


int main(int argc, char* argv[]) {
    Teuchos::GlobalMPISession mpi(&argc, &argv);
	Teuchos::RCP<const Teuchos::Comm<int> > comm = Teuchos::DefaultComm<int>::getComm();
    Teuchos::oblackholestream blackHole;
    const int procRank = comm->getRank();
    const int numProcs = comm->getSize();
    
    std::ostream& out = (procRank == 0) ? std::cout : blackHole;
    
    out << "*** Compadre unit test: SimpleMPI ***" << std::endl;
    out << "1. Basic functions and load balancing." << std::endl;
    
    const int nn = 2500;
    
    Compadre::SimpleMPI mpiWork(nn, numProcs);
    
    out << "\t" << mpiWork << std::endl;
    
    mpiWork.loadBalance(nn/2, numProcs);
    
    out << "\t" << mpiWork << std::endl;

    out << "End Result: TEST PASSED" << std::endl;
return 0;
}
