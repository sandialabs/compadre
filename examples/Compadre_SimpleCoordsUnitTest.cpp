#include "CompadreHarness_Config.h"
#include "CompadreHarness_Typedefs.hpp"
#include "Compadre_GlobalConstants.hpp"
#include "Compadre_SimpleSphericalCoords.hpp"
#include "Compadre_SimpleEuclidean.hpp"
#include "Compadre_XyzVector.hpp"

#include <Tpetra_Core.hpp>

using Compadre::scalar_type;
using Compadre::local_index_type;
typedef Compadre::XyzVector xyz_type;

int main(int argc, char* argv[]) {
    Teuchos::GlobalMPISession mpi(&argc, &argv);
    Teuchos::RCP<const Teuchos::Comm<int> > comm =  Teuchos::DefaultComm<int>::getComm();
    Teuchos::oblackholestream blackHole;
    const int procRank = comm->getRank();
    
    std::ostream& out = (procRank == 0) ? std::cout : blackHole;
    
    Kokkos::initialize(argc, argv);
{
    
    out << "*** Compadre unit test: SimpleCoords ***" << std::endl;
    out << "1. Spherical Coordinates." << std::endl;
    /*
        CONSTRUCT a spherical set of particles
    */
    Compadre::SimpleSphericalCoords sc(14, "sphTest");
    sc.insertXyz( 0.577350269189626,	-0.577350269189626,	0.577350269189626);//0
	sc.insertXyz( 0.577350269189626,	-0.577350269189626,	-0.577350269189626);//1
	sc.insertXyz( 0.577350269189626,	0.577350269189626,	-0.577350269189626);//2
	sc.insertXyz( 0.577350269189626,	0.577350269189626,	0.577350269189626);//3
	sc.insertXyz( -0.577350269189626,	0.577350269189626,	-0.577350269189626);//4
	sc.insertXyz( -0.577350269189626,	0.577350269189626,	0.577350269189626);//5
	sc.insertXyz( -0.577350269189626,	-0.577350269189626,	-0.577350269189626);//6
	sc.insertXyz( -0.577350269189626,	-0.577350269189626,	0.577350269189626);//7
	sc.insertXyz( 1.0, 0.0, 0.0);//8
	sc.insertXyz( 0.0, 1.0, 0.0);//9
	sc.insertXyz( -1.0, 0.0, 0.0);//10
	sc.insertXyz( 0.0, -1.0, 0.0);//11
	sc.insertXyz( 0.0, 0.0, 1.0);//12
	sc.insertXyz( 0.0, 0.0, -1.0);//13
	
	out << sc << std::endl;
	
	for (int i = 0; i < sc.nUsed(); ++i)
	    out << "\tcoord " << i << " has magnitude " << sc.xyz(i).magnitude() << std::endl;
    
    out << "\tsphere midpoint of coord 0 and coord 2 should be (1,0,0); computed midpoint is " <<
        sc.midpoint(0,2) << std::endl;
        
    out << "\tinput position of coord 3 = (0.57735, 0.57735, 0.57735); output position is " << sc.xyz(3) << std::endl;
    
    /* 
        CONSTRUCT A Spherical Quad
    */
    std::vector<local_index_type> vertInds = {0, 3, 5, 7};
    for (int i = 0; i < 4; ++i) {
        out << "\tvertInd = " << vertInds[i] << "; coord: " << sc.xyz(vertInds[i]) << std::endl;
    }
    out << "\tcentroid of particles 0, 3, 5, 7 should be (0, 0, 1). Computed centroid is " << 
        sc.centroid(vertInds) << std::endl;
    
    /*
        CHeck area of sphere
    */
    scalar_type surfArea = 0.0;
    scalar_type surfAreaExact;
    {
        Compadre::GlobalConstants consts;
        surfAreaExact = 4.0 * consts.Pi();
    }
    // face 1 area
	surfArea += sc.triArea(0,8,1);
	surfArea += sc.triArea(1,8,2);
	surfArea += sc.triArea(2,8,3);
	surfArea += sc.triArea(3,8,0);
	// face 2
	surfArea += sc.triArea(3,9,2);
	surfArea += sc.triArea(2,9,4);
	surfArea += sc.triArea(4,9,5);
	surfArea += sc.triArea(5,9,3);
	// face 3
	surfArea += sc.triArea(5,10,4);
	surfArea += sc.triArea(4,10,6);
	surfArea += sc.triArea(6,10,7);
	surfArea += sc.triArea(7,10,5);
	// face 4
	surfArea += sc.triArea(7,11,6);
	surfArea += sc.triArea(6,11,1);
	surfArea += sc.triArea(1,11,0);
	surfArea += sc.triArea(0,11,7);
	// face 5
	surfArea += sc.triArea(7,12,0);
	surfArea += sc.triArea(0,12,3);
	surfArea += sc.triArea(3,12,5);
	surfArea += sc.triArea(5,12,7);
	// face 6
	surfArea += sc.triArea(1,13,6);
	surfArea += sc.triArea(6,13,4);
	surfArea += sc.triArea(4,13,2);
	surfArea += sc.triArea(2,13,1);
	
	out << "\tsphere surface area should be " << surfAreaExact << "; computed surface area is " << surfArea << std::endl;
    
    {
        out << "\toutputting coordinates to .m file.\n";
        std::ofstream mFile("../test_data/simpleSphere.m");
        sc.writeToMatlab(mFile, "cubeSph", 0);
        mFile.close();
    }
    
    out << "2. Euclidean coordinates." << std::endl;
    
    Compadre::SimpleEuclideanCoords ec(8, "unitCube");
    ec.insertXyz(0,0,0); //0
    ec.insertXyz(1,0,0); //1
    ec.insertXyz(1,1,0); //2
    ec.insertXyz(0,1,0); //3
    ec.insertXyz(0,0,1); //4
    ec.insertXyz(1,0,1); //5
    ec.insertXyz(1,1,1); //6
    ec.insertXyz(0,1,1); //7
    
    out << "\tcoord 0 and coord 1 should have midpoint = (0.5, 0, 0); computed midpoint is :" << ec.midpoint(0,1) << std::endl;
    std::vector<xyz_type> vecs;
    for (int i = 0; i < 4; ++i)
        vecs.push_back(ec.xyz(vertInds[i]));
    out << "\tcentroid of coords 0, 3, 5, 7 should be (0.25, 0.5, 0.5); computed centroid is :" << ec.centroid(vecs) << std::endl;
    
    surfArea = ec.triArea(0,1,2) + ec.triArea(2,3,0);
    out << "\tarea of bottom face should be 1; computed area is " << surfArea << std::endl;
    
    out << "End Result: TEST PASSED" << std::endl;
}
    Kokkos::finalize();
return 0;
}
