#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_CommandLineProcessor.hpp>
#include <Tpetra_Core.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_TeuchosCommAdapters.hpp>
#include "CompadreHarness_Typedefs.hpp"
#include "Compadre_CoordsT.hpp"
#include "Compadre_ParticlesT.hpp"
#include "Compadre_FieldT.hpp"
#include <Compadre_FieldManager.hpp>
#include "Compadre_SphericalCoordsT.hpp"
#include "Compadre_FileIO.hpp"
#include "Compadre_AnalyticFunctions.hpp"
#include "Compadre_ParameterManager.hpp"
#include <iostream>
#include <string>
#define _PI_ 3.1415926535897932384626433832795027975
#define _FLOAT_ZERO_ 1.0e-14

using Compadre::global_index_type;
using Compadre::local_index_type;
using Compadre::scalar_type;

typedef Compadre::device_view_type device_view_type;

struct greensFn {
    typedef Compadre::scalar_type scalar_type;
    device_view_type srcCoords;
    device_view_type tgtCoords;
    device_view_type tgtPotential;
    device_view_type srcSource;
    device_view_type srcArea;

    greensFn(device_view_type tgtCrds, device_view_type srcCrds,
        device_view_type tgtPot,  device_view_type srcSrc, device_view_type srcA) :
        srcCoords(srcCrds), tgtCoords(tgtCrds), tgtPotential(tgtPot), srcSource(srcSrc), srcArea(srcA) {};

    KOKKOS_INLINE_FUNCTION
    void operator() (const int& i) const {
        for (int j = 0; j < srcCoords.extent(0); ++j) {
            scalar_type dotProd = 0.0;
            for (int k = 0; k < 3; ++k)
                dotProd += tgtCoords(i,k) * srcCoords(j,k);
            if ( std::abs(1.0 - dotProd) > _FLOAT_ZERO_ )
                tgtPotential(i,0) -= std::log(1.0 - dotProd) * srcSource(j,0) * srcArea(j,0) / (4.0 * _PI_);
        }
    }
};

int main(int argc, char* args[]) {
    Teuchos::GlobalMPISession mpi(&argc, &args);
	Teuchos::oblackholestream bstream;
	Teuchos::RCP<const Teuchos::Comm<int> > comm = Teuchos::DefaultComm<int>::getComm();
	const int procRank = comm->getRank();
	const int numProcs = comm->getSize();
	
	Kokkos::initialize(argc, args);
	
	{
	
		std::ostream& out = (procRank == 0) ? std::cout : bstream;

		// PARSE COMMAND LINE ARGUMENTS
		Teuchos::CommandLineProcessor clp;
		clp.setDocString("Poisson solver for the sphere. Argument --filename=afile is required to define a particle set.");
		std::string fname = "../test_data/grids/icos_tri_sphere/icosTri_5.nc";
		clp.setOption("filename", &fname, "particle set .nc filename", false);
		clp.throwExceptions(false);

		Teuchos::CommandLineProcessor::EParseCommandLineReturn parseReturn = clp.parse(argc, args);
		if (parseReturn == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED)
			return 0;
		if (parseReturn != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
			return 1;

		out << "Particle set filename = " << fname << std::endl;
	
		typedef Compadre::SphericalCoordsT CT;

		Teuchos::RCP<Compadre::ParameterManager> parameter_manager = Teuchos::rcp(new Compadre::ParameterManager());
		Teuchos::RCP<Teuchos::ParameterList> parameters = parameter_manager->getList();
		Teuchos::ParameterList coordList = parameters->get<Teuchos::ParameterList>("coordinates");
		coordList.set("type","spherical");

		// DEFINE PARTICLES: Read from .nc file
		Teuchos::RCP<CT> sphCoords = Teuchos::rcp(new CT(0, comm));
		Teuchos::RCP<Compadre::ParticlesT> particles =
			Teuchos::rcp( new Compadre::ParticlesT(parameters, Teuchos::rcp_dynamic_cast<Compadre::CoordsT>(sphCoords)));

		Compadre::FileManager fm;
		fm.setReader(fname, particles);
		fm.read();
		const global_index_type nParticles = particles->getCoordsConst()->nGlobal();
	
		particles->zoltan2Initialize();
	//     out << "Fields registered from input file:\n";
	// 	particles->listFields(out);
	//
		particles->getFieldManager()->createField(1, "computedPotential");
		particles->getFieldManager()->createField(1, "exactPotential");
		particles->getFieldManager()->createField(1, "computed-exact");

		const int harm_m = 4;
		const int harm_n = 5;
		particles->getFieldManager()->createField(1, "source");

		particles->createDOFManager();
	
		std::vector<Teuchos::RCP<Compadre::FieldT> > fields = particles->getFieldManagerConst()->getVectorOfFields();
		int srcID;
		int exactID;
		int computID;
		int areaID;
		int errID;
		for (int i = 0; i < fields.size(); ++i) {
			if (fields[i]->getName() == "source")
				srcID = i;
			if (fields[i]->getName() == "computedPotential")
				computID = i;
			if (fields[i]->getName() == "exactPotential")
				exactID = i;
			if (fields[i]->getName() == "area")
				areaID = i;
			if (fields[i]->getName() == "computed-exact")
				errID = i;
		}
		Compadre::SphereHarmonic harm(harm_m, harm_n);
		fields[srcID]->localInitFromScalarFunction(&harm);
		fields[exactID]->localInitFromScalarFunction(&harm);
		fields[exactID]->scale(1.0 / (harm_n * (harm_n + 1)));


		device_view_type targetCoords = particles->getCoords()->getPts()->getLocalView<device_view_type>();
		device_view_type sourceCoords = particles->getCoords()->getPts()->getLocalView<device_view_type>();
		device_view_type targetPotential = fields[computID]->getDeviceView();
		device_view_type poissonSource = fields[srcID]->getDeviceView();
		device_view_type particleArea = fields[areaID]->getDeviceView();

	
		Teuchos::RCP<Teuchos::Time> GFTime = Teuchos::TimeMonitor::getNewCounter ("GreensFn Time");
		GFTime->start();
		// local sums
		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,particles->getCoords()->nLocal()),
			greensFn(targetCoords, sourceCoords,
				targetPotential, poissonSource, particleArea));
		GFTime->stop();

		if (numProcs > 1 ){
			// communication section
			typedef Kokkos::View<global_index_type*> gid_view_type;
			typedef Kokkos::View<const global_index_type*> const_gid_view_type;

			const_gid_view_type send_gids_const = particles->getCoordMap()->getMyGlobalIndices();
			const local_index_type sendSize = send_gids_const.extent(0);

			std::vector<global_index_type> send_gids(sendSize);
			for (global_index_type j = 0; j < sendSize; ++j)
				send_gids[j] = send_gids_const[j];

	
			Teuchos::ArrayView<global_index_type> sendView(send_gids);
	
			out << "ready for communication loop." << std::endl;

			for (int p = 0; p < numProcs - 1; ++p) {
				// proc procRank receives global indices from procRank + p - 1
				const int srcRank = (procRank - p - 1 + numProcs)%numProcs;
				std::cout << "proc " << procRank << " iteration " << p << " receiving from proc " << srcRank << std::endl;
				// proc procRank sends its global indices to procRank + p + 1
				const int destRank = (procRank + p + 1)%numProcs;
				std::cout << "proc " << procRank << " iteration " << p << " sending to proc " << destRank << std::endl;

	
				local_index_type recvSize;

				Teuchos::send<int, local_index_type>(*comm, sendSize, destRank);
				int recStat = Teuchos::receive<int, local_index_type>(*comm, srcRank, &recvSize);
				TEUCHOS_TEST_FOR_EXCEPT_MSG(recStat != srcRank, "MPI comm fail.");

				for (int i = 0; i < numProcs; ++i) {
					comm->barrier();

					if (procRank == i) {
						std::cout << "proc " << i << " sends " << sendSize << " gids to proc " << destRank << std::endl;
						std::cout << "proc " << i << " recvs " << recvSize << " gids from proc " << srcRank << std::endl;
						std::cout << std::endl;
					}
				}

				std::vector<global_index_type> receive_gids(recvSize);
				Teuchos::ArrayView<global_index_type> recView(receive_gids);

				Teuchos::send(*comm, sendSize, sendView.getRawPtr(), destRank);
				recStat = Teuchos::receive(*comm, srcRank, recvSize, recView.getRawPtr());
				TEUCHOS_TEST_FOR_EXCEPT_MSG(recStat != srcRank, "MPI comm fail.");

				for (int i = 0; i < numProcs; ++i) {
					comm->barrier();

					if (procRank == i) {
						std::cout << "proc " << i << " recv gids (" << recvSize << "):" << std::endl << "\t";
						for (int j = 0; j < send_gids_const.extent(0); ++j)
							std::cout << send_gids_const(j,0) << " ";
						std::cout << std::endl;
					}
				}
	
				out << " iteration " << p << ": communications complete\n";

				// build the cyclic perturbation map for the particles' multivectors
				Teuchos::RCP<Compadre::map_type> cycle_map = rcp(new Compadre::map_type(nParticles, recView.getRawPtr(),
					 recvSize, 0, comm));

				for (int i = 0; i < numProcs; ++i) {
					send_gids_const = cycle_map->getMyGlobalIndices();
					comm->barrier();

					if (procRank == i) {
						std::cout << "proc " << i << " has global indices in map (" << recvSize << "):" << std::endl << "\t";
						for (int j = 0; j < send_gids_const.extent(0); ++j)
							std::cout << send_gids_const(j) << " ";
						std::cout << std::endl;
					}
				}

				Teuchos::RCP<Compadre::importer_type> cycle_importer = rcp(new Compadre::importer_type(particles->getCoordMap(),
					cycle_map));
	//             Teuchos::RCP<Tpetra::Export<Compadre::local_index_type, Compadre::global_index_type>> cycle_exporter =
	//                 rcp(new Tpetra::Export<Compadre::local_index_type, Compadre::global_index_type>(particles->getCoordMap(),
	//                     cycle_map));

				Teuchos::RCP<Compadre::mvec_type> srcCrdMVec = Teuchos::rcp(new Compadre::mvec_type(cycle_map, 3, false));
				Teuchos::RCP<Compadre::mvec_type> srcPoissonMVec = Teuchos::rcp(new Compadre::mvec_type(cycle_map, 1, false));
				Teuchos::RCP<Compadre::mvec_type> srcAreaMVec = Teuchos::rcp(new Compadre::mvec_type(cycle_map, 1, false));

				srcCrdMVec->doImport(*(particles->getCoordsConst()->getPts()), *cycle_importer, Tpetra::CombineMode::REPLACE);
				srcPoissonMVec->doImport(*(fields[srcID]->getLocalVectorVals()), *cycle_importer, Tpetra::CombineMode::REPLACE);
				srcAreaMVec->doImport(*(fields[areaID]->getLocalVectorVals()), *cycle_importer, Tpetra::CombineMode::REPLACE);

	//             srcCrdMVec->doExport(*(particles->getCoordsConst()->getPts()), *cycle_exporter, Tpetra::CombineMode::REPLACE);
	//             srcPoissonMVec->doExport(*(fields[srcID]->getLocalVectorVals()), *cycle_exporter, Tpetra::CombineMode::REPLACE);
	//             srcAreaMVec->doExport(*(fields[areaID]->getLocalVectorVals()), *cycle_exporter, Tpetra::CombineMode::REPLACE);

				out << " iteration " << p << " imports complete." << std::endl;

				sourceCoords = srcCrdMVec->getLocalView<device_view_type>();
				poissonSource = srcPoissonMVec->getLocalView<device_view_type>();
				particleArea = srcAreaMVec->getLocalView<device_view_type>();

	//             for (int i = 0; i < numProcs; ++i) {
	//                 comm->barrier();
	//
	//                 if (procRank == i) {
	//                     std::cout << "proc " << i << " has f = (" << send_gids_const.extent(0) << "):" << std::endl << "\t";
	//                     for (int j = 0; j < sourceCoords.extent(0); ++j) {
	//                         std::cout << "(";
	//                         for (int k = 0; k < 3; ++k)
	//                             std::cout << sourceCoords(j,k) << (k < 2 ? ", " : ")");
	//                         std::cout << ", ";
	//                     }
	//                     std::cout << std::endl;
	//                 }
	//             }
	//
				GFTime->start();
				Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,particles->getCoords()->nLocal()),
						greensFn(targetCoords, sourceCoords, targetPotential, poissonSource, particleArea));
				GFTime->stop();
			}
		}
		fields[errID]->updateMultiVector(1.0, fields[computID], -1.0, fields[exactID], 0.0);
		std::vector<scalar_type> infErr = fields[errID]->normInf();

		fm.setWriter("sphPoissonTest.nc", particles);
		fm.write();

		out << "PROGRAM COMPLETE\n";
		out << "\tinfErr = " << infErr[0] << std::endl;
	}

	Teuchos::TimeMonitor::summarize();
    Kokkos::finalize();
return 0;
}
