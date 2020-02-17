#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_Core.hpp>
#include <Kokkos_Core.hpp>
#include <Compadre_GlobalConstants.hpp>
#include <Compadre_ProblemT.hpp>
#include <Compadre_ParticlesT.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_RemapManager.hpp>
#include <Compadre_EuclideanCoordsT.hpp>
#include <Compadre_SphericalCoordsT.hpp>
#include <Compadre_FieldT.hpp>
#include <Compadre_XyzVector.hpp>
#include <Compadre_AnalyticFunctions.hpp>
#include <Compadre_FileIO.hpp>
#include <Compadre_ParameterManager.hpp>
#include <Compadre_Evaluator.hpp>
#include <Compadre_NeighborhoodT.hpp>

#include <iostream>

using namespace Compadre;
typedef int LO;
typedef long GO;
typedef double ST;

typedef Compadre::NeighborhoodT neighbors_type;

typedef Compadre::CoordsT coords_type;
typedef Compadre::XyzVector xyz_type;

int main (int argc, char* args[]) {

    int dim = 2;

	Teuchos::GlobalMPISession mpi(&argc, &args);
	Teuchos::oblackholestream bstream;
	Teuchos::RCP<const Teuchos::Comm<int> > comm = Teuchos::DefaultComm<int>::getComm();

	Kokkos::initialize(argc, args);
	
    {
	Teuchos::RCP<Teuchos::Time> ParameterTime = Teuchos::TimeMonitor::getNewCounter ("Parameter Initialization");
	Teuchos::RCP<Teuchos::Time> MiscTime = Teuchos::TimeMonitor::getNewCounter ("Miscellaneous");
	Teuchos::RCP<Teuchos::Time> NormTime = Teuchos::TimeMonitor::getNewCounter ("Norm calculation");
	Teuchos::RCP<Teuchos::Time> CoordinateInsertionTime = Teuchos::TimeMonitor::getNewCounter ("Coordinate Insertion Time");
	Teuchos::RCP<Teuchos::Time> ParticleInsertionTime = Teuchos::TimeMonitor::getNewCounter ("Particle Insertion Time");

	//********* BASIC PARAMETER SETUP FOR ANY PROBLEM
	ParameterTime->start();
	Teuchos::RCP<Compadre::ParameterManager> parameter_manager;
	if (argc > 1)
		parameter_manager = Teuchos::rcp(new Compadre::ParameterManager(argc, args));
	else {
		parameter_manager = Teuchos::rcp(new Compadre::ParameterManager());
		std::cout << "WARNING: No parameter list given. Default parameters used." << std::endl;
	}
	if (parameter_manager->helpRequested()) return 0;
	if (parameter_manager->parseError()) return -1;

	Teuchos::RCP<Teuchos::ParameterList> parameters = parameter_manager->getList();
	//parameters->print();
	ParameterTime->stop();
	// There is a balance between "neighbors needed multiplier" and "cutoff multiplier"
	// Their product must increase somewhat with "porder", but "neighbors needed multiplier"
	// requires more of an increment than "cutoff multiplier" to achieve the required threshold (the other staying the same)
	//*********
    
    Teuchos::RCP<Compadre::ParticlesT> particles =
    		Teuchos::rcp( new Compadre::ParticlesT(parameters, comm));

    double h_size = 0.083;
    int index_manipulated = 92;
    {
        Compadre::FileManager fm;
        

        // NEED Cell centered data
        fm.setReader("../test_data/grids/cells.nc", particles);
        fm.read();
	    // build halo data
	    ST halo_size = h_size *
	    		( parameters->get<Teuchos::ParameterList>("halo").get<double>("multiplier")
	    				+ (ST)(parameters->get<Teuchos::ParameterList>("remap").get<int>("porder")));
	    particles->zoltan2Initialize();
	    particles->buildHalo(halo_size);
        particles->getFieldManager()->createField(1, "remapped_basis", "na");

        // get view and set just one value to 1
        
        host_view_type values_holder = particles->getFieldManagerConst()->
            getFieldByName("remapped_basis")->getMultiVectorPtrConst()->getLocalView<const host_view_type>();
        particles->getFieldManager()->
            getFieldByName("remapped_basis")->getMultiVectorPtr()->putScalar(0.0);

        values_holder(index_manipulated,0) = 1.0;

	    particles->getFieldManager()->updateFieldsHaloData();
    }


    Teuchos::RCP<Compadre::ParticlesT> eval_particles =
    		Teuchos::rcp( new Compadre::ParticlesT(parameters, comm));
    //{
    //    //Compadre::FileManager fm;
    //    //fm.setReader("../test_data/grids/patch.nc", eval_particles);
    //    ////fm.setReader("../test_data/grids/cells.nc", eval_particles);
    //    //fm.read();
	//    // build halo data
	//    ST halo_size = h_size *
	//    		( parameters->get<Teuchos::ParameterList>("halo").get<double>("multiplier")
	//    				+ (ST)(parameters->get<Teuchos::ParameterList>("remap").get<int>("porder")));

	//	//std::vector<Compadre::XyzVector> verts_to_insert;
	//	//verts_to_insert.push_back(Compadre::XyzVector(0.266667,0.35,0));
    //    //eval_particles->resize(150);

    //    //auto coords = particles2->getCoords();
	//	//coords->insertCoords(verts_to_insert);
	//	//particles2->resetWithSameCoords(); // must be called because particles doesn't know about coordinate insertions

    //    eval_particles->getFieldManager()->createField(1, "remapped_basis2", "na");
	//    eval_particles->zoltan2Initialize();
	//    eval_particles->buildHalo(halo_size);
    //    //particles2->getFieldManager()->createField(1, "remapped_basis", "na");
	//    //particles2->getFieldManager()->updateFieldsHaloData();
    //}
    auto trg_particles = particles;
    auto src_particles = particles;


    { // SETUP GMLS AND SOLVE

        std::string method =parameters->get<Teuchos::ParameterList>("neighborhood").get<std::string>("method");
        transform(method.begin(), method.end(), method.begin(), ::tolower);
    
    
        local_index_type maxLeaf = parameters->get<Teuchos::ParameterList>("neighborhood").get<int>("max leaf");
        auto _neighborhoodInfo = Teuchos::rcp_static_cast<neighbors_type>(Teuchos::rcp(
                    new neighbors_type(src_particles.getRawPtr(), trg_particles.getRawPtr(), 
                        false /*use physical coords*/, maxLeaf)));
    
        local_index_type neighbors_needed;
    
        std::string solver_type_to_lower = parameters->get<Teuchos::ParameterList>("remap").get<std::string>("dense solver type");
        transform(solver_type_to_lower.begin(), solver_type_to_lower.end(), solver_type_to_lower.begin(), ::tolower);
    
        if (solver_type_to_lower == "manifold") {
            int max_order = std::max(parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"), parameters->get<Teuchos::ParameterList>("remap").get<int>("curvature porder"));
            neighbors_needed = GMLS::getNP(max_order, dim-1);
        } else {
            neighbors_needed = GMLS::getNP(parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"), dim);
        }
    
 		_neighborhoodInfo->constructAllNeighborLists(1e+16,
            parameters->get<Teuchos::ParameterList>("neighborhood").get<std::string>("search type"),
            true /*dry run for sizes*/,
            neighbors_needed,
            parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("cutoff multiplier"),
            parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("size"),
            parameters->get<Teuchos::ParameterList>("neighborhood").get<bool>("uniform radii"),
            parameters->get<Teuchos::ParameterList>("neighborhood").get<double>("radii post search scaling"),
            false);
        Kokkos::fence();
    
    
        //****************
        //
        //  Copying data from particles (std::vector's, multivectors, etc....) to views used by local reconstruction class
        //
        //****************
    
        size_t max_num_neighbors = _neighborhoodInfo->computeMaxNumNeighbors(false /*local process max*/);
    
        // generate the interpolation operator and call the coefficients needed (storing them)
        const coords_type* target_coords = trg_particles->getCoordsConst();
        const coords_type* source_coords = src_particles->getCoordsConst();
    
        Kokkos::View<int**> kokkos_neighbor_lists("neighbor lists", target_coords->nLocal(), max_num_neighbors+1);
        Kokkos::View<int**>::HostMirror kokkos_neighbor_lists_host = Kokkos::create_mirror_view(kokkos_neighbor_lists);
    
        // fill in the neighbor lists into a kokkos view. First entry is # of neighbors for that target
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
            const int num_i_neighbors = _neighborhoodInfo->getNumNeighbors(i);
            for (int j=1; j<num_i_neighbors+1; ++j) {
                kokkos_neighbor_lists_host(i,j) = _neighborhoodInfo->getNeighbor(i,j-1);
            }
            kokkos_neighbor_lists_host(i,0) = num_i_neighbors;
        });
    
        Kokkos::View<double**> kokkos_augmented_source_coordinates("source_coordinates", source_coords->nLocal(true /* include halo in count */), source_coords->nDim());
        Kokkos::View<double**>::HostMirror kokkos_augmented_source_coordinates_host = Kokkos::create_mirror_view(kokkos_augmented_source_coordinates);
    
        // fill in the source coords, adding regular with halo coordiantes into a kokkos view
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,source_coords->nLocal(true /* include halo in count*/)), KOKKOS_LAMBDA(const int i) {
            xyz_type coordinate = source_coords->getLocalCoords(i, true /*include halo*/, false);
            kokkos_augmented_source_coordinates_host(i,0) = coordinate.x;
            kokkos_augmented_source_coordinates_host(i,1) = coordinate.y;
            kokkos_augmented_source_coordinates_host(i,2) = coordinate.z;
        });
    
        Kokkos::View<double**> kokkos_target_coordinates("target_coordinates", target_coords->nLocal(), target_coords->nDim());
        Kokkos::View<double**>::HostMirror kokkos_target_coordinates_host = Kokkos::create_mirror_view(kokkos_target_coordinates);
        // fill in the target, adding regular coordiantes only into a kokkos view
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
            xyz_type coordinate = target_coords->getLocalCoords(i, false /*include halo*/, false);
            kokkos_target_coordinates_host(i,0) = coordinate.x;
            kokkos_target_coordinates_host(i,1) = coordinate.y;
            kokkos_target_coordinates_host(i,2) = coordinate.z;
        });
    
        auto epsilons = _neighborhoodInfo->getHSupportSizes()->getLocalView<const host_view_type>();
        Kokkos::View<double*> kokkos_epsilons("epsilons",target_coords->nLocal());
        Kokkos::View<double*>::HostMirror kokkos_epsilons_host = Kokkos::create_mirror_view(kokkos_epsilons);
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
            kokkos_epsilons_host(i) = epsilons(i,0);
        });




        auto num_triangles = target_coords->nLocal(false);
        int rows = 40;
        int num_per_triangle = rows*(rows+1)/2;
        const host_view_type coords_values_holder = src_particles->getFieldManagerConst()->
                getFieldByName("basis")->getMultiVectorPtrConst()->getLocalView<const host_view_type>();


        Kokkos::View<double**> kokkos_eval_coordinates("eval_coordinates", num_triangles*num_per_triangle, 3);
        Kokkos::View<double**>::HostMirror kokkos_eval_coordinates_host = Kokkos::create_mirror_view(kokkos_eval_coordinates);
        Kokkos::View<int**> kokkos_eval_lists("eval lists", target_coords->nLocal(), num_per_triangle+1);
        Kokkos::View<int**>::HostMirror kokkos_eval_lists_host = Kokkos::create_mirror_view(kokkos_eval_lists);

        int eval_index = 0;
        for (int i=0; i<num_triangles; ++i) {
            kokkos_eval_lists_host(i,0) = num_per_triangle;
            int local_created = 0;
            for (int k=0; k<rows; ++k) {
                for (int l=0; l<=k; ++l) {
                    double t_y = l*(1./(rows-1));
                    double t_x = 1-k*(1./(rows-1));
                    printf("t_x: %f t_y: %f\n", t_x, t_y);
                    // maps (1,0) onto first point  -> 
                    // maps (0,1) onto second point -> 
                    double a_01 = coords_values_holder(i,0)-coords_values_holder(i,4);
                    double a_11 = coords_values_holder(i,1)-coords_values_holder(i,5);
                    double a_00 = coords_values_holder(i,2)-coords_values_holder(i,4);
                    double a_10 = coords_values_holder(i,3)-coords_values_holder(i,5);
                    kokkos_eval_coordinates_host(eval_index,0) = a_00*t_x + a_01*t_y + coords_values_holder(i,4);
                    kokkos_eval_coordinates_host(eval_index,1) = a_10*t_x + a_11*t_y + coords_values_holder(i,5);
                    kokkos_eval_lists_host(i,local_created+1) = eval_index;
                    local_created++;
                    eval_index++;
                }
            }
        }

        //// fill in the target, adding regular coordiantes only into a kokkos view
        //Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,eval_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
        //    xyz_type coordinate = eval_coords->getLocalCoords(i, false /*include halo*/, false);
        //    kokkos_eval_coordinates_host(i,0) = coordinate.x;
        //    kokkos_eval_coordinates_host(i,1) = coordinate.y;
        //    kokkos_eval_coordinates_host(i,2) = coordinate.z;
        //});

        //// fill in the neighbor lists into a kokkos view. First entry is # of neighbors for that target
        //Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,target_coords->nLocal()), KOKKOS_LAMBDA(const int i) {
        //    if (i==10) {
        //        for (int j=1; j<eval_coords->nLocal()+1; ++j) {
        //            kokkos_eval_lists_host(i,j) = j-1;
        //        }
        //        kokkos_eval_lists_host(i,0) = eval_coords->nLocal();
        //    } else {
        //        kokkos_eval_lists_host(i,0) = 0;
        //    }
        //    //if (i==0) {
        //    //    printf("NEIGHBORS for 0: %d\n", kokkos_eval_lists_host(i,0));
        //    //}
        //});
    
        //****************
        //
        //  End of data copying
        //
        //****************



        auto _GMLS = Teuchos::rcp(new GMLS(ReconstructionSpace::ScalarTaylorPolynomial,
                PointSample,
                parameters->get<Teuchos::ParameterList>("remap").get<int>("porder"),
                dim,
                parameters->get<Teuchos::ParameterList>("remap").get<std::string>("dense solver type"),
                parameters->get<Teuchos::ParameterList>("remap").get<std::string>("problem type"),
                parameters->get<Teuchos::ParameterList>("remap").get<std::string>("constraint type"),
                parameters->get<Teuchos::ParameterList>("remap").get<int>("curvature porder")));
    
        _GMLS->setProblemData(kokkos_neighbor_lists_host,
                kokkos_augmented_source_coordinates_host,
                kokkos_target_coordinates,
                kokkos_epsilons_host);

        // set up additional sites to evaluate target operators
        _GMLS->setAdditionalEvaluationSitesData(kokkos_eval_lists_host, kokkos_eval_coordinates_host);

    
        _GMLS->setWeightingType(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("weighting type"));
        _GMLS->setWeightingPower(parameters->get<Teuchos::ParameterList>("remap").get<int>("weighting power"));
        _GMLS->setCurvatureWeightingType(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("curvature weighting type"));
        _GMLS->setCurvatureWeightingPower(parameters->get<Teuchos::ParameterList>("remap").get<int>("curvature weighting power"));
		_GMLS->setOrderOfQuadraturePoints(parameters->get<Teuchos::ParameterList>("remap").get<int>("quadrature order"));
		_GMLS->setDimensionOfQuadraturePoints(parameters->get<Teuchos::ParameterList>("remap").get<int>("quadrature dimension"));
		_GMLS->setQuadratureType(parameters->get<Teuchos::ParameterList>("remap").get<std::string>("quadrature type"));

        std::vector<TargetOperation> lro(1);
        lro[0]=TargetOperation::ScalarPointEvaluation;
    
        _GMLS->addTargets(lro);
        _GMLS->generateAlphas(); // all operations requested

        Evaluator remap_agent(_GMLS.getRawPtr());










    
        const host_view_type source_values_holder = src_particles->getFieldManagerConst()->
                getFieldByName("remapped_basis")->getMultiVectorPtrConst()->getLocalView<const host_view_type>();
        const host_view_type source_halo_values_holder = src_particles->getFieldManagerConst()->
                getFieldByName("remapped_basis")->getHaloMultiVectorPtrConst()->getLocalView<const host_view_type>();
    
        const local_index_type num_local_particles = source_values_holder.extent(0);
    
    
        // fill in the source values, adding regular with halo values into a kokkos view
        Kokkos::View<double**, Kokkos::HostSpace> kokkos_source_values("source values", source_coords->nLocal(true /* include halo in count */), 1);
    
        // copy data local owned by this process
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,source_coords->nLocal(false /* include halo in count*/)), KOKKOS_LAMBDA(const int j) {
            for (local_index_type k=0; k<1; ++k) {
                kokkos_source_values(j,k) = source_values_holder(j,k);
            }
        });
        // copy halo data 
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(source_coords->nLocal(false /* not includinghalo in count*/), source_coords->nLocal(true /*include halo in count */)), KOKKOS_LAMBDA(const int j) {
            for (local_index_type k=0; k<1; ++k) {
                kokkos_source_values(j,k) = source_halo_values_holder(j-num_local_particles,k);
            }
        });
        Kokkos::fence();








		std::vector<Compadre::XyzVector> verts_to_insert;
        // loop through extras list for each cell
        for (int i=0; i<num_triangles; ++i) {
            for (int j=0; j<num_per_triangle; ++j) {
                auto this_index = kokkos_eval_lists_host(i,j+1);
		        verts_to_insert.push_back(Compadre::XyzVector(kokkos_eval_coordinates_host(this_index,0), kokkos_eval_coordinates_host(this_index,1), kokkos_eval_coordinates_host(this_index,2)));
            }
        }
        auto eval_coords = eval_particles->getCoords();
		eval_coords->insertCoords(verts_to_insert);
		eval_particles->resetWithSameCoords(); 


        eval_particles->getFieldManager()->createField(1, "remapped_basis2", "na");
	    eval_particles->zoltan2Initialize();
	    ST halo_size = h_size *
	    		( parameters->get<Teuchos::ParameterList>("halo").get<double>("multiplier")
	    				+ (ST)(parameters->get<Teuchos::ParameterList>("remap").get<int>("porder")));
	    eval_particles->buildHalo(halo_size);


        // should write from here nper_triangle * num_triangles ---->  a grid of that size
        host_view_type target_values = eval_particles->getFieldManager()->
                getFieldByName("remapped_basis2")->getMultiVectorPtr()->getLocalView<host_view_type>();



        for (int j=0; j<num_per_triangle; ++j) {
            auto output_vector = remap_agent.applyAlphasToDataAllComponentsAllTargetSites<double**, Kokkos::HostSpace>
                (kokkos_source_values, TargetOperation::ScalarPointEvaluation, PointSample, true, j+1);
            for (int i=0; i<num_triangles; ++i) {
                target_values(j + i*num_per_triangle,0) = output_vector(i,0);
            }
        }

        //for (int j=0; j<kokkos_eval_lists_host(index_manipulated,0); ++j) {
        //    auto output_vector = remap_agent.applyAlphasToDataAllComponentsAllTargetSites<double**, Kokkos::HostSpace>
        //        (kokkos_source_values, TargetOperation::ScalarPointEvaluation, PointSample, true, j+1);
        //    // copy remapped solution back into field
        //    //target_values(kokkos_eval_lists_host(index_manipulated,j+1),0) = output_vector(index_manipulated,0);
        //}
    
    



    }







	if (comm->getRank()==0) parameters->print();
	//Teuchos::TimeMonitor::summarize();

    {
        //Compadre::FileManager fm;
	    //fm.setWriter("./out.nc", particles);
	    //fm.write();
        Compadre::FileManager fm;
	    fm.setWriter("./out.nc", eval_particles);
	    fm.write();
    }


    }
	Kokkos::finalize();
return 0;
}

