#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_Core.hpp>
#include <Kokkos_Core.hpp>
#include <Compadre_GlobalConstants.hpp>
#include <Compadre_ProblemT.hpp>
#include <Compadre_ParticlesT.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_EuclideanCoordsT.hpp>
#include <Compadre_SphericalCoordsT.hpp>
#include <Compadre_NeighborhoodT.hpp>
#include <Compadre_FieldT.hpp>
#include <Compadre_XyzVector.hpp>
#include <Compadre_AnalyticFunctions.hpp>
#include <Compadre_FileIO.hpp>
#include <Compadre_ParameterManager.hpp>

#include <Compadre_PinnedGraphLaplacian_Operator.hpp>
#include <Compadre_PinnedGraphLaplacian_Sources.hpp>
#include <Compadre_PinnedGraphLaplacian_BoundaryConditions.hpp>

#include <iostream>

typedef int LO;
typedef long GO;
typedef double ST;


int main (int argc, char* args[]) {

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

    Teuchos::GlobalMPISession mpi(&argc, &args);
    Teuchos::oblackholestream bstream;
    Teuchos::RCP<const Teuchos::Comm<int> > comm = Teuchos::DefaultComm<int>::getComm();
    
    Kokkos::initialize(argc, args);

    // test of analytic functions and their derivatives
    {
        std::string testfilename(parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file prefix") + parameters->get<Teuchos::ParameterList>("io").get<std::string>("input file"));

        Teuchos::RCP<Compadre::ParticlesT> particles =
                Teuchos::rcp( new Compadre::ParticlesT(parameters, comm));

        Compadre::FileManager fm;
        fm.setReader(testfilename, particles);
        fm.read();

		auto coords = (Compadre::CoordsT*)(particles->getCoordsConst());

        auto x = Teuchos::rcp(new Compadre::LinearInX());
        auto y = Teuchos::rcp(new Compadre::LinearInY());
        auto z = Teuchos::rcp(new Compadre::LinearInZ());

        auto f_x = pow(x,7)*y-27*pow(z,3)*y;
        //auto f_x = x*x*x*x*x*x*x*y-27*z*z*z*y;

        auto func_df_dx = 7*pow(x,6)*y;
        auto func_df_dy = pow(x,7)-27*pow(z,3);
        auto func_df_dz = -81*pow(z,2)*y;

        auto func_d2f_dxx = 42*pow(x,5)*y;
        auto func_d2f_dxy = 7*pow(x,6);
        auto func_d2f_dxz = 0*x;

        auto func_d2f_dyx = 7*pow(x,6);
        auto func_d2f_dyy = 0*x;
        auto func_d2f_dyz = -81*pow(z,2);

        auto func_d2f_dzx = 0*y;
        auto func_d2f_dzy = -81*pow(z,2);
        auto func_d2f_dzz = -162*z*y;

        double tol=1e-14;
        for(int j =0; j<coords->nLocal(); j++){

            auto xyz = coords->getLocalCoords(j, true /* use_halo */, true);

            // function evaluations of exact solutions (derivatives)
            auto f_df_dx = func_df_dx->evalScalar(xyz,0,parameters->get<Teuchos::ParameterList>("time").get<double>("t_end"));
            auto f_df_dy = func_df_dy->evalScalar(xyz,0,parameters->get<Teuchos::ParameterList>("time").get<double>("t_end"));
            auto f_df_dz = func_df_dz->evalScalar(xyz,0,parameters->get<Teuchos::ParameterList>("time").get<double>("t_end"));

            // function evaluations of exact solutions (hessian)
            auto f_d2f_dxx = func_d2f_dxx->evalScalar(xyz,0,parameters->get<Teuchos::ParameterList>("time").get<double>("t_end"));
            auto f_d2f_dxy = func_d2f_dxy->evalScalar(xyz,0,parameters->get<Teuchos::ParameterList>("time").get<double>("t_end"));
            auto f_d2f_dxz = func_d2f_dxz->evalScalar(xyz,0,parameters->get<Teuchos::ParameterList>("time").get<double>("t_end"));
            auto f_d2f_dyx = func_d2f_dyx->evalScalar(xyz,0,parameters->get<Teuchos::ParameterList>("time").get<double>("t_end"));
            auto f_d2f_dyy = func_d2f_dyy->evalScalar(xyz,0,parameters->get<Teuchos::ParameterList>("time").get<double>("t_end"));
            auto f_d2f_dyz = func_d2f_dyz->evalScalar(xyz,0,parameters->get<Teuchos::ParameterList>("time").get<double>("t_end"));
            auto f_d2f_dzx = func_d2f_dzx->evalScalar(xyz,0,parameters->get<Teuchos::ParameterList>("time").get<double>("t_end"));
            auto f_d2f_dzy = func_d2f_dzy->evalScalar(xyz,0,parameters->get<Teuchos::ParameterList>("time").get<double>("t_end"));
            auto f_d2f_dzz = func_d2f_dzz->evalScalar(xyz,0,parameters->get<Teuchos::ParameterList>("time").get<double>("t_end"));
            
            // exact solution directly computed (derivatives and hessians)
            auto df_dx = 7*pow(xyz.x,6)*xyz.y;
            auto df_dy = pow(xyz.x,7)-27*pow(xyz.z,3);
            auto df_dz = -81*pow(xyz.z,2)*xyz.y;

            auto d2f_dxx = 42*pow(xyz.x,5)*xyz.y;
            auto d2f_dxy = 7*pow(xyz.x,6);
            auto d2f_dxz = 0*xyz.x;

            auto d2f_dyx = 7*pow(xyz.x,6);
            auto d2f_dyy = 0*xyz.x;
            auto d2f_dyz = -81*pow(xyz.z,2);

            auto d2f_dzx = 0*xyz.y;
            auto d2f_dzy = -81*pow(xyz.z,2);
            auto d2f_dzz = -162*xyz.z*xyz.y;




            // compute derivatives and hessians
            auto df_computed = f_x->evalScalarDerivative(xyz,0,parameters->get<Teuchos::ParameterList>("time").get<double>("t_end"));
            auto d2f_computed = f_x->evalScalarHessian(xyz,0,parameters->get<Teuchos::ParameterList>("time").get<double>("t_end"));

            //printf("df_exact={%f,%f,%f} df_computed={%f,%f,%f}\n", df_dx, df_dy, df_dz, df_computed[0], df_computed[1], df_computed[2]);
            //printf("d2f_exact={(%f,%f,%f),(%f,%f,%f),(%f,%f,%f)} d2f_computed={(%f,%f,%f),(%f,%f,%f),(%f,%f,%f)}\n", d2f_dxx, d2f_dxy, d2f_dxz, d2f_dyx, d2f_dyy, d2f_dyz, d2f_dzx, d2f_dzy, d2f_dzz, d2f_computed[0][0], d2f_computed[0][1], d2f_computed[0][2], d2f_computed[1][0], d2f_computed[1][1], d2f_computed[1][2], d2f_computed[2][0], d2f_computed[2][1], d2f_computed[2][2]);

            // compare first derivatives
            TEUCHOS_ASSERT(std::abs(df_dx-df_computed.x)<tol);
            TEUCHOS_ASSERT(std::abs(df_dy-df_computed.y)<tol);
            TEUCHOS_ASSERT(std::abs(df_dz-df_computed.z)<tol);

            TEUCHOS_ASSERT(std::abs(df_dx-f_df_dx)<tol);
            TEUCHOS_ASSERT(std::abs(df_dy-f_df_dy)<tol);
            TEUCHOS_ASSERT(std::abs(df_dz-f_df_dz)<tol);

            // compare hessians
            TEUCHOS_ASSERT(std::abs(d2f_dxx-f_d2f_dxx)<tol);
            TEUCHOS_ASSERT(std::abs(d2f_dxy-f_d2f_dxy)<tol);
            TEUCHOS_ASSERT(std::abs(d2f_dxz-f_d2f_dxz)<tol);
            TEUCHOS_ASSERT(std::abs(d2f_dyx-f_d2f_dyx)<tol);
            TEUCHOS_ASSERT(std::abs(d2f_dyy-f_d2f_dyy)<tol);
            TEUCHOS_ASSERT(std::abs(d2f_dyz-f_d2f_dyz)<tol);
            TEUCHOS_ASSERT(std::abs(d2f_dzx-f_d2f_dzx)<tol);
            TEUCHOS_ASSERT(std::abs(d2f_dzy-f_d2f_dzy)<tol);
            TEUCHOS_ASSERT(std::abs(d2f_dzz-f_d2f_dzz)<tol);


            TEUCHOS_ASSERT(std::abs(d2f_dxx-d2f_computed[0][0])<tol);
            TEUCHOS_ASSERT(std::abs(d2f_dxy-d2f_computed[0][1])<tol);
            TEUCHOS_ASSERT(std::abs(d2f_dxz-d2f_computed[0][2])<tol);
            TEUCHOS_ASSERT(std::abs(d2f_dyx-d2f_computed[1][0])<tol);
            TEUCHOS_ASSERT(std::abs(d2f_dyy-d2f_computed[1][1])<tol);
            TEUCHOS_ASSERT(std::abs(d2f_dyz-d2f_computed[1][2])<tol);
            TEUCHOS_ASSERT(std::abs(d2f_dzx-d2f_computed[2][0])<tol);
            TEUCHOS_ASSERT(std::abs(d2f_dzy-d2f_computed[2][1])<tol);
            TEUCHOS_ASSERT(std::abs(d2f_dzz-d2f_computed[2][2])<tol);


        }
    }
    Kokkos::finalize();
return 0;
}
