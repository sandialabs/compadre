#include "GMLS_boost.hpp"

#include <cstdio>

#ifdef COMPADRE_USE_BOOST
#include <boost/numeric/ublas/lu.hpp>

vector_type GMLS_T_BOOST::get_relative_coord(GMLS_ParticleT_BOOST& t_particle){
    return t_particle.get_coordinates()-m_particle.get_coordinates();
}

void GMLS_T_BOOST::invert_M(matrix_type& M_data,  int N){

	boost::numeric::ublas::permutation_matrix<matrix_type::size_type> p(N);
	int attempt = boost::numeric::ublas::lu_factorize(M_data, p);
	boost::numeric::ublas::lu_substitute(M_data, p, m_M_inv);

}


double GMLS_T_BOOST::Wab(double r,double h,int sigma){
    
    int p = 8;
    // return double(1.0-std::abs(r/(3*h))>0.0);
    return pow(1.0-std::abs(r/(3*h)),p)*double(1.0-std::abs(r/(3*h))>0.0);
    //return pow((r+1e-15)/(3*h),-p)*double(1-std::abs(r/(3*h))>0);
    //return (1.0-std::abs(r/(3.*h)))*double(1.0-std::abs(r/(3*h))>0)/3.0*h;
   // return p;
}


double GMLS_T_BOOST::factorial(int n){
    double f = 1.0;
    int i;
    for(i = 1; i <= n; i++){
        f *=i;
    }
    return f;
}


vector_type GMLS_T_BOOST::calcWij(GMLS_ParticleT_BOOST& neighbor){
	vector_type delta(NP, 0);
	vector_type relative_coord = get_relative_coord(neighbor);
    int Porder = Poly_order;
    double cutoff_p = m_epsilon;
    int alphax, alphay, alphaz;
    double alphaf;
    int i = 0;
    for (int n = 0; n <= Porder; n++){
        for (alphax = 0; alphax <= n; alphax++){
            int s = n - alphax;
            
            for (alphay = 0; alphay <= s; alphay++){
                alphaz = s - alphay;
                alphaf = factorial(alphax)*factorial(alphay)*factorial(alphaz);
                delta[i] = std::pow(relative_coord[0]/cutoff_p,alphax)
							*std::pow(relative_coord[1]/cutoff_p,alphay)
							*std::pow(relative_coord[2]/cutoff_p,alphaz)/alphaf;
                i++;
            }
            
        }
    }
    return delta;
}


void GMLS_T_BOOST::get_M_inverse(){
    int i;
    double r, W;
    double m_kl;
    vector_type delta(NP, 0);
    matrix_type M_data(NP, NP, 0);
    
    for(i = 0; i < m_Nneighbors; i++){
        r = m_particle.distance(m_neighbor_data[i]);

        delta = calcWij(m_neighbor_data[i]);
        W = Wab(r,m_epsilon,2);

        // fastest so far is to keep for loops rather than a matrix update
        // prestoring m_kl then adding it
        for(int k = 0; k < NP; k++){
            for(int l = 0; l < NP; l++){
                m_kl = M_data(k,l);
                M_data(k,l) = m_kl + delta[l]*delta[k]*W;
            }
        }
    }
    
    invert_M(M_data, NP);
//    getSVD(M_data, NP);
}


void GMLS_T_BOOST::print_neighbors(){
    fprintf(stdout, "My neighbors: %f", 0.0f);
    
    // for(int i = 1; i < m_Nneighbors; i++){
      //  fprintf(stdout, ", %f ", gsl_vector_get(t_particle->get_coordinates(), i));
        
	//}
    fprintf(stdout, "\n");
}


void GMLS_T_BOOST::print_neighbor_data(){
    int i;
    
    fprintf(stdout, "my location: ");
    vector_type my_loc = m_particle.get_coordinates();
    for(i = 0; i < 3; i++){
        
        fprintf(stdout, " %f ", my_loc[i]);
    }
    fprintf(stdout, "\n");



    
    for(int j = 0; j < m_Nneighbors; j++){
      fprintf(stdout, "%d", j); 
    for(i = 0; i < 3; i++){
        
      fprintf(stdout, "%f ", m_neighbor_data[j].get_coordinates()[i]);
    }
    fprintf(stdout, "\n");
    }
}


//void GMLS_T_BOOST::getSVD(matrix_type& M_data, int NP){
//   // gsl_matrix * M_inv = gsl_matrix_alloc(NP, NP);
//	boost::numeric::ublas::permutation_matrix<matrix_type::size_type> p(N);
//	int attempt = boost::numeric::ublas::lu_factorize(M_data, p);
//	boost::numeric::ublas::lu_substitute(M_data, p, m_M_inv);
//
//    double M_inv_ae;
//
////    matrix_type U(NP,NP,0);
//	matrix_type U(M_data);
//    matrix_type V(NP,NP,0);
//	vector_type S(NP,0);
//	vector_type work(NP,0);
//
////        gsl_matrix* U;
////        gsl_matrix* V;
////        gsl_vector* S;
////        gsl_vector* work;
//
//	int i,j;//,N;
//
////        work = gsl_vector_alloc(NP);
////        S    = gsl_vector_alloc(NP);
////        U    = gsl_matrix_alloc(NP,NP);
////        V    = gsl_matrix_alloc(NP,NP);
//
//
////        gsl_matrix_memcpy(U, M_data);
//
//
//        gsl_linalg_SV_decomp(U,V,S,work);
//        svd_moore_penrose_pseudoinverse
//
//
//        /*
//         //For post processing condition number
//         printf("S = [");
//         for(int i = 0; i < N; i++){
//         printf(" %g",gsl_vector_get(S,i));
//         }
//         printf("]\n");
//         cout << "condNumber: " << gsl_vector_get(S,0)/gsl_vector_get(S,N-1) << endl;
//         */
//        stl_vector_type SS;
//        SS.resize(NP);
//        for(int i = 0; i < NP; i++){
//            SS[i] = S[i];//gsl_vector_get(S,i);
//            SS[i] = (std::abs(SS[i])>1e-10) ? 1.0/SS[i] : 0.0;
//            //      SS[i] = 1.0/(SS[i]+1e-12);
//        }
//
//       // gsl_matrix_set_zero(m_M_inv);
//
//        for(int b = 0; b < NP; b++){
//            for(int a = 0; a < NP; a++){
//                for(int e = 0; e < NP; e++){
//                   // M_inv_ae = gsl_matrix_get(m_M_inv, a, e);
//                    M_inv_ae = m_M_inv.at_element(a,e);
//                    m_M_inv(a,e) += V(a,b)*U(e,b)*SS[b];
////                    gsl_matrix_set(m_M_inv, a,e, M_inv_ae+gsl_matrix_get(V,a,b)*gsl_matrix_get(U,e,b)*SS[b]);
//                }
//            }
//        }
////        gsl_vector_free(S);
////        gsl_vector_free(work);
////        gsl_matrix_free(U);
////        gsl_matrix_free(V);
//
//    //return M_inv;
//}


stl_vector_type GMLS_T_BOOST::get_alphaij(){
    vector_type b_data(NP,0);
    stl_vector_type alpha(m_Nneighbors,0);
	vector_type P(NP);
	vector_type delta(NP);
    
    double r, W;
    double alpha_ij;
    P = calcWij(m_particle);

    int i;
    for(i = 0; i < m_Nneighbors; i++){

        r = m_particle.distance(m_neighbor_data[i]);
        delta = calcWij(m_neighbor_data[i]);
        W = Wab(r,m_epsilon,2);
        delta *= W;

        b_data = boost::numeric::ublas::prod(m_M_inv, delta); // axpy_prod is MUCH slower

        alpha_ij = boost::numeric::ublas::inner_prod(P,b_data);
        alpha[i] = alpha_ij;
    }

    return alpha;
}


stl_vector_type GMLS_T_BOOST::get_Laplacian_alphaij(){
    vector_type b_data(NP,0);
    stl_vector_type alpha(m_Nneighbors,0);
	vector_type ddP(NP,0);
	vector_type delta(NP);
    
	ddP[9]=std::pow(m_epsilon, -2.0);
	ddP[6]=std::pow(m_epsilon, -2.0);
	ddP[4]=std::pow(m_epsilon, -2.0);

    double r, W;
    double alpha_ij;

    for(int i = 0; i < m_Nneighbors; i++){

        r = m_particle.distance(m_neighbor_data[i]);
        delta = calcWij(m_neighbor_data[i]);
        W = Wab(r,m_epsilon,2);
        delta *= W;

        b_data = boost::numeric::ublas::prod(m_M_inv, delta); // axpy_prod is MUCH slower

        alpha_ij = boost::numeric::ublas::inner_prod(ddP,b_data);
        alpha[i] = alpha_ij;
    }
    
    return alpha;
}
#endif
