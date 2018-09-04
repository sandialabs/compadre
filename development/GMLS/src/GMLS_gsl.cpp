#include "GMLS_gsl.hpp"

#ifdef USE_GSL
gsl_vector * GMLS_T_GSL::get_relative_coord(GMLS_ParticleT_GSL * t_particle){
    gsl_vector * s = gsl_vector_alloc (m_nDim);
    gsl_vector_memcpy(s, t_particle->get_coordinates());
    gsl_vector_sub(s, m_particle->get_coordinates());
    return s;
}


void GMLS_T_GSL::invert_M(gsl_matrix * M_data,  int N){
    int s;

    gsl_permutation * p = gsl_permutation_alloc (N);
    gsl_linalg_LU_decomp(M_data, p, &s);
    gsl_linalg_LU_invert(M_data, p, m_M_inv);
    gsl_permutation_free (p);

}


double GMLS_T_GSL::Wab(double r,double h,int sigma){
    
    int p = 6;
    // return double(1.0-std::abs(r/(3*h))>0.0);
    return pow(1.0-std::abs(r/(3*h)),p)*double(1.0-std::abs(r/(3*h))>0.0);
    //return pow((r+1e-15)/(3*h),-p)*double(1-std::abs(r/(3*h))>0);
    //return (1.0-std::abs(r/(3.*h)))*double(1.0-std::abs(r/(3*h))>0)/3.0*h;
   // return p;
}


double GMLS_T_GSL::factorial(int n){
    double f = 1.0;
    int i;
    for(i = 1; i <= n; i++){
        f *=i;
    }
    return f;
}


gsl_vector * GMLS_T_GSL::calcWij(GMLS_ParticleT_GSL * neighbor){
    gsl_vector * delta = gsl_vector_alloc(NP);
    gsl_vector * relative_coord = get_relative_coord(neighbor);
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
                gsl_vector_set(delta, i, pow(gsl_vector_get(relative_coord, 0)/cutoff_p,alphax)
                                    *pow(gsl_vector_get(relative_coord, 1)/cutoff_p,alphay)
                                    *pow(gsl_vector_get(relative_coord, 2)/cutoff_p,alphaz)/alphaf);
                i++;
            }
            
        }
    }
    gsl_vector_free(relative_coord);
    return delta;
}


void GMLS_T_GSL::get_M_inverse(){
    int i;
    double r, W;
    double m_kl;
    gsl_vector * delta = gsl_vector_alloc(NP);
    gsl_matrix * M_data  = gsl_matrix_alloc(NP, NP);
    gsl_matrix_set_zero(M_data);
    
    for(i = 0; i < m_Nneighbors; i++){
        r = m_particle->distance(m_neighbor_data[i]);

        delta = calcWij(m_neighbor_data[i]);
        W =Wab(r,m_epsilon,2);

        for(int k = 0; k < NP; k++){
            for(int l = 0; l < NP; l++){
                m_kl = gsl_matrix_get(M_data, k, l);
                gsl_matrix_set(M_data, k, l, m_kl +gsl_vector_get(delta, l)*gsl_vector_get(delta, k)*W);
            }
        }
    }
    
//    invert_M(M_data, NP);
    getSVD(M_data, NP);

    gsl_vector_free(delta);
    gsl_matrix_free(M_data);


}


void GMLS_T_GSL::print_neighbors(){
    fprintf(stdout, "My neighbors: %f", gsl_vector_get(m_neighbors, 0));
    
    // for(i = 1; i < m_Nneighbors; i++){
      //  fprintf(stdout, ", %f ", gsl_vector_get(t_particle->get_coordinates(), i));
        
	//}
    fprintf(stdout, "\n");
}


void GMLS_T_GSL::print_neighbor_data(){
    int i;
    
    fprintf(stdout, "my location: ");
    gsl_vector * my_loc = m_particle->get_coordinates();
    for(i = 0; i < 3; i++){
        
        fprintf(stdout, " %f ", gsl_vector_get(my_loc, i));
    }
    fprintf(stdout, "\n");



    
    for(int j = 0; j < m_Nneighbors; j++){
      fprintf(stdout, "%d", j); 
    for(i = 0; i < 3; i++){
        
      fprintf(stdout, "%f ", gsl_vector_get(m_neighbor_data[j]->get_coordinates(), i));
    }
    fprintf(stdout, "\n");
    }
}


void GMLS_T_GSL::getSVD(gsl_matrix* M_data, int NP){
   // gsl_matrix * M_inv = gsl_matrix_alloc(NP, NP);
    double M_inv_ae;
    
        gsl_matrix* U;
        gsl_matrix* V;
        gsl_vector* S;
        gsl_vector* work;
        
        work = gsl_vector_alloc(NP);
        S    = gsl_vector_alloc(NP);
        U    = gsl_matrix_alloc(NP,NP);
        V    = gsl_matrix_alloc(NP,NP);
        gsl_matrix_memcpy(U, M_data);
        
        gsl_linalg_SV_decomp(U,V,S,work);
        
        /*
         //For post processing condition number
         printf("S = [");
         for(int i = 0; i < N; i++){
         printf(" %g",gsl_vector_get(S,i));
         }
         printf("]\n");
         cout << "condNumber: " << gsl_vector_get(S,0)/gsl_vector_get(S,N-1) << endl;
         */
        std::vector<double> SS;
        SS.resize(NP);
        for(int i = 0; i < NP; i++){
            SS[i] = gsl_vector_get(S,i);
            SS[i] = (std::abs(SS[i])>1e-12)?1.0/SS[i]:0.0;
            //      SS[i] = 1.0/(SS[i]+1e-12);
        }

       // gsl_matrix_set_zero(m_M_inv);
        
        for(int b = 0; b < NP; b++){
            for(int a = 0; a < NP; a++){
                for(int e = 0; e < NP; e++){
                    M_inv_ae = gsl_matrix_get(m_M_inv, a, e);
                    gsl_matrix_set(m_M_inv, a,e, M_inv_ae+gsl_matrix_get(V,a,b)*gsl_matrix_get(U,e,b)*SS[b]);
                }
            }
        }
        gsl_vector_free(S);
        gsl_vector_free(work);
        gsl_matrix_free(U);
        gsl_matrix_free(V);

    //return M_inv;
}


std::vector<double> GMLS_T_GSL::get_alphaij(){
    std::vector<double> alpha_vals(m_Nneighbors,0);
    gsl_vector * b_data = gsl_vector_alloc(NP);
    gsl_vector * P = gsl_vector_alloc(NP);
    gsl_vector * delta = gsl_vector_alloc(NP);
    
    double r, W;
    double alpha_ij;
    P = calcWij(m_particle);

    int i;
    for(i = 0; i < m_Nneighbors; i++){
        gsl_vector_set_zero(b_data);
        r = m_particle->distance(m_neighbor_data[i]);
        delta = calcWij(m_neighbor_data[i]);
        W =Wab(r,m_epsilon,2);
        gsl_vector_scale(delta, W);
        gsl_blas_dgemv(CblasNoTrans, 1.0, m_M_inv, delta, 0.0, b_data);
        gsl_blas_ddot(P, b_data, &alpha_ij);
        alpha_vals[i] = alpha_ij;
    }

    gsl_vector_free(b_data);
    gsl_vector_free(P);
    gsl_vector_free(delta);

    return alpha_vals;
}


std::vector<double> GMLS_T_GSL::get_Laplacian_alphaij(){

    std::vector<double> alpha_vals(m_Nneighbors,0);
    gsl_vector * b_data = gsl_vector_alloc(NP);
    gsl_vector * ddP= gsl_vector_alloc(NP);
    gsl_vector * delta = gsl_vector_alloc(NP);
    
    gsl_vector_set_zero(b_data);
    gsl_vector_set_zero(ddP);
    gsl_vector_set(ddP, 9, pow(m_epsilon, -2.0));
    gsl_vector_set(ddP, 6, pow(m_epsilon, -2.0));
    gsl_vector_set(ddP, 4, pow(m_epsilon, -2.0));

    double r, W;
    double alpha_ij;

    for(int i = 0; i < m_Nneighbors; i++){
        r = m_particle->distance(m_neighbor_data[i]);
        delta = calcWij(m_neighbor_data[i]);
        W =Wab(r,m_epsilon,2);
        gsl_vector_scale(delta, W);
        gsl_blas_dgemv(CblasNoTrans, 1.0, m_M_inv, delta, 0.0, b_data);

        gsl_blas_ddot (ddP, b_data, &alpha_ij);

        alpha_vals[i] = alpha_ij;
    }
    
    gsl_vector_free(b_data);
    gsl_vector_free(ddP);
    gsl_vector_free(delta);
    
    return alpha_vals;
}
#endif
