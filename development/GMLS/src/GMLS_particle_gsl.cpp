#include "GMLS_particle_gsl.hpp"

#ifdef USE_GSL
double GMLS_ParticleT_GSL::distance(GMLS_ParticleT_GSL * neighbor){
    gsl_vector * s = gsl_vector_alloc (m_nDim);
    gsl_vector_memcpy(s, neighbor->get_coordinates());
    gsl_vector_sub(s, m_coordinates);
    double ddot = 0.0;
    gsl_blas_ddot (s, s, &ddot);
    gsl_vector_free(s);
    return sqrt(ddot);
}

void GMLS_ParticleT_GSL::print_coordinates(){
    int i;
    for(i = 0; i < m_nDim; i++){
        fprintf(stdout, " %f ", gsl_vector_get(m_coordinates, i));
    }
    fprintf(stdout, "\n");
}
#endif
