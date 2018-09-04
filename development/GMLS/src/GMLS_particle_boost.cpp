#include "GMLS_particle_boost.hpp"

#include <cstdio>

#ifdef COMPADRE_USE_BOOST
double GMLS_ParticleT_BOOST::distance(GMLS_ParticleT_BOOST& neighbor){
	vector_type s = neighbor.get_coordinates() - m_coordinates;
	return std::sqrt(boost::numeric::ublas::inner_prod(s,s));
}

void GMLS_ParticleT_BOOST::print_coordinates(){
    for(int i=0, n=m_coordinates.size(); i < n; i++){
        fprintf(stdout, " %f ", m_coordinates[i]);
    }
    fprintf(stdout, "\n");
}
#endif
