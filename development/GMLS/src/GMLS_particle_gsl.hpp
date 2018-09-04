#ifndef _GMLS_PARTICLE_GSL_HPP_
#define _GMLS_PARTICLE_GSL_HPP_

#include <iostream>
#include <string>
#include <vector>
#include <map>

#include "GMLS_Config.h"

#ifdef USE_GSL
#include <gsl/gsl_integration.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_blas.h>

typedef std::vector<double> stl_vector_type;

class GMLS_ParticleT_GSL {
    protected:
        gsl_vector * m_coordinates;
        int m_ID;
        int m_nDim;
        
    public:
        GMLS_ParticleT_GSL();
        GMLS_ParticleT_GSL(gsl_vector * t_coords, int t_ID){
            m_nDim = t_coords->size;
            m_coordinates = gsl_vector_alloc (m_nDim);
            gsl_vector_memcpy(m_coordinates, t_coords);
            m_ID = t_ID;
        };

        ~GMLS_ParticleT_GSL() {}
    
        double distance(GMLS_ParticleT_GSL * neighbor);
        int ID() {return m_ID;}
        gsl_vector * get_coordinates() {return m_coordinates;}
        void print_coordinates();
    
    private:
};
#endif

#endif
