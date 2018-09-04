#ifndef _GMLS_PARTICLE_BOOST_HPP_
#define _GMLS_PARTICLE_BOOST_HPP_

#include "GMLS_Config.h"

#ifdef COMPADRE_USE_BOOST
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

typedef std::vector<double> stl_vector_type;
typedef boost::numeric::ublas::vector<double> vector_type;
typedef boost::numeric::ublas::matrix<double> matrix_type;

class GMLS_ParticleT_BOOST {
    protected:
        vector_type m_coordinates;
        int m_ID;
        
    public:
        GMLS_ParticleT_BOOST();
        GMLS_ParticleT_BOOST(vector_type& t_coords, int t_ID) : m_coordinates(t_coords) {
            m_ID = t_ID;
        };

        ~GMLS_ParticleT_BOOST() {}
    
        double distance(GMLS_ParticleT_BOOST& neighbor);
        int ID() {return m_ID;}
        vector_type get_coordinates() {return m_coordinates;}
        void print_coordinates();
    
    private:
};
#endif

#endif
