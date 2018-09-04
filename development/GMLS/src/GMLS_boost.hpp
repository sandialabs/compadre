#ifndef _GMLS_BOOST_HPP_
#define _GMLS_BOOST_HPP_

#define BOOST_UBLAS_NDEBUG 1

#include "GMLS_Config.h"

#ifdef COMPADRE_USE_BOOST
#include "GMLS_particle_boost.hpp" // boost includes are in here

class GMLS_T_BOOST {
protected:
    int m_type; //Discretization type. TO DO: Create more types
    vector_type calcWij(GMLS_ParticleT_BOOST& neighbor);
    double Wab(double r,double h,int sigma); //Calculates weighting function
    int NP;
    int m_Nneighbors; //Number of neighbors
    GMLS_ParticleT_BOOST& m_particle; //Current particle
    std::vector<GMLS_ParticleT_BOOST>& m_neighbor_data; //Needed data per neighbor
    double m_epsilon;
    matrix_type m_M_inv;
    
public:
    GMLS_T_BOOST(std::vector<GMLS_ParticleT_BOOST>& t_neighbor_data, GMLS_ParticleT_BOOST& my_particle,
				int t_type, double t_epsilon, int t_Poly_order)
				: m_particle(my_particle), m_neighbor_data(t_neighbor_data) {

        m_Nneighbors = t_neighbor_data.size();
        m_type = t_type;
        m_epsilon = t_epsilon;
        Poly_order = t_Poly_order;
        
        NP = this->getNP(Poly_order);

        boost::numeric::ublas::identity_matrix<double> identity_matrix(NP,NP);
        m_M_inv = matrix_type(NP,NP);
        m_M_inv.assign(identity_matrix);

        this->get_M_inverse();
    };

    
    ~GMLS_T_BOOST(){
    };
    
    inline static int getNP(int m) { return (m+1)*(m+2)*(m+3)/6; }

    stl_vector_type get_alphaij();
    stl_vector_type get_Laplacian_alphaij();
    int get_Nneighbors() {return m_Nneighbors;}


    void print_neighbors(); //IO function
    void print_neighbor_data(); //IO function
    void get_M_inverse();


    
private:
    double factorial(int n);
    int Poly_order; //order of polynomial reconstruction
    
    vector_type get_relative_coord(GMLS_ParticleT_BOOST& t_particle);
    void getSVD(matrix_type& U, int NP);
    void invert_M(matrix_type& M, int N);



};
#endif

#endif



