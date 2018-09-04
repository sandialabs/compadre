#ifndef _GMLS_GSL_HPP_
#define _GMLS_GSL_HPP_

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <cmath>

#include "GMLS_Config.h"

#ifdef USE_GSL
#include "GMLS_particle_gsl.hpp"
#include <gsl/gsl_integration.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_blas.h>

class GMLS_T_GSL {
protected:
    int m_type; //Discretization type. TO DO: Create more types
    gsl_vector * calcWij(GMLS_ParticleT_GSL * neighbor);
    double Wab(double r,double h,int sigma); //Calculates weighting function
    int NP;
    int m_Nneighbors; //Number of neighbors
    GMLS_ParticleT_GSL * m_particle; //Current particle
    gsl_vector * m_neighbors; // List of neighbor IDs
    std::vector<GMLS_ParticleT_GSL *> m_neighbor_data; //Needed data per neighbor
    double m_epsilon;
    gsl_matrix * m_M_inv;
    
public:
    GMLS_T_GSL(GMLS_ParticleT_GSL t_particle, gsl_vector * t_neighbors, std::vector<GMLS_ParticleT_GSL *> t_neighbor_data, int t_type, double t_epsilon, int t_Poly_order){

        
        m_particle = new GMLS_ParticleT_GSL(t_particle.get_coordinates(), t_particle.ID());
        m_nDim = m_particle->get_coordinates()->size,
        
        m_neighbors = t_neighbors;
        m_Nneighbors = t_neighbors->size;
        m_type = t_type;
        m_neighbor_data = t_neighbor_data;
        m_epsilon = t_epsilon;
        Poly_order = t_Poly_order;
        
        NP = this->getNP(Poly_order);
        m_M_inv = gsl_matrix_alloc(NP, NP);
        gsl_matrix_set_zero(m_M_inv);

        this->get_M_inverse();
    };

    
    ~GMLS_T_GSL(){
    };
    
    inline static int getNP(int m) { return (m+1)*(m+2)*(m+3)/6; }

    stl_vector_type get_alphaij();
    stl_vector_type get_Laplacian_alphaij();
    gsl_vector * get_neighbors() {return m_neighbors;}
    int get_Nneighbors() {return m_Nneighbors;}


    void print_neighbors(); //IO function
    void print_neighbor_data(); //IO function
    void get_M_inverse();


    
private:
    double factorial(int n);
    int Poly_order; //order of polynomial reconstruction
    int m_nDim;
    
    gsl_vector * get_relative_coord(GMLS_ParticleT_GSL * t_particle);
    void getSVD(gsl_matrix* U, int NP);
    void invert_M(gsl_matrix * M, int N);



};
#endif

#endif



