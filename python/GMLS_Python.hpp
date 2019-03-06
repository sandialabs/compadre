#ifndef _GMLS_PYTHON_HPP_
#define _GMLS_PYTHON_HPP_

#include <Compadre_GMLS.hpp>
#include <Compadre_Evaluator.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <assert.h>
#ifdef COMPADRE_USE_MPI
    #include <mpi.h>
#endif

void initializeKokkos() {
    Kokkos::initialize();
}

void finalizeKokkos() {
    Kokkos::finalize(); 
}

class GMLS_Python {

private:

    Compadre::GMLS* gmls_object;

public:

    GMLS_Python(const int poly_order, std::string dense_solver_type, const int curvature_poly_order, const int dimensions) {
        gmls_object = new Compadre::GMLS(poly_order, dense_solver_type, curvature_poly_order, dimensions);
        // initialized, but values not set
    }

    ~GMLS_Python() { delete gmls_object; }

    void setWeightingOrder(int regular_weight, int curvature_weight = -1) {
        if (curvature_weight < 0) curvature_weight = regular_weight;
        gmls_object->setCurvatureWeightingPower(curvature_weight);
        gmls_object->setWeightingPower(regular_weight);
    }

    void setNeighbors(PyObject* pyObjectArray_in) {
        // cast as a numpy array
        PyArrayObject *np_arr_in = reinterpret_cast<PyArrayObject*>(pyObjectArray_in);
    
        // copy data into Kokkos View
        // read in size in each dimension
        npy_intp* dims_in = PyArray_DIMS(np_arr_in);
        int npyLength1D = dims_in[0];
        int npyLength2D = dims_in[1];
    
        // create Kokkos View on host to copy into
        Kokkos::View<int**, Kokkos::HostSpace>    neighbor_lists("neighbor lists", npyLength1D, npyLength2D); // first column is # of neighbors
    
        // overwrite existing data assuming a 2D layout
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,npyLength1D), [=](int i) {
            for (int j = 0; j < npyLength2D; ++j)
            {
                int* val = (int*)PyArray_GETPTR2(np_arr_in, i, j);
                neighbor_lists(i,j) = *val;
            }
        });
    
        // set values from Kokkos View
        gmls_object->setNeighborLists(neighbor_lists);
    }

    void setSourceSites(PyObject* pyObjectArray_in) {
        // cast as a numpy array
        PyArrayObject *np_arr_in = reinterpret_cast<PyArrayObject*>(pyObjectArray_in);
        
        // copy data into Kokkos View
        // read in size in each dimension
        npy_intp* dims_in = PyArray_DIMS(np_arr_in);
        int npyLength1D = dims_in[0];
        int npyLength2D = dims_in[1];
        
        //  assert(npyLength2Dd == gmls_object->getDimensions());
        
        // create Kokkos View on host to copy into
        Kokkos::View<double**, Kokkos::HostSpace> source_coords("neighbor coordinates", npyLength1D, npyLength2D);
        
        // overwrite existing data assuming a 2D layout
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,npyLength1D), [=](int i) {
            for (int j = 0; j < npyLength2D; ++j)
            {
                double* val = (double*)PyArray_GETPTR2(np_arr_in, i, j);
                source_coords(i,j) = *val;
            }
        });
        
        // set values from Kokkos View
        gmls_object->setSourceSites(source_coords);
    }

    void setTargetSites(PyObject* pyObjectArray_in) {
        // cast as a numpy array
        PyArrayObject *np_arr_in = reinterpret_cast<PyArrayObject*>(pyObjectArray_in);
        
        // copy data into Kokkos View
        // read in size in each dimension
        npy_intp* dims_in = PyArray_DIMS(np_arr_in);
        int npyLength1D = dims_in[0];
        int npyLength2D = dims_in[1];
        
        // create Kokkos View on host to copy into
        Kokkos::View<double**, Kokkos::HostSpace> target_coords("neighbor coordinates", npyLength1D, npyLength2D);
        
        // assert(npyLength2Dd == gmls_object->getDimensions());
        
        // overwrite existing data assuming a 2D layout
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,npyLength1D), [=](int i) {
            for (int j = 0; j < npyLength2D; ++j)
            {
                double* val = (double*)PyArray_GETPTR2(np_arr_in, i, j);
                target_coords(i,j) = *val;
            }
        });
        
        // set values from Kokkos View
        gmls_object->setTargetSites(target_coords);
    }

    void setWindowSizes(PyObject* pyObjectArray_in) {
        // cast as a numpy array
        PyArrayObject *np_arr_in = reinterpret_cast<PyArrayObject*>(pyObjectArray_in);
        
        // copy data into Kokkos View
        // read in size in each dimension
        npy_intp* dims_in = PyArray_DIMS(np_arr_in);
        int npyLength1D = dims_in[0];
        
        // create Kokkos View on host to copy into
        Kokkos::View<double*, Kokkos::HostSpace> epsilon("h supports", npyLength1D);
        
        // overwrite existing data assuming a 2D layout
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,npyLength1D), [=](int i) {
            double* val = (double*)PyArray_GETPTR1(np_arr_in, i);
            epsilon(i) = *val;
        });
        
        // set values from Kokkos View
        gmls_object->setWindowSizes(epsilon);
    }

    void generatePointEvaluationStencil() {
        gmls_object->addTargets(Compadre::TargetOperation::ScalarPointEvaluation);
        gmls_object->generateAlphas();
    }

    PyObject* getPolynomialCoefficients(PyObject* pyObjectArray_in) {

        // cast as a numpy array
        PyArrayObject *np_arr_in = reinterpret_cast<PyArrayObject*>(pyObjectArray_in);
    
        // copy data into Kokkos View
        // read in size in each dimension
        const int num_dims_in = PyArray_NDIM(np_arr_in);
        npy_intp* dims_in = PyArray_DIMS(np_arr_in);
        int npyLength1D = dims_in[0];
        int npyLength2D = (num_dims_in > 1) ? dims_in[1] : 1;
    
        // create Kokkos View on host to copy into
        Kokkos::View<double**, Kokkos::HostSpace> source_data("source data", npyLength1D, npyLength2D); 
    
        // overwrite existing data assuming a 2D layout
        if (num_dims_in == 1) {
            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,npyLength1D), [=](int i) {
                double* val = (double*)PyArray_GETPTR1(np_arr_in, i);
                source_data(i,0) = *val;
            });
        } else {
            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,npyLength1D), [=](int i) {
                for (int j = 0; j < npyLength2D; ++j) {
                    double* val = (double*)PyArray_GETPTR2(np_arr_in, i, j);
                    source_data(i,j) = *val;
                }
            });
        }



        // get polynomial coefficient size
        const int NP = gmls_object->getPolynomialCoefficientsSize();
        // get number of target sites
        const int NT = gmls_object->getNeighborLists().dimension_0();

        // copy data into Kokkos View
        // set dimensions
        npy_intp dims_out[2] = {static_cast<npy_intp>(NT), static_cast<npy_intp>(NP)};

        // allocate memory for array 
        PyObject *pyObjectArray_out = PyArray_SimpleNew(2, dims_out, NPY_DOUBLE);
        if (!pyObjectArray_out) {
                printf("Out of memory.\n");
        }

        // recast as a numpy array and write assuming a 1D layout
        PyArrayObject *np_arr_out = reinterpret_cast<PyArrayObject*>(pyObjectArray_out);


        Compadre::Evaluator gmls_evaluator(gmls_object);
        auto polynomial_coefficients = gmls_evaluator.applyFullPolynomialCoefficientsBasisToDataAllComponents<double**, Kokkos::HostSpace>
            (source_data);

        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,NP), [=](int i) {
            for (int target_num=0; target_num<NT; target_num++) {
                double polynomial_coefficient = polynomial_coefficients(target_num, i);
                double* val = (double*)PyArray_GETPTR2(np_arr_out, target_num, i);
                *val = polynomial_coefficient;
            }
        });

        // return the Python object
        return pyObjectArray_out;

    }

    PyObject* getAlphas0Tensor(int target_num, PyObject* pyObjectArray_neighborList) {
        // cast as a numpy array
        PyArrayObject *np_arr_neighborlist = reinterpret_cast<PyArrayObject*>(pyObjectArray_neighborList);

        int* loop_size = (int*)PyArray_GETPTR2(np_arr_neighborlist, target_num, 0);

        // copy data into Kokkos View
        // set dimensions
        npy_intp dims_out[1] = {*loop_size};

        // allocate memory for array 
        PyObject *pyObjectArray_out = PyArray_SimpleNew(1, dims_out, NPY_DOUBLE);
        if (!pyObjectArray_out) {
                printf("Out of memory.\n");
        }

        // recast as a numpy array and write assuming a 1D layout
        PyArrayObject *np_arr_out = reinterpret_cast<PyArrayObject*>(pyObjectArray_out);

        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,*loop_size), [=](int i) {
            int* neighbor_id = (int*)PyArray_GETPTR2(np_arr_neighborlist, target_num, i+1); // first index is size in neighborlist
            double alpha_evaluation = gmls_object->getAlpha0TensorTo0Tensor(Compadre::TargetOperation::ScalarPointEvaluation, target_num, i);
            double* val = (double*)PyArray_GETPTR1(np_arr_out, i);
            *val = alpha_evaluation;
        });

        // return the Python object
        return pyObjectArray_out;
    }

    PyObject* applyStencil(PyObject* pyObjectArray_in) {
        // this is the preferred method for performing the evaluation of a GMLS operator
        // currently, it only supports PointEvaluation, can easily be expanded in the future

        // cast as a numpy array
        PyArrayObject *np_arr_in = reinterpret_cast<PyArrayObject*>(pyObjectArray_in);
    
        // copy data into Kokkos View
        // read in size in each dimension
        const int num_dims_in = PyArray_NDIM(np_arr_in);
        npy_intp* dims_in = PyArray_DIMS(np_arr_in);
        int npyLength1D = dims_in[0];
        int npyLength2D = (num_dims_in > 1) ? dims_in[1] : 1;
    
        // create Kokkos View on host to copy into
        Kokkos::View<double**, Kokkos::HostSpace> source_data("source data", npyLength1D, npyLength2D); 
    
        // overwrite existing data assuming a 2D layout
        if (num_dims_in == 1) {
            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,npyLength1D), [=](int i) {
                double* val = (double*)PyArray_GETPTR1(np_arr_in, i);
                source_data(i,0) = *val;
            });
        } else {
            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,npyLength1D), [=](int i) {
                for (int j = 0; j < npyLength2D; ++j) {
                    double* val = (double*)PyArray_GETPTR2(np_arr_in, i, j);
                    source_data(i,j) = *val;
                }
            });
        }

        Compadre::Evaluator gmls_evaluator(gmls_object);
        auto output_values = gmls_evaluator.applyAlphasToDataAllComponentsAllTargetSites<double**, Kokkos::HostSpace>
            (source_data, Compadre::TargetOperation::ScalarPointEvaluation);

        auto dim_out_0 = output_values.dimension_0();
        auto dim_out_1 = output_values.dimension_1();

        if (dim_out_1 == 1) {

            // allocate memory for array 
            npy_intp dims_out[1] = {static_cast<npy_intp>(dim_out_0)};
            PyObject *pyObjectArray_out = PyArray_SimpleNew(1, dims_out, NPY_DOUBLE);
            if (!pyObjectArray_out) {
                    printf("Out of memory.\n");
            }

            // recast as a numpy array and write assuming a 1D layout
            PyArrayObject *np_arr_out = reinterpret_cast<PyArrayObject*>(pyObjectArray_out);

            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,dim_out_0), [=](int i) {
                double* val = (double*)PyArray_GETPTR1(np_arr_out, i);
                *val = output_values(i,0);
            });
            Kokkos::fence();

            // return the 1D Python object
            return pyObjectArray_out;

        } else {

            // allocate memory for array 
            npy_intp dims_out[2] = {static_cast<npy_intp>(dim_out_0), static_cast<npy_intp>(dim_out_1)};
            PyObject *pyObjectArray_out = PyArray_SimpleNew(2, dims_out, NPY_DOUBLE);
            if (!pyObjectArray_out) {
                    printf("Out of memory.\n");
            }

            // recast as a numpy array and write assuming a 1D layout
            PyArrayObject *np_arr_out = reinterpret_cast<PyArrayObject*>(pyObjectArray_out);

            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,dim_out_0), [=](int i) {
                for (int j=0; j<dim_out_1; ++j) {
                    double* val = (double*)PyArray_GETPTR2(np_arr_out, i, j);
                    *val = output_values(i,j);
                }
            });
            Kokkos::fence();

            // return the 2D Python object
            return pyObjectArray_out;

        }
    }

};

int getNP(const int poly_order, const int dimensions) {
    // size of basis for given polynomial order and dimension
    return Compadre::GMLS::getNP(poly_order, dimensions);
}

int getNN(const int poly_order, const int dimensions) {
    // heuristic number of neighbors
    return Compadre::GMLS::getNN(poly_order, dimensions);
}

#endif


