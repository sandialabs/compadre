#ifndef _GMLS_PYTHON_HPP_
#define _GMLS_PYTHON_HPP_

#include <Compadre_GMLS.hpp>
#include <Compadre_Evaluator.hpp>
#include <Compadre_PointCloudSearch.hpp>
#include <Compadre_KokkosParser.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <assert.h>
#ifdef COMPADRE_USE_MPI
    #include <mpi.h>
#endif

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>

using namespace Compadre;
namespace py = pybind11;


class ParticleHelper {
public:

    typedef Kokkos::View<double***, Kokkos::HostSpace> double_3d_view_type;
    typedef Kokkos::View<double**, Kokkos::HostSpace> double_2d_view_type;
    typedef Kokkos::View<int**, Kokkos::HostSpace> int_2d_view_type;
    typedef Kokkos::View<double*, Kokkos::HostSpace> double_1d_view_type;
    typedef Kokkos::View<int*, Kokkos::HostSpace> int_1d_view_type;
    typedef Kokkos::View<int*> int_1d_view_type_in_gmls;

private:

    Compadre::GMLS* gmls_object;
    Compadre::NeighborLists<int_1d_view_type_in_gmls>* nl;

    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, Compadre::PointCloudSearch<double_2d_view_type> >, 
            Compadre::PointCloudSearch<double_2d_view_type>, 3> tree_type;
    std::shared_ptr<Compadre::PointCloudSearch<double_2d_view_type> > point_cloud_search;

    double_2d_view_type _source_coords;
    double_2d_view_type _target_coords;
    double_1d_view_type _epsilon;

    // optional
    double_3d_view_type _tangent_bundle;
    double_2d_view_type _reference_normal_directions;
    double_2d_view_type _additional_evaluation_coords;
    Compadre::NeighborLists<int_1d_view_type_in_gmls>* a_nl;

public:

    ParticleHelper(GMLS& gmls_instance) {
        gmls_object = &gmls_instance;
        nl = gmls_object->getNeighborLists();
        a_nl = gmls_object->getAdditionalEvaluationIndices();
    }

    void setNeighbors(py::array_t<local_index_type> input) {
        py::buffer_info buf = input.request();

        if (buf.ndim != 2) {
            throw std::runtime_error("Number of dimensions must be two");
        }
        
        // create Kokkos View on host to copy into
        Kokkos::View<local_index_type**, Kokkos::HostSpace> neighbor_lists("neighbor lists", input.shape(0), input.shape(1));
        
        // overwrite existing data assuming a 2D layout
        auto data = input.unchecked<2>();
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,input.shape(0)), [=](int i) {
            for (int j = 0; j < input.shape(1); ++j)
            {
                neighbor_lists(i, j) = data(i, j);
            }
        });
        Kokkos::fence();
    
        // set values from Kokkos View
        gmls_object->setNeighborLists(neighbor_lists);
        nl = gmls_object->getNeighborLists();
    }

    decltype(nl) getNeighborLists() {
        compadre_assert_release((nl->getNumberOfTargets()>0) && "getNeighborLists() called, but neighbor lists were never set.");
        return gmls_object->getNeighborLists();
    }

    decltype(a_nl) getAdditionalEvaluationIndices() {
        compadre_assert_release((a_nl->getNumberOfTargets()>0) && "getAdditionalEvaluationIndices() called, but additional evaluation indices were never set.");
        return gmls_object->getAdditionalEvaluationIndices();
    }

    void setSourceSites(py::array_t<double> input) {
        py::buffer_info buf = input.request();

        if (buf.ndim != 2) {
            throw std::runtime_error("Number of dimensions must be two");
        }

        if (gmls_object->getGlobalDimensions()!=input.shape(1)) {
            throw std::runtime_error("Second dimension must be the same as GMLS spatial dimension");
        }
        
        // create Kokkos View on host to copy into
        Kokkos::View<double**, Kokkos::HostSpace> source_coords("neighbor coordinates", input.shape(0), input.shape(1));
        
        // overwrite existing data assuming a 2D layout
        auto data = input.unchecked<2>();
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,input.shape(0)), [=](int i) {
            for (int j = 0; j < input.shape(1); ++j)
            {
                source_coords(i, j) = data(i, j);
            }
        });
        Kokkos::fence();
        
        // set values from Kokkos View
        gmls_object->setSourceSites(source_coords);
        _source_coords = source_coords;
    }

    py::array_t<double> getSourceSites() {
        compadre_assert_release((_source_coords.extent(0)>0) && "getSourceSites() called, but source sites were never set.");

        auto dim_out_0 = _source_coords.extent(0);
        auto dim_out_1 = _source_coords.extent(1);

        auto result = py::array_t<double>(dim_out_0*dim_out_1);
        py::buffer_info buf_out = result.request();

        double *ptr = (double *) buf_out.ptr;
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,dim_out_0*dim_out_1), [&](int i) {
            ptr[i] = *(_source_coords.data()+i);
        });
        Kokkos::fence();

        result.resize({dim_out_0,dim_out_1});
        return result;
    }

    void setTargetSites(py::array_t<double> input) {
        py::buffer_info buf = input.request();

        if (buf.ndim != 2) {
            throw std::runtime_error("Number of dimensions must be two");
        }

        if (gmls_object->getGlobalDimensions()!=input.shape(1)) {
            throw std::runtime_error("Second dimension must be the same as GMLS spatial dimension");
        }
        
        // create Kokkos View on host to copy into
        Kokkos::View<double**, Kokkos::HostSpace> target_coords("target coordinates", input.shape(0), input.shape(1));
        
        // overwrite existing data assuming a 2D layout
        auto data = input.unchecked<2>();
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,input.shape(0)), [=](int i) {
            for (int j = 0; j < input.shape(1); ++j)
            {
                target_coords(i, j) = data(i, j);
            }
        });
        Kokkos::fence();
        
        // set values from Kokkos View
        gmls_object->setTargetSites(target_coords);
        _target_coords = target_coords;
    }

    py::array_t<double> getTargetSites() {
        compadre_assert_release((_target_coords.extent(0)>0) && "getTargetSites() called, but target sites were never set.");

        auto dim_out_0 = _target_coords.extent(0);
        auto dim_out_1 = _target_coords.extent(1);

        auto result = py::array_t<double>(dim_out_0*dim_out_1);
        py::buffer_info buf_out = result.request();

        double *ptr = (double *) buf_out.ptr;
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,dim_out_0*dim_out_1), [&](int i) {
            ptr[i] = *(_target_coords.data()+i);
        });
        Kokkos::fence();

        result.resize({dim_out_0,dim_out_1});
        return result;
    }

    void setWindowSizes(py::array_t<double> input) {
        py::buffer_info buf = input.request();

        if (buf.ndim != 1) {
            throw std::runtime_error("Number of dimensions must be one");
        }
        
        // create Kokkos View on host to copy into
        Kokkos::View<double*, Kokkos::HostSpace> epsilon("h supports", input.shape(0));
        
        // overwrite existing data assuming a 2D layout
        auto data = input.unchecked<1>();
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,input.shape(0)), [=](int i) {
            epsilon(i) = data(i);
        });
        Kokkos::fence();
        
        // set values from Kokkos View
        gmls_object->setWindowSizes(epsilon);
        _epsilon = epsilon;
    }
 
    py::array_t<double> getWindowSizes() {
        if (_epsilon.extent(0)<=0) {
            throw std::runtime_error("getWindowSizes() called, but window sizes were never set.");
        }

        auto result = py::array_t<double>(_epsilon.extent(0));
        py::buffer_info buf = result.request();

        double *ptr = (double *) buf.ptr;
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,buf.shape[0]), [&](int i) {
            ptr[i] = _epsilon(i);
        });
        Kokkos::fence();

        return result;
    }

    void setTangentBundle(py::array_t<double> input) {
        py::buffer_info buf = input.request();

        if (buf.ndim != 3) {
            throw std::runtime_error("Number of dimensions must be three");
        }

        if (gmls_object->getGlobalDimensions()!=input.shape(1)) {
            throw std::runtime_error("Second dimension must be the same as GMLS spatial dimension");
        }

        if (gmls_object->getGlobalDimensions()!=input.shape(2)) {
            throw std::runtime_error("Second dimension must be the same as GMLS spatial dimension");
        }
        
        // create Kokkos View on host to copy into
        Kokkos::View<double***, Kokkos::HostSpace> tangent_bundle("tangent bundle", input.shape(0), input.shape(1), input.shape(2));
        
        // overwrite existing data assuming a 2D layout
        auto data = input.unchecked<3>();
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,input.shape(0)), [=](int i) {
            for (int j = 0; j < input.shape(1); ++j)
            {
                for (int k = 0; k < input.shape(2); ++k) {
                    tangent_bundle(i, j, k) = data(i, j, k);
                }
            }
        });
        Kokkos::fence();
        
        // set values from Kokkos View
        gmls_object->setTangentBundle(tangent_bundle);
        _tangent_bundle = tangent_bundle;
    }

    py::array_t<double> getTangentBundle() {
        compadre_assert_release((_tangent_bundle.extent(0)>0) && "getTangentBundle() called, but tangent bundle was never set.");

        auto dim_out_0 = _tangent_bundle.extent(0);
        auto dim_out_1 = _tangent_bundle.extent(1);
        auto dim_out_2 = _tangent_bundle.extent(2);

        auto result = py::array_t<double>(dim_out_0*dim_out_1*dim_out_2);
        py::buffer_info buf_out = result.request();

        double *ptr = (double *) buf_out.ptr;
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,dim_out_0*dim_out_1*dim_out_2), [&](int i) {
            ptr[i] = *(_tangent_bundle.data()+i);
        });
        Kokkos::fence();

        result.resize({dim_out_0,dim_out_1,dim_out_2});
        return result;
    }

    void setReferenceOutwardNormalDirection(py::array_t<double> input, bool use_to_orient_surface = true) {
        py::buffer_info buf = input.request();

        if (buf.ndim != 2) {
            throw std::runtime_error("Number of dimensions must be two");
        }

        if (gmls_object->getGlobalDimensions()!=input.shape(1)) {
            throw std::runtime_error("Second dimension must be the same as GMLS spatial dimension");
        }
        
        // create Kokkos View on host to copy into
        Kokkos::View<double**, Kokkos::HostSpace> reference_normal_directions("reference normal directions", input.shape(0), input.shape(1));
        
        // overwrite existing data assuming a 2D layout
        auto data = input.unchecked<2>();
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,input.shape(0)), [=](int i) {
            for (int j = 0; j < input.shape(1); ++j)
            {
                reference_normal_directions(i, j) = data(i, j);
            }
        });
        Kokkos::fence();
        
        // set values from Kokkos View
        gmls_object->setReferenceOutwardNormalDirection(reference_normal_directions, true /* use to orient surface*/);
        _reference_normal_directions = reference_normal_directions;
    }

    py::array_t<double> getReferenceOutwardNormalDirection() {
        compadre_assert_release((_reference_normal_directions.extent(0)>0) && "getReferenceNormalDirections() called, but normal directions were never set.");

        auto dim_out_0 = _reference_normal_directions.extent(0);
        auto dim_out_1 = _reference_normal_directions.extent(1);

        auto result = py::array_t<double>(dim_out_0*dim_out_1);
        py::buffer_info buf_out = result.request();

        double *ptr = (double *) buf_out.ptr;
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,dim_out_0*dim_out_1), [&](int i) {
            ptr[i] = *(_reference_normal_directions.data()+i);
        });
        Kokkos::fence();

        result.resize({dim_out_0,dim_out_1});
        return result;
    }

    void generateKDTree(py::array_t<double> input) {
        py::buffer_info buf = input.request();

        if (buf.ndim != 2) {
            throw std::runtime_error("Number of dimensions must be two");
        }

        if (gmls_object->getGlobalDimensions()!=input.shape(1)) {
            throw std::runtime_error("Second dimension must be the same as GMLS spatial dimension");
        }

        // create Kokkos View on host to copy into
        Kokkos::View<double**, Kokkos::HostSpace> source_coords("neighbor coordinates", input.shape(0), input.shape(1));
        
        // overwrite existing data assuming a 2D layout
        auto data = input.unchecked<2>();
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, input.shape(0)), [=](int i) {
            for (int j = 0; j < input.shape(1); ++j)
            {
                source_coords(i, j) = data(i, j);
            }
        });
        point_cloud_search = std::shared_ptr<Compadre::PointCloudSearch<double_2d_view_type> >(new Compadre::PointCloudSearch<double_2d_view_type>(source_coords, gmls_object->getGlobalDimensions()));

        _source_coords = source_coords;
        gmls_object->setSourceSites(source_coords);
    }

    void generateNeighborListsFromKNNSearchAndSet(py::array_t<double> input, int poly_order, int dimension = 3, double epsilon_multiplier = 1.6, double max_search_radius = 0.0, bool scale_k_neighbor_radius = true, bool scale_num_neighbors = false) {

        int neighbors_needed = Compadre::GMLS::getNP(poly_order, dimension);

        py::buffer_info buf = input.request();

        if (buf.ndim != 2) {
            throw std::runtime_error("Number of dimensions must be two");
        }

        if (gmls_object->getGlobalDimensions()!=input.shape(1)) {
            throw std::runtime_error("Second dimension must be the same as GMLS spatial dimension");
        }
        
        // create Kokkos View on host to copy into
        Kokkos::View<double**, Kokkos::HostSpace> target_coords("target site coordinates", input.shape(0), input.shape(1));
        
        // overwrite existing data assuming a 2D layout
        auto data = input.unchecked<2>();
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,input.shape(0)), [=](int i) {
            for (int j = 0; j < input.shape(1); ++j)
            {
                target_coords(i, j) = data(i, j);
            }
        });

        // how many target sites
        int number_target_coords = target_coords.extent(0);

        // make empty neighbor list kokkos view
        int_1d_view_type neighbor_lists("neighbor lists", 0);
        
        // holds number of neighbors for each target
        int_1d_view_type number_of_neighbors_list("number of neighbors list", number_target_coords);
        
        // make epsilons kokkos view
        double_1d_view_type epsilon("h supports", number_target_coords);
        
        // call point_cloud_search using targets
        // use these neighbor lists and epsilons to set the gmls object
        
        compadre_assert_release(((int)scale_k_neighbor_radius + (int)scale_num_neighbors==1) && "One and only of scale kth neighbor's radius (scale_k_neighbor_radius) or scale number of neighbors (scale_num_neighbors) can be set true.");
        if (scale_num_neighbors) { 
            neighbors_needed = std::ceil(neighbors_needed*epsilon_multiplier);
            epsilon_multiplier = 1.0+1e-14;
        }

        size_t total_storage = point_cloud_search->generateCRNeighborListsFromKNNSearch(true /* is a dry run*/, target_coords, neighbor_lists, 
                number_of_neighbors_list, epsilon, neighbors_needed, epsilon_multiplier, max_search_radius);

        Kokkos::resize(neighbor_lists, total_storage);
        Kokkos::fence();

        total_storage = point_cloud_search->generateCRNeighborListsFromKNNSearch(false /* not a dry run*/, target_coords, neighbor_lists, 
                number_of_neighbors_list, epsilon, neighbors_needed, epsilon_multiplier, max_search_radius);
        Kokkos::fence();

        // set these views in the GMLS object
        gmls_object->setTargetSites(target_coords);
        gmls_object->setNeighborLists(neighbor_lists, number_of_neighbors_list);
        gmls_object->setWindowSizes(epsilon);
        nl = gmls_object->getNeighborLists();

        _target_coords = target_coords;
        _epsilon = epsilon;

    }
    
    void setAdditionalEvaluationSitesData(py::array_t<local_index_type> input_indices, py::array_t<double> input_coordinates) {
        py::buffer_info buf_indices = input_indices.request();
        py::buffer_info buf_coordinates = input_coordinates.request();

        if (buf_indices.ndim != 2) {
            throw std::runtime_error("Number of dimensions of input indices must be two");
        }
        if (buf_coordinates.ndim != 2) {
            throw std::runtime_error("Number of dimensions of input coordinates must be two");
        }
        if (gmls_object->getGlobalDimensions()!=input_coordinates.shape(1)) {
            throw std::runtime_error("Second dimension of input coordinates must be the same as GMLS spatial dimension");
        }
        
        // create Kokkos View on host to copy into
        Kokkos::View<local_index_type**, Kokkos::HostSpace> neighbor_lists("neighbor lists", input_indices.shape(0), input_indices.shape(1));

        // create Kokkos View on host to copy into
        Kokkos::View<double**, Kokkos::HostSpace> extra_coords("neighbor coordinates", input_coordinates.shape(0), input_coordinates.shape(1));
        
        // overwrite existing data assuming a 2D layout
        auto data_indices = input_indices.unchecked<2>();
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,input_indices.shape(0)), [=](int i) {
            for (int j = 0; j < input_indices.shape(1); ++j)
            {
                neighbor_lists(i, j) = data_indices(i, j);
            }
        });
        // overwrite existing data assuming a 2D layout
        auto data_coordinates = input_coordinates.unchecked<2>();
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,input_coordinates.shape(0)), [=](int i) {
            for (int j = 0; j < input_coordinates.shape(1); ++j)
            {
                extra_coords(i, j) = data_coordinates(i, j);
            }
        });
        Kokkos::fence();
    
        // set values from Kokkos View
        gmls_object->setAdditionalEvaluationSitesData(neighbor_lists, extra_coords);
        // store in ParticleHelper
        _additional_evaluation_coords  = extra_coords;
        a_nl = gmls_object->getAdditionalEvaluationIndices();
    }

    py::array_t<double> getPolynomialCoefficients(py::array_t<double> input) {
        py::buffer_info buf = input.request();

        // create Kokkos View on host to copy into
        Kokkos::View<double**, Kokkos::HostSpace> source_data("source data", input.shape(0), (buf.ndim>1) ? input.shape(1) : 1); 

        if (buf.ndim==1) {
            // overwrite existing data assuming a 2D layout
            auto data = input.unchecked<1>();
            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, input.shape(0)), [=](int i) {
                source_data(i, 0) = data(i);
            });
        } else if (buf.ndim>1) {
            // overwrite existing data assuming a 2D layout
            auto data = input.unchecked<2>();
            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, input.shape(0)), [=](int i) {
                for (int j = 0; j < input.shape(1); ++j)
                {
                    source_data(i, j) = data(i, j);
                }
            });
        }
        Kokkos::fence();

        Compadre::Evaluator gmls_evaluator(gmls_object);
        auto polynomial_coefficients = gmls_evaluator.applyFullPolynomialCoefficientsBasisToDataAllComponents<double**, Kokkos::HostSpace>
            (source_data);

        auto dim_out_0 = polynomial_coefficients.extent(0);
        auto dim_out_1 = polynomial_coefficients.extent(1);

        auto result = py::array_t<double>(dim_out_0*dim_out_1);
        py::buffer_info buf_out = result.request();

        double *ptr = (double *) buf_out.ptr;
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,dim_out_0*dim_out_1), [&](int i) {
            ptr[i] = *(polynomial_coefficients.data()+i);
        });
        Kokkos::fence();

        if (dim_out_1>1) {
            result.resize({dim_out_0,dim_out_1});
        }
        return result;
    }
 
    py::array_t<double> applyStencil(const py::array_t<double, py::array::f_style | py::array::forcecast> input, const TargetOperation lro, const SamplingFunctional sro, const int evaluation_site_local_index = 0) const {
        py::buffer_info buf = input.request();
 
        // cast numpy data as Kokkos View
        host_scratch_matrix_left_type source_data((double *) buf.ptr, input.shape(0), (buf.ndim>1) ? input.shape(1) : 1);

        Compadre::Evaluator gmls_evaluator(gmls_object);
        if (lro==Compadre::TargetOperation::PartialYOfScalarPointEvaluation) {
            compadre_assert_release((gmls_object->getGlobalDimensions() > 1) && "Partial derivative w.r.t. y requested, but less than 2D problem.");
        } else if (lro==Compadre::TargetOperation::PartialZOfScalarPointEvaluation) {
            compadre_assert_release((gmls_object->getGlobalDimensions() > 2) && "Partial derivative w.r.t. z requested, but less than 3D problem.");
        }
        auto output_values = gmls_evaluator.applyAlphasToDataAllComponentsAllTargetSites<double**, Kokkos::HostSpace>
            (source_data, lro, sro, true /*scalar_as_vector_if_needed*/, evaluation_site_local_index);

        auto dim_out_0 = output_values.extent(0);
        auto dim_out_1 = output_values.extent(1);

        auto result = py::array_t<double>(dim_out_0*dim_out_1);

        if (dim_out_1==2) {
            result.resize({dim_out_0,dim_out_1});
            auto result_data = result.mutable_unchecked<2>();
            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,dim_out_0), [&](int i) {
                for (size_t j=0; j<dim_out_1; ++j) {
                    result_data(i,j) = output_values(i,j);
                }
            });
            Kokkos::fence();
        }
        else if (dim_out_1==1) {
            auto result_data = result.mutable_unchecked<1>();
            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,dim_out_0), [&](int i) {
                result_data(i) = output_values(i,0);
            });
            Kokkos::fence();
        }

        return result;
    }
    
    py::array_t<double> applyStencilAllTargetsAllAdditionalEvaluationSites(const py::array_t<double, py::array::f_style | py::array::forcecast> input, const TargetOperation lro, const SamplingFunctional sro) const {
        py::buffer_info buf = input.request();
 
        // cast numpy data as Kokkos View
        host_scratch_matrix_left_type source_data((double *) buf.ptr, input.shape(0), (buf.ndim>1) ? input.shape(1) : 1);

        Compadre::Evaluator gmls_evaluator(gmls_object);
        if (lro==Compadre::TargetOperation::PartialYOfScalarPointEvaluation) {
            compadre_assert_release((gmls_object->getGlobalDimensions() > 1) && "Partial derivative w.r.t. y requested, but less than 2D problem.");
        } else if (lro==Compadre::TargetOperation::PartialZOfScalarPointEvaluation) {
            compadre_assert_release((gmls_object->getGlobalDimensions() > 2) && "Partial derivative w.r.t. z requested, but less than 3D problem.");
        }
        // run one problem at extra site 0 (=target site) to get output size
        auto output_values = gmls_evaluator.applyAlphasToDataAllComponentsAllTargetSites<double**, Kokkos::HostSpace>
            (source_data, lro, sro, true /*scalar_as_vector_if_needed*/, 0);

        // get maximum number of additional sites plus target site (+1) per target site
        int max_additional_sites = a_nl->getMaxNumNeighbors() + 1;
        a_nl->computeMinNumNeighbors();
        int min_additional_sites = a_nl->getMinNumNeighbors();

        // set dim_out_0 to 1 if setAdditionalEvaluationSitesData never called
        size_t dim_out_0 = (_additional_evaluation_coords.size()==0) ? 1 : max_additional_sites;
        auto dim_out_1 = output_values.extent(0);
        auto dim_out_2 = output_values.extent(1);

        auto result = py::array_t<double>(dim_out_0*dim_out_1*dim_out_2);

        if (dim_out_2==2) {
            result.resize({dim_out_0,dim_out_1,dim_out_2});
            auto result_data = result.mutable_unchecked<3>();
            for (size_t k=0; k<max_additional_sites; ++k) {
                if (k>0) {
                    output_values = gmls_evaluator.applyAlphasToDataAllComponentsAllTargetSites<double**, Kokkos::HostSpace>
                        (source_data, lro, sro, true /*scalar_as_vector_if_needed*/, k);
                }
                Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,dim_out_1), [&](int i) {
                    for (size_t j=0; j<dim_out_2; ++j) {
                            result_data(k,i,j) = output_values(i,j);
                        }
                });
                Kokkos::fence();
            }
        }
        else if (dim_out_2==1) {
            result.resize({dim_out_0,dim_out_1});
            auto result_data = result.mutable_unchecked<2>();
            for (size_t k=0; k<max_additional_sites; ++k) {
                if (k>0) {
                    output_values = gmls_evaluator.applyAlphasToDataAllComponentsAllTargetSites<double**, Kokkos::HostSpace>
                        (source_data, lro, sro, true /*scalar_as_vector_if_needed*/, k);
                }
                Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,dim_out_1), [&](int i) {
                    result_data(k,i) = output_values(i,0);
                });
                Kokkos::fence();
            }
        }

        return result;
    }

    double applyStencilSingleTarget(const py::array_t<double, py::array::f_style | py::array::forcecast> input, const TargetOperation lro, const SamplingFunctional sro, const int evaluation_site_local_index = 0) const {
        py::buffer_info buf = input.request();
 
        // cast numpy data as Kokkos View
        host_scratch_matrix_left_type source_data((double *) buf.ptr, input.shape(0), (buf.ndim>1) ? input.shape(1) : 1);

        compadre_assert_release(buf.ndim==1 && "Input given with dimensions > 1");

        Compadre::Evaluator gmls_evaluator(gmls_object);

        return gmls_evaluator.applyAlphasToDataSingleComponentSingleTargetSite(source_data, 0, lro, 0, evaluation_site_local_index, 0, 0, 0, 0, false);
    }
};

template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

PYBIND11_MODULE(_pycompadre, m) {
    m.doc() = R"pbdoc(
PyCOMPADRE: Compadre Toolkit
-----------------------------

Important:
Be sure to initialize Kokkos before setting up any objects from this library,
by created a scoped KokkosParser object, i.e:

.. code-block:: python

    kp = pycompadre.KokkosParser()

When `kp` goes out of scope, then Kokkos will be finalized.

GMLS type objects are passed to ParticleHelper objects.
Be sure to deallocate in reverse order, i.e.:

.. code-block:: python

    >> kp = pycompadre.KokkosParser()
    >> gmls_obj = GMLS(...)
    >> gmls_helper = ParticleHelper(gmls_object)
    ...
    ...
    >> del gmls_obj
    >> del gmls_helper
    >> del kp

Project details at:
https://github.com/sandialabs/compadre

Implementation details at:
https://github.com/sandialabs/compadre/blob/master/pycompadre/pycompadre.cpp

)pbdoc";

    py::class_<SamplingFunctional>(m, "SamplingFunctional");
    py::dict sampling_functional;
    sampling_functional["PointSample"] = PointSample;
    sampling_functional["VectorPointSample"] = VectorPointSample;
    sampling_functional["ManifoldVectorPointSample"] = ManifoldVectorPointSample;
    sampling_functional["StaggeredEdgeAnalyticGradientIntegralSample"] = StaggeredEdgeAnalyticGradientIntegralSample;
    sampling_functional["StaggeredEdgeIntegralSample"] = StaggeredEdgeIntegralSample;
    sampling_functional["VaryingManifoldVectorPointSample"] = VaryingManifoldVectorPointSample;
    sampling_functional["FaceNormalIntegralSample"] = FaceNormalIntegralSample;
    sampling_functional["FaceNormalPointSample"] = FaceNormalPointSample;
    sampling_functional["FaceTangentIntegralSample"] = FaceTangentIntegralSample;
    sampling_functional["FaceTangentPointSample"] = FaceTangentPointSample;
    sampling_functional["ScalarFaceAverageSample"] = ScalarFaceAverageSample;
    m.attr("SamplingFunctional") = sampling_functional;

    py::enum_<TargetOperation>(m, "TargetOperation")
    .value("ScalarPointEvaluation", TargetOperation::ScalarPointEvaluation)
    .value("VectorPointEvaluation", TargetOperation::VectorPointEvaluation)
    .value("LaplacianOfScalarPointEvaluation", TargetOperation::LaplacianOfScalarPointEvaluation)
    .value("VectorLaplacianPointEvaluation", TargetOperation::VectorLaplacianPointEvaluation)
    .value("GradientOfScalarPointEvaluation", TargetOperation::GradientOfScalarPointEvaluation)
    .value("GradientOfVectorPointEvaluation", TargetOperation::GradientOfVectorPointEvaluation)
    .value("DivergenceOfVectorPointEvaluation", TargetOperation::DivergenceOfVectorPointEvaluation)
    .value("CurlOfVectorPointEvaluation", TargetOperation::CurlOfVectorPointEvaluation)
    .value("CurlCurlOfVectorPointEvaluation", TargetOperation::CurlCurlOfVectorPointEvaluation)
    .value("PartialXOfScalarPointEvaluation", TargetOperation::PartialXOfScalarPointEvaluation)
    .value("PartialYOfScalarPointEvaluation", TargetOperation::PartialYOfScalarPointEvaluation)
    .value("PartialZOfScalarPointEvaluation", TargetOperation::PartialZOfScalarPointEvaluation)
    .value("ChainedStaggeredLaplacianOfScalarPointEvaluation", TargetOperation::ChainedStaggeredLaplacianOfScalarPointEvaluation)
    .value("GaussianCurvaturePointEvaluation", TargetOperation::GaussianCurvaturePointEvaluation)
    .value("ScalarFaceAverageEvaluation", TargetOperation::ScalarFaceAverageEvaluation)
    .export_values();


    py::enum_<ReconstructionSpace>(m, "ReconstructionSpace")
    .value("ScalarTaylorPolynomial", ReconstructionSpace::ScalarTaylorPolynomial)
    .value("VectorTaylorPolynomial", ReconstructionSpace::VectorTaylorPolynomial)
    .value("VectorOfScalarClonesTaylorPolynomial", ReconstructionSpace::VectorOfScalarClonesTaylorPolynomial)
    .value("DivergenceFreeVectorTaylorPolynomial", ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial)
    .export_values();

    py::enum_<WeightingFunctionType>(m, "WeightingFunctionType")
    .value("Power", WeightingFunctionType::Power)
    .value("Gaussian", WeightingFunctionType::Gaussian)
    .value("CubicSpline", WeightingFunctionType::CubicSpline)
    .value("Cosine", WeightingFunctionType::Cosine)
    .value("Sigmoid", WeightingFunctionType::Sigmoid)
    .export_values();

    // helper functions
    py::class_<ParticleHelper>(m, "ParticleHelper", R"pbdoc(
        Class to manage calling PointCloudSearch, moving data to/from Numpy arrays in Kokkos::Views,
        and applying GMLS solutions to multidimensional data arrays
    )pbdoc")
    .def(py::init<GMLS&>(),py::arg("gmls_instance"))
    .def("generateKDTree", &ParticleHelper::generateKDTree)
    .def("generateNeighborListsFromKNNSearchAndSet", &ParticleHelper::generateNeighborListsFromKNNSearchAndSet, 
            py::arg("target_sites"), py::arg("poly_order"), py::arg("dimension") = 3, py::arg("epsilon_multiplier") = 1.6, 
            py::arg("max_search_radius") = 0.0, py::arg("scale_k_neighbor_radius") = true, 
            py::arg("scale_num_neighbors") = false)
    .def("setNeighbors", &ParticleHelper::setNeighbors, py::arg("neighbor_lists"), 
            "Sets neighbor lists from 2D array where first column is number of neighbors for corresponding row's target site.")
    .def("getAdditionalEvaluationIndices", &ParticleHelper::getAdditionalEvaluationIndices, py::return_value_policy::reference_internal)
    .def("getNeighborLists", &ParticleHelper::getNeighborLists, py::return_value_policy::reference_internal)
    .def("setWindowSizes", &ParticleHelper::setWindowSizes, py::arg("window_sizes"))
    .def("getWindowSizes", &ParticleHelper::getWindowSizes, py::return_value_policy::take_ownership)
    .def("setSourceSites", &ParticleHelper::setSourceSites, py::arg("source_coordinates"))
    .def("getSourceSites", &ParticleHelper::getSourceSites, py::return_value_policy::take_ownership)
    .def("setTargetSites", &ParticleHelper::setTargetSites, py::arg("target_coordinates"))
    .def("getTargetSites", &ParticleHelper::getTargetSites, py::return_value_policy::take_ownership)
    .def("setTangentBundle", &ParticleHelper::setTangentBundle, py::arg("tangent_bundle"))
    .def("getTangentBundle", &ParticleHelper::getTangentBundle, py::return_value_policy::take_ownership)
    .def("setReferenceOutwardNormalDirection", &ParticleHelper::setReferenceOutwardNormalDirection, py::arg("reference_normal_directions"), py::arg("use_to_orient_surface") = true)
    .def("getReferenceOutwardNormalDirection", &ParticleHelper::getReferenceOutwardNormalDirection, py::return_value_policy::take_ownership)
    .def("setAdditionalEvaluationSitesData", &ParticleHelper::setAdditionalEvaluationSitesData, py::arg("additional_evaluation_sites_neighbor_lists"), py::arg("additional_evaluation_sites_coordinates"))
    .def("getPolynomialCoefficients", &ParticleHelper::getPolynomialCoefficients, py::arg("input_data"), py::return_value_policy::take_ownership)
    .def("applyStencilSingleTarget", &ParticleHelper::applyStencilSingleTarget, py::arg("input_data"), py::arg("target_operation")=TargetOperation::ScalarPointEvaluation, py::arg("sampling_functional")=PointSample, py::arg("evaluation_site_local_index")=0)
    .def("applyStencilAllTargetsAllAdditionalEvaluationSites", &ParticleHelper::applyStencilAllTargetsAllAdditionalEvaluationSites, py::arg("input_data"), py::arg("target_operation")=TargetOperation::ScalarPointEvaluation, py::arg("sampling_functional")=PointSample, py::return_value_policy::take_ownership)
    .def("applyStencil", &ParticleHelper::applyStencil, py::arg("input_data"), py::arg("target_operation")=TargetOperation::ScalarPointEvaluation, py::arg("sampling_functional")=PointSample, py::arg("evaluation_site_local_index")=0, py::return_value_policy::take_ownership);
    
    py::class_<SolutionSet<host_memory_space> >(m, "SolutionSet", R"pbdoc(
        Class containing solution data from GMLS problems
    )pbdoc")
    .def("getAlpha", &SolutionSet<host_memory_space>::getAlpha<>, py::arg("lro"), py::arg("target_index"), py::arg("output_component_axis_1"), py::arg("output_component_axis_2"), py::arg("neighbor_index"), py::arg("input_component_axis_1"), py::arg("input_component_axis_2"), py::arg("additional_evaluation_site")=0);

    py::class_<GMLS>(m, "GMLS")
    .def(py::init<int,int,std::string,std::string,std::string,int>(),
            py::arg("poly_order"),py::arg("dimension")=3,py::arg("dense_solver_type")="QR", 
            py::arg("problem_type")="STANDARD", py::arg("constraint_type")="NO_CONSTRAINT", 
            py::arg("curvature_poly_order")=2)
    .def(py::init<ReconstructionSpace,SamplingFunctional,int,int,std::string,std::string,std::string,int>(),
            py::arg("reconstruction_space"),py::arg("sampling_functional"),
            py::arg("poly_order"),py::arg("dimension")=3,py::arg("dense_solver_type")="QR", 
            py::arg("problem_type")="STANDARD", py::arg("constraint_type")="NO_CONSTRAINT", 
            py::arg("curvature_poly_order")=2)
    .def("getWeightingParameter", &GMLS::getWeightingParameter, py::arg("parameter index")=0, "Get weighting kernel parameter[index].")
    .def("setWeightingParameter", &GMLS::setWeightingParameter, py::arg("parameter value"), py::arg("parameter index")=0, "Set weighting kernel parameter[index] to parameter value.")
    .def("setWeightingPower", &GMLS::setWeightingParameter, py::arg("parameter value"), py::arg("parameter index")=0, "Set weighting kernel parameter[index] to parameter value. [DEPRECATED]")
    .def("getWeightingType", &GMLS::getWeightingType, "Get the weighting type.")
    .def("setWeightingType", overload_cast_<const std::string&>()(&GMLS::setWeightingType), "Set the weighting type with a string.")
    .def("setWeightingType", overload_cast_<WeightingFunctionType>()(&GMLS::setWeightingType), "Set the weighting type with a WeightingFunctionType.")
    .def("addTargets", overload_cast_<TargetOperation>()(&GMLS::addTargets), "Add a target operation.")
    .def("addTargets", overload_cast_<std::vector<TargetOperation> >()(&GMLS::addTargets), "Add a list of target operations.")
    .def("generateAlphas", &GMLS::generateAlphas, py::arg("number_of_batches")=1, py::arg("keep_coefficients")=false)
    .def("getSolutionSet", &GMLS::getSolutionSetHost, py::return_value_policy::reference_internal)
    .def("getNP", &GMLS::getNP, "Get size of basis.")
    .def("getNN", &GMLS::getNN, "Heuristic number of neighbors.");

    py::class_<NeighborLists<ParticleHelper::int_1d_view_type_in_gmls> >(m, "NeighborLists")
    .def("computeMaxNumNeighbors", &NeighborLists<ParticleHelper::int_1d_view_type_in_gmls>::computeMaxNumNeighbors, "Compute maximum number of neighbors over all neighborhoods.")
    .def("computeMinNumNeighbors", &NeighborLists<ParticleHelper::int_1d_view_type_in_gmls>::computeMinNumNeighbors, "Compute minimum number of neighbors over all neighborhoods.")
    .def("getNumberOfTargets", &NeighborLists<ParticleHelper::int_1d_view_type_in_gmls>::getNumberOfTargets, "Number of targets.")
    .def("getNumberOfNeighbors", &NeighborLists<ParticleHelper::int_1d_view_type_in_gmls>::getNumberOfNeighborsHost, py::arg("target index"), "Get number of neighbors for target.")
    .def("getMaxNumNeighbors", &NeighborLists<ParticleHelper::int_1d_view_type_in_gmls>::getMaxNumNeighbors, "Get maximum number of neighbors over all neighborhoods.")
    .def("getMinNumNeighbors", &NeighborLists<ParticleHelper::int_1d_view_type_in_gmls>::getMinNumNeighbors, "Get minimum number of neighbors over all neighborhoods.")
    .def("getNeighbor", &NeighborLists<ParticleHelper::int_1d_view_type_in_gmls>::getNeighborHost, py::arg("target index"), py::arg("local neighbor number"), "Get neighbor index from target index and local neighbor number.")
    .def("getTotalNeighborsOverAllLists", &NeighborLists<ParticleHelper::int_1d_view_type_in_gmls>::getTotalNeighborsOverAllListsHost, "Get total storage size of all neighbor lists combined.");

    py::class_<KokkosParser>(m, "KokkosParser")
    .def(py::init<std::vector<std::string>,bool>(), py::arg("args"), py::arg("print") = false)
    .def(py::init<bool>(), py::arg("print") = false)
    .def("status", &KokkosParser::status);

    m.def("getNP", &GMLS::getNP, R"pbdoc(
        Get size of basis.
    )pbdoc");

    m.def("getNN", &GMLS::getNN, R"pbdoc(
        Heuristic number of neighbors.
    )pbdoc");

    m.def("Wab", &GMLS::Wab, py::arg("r"), py::arg("h"), py::arg("weighting type"), py::arg("p"), py::arg("n"), "Evaluate weighting kernel.");

    m.def("is_unique_installation", [] {
        py::exec(R"(
import sys
import os

pycompadre_count = 0
instances = list()
for path in sys.path:
    if os.getcwd()==path:
        print("directory same as working directory, so disregarded:\n  - " + os.getcwd())
    elif 'pycompadre' in path:
        if (os.path.basename(path)) not in instances:
            # a `python setup.py install` will end up here
            print("found pycompadre in directory containing pycompadre name:\n  - ", path)
            instances.append(os.path.basename(path))
            pycompadre_count += 1
    else:
        if os.path.isdir(path):
            for fname in os.listdir(path):
                fname_splitext = os.path.splitext(fname)
                if 'pycompadre' in fname and len(fname_splitext)==2 and fname_splitext[1] in ['.so', '.egg', '.dist-info', '.egg-info']:
                    # a `pip install pycompadre*` or `python setup.py install` will end up here
                    if (os.path.basename(fname)) not in instances:
                        print("found pycompadre*.{so,egg,dist-info,egg-info} at:\n  - " + path + os.sep + fname)
                        instances.append(os.path.basename(fname))
                        pycompadre_count += 1

if pycompadre_count == 1:
    print("SUCCESS. " + str(pycompadre_count) + " instances of pycompadre found.")
elif pycompadre_count == 0:
    assert False, "FAILURE. pycompadre not found, but being called from pycompadre. Please report this to the developers."
elif pycompadre_count == 2:
    assert pycompadre_count==1, "FAILURE. Both pycompadre and pycompadre-serial are installed. Unexpected behavior when calling `import pycompadre`."
else:
    assert False, "FAILURE. More than two instances of pycompadre found, which should not be possible. Please report this to the developers."
        )");
        // can only make it this far if successful
        return true;
    });

    m.def("test", [] {
        py::module_ unittest;
        try {
            unittest = py::module_::import("unittest");
        } catch (...) {
            std::cerr << "test() requires 'unittest' module." << std::endl;
        }

        py::module_ os;
        try {
            os = py::module::import("os");
        } catch (...) {
            std::cerr << "test() requires 'os' module." << std::endl;
        }

        std::string examples_path;
        try {
            py::object this_pycompadre = py::module::import("pycompadre");
            auto location = this_pycompadre.attr("__file__").cast<std::string>();
            py::object path = os.attr("path").attr("dirname")(location);
#ifdef PYTHON_CALLING_BUILD
            examples_path = path.cast<std::string>() + "/examples";
#else
            examples_path = path.cast<std::string>() + "/pycompadre/examples";
#endif
        } catch (...) {
            std::cerr << "Error getting examples path from pycompadre module." << std::endl;
        }

        bool result = false;
        try {
            printf("Running examples in: %s\n", examples_path.c_str());
            py::object loader = unittest.attr("TestLoader")();
            py::object tests  = loader.attr("discover")(examples_path);
            py::object runner = unittest.attr("runner").attr("TextTestRunner")();
            result = runner.attr("run")(tests).attr("wasSuccessful")().cast<bool>();
        } catch (...) {
            std::cerr << "Error running unittests." << std::endl;
        }
        return result;
    });

#ifdef COMPADRE_VERSION_MAJOR
    m.attr("__version__") = std::to_string(COMPADRE_VERSION_MAJOR) + "." + std::to_string(COMPADRE_VERSION_MINOR) + "." + std::to_string(COMPADRE_VERSION_PATCH);
#else
    m.attr("__version__") = "dev";
#endif
}

#endif


