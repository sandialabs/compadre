#ifndef _GMLS_PYTHON_HPP_
#define _GMLS_PYTHON_HPP_

#include <Compadre_GMLS.hpp>
#include <Compadre_Evaluator.hpp>
#include <Compadre_PointCloudSearch.hpp>
#include <Compadre_KokkosParser.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <assert.h>
#include <memory>
#ifdef COMPADRE_USE_MPI
    #include <mpi.h>
#endif

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>

using namespace Compadre;
namespace py = pybind11;

// conversion of numpy arrays to kokkos arrays

template<typename space=Kokkos::HostSpace, typename T>
Kokkos::View<T*, space> convert_np_to_kokkos_1d(py::array_t<T> array, 
        std::string new_label="convert_np_to_kokkos_1d array") {

    compadre_assert_release(array.ndim()==1 && "array must be of rank 1");
    auto np_array = array.template unchecked<1>();

    Kokkos::View<T*, space> kokkos_array_device(new_label, array.shape(0));
    auto kokkos_array_host = Kokkos::create_mirror_view(kokkos_array_device);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, array.shape(0)), [&](int i) {
        kokkos_array_host(i) = np_array(i);
    });
    Kokkos::fence();
    Kokkos::deep_copy(kokkos_array_device, kokkos_array_host);

    return kokkos_array_device;

}

template<typename space=Kokkos::HostSpace, typename layout=Kokkos::LayoutRight, typename T>
Kokkos::View<T**, layout, space> convert_np_to_kokkos_2d(py::array_t<T> array,
        std::string new_label="convert_np_to_kokkos_2d array") {

    compadre_assert_release(array.ndim()==2 && "array must be of rank 1");
    auto np_array = array.template unchecked<2>();

    Kokkos::View<T**, layout, space> kokkos_array_device(new_label, array.shape(0), array.shape(1));
    auto kokkos_array_host = Kokkos::create_mirror_view(kokkos_array_device);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, array.shape(0)), [&](int i) {
        for (int j=0; j<array.shape(1); ++j) {
            kokkos_array_host(i,j) = np_array(i,j);
        }
    });
    Kokkos::fence();
    Kokkos::deep_copy(kokkos_array_device, kokkos_array_host);

    return kokkos_array_device;

}

template<typename space=Kokkos::HostSpace, typename layout=Kokkos::LayoutRight, typename T>
Kokkos::View<T***, layout, space> convert_np_to_kokkos_3d(py::array_t<T> array,
        std::string new_label="convert_np_to_kokkos_3d array") {

    compadre_assert_release(array.ndim()==3 && "array must be of rank 1");
    auto np_array = array.template unchecked<3>();

    Kokkos::View<T***, layout, space> kokkos_array_device(new_label, array.shape(0), array.shape(1), array.shape(2));
    auto kokkos_array_host = Kokkos::create_mirror_view(kokkos_array_device);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, array.shape(0)), [&](int i) {
        for (int j=0; j<array.shape(1); ++j) {
            for (int k=0; k<array.shape(2); ++k) {
                kokkos_array_host(i,j,k) = np_array(i,j,k);
            }
        }
    });
    Kokkos::fence();
    Kokkos::deep_copy(kokkos_array_device, kokkos_array_host);

    return kokkos_array_device;

}


// conversion of kokkos arrays to numpy arrays

template<typename T, typename T2=void>
struct cknp1d {
    py::array_t<typename T::value_type> result;
    cknp1d (T kokkos_array_host) {

        auto dim_out_0 = kokkos_array_host.extent(0);
        result = py::array_t<typename T::value_type>(dim_out_0);
        auto data = result.template mutable_unchecked<1>();
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,dim_out_0), [&](int i) {
            data(i) = kokkos_array_host(i);
        });
        Kokkos::fence();

    }
    py::array_t<typename T::value_type> convert() { return result; }
};

template<typename T>
struct cknp1d<T, enable_if_t<(T::rank!=1)> > {
    py::array_t<typename T::value_type> result;
    cknp1d (T kokkos_array_host) {
        result = py::array_t<typename T::value_type>(0);
    }
    py::array_t<typename T::value_type> convert() { return result; }
};

template<typename T, typename T2=void>
struct cknp2d {
    py::array_t<typename T::value_type> result;
    cknp2d (T kokkos_array_host) {

        auto dim_out_0 = kokkos_array_host.extent(0);
        auto dim_out_1 = kokkos_array_host.extent(1);

        result = py::array_t<typename T::value_type>(dim_out_0*dim_out_1);
        result.resize({dim_out_0,dim_out_1});
        auto data = result.template mutable_unchecked<T::rank>();
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,dim_out_0), [&](int i) {
            for (int j=0; j<dim_out_1; ++j) {
                data(i,j) = kokkos_array_host(i,j);
            }
        });
        Kokkos::fence();

    }
    py::array_t<typename T::value_type> convert() { return result; }
};

template<typename T>
struct cknp2d<T, enable_if_t<(T::rank!=2)> > {
    py::array_t<typename T::value_type> result;
    cknp2d (T kokkos_array_host) {
        result = py::array_t<typename T::value_type>(0);
    }
    py::array_t<typename T::value_type> convert() { return result; }
};

template<typename T, typename T2=void>
struct cknp3d {
    py::array_t<typename T::value_type> result;
    cknp3d (T kokkos_array_host) {

        auto dim_out_0 = kokkos_array_host.extent(0);
        auto dim_out_1 = kokkos_array_host.extent(1);
        auto dim_out_2 = kokkos_array_host.extent(2);

        result = py::array_t<typename T::value_type>(dim_out_0*dim_out_1*dim_out_2);
        result.resize({dim_out_0,dim_out_1,dim_out_2});
        auto data = result.template mutable_unchecked<T::rank>();
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,dim_out_0), [&](int i) {
            for (int j=0; j<dim_out_1; ++j) {
                for (int k=0; k<dim_out_2; ++k) {
                    data(i,j,k) = kokkos_array_host(i,j,k);
                }
            }
        });
        Kokkos::fence();

    }
    py::array_t<typename T::value_type> convert() { return result; }
};

template<typename T>
struct cknp3d<T, enable_if_t<(T::rank!=3)> > {
    py::array_t<typename T::value_type> result;
    cknp3d (T kokkos_array_host) {
        result = py::array_t<typename T::value_type>(0);
    }
    py::array_t<typename T::value_type> convert() { return result; }
};


template<typename T>
py::array_t<typename T::value_type> convert_kokkos_to_np(T kokkos_array_device) {

    // ensure data is accessible
    auto kokkos_array_host =
        Kokkos::create_mirror_view(kokkos_array_device);
    Kokkos::deep_copy(kokkos_array_host, kokkos_array_device);

    py::array_t<typename T::value_type> result;
    if (T::rank==1) {
        result = cknp1d<decltype(kokkos_array_host)>(kokkos_array_host).convert();
    } else if (T::rank==2) {
        result = cknp2d<decltype(kokkos_array_host)>(kokkos_array_host).convert();
    } else if (T::rank==3) {
        result = cknp3d<decltype(kokkos_array_host)>(kokkos_array_host).convert();
    } else {
        compadre_assert_release(false && "Only (1 <= rank <= 3) supported.");
        result = py::array_t<typename T::value_type>(0);
    }
    return result;

}


class ParticleHelper {
public:

    typedef Kokkos::View<double**, Kokkos::HostSpace> double_2d_view_type;
    typedef Kokkos::View<double*, Kokkos::HostSpace> double_1d_view_type;
    typedef Kokkos::View<int*, Kokkos::HostSpace> int_1d_view_type;
    typedef Kokkos::View<int*> int_1d_view_type_in_gmls;

private:

    // gmls_object passed at instantiation 
    // or set after instantiation
    Compadre::GMLS* gmls_object;

    std::shared_ptr<Compadre::PointCloudSearch<double_2d_view_type> > point_cloud_search;

public:

    ParticleHelper() {

        gmls_object = nullptr;

    }

    // handles an lvalued argument (scope of GMLS is bound externally)
    ParticleHelper(GMLS* gmls_instance) : gmls_object(nullptr) {
        setGMLSObject(gmls_instance);
    }

    void setGMLSObject(GMLS* gmls_instance) {
        if (gmls_instance != gmls_object) {
            gmls_object = gmls_instance;
            // generateKDTree only called if new GMLS object provided
            this->generateKDTree();
        }
    }

    decltype(gmls_object) getGMLSObject() const {
        compadre_assert_release(gmls_object!=nullptr &&
                "getGMLSObject() called before setting the GMLS object");
        return gmls_object;
    }

    void setNeighbors(py::array_t<local_index_type> input) {
        py::buffer_info buf = input.request();

        if (buf.ndim != 2) {
            throw std::runtime_error("Number of dimensions must be two");
        }

        auto neighbor_lists = convert_np_to_kokkos_2d(input);
        
        // set values from Kokkos View
        compadre_assert_release(gmls_object!=nullptr &&
                "ParticleHelper used without an internal GMLS object set");
        gmls_object->setNeighborLists(neighbor_lists);
    }

    GMLS::neighbor_lists_type getNeighborLists() {
        compadre_assert_release(gmls_object!=nullptr &&
                "ParticleHelper used without an internal GMLS object set");
        compadre_assert_release((gmls_object->getNeighborLists()->getNumberOfTargets()>0) && 
                "getNeighborLists() called, but neighbor lists were never set.");
        return *(gmls_object->getNeighborLists());
    }

    GMLS::neighbor_lists_type getAdditionalEvaluationIndices() {
        compadre_assert_release(gmls_object!=nullptr &&
                "ParticleHelper used without an internal GMLS object set");
        compadre_assert_release((gmls_object->getAdditionalEvaluationIndices()->getNumberOfTargets()>0) && 
                "getAdditionalEvaluationIndices() called, but additional evaluation indices were never set.");
        return *(gmls_object->getAdditionalEvaluationIndices());
    }

    py::array_t<double> getSourceSites() {
        compadre_assert_release(gmls_object!=nullptr &&
                "ParticleHelper used without an internal GMLS object set");
        compadre_assert_release((gmls_object->getPointConnections()->_source_coordinates.extent(0)>0) && 
                "getSourceSites() called, but source sites were never set.");
        return convert_kokkos_to_np(gmls_object->getPointConnections()->_source_coordinates);
    }

    void setTargetSites(py::array_t<double> input) {
        py::buffer_info buf = input.request();

        if (buf.ndim != 2) {
            throw std::runtime_error("Number of dimensions must be two");
        }

        compadre_assert_release(gmls_object!=nullptr &&
                "ParticleHelper used without an internal GMLS object set");
        if (gmls_object->getGlobalDimensions()!=input.shape(1)) {
            throw std::runtime_error("Second dimension must be the same as GMLS spatial dimension");
        }

        auto target_coords = convert_np_to_kokkos_2d<device_memory_space>(input);
        
        // set values from Kokkos View
        gmls_object->setTargetSites(target_coords);
    }

    py::array_t<double> getTargetSites() {
        compadre_assert_release(gmls_object!=nullptr &&
                "ParticleHelper used without an internal GMLS object set");
        compadre_assert_release((gmls_object->getPointConnections()->_target_coordinates.extent(0)>0) && 
                "getTargetSites() called, but target sites were never set.");
        return convert_kokkos_to_np(gmls_object->getPointConnections()->_target_coordinates);
    }

    void setWindowSizes(py::array_t<double> input) {
        py::buffer_info buf = input.request();

        if (buf.ndim != 1) {
            throw std::runtime_error("Number of dimensions must be one");
        }
        
        auto epsilon = convert_np_to_kokkos_1d<device_memory_space>(input);
        
        compadre_assert_release(gmls_object!=nullptr &&
                "ParticleHelper used without an internal GMLS object set");
        // set values from Kokkos View
        gmls_object->setWindowSizes(epsilon);
    }
 
    py::array_t<double> getWindowSizes() {
        compadre_assert_release(gmls_object!=nullptr &&
                "ParticleHelper used without an internal GMLS object set");
        if (gmls_object->getWindowSizes()->extent(0)<=0) {
            throw std::runtime_error("getWindowSizes() called, but window sizes were never set.");
        }
        return convert_kokkos_to_np(*gmls_object->getWindowSizes());
    }

    void setTangentBundle(py::array_t<double> input) {
        py::buffer_info buf = input.request();

        if (buf.ndim != 3) {
            throw std::runtime_error("Number of dimensions must be three");
        }

        compadre_assert_release(gmls_object!=nullptr &&
                "ParticleHelper used without an internal GMLS object set");
        if (gmls_object->getGlobalDimensions()!=input.shape(1)) {
            throw std::runtime_error("Second dimension must be the same as GMLS spatial dimension");
        }

        if (gmls_object->getGlobalDimensions()!=input.shape(2)) {
            throw std::runtime_error("Second dimension must be the same as GMLS spatial dimension");
        }
        
        auto device_tangent_bundle = convert_np_to_kokkos_3d<device_memory_space>(input);
        
        // set values from Kokkos View
        gmls_object->setTangentBundle(device_tangent_bundle);
    }

    py::array_t<double> getTangentBundle() {
        compadre_assert_release(gmls_object!=nullptr &&
                "ParticleHelper used without an internal GMLS object set");
        compadre_assert_release((gmls_object->getTangentDirections()->extent(0)>0) && 
                "getTangentBundle() called, but tangent bundle was never set.");
        return convert_kokkos_to_np(*gmls_object->getTangentDirections());
    }

    void setReferenceOutwardNormalDirection(py::array_t<double> input, bool use_to_orient_surface = true) {
        py::buffer_info buf = input.request();

        if (buf.ndim != 2) {
            throw std::runtime_error("Number of dimensions must be two");
        }

        compadre_assert_release(gmls_object!=nullptr &&
                "ParticleHelper used without an internal GMLS object set");
        if (gmls_object->getGlobalDimensions()!=input.shape(1)) {
            throw std::runtime_error("Second dimension must be the same as GMLS spatial dimension");
        }
        
        auto device_reference_normal_directions = convert_np_to_kokkos_2d<device_memory_space>(input);
        
        // set values from Kokkos View
        gmls_object->setReferenceOutwardNormalDirection(device_reference_normal_directions, true /* use to orient surface*/);
    }

    py::array_t<double> getReferenceOutwardNormalDirection() {
        compadre_assert_release(gmls_object!=nullptr &&
                "ParticleHelper used without an internal GMLS object set");
        compadre_assert_release((gmls_object->getReferenceNormalDirections()->extent(0)>0) && 
                "getReferenceNormalDirections() called, but normal directions were never set.");
        return convert_kokkos_to_np(*gmls_object->getReferenceNormalDirections());
    }

    void generateKDTree() {
        compadre_assert_release(gmls_object!=nullptr &&
                "ParticleHelper used without an internal GMLS object set");
        point_cloud_search = std::shared_ptr<Compadre::PointCloudSearch<double_2d_view_type> >(
                new Compadre::PointCloudSearch<double_2d_view_type>(
                    convert_np_to_kokkos_2d<host_memory_space>(
                        convert_kokkos_to_np(gmls_object->getPointConnections()->_source_coordinates)), 
                    gmls_object->getGlobalDimensions()
                )
        );
    }

    void generateKDTree(py::array_t<double> input) {
        py::buffer_info buf = input.request();

        if (buf.ndim != 2) {
            throw std::runtime_error("Number of dimensions must be two");
        }

        compadre_assert_release(gmls_object!=nullptr &&
                "ParticleHelper used without an internal GMLS object set");
        if (gmls_object->getGlobalDimensions()!=input.shape(1)) {
            throw std::runtime_error("Second dimension must be the same as GMLS spatial dimension");
        }

        auto source_coords = convert_np_to_kokkos_2d<device_memory_space>(input);
        
        // set values from Kokkos View
        gmls_object->setSourceSites(source_coords);
        this->generateKDTree();
    }

    void generateNeighborListsFromKNNSearchAndSet(py::array_t<double> input, int poly_order, int dimension = 3, double epsilon_multiplier = 1.6, double max_search_radius = 0.0, bool scale_k_neighbor_radius = true, bool scale_num_neighbors = false) {

        compadre_assert_release(gmls_object!=nullptr &&
                "ParticleHelper used without an internal GMLS object set");
        int neighbors_needed = Compadre::GMLS::getNP(poly_order, dimension, gmls_object->getReconstructionSpace());

        py::buffer_info buf = input.request();

        if (buf.ndim != 2) {
            throw std::runtime_error("Number of dimensions must be two");
        }

        if (gmls_object->getGlobalDimensions()!=input.shape(1)) {
            throw std::runtime_error("Second dimension must be the same as GMLS spatial dimension");
        }
        
        auto target_coords = convert_np_to_kokkos_2d<host_memory_space>(input);
        gmls_object->setTargetSites(target_coords);

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
        gmls_object->setNeighborLists(neighbor_lists, number_of_neighbors_list);
        gmls_object->setWindowSizes(epsilon);

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
        compadre_assert_release(gmls_object!=nullptr &&
                "ParticleHelper used without an internal GMLS object set");
        if (gmls_object->getGlobalDimensions()!=input_coordinates.shape(1)) {
            throw std::runtime_error("Second dimension of input coordinates must be the same as GMLS spatial dimension");
        }
        
        // overwrite existing data assuming a 2D layout
        auto neighbor_lists = convert_np_to_kokkos_2d(input_indices);
        // overwrite existing data assuming a 2D layout
        auto extra_coords = convert_np_to_kokkos_2d(input_coordinates);
    
        // set values from Kokkos View
        gmls_object->setAdditionalEvaluationSitesData(neighbor_lists, extra_coords);
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

        compadre_assert_release(gmls_object!=nullptr &&
                "ParticleHelper used without an internal GMLS object set");
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

        compadre_assert_release(gmls_object!=nullptr &&
                "ParticleHelper used without an internal GMLS object set");
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

        if (dim_out_1>=2) {
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

        compadre_assert_release(gmls_object!=nullptr &&
                "ParticleHelper used without an internal GMLS object set");
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
        int max_additional_sites = gmls_object->getAdditionalEvaluationIndices()->getMaxNumNeighbors() + 1;
        gmls_object->getAdditionalEvaluationIndices()->computeMinNumNeighbors();
        int min_additional_sites = gmls_object->getAdditionalEvaluationIndices()->getMinNumNeighbors();

        // set dim_out_0 to 1 if setAdditionalEvaluationSitesData never called
        auto additional_evaluation_coords_size = gmls_object->getAdditionalPointConnections()->_source_coordinates.size();
        size_t dim_out_0 = (additional_evaluation_coords_size==0) ? 1 : max_additional_sites;
        auto dim_out_1 = output_values.extent(0);
        auto dim_out_2 = output_values.extent(1);

        auto result = py::array_t<double>(dim_out_0*dim_out_1*dim_out_2);

        if (dim_out_2>=2) {
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

        compadre_assert_release(gmls_object!=nullptr &&
                "ParticleHelper used without an internal GMLS object set");
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

    py::class_<SamplingFunctional>(m, "SamplingFunctional")
    .def(py::pickle(
        [](const SamplingFunctional &sf) { // __getstate__
            return py::make_tuple(sf.input_rank, sf.output_rank, sf.use_target_site_weights,
                                  sf.nontrivial_nullspace, sf.transform_type, sf.id);
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 6)
                throw std::runtime_error("Invalid state!");
            SamplingFunctional sf(t[0].cast<int>(), t[1].cast<int>(), t[2].cast<bool>(),
                                  t[3].cast<bool>(), t[4].cast<int>(), t[5].cast<int>());
            return sf;
        }
    ));
    py::dict sampling_functional;
    sampling_functional["PointSample"] = PointSample;
    sampling_functional["VectorPointSample"] = VectorPointSample;
    sampling_functional["ManifoldVectorPointSample"] = ManifoldVectorPointSample;
    sampling_functional["StaggeredEdgeAnalyticGradientIntegralSample"] = StaggeredEdgeAnalyticGradientIntegralSample;
    sampling_functional["StaggeredEdgeIntegralSample"] = StaggeredEdgeIntegralSample;
    sampling_functional["VaryingManifoldVectorPointSample"] = VaryingManifoldVectorPointSample;
    sampling_functional["FaceNormalIntegralSample"] = FaceNormalIntegralSample;
    sampling_functional["FaceNormalAverageSample"] = FaceNormalAverageSample;
    sampling_functional["EdgeTangentIntegralSample"] = EdgeTangentIntegralSample;
    sampling_functional["EdgeTangentAverageSample"] = EdgeTangentAverageSample;
    sampling_functional["CellAverageSample"] = CellAverageSample;
    sampling_functional["CellIntegralSample"] = CellIntegralSample;
    m.attr("SamplingFunctionals") = sampling_functional;

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
    .value("CellAverageEvaluation", TargetOperation::CellAverageEvaluation)
    .value("CellIntegralEvaluation", TargetOperation::CellIntegralEvaluation)
    .value("FaceNormalAverageEvaluation", TargetOperation::FaceNormalAverageEvaluation)
    .value("FaceNormalIntegralEvaluation", TargetOperation::FaceNormalIntegralEvaluation)
    .value("EdgeTangentAverageEvaluation", TargetOperation::EdgeTangentAverageEvaluation)
    .value("EdgeTangentIntegralEvaluation", TargetOperation::EdgeTangentIntegralEvaluation)
    .export_values();


    py::enum_<ReconstructionSpace>(m, "ReconstructionSpace")
    .value("ScalarTaylorPolynomial", ReconstructionSpace::ScalarTaylorPolynomial)
    .value("VectorTaylorPolynomial", ReconstructionSpace::VectorTaylorPolynomial)
    .value("VectorOfScalarClonesTaylorPolynomial", ReconstructionSpace::VectorOfScalarClonesTaylorPolynomial)
    .value("DivergenceFreeVectorTaylorPolynomial", ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial)
    .value("BernsteinPolynomial", ReconstructionSpace::BernsteinPolynomial)
    .export_values();

    py::enum_<WeightingFunctionType>(m, "WeightingFunctionType")
    .value("Power", WeightingFunctionType::Power)
    .value("Gaussian", WeightingFunctionType::Gaussian)
    .value("CubicSpline", WeightingFunctionType::CubicSpline)
    .value("Cosine", WeightingFunctionType::Cosine)
    .value("Sigmoid", WeightingFunctionType::Sigmoid)
    .export_values();

    py::enum_<DenseSolverType>(m, "DenseSolverType")
    .value("QR", DenseSolverType::QR)
    .value("LU", DenseSolverType::LU)
    .export_values();

    py::enum_<ProblemType>(m, "ProblemType")
    .value("STANDARD", ProblemType::STANDARD)
    .value("MANIFOLD", ProblemType::MANIFOLD)
    .export_values();

    py::enum_<ConstraintType>(m, "ConstraintType")
    .value("NO_CONSTRAINT", ConstraintType::NO_CONSTRAINT)
    .value("NEUMANN_GRAD_SCALAR", ConstraintType::NEUMANN_GRAD_SCALAR)
    .export_values();

    py::enum_<QuadratureType>(m, "QuadratureType")
    .value("INVALID", QuadratureType::INVALID)
    .value("LINE", QuadratureType::LINE)
    .value("TRI", QuadratureType::TRI)
    .value("QUAD", QuadratureType::QUAD)
    .value("TET", QuadratureType::TET)
    .value("HEX", QuadratureType::HEX)
    .export_values();
    
    py::class_<SolutionSet<host_memory_space> >(m, "SolutionSet", R"pbdoc(
        Class containing solution data from GMLS problems
    )pbdoc")
    .def("getAlpha", &SolutionSet<host_memory_space>::getAlpha<>, py::arg("lro"), py::arg("target_index"), py::arg("output_component_axis_1"), py::arg("output_component_axis_2"), py::arg("neighbor_index"), py::arg("input_component_axis_1"), py::arg("input_component_axis_2"), py::arg("additional_evaluation_site")=0);

    // helper functions
    py::class_<ParticleHelper>(m, "ParticleHelper", R"pbdoc(
        Class to manage calling PointCloudSearch, moving data to/from Numpy arrays in Kokkos::Views,
        and applying GMLS solutions to multidimensional data arrays 
    )pbdoc")
    .def(py::init<>(), R"pbdoc(
        Requires follow-on call to setGMLSObject(gmls_obj) before use
    )pbdoc")
    .def(py::init<GMLS*>(), py::arg("gmls_instance"))
    .def("generateKDTree", overload_cast_<py::array_t<double> >()(&ParticleHelper::generateKDTree))
    .def("generateNeighborListsFromKNNSearchAndSet", &ParticleHelper::generateNeighborListsFromKNNSearchAndSet, 
            py::arg("target_sites"), py::arg("poly_order"), py::arg("dimension") = 3, py::arg("epsilon_multiplier") = 1.6, 
            py::arg("max_search_radius") = 0.0, py::arg("scale_k_neighbor_radius") = true, 
            py::arg("scale_num_neighbors") = false)
    .def("setNeighbors", &ParticleHelper::setNeighbors, py::arg("neighbor_lists"), 
            "Sets neighbor lists from 2D array where first column is number of neighbors for corresponding row's target site.")
    .def("setGMLSObject", &ParticleHelper::setGMLSObject, py::arg("gmls_object"))
    .def("getGMLSObject", &ParticleHelper::getGMLSObject, py::return_value_policy::reference_internal)
    .def("getAdditionalEvaluationIndices", &ParticleHelper::getAdditionalEvaluationIndices, py::return_value_policy::reference_internal)
    .def("getNeighborLists", &ParticleHelper::getNeighborLists, py::return_value_policy::reference_internal)
    .def("setWindowSizes", &ParticleHelper::setWindowSizes, py::arg("window_sizes"))
    .def("getWindowSizes", &ParticleHelper::getWindowSizes, py::return_value_policy::take_ownership)
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
    .def("applyStencil", &ParticleHelper::applyStencil, py::arg("input_data"), py::arg("target_operation")=TargetOperation::ScalarPointEvaluation, py::arg("sampling_functional")=PointSample, py::arg("evaluation_site_local_index")=0, py::return_value_policy::take_ownership)
    .def(py::pickle(
        [](const ParticleHelper &helper) { // __getstate__
            return py::make_tuple();
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 0)
                throw std::runtime_error("Invalid state!");
            ParticleHelper helper;
            return helper;
        }
    ));

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
    .def(py::init<ReconstructionSpace,SamplingFunctional,SamplingFunctional,int,int,std::string,std::string,std::string,int>(),
            py::arg("reconstruction_space"),py::arg("polynomial_sampling_functional"),py::arg("data_sampling_functional"),
            py::arg("poly_order"),py::arg("dimension")=3,py::arg("dense_solver_type")="QR", 
            py::arg("problem_type")="STANDARD", py::arg("constraint_type")="NO_CONSTRAINT", 
            py::arg("curvature_poly_order")=2)
    .def("getWeightingParameter", &GMLS::getWeightingParameter, py::arg("parameter index")=0, "Get weighting kernel parameter[index].")
    .def("setWeightingParameter", &GMLS::setWeightingParameter, py::arg("parameter value"), py::arg("parameter index")=0, "Set weighting kernel parameter[index] to parameter value.")
    .def("setWeightingPower", &GMLS::setWeightingParameter, py::arg("parameter value"), py::arg("parameter index")=0, "Set weighting kernel parameter[index] to parameter value. [DEPRECATED]")
    .def("getWeightingType", &GMLS::getWeightingType, "Get the weighting type.")
    .def("setWeightingType", overload_cast_<const std::string&>()(&GMLS::setWeightingType), "Set the weighting type with a string.")
    .def("setWeightingType", overload_cast_<WeightingFunctionType>()(&GMLS::setWeightingType), "Set the weighting type with a WeightingFunctionType.")
    .def("setOrderOfQuadraturePoints", &GMLS::setOrderOfQuadraturePoints, py::arg("order"))
    .def("setDimensionOfQuadraturePoints", &GMLS::setDimensionOfQuadraturePoints, py::arg("dim"))
    .def("setQuadratureType", &GMLS::setQuadratureType, py::arg("quadrature_type"))
    .def("addTargets", overload_cast_<TargetOperation>()(&GMLS::addTargets), "Add a target operation.")
    .def("addTargets", overload_cast_<std::vector<TargetOperation> >()(&GMLS::addTargets), "Add a list of target operations.")
    .def("generateAlphas", &GMLS::generateAlphas, py::arg("number_of_batches")=1, py::arg("keep_coefficients")=false, py::arg("clear_cache")=true)
    .def("containsValidAlphas", &GMLS::containsValidAlphas)
    .def("getSolutionSet", &GMLS::getSolutionSetHost, py::arg("validity_check")=true, py::return_value_policy::reference_internal)
    .def("getNP", &GMLS::getNP, "Get size of basis.")
    .def("getNN", &GMLS::getNN, "Heuristic number of neighbors.")
    .def("setTargetExtraData", [] (GMLS* gmls, py::array_t<double> extra_data) {
        py::buffer_info buf = extra_data.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("Number of dimensions must be two");
        }
        gmls->setTargetExtraData(convert_np_to_kokkos_2d(extra_data));
    })
    .def("setSourceExtraData", [] (GMLS* gmls, py::array_t<double> extra_data) {
        py::buffer_info buf = extra_data.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("Number of dimensions must be two");
        }
        gmls->setSourceExtraData(convert_np_to_kokkos_2d(extra_data));
    })
    .def(py::pickle(
        [](const GMLS &gmls) { // __getstate__

            // values needed for constructor
            auto rs   = gmls.getReconstructionSpace();
            auto psf  = gmls.getPolynomialSamplingFunctional();
            auto dsf  = gmls.getDataSamplingFunctional();
            auto po   = gmls.getPolynomialOrder();
            auto gdim = gmls.getGlobalDimensions();
            auto dst  = gmls.getDenseSolverType();
            auto pt   = gmls.getProblemType();
            auto ct   = gmls.getConstraintType();
            auto cpo  = gmls.getCurvaturePolynomialOrder();

            // values needed to get weighting types the same
            auto wt   = gmls.getWeightingType();
            auto wp0  = gmls.getWeightingParameter(0);
            auto wp1  = gmls.getWeightingParameter(1);

            auto mwt  = gmls.getManifoldWeightingType();
            auto mwp0 = gmls.getManifoldWeightingParameter(0);
            auto mwp1 = gmls.getManifoldWeightingParameter(1);

            // get all previously added targets
            auto d_lro  = const_cast<GMLS&>(gmls).getSolutionSetDevice(false)->_lro;
            auto lro    = Kokkos::create_mirror_view(d_lro);
            Kokkos::deep_copy(lro, d_lro);
            auto lro_list = py::list();
            for (int i=0; i<lro.extent(0); ++i) {
                lro_list.append(lro[i]);
            }

            auto nonconst_gmls = *const_cast<Compadre::GMLS*>(&gmls);

            // get all source sites
            auto ss     = nonconst_gmls.getPointConnections()->_source_coordinates;
            auto np_ss  = convert_kokkos_to_np(ss);

            // get all target sites
            auto ts     = nonconst_gmls.getPointConnections()->_target_coordinates;
            auto np_ts  = convert_kokkos_to_np(ts);

            // get all window sizes
            auto ws     = nonconst_gmls.getWindowSizes();
            auto np_ws  = convert_kokkos_to_np(*ws);

            // get all neighbors
            auto nl     = nonconst_gmls.getNeighborLists();
            auto cr     = nl->getNeighborLists();
            auto np_cr  = convert_kokkos_to_np(cr);
            auto nnl    = nl->getNumberOfNeighborsList();
            auto np_nnl = convert_kokkos_to_np(nnl);

            // get all additional evaluation indices
            auto a_nl     = nonconst_gmls.getAdditionalEvaluationIndices();
            auto a_cr     = a_nl->getNeighborLists();
            auto np_a_cr  = convert_kokkos_to_np(a_cr);
            auto a_nnl    = a_nl->getNumberOfNeighborsList();
            auto np_a_nnl = convert_kokkos_to_np(a_nnl);

            // get all additional evaluation coordinates
            auto a_es = nonconst_gmls.getAdditionalPointConnections()->_source_coordinates;
            auto np_a_es  = convert_kokkos_to_np(a_es);
 
            // get all relevant details from GMLS class
            return py::make_tuple(rs, psf, dsf, po, gdim, dst, pt, ct, cpo, 
                                  wt, wp0, wp1, mwt, mwp0, mwp1, lro_list,
                                  np_ss, np_ts, np_ws, np_cr, np_nnl,
                                  np_a_cr, np_a_nnl, np_a_es);

        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 24)
                throw std::runtime_error("Invalid state!");

            // reinstantiate with details
            // no need to keep solutions
            GMLS gmls(t[0].cast<const ReconstructionSpace>(),
                 t[1].cast<const SamplingFunctional>(),
                 t[2].cast<const SamplingFunctional>(),
                 t[3].cast<const int>(),
                 t[4].cast<const int>(),
                 t[5].cast<const DenseSolverType>(),
                 t[6].cast<const ProblemType>(),
                 t[7].cast<const ConstraintType>(),
                 t[8].cast<const int>());

            gmls.setWeightingType(t[9].cast<WeightingFunctionType>());
            gmls.setWeightingParameter(t[10].cast<int>(), 0);
            gmls.setWeightingParameter(t[11].cast<int>(), 1);

            gmls.setCurvatureWeightingType(t[12].cast<WeightingFunctionType>());
            gmls.setCurvatureWeightingParameter(t[13].cast<int>(), 0);
            gmls.setCurvatureWeightingParameter(t[14].cast<int>(), 1);

            auto lro_list = t[15].cast<py::list>();
            for (py::handle obj_i : lro_list) {
                gmls.addTargets(obj_i.cast<TargetOperation>());
            }
            
            // need to convert from numpy back to kokkos before setting problem data
            auto ss      = t[16].cast<py::array_t<double> >();
            auto k_ss    = convert_np_to_kokkos_2d(ss);
            auto ts      = t[17].cast<py::array_t<double> >();
            auto k_ts    = convert_np_to_kokkos_2d(ts);
            auto ws      = t[18].cast<py::array_t<double> >();
            auto k_ws    = convert_np_to_kokkos_1d(ws);
            auto cr      = t[19].cast<py::array_t<int> >();
            auto k_cr    = convert_np_to_kokkos_1d(cr);
            auto nnl     = t[20].cast<py::array_t<int> >();
            auto k_nnl   = convert_np_to_kokkos_1d(nnl);
            auto a_cr    = t[21].cast<py::array_t<int> >();
            auto k_a_cr  = convert_np_to_kokkos_1d(a_cr);
            auto a_nnl   = t[22].cast<py::array_t<int> >();
            auto k_a_nnl = convert_np_to_kokkos_1d(a_nnl);
            auto a_es    = t[23].cast<py::array_t<double> >();
            auto k_a_es  = convert_np_to_kokkos_2d(a_es);

            gmls.setAdditionalEvaluationSitesData(k_a_cr, k_a_nnl, k_a_es);
            gmls.setProblemData(k_cr, k_nnl, k_ss, k_ts, k_ws);

            // this object will be copy constructed
            return gmls;
        }
    ));

    py::class_<NeighborLists<ParticleHelper::int_1d_view_type_in_gmls> >(m, "NeighborLists")
    .def("computeMaxNumNeighbors", &NeighborLists<ParticleHelper::int_1d_view_type_in_gmls>::computeMaxNumNeighbors, "Compute maximum number of neighbors over all neighborhoods.")
    .def("computeMinNumNeighbors", &NeighborLists<ParticleHelper::int_1d_view_type_in_gmls>::computeMinNumNeighbors, "Compute minimum number of neighbors over all neighborhoods.")
    .def("getNumberOfTargets", &NeighborLists<ParticleHelper::int_1d_view_type_in_gmls>::getNumberOfTargets, "Number of targets.")
    .def("getNumberOfNeighbors", &NeighborLists<ParticleHelper::int_1d_view_type_in_gmls>::getNumberOfNeighborsHost, py::arg("target index"), "Get number of neighbors for target.")
    .def("getMaxNumNeighbors", &NeighborLists<ParticleHelper::int_1d_view_type_in_gmls>::getMaxNumNeighbors, "Get maximum number of neighbors over all neighborhoods.")
    .def("getMinNumNeighbors", &NeighborLists<ParticleHelper::int_1d_view_type_in_gmls>::getMinNumNeighbors, "Get minimum number of neighbors over all neighborhoods.")
    .def("getNeighbor", &NeighborLists<ParticleHelper::int_1d_view_type_in_gmls>::getNeighborHost, py::arg("target index"), py::arg("local neighbor number"), "Get neighbor index from target index and local neighbor number.")
    .def("getTotalNeighborsOverAllLists", &NeighborLists<ParticleHelper::int_1d_view_type_in_gmls>::getTotalNeighborsOverAllListsHost, "Get total storage size of all neighbor lists combined.");

    py::class_<Quadrature>(m, "Quadrature")
    .def(py::init<int, int, std::string>(), py::arg("order_of_quadrature_points"), py::arg("dimension_of_quadrature_points"), 
         py::arg("quadrature_type") = "LINE")
    .def("validQuadrature", &Quadrature::validQuadrature, "Has quadrature been generated.")
    .def("getNumberOfQuadraturePoints", &Quadrature::getNumberOfQuadraturePoints)
    .def("getOrderOfQuadraturePoints", &Quadrature::getOrderOfQuadraturePoints)
    .def("getDimensionOfQuadraturePoints", &Quadrature::getDimensionOfQuadraturePoints)
    .def("getQuadratureType", &Quadrature::getQuadratureType)
    .def("getWeights", [] (const Quadrature &quadrature) {
        return convert_kokkos_to_np(quadrature.getWeights());
    })
    .def("getSites", [] (const Quadrature &quadrature) {
        return convert_kokkos_to_np(quadrature.getSites());
    });

    auto m_kokkos = m.def_submodule("Kokkos", R"pbdoc(
        Namespace with a subset of Kokkos functionality. KokkosParser class should be preferred.
    )pbdoc");
    py::class_<Kokkos::InitArguments>(m_kokkos, "InitArguments")
    .def(py::init<int, int, int, bool, bool>(), 
            py::arg("num_threads") = -1, py::arg("num_numa") = -1, py::arg("device_id") = -1,
            py::arg("disable_warnings") = false, py::arg("tune_internals") = false);

    m_kokkos
    .def("initialize", overload_cast_<Kokkos::InitArguments>()(&Kokkos::initialize), 
            py::arg("init_args") = Kokkos::InitArguments())
    .def("initialize", [](py::list args) {
        std::vector<char*> char_args;
        for (py::handle arg : args) {
            char_args.push_back((char*)py::cast<std::string>(arg).data());
        }
        char_args.push_back(nullptr);
        int argc = (int)args.size();
        Kokkos::initialize(argc, char_args.data());     
    })
    .def("finalize", &Kokkos::finalize)
    .def("is_initialized", &Kokkos::is_initialized)
    .def("status", &KokkosParser::status);

    py::class_<KokkosParser>(m, "KokkosParser", R"pbdoc(
        Class wrapping the functionality of Kokkos::ScopeGuard. Multiple instances can  
        can be instantiated and go out of scope with only one instance initiating Kokkos
        and the last instance finalizing Kokkos when it goes out of scope.
    )pbdoc")
    .def(py::init<Kokkos::InitArguments,bool>(), py::arg("init_args"), py::arg("print") = false)
    .def(py::init<std::vector<std::string>,bool>(), py::arg("args"), py::arg("print") = false)
    .def(py::init<bool>(), py::arg("print") = false)
    .def_static("status", &KokkosParser::status);

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


