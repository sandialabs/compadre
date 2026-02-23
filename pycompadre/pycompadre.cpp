#ifndef _GMLS_PYTHON_HPP_
#define _GMLS_PYTHON_HPP_

//#include "headers.hpp"
//#<nanobind/nanobind.h>
#include <nanobind/nanobind.h>
#include <vector>

NB_MAKE_OPAQUE(std::vector<int>);
NB_MAKE_OPAQUE(std::vector<double>);
NB_MAKE_OPAQUE(std::vector<size_t>);

#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/eval.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <Compadre_GMLS.hpp>
#include <Compadre_Evaluator.hpp>
#include <Compadre_PointCloudSearch.hpp>
#include <Compadre_KokkosParser.hpp>
#include <Compadre_Typedefs.hpp>
#include <Kokkos_Core.hpp>
#include <assert.h>
#include <memory>
#include <new>
#include <iostream>
#ifdef COMPADRE_USE_MPI
    #include <mpi.h>
#endif


//template< bool B, class T = void >
//using enable_if_t = typename std::enable_if<B,T>::type;

using namespace Compadre;
namespace nb = nanobind;

template<typename T>
nb::ndarray<nb::numpy, T> create_new_array(const std::vector<size_t> dim_out) {
    size_t total_entries = 1;
    for (size_t i=0; i<dim_out.size(); ++i) {
        total_entries *= dim_out[i];
    }
    auto data_vec = new std::vector<T>(total_entries);
    std::fill(data_vec->begin(), data_vec->end(), 0.0);
    nb::capsule owner(data_vec, [](void *v) noexcept {
        delete static_cast<std::vector<T> *>(v);
    });
    return nb::ndarray<nb::numpy, T>(
        data_vec->data(),
        dim_out.size(),
        dim_out.data(),
        std::move(owner)
    );
}


// conversion of numpy arrays to kokkos arrays

template<typename space=Kokkos::HostSpace, typename T>
Kokkos::View<T*, space> convert_np_to_kokkos_1d(nb::ndarray<nb::numpy, T> array, 
        std::string new_label="convert_np_to_kokkos_1d array") {

    compadre_assert_release(array.ndim()==1 && "array must be of rank 1");
    size_t stride_0 = array.stride(0);
    if (stride_0!=1) {
        throw std::runtime_error("convert_np_to_kokkos_1d was passed an array with noncontiguous data. Please reformat array.");
    }

    Kokkos::View<T*, space> kokkos_array_device(new_label, array.shape(0));
    auto kokkos_array_host = Kokkos::create_mirror_view(kokkos_array_device);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, array.shape(0)), [&](int i) {
        //kokkos_array_host(i) = array(i);
        //kokkos_array_host(i) = *(array.data() + array(i));
        kokkos_array_host(i) = *(T*)((T*)array.data() + i);
        //kokkos_array_host(i) = *(array.data(i));

    });
    Kokkos::fence();
    Kokkos::deep_copy(kokkos_array_device, kokkos_array_host);

    return kokkos_array_device;

}

template<typename space=Kokkos::HostSpace, typename layout=Kokkos::LayoutRight, typename T>
Kokkos::View<T**, layout, space> convert_np_to_kokkos_2d(nb::ndarray<nb::numpy, T> array,
        std::string new_label="convert_np_to_kokkos_2d array") {

    compadre_assert_release(array.ndim()==2 && "array must be of rank 2");
    size_t cols = array.shape(1);
    size_t stride_0 = array.stride(0);
    size_t stride_1 = array.stride(1);
    if (stride_0!=cols || stride_1!=1) {
        throw std::runtime_error("convert_np_to_kokkos_2d was passed an array with noncontiguous data. Please reformat array.");
    }

    Kokkos::View<T**, layout, space> kokkos_array_device(new_label, array.shape(0), array.shape(1));
    auto kokkos_array_host = Kokkos::create_mirror_view(kokkos_array_device);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, array.shape(0)), [&](int i) {
        for (int j=0; j<array.shape(1); ++j) {
            //kokkos_array_host(i,j) = array(i*array.shape(1)+j);
            //kokkos_array_host(i,j) = *(array.data() + array(i*array.shape(1)+j));
            kokkos_array_host(i,j) = *(T*)((T*)array.data() + i*array.shape(1)+j);
            //kokkos_array_host(i,j) = *(array.data(i,j));
        }
    });
    Kokkos::fence();
    Kokkos::deep_copy(kokkos_array_device, kokkos_array_host);

    return kokkos_array_device;

}

template<typename space=Kokkos::HostSpace, typename layout=Kokkos::LayoutRight, typename T>
Kokkos::View<T***, layout, space> convert_np_to_kokkos_3d(nb::ndarray<nb::numpy, T> array,
        std::string new_label="convert_np_to_kokkos_3d array") {

    compadre_assert_release(array.ndim()==3 && "array must be of rank 3");
    size_t rows = array.shape(0);
    size_t cols = array.shape(1);
    size_t nextdim = array.shape(2);
    size_t stride_0 = array.stride(0);
    size_t stride_1 = array.stride(1);
    size_t stride_2 = array.stride(2);
    if (stride_0!=cols*nextdim || stride_1!=nextdim || stride_2!=1) {
        throw std::runtime_error("convert_np_to_kokkos_3d was passed an array with noncontiguous data. Please reformat array.");
    }


    Kokkos::View<T***, layout, space> kokkos_array_device(new_label, array.shape(0), array.shape(1), array.shape(2));
    auto kokkos_array_host = Kokkos::create_mirror_view(kokkos_array_device);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, array.shape(0)), [&](int i) {
        for (int j=0; j<array.shape(1); ++j) {
            for (int k=0; k<array.shape(2); ++k) {
                //kokkos_array_host(i,j,k) = array(i,j,k);
                //kokkos_array_host(i,j,k) = *(array.data() + array(i*array.shape(1)*array.shape(2)+j*array.shape(2)*j+k));
                kokkos_array_host(i,j,k) = *(T*)((T*)array.data() + i*array.shape(1)*array.shape(2)+j*array.shape(2)+k);
                //kokkos_array_host(i,j,k) = *(array.data(i,j,k));
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
    nb::ndarray<nb::numpy, typename T::value_type> result;
    cknp1d (T kokkos_array_host) {

        auto dim_out_0 = kokkos_array_host.extent(0);
        std::vector<size_t> dim_out = {dim_out_0};
        result = create_new_array<typename T::value_type>(dim_out);
        typename T::value_type* data_ptr = (typename T::value_type*)result.data();
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,dim_out_0), [&](int i) {
            //result(i) = kokkos_array_host(i);
            data_ptr[i] = kokkos_array_host(i);
        });
        Kokkos::fence();

    }
    nb::ndarray<nb::numpy, typename T::value_type> convert() { return result; }
};

template<typename T>
struct cknp1d<T, Compadre::enable_if_t<(T::rank!=1)> > {
    nb::ndarray<nb::numpy, typename T::value_type> result;
    cknp1d (T kokkos_array_host) {
        std::vector<size_t> dim_out = {0};
        result = create_new_array<typename T::value_type>(dim_out);
    }
    nb::ndarray<nb::numpy, typename T::value_type> convert() { return result; }
};

template<typename T, typename T2=void>
struct cknp2d {
    nb::ndarray<nb::numpy, typename T::value_type> result;
    cknp2d (T kokkos_array_host) {

        size_t dim_out_0 = kokkos_array_host.extent(0);
        size_t dim_out_1 = kokkos_array_host.extent(1);

        std::vector<size_t> dim_out = {dim_out_0, dim_out_1};
        result = create_new_array<typename T::value_type>(dim_out);
        typename T::value_type* data_ptr = (typename T::value_type*)result.data();
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,dim_out_0), [&](size_t i) {
            for (size_t j=0; j<dim_out_1; ++j) {
                //result(i,j) = kokkos_array_host(i,j);
                //*(typename T*)((typename T*)result.data() + i*dim_out_1 + j) = kokkos_array_host(i,j);
                data_ptr[i*dim_out_1 + j] = kokkos_array_host(i,j);
                //*(result.data(i,j)) = kokkos_array_host(i,j);
            }
        });
        Kokkos::fence();

    }
    nb::ndarray<nb::numpy, typename T::value_type> convert() { return result; }
};

template<typename T>
struct cknp2d<T, Compadre::enable_if_t<(T::rank!=2)> > {
    nb::ndarray<nb::numpy, typename T::value_type> result;
    cknp2d (T kokkos_array_host) {
        //result = create_new_1d_array<typename T::value_type>(0);
        std::vector<size_t> dim_out = {0, 0};//dim_out_0, dim_out_1};
        result = create_new_array<typename T::value_type>(dim_out);
    }
    nb::ndarray<nb::numpy, typename T::value_type> convert() { return result; }
};

template<typename T, typename T2=void>
struct cknp3d {
    nb::ndarray<nb::numpy, typename T::value_type> result;
    cknp3d (T kokkos_array_host) {

        auto dim_out_0 = kokkos_array_host.extent(0);
        auto dim_out_1 = kokkos_array_host.extent(1);
        auto dim_out_2 = kokkos_array_host.extent(2);

        std::vector<size_t> dim_out = {dim_out_0, dim_out_1, dim_out_2};
        result = create_new_array<typename T::value_type>(dim_out);
        typename T::value_type* data_ptr = (typename T::value_type*)result.data();
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,dim_out_0), [&](int i) {
            for (int j=0; j<dim_out_1; ++j) {
                for (int k=0; k<dim_out_2; ++k) {
                    //result(i,j,k) = kokkos_array_host(i,j,k);
                    //*(result.data(i,j,k)) = kokkos_array_host(i,j,k);
                    //*(T*)((T*)result.data() + i*result.shape(1)*result.shape(2)+j*result.shape(2)*j+k) = kokkos_array_host(i,j,k);
                    data_ptr[i*result.shape(1)*result.shape(2)+j*result.shape(2)+k] = kokkos_array_host(i,j,k);
                }
            }
        });
        Kokkos::fence();

    }
    nb::ndarray<nb::numpy, typename T::value_type, nb::ndim<3> > convert() { return result; }
};

template<typename T>
struct cknp3d<T, Compadre::enable_if_t<(T::rank!=3)> > {
    nb::ndarray<nb::numpy, typename T::value_type> result;
    cknp3d (T kokkos_array_host) {
        std::vector<size_t> dim_out = {0, 0, 0};
        result = create_new_array<typename T::value_type>(dim_out);
    }
    nb::ndarray<nb::numpy, typename T::value_type> convert() { return result; }
};


template<typename T>
nb::ndarray<nb::numpy, typename T::value_type> convert_kokkos_to_np(T kokkos_array_device) {

    // ensure data is accessible
    auto kokkos_array_host =
        Kokkos::create_mirror_view(kokkos_array_device);
    Kokkos::deep_copy(kokkos_array_host, kokkos_array_device);

    nb::ndarray<nb::numpy, typename T::value_type> result;
    if (T::rank==1) {
        result = cknp1d<decltype(kokkos_array_host)>(kokkos_array_host).convert();
    } else if (T::rank==2) {
        result = cknp2d<decltype(kokkos_array_host)>(kokkos_array_host).convert();
    } else if (T::rank==3) {
        result = cknp3d<decltype(kokkos_array_host)>(kokkos_array_host).convert();
    } else {
        //compadre_assert_release(false && "Only (1 <= rank <= 3) supported.");
        std::vector<size_t> dim_out = {0};
        result = create_new_array<typename T::value_type>(dim_out);
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

    void setNeighbors(nb::ndarray<nb::numpy, local_index_type> input) {

        if (input.ndim() != 2) {
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

    nb::ndarray<nb::numpy, double> getSourceSites() {
        compadre_assert_release(gmls_object!=nullptr &&
                "ParticleHelper used without an internal GMLS object set");
        compadre_assert_release((gmls_object->getPointConnections()->_source_coordinates.extent(0)>0) && 
                "getSourceSites() called, but source sites were never set.");
        return convert_kokkos_to_np(gmls_object->getPointConnections()->_source_coordinates);
    }

    void setTargetSites(nb::ndarray<nb::numpy, double> input) {

        if (input.ndim() != 2) {
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

    nb::ndarray<nb::numpy, double> getTargetSites() {
        compadre_assert_release(gmls_object!=nullptr &&
                "ParticleHelper used without an internal GMLS object set");
        compadre_assert_release((gmls_object->getPointConnections()->_target_coordinates.extent(0)>0) && 
                "getTargetSites() called, but target sites were never set.");
        return convert_kokkos_to_np(gmls_object->getPointConnections()->_target_coordinates);
    }

    void setWindowSizes(nb::ndarray<nb::numpy, double> input) {

        if (input.ndim() != 1) {
            throw std::runtime_error("Number of dimensions must be one");
        }
        
        auto epsilon = convert_np_to_kokkos_1d<device_memory_space>(input);
        
        compadre_assert_release(gmls_object!=nullptr &&
                "ParticleHelper used without an internal GMLS object set");
        // set values from Kokkos View
        gmls_object->setWindowSizes(epsilon);
    }
 
    nb::ndarray<nb::numpy, double> getWindowSizes() {
        compadre_assert_release(gmls_object!=nullptr &&
                "ParticleHelper used without an internal GMLS object set");
        if (gmls_object->getWindowSizes()->extent(0)<=0) {
            throw std::runtime_error("getWindowSizes() called, but window sizes were never set.");
        }
        return convert_kokkos_to_np(*gmls_object->getWindowSizes());
    }

    void setTangentBundle(nb::ndarray<nb::numpy, double> input) {

        if (input.ndim() != 3) {
            throw std::runtime_error("Number of dimensions must be three");
        }

        compadre_assert_release(gmls_object!=nullptr &&
                "ParticleHelper used without an internal GMLS object set");
        if (gmls_object->getGlobalDimensions()!=input.shape(1)) {
            throw std::runtime_error("Second dimension must be the same as GMLS spatial dimension");
        }

        if (gmls_object->getGlobalDimensions()!=input.shape(2)) {
            throw std::runtime_error("Third dimension must be the same as GMLS spatial dimension");
        }
        
        auto device_tangent_bundle = convert_np_to_kokkos_3d<device_memory_space>(input);
        
        // set values from Kokkos View
        gmls_object->setTangentBundle(device_tangent_bundle);
    }

    nb::ndarray<nb::numpy, double> getTangentBundle() {
        compadre_assert_release(gmls_object!=nullptr &&
                "ParticleHelper used without an internal GMLS object set");
        compadre_assert_release((gmls_object->getTangentDirections()->extent(0)>0) && 
                "getTangentBundle() called, but tangent bundle was never set.");
        return convert_kokkos_to_np(*gmls_object->getTangentDirections());
    }

    void setReferenceOutwardNormalDirection(nb::ndarray<nb::numpy, double> input, bool use_to_orient_surface = true) {

        if (input.ndim() != 2) {
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

    nb::ndarray<nb::numpy, double> getReferenceOutwardNormalDirection() {
        compadre_assert_release(gmls_object!=nullptr &&
                "ParticleHelper used without an internal GMLS object set");
        compadre_assert_release((gmls_object->getReferenceNormalDirections()->extent(0)>0) && 
                "getReferenceNormalDirections() called, but normal directions were never set.");
        return convert_kokkos_to_np(*gmls_object->getReferenceNormalDirections());
    }

    void generateKDTree() {
        compadre_assert_release(gmls_object!=nullptr &&
                "ParticleHelper used without an internal GMLS object set");
        //auto d_val = gmls_object->getPointConnections()->_source_coordinates;
        ////compadre_assert_release((gmls_object->getPointConnections()->_source_coordinates.extent(0)>0) &&
        ////"getSourceSites() called, but source sites were never set.");
        //std::cout << "T" << d_val.extent(0) << std::endl;
        //auto h_val = 
        //    Kokkos::create_mirror_view(d_val);
        //Kokkos::deep_copy(h_val, d_val);
        //if (d_val.extent(0)>0) {
        //    std::cout << "N" << h_val(0,0) << std::endl;
        //    point_cloud_search = std::shared_ptr<Compadre::PointCloudSearch<double_2d_view_type> >(
        //            new Compadre::PointCloudSearch<double_2d_view_type>(
        //                h_val,
        //                gmls_object->getGlobalDimensions()
        //            )
        //    );
        //}
        //auto h_val = gmls_object->getPointConnections()->_source_coordinates;
        point_cloud_search = std::shared_ptr<Compadre::PointCloudSearch<double_2d_view_type> >(
                new Compadre::PointCloudSearch<double_2d_view_type>(
                    convert_np_to_kokkos_2d<host_memory_space>(
                        convert_kokkos_to_np(gmls_object->getPointConnections()->_source_coordinates)), 
                    gmls_object->getGlobalDimensions()
                )
        );
    }

    void generateKDTree(nb::ndarray<nb::numpy, double> input) {

        if (input.ndim() != 2) {
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

    void generateNeighborListsFromKNNSearchAndSet(nb::ndarray<nb::numpy, double> input, int poly_order, int dimension = 3, double epsilon_multiplier = 1.6, double max_search_radius = 0.0, bool scale_k_neighbor_radius = true, bool scale_num_neighbors = false) {

        compadre_assert_release(gmls_object!=nullptr &&
                "ParticleHelper used without an internal GMLS object set");
        int neighbors_needed = Compadre::GMLS::getNP(poly_order, dimension, gmls_object->getReconstructionSpace());

        if (input.ndim() != 2) {
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
    
    void setAdditionalEvaluationSitesData(nb::ndarray<nb::numpy, local_index_type> input_indices, nb::ndarray<nb::numpy, double> input_coordinates) {

        if (input_indices.ndim() != 2) {
            throw std::runtime_error("Number of dimensions of input indices must be two");
        }
        if (input_coordinates.ndim() != 2) {
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

    nb::ndarray<nb::numpy, double> getPolynomialCoefficients(nb::ndarray<nb::numpy, double> input) {

        size_t rows = input.shape(0);
        size_t cols = (input.ndim() > 1) ? input.shape(1) : 1;
        size_t stride_0 = input.stride(0);
        size_t stride_1 = (input.ndim() > 1) ? input.stride(1) : 1;
        if ((input.ndim() == 1 && stride_0!=1) || (input.ndim() == 2 && (stride_0!=cols || stride_1!=1))) {
        //if (stride_0!=rows || stride_1!=cols) {
            throw std::runtime_error("getPolynomialCoefficients was passed an input array with noncontiguous data. Please reformat input before passing to getPolynomialCoefficients.");
        }

        // create Kokkos View on host to copy into
        Kokkos::View<double**, Kokkos::HostSpace> source_data("source data", input.shape(0), (input.ndim()>1) ? input.shape(1) : 1); 

        if (input.ndim()==1) {
            // overwrite existing data assuming a 2D layout
            double* data_ptr = (double*)input.data();
            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, input.shape(0)), [=](int i) {
                source_data(i, 0) = data_ptr[i];
            });
        } else if (input.ndim()>1) {
            double* data_ptr = (double*)input.data();
            // overwrite existing data assuming a 2D layout
            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, input.shape(0)), [=](int i) {
                for (int j = 0; j < input.shape(1); ++j)
                {
                    //source_data(i, j) = input(i, j);
                    source_data(i, j) = data_ptr[i*input.shape(1) + j];
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

        std::vector<size_t> dim_out = {dim_out_0, dim_out_1};
        auto result = create_new_array<double>(dim_out);
        double* data_ptr = (double*)result.data();
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, dim_out_0), [&](int i) {
            for (int j = 0; j < dim_out_1; ++j)
            {
                //result(i, j) = polynomial_coefficients(i, j);
                data_ptr[i*dim_out_1 + j] = polynomial_coefficients(i, j);
            }
        });
        Kokkos::fence();

        return result;
    }
 
    nb::ndarray<nb::numpy, double> applyStencil(const nb::ndarray<nb::numpy, double> input, const TargetOperation lro, const SamplingFunctional sro, const int evaluation_site_local_index = 0) const {
 
        size_t rows = input.shape(0);
        size_t cols = (input.ndim() > 1) ? input.shape(1) : 1;
        size_t stride_0 = input.stride(0);
        size_t stride_1 = (input.ndim() > 1) ? input.stride(1) : 1;
        if ((input.ndim() == 1 && stride_0!=1) || (input.ndim() == 2 && (stride_0!=cols || stride_1!=1))) {
            throw std::runtime_error("applyStencil was passed an input array with noncontiguous data. Please reformat input before passing to applyStencil.");
        }

        //if (input.ndim() != 2) {
        //    throw std::runtime_error("Number of dimensions of input indices must be two");
        //}

        // cast numpy data as Kokkos View
        host_scratch_matrix_right_type source_data((double *) input.data(), input.shape(0), (input.ndim()>1) ? input.shape(1) : 1);

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

        nb::ndarray<nb::numpy, double> result;

        if (dim_out_1>=2) {
            std::vector<size_t> dim_out = {dim_out_0, dim_out_1};
            result = create_new_array<double>(dim_out);
            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,dim_out_0), [&](int i) {
                for (size_t j=0; j<dim_out_1; ++j) {
                    //result(i,j) = output_values(i,j);
                    *(double*)((double*)result.data() + i*dim_out_1 + j) = output_values(i,j);
                }
            });
            Kokkos::fence();
        }
        else if (dim_out_1==1) {
            std::vector<size_t> dim_out = {dim_out_0};
            result = create_new_array<double>(dim_out);
            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,dim_out_0), [&](int i) {
                //result(i) = output_values(i,0);
                *(double*)((double*)result.data() + i) = output_values(i,0);
            });
            Kokkos::fence();
        }

        return result;
    }
    
    nb::ndarray<nb::numpy, double> applyStencilAllTargetsAllAdditionalEvaluationSites(const nb::ndarray<nb::numpy, double> input, const TargetOperation lro, const SamplingFunctional sro) const {

        size_t rows = input.shape(0);
        size_t cols = (input.ndim() > 1) ? input.shape(1) : 1;
        size_t stride_0 = input.stride(0);
        size_t stride_1 = (input.ndim() > 1) ? input.stride(1) : 1;
        //if (stride_0!=rows || stride_1!=cols) {
        if ((input.ndim() == 1 && stride_0!=1) || (input.ndim() == 2 && (stride_0!=cols || stride_1!=1))) {
            throw std::runtime_error("applyStencilAllTargetsAllAdditionalEvaluationSites was passed an input array with noncontiguous data. Please reformat input before passing to applyStencil.");
        }
 
        // cast numpy data as Kokkos View
        host_scratch_matrix_left_type source_data((double *) input.data(), input.shape(0), (input.ndim()>1) ? input.shape(1) : 1);

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
        size_t max_additional_sites = gmls_object->getAdditionalEvaluationIndices()->getMaxNumNeighbors() + 1;
        gmls_object->getAdditionalEvaluationIndices()->computeMinNumNeighbors();

        // set dim_out_0 to 1 if setAdditionalEvaluationSitesData never called
        auto additional_evaluation_coords_size = gmls_object->getAdditionalPointConnections()->_source_coordinates.size();
        size_t dim_out_0 = (additional_evaluation_coords_size==0) ? 1 : max_additional_sites;
        auto dim_out_1 = output_values.extent(0);
        auto dim_out_2 = output_values.extent(1);

        nb::ndarray<nb::numpy, double> result;

        if (dim_out_2>=2) {
            std::vector<size_t> dim_out = {dim_out_0, dim_out_1, dim_out_2};
            result = create_new_array<double>(dim_out);
            for (size_t k=0; k<max_additional_sites; ++k) {
                if (k>0) {
                    output_values = gmls_evaluator.applyAlphasToDataAllComponentsAllTargetSites<double**, Kokkos::HostSpace>
                        (source_data, lro, sro, true /*scalar_as_vector_if_needed*/, k);
                }
                Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,dim_out_1), [&](int i) {
                    for (size_t j=0; j<dim_out_2; ++j) {
                            //result(k,i,j) = output_values(i,j);
                            *(double*)((double*)result.data() + k*dim_out_1*dim_out_2 + i*dim_out_2 + j) = output_values(i,j);
                        }
                });
                Kokkos::fence();
            }
        }
        else if (dim_out_2==1) {
            std::vector<size_t> dim_out = {dim_out_0, dim_out_1};
            result = create_new_array<double>(dim_out);
            for (size_t k=0; k<max_additional_sites; ++k) {
                if (k>0) {
                    output_values = gmls_evaluator.applyAlphasToDataAllComponentsAllTargetSites<double**, Kokkos::HostSpace>
                        (source_data, lro, sro, true /*scalar_as_vector_if_needed*/, k);
                }
                Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,dim_out_1), [&](int i) {
                    //result(k,i) = output_values(i,0);
                    *(double*)((double*)result.data() + k*dim_out_1 + i) = output_values(i,0);
                });
                Kokkos::fence();
            }
        }

        return result;
    }

    double applyStencilSingleTarget(const nb::ndarray<nb::numpy, double> input, const TargetOperation lro, const SamplingFunctional sro, const int evaluation_site_local_index = 0) const {

        size_t rows = input.shape(0);
        size_t cols = (input.ndim() > 1) ? input.shape(1) : 1;
        size_t stride_0 = input.stride(0);
        size_t stride_1 = (input.ndim() > 1) ? input.stride(1) : 1;
        if ((input.ndim() == 1 && stride_0!=1) || (input.ndim() == 2 && (stride_0!=cols || stride_1!=1))) {
        //if (stride_0!=rows || stride_1!=cols) {
            throw std::runtime_error("applyStencilSingleTarget was passed an input array with noncontiguous data. Please reformat input before passing to applyStencil.");
        }
 
        // cast numpy data as Kokkos View
        host_scratch_matrix_left_type source_data((double *) input.data(), input.shape(0), (input.ndim()>1) ? input.shape(1) : 1);

        compadre_assert_release(input.ndim()==1 && "Input given with dimensions > 1");

        compadre_assert_release(gmls_object!=nullptr &&
                "ParticleHelper used without an internal GMLS object set");
        Compadre::Evaluator gmls_evaluator(gmls_object);

        return gmls_evaluator.applyAlphasToDataSingleComponentSingleTargetSite(source_data, 0, lro, 0, evaluation_site_local_index, 0, 0, 0, 0, false);
    }
};

//GMLS factory_function(const ReconstructionSpace rs,
//                      const SamplingFunctional psf,
//                      const SamplingFunctional dsf,
//                      const int po,
//                      const int gdim,
//                      const DenseSolverType dst,
//                      const ProblemType pt,
//                      const ConstraintType ct,
//                      const int cpo) {
//
//    // reinstantiate with details
//    // no need to keep solutions
//    GMLS gmls(rs, psf, dsf, po, gdim, dst, pt, ct, cpo);
//
//    return gmls;
//}

void set_gmls_state(GMLS& gmls, nb::tuple t) {

    if (t.size() != 15)
        throw std::runtime_error("Invalid state!");
    
    
    gmls.setWeightingType(nb::cast<WeightingFunctionType>(t[0]));
    gmls.setWeightingParameter(nb::cast<int>(t[1]), 0);
    gmls.setWeightingParameter(nb::cast<int>(t[2]), 1);
    
    gmls.setCurvatureWeightingType(nb::cast<WeightingFunctionType>(t[3]));
    gmls.setCurvatureWeightingParameter(nb::cast<int>(t[4]), 0);
    gmls.setCurvatureWeightingParameter(nb::cast<int>(t[5]), 1);
    
    auto lro_list = nb::cast<nb::list>(t[6]);
    for (nb::handle obj_i : lro_list) {
        gmls.addTargets(nb::cast<TargetOperation>(obj_i));
    }
    
    // need to convert from numpy back to kokkos before setting problem data
    auto ss      = nb::cast<nb::ndarray<nb::numpy, double> >(t[7]);
    auto k_ss    = convert_np_to_kokkos_2d(ss);
    auto ts      = nb::cast<nb::ndarray<nb::numpy, double> >(t[8]);
    auto k_ts    = convert_np_to_kokkos_2d(ts);
    auto ws      = nb::cast<nb::ndarray<nb::numpy, double> >(t[9]);
    auto k_ws    = convert_np_to_kokkos_1d(ws);
    auto cr      = nb::cast<nb::ndarray<nb::numpy, int> >(t[10]);
    auto k_cr    = convert_np_to_kokkos_1d(cr);
    auto nnl     = nb::cast<nb::ndarray<nb::numpy, int> >(t[11]);
    auto k_nnl   = convert_np_to_kokkos_1d(nnl);
    auto a_cr    = nb::cast<nb::ndarray<nb::numpy, int> >(t[12]);
    auto k_a_cr  = convert_np_to_kokkos_1d(a_cr);
    auto a_nnl   = nb::cast<nb::ndarray<nb::numpy, int> >(t[13]);
    auto k_a_nnl = convert_np_to_kokkos_1d(a_nnl);
    auto a_es    = nb::cast<nb::ndarray<nb::numpy, double> >(t[14]);
    auto k_a_es  = convert_np_to_kokkos_2d(a_es);
    
    gmls.setAdditionalEvaluationSitesData(k_a_cr, k_a_nnl, k_a_es);
    gmls.setProblemData(k_cr, k_nnl, k_ss, k_ts, k_ws);
    
}

//nb::tuple get_gmls_state(GMLS& gmls) {
//    // values needed for base constructor
//    auto rs   = gmls.getReconstructionSpace();
//    auto psf  = gmls.getPolynomialSamplingFunctional();
//    auto dsf  = gmls.getDataSamplingFunctional();
//    auto po   = gmls.getPolynomialOrder();
//    auto gdim = gmls.getGlobalDimensions();
//    auto dst  = gmls.getDenseSolverType();
//    auto pt   = gmls.getProblemType();
//    auto ct   = gmls.getConstraintType();
//    auto cpo  = gmls.getCurvaturePolynomialOrder();
//
//    // values needed to get weighting types the same
//    auto wt   = gmls.getWeightingType();
//    auto wp0  = gmls.getWeightingParameter(0);
//    auto wp1  = gmls.getWeightingParameter(1);
//
//    auto mwt  = gmls.getManifoldWeightingType();
//    auto mwp0 = gmls.getManifoldWeightingParameter(0);
//    auto mwp1 = gmls.getManifoldWeightingParameter(1);
//
//    // get all previously added targets
//    auto d_lro  = const_cast<GMLS&>(gmls).getSolutionSetDevice(false)->_lro;
//    auto lro    = Kokkos::create_mirror_view(d_lro);
//    Kokkos::deep_copy(lro, d_lro);
//    auto lro_list = nb::list();
//    for (int i=0; i<lro.extent_int(0); ++i) {
//        lro_list.append(lro[i]);
//    }
//
//    auto nonconst_gmls = *const_cast<Compadre::GMLS*>(&gmls);
//
//    // get all source sites
//    auto ss     = nonconst_gmls.getPointConnections()->_source_coordinates;
//    auto np_ss  = convert_kokkos_to_np(ss);
//
//    // get all target sites
//    auto ts     = nonconst_gmls.getPointConnections()->_target_coordinates;
//    auto np_ts  = convert_kokkos_to_np(ts);
//
//    // get all window sizes
//    auto ws     = nonconst_gmls.getWindowSizes();
//    auto np_ws  = convert_kokkos_to_np(*ws);
//
//    // get all neighbors
//    auto nl     = nonconst_gmls.getNeighborLists();
//    auto cr     = nl->getNeighborLists();
//    auto np_cr  = convert_kokkos_to_np(cr);
//    auto nnl    = nl->getNumberOfNeighborsList();
//    auto np_nnl = convert_kokkos_to_np(nnl);
//
//    // get all additional evaluation indices
//    auto a_nl     = nonconst_gmls.getAdditionalEvaluationIndices();
//    auto a_cr     = a_nl->getNeighborLists();
//    auto np_a_cr  = convert_kokkos_to_np(a_cr);
//    auto a_nnl    = a_nl->getNumberOfNeighborsList();
//    auto np_a_nnl = convert_kokkos_to_np(a_nnl);
//
//    // get all additional evaluation coordinates
//    auto a_es = nonconst_gmls.getAdditionalPointConnections()->_source_coordinates;
//    auto np_a_es  = convert_kokkos_to_np(a_es);
// 
//    return std::make_tuple(rs, psf, dsf, po, gdim, dst, pt, ct, cpo, 
//                           wt, wp0, wp1, mwt, mwp0, mwp1, lro_list,
//                           np_ss, np_ts, np_ws, np_cr, np_nnl,
//                           np_a_cr, np_a_nnl, np_a_es);
//}

//std::tuple<nb::object, nb::tuple, nb::tuple> gmls_reduce(const GMLS& gmls) {
//
//    nb::object factory = nb::cpp_function(&factory_function);
//
//    // values needed for base constructor
//    auto rs   = gmls.getReconstructionSpace();
//    auto psf  = gmls.getPolynomialSamplingFunctional();
//    auto dsf  = gmls.getDataSamplingFunctional();
//    auto po   = gmls.getPolynomialOrder();
//    auto gdim = gmls.getGlobalDimensions();
//    auto dst  = gmls.getDenseSolverType();
//    auto pt   = gmls.getProblemType();
//    auto ct   = gmls.getConstraintType();
//    auto cpo  = gmls.getCurvaturePolynomialOrder();
//
//    nb::tuple base_args = nb::make_tuple(rs, psf, dsf, po, gdim, dst, pt, ct, cpo);
//
//    // values needed to get weighting types the same
//    auto wt   = gmls.getWeightingType();
//    auto wp0  = gmls.getWeightingParameter(0);
//    auto wp1  = gmls.getWeightingParameter(1);
//
//    auto mwt  = gmls.getManifoldWeightingType();
//    auto mwp0 = gmls.getManifoldWeightingParameter(0);
//    auto mwp1 = gmls.getManifoldWeightingParameter(1);
//
//    // get all previously added targets
//    auto d_lro  = const_cast<GMLS&>(gmls).getSolutionSetDevice(false)->_lro;
//    auto lro    = Kokkos::create_mirror_view(d_lro);
//    Kokkos::deep_copy(lro, d_lro);
//    auto lro_list = nb::list();
//    for (int i=0; i<lro.extent_int(0); ++i) {
//        lro_list.append(lro[i]);
//    }
//
//    auto nonconst_gmls = *const_cast<Compadre::GMLS*>(&gmls);
//
//    // get all source sites
//    auto ss     = nonconst_gmls.getPointConnections()->_source_coordinates;
//    auto np_ss  = convert_kokkos_to_np(ss);
//
//    // get all target sites
//    auto ts     = nonconst_gmls.getPointConnections()->_target_coordinates;
//    auto np_ts  = convert_kokkos_to_np(ts);
//
//    // get all window sizes
//    auto ws     = nonconst_gmls.getWindowSizes();
//    auto np_ws  = convert_kokkos_to_np(*ws);
//
//    // get all neighbors
//    auto nl     = nonconst_gmls.getNeighborLists();
//    auto cr     = nl->getNeighborLists();
//    auto np_cr  = convert_kokkos_to_np(cr);
//    auto nnl    = nl->getNumberOfNeighborsList();
//    auto np_nnl = convert_kokkos_to_np(nnl);
//
//    // get all additional evaluation indices
//    auto a_nl     = nonconst_gmls.getAdditionalEvaluationIndices();
//    auto a_cr     = a_nl->getNeighborLists();
//    auto np_a_cr  = convert_kokkos_to_np(a_cr);
//    auto a_nnl    = a_nl->getNumberOfNeighborsList();
//    auto np_a_nnl = convert_kokkos_to_np(a_nnl);
//
//    // get all additional evaluation coordinates
//    auto a_es = nonconst_gmls.getAdditionalPointConnections()->_source_coordinates;
//    auto np_a_es  = convert_kokkos_to_np(a_es);
// 
//    // get all relevant details from GMLS class
//    auto extra_args = nb::make_tuple(wt, wp0, wp1, mwt, mwp0, mwp1, lro_list,
//                                     np_ss, np_ts, np_ws, np_cr, np_nnl,
//                                     np_a_cr, np_a_nnl, np_a_es);
//    //return std::make_tuple(rs, psf, dsf, po, gdim, dst, pt, ct, cpo, 
//    //                       wt, wp0, wp1, mwt, mwp0, mwp1, lro_list,
//    //                       np_ss, np_ts, np_ws, np_cr, np_nnl,
//    //                       np_a_cr, np_a_nnl, np_a_es);
//
//    return std::make_tuple(factory, base_args, extra_args);
//
//}


NB_MODULE(_pycompadre, m) {
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

    nb::class_<SamplingFunctional>(m, "SamplingFunctional")
    .def("__setstate__", [](nb::handle self, nb::tuple t) { // __setstate__
            if (t.size() != 6)
                throw std::runtime_error("Invalid state!");
            SamplingFunctional *sf = nb::inst_ptr<SamplingFunctional>(self);
            new (sf) SamplingFunctional(nb::cast<int>(t[0]), nb::cast<int>(t[1]), nb::cast<bool>(t[2]),
                                        nb::cast<bool>(t[3]), nb::cast<int>(t[4]), nb::cast<int>(t[5]));
            nb::inst_mark_ready(self);
        })
    .def("__getstate__", [](const SamplingFunctional &sf) { // __getstate__
            return nb::make_tuple(sf.input_rank, sf.output_rank, sf.use_target_site_weights,
                                  sf.nontrivial_nullspace, sf.transform_type, sf.id);
        });
    nb::dict sampling_functional;
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

    nb::enum_<TargetOperation>(m, "TargetOperation")
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


    nb::enum_<ReconstructionSpace>(m, "ReconstructionSpace")
    .value("ScalarTaylorPolynomial", ReconstructionSpace::ScalarTaylorPolynomial)
    .value("VectorTaylorPolynomial", ReconstructionSpace::VectorTaylorPolynomial)
    .value("VectorOfScalarClonesTaylorPolynomial", ReconstructionSpace::VectorOfScalarClonesTaylorPolynomial)
    .value("DivergenceFreeVectorTaylorPolynomial", ReconstructionSpace::DivergenceFreeVectorTaylorPolynomial)
    .value("BernsteinPolynomial", ReconstructionSpace::BernsteinPolynomial)
    .export_values();

    nb::enum_<WeightingFunctionType>(m, "WeightingFunctionType")
    .value("Power", WeightingFunctionType::Power)
    .value("Gaussian", WeightingFunctionType::Gaussian)
    .value("CubicSpline", WeightingFunctionType::CubicSpline)
    .value("CardinalCubicBSpline", WeightingFunctionType::CardinalCubicBSpline)
    .value("Cosine", WeightingFunctionType::Cosine)
    .value("Sigmoid", WeightingFunctionType::Sigmoid)
    .export_values();

    nb::enum_<DenseSolverType>(m, "DenseSolverType")
    .value("QR", DenseSolverType::QR)
    .value("LU", DenseSolverType::LU)
    .export_values();

    nb::enum_<ProblemType>(m, "ProblemType")
    .value("STANDARD", ProblemType::STANDARD)
    .value("MANIFOLD", ProblemType::MANIFOLD)
    .export_values();

    nb::enum_<ConstraintType>(m, "ConstraintType")
    .value("NO_CONSTRAINT", ConstraintType::NO_CONSTRAINT)
    .value("NEUMANN_GRAD_SCALAR", ConstraintType::NEUMANN_GRAD_SCALAR)
    .export_values();

    nb::enum_<QuadratureType>(m, "QuadratureType")
    .value("INVALID", QuadratureType::INVALID)
    .value("LINE", QuadratureType::LINE)
    .value("TRI", QuadratureType::TRI)
    .value("QUAD", QuadratureType::QUAD)
    .value("TET", QuadratureType::TET)
    .value("HEX", QuadratureType::HEX)
    .export_values();
    
    nb::class_<SolutionSet<host_memory_space> >(m, "SolutionSet", R"pbdoc(
        Class containing solution data from GMLS problems
    )pbdoc")
    .def("getAlpha", &SolutionSet<host_memory_space>::getAlpha<>, nb::arg("lro"), nb::arg("target_index"), nb::arg("output_component_axis_1"), nb::arg("output_component_axis_2"), nb::arg("neighbor_index"), nb::arg("input_component_axis_1"), nb::arg("input_component_axis_2"), nb::arg("additional_evaluation_site")=0);

    // helper functions
    nb::class_<ParticleHelper>(m, "ParticleHelper", R"pbdoc(
        Class to manage calling PointCloudSearch, moving data to/from Numpy arrays in Kokkos::Views,
        and applying GMLS solutions to multidimensional data arrays 
    )pbdoc")
    .def(nb::init<>(), R"pbdoc(
        Requires follow-on call to setGMLSObject(gmls_obj) before use
    )pbdoc")
    .def(nb::init<GMLS*>(), nb::arg("gmls_instance"))
    .def("generateKDTree", static_cast<void (ParticleHelper::*)(nb::ndarray<nb::numpy, double>)>(&ParticleHelper::generateKDTree))
    .def("generateNeighborListsFromKNNSearchAndSet", &ParticleHelper::generateNeighborListsFromKNNSearchAndSet, 
            nb::arg("target_sites"), nb::arg("poly_order"), nb::arg("dimension") = 3, nb::arg("epsilon_multiplier") = 1.6, 
            nb::arg("max_search_radius") = 0.0, nb::arg("scale_k_neighbor_radius") = true, 
            nb::arg("scale_num_neighbors") = false)
    .def("setNeighbors", &ParticleHelper::setNeighbors, nb::arg("neighbor_lists"), 
            "Sets neighbor lists from 2D array where first column is number of neighbors for corresponding row's target site.")
    .def("setGMLSObject", &ParticleHelper::setGMLSObject, nb::arg("gmls_object"))
    .def("getGMLSObject", &ParticleHelper::getGMLSObject, nb::rv_policy::reference_internal)
    .def("getAdditionalEvaluationIndices", &ParticleHelper::getAdditionalEvaluationIndices, nb::rv_policy::reference_internal)
    .def("getNeighborLists", &ParticleHelper::getNeighborLists, nb::rv_policy::reference_internal)
    .def("setWindowSizes", &ParticleHelper::setWindowSizes, nb::arg("window_sizes"))
    .def("getWindowSizes", &ParticleHelper::getWindowSizes)
    .def("getSourceSites", &ParticleHelper::getSourceSites)
    .def("setTargetSites", &ParticleHelper::setTargetSites, nb::arg("target_coordinates"))
    .def("getTargetSites", &ParticleHelper::getTargetSites)
    .def("setTangentBundle", &ParticleHelper::setTangentBundle, nb::arg("tangent_bundle"))
    .def("getTangentBundle", &ParticleHelper::getTangentBundle)
    .def("setReferenceOutwardNormalDirection", &ParticleHelper::setReferenceOutwardNormalDirection, nb::arg("reference_normal_directions"), nb::arg("use_to_orient_surface") = true)
    .def("getReferenceOutwardNormalDirection", &ParticleHelper::getReferenceOutwardNormalDirection)
    .def("setAdditionalEvaluationSitesData", &ParticleHelper::setAdditionalEvaluationSitesData, nb::arg("additional_evaluation_sites_neighbor_lists"), nb::arg("additional_evaluation_sites_coordinates"))
    .def("getPolynomialCoefficients", &ParticleHelper::getPolynomialCoefficients, nb::arg("input_data"))
    .def("applyStencilSingleTarget", &ParticleHelper::applyStencilSingleTarget, nb::arg("input_data"), nb::arg("target_operation")=TargetOperation::ScalarPointEvaluation, nb::arg("sampling_functional")=PointSample, nb::arg("evaluation_site_local_index")=0)
    .def("applyStencilAllTargetsAllAdditionalEvaluationSites", &ParticleHelper::applyStencilAllTargetsAllAdditionalEvaluationSites, nb::arg("input_data"), nb::arg("target_operation")=TargetOperation::ScalarPointEvaluation, nb::arg("sampling_functional")=PointSample)
    .def("applyStencil", &ParticleHelper::applyStencil, nb::arg("input_data"), nb::arg("target_operation")=TargetOperation::ScalarPointEvaluation, nb::arg("sampling_functional")=PointSample, nb::arg("evaluation_site_local_index")=0)
    .def("__setstate__", [](nb::handle self, nb::tuple t) { // __setstate__
            if (t.size() != 0)
                throw std::runtime_error("Invalid state!");
            ParticleHelper *ph = nb::inst_ptr<ParticleHelper>(self);
            new (ph) ParticleHelper();

            nb::inst_mark_ready(self);
        })
    .def("__getstate__", [](const ParticleHelper &ph) { // __getstate__
            return nb::make_tuple();
        });

    nb::class_<GMLS>(m, "GMLS")
    .def(nb::init<int,int,std::string,std::string,std::string,int>(),
            nb::arg("poly_order"),nb::arg("dimension")=3,nb::arg("dense_solver_type")="QR", 
            nb::arg("problem_type")="STANDARD", nb::arg("constraint_type")="NO_CONSTRAINT", 
            nb::arg("curvature_poly_order")=2)
    .def(nb::init<ReconstructionSpace,SamplingFunctional,int,int,std::string,std::string,std::string,int>(),
            nb::arg("reconstruction_space"),nb::arg("sampling_functional"),
            nb::arg("poly_order"),nb::arg("dimension")=3,nb::arg("dense_solver_type")="QR", 
            nb::arg("problem_type")="STANDARD", nb::arg("constraint_type")="NO_CONSTRAINT", 
            nb::arg("curvature_poly_order")=2)
    .def(nb::init<ReconstructionSpace,SamplingFunctional,SamplingFunctional,int,int,std::string,std::string,std::string,int>(),
            nb::arg("reconstruction_space"),nb::arg("polynomial_sampling_functional"),nb::arg("data_sampling_functional"),
            nb::arg("poly_order"),nb::arg("dimension")=3,nb::arg("dense_solver_type")="QR", 
            nb::arg("problem_type")="STANDARD", nb::arg("constraint_type")="NO_CONSTRAINT", 
            nb::arg("curvature_poly_order")=2)
    .def("getWeightingParameter", &GMLS::getWeightingParameter, nb::arg("parameter index")=0, "Get weighting kernel parameter[index].")
    .def("setWeightingParameter", &GMLS::setWeightingParameter, nb::arg("parameter value"), nb::arg("parameter index")=0, "Set weighting kernel parameter[index] to parameter value.")
    .def("setWeightingPower", &GMLS::setWeightingParameter, nb::arg("parameter value"), nb::arg("parameter index")=0, "Set weighting kernel parameter[index] to parameter value. [DEPRECATED]")
    .def("getWeightingType", &GMLS::getWeightingType, "Get the weighting type.")
    .def("setWeightingType", static_cast<void (GMLS::*)(const std::string&)>(&GMLS::setWeightingType), "Set the weighting type with a string.")
    .def("setWeightingType", static_cast<void (GMLS::*)(WeightingFunctionType)>(&GMLS::setWeightingType), "Set the weighting type with a WeightingFunctionType.")
    .def("setOrderOfQuadraturePoints", &GMLS::setOrderOfQuadraturePoints, nb::arg("order"))
    .def("setDimensionOfQuadraturePoints", &GMLS::setDimensionOfQuadraturePoints, nb::arg("dim"))
    .def("setQuadratureType", &GMLS::setQuadratureType, nb::arg("quadrature_type"))
    .def("addTargets", static_cast<void (GMLS::*)(TargetOperation)>(&GMLS::addTargets), "Add a target operation.")
    .def("addTargets", static_cast<void (GMLS::*)(std::vector<TargetOperation>)>(&GMLS::addTargets), "Add a list of target operations.")
    .def("generateAlphas", &GMLS::generateAlphas, nb::arg("number_of_batches")=1, nb::arg("keep_coefficients")=false, nb::arg("clear_cache")=true)
    .def("containsValidAlphas", &GMLS::containsValidAlphas)
    .def("getSolutionSet", &GMLS::getSolutionSetHost, nb::arg("validity_check")=true, nb::rv_policy::reference_internal)
    .def("getNP", &GMLS::getNP, "Get size of basis.")
    .def("getNN", &GMLS::getNN, "Heuristic number of neighbors.")
    .def("setTargetExtraData", [] (GMLS* gmls, nb::ndarray<nb::numpy, double> extra_data) {
        if (extra_data.ndim() != 2) {
            throw std::runtime_error("Number of dimensions must be two");
        }
        gmls->setTargetExtraData(convert_np_to_kokkos_2d(extra_data));
    })
    .def("setSourceExtraData", [] (GMLS* gmls, nb::ndarray<nb::numpy, double> extra_data) {
        if (extra_data.ndim() != 2) {
            throw std::runtime_error("Number of dimensions must be two");
        }
        gmls->setSourceExtraData(convert_np_to_kokkos_2d(extra_data));
    })
    .def("__setstate__", [](nb::handle self, nb::tuple t) { // __setstate__
            if (t.size() != 24)
                throw std::runtime_error("Invalid state!");

            GMLS *gmls = nb::inst_ptr<GMLS>(self);

            new (gmls) GMLS(nb::cast<const ReconstructionSpace>(t[0]),
                            nb::cast<const SamplingFunctional>(t[1]),
                            nb::cast<const SamplingFunctional>(t[2]),
                            nb::cast<const int>(t[3]),
                            nb::cast<const int>(t[4]),
                            nb::cast<const DenseSolverType>(t[5]),
                            nb::cast<const ProblemType>(t[6]),
                            nb::cast<const ConstraintType>(t[7]),
                            nb::cast<const int>(t[8]));

            gmls->setWeightingType(nb::cast<WeightingFunctionType>(t[9]));
            gmls->setWeightingParameter(nb::cast<int>(t[10]), 0);
            gmls->setWeightingParameter(nb::cast<int>(t[11]), 1);
            
            gmls->setCurvatureWeightingType(nb::cast<WeightingFunctionType>(t[12]));
            gmls->setCurvatureWeightingParameter(nb::cast<int>(t[13]), 0);
            gmls->setCurvatureWeightingParameter(nb::cast<int>(t[14]), 1);
            
            auto lro_list = nb::cast<nb::list>(t[15]);
            for (nb::handle obj_i : lro_list) {
                gmls->addTargets(nb::cast<TargetOperation>(obj_i));
            }
            
            // need to convert from numpy back to kokkos before setting problem data
            auto ss      = nb::cast<nb::ndarray<nb::numpy, double> >(t[16]);
            auto k_ss    = convert_np_to_kokkos_2d(ss);
            auto ts      = nb::cast<nb::ndarray<nb::numpy, double> >(t[17]);
            auto k_ts    = convert_np_to_kokkos_2d(ts);
            auto ws      = nb::cast<nb::ndarray<nb::numpy, double> >(t[18]);
            auto k_ws    = convert_np_to_kokkos_1d(ws);
            auto cr      = nb::cast<nb::ndarray<nb::numpy, int> >(t[19]);
            auto k_cr    = convert_np_to_kokkos_1d(cr);
            auto nnl     = nb::cast<nb::ndarray<nb::numpy, int> >(t[20]);
            auto k_nnl   = convert_np_to_kokkos_1d(nnl);
            auto a_cr    = nb::cast<nb::ndarray<nb::numpy, int> >(t[21]);
            auto k_a_cr  = convert_np_to_kokkos_1d(a_cr);
            auto a_nnl   = nb::cast<nb::ndarray<nb::numpy, int> >(t[22]);
            auto k_a_nnl = convert_np_to_kokkos_1d(a_nnl);
            auto a_es    = nb::cast<nb::ndarray<nb::numpy, double> >(t[23]);
            auto k_a_es  = convert_np_to_kokkos_2d(a_es);
            
            gmls->setAdditionalEvaluationSitesData(k_a_cr, k_a_nnl, k_a_es);
            gmls->setProblemData(k_cr, k_nnl, k_ss, k_ts, k_ws);

            nb::inst_mark_ready(self);
        })
    .def("__getstate__", [](const GMLS &gmls) { // __getstate__
            // values needed for base constructor
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
            auto lro_list = nb::list();
            for (int i=0; i<lro.extent_int(0); ++i) {
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
            return std::make_tuple(rs, psf, dsf, po, gdim, dst, pt, ct, cpo, 
                                   wt, wp0, wp1, mwt, mwp0, mwp1, lro_list,
                                   np_ss, np_ts, np_ws, np_cr, np_nnl,
                                   np_a_cr, np_a_nnl, np_a_es);
        });
    //.def("__setstate__", &set_gmls_state)
    ////.def("__getstate__", &get_gmls_state)
    //.def("__reduce__", &gmls_reduce);
    

    nb::class_<NeighborLists<ParticleHelper::int_1d_view_type_in_gmls> >(m, "NeighborLists")
    .def("computeMaxNumNeighbors", &NeighborLists<ParticleHelper::int_1d_view_type_in_gmls>::computeMaxNumNeighbors, "Compute maximum number of neighbors over all neighborhoods.")
    .def("computeMinNumNeighbors", &NeighborLists<ParticleHelper::int_1d_view_type_in_gmls>::computeMinNumNeighbors, "Compute minimum number of neighbors over all neighborhoods.")
    .def("getNumberOfTargets", &NeighborLists<ParticleHelper::int_1d_view_type_in_gmls>::getNumberOfTargets, "Number of targets.")
    .def("getNumberOfNeighbors", &NeighborLists<ParticleHelper::int_1d_view_type_in_gmls>::getNumberOfNeighborsHost, nb::arg("target index"), "Get number of neighbors for target.")
    .def("getMaxNumNeighbors", &NeighborLists<ParticleHelper::int_1d_view_type_in_gmls>::getMaxNumNeighbors, "Get maximum number of neighbors over all neighborhoods.")
    .def("getMinNumNeighbors", &NeighborLists<ParticleHelper::int_1d_view_type_in_gmls>::getMinNumNeighbors, "Get minimum number of neighbors over all neighborhoods.")
    .def("getNeighbor", &NeighborLists<ParticleHelper::int_1d_view_type_in_gmls>::getNeighborHost, nb::arg("target index"), nb::arg("local neighbor number"), "Get neighbor index from target index and local neighbor number.")
    .def("getTotalNeighborsOverAllLists", &NeighborLists<ParticleHelper::int_1d_view_type_in_gmls>::getTotalNeighborsOverAllListsHost, "Get total storage size of all neighbor lists combined.");

    nb::class_<Quadrature>(m, "Quadrature")
    .def(nb::init<int, int, std::string>(), nb::arg("order_of_quadrature_points"), nb::arg("dimension_of_quadrature_points"), 
         nb::arg("quadrature_type") = "LINE")
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

    m_kokkos
    .def("initialize", [](nb::list args) {
        std::vector<char*> char_args;
        for (nb::handle arg : args) {
            char_args.push_back((char*)nb::cast<std::string>(arg).data());
        }
        char_args.push_back(nullptr);
        int argc = (int)args.size();
        Kokkos::initialize(argc, char_args.data());     
    })
    .def("finalize", &Kokkos::finalize)
    .def("is_initialized", &Kokkos::is_initialized)
    .def("status", &KokkosParser::status);

    nb::class_<KokkosParser>(m, "KokkosParser", R"pbdoc(
        Class wrapping the functionality of Kokkos::ScopeGuard. Multiple instances can  
        can be instantiated and go out of scope with only one instance initiating Kokkos
        and the last instance finalizing Kokkos when it goes out of scope.
    )pbdoc")
    .def(nb::init<std::vector<std::string>,bool>(), nb::arg("args"), nb::arg("print") = false)
    .def(nb::init<bool>(), nb::arg("print") = false)
    .def_static("status", &KokkosParser::status);

    m.def("getNP", &GMLS::getNP, R"pbdoc(
        Get size of basis.
    )pbdoc");

    m.def("getNN", &GMLS::getNN, R"pbdoc(
        Heuristic number of neighbors.
    )pbdoc");

    m.def("Wab", &GMLS::Wab, nb::arg("r"), nb::arg("h"), nb::arg("weighting type"), nb::arg("p"), nb::arg("n"), "Evaluate weighting kernel.");

    m.def("is_unique_installation", [] {
        nb::exec(R"(
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
        auto unittest = nb::module_::import_("unittest");
        auto os = nb::module_::import_("os");

        std::string examples_path;
        try {
            nb::object this_pycompadre = nb::module_::import_("pycompadre");
            auto location = nb::cast<std::string>(this_pycompadre.attr("__file__"));
            nb::object path = os.attr("path").attr("dirname")(location);
            examples_path = nb::cast<std::string>(path) + "/examples";
        } catch (...) {
            std::cerr << "Error getting examples path from pycompadre module." << std::endl;
        }

        bool result = false;
        try {
            printf("Running examples in: %s\n", examples_path.c_str());
            nb::object loader = unittest.attr("TestLoader")();
            nb::object tests  = loader.attr("discover")(examples_path);
            nb::object runner = unittest.attr("runner").attr("TextTestRunner")();
            result = nb::cast<bool>(runner.attr("run")(tests).attr("wasSuccessful")());
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


