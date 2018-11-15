#ifndef _COMPADRE_REMAP_HPP_
#define _COMPADRE_REMAP_HPP_

#include "Compadre_Typedefs.hpp"
#include "Compadre_GMLS.hpp"

template< bool B, class T = void >
using enable_if_t = typename std::enable_if<B,T>::type;

namespace Compadre {

//! Creates 1D subviews of data from a 2D view, generally constructed with Create1DSliceOnDeviceView
template<typename T, typename T2, typename T3=void>
struct Subview1D { 
    
    T _data_in;
    T2 _data_original_view;

    Subview1D(T data_in, T2 data_original_view) {
        _data_in = data_in;
        _data_original_view = data_original_view;
    }

    auto get1DView(const int column_num) -> decltype(Kokkos::subview(_data_in, Kokkos::ALL, column_num)) {
        assert((column_num<_data_in.dimension_1()) && "Subview asked for column > second dimension of input data.");
        return Kokkos::subview(_data_in, Kokkos::ALL, column_num);
    }

    T2 copyToAndReturnOriginalView() {
        Kokkos::deep_copy(_data_original_view, _data_in);
        Kokkos::fence();
        return _data_original_view;
    }

};

//! Creates 1D subviews of data from a 1D view, generally constructed with Create1DSliceOnDeviceView
template<typename T, typename T2>
struct Subview1D<T, T2, enable_if_t<(T::rank<2)> >
{ 

    T _data_in;
    T2 _data_original_view;

    Subview1D(T data_in, T2 data_original_view) {
        _data_in = data_in;
        _data_original_view = data_original_view;
    }

    auto get1DView(const int column_num) -> decltype(Kokkos::subview(_data_in, Kokkos::ALL)) {
        assert((column_num==0) && "Subview asked for column column_num!=0, but _data_in is rank 1.");
        return Kokkos::subview(_data_in, Kokkos::ALL);
    }

    T2 copyToAndReturnOriginalView() {
        Kokkos::deep_copy(_data_original_view, _data_in);
        Kokkos::fence();
        return _data_original_view;
    }

};

//! Copies data_in to the device, and then allows for access to 1D columns of data on device.
//! Handles either 2D or 1D views as input, and they can be on the host or the device.
template <typename T>
auto Create1DSliceOnDeviceView(T sampling_input_data_host_or_device) -> Subview1D<decltype(Kokkos::create_mirror_view(
                    Kokkos::DefaultExecutionSpace::memory_space(), sampling_input_data_host_or_device)), T> {

    // makes view on the device (does nothing if already on the device)
    auto sampling_input_data_device = Kokkos::create_mirror_view(
        Kokkos::DefaultExecutionSpace::memory_space(), sampling_input_data_host_or_device);
    Kokkos::deep_copy(sampling_input_data_device, sampling_input_data_host_or_device);
    Kokkos::fence();

    return Subview1D<decltype(sampling_input_data_device),T>(sampling_input_data_device, sampling_input_data_host_or_device);
}




//! \brief Lightweight Remap Helper
//! This class is a lightweight wrapper for extracting information to remap from a GMLS class
//! and applying it to data. 
class Remap {

private:

    GMLS *_gmls;

public:

    Remap(GMLS *gmls) : _gmls(gmls) {
        Kokkos::fence();
    };

    ~Remap() {};

    //! Dot product of alphas with sampling data, FOR A SINGLE target_index,  where sampling data is in a 1D Kokkos View
    //! 
    //! This function is to be used when the alpha values have already been calculated and stored for use 
    //!
    //! Only supports one output component / input component at a time. The user will need to loop over the output 
    //! components in order to fill a vector target or matrix target.
    //! 
    //! Assumptions on input data:
    //! \param sampling_input_data      [in] - 1D Kokkos View (no restriction on memory space)
    //! \param lro                      [in] - Target operation from the TargetOperation enum
    //! \param target_index             [in] - Target # user wants to reconstruct target functional at, corresponds to row number of neighbor_lists
    //! \param output_component_axis_1  [in] - Row for a rank 2 tensor or rank 1 tensor, 0 for a scalar output
    //! \param output_component_axis_2  [in] - Columns for a rank 2 tensor, 0 for rank less than 2 output tensor
    //! \param input_component_axis_1   [in] - Row for a rank 2 tensor or rank 1 tensor, 0 for a scalar input
    //! \param input_component_axis_2   [in] - Columns for a rank 2 tensor, 0 for rank less than 2 input tensor
    template <typename view_type_data>
    double applyAlphasToDataSingleComponentSingleTargetSite(view_type_data sampling_input_data, const int column_of_input, TargetOperation lro, const int target_index, const int output_component_axis_1, const int output_component_axis_2, const int input_component_axis_1, const int input_component_axis_2) const {

        double value = 0;

        const int alpha_column_base_multiplier = _gmls->getAlphaColumnOffset(lro, output_component_axis_1, 
                output_component_axis_2, input_component_axis_1, input_component_axis_2);

        auto sampling_subview_maker = Create1DSliceOnDeviceView(sampling_input_data);

        
        // gather needed information for remap
        auto neighbor_lists = _gmls->getNeighborLists();
        auto alphas         = _gmls->getAlphas();
        auto neighbor_lists_lengths = _gmls->getNeighborListsLengths();
        auto sampling_data_device = sampling_subview_maker.get1DView(column_of_input);
        
        // loop through neighbor list for this target_index
        // grabbing data from that entry of data
        Kokkos::parallel_reduce("applyAlphasToData::Device", 
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,neighbor_lists_lengths(target_index)), 
                KOKKOS_LAMBDA(const int i, double& t_value) {

            t_value += sampling_data_device(neighbor_lists(target_index, i+1))
                *alphas(ORDER_INDICES(target_index, alpha_column_base_multiplier*neighbor_lists(target_index, 0) + i));

        }, value );
        Kokkos::fence();

        return value;
    }

    //! Dot product of alphas with sampling data where sampling data is in a 1D Kokkos View and output view is also a 1D Kokkos View
    //! 
    //! This function is to be used when the alpha values have already been calculated and stored for use.
    //!
    //! Only supports one output component / input component at a time. The user will need to loop over the output 
    //! components in order to fill a vector target or matrix target.
    //! 
    //! Assumptions on input data:
    //! \param output_data_single_column       [out] - 1D Kokkos View (memory space must match sampling_data_single_column)
    //! \param sampling_data_single_column      [in] - 1D Kokkos View (memory space must match output_data_single_column)
    //! \param lro                              [in] - Target operation from the TargetOperation enum
    //! \param target_index                     [in] - Target # user wants to reconstruct target functional at, corresponds to row number of neighbor_lists
    //! \param output_component_axis_1          [in] - Row for a rank 2 tensor or rank 1 tensor, 0 for a scalar output
    //! \param output_component_axis_2          [in] - Columns for a rank 2 tensor, 0 for rank less than 2 output tensor
    //! \param input_component_axis_1           [in] - Row for a rank 2 tensor or rank 1 tensor, 0 for a scalar input
    //! \param input_component_axis_2           [in] - Columns for a rank 2 tensor, 0 for rank less than 2 input tensor
    template <typename view_type_data_out, typename view_type_data_in>
    void applyAlphasToDataSingleComponentAllTargetSitesWithPreAndPostTransform(view_type_data_out output_data_single_column, view_type_data_in sampling_data_single_column, TargetOperation lro, const int output_component_axis_1, const int output_component_axis_2, const int input_component_axis_1, const int input_component_axis_2, const int pre_transform_local_index = -1, const int pre_transform_global_index = -1, const int post_transform_local_index = -1, const int post_transform_global_index = -1) const {

        const int alpha_column_base_multiplier = _gmls->getAlphaColumnOffset(lro, output_component_axis_1, 
                output_component_axis_2, input_component_axis_1, input_component_axis_2);

        auto global_dimensions = _gmls->getGlobalDimensions();

        // gather needed information for remap
        auto neighbor_lists = _gmls->getNeighborLists();
        auto alphas         = _gmls->getAlphas();
        auto tangent_directions = _gmls->getTangentDirections();

        const int num_targets = neighbor_lists.dimension_0(); // one row for each target
        
        //const int this_host_lro_total_offset = this->_host_lro_total_offsets[lro_number];
        
        
        //const int lro_number = _lro_lookup[(int)lro];
        //const int input_index = getTargetInputIndex((int)lro, input_component_axis_1, input_component_axis_2);
        //const int output_index = getTargetOutputIndex((int)lro, output_component_axis_1, output_component_axis_2);
        //const int this_host_lro_output_tile_size = this->_host_lro_output_tile_size[lro_number];

        // make sure input and output views have same memory space
        assert((std::is_same<typename view_type_data_out::memory_space, typename view_type_data_in::memory_space>::value) && 
                "output_data_single_column view and input_data_single_column view have difference memory spaces.");

        bool weight_with_pre_T = (pre_transform_local_index>=0 && pre_transform_global_index>=0) ? true : false;
        bool weight_with_post_T = (post_transform_local_index>=0 && post_transform_global_index>=0) ? true : false;

        //// It is possible to call this function with a view having a memory space of the host
        //// this first case takes case of that scenario. If this function is called by applyTargetToData,
        //// then it will always provide a view having memory space of the device (else case)
        //if (std::is_same<typename view_type_data_in::memory_space, Kokkos::HostSpace>::value) {
        //    typedef Kokkos::TeamPolicy<Kokkos::DefaultHostExecutionSpace> alpha_policy;
        //    typedef Kokkos::TeamPolicy<Kokkos::DefaultHostExecutionSpace>::member_type alpha_member_type;

        //    auto this_T = this->_T;

        //    // loops over target_indexes
        //    Kokkos::parallel_for(alpha_policy(num_targets, Kokkos::AUTO), 
        //            KOKKOS_LAMBDA(const alpha_member_type& teamMember) {
        //        double value = 0;
        //        const int target_index = teamMember.league_rank();
        //        Kokkos::View<double**, layout_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> > T(this_T.data() + target_index*_dimensions*_dimensions, _dimensions, _dimensions);
        //        const double previous_value = output_data_single_column(target_index);
        //        teamMember.team_barrier();
        //        // loops over neighbors of target_index
        //        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember,_host_neighbor_lists(target_index,0)), [=](const int i, double& t_value) {
        //            t_value += sampling_data_single_column(_host_neighbor_lists(target_index, i+1))*_host_alphas(ORDER_INDICES(target_index,
        //                (this_host_lro_total_offset + input_index*output_dimension_of_operator + output_index)
        //                *_number_of_neighbors_list(target_index) + i));
        //        }, value);
        //        double pre_T = (weight_with_pre_T) ? 
        //            this_T(target_index, pre_transform_local_index, pre_transform_global_index) : 1;
        //        double post_T = (weight_with_post_T) ? 
        //            this_T(target_index, post_transform_local_index, post_transform_global_index) : 1;

        //        output_data_single_column(target_index) = previous_value + pre_T*post_T*value;
        //    });
        //} else {
            typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> alpha_policy;
            typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type alpha_member_type;

            // this function is not a functor, so no capture on *this takes place, and therefore memory addresses 
            // will be illegal accesses if made on the device, even if this->... is on the device, so we create a 
            // name that will be captured by KOKKOS_LAMBDA
            //auto neighbor_lists = this->_neighbor_lists;
            //auto alphas = this->_alphas;
            //auto dimensions = this->_dimensions;
            //auto this_T = this->_T;

            // loops over target indices
            Kokkos::parallel_for(alpha_policy(num_targets, Kokkos::AUTO), 
                    KOKKOS_LAMBDA(const alpha_member_type& teamMember) {

                double value = 0;
                const int target_index = teamMember.league_rank();

                Kokkos::View<double**, layout_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> > T
                        (tangent_directions.data() + target_index*global_dimensions*global_dimensions, 
                         global_dimensions, global_dimensions);
                teamMember.team_barrier();

                // loops over neighbors of target_index
                const double previous_value = output_data_single_column(target_index);
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, neighbor_lists(target_index,0)), [=](const int i, double& t_value) {
                    t_value += sampling_data_single_column(neighbor_lists(target_index, i+1))
                        *alphas(ORDER_INDICES(target_index, alpha_column_base_multiplier*neighbor_lists(target_index,0) +i));

                }, value );

                double pre_T = (weight_with_pre_T) ? T(pre_transform_local_index, pre_transform_global_index) : 1.0;
                double post_T = (weight_with_post_T) ? T(post_transform_local_index, post_transform_global_index) : 1.0;
                double added_value = pre_T*post_T*value;
                Kokkos::single(Kokkos::PerTeam(teamMember), [=] () {
                    output_data_single_column(target_index) = previous_value + added_value;
                });
            });
        //}
        Kokkos::fence();
    }


    //! Transformation of data under GMLS
    //! 
    //! This function is to be used when the alpha values have already been calculated and stored for use.
    //!
    //! Produces a Kokkos View as output with a Kokkos memory_space provided as a template tag by the caller. 
    //! The data type (double* or double**) must also be specified as a template type if one wish to get a 1D 
    //! Kokkos View back that can be indexed into with only one ordinal.
    //! 
    //! Assumptions on input data:
    //! \param sampling_data            [in] - 1D or 2D Kokkos View that has the layout #targets * columns of data. Memory space for data can be host or device. It is assumed that this data has already been transformed by the sampling functional.
    //! \param lro                      [in] - Target operation from the TargetOperation enum
    template <typename output_data_type = double**, typename output_memory_space, typename view_type_input_data, typename output_array_layout = typename view_type_input_data::array_layout>
    Kokkos::View<output_data_type, output_array_layout, output_memory_space>  // shares layout of input by default
            applyAlphasToDataAllComponentsAllTargetSites(view_type_input_data sampling_data, TargetOperation lro, SamplingFunctional sro = SamplingFunctional::PointSample, CoordinatesType coords_in = CoordinatesType::Ambient, CoordinatesType coords_out = CoordinatesType::Ambient) const {

        // output can be device or host
        // input can be device or host
        // move everything to device and calculate there, then move back to host if necessary

        typedef Kokkos::View<output_data_type, output_array_layout, output_memory_space> output_view_type;

        auto problem_type = _gmls->getProblemType();
        auto global_dimensions = _gmls->getGlobalDimensions();
        auto output_dimension_of_operator = _gmls->getOutputDimensionOfOperation(lro);
        auto input_dimension_of_operator = _gmls->getInputDimensionOfOperation(lro);

        // gather needed information for remap
        auto neighbor_lists = _gmls->getNeighborLists();

        
        int output_dimensions;
        if (coords_out==CoordinatesType::Ambient && problem_type==MANIFOLD && TargetOutputTensorRank[(int)lro]==1) {
            output_dimensions = global_dimensions;
        } else {
            output_dimensions = output_dimension_of_operator;
        }

        // create view on whatever memory space the user specified with their template argument when calling this function
        output_view_type target_output("output of target", neighbor_lists.dimension_0() /* number of targets */, 
                output_dimensions);

        // create device mirror and write into it then copy back at the end
        //auto target_output_device_mirror = Kokkos::create_mirror(Kokkos::DefaultExecutionSpace::memory_space(), target_output);
        //auto sampling_data_device_mirror = Kokkos::create_mirror(Kokkos::DefaultExecutionSpace::memory_space(), sampling_data);

        // copy sampling data from whatever memory space it is in to the device (does nothing if already on the device)
        //Kokkos::deep_copy(sampling_data_device_mirror, sampling_data);
        //Kokkos::deep_copy(target_output_device_mirror, 0);
        //Kokkos::fence();

        // make sure input and output columns make sense under the target operation
        assert(((output_dimension_of_operator==1 && output_view_type::rank==1) || output_view_type::rank!=1) && 
                "Output view is requested as rank 1, but the target requires a rank larger than 1. Try double** as template argument.");

        // we need to specialize a template on the rank of the output view type and the input view type
        auto sampling_subview_maker = Create1DSliceOnDeviceView(sampling_data);
        auto output_subview_maker = Create1DSliceOnDeviceView(target_output);
        
        // only written for up to rank 1 to rank 1 remap
        // loop over components of output of the target operation
        for (int i=0; i<output_dimension_of_operator; ++i) {
            const int output_component_axis_1 = i;
            const int output_component_axis_2 = 0;
            // loop over components of input of the target operation
            for (int j=0; j<input_dimension_of_operator; ++j) {
                const int input_component_axis_1 = j;
                const int input_component_axis_2 = 0;

                if ((coords_in==CoordinatesType::Ambient && sro==SamplingFunctional::ManifoldVectorSample) &&
                        (coords_out==CoordinatesType::Ambient && problem_type==MANIFOLD 
                         && TargetOutputTensorRank[(int)lro]==1)) {
                    for (int k=0; k<global_dimensions; ++k) {
                        for (int l=0; l<global_dimensions; ++l) {

                            //auto sub_out = Kokkos::subview(out, Kokkos::ALL, column_out);
                            //auto sampling_data_device = sampling_subview_maker.get1DView(column_of_input);
                            this->applyAlphasToDataSingleComponentAllTargetSitesWithPreAndPostTransform(output_subview_maker.get1DView(k), 
                                sampling_subview_maker.get1DView(l), lro, output_component_axis_1, 
                                output_component_axis_2, input_component_axis_1, input_component_axis_2,
                                j, k, i, l);


                            // creates subviews if necessary so that only a 1D Kokkos View is exposed as the input and 
                            // output for applyAlphasToDataSingleComponentAllTargets
                            //sm_w_pre_post_transform.execute(this, target_output_device_mirror, k, sampling_data_device_mirror, 
                            //        l, lro, output_component_axis_1, output_component_axis_2, input_component_axis_1, 
                            //        input_component_axis_2, j, k, i, l);
                            //Kokkos::fence();
                        }
                    }

                } else if (coords_out==CoordinatesType::Ambient && problem_type==MANIFOLD && 
                        TargetOutputTensorRank[(int)lro]==1) {

                    for (int k=0; k<global_dimensions; ++k) {
                        this->applyAlphasToDataSingleComponentAllTargetSitesWithPreAndPostTransform(output_subview_maker.get1DView(k), 
                            sampling_subview_maker.get1DView(j), lro, output_component_axis_1, 
                            output_component_axis_2, input_component_axis_1, input_component_axis_2,
                            -1, -1, i, k);
                        // creates subviews if necessary so that only a 1D Kokkos View is exposed as the input and 
                        // output for applyAlphasToDataSingleComponentAllTargets
                        //sm_w_pre_post_transform.execute(this, target_output_device_mirror, k, sampling_data_device_mirror, 
                        //        j, lro, output_component_axis_1, output_component_axis_2, input_component_axis_1, 
                        //        input_component_axis_2, -1, -1, i, k);
                    }

                } else if (coords_in==CoordinatesType::Ambient && sro==SamplingFunctional::ManifoldVectorSample) {

                    for (int k=0; k<global_dimensions; ++k) {
                        this->applyAlphasToDataSingleComponentAllTargetSitesWithPreAndPostTransform(output_subview_maker.get1DView(i), 
                            sampling_subview_maker.get1DView(k), lro, output_component_axis_1, 
                            output_component_axis_2, input_component_axis_1, input_component_axis_2,
                            j, k, -1, -1);
                        // creates subviews if necessary so that only a 1D Kokkos View is exposed as the input and 
                        // output for applyAlphasToDataSingleComponentAllTargets
                        //sm_w_pre_post_transform.execute(this, target_output_device_mirror, i, sampling_data_device_mirror, 
                        //        k, lro, output_component_axis_1, output_component_axis_2, input_component_axis_1, 
                        //        input_component_axis_2, j, k, -1, -1);
                    }

                } else { // standard
                    this->applyAlphasToDataSingleComponentAllTargetSitesWithPreAndPostTransform(output_subview_maker.get1DView(i), 
                        sampling_subview_maker.get1DView(j), lro, output_component_axis_1, 
                        output_component_axis_2, input_component_axis_1, input_component_axis_2);
                    // creates subviews if necessary so that only a 1D Kokkos View is exposed as the input and 
                    // output for applyAlphasToDataSingleComponentAllTargets
                    //sm_w_pre_post_transform.execute(this, target_output_device_mirror, i, sampling_data_device_mirror, 
                    //        j, lro, output_component_axis_1, output_component_axis_2, input_component_axis_1, 
                    //        input_component_axis_2, -1, -1, -1, -1);
                }
            }
        }

        // copy back to whatever memory space the user requester through templating from the device
        Kokkos::deep_copy(target_output, output_subview_maker.copyToAndReturnOriginalView());
        return target_output;
    }

//    //! like applyTargetToData above, but will write to the users provided view
//    template <typename view_type_output, typename view_type_mapping, typename view_type_data>
//    double applyTargetToData(view_type_output output_data, view_type_data sampling_data, TargetOperation lro, view_type_mapping target_mapping = Kokkos::View<int*>()) const {
//        // TODO fill this in
//    }
//    //! helper struct allowing for subviews of 1D or 2D Kokkos Views with partial template instantiation
//    template <typename view_type_out, int rank_out, typename view_type_in, int rank_in> 
//    struct SubviewMaker {
//        void execute(const Remap* this_gmls_class, view_type_out out, const int column_out, view_type_in in, const int column_in, TargetOperation lro, const int output_component_axis_1, const int output_component_axis_2, const int input_component_axis_1, const int input_component_axis_2, const int pre_transform_local_index = -1, const int pre_transform_global_index = -1, const int post_transform_local_index = -1, const int post_transform_global_index = -1) {
//            auto sub_out = Kokkos::subview(out, Kokkos::ALL, column_out);
//            auto sub_in = Kokkos::subview(in, Kokkos::ALL, column_in);
//            this_gmls_class->applyAlphasToDataSingleComponentAllTargetSitesWithPreAndPostTransform(sub_out, sub_in, lro, output_component_axis_1, output_component_axis_2, input_component_axis_1, input_component_axis_2, pre_transform_local_index, pre_transform_global_index, post_transform_local_index, post_transform_global_index);
//        }
//    };

}; // Remap

}; // Compadre

#endif
