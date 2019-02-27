#ifndef _COMPADRE_EVALUATOR_HPP_
#define _COMPADRE_EVALUATOR_HPP_

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
    bool _scalar_as_vector_if_needed;

    Subview1D(T data_in, T2 data_original_view, bool scalar_as_vector_if_needed) {
        _data_in = data_in;
        _data_original_view = data_original_view;
        _scalar_as_vector_if_needed = scalar_as_vector_if_needed; 
    }

    auto get1DView(const int column_num) -> decltype(Kokkos::subview(_data_in, Kokkos::ALL, column_num)) {
        if (!_scalar_as_vector_if_needed) {
            compadre_assert_debug((column_num<_data_in.dimension_1()) && "Subview asked for column > second dimension of input data.");
        }
        if (column_num<_data_in.dimension_1())
            return Kokkos::subview(_data_in, Kokkos::ALL, column_num);
        else // scalar treated as a vector (being reused for each component of the vector input that was expected)
            return Kokkos::subview(_data_in, Kokkos::ALL, 0);
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
    bool _scalar_as_vector_if_needed;

    Subview1D(T data_in, T2 data_original_view, bool scalar_as_vector_if_needed) {
        _data_in = data_in;
        _data_original_view = data_original_view;
        _scalar_as_vector_if_needed = scalar_as_vector_if_needed; 
    }

    auto get1DView(const int column_num) -> decltype(Kokkos::subview(_data_in, Kokkos::ALL)) {
        // TODO: There is a valid use case for violating this assert, so in the future we may want
        // to add other logic to the evaluator function calling this so that it knows to do nothing with
        // this data.
        if (!_scalar_as_vector_if_needed) {
            compadre_assert_debug((column_num==0) && "Subview asked for column column_num!=0, but _data_in is rank 1.");
        }
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
auto Create1DSliceOnDeviceView(T sampling_input_data_host_or_device, bool scalar_as_vector_if_needed) -> Subview1D<decltype(Kokkos::create_mirror_view(
                    Kokkos::DefaultExecutionSpace::memory_space(), sampling_input_data_host_or_device)), T> {

    // makes view on the device (does nothing if already on the device)
    auto sampling_input_data_device = Kokkos::create_mirror_view(
        Kokkos::DefaultExecutionSpace::memory_space(), sampling_input_data_host_or_device);
    Kokkos::deep_copy(sampling_input_data_device, sampling_input_data_host_or_device);
    Kokkos::fence();

    return Subview1D<decltype(sampling_input_data_device),T>(sampling_input_data_device, 
            sampling_input_data_host_or_device, scalar_as_vector_if_needed);
}




//! \brief Lightweight Evaluator Helper
//! This class is a lightweight wrapper for extracting and applying all relevant data from a GMLS class
//! in order to transform data into a form that can be acted on by the GMLS operator, apply the action of
//! the GMLS operator, and then transform data again (only if on a manifold)
class Evaluator {

private:

    GMLS *_gmls;


public:

    Evaluator(GMLS *gmls) : _gmls(gmls) {
        Kokkos::fence();
    };

    ~Evaluator() {};

    //! Dot product of alphas with sampling data, FOR A SINGLE target_index,  where sampling data is in a 1D/2D Kokkos View
    //! 
    //! This function is to be used when the alpha values have already been calculated and stored for use 
    //!
    //! Only supports one output component / input component at a time. The user will need to loop over the output 
    //! components in order to fill a vector target or matrix target.
    //! 
    //! Assumptions on input data:
    //! \param sampling_input_data      [in] - 1D/2D Kokkos View (no restriction on memory space)
    //! \param column_of_input          [in] - Column of sampling_input_data to use for this input component
    //! \param lro                      [in] - Target operation from the TargetOperation enum
    //! \param target_index             [in] - Target # user wants to reconstruct target functional at, corresponds to row number of neighbor_lists
    //! \param output_component_axis_1  [in] - Row for a rank 2 tensor or rank 1 tensor, 0 for a scalar output
    //! \param output_component_axis_2  [in] - Columns for a rank 2 tensor, 0 for rank less than 2 output tensor
    //! \param input_component_axis_1   [in] - Row for a rank 2 tensor or rank 1 tensor, 0 for a scalar input
    //! \param input_component_axis_2   [in] - Columns for a rank 2 tensor, 0 for rank less than 2 input tensor
    //! \param scalar_as_vector_if_needed [in] - If a 1D view is given, where a 2D view is expected (scalar values given where a vector was expected), then the scalar will be repeated for as many components as the vector has
    template <typename view_type_data>
    double applyAlphasToDataSingleComponentSingleTargetSite(view_type_data sampling_input_data, const int column_of_input, TargetOperation lro, const int target_index, const int evaluation_site_local_index, const int output_component_axis_1, const int output_component_axis_2, const int input_component_axis_1, const int input_component_axis_2, bool scalar_as_vector_if_needed = true) const {

        double value = 0;

        const int alpha_column_base_multiplier = _gmls->getAlphaColumnOffset(lro, output_component_axis_1, 
                output_component_axis_2, input_component_axis_1, input_component_axis_2, evaluation_site_local_index);

        auto sampling_subview_maker = Create1DSliceOnDeviceView(sampling_input_data, scalar_as_vector_if_needed);

        
        // gather needed information for evaluation
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

    //! Dot product of alphas with sampling data where sampling data is in a 1D/2D Kokkos View and output view is also 
    //! a 1D/2D Kokkos View, however THE SAMPLING DATA and OUTPUT VIEW MUST BE ON THE DEVICE!
    //! 
    //! This function is to be used when the alpha values have already been calculated and stored for use.
    //!
    //! Only supports one output component / input component at a time. The user will need to loop over the output 
    //! components in order to fill a vector target or matrix target.
    //! 
    //! Assumptions on input data:
    //! \param output_data_single_column       [out] - 1D Kokkos View (memory space must be Kokkos::DefaultExecutionSpace::memory_space())
    //! \param sampling_data_single_column      [in] - 1D Kokkos View (memory space must match output_data_single_column)
    //! \param lro                              [in] - Target operation from the TargetOperation enum
    //! \param sro                              [in] - Sampling functional from the SamplingFunctional enum
    //! \param target_index                     [in] - Target # user wants to reconstruct target functional at, corresponds to row number of neighbor_lists
    //! \param output_component_axis_1          [in] - Row for a rank 2 tensor or rank 1 tensor, 0 for a scalar output
    //! \param output_component_axis_2          [in] - Columns for a rank 2 tensor, 0 for rank less than 2 output tensor
    //! \param input_component_axis_1           [in] - Row for a rank 2 tensor or rank 1 tensor, 0 for a scalar input
    //! \param input_component_axis_2           [in] - Columns for a rank 2 tensor, 0 for rank less than 2 input tensor
    //! \param pre_transform_local_index        [in] - For manifold problems, this is the local coordinate direction that sampling data may need to be transformed to before the application of GMLS
    //! \param pre_transform_global_index       [in] - For manifold problems, this is the global coordinate direction that sampling data can be represented in
    //! \param post_transform_local_index       [in] - For manifold problems, this is the local coordinate direction that vector output target functionals from GMLS will output into
    //! \param post_transform_global_index      [in] - For manifold problems, this is the global coordinate direction that the target functional output from GMLS will be transformed into
    //! \param transform_output_ambient         [in] - Whether or not a 1D output from GMLS is on the manifold and needs to be mapped to ambient space
    //! \param vary_on_target                   [in] - Whether the sampling functional has a tensor to act on sampling data that varies with each target site
    //! \param vary_on_neighbor                 [in] - Whether the sampling functional has a tensor to act on sampling data that varies with each neighbor site in addition to varying wit each target site
    template <typename view_type_data_out, typename view_type_data_in>
    void applyAlphasToDataSingleComponentAllTargetSitesWithPreAndPostTransform(view_type_data_out output_data_single_column, view_type_data_in sampling_data_single_column, TargetOperation lro, SamplingFunctional sro, const int evaluation_site_local_index, const int output_component_axis_1, const int output_component_axis_2, const int input_component_axis_1, const int input_component_axis_2, const int pre_transform_local_index = -1, const int pre_transform_global_index = -1, const int post_transform_local_index = -1, const int post_transform_global_index = -1, bool transform_output_ambient = false, bool vary_on_target = false, bool vary_on_neighbor = false) const {

        const int alpha_column_base_multiplier = _gmls->getAlphaColumnOffset(lro, output_component_axis_1, 
                output_component_axis_2, input_component_axis_1, input_component_axis_2, evaluation_site_local_index);
        const int alpha_column_base_multiplier2 = alpha_column_base_multiplier;

        auto global_dimensions = _gmls->getGlobalDimensions();

        // gather needed information for evaluation
        auto neighbor_lists = _gmls->getNeighborLists();
        auto alphas         = _gmls->getAlphas();
        auto tangent_directions = _gmls->getTangentDirections();
        auto prestencil_weights = _gmls->getPrestencilWeights();

        const int num_targets = neighbor_lists.dimension_0(); // one row for each target

        // make sure input and output views have same memory space
        compadre_assert_debug((std::is_same<typename view_type_data_out::memory_space, typename view_type_data_in::memory_space>::value) && 
                "output_data_single_column view and input_data_single_column view have difference memory spaces.");

        bool weight_with_pre_T = (pre_transform_local_index>=0 && pre_transform_global_index>=0) ? true : false;
        bool target_plus_neighbor_staggered_schema = SamplingTensorForTargetSite[(int)sro];

        typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> alpha_policy;
        typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type alpha_member_type;

        // loops over target indices
        Kokkos::parallel_for(alpha_policy(num_targets, Kokkos::AUTO),
                KOKKOS_LAMBDA(const alpha_member_type& teamMember) {

            const int target_index = teamMember.league_rank();

            Kokkos::View<double**, layout_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> > T
                    (tangent_directions.data() + target_index*global_dimensions*global_dimensions, 
                     global_dimensions, global_dimensions);
            teamMember.team_barrier();


            const double previous_value = output_data_single_column(target_index);

            // loops over neighbors of target_index
            double gmls_value = 0;
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, neighbor_lists(target_index,0)), [=](const int i, double& t_value) {
                const double neighbor_varying_pre_T =  (weight_with_pre_T && vary_on_neighbor) ?
                    prestencil_weights(0, target_index, i, pre_transform_local_index, pre_transform_global_index)
                    : 1.0;

                t_value += neighbor_varying_pre_T * sampling_data_single_column(neighbor_lists(target_index, i+1))
                    *alphas(ORDER_INDICES(target_index, alpha_column_base_multiplier*neighbor_lists(target_index,0) +i));

            }, gmls_value );

            // data contract for sampling functional
            double pre_T = 1.0;
            if (weight_with_pre_T) {
                if (!vary_on_neighbor && vary_on_target) {
                    pre_T = prestencil_weights(0, target_index, 0, pre_transform_local_index, 
                            pre_transform_global_index); 
                } else if (!vary_on_target) { // doesn't vary on target or neighbor
                    pre_T = prestencil_weights(0, 0, 0, pre_transform_local_index, 
                            pre_transform_global_index); 
                }
            }

            double staggered_value_from_targets = 0;
            double pre_T_staggered = 1.0;
            // loops over target_index for each neighbor for staggered approaches
            if (target_plus_neighbor_staggered_schema) {
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, neighbor_lists(target_index,0)), [=](const int i, double& t_value) {
                    const double neighbor_varying_pre_T_staggered =  (weight_with_pre_T && vary_on_neighbor) ?
                        prestencil_weights(1, target_index, i, pre_transform_local_index, pre_transform_global_index)
                        : 1.0;

                    t_value += neighbor_varying_pre_T_staggered * sampling_data_single_column(neighbor_lists(target_index, 1))
                        *alphas(ORDER_INDICES(target_index, alpha_column_base_multiplier2*neighbor_lists(target_index,0) +i));

                }, staggered_value_from_targets );

                // for staggered approaches that transform source data for the target and neighbors
                if (weight_with_pre_T) {
                    if (!vary_on_neighbor && vary_on_target) {
                        pre_T_staggered = prestencil_weights(1, target_index, 0, pre_transform_local_index, 
                                pre_transform_global_index); 
                    } else if (!vary_on_target) { // doesn't vary on target or neighbor
                        pre_T_staggered = prestencil_weights(1, 0, 0, pre_transform_local_index, 
                                pre_transform_global_index); 
                    }
                }
            }

            double post_T = (transform_output_ambient) ? T(post_transform_local_index, post_transform_global_index) : 1.0;
            double added_value = post_T*(pre_T*gmls_value + pre_T_staggered*staggered_value_from_targets);
            Kokkos::single(Kokkos::PerTeam(teamMember), [=] () {
                output_data_single_column(target_index) = previous_value + added_value;
            });
        });
        Kokkos::fence();
    }

    //! Transformation of data under GMLS
    //! 
    //! This function is the go to function to be used when the alpha values have already been calculated and stored for use. The sampling functional provided instructs how a data transformation tensor is to be used on source data before it is provided to the GMLS operator. Once the sampling functional (if applicable) and the GMLS operator have been applied, this function also handles mapping the local vector back to the ambient space if working on a manifold problem and a target functional who has rank 1 output.
    //!
    //! Produces a Kokkos View as output with a Kokkos memory_space provided as a template tag by the caller. 
    //! The data type (double* or double**) must also be specified as a template type if one wish to get a 1D 
    //! Kokkos View back that can be indexed into with only one ordinal.
    //! 
    //! Assumptions on input data:
    //! \param sampling_data              [in] - 1D or 2D Kokkos View that has the layout #targets * columns of data. Memory space for data can be host or device. 
    //! \param lro                        [in] - Target operation from the TargetOperation enum
    //! \param sro                        [in] - Sampling functional from the SamplingFunctional enum
    //! \param scalar_as_vector_if_needed [in] - If a 1D view is given, where a 2D view is expected (scalar values given where a vector was expected), then the scalar will be repeated for as many components as the vector has
    template <typename output_data_type = double**, typename output_memory_space, typename view_type_input_data, typename output_array_layout = typename view_type_input_data::array_layout>
    Kokkos::View<output_data_type, output_array_layout, output_memory_space>  // shares layout of input by default
            applyAlphasToDataAllComponentsAllTargetSites(view_type_input_data sampling_data, TargetOperation lro, SamplingFunctional sro = SamplingFunctional::PointSample, bool scalar_as_vector_if_needed = true, const int evaluation_site_local_index = 0) const {


        // output can be device or host
        // input can be device or host
        // move everything to device and calculate there, then move back to host if necessary

        typedef Kokkos::View<output_data_type, output_array_layout, output_memory_space> output_view_type;

        auto problem_type = _gmls->getProblemType();
        auto global_dimensions = _gmls->getGlobalDimensions();
        auto output_dimension_of_operator = _gmls->getOutputDimensionOfOperation(lro);
        auto input_dimension_of_operator = _gmls->getInputDimensionOfOperation(lro);

        // gather needed information for evaluation
        auto neighbor_lists = _gmls->getNeighborLists();

        // determines the number of columns needed for output after action of the target functional
        int output_dimensions;
        if (problem_type==MANIFOLD && TargetOutputTensorRank[(int)lro]==1) {
            output_dimensions = global_dimensions;
        } else {
            output_dimensions = output_dimension_of_operator;
        }

        // special case for VectorPointSample, because if it is on a manifold it includes data transform to local charts
        if (problem_type==MANIFOLD && sro==VectorPointSample) {
            sro = ManifoldVectorPointSample;
        }

        // create view on whatever memory space the user specified with their template argument when calling this function
        output_view_type target_output("output of target", neighbor_lists.dimension_0() /* number of targets */, 
                output_dimensions);

        // make sure input and output columns make sense under the target operation
        compadre_assert_debug(((output_dimension_of_operator==1 && output_view_type::rank==1) || output_view_type::rank!=1) && 
                "Output view is requested as rank 1, but the target requires a rank larger than 1. Try double** as template argument.");

        // we need to specialize a template on the rank of the output view type and the input view type
        auto sampling_subview_maker = Create1DSliceOnDeviceView(sampling_data, scalar_as_vector_if_needed);
        auto output_subview_maker = Create1DSliceOnDeviceView(target_output, false); // output will always be the correct dimension

        // figure out preprocessing and postprocessing
        auto prestencil_weights = _gmls->getPrestencilWeights();

        // all loop logic based on transforming data under a sampling functional
        // into something that is valid input for GMLS
        bool vary_on_target, vary_on_neighbor;
        auto sro_style = SamplingTensorStyle[(int)sro];
        bool loop_global_dimensions = SamplingInputTensorRank[(int)sro]>0 && sro_style!=Identity; 


        if (SamplingTensorStyle[(int)sro] == Identity || SamplingTensorStyle[(int)sro] == SameForAll) {
            vary_on_target = false;
            vary_on_neighbor = false;
        } else if (SamplingTensorStyle[(int)sro] == DifferentEachTarget) {
            vary_on_target = true;
            vary_on_neighbor = false;
        } else if (SamplingTensorStyle[(int)sro] == DifferentEachNeighbor) {
            vary_on_target = true;
            vary_on_neighbor = true;
        }


        bool transform_gmls_output_to_ambient = (problem_type==MANIFOLD && TargetOutputTensorRank[(int)lro]==1);


        // only written for up to rank 1 to rank 1 (in / out)
        // loop over components of output of the target operation
        for (int i=0; i<output_dimension_of_operator; ++i) {
            const int output_component_axis_1 = i;
            const int output_component_axis_2 = 0;
            // loop over components of input of the target operation
            for (int j=0; j<input_dimension_of_operator; ++j) {
                const int input_component_axis_1 = j;
                const int input_component_axis_2 = 0;

                if (loop_global_dimensions && transform_gmls_output_to_ambient) {
                    for (int k=0; k<global_dimensions; ++k) { // loop for handling sampling functional
                        for (int l=0; l<global_dimensions; ++l) { // loop for transforming output of GMLS to ambient
                            this->applyAlphasToDataSingleComponentAllTargetSitesWithPreAndPostTransform(
                                    output_subview_maker.get1DView(k), sampling_subview_maker.get1DView(l), 
                                    lro, sro, evaluation_site_local_index, output_component_axis_1, output_component_axis_2, 
                                    input_component_axis_1, input_component_axis_2, j, k, i, l,
                                    transform_gmls_output_to_ambient, vary_on_target, vary_on_neighbor);
                        }
                    }
                } else if (transform_gmls_output_to_ambient) {
                    for (int k=0; k<global_dimensions; ++k) { // loop for transforming output of GMLS to ambient
                        this->applyAlphasToDataSingleComponentAllTargetSitesWithPreAndPostTransform(
                                output_subview_maker.get1DView(k), sampling_subview_maker.get1DView(j), lro, sro, 
                                evaluation_site_local_index, output_component_axis_1, output_component_axis_2, input_component_axis_1, 
                                input_component_axis_2, -1, -1, i, k,
                                transform_gmls_output_to_ambient, vary_on_target, vary_on_neighbor);
                    }
                } else if (loop_global_dimensions) {
                    for (int k=0; k<global_dimensions; ++k) { // loop for handling sampling functional
                        this->applyAlphasToDataSingleComponentAllTargetSitesWithPreAndPostTransform(
                                output_subview_maker.get1DView(i), sampling_subview_maker.get1DView(k), lro, sro, 
                                evaluation_site_local_index, output_component_axis_1, output_component_axis_2, input_component_axis_1, 
                                input_component_axis_2, j, k, -1, -1, transform_gmls_output_to_ambient,
                                vary_on_target, vary_on_neighbor);
                    }
                } else if (sro_style != Identity) {
                    this->applyAlphasToDataSingleComponentAllTargetSitesWithPreAndPostTransform(
                            output_subview_maker.get1DView(i), sampling_subview_maker.get1DView(j), lro, sro, 
                            evaluation_site_local_index, output_component_axis_1, output_component_axis_2, input_component_axis_1, 
                            input_component_axis_2, 0, 0, -1, -1,
                            transform_gmls_output_to_ambient, vary_on_target, vary_on_neighbor);
                } else { // standard
                    this->applyAlphasToDataSingleComponentAllTargetSitesWithPreAndPostTransform(
                            output_subview_maker.get1DView(i), sampling_subview_maker.get1DView(j), lro, sro, 
                            evaluation_site_local_index, output_component_axis_1, output_component_axis_2, input_component_axis_1, 
                            input_component_axis_2);
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

}; // Evaluator

}; // Compadre

#endif
