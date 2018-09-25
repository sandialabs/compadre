#ifndef _GMLS_HPP_
#define _GMLS_HPP_


#include "GMLS_Config.h"
#include <assert.h>
#include "GMLS_LinearAlgebra_Definitions.hpp"

#ifdef COMPADRE_USE_KOKKOSCORE

struct XYZ {
	KOKKOS_INLINE_FUNCTION
	XYZ() : x(0), y(0), z(0) {}

	KOKKOS_INLINE_FUNCTION
	XYZ(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}

	double x;
	double y;
	double z;

	KOKKOS_INLINE_FUNCTION
	double& operator [](const int i) {
		switch (i) {
			case 0:
				return x;
			case 1:
				return y;
			default:
				return z;
		}
	}

	KOKKOS_INLINE_FUNCTION
	XYZ operator *(double scalar) {
		XYZ result;
		result.x = scalar*x;
		result.y = scalar*y;
		result.z = scalar*z;
		return result;
	}
};

namespace ReconstructionOperator {

	enum TargetOperation {
		ScalarPointEvaluation,
		VectorPointEvaluation, // reconstructs entire vector at once
		LaplacianOfScalarPointEvaluation,
		LaplacianOfVectorPointEvaluation,
		GradientOfScalarPointEvaluation,
		GradientOfVectorPointEvaluation,
		DivergenceOfVectorPointEvaluation,
		CurlOfVectorPointEvaluation,
		PartialXOfScalarPointEvaluation,
		PartialYOfScalarPointEvaluation,
		PartialZOfScalarPointEvaluation,
		DivergenceOfScalarPointEvaluation,
		ChainedStaggeredLaplacianOfScalarPointEvaluation,
		COUNT=13,
	};

	enum ReconstructionSpace {
		ScalarTaylorPolynomial,
		VectorTaylorPolynomial,
		DivergenceFreeVectorPolynomial,
	};

	enum SamplingFunctional {
		PointSample,
		ManifoldVectorSample,
		ManifoldGradientVectorSample,
		StaggeredEdgeAnalyticGradientIntegralSample,
		StaggeredEdgeIntegralSample,
	};

	enum DenseSolverType {
		QR,
		LU,
		SVD,
		MANIFOLD,
	};

	const int TargetInputTensorRank[] = {
		0, // ScalarPointEvaluation
		1, // VectorPointEvaluation
		0, // LaplacianOfScalarPointEvaluation
		1, // LaplacianOfVectorPointEvaluation
		0, // GradientOfScalarPointEvaluation
		1, // GradientOfVectorPointEvaluation
		1, // DivergenceOfVectorPointEvaluation
		1, // CurlOfVectorPointEvaluation
		0, // PartialXOfScalarPointEvaluation
		0, // PartialYOfScalarPointEvaluation
		0, // PartialZOfScalarPointEvaluation
		0, // DivergenceOfScalarPointEvaluation
		0, // ChainedStaggeredLaplacianOfScalarPointEvaluation
	};

	const int TargetOutputTensorRank[] {
		0, // PointEvaluation
		1, // VectorPointEvaluation
		0, // LaplacianOfScalarPointEvaluation
		1, // LaplacianOfVectorPointEvaluation
		1, // GradientOfScalarPointEvaluation
		1, // GradientOfVectorPointEvaluation
		0, // DivergenceOfVectorPointEvaluation
		1, // CurlOfVectorPointEvaluation
		0, // PartialXOfScalarPointEvaluation
		0, // PartialYOfScalarPointEvaluation
		0, // PartialZOfScalarPointEvaluation
		0, // DivergenceOfScalarPointEvaluation
		0, // ChainedStaggeredLaplacianOfScalarPointEvaluation
	};

	const int ReconstructionSpaceRank[] = {
		0, // ScalarTaylorPolynomial
		1, // VectorTaylorPolynomial
		1, // DivergenceFreeVectorPolynomial
		0, // ScalarBernsteinPolynomial
		1, // VectorBernsteinPolynomial
	};

	const int SamplingInputTensorRank[] = {
		0, // PointSample
		1, // ManifoldVectorSample
		1, // ManifoldGradientVectorSample
		0, // StaggeredEdgeAnalyticGradientIntegralSample,
		1, // StaggeredEdgeIntegralSample
	};

	const int SamplingOutputTensorRank[] {
		0, // PointSample
		1, // ManifoldVectorSample
		1, // ManifoldGradientVectorSample
		0, // StaggeredEdgeAnalyticGradientIntegralSample,
		0, // StaggeredEdgeIntegralSample
	};

	const int SamplingNontrivialNullspace[] {
		// does the sample over polynomials result in an operator
		// with a nontrivial nullspace requiring SVD
		0, // PointSample
		0, // ManifoldVectorSample
		0, // ManifoldGradientVectorSample
		1, // StaggeredEdgeAnalyticGradientIntegralSample,
		1, // StaggeredEdgeIntegralSample
	};

	static int getTargetInputIndex(const int operation_num, const int input_component_axis_1, const int input_component_axis_2) {
		const int axis_1_size = (TargetInputTensorRank[operation_num] > 1) ? TargetInputTensorRank[operation_num] : 1;
		return axis_1_size*input_component_axis_1 + input_component_axis_2; // 0 for scalar, 0 for vector;
	}

	static int getTargetOutputIndex(const int operation_num, const int output_component_axis_1, const int output_component_axis_2) {
		const int axis_1_size = (TargetOutputTensorRank[operation_num] > 1) ? TargetOutputTensorRank[operation_num] : 1;
		return axis_1_size*output_component_axis_1 + output_component_axis_2; // 0 for scalar, 0 for vector;
	}

	static int getSamplingInputIndex(const int operation_num, const int input_component_axis_1, const int input_component_axis_2) {
		const int axis_1_size = (SamplingInputTensorRank[operation_num] > 1) ? SamplingInputTensorRank[operation_num] : 1;
		return axis_1_size*input_component_axis_1 + input_component_axis_2; // 0 for scalar, 0 for vector;
	}

	static int getSamplingOutputIndex(const int operation_num, const int output_component_axis_1, const int output_component_axis_2) {
		const int axis_1_size = (SamplingOutputTensorRank[operation_num] > 1) ? SamplingOutputTensorRank[operation_num] : 1;
		return axis_1_size*output_component_axis_1 + output_component_axis_2; // 0 for scalar, 0 for vector;
	}

	static bool validTargetSpaceSample(TargetOperation to, ReconstructionSpace rs, SamplingFunctional sf) {
		// all valid combinations to be added here
		return true;
	}

	enum WeightingFunctionType {
		Power,
		Gaussian
	};

} // namespace ReconstructionOperator




class GMLS {
protected:

	//	std::cout << "_NP*_host_operations.size()" << _NP*_host_operations.size() << std::endl;
	Kokkos::View<int**, layout_type> _neighbor_lists; // contains local ids of neighbors to get coords from _source_coordinates
	Kokkos::View<int**, layout_type>::HostMirror _host_neighbor_lists; // contains local ids of neighbors to get coords from _source_coordinates
	Kokkos::View<int*, Kokkos::HostSpace> _number_of_neighbors_list; // contains the # of neighbors for each target
	Kokkos::View<double**, layout_type> _source_coordinates; // all coordinates for the source for which _neighbor_lists refers
	Kokkos::View<double**, layout_type>::HostMirror _host_source_coordinates; // all coordinates for the source for which _neighbor_lists refers
	Kokkos::View<double**, layout_type> _target_coordinates; // same number of rows as _neighbor_lists
	Kokkos::View<double**, layout_type>::HostMirror _host_target_coordinates; // same number of rows as _neighbor_lists
    Kokkos::View<double*> _epsilons; // h supports determined through neighbor search, same number of rows as _neighbor_lists
    Kokkos::View<double*>::HostMirror _host_epsilons; // h supports determined through neighbor search, same number of rows as _neighbor_lists

    Kokkos::View<double**, layout_type> _alphas; // generated coefficients
    Kokkos::View<const double**, layout_type>::HostMirror _host_alphas;

    Kokkos::View<double**, layout_type> _prestencil_weights; // generated weights for nontraditional samples
    Kokkos::View<const double**, layout_type>::HostMirror _host_prestencil_weights;

    Kokkos::View<double**, layout_type> _operator_coefficients; // coefficients for operators or prestencils

    int _type; // reconstruction type
    int _poly_order; // order of polynomial reconstruction
    int _manifold_poly_order; // order of manifold polynomial reconstruction
    int _NP;

    int _scratch_team_level;
    int _scratch_thread_level;

    int _dimensions;

    ReconstructionOperator::ReconstructionSpace _reconstruction_space;
    ReconstructionOperator::SamplingFunctional _polynomial_sampling_functional;
    ReconstructionOperator::SamplingFunctional _data_sampling_functional;

    Kokkos::View<ReconstructionOperator::TargetOperation*> _manifold_support_operations;
    Kokkos::View<ReconstructionOperator::TargetOperation*> _operations;
    Kokkos::View<ReconstructionOperator::TargetOperation*>::HostMirror _host_operations;

    std::vector<int> _lro_lookup; // gets the # for an operation from a given operation

    Kokkos::View<int*, layout_type> _lro_total_offsets; // index for where this operation begins for alpha coefficients
    Kokkos::View<int*, layout_type> _lro_output_tile_size; // size of this output dependent on tensor rank
    Kokkos::View<int*, layout_type> _lro_input_tile_size; // size of this input dependent on tensor rank
    Kokkos::View<int*, layout_type> _lro_input_tensor_rank;
    Kokkos::View<int*, layout_type> _lro_output_tensor_rank;

    Kokkos::View<int*, layout_type>::HostMirror _host_lro_total_offsets; // index for where this operation begins for alpha coefficients
	Kokkos::View<int*, layout_type>::HostMirror _host_lro_output_tile_size; // size of this output dependent on tensor rank
	Kokkos::View<int*, layout_type>::HostMirror _host_lro_input_tile_size; // size of this input dependent on tensor rank
    Kokkos::View<int*, layout_type>::HostMirror _host_lro_input_tensor_rank;
    Kokkos::View<int*, layout_type>::HostMirror _host_lro_output_tensor_rank;

	Kokkos::View<double*, layout_type> _quadrature_weights;
	Kokkos::View<double*, layout_type> _parameterized_quadrature_sites;

    int _max_target_tile_size;

    std::vector<ReconstructionOperator::TargetOperation> _lro; // user requested target operations

    int _total_alpha_values;
    int _total_output_values;

    ReconstructionOperator::DenseSolverType _dense_solver_type;
    ReconstructionOperator::WeightingFunctionType _weighting_type;
    ReconstructionOperator::WeightingFunctionType _manifold_weighting_type;
    int _weighting_power;
    int _manifold_weighting_power;

	int _basis_multiplier;
	int _sampling_multiplier;
	int _number_of_quadrature_points;

	bool _nontrivial_nullspace;


    // PRIVATE MEMBER FUNCTIONS


    KOKKOS_INLINE_FUNCTION
    void calcWij(double* delta, const int target_index, int neighbor_index, const double alpha, const int dimension, const int poly_order, bool specific_order_only = false, scratch_matrix_type* V = NULL, scratch_matrix_type* T = NULL, const ReconstructionOperator::SamplingFunctional sampling_strategy = ReconstructionOperator::SamplingFunctional::PointSample, scratch_vector_type* target_manifold_gradient = NULL, scratch_matrix_type* quadrature_manifold_gradients = NULL) const;

    KOKKOS_INLINE_FUNCTION
    void calcGradientWij(double* delta, const int target_index, const int neighbor_index, const double alpha, const int partial_direction, const int dimension, const int poly_order, bool specific_order_only, scratch_matrix_type* V, const ReconstructionOperator::SamplingFunctional sampling_strategy) const;

    KOKKOS_INLINE_FUNCTION
    double Wab(const double r, const double h, const ReconstructionOperator::WeightingFunctionType& weighting_type, const int power) const; //Calculates weighting function
    
    KOKKOS_INLINE_FUNCTION
    double factorial(const int n) const;

    KOKKOS_INLINE_FUNCTION
    void createWeightsAndP(const member_type& teamMember, scratch_vector_type delta, scratch_matrix_type P, scratch_vector_type w, const int dimension, int polynomial_order, bool weight_p = false, scratch_matrix_type* V = NULL, scratch_matrix_type* T = NULL, const ReconstructionOperator::SamplingFunctional sampling_strategy = ReconstructionOperator::SamplingFunctional::PointSample, scratch_vector_type* target_manifold_gradient = NULL, scratch_matrix_type* quadrature_manifold_gradients = NULL) const;

    KOKKOS_INLINE_FUNCTION
	void createWeightsAndPForCurvature(const member_type& teamMember, scratch_vector_type delta, scratch_matrix_type P, scratch_vector_type w, const int dimension, bool only_specific_order, scratch_matrix_type* V = NULL) const;

    KOKKOS_INLINE_FUNCTION
	void computeTargetFunctionals(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_matrix_type P_target_row, const int basis_multiplier_component = 0) const;

    KOKKOS_INLINE_FUNCTION
	void computeManifoldFunctionals(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_matrix_type P_target_row, scratch_matrix_type* V, const int neighbor_index, const double alpha, const int basis_multiplier_component = 0) const;

    KOKKOS_INLINE_FUNCTION
	void computeTargetFunctionalsOnManifold(const member_type& teamMember, scratch_vector_type t1, scratch_vector_type t2, scratch_matrix_type P_target_row, Kokkos::View<ReconstructionOperator::TargetOperation*> operations, scratch_matrix_type V, scratch_matrix_type T, scratch_matrix_type G_inv, scratch_vector_type manifold_coefficients, scratch_vector_type manifold_gradients, const int basis_multiplier_component = 0) const;

    void generateLaplacianAlpha(const int target_index);

	void generate1DQuadrature();

    // PRIVATE ACCESSORS
	// private function because information lives on the device
	// host should know how many neighbors for each target because they provided the information

    KOKKOS_INLINE_FUNCTION
    int getNNeighbors(const int target_index) const {
		return _neighbor_lists(target_index, 0);
    }

    void printNeighbors(const int target_index) const; //IO function

    void printNeighborData(const int target_index, const int dimensions) const; //IO function

    KOKKOS_INLINE_FUNCTION
    int getNeighborIndex(const int target_index, const int neighbor_list_num) const {
		return _neighbor_lists(target_index, neighbor_list_num+1);
    }

    KOKKOS_INLINE_FUNCTION
    double EuclideanVectorLength(const XYZ& delta_vector, const int dimension) const {

    	double inside_val = delta_vector.x*delta_vector.x;
    	switch (dimension) {
    	case 2:
    		inside_val += delta_vector.y*delta_vector.y;
    		// no break is intentional
    	case 3:
    		inside_val += delta_vector.z*delta_vector.z;
    		// no break is intentional
    	default:
    		break;
    	}
		return std::sqrt(inside_val);

    }

    KOKKOS_INLINE_FUNCTION
    double getTargetCoordinate(const int target_index, const int dim, const scratch_matrix_type* V = NULL) const {
    	if (V==NULL) {
    		return _target_coordinates(target_index, dim);
    	} else {
    		XYZ target_coord = XYZ(_target_coordinates(target_index, 0), _target_coordinates(target_index, 1), _target_coordinates(target_index, 2));
    		return this->convertGlobalToLocalCoordinate(target_coord, dim, V);
    	}
    }

    KOKKOS_INLINE_FUNCTION
    double getNeighborCoordinate(const int target_index, const int neighbor_list_num, const int dim, const scratch_matrix_type* V = NULL) const {
    	if (V==NULL) {
    		return _source_coordinates(this->getNeighborIndex(target_index, neighbor_list_num), dim);
    	} else {
    		XYZ neighbor_coord = XYZ(_source_coordinates(this->getNeighborIndex(target_index, neighbor_list_num), 0), _source_coordinates(this->getNeighborIndex(target_index, neighbor_list_num), 1), _source_coordinates(this->getNeighborIndex(target_index, neighbor_list_num), 2));
    		return this->convertGlobalToLocalCoordinate(neighbor_coord, dim, V);
    	}
    }

    KOKKOS_INLINE_FUNCTION
    XYZ getRelativeCoord(const int target_index, const int neighbor_list_num, const int dimension, const scratch_matrix_type* V = NULL) const {
		XYZ coordinate_delta;

		coordinate_delta.x = this->getNeighborCoordinate(target_index, neighbor_list_num, 0, V) - this->getTargetCoordinate(target_index, 0, V);
		if (dimension>1) coordinate_delta.y = this->getNeighborCoordinate(target_index, neighbor_list_num, 1, V) - this->getTargetCoordinate(target_index, 1, V);
		if (dimension>2) coordinate_delta.z = this->getNeighborCoordinate(target_index, neighbor_list_num, 2, V) - this->getTargetCoordinate(target_index, 2, V);

        return coordinate_delta;
    }

    KOKKOS_INLINE_FUNCTION
    double convertGlobalToLocalCoordinate(const XYZ global_coord, const int dim, const scratch_matrix_type* V) const {
    	// only written for 2d manifold in 3d space
		double val = 0;
		val += global_coord.x * (*V)(0, dim);
		val += global_coord.y * (*V)(1, dim); // can't be called from dimension 1 problem
		if (_dimensions>2) val += global_coord.z * (*V)(2, dim);
		return val;
    }

    KOKKOS_INLINE_FUNCTION
    double convertLocalToGlobalCoordinate(const XYZ local_coord, const int dim, const scratch_matrix_type* V) const {
    	// only written for 2d manifold in 3d space
		double val;
		if (dim == 0 && _dimensions==2) { // 2D problem with 1D manifold
			val = local_coord.x * (*V)(dim, 0);
		} else if (dim == 0) { // 3D problem with 2D manifold
			val = local_coord.x * ((*V)(dim, 0) + (*V)(dim, 1));
		} else if (dim == 1) { // 3D problem with 2D manifold
			val = local_coord.y * ((*V)(dim, 0) + (*V)(dim, 1));
		}
		return val;
    }


public:
    GMLS(const int poly_order,
			const std::string dense_solver_type = std::string("QR"),
			const int manifold_poly_order = 2,
			const int dimensions = 3) : _poly_order(poly_order), _manifold_poly_order(manifold_poly_order), _dimensions(dimensions) {

        _NP = this->getNP(_poly_order, dimensions);
        Kokkos::fence();

#ifdef KOKKOS_HAVE_CUDA
		_scratch_team_level = 1;
		_scratch_thread_level = 1;
#else
		_scratch_team_level = 0;
		_scratch_thread_level = 0;
#endif

		// temporary, just to avoid warning
		_dense_solver_type = ReconstructionOperator::DenseSolverType::QR;
		// set based on input
		this->setSolverType(dense_solver_type);

	    _lro_lookup = std::vector<int>(ReconstructionOperator::TargetOperation::COUNT,-1); // hard coded against number of operations defined
	    _lro = std::vector<ReconstructionOperator::TargetOperation>();

	    // various initializations
	    _total_output_values = 0;
	    _max_target_tile_size = 0;
	    _type = 1;
	    _total_alpha_values = 0;

	    _weighting_type = ReconstructionOperator::WeightingFunctionType::Power;
	    _manifold_weighting_type = ReconstructionOperator::WeightingFunctionType::Power;
	    _weighting_power = 2;
	    _manifold_weighting_power = 2;

	    _reconstruction_space = ReconstructionOperator::ReconstructionSpace::ScalarTaylorPolynomial;
	    _polynomial_sampling_functional = ReconstructionOperator::SamplingFunctional::PointSample;
	    _data_sampling_functional = ReconstructionOperator::SamplingFunctional::PointSample;

	    _basis_multiplier = 1;
	    _sampling_multiplier = 1;
	    _number_of_quadrature_points = 2;

	    _nontrivial_nullspace = false;
    }


    GMLS(ReconstructionOperator::ReconstructionSpace reconstruction_space,
		ReconstructionOperator::SamplingFunctional polynomial_sampling_strategy,
		ReconstructionOperator::SamplingFunctional data_sampling_strategy,
		const int poly_order,
		const std::string dense_solver_type = std::string("QR"),
		const int manifold_poly_order = 2,
		const int dimensions = 3)
			: GMLS(poly_order, dense_solver_type, manifold_poly_order, dimensions) {

	_reconstruction_space = reconstruction_space;
	_polynomial_sampling_functional = polynomial_sampling_strategy;
	_data_sampling_functional = data_sampling_strategy;
    };

    GMLS(ReconstructionOperator::ReconstructionSpace reconstruction_space,
    		ReconstructionOperator::SamplingFunctional dual_sampling_strategy,
    		const int poly_order,
    		const std::string dense_solver_type = std::string("QR"),
    		const int manifold_poly_order = 2,
    		const int dimensions = 3)
    			: GMLS(reconstruction_space, dual_sampling_strategy, dual_sampling_strategy, poly_order, dense_solver_type, manifold_poly_order, dimensions) {}

    ~GMLS(){
    };


    template<typename view_type_1, typename view_type_2, typename view_type_3, typename view_type_4>
    void setProblemData(
    		view_type_1 neighbor_lists,
    		view_type_2 source_coordinates,
    		view_type_3 target_coordinates,
    		view_type_4 epsilons) {
		this->setNeighborLists<view_type_1>(neighbor_lists);
		this->setSourceSites<view_type_2>(source_coordinates);
		this->setTargetSites<view_type_3>(target_coordinates);
		this->setWindowSizes<view_type_4>(epsilons);
    }

    template <typename view_type>
    void setNeighborLists(view_type neighbor_lists) {
    // Catches Kokkos::View<int**, Kokkos::DefaultHostExecutionSpace
	// allocate memory on device
	_neighbor_lists = Kokkos::View<int**, layout_type>("device neighbor lists",
		neighbor_lists.dimension_0(), neighbor_lists.dimension_1());

	_host_neighbor_lists = Kokkos::create_mirror_view(_neighbor_lists);
	Kokkos::deep_copy(_host_neighbor_lists, neighbor_lists);
	// copy data from host to device
	Kokkos::deep_copy(_neighbor_lists, _host_neighbor_lists);

	_number_of_neighbors_list = Kokkos::View<int*, Kokkos::HostSpace>("number of neighbors", neighbor_lists.dimension_0());
	for (int i=0; i<_neighbor_lists.dimension_0(); ++i) {
		_number_of_neighbors_list(i) = neighbor_lists(i,0);
	}
    }

    template <typename view_type>
    void setNeighborLists(Kokkos::View<int**, Kokkos::DefaultExecutionSpace> neighbor_lists) {
    	// allocate memory on device
    	_neighbor_lists = neighbor_lists;
    	Kokkos::View<int**>::HostMirror host_neighbor_lists = Kokkos::create_mirror_view(_neighbor_lists);
    	Kokkos::deep_copy(host_neighbor_lists, _neighbor_lists);

    	_number_of_neighbors_list = Kokkos::View<int*, Kokkos::HostSpace>("number of neighbors", neighbor_lists.dimension_0());
    	for (int i=0; i<_neighbor_lists.dimension_0(); ++i) {
    		_number_of_neighbors_list(i) = host_neighbor_lists(i,0);
    	}
    }

    template<typename view_type>
    void setSourceSites(view_type source_coordinates) {

		// allocate memory on device
		_source_coordinates = Kokkos::View<double**, layout_type>("device neighbor coordinates",
				source_coordinates.dimension_0(), source_coordinates.dimension_1());

		_host_source_coordinates = Kokkos::create_mirror_view(_source_coordinates);
		Kokkos::deep_copy(_host_source_coordinates, source_coordinates);
		// copy data from host to device
		Kokkos::deep_copy(_source_coordinates, _host_source_coordinates);

    }

    template<typename view_type>
	void setSourceSites(Kokkos::View<double**, Kokkos::DefaultExecutionSpace> source_coordinates) {
		// allocate memory on device
		_source_coordinates = source_coordinates;
	}

    template<typename view_type>
    void setTargetSites(view_type target_coordinates) {
		// allocate memory on device
		_target_coordinates = Kokkos::View<double**, layout_type>("device target coordinates",
				target_coordinates.dimension_0(), target_coordinates.dimension_1());

		_host_target_coordinates = Kokkos::create_mirror_view(_target_coordinates);
		Kokkos::deep_copy(_host_target_coordinates, target_coordinates);
		// copy data from host to device
		Kokkos::deep_copy(_target_coordinates, _host_target_coordinates);
    }

    template<typename view_type>
    void setTargetSites(Kokkos::View<double**, Kokkos::DefaultExecutionSpace> target_coordinates) {
    	// allocate memory on device
    	_target_coordinates = target_coordinates;
    }

    template<typename view_type>
    void setWindowSizes(view_type epsilons) {

    	// allocate memory on device
		_epsilons = Kokkos::View<double*>("device epsilons",
						epsilons.dimension_0(), epsilons.dimension_1());

		_host_epsilons = Kokkos::create_mirror_view(_epsilons);
		Kokkos::deep_copy(_host_epsilons, epsilons);
		// copy data from host to device
		Kokkos::deep_copy(_epsilons, _host_epsilons);
    }

    template<typename view_type>
    void setWindowSizes(Kokkos::View<double*, Kokkos::DefaultExecutionSpace> epsilons) {
    	// allocate memory on device
    	_epsilons = epsilons;
    }

    void setOperatorCoefficients(Kokkos::View<double**, Kokkos::HostSpace> operator_coefficients) {
    	// allocate memory on device
		_operator_coefficients = Kokkos::View<double**, layout_type>("device operator coefficients",
				operator_coefficients.dimension_0(), operator_coefficients.dimension_1());
		// copy data from host to device
		Kokkos::deep_copy(_operator_coefficients, operator_coefficients);
    }

    void setPolynomialOrder(const int poly_order) {
    	_poly_order = poly_order;
    	_NP = this->getNP(_poly_order);
    }

    void setManifoldPolynomialOrder(const int manifold_poly_order) {
    	_manifold_poly_order = manifold_poly_order;
    }

    void setSolverType(const std::string& dense_solver_type) {
		std::string solver_type_to_lower = dense_solver_type;
		transform(solver_type_to_lower.begin(), solver_type_to_lower.end(), solver_type_to_lower.begin(), ::tolower);
		if (solver_type_to_lower == "lu") {
			_dense_solver_type = ReconstructionOperator::DenseSolverType::LU;
		} else if (solver_type_to_lower == "svd") {
			_dense_solver_type = ReconstructionOperator::DenseSolverType::SVD;
		} else if (solver_type_to_lower == "manifold") {
			_dense_solver_type = ReconstructionOperator::DenseSolverType::MANIFOLD;
			_manifold_support_operations = Kokkos::View<ReconstructionOperator::TargetOperation*>("operations needed for manifold gradient reconstruction", 1);
			_manifold_support_operations[0] = ReconstructionOperator::TargetOperation::GradientOfScalarPointEvaluation;
		} else {
			_dense_solver_type = ReconstructionOperator::DenseSolverType::QR;
		}
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const member_type& teamMember) const;

    KOKKOS_INLINE_FUNCTION
    static int getNP(const int m, const int dimension = 3) {
    	if (dimension == 3) return (m+1)*(m+2)*(m+3)/6;
    	else if (dimension == 2) return (m+1)*(m+2)/2;
    	else return m+1;
    }

    void addTargets(ReconstructionOperator::TargetOperation lro, const int dimension = 3) {
    	std::vector<ReconstructionOperator::TargetOperation> temporary_lro_vector(1, lro);
    	this->addTargets(temporary_lro_vector, dimension);
    }

    void addTargets(std::vector<ReconstructionOperator::TargetOperation> lro, const int dimension = 3) {
    	// if called multiple times with different dimensions, only the last
    	// dimension called with is used for all

    	// loop over requested targets
	    for (int i=0; i<lro.size(); ++i) {

	    	bool operation_found = false;
	    	// loop over existing targets registered
	    	for (int j=0; j<_lro.size(); ++j) {

	    		// if found
	    		if (_lro[j]==lro[i]) {

	    			operation_found = true;

	    			// the operation should now point to where the operation is stored
	    			_lro_lookup[(int)lro[i]] = j;

	    			break;

	    		}
	    	}

	    	if (!operation_found) {
				_lro_lookup[(int)lro[i]] = _lro.size();
				_lro.push_back(lro[i]);
	    	}
	    }

		_lro_total_offsets = Kokkos::View<int*, layout_type>("total offsets for alphas", _lro.size());
		_lro_output_tile_size = Kokkos::View<int*, layout_type>("output tile size for each operation", _lro.size());
		_lro_input_tile_size = Kokkos::View<int*, layout_type>("output tile size for each operation", _lro.size());
		_lro_output_tensor_rank = Kokkos::View<int*, layout_type>("output tensor rank", _lro.size());
		_lro_input_tensor_rank = Kokkos::View<int*, layout_type>("input tensor rank", _lro.size());

	    _host_lro_total_offsets = create_mirror(_lro_total_offsets);
	    _host_lro_output_tile_size = create_mirror(_lro_output_tile_size);
	    _host_lro_input_tile_size = create_mirror(_lro_input_tile_size);
	    _host_lro_output_tensor_rank = create_mirror(_lro_output_tensor_rank);
	    _host_lro_input_tensor_rank = create_mirror(_lro_input_tensor_rank);

	    int total_offset = 0; // need total offset
	    int output_offset = 0;
	    int input_offset = 0;
	    _max_target_tile_size = 0;
	    for (int i=0; i<_lro.size(); ++i) {
	    	_host_lro_total_offsets(i) = total_offset;

	    	// allows for a tile of the product of dimension^input_tensor_rank * dimension^output_tensor_rank * the number of neighbors
	    	int output_tile_size = std::pow(dimension, ReconstructionOperator::TargetOutputTensorRank[(int)_lro[i]]);
	    	int input_tile_size = std::pow(dimension, ReconstructionOperator::TargetInputTensorRank[(int)_lro[i]]);
	    	_max_target_tile_size = std::max(_max_target_tile_size, output_tile_size);
	    	_host_lro_output_tile_size(i) = output_tile_size;
	    	_host_lro_input_tile_size(i) = input_tile_size;

	    	total_offset += input_tile_size * output_tile_size;
	    	output_offset += output_tile_size;
	    	input_offset += input_tile_size;

	    	_host_lro_input_tensor_rank(i) = ReconstructionOperator::TargetInputTensorRank[(int)_lro[i]];
	    	_host_lro_output_tensor_rank(i) = ReconstructionOperator::TargetOutputTensorRank[(int)_lro[i]];
	    }

	    _total_alpha_values = total_offset;
	    _total_output_values = output_offset;

	    Kokkos::deep_copy(_lro_total_offsets, _host_lro_total_offsets);
	    Kokkos::deep_copy(_lro_output_tile_size, _host_lro_output_tile_size);
	    Kokkos::deep_copy(_lro_input_tile_size, _host_lro_input_tile_size);
	    Kokkos::deep_copy(_lro_output_tensor_rank, _host_lro_output_tensor_rank);
	    Kokkos::deep_copy(_lro_input_tensor_rank, _host_lro_input_tensor_rank);

    }

    void clearTargets() {
    	_lro.clear();
    	for (int i=0; i<ReconstructionOperator::TargetOperation::COUNT; ++i) {
    		_lro_lookup[i] = -1;
    	}
    }

    int getOutputDimensionOfOperation(ReconstructionOperator::TargetOperation lro) {
    	return this->_lro_output_tile_size[_lro_lookup[(int)lro]];
    }

    int getInputDimensionOfOperation(ReconstructionOperator::TargetOperation lro) {
    	return this->_lro_input_tile_size[_lro_lookup[(int)lro]];
    }

    int getOutputDimensionOfSampling(ReconstructionOperator::SamplingFunctional sro) {
    	return std::pow(_dimensions, ReconstructionOperator::SamplingOutputTensorRank[(int)sro]);
    }

    int getInputDimensionOfSampling(ReconstructionOperator::SamplingFunctional sro) {
    	return std::pow(_dimensions, ReconstructionOperator::SamplingInputTensorRank[(int)sro]);
    }

    int getOutputRankOfSampling(ReconstructionOperator::SamplingFunctional sro) {
    	return ReconstructionOperator::SamplingOutputTensorRank[(int)sro];
    }

    int getInputRankOfSampling(ReconstructionOperator::SamplingFunctional sro) {
    	return ReconstructionOperator::SamplingInputTensorRank[(int)sro];
    }

    void generateAlphas();

    double getAlpha0TensorTo0Tensor(ReconstructionOperator::TargetOperation lro, const int target_index, const int neighbor_index) const {
    	// e.g. Dirac Delta target of a scalar field
    	return getAlpha(lro, target_index, 0, 0, neighbor_index, 0, 0);
    }

    double getAlpha0TensorTo1Tensor(ReconstructionOperator::TargetOperation lro, const int target_index, const int output_component, const int neighbor_index) const {
    	// e.g. gradient of a scalar field
    	return getAlpha(lro, target_index, output_component, 0, neighbor_index, 0, 0);
    }

    double getAlpha0TensorTo2Tensor(ReconstructionOperator::TargetOperation lro, const int target_index, const int output_component_axis_1, const int output_component_axis_2, const int neighbor_index) const {
    	return getAlpha(lro, target_index, output_component_axis_1, output_component_axis_2, neighbor_index, 0, 0);
    }

    double getAlpha1TensorTo0Tensor(ReconstructionOperator::TargetOperation lro, const int target_index, const int neighbor_index, const int input_component) const {
    	// e.g. divergence of a vector field
    	return getAlpha(lro, target_index, 0, 0, neighbor_index, input_component, 0);
    }

    double getAlpha1TensorTo1Tensor(ReconstructionOperator::TargetOperation lro, const int target_index, const int output_component, const int neighbor_index, const int input_component) const {
    	// e.g. curl of a vector field
    	return getAlpha(lro, target_index, output_component, 0, neighbor_index, input_component, 0);
    }

    double getAlpha1TensorTo2Tensor(ReconstructionOperator::TargetOperation lro, const int target_index, const int output_component_axis_1, const int output_component_axis_2, const int neighbor_index, const int input_component) const {
    	// e.g. gradient of a vector field
    	return getAlpha(lro, target_index, output_component_axis_1, output_component_axis_2, neighbor_index, input_component, 0);
    }

    double getAlpha2TensorTo0Tensor(ReconstructionOperator::TargetOperation lro, const int target_index, const int neighbor_index, const int input_component_axis_1, const int input_component_axis_2) const {
    	return getAlpha(lro, target_index, 0, 0, neighbor_index, input_component_axis_1, input_component_axis_2);
    }

    double getAlpha2TensorTo1Tensor(ReconstructionOperator::TargetOperation lro, const int target_index, const int output_component, const int neighbor_index, const int input_component_axis_1, const int input_component_axis_2) const {
    	return getAlpha(lro, target_index, output_component, 0, neighbor_index, input_component_axis_1, input_component_axis_2);
    }

    double getAlpha2TensorTo2Tensor(ReconstructionOperator::TargetOperation lro, const int target_index, const int output_component_axis_1, const int output_component_axis_2, const int neighbor_index, const int input_component_axis_1, const int input_component_axis_2) const {
    	return getAlpha(lro, target_index, output_component_axis_1, output_component_axis_2, neighbor_index, input_component_axis_1, input_component_axis_2);
    }

    double getAlpha(ReconstructionOperator::TargetOperation lro, const int target_index, const int output_component_axis_1, const int output_component_axis_2, const int neighbor_index, const int input_component_axis_1, const int input_component_axis_2) const {
		// lro - the operator from ReconstructionOperator::TargetOperations
		// target_index - the # for the target site where information is required
		// neighbor_index - the # for the neighbor of the target
    	//
    	// This code support up to rank 2 tensors for inputs and outputs
		//
		// scalar reconstruction from scalar data: rank 0 to rank 0
		//   provides 1 piece of information for each neighbor
		// scalar reconstruction from vector data (e.g. divergence): rank 1 to rank 0
		//   provides 'd' pieces of information for each neighbor
		// vector reconstruction from scalar data (e.g. gradient): rank 0 to rank 1
		//   provides 'd' piece of information for each neighbor
		// vector reconstruction from vector data (e.g. curl): rank 1 to rank 1
		//   provides 'd'x'd' pieces of information for each neighbor
		//
    	// This function would more reasonably be called from one of the getAlphaNTensorFromNTensor
    	// which is much easier to understand with respect to indexing and only requesting indices
    	// that are relavent to the operator in question.
		//

    	const int lro_number = _lro_lookup[(int)lro];
    	const int input_index = ReconstructionOperator::getTargetInputIndex((int)lro, input_component_axis_1, input_component_axis_2);
    	const int output_index = ReconstructionOperator::getTargetOutputIndex((int)lro, output_component_axis_1, output_component_axis_2);

    	return _host_alphas(target_index,
    			(_host_lro_total_offsets[lro_number] + input_index*_host_lro_output_tile_size[lro_number] + output_index)*_number_of_neighbors_list(target_index)
						+ neighbor_index);
    }

    double getPreStencilWeight(ReconstructionOperator::SamplingFunctional sro, const int target_index, const int neighbor_index, bool for_target, const int output_component = 0, const int input_component = 0) const {
    	// for certain sampling strategies, linear combinations of the neighbor and target value are needed
    	// for the traditional PointSample, this value is 1 for the neighbor and 0 for the target
    	if (sro == ReconstructionOperator::SamplingFunctional::PointSample ) {
    		if (for_target) return 0; else return 1;
    	}
    	// 2 is because there is one value for each neighbor and one value for the target, for each target
    	return _host_prestencil_weights(target_index, output_component*_dimensions*2*(_neighbor_lists.dimension_1()-1) + input_component*2*(_neighbor_lists.dimension_1()-1) + 2*neighbor_index + (int)for_target);
    }

    ReconstructionOperator::WeightingFunctionType getWeightingType() const { return _weighting_type; }
    void setWeightingType( const std::string &wt) {
    	if (wt == "power") {
    		_weighting_type = ReconstructionOperator::WeightingFunctionType::Power;
    	} else {
    		_weighting_type = ReconstructionOperator::WeightingFunctionType::Gaussian;
    	}
    }

    ReconstructionOperator::WeightingFunctionType getManifoldWeightingType() const { return _manifold_weighting_type; }
    void setManifoldWeightingType( const std::string &wt) {
    	if (wt == "power") {
    		_manifold_weighting_type = ReconstructionOperator::WeightingFunctionType::Power;
    	} else {
    		_manifold_weighting_type = ReconstructionOperator::WeightingFunctionType::Gaussian;
    	}
    }

    int getWeightingPower() const { return _weighting_power; }
    void setWeightingPower(int wp) { _weighting_power = wp; }

    int getManifoldWeightingPower() const { return _manifold_weighting_power; }
    void setManifoldWeightingPower(int wp) { _manifold_weighting_power = wp; }

    int getDimensions() const { return _dimensions; }

    int getNumberOfQuadraturePoints() const { return _number_of_quadrature_points; }
    void setNumberOfQuadraturePoints(int np) { _number_of_quadrature_points = np; }

};

#endif

#endif


