#ifndef _GMLS_OPERATORS_HPP_
#define _GMLS_OPERATORS_HPP_

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

#endif
