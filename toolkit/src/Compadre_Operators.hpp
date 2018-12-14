#ifndef _COMPADRE_OPERATORS_HPP_
#define _COMPADRE_OPERATORS_HPP_

namespace Compadre {

    //! Available target functionals
    enum TargetOperation {
        //! Point evaluation of a scalar
        ScalarPointEvaluation,
        //! Point evaluation of a vector (reconstructs entire vector at once, requiring a 
        //! ReconstructionSpace having a sufficient number of components in the basis)
        VectorPointEvaluation, 
        //! Point evaluation of the laplacian of a scalar (could be on a manifold or not)
        LaplacianOfScalarPointEvaluation,
        //! Point evaluation of the laplacian of each component of a vector
        VectorLaplacianPointEvaluation,
        //! Point evaluation of the gradient of a scalar
        GradientOfScalarPointEvaluation,
        //! Point evaluation of the gradient of a vector (results in a matrix, NOT CURRENTLY IMPLEMENTED)
        GradientOfVectorPointEvaluation,
        //! Point evaluation of the divergence of a vector (results in a scalar)
        DivergenceOfVectorPointEvaluation,
        //! Point evaluation of the curl of a vector (results in a vector)
        CurlOfVectorPointEvaluation,
        //! Point evaluation of the partial with respect to x of a scalar
        PartialXOfScalarPointEvaluation,
        //! Point evaluation of the partial with respect to y of a scalar
        PartialYOfScalarPointEvaluation,
        //! Point evaluation of the partial with respect to z of a scalar
        PartialZOfScalarPointEvaluation,
        //! Point evaluation of the chained staggered Laplacian acting on VectorTaylorPolynomial 
        //! basis + StaggeredEdgeIntegralSample sampling functional
        ChainedStaggeredLaplacianOfScalarPointEvaluation,
        //! Should be the total count of all available target functionals
        COUNT=12,
    };

    //! Rank of target functional output for each TargetOperation 
    //! Rank of target functional input for each TargetOperation is based on the output
    //! rank of the SamplingFunctional used on the polynomial basis
    const int TargetOutputTensorRank[] {
        0, ///< PointEvaluation
        1, ///< VectorPointEvaluation
        0, ///< LaplacianOfScalarPointEvaluation
        1, ///< VectorLaplacianPointEvaluation
        1, ///< GradientOfScalarPointEvaluation
        2, ///< GradientOfVectorPointEvaluation
        0, ///< DivergenceOfVectorPointEvaluation
        1, ///< CurlOfVectorPointEvaluation
        0, ///< PartialXOfScalarPointEvaluation
        0, ///< PartialYOfScalarPointEvaluation
        0, ///< PartialZOfScalarPointEvaluation
        0, ///< ChainedStaggeredLaplacianOfScalarPointEvaluation
    };

    //! Space in which to reconstruct polynomial
    enum ReconstructionSpace {
        //! Scalar polynomial basis centered at the target site and scaled by sum of basis powers 
        //! e.g. \f$(x-x_t)^2*(y-y_t)*(z-z_t)^3/factorial(2+1+3)\f$ would be a member of 3rd order in 3D, where 
        //! \f$(x_t,y_t,z_t)\f$ is the coordinate of the target site in 3D coordinates.
        ScalarTaylorPolynomial,
        //! Vector polynomial basis having # of components _dimensions, or (_dimensions-1) in the case of manifolds)
        VectorTaylorPolynomial,
        //! Scalar basis reused as many times as there are components in the vector
        //! resulting in a much cheaper polynomial reconstruction
        VectorOfScalarClonesTaylorPolynomial,
    };

    //! Number of effective components in the ReconstructionSpace (behaves like a reconstruction of this rank)
    const int EffectiveReconstructionSpaceRank[] = {
        0, ///< ScalarTaylorPolynomial
        1, ///< VectorTaylorPolynomial
        1, ///< VectorOfScalarClonesTaylorPolynomial
    };

    //! Number of actual components in the ReconstructionSpace
    const int ActualReconstructionSpaceRank[] = {
        0, ///< ScalarTaylorPolynomial
        1, ///< VectorTaylorPolynomial
        0, ///< VectorOfScalarClonesTaylorPolynomial
    };

    //! Available sampling functionals
    enum SamplingFunctional {
        //! Point evaluations of the scalar source function
        PointSample,
        //! Point evaluations of the entire vector source function
        VectorPointSample,
        //! Point evaluations of the entire vector source function 
        //! (but on a manifold, so it includes a transform into local coordinates)
        ManifoldVectorPointSample,
        //! Analytical integral of a gradient source vector is just a difference of the scalar source at neighbor and target
        StaggeredEdgeAnalyticGradientIntegralSample,
        //! Samples consist of the result of integrals of a vector dotted with the tangent along edges between neighbor and target
        StaggeredEdgeIntegralSample,
    };

    //! Rank of sampling functional input for each SamplingFunctional
    const int SamplingInputTensorRank[] = {
        0, ///< PointSample
        1, ///< VectorPointSample
        1, ///< ManifoldVectorPointSample
        0, ///< StaggeredEdgeAnalyticGradientIntegralSample,
        1, ///< StaggeredEdgeIntegralSample
    };

    //! Rank of sampling functional output for each SamplingFunctional
    const int SamplingOutputTensorRank[] {
        0, ///< PointSample
        1, ///< VectorPointSample
        1, ///< ManifoldVectorPointSample
        0, ///< StaggeredEdgeAnalyticGradientIntegralSample,
        0, ///< StaggeredEdgeIntegralSample
    };

    //! Describes the SamplingFunction relationship to targets, neighbors
    enum SamplingTransformType {
        Identity,           ///< No action performed on data before GMLS target operation
        SameForAll,         ///< Each neighbor for each target all apply the same transform
        DifferentEachTarget,   ///< Each target applies a different data transform, but the same to each neighbor
        DifferentEachNeighbor, ///< Each target applies a different transform for each neighbor
    };

    //! Rank of sampling functional output for each SamplingFunctional
    const int SamplingTensorStyle[] {
        (int)Identity,              ///< PointSample
        (int)Identity,              ///< VectorPointSample
        (int)DifferentEachTarget,   ///< ManifoldVectorPointSample
        (int)SameForAll,            ///< StaggeredEdgeAnalyticGradientIntegralSample,
        (int)DifferentEachNeighbor, ///< StaggeredEdgeIntegralSample
    };

    //! Whether or not the SamplingTensor acts on the target site as well as the neighbors.
    //! This makes sense only in staggered schemes, when each target site is also a source site
    const int SamplingTensorForTargetSite[] {
        0, ///< PointSample
        0, ///< VectorPointSample
        0, ///< ManifoldVectorPointSample
        1, ///< StaggeredEdgeAnalyticGradientIntegralSample,
        1, ///< StaggeredEdgeIntegralSample
    };

    //! Whether the SamplingFunctional + ReconstructionSpace results in a nontrivial nullspace requiring SVD
    const int SamplingNontrivialNullspace[] {
        // does the sample over polynomials result in an operator
        // with a nontrivial nullspace requiring SVD
        0, ///< PointSample
        0, ///< VectorPointSample
        0, ///< ManifoldVectorPointSample
        1, ///< StaggeredEdgeAnalyticGradientIntegralSample,
        1, ///< StaggeredEdgeIntegralSample
    };

    //! Dense solver type, that optionally can also handle manifolds
    enum DenseSolverType {
        //! QR factorization performed on P*sqrt(w) matrix
        QR, 
        //! SVD factorization performed on P*sqrt(w) matrix
        SVD, 
        //! Solve GMLS problem on a manifold (will use QR or SVD to solve the resultant GMLS 
        //! problem dependent on SamplingNontrivialNullspace
        MANIFOLD, 
    };

    //! Available weighting kernel function types
    enum WeightingFunctionType {
        Power,
        Gaussian
    };

    //! Coordinate type for input and output format of vector data on manifold problems.
    //! Anything without a manifold is always Ambient.
    enum CoordinatesType {
        Ambient, ///< a 2D manifold in 3D in ambient coordinates would have 3 components for a vector
        Local,   ///< a 2D manifold in 3D in local coordinates would have 2 components for a vector
    };

    //! Helper function for finding alpha coefficients
    static int getTargetOutputIndex(const int operation_num, const int output_component_axis_1, const int output_component_axis_2) {
        const int axis_1_size = (TargetOutputTensorRank[operation_num] > 1) ? TargetOutputTensorRank[operation_num] : 1;
        return axis_1_size*output_component_axis_1 + output_component_axis_2; // 0 for scalar, 0 for vector;
    }

    //static int getSamplingInputIndex(const int operation_num, const int input_component_axis_1, const int input_component_axis_2) {
    //    const int axis_1_size = (SamplingInputTensorRank[operation_num] > 1) ? SamplingInputTensorRank[operation_num] : 1;
    //    return axis_1_size*input_component_axis_1 + input_component_axis_2; // 0 for scalar, 0 for vector;
    //}

    //! Helper function for finding alpha coefficients
    static int getSamplingOutputIndex(const int operation_num, const int output_component_axis_1, const int output_component_axis_2) {
        const int axis_1_size = (SamplingOutputTensorRank[operation_num] > 1) ? SamplingOutputTensorRank[operation_num] : 1;
        return axis_1_size*output_component_axis_1 + output_component_axis_2; // 0 for scalar, 0 for vector;
    }

    //static bool validTargetSpaceSample(TargetOperation to, ReconstructionSpace rs, SamplingFunctional sf) {
    //    // all valid combinations to be added here
    //    return true;
    //}

    /*
       \page pageGMLSOperators GMLS Target Operations, Spaces, and Sampling Functionals
       \section sectionGMLSTarget Target Operations

       \section sectionGMLSSpace Reconstruction Spaces

       \section sectionGMLSSampling Sampling Functionals 
    */


}; // Compadre

#endif
