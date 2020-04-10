#ifndef _COMPADRE_ANALYTIC_FUNCTIONS_HPP_
#define _COMPADRE_ANALYTIC_FUNCTIONS_HPP_

#include "CompadreHarness_Config.h"
#include "CompadreHarness_Typedefs.hpp"
#include "Compadre_XyzVector.hpp"
#include "Compadre_CoordsT.hpp"
#include "Compadre_GlobalConstants.hpp"

namespace Compadre {

// NOTE, operator() functions could be in cpp if KOKKOS_INLINE_FUNCTION permitted it
// but it doesn't, so we leave it in here. This requires including XyzVector and CoordsT in the hpp

class AnalyticFunction {

	protected : 

		typedef XyzVector xyz_type;

	    const local_index_type _dim;
	
	public :

		AnalyticFunction(const local_index_type dim = 3) : _dim(dim) {}

		virtual ~AnalyticFunction() {}

        // these are defined by deriving symbolically
		virtual scalar_type evalScalar(const xyz_type& xyzIn, const local_index_type input_comp = 0) const;
		virtual xyz_type evalScalarDerivative(const xyz_type& xyzIn, const local_index_type input_comp = 0) const;
		virtual std::vector<xyz_type> evalScalarHessian(const xyz_type& xyzIn, const local_index_type input_comp = 0) const;

        // function of evalScalar
		virtual xyz_type evalVector(const xyz_type& xyzIn) const;

        // function of evalScalarDerivative
		virtual std::vector<xyz_type> evalJacobian(const xyz_type& xyzIn) const;

        // function of evalHessian
		virtual scalar_type evalScalarLaplacian(const xyz_type& xyzIn, const local_index_type input_comp = 0) const;
		virtual xyz_type evalVectorLaplacian(const xyz_type& xyzIn) const;
        virtual scalar_type evalScalarReactionDiffusionRHS(const xyz_type& xyzIn, const scalar_type reaction, const scalar_type diffusion) const;
        virtual scalar_type evalLinearElasticityRHS(const xyz_type& xyzIn, const local_index_type output_comp, const scalar_type shear_modulus, 
                                                    const scalar_type lambda) const;

};


struct evaluateScalar {
	typedef AnalyticFunction function_type;
	typedef CoordsT coords_type;
	typedef XyzVector xyz_type;
	
	device_view_type vals;
	function_type* fn;
	device_view_type coords;
	
	evaluateScalar(device_view_type vals_, device_view_type coords_, function_type* fn_) :
		vals(vals_), fn(fn_), coords(coords_) {};

	void operator() (int i) const {
		xyz_type xyz;
		xyz.x = coords(i,0);
		xyz.y = coords(i,1);
		xyz.z = coords(i,2);
		vals(i,0) = fn->evalScalar(xyz);
	}
};


struct evaluateVector {
	typedef AnalyticFunction function_type;
	typedef CoordsT coords_type;
	typedef XyzVector xyz_type;
	
	device_view_type vals;
	function_type* fn;
	device_view_type coords;
	const int operation_type;
	
	evaluateVector(device_view_type vals_, device_view_type coords_, function_type* fn_, const int operation_type_ = 0) :
		vals(vals_), fn(fn_), coords(coords_), operation_type(operation_type_) {};

	void operator() (int i) const {
		xyz_type xyz;
		xyz.x = coords(i,0);
		xyz.y = coords(i,1);
		xyz.z = coords(i,2);
		xyz_type vecval;
		switch (operation_type) {
		case 1:
			vecval = fn->evalScalarDerivative(xyz);
			break;
		default:
			vecval = fn->evalVector(xyz);
		}
		switch (vals.extent(1)) {
		case 3:
			vals(i,2) = vecval.z;
		case 2:
			vals(i,1) = vecval.y;
		case 1:
			vals(i,0) = vecval.x;
		default:
			break;
		}
	}
};

struct incrementByEvaluateVector {
	typedef AnalyticFunction function_type;
	typedef CoordsT coords_type;
	typedef XyzVector xyz_type;

	device_view_type vals;
	function_type* fn;
	device_view_type coords;

	incrementByEvaluateVector(device_view_type vals_, device_view_type coords_, function_type* fn_) :
		vals(vals_), fn(fn_), coords(coords_) {};

	void operator() (int i) const {
		xyz_type xyz;
		xyz.x = coords(i,0);
		xyz.y = coords(i,1);
		xyz.z = coords(i,2);
		const xyz_type vecval = fn->evalVector(xyz);
		vals(i,0) += vecval.x;
		vals(i,1) += vecval.y;
		vals(i,2) += vecval.z;
	}
};

class Gaussian3D : public AnalyticFunction {

	typedef XyzVector xyz_type;

	public:

		virtual scalar_type evalScalar(const xyz_type& xyzIn, const local_index_type input_comp = 0) const;

};

class SphereHarmonic : public AnalyticFunction {

	typedef XyzVector xyz_type;

	protected:

		int _m;
		int _n;
    
	public:
		SphereHarmonic(const int legendreM, const int legendreN) : _m(legendreM), _n(legendreN) {};

		scalar_type evalScalar(const xyz_type& xyzIn, const local_index_type input_comp = 0) const;

		xyz_type evalScalarDerivative(const xyz_type& xyzIn, const local_index_type input_comp = 0) const;
};

class SphereTestVelocity : public AnalyticFunction {

	// only for use with SphereHarmonic(4,5)

	typedef XyzVector xyz_type;

	public:
		SphereTestVelocity() {};

		virtual xyz_type evalVector(const xyz_type& xyzIn) const;

};

class SphereRigidRotationVelocity : public AnalyticFunction {

	typedef XyzVector xyz_type;

	public :

		virtual xyz_type evalVector(const xyz_type& xIn) const;

};

class ShallowWaterTestCases : public AnalyticFunction {

	typedef XyzVector xyz_type;

	GlobalConstants _gc;
	local_index_type _test_case;
	double _alpha;
	double _gh0;
	double _u0;
	double _R_earth;
	double _Omega;
	double _g;
	local_index_type _atmosphere_or_mountain; // 0 for atmosphere, 1 for mountain

	public:
		ShallowWaterTestCases(local_index_type test_case, double alpha = 0): _test_case(test_case), _alpha(alpha) {
			if (test_case == 2) {
				_gh0 = 2.94e+4; //0.0046145008;// gh0 o 2.94e+4 / earth's radius
				_R_earth = _gc.EarthRadius(); //1.0;
				_Omega = 7.292e-5; // 2*_gc.Pi()/86164.1;//7.292e-5; // 2*_gc.Pi() / 86164.1; // s^-1
				_g = _gc.Gravity();
				_u0 = 2.0*_gc.Pi()*_R_earth / (12.0*86164.1);
			} else if (test_case == 5) {
				_gh0 = 5400;
				_R_earth = _gc.EarthRadius(); 
				_Omega = 7.292e-5;
				_g = _gc.Gravity();
				_u0 = 20.0;
			}
			_atmosphere_or_mountain = 0; // preset to atmosphere
		};

		// returns h
		virtual scalar_type evalScalar(const xyz_type& xIn, const local_index_type input_comp = 0) const;

		// returns u, v, and w
		virtual xyz_type evalVector(const xyz_type& xyzIn) const;

		double getOmega() const { return _Omega; }

		double getGravity() const { return _g; }

		double getRadius() const { return _R_earth; }

		void setAtmosphere() { _atmosphere_or_mountain = 0; }

		void setMountain() { _atmosphere_or_mountain = 1; }
};

class CoriolisForce : public AnalyticFunction {

	typedef XyzVector xyz_type;

	double _Omega;
	double _alpha;

	public:
		CoriolisForce(double Omega, double alpha = 0): _Omega(Omega), _alpha(alpha) {};

		// returns f
		virtual scalar_type evalScalar(const xyz_type& xIn, const local_index_type input_comp = 0) const;
};

class DiscontinuousOnSphere : public AnalyticFunction {

	typedef XyzVector xyz_type;

	GlobalConstants _gc;

	public:
		DiscontinuousOnSphere() {};

		// returns f
		virtual scalar_type evalScalar(const xyz_type& xIn, const local_index_type input_comp = 0) const;
};

class FiveStripOnSphere : public AnalyticFunction {

	typedef XyzVector xyz_type;

	GlobalConstants _gc;

	public:
		FiveStripOnSphere() {};

		// returns f
		virtual scalar_type evalScalar(const xyz_type& xIn, const local_index_type input_comp = 0) const;

		virtual xyz_type evalVector(const xyz_type& xIn) const;

		scalar_type evalDiffusionCoefficient(const xyz_type& xIn) const;
};

class CylinderSinLonCosZ : public AnalyticFunction {

	typedef XyzVector xyz_type;

	public :

		virtual scalar_type evalScalar(const xyz_type& xIn, const local_index_type input_comp = 0) const;
};

class CylinderSinLonCosZRHS : public AnalyticFunction {

	typedef XyzVector xyz_type;

	public :

		virtual scalar_type evalScalar(const xyz_type& xIn, const local_index_type input_comp = 0) const;
};

class DivFreeSineCos : public AnalyticFunction {

	typedef XyzVector xyz_type;

	public :

	    DivFreeSineCos(const local_index_type dim = 3) : AnalyticFunction(dim) {}

		virtual scalar_type evalScalar(const xyz_type& xIn, const local_index_type input_comp = 0) const;

		virtual xyz_type evalScalarDerivative(const xyz_type& xIn, const local_index_type input_comp = 0) const;

		std::vector<xyz_type> evalScalarHessian(const xyz_type& xyzIn, const local_index_type input_comp = 0) const;

};

class SineProducts : public AnalyticFunction {

	typedef XyzVector xyz_type;

	public :

	    SineProducts(const local_index_type dim = 3) : AnalyticFunction(dim) {}

		virtual scalar_type evalScalar(const xyz_type& xIn, const local_index_type input_comp = 0) const;

		virtual xyz_type evalScalarDerivative(const xyz_type& xIn, const local_index_type input_comp = 0) const;

		virtual std::vector<xyz_type> evalScalarHessian(const xyz_type& xyzIn, const local_index_type input_comp = 0) const;

		virtual scalar_type evalScalarLaplacian(const xyz_type& xyzIn, const local_index_type input_comp = 0) const;

};

class FirstOrderBasis : public AnalyticFunction {

	typedef XyzVector xyz_type;

	public :

	    FirstOrderBasis(const local_index_type dim = 3) : AnalyticFunction(dim) {}

		virtual scalar_type evalScalar(const xyz_type& xIn, const local_index_type input_comp = 0) const;

		virtual xyz_type evalScalarDerivative(const xyz_type& xyzIn, const local_index_type input_comp = 0) const;

		virtual std::vector<xyz_type> evalScalarHessian(const xyz_type& xyzIn, const local_index_type input_comp = 0) const;

        virtual scalar_type evalScalarLaplacian(const xyz_type& xyzIn, const local_index_type input_comp = 0) const;

};

class DivFreeSecondOrderBasis : public AnalyticFunction {

	typedef XyzVector xyz_type;

	public :

	    DivFreeSecondOrderBasis(const local_index_type dim = 3) : AnalyticFunction(dim) {}

		virtual scalar_type evalScalar(const xyz_type& xIn, const local_index_type input_comp = 0) const;

		virtual xyz_type evalScalarDerivative(const xyz_type& xyzIn, const local_index_type input_comp = 0) const;

		virtual std::vector<xyz_type> evalScalarHessian(const xyz_type& xyzIn, const local_index_type input_comp = 0) const;

        virtual scalar_type evalScalarLaplacian(const xyz_type& xyzIn, const local_index_type input_comp = 0) const;

};

class SecondOrderBasis : public AnalyticFunction {

	typedef XyzVector xyz_type;

	public :

	    SecondOrderBasis(const local_index_type dim = 3) : AnalyticFunction(dim) {}

		virtual scalar_type evalScalar(const xyz_type& xIn, const local_index_type input_comp = 0) const;

		virtual xyz_type evalScalarDerivative(const xyz_type& xyzIn, const local_index_type input_comp = 0) const;

		virtual std::vector<xyz_type> evalScalarHessian(const xyz_type& xyzIn, const local_index_type input_comp = 0) const;

        virtual scalar_type evalScalarLaplacian(const xyz_type& xyzIn, const local_index_type input_comp = 0) const;

};

class ThirdOrderBasis : public AnalyticFunction {

	typedef XyzVector xyz_type;

	public :

	    ThirdOrderBasis(const local_index_type dim = 3) : AnalyticFunction(dim) {}

		virtual scalar_type evalScalar(const xyz_type& xIn, const local_index_type input_comp = 0) const;

		virtual xyz_type evalScalarDerivative(const xyz_type& xyzIn, const local_index_type input_comp = 0) const;

		virtual std::vector<xyz_type> evalScalarHessian(const xyz_type& xyzIn, const local_index_type input_comp = 0) const;

		virtual scalar_type evalScalarLaplacian(const xyz_type& xyzIn, const local_index_type input_comp = 0) const;

};

class ConstantEachDimension : public AnalyticFunction {

	typedef XyzVector xyz_type;
	std::vector<scalar_type> _scaling_factors;

	public :
		ConstantEachDimension(std::vector<scalar_type> scaling_factors) {
			TEUCHOS_TEST_FOR_EXCEPT_MSG(scaling_factors.size()==3, "std::vector should be of length 3.");
			_scaling_factors = scaling_factors;
		}

		ConstantEachDimension(scalar_type x_scale, scalar_type y_scale, scalar_type z_scale) {
			_scaling_factors = std::vector<scalar_type>(3);
			_scaling_factors[0] = x_scale;
			_scaling_factors[1] = y_scale;
			_scaling_factors[2] = z_scale;
		}

		virtual scalar_type evalScalar(const xyz_type& xIn, const local_index_type input_comp = 0) const;

		virtual xyz_type evalScalarDerivative(const xyz_type& xIn, const local_index_type input_comp = 0) const;

		virtual scalar_type evalScalarLaplacian(const xyz_type& xIn, const local_index_type input_comp = 0) const;

};

class ScaleOfEachDimension : public AnalyticFunction {

	typedef XyzVector xyz_type;
	std::vector<scalar_type> _scaling_factors;

	public :
		ScaleOfEachDimension(std::vector<scalar_type> scaling_factors) {
			TEUCHOS_TEST_FOR_EXCEPT_MSG(scaling_factors.size()==3, "std::vector should be of length 3.");
			_scaling_factors = scaling_factors;
		}

		ScaleOfEachDimension(scalar_type x_scale, scalar_type y_scale, scalar_type z_scale) {
			_scaling_factors = std::vector<scalar_type>(3);
			_scaling_factors[0] = x_scale;
			_scaling_factors[1] = y_scale;
			_scaling_factors[2] = z_scale;
		}

		virtual xyz_type evalVector(const xyz_type& xIn) const;
};

class CangaSphereTransform : public AnalyticFunction {

	typedef XyzVector xyz_type;

	GlobalConstants _gc;
	bool _in_degrees;

	public :

		CangaSphereTransform(bool in_degrees = false) : _in_degrees(in_degrees) {}

		virtual xyz_type evalVector(const xyz_type& latLonIn) const;
};

class CurlCurlSineTestRHS : public AnalyticFunction {

	typedef XyzVector xyz_type;

	public :

		virtual xyz_type evalVector(const xyz_type& xyzIn) const;
};

class CurlCurlSineTest : public AnalyticFunction {

	typedef XyzVector xyz_type;

	public :

		virtual xyz_type evalVector(const xyz_type& xyzIn) const;
};

class CurlCurlPolyTestRHS : public AnalyticFunction {

	typedef XyzVector xyz_type;

	public :

		virtual xyz_type evalVector(const xyz_type& xyzIn) const;
};

class CurlCurlPolyTest : public AnalyticFunction {

	typedef XyzVector xyz_type;

	public :

		virtual xyz_type evalVector(const xyz_type& xyzIn) const;
};

}
#endif 
