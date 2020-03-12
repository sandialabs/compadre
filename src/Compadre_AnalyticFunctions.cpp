#include "Compadre_AnalyticFunctions.hpp"
#include <cmath>
//#include "Compadre_XyzVector.hpp"
//#include "Compadre_CoordsT.hpp"

namespace Compadre {

typedef XyzVector xyz_type;

scalar_type AnalyticFunction::evalScalar(const xyz_type& xyzIn) const { TEUCHOS_ASSERT(false); return 0.0; }
xyz_type AnalyticFunction::evalVector(const xyz_type& xyzIn) const { TEUCHOS_ASSERT(false); return 0.0; }
xyz_type AnalyticFunction::evalScalarDerivative(const xyz_type& xyzIn) const { TEUCHOS_ASSERT(false); return 0.0; }
std::vector<xyz_type> AnalyticFunction::evalJacobian(const xyz_type& xyzIn) const { TEUCHOS_ASSERT(false); return std::vector<xyz_type>(); }
scalar_type AnalyticFunction::evalScalarLaplacian(const xyz_type& xyzIn) const { TEUCHOS_ASSERT(false); return 0.0; }
xyz_type AnalyticFunction::evalVectorLaplacian(const xyz_type& xyzIn) const{ TEUCHOS_ASSERT(false); return 0.0; }

//KOKKOS_INLINE_FUNCTION
//void evaluateScalar::operator() (int i) const {
//	const xyz_type xyz = coords->getLocalCoords(i);
//	vals(i,0) = fn->evalScalar(xyz);
//}
//
//
//KOKKOS_INLINE_FUNCTION
//void evaluateVector::operator() (int i) const {
//	const xyz_type xyz = coords->getLocalCoords(i);
//	const xyz_type vecval = fn->evalVector(xyz);
//	vals(i,0) = vecval.x;
//	vals(i,1) = vecval.y;
//	vals(i,2) = vecval.z;
//}


scalar_type Gaussian3D::evalScalar(const xyz_type& xyzIn) const {
	const scalar_type x0 = 1.0;
	const scalar_type y0 = 1.0;
	const scalar_type z0 = 1.0;
	return std::exp(-((xyzIn.x - x0) * (xyzIn.x - x0) + (xyzIn.y - y0) * (xyzIn.y - y0) +
						(xyzIn.z - z0) * (xyzIn.z - z0)));
};

xyz_type Gaussian3D::evalVector(const xyz_type& xyzIn) const {return xyz_type();}

scalar_type legendre54(const scalar_type& z) {
    return z * (  z * z - 1.0 ) * ( z * z - 1.0 );
}

scalar_type SphereHarmonic::evalScalar(const xyz_type& xyzIn) const {
    const scalar_type lon = xyzIn.longitude();
    return 30.0 * std::cos(4.0 * lon) * legendre54(xyzIn.z);
}

xyz_type SphereHarmonic::evalScalarDerivative(const xyz_type& xyzIn) const {
	const scalar_type lat = xyzIn.latitude(); // phi
	const scalar_type lon = xyzIn.longitude(); // lambda

	const scalar_type A = -4.0 * std::pow(std::cos(lat),3) * std::sin(4.0 * lon) * std::sin(lat);
	const scalar_type B = 0.5* std::cos(4.0 * lon) * std::pow(std::cos(lat),3) * ( 5 * std::cos(2.0 * lat) - 3.0 );

    return xyz_type( -A * std::sin(lon) - B * std::sin(lat) * std::cos(lon),
    					A * std::cos(lon) - B * std::sin(lat) * std::sin(lon),
						B * std::cos(lat) );
}

xyz_type SphereTestVelocity::evalVector(const xyz_type& xyzIn) const {
	const scalar_type lat = xyzIn.latitude(); // phi
	const scalar_type lon = xyzIn.longitude(); // lambda

	const scalar_type U = 0.5* std::cos(4.0 * lon) * std::pow(std::cos(lat),3) * ( 5 * std::cos(2.0 * lat) - 3.0 );
	const scalar_type V = 4.0 * std::pow(std::cos(lat),3) * std::sin(4.0 * lon) * std::sin(lat);

    return xyz_type( -U * std::sin(lon) - V * std::sin(lat) * std::cos(lon),
    					U * std::cos(lon) - V * std::sin(lat) * std::sin(lon),
						V * std::cos(lat) );
}

xyz_type SphereRigidRotationVelocity::evalVector(const xyz_type& xIn) const {
	return xyz_type(-xIn.y, xIn.x, 0.0);
}

double ShallowWaterTestCases::evalScalar(const xyz_type& xIn) const {
	double return_val = 0;
	if (_test_case == 2) {
		// gh
		const scalar_type lat = xIn.latitude(); // phi
		const scalar_type lon = xIn.longitude(); // lambda

		double term_2 = ( - std::cos(lon) * std::cos(lat) * std::sin(_alpha) + std::sin(lat) * std::cos(_alpha));
		return_val = _gh0 - (_R_earth*_Omega*_u0 + 0.5*_u0*_u0)*term_2*term_2;
	} else if (_test_case == 5) {

		double mnt_h = 0;

		// h_mountain
		scalar_type lat = xIn.latitude(); // phi
		scalar_type lon = xIn.longitude(); // lambda

		double term_2 = ( - std::cos(lon) * std::cos(lat) * std::sin(_alpha) + std::sin(lat) * std::cos(_alpha));

		const scalar_type hs0 = 2000; // meters

		lat -= 1./6.*_gc.Pi();
		lon -= 1.5*_gc.Pi();

		const scalar_type R = _gc.Pi()/9.0;
		scalar_type r = std::sqrt(lat*lat + lon*lon);
		r = (r > R) ? R : r;

//		// C0 causing ringing
		//mnt_h = hs0*(1-r/R);

		// C^{\infty}
		mnt_h = hs0*std::exp(-(2.8*r/R)*(2.8*r/R));


		if (_atmosphere_or_mountain == 0) { // atmosphere

			// h_atmosphere
			return_val = (_gh0 - 1./_g*(_R_earth*_Omega*_u0 + 0.5*_u0*_u0)*term_2*term_2) - mnt_h;

		} else { // mountain

			return_val = mnt_h;

		}
	}
	return return_val;
}

xyz_type ShallowWaterTestCases::evalVector(const xyz_type& xIn) const {
	// u, v, w
	const scalar_type lat = xIn.latitude(); // phi
	const scalar_type lon = xIn.longitude(); // lambda

	const scalar_type U = _u0*(std::cos(lat)*std::cos(_alpha) +  std::cos(lon)*std::sin(lat)*std::sin(_alpha));
	const scalar_type V = -_u0*std::sin(lon)*std::sin(_alpha);

    return xyz_type( -U * std::sin(lon) - V * std::sin(lat) * std::cos(lon),
    					U * std::cos(lon) - V * std::sin(lat) * std::sin(lon),
						V * std::cos(lat) );
}

double CoriolisForce::evalScalar(const xyz_type& xIn) const {
	// f
	const scalar_type lat = xIn.latitude(); // phi
	const scalar_type lon = xIn.longitude(); // lambda

	return 2.0*_Omega*( - std::cos(lon) * std::cos(lat) * std::sin(_alpha) + std::sin(lat) * std::cos(_alpha));
}

double DiscontinuousOnSphere::evalScalar(const xyz_type& xIn) const {
	const scalar_type lon = xIn.longitude(); // lambda
	const scalar_type eps = 0;

	return (lon > eps && lon < _gc.Pi()-eps) ? 0 : 1;
}

scalar_type FiveStripOnSphere::evalScalar(const xyz_type& xIn) const {
	xyz_type temp_xyz(xIn.x, xIn.y, 0);
	const scalar_type lon = temp_xyz.longitude(); // lambda
	return lon;
}

scalar_type FiveStripOnSphere::evalDiffusionCoefficient(const xyz_type& xIn) const {
	if (xIn.z < .4) { // 1st slice
		return 16;
	} else if (xIn.z < .8) { // 2nd slice
		return 6;
	} else if (xIn.z < 1.2) { // 3rd slice
		return 1;
	} else if (xIn.z < 1.6) { // 4th slice
		return 10;
	} else {
		return 2;
	}
}

xyz_type  FiveStripOnSphere::evalVector(const xyz_type& xIn) const {
	const scalar_type lon = xIn.longitude(); // lambda
	double kappa = 0;
	if (xIn.z < .4) { // 1st slice
		kappa = 16;
	} else if (xIn.z < .8) { // 2nd slice
		kappa = 6;
	} else if (xIn.z < 1.2) { // 3rd slice
		kappa = 1;
	} else if (xIn.z < 1.6) { // 4th slice
		kappa = 10;
	} else {
		kappa = 2;
	}

	double multiplier = kappa;
	return xyz_type(multiplier*-sin(lon),multiplier*cos(lon),0);
}

scalar_type CylinderSinLonCosZ::evalScalar(const xyz_type& xyzIn) const {
	return xyzIn.x*cos(xyzIn.z);
}

scalar_type CylinderSinLonCosZRHS::evalScalar(const xyz_type& xyzIn) const {
	const scalar_type s = xyzIn.z;
	const scalar_type denom = std::sqrt(xyzIn.x*xyzIn.x+xyzIn.y*xyzIn.y);
	// only negatived because drivers expect a negated version of the laplacian rhs
	return -(-cos(s)*xyzIn.x/denom - cos(s)*xyzIn.x*denom)/denom;
}


scalar_type SineProducts::evalScalar(const xyz_type& xyzIn) const {
    if (_dim==3) {
        return sin(0.5*xyzIn.x)*sin(0.5*xyzIn.y)*sin(0.5*xyzIn.z);
    } else {
        return sin(xyzIn.x)*sin(xyzIn.y);
    }
}

xyz_type SineProducts::evalScalarDerivative(const xyz_type& xyzIn) const {
    if (_dim==3) {
        return xyz_type(0.5*cos(0.5*xyzIn.x)*sin(0.5*xyzIn.y)*sin(0.5*xyzIn.z),
                        0.5*sin(0.5*xyzIn.x)*cos(0.5*xyzIn.y)*sin(0.5*xyzIn.z),
                        0.5*sin(0.5*xyzIn.x)*sin(0.5*xyzIn.y)*cos(0.5*xyzIn.z));
    } else {
        return xyz_type(cos(xyzIn.x)*sin(xyzIn.y),
                        sin(xyzIn.x)*cos(xyzIn.y));
    }
}

xyz_type SineProducts::evalVector(const xyz_type& xyzIn) const {
    if (_dim==3) {
        return xyz_type(evalScalar(xyzIn),-evalScalar(xyzIn),2*evalScalar(xyzIn));
    } else
      {
        return xyz_type(evalScalar(xyzIn),-evalScalar(xyzIn),0);
    }
}

scalar_type SineProducts::evalScalarLaplacian(const xyz_type& xyzIn) const {
    if (_dim==3) {
        return -0.75*sin(0.5*xyzIn.x)*sin(0.5*xyzIn.y)*sin(0.5*xyzIn.z);
    } else {
        return -2*sin(xyzIn.x)*sin(xyzIn.y);
    }
}

scalar_type SineProducts::evalAdvectionDiffusionRHS(const xyz_type& xyzIn, const scalar_type diffusion, 
        const xyz_type& advection_field) const {
    if (_dim==3) {
        double grad[3];
        grad[0] = cos(xyzIn.x)*sin(xyzIn.y)*sin(xyzIn.z);
        grad[1] = cos(xyzIn.y)*sin(xyzIn.x)*sin(xyzIn.z);
        grad[2] = cos(xyzIn.z)*sin(xyzIn.x)*sin(xyzIn.y);
        return -diffusion*3*sin(xyzIn.x)*sin(xyzIn.y)*sin(xyzIn.z) + 
            (advection_field.x*grad[0] + advection_field.y*grad[1] + advection_field.z*grad[2]);
    } else {
        double grad[2];
        grad[0] = cos(xyzIn.x)*sin(xyzIn.y);
        grad[1] = cos(xyzIn.y)*sin(xyzIn.x);
        return -diffusion*2*sin(xyzIn.x)*sin(xyzIn.y) + (advection_field.x*grad[0] + advection_field.y*grad[1]);
    }
}

scalar_type SecondOrderBasis::evalScalar(const xyz_type& xyzIn) const {
    if (_dim==2) {
        return xyzIn.x*(1 + xyzIn.x + xyzIn.y) + xyzIn.y*(1 + xyzIn.y);
    } else {
        return xyzIn.x*(1 + xyzIn.x + xyzIn.y + xyzIn.z) + xyzIn.y*(1 + xyzIn.y + xyzIn.z) + xyzIn.z*(1 + xyzIn.z);
    }
}

xyz_type SecondOrderBasis::evalScalarDerivative(const xyz_type& xyzIn) const {
    if (_dim==2) {
        return xyz_type(2.0*xyzIn.x + xyzIn.y + 1.0,
                        xyzIn.x + 2.0*xyzIn.y + 1.0);
    } else {
        return xyz_type(2.0*xyzIn.x + xyzIn.y + xyzIn.z + 1.0,
                        xyzIn.x + 2.0*xyzIn.y + xyzIn.z + 1.0,
                        xyzIn.x + xyzIn.y + 2.0*xyzIn.z + 1.0);
    }
}

xyz_type SecondOrderBasis::evalVector(const xyz_type& xyzIn) const {
    return xyz_type(evalScalar(xyzIn),-evalScalar(xyzIn),2*evalScalar(xyzIn));
}

scalar_type SecondOrderBasis::evalScalarLaplacian(const xyz_type& xyzIn) const {
	return 6;
}

scalar_type ConstantEachDimension::evalScalarLaplacian(const xyz_type& xyzIn) const {
	return 0;
}

scalar_type ConstantEachDimension::evalScalar(const xyz_type& xyzIn) const {
    return _scaling_factors[0];
}

xyz_type ConstantEachDimension::evalVector(const xyz_type& xyzIn) const {
    return xyz_type(_scaling_factors[0],_scaling_factors[1],_scaling_factors[2]);
}

xyz_type ScaleOfEachDimension::evalVector(const xyz_type& xyzIn) const {
    return xyz_type(_scaling_factors[0]*xyzIn.x,_scaling_factors[1]*xyzIn.y,_scaling_factors[2]*xyzIn.z);
}

xyz_type CangaSphereTransform::evalVector(const xyz_type& latLonIn) const {
	const double lat = (_in_degrees) ? latLonIn.x * _gc.Pi() / 180.0 : latLonIn.x;
	const double lon = (_in_degrees) ? latLonIn.y * _gc.Pi() / 180.0 : latLonIn.y;
    return xyz_type(cos(lon)*cos(lat), sin(lon)*cos(lat), sin(lat));
}

xyz_type CurlCurlSineTestRHS::evalVector(const xyz_type& xyzIn) const {
    return xyz_type(0.5*sin(0.5*xyzIn.y)*sin(0.5*xyzIn.z),
                    0.5*sin(0.5*xyzIn.x)*sin(0.5*xyzIn.z),
                    0.5*sin(0.5*xyzIn.x)*sin(0.5*xyzIn.y));
}

xyz_type CurlCurlSineTest::evalVector(const xyz_type& xyzIn) const {
    return xyz_type(sin(0.5*xyzIn.y)*sin(0.5*xyzIn.z),
                    sin(0.5*xyzIn.x)*sin(0.5*xyzIn.z),
                    sin(0.5*xyzIn.x)*sin(0.5*xyzIn.y));
}

xyz_type CurlCurlPolyTestRHS::evalVector(const xyz_type& xyzIn) const {
    return xyz_type(-6.0,
                    -6.0,
                    -6.0);
}

xyz_type CurlCurlPolyTest::evalVector(const xyz_type& xyzIn) const {
    return xyz_type(2.0*xyzIn.x + xyzIn.y + xyzIn.z + xyzIn.x*xyzIn.x + xyzIn.y*xyzIn.y + xyzIn.z*xyzIn.z - xyzIn.x*xyzIn.y + xyzIn.y*xyzIn.z - 2.0*xyzIn.x*xyzIn.z + 1.0,
                    xyzIn.x - xyzIn.y + xyzIn.z + xyzIn.x*xyzIn.x + xyzIn.y*xyzIn.y + xyzIn.z*xyzIn.z - xyzIn.x*xyzIn.y + xyzIn.x*xyzIn.z + 1.0,
                    xyzIn.x + xyzIn.y - xyzIn.z + xyzIn.x*xyzIn.x + xyzIn.y*xyzIn.y + xyzIn.z*xyzIn.z + xyzIn.x*xyzIn.y - xyzIn.y*xyzIn.z - xyzIn.x*xyzIn.z + 1.0);
}

}
