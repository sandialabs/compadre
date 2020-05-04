#include "Compadre_AnalyticFunctions.hpp"
#include <cmath>

namespace Compadre {

typedef XyzVector xyz_type;

// must be defined for each function
scalar_type AnalyticFunction::evalScalar(const xyz_type& xyzIn, const local_index_type input_comp) const { 
    TEUCHOS_ASSERT(false); return 0.0; 
}

xyz_type AnalyticFunction::evalScalarDerivative(const xyz_type& xyzIn, const local_index_type input_comp) const { 
    TEUCHOS_ASSERT(false); return xyz_type();
}

std::vector<xyz_type> AnalyticFunction::evalScalarHessian(const xyz_type& xyzIn, const local_index_type input_comp) const { 
    TEUCHOS_ASSERT(false); std::vector<xyz_type>();
}

// products of the previous three functions

xyz_type AnalyticFunction::evalVector(const xyz_type& xyzIn) const { 
    xyz_type return_vec;
    for (local_index_type i=0; i<_dim; ++i) {
        return_vec[i] = this->evalScalar(xyzIn, i);
    }    
    return return_vec;
}

std::vector<xyz_type> AnalyticFunction::evalJacobian(const xyz_type& xyzIn) const { 
    std::vector<xyz_type> return_vec(_dim);
    for (local_index_type i=0; i<_dim; ++i) {
        return_vec[i] = this->evalScalarDerivative(xyzIn, i);
    }    
    return return_vec;
}

scalar_type AnalyticFunction::evalScalarLaplacian(const xyz_type& xyzIn, const local_index_type input_comp) const { 
    auto this_scalar_hessian = this->evalScalarHessian(xyzIn, input_comp);
    scalar_type return_val = 0.0;
    for (local_index_type i=0; i<_dim; ++i) {
        return_val += this_scalar_hessian[i][i];
    }    
    return return_val;
}

xyz_type AnalyticFunction::evalVectorLaplacian(const xyz_type& xyzIn) const{ 
    xyz_type return_vec;
    for (local_index_type i=0; i<_dim; ++i) {
        return_vec[i] = this->evalScalarLaplacian(xyzIn, i);
    }    
    return return_vec;
}

scalar_type AnalyticFunction::evalScalarReactionDiffusionRHS(const xyz_type& xyzIn, const scalar_type reaction, const scalar_type diffusion) const {
    auto laplace_u = this->evalScalarLaplacian(xyzIn, 0);
    return -diffusion*laplace_u + reaction*this->evalScalar(xyzIn, 0);
}

scalar_type AnalyticFunction::evalLinearElasticityRHS(const xyz_type& xyzIn, const local_index_type output_comp, const scalar_type shear_modulus, const scalar_type lambda) const {
    if (_dim==3) {
        TEUCHOS_ASSERT(false);
        return 0;
    } else {
        std::vector<std::vector<xyz_type> > vector_hessian(2, std::vector<xyz_type>());
        for (local_index_type i=0; i<_dim; ++i) {
            vector_hessian[i] = this->evalScalarHessian(xyzIn, i);
        }
        // - div(2*shear*D(u) + lambda* div(u))
        if (output_comp==0) {
            return -lambda*(vector_hessian[0][0][0] + vector_hessian[1][1][0]) 
                - 2*shear_modulus*(vector_hessian[0][0][0] + 0.5*(vector_hessian[1][0][1]+vector_hessian[0][1][1]));
        } else {
            return -lambda*(vector_hessian[1][1][1] + vector_hessian[0][0][1]) 
                - 2*shear_modulus*(vector_hessian[1][1][1] + 0.5*(vector_hessian[0][1][0]+vector_hessian[1][0][0]));
        }
    }
}

//
// Gaussian3D
//

scalar_type Gaussian3D::evalScalar(const xyz_type& xyzIn, const local_index_type input_comp) const {
	const scalar_type x0 = 1.0;
	const scalar_type y0 = 1.0;
	const scalar_type z0 = 1.0;
	return std::exp(-((xyzIn.x - x0) * (xyzIn.x - x0) + (xyzIn.y - y0) * (xyzIn.y - y0) +
						(xyzIn.z - z0) * (xyzIn.z - z0)));
};

//
// SphereHarmonic
//

scalar_type legendre54(const scalar_type& z) {
    return z * (  z * z - 1.0 ) * ( z * z - 1.0 );
}

scalar_type SphereHarmonic::evalScalar(const xyz_type& xyzIn, const local_index_type input_comp) const {
    const scalar_type lon = xyzIn.longitude();
    return 30.0 * std::cos(4.0 * lon) * legendre54(xyzIn.z);
}

xyz_type SphereHarmonic::evalScalarDerivative(const xyz_type& xyzIn, const local_index_type input_comp) const {
	const scalar_type lat = xyzIn.latitude(); // phi
	const scalar_type lon = xyzIn.longitude(); // lambda

	const scalar_type A = -4.0 * std::pow(std::cos(lat),3) * std::sin(4.0 * lon) * std::sin(lat);
	const scalar_type B = 0.5* std::cos(4.0 * lon) * std::pow(std::cos(lat),3) * ( 5 * std::cos(2.0 * lat) - 3.0 );

    return xyz_type( -A * std::sin(lon) - B * std::sin(lat) * std::cos(lon),
    					A * std::cos(lon) - B * std::sin(lat) * std::sin(lon),
						B * std::cos(lat) );
}

//
// SphereTestVelocity
//

xyz_type SphereTestVelocity::evalVector(const xyz_type& xyzIn) const {
	const scalar_type lat = xyzIn.latitude(); // phi
	const scalar_type lon = xyzIn.longitude(); // lambda

	const scalar_type U = 0.5* std::cos(4.0 * lon) * std::pow(std::cos(lat),3) * ( 5 * std::cos(2.0 * lat) - 3.0 );
	const scalar_type V = 4.0 * std::pow(std::cos(lat),3) * std::sin(4.0 * lon) * std::sin(lat);

    return xyz_type( -U * std::sin(lon) - V * std::sin(lat) * std::cos(lon),
    					U * std::cos(lon) - V * std::sin(lat) * std::sin(lon),
						V * std::cos(lat) );
}

//
// SphereRigidRotationVelocity
//

xyz_type SphereRigidRotationVelocity::evalVector(const xyz_type& xIn) const {
	return xyz_type(-xIn.y, xIn.x, 0.0);
}

//
// ShallowWaterTestCases
//

double ShallowWaterTestCases::evalScalar(const xyz_type& xIn, const local_index_type input_comp) const {
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

//
// CoriolisForce
//

double CoriolisForce::evalScalar(const xyz_type& xIn, const local_index_type input_comp) const {
	// f
	const scalar_type lat = xIn.latitude(); // phi
	const scalar_type lon = xIn.longitude(); // lambda

	return 2.0*_Omega*( - std::cos(lon) * std::cos(lat) * std::sin(_alpha) + std::sin(lat) * std::cos(_alpha));
}

//
// DiscontinuousOnSphere
//

double DiscontinuousOnSphere::evalScalar(const xyz_type& xIn, const local_index_type input_comp) const {
	const scalar_type lon = xIn.longitude(); // lambda
	const scalar_type eps = 0;

	return (lon > eps && lon < _gc.Pi()-eps) ? 0 : 1;
}

//
// FiveStripOnSphere
//

scalar_type FiveStripOnSphere::evalScalar(const xyz_type& xIn, const local_index_type input_comp) const {
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

//
// CylinderSinLonCosZ
//

scalar_type CylinderSinLonCosZ::evalScalar(const xyz_type& xyzIn, const local_index_type input_comp) const {
	return xyzIn.x*cos(xyzIn.z);
}

scalar_type CylinderSinLonCosZRHS::evalScalar(const xyz_type& xyzIn, const local_index_type input_comp) const {
	const scalar_type s = xyzIn.z;
	const scalar_type denom = std::sqrt(xyzIn.x*xyzIn.x+xyzIn.y*xyzIn.y);
	// only negatived because drivers expect a negated version of the laplacian rhs
	return -(-cos(s)*xyzIn.x/denom - cos(s)*xyzIn.x*denom)/denom;
}

//
// DivFreeSineCos
//

scalar_type DivFreeSineCos::evalScalar(const xyz_type& xyzIn, const local_index_type input_comp) const {
    if (_dim==3) {
        TEUCHOS_ASSERT(false);
        return 0;
    } else {
        if (input_comp==0) {
            return cos(xyzIn.x)*sin(xyzIn.y);
        } else {
            return -sin(xyzIn.x)*cos(xyzIn.y);
        }
    }
}

xyz_type DivFreeSineCos::evalScalarDerivative(const xyz_type& xyzIn, const local_index_type input_comp) const {
    if (_dim==3) {
        TEUCHOS_ASSERT(false);
        return xyz_type();
    } else {
        if (input_comp==0) {
            return xyz_type(-sin(xyzIn.x)*sin(xyzIn.y),
                            cos(xyzIn.x)*cos(xyzIn.y));
        } else {
            return xyz_type(-cos(xyzIn.x)*cos(xyzIn.y),
                            sin(xyzIn.x)*sin(xyzIn.y));
        }
    }
}

std::vector<xyz_type> DivFreeSineCos::evalScalarHessian(const xyz_type& xyzIn, const local_index_type input_comp) const {
    std::vector<xyz_type> hessian(_dim);
    if (_dim==3) {
        TEUCHOS_ASSERT(false);
        return hessian;
    } else {

        if (input_comp==0) {
            // u_xd = (u_xx, u_xy)
            xyz_type u_xd(-cos(xyzIn.x)*sin(xyzIn.y),-sin(xyzIn.x)*cos(xyzIn.y));
            // u_yd = (u_yx, u_yy)
            xyz_type u_yd(-sin(xyzIn.x)*cos(xyzIn.y),-cos(xyzIn.x)*sin(xyzIn.y));
            hessian[0] = u_xd;
            hessian[1] = u_yd;
        } else {
            // u_xd = (u_xx, u_xy)
            xyz_type u_xd(sin(xyzIn.x)*cos(xyzIn.y),cos(xyzIn.x)*sin(xyzIn.y));
            // u_yd = (u_yx, u_yy)
            xyz_type u_yd(cos(xyzIn.x)*sin(xyzIn.y),sin(xyzIn.x)*cos(xyzIn.y));
            hessian[0] = u_xd;
            hessian[1] = u_yd;
        }
    }
    return hessian;
}

//
// SineProducts
//

scalar_type SineProducts::evalScalar(const xyz_type& xyzIn, const local_index_type input_comp) const {
    if (_dim==3) {
        if (input_comp==0) {
            return sin(xyzIn.x)*sin(xyzIn.y)*sin(xyzIn.z);
        } else if (input_comp==1) {
            return -1*(sin(xyzIn.x)*sin(xyzIn.y)*sin(xyzIn.z));
        } else {
            return 2*(sin(xyzIn.x)*sin(xyzIn.y)*sin(xyzIn.z));
        }
    } else {
        if (input_comp==0) {
            return sin(xyzIn.x)*sin(xyzIn.y);
        } else {
            return -1*sin(xyzIn.x)*sin(xyzIn.y);
        }
    }
}

xyz_type SineProducts::evalScalarDerivative(const xyz_type& xyzIn, const local_index_type input_comp) const {
    if (_dim==3) {
        TEUCHOS_ASSERT(input_comp==0);
        if (input_comp==0) {
            return xyz_type(cos(xyzIn.x)*sin(xyzIn.y)*sin(xyzIn.z),
                            sin(xyzIn.x)*cos(xyzIn.y)*sin(xyzIn.z),
                            sin(xyzIn.x)*sin(xyzIn.y)*cos(xyzIn.z));
        } else if (input_comp==1) {
            return -1*xyz_type(cos(xyzIn.x)*sin(xyzIn.y)*sin(xyzIn.z),
                               sin(xyzIn.x)*cos(xyzIn.y)*sin(xyzIn.z),
                               sin(xyzIn.x)*sin(xyzIn.y)*cos(xyzIn.z));
        } else {
            return 2*xyz_type(cos(xyzIn.x)*sin(xyzIn.y)*sin(xyzIn.z),
                              sin(xyzIn.x)*cos(xyzIn.y)*sin(xyzIn.z),
                              sin(xyzIn.x)*sin(xyzIn.y)*cos(xyzIn.z));
        }
    } else {
        if (input_comp==0) {
            return xyz_type(cos(xyzIn.x)*sin(xyzIn.y),
                            sin(xyzIn.x)*cos(xyzIn.y));
        } else {
            return -1*xyz_type(cos(xyzIn.x)*sin(xyzIn.y),
                               sin(xyzIn.x)*cos(xyzIn.y));
        }
    }
}

std::vector<xyz_type> SineProducts::evalScalarHessian(const xyz_type& xyzIn, const local_index_type input_comp) const {
    std::vector<xyz_type> hessian(_dim);
    if (_dim==3) {
        // first component
        // u_xd = (u_xx, u_xy, u_xz)
        xyz_type u_xd(-sin(xyzIn.x)*sin(xyzIn.y)*sin(xyzIn.z),cos(xyzIn.x)*cos(xyzIn.y)*sin(xyzIn.z),cos(xyzIn.x)*sin(xyzIn.y)*cos(xyzIn.z));
        // u_yd = (u_yx, u_yy, u_yz)
        xyz_type u_yd(cos(xyzIn.x)*cos(xyzIn.y)*sin(xyzIn.z),-sin(xyzIn.x)*sin(xyzIn.y)*sin(xyzIn.z),sin(xyzIn.x)*cos(xyzIn.y)*cos(xyzIn.z));
        // u_zd = (u_zx, u_zy, u_zz))
        xyz_type u_zd(cos(xyzIn.x)*sin(xyzIn.y)*cos(xyzIn.z),sin(xyzIn.x)*cos(xyzIn.y)*cos(xyzIn.z),-sin(xyzIn.x)*sin(xyzIn.y)*sin(xyzIn.z));
        if (input_comp==0) {
            hessian[0] = u_xd;
            hessian[1] = u_yd;
            hessian[2] = u_zd;
        } else if (input_comp==1) {
            hessian[0] = -1*u_xd;
            hessian[1] = -1*u_yd;
            hessian[2] = -1*u_zd;
        } else {
            hessian[0] = 2*u_xd;
            hessian[1] = 2*u_yd;
            hessian[2] = 2*u_zd;
        }
    } else {
        // u_xd = (u_xx, u_xy)
        xyz_type u_xd(-sin(xyzIn.x)*sin(xyzIn.y),cos(xyzIn.x)*cos(xyzIn.y));
        // u_yd = (u_yx, u_yy)
        xyz_type u_yd(cos(xyzIn.x)*cos(xyzIn.y),-sin(xyzIn.x)*sin(xyzIn.y));

        if (input_comp==0) {
            hessian[0] = u_xd;
            hessian[1] = u_yd;
        } else {
            hessian[0] = -1*u_xd;
            hessian[1] = -1*u_yd;
        }
    }
    return hessian;
}

scalar_type SineProducts::evalScalarLaplacian(const xyz_type& xyzIn, const local_index_type input_comp) const {
    if (input_comp==0) {
        if (_dim==3) {
            return -3*sin(xyzIn.x)*sin(xyzIn.y)*sin(xyzIn.z);
        } else {
            return -2*sin(xyzIn.x)*sin(xyzIn.y);
        }
    } if (input_comp==1) {
        if (_dim==3) {
            return 3*sin(xyzIn.x)*sin(xyzIn.y)*sin(xyzIn.z);
        } else {
            return 2*sin(xyzIn.x)*sin(xyzIn.y);
        }
    } else {
        if (_dim==3) {
            return -6*sin(xyzIn.x)*sin(xyzIn.y)*sin(xyzIn.z);
        } else {
            return -4*sin(xyzIn.x)*sin(xyzIn.y);
        }
    }
}

//
// FirstOrderBasis
//

scalar_type FirstOrderBasis::evalScalar(const xyz_type& xyzIn, const local_index_type input_comp) const {
    if (_dim==3) {
        if (input_comp==0) {
            return 2*xyzIn.x + xyzIn.y + xyzIn.z;
        } else if (input_comp==1) {
            return -1*(xyzIn.x + xyzIn.y + xyzIn.z);
        } else {
            return xyzIn.x - xyzIn.y - xyzIn.z;
        }
    } else if (_dim==2) {
        if (input_comp==0) {
            return xyzIn.x + xyzIn.y;
        } else {
            return -1*(xyzIn.x + xyzIn.y);
        }
    } else {
        TEUCHOS_ASSERT(false);
        return false;
    }
}

xyz_type FirstOrderBasis::evalScalarDerivative(const xyz_type& xyzIn, const local_index_type input_comp) const {
    if (_dim==3) {
        if (input_comp==0) {
            return xyz_type(2,1,1);
        } else if (input_comp==1) {
            return xyz_type(-1,-1,-1);
        } else {
            return xyz_type(1,-1,-1);
        }
    } else {
        if (input_comp==0) {
            return xyz_type(1,1);
        } else {
            return xyz_type(-1,-1);
        }
    }
}

std::vector<xyz_type> FirstOrderBasis::evalScalarHessian(const xyz_type& xyzIn, const local_index_type input_comp) const {
    std::vector<xyz_type> hessian(_dim);
    if (_dim==3) {
        xyz_type u_xd;
        xyz_type u_yd;
        xyz_type u_zd;
        hessian[0] = u_xd;
        hessian[1] = u_yd;
        hessian[2] = u_zd;
    } else {
        // first component
        // u_xd = (u_xx, u_xy)
        xyz_type u_xd(0,0,0);
        // u_yd = (u_yx, u_yy)
        xyz_type u_yd(0,0,0);
        if (input_comp==0) {
            hessian[0] = u_xd;
            hessian[1] = u_yd;
        } else {
            hessian[0] = u_xd;
            hessian[1] = u_yd;
        }
    }
    return hessian;
}

scalar_type FirstOrderBasis::evalScalarLaplacian(const xyz_type& xyzIn, const local_index_type input_comp) const {
    return 0;
}

//
// Divergence Free SecondOrderBasis
//

scalar_type DivFreeSecondOrderBasis::evalScalar(const xyz_type& xyzIn, const local_index_type input_comp) const {

    if (_dim==3) {
        TEUCHOS_ASSERT(false);
        return 0;
    } else if (_dim==2) {
        // u= [ 1 + x + y + x^2 + 2*xy + y^2
        //     -1 - x - y - x^2 - 2*xy - y^2 ]
        if (input_comp==0) {
            return 1 + xyzIn.x + xyzIn.y + xyzIn.x*xyzIn.x + 2*xyzIn.x*xyzIn.y + xyzIn.y*xyzIn.y;
        } else {
            return -1*(1 + xyzIn.x + xyzIn.y + xyzIn.x*xyzIn.x + 2*xyzIn.x*xyzIn.y + xyzIn.y*xyzIn.y);
        }
    } else {
        TEUCHOS_ASSERT(false);
        return false;
    }
}

xyz_type DivFreeSecondOrderBasis::evalScalarDerivative(const xyz_type& xyzIn, const local_index_type input_comp) const {
    if (_dim==3) {
        TEUCHOS_ASSERT(false);
        return xyz_type();
    } else {
        if (input_comp==0) {
            return xyz_type(1 + 2*xyzIn.x + 2*xyzIn.y,
                            1 + 2*xyzIn.x + 2*xyzIn.y);
        } else {
            return -1*xyz_type(1 + 2*xyzIn.x + 2*xyzIn.y,
                               1 + 2*xyzIn.x + 2*xyzIn.y);
        }
    }
}

std::vector<xyz_type> DivFreeSecondOrderBasis::evalScalarHessian(const xyz_type& xyzIn, const local_index_type input_comp) const {
    std::vector<xyz_type> hessian(_dim);
    if (_dim==3) {
        TEUCHOS_ASSERT(false);
        return hessian;
    } else {
        // first component
        // u_xd = (u_xx, u_xy)
        xyz_type u_xd(2,2);
        // u_yd = (u_yx, u_yy)
        xyz_type u_yd(2,2);
        if (input_comp==0) {
            hessian[0] = u_xd;
            hessian[1] = u_yd;
        } else {
            hessian[0] = -1*u_xd;
            hessian[1] = -1*u_yd;
        }
    }
    return hessian;
}

scalar_type DivFreeSecondOrderBasis::evalScalarLaplacian(const xyz_type& xyzIn, const local_index_type input_comp) const {
    if (_dim==3) {
        TEUCHOS_ASSERT(false);
        return 0;
    } else {
        if (input_comp==0) {
            return 4;
        } else if (input_comp==1) {
            return -4;
        }
        return 0;
    }
}

//
// SecondOrderBasis
//

scalar_type SecondOrderBasis::evalScalar(const xyz_type& xyzIn, const local_index_type input_comp) const {
    if (_dim==3) {
        if (input_comp==0) {
            return xyzIn.x*(1 + xyzIn.x + xyzIn.y + xyzIn.z) + xyzIn.y*(1 + xyzIn.y + xyzIn.z) + xyzIn.z*(1 + xyzIn.z);
        } else if (input_comp==1) {
            return -1*(xyzIn.x*(1 + xyzIn.x + xyzIn.y + xyzIn.z) + xyzIn.y*(1 + xyzIn.y + xyzIn.z) + xyzIn.z*(1 + xyzIn.z));
        } else {
            return 2*(xyzIn.x*(1 + xyzIn.x + xyzIn.y + xyzIn.z) + xyzIn.y*(1 + xyzIn.y + xyzIn.z) + xyzIn.z*(1 + xyzIn.z));
        }
    } else if (_dim==2) {
        if (input_comp==0) {
            return xyzIn.x*(1 + xyzIn.x + xyzIn.y) + xyzIn.y*(1 + xyzIn.y);
        } else {
            return -1*(xyzIn.x*(1 + xyzIn.x + xyzIn.y) + xyzIn.y*(1 + xyzIn.y));
        }
    } else {
        TEUCHOS_ASSERT(false);
        return false;
    }
}

xyz_type SecondOrderBasis::evalScalarDerivative(const xyz_type& xyzIn, const local_index_type input_comp) const {
    if (_dim==3) {
        if (input_comp==0) {
            return xyz_type((1 + 2*xyzIn.x + xyzIn.y + xyzIn.z),
                            (xyzIn.x + 1 + 2*xyzIn.y + xyzIn.z),
                            (xyzIn.x + xyzIn.y + 1 + 2*xyzIn.z));
        } else if (input_comp==1) {
            return -1*xyz_type((1 + 2*xyzIn.x + xyzIn.y + xyzIn.z),
                               (xyzIn.x + 1 + 2*xyzIn.y + xyzIn.z),
                               (xyzIn.x + xyzIn.y + 1 + 2*xyzIn.z));
        } else {
            return 2*xyz_type((1 + 2*xyzIn.x + xyzIn.y + xyzIn.z),
                              (xyzIn.x + 1 + 2*xyzIn.y + xyzIn.z),
                              (xyzIn.x + xyzIn.y + 1 + 2*xyzIn.z));
        }
    } else {
        if (input_comp==0) {
            return xyz_type((1 + 2*xyzIn.x + xyzIn.y),
                            (xyzIn.x + 1 + 2*xyzIn.y));
        } else {
            return -1*xyz_type((1 + 2*xyzIn.x + xyzIn.y),
                               (xyzIn.x + 1 + 2*xyzIn.y));
        }
    }
}

std::vector<xyz_type> SecondOrderBasis::evalScalarHessian(const xyz_type& xyzIn, const local_index_type input_comp) const {
    std::vector<xyz_type> hessian(_dim);
    if (_dim==3) {
        // first component
        // u_xd = (u_xx, u_xy, u_xz)
        xyz_type u_xd(2,1,1);
        // u_yd = (u_yx, u_yy, u_yz)
        xyz_type u_yd(1,2,1);
        // u_zd = (u_zx, u_zy, u_zz))
        xyz_type u_zd(1,1,2);
        if (input_comp==0) {
            hessian[0] = u_xd;
            hessian[1] = u_yd;
            hessian[2] = u_zd;
        } else if (input_comp==1) {
            hessian[0] = -1*u_xd;
            hessian[1] = -1*u_yd;
            hessian[2] = -1*u_zd;
        } else {
            hessian[0] = 2*u_xd;
            hessian[1] = 2*u_yd;
            hessian[2] = 2*u_zd;
        }
    } else {
        // first component
        // u_xd = (u_xx, u_xy)
        xyz_type u_xd(2,1);
        // u_yd = (u_yx, u_yy)
        xyz_type u_yd(1,2);
        if (input_comp==0) {
            hessian[0] = u_xd;
            hessian[1] = u_yd;
        } else {
            hessian[0] = -1*u_xd;
            hessian[1] = -1*u_yd;
        }
    }
    return hessian;
}

scalar_type SecondOrderBasis::evalScalarLaplacian(const xyz_type& xyzIn, const local_index_type input_comp) const {
    if (input_comp==0) {
        if (_dim==3) {
            return 6;
        } else {
            return 4;
        }
    } else if (input_comp==1) {
        if (_dim==3) {
            return -6;
        } else {
            return -4;
        }
    } else {
        return 12;
    }
}

//
// ThirdOrderBasis
//

scalar_type ThirdOrderBasis::evalScalar(const xyz_type& xyzIn, const local_index_type input_comp) const {
    scalar_type x = xyzIn.x;
    scalar_type y = xyzIn.y;
    scalar_type z = xyzIn.z;
    if (_dim==3) {
        if (input_comp==0) {
            return 1 + x + y + z + x*x + x*y + x*z + y*y + y*z + z*z + x*x*x + x*x*y + x*x*z + y*y*x + y*y*y + y*y*z + z*z*x + z*z*y + z*z*z + x*y*z;
        } else if (input_comp==1) {
            return -1*(1 + x + y + z + x*x + x*y + x*z + y*y + y*z + z*z + x*x*x + x*x*y + x*x*z + y*y*x + y*y*y + y*y*z + z*z*x + z*z*y + z*z*z + x*y*z);
        } else {
            return 2*(1 + x + y + z + x*x + x*y + x*z + y*y + y*z + z*z + x*x*x + x*x*y + x*x*z + y*y*x + y*y*y + y*y*z + z*z*x + z*z*y + z*z*z + x*y*z);
        }
    } else if (_dim==2) {
        if (input_comp==0) {
            return 1 + x + y + x*x + x*y + y*y + x*x*x + x*x*y + x*y*y + y*y*y;
        } else {
            return -1*(1 + x + y + x*x + x*y + y*y + x*x*x + x*x*y + x*y*y + y*y*y);
        }
    } else {
        TEUCHOS_ASSERT(false);
        return false;
    }
}

xyz_type ThirdOrderBasis::evalScalarDerivative(const xyz_type& xyzIn, const local_index_type input_comp) const {
    scalar_type x = xyzIn.x;
    scalar_type y = xyzIn.y;
    scalar_type z = xyzIn.z;
    if (_dim==3) {
        TEUCHOS_ASSERT(input_comp==0);
        if (input_comp==0) {
            return xyz_type((1 + 2*x + y + z + 3*x*x + 2*x*y + 2*x*z + y*y + z*z + y*z),
                            (1 + x + 2*y + z + x*x + 2*y*x + 3*y*y + 2*y*z + z*z + x*z),
                            (1 + x + y + 2*z + x*x + y*y + 2*z*x + 2*z*y + 3*z*z + x*y));
        } else if (input_comp==1) {
            return -1*xyz_type((1 + 2*x + y + z + 3*x*x + 2*x*y + 2*x*z + y*y + z*z + y*z),
                               (1 + x + 2*y + z + x*x + 2*y*x + 3*y*y + 2*y*z + z*z + x*z),
                               (1 + x + y + 2*z + x*x + y*y + 2*z*x + 2*z*y + 3*z*z + x*y));
        } else {
            return 2*xyz_type((1 + 2*x + y + z + 3*x*x + 2*x*y + 2*x*z + y*y + z*z + y*z),
                              (1 + x + 2*y + z + x*x + 2*y*x + 3*y*y + 2*y*z + z*z + x*z),
                              (1 + x + y + 2*z + x*x + y*y + 2*z*x + 2*z*y + 3*z*z + x*y));
        }
    } else {
        if (input_comp==0) {
            return xyz_type((1 + 2*x + y + 3*x*x + 2*x*y + y*y),
                            (1 + x + 2*y + x*x + 2*x*y + 3*y*y));
        } else {
            return -1*xyz_type((1 + 2*x + y + 3*x*x + 2*x*y + y*y),
                               (1 + x + 2*y + x*x + 2*x*y + 3*y*y));
        }
    }
}

std::vector<xyz_type> ThirdOrderBasis::evalScalarHessian(const xyz_type& xyzIn, const local_index_type input_comp) const {
    scalar_type x = xyzIn.x;
    scalar_type y = xyzIn.y;
    scalar_type z = xyzIn.z;
    std::vector<xyz_type> hessian(_dim);
    if (_dim==3) {
        // u_xd = (u_xx, u_xy, u_xz)
        xyz_type u_xd(2+6*x+2*y+2*z,   1+2*x+2*y+z, 1+2*x+2*z+y);
        // u_yd = (u_yx, u_yy, u_yz)
        xyz_type u_yd(  1+2*x+2*y+z, 2+2*x+6*y+2*z, 1+2*y+2*z+x);
        // u_zd = (u_zx, u_zy, u_zz)
        xyz_type u_zd(  1+2*x+2*z+y,   1+2*y+2*z+x, 2+2*x+2*y+6*z);
        if (input_comp==0) {
            hessian[0] = u_xd;
            hessian[1] = u_yd;
            hessian[2] = u_zd;
        } else if (input_comp==1) {
            hessian[0] = -1*u_xd;
            hessian[1] = -1*u_yd;
            hessian[2] = -1*u_zd;
        } else {
            hessian[0] = 2*u_xd;
            hessian[1] = 2*u_yd;
            hessian[2] = 2*u_zd;
        }
    } else {
        // u_xd = (u_xx, u_xy)
        xyz_type u_xd(2+6*x+2*y, 1+2*x+2*y);
        // u_yd = (u_yx, u_yy)
        xyz_type u_yd(1+2*x+2*y, 2+2*x+6*y);
        if (input_comp==0) {
            hessian[0] = u_xd;
            hessian[1] = u_yd;
        } else {
            hessian[0] = -1*u_xd;
            hessian[1] = -1*u_yd;
        }
    }
    return hessian;
}

scalar_type ThirdOrderBasis::evalScalarLaplacian(const xyz_type& xyzIn, const local_index_type input_comp) const {
    scalar_type x = xyzIn.x;
    scalar_type y = xyzIn.y;
    scalar_type z = xyzIn.z;
    if (input_comp==0) {
        if (_dim==3) {
            return (2 + 6*x + 2*y + 2*z) + (2 + 2*x + 6*y + 2*z) + (2 + 2*x + 2*y + 6*z);
        } else {
            return (2 + 6*x + 2*y) + (2 + 2*x + 6*y);
        }
    } else if (input_comp==1) {
        if (_dim==3) {
            return -1*((2 + 6*x + 2*y + 2*z) + (2 + 2*x + 6*y + 2*z) + (2 + 2*x + 2*y + 6*z));
        } else {
            return -1*((2 + 6*x + 2*y) + (2 + 2*x + 6*y));
        }
    } else {
        if (_dim==3) {
            return 2*((2 + 6*x + 2*y + 2*z) + (2 + 2*x + 6*y + 2*z) + (2 + 2*x + 2*y + 6*z));
        } else {
            return 2*((2 + 6*x + 2*y) + (2 + 2*x + 6*y));
        }
    }
}

//
// ConstantEachDimension
//

scalar_type ConstantEachDimension::evalScalar(const xyz_type& xyzIn, const local_index_type input_comp) const {
    return _scaling_factors[input_comp];
}

xyz_type ConstantEachDimension::evalScalarDerivative(const xyz_type& xyzIn, const local_index_type input_comp) const {
	return xyz_type(); // zero vector
}

scalar_type ConstantEachDimension::evalScalarLaplacian(const xyz_type& xyzIn, const local_index_type input_comp) const {
	return 0;
}

//
// ScaleOfEachDimension
//

xyz_type ScaleOfEachDimension::evalVector(const xyz_type& xyzIn) const {
    return xyz_type(_scaling_factors[0]*xyzIn.x,_scaling_factors[1]*xyzIn.y,_scaling_factors[2]*xyzIn.z);
}

//
// CangaSphereTransform
//

xyz_type CangaSphereTransform::evalVector(const xyz_type& latLonIn) const {
	const double lat = (_in_degrees) ? latLonIn.x * _gc.Pi() / 180.0 : latLonIn.x;
	const double lon = (_in_degrees) ? latLonIn.y * _gc.Pi() / 180.0 : latLonIn.y;
    return xyz_type(cos(lon)*cos(lat), sin(lon)*cos(lat), sin(lat));
}

//
// CurlCurlSineTest
//

xyz_type CurlCurlSineTestRHS::evalVector(const xyz_type& xyzIn) const {
    return xyz_type(2.0*sin(xyzIn.y)*sin(xyzIn.z),
                    2.0*sin(xyzIn.x)*sin(xyzIn.z),
                    2.0*sin(xyzIn.x)*sin(xyzIn.y));
}

xyz_type CurlCurlSineTest::evalVector(const xyz_type& xyzIn) const {
    return xyz_type(sin(xyzIn.y)*sin(xyzIn.z),
                    sin(xyzIn.x)*sin(xyzIn.z),
                    sin(xyzIn.x)*sin(xyzIn.y));
}

xyz_type CurlCurlPolyTestRHS::evalVector(const xyz_type& xyzIn) const {
    return xyz_type(-14.0*xyzIn.x - 12.0*xyzIn.z,
                    14.0*xyzIn.y,
                    12.0*xyzIn.x);
}

xyz_type CurlCurlPolyTest::evalVector(const xyz_type& xyzIn) const {
    return xyz_type(7.0*xyzIn.x*xyzIn.z*xyzIn.z + 6.0*xyzIn.y*xyzIn.y*xyzIn.z,
                    -7.0*xyzIn.y*xyzIn.z*xyzIn.z,
                    -2.0*xyzIn.x*xyzIn.x*xyzIn.x);
} 

}
