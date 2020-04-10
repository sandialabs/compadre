#ifndef _COMPADRE_XYZVECTOR_HPP_
#define _COMPADRE_XYZVECTOR_HPP_

#include "CompadreHarness_Config.h"
#include "CompadreHarness_Typedefs.hpp"

namespace Compadre {

/*	Class for vectors in R3, and various functions for computing geometric quantities in R3. */
class XyzVector
{

	public:

		scalar_type x;
		scalar_type y;
		scalar_type z;
		
		KOKKOS_INLINE_FUNCTION
		XyzVector( const scalar_type nx = 0.0, const scalar_type ny = 0.0, const scalar_type nz = 0.0 ) :
			x(nx), y(ny), z(nz) {};
		
		XyzVector(const scalar_type* arry) : x(arry[0]), y(arry[1]), z(arry[2]) {};
		
		XyzVector(const std::vector<scalar_type>& vec) : x(vec[0]), y(vec[1]), z(vec[2]) {};
		
		inline XyzVector operator += (const XyzVector& other)
			{ x += other.x;
			  y += other.y;
			  z += other.z;
			  return *this;	}
		inline XyzVector operator += (XyzVector& other)
			{ x += other.x;
			  y += other.y;
			  z += other.z;
			  return *this;	}
		inline XyzVector operator -= (const XyzVector& other)
			{ x -= other.x;
			  y -= other.y;
			  z -= other.z;
			  return *this;	}
		inline XyzVector operator -= (XyzVector& other)
			{ x -= other.x;
			  y -= other.y;
			  z -= other.z;
			  return *this;	}
		inline XyzVector operator *= (const double& scaling)
			{ x *= scaling;
			  y *= scaling;
			  z *= scaling;
			  return *this;	}

		inline scalar_type& operator [](const int i) {
			switch (i) {
				case 0:
					return x;
				case 1:
					return y;
				default:
					return z;
			}
		}

		inline scalar_type operator [](const int i) const {
			switch (i) {
				case 0:
					return x;
				case 1:
					return y;
				default:
					return z;
			}
		}

		inline scalar_type magnitude() const { return std::sqrt( x*x + y*y + z*z); }

		inline scalar_type dotProduct( const XyzVector& other) const { return x*other.x + y*other.y + z*other.z	; }
		
		scalar_type latitude() const;
		scalar_type longitude() const;
		scalar_type colatitude() const;

		inline XyzVector crossProduct( const XyzVector& other) const
			{ return XyzVector( y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x); }

		inline void scale( const scalar_type multiplier ){ x *= multiplier; y *= multiplier; z *= multiplier; }

		inline void normalize() { const scalar_type norm = magnitude(); x /= norm; y /= norm; z /= norm; }

		inline void convertToStdVector(std::vector<scalar_type>& vec) const { vec[0] = x; vec[1] = y; vec[2] = z; }

		inline void convertToArray(scalar_type* arry) const {arry[0] = x; arry[1] = y; arry[2] = z;}
		
};

XyzVector operator + ( const XyzVector& vecA, const XyzVector& vecB );

XyzVector operator - ( const XyzVector& vecA, const XyzVector& vecB );
	
XyzVector operator * ( const XyzVector& vecA, const XyzVector& vecB );

XyzVector operator + ( const XyzVector& vecA, const scalar_type& constant );

XyzVector operator + ( const scalar_type& constant, const XyzVector& vecA );

XyzVector operator - ( const XyzVector& vecA, const scalar_type& constant );

XyzVector operator - ( const scalar_type& constant, const XyzVector& vecA );

XyzVector operator * ( const XyzVector& vecA, const scalar_type& constant );

XyzVector operator * ( const scalar_type& constant, const XyzVector& vecA );

XyzVector operator / ( const XyzVector& vecA, const scalar_type& constant );

std::ostream& operator << ( std::ostream& os, const XyzVector& vec );

/** @brief Inverse tangent with branch cut at 0 and 2*pi, rather than at pi and -pi.	
*/	
scalar_type atan4( const scalar_type y, const scalar_type x );

bool operator == ( const XyzVector& vecA, const XyzVector& vecB );

XyzVector euclideanMidpoint( const XyzVector& vecA, const XyzVector& vecB );

XyzVector euclideanCentroid( const std::vector<XyzVector > vecs );

scalar_type euclideanDistance( const XyzVector& vecA, const XyzVector& vecB);

/**
	@ingroup XyzVector
	@brief Computes the area of the triangle defined by three points in R3.
*/
scalar_type euclideanTriArea( const XyzVector& vecA, const XyzVector& vecB, const XyzVector& vecC);

/**
	@ingroup XyzVector
	@brief Computes the great-circle distance between two points on the sphere.
*/
scalar_type sphereDistance( const XyzVector& vecA, const XyzVector& vecB,
	const scalar_type radius = 1.0 );

/**
	@ingroup XyzVector
	@brief Computes the area of a spherical triangle whose vertices are given by the function arguments.
*/
scalar_type sphereTriArea( const XyzVector& vecA,
	const XyzVector& vecB, const XyzVector& vecC, const scalar_type radius = 1.0);

/**
	@ingroup XyzVector
	@brief Computes the centroid in R3 of a group of points on the sphere, then projects the centroid to the sphere.
*/
XyzVector sphereCentroid( const std::vector<XyzVector > vecs,
	const scalar_type radius = 1.0 );

XyzVector sphereMidpoint( const XyzVector& vecA,
	const XyzVector& vecB, const scalar_type radius = 1.0 );

}

#endif
