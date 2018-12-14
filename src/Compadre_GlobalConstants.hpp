//
//  Compadre_GlobalConstants.hpp
//
//  Created by Peter Bosler on 11/3/16.
//  Copyright (c) 2016 Peter Bosler. All rights reserved.
//
#ifndef _COMPADRE_GlobalConstants_HPP_
#define _COMPADRE_GlobalConstants_HPP_

#include "CompadreHarness_Typedefs.hpp"

namespace Compadre {

/**
	@brief Provides a templated data structure for holding/retrieving floating point 
	constants used throughout the rest of the code.
	All constants are listed as a `std::pair`; the first entry is the value, the second is its units (if applicable).
*/
class GlobalConstants {

	public :

		typedef std::pair<scalar_type, std::string> pair_type;		
		
		/// Pi
		inline scalar_type Pi() const {return _pi.first;}
		/// Conversion factor, radians to degrees
		inline scalar_type Rad2Deg() const {return _rad2deg.first;}
		/// Conversion factor, degrees to radians
		inline scalar_type Deg2Rad() const {return _deg2rad.first;}
		
		/// Gravitational constant
		inline scalar_type Gravity() const {return _gravity.first;}
		inline std::string GravityUnits() const {return _gravity.second;}
		
		/// Mean sea level radius of the Earth
		inline scalar_type EarthRadius() const {return _earthRadius.first;}
		inline std::string EarthRadiusUnits() const {return _earthRadius.second;}
		
		/// Floating point zero tolerance, used for comparing two floating point numbers
		inline scalar_type ZeroTol() const {return _zeroTol.first;}
		void setZeroTol(const scalar_type new_tol) {_zeroTol.first = new_tol;}
		
		/// Universal gas constant for dry air
		inline scalar_type RdDryAir() const {return _RdDryAir.first;}
		inline std::string RdDryAirUnits() const {return _RdDryAir.second;}
		
		
		GlobalConstants() : _pi(3.1415926535897932384626433832795027975, "n/a"), 
							_rad2deg(180.0 / _pi.first, "n/a"), 
							_deg2rad(_pi.first / 180.0, "n/a"), 
							_gravity(9.80616, "m s^{-2}"),
							_earthRadius(6371220.0, "m"),
							_zeroTol(1.0e-14, "n/a"),
							_RdDryAir(287.0, "J Kg^{-1} K^{-1}") {};

	private :

		pair_type _pi;
		pair_type _rad2deg;
		pair_type _deg2rad;
		pair_type _gravity;
		pair_type _earthRadius;
		pair_type _zeroTol;
		pair_type _RdDryAir;

};

};

#endif 
