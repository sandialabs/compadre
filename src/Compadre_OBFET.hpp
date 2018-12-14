#ifndef _COMPADRE_OBFET_HPP_
#define _COMPADRE_OBFET_HPP_

#include "CompadreHarness_Config.h"
#include "CompadreHarness_Typedefs.hpp"

namespace Compadre {

/*
*  This class loads data for OBFET
*/
class NeighborhoodT;
class CoordsT;
class ParticlesT;
class FieldT;
class XyzVector;

class OBFET {

	/*! \brief OBFET minimizes deviation from target solution while conserving total mass and staying within bounds
	 *
	 *  OBFET minimizes deviation from target solution while conserving total mass and staying within bounds
	 *
	 *  TODO: currently only written to run in serial
	 */

	protected:

		typedef Compadre::NeighborhoodT neighbors_type;
		typedef Compadre::CoordsT coords_type;
		typedef Compadre::FieldT field_type;
		typedef Compadre::ParticlesT particles_type;
		typedef Compadre::XyzVector xyz_type;

		const particles_type* _source_particles;
		particles_type* _target_particles;

		local_index_type _source_field_num;
		local_index_type _target_field_num;

		std::string _source_weighting_name, _target_weighting_name;

		const neighbors_type* _neighborhood_info;

	public:

		OBFET( const particles_type* source_particles, particles_type* target_particles, local_index_type source_field_num, local_index_type target_field_num, std::string source_weighting_name, std::string target_weighting_name, const neighbors_type* neighborhood_info)
			: _source_particles(source_particles), _target_particles(target_particles), _source_field_num(source_field_num), _target_field_num(target_field_num), _source_weighting_name(source_weighting_name), _target_weighting_name(target_weighting_name), _neighborhood_info(neighborhood_info) {}

		virtual ~OBFET() {};

		void solveAndUpdate();

};


}

#endif
