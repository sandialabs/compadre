#ifndef _COMPADRE_OPTIMIZATIONMANAGER_HPP_
#define _COMPADRE_OPTIMIZATIONMANAGER_HPP_

#include "CompadreHarness_Config.h"
#include "CompadreHarness_Typedefs.hpp"

namespace Compadre {

/*
*  This class loads data for Optimization
*/
class NeighborhoodT;
class CoordsT;
class ParticlesT;
class FieldT;
class XyzVector;

/*!
*  OBFET (L2 minimizer) only runs in serial, while CAAS runs in parallel
*/
enum OptimizationAlgorithm { NONE, OBFET, CAAS };

struct OptimizationObject {

    OptimizationAlgorithm _optimization_algorithm;
     
    bool _single_linear_bound_constraint;
    bool _bounds_preservation;

    scalar_type _global_lower_bound;
    scalar_type _global_upper_bound;

    OptimizationObject(OptimizationAlgorithm optimization_algorithm = NONE, bool single_linear_bound_constraint = true, 
            bool bounds_preservation = true, scalar_type global_lower_bound = std::numeric_limits<scalar_type>::lowest(), 
            scalar_type global_upper_bound = std::numeric_limits<scalar_type>::max()) :
        _optimization_algorithm(optimization_algorithm), _single_linear_bound_constraint(single_linear_bound_constraint),
        _bounds_preservation(bounds_preservation), _global_lower_bound(global_lower_bound), 
        _global_upper_bound(global_upper_bound) {}

    OptimizationObject(std::string optimization_algorithm_str, bool single_linear_bound_constraint = true, 
            bool bounds_preservation = true, scalar_type global_lower_bound = std::numeric_limits<scalar_type>::lowest(), 
            scalar_type global_upper_bound = std::numeric_limits<scalar_type>::max()) :
        _single_linear_bound_constraint(single_linear_bound_constraint),
        _bounds_preservation(bounds_preservation), _global_lower_bound(global_lower_bound), 
        _global_upper_bound(global_upper_bound) {

	    transform(optimization_algorithm_str.begin(), optimization_algorithm_str.end(), 
                optimization_algorithm_str.begin(), ::tolower);

        if (optimization_algorithm_str == "obfet") {
            _optimization_algorithm = OptimizationAlgorithm::OBFET;
        } else if (optimization_algorithm_str == "caas") {
            _optimization_algorithm = OptimizationAlgorithm::CAAS;
        } else {
            _optimization_algorithm = OptimizationAlgorithm::NONE;
        }

    }

    ~OptimizationObject() {};
    
};

class OptimizationManager {

	/*! \brief OptimizationManager minimizes deviation from target solution while conserving total mass and staying within bounds
	 *
	 *  OptimizationManager minimizes deviation from target solution while conserving total mass and staying within bounds
	 *
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

        OptimizationObject _optimization_object;

	public:

		OptimizationManager( const particles_type* source_particles, particles_type* target_particles, 
                local_index_type source_field_num, local_index_type target_field_num, std::string source_weighting_name, 
                std::string target_weighting_name, const neighbors_type* neighborhood_info, 
                OptimizationObject optimization_object = OptimizationObject()) :
			    _source_particles(source_particles), _target_particles(target_particles), _source_field_num(source_field_num), 
                _target_field_num(target_field_num), _source_weighting_name(source_weighting_name), 
                _target_weighting_name(target_weighting_name), _neighborhood_info(neighborhood_info), 
                _optimization_object(optimization_object) {}

		virtual ~OptimizationManager() {};

		void solveAndUpdate();

        void setOptimizationObject(OptimizationObject optimization_object) { _optimization_object = optimization_object; }

};


}

#endif
