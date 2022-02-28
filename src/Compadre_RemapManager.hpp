#ifndef _COMPADRE_REMAPMANAGER_HPP_
#define _COMPADRE_REMAPMANAGER_HPP_

#include "CompadreHarness_Config.h"
#include "CompadreHarness_Typedefs.hpp"
#include "Compadre_OptimizationManager.hpp"

#include <Compadre_GMLS.hpp>

namespace Compadre {

class ParticlesT;
class NeighborhoodT;

struct RemapObject {

	public:
		std::string src_fieldname;
		std::string trg_fieldname;
		local_index_type src_fieldnum;
		local_index_type trg_fieldnum;
        local_index_type src_dim;
        local_index_type trg_dim;

		TargetOperation _target_operation;
		ReconstructionSpace _reconstruction_space;
		SamplingFunctional _polynomial_sampling_functional;
		SamplingFunctional _data_sampling_functional;

		std::string _operator_coefficients_fieldname;
		std::string _reference_normal_directions_fieldname;
		std::string _source_extra_data_fieldname;
		std::string _target_extra_data_fieldname;

		OptimizationObject _optimization_object;


		// basic instantiation with differing polynomial and data sampling functionals
		RemapObject(TargetOperation target_operation, ReconstructionSpace reconstruction_space, SamplingFunctional polynomial_sampling_functional, SamplingFunctional data_sampling_functional, OptimizationObject optimization_object = OptimizationObject()) : _polynomial_sampling_functional(polynomial_sampling_functional), _data_sampling_functional(data_sampling_functional) {
			src_fieldnum = -1;
			trg_fieldnum = -1;
			_target_operation = target_operation;
			_reconstruction_space = reconstruction_space;
			_optimization_object = optimization_object;
		}
		// basic instantiation with SAME polynomial and data sampling functionals
		RemapObject(TargetOperation target_operation = TargetOperation::ScalarPointEvaluation, ReconstructionSpace reconstruction_space = ReconstructionSpace::ScalarTaylorPolynomial, SamplingFunctional sampling_functional = PointSample, OptimizationObject optimization_object = OptimizationObject())
			: RemapObject(target_operation, reconstruction_space, sampling_functional, sampling_functional, optimization_object) {}


		RemapObject(const std::string& source_name, const std::string& target_name, TargetOperation target_operation, ReconstructionSpace reconstruction_space, SamplingFunctional polynomial_sampling_functional, SamplingFunctional data_sampling_functional, OptimizationObject optimization_object = OptimizationObject())
			: RemapObject(target_operation, reconstruction_space, polynomial_sampling_functional, data_sampling_functional, optimization_object) {
			src_fieldname = source_name;
			if (target_name.empty()) trg_fieldname = source_name;
			else trg_fieldname = target_name;
		}
		RemapObject(const std::string& source_name, const std::string& target_name = std::string(), TargetOperation target_operation = TargetOperation::ScalarPointEvaluation, ReconstructionSpace reconstruction_space = ReconstructionSpace::ScalarTaylorPolynomial, SamplingFunctional sampling_functional = PointSample, OptimizationObject optimization_object = OptimizationObject())
			: RemapObject(source_name, target_name, target_operation, reconstruction_space, sampling_functional, sampling_functional, optimization_object) {}


		RemapObject(const local_index_type source_num, const local_index_type target_num, TargetOperation target_operation, ReconstructionSpace reconstruction_space, SamplingFunctional polynomial_sampling_functional, SamplingFunctional data_sampling_functional, OptimizationObject optimization_object = OptimizationObject())
			: RemapObject(target_operation, reconstruction_space, polynomial_sampling_functional, data_sampling_functional, optimization_object) {
			src_fieldnum = source_num;
			if (target_num < 0) trg_fieldnum = source_num;
		}
		RemapObject(const local_index_type source_num, const local_index_type target_num = -1, TargetOperation target_operation = TargetOperation::ScalarPointEvaluation, ReconstructionSpace reconstruction_space = ReconstructionSpace::ScalarTaylorPolynomial, SamplingFunctional sampling_functional = PointSample, OptimizationObject optimization_object = OptimizationObject())
			: RemapObject(source_num, target_num, target_operation, reconstruction_space, sampling_functional, sampling_functional, optimization_object) {}


		void setOptimizationObject(OptimizationObject optimization_object) {
			_optimization_object = optimization_object;
		}

		OptimizationObject getOptimizationObject() {
			return _optimization_object;
		}

		void setOperatorCoefficients(std::string operator_coefficients_fieldname) {
			_operator_coefficients_fieldname = operator_coefficients_fieldname;
		}

		std::string& getOperatorCoefficients() {
			return _operator_coefficients_fieldname;
		}

		void setNormalDirections(std::string reference_normal_directions_fieldname) {
			_reference_normal_directions_fieldname = reference_normal_directions_fieldname;
		}

		std::string& getNormalDirections() {
			return _reference_normal_directions_fieldname;
		}

		void setSourceExtraData(std::string source_extra_data_fieldname) {
			_source_extra_data_fieldname = source_extra_data_fieldname;
        }

		std::string& getSourceExtraData() {
			return _source_extra_data_fieldname;
		}

		void setTargetExtraData(std::string target_extra_data_fieldname) {
			_target_extra_data_fieldname = target_extra_data_fieldname;
		}

		std::string& getTargetExtraData() {
			return _target_extra_data_fieldname;
		}

};

/*!
 *  RemapManager is a queue for field remappings that need to take place.
 *
 *  It controls adding, clearing, and executing the queue.
 *
 */
class RemapManager {

	protected:

		typedef Compadre::ParticlesT particles_type;
		typedef Compadre::NeighborhoodT neighbors_type;

		Teuchos::RCP<Teuchos::ParameterList> _parameters;

		const particles_type* _src_particles;
		particles_type* _trg_particles;

		std::vector<RemapObject> _queue;
		const scalar_type _max_radius;

		Teuchos::RCP<neighbors_type> _neighborhoodInfo;
		Teuchos::RCP<neighbors_type> _localBoundsNeighborhoodInfo;
		std::vector<local_index_type> _local_operations_num;
		Teuchos::RCP<GMLS> _GMLS;

	public:

		RemapManager(Teuchos::RCP<Teuchos::ParameterList> parameters,
				const particles_type* src_particles, particles_type* trg_particles,
				const scalar_type max_radius) :	_parameters(parameters), _src_particles(src_particles),
				_trg_particles(trg_particles), _max_radius(max_radius) {}

		~RemapManager() {};

		void add(RemapObject obj);

		void clear() {
			_queue = std::vector<RemapObject>();
            // changing the queue invalidates reusing GMLS object
            this->_GMLS = Teuchos::null;
		}

		void execute(bool keep_neighborhoods = true, bool keep_GMLS = false, bool reuse_neighborhoods = true, bool reuse_GMLS = false, bool use_physical_coords = true);

		std::string queueToString() {
			std::string output;
			for (auto obj: this->_queue) {
				output += "source field#: " + std::to_string(obj.src_fieldnum) +  ", target field#: " + std::to_string(obj.trg_fieldnum) + ", target_operation#: " + std::to_string(obj._target_operation) + ", reconstruction_space#: " + std::to_string(obj._reconstruction_space) + ", polynomial_source_sample#: " + std::to_string(obj._polynomial_sampling_functional.id) + ", data_source_sample#: " + std::to_string(obj._data_sampling_functional.id) + "\n";
			}
			return output;
		}

        bool isCompatible(const RemapObject obj_1, const RemapObject obj_2) const;
};

}

#endif
