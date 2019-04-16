#ifndef _COMPADRE_REMAPMANAGER_HPP_
#define _COMPADRE_REMAPMANAGER_HPP_

#include "CompadreHarness_Config.h"
#include "CompadreHarness_Typedefs.hpp"

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
		std::string _extra_data_fieldname;

		bool _obfet;


		// basic instantiation with differing polynomial and data sampling functionals
		RemapObject(TargetOperation target_operation, ReconstructionSpace reconstruction_space, SamplingFunctional polynomial_sampling_functional, SamplingFunctional data_sampling_functional, bool obfet = false) {
			src_fieldnum = -1;
			trg_fieldnum = -1;
			_target_operation = target_operation;
			_reconstruction_space = reconstruction_space;
			_polynomial_sampling_functional = polynomial_sampling_functional;
			_data_sampling_functional = data_sampling_functional;
			_obfet = obfet;
		}
		// basic instantiation with SAME polynomial and data sampling functionals
		RemapObject(TargetOperation target_operation = TargetOperation::ScalarPointEvaluation, ReconstructionSpace reconstruction_space = ReconstructionSpace::ScalarTaylorPolynomial, SamplingFunctional sampling_functional = SamplingFunctional::PointSample, bool obfet = false)
			: RemapObject(target_operation, reconstruction_space, sampling_functional, sampling_functional, obfet) {}


		RemapObject(const std::string& source_name, const std::string& target_name, TargetOperation target_operation, ReconstructionSpace reconstruction_space, SamplingFunctional polynomial_sampling_functional, SamplingFunctional data_sampling_functional, bool obfet = false)
			: RemapObject(target_operation, reconstruction_space, polynomial_sampling_functional, data_sampling_functional, obfet) {
			src_fieldname = source_name;
			if (target_name.empty()) trg_fieldname = source_name;
			else trg_fieldname = target_name;
		}
		RemapObject(const std::string& source_name, const std::string& target_name = std::string(), TargetOperation target_operation = TargetOperation::ScalarPointEvaluation, ReconstructionSpace reconstruction_space = ReconstructionSpace::ScalarTaylorPolynomial, SamplingFunctional sampling_functional = SamplingFunctional::PointSample, bool obfet = false)
			: RemapObject(source_name, target_name, target_operation, reconstruction_space, sampling_functional, sampling_functional, obfet) {}


		RemapObject(const local_index_type source_num, const local_index_type target_num, TargetOperation target_operation, ReconstructionSpace reconstruction_space, SamplingFunctional polynomial_sampling_functional, SamplingFunctional data_sampling_functional, bool obfet = false)
			: RemapObject(target_operation, reconstruction_space, polynomial_sampling_functional, data_sampling_functional, obfet) {
			src_fieldnum = source_num;
			if (target_num < 0) trg_fieldnum = source_num;
		}
		RemapObject(const local_index_type source_num, const local_index_type target_num = -1, TargetOperation target_operation = TargetOperation::ScalarPointEvaluation, ReconstructionSpace reconstruction_space = ReconstructionSpace::ScalarTaylorPolynomial, SamplingFunctional sampling_functional = SamplingFunctional::PointSample, bool obfet = false)
			: RemapObject(source_num, target_num, target_operation, reconstruction_space, sampling_functional, sampling_functional, obfet) {}


		void setOBFET(bool value) {
			_obfet = value;
		}

		bool getOBFET() {
			return _obfet;
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

		void setExtraData(std::string extra_data_fieldname) {
			_extra_data_fieldname = extra_data_fieldname;
		}

		std::string& getExtraData() {
			return _extra_data_fieldname;
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
		}

		void execute(bool keep_neighborhoods = false, bool use_physical_coords = true);

		std::string queueToString() {
			std::string output;
			for (auto obj: this->_queue) {
				output += "source field#: " + std::to_string(obj.src_fieldnum) +  ", target field#: " + std::to_string(obj.trg_fieldnum) + ", target_operation#: " + std::to_string(obj._target_operation) + ", reconstruction_space#: " + std::to_string(obj._reconstruction_space) + ", polynomial_source_sample#: " + std::to_string(obj._polynomial_sampling_functional)+ ", data_source_sample#: " + std::to_string(obj._data_sampling_functional) + "\n";
			}
			return output;
		}

        bool isCompatible(const RemapObject obj_1, const RemapObject obj_2) const;
};

}

#endif
