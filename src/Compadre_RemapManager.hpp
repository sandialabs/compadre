#ifndef _COMPADRE_REMAPMANAGER_HPP_
#define _COMPADRE_REMAPMANAGER_HPP_

#include "Compadre_Config.h"
#include "Compadre_Typedefs.hpp"

#include <GMLS.hpp>

namespace Compadre {

class ParticlesT;
class NeighborhoodT;

struct RemapObject {

	public:
		std::string src_fieldname;
		std::string trg_fieldname;
		local_index_type src_fieldnum;
		local_index_type trg_fieldnum;

		ReconstructionOperator::TargetOperation _target_operation;
		ReconstructionOperator::ReconstructionSpace _reconstruction_space;
		ReconstructionOperator::SamplingFunctional _polynomial_sampling_functional;
		ReconstructionOperator::SamplingFunctional _data_sampling_functional;

		std::string _operator_coefficients_fieldname;

		bool _obfet;


		// basic instantiation with differing polynomial and data sampling functionals
		RemapObject(ReconstructionOperator::TargetOperation target_operation, ReconstructionOperator::ReconstructionSpace reconstruction_space, ReconstructionOperator::SamplingFunctional polynomial_sampling_functional, ReconstructionOperator::SamplingFunctional data_sampling_functional, bool obfet = false) {
			src_fieldnum = -1;
			trg_fieldnum = -1;
			_target_operation = target_operation;
			_reconstruction_space = reconstruction_space;
			_polynomial_sampling_functional = polynomial_sampling_functional;
			_data_sampling_functional = data_sampling_functional;
			_obfet = obfet;
		}
		// basic instantiation with SAME polynomial and data sampling functionals
		RemapObject(ReconstructionOperator::TargetOperation target_operation = ReconstructionOperator::TargetOperation::ScalarPointEvaluation, ReconstructionOperator::ReconstructionSpace reconstruction_space = ReconstructionOperator::ReconstructionSpace::ScalarTaylorPolynomial, ReconstructionOperator::SamplingFunctional sampling_functional = ReconstructionOperator::SamplingFunctional::PointSample, bool obfet = false)
			: RemapObject(target_operation, reconstruction_space, sampling_functional, sampling_functional, obfet) {}


		RemapObject(const std::string& source_name, const std::string& target_name, ReconstructionOperator::TargetOperation target_operation, ReconstructionOperator::ReconstructionSpace reconstruction_space, ReconstructionOperator::SamplingFunctional polynomial_sampling_functional, ReconstructionOperator::SamplingFunctional data_sampling_functional, bool obfet = false)
			: RemapObject(target_operation, reconstruction_space, polynomial_sampling_functional, data_sampling_functional, obfet) {
			src_fieldname = source_name;
			if (target_name.empty()) trg_fieldname = source_name;
			else trg_fieldname = target_name;
		}
		RemapObject(const std::string& source_name, const std::string& target_name = std::string(), ReconstructionOperator::TargetOperation target_operation = ReconstructionOperator::TargetOperation::ScalarPointEvaluation, ReconstructionOperator::ReconstructionSpace reconstruction_space = ReconstructionOperator::ReconstructionSpace::ScalarTaylorPolynomial, ReconstructionOperator::SamplingFunctional sampling_functional = ReconstructionOperator::SamplingFunctional::PointSample, bool obfet = false)
			: RemapObject(source_name, target_name, target_operation, reconstruction_space, sampling_functional, sampling_functional, obfet) {}


		RemapObject(const local_index_type source_num, const local_index_type target_num, ReconstructionOperator::TargetOperation target_operation, ReconstructionOperator::ReconstructionSpace reconstruction_space, ReconstructionOperator::SamplingFunctional polynomial_sampling_functional, ReconstructionOperator::SamplingFunctional data_sampling_functional, bool obfet = false)
			: RemapObject(target_operation, reconstruction_space, polynomial_sampling_functional, data_sampling_functional, obfet) {
			src_fieldnum = source_num;
			if (target_num < 0) trg_fieldnum = source_num;
		}
		RemapObject(const local_index_type source_num, const local_index_type target_num = -1, ReconstructionOperator::TargetOperation target_operation = ReconstructionOperator::TargetOperation::ScalarPointEvaluation, ReconstructionOperator::ReconstructionSpace reconstruction_space = ReconstructionOperator::ReconstructionSpace::ScalarTaylorPolynomial, ReconstructionOperator::SamplingFunctional sampling_functional = ReconstructionOperator::SamplingFunctional::PointSample, bool obfet = false)
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

		void add(RemapObject obj) {
			// This function sorts through insert by checking the sampling strategy type
			// and inserting in the correct position

			// if empty, then add the remap object
			if (_queue.size()==0) {
				_queue.push_back(obj);
				return;
			}
			// loop through remap objects until we find one with the same samplings and reconstruction space, then add to the end of it
			bool obj_inserted = false;
			local_index_type index_with_last_match = -1;

			for (local_index_type i=0; i<_queue.size(); ++i) {
				if ((_queue[i]._reconstruction_space == obj._reconstruction_space) && (_queue[i]._polynomial_sampling_functional == obj._polynomial_sampling_functional) && (_queue[i]._data_sampling_functional == obj._data_sampling_functional)) {
					index_with_last_match = i;
				} else if (index_with_last_match > -1) { // first time they differ in strategies after finding the sampling / space match
					// insert at this index
					_queue.insert(_queue.begin()+i, obj); // end of previously found location
					obj_inserted = true;
					break;
				}
			}
			if (!obj_inserted) _queue.push_back(obj); // also takes care of case where you match up to the last item in the queue, but never found one that differed
		}

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
};

}

#endif
