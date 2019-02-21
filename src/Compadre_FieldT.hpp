#ifndef _COMPADRE_FIELDT_HPP_
#define _COMPADRE_FIELDT_HPP_

#include "CompadreHarness_Config.h"
#include "CompadreHarness_Typedefs.hpp"

namespace Compadre {

class CoordsT;
class AnalyticFunction;
class XyzVector;

class FieldT {

	friend class CoordsT;

	protected :

		typedef CoordsT coords_type;
		typedef AnalyticFunction function_type;
		typedef XyzVector xyz_type;
		
		std::string _name;
		std::string _units;
		int _nDim;
		
		const coords_type* _coords;
		Teuchos::RCP<mvec_type> _vals;
		Teuchos::RCP<mvec_type> _halo_vals;

        const FieldSparsityType _field_sparsity;

	public :

		FieldT(const coords_type* coords, const int nDim, const std::string name = "noName",
			   const std::string units = "null", const FieldSparsityType fst = FieldSparsityType::Banded);
		
		virtual ~FieldT() {};

		std::string getName() const { return _name; }

		std::string getUnits() const { return _units; }

		local_index_type nDim() const { return _nDim; }
		
		device_view_type getDeviceView();
		
		const FieldSparsityType getFieldSparsityType() { return _field_sparsity; }

		void resize();

		void resetCoords(const coords_type* coords);

		void applyZoltan2Partition();

		void insertParticles(const coords_type * coords, const std::vector<xyz_type>& new_pts_vector, const host_view_type& inserted_field_values);

		void removeParticles(const coords_type * coords, const std::vector<local_index_type>& coord_ids);

		void updateHalo();

		void localInitFromScalarFunction(function_type* fn, bool use_physical_coords = true);
		
		void localInitFromScalarFunctionGradient(function_type* fn, bool use_physical_coords = true);

		std::vector<scalar_type> normInf() const;
		
		std::vector<scalar_type> norm2() const;
		
		void localInitFromVectorFunction(function_type* fn, bool use_physical_coords = true);
		
		scalar_type getLocalScalarVal(const local_index_type idx, const local_index_type component = 0, bool for_halo = false) const;

		std::vector<scalar_type> getLocalVectorVal(const local_index_type idx, bool for_halo = false) const;

		const Teuchos::RCP<mvec_type>& getLocalVectorVals(bool for_halo = false) const;
		
		Teuchos::RCP<mvec_type> getMultiVectorPtr();  // this is perhaps a bad idea, as it violates encapsulation 

		const mvec_type* getMultiVectorPtrConst() const;

		Teuchos::RCP<mvec_type> getHaloMultiVectorPtr();  // this is perhaps a bad idea, as it violates encapsulation

		const mvec_type* getHaloMultiVectorPtrConst() const;

		void updateHaloData();

		void syncMemory();

		void print(std::ostream& os) const;
		
		void scale(const scalar_type a);
		
		void setLocalScalarVal(const local_index_type idx, const local_index_type component, const double value);

		void updateMultiVector(const scalar_type& alpha, const Teuchos::RCP<FieldT> fieldA, const scalar_type& beta);
		
		void updateMultiVector(const scalar_type& alpha, const Teuchos::RCP<FieldT> fieldA, const scalar_type& beta, 
		    const Teuchos::RCP<FieldT> fieldB, const scalar_type& gamma);
};

}

#endif
