#include <Compadre_GMLS_Stokes_BoundaryConditions.hpp>

#include <Compadre_CoordsT.hpp>
#include <Compadre_ParticlesT.hpp>
#include <Compadre_FieldT.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_DOFManager.hpp>
#include <Compadre_XyzVector.hpp>
#include <Compadre_AnalyticFunctions.hpp>

namespace Compadre {

typedef Compadre::FieldT fields_type;
typedef Compadre::XyzVector xyz_type;

void GMLS_StokesBoundaryConditions::flagBoundaries() {
}

void GMLS_StokesBoundaryConditions::applyBoundaries(local_index_type field_one, local_index_type field_two, scalar_type time) {
    Teuchos::RCP<Compadre::AnalyticFunction> velocity_function, pressure_function;
    if (_parameters->get<std::string>("solution type")=="sine") {
        velocity_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::CurlCurlSineTest));
        pressure_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::SineProducts));
    } else {
        velocity_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::CurlCurlPolyTest));
        pressure_function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::SecondOrderBasis));
    }

    TEUCHOS_TEST_FOR_EXCEPT_MSG(_b==NULL, "Tpetra Multivector for BCS not yet specified.");
    if (field_two == -1) {
        field_two = field_one;
    }

    host_view_local_index_type bc_id = this->_particles->getFlags()->getLocalView<host_view_local_index_type>();
    host_view_type rhs_vals = this->_b->getLocalView<host_view_type>();
    host_view_type pts = this->_coords->getPts()->getLocalView<host_view_type>();

    const local_index_type nlocal = static_cast<local_index_type>(this->_coords->nLocal());
    const std::vector<Teuchos::RCP<fields_type>>& fields = this->_particles->getFieldManagerConst()->getVectorOfFields();
    const local_dof_map_view_type local_to_dof_map = _dof_data->getDOFMap();

    // obtaint the ID for the velocity and the pressure field
    auto velocity_field_id = _particles->getFieldManagerConst()->getIDOfFieldFromName("velocity");
    auto pressure_field_id = _particles->getFieldManagerConst()->getIDOfFieldFromName("pressure");

    for (local_index_type i=0; i<nlocal; i++) {
        for (local_index_type k=0; k<fields[field_one]->nDim(); k++) {
            const local_index_type dof = local_to_dof_map(i, field_one, k);
            xyz_type pt(pts(i, 0), pts(i, 1), pts(i, 2));
            if (field_one == velocity_field_id && field_two == velocity_field_id) {
                if (bc_id(i, 0) == 1) {
                    rhs_vals(dof, 0) = velocity_function->evalVector(pt)[k];
                }
            }
        }
    }
}

std::vector<InteractingFields> GMLS_StokesBoundaryConditions::gatherFieldInteractions() {
    std::vector<InteractingFields> field_interactions;
    return field_interactions;
}

}
