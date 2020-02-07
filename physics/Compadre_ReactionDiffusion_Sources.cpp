#include <Compadre_ReactionDiffusion_Sources.hpp>

#include <Compadre_CoordsT.hpp>
#include <Compadre_FieldT.hpp>
#include <Compadre_FieldManager.hpp>
#include <Compadre_DOFManager.hpp>
#include <Compadre_ParticlesT.hpp>
#include <Compadre_AnalyticFunctions.hpp>
#include <Compadre_XyzVector.hpp>
#include <Compadre_NeighborhoodT.hpp>
#include <Compadre_ReactionDiffusion_Operator.hpp>
#include <Compadre_GMLS.hpp>

namespace Compadre {

typedef Compadre::FieldT fields_type;
typedef Compadre::XyzVector xyz_type;

void ReactionDiffusionSources::evaluateRHS(local_index_type field_one, local_index_type field_two, scalar_type time) {

    bool l2_op = (_parameters->get<Teuchos::ParameterList>("physics").get<std::string>("operator")=="l2");
    bool rd_op = (_parameters->get<Teuchos::ParameterList>("physics").get<std::string>("operator")=="rd");
    bool le_op = (_parameters->get<Teuchos::ParameterList>("physics").get<std::string>("operator")=="le");

    TEUCHOS_TEST_FOR_EXCEPT_MSG(_b==NULL, "Tpetra Multivector for RHS not yet specified.");
    if (field_two == -1) {
        field_two = field_one;
    }

    Teuchos::RCP<Compadre::AnalyticFunction> function;
    if (_parameters->get<Teuchos::ParameterList>("physics").get<std::string>("solution")=="polynomial") {
	    function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::SecondOrderBasis(2 /*dimension*/)));
    } else if (_parameters->get<Teuchos::ParameterList>("physics").get<std::string>("solution")=="polynomial_3") {
	    function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::ThirdOrderBasis(2 /*dimension*/)));
    } else {
	    function = Teuchos::rcp_static_cast<Compadre::AnalyticFunction>(Teuchos::rcp(new Compadre::SineProducts(2 /*dimension*/)));
    }

	host_view_type rhs_vals = this->_b->getLocalView<host_view_type>();
    const local_index_type nlocal = static_cast<local_index_type>(this->_coords->nLocal());
    const std::vector<Teuchos::RCP<fields_type> >& fields = this->_particles->getFieldManagerConst()->getVectorOfFields();
    const local_dof_map_view_type local_to_dof_map = _dof_data->getDOFMap();
    const host_view_local_index_type bc_id = this->_particles->getFlags()->getLocalView<host_view_local_index_type>();

    // quantities contained on cells (mesh)
    auto quadrature_points = _physics->_cells->getFieldManager()->getFieldByName("quadrature_points")->getMultiVectorPtr()->getLocalView<host_view_type>();
    auto quadrature_weights = _physics->_cells->getFieldManager()->getFieldByName("quadrature_weights")->getMultiVectorPtr()->getLocalView<host_view_type>();
    auto quadrature_type = _physics->_cells->getFieldManager()->getFieldByName("interior")->getMultiVectorPtr()->getLocalView<host_view_type>();
    auto unit_normals = _physics->_cells->getFieldManager()->getFieldByName("unit_normal")->getMultiVectorPtr()->getLocalView<host_view_type>();
    auto adjacent_elements = _physics->_cells->getFieldManager()->getFieldByName("adjacent_elements")->getMultiVectorPtr()->getLocalView<host_view_type>();

    auto halo_quadrature_points = _physics->_cells->getFieldManager()->getFieldByName("quadrature_points")->getHaloMultiVectorPtr()->getLocalView<host_view_type>();
    auto halo_quadrature_weights = _physics->_cells->getFieldManager()->getFieldByName("quadrature_weights")->getHaloMultiVectorPtr()->getLocalView<host_view_type>();
    auto halo_quadrature_type = _physics->_cells->getFieldManager()->getFieldByName("interior")->getHaloMultiVectorPtr()->getLocalView<host_view_type>();
    auto halo_unit_normals = _physics->_cells->getFieldManager()->getFieldByName("unit_normal")->getHaloMultiVectorPtr()->getLocalView<host_view_type>();
    auto halo_adjacent_elements = _physics->_cells->getFieldManager()->getFieldByName("adjacent_elements")->getHaloMultiVectorPtr()->getLocalView<host_view_type>();

    auto penalty = (_parameters->get<Teuchos::ParameterList>("remap").get<int>("porder")+1)*_parameters->get<Teuchos::ParameterList>("physics").get<double>("penalty")/_physics->_cell_particles_max_h;
    // each element must have the same number of edges
    auto num_edges = adjacent_elements.extent(1);
    auto num_interior_quadrature = 0;
    for (int q=0; q<_physics->_weights_ndim; ++q) {
        if (quadrature_type(0,q)!=1) { // some type of edge quadrature
            num_interior_quadrature = q;
            break;
        }
    }
    auto num_exterior_quadrature_per_edge = (_physics->_weights_ndim - num_interior_quadrature)/num_edges;

    // zero out rhs before starting
    Kokkos::deep_copy(rhs_vals, 0.0);

    // get an unmanaged atomic view that can be added to in a parallel for loop
    Kokkos::View<scalar_type**, decltype(rhs_vals)::memory_space, Kokkos::MemoryTraits<Kokkos::Atomic|Kokkos::Unmanaged> > 
        rhs_vals_atomic(rhs_vals.data(), rhs_vals.extent(0), rhs_vals.extent(1));


    // assembly over cells, then particles
    // loop over cells including halo cells 
    //for (int i=0; i<_physics->_cells->getCoordsConst()->nLocal(true /* include halo in count */); ++i) {
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,_physics->_cells->getCoordsConst()->nLocal(true /* include halo in count */)), KOKKOS_LAMBDA(const int i) {
    //for (int i=0; i<_physics->_cells->getCoordsConst()->nLocal(); ++i) {
        for (local_index_type comp = 0; comp < fields[field_one]->nDim(); ++comp) {

            // filter out (skip) all halo cells that are not able to be seen from a locally owned particle
            auto halo_i = (i<nlocal) ? -1 : _physics->_halo_big_to_small(i-nlocal,0);
            if (i>=nlocal /* halo */  && halo_i<0 /* not one found within max_h of locally owned particles */) return;//continue;

            //TEUCHOS_ASSERT(i<nlocal);
            // need to have a halo gmls object that can be queried for cells that do not live on processor
            // this is also needed for the operator file in order to finish jump terms

            // loop over cells touching that particle
            local_index_type num_neighbors = (i<nlocal) ? _physics->_cell_particles_neighborhood->getNumNeighbors(i)
                : _physics->_halo_cell_particles_neighborhood->getNumNeighbors(halo_i);
            for (local_index_type j=0; j<num_neighbors; ++j) {
                auto particle_j = (i<nlocal) ? _physics->_cell_particles_neighborhood->getNeighbor(i,j)
                    : _physics->_halo_cell_particles_neighborhood->getNeighbor(halo_i,j);
                // particle_j is an index from 0..nlocal+nhalo

                // filter out (skip) all halo particles that are not a local DOF
                if (particle_j >= nlocal) continue; // particle is in halo region, but we only want dofs that are rows (locally owned))


                local_index_type row = local_to_dof_map(particle_j, field_one, comp);

                // add the contribution to the row for the particle
                double contribution = 0;
                if (l2_op) {
                    for (int q=0; q<_physics->_weights_ndim; ++q) {
                        auto q_type = (i<nlocal) ? quadrature_type(i,q) : halo_quadrature_type(_physics->_halo_small_to_big(halo_i,0),q);
                        if (q_type==1) { // interior
                            // integrating RHS function
                            double v = (i<nlocal) ? _physics->_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, j, q+1)
                                : _physics->_halo_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, halo_i, j, q+1);

                            xyz_type pt = (i<nlocal) ? xyz_type(quadrature_points(i,2*q),quadrature_points(i,2*q+1),0) 
                                : xyz_type(halo_quadrature_points(_physics->_halo_small_to_big(halo_i,0),2*q), halo_quadrature_points(_physics->_halo_small_to_big(halo_i,0),2*q+1),0);

                            double q_wt = (i<nlocal) ? quadrature_weights(i,q) : halo_quadrature_weights(_physics->_halo_small_to_big(halo_i,0),q);

                            contribution += q_wt * v * function->evalScalar(pt);
                        } 
                    }
                } else {
                    for (int q=0; q<_physics->_weights_ndim; ++q) {
                        auto q_type = (i<nlocal) ? quadrature_type(i,q) : halo_quadrature_type(_physics->_halo_small_to_big(halo_i,0),q);
                        double v = (i<nlocal) ? _physics->_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, j, q+1)
                            : _physics->_halo_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, halo_i, j, q+1);
                        double q_wt = (i<nlocal) ? quadrature_weights(i,q) : halo_quadrature_weights(_physics->_halo_small_to_big(halo_i,0),q);
                        xyz_type pt = (i<nlocal) ? xyz_type(quadrature_points(i,2*q),quadrature_points(i,2*q+1),0) 
                            : xyz_type(halo_quadrature_points(_physics->_halo_small_to_big(halo_i,0),2*q), halo_quadrature_points(_physics->_halo_small_to_big(halo_i,0),2*q+1),0);

                        if (q_type==1) { // interior
                            double rhs_eval = 0;
                            if (rd_op) {
                                // integrating RHS function
                                if (_parameters->get<Teuchos::ParameterList>("physics").get<std::string>("solution")=="polynomial") {
                                    auto cast_to_poly_2 = dynamic_cast<SecondOrderBasis*>(function.getRawPtr());
                                    rhs_eval = cast_to_poly_2->evalReactionDiffusionRHS(pt,_physics->_reaction,_physics->_diffusion);
                                } else if (_parameters->get<Teuchos::ParameterList>("physics").get<std::string>("solution")=="polynomial_3") {
                                    auto cast_to_poly_3 = dynamic_cast<ThirdOrderBasis*>(function.getRawPtr());
                                    rhs_eval = cast_to_poly_3->evalReactionDiffusionRHS(pt,_physics->_reaction,_physics->_diffusion);
                                } else {
                                    auto cast_to_sine = dynamic_cast<SineProducts*>(function.getRawPtr());
                                    rhs_eval = cast_to_sine->evalReactionDiffusionRHS(pt,_physics->_reaction,_physics->_diffusion);
                                }
                            } else if (le_op) {
                                if (_parameters->get<Teuchos::ParameterList>("physics").get<std::string>("solution")=="polynomial") {
                                    auto cast_to_poly_2 = dynamic_cast<SecondOrderBasis*>(function.getRawPtr());
                                    rhs_eval = cast_to_poly_2->evalLinearElasticityRHS(pt,comp,_physics->_shear,_physics->_lambda);
                                } else if (_parameters->get<Teuchos::ParameterList>("physics").get<std::string>("solution")=="polynomial_3") {
                                    auto cast_to_poly_3 = dynamic_cast<ThirdOrderBasis*>(function.getRawPtr());
                                    rhs_eval = cast_to_poly_3->evalLinearElasticityRHS(pt,comp,_physics->_shear,_physics->_lambda);
                                } else {
                                    auto cast_to_sine = dynamic_cast<SineProducts*>(function.getRawPtr());
                                    rhs_eval = cast_to_sine->evalLinearElasticityRHS(pt,comp,_physics->_shear,_physics->_lambda);
                                }
                            }
                            contribution += q_wt * v * rhs_eval;
                        } 
                        else if (q_type==2) { // edge on exterior
                            // DG enforcement of Dirichlet BCs
                            // penalties for edges of mass
                            double v_x = (i<nlocal) ? _physics->_gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 0, j, q+1)
                                : _physics->_halo_gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, halo_i, 0, j, q+1);
                            double v_y = (i<nlocal) ? _physics->_gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, 1, j, q+1)
                                : _physics->_halo_gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, halo_i, 1, j, q+1);
                            double n_x = (i<nlocal) ? unit_normals(i,2*q+0) : halo_unit_normals(_physics->_halo_small_to_big(halo_i,0), 2*q+0);
                            double n_y = (i<nlocal) ? unit_normals(i,2*q+1) : halo_unit_normals(_physics->_halo_small_to_big(halo_i,0), 2*q+1);
                            double exact_eval = 0;
                            if (rd_op) {
                                exact_eval = function->evalScalar(pt);
                            } else if (le_op) {
                                auto exact = function->evalVector(pt);
                                exact_eval = exact[comp];
                            }
                            contribution += penalty * q_wt * v * exact_eval;

                            if (rd_op) {
                                contribution -= _physics->_diffusion * q_wt * ( n_x * v_x + n_y * v_y) * exact_eval;
                            } else if (le_op) {
                                auto exact = function->evalVector(pt);
                                contribution -= q_wt * (
                                      2 * _physics->_shear * (n_x*v_x*(comp==0) + 0.5*n_y*(v_y*(comp==0) + v_x*(comp==1))) * exact[0]  
                                    + 2 * _physics->_shear * (n_y*v_y*(comp==1) + 0.5*n_x*(v_x*(comp==1) + v_y*(comp==0))) * exact[1]
                                    + _physics->_lambda * (v_x*(comp==0) + v_y*(comp==1)) * (n_x*exact[0] + n_y*exact[1]));
                            }
                        }
                    }

                }
                
                rhs_vals_atomic(row,0) += contribution;
                //rhs_vals(row,0) += contribution;
            }
        }
    });
    //});
    // DIAGNOSTIC:: loop particles locally owned
    //for (int i=0; i<_physics->_cells->getCoordsConst()->nLocal(); ++i) {
    //    bool i_found_in_cell_list = false;
    //    local_index_type num_neighbors = _physics->_cell_particles_neighborhood->getNumNeighbors(i);
    //    for (local_index_type j=0; j<num_neighbors; ++j) {
    //        auto cell_j = _physics->_cell_particles_neighborhood->getNeighbor(i,j);
    //        // this cell_j is in support of particle i
    //        
    //        auto halo_j = (cell_j<nlocal) ? -1 : _physics->_halo_big_to_small(cell_j-nlocal,0);
    //        if (cell_j>=nlocal /* halo */  && halo_j<0 /* not one found within max_h of locally owned particles */) continue;

    //        local_index_type num_neighbors_j = (cell_j<nlocal) ? _physics->_cell_particles_neighborhood->getNumNeighbors(cell_j)
    //            : _physics->_halo_cell_particles_neighborhood->getNumNeighbors(halo_j);
    //        for (local_index_type k=0; k<num_neighbors_j; ++k) {
    //            auto particle_k = (cell_j<nlocal) ? _physics->_cell_particles_neighborhood->getNeighbor(cell_j,k)
    //                : _physics->_halo_cell_particles_neighborhood->getNeighbor(halo_j,k);
    //            if (particle_k==i) i_found_in_cell_list = true;
    //        }
    //    }
    //    printf("%d: %d\n", i, i_found_in_cell_list);
    //    TEUCHOS_ASSERT(i_found_in_cell_list);
    //}


    if (_parameters->get<Teuchos::ParameterList>("physics").get<std::string>("solution")=="polynomial") {
        auto reference_global_force = 22.666666666666666666;//4.0;//1.9166666666666665;//0.21132196999014932;
        double force = 0;
        for (int i=0; i<_physics->_cells->getCoordsConst()->nLocal(); ++i) {
            local_index_type row = local_to_dof_map(i, field_one, 0 /* component 0*/);
            force += rhs_vals(row,0);
        }
	    scalar_type global_force;
	    Teuchos::Ptr<scalar_type> global_force_ptr(&global_force);
	    Teuchos::reduceAll<int, scalar_type>(*(_physics->_cells->getCoordsConst()->getComm()), Teuchos::REDUCE_SUM, force, global_force_ptr);
        if (_physics->_cells->getCoordsConst()->getComm()->getRank()==0) {
            printf("Global RHS SUM calculated: %.16f, reference RHS SUM: %.16f\n", global_force, reference_global_force);
        }
    }
}

std::vector<InteractingFields> ReactionDiffusionSources::gatherFieldInteractions() {
	std::vector<InteractingFields> field_interactions;
    //field_interactions.push_back(InteractingFields(op_needing_interaction::source, _particles->getFieldManagerConst()->getIDOfFieldFromName("solution")));
	return field_interactions;
}

}
