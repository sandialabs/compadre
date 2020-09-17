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

    auto cell_vertices = _physics->_cells->getFieldManager()->getFieldByName("vertex_points")->getMultiVectorPtr()->getLocalView<host_view_type>();
    if (field_one != field_two) return;
    TEUCHOS_TEST_FOR_EXCEPT_MSG(_b==NULL, "Tpetra Multivector for RHS not yet specified.");

    // get problem specific parameters, operators and manufactured solution functions from _physics
 
    bool _l2_op = _physics->_l2_op;
    bool _rd_op = _physics->_rd_op;
    bool _le_op = _physics->_le_op;
    bool _vl_op = _physics->_vl_op;
    bool _st_op = _physics->_st_op;
    bool _mix_le_op = _physics->_mix_le_op;

    bool _velocity_basis_type_vector = _physics->_velocity_basis_type_vector;

    bool _use_vector_gmls = _physics->_use_vector_gmls;
    bool _use_vector_grad_gmls = _physics->_use_vector_grad_gmls;
    bool _use_sip = _physics->_use_sip;
    bool _use_vms = _physics->_use_vms;
    bool _use_neighbor_weighting = _physics->_use_neighbor_weighting;
    bool _use_side_weighting = _physics->_use_side_weighting;

    int _velocity_field_id = _physics->_velocity_field_id;
    int _pressure_field_id = _physics->_pressure_field_id;
    int _lagrange_field_id = _physics->_lagrange_field_id;
    if (field_one == _lagrange_field_id) return; // no work to do for lagrange multiplier RHS

    AnalyticFunction* velocity_function = _physics->_velocity_function;
    AnalyticFunction* pressure_function = _physics->_pressure_function;

    local_index_type _ndim_requested = _physics->_ndim_requested;
    if (_ndim_requested==3) printf("RHS started.\n");

    host_view_type rhs_vals = this->_b->getLocalView<host_view_type>();
    const local_index_type nlocal_cells = static_cast<local_index_type>(_physics->_cells->getCoords()->nLocal());
    const local_index_type nlocal_particles = static_cast<local_index_type>(_physics->getParticles()->getCoords()->nLocal());
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

    auto base_penalty = (_parameters->get<Teuchos::ParameterList>("remap").get<int>("porder")+1)*_parameters->get<Teuchos::ParameterList>("physics").get<double>("penalty");
    if (_use_neighbor_weighting) base_penalty /= (_ndim_requested==2) ? _physics->_cell_particles_max_h : 1.772453850905516*_physics->_cell_particles_max_h; // either h or sqrt(pi)*h

    // each element must have the same number of sides
    auto num_sides = adjacent_elements.extent(1);
    auto num_interior_quadrature = 0;
    int _weights_ndim = _physics->_weights_ndim;
    for (int q=0; q<_weights_ndim; ++q) {
        if (quadrature_type(0,q)!=1) { // some type of side quadrature
            num_interior_quadrature = q;
            break;
        }
    }
    auto num_exterior_quadrature_per_side = (_weights_ndim - num_interior_quadrature)/num_sides;

    //if (field_one == field_two && field_one == _velocity_field_id) {
    //    // zero out rhs before starting
    //    Kokkos::deep_copy(rhs_vals, 0.0);
    //}

    // get an unmanaged atomic view that can be added to in a parallel for loop
    Kokkos::View<scalar_type**, decltype(rhs_vals)::memory_space, Kokkos::MemoryTraits<Kokkos::Atomic|Kokkos::Unmanaged> > 
        rhs_vals_atomic(rhs_vals.data(), rhs_vals.extent(0), rhs_vals.extent(1));


    // assembly over cells, then particles
    // loop over cells including halo cells 
    //for (int i=0; i<_physics->_cells->getCoordsConst()->nLocal(true /* include halo in count */); ++i) {
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,_physics->_cells->getCoordsConst()->nLocal(true /* include halo in count */)), KOKKOS_LAMBDA(const int i) {
    //for (int i=0; i<_physics->_cells->getCoordsConst()->nLocal(); ++i) {
        std::vector<double> side_lengths(num_sides);
        {
            for (int current_side_num = 0; current_side_num < num_sides; ++current_side_num) {
               for (int q=0; q<_weights_ndim; ++q) {
                    auto q_type = quadrature_type(i,q);
                    if (q_type!=1) { // side
                        auto q_wt = quadrature_weights(i,q);
                        int side_num = (q - num_interior_quadrature)/num_exterior_quadrature_per_side;
                        if (side_num==current_side_num) {
                            side_lengths[current_side_num] += q_wt;
                        }
                    }
                }
            }
        }
        for (local_index_type comp_out = 0; comp_out < fields[field_one]->nDim(); ++comp_out) {
        for (local_index_type comp_in = 0; comp_in < fields[field_one]->nDim(); ++comp_in) {
            if ((comp_out!=comp_in) && ((!_velocity_basis_type_vector && field_one == _velocity_field_id) || (field_one == _pressure_field_id))) continue;

            // filter out (skip) all halo cells that are not able to be seen from a locally owned particle
            auto halo_i = (i<nlocal_cells) ? -1 : _physics->_halo_big_to_small(i-nlocal_cells,0);
            if (i>=nlocal_cells /* halo */  && halo_i<0 /* not one found within max_h of locally owned particles */) return;//continue;

            //TEUCHOS_ASSERT(i<nlocal);
            // need to have a halo gmls object that can be queried for cells that do not live on processor
            // this is also needed for the operator file in order to finish jump terms

            // loop over cells touching that particle
            local_index_type num_neighbors = (i<nlocal_cells) ? _physics->_cell_particles_neighborhood->getNumNeighbors(i)
                : _physics->_halo_cell_particles_neighborhood->getNumNeighbors(halo_i);
            for (local_index_type j=0; j<num_neighbors; ++j) {
                auto particle_j = (i<nlocal_cells) ? _physics->_cell_particles_neighborhood->getNeighbor(i,j)
                    : _physics->_halo_cell_particles_neighborhood->getNeighbor(halo_i,j);
                // particle_j is an index from 0..nlocal+nhalo

                // filter out (skip) all halo particles that are not a local DOF
                if (particle_j >= nlocal_particles) continue; // particle is in halo region, but we only want dofs that are rows (locally owned))


                local_index_type row = local_to_dof_map(particle_j, field_one, comp_in);

                // add the contribution to the row for the particle
                double contribution = 0;
                if (_l2_op) {
                    for (int qn=0; qn<_weights_ndim; ++qn) {
                        auto q_type = (i<nlocal_cells) ? quadrature_type(i,qn) : halo_quadrature_type(_physics->_halo_small_to_big(halo_i,0),qn);
                        if (q_type==1) { // interior
                            // integrating RHS velocity_function
                            double v;
                            if (_use_vector_gmls) {
                                v = (i<nlocal_cells) ? _physics->_vel_gmls->getAlpha1TensorTo1Tensor(TargetOperation::VectorPointEvaluation, i, comp_out, j, comp_in, qn+1)
                                    : _physics->_halo_vel_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, halo_i, j, qn+1);
                            } else {
                                v = (i<nlocal_cells) ? _physics->_vel_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, j, qn+1)
                                    : _physics->_halo_vel_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, halo_i, j, qn+1);
                            }

                            xyz_type pt = (i<nlocal_cells) ? xyz_type(quadrature_points(i,_ndim_requested*qn),
                                                                quadrature_points(i,_ndim_requested*qn+1),
                                                                (_ndim_requested==3) ? quadrature_points(i,_ndim_requested*qn+2) : 0) 
                                : xyz_type(halo_quadrature_points(_physics->_halo_small_to_big(halo_i,0),_ndim_requested*qn), 
                                               halo_quadrature_points(_physics->_halo_small_to_big(halo_i,0),_ndim_requested*qn+1),
                                               (_ndim_requested==3) ? quadrature_points(i,_ndim_requested*qn+2) : 0);

                            double q_wt = (i<nlocal_cells) ? quadrature_weights(i,qn) : halo_quadrature_weights(_physics->_halo_small_to_big(halo_i,0),qn);

                            contribution += q_wt * v * velocity_function->evalScalar(pt);
                        } 
                    }
                } else {
                    for (int qn=0; qn<_weights_ndim; ++qn) {
                        auto q_type = (i<nlocal_cells) ? quadrature_type(i,qn) : halo_quadrature_type(_physics->_halo_small_to_big(halo_i,0),qn);
                        double v;
                        if (_use_vector_gmls) {
                            v = (i<nlocal_cells) ? _physics->_vel_gmls->getAlpha1TensorTo1Tensor(TargetOperation::VectorPointEvaluation, i, comp_out, j, comp_in, qn+1)
                            : _physics->_halo_vel_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, halo_i, j, qn+1);
                        }
                        else {
                            v = (i<nlocal_cells) ? _physics->_vel_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, j, qn+1)
                            : _physics->_halo_vel_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, halo_i, j, qn+1);
                        }
                        double q_wt = (i<nlocal_cells) ? quadrature_weights(i,qn) : halo_quadrature_weights(_physics->_halo_small_to_big(halo_i,0),qn);
                        xyz_type pt = (i<nlocal_cells) ? xyz_type(quadrature_points(i,_ndim_requested*qn),
                                                            quadrature_points(i,_ndim_requested*qn+1),
                                                            (_ndim_requested==3) ? quadrature_points(i,_ndim_requested*qn+2) : 0) 
                            : xyz_type(halo_quadrature_points(_physics->_halo_small_to_big(halo_i,0),_ndim_requested*qn), 
                                       halo_quadrature_points(_physics->_halo_small_to_big(halo_i,0),_ndim_requested*qn+1),
                                       (_ndim_requested) ? quadrature_points(i,_ndim_requested*qn+2) : 0);

                        if (q_type==1) { // interior
                            double rhs_eval = 0;
                            if (_rd_op) {
                                rhs_eval = velocity_function->evalScalarReactionDiffusionRHS(pt,_physics->_reaction,_physics->_diffusion);
                            } else if (_le_op || _mix_le_op) {
                                if (_velocity_field_id == field_one && _velocity_field_id == field_two) {
                                    rhs_eval = velocity_function->evalLinearElasticityRHS(pt,comp_out,_physics->_shear,_physics->_lambda);
                                }
                            } else if (_vl_op) {
                                rhs_eval = velocity_function->evalScalarLaplacian(pt, comp_out);
                                rhs_eval *= -_physics->_shear;
                            } else if (_st_op) {
                                if (_velocity_field_id == field_one && _velocity_field_id == field_two) {
                                    rhs_eval = velocity_function->evalScalarLaplacian(pt, comp_out);
                                    rhs_eval *= -_physics->_diffusion;
                                    xyz_type grad_p_vals = pressure_function->evalScalarDerivative(pt);
                                    rhs_eval += grad_p_vals[comp_out];
                                }
                            }
                            contribution += q_wt * v * rhs_eval;
                        } 
                        else if (q_type==2) { // side on exterior
                            int current_side_num = (qn - num_interior_quadrature)/num_exterior_quadrature_per_side;
                            double penalty = (_use_side_weighting) ? base_penalty/side_lengths[current_side_num] : base_penalty;

                            // DG enforcement of Dirichlet BCs
                            XYZ grad_v;
                            if (_use_vector_grad_gmls) {
                                for (int d=0; d<_ndim_requested; ++d) {
                                    grad_v[d] = (i<nlocal_cells) ? _physics->_vel_gmls->getAlpha1TensorTo2Tensor(TargetOperation::GradientOfVectorPointEvaluation, i, comp_out, d, j, comp_in, qn+1)
                                        : _physics->_halo_vel_gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, halo_i, d, j, qn+1);
                                }
                            } else {
                                for (int d=0; d<_ndim_requested; ++d) {
                                    grad_v[d] = (i<nlocal_cells) ? _physics->_vel_gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, i, d, j, qn+1)
                                        : _physics->_halo_vel_gmls->getAlpha0TensorTo1Tensor(TargetOperation::GradientOfScalarPointEvaluation, halo_i, d, j, qn+1);
                                }
                            }
                            XYZ n;
                            for (int d=0; d<_ndim_requested; ++d) {
                                n[d] = (i<nlocal_cells) ? unit_normals(i,_ndim_requested*qn+d) : halo_unit_normals(_physics->_halo_small_to_big(halo_i,0), _ndim_requested*qn+d);
                            }
                            double exact_eval = 0;

                            if (_rd_op) {
                                exact_eval = velocity_function->evalScalar(pt);
                                contribution += penalty * q_wt * v * exact_eval;
                                double n_dot_grad = 0.0;
                                for (int d=0; d<_ndim_requested; ++d) {
                                    n_dot_grad += n[d]*grad_v[d];
                                }
                                contribution -= _physics->_diffusion * q_wt * n_dot_grad * exact_eval;
                            } else if (_le_op) {
                                auto exact = velocity_function->evalVector(pt);

                                if (_use_vms) {
                                    int current_side_num_in_current_cell = (qn - num_interior_quadrature)/num_exterior_quadrature_per_side;
                                    Teuchos :: SerialDenseMatrix <local_index_type, scalar_type> tau_side(_ndim_requested, _ndim_requested);
                                    Teuchos :: SerialDenseSolver <local_index_type, scalar_type> vmsdg_solver;
                                    tau_side.putScalar(0.0);
                                    for (int j1 = 0; j1 < _ndim_requested; ++j1) {
                                        for (int j2 = 0; j2 < _ndim_requested; ++j2) {
                                            tau_side(j1, j2) = _physics->_tau(i, current_side_num_in_current_cell, j1, j2);    //tau^(1)_s from 'current' elem
                                        }
                                    }
                                    vmsdg_solver.setMatrix(Teuchos::rcp(&tau_side, false));
                                    auto info = vmsdg_solver.invert();
                                    double vmsBCfactor = 2.0;
                                    contribution += q_wt * v * vmsBCfactor * ( tau_side(comp_out, 0) * velocity_function->evalScalar(pt, 0) + 
                                                                               tau_side(comp_out, 1) * velocity_function->evalScalar(pt, 1) );
                                } else if (_use_sip) {
                                    contribution += penalty * q_wt * v * exact[comp_out];
                                }

                                double n_dot_exact = 0.0;
                                for (int d=0; d<_ndim_requested; ++d) {
                                    n_dot_exact += n[d] * exact[d];
                                }
                                double mu_sigma_v_dot_n_dot_exact = 0.0;
                                for (int d1=0; d1<_ndim_requested; ++d1) {
                                    for (int d2=0; d2<_ndim_requested; ++d2) {
                                        mu_sigma_v_dot_n_dot_exact += 0.5*n[d1]*exact[d2]*(grad_v[d1]*(comp_out==d2) + grad_v[d2]*(comp_out==d1));
                                    }
                                }
                                double lambda_sigma_v_dot_n_dot_exact = 0.0;
                                for (int d=0; d<_ndim_requested; ++d) {
                                    lambda_sigma_v_dot_n_dot_exact += grad_v[d]*(comp_out==d) * n_dot_exact;
                                }
                                contribution -= q_wt * 2 * _physics->_shear * mu_sigma_v_dot_n_dot_exact;
                                contribution -= q_wt * _physics->_lambda * lambda_sigma_v_dot_n_dot_exact;
                                //contribution -= q_wt * (
                                //      2 * _physics->_shear * (n[0]*grad_v[0]*(comp_out==0) + 0.5*n[1]*(grad_v[1]*(comp_out==0) + grad_v[0]*(comp_out==1))) * exact[0]  
                                //    + 2 * _physics->_shear * (n[1]*grad_v[1]*(comp_out==1) + 0.5*n[0]*(grad_v[0]*(comp_out==1) + grad_v[1]*(comp_out==0))) * exact[1]
                                //    + _physics->_lambda * (grad_v[0]*(comp_out==0) + grad_v[1]*(comp_out==1)) * (n[0]*exact[0] + n[1]*exact[1]));
                            } else if (_mix_le_op) {
                                if ((_velocity_field_id == field_one) && (_velocity_field_id == field_two)) {
                                    exact_eval = velocity_function->evalScalar(pt, comp_out);
                                    contribution += penalty * q_wt * v * exact_eval;
                                    auto exact = velocity_function->evalVector(pt);
                                    double n_dot_exact = 0.0;
                                    for (int d=0; d<_ndim_requested; ++d) {
                                        n_dot_exact += n[d] * exact[d];
                                    }
                                    double mu_sigma_v_dot_n_dot_exact = 0.0;
                                    for (int d1=0; d1<_ndim_requested; ++d1) {
                                        for (int d2=0; d2<_ndim_requested; ++d2) {
                                            mu_sigma_v_dot_n_dot_exact += 0.5*n[d1]*exact[d2]*(grad_v[d1]*(comp_out==d2) + grad_v[d2]*(comp_out==d1));
                                        }
                                    }
                                    double lambda_sigma_v_dot_n_dot_exact = 0.0;
                                    for (int d=0; d<_ndim_requested; ++d) {
                                        lambda_sigma_v_dot_n_dot_exact += grad_v[d]*(comp_out==d) * n_dot_exact;
                                    }
                                    contribution += q_wt * 2./(scalar_type)(_ndim_requested) * _physics->_shear * lambda_sigma_v_dot_n_dot_exact;
                                    contribution -= q_wt * 2 * _physics->_shear * mu_sigma_v_dot_n_dot_exact;
                                    //contribution -= q_wt * (
                                    //      2 * _physics->_shear * (n[0]*grad_v[0]*(comp_out==0) + 0.5*n[1]*(grad_v[1]*(comp_out==0) + grad_v[0]*(comp_out==1))) * exact[0]  
                                    //    + 2 * _physics->_shear * (n[1]*grad_v[1]*(comp_out==1) + 0.5*n[0]*(grad_v[0]*(comp_out==1) + grad_v[1]*(comp_out==0))) * exact[1]
                                    //    - 2./(scalar_type)(_ndim_requested) * _physics->_shear * (grad_v[0]*(comp_out==0) + grad_v[1]*(comp_out==1)) * (n[0]*exact[0] + n[1]*exact[1]));
                                } else if (_pressure_field_id == field_one && _pressure_field_id == field_two) {
                                    double q = (i<nlocal_cells) ? _physics->_pressure_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, j, qn+1)
                                        : _physics->_halo_pressure_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, halo_i, j, qn+1);
                                    auto exact = velocity_function->evalVector(pt);
                                    double n_dot_exact = 0.0;
                                    for (int d=0; d<_ndim_requested; ++d) {
                                        n_dot_exact += n[d] * exact[d];
                                    }
                                    contribution += q_wt * ( q * ( n_dot_exact ) );
                                    //contribution += q_wt * (
                                    //      pressure_coeff * q * ( n_x * exact[0] + n_y * exact[1] ) );
                                }
                            } else if (_vl_op) {
                                exact_eval = velocity_function->evalScalar(pt, comp_out);
                                contribution += penalty * q_wt * v * exact_eval;
                                auto exact = velocity_function->evalVector(pt);
                                double sigma_v_dot_n_dot_exact = 0.0;
                                for (int d1=0; d1<_ndim_requested; ++d1) {
                                    for (int d2=0; d2<_ndim_requested; ++d2) {
                                        sigma_v_dot_n_dot_exact += n[d1]*grad_v[d1]*(comp_out==d2) * exact[d2];
                                    }
                                }
                                contribution -= q_wt * _physics->_shear * sigma_v_dot_n_dot_exact;
                                //contribution -= q_wt * (
                                //      _physics->_shear * (n_x*v_x*(comp_out==0) + n_y*v_y*(comp_out==0)) * exact[0]  
                                //    + _physics->_shear * (n_x*v_x*(comp_out==1) + n_y*v_y*(comp_out==1)) * exact[1] );
                            } else if (_st_op) {
                                if ((_velocity_field_id == field_one) && (_velocity_field_id == field_two)) {
                                    exact_eval = velocity_function->evalScalar(pt, comp_out);
                                    contribution += penalty * q_wt * v * exact_eval;
                                    auto exact = velocity_function->evalVector(pt);
                                    double grad_v_dot_n_dot_exact = 0.0;
                                    for (int d1=0; d1<_ndim_requested; ++d1) {
                                        for (int d2=0; d2<_ndim_requested; ++d2) {
                                            grad_v_dot_n_dot_exact += n[d1]*grad_v[d1]*(comp_out==d2) * exact[d2];
                                        }
                                    }
                                    contribution -= q_wt * _physics->_diffusion * grad_v_dot_n_dot_exact;
                                    //contribution -= q_wt * (
                                    //      _physics->_diffusion * (n_x*v_x*(comp_out==0) + n_y*v_y*(comp_out==0)) * exact[0]  
                                    //    + _physics->_diffusion * (n_x*v_x*(comp_out==1) + n_y*v_y*(comp_out==1)) * exact[1] );
                                } else if (_pressure_field_id == field_one && _pressure_field_id == field_two) {
                                    double q = (i<nlocal_cells) ? _physics->_pressure_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, i, j, qn+1)
                                        : _physics->_halo_pressure_gmls->getAlpha0TensorTo0Tensor(TargetOperation::ScalarPointEvaluation, halo_i, j, qn+1);
                                    auto exact = velocity_function->evalVector(pt);
                                    double n_dot_exact = 0.0;
                                    for (int d=0; d<_ndim_requested; ++d) {
                                        n_dot_exact += n[d] * exact[d];
                                    }
                                    contribution += q_wt * (
                                          q * ( n_dot_exact ) );
                                }
                            }
                        }
                    }

                }
                
                rhs_vals_atomic(row,0) += contribution;
                //rhs_vals(row,0) += contribution;
            }
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


    //if (polynomial_solution && field_one==_physics->_velocity_field_id) {
    //    auto reference_global_force = 22.666666666666666666;//4.0;//1.9166666666666665;//0.21132196999014932;
    //    double force = 0;
    //    for (int i=0; i<_physics->_cells->getCoordsConst()->nLocal(); ++i) {
    //        local_index_type row = local_to_dof_map(i, field_one, 0 /* component 0*/);
    //        force += rhs_vals(row,0);
    //    }
    //    scalar_type global_force;
    //    Teuchos::Ptr<scalar_type> global_force_ptr(&global_force);
    //    Teuchos::reduceAll<int, scalar_type>(*(_physics->_cells->getCoordsConst()->getComm()), Teuchos::REDUCE_SUM, force, global_force_ptr);
    //    if (_physics->_cells->getCoordsConst()->getComm()->getRank()==0) {
    //        printf("Global RHS SUM calculated: %.16f, reference RHS SUM: %.16f\n", global_force, reference_global_force);
    //    }
    //}
}

std::vector<InteractingFields> ReactionDiffusionSources::gatherFieldInteractions() {
    std::vector<InteractingFields> field_interactions;
    //field_interactions.push_back(InteractingFields(op_needing_interaction::source, _particles->getFieldManagerConst()->getIDOfFieldFromName("solution")));
    return field_interactions;
}

}
