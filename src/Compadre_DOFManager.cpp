#include "Compadre_DOFManager.hpp"

#include "Compadre_CoordsT.hpp"
#include "Compadre_ParticlesT.hpp"
#include "Compadre_FieldT.hpp"
#include "Compadre_FieldManager.hpp"
#include "Compadre_DOFData.hpp"

namespace Compadre {

DOFManager::DOFManager (const particles_type* particles, Teuchos::RCP<Teuchos::ParameterList> parameters) :
		_parameters(parameters), _field_manager(particles->getFieldManagerConst()),
		_fields(_field_manager->getVectorOfFields()), _particles(particles)
{
	_particle_dof_data = this->generateDOFMap();
}

// dof manager
// row map and column map produced through dof manager

Teuchos::RCP<const DOFManager::dof_data_type> DOFManager::generateDOFMap(std::vector<local_index_type> field_numbers) {
	// generates dof_row_map and dof_col_map
	// can be used to either construct the maps for all particles and fields,
	// which is used by the ParticlesT class, or can generate a DOFData
	// object with a subset of these fields, which is useful for row
	// and column maps need for linear algebra objects

	Teuchos::RCP<DOFManager::dof_data_type> dof_data;
	if (field_numbers.size()==0) {
		if (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")) {
			dof_data = Teuchos::rcp<DOFManager::dof_data_type>(new DOFManager::dof_data_type(_parameters, _fields.size()));
		} else {
			dof_data = Teuchos::rcp<DOFManager::dof_data_type>(new DOFManager::dof_data_type(_parameters));
		}
	} else {
		dof_data = Teuchos::rcp<DOFManager::dof_data_type>(new DOFManager::dof_data_type(_parameters, field_numbers.size()));
	}

	typedef Kokkos::View<global_index_type*> gid_view_type;
	typedef Kokkos::View<const global_index_type*> const_gid_view_type;

	Teuchos::RCP<const map_type> particle_map = _particles->getCoordsConst()->getMapConst(false /* no halo */);
	const_gid_view_type particle_gids_locally_owned = particle_map->getMyGlobalIndices();

	Teuchos::RCP<const map_type> halo_particle_map = _particles->getCoordsConst()->getMapConst(true /* only halo */);
	const_gid_view_type halo_particle_gids_locally_owned;
	local_index_type num_halo_particle_gids = 0;

	if (!(halo_particle_map.is_null())) {
		halo_particle_gids_locally_owned = halo_particle_map->getMyGlobalIndices();
		num_halo_particle_gids = halo_particle_gids_locally_owned.dimension_0();
	}
	// above is particles and halo particles, but not DOFS

	// if no specific fields are requested, then all fields are requested
	if (field_numbers.size()==0) {
		field_numbers.resize(_fields.size());
		for (size_t i=0; i<_fields.size(); i++) {
			field_numbers[i] = i;
		}
	}

	std::vector<std::vector<std::vector<local_index_type> > > particle_field_component_to_dof_map = std::vector<std::vector<std::vector<local_index_type> > >(particle_gids_locally_owned.dimension_0() + num_halo_particle_gids, std::vector<std::vector<local_index_type> >(_fields.size()));

	std::vector<gid_view_type> row_map_gids(field_numbers.size());
	std::vector<gid_view_type> col_map_gids(field_numbers.size());

	local_index_type field_dimensions_total_size = 0;
	for (size_t i=0; i<field_numbers.size(); i++) {
		field_dimensions_total_size += _fields[field_numbers[i]]->nDim();
	}

	if (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")) {
		for (size_t i=0; i<field_numbers.size(); i++) {
            // check field type here to determine if particle_gids_locally_owned is appropriate
			row_map_gids[i] = gid_view_type("row gids", _fields[field_numbers[i]]->nDim() * particle_gids_locally_owned.dimension_0());
			col_map_gids[i] = gid_view_type("col gids", _fields[field_numbers[i]]->nDim() * (particle_gids_locally_owned.dimension_0() + num_halo_particle_gids) );
		}
	} else {
        //    check field type here to determine if particle_gids_locally_owned is appropriate
		row_map_gids[0] = gid_view_type("row gids", field_dimensions_total_size * particle_gids_locally_owned.dimension_0());
		col_map_gids[0] = gid_view_type("col gids", field_dimensions_total_size * (particle_gids_locally_owned.dimension_0() + num_halo_particle_gids) );
	}

	if (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")) {
		// copy data to build column map
//		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,particle_gids_locally_owned.dimension_0()), KOKKOS_LAMBDA(const int i) {
		for (size_t i=0; i<particle_gids_locally_owned.dimension_0(); ++i) {
			local_index_type field_j_offset = 0;
			for (size_t j=0; j<field_numbers.size(); j++) {
				particle_field_component_to_dof_map[i][field_numbers[j]].resize(_fields[field_numbers[j]]->nDim());
				for (local_index_type k=0; k<_fields[field_numbers[j]]->nDim(); k++) {
					global_index_type global_value = particle_gids_locally_owned(i)*field_dimensions_total_size + field_j_offset + k;
					local_index_type local_value   = i*_fields[field_numbers[j]]->nDim() + k;

					particle_field_component_to_dof_map[i][field_numbers[j]][k] = local_value;
					row_map_gids[j](local_value) = global_value;
					col_map_gids[j](local_value) = global_value;
				}
				field_j_offset += _fields[field_numbers[j]]->nDim();
			}
//		});
		}
		local_index_type offset = particle_gids_locally_owned.dimension_0();
//		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,num_halo_particle_gids), KOKKOS_LAMBDA(const int i) {
		for (local_index_type i=0; i<num_halo_particle_gids; ++i) {
			local_index_type field_j_offset = 0;
			for (size_t j=0; j<field_numbers.size(); j++) {
				particle_field_component_to_dof_map[i+offset][field_numbers[j]].resize(_fields[field_numbers[j]]->nDim());
				for (local_index_type k=0; k<_fields[field_numbers[j]]->nDim(); k++) {
					global_index_type global_value = halo_particle_gids_locally_owned(i)*field_dimensions_total_size + field_j_offset + k;
					local_index_type local_value   = (i+offset)*_fields[field_numbers[j]]->nDim() + k;

					particle_field_component_to_dof_map[i+offset][field_numbers[j]][k] = local_value;
					col_map_gids[j](local_value) = global_value;
				}
				field_j_offset += _fields[field_numbers[j]]->nDim();
			}
//		});
		}
	} else {
		// copy data to build column map
//		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,particle_gids_locally_owned.dimension_0()), KOKKOS_LAMBDA(const int i) {
		for (size_t i=0; i<particle_gids_locally_owned.extent(0); ++i) {
			local_index_type field_j_offset = 0;
			for (size_t j=0; j<field_numbers.size(); j++) {
				particle_field_component_to_dof_map[i][field_numbers[j]].resize(_fields[field_numbers[j]]->nDim()); // global over fields
				for (local_index_type k=0; k<_fields[field_numbers[j]]->nDim(); k++) {
					global_index_type global_value = particle_gids_locally_owned(i)*field_dimensions_total_size + field_j_offset + k;
					local_index_type local_value   = i*field_dimensions_total_size + field_j_offset + k;

					particle_field_component_to_dof_map[i][field_numbers[j]][k] = local_value;
					row_map_gids[0](local_value) = global_value;
					col_map_gids[0](local_value) = global_value;
				}
				field_j_offset += _fields[field_numbers[j]]->nDim();
			}
//		});
		}
		local_index_type offset = particle_gids_locally_owned.extent(0);
//		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,num_halo_particle_gids), KOKKOS_LAMBDA(const int i) {
		for (local_index_type i=0; i<num_halo_particle_gids; ++i) {
			local_index_type field_j_offset = 0;
			for (size_t j=0; j<field_numbers.size(); j++) {
				particle_field_component_to_dof_map[i+offset][field_numbers[j]].resize(_fields[field_numbers[j]]->nDim()); // global over fields
				for (local_index_type k=0; k<_fields[field_numbers[j]]->nDim(); k++) {
					global_index_type global_value = halo_particle_gids_locally_owned(i)*field_dimensions_total_size + field_j_offset + k;
					local_index_type local_value   = (i+offset)*field_dimensions_total_size + field_j_offset + k;

					particle_field_component_to_dof_map[i+offset][field_numbers[j]][k] = local_value;
					col_map_gids[0](local_value) = global_value;
				}
				field_j_offset += _fields[field_numbers[j]]->nDim();
			}
//		});
		}
	}



	if (_parameters->get<Teuchos::ParameterList>("solver").get<bool>("blocked")) {

        global_index_type field_row_offset = 0;

		std::vector<Teuchos::RCP<map_type> > dof_row_map = std::vector<Teuchos::RCP<map_type> >(field_numbers.size());
		std::vector<Teuchos::RCP<map_type> > dof_col_map = std::vector<Teuchos::RCP<map_type> >(field_numbers.size());
		for (size_t i=0; i<field_numbers.size(); i++) {

            auto t_row_map_gids = row_map_gids[i];
            auto t_col_map_gids = col_map_gids[i];

            // increment  row_map_gids and col_map_gids by row offsets
            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,t_row_map_gids.extent(0)), 
                    KOKKOS_LAMBDA(const int j) {
                t_row_map_gids(j) += field_row_offset;
            });
            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,t_col_map_gids.extent(0)), 
                    KOKKOS_LAMBDA(const int j) {
                t_col_map_gids(j) += field_row_offset;
            });

			dof_row_map[i] = Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
															row_map_gids[i],
															field_row_offset,
															_particles->getCoordsConst()->getComm()));

			dof_col_map[i] = Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
															col_map_gids[i],
															field_row_offset,
															_particles->getCoordsConst()->getComm()));

            field_row_offset += dof_row_map[i]->getGlobalNumElements();

			dof_data->setRowMap(dof_row_map[i], i);
			dof_data->setColMap(dof_col_map[i], i);
		}
	} else {
		std::vector<Teuchos::RCP<map_type> > dof_row_map = std::vector<Teuchos::RCP<map_type> >(1);
		std::vector<Teuchos::RCP<map_type> > dof_col_map = std::vector<Teuchos::RCP<map_type> >(1);
		dof_row_map[0] = Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
														row_map_gids[0],
														0,
														_particles->getCoordsConst()->getComm()));

		dof_col_map[0] = Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
														col_map_gids[0],
														0,
														_particles->getCoordsConst()->getComm()));

		dof_data->setRowMap(dof_row_map[0]);
		dof_data->setColMap(dof_col_map[0]);
	}

	dof_data->setDOFMap(particle_field_component_to_dof_map);

	return dof_data;
}

}
