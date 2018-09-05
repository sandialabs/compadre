#include "Compadre_VTKInformation.hpp"

#include "Compadre_XyzVector.hpp"
#include "Compadre_CoordsT.hpp"
#include "Compadre_ParticlesT.hpp"
#include "Compadre_FileIO.hpp" // for VTKIO

#ifdef COMPADRE_USE_VTK
#include <vtkDataSet.h>
#include <vtkPolyData.h>
#include <vtkKdTreePointLocator.h>
#endif

#ifdef COMPADRE_USE_BOOST
#include <boost/bind.hpp>
#endif

#ifdef COMPADRE_USE_OPENMP
#include <omp.h>
#endif

#ifdef COMPADRE_USE_VTK
namespace Compadre {

typedef Compadre::XyzVector xyz_type;
typedef Compadre::CoordsT coords_type;
typedef Compadre::CoordsT particles_type;

VTKInformation::VTKInformation(const particles_type* source_particles, Teuchos::RCP<Teuchos::ParameterList> parameters, const particles_type* target_particles, bool use_physical_coords) :
		NeighborhoodT(source_particles, parameters, target_particles)
{
	_vtk_data = Teuchos::rcp(new Compadre::VTKData(_source_particles, parameters));
	_vtk_data->generateDataSet(true /*halo*/, false /*for_writing_output*/, use_physical_coords);
	_vtk_data->generateCombinedDataSet();

	vtkSmartPointer<vtkDataSet> ds = _vtk_data->getCombinedDataSet();
	_cloud = vtkSmartPointer<vtkPolyData>::New();
	_cloud->ShallowCopy(ds);

	_index = vtkSmartPointer<vtkKdTreePointLocator>::New();
	_index->SetDataSet(_cloud);
	_index->BuildLocator();
}

void VTKInformation::constructSingleNeighborList(const scalar_type* coordinate,
		const scalar_type radius, std::vector<std::pair<size_t, scalar_type> >& neighbors, bool use_physical_coords) const {
	vtkSmartPointer<vtkIdList> result = vtkSmartPointer<vtkIdList>::New();
	std::vector<std::pair<size_t, scalar_type> >& new_neighbors = neighbors;

//	std::cout << "radius: " << radius << std::endl;
	_index->FindPointsWithinRadius (radius, &coordinate[0], result);

	new_neighbors.resize(result->GetNumberOfIds());
	const xyz_type current_coord(coordinate);

	for (vtkIdType i=0, n=result->GetNumberOfIds(); i<n; i++) {
		const size_t other_id = (size_t)(result->GetId(i));
		std::pair<size_t, scalar_type> pair(other_id, _source_coords->distance((local_index_type)other_id, current_coord, use_physical_coords) /* distance*/);
		new_neighbors[i] = pair;
//		std::cout << other_id << ", ";
	}
//	std::cout << std::endl;

#ifdef USE_BOOST
	std::sort(new_neighbors.begin(), new_neighbors.end(),
	          boost::bind(&std::pair<size_t, scalar_type>::second, _1) < boost::bind(&std::pair<size_t, scalar_type>::second, _2));
#else
	TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "Sort called in VTKInformation.hpp which relies on Boost, but build without Boost."); 
#endif

//	for (vtkIdType i=0, n=result->GetNumberOfIds(); i<n; i++) {
//		const xyz_type current_coord(coordinate);
//		const size_t other_id = new_neighbors[i].first;
//		const xyz_type other_coord = _coords->getLocalCoords((local_index_type)other_id, true);
//
//		std::cout << "i: " << i << " my_pt: " << current_coord << " dist: " << _coords->distance((local_index_type)other_id, current_coord) << " ot_pt: " << other_coord << std::endl;
//		std::cout << other_id << ", vs " << new_neighbors[i].second << std::endl;
//	}

}

}
#endif
