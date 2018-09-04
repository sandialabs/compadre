#ifndef _COMPADRE_VTKINFORMATION_HPP_
#define _COMPADRE_VTKINFORMATION_HPP_

#include "Compadre_NeighborhoodT.hpp"
#ifdef COMPADRE_USE_VTK
	#include <vtkSmartPointer.h>
#endif

class vtkPolyData;
class vtkKdTreePointLocator;

#ifdef COMPADRE_USE_VTK
namespace Compadre {

class CoordsT;
class ParticlesT;
class LocalPointCloudT;
class VTKData;

/**
 * Derived class of NeighborhoodT specialized for using VTK to perform KdTree search.
*/

class VTKInformation : public NeighborhoodT  {

	protected:

		Teuchos::RCP<Compadre::VTKData> _vtk_data;
		vtkSmartPointer<vtkPolyData> _cloud;
		vtkSmartPointer<vtkKdTreePointLocator> _index;

	public:

		VTKInformation(const particles_type* source_particles, Teuchos::RCP<Teuchos::ParameterList> parameters, const particles_type* target_particles = NULL, bool use_physical_coords = true);

		virtual ~VTKInformation() {};

		virtual void constructSingleNeighborList(const scalar_type* coordinate,
				const scalar_type radius, std::vector<std::pair<size_t, scalar_type> >& neighbors, bool use_physical_coords = true) const;

};



}
#endif

#endif
