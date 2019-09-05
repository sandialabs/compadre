#ifndef _COMPADRE_GMLS_CURLCURLSOURCES_HPP_
#define _COMPADRE_GMLS_CURLCURLSOURCES_HPP_

#include <Compadre_SourcesT.hpp>

namespace Compadre {

class GMLS_CurlCurlSources : public SourcesT {

	protected:

		typedef Compadre::ParticlesT particle_type;

	public:

		GMLS_CurlCurlSources(	Teuchos::RCP<particle_type> particles,
                                        Teuchos::RCP<mvec_type> b = Teuchos::null) :
                  SourcesT(particles, b)
		{}

		virtual ~GMLS_CurlCurlSources() {};

		virtual void evaluateRHS(local_index_type field_one, local_index_type field_two = -1, scalar_type time = 0.0);

		virtual std::vector<InteractingFields> gatherFieldInteractions();

};

}

#endif
