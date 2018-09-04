#ifndef _COMPADRE_PARAMETERMANAGER_HPP_
#define _COMPADRE_PARAMETERMANAGER_HPP_

#include "Compadre_Config.h"
#include "Compadre_Typedefs.hpp"

namespace Compadre {

/*
* Eventually, this class should read in a file of type XML or YAML, set default values if not specified
*/
class ParameterManager {

	protected:

		Teuchos::RCP<Teuchos::ParameterList> _parameter_list;
		bool _help_requested;
		bool _parse_error;

	public:

		ParameterManager();

		ParameterManager(std::string const & filename);

		ParameterManager(int argc, char* argv[]);

		~ParameterManager() {};

		Teuchos::RCP<Teuchos::ParameterList> getList() const { return _parameter_list; }

		bool helpRequested() const { return _help_requested; }

		bool parseError() const { return _parse_error; }

		static Teuchos::RCP<Teuchos::ParameterList> readInXML(std::string const & filename);

		static Teuchos::RCP<Teuchos::ParameterList> readInYAML(std::string const & filename);

	private:

		void setDefaultParameters();

		void buildFromFile(std::string const & filename);

};

}

#endif
