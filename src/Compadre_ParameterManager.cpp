#include "Compadre_ParameterManager.hpp"

#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_XMLParameterListCoreHelpers.hpp>

//#include <Teuchos_YamlParameterListCoreHelpers.hpp>

namespace Compadre {

ParameterManager::ParameterManager(int argc, char* argv[]) : _parameter_list(Teuchos::rcp(new Teuchos::ParameterList("Compadre")))
{
	Teuchos::RCP<Teuchos::CommandLineProcessor> command_line_processor = Teuchos::rcp(new Teuchos::CommandLineProcessor());
	command_line_processor->setDocString(
		"GMLS Laplacian test.\n"
	);

    std::string input_file_parameters;
    command_line_processor->setOption("input", &input_file_parameters, "Input file with parameters.");
    command_line_processor->setOption("i", &input_file_parameters, "Input file with parameters.");

    command_line_processor->recogniseAllOptions(false);
    command_line_processor->throwExceptions(false);

    Teuchos::CommandLineProcessor::EParseCommandLineReturn
		parseReturn = command_line_processor->parse( argc, argv );

    if( parseReturn == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED ) {
		_help_requested = true;
    } else _help_requested = false;
    if( parseReturn != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL   ) {
		_parse_error = true;
    } else _parse_error = false;

    buildFromFile(input_file_parameters);
}

ParameterManager::ParameterManager(std::string const & filename) : _parameter_list(Teuchos::rcp(new Teuchos::ParameterList("Compadre")))
{
	_help_requested = false;
	_parse_error = false;
	buildFromFile(filename);
}

ParameterManager::ParameterManager() : _parameter_list(Teuchos::rcp(new Teuchos::ParameterList("Compadre")))
{
	_help_requested = false;
	_parse_error = false;
	setDefaultParameters();
}

void ParameterManager::buildFromFile(std::string const & filename) {
	setDefaultParameters();
	// check file type
	size_t pos = filename.rfind('.', filename.length());
	std::string extension = filename.substr(pos+1, filename.length() - pos);
	transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
	Teuchos::RCP<Teuchos::ParameterList> file_parameters;
	if (extension == "xml") {
		file_parameters = readInXML(filename);
	} else if (extension == "yaml" || extension == "yml") {
		file_parameters = readInYAML(filename);
	} else {
		std::ostringstream msg;
		msg << "Invalid parameter list file-type specified: " << extension.c_str() << std::endl;
		TEUCHOS_TEST_FOR_EXCEPT_MSG(true, msg.str());
	}
	// copy all parameters missing from the file
	file_parameters->setParametersNotAlreadySet(*_parameter_list);
	// file_parameters->print();
	_parameter_list = file_parameters; // copy
}

void ParameterManager::setDefaultParameters() {

	_parameter_list->set("loop size", (int)1);

	// Lagrangian simulation, coordinate evolution
	_parameter_list->set("simulation type", "eulerian"); // alternate is Lagrangian

	// Solver Details
	Teuchos::RCP<Teuchos::ParameterList> solverList = Teuchos::rcp(new Teuchos::ParameterList("solver"));
	solverList->set("type", "iterative");
	solverList->set("file", "");
	solverList->set("save residual over solution", (bool)false);
	solverList->set("blocked", false); // block assembly and solve?
	solverList->set("blocks", 1); // how many blocks

	// Coordinate Details
	Teuchos::RCP<Teuchos::ParameterList> coordinatesList = Teuchos::rcp(new Teuchos::ParameterList("coordinates"));
	coordinatesList->set("type", "euclidean");
	coordinatesList->set("units", "cm");

	// Remap Details
	Teuchos::RCP<Teuchos::ParameterList> remapList = Teuchos::rcp(new Teuchos::ParameterList("remap"));
	remapList->set("porder", 2);
	remapList->set("curvature porder", 0); // order to reconstruct manifold with when requested
	remapList->set("neighbors needed multiplier", 1.2);
	remapList->set("dense linear solver", "QR");
	remapList->set("weighting power", (local_index_type)8);
	remapList->set("weighting type", "power");
	// weighting type for covariance matrix from which a tangent plane to the manifold is derived
	remapList->set("curvature weighting power", (local_index_type)8);
	// weighting power for covariance matrix from which a tangent plane to the manifold is derived
	remapList->set("curvature weighting type", "power");
	remapList->set("obfet", (bool)false);
	remapList->set("source weighting field name", "");
	remapList->set("target weighting field name", "");
	remapList->set("quadrature points", (local_index_type)2); // number of quadrature points for integral sampling

	// Neighborhood Details
	Teuchos::RCP<Teuchos::ParameterList> neighborList = Teuchos::rcp(new Teuchos::ParameterList("neighborhood"));
	neighborList->set("method", "nanoflann"); // default neighbor search program is vtk
	neighborList->set("dynamic radius", (bool)true);
	// allow for dynamically enlarging the radius if more neighbors are needed than are found
	neighborList->set("spatially varying radius", (bool)true);
	// allow each target to have a neighborhood with a different support radius than other targets
	neighborList->set("cutoff multiplier", 1.1);
	// multiplier times the quantitiy that is the distance from target, for the location that is the nth#required neighbor
	neighborList->set("multiplier", 1.4); // multiplier for dynamically increasing neighbor search after failing to find sufficient neighbors
	neighborList->set("size", 0.2); // initial search radius used
	neighborList->set("max leaf", (int)10); // used by nanoflann for kdtree search

	// Halo Details
	Teuchos::RCP<Teuchos::ParameterList> haloList = Teuchos::rcp(new Teuchos::ParameterList("halo"));
	haloList->set("dynamic", (bool)true);
	// dynamic or not for halo list is used by applications in determining a halo_size, but
	// TODO: this will be used in building a larger halo region if insufficient neighbors are found in the neighbor search
	haloList->set("multiplier", 1.5); // used by applications now, but will be used like neighborhood's multiplier
	haloList->set("size", 0.2); // only used if dynamic set to false

	// Physics Details
	Teuchos::RCP<Teuchos::ParameterList> physicsList = Teuchos::rcp(new Teuchos::ParameterList("physics"));
	physicsList->set("artificial viscosity", (double)0.0); // default
	physicsList->set("some val", 0.1); // default

	Teuchos::RCP<Teuchos::ParameterList> rhsList = Teuchos::rcp(new Teuchos::ParameterList("rhs"));
	rhsList->set("some val", 0.2); // default

	Teuchos::RCP<Teuchos::ParameterList> bcsList = Teuchos::rcp(new Teuchos::ParameterList("bcs"));
	bcsList->set("some val", 0.3); // default

	// Timestepping Details
	Teuchos::RCP<Teuchos::ParameterList> timeList = Teuchos::rcp(new Teuchos::ParameterList("time"));
	timeList->set("dt", (scalar_type)-1); // explicitly setting the timestep, negative means not set
	timeList->set("dt multiplier", (scalar_type)-1); // multiplier times the CFL
	timeList->set("rk order", (local_index_type)4); // runge-kutta order
	timeList->set("t_0", (scalar_type)0.0); // initial simulation time
	timeList->set("t_end", (scalar_type)0.0); // ending simulation time

	// Input/Output Details
	Teuchos::RCP<Teuchos::ParameterList> inoutFileList = Teuchos::rcp(new Teuchos::ParameterList("io"));

	inoutFileList->set("input file prefix", "");
	inoutFileList->set("input file", "");

	inoutFileList->set("output file prefix", "");
	inoutFileList->set("output file", "out.pvtp");

	// both - outputs both Lagrangian and physical coordinates if a Lagrangian simulation
	// reference - outputs only Lagrangian coordinates (or Eulerian coordinates if in Eulerian frame
	// physical - outputs only physical coordinates in Lagrangian simulation
	// by default, using ``both'' will result in reference for an Eulerian simulation
	inoutFileList->set("coordinate type", "physical");

	inoutFileList->set("vtk produce mesh", (bool)false);

	inoutFileList->set("write every", (local_index_type)-1); // if negative, do not write to file, 0 means write the last time step, and n means every n time steps
	inoutFileList->set("keep original lat lon", (bool)false); // keep original as fields when reading in HOMME/MPAS files

	_parameter_list->set("solver", *solverList);
	_parameter_list->set("coordinates", *coordinatesList);
	_parameter_list->set("remap", *remapList);
	_parameter_list->set("neighborhood", *neighborList);
	_parameter_list->set("halo", *haloList);
	_parameter_list->set("physics", *physicsList);
	_parameter_list->set("rhs", *rhsList);
	_parameter_list->set("bcs", *bcsList);
	_parameter_list->set("time", *timeList);
	_parameter_list->set("io", *inoutFileList);

}

Teuchos::RCP<Teuchos::ParameterList> ParameterManager::readInXML(const std::string & filename) {
	Teuchos::RCP<Teuchos::ParameterList> file_parameter_list = Teuchos::getParametersFromXmlFile(filename);
	//file_parameter_list->print();
	return file_parameter_list;

//	std::ostringstream msg;
//	msg << "Not yet implemented." << std::endl;
//	TEUCHOS_TEST_FOR_EXCEPT_MSG(true, msg.str());
//	return Teuchos::RCP<Teuchos::ParameterList>();
}

Teuchos::RCP<Teuchos::ParameterList> ParameterManager::readInYAML(std::string const & filename) {
//	Teuchos::RCP<Teuchos::ParameterList> file_parameter_list = Teuchos::getParametersFromYamlFile(filename);
//	//file_parameter_list->print();
//	return file_parameter_list;

	std::ostringstream msg;
	msg << "Not yet implemented." << std::endl;
	TEUCHOS_TEST_FOR_EXCEPT_MSG(true, msg.str());
	return Teuchos::RCP<Teuchos::ParameterList>();
}

}
