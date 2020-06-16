% check version to see if compatible
V=regexp(version,'\d*','Match');
if (str2num(V{length(V)}) > 2000 && str2num(V{length(V)}) < 2017)
    error('Error: Matlab version is too old to interface correctly to Python. Try Matlab 2017a or newer.')
elseif (str2num(V{length(V)}) < 2000 && str2num(V{length(V)-1}) < 2017)
    error('Error: Matlab version is too old to interface correctly to Python. Try Matlab 2017a or newer.')
end

% add current folder to path
if count(py.sys.path,'.') == 0  
    insert(py.sys.path,int32(0),'');
end

% import numpy
np = py.importlib.import_module('numpy');

% import Compadre Toolkit
py.importlib.import_module('pycompadre');

% initialize Kokkos
py.pycompadre.KokkosParser();

% polynomial degree of reconstruction
poly_order = py.int(2);

% spatial dimension for polynomial reconstruction
spatial_dimensions = py.int(1);

% initialize and instance of the GMLS class in Compadre Toolkit
my_gmls = py.pycompadre.GMLS(poly_order, spatial_dimensions, py.str("QR"), py.str("STANDARD"));
% initialize a helper that manipulates the GMLS class (neighbor search, transforming data, applying solution to data, etc...)
gmls_helper = py.pycompadre.ParticleHelper(my_gmls);

% set the weighting order
regular_weight = py.int(4);
my_gmls.setWeightingPower(regular_weight);

% generate some 1d source points and three target sites
x=-10:.001:10;
y=[-1 0 1];

% reshape data inside of python
np_x = np.array(x);
np_x = np_x.reshape(py.list({py.int(length(x)),py.int(1)}));
np_y = np.array(y);
np_y = np_y.reshape(py.list({py.int(length(y)),py.int(1)}));

% returns a dictionary with epsilons and with neighbor_lists
epsilon_multiplier = 1.5;

% generate a KD tree from source sites and also sets source sites for GMLS class
gmls_helper.generateKDTree(np_x);

% do a neighbor search for target sites
gmls_helper.generateNeighborListsFromKNNSearchAndSet(np_y, poly_order, spatial_dimensions, epsilon_multiplier);

% add TargetOperation 0 and 9 from ENUM which are ScalarPointEvaluation and PartialXDerivativePointEvaluation
my_gmls.addTargets(py.pycompadre.TargetOperation(py.int(0)));
my_gmls.addTargets(py.pycompadre.TargetOperation(py.int(9)));

% generate alphas in 1 batch and don't keept the polynomial coefficients (not needed)
my_gmls.generateAlphas();

% generate data value for each source site (1 at each site)
np_data_vector = np.array(ones(1,length(x)));
np_data_vector = np_data_vector.reshape(py.list({py.int(length(x)),py.int(1)}));

% apply GMLS solution to data through helper
np_computed_answer = gmls_helper.applyStencil(np_data_vector,py.pycompadre.TargetOperation(py.int(0)));

computed_answers = double(py.array.array('d',py.numpy.nditer(np_computed_answer)));

% check that answer is correct
tolerance = 1e-14;
assert(norm(computed_answers-1)<tolerance, 'Failed test : computed answers should be 1, but at least one is not')
fprintf("Passed test.\n")



% delete gmls_helper (which uses gmls), then GMLS instance
clear gmls_helper
clear my_gmls

% KokkosParser cleans up when it goes out of scope

% if needed, py.reload(compadre_py_util);
