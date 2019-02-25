% check version to see if compatible
V=regexp(version,'\d*','Match');
if (str2num(V{length(V)}) < 2017)
    fprintf('Error: Matlab version is too old to interface correctly to Python. Try Matlab 2017a or newer.')
    throw()
end

% add current folder to path
if count(py.sys.path,'.') == 0  
    insert(py.sys.path,int32(0),'');
end

% import numpy
np = py.importlib.import_module('numpy');

% import Compadre Toolkit
py.importlib.import_module('GMLS_Module');

% initialize Kokkos
py.GMLS_Module.initializeKokkos();

% set the polynomial order for the basis and the curvature polynomial order
% (if on a manifold)
poly_order = py.int(2);
curvature_poly_order = py.int(2);

dense_solver_type = py.str("QR");

% spatial dimension for polynomial reconstruction
spatial_dimensions = py.int(1);

% initialize and instance of the GMLS class in Compadre Toolkit
my_gmls = py.GMLS_Module.GMLS_Python(poly_order, dense_solver_type, curvature_poly_order, spatial_dimensions);

% set the weighting order
regular_weight = py.int(12);
my_gmls.setWeightingOrder(regular_weight);

% import the compadre_py_util module which uses kdtree in scipy
compadre_py_util = py.importlib.import_module('compadre_py_util');

% generate some 1d source points and a single target site
x=-10:.001:10;
x=[x' zeros(length(x),2)];
y=0;
y=[y' zeros(length(y),2)];

% flatten data to a 1D vector to be compatible with Matlab/Python 2017
flat_x = x(:)';
flat_y = y(:)';

% reshape data inside of python
np_x = compadre_py_util.get_2D_numpy_array(flat_x, py.int(length(flat_x)/3), py.int(3));
np_y = compadre_py_util.get_2D_numpy_array(flat_y, py.int(length(flat_y)/3), py.int(3));

% returns a dictionary with epsilons and with neighbor_lists
d = compadre_py_util.get_neighborlist(np_x,np_y,poly_order,spatial_dimensions);

% set source, targets, window sizes, and neighbor lists
my_gmls.setSourceSites(np_x);
my_gmls.setTargetSites(np_y);
my_gmls.setWindowSizes(d{'epsilons'});
my_gmls.setNeighbors(d{'neighbor_lists'});

% generates stencil
my_gmls.generatePointEvaluationStencil();

% apply stencil to sample data for all targets
data_vector = ones(size(x,1),1);
np_data_vector = np.array(data_vector');
np_computed_answer = my_gmls.applyStencil(np_data_vector)
computed_answer = double(py.array.array('d',py.numpy.nditer(np_computed_answer)));

% check that answer is correct
tolerance = 1e-14;
assert(abs(computed_answer-1)<tolerance, 'Computed answer should be 1, but it is not')



% finalize kokkos
clear my_gmls
py.GMLS_Module.finalizeKokkos();


% if needed, py.reload(compadre_py_util);
