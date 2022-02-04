import time
import pycompadre
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons

''' Interactive example of pycompadre in two dimensions

By changing the evaluation location, users can see the local polynomial approximation
and observe how it changes as it is drug left or right.

Additional, users can get a feel for the effect of various kernels, polynomial orders,
and epsilon multiplier choices.
'''

PRINT_SETUP_TIME = False 
PRINT_SOLVE_TIME = False
PRINT_PLOT_TIME  = False 

# initialize Kokkos
kp = pycompadre.KokkosParser()

# initialize parameters
tic = time.perf_counter()
polynomial_order = 2
input_dimensions = 2
epsilon_multiplier = 1.6
global weighting_type, function, num_data_points
weighting_type = 'power'
function = lambda x, y: pow(x,2)*pow(y,2)
num_data_points = 10
global x, y, X, Y, XY_ravel, Z, Z_ravel
x = np.linspace(0,4,num_data_points)
y = np.linspace(0,4,num_data_points)
X, Y = np.meshgrid(x,y)
XY_ravel = np.vstack((X.ravel(order = 'C'),Y.ravel(order = 'C'))).T
Z = function(X,Y)
Z_ravel = Z.ravel(order='C')
x_pred = np.linspace(0,4,200)
y_pred = np.linspace(0,4,200)
X_pred, Y_pred = np.meshgrid(x_pred,y_pred)
XY_pred_ravel = np.vstack((X_pred.ravel(order = 'C'),Y_pred.ravel(order = 'C'))).T
extra_sites_coords_X = np.copy(X_pred)
extra_sites_coords_Y = np.copy(Y_pred)
toc = time.perf_counter()
PRINT_SETUP_TIME and print("Setup data in %0.6f seconds"%(toc-tic,))
    
# get GMLS approximate at all x_pred, as well as reconstruction about attempt_center_about_coord
def approximate(porder, wpower, wtype, epsilon_multiplier, attempt_center_about_coord):
    tic = time.perf_counter()
    gmls_obj=pycompadre.GMLS(porder, input_dimensions, "QR", "STANDARD")
    gmls_obj.setWeightingParameter(wpower)
    gmls_obj.setWeightingType(wtype)
    gmls_helper = pycompadre.ParticleHelper(gmls_obj)

    gmls_helper.generateKDTree(XY_ravel)
    gmls_obj.addTargets(pycompadre.TargetOperation.ScalarPointEvaluation)
    
    gmls_helper.generateNeighborListsFromKNNSearchAndSet(XY_pred_ravel, porder, input_dimensions, epsilon_multiplier)
    gmls_obj.generateAlphas(1, False)
    
    # helper function for applying of alphas
    Z_pred = gmls_helper.applyStencil(Z_ravel, pycompadre.TargetOperation.ScalarPointEvaluation)

    center_about_idx   = np.sum(np.abs(XY_pred_ravel - attempt_center_about_coord), axis=1).argmin()
    center_about_coord = XY_pred_ravel[center_about_idx]
    extra_sites_coords = np.copy(XY_pred_ravel)

    gmls_obj_2=pycompadre.GMLS(porder, input_dimensions, "QR", "STANDARD")
    gmls_obj_2.setWeightingParameter(wpower)
    gmls_obj_2.setWeightingType(wtype)
    gmls_helper_2 = pycompadre.ParticleHelper(gmls_obj_2)
    gmls_helper_2.generateKDTree(XY_ravel)
    gmls_obj_2.addTargets(pycompadre.TargetOperation.ScalarPointEvaluation)
    gmls_helper_2.generateNeighborListsFromKNNSearchAndSet(np.atleast_2d(XY_pred_ravel[center_about_idx]), porder, input_dimensions, epsilon_multiplier)

    extra_sites_idx = np.zeros(shape=(1,extra_sites_coords.shape[0]+1), dtype='i4')
    extra_sites_idx[0,0] = extra_sites_coords.shape[0]
    extra_sites_idx[0,1:] = np.arange(extra_sites_coords.shape[0])
    gmls_helper_2.setAdditionalEvaluationSitesData(extra_sites_idx, extra_sites_coords)

    gmls_obj_2.generateAlphas(1, False)
    
    # manual applying of alphas
    nl = gmls_helper_2.getNeighborLists()
    computed_answer = np.zeros(shape=(len(extra_sites_coords),), dtype='f8')
    sf = pycompadre.SamplingFunctionals['PointSample']

    colors = len(XY_ravel)*['black']
    for k in range(nl.getNumberOfNeighbors(0)):
        colors[nl.getNeighbor(0,k)] = 'red'
    for j in range(extra_sites_idx[0,0]):
        computed_answer[j] = gmls_helper_2.applyStencilSingleTarget(Z_ravel, pycompadre.TargetOperation.ScalarPointEvaluation, sf, j+1)

    center_about_extra_idx   = np.sum(np.abs(extra_sites_coords - center_about_coord), axis=1).argmin()
    center_about_extra_coord = extra_sites_coords[center_about_extra_idx]
    del nl
    del gmls_obj
    del gmls_obj_2
    del gmls_helper
    del gmls_helper_2
    toc = time.perf_counter()
    PRINT_SOLVE_TIME and print("Solve GMLS in %0.6f seconds"%(toc-tic,))
    return (np.reshape(Z_pred, newshape=(len(x_pred), len(y_pred))), np.reshape(computed_answer, newshape=(len(x_pred), len(y_pred))), center_about_extra_idx, center_about_extra_coord, colors)
    
# get initial data for plotting
Z_pred, computed_answer, center_about_extra_idx, center_about_extra_coord, colors = approximate(polynomial_order, 3, 'power', epsilon_multiplier, np.atleast_2d([1.0,1.0]))

tic = time.perf_counter()
# plot initial data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.30, bottom=0.30)

# plot just data to get bounds for axes
global d, l, s, p
p = ax.scatter(X.ravel(order='C'), Y.ravel(order='C'), Z_ravel, c='#000000', marker='D', zorder=2)
ax.autoscale(True)
ax.relim()
ax.autoscale_view()
ax.autoscale(False)
p.remove()

# plot again
d = ax.plot_surface(X_pred, Y_pred, Z_pred, color='b', zorder=1, alpha=0.7)
p = ax.scatter(X.ravel(order='C'), Y.ravel(order='C'), Z_ravel, c='#000000', marker='D', zorder=2)
p.set_color(colors)
l = ax.plot_surface(X_pred, Y_pred, computed_answer, color='#00FF00', zorder=2, alpha=0.3)
s = ax.scatter([center_about_extra_coord[0],], [center_about_extra_coord[1],], [computed_answer.ravel(order='C')[center_about_extra_idx],], c='#FF0000', marker='o', zorder=4, edgecolor='black', s=180, alpha=1.0)
ax.set(xlabel='x', ylabel='GMLS approximation')
ax.grid()
ax.margins(x=0)
axcolor = 'lightgoldenrodyellow'

# axes for sliders and radio buttons
ax_location_check = plt.axes([0.05, 0.093, 0.15, 0.15], facecolor='white', frameon=False)
ax_location_x = plt.axes([0.25, 0.175, 0.65, 0.03], facecolor='white')
ax_location_y = plt.axes([0.25, 0.125, 0.65, 0.03], facecolor='white')
ax_weighting_power = plt.axes([0.7, 0.025, 0.2, 0.03], facecolor=axcolor)
ax_num_data_points = plt.axes([0.25, 0.025, 0.2, 0.03], facecolor=axcolor)
ax_epsilon = plt.axes([0.7, 0.075, 0.2, 0.03])
ax_polynomial_order = plt.axes([0.25, 0.075, 0.2, 0.03])
ax_weighting_type = plt.axes([0.015, 0.25, 0.25, 0.15], facecolor=axcolor)
ax_func_type = plt.axes([0.015, 0.40, 0.25, 0.15], facecolor=axcolor)

# sliders
sl_location_check = CheckButtons(ax_location_check, ["",], [True,])
delta_f = 4.0/200
sl_location_x = Slider(ax_location_x, 'Location (x)', valmin=0.0, valmax=4.0, valinit=0.0, valstep=delta_f, color=None, initcolor='black')
sl_location_y = Slider(ax_location_y, 'Location (y)', valmin=0.0, valmax=4.0, valinit=0.0, valstep=delta_f, color=None, initcolor='black')
sl_location_x.set_val(1.0)
sl_location_y.set_val(1.0)
sl_weighting_power = Slider(ax_weighting_power, 'Weighting P.', valmin=0, valmax=5, valinit=3, valstep=1)
sl_num_data_points = Slider(ax_num_data_points, 'Number of Points', valmin=6, valmax=50, valinit=10, valstep=1)
sl_epsilon = Slider(ax_epsilon, 'Epsilon Multipler', valmin=1.0001, valmax=5.0, valinit=1.6, valstep=.1)
sl_polynomial_order = Slider(ax_polynomial_order, 'Polynomial Order', valmin=0, valmax=6, valinit=2, valstep=1)

#radios
rad_weighting_type = RadioButtons(ax_weighting_type, ('Power', 'Cubic Spl.', 'Gaussian'), active=0)
rad_func_type = RadioButtons(ax_func_type, ('x^2*y^2', 'sin(2x)cos(2y)'), active=0)
toc = time.perf_counter()
PRINT_PLOT_TIME and print("Setup graphics in %0.6f seconds"%(toc-tic,))

def update(val):
    global weighting_type
    Z_pred, computed_answer, center_about_extra_idx, center_about_extra_coord, colors = approximate(sl_polynomial_order.val, sl_weighting_power.val, weighting_type, sl_epsilon.val, np.atleast_2d([sl_location_x.val, sl_location_y.val]))
    global d, l, s
    try:
        d.remove()
    except:
        pass
    d = ax.plot_surface(X_pred, Y_pred, Z_pred, color='b', zorder=1,alpha=0.7)
    try:
        l.remove()
    except:
        pass
    l = ax.plot_surface(X_pred, Y_pred, computed_answer, color='#00FF00', zorder=2, alpha=0.3, visible=sl_location_check.get_status()[0])
    try:
        s.remove()
    except:
        pass
    s = ax.scatter([center_about_extra_coord[0],], [center_about_extra_coord[1],], [computed_answer.ravel(order='C')[center_about_extra_idx],], c='#FF0000', marker='o', zorder=4, edgecolor='black', s=180, alpha=1.0)
    if sl_location_check.get_status()[0]:
        p.set_color(colors)
    fig.canvas.draw_idle()
# register objects using update
sl_location_x.on_changed(update)
sl_location_y.on_changed(update)
sl_weighting_power.on_changed(update)
sl_epsilon.on_changed(update)
sl_polynomial_order.on_changed(update)

# called from changefunc
def updatepoints(val):
    global num_data_points, x, y, X, Y, XY_ravel
    num_data_points = sl_num_data_points.val
    x = np.resize(x, new_shape=(num_data_points,))
    y = np.resize(y, new_shape=(num_data_points,))
    x[:] = np.linspace(0,4,num_data_points)
    y[:] = np.linspace(0,4,num_data_points)
    X, Y = np.meshgrid(x,y)
    XY_ravel = np.vstack((X.ravel(order = 'C'),Y.ravel(order = 'C'))).T

def weighting_type_update(label):
    weighting_type_dict = {'Power': 'power', 'Cubic Spl.': 'cubicspline', 'Gaussian' : 'gaussian'}
    global weighting_type
    weighting_type = weighting_type_dict[label]
    if (weighting_type=='power'):
        ax_weighting_power.set_visible(True)
    else:
        ax_weighting_power.set_visible(False)
    update(0)
# register objects using weighting_type_update
rad_weighting_type.on_clicked(weighting_type_update)

def changefunc(label):
    updatepoints(0)
    global weighting_type
    global function
    if type(label)==str:
        func_type_dict = {'x^2*y^2': lambda x,y: pow(x,2)*pow(y,2), 'sin(2x)cos(2y)': lambda x,y: np.sin(2*x)*np.cos(2*y)}
        function = func_type_dict[label]
    global X, Y, Z, Z_ravel
    Z = function(X,Y)
    Z_ravel = Z.ravel(order='C')
    update(0)
    global p, l, s, d
    d.remove()
    l.remove()
    s.remove()
    p.remove()
    p = ax.scatter(X.ravel(order='C'), Y.ravel(order='C'), Z_ravel, c='#000000', marker='D', zorder=2)
    ax.autoscale(True)
    if type(label)==str:
        ax.relim()
        ax.autoscale_view()
    ax.autoscale(False)
    update(0)
    fig.canvas.draw_idle()

# register objects using changefunc
rad_func_type.on_clicked(changefunc)
sl_num_data_points.on_changed(changefunc)

def changelocationviz(label):
    l.set_visible(sl_location_check.get_status()[0])
    fig.canvas.draw_idle()
    if sl_location_check.get_status()[0]:
        update(0)
    else:
        p.set_color('black')
sl_location_check.on_clicked(changelocationviz)

plt.show()
del kp
