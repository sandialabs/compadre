import pycompadre
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons

''' Interactive example of pycompadre in one dimension

By changing the evaluation location, users can see the local polynomial approximation
and observe how it changes as it is drug left or right.

Additional, users can get a feel for the effect of various kernels, polynomial orders,
and epsilon multiplier choices.
'''

# initialize Kokkos
kp = pycompadre.KokkosParser()

# initialize parameters
solver_type = 'QR'
polynomial_order = 2
input_dimensions = 1
epsilon_multiplier = 1.6
global weighting_type, function, num_data_points
weighting_type = 'power'
function = lambda x: pow(x,2)
num_data_points = 10
x = np.linspace(0,4,num_data_points)
y = function(x)
x_pred = np.linspace(0,4,2000)
extra_sites_coords = np.atleast_2d(np.linspace(0,4,200)).T

# get GMLS approximate at all x_pred, as well as reconstruction about attempt_center_about_coord
def approximate(solver_type, porder, wpower0, wpower1, wtype, epsilon_multiplier, attempt_center_about_coord):
    gmls_obj=pycompadre.GMLS(porder, input_dimensions, solver_type, "STANDARD")
    gmls_obj.setWeightingParameter(wpower0,0)
    gmls_obj.setWeightingParameter(wpower1,1)
    gmls_obj.setWeightingType(wtype)
    gmls_helper = pycompadre.ParticleHelper(gmls_obj)
    gmls_helper.generateKDTree(np.atleast_2d(x).T)
    gmls_obj.addTargets(pycompadre.TargetOperation.ScalarPointEvaluation)
    
    gmls_helper.generateNeighborListsFromKNNSearchAndSet(np.atleast_2d(x_pred).T, porder, input_dimensions, epsilon_multiplier, 0.0, False, True)
    
    center_about_idx   = (np.abs(x_pred - attempt_center_about_coord)).argmin()
    center_about_coord = x_pred[center_about_idx]
    extra_sites_idx    = np.zeros(shape=(len(x_pred),len(extra_sites_coords)+1), dtype='i4')
    extra_sites_idx[center_about_idx,0] = len(extra_sites_coords)
    extra_sites_idx[center_about_idx,1:] = np.arange(len(extra_sites_coords))
    gmls_helper.setAdditionalEvaluationSitesData(extra_sites_idx, extra_sites_coords)
    
    gmls_obj.generateAlphas(1, False)
    
    # helper function for applying of alphas
    y_pred = gmls_helper.applyStencil(np.atleast_2d(y).T, pycompadre.TargetOperation.ScalarPointEvaluation)
    
    # manual applying of alphas
    nl = gmls_helper.getNeighborLists()
    for i in range(len(x_pred)):
        if (i==center_about_idx):
            computed_answer = np.zeros(shape=(len(extra_sites_coords),), dtype='f8')
            colors = len(x)*['black']
            for j in range(extra_sites_idx[i,0]):
                computed_answer[j] = 0.0
                for k in range(nl.getNumberOfNeighbors(i)):
                    colors[nl.getNeighbor(i,k)] = 'red'
                    computed_answer[j] += gmls_obj.getSolutionSet().getAlpha(pycompadre.TargetOperation.ScalarPointEvaluation, 
                                          i, 0, 0, k, 0, 0, j+1)*y[nl.getNeighbor(i,k)]
    center_about_extra_idx   = (np.abs(extra_sites_coords - center_about_coord)).argmin()
    center_about_extra_coord = extra_sites_coords[center_about_extra_idx]
    del nl
    del gmls_obj
    del gmls_helper
    return (y_pred, computed_answer, center_about_extra_idx, center_about_extra_coord, colors)

# get initial data for plotting
y_pred, computed_answer, center_about_extra_idx, center_about_extra_coord, colors = approximate(solver_type, polynomial_order, 2, 1, 'power', epsilon_multiplier, 1.2)
# plot initial data
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
p = ax.scatter(x, y, c='#000000', marker='D', zorder=2)
p.set_color(colors)
d, = plt.plot(x_pred, y_pred, c='#0000FF', linewidth=4, zorder=1)
l, = plt.plot(extra_sites_coords, computed_answer, c='#00FF00', lw=2, zorder=3)
s = plt.scatter([center_about_extra_coord,], computed_answer[center_about_extra_idx], c='#00FF00', marker='o', zorder=4, edgecolor='black', s=50)
ax.set(xlabel='x', ylabel='',
       title='GMLS Approximation')
ax.grid()
ax.margins(x=0)
axcolor = 'lightgoldenrodyellow'
plt.legend((p,d,l,s), ('Data', 'GMLS', 'Local Polynomial Fit', 'Target Site'))

# axes for sliders and radio buttons
ax_location_check = plt.axes([0.10, 0.068, 0.15, 0.15], facecolor='white', frameon=False)
ax_location = plt.axes([0.25, 0.125, 0.65, 0.03], facecolor='white')
ax_weighting_power_0 = plt.axes([0.6175, 0.025, 0.075, 0.03], facecolor=axcolor)
ax_weighting_power_1 = plt.axes([0.825, 0.025, 0.075, 0.03], facecolor=axcolor)
ax_num_data_points = plt.axes([0.25, 0.025, 0.2, 0.03], facecolor=axcolor)
ax_epsilon = plt.axes([0.7, 0.075, 0.2, 0.03])
ax_polynomial_order = plt.axes([0.25, 0.075, 0.2, 0.03])
ax_weighting_type = plt.axes([0.015, 0.20, 0.15, 0.20], facecolor=axcolor)
ax_func_type = plt.axes([0.015, 0.40, 0.15, 0.15], facecolor=axcolor)
ax_solver_type = plt.axes([0.015, 0.55, 0.15, 0.20], facecolor=axcolor)

# sliders
sl_location_check = CheckButtons(ax_location_check, ["",], [True,])
delta_f = 4.0/200
sl_location = Slider(ax_location, 'Location', valmin=0.0, valmax=4.0, valinit=0.0, valstep=delta_f, color=None, initcolor='black')
sl_location.set_val(1.2)
sl_weighting_power_0 = Slider(ax_weighting_power_0, 'Wgt. P(0)', valmin=0, valmax=6, valinit=2, valstep=1)
sl_weighting_power_1 = Slider(ax_weighting_power_1, 'Wgt. P(1)', valmin=0, valmax=6, valinit=1, valstep=1)
sl_num_data_points = Slider(ax_num_data_points, 'Number of Points', valmin=6, valmax=50, valinit=10, valstep=1)
sl_epsilon = Slider(ax_epsilon, 'Epsilon Multipler', valmin=1.0001, valmax=5.0, valinit=1.6, valstep=.1)
sl_polynomial_order = Slider(ax_polynomial_order, 'Polynomial Order', valmin=0, valmax=6, valinit=2, valstep=1)

#radios
rad_weighting_type = RadioButtons(ax_weighting_type, ('Power', 'Cubic Spl.', 'Cosine', 'Gaussian', 'Sigmoid'), active=0)
rad_func_type = RadioButtons(ax_func_type, ('sin(x)', 'x*sin(20x)', 'x^2'), active=2)
rad_solver_type = RadioButtons(ax_solver_type, ('QR', 'LU'), active=0)

def update(val, relim=False):
    sl_location.valinit=2.0
    global weighting_type
    y_pred, computed_answer, center_about_extra_idx, center_about_extra_coord, colors = approximate(rad_solver_type.value_selected, sl_polynomial_order.val, sl_weighting_power_0.val, sl_weighting_power_1.val, weighting_type, sl_epsilon.val, sl_location.val)
    l.set_ydata(computed_answer)
    d.set_ydata(y_pred)
    s.set_offsets([center_about_extra_coord, computed_answer[center_about_extra_idx]])
    if sl_location_check.get_status()[0]:
        p.set_color(colors)
    if relim:
        ax.ignore_existing_data_limits = True
        ax.update_datalim(p.get_datalim(ax.transData))
        ax.autoscale(True)
        ax.autoscale_view()
        ax.autoscale(False)
    fig.canvas.draw_idle()
# register objects using update
sl_location.on_changed(update)
sl_weighting_power_0.on_changed(update)
sl_weighting_power_1.on_changed(update)
sl_epsilon.on_changed(update)
sl_polynomial_order.on_changed(update)
rad_solver_type.on_clicked(update)

# called from changefunc
def updatepoints(val):
    global num_data_points, x, y 
    num_data_points = sl_num_data_points.val
    x = np.resize(x, new_shape=(num_data_points,))
    y = np.resize(y, new_shape=(num_data_points,))
    x[:] = np.linspace(0,4,num_data_points)

def weighting_type_update(label):
    weighting_type_dict = {'Power': 'power', 'Cubic Spl.': 'cubicspline', 'Cosine': 'cosine', 'Gaussian': 'gaussian', 'Sigmoid': 'sigmoid'}
    global weighting_type
    weighting_type = weighting_type_dict[label]
    if (label=='Power'):
        ax_weighting_power_0.set_visible(True)
        ax_weighting_power_1.set_visible(True)
    elif (label=='Cubic Spl.'):
        ax_weighting_power_0.set_visible(False)
        ax_weighting_power_1.set_visible(False)
    elif (label=='Cosine'):
        ax_weighting_power_0.set_visible(False)
        ax_weighting_power_1.set_visible(False)
    elif (label=='Gaussian'):
        ax_weighting_power_0.set_visible(True)
        ax_weighting_power_1.set_visible(False)
    elif (label=='Sigmoid'):
        ax_weighting_power_0.set_visible(True)
        ax_weighting_power_1.set_visible(True)
    update(0)
# register objects using weighting_type_update
rad_weighting_type.on_clicked(weighting_type_update)

def changefunc(label):
    updatepoints(0)
    global weighting_type
    global function
    if type(label)==str:
        func_type_dict = {'sin(x)': lambda x: np.sin(x), 'x*sin(20x)': lambda x: x*np.sin(20*x), 'x^2' : lambda x: pow(x,2)}
        function = func_type_dict[label]
    y[:] = function(x)
    p.set_offsets(np.vstack((x,y)).T)
    update(0, True)
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
