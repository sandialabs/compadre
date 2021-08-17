#!pip install cmake
#!pip install pycompadre

import pycompadre
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

kp = pycompadre.KokkosParser()

polynomial_order = 2
input_dimensions = 1
epsilon_multiplier = 2.4
global weighting_type
weighting_type = 'power'

global function
function = lambda x: pow(x,2)
global num_data_points
num_data_points = 10
x = np.linspace(0,4,num_data_points)
y = function(x)
#y = pow(x,3)
x_pred = np.linspace(0,4,200)
extra_sites_coords = np.atleast_2d(np.linspace(0,4,200)).T


def approximate(porder, wpower, wtype, epsilon_multiplier, attempt_center_about_coord):
    #help(pycompadre.GMLS)
    gmls_obj=pycompadre.GMLS(porder, input_dimensions, "QR", "STANDARD")
    #gmls_obj.setWeightingType("cubicspline")
    gmls_obj.setWeightingPower(wpower)
    gmls_obj.setWeightingType(wtype)
    gmls_helper = pycompadre.ParticleHelper(gmls_obj)
    gmls_helper.generateKDTree(np.atleast_2d(x).T)
    gmls_obj.addTargets(pycompadre.TargetOperation.ScalarPointEvaluation)
    
    gmls_helper.generateNeighborListsFromKNNSearchAndSet(np.atleast_2d(x_pred).T, porder, input_dimensions, epsilon_multiplier)
    
    center_about_idx   = (np.abs(x_pred - attempt_center_about_coord)).argmin()
    center_about_coord = x_pred[center_about_idx]
    print('Centered About:',center_about_coord)
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
            for j in range(extra_sites_idx[i,0]):
                computed_answer[j] = 0.0
                for k in range(nl.getNumberOfNeighbors(i)):
                    computed_answer[j] += gmls_obj.getAlpha(pycompadre.TargetOperation.ScalarPointEvaluation, 
                                          i, 0, 0, k, 0, 0, j+1)*y[nl.getNeighbor(i,k)]
    center_about_extra_idx   = (np.abs(extra_sites_coords - center_about_coord)).argmin()
    center_about_extra_coord = extra_sites_coords[center_about_extra_idx]
    del nl
    del gmls_obj
    del gmls_helper
    return (y_pred, computed_answer, center_about_extra_idx, center_about_extra_coord)

y_pred, computed_answer, center_about_extra_idx, center_about_extra_coord = approximate(polynomial_order, 3, 'power', epsilon_multiplier, 1.2)
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
p = plt.scatter(x, y, c='#000000', marker='D', zorder=2)
d, = plt.plot(x_pred, y_pred, c='#0000FF', linewidth=4, zorder=1)
#ax.plot(extra_sites_coords, computed_answer, c='#00FF00', linewidth=2)
l, = plt.plot(extra_sites_coords, computed_answer, c='#00FF00', lw=2, zorder=3)
s = plt.scatter([center_about_extra_coord,], computed_answer[center_about_extra_idx], c='#00FF00', marker='o', zorder=4)
ax.set(xlabel='x', ylabel='GMLS approximation',
       title='Approximation of sin(x)')
ax.grid()
#fig.savefig("test.png")

ax.margins(x=0)

axcolor = 'lightgoldenrodyellow'
ax_location = plt.axes([0.25, 0.125, 0.65, 0.03], facecolor='white')
ax_weighting_power = plt.axes([0.7, 0.025, 0.2, 0.03], facecolor=axcolor)

#a0 = 5
delta_f = 4.0/200
sl_location = Slider(ax_location, 'Location', valmin=0.0, valmax=4.0, valinit=0.0, valstep=delta_f, color=None, initcolor='black')#,hovercolor='skyblue')
sl_location.set_val(1.2)
sl_weighting_power = Slider(ax_weighting_power, 'Weighting P.', valmin=0, valmax=5, valinit=3, valstep=1)#, color='gray', initcolor='gray')#0.1, 10.0, valinit=a0)


ax_num_data_points = plt.axes([0.25, 0.025, 0.2, 0.03], facecolor=axcolor)
spl_num_data_points = Slider(ax_num_data_points, 'Number of Points', valmin=6, valmax=50, valinit=10, valstep=1)#, color='gray', initcolor='gray')#0.1, 10.0, valinit=a0)
#help(sl_weighting_power)
#sl_weighting_power.set(color='gray')
#help(ax_weighting_power)


def update(val):
    sl_location.valinit=2.0
    global weighting_type
    y_pred, computed_answer, center_about_extra_idx, center_about_extra_coord = approximate(sorder.val, sl_weighting_power.val, weighting_type, sepsilon.val, sl_location.val)
    l.set_ydata(computed_answer)
    d.set_ydata(y_pred)
    s.set_offsets([center_about_extra_coord, computed_answer[center_about_extra_idx]])
    fig.canvas.draw_idle()


sl_location.on_changed(update)
sl_weighting_power.on_changed(update)

epsilonax = plt.axes([0.7, 0.075, 0.2, 0.03])
sepsilon = Slider(epsilonax, 'Epsilon Multipler', valmin=1.0001, valmax=5.0, valinit=1.6, valstep=.1)

powerax = plt.axes([0.25, 0.075, 0.2, 0.03])
sorder = Slider(powerax, 'Polynomial Order', valmin=0, valmax=6, valinit=2, valstep=1)
#button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    sl_location.reset()
    sl_weighting_power.reset()
sepsilon.on_changed(update)
#button.on_clicked(reset)
sorder.on_changed(update)

ax_weighting_type = plt.axes([0.015, 0.25, 0.15, 0.15], facecolor=axcolor)
##rad_weighting_type = RadioButtons(ax_weighting_type, ('red', 'blue', 'green'), active=0)
rad_weighting_type = RadioButtons(ax_weighting_type, ('Power', 'Cubic Spl.', 'Gaussian'), active=0)

ax_func_type = plt.axes([0.015, 0.6, 0.15, 0.15], facecolor=axcolor)
##radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)
rad_func_type = RadioButtons(ax_func_type, ('sin(x)', 'x*sin(20x)', 'x^2'), active=0)

# called from changefunc
def updatepoints(val):
    global num_data_points, x, y 
    num_data_points = spl_num_data_points.val
    x = np.resize(x, new_shape=(num_data_points,))
    y = np.resize(y, new_shape=(num_data_points,))
    x[:] = np.linspace(0,4,num_data_points)

def colorfunc(label):
    #order_dict = {'Order 0': 0, 'Order 1': 1, 'Order 2': 2, 'Order 3': 3}
    weighting_type_dict = {'Power': 'power', 'Cubic Spl.': 'cubicspline', 'Gaussian' : 'gaussian'}
    #global polynomial_order
    global weighting_type
    weighting_type = weighting_type_dict[label]
    if (weighting_type=='power'):
        ax_weighting_power.set_visible(True)
        #ax_weighting_power.set(facecolor='yellowgoldenrod')
    #    sl_weighting_power = Slider(ax_weighting_power, 'Weighting (Power Only)', valmin=0, valmax=5, valinit=3, valstep=1)#, color='gray', initcolor='gray')#0.1, 10.0, valinit=a0)
    else:
        ax_weighting_power.set_visible(False)
        #ax_weighting_power.set(facecolor='gray')
    #    sl_weighting_power = Slider(ax_weighting_power, 'Weighting (Power Only)', valmin=0, valmax=5, valinit=3, valstep=1, color='gray', initcolor='gray')#0.1, 10.0, valinit=a0)
    #sl_weighting_power.on_changed(update)
    y_pred, computed_answer, center_about_extra_idx, center_about_extra_coord = approximate(sorder.val, sl_weighting_power.val, weighting_type, sepsilon.val, sl_location.val)
    #y_pred, computed_answer, center_about_extra_idx, center_about_extra_coord = approximate(polynomial_order, sl_location.val)
    #l.set_color(label)
    #s.set_color(label)
    l.set_ydata(computed_answer)
    d.set_ydata(y_pred)
    fig.canvas.draw_idle()
rad_weighting_type.on_clicked(colorfunc)

def changefunc(label):
    updatepoints(0)
    global weighting_type
    global function
    if type(label)==str:
        func_type_dict = {'sin(x)': lambda x: np.sin(x), 'x*sin(20x)': lambda x: x*np.sin(20*x), 'x^2' : lambda x: pow(x,2)}
        function = func_type_dict[label]
    y[:] = function(x)
    y_pred, computed_answer, center_about_extra_idx, center_about_extra_coord = approximate(sorder.val, sl_weighting_power.val, weighting_type, sepsilon.val, sl_location.val)
    l.set_ydata(computed_answer)
    d.set_ydata(y_pred)
    p.set_offsets(np.vstack((x,y)).T)
    if type(label)==str:
        ax.relim()
        ax.autoscale_view()
    update(0)
    fig.canvas.draw_idle()
rad_func_type.on_clicked(changefunc)
spl_num_data_points.on_changed(changefunc)

plt.show()
del kp
