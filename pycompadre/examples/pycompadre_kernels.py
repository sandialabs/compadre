import pycompadre
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons

''' Interactive example of weighting kernels in pycompadre

The weighting kernels used in the GMLS class in the function Wab allow users
to vary two parameters. This tool gives users the ability to vary these two
parameters and weighting type and see the effect visually.
'''

# initialize Kokkos
kp = pycompadre.KokkosParser()

# initialize parameters
wt = pycompadre.WeightingFunctionType.Power
x = np.linspace(-2.0, 2.0, 700)
h = 1.0
p = 2
n = 1

def approximate(wt, h, p, n):
    y = np.array([pycompadre.Wab(xin,h,wt,p,n) for xin in x], dtype='f8')
    return y

# get initial data
y = approximate(wt, h, p, n)

# plot initial data
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

# resize to make font more readable when large
fig.set_size_inches(1600/fig.dpi,900/fig.dpi)
global plot_size
plot_size = fig.get_size_inches()*fig.dpi

d, = plt.plot(x, y, c='#0000FF', linewidth=4, zorder=1)
ax.set(xlabel='r', ylabel='',
       title='Kernel Evaluation')
ax.grid()
ax.margins(x=0)
axcolor = 'lightgoldenrodyellow'
ax.autoscale(True)
ax.relim()
ax.autoscale_view()
ax.autoscale(False)

# axes for sliders and radio buttons
ax_p = plt.axes([0.25, 0.125, 0.65, 0.03], facecolor='white')
ax_n = plt.axes([0.25, 0.075, 0.65, 0.03], facecolor='white')
ax_weighting_type = plt.axes([0.015, 0.25, 0.15, 0.35], facecolor=axcolor)

# sliders
delta_f = 4.0/200
sl_p = Slider(ax_p, 'P', valmin=0.0, valmax=6.0, valinit=2, valstep=1, color=None, initcolor='black')
sl_n = Slider(ax_n, 'N', valmin=0.0, valmax=6.0, valinit=1, valstep=1, color=None, initcolor='black')

fontsize=10.0/800*plot_size[0]
t_power = plt.text(0.02, 0.05, '$$\\left(1-\\left(\\frac{|r|}{h}\\right)^N\\right)^P$$', fontsize=fontsize, transform=plt.gcf().transFigure,usetex=True)
t_gaussian = plt.text(0.02, 0.05, '$$\\frac{1}{\\frac{h}{P} \\sqrt(2 \\pi)} e^{\\frac{-1}{2}r^2/{\\left(\\frac{h}{P}\\right)^2}}$$', fontsize=fontsize, transform=plt.gcf().transFigure,usetex=True)
t_cubic = plt.text(0.02, 0.05, '$$((1-\\frac{r}{h})+\\frac{r}{h} (1-\\frac{r}{h}) (1-2 \\frac{r}{h}))$$', fontsize=fontsize, transform=plt.gcf().transFigure,usetex=True)
t_cosine = plt.text(0.02, 0.05, '$$\\cos(\\frac{\\pi r}{2h})$$', fontsize=fontsize, transform=plt.gcf().transFigure,usetex=True)
t_sigmoid = plt.text(0.02, 0.05, '$$\\frac{1}{e^{Pr} + e^{-Pr} + N}$$', fontsize=fontsize, transform=plt.gcf().transFigure,usetex=True)

t_labels = {'Power':t_power, 'Gaussian':t_gaussian, 'Cubic Spl.':t_cubic, 'Cosine':t_cosine, 'Sigmoid':t_sigmoid}
for item in t_labels.keys():
    if item!='Power':
        t_labels[item].update({'visible':False})

#radios
rad_weighting_type = RadioButtons(ax_weighting_type, ('Power', 'Cubic Spl.', 'Cosine', 'Gaussian', 'Sigmoid'), active=0)

def update(val):
    global wt
    y = approximate(wt, h, int(sl_p.val), int(sl_n.val))
    d.set_ydata(y)
    fig.canvas.draw_idle()

# register objects using update
sl_p.on_changed(update)
sl_n.on_changed(update)

def weighting_type_update(label):
    weighting_type_dict = {'Power': pycompadre.WeightingFunctionType.Power, 'Cubic Spl.': pycompadre.WeightingFunctionType.CubicSpline, 'Cosine': pycompadre.WeightingFunctionType.Cosine, 'Gaussian': pycompadre.WeightingFunctionType.Gaussian, 'Sigmoid': pycompadre.WeightingFunctionType.Sigmoid}
    global wt
    wt = weighting_type_dict[label]
    for item in t_labels.keys():
        t_labels[item].update({'visible':False})
    t_labels[label].update({'visible':True})
    if (label=='Power'):
        ax_p.set_visible(True)
        ax_n.set_visible(True)
    elif (label=='Cubic Spl.'):
        ax_p.set_visible(False)
        ax_n.set_visible(False)
    elif (label=='Cosine'):
        ax_p.set_visible(False)
        ax_n.set_visible(False)
    elif (label=='Gaussian'):
        ax_p.set_visible(True)
        ax_n.set_visible(False)
    elif (label=='Sigmoid'):
        ax_p.set_visible(True)
        ax_n.set_visible(True)

    y0 = approximate(wt, h, 0, 0)
    y1 = approximate(wt, h, 0, 6)
    y2 = approximate(wt, h, 6, 0)
    y3 = approximate(wt, h, 6, 6)

    dy0, = ax.plot(x, y0, c='#0000FF', linewidth=4, zorder=1)
    dy1, = ax.plot(x, y1, c='#0000FF', linewidth=4, zorder=1)
    dy2, = ax.plot(x, y2, c='#0000FF', linewidth=4, zorder=1)
    dy3, = ax.plot(x, y3, c='#0000FF', linewidth=4, zorder=1)
    ax.autoscale(True)
    d.set_ydata(y0)
    ax.relim()
    ax.autoscale_view()
    ax.autoscale(False)

    dy0.remove()
    dy1.remove()
    dy2.remove()
    dy3.remove()

    update(0)

# register objects using weighting_type_update
rad_weighting_type.on_clicked(weighting_type_update)

def on_move(event):
    if event.inaxes:
        new_plot_size = fig.get_size_inches()*fig.dpi
        global plot_size
        if (not np.array_equal(new_plot_size, plot_size)):
            for item in t_labels.keys():
                t_labels[item].update({'fontsize':10.0/1600*new_plot_size[0]})
            plot_size = new_plot_size
            fig.canvas.draw_idle()

plt.connect('motion_notify_event', on_move)

plt.show()
del kp
