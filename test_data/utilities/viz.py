import matplotlib.pyplot as plt
from mpltools import annotation
import os
#import glob
import numpy as np
from pathlib import Path
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Visualize rates from table data.')
parser.add_argument('fname', type=str, help='file name to open')
parser.add_argument('--title', dest='title', type=str, nargs='?', default='', help='title for plot')
parser.add_argument('--footer', dest='footer', type=str, nargs='?', default='', help='footer for plot')
args = parser.parse_args()
if args.title=="":
    args.title = os.path.basename(Path(args.fname).with_suffix(''))

df = pd.read_csv(args.fname)
args.title = args.title.replace("\\n","\n")
plt.title(args.title)
xlabel = "mesh spacing"
args.footer = args.footer.replace("\\n","\n")
if args.footer!="":
    xlabel = xlabel + "\n" + args.footer
plt.xlabel(xlabel)
plt.ylabel("error")
for key,field in enumerate(df):
    if key!=0:
        data_line = plt.loglog(df['h'][0:-1].to_numpy(), df[field][0:-1].to_numpy(), '-*')
plt.autoscale(enable=True,axis='x',tight=True)
ax = plt.gca()

all_h = df.to_numpy()[:,0:1]
all_errors = df.to_numpy()[:,1::]

rate_ordering = all_errors[-1,:].argsort()

lower_rate = all_errors[-1,rate_ordering[0]]
higher_rate = all_errors[-1,rate_ordering[-1]]

low_rate_indices = np.where(all_errors[-1,:]==lower_rate)
high_rate_indices = np.where(all_errors[-1,:]==higher_rate)

print(lower_rate,higher_rate)
print(low_rate_indices)
print(high_rate_indices)

low_rate_sorted = all_errors[-2,low_rate_indices].argsort()
print(low_rate_sorted[0])
low_rate_indices_ordered = low_rate_indices[0][low_rate_sorted[0]]
print(low_rate_indices_ordered)

high_rate_sorted = all_errors[-2,high_rate_indices].argsort()
print(high_rate_sorted[0])
high_rate_indices_ordered = high_rate_indices[0][high_rate_sorted[0]]
print(high_rate_indices_ordered)

min_low = all_errors[-2,low_rate_indices_ordered[0]]
max_low = all_errors[-2,low_rate_indices_ordered[-1]]
min_high = all_errors[-2,high_rate_indices_ordered[0]]
max_high = all_errors[-2,high_rate_indices_ordered[-1]]
print(min_low,max_low,min_high,max_high)


print(all_h)
annotation.slope_marker((all_h[-2][0], max_low), (int(lower_rate), 1), ax=ax, poly_kwargs={'facecolor': (0.0, 0.0, 0.0)})
annotation.slope_marker((all_h[-2][0], min_high), (int(higher_rate), 1), ax=ax, poly_kwargs={'facecolor': (0.0, 0.0, 0.0)})

plt.legend([field for field in list(df)[1::]])
plt.show()
