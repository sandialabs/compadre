import pandas as pd
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Concatenate data file (optional) and calculate converge rate (optional).')
parser.add_argument('source_files', metavar='source_files', type=str, nargs='+',
                            help='files to from which to get data')
parser.add_argument('--target', dest='target_file', type=str, nargs='?', help='file to write to')
parser.add_argument('--norm', dest='norm', type=str, nargs='*', help='norms to convert rate for')
args = parser.parse_args()
print(args)

target_data = None
for fname in args.source_files:
    if (target_data is None):
        target_data = pd.read_csv(fname) 
    else:
        new_data = pd.read_csv(fname)
        target_data = pd.concat([target_data, new_data], axis=0)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(target_data)
# save to target_file, if specified
if (args.target_file is not None):
    target_data.to_csv(args.target_file, index = False, header=True)

# compute specified norms
if (args.norm is not None):
    for norm_name in args.norm:
        for col_name in target_data.columns.tolist():
            if (norm_name.lower()==col_name.lower()):
                print(col_name+' rates:')
                col_data = target_data[col_name].to_numpy()
                for i in range(len(col_data)-1):
                    print(np.log(col_data[i+1]/col_data[i])/np.log(.5))
                break
