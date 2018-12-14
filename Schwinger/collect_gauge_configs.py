import glob
import numpy as np
import os

path = os.getcwd()
config_files = glob.glob(path + '/configs/gauge*')

out_data = 'pion_correlator' # ['fermion_determinant', 'top_charge', 'pion_correlator']

lattice_dim = 18

if out_data == 'fermion_determinant':
    cnt_output = 0
    len_output = 1
elif out_data == 'top_charge':
    cnt_output = 2
    len_output = 1
elif out_data == 'pion_correlator':
    cnt_output = 4
    len_output = lattice_dim

cnt_input = 6
output_features = []
input_features = []

for file in config_files:
    ff = open(file, 'r')
    line_count = 0
    for x in ff:
        if line_count == cnt_output:
            output_features.append(map(float, x.split()))
        
        if line_count >= cnt_input:
            input_features.append(map(float, x.split())[2:])
    
        line_count +=1
    
    ff.close()

output_sve_array = np.asarray(output_features).flatten()
input_sve_array = np.asarray(input_features).flatten()


file_tag = 'NN_data/Schwinger_' + out_data

np.savetxt(file_tag + '_input_data.dat', input_sve_array)
np.savetxt(file_tag + '_output_data.dat', output_sve_array)
