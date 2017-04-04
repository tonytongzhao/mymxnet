import numpy 
import sys
import os
resdir=sys.argv[1]

tr_res_list={}
val_res_list={}
for filename in os.listdir(resdir):
    print filename
    tr_res_list[filename]=[5]
    val_res_list[filename]=[5]
    with open(os.path.join(resdir, filename), 'r') as f:
        for line in f:
            if 'Train-RMSE' in line:
                tr_res_list[filename].append(float(line.split('=')[1]))
            elif 'Validation-RMSE' in line:
                val_res_list[filename].append(float(line.split('=')[1]))
print 'Best Tr RMSE'
for i in tr_res_list:
    print i, min(tr_res_list[i])
      

print 'Best Val RMSE'
for i in tr_res_list:
    print i, min(val_res_list[i])
            
