import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mimage
import seaborn as sns

#Import results
testpath ='/Users/lacan/Documents/Cosmostat/Codes/BlendHunter'
#testpath ='/Users/alacan/Documents/Cosmostat/Codes/BlendHunter'
path_results = testpath+'/bh/bh_results'

#Different noise realisations
sigmas = np.array([[5,51,52 ,53, 54],[14,141,142,143,144], [18,181,182,183,184],
                  [26,261,262,263,264], [35,351,352,353,354], [40,401,402,403,404]])

#Import function
def import_(path):
    img = np.load(path, allow_pickle=True)
    return img

def mb_(data=None): #Missed blends
    return len(np.where(data[0:4000] == 0)[0])

def fp_(data=None): #Single galaxy mistaken for a blend
    return len(np.where(data[4000:8000] == 1)[0])

def uni_(data=None): #Unidentified by SExtractor
    return len(np.where(data[0:4000] == 16)[0])+len(np.where(data[4000:8000] == 16)[0])

####Separated objects
def sep_(data=None, sep_results= False, dist=distance): 
    if sep_results:
        return len([i for i in np.where(data[0:4000] != 1)[0] if distance[i] > 20.0])
    else:
        return len([i for i in np.where(data[0:4000] != 'blended')[0] if dist[i] > 20.0])

####Close objects
def cls_(data=None, sep_results= False, dist=distance):
    if sep_results:
        return len([i for i in np.where(data[0:4000] != 1)[0] if  6.0 < distance[i] <= 20.0])
    else:
        return len([i for i in np.where(data[0:4000] != 'blended')[0] if  6.0 < dist[i] <= 20.0])

####Overlapping objects
def ovlps_(data=None, sep_results= False, dist=distance):
    if sep_results:
        return len([i for i in np.where(data[0:4000] != 1)[0] if distance[i] <= 6.0 ])
    else:
        return len([i for i in np.where(data[0:4000] != 'blended')[0] if dist[i] <= 6.0 ])

def get_mean(data=None):
    return np.mean(data)

def get_std(data=None):
    return np.std(data)

#Import bh results
results5 = [np.load(path_results+'/pred_{}.npy'.format(i), allow_pickle = True) for i in sigmas[0]]
results14 = [np.load(path_results+'/pred_{}.npy'.format(i), allow_pickle = True) for i in sigmas[1]]
results18 = [np.load(path_results+'/pred_{}.npy'.format(i), allow_pickle = True) for i in sigmas[2]]
results26 = [np.load(path_results+'/pred_{}.npy'.format(i), allow_pickle = True) for i in sigmas[3]]
results35 = [np.load(path_results+'/pred_{}.npy'.format(i), allow_pickle = True) for i in sigmas[4]]
results40 = [np.load(path_results+'/pred_{}.npy'.format(i), allow_pickle = True) for i in sigmas[5]]

#Paths flags for sep results
paths_flags5 = np.array([testpath+'/sep_results_8000/flags{}.npy'.format(i) for i in sigmas[0]])
paths_flags14 = np.array([testpath+'/sep_results_8000/flags{}.npy'.format(i) for i in sigmas[1]])
paths_flags18 = np.array([testpath+'/sep_results_8000/flags{}.npy'.format(i) for i in sigmas[2]])
paths_flags26 = np.array([testpath+'/sep_results_8000/flags{}.npy'.format(i) for i in sigmas[3]])
paths_flags35 = np.array([testpath+'/sep_results_8000/flags{}.npy'.format(i) for i in sigmas[4]])
paths_flags40 = np.array([testpath+'/sep_results_8000/flags{}.npy'.format(i) for i in sigmas[5]])

#Import sep results
flags5 = [import_(paths_flags5[j]) for j in range(5)]
flags14 = [import_(paths_flags14[j]) for j in range(5)]
flags18 = [import_(paths_flags18[j]) for j in range(5)]
flags26 = [import_(paths_flags26[j]) for j in range(5)]
flags35 = [import_(paths_flags35[j]) for j in range(5)]
flags40 = [import_(paths_flags40[j]) for j in range(5)]

#Results list
results_bh = [results5, results14, results18, results26, results35, results40]
results_sep = [flags5, flags14,flags18, flags26, flags35,flags40]

#Import distance between galaxies
param_x = np.load(testpath+"/bh/param_x_total.npy", allow_pickle=True)[36000:40000] #X parameter extracted from test images

param_y = np.load(testpath+"/bh/param_y_total.npy", allow_pickle=True)[36000:40000] #Y parameter extracted from test images

distance = np.sqrt(param_x**2 + param_y**2)
distance = np.array([distance[i][0] for i in range(len(distance))])


#####Missed blends and false positives for bh and sep
#Blendhunter
mb = [[len(np.where(data[j][0:4000] != 'blended')[0]) for j in range(5)] for data in results_bh]
fp = [[len(np.where(data[j][4000:8000] != 'not_blended')[0]) for j in range(5)] for data in results_bh]
#SExtractor
mb_sep = [[mb_(data=flags[i]) for i in range(len(flags))] for flags in results_sep]
fp_sep = [[fp_(data=flags[i]) for i in range(len(flags))] for flags in results_sep]
uni_sep = [[uni_(data=flags[i]) for i in range(len(flags))] for flags in results_sep]

#####Separated, close and overlapping objects in errors
#Blendhunter
sep_obj = [[sep_(data[k]) for k in range(len(data))] for data in results_bh]
cls_obj = [[cls_(data[k]) for k in range(len(data))] for data in results_bh]
ovlps_obj= [[ovlps(data[k]) for k in range(len(data))] for data in results_bh]

#SExtractor
sep_sep = [[sep_(data=flags[i], sep_results=True) for i in range(len(flags))] for flags in results_sep]
cls_sep = [[cls_(data=flags[i], sep_results=True) for i in range(len(flags))] for flags in results_sep]
ovlps_sep = [[ovlps_(data=flags[i], sep_results=True) for i in range(len(flags))] for flags in results_sep]

##########Get means and std deviation for error bars

#########MB and FP
means_mb_bh = [get_mean(data=i) for i in mb]
std_mb_bh = [get_std(data=i) for i in mb]

means_fp_bh = [get_mean(data=i) for i in fp]
std_fp_bh = [get_std(data=i) for i in fp]

means_mb_sep = [get_mean(data=i) for i in mb_sep]
std_mb_sep = [get_std(data=i) for i in mb_sep]

means_fp_sep = [get_mean(data=i) for i in fp_sep]
std_fp_sep = [get_std(data=i) for i in fp_sep]

means_uni_sep = [get_mean(data=i) for i in uni_sep]
std_uni_sep = [get_std(data=i) for i in uni_sep]


########SEP, CLOSE and OVERLAPS
means_sep_bh = [get_mean(data=i) for i in sep_obj]
std_sep_bh = [get_std(data=i) for i in sep_obj]

means_cls_bh = [get_mean(data=i) for i in cls_obj]
std_cls_bh = [get_std(data=i) for i in cls_obj]

means_ovlps_bh = [get_mean(data=i) for i in ovlps_obj]
std_ovlps_bh = [get_std(data=i) for i in ovlps_obj]

means_sep_sep = [get_mean(data=i) for i in sep_sep]
std_sep_sep = [get_std(data=i) for i in sep_sep]

means_cls_sep = [get_mean(data=i) for i in cls_sep]
std_cls_sep = [get_std(data=i) for i in cls_sep]

means_ovlps_sep = [get_mean(data=i) for i in ovlps_sep]
std_ovlps_sep = [get_std(data=i) for i in ovlps_sep]



##### SAVE RESULTS (create 'errors_stats' folder)
np.save(testpath+'/total_sep_bh.npy', sep_obj)
np.save(testpath+'/total_cls_bh.npy', cls_obj)
np.save(testpath+'/total_ovlps_bh.npy', ovlps_obj)

np.save(testpath+'/total_sep_sep.npy', sep_sep)
np.save(testpath+'/total_cls_sep.npy', cls_sep)
np.save(testpath+'/total_ovlps_sep.npy', ovlps_sep)
###################################################
np.save(testpath+'/errors_stats/means_mb_bh.npy', means_mb_bh)
np.save(testpath+'/errors_stats/std_mb_bh.npy', std_mb_bh)
np.save(testpath+'/errors_stats/means_fp_bh.npy', means_fp_bh)
np.save(testpath+'/errors_stats/std_fp_bh.npy', std_fp_bh)

np.save(testpath+'/errors_stats/means_mb_sep.npy', means_mb_sep)
np.save(testpath+'/errors_stats/means_fp_sep.npy', means_fp_sep)
np.save(testpath+'/errors_stats/std_mb_sep.npy', std_mb_sep)
np.save(testpath+'/errors_stats/std_fp_sep.npy', std_fp_sep)
np.save(testpath+'/errors_stats/means_uni_sep.npy', means_uni_sep)
np.save(testpath+'/errors_stats/std_uni_sep.npy', std_uni_sep)
###############################################################
np.save(testpath+'/errors_stats/means_sep_bh.npy', means_sep_bh)
np.save(testpath+'/errors_stats/means_cls_bh.npy', means_cls_bh)
np.save(testpath+'/errors_stats/means_ovlps_bh.npy', means_ovlps_bh)
np.save(testpath+'/errors_stats/std_sep_bh.npy', std_sep_bh)
np.save(testpath+'/errors_stats/std_cls_bh.npy', std_cls_bh)
np.save(testpath+'/errors_stats/std_ovlps_bh.npy', std_ovlps_bh)

np.save(testpath+'/errors_stats/means_sep_sep.npy', means_sep_sep)
np.save(testpath+'/errors_stats/means_cls_sep.npy', means_cls_sep)
np.save(testpath+'/errors_stats/means_ovlps_sep.npy', means_ovlps_sep)
np.save(testpath+'/errors_stats/std_sep_sep.npy', std_sep_sep)
np.save(testpath+'/errors_stats/std_cls_sep.npy', std_cls_sep)
np.save(testpath+'/errors_stats/std_ovlps_sep.npy', std_ovlps_sep)
