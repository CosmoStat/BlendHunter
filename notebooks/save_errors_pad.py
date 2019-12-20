import numpy as np

from annex_new import import_
from annex_new import get_distance
from annex_new import get_bh_errors
from annex_new import get_sep_errors
from annex_new import get_distance_errors
from annex_new import count_per_bin
from annex_new import get_bh_results
from annex_new import get_sep_results

from os.path import expanduser
user_home = expanduser("~")
path = user_home+'/Documents/Cosmostat/Codes/BlendHunter'

#Different noise realisations
sigmas = [5,14,18,26,35,40]
noise_realisation = ['',1,2,3,4]
datasets = [[str(j)+str(i) for i in noise_realisation]  for j in sigmas]


def mb_(data=None, sep=False): #Missed blends
    if sep:
        return len(np.where(data[0:4000] == 0)[0])
    else:
        return len(np.where(data[0:4000] != 'blended')[0])

def fp_(data=None, sep=False): #Single galaxy mistaken for a blend
    if sep:
        return len(np.where(data[4000:8000] == 1)[0])
    else:
        return len(np.where(data[4000:8000] != 'not_blended')[0])

def uni_(data=None): #Unidentified by SExtractor
    return len(np.where(data[0:4000] == 16)[0])+len(np.where(data[4000:8000] == 16)[0])


####Separated objects
def sep_(data=None, sep_results= False, dist=None): 
    if sep_results:
        return len([i for i in np.where(data[0:4000] != 1)[0] if distance[i] > 20.0])
    else:
        return len([i for i in np.where(data[0:4000] != 'blended')[0] if dist[i] > 20.0])

####Close objects
def cls_(data=None, sep_results= False, dist=None):
    if sep_results:
        return len([i for i in np.where(data[0:4000] != 1)[0] if  6.0 < distance[i] <= 20.0])
    else:
        return len([i for i in np.where(data[0:4000] != 'blended')[0] if  6.0 < dist[i] <= 20.0])

####Overlapping objects
def ovlps_(data=None, sep_results= False, dist=None):
    if sep_results:
        return len([i for i in np.where(data[0:4000] != 1)[0] if distance[i] <= 6.0 ])
    else:
        return len([i for i in np.where(data[0:4000] != 'blended')[0] if dist[i] <= 6.0 ])

def get_mean(data=None):
    return np.mean(data)

def get_std(data=None):
    return np.std(data)

#Results
bh_pad = get_bh_results(path_bh_results =path+'/bh_pad_results', pad_images=True)
sep_pad = get_sep_results(path_sep_results =path+'/sep_pad_results', pad_images=True)

#Import distance between galaxies
distance = get_distance(path=path+'/bh')

#####Missed blends and false positives for bh and sep
bh_mb = [[mb_(data=x[j]) for j in range(len(x))] for x in bh_pad]
sep_mb = [[mb_(data=x[j], sep=True) for j in range(len(x))] for x in sep_pad]

bh_fp = [[fp_(data=x[j]) for j in range(len(x))] for x in bh_pad]
sep_fp = [[fp_(data=x[j], sep=True) for j in range(len(x))] for x in sep_pad]

#Unidentified objects by sep
sep_uni = [[uni_(data=x[j]) for j in range(len(x))] for x in sep_pad]


#####Separated, close and overlapping objects in errors
#Blendhunter
sep_obj = [[sep_(data[k], dist=distance) for k in range(len(data))] for data in bh_pad]
cls_obj = [[cls_(data[k], dist=distance) for k in range(len(data))] for data in bh_pad]
ovlps_obj= [[ovlps_(data[k], dist=distance) for k in range(len(data))] for data in bh_pad]

#SExtractor
sep_sep = [[sep_(data=flags[i], sep_results=True, dist=distance) for i in range(len(flags))] for flags in sep_pad]
cls_sep = [[cls_(data=flags[i], sep_results=True, dist=distance) for i in range(len(flags))] for flags in sep_pad]
ovlps_sep = [[ovlps_(data=flags[i], sep_results=True, dist=distance) for i in range(len(flags))] for flags in sep_pad]

##########Get means and std deviation for error bars

#########MB and FP
means_mb_bh = [get_mean(data=i) for i in bh_mb]
std_mb_bh = [get_std(data=i) for i in bh_mb]

means_fp_bh = [get_mean(data=i) for i in bh_fp]
std_fp_bh = [get_std(data=i) for i in bh_fp]

means_mb_sep = [get_mean(data=i) for i in sep_mb]
std_mb_sep = [get_std(data=i) for i in sep_mb]

means_fp_sep = [get_mean(data=i) for i in sep_fp]
std_fp_sep = [get_std(data=i) for i in sep_fp]

means_uni_sep = [get_mean(data=i) for i in sep_uni]
std_uni_sep = [get_std(data=i) for i in sep_uni]


########SEPERATED, CLOSE and OVERLAPS
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
np.save(path+'/errors_stats_pad/sep_bh_pad.npy', sep_obj)
np.save(path+'/errors_stats_pad/cls_bh_pad.npy', cls_obj)
np.save(path+'/errors_stats_pad/ovlps_bh_pad.npy', ovlps_obj)

np.save(path+'/errors_stats_pad/sep_sep_pad.npy', sep_sep)
np.save(path+'/errors_stats_pad/cls_sep_pad.npy', cls_sep)
np.save(path+'/errors_stats_pad/ovlps_sep_pad.npy', ovlps_sep)
###################################################
np.save(path+'/errors_stats_pad/means_mb_bh_pad.npy', means_mb_bh)
np.save(path+'/errors_stats_pad/std_mb_bh_pad.npy', std_mb_bh)
np.save(path+'/errors_stats_pad/means_fp_bh_pad.npy', means_fp_bh)
np.save(path+'/errors_stats_pad/std_fp_bh_pad.npy', std_fp_bh)

np.save(path+'/errors_stats_pad/means_mb_sep_pad.npy', means_mb_sep)
np.save(path+'/errors_stats_pad/means_fp_sep_pad.npy', means_fp_sep)
np.save(path+'/errors_stats_pad/std_mb_sep_pad.npy', std_mb_sep)
np.save(path+'/errors_stats_pad/std_fp_sep_pad.npy', std_fp_sep)
np.save(path+'/errors_stats_pad/means_uni_sep_pad.npy', means_uni_sep)
np.save(path+'/errors_stats_pad/std_uni_sep_pad.npy', std_uni_sep)
###############################################################
np.save(path+'/errors_stats_pad/means_sep_bh_pad.npy', means_sep_bh)
np.save(path+'/errors_stats_pad/means_cls_bh_pad.npy', means_cls_bh)
np.save(path+'/errors_stats_pad/means_ovlps_bh_pad.npy', means_ovlps_bh)
np.save(path+'/errors_stats_pad/std_sep_bh_pad.npy', std_sep_bh)
np.save(path+'/errors_stats_pad/std_cls_bh_pad.npy', std_cls_bh)
np.save(path+'/errors_stats_pad/std_ovlps_bh_pad.npy', std_ovlps_bh)

np.save(path+'/errors_stats_pad/means_sep_sep_pad.npy', means_sep_sep)
np.save(path+'/errors_stats_pad/means_cls_sep_pad.npy', means_cls_sep)
np.save(path+'/errors_stats_pad/means_ovlps_sep_pad.npy', means_ovlps_sep)
np.save(path+'/errors_stats_pad/std_sep_sep_pad.npy', std_sep_sep)
np.save(path+'/errors_stats_pad/std_cls_sep_pad.npy', std_cls_sep)
np.save(path+'/errors_stats_pad/std_ovlps_sep_pad.npy', std_ovlps_sep)

