#Script with annex codes to import in notebooks for results visualisation
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns #install seaborn

#Import function
def import_(path):
    img = np.load(path, allow_pickle=True)
    return img

def get_bh_results(path_bh_results = None, pad_images=False):
    sigmas = [5,14,18,26,35,40]
    noise_realisation = ['',1,2,3,4]
    datasets = [[str(j)+str(i) for i in noise_realisation]  for j in sigmas]
    
    if pad_images:
        return [[import_(path_bh_results+'/preds_pad{}.npy'.format(i)) for i in datasets[j]] for j in range(len(datasets))]
    
    else:
        return [[import_(path_bh_results+'/pred_{}.npy'.format(i)) for i in datasets[j]] for j in range(len(datasets))]

def get_sep_results(path_sep_results=None, pad_images=False):
    sigmas = [5,14,18,26,35,40]
    noise_realisation = ['',1,2,3,4]
    datasets = [[str(j)+str(i) for i in noise_realisation]  for j in sigmas]
    
    if pad_images:
        return [[import_(path_sep_results+'/flags_pad{}.npy'.format(i)) for i in datasets[j]] for j in range(len(datasets))]
    else:
        return [[import_(path_sep_results+'/flags{}.npy'.format(i)) for i in datasets[j]] for j in range(len(datasets))]

#Get distance between galaxies
def get_distance(path):
    #X parameter extracted from test images
    param_x = np.load(path+"/param_x_total.npy", allow_pickle=True)[36000:40000]    
    #Y parameter extracted from test images
    param_y = np.load(path+"/param_y_total.npy", allow_pickle=True)[36000:40000]    
    #Compute distance
    distance = np.array([np.sqrt(param_x**2 + param_y**2)[i][0] for i in range(4000)])
    
    return distance

#Get ellipticity components
def get_ellipticity(path = None, get_e1=False, get_e2=False):
    if get_e1:
        e1 = np.load(path+"/bh/e1_total.npy", allow_pickle=True)      
        return e1
    
    if get_e2:
        e2 = np.load(path+"/bh/e2_total.npy", allow_pickle=True)[36000:40000]
        return e2
    else:
        e1 = np.load(path+"/bh/e1_total.npy", allow_pickle=True)
        e2 = np.load(path+"/bh/e2_total.npy", allow_pickle=True)[36000:40000]
        
        return e1, e2  
    
#Get bh errors 
def get_bh_errors(results=None, get_false_positives=False):

    if get_false_positives:
        return np.where(results[4000:8000] != 'not_blended')[0]
    
    else:
        return np.where(results[0:4000] != 'blended')[0]

    
#Get sep errors
def get_sep_errors(results=None, get_false_positives=False, get_unidentified=False):
    
    if get_false_positives:
        return np.where(results[4000:8000] != 0)[0]
    
    if get_unidentified:
        return np.where(results[0:8000] == 16)[0]
    
    else:
        return np.where(results[0:4000] != 1)[0]
    
#Get distance for missed blends by bh and sep
def get_distance_errors(distances=None, errors=None):
    return [distances[i] for i in errors]


# How to retrieve informations from each bin in total dataset
def count_per_bin(data =None, get_bins =False, bins_=int(180/3)):
    (n, bins, patches) = plt.hist(data, bins = bins_)
    if get_bins:
        return n, bins[1:], bins
    else:
        return n
    
# Computation of error ratios
def acc_ratio_bins(data=None, N=None, bins=int(180/3)):
    n = count_per_bin(data, bins_=bins)
    ratio = 1 - (n/N)   
    return ratio

## Compute means 
def get_mean_acc(data=None, data_total=None, nb_ratios=60, get_mean_total=False):
    
    if get_mean_total:
        #Get total number per bin and mean distance per bin for the whole test set
        n_total, mean_dist, bin_edges = count_per_bin(data=data_total, get_bins=True)  
        return mean_dist
    else:
        #Get total number per bin and mean distance per bin for the whole test set
        n_total, mean_dist, bin_edges = count_per_bin(data=data_total, get_bins=True)  
        #Compute accuracy   
        acc_ratios = [[acc_ratio_bins(x[j], N= n_total , bins=bin_edges) for j in range(len(x))] for x in data]
        
        sub_ratios = [[np.array([acc_ratios[k][i][j] for i in range(len(acc_ratios[k]))]) for j in range(nb_ratios)] for k in                                   range(len(acc_ratios))]

        return [np.array([np.mean(k[i]) for i in range(len(k))]) for k in sub_ratios]


def plot_distribution(total_data, bh_data=None, sep_data=None, bh_distribution=False, sep_distribution=False,
                     font_size=13, TITLE='None', nb_col=2, nb_lines=3, sigma_val=None, 
                     xlabel='Distance between objects (pixels)', ylabel='Number of images', legend_size=18, size_plot= (16,20)):
    #Font dictionnary
    font = {'family': 'monospace',
            'color':  'k',
            'weight': 'normal',
            'size': font_size}
    
    if bh_distribution:
        #Seaborn theme
        sns.set(context='notebook', style='whitegrid', palette='deep')
        #Start plot
        fig, axs = plt.subplots(nb_lines,nb_col, figsize=size_plot, sharex=False)
        #Title
        fig.suptitle(TITLE, fontdict = {'family': 'serif','color':  'k','weight': 'heavy','size': 23})
        axs = axs.ravel()
        for i,j,k in zip(range(len(sigma_val)), range(len(bh_data)), sigma_val):
            axs[i].set_title('$\sigma_{noise}$ = {}'.format(k), fontdict=font, fontsize=15.5)
            axs[i].hist(total_data, color = 'steelblue', edgecolor = 'black', bins = int(180/3), label='Total dataset')
            axs[i].hist(bh_data, color = 'y', edgecolor = 'black', bins = int(180/3), label='BH errors')
            axs[i].set_ylabel(ylabel, fontdict = font)
            axs[i].set_xlabel(xlabel, fontdict = font)
            axs[i].tick_params(axis='both', which='major', labelsize=12)
            axs[i].legend(borderaxespad=0.1, loc="upper left", fontsize=legend_size, prop ={'family': 'monospace'})
    
        plt.show()
        
    if sep_distribution:
        sns.set(context='notebook', style='whitegrid', palette='deep')
        fig, axs = plt.subplots(nb_lines,nb_col, figsize=size_plot, sharex=False)
        fig.suptitle(TITLE, fontdict = {'family': 'serif','color':  'k','weight': 'heavy','size': 23})
        axs = axs.ravel()
        for i,j,k in zip(range(len(sigma_val)), range(len(sep_data)), sigma_val):
            axs[i].set_title('$\sigma_{noise}$ = {}'.format(k), fontdict=font, fontsize=15.5)
            axs[i].hist(total_data, color = 'steelblue', edgecolor = 'black', bins = int(180/3), label='Total dataset')
            axs[i].hist(sep_data, color = 'r', edgecolor = 'black', bins = int(180/3), label='SExtractor errors')
            axs[i].set_ylabel(ylabel, fontdict = font)
            axs[i].set_xlabel(xlabel, fontdict = font)
            axs[i].tick_params(axis='both', which='major', labelsize=12)
            axs[i].legend(borderaxespad=0.1, loc="upper left", fontsize=legend_size, prop ={'family': 'monospace'})
    
        plt.show()
    
    else:
        sns.set(context='notebook', style='whitegrid', palette='deep')
        fig, axs = plt.subplots(nb_lines,nb_col,figsize=size_plot, sharex=False)
        fig.suptitle(TITLE, fontdict = {'family': 'serif','color':  'k','weight': 'heavy','size': 26})
        axs = axs.ravel()
        for i,j,k in zip(range(len(sigma_val)), range(len(bh_data)), sigma_val):
            axs[i].set_title('sigma_noise = {}'.format(k), fontdict=font, fontsize=15.5)
            axs[i].hist(total_data, color = 'steelblue', edgecolor = 'black', bins = int(180/3), label='Total dataset')
            axs[i].hist(sep_data[j], color = 'r', edgecolor = 'black', bins = int(180/3), label='SExtractor errors')
            axs[i].hist(bh_data[j], color = 'y', edgecolor = 'black', bins = int(180/3), label='BH errors')
            axs[i].set_ylabel(ylabel, fontdict = font)
            axs[i].set_xlabel(xlabel, fontdict = font)
            axs[i].tick_params(axis='both', which='major', labelsize=12)
            axs[i].legend(borderaxespad=0.1, loc="upper left", fontsize=legend_size, prop ={'family': 'monospace'})

    
    plt.show()
    
def plot_acc_distances(x_axis =None, accuracy_bh=None, accuracy_sep=None,
                       TITLE = 'None', nb_col=2, nb_lines=3, size_plot=(20,17) , sigma_val=None,
                      xlabel = 'Distance between objects (pixels)', ylabel = 'Accuracy (%)', filling_label='Gain on SExtractor',
                      label2='SExtractor'):    
    font = {'family': 'monospace',
        'color':  'k',
        'weight': 'normal',
        'size': 15}

    sns.set(context='notebook', style='whitegrid', palette='deep')
    fig, axs = plt.subplots(nb_lines,nb_col,figsize=size_plot, sharex=False)
    fig.suptitle(TITLE, fontdict = {'family': 'serif','color':  'k','weight': 'heavy','size': 23})
    axs = axs.ravel()
    for i,j,k in zip(range(len(sigma_val)), range(len(accuracy_bh)), sigma_val):
        axs[i].set_title('sigma_noise = {}'.format(k), fontdict=font, fontsize=18.5)
        axs[i].plot(x_axis, 100*accuracy_bh[j], color = 'k', marker='.', label='BH')
        axs[i].plot(x_axis, 100*accuracy_sep[j], color = 'steelblue', marker='.', label=label2)
        axs[i].set_ylabel(ylabel, fontdict = font)
        axs[i].set_xlabel(xlabel, fontdict = font)
        axs[i].set_ylim(0,100)
        axs[i].tick_params(axis='both', which='major', labelsize=15)
        axs[i].fill_between(x_axis, 100*accuracy_bh[j], y2=100*accuracy_sep[j],
                     where=100*accuracy_bh[j] > 100*accuracy_sep[j], 
                            interpolate=True, hatch="/", edgecolor="k", alpha=0.3 ,label=filling_label)
        axs[i].legend(borderaxespad=0.1, loc="lower center", fontsize=18, prop ={'family': 'monospace','size': 15})

    plt.subplots_adjust(hspace=0.4)
    plt.show()
    