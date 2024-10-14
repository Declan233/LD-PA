import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from torch import nn
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from sca_metrics import SNR_fit



AES_Sbox = np.array([99, 124, 119, 123, 242, 107, 111, 197, 48, 1, 103, 43, 254, 215, 171, 118, 202, 130, 201, 125, 250, 89, 71,
        240, 173, 212, 162, 175, 156, 164, 114, 192, 183, 253, 147, 38, 54, 63, 247, 204, 52, 165, 229, 241, 113, 216,
        49, 21, 4, 199, 35, 195, 24, 150, 5, 154, 7, 18, 128, 226, 235, 39, 178, 117, 9, 131, 44, 26, 27, 110, 90, 160,
        82, 59, 214, 179, 41, 227, 47, 132, 83, 209, 0, 237, 32, 252, 177, 91, 106, 203, 190, 57, 74, 76, 88, 207, 208,
        239, 170, 251, 67, 77, 51, 133, 69, 249, 2, 127, 80, 60, 159, 168, 81, 163, 64, 143, 146, 157, 56, 245, 188,
        182, 218, 33, 16, 255, 243, 210, 205, 12, 19, 236, 95, 151, 68, 23, 196, 167, 126, 61, 100, 93, 25, 115, 96,
        129, 79, 220, 34, 42, 144, 136, 70, 238, 184, 20, 222, 94, 11, 219, 224, 50, 58, 10, 73, 6, 36, 92, 194, 211,
        172, 98, 145, 149, 228, 121, 231, 200, 55, 109, 141, 213, 78, 169, 108, 86, 244, 234, 101, 122, 174, 8, 186,
        120, 37, 46, 28, 166, 180, 198, 232, 221, 116, 31, 75, 189, 139, 138, 112, 62, 181, 102, 72, 3, 246, 14, 97,
        53, 87, 185, 134, 193, 29, 158, 225, 248, 152, 17, 105, 217, 142, 148, 155, 30, 135, 233, 206, 85, 40, 223, 140,
        161, 137, 13, 191, 230, 66, 104, 65, 153, 45, 15, 176, 84, 187, 22])

HW_byte = np.array([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2,
            3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3,
            3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3,
            4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4,
            3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5,
            6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4,
            4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5,
            6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8])



def load_ref_file(filename, nb_trs:int=None, refidx:int=0):
    '''reference h5 format:
    - Reference0
    -- traces
    -- labels
    - Reference1
    -- traces
    -- labels
    '''
    file = h5py.File(filename, 'r')
    file = file[f"Reference{refidx}"]
    X = file['traces']
    L = np.array(file['labels'], dtype=np.int16).squeeze()
    if nb_trs is not None:
        X, L = X[:nb_trs], L[:nb_trs]
    X = np.array(X, dtype=np.float64)
    return X, L

def load_file(filename, trs_start:int=0, nb_trs:int=None, profile:bool=True):
    '''Load target datasets in h5 format:
    - Profiling_traces
    -- traces
    -- metadata
    --- plaintext
    --- key
    --- masks
    - Attack_traces
    -- traces
    -- metadata
    --- plaintext
    --- key
    --- masks
    '''
    file = h5py.File(filename, 'r')
    file = file["Profiling_traces"] if profile else file["Attack_traces"]
    X = file['traces']
    P = np.array(file['metadata']['plaintext'], dtype=np.int16).squeeze()
    K = np.array(file['metadata']['key'], dtype=np.int16).squeeze()
    try:
        masks_group = file['metadata']['masks']
        masks_exist = True
    except (KeyError, ValueError):
        masks_exist = False
    R = np.array(masks_group, dtype=np.int16).squeeze() if masks_exist else None
    if nb_trs is not None:
        X = X[trs_start:trs_start+nb_trs]
        K = K[trs_start:trs_start+nb_trs]
        P = P[trs_start:trs_start+nb_trs]
        if R is not None:
            R = R[trs_start:trs_start+nb_trs]
    X = np.array(X, dtype=np.float64)
    return X, P, K, R


def compute_iv(P, K, R=None, leakage_model='ID'):
    '''
    Compute the target intermediate variable with respect to leakage model.
    '''
    num_samples = len(P)
    iv_size = 2 if R is not None else 1
    iv = np.zeros((num_samples, iv_size), dtype=np.int16)
    
    if R is not None:
        if leakage_model == "ID":
            iv[:, 0] = R
            iv[:, 1] = AES_Sbox[P ^ K] ^ R
        else:
            iv[:, 0] = HW_byte[R]
            iv[:, 1] = HW_byte[AES_Sbox[P ^ K] ^ R]
    else:
        if leakage_model == "ID":
            iv[:, 0] = AES_Sbox[P ^ K]
        else:
            iv[:, 0] = HW_byte[AES_Sbox[P ^ K]]
    return iv


def compute_label_key_guess(P, leakage_model='ID'):
    '''Compute all hypothesis of label for each key guess'''
    label_key_guess = np.zeros([256, len(P)])
    for key_guess in range(256):
        K = np.ones_like(P)*key_guess
        label = compute_iv(P, K, None, leakage_model)
        label_key_guess[key_guess] = label.squeeze()
    return label_key_guess



def snr_poi(X, iv, leakage_model:str='ID', savefile:str=None, flag:str=None, model:nn.Module=None):
    '''Compute and plot the SNR between traces/features and shares/intermediate variable.'''               
    if model is None:
        snr_val = SNR_fit(X, nb_cls=256 if leakage_model == "ID" else 9, iv=iv)
    else:
        with torch.no_grad():
            f_tar = model(torch.tensor(X).float().cuda())
            f_tar = f_tar.cpu().numpy()
        snr_val = SNR_fit(f_tar, nb_cls=256 if leakage_model == "ID" else 9, iv=iv)

    plot_snr(snr_val, savefile, flag=flag)
    return snr_val


def plot_snr(snr_val, savefile:str=None, show:bool=True, flag=None):
    '''Plot the SNR values'''
    l,n = snr_val.shape
    x = np.arange(0,n)+1
    plt.figure(figsize=(6, 4))
    for i in range(0, l):
        plt.plot(x, snr_val[i], label=f'share {i}' if flag is None else flag[i])
    plt.legend()
    plt.xlim(0, n+1)
    plt.ylabel('SNR values')
    plt.xlabel('Features')
    plt.tight_layout(pad=0)
    if savefile is not None:
        plt.savefig(savefile)
    if show:
        plt.show()
    plt.close()


def plot_log(log_file, savefile:str=None, zoom:bool=True, zoominbound:list=[0, 1.0], show:bool=True):
    df = pd.read_csv(log_file).drop('epoch', axis=1)
    _, ax = plt.subplots()
    for column in df.columns:
        ax.plot(df.index, df[column], label=column)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend(loc='upper right')
    
    if zoom:
        total_epoch = df.shape[0]
        left, bottom, width, height = 0.03, 0.47, 0.5, 0.5
        axins = ax.inset_axes([left, bottom, width, height], transform=ax.transAxes)
        for column in df.columns:
            axins.plot(df.index, df[column], label=column)
        axins.set_xlim(total_epoch-50, total_epoch)
        axins.set_ylim(zoominbound[0], zoominbound[1])
        axins.yaxis.tick_right()
        for spine in axins.spines.values():
            spine.set_edgecolor('black')

        ax.annotate("", xy=(total_epoch-50, zoominbound[0]), xycoords='data', xytext=(left, bottom), textcoords='axes fraction',
                    arrowprops=dict(arrowstyle="-", connectionstyle="arc3"))
        ax.annotate("", xy=(total_epoch, zoominbound[1]), xycoords='data', xytext=(left+width, bottom+height), textcoords='axes fraction',
                    arrowprops=dict(arrowstyle="-", connectionstyle="arc3"))
        ax.plot([total_epoch-50,total_epoch-50], zoominbound, 'black')
        ax.plot([total_epoch,total_epoch], zoominbound, 'black')
        ax.plot([total_epoch-50,total_epoch], [zoominbound[0],zoominbound[0]], 'black')
        ax.plot([total_epoch-50,total_epoch], [zoominbound[1],zoominbound[1]], 'black')

    if savefile is not None:
        plt.savefig(savefile)
    if show:
        plt.show()
    plt.close()


def plot_GEnSR(GE, SR, savefile:str=None, show:bool=True):
    assert GE is not None or SR is not None, print("GE and SR cannot both be None.")
    x = np.arange(0,len(GE))+1
    _, ax1 = plt.subplots()
    if SR is not None and GE is not None:
        ax1.plot(x, GE, color='b', label='GE')
        if GE[-1] < 2: # converged
            result_number_of_traces_ge_1 = len(GE) - np.argmax(GE[::-1] > 2)
            ax1.axvline(x=result_number_of_traces_ge_1, color='g', linestyle='--', label=f'#traces for GE=1: {result_number_of_traces_ge_1}')
        ax1.set_xlabel('num_traces')
        ax1.set_ylabel('Guseesing Entropy', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax2 = ax1.twinx()
        ax2.plot(x, SR, color='r', label='SR')
        ax2.set_ylabel('Success Rate', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1, labels1, loc='upper left') 
        ax2.legend(lines2, labels2, loc='upper right')
    elif SR is not None:
        ax1 = plt.gca()
        ax1.plot(x, SR, label='SR')
        ax1.legend()
    elif GE is not None:
        ax1 = plt.gca()
        ax1.plot(x, GE, label='GE') 
        if GE[-1] < 2: # converged
            result_number_of_traces_ge_1 = len(GE) - np.argmax(GE[::-1] > 2)
            ax1.axvline(x=result_number_of_traces_ge_1, color='g', linestyle='--', label=f'#traces for GE=1: {result_number_of_traces_ge_1}')
        ax1.legend()

    if savefile is not None:
        plt.savefig(savefile)
    if show:
        plt.show()
    plt.close()


def get_filename(type, ref, target, target_byte=None, flag=None):
    '''File name generation for different experiments.'''
    if type=='model':
        format = '.pth'
    elif type=='log':
        format = '.log'
    elif type in ['snr', 'ge']:
        format = '.csv'
    else:
        format = '.eps'
    filename = f'{type}_{ref}-{target}'
    if target_byte is not None:
        filename += f'{target_byte}'
    if flag is not None:
        filename += f'_{flag}'
    filename += f'{format}'
    return filename