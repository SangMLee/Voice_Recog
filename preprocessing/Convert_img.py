import os
import glob
import random
import pickle
import librosa
import librosa.display 

import numpy as np
import pandas as pd
import tensorflow as tf

import IPython.display as ipd
import matplotlib.pyplot as plt

#%matplotlib inline
random.seed(123)

#Creating lists of directors for the file paths
sor_dir = "../../audio/data/VOiCES/source-16k"
hol_dir = "../../audio/data/Hold_Out_Set/source-16k"
bkg_rm1_dir = "../../audio/data/VOiCES/distant-16k/distractors"

sFile_lst = glob.glob("{}/*/*.wav".format(sor_dir))
hFile_lst = glob.glob("{}/*/*.wav".format(hol_dir))
bFile_lst = glob.glob("{}/rm1/*.wav".format(bkg_rm1_dir))

len(bFile_lst)

rem = 0
for i in range(len(bFile_lst)):
    if 'none' in bFile_lst[i-rem]:
        bFile_lst.pop(i-rem)
        rem += 1

for i in bFile_lst:
    if 'none' in i:
        print(i)

# Initial test - 720 Files 
#bFile_lst = bFile_lst[:20]
sFile_lst = sFile_lst[:1297]

for i in range(4):
    random.shuffle(bFile_lst)

# Make sure source file is loading properly.
fig = plt.gcf()
DPI = fig.get_dpi()
# Make a list of speakers for each file ( Creating Labels )
def find_id(sfname):
    #Creating a list of speaker labels to be used as labels for the Deep Learning Model
    sspker_label = [name[name.find("-sp")+1:name.find("-ch")] for name in sfname]
    return sspker_label

def find_sinfo(sfname):
    #Creating a list of source information to be used as image names for the Deep Learning Model
    info = [name[name.find("-sp")+1:name.find(".wav")] for name in sfname]
    return info

def find_binfo(sfname):
    #Creating a list of source information to be used as image names for the Deep Learning Model
    info = [name[name.find("-rm")+1:name.find(".wav")] for name in sfname]
    return info

# Creating list of path names to source file  
#sf_lst = sFile_lst + hFile_lst 
sf_lst = sFile_lst

b_len = len(bFile_lst)
s_len = (len(sf_lst))
sf_lst = np.repeat(sf_lst, b_len)
# Creating list of bFiles the length of the other 
bkg_lst = bFile_lst * s_len

# Creating list of speaker ID for each file path
spkr_id = find_id(sf_lst)
s_info = find_sinfo(sf_lst)
b_info = find_binfo(bkg_lst)

df = pd.DataFrame({'filePath': sf_lst,'bkg_noise': bkg_lst ,'spkr_id':spkr_id, 's_info':s_info, 'b_info':b_info})

#Create list of speaker ids 
spkr_lst = df['spkr_id'].unique()

len(spkr_lst)

#Check contents of DataFrame 
print(df.head())
print(df.filePath[0])
print(df.filePath[30])
sor_dir = "../../audio/data/VOiCES/source-16k"
hol_dir = "../../audio/data/Hold_Out_Set/source-16k"
bkg_rm1_dir = "../../audio/data/VOiCES/distant-16k/distractors"

def img_dir(spkr_lst):
    for i in spkr_lst:
        if not os.path.exists('../data/test/{}'.format(i)):
             os.makedirs('../data/test/{}'.format(i))
        if not os.path.exists('../data/train/{}'.format(i)):
             os.makedirs('../data/train/{}'.format(i))
        if not os.path.exists('../data/vali/{}'.format(i)):
             os.makedirs('../data/vali/{}'.format(i))

img_dir(spkr_lst)

import os.path as path
import gc

random.seed(123)

def wav_img(df, dpi, size=(224,224)): 
    count = 0
    vali_count = 0
    current = ''
    for i in range(len(df)):
        if df['spkr_id'][i] not in current:
            print(df['spkr_id'][i])
            current = df['spkr_id'][i]
        test = '../data/test/{}/{}_{}.png'.format(df['spkr_id'][i], df['s_info'][i], df['b_info'][i])
        vali = '../data/vali/{}/{}_{}.png'.format(df['spkr_id'][i], df['s_info'][i], df['b_info'][i])
        train = '../data/train/{}/{}_{}.png'.format(df['spkr_id'][i],df['s_info'][i], df['b_info'][i])
        if (not path.exists(train)) and (not path.exists(test)) and (not path.exists(vali)):
            plt.figure(figsize=(5,5))   
            # use librosa library to read in the wave files
            src_x, src_sr = librosa.load(df['filePath'][i])

            # Cut bkground file to the same size of audio file
            src_len = librosa.get_duration(filename=df['filePath'][i])
            bkg_len = librosa.get_duration(filename=df['bkg_noise'][i])
            time_diff = int(bkg_len - (src_len+1))
            rand_time = random.uniform(0, time_diff)

            # Read bkgfile
            bkg_x, _ = librosa.load(df['bkg_noise'][i], offset=rand_time, duration=src_len+10)
            
            if len(bkg_x[:len(src_x)]) < len(src_x):
                print("Debugging .....")
                
                print("Length of src file:", src_len)
                print("Length of src array:", len(src_x))
                
                print("Length of src file:", bkg_len)
                print("Length of src array:", len(bkg_x))
                print("New Length of bkg array:", bkg_x[:len(src_x)])
                
                print("Time difference between src and bkg:", time_diff)
                print("Random starting time for bkg:", rand_time)
                
            
            # Create list with source and bkg
            n_x = ((bkg_x[:len(src_x)]*15)+(src_x/2)) # <---- Try with bkg increase by factor of 10
            n_sr = src_sr

            # Create spectrogram 
            n_ft = librosa.stft(n_x)
            n_db = librosa.amplitude_to_db(abs(n_ft))
            librosa.display.specshow(n_db, sr=n_sr, y_axis='hz')
            plt.ylim(0, 8000)
            plt.ylabel("")
            plt.yticks([])
            
            if count != 0 and count % 5 == 0:
                if vali_count != 0 and vali_count % 4 == 0:
                    plt.savefig(test, dpi=dpi, bbox_inches='tight', transparent=True)
                else:
                    plt.savefig(vali, dpi=dpi, bbox_inches='tight', transparent=True)
                vali_count += 1
                
            else:
                plt.savefig(train, dpi=dpi, bbox_inches='tight', transparent=True)
                
            plt.close()
            if count/len(df)*100 % 10 == 0:
                print("Progress :", count/len(df)*100,"%")
            count += 1 
            # Make the random seed placement the same
            if count % 50 ==0:
              gc.collect()
        else:
            rand_time = random.uniform(0, 100)
            count += 1

wav_img(df, DPI)

