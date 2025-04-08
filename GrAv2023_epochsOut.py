### FILE: GrAv2023_epochsOut.py
### UPDATED: 08/04/2025
### AUTHOR: pbenedetti@itba.edu.ar
### TASK: Takes epochs as an input and having bads_preICA and bads_postICA
###         for a given band, for each subject and each condition, identifies
###         epochs wich PSD is an outlier. It does so by computing the means
###         of the epoch for all channels and freq of the band. Bads channels
###         are excluded.
### OUTPUT: psd_outliers_{band_name}.csv
### protocol2023_V2 --> GrAv2023_epochsOut.py

#%% LIBRARIES
import mne
import numpy as np
import pandas as pd
import mne
import numpy as np
import pandas as pd
import os
from my_functions import find_outliers

#%% CONFIG
filepath = "E:/Doctorado/protocol2023/epochs/"  # PATH
subjects = [ "01_test_2023",
    "02_test_2023", 
    "04_test_2023",
    "05_test_2023",
    "06_test_2023",
    "07_test_2023",
    "10_test_2023",
    "11_test_2023",
    "13_test_2023",
    "14_test_2023",
    "15_test_2023",
    "16_test_2023", 
    "20_test_2023", 
    "21_test_2023",
    "22_test_2023",
    "23_test_2023",
    "24_test_2023",
    "25_test_2023", 
    "26_test_2023",
    "28_test_2023",
    "29_test_2023",
    "30_test_2023_full",
    "31_test_2023",
    "34_test_2023"]  # Lista de sujetos (ajustar según corresponda)

conditions = ["Resting", "REY Thinking", "REY", "AUT Thinking", "AUT"]  # CONDITIONS
bands = {"Theta": (4, 8), "Alpha": (8, 12)}  # BANDS

# BAD CHANNELS
bads_preICA = [ ["B2","C4","C30",'D4', "D5",'D10','D12'],  # s=0 - 01
    ["C10", "D8", "D9", "D24", "D25"],  # s=1 - 02
    ["B4", "B8", "B9", "C29", "D5", "D23"],  # s=2 - 04
    ["B4", "B8", "B9", "C29", "D5", "D23"],  # s=3 - 05
    ["B6", "D10", "D25", "D29"],  # s=4 - 06
    ["C8"],  # s=5 - 07
    ["A10", "A17", "A27", "B4", "B30", "B31", "C17", "C29", "C32", "D19"],  # s=6 - 10
    ['B13'], #s= 7 - 11_test_2023 
    ['A23','B20','B21','B23','C16','C29','C30','C32','D3','D23'], #s=9 - 13 
    ['A6','A7','A12','A13','A26','D22','D23'],#s= 10 - 14_test_2023
    ['A9','A30','B7','C21','C22','D18','C8','C14'],#s = 11 - 15_test_2023
    ['B24','D31','D32'],#s = 12 - 16_test_2023
    ['A6','D3','D27'],#s = 13 - 20_test_2023
    ['A12','A13','A14','B8','B9','B28','D11','D23'],#s = 14 - 21_test_2023
    ['A32','C4','C5','C6','C7','C8','C16','C17','C29'],#s = 15 - 22_test_2023
    ['C14','C15','D4'],#s = 16 - 23_test_2023 #MUY FEO.
    ['A32','B9','B27','C16','D20','D30','D32'],
    ['A10','A24','A25','A32','B23','B25','C18','C23','C24','C28','D31','D32'],#s = 18 - 25_test_2023
    ['A17','A25','C6','D17','D19','D22','D23','D24','D28','D32'],#s = 19 - 26_test_2023
    ['B8','B9','B26','C16'],#s = 20 - 28_test_2023
    ['A14','A21','A22','A31','B3','B4','B24','C2','C23','D22','D23'],#s = 21 - 29_test_2023
    ['A15', 'D3', 'D11', 'A6', 'B1','B13' , 'C26', 'B6', 'A20'],#s = 22 - 30_test_2023
    ['A11','B21','C16','C17','C29','C30','D5'],#s = 24 - 31_test_2023
    ['A24','B1','B2','B18','B19','B28','B32','C6','C28','D8','D11','D12','D26']#s = 25 - 34_test_2023
    ]  # Canales malos antes de ICA por sujeto
bads_postICA = [ [],  # s=0 - 01
    ["B25"],  # s=1 - 02
    [],  # s=2 - 04
    [],  # s=3 - 05
    [],  # s=4 - 06
    [],  # s=5 - 07
    [],  # s=6 - 10
    [],
    [],#s= 9 - 13_test_2023
    [],#s= 10 - 14_test_2023
    ["A31","B6"],#s = 11 - 15_test_2023
    ["B26"],#s = 12 - 16_test_2023
    [],
    [],#s = 14 - 21_test_2023
    [],#s = 15 - 22_test_2023
    [],#s = 16 - 23_test_2023
    [],#s = 17 - 24_test_2023
    [],#s = 18 - 25_test_2023
    [],#s = 19 - 26_test_2023
    [],#s = 20 - 28_test_2023
    [],#s = 21 - 29_test_2023
    ['C30'],#s = 22 - 30_test_2023
    [],#s = 24 - 31_test_2023
    ["A11","B7","B22"]#s = 25 - 34_test_2023
    ]  # Canales malos después de ICA por sujeto

# MERGE BAD CHANNELS
bads_total = [list(set(pre + post)) for pre, post in zip(bads_preICA, bads_postICA)]

for band_name, (fmin, fmax) in bands.items():
    # DIC TO STORE OUTLIERS
    outliers_dict = {subject: {condition: [] for condition in conditions} for subject in subjects}
    
    for i, subject in enumerate(subjects):
        # LOADS EPOCHS OFTHE SUBJECT
        epochs_path = os.path.join(filepath, f"{subject}-epo.fif")
        epochs = mne.read_epochs(epochs_path, preload=True)
        
        # EXCLUDES BAD CHANNELS
        epochs.drop_channels(bads_total[i])
        
        for condition in conditions:
            if condition in epochs.event_id:
                # COMPUTES PSDFOT THE EPOCH. SHAPE: (n_epochs, n_channels, n_freqs)
                psd = epochs[condition].compute_psd()
                psd_data = psd.get_data()  # Forma: (n_epochs, n_channels, n_freqs)
                freqs = psd.freqs
                
                band_mask = (freqs >= fmin) & (freqs <= fmax) #SELECTS ONLY BAND OF INTEREST
                psd_band = psd_data[:, :, band_mask].mean(axis=2)  # FRECUENCY MEAN
                psd_mean = psd_band.mean(axis=1)  # CHANNELS MEAN
                
                # IDENTIFIES OUTLIERS WITHIN CONDITION AND SUBJECT.
                _, outliers = find_outliers(psd_mean)
                outliers_dict[subject][condition].extend(outliers[0]) #SAVES OUTLIERS EPOCHS.
                
                print(f"{subject} - {condition} - {band_name}: {len(outliers[0])} outliers detectados")
    
    #%% CONVERTS TO DataFrame y SAVE AS CSV
    df_outliers = pd.DataFrame.from_dict(outliers_dict, orient="index")
    df_outliers = df_outliers.applymap(lambda x: ", ".join(map(str, x)) if x else "")
    csv_filename = f"Outputs/psd_outliers_{band_name}.csv"
    df_outliers.to_csv(csv_filename)
    print(f"Análisis completado y guardado en {csv_filename}")
