### FILE: GrAv2023_means.py
### UPDATED: 08/04/2025
### AUTHOR: pbenedetti@itba.edu.ar
### TASK: Takes epochs as an input and having bads_preICA, bads_postICA
###         and psd_outliers_{band_name}.csv. For each subject, condiion 
###         and channel computes a mean value for a given band.
###         OUTPUT: psd_cleaned_avg_{band_name}.csv
### protocol2023_V2 --> GrAv2023_epochsOut.py --> GrAv2023_means.py
#%% LIBRARIES
import mne
import numpy as np
import pandas as pd
import os

#%% CONFIG
filepath = "E:/Doctorado/protocol2023/epochs/"  # Ruta de los archivos de épocas
subjects = [ "01_test_2023", "02_test_2023", "04_test_2023", "05_test_2023",
    "06_test_2023","07_test_2023", "10_test_2023", "11_test_2023",
    "13_test_2023","14_test_2023","15_test_2023","16_test_2023", 
    "20_test_2023", "21_test_2023","22_test_2023","23_test_2023",
    "24_test_2023","25_test_2023", "26_test_2023","28_test_2023",
    "29_test_2023","30_test_2023_full","31_test_2023","34_test_2023"]

conditions = ["Resting", "REY Thinking", "REY",
              "AUT Thinking", "AUT"]  # Condiciones
bands = {"Theta": (4, 8), "Alpha": (8, 12)}  # Bandas de frecuencia

# CHANNELS LIST
all_channels = [
    "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15", "A16", "A17", "A18", "A19", "A20", "A21", "A22", "A23", "A24", "A25", "A26", "A27", "A28", "A29", "A30", "A31", "A32", 
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "B12", "B13", "B14", "B15", "B16", "B17", "B18", "B19", "B20", "B21", "B22", "B23", "B24", "B25", "B26", "B27", "B28", "B29", "B30", "B31", "B32", 
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26", "C27", "C28", "C29", "C30", "C31", "C32", 
    "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14", "D15", "D16", "D17", "D18", "D19", "D20", "D21", "D22", "D23", "D24", "D25", "D26", "D27", "D28", "D29", "D30", "D31", "D32"
]
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

# Diccionario con los canales excluidos por sujeto
excluded_channels = {
    subject: list(set(bads_preICA[i] + bads_postICA[i])) for i, subject in enumerate(subjects)
}

# Cargar los archivos de outliers
df_outliers_alpha = pd.read_csv("Outputs/psd_outliers_Alpha.csv", index_col=0)
df_outliers_theta = pd.read_csv("Outputs/psd_outliers_Theta.csv", index_col=0)

for band_name, (fmin, fmax) in bands.items():
    df_outliers = df_outliers_alpha if band_name == "Alpha" else df_outliers_theta
    
    psd_cleaned_avg = {subject: {condition: None for condition in conditions} for subject in subjects}
    results = []
    
    for subject in subjects:
        epochs_path = os.path.join(filepath, f"{subject}-epo.fif") #LOADS EPOCHS OF THE SUBJECT
        if not os.path.exists(epochs_path):
            print(f"WARNING: No epochs found for {subject}.")
            continue
        
        epochs = mne.read_epochs(epochs_path, preload=True)
        
        for condition in conditions:
            if condition in epochs.event_id:
                psd = epochs[condition].compute_psd()
                psd_data = psd.get_data()
                freqs = psd.freqs
                
                band_mask = (freqs >= fmin) & (freqs <= fmax)
                psd_band = psd_data[:, :, band_mask].mean(axis=2) #Band Mean
                
                outliers_str = df_outliers.loc[subject, condition] if condition in df_outliers.columns else ""
                outliers = list(map(int, outliers_str.split(", "))) if isinstance(outliers_str, str) and outliers_str else []
                
                if outliers:
                    print(f"Deleting {len(outliers)} outliers in {subject} - {condition} - {band_name}: {outliers}")
                    psd_band = np.delete(psd_band, outliers, axis=0)
                
                if psd_band.size == 0:
                    print(f"WARNING: Todas las épocas de {subject} en {condition} fueron eliminadas por outliers.")
                    continue
                
                psd_mean_dict = {ch: np.nan for ch in all_channels}
                valid_channels = [ch for ch in all_channels if ch not in excluded_channels[subject]]
                
                if len(psd_band.mean(axis=0)) != len(valid_channels):
                    print(f"Error en {subject} - {condition}: Desajuste entre valid_channels y psd_band.mean().")
                    # warning raised if number of channels in epochs is not matched with 128-badschannels
                    continue
                
                for ch, value in zip(valid_channels, psd_band.mean(axis=0)):
                    psd_mean_dict[ch] = value
                
                psd_cleaned_avg[subject][condition] = psd_mean_dict
                
                for channel_name in all_channels:
                    results.append([subject, condition, channel_name, psd_mean_dict[channel_name]])
    
    df_psd_cleaned = pd.DataFrame(results, columns=["Subject", "Condition", "Channel", "PSD_Mean"])
    csv_filename = f"Outputs/psd_cleaned_avg_{band_name}.csv"
    df_psd_cleaned.to_csv(csv_filename, index=False)
    print(f"Proceso completado. Resultados guardados en {csv_filename}")