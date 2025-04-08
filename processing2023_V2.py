#ESTA ES LA DEFINITIVA AL 20/3/2025
# COPIA DEL NOTEBOOK
# COMENTAR LO NUEVO DE DROPS.

# %% NOTEBOOK EN DRIVE
# from google.colab import drive
# drive.mount('/content/drive')
#!pip install mne --quiet

# %% ######LIBRARIES#######
import mne
import mne
print(mne.__version__)
import pandas as pd
import matplotlib.pyplot as plt
import sys
from autoreject import get_rejection_threshold
#sys.path.append("/content/drive/MyDrive/Doctorado/IMAGINACION/Script/")
from my_functions import (
    preprocessing_mne,
    make_ICA,
    prom_areas,
    my_grand_average,
    df_prom_areas,
    find_outliers,
)
import numpy as np
import pandas as pd
import random
import os


# %% PARTE A CAMBIAR
s = 26  # Sujeto en posicion de array Subjects.
condition = ""
bands = {"Alpha": (8, 12)}
freqs_psd = bands["Alpha"]
lims_graph = (None, None)

# %% INFORMACION
subjects = [
    "01_test_2023",
    "02_test_2023", 
    "04_test_2023",
    "05_test_2023",
    "06_test_2023",
    "07_test_2023",
    "10_test_2023",
    "11_test_2023",
    "12_test_2023",    #s= 8 - 12_test_2023 ANULADO
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
    "30_test_2023",
    "30_test_2023_bis",
    "31_test_2023",
    "34_test_2023",
    "30_test_2023_Full"
]

bads_preICA = [
    ["B2","C4","C30",'D4', "D5",'D10','D12'],  # s=0 - 01
    ["C10", "D8", "D9", "D24", "D25"],  # s=1 - 02
    ["B4", "B8", "B9", "C29", "D5", "D23"],  # s=2 - 04
    ["B4", "B8", "B9", "C29", "D5", "D23"],  # s=3 - 05
    ["B6", "D10", "D25", "D29"],  # s=4 - 06
    ["C8"],  # s=5 - 07
    ["A10", "A17", "A27", "B4", "B30", "B31", "C17", "C29", "C32", "D19"],  # s=6 - 10
    ['B13'], #s= 7 - 11_test_2023 RARISIMO EL PSD
    [],     #s= 8 - 12_test_2023 ANULADO
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
    ['A6','A15','B1','B13','C26','D3'],#s = 22 - 30_test_2023
    ['A15','A20','B6','B13','C30','D3'],#s = 23 - 30_test_2023_bis
    ['A11','B21','C16','C17','C29','C30','D5'],#s = 24 - 31_test_2023
    ['A24','B1','B2','B18','B19','B28','B32','C6','C28','D8','D11','D12','D26'],#s = 25 - 34_test_2023
    []
]

bads_ICA = [
    [0,7,8,22],  # s=0 - 01 
    [0,4,8,13,21,22,23],  # s=1 - 02
    [0,2,8,11,17,19,23],  # s=2 - 04
    [0,3,6,7,12,13,15,23],  # s=3 - 05
    [0,1,3,4,5,14,15,20,22],  # s=04 - 06
    [0,4,5,6,10,12,14,23],  # s=5 - 07
    [0,5,6,10,12,13,14,16,17,21,23],  # s=6 - 10
    [0,1,8,16,18,19,22], #s= 7 - 11_test_2023
    [0,1,4,11,16,20,22,23],#s= 8 - 12_test_2023 ANULADO
    [1,3,12,21,22,23],#s=9 - 13 
    [0,1,2,6,7,11,12,13,19,21,22,23],#s= 10 - 14_test_2023
    [0,5,21,22,23],#s = 11 - 15_test_2023
    [0,3,4,5,7,8,9,12,22,23],#s = 12 - 16_test_2023 
    [0,1,10,17,20,22,23],#s = 13 - 16_test_2023
    [0,1,2,7,14,18],#s = 14 - 21_test_2023
    [1,4,9,15,17,18,21],#s = 15 - 22_test_2023
    [3,7,14,15,19,20,21],#s = 16 - 23_test_2023
    [0,2,4,12,14,16,17,19],#s = 17 - 24_test_2023
    [0,4,6,10,11,12,18,19,21,22,23],#s = 18 - 25_test_2023
    [0,3,4,11,15,16,17,19,21,23],#s = 19 - 26_test_2023
    [1,3,4,6,9,11,17],#s = 20 - 28_test_2023
    [0,3,6,9,15,17,18,20,21,23],#s = 21 - 29_test_2023
    [2,4,6,11,12,17,20],#s = 22 - 30_test_2023
    [0,1,5,8,10,13,15,18],#s = 23 - 30_test_2023_bis
    [0,4,5,13,15,16,17,18,21,22,23],#s = 24 - 31_test_2023
    [0,2,3,4,5,7,14,21,23]#s = 25 - 34_test_2023
    ,[]
]

bads_postICA = [
    [],  # s=0 - 01
    ["B25"],  # s=1 - 02
    [],  # s=2 - 04
    [],  # s=3 - 05
    [],  # s=4 - 06
    [],  # s=5 - 07
    [],  # s=6 - 10
    [],
    [], #s= 8 - 12_test_2023 ANULADO
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
    [],#s = 22 - 30_test_2023
    [],#s = 23 - 30_test_2023_bis
    [],#s = 24 - 31_test_2023
    ["A11","B7","B22"]#s = 25 - 34_test_2023
    ,[]
]

bads_epochs = [
    [12,25,34,43,49,63,68,74,78,83,84,87,100,105],  # s=0-01
    [16, 17, 19, 62, 66, 68, 73, 96, 105],  # s=1-02
    [1, 15, 24, 55, 77, 104, 110, 112, 118],  # s=2-04
    [7,10,11,21,22,25,52,55,63,66,70,71,80,87,94,96,97,98,99,106,108,115],  # s=3-05
    [10, 22, 25],  # s=4-06
    [5, 8, 9, 25, 29, 36, 60, 68, 72, 76, 114],  # s=5-07
    [6, 9, 13, 16, 18, 24, 27, 29, 48, 56, 71, 79, 85, 86, 89, 109],  # s=6 - 10
    [1, 23, 39, 49, 59, 69, 87, 98, 111], #s= 7 - 11_test_2023
    [[7, 22, 24, 28, 29, 30, 31, 32, 38, 39, 53, 54, 55, 56, 57, 58, 60, 61, 62, 63, 67, 69, 70, 74, 77, 79, 80, 81, 82, 84, 91, 92, 93, 94, 97, 98, 99]], #s= 8 - 12_test_2023 ANULADO
    [12, 34, 37, 50, 58, 72, 73, 74, 76, 78, 79, 80, 83, 84, 88, 89, 93, 94, 97, 101, 105, 114, 115, 116, 117, 118, 119],#s=9 - 13 
    [54, 91, 103, 111],#s= 10 - 14_test_2023
    [24, 25, 46, 48, 49, 50, 51, 52, 92, 97],#s = 11 - 15_test_2023
    [41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 53, 54, 55, 58, 77, 80, 82, 93, 110],#s = 12 - 16_test_2023
    [7, 17, 19, 20, 24, 27, 29, 30, 33, 34, 51, 52, 53, 54, 56, 57, 58, 59, 60, 65, 66, 70, 82, 89, 90],
    [52, 66, 79, 82, 97],#s = 14 - 21_test_2023
    [0, 7, 20, 27, 37, 39, 41, 47, 56, 59, 67, 76, 81, 95, 99, 100, 104, 105, 106, 109],#s = 15 - 22_test_2023
    [3, 4, 9, 10, 12, 24, 25, 32, 33, 35, 37, 47, 55, 60, 87, 90, 98, 99, 100, 102, 110, 111, 112, 113, 115, 118],#s = 16 - 23_test_2023
    [43, 47, 48, 64, 65, 84, 86, 91, 95, 96, 106],#s = 17 - 24_test_2023
    [12, 28, 60, 66, 82, 97, 102, 104, 113, 117],#s = 18 - 25_test_2023
    [12, 24, 39, 42, 49, 52, 55, 58, 62, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 89, 95, 96, 97],#s = 19 - 26_test_2023
    [6, 7, 14, 15, 18, 19, 20, 21, 22, 23, 24, 32, 33, 42, 57, 69, 70, 72, 83, 84, 85, 87, 88, 91, 94, 95, 98, 99, 110, 113, 115, 117],#s = 20 - 28_test_2023
    [14, 19, 20, 24, 25, 28, 30, 34, 35, 36, 40, 41, 48, 49, 50, 51, 53, 55, 56, 58, 59, 63, 64, 67, 68, 69, 70, 71, 72, 75, 78, 79, 80, 82, 97],#s = 21 - 29_test_2023
    [12],#s = 22 - 30_test_2023
    [1],#s = 23 - 30_test_2023_bis
    [25, 47, 50, 60, 73, 75, 77, 78, 80, 82, 83, 84, 85, 88, 95, 103, 104, 111, 112, 119],#s = 24 - 31_test_2023
    [22, 32, 57]#s = 25 - 34_test_2023
    ,[]
]

print("#s =",s,"-",subjects[s])
print("bads_preICA:",bads_preICA[s])
print("bads_ICA:",bads_ICA[s])
print("bads_postICA:",bads_postICA[s])
print("bads_epochs:",bads_epochs[s])
# %% CONFIGURACION
misc = ["EXG1", "EXG2"]
eog = []
bands = {"Theta": (4, 8), "Alpha": (8, 12)}
filepath = "E:/Doctorado/protocol2023/"
excluded = ["EXG3", "EXG4", "EXG5", "EXG6", "EXG7", "EXG8"]
count_subs = 0
evokeds = []
bands = {"Theta (4-8 Hz)": (4, 8), "Alpha (8-12 Hz)": (8, 12)}
my_events_dict = {
    "Resting": 40,
    "REY Thinking": 60,
    "REY": 70,
    "AUT Thinking": 100,
    "AUT": 110,
}
marks_events = [
    40, 60, 70,100, 110]

# %% PROCESAMIENTO
raw = preprocessing_mne(
    filepath,
    subjects[s],
    excluded=excluded,
    bads=bads_preICA[s],
    lowpass_cut=1,
    highpass_cut=30,
    raw_plot=False,
    filtered_plot=False,
    psd_plot=False,
    edit_marks=True,
)

ica, raw_clean = make_ICA(
    raw,
    method="infomax",
    n_components=24,
    decim=3,
    random_state=23,
    reject_limit=250e-6,
    bad_ica_channels=bads_ICA[s],
    plot_ica_topo=True,
    plot_ica_time=True,
    plot_raw=True,
)
#Infomax: Funciona mejor con datos biológicos porque detecta tanto señales supergaussianas (artefactos) como subgaussianas (actividad neuronal real). 
# Separa mejor los artefactos fisiológicos como parpadeos, movimientos musculares y latidos cardíacos.

# mne_icalabel.iclabel.get_iclabel_features(raw,ica)
# The provided ICA instance was fitted with a 'fastica' algorithm. ICLabel was designed with extended infomax ICA decompositions. To use the extended infomax algorithm, use the 'mne.preprocessing.ICA' instance with the arguments 'ICA(method='infomax', fit_params=dict(extended=True))' (scikit-learn) or 'ICA(method='picard', fit_params=dict(ortho=False, extended=True))' (python-picard).
# mne_icalabel.iclabel.get_iclabel_features(raw,ica)
plt.show()

# gui = label_ica_components(raw, ica)
# print(ica.labels_)
# plt.show()

events = mne.find_events(raw_clean, stim_channel="Status")
raw_clean.info["bads"].extend(bads_postICA[s])
my_events = mne.pick_events(events, include=marks_events)
#reject_criteria = dict(eeg=250e-6)
epochs = mne.Epochs(
    raw_clean,
    my_events,
    tmin=0,
    tmax=5,
    baseline=None,
    event_id=my_events_dict,
    preload=False,
)
print(epochs)
epochs.plot(events=my_events, event_id=my_events_dict)
plt.show()
# Obtener las posiciones donde epochs.drop_log contiene ('USER',)
user_rejected_epochs = [i for i, log in enumerate(epochs.drop_log) if log == ('USER',)]
# Imprimir las posiciones de las épocas rechazadas manualmente
print(f"Épocas rechazadas manualmente ('USER'): {user_rejected_epochs}")
epochs.drop(bads_epochs[s])
print(epochs)
reject_criteria = get_rejection_threshold(epochs, decim=2)
print(reject_criteria)
epochs.drop_bad(reject=reject_criteria)
epochs.plot_drop_log()
print(epochs)
# %% GUARDAR
epochs.save("E:/Doctorado/protocol2023/epochs/"+subjects[s]+"-epo.fif",overwrite=True)

#Si simplemente promediás todas las épocas antes de calcular la PSD, las condiciones con más ensayos tendrán más influencia en el promedio general.
#Esta estrategia asegura que cada condición tenga el mismo peso, sin importar cuántas épocas tenga.

# %% GRAFICO

spectrum = epochs["Resting"].compute_psd()
spectrum.plot_topomap(bands=bands, show_names=False, cmap="coolwarm", vlim=lims_graph)

spectrum = epochs["REY Thinking"].compute_psd()
spectrum.plot_topomap(bands=bands, show_names=False, cmap="coolwarm", vlim=lims_graph)

spectrum = epochs["REY"].compute_psd()
spectrum.plot_topomap(bands=bands, show_names=False, cmap="coolwarm", vlim=lims_graph)

spectrum = epochs["AUT Thinking"].compute_psd()
spectrum.plot_topomap(bands=bands, show_names=False, cmap="coolwarm", vlim=lims_graph)

spectrum = epochs["AUT"].compute_psd()
spectrum.plot_topomap(bands=bands, show_names=False, cmap="coolwarm", vlim=lims_graph)

plt.show()


# Convertir listas en strings asegurando que todos los elementos sean texto
data = {
    "Subject": subjects,
    "Bads_PreICA": [", ".join(map(str, b)) if b else "" for b in bads_preICA],
    "Bads_ICA": [", ".join(map(str, b)) if b else "" for b in bads_ICA],
    "Bads_PostICA": [", ".join(map(str, b)) if b else "" for b in bads_postICA],
    "Bads_Epochs": [", ".join(map(str, b)) if b else "" for b in bads_epochs]
}

# Convertir a DataFrame de Pandas
df = pd.DataFrame(data)

# Guardar como CSV
csv_filename = "Outputs/bads_summary.csv"
df.to_csv(csv_filename, index=False)

print(f"Archivo guardado: {csv_filename}")
