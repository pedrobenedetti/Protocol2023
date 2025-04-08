### FILE: zones.py
### UPDATED: 08/04/2025
### AUTHOR: pbenedetti@itba.edu.ar
### TASK: Express each PSD value as a percetage of activation of the 
###         zone within condition and subject. 
### OUTPUT: Outputs/SubsAndZones{Band}_Porcentajes.csv
### protocol2023_V2 --> GrAv2023_epochsOut.py --> GrAv2023_means.py --> zones.py --> porcPSDzona
import pandas as pd

# Cargar los datos desde la ruta correcta
df = pd.read_csv('Outputs/SubsAndZonesAlpha.csv')

# Convertir explícitamente la columna PSD_Mean a numérica
df['PSD_Mean'] = pd.to_numeric(df['PSD_Mean'], errors='coerce')

# Calcular el total del PSD para cada sujeto y condición
df['PSD_total'] = df.groupby(['Subject', 'Condition'])['PSD_Mean'].transform('sum')

# Calcular el porcentaje del PSD respecto del total
df['PSD_pct'] = (df['PSD_Mean'] / df['PSD_total']) * 100

# Guardar el resultado en un nuevo archivo
df.to_csv('Outputs/SubsAndZonesAlpha_Porcentajes.csv', index=False)

print(df.head())

