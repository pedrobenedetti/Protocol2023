### FILE: zones.py
### UPDATED: 08/04/2025
### AUTHOR: pbenedetti@itba.edu.ar
### TASK: Computes mean values for each zone in each condition of each 
###         subject. Classifies sujects for Treatment.
### OUTPUT: SubsAndZones"+banda+".csv
### protocol2023_V2 --> GrAv2023_epochsOut.py --> GrAv2023_means.py --> zones.py

import pandas as pd
banda = "Theta"
#banda = "Alpha"
# Archivo de entrada con columnas: [Subject, Condition, Channel, PSD_Mean]
input_csv = "Outputs/psd_cleaned_avg_"+banda+".csv"  # Modificar si corresponde
output_csv = "Outputs/SubsAndZones"+banda+".csv"

# Listas de sujetos por tratamiento
habituados = [
    "01_test_2023", "02_test_2023", "07_test_2023", "12_test_2023",
    "13_test_2023", "14_test_2023", "15_test_2023", "16_test_2023",
    "20_test_2023", "23_test_2023", "28_test_2023", "34_test_2023"
]

novedados = [
    "04_test_2023", "05_test_2023", "06_test_2023", "10_test_2023",
    "11_test_2023", "21_test_2023", "22_test_2023", "24_test_2023",
    "25_test_2023", "26_test_2023", "29_test_2023", "30_test_2023_full",
    "31_test_2023", "34_test_2023"
]

# Definir las zonas
zones_dict = {
    "Frontales": [
        "C8", "C9", "C10", "C12", "C13", "C14", "C15", "C16", "C17", "C18",
        "C19", "C20", "C21", "C25", "C26", "C27", "C28", "C29", "C30", "C31", "C32"
    ],
    "Temporales_Izq": [
        "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D21", "D22", "D23", "D24",
        "D25", "D26", "D29", "D30", "D31", "D32"
    ],
    "Temporales_Der": [
        "C4", "C5", "C6", "C7", "B10", "B11", "B12", "B13", "B14", "B15", "B16",
        "B24", "B25", "B26", "B27", "B28", "B29", "B30"
    ],
    "Occipital": [
        "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15", "A16", "A17", "A18",
        "A20", "A21", "A22", "A23", "A24", "A25", "A26", "A27", "A28", "A29", "A30",
        "A31", "B5", "B6", "B7", "B8", "B9"
    ],
    "Parietal": [
        "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A19", "A32", "B1", "B2", "B3",
        "B4", "B17", "B18", "B19", "B20", "B21", "B22", "B23", "B31", "B32", "C1",
        "C2", "C3", "C11", "C22", "C23", "C24", "D1", "D2", "D3", "D12", "D13",
        "D14", "D15", "D16", "D17", "D18", "D19", "D20", "D27", "D28"
    ]
}

# Función que retorna el tratamiento de un sujeto
# Si un sujeto aparece en ambos, prevalece la lista "habituados".
# Si no aparece en ninguno, retorna "Desconocido".

def get_tratamiento(subject):
    if subject in habituados:
        return "Habituados"
    elif subject in novedados:
        return "Novedados"
    else:
        return "Desconocido"

# Cargar el DataFrame
df = pd.read_csv(input_csv)

# Asignar el tratamiento a cada fila
df["Tratamiento"] = df["Subject"].apply(get_tratamiento)

# Definir una función para ubicar la zona del canal

def get_zone(channel):
    for zone_name, zone_list in zones_dict.items():
        if channel in zone_list:
            return zone_name
    return None  # Si no está en ninguna zona

# Asignar la zona
df["Zone"] = df["Channel"].apply(get_zone)

# Agrupar por [Subject, Tratamiento, Condition, Zone], promediando PSD_Mean
df_zones = df.groupby(["Subject", "Tratamiento", "Condition", "Zone"], as_index=False)["PSD_Mean"].mean()

# Guardar resultado
df_zones.to_csv(output_csv, index=False)
print(f"Archivo guardado: {output_csv}")

# Estructura final:
print(df_zones.head(10))
