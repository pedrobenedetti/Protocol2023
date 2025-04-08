# Protocol2023
AUTHOR: pbenedetti@itba.edu.ar
## processing2023_V2

## GrAv2023_epochsOut
TASK: Takes epochs as an input and having bads_preICA and bads_postICA for a given band, for each subject and each condition, identifies epochs wich PSD is an outlier. It does so by computing the means of the epoch for all channels and freq of the band. Bads channels are excluded.
OUTPUT: psd_outliers_{band_name}.csv
PIPELINE: protocol2023_V2 --> GrAv2023_epochsOut.py

## GrAv2023_means
TASK: Takes epochs as an input and has bads_preICA, bads_postICA, and psd_outliers_{band_name}.csv. For each subject, condition, and channel computes a mean value for a given band.
OUTPUT: psd_cleaned_avg_{band_name}.csv
PIPELINE: Protocol2023_V2 --> GrAv2023_epochsOut.py --> GrAv2023_means.py

## zones
TASK: Computes mean values for each zone in each condition of each subject. Classifies subjects for Treatment.
OUTPUT: SubsAndZones"+banda+".csv
PIPELINE: protocol2023_V2 --> GrAv2023_epochsOut.py --> GrAv2023_means.py --> zones.py

## porcPSDzona
TASK: Express each PSD value as a percentage of activation of the zone within the condition and subject. 
OUTPUT: Outputs/SubsAndZones{Band}_Porcentajes.csv
PIPELINE: protocol2023_V2 --> GrAv2023_epochsOut.py --> GrAv2023_means.py --> zones.py --> porcPSDzona
