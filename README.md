# 2p_valence

Stimuli presentations were in this order:
Day1:
30 odor trials (15 coyote, 15 female urine), pseudorandomly presented
30 sucrose deliveries (40 for 2018 collective). Not all deliveries were necessarily consumed
10 shock trials
10 sound blasts (2020 collective only)

Day2:
same as Day1

Many notebooks call other notebooks, and all rely on variables defined in 'preprocessing' notebook. Python 2.7

the most up-to-date notebooks are in c10m6. The other animals can be used for data, but any notebooks in other animals should be updated/overwritten
   
*C.txt* = denoised trace (the red trace)  
*C_raw.txt* = raw trace (the blue trace)  
*C_df.txt* = not used  
*S.txt* = spiking events deconvolved from traces  
*A.txt* = spatial footprints of ROIs. This file is missing from some datasets (too large to upload to github)...
*cnn.txt* = also for spatial footprints, but I believe current notebooks no longer use  
*behavior_codes* = behavior file (arduino output containing timestamps of when behavioral (licks) and sensory (odor, sucrose deliveries) events occured  
*tseries.xml* = imaging file with timestamps of images  
