# 2p_valence
 
Many notebooks call other notebooks, and all rely on variables defined in 'preprocessing' notebook

*JSB is currenlty working on a generalized preprocessing notebook that can be used for any of the datasets (eg, whether odor and sucrose/shock presentations were collected in separate or same session)*
   
*C.txt* = denoised trace (the red trace)  
*C_raw.txt* = raw trace (the blue trace)  
*C_df.txt* = not used  
*S.txt* = spiking events deconvolved from traces  
*A.txt* = spatial footprints of ROIs. This file is missing from some datasets (too large to upload to github)...
*cnn.txt* = also for spatial footprints, but I believe current notebooks no longer use  
*behavior_codes* = behavior file (arduino output containing timestamps of when behavioral (licks) and sensory (odor, sucrose deliveries) events occured  
*tseries.xml* = imaging file with timestamps of images  
   
991 is a complete dataset. 91 is lacking all notebooks, and CellReg data isn't in proper format for NBs to read
