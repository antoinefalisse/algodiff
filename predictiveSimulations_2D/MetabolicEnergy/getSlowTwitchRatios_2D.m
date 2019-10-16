% This function returns the percentage of slow twitch fibers in the muscles
% The data come from Uchida et al. (2016).
% We used 0.5 when no data were available.
%
% Author: Antoine Falisse
% Date: 12/19/2018
%     
function pctst = getSlowTwitchRatios_2D(muscleNames)    
    pctst_data.hamstrings_r = 0.5425;
    pctst_data.bifemsh_r = 0.529;    
    pctst_data.glut_max_r = 0.55;
    pctst_data.iliopsoas_r = 0.50;    
    pctst_data.rect_fem_r = 0.3865;
    pctst_data.vasti_r = 0.543;
    pctst_data.gastroc_r = 0.566;
    pctst_data.soleus_r = 0.803;    
    pctst_data.tib_ant_r = 0.70;  
    pctst = zeros(length(muscleNames),1);
    for i = 1:length(muscleNames)
        pctst(i,1) = pctst_data.(muscleNames{i});
    end
    
end   
