% This function returns the specific tensions in the muscles
% The data come from Uchida et al. (2016).
% 
% Author: Antoine Falisse
% Date: 12/19/2018
%         
function sigma = getSpecificTensions_2D(muscleNames)   
    sigma_data.hamstrings_r = 0.62222;    
    sigma_data.bifemsh_r = 1.00500;
    sigma_data.glut_max_r = 0.74455;
    sigma_data.iliopsoas_r = 1.5041;
    sigma_data.rect_fem_r = 0.74936;
    sigma_data.vasti_r = 0.55263;
    sigma_data.gastroc_r = 0.69865;
    sigma_data.soleus_r = 0.62703;
    sigma_data.tib_ant_r = 0.75417;    
    
    sigma = zeros(length(muscleNames),1);
    for i = 1:length(muscleNames)
        sigma(i,1) = sigma_data.(muscleNames{i});
    end
    
end    
