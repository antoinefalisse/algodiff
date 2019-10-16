% This script plots results users might be intersted in (e.g.,joint angles
% and torques, ground reaction forces and torques, muscle activations, contact
% model parameters, CPU times)
%
% Author: Antoine Falisse
% Date: 9/9/2019
%
clear all
close all
clc

%% User settings
% Select trials, for example
ww_3D  = [1,2,3];

%% Other settings
% Load pre-defined settings
pathmain = pwd;
[pathRepo_3D,~,~] = fileparts(pathmain);
pathSettings = [pathRepo_3D,'\Settings'];
addpath(genpath(pathSettings));
settings_3D
subject = 'subject1';
body_mass = 62;
body_weight = body_mass*9.81;
setup.ocp = 'TrackSim_3D';
time_opt = [3.45,4.25];
trial = 'gait_1';

%% Load results
% Pre-allocation structures
Qs_opt_3D           = struct('m',[]);
Ts_opt_3D           = struct('m',[]);
GRFs_opt_3D         = struct('m',[]);
GRMs_opt_3D         = struct('m',[]);
Qs_toTrack_3D       = struct('m',[]);
Ts_toTrack_3D       = struct('m',[]);
GRFs_toTrack_3D     = struct('m',[]);
GRMs_toTrack_3D     = struct('m',[]);
Stats_3D            = struct('m',[]);
As_opt_3D           = struct('m',[]);
ParamsCM_opt_3D     = struct('m',[]);
ParamsCM_gen_3D     = struct('m',[]);
legend_case         = cell(1,length(ww_3D));
% Loop over cases
for k = 1:length(ww_3D) 
    data_3D;
    legend_case{k} = ['Case ',num2str(ww_3D(k))]; 
end

%% Helper names
RefData_str_tit = {'Pelvis tilt','Pelvis list','Pelvis rotation','Pelvis tx',...
    'Pelvis ty','Pelvis tz','Hip flex L','Hip add L','Hip rot L',...
    'Hip flex R','Hip add R','Hip rot R','Knee L','Knee R','Ankle L',...
    'Ankle R','Subtalar L','Subtalar R','Lumbar ext','Lumbar ben',...
    'Lumbar rot','Arm flex L','Arm add L','Arm rot L','Arm Flex R',...
    'Arm add R','Arm rot R','Elbow flex L','Elbow flex R'};

%% Common settings for plots
label_fontsize  = 14;
line_linewidth  = 3;
sup_fontsize  = 24;
col = hsv(length(ww_3D));

%% Plot joint angles
idx_Qs = 1:29;
NumTicks_Qs = 2;
figure()
for i = 1:length(idx_Qs)
    subplot(5,6,i)
    % Experimental data
    Qs_toTrack_deg = Qs_toTrack_3D(ww_3D(k)).m;
    Qs_toTrack_deg(:,[2:4,8:end]) = Qs_toTrack_deg(:,[2:4,8:end])*180/pi;    
    plot(Qs_toTrack_deg(:,1),Qs_toTrack_deg(:,idx_Qs(i)+1),...
        'k','linewidth',line_linewidth);  
    hold on
    % Simulation results
    p = gobjects(1,length(ww_3D)); 
    for k = 1:length(ww_3D)
        p(k) = plot(Qs_toTrack_3D(ww_3D(k)).m(:,1),...
            Qs_opt_3D(ww_3D(k)).m(:,idx_Qs(i)),...
            'color',col(k,:),'linestyle',':','linewidth',line_linewidth);
        hold on;          
    end    
    % Plot settings 
    set(gca,'Fontsize',label_fontsize);    
    title(RefData_str_tit{idx_Qs(i)},'Fontsize',label_fontsize);  
    % Y-axis
    L = get(gca,'YLim');
    set(gca,'YTick',linspace(L(1),L(2),NumTicks_Qs));       
    if i == 1 || i == 7 || i == 13 || i == 19 || i == 25
        ylabel('(°)','Fontsize',label_fontsize);
    elseif i == 4
        ylabel('(m)','Fontsize',label_fontsize);
    end      
    % X-axis
    xlim([Qs_toTrack_deg(1,1),Qs_toTrack_deg(end,1)])
    if i > 23
        set(gca,'XTick',linspace(L(1),L(2),NumTicks_Qs))
        xlabel('Time (s)','Fontsize',label_fontsize);
    else
        set(gca,'XTick',[]);
    end
    box off;
end 
l = legend(p,legend_case);
set(l,'Fontsize',label_fontsize)
sp = suptitle('Joint angles');
set(sp,'Fontsize',sup_fontsize);   

%% Plot ground reaction forces
ylim_GRF = [-50,50;0 150;-25 25;-50,50;0 150;-25 25];
NumTicks_GRF = 2;
temp = GRFs_toTrack_3D(ww_3D(1)).m(:,[3,6]);
temp(temp<0) = 0;
GRFs_toTrack_3D(ww_3D(1)).m(:,[3,6])= temp;
GRF_str = {'Fore-aft R','Vertical R','Lateral R',...
    'Fore-aft L','Vertical L','Lateral L'};
figure()
for i = 1:length(GRF_str)
    subplot(2,3,i)
    % Experimental data
    plot(GRFs_toTrack_3D(ww_3D(k)).m(:,1),...
        GRFs_toTrack_3D(ww_3D(k)).m(:,i+1)./(body_weight/100),...
        'k','linewidth',line_linewidth);
    hold on;
    % Simulation results
    p = gobjects(1,length(ww_3D)); 
    for k = 1:length(ww_3D)
        p(k) = plot(GRFs_toTrack_3D(ww_3D(k)).m(:,1),...
            GRFs_opt_3D(ww_3D(k)).m(:,i)./(body_weight/100),...
            'color',col(k,:),'linestyle',':','linewidth',line_linewidth);
        hold on;    
    end
    % Plot settings 
    set(gca,'Fontsize',label_fontsize);    
    title(GRF_str{i},'Fontsize',label_fontsize);  
    % Y-axis
    ylim([ylim_GRF(i,1) ylim_GRF(i,2)]);
    L = get(gca,'YLim');
    set(gca,'YTick',linspace(L(1),L(2),NumTicks_GRF));       
    if i == 1 || i == 4 
        ylabel('(%BW)','Fontsize',label_fontsize);
    end      
    % X-axis
    xlim([Qs_toTrack_deg(1,1),Qs_toTrack_deg(end,1)])
    L = get(gca,'XLim');
    if i == 4 || i == 5 || i == 6
        set(gca,'XTick',linspace(L(1),L(2),NumTicks_GRF))
        xlabel('Time (s)','Fontsize',label_fontsize);
    else
        set(gca,'XTick',[]);
    end
    box off;   
end
l = legend(p,legend_case);
set(l,'Fontsize',16)
sp = suptitle('Ground reaction forces');
set(sp,'Fontsize',sup_fontsize);

%% Plot ground reaction torques
% Expressed with respect to the ground frame origin
ylim_GRM = [-20,200;-50 50;-250 250;-20,200;-50 50;-250 250];
figure()
for i = 1:length(GRF_str)
    subplot(2,3,i)
    % Experimental data
    plot(GRMs_toTrack_3D(ww_3D(k)).m(:,1),GRMs_toTrack_3D(ww_3D(k)).m(:,i+1),...
        'k','linewidth',line_linewidth);
    hold on;
    % Simulation results
    p = gobjects(1,length(ww_3D)); 
    for k = 1:length(ww_3D)
        p(k) = plot(GRMs_toTrack_3D(ww_3D(k)).m(:,1),...
            GRMs_opt_3D(ww_3D(k)).m(:,i),'color',col(k,:),'linestyle',':',...
            'linewidth',line_linewidth);
        hold on;    
    end
    % Plot settings 
    set(gca,'Fontsize',label_fontsize);    
    title(GRF_str{i},'Fontsize',label_fontsize);  
    % Y-axis
    ylim([ylim_GRM(i,1) ylim_GRM(i,2)]);
    L = get(gca,'YLim');
    set(gca,'YTick',linspace(L(1),L(2),NumTicks_GRF));       
    if i == 1 || i == 4 
        ylabel('(Nm)','Fontsize',label_fontsize);
    end      
    % X-axis
    xlim([Qs_toTrack_deg(1,1),Qs_toTrack_deg(end,1)])
    L = get(gca,'XLim');
    if i == 4 || i == 5 || i == 6
        set(gca,'XTick',linspace(L(1),L(2),NumTicks_GRF))
        xlabel('Time (s)','Fontsize',label_fontsize);
    else
        set(gca,'XTick',[]);
    end
    box off;    
end
l = legend(p,legend_case);
set(l,'Fontsize',16)
sp = suptitle('Ground reaction torques');
set(sp,'Fontsize',sup_fontsize);

%% Plot joint torques
idx_Qdots = 1:29;
figure()
for i = 1:length(idx_Qdots)
    subplot(5,6,i)
    % Experimental data
    plot(Ts_toTrack_3D(ww_3D(k)).m(:,1),...
        Ts_toTrack_3D(ww_3D(k)).m(:,idx_Qdots(i)+1),...
        'k','linewidth',line_linewidth);
    hold on
    % Simulation results
    p = gobjects(1,length(ww_3D)); 
    for k = 1:length(ww_3D)
        p(k) = plot(Ts_toTrack_3D(ww_3D(k)).m(:,1),...
            Ts_opt_3D(ww_3D(k)).m(:,idx_Qdots(i)),...
            'color',col(k,:),'linestyle',':','linewidth',line_linewidth);
        hold on;            
    end
    % Plot settings 
    set(gca,'Fontsize',label_fontsize);    
    title(RefData_str_tit{idx_Qs(i)},'Fontsize',label_fontsize);  
    % Y-axis
    L = get(gca,'YLim');
    set(gca,'YTick',linspace(L(1),L(2),NumTicks_Qs));       
    if i == 1 || i == 7 || i == 13 || i == 19 || i == 25
        ylabel('(Nm)','Fontsize',label_fontsize);
    end      
    % X-axis
    xlim([Qs_toTrack_deg(1,1),Qs_toTrack_deg(end,1)])
    if i > 23
        set(gca,'XTick',linspace(L(1),L(2),NumTicks_Qs))
        xlabel('Time (s)','Fontsize',label_fontsize);
    else
        set(gca,'XTick',[]);
    end
    box off;
end    
l = legend(p,legend_case);
set(l,'Fontsize',16)
sp = suptitle('Joint torques');
set(sp,'Fontsize',sup_fontsize);

%% Plot muscle activations: right
muscleNames = {'Glut med 1','Glut med 2','Glut med 3',...
        'Glut min 1','Glut min 2','Glut min 3','Semimem',...
        'Semiten','bifemlh','Bic fem sh','Sar','Add long',...
        'Add brev','Add mag 1','Add mag 2','Add mag 3','TFL',...
        'Pect','Grac','Glut max 1','Glut max 2','Glut max 3',......
        'Iliacus','Psoas','Quad fem','Gem','Peri',...
        'Rect fem','Vas med','Vas int','Vas lat','Med gas',...
        'Lat gas','Soleus','Tib post','Flex dig','Flex hal',...
        'Tib ant','Per brev','Per long','Per tert','Ext dig',...
        'Ext hal','Ercspn','Intobl','Extobl'};    
EMGchannel_r = {'GluMed_r','GluMed_r','GluMed_r',...
    'no','no','no','HamM_r',...
    'HamM_r','HamL_r','HamL_r','no','no',...
    'no','no','no','no','no',...
    'no','no','no','no','no',...
    'no','no','no','no','no',...
    'RF_r','VM_r','no','VL_r','no',...
    'GL_r','Sol_r','no','no','no',...
    'TA_r','no','PerL_r','no','no',...
    'no','no','no','no'};
EMGcol_r = ones(1,length(muscleNames));
for i = 1:length(muscleNames)
    if strcmp(EMGchannel_r{i},'no')
        EMGcol_r(i) = 0;
    end
end
% Load data
pathmain = pwd;
[pathRepo_3D,~,~] = fileparts(pathmain);
pathData = [pathRepo_3D,'\OpenSimModel\',subject];
load([pathData,'\EMG\EMG_',trial,'.mat'],'EMG');
% Time window
id_i = find(EMG.data(:,1)==time_opt(1));
id_f = find(EMG.data(:,1)==time_opt(2));
EMGsel = EMG.data(id_i:id_f,:);    
figure()
NMuscle = size(As_opt_3D(ww_3D(k)).m,2);
for i = 1:NMuscle/2
    subplot(7,7,i)
    p = gobjects(1,length(ww_3D));   
    % Simulation results
    for k = 1:length(ww_3D)
        x = 1:(100-1)/(size(As_opt_3D(ww_3D(k)).m,1)-1):100;
        p(k) = plot(x,As_opt_3D(ww_3D(k)).m(:,i+NMuscle/2),'color',...
            col(k,:),'linestyle',':','linewidth',line_linewidth);
        hold on;
    end
    if EMGcol_r(i)
        % Normalize peak EMG to peak muscle activation
        a_peak = max(As_opt_3D(ww_3D(k)).m(:,i+NMuscle/2));
        emg_peak = max(EMGsel(:,strcmp(EMG.colheaders,EMGchannel_r{i})));
        norm_f = a_peak./emg_peak;       
        step = (EMGsel(end,1)-EMGsel(1,1))/(size(As_opt_3D(ww_3D(k)).m,1)-1);
        intervalInterp = EMGsel(1,1):step:EMGsel(end,1);        
        EMGselinterp = interp1(EMGsel(:,1),EMGsel(:,2:end),intervalInterp); 
        EMGselinterpInt = [intervalInterp',EMGselinterp];
        plot(x,EMGselinterpInt(:,strcmp(EMG.colheaders,...
            EMGchannel_r{i}))*norm_f,'k','linewidth',line_linewidth);        
    end
    % Plot settings
    set(gca,'Fontsize',label_fontsize)
    title(muscleNames{i},'Fontsize',label_fontsize);    
    % X-axis
    L = get(gca,'XLim');
    NumTicks = 3;
    if i > 39
        set(gca,'XTick',linspace(L(1),L(2),NumTicks))
        xlabel('Gait cycle (%)','Fontsize',label_fontsize);
    else
        set(gca,'XTick',[]);
    end
    % Y-axis
    ylim([0,1]);
    NumTicks = 2;
    LY = get(gca,'YLim');
    set(gca,'YTick',linspace(LY(1),LY(2),NumTicks))
    if i == 1 || i == 8 || i == 15 || i == 22 || i == 29 || i == 36 || i == 43
        ylabel('(-)','Fontsize',label_fontsize);
    end  
    box off;
end
l = legend(p,legend_case);
set(l,'Fontsize',label_fontsize)
sp = suptitle('Muscle activations: right');
set(sp,'Fontsize',sup_fontsize);

%% Plot muscle activations: left
EMGchannel_l = {'GluMed_l','GluMed_l','GluMed_l',...
    'no','no','no','HamM_l',...
    'HamM_l','HamL_l','HamL_l','no','AddL_l',...
    'no','no','no','no','TFL_l',...
    'no','no','no','no','no',...
    'no','no','no','no','no',...
    'RF_l','VM_l','no','VL_l','GM_l',...
    'GL_l','Sol_l','no','no','no',...
    'TA_l','PerB_l','PerL_l','no','no',...
    'no','no','no','no'};
EMGcol_l = ones(1,length(muscleNames));
for i = 1:length(muscleNames)
    if strcmp(EMGchannel_l{i},'no')
        EMGcol_l(i) = 0;
    end
end

figure()
NMuscle = size(As_opt_3D(ww_3D(k)).m,2);
for i = 1:NMuscle/2
    subplot(7,7,i)
    p = gobjects(1,length(ww_3D));   
    % Simulation results
    for k = 1:length(ww_3D)
        x = 1:(100-1)/(size(As_opt_3D(ww_3D(k)).m,1)-1):100;
        p(k) = plot(x,As_opt_3D(ww_3D(k)).m(:,i),'color',...
            col(k,:),'linestyle',':','linewidth',line_linewidth);
        hold on;
    end
    if EMGcol_l(i)
        % Normalize peak EMG to peak muscle activation
        a_peak = max(As_opt_3D(ww_3D(k)).m(:,i+NMuscle/2));
        emg_peak = max(EMGsel(:,strcmp(EMG.colheaders,EMGchannel_l{i})));
        norm_f = a_peak./emg_peak;       
        step = (EMGsel(end,1)-EMGsel(1,1))/(size(As_opt_3D(ww_3D(k)).m,1)-1);
        intervalInterp = EMGsel(1,1):step:EMGsel(end,1);        
        EMGselinterp = interp1(EMGsel(:,1),EMGsel(:,2:end),intervalInterp); 
        EMGselinterpInt = [intervalInterp',EMGselinterp];
        plot(x,EMGselinterpInt(:,strcmp(EMG.colheaders,...
            EMGchannel_l{i}))*norm_f,'k','linewidth',line_linewidth);        
    end
    % Plot settings
    set(gca,'Fontsize',label_fontsize)
    title(muscleNames{i},'Fontsize',label_fontsize);    
    % X-axis
    L = get(gca,'XLim');
    NumTicks = 3;
    if i > 39
        set(gca,'XTick',linspace(L(1),L(2),NumTicks))
        xlabel('Gait cycle (%)','Fontsize',label_fontsize);
    else
        set(gca,'XTick',[]);
    end
    % Y-axis
    ylim([0,1]);
    NumTicks = 2;
    LY = get(gca,'YLim');
    set(gca,'YTick',linspace(LY(1),LY(2),NumTicks))
    if i == 1 || i == 8 || i == 15 || i == 22 || i == 29 || i == 36 || i == 43
        ylabel('(-)','Fontsize',label_fontsize);
    end  
    box off;
end
l = legend(p,legend_case);
set(l,'Fontsize',label_fontsize)
sp = suptitle('Muscle activations: left');
set(sp,'Fontsize',sup_fontsize);

%% Plot parameters
figure()
p = gobjects(1,length(ww_3D)); 
for k = 1:length(ww_3D)
    % Location contact spheres
    subplot(1,2,1)
    p(k) = scatter(1:12,ParamsCM_opt_3D(ww_3D(k)).m(1:12),'filled',...
        'MarkerFaceColor',col(k,:),'linewidth',2); hold on
    scatter(1:12,ParamsCM_gen_3D(ww_3D(k)).m(1:12),'k','+','linewidth',2);
    % bounds
    dev_cm.loc = 25;
    dev_cm.rad = 50;
    loc_upper = ParamsCM_gen_3D(ww_3D(k)).m(1:12) + dev_cm.loc/1000;
    loc_lower = ParamsCM_gen_3D(ww_3D(k)).m(1:12) - dev_cm.loc/1000;    
    radii = ParamsCM_gen_3D(ww_3D(k)).m(13:18);
    radii_upper = radii + dev_cm.rad/100*radii;
    radii_lower = radii - dev_cm.rad/100*radii;
    scatter(1:12,loc_upper,'b','^','linewidth',2);
    scatter(1:12,loc_lower,'r','v','linewidth',2);
    % Plot settings
    set(gca,'Fontsize',label_fontsize)
    title('Location contact spheres','Fontsize',label_fontsize);  
    % Y-axis
    ylabel('Position in body','Fontsize',label_fontsize);
    % X-axis
    size_x = 12 + 1;
    Xt = 0:size_x;
    Xl = [0 size_x];
    set(gca,'XTick',Xt,'XLim',Xl);
    temp = cell(1,size_x+1);
    temp(2:end-1) = {'s1x','s1z','s2x','s2z','s3x','s3z','s4x','s4z','s5x',...
        's5z','s6x','s6z'};
    set(gca,'XTickLabel',temp,'Fontsize',label_fontsize');
    box off;
    % Radii contact spheres
    subplot(1,2,2)
    scatter(1:6,ParamsCM_opt_3D(ww_3D(k)).m(13:18),'filled',...
        'MarkerFaceColor',col(k,:),'linewidth',2); hold on
    scatter(1:6,ParamsCM_gen_3D(ww_3D(k)).m(13:18),'k','+','linewidth',2);        
    scatter(1:6,radii_upper,'b','^','linewidth',2);
    scatter(1:6,radii_lower,'r','v','linewidth',2);
    % Plot settings
    set(gca,'Fontsize',label_fontsize)
    title('Radius contact spheres','Fontsize',label_fontsize);
    % Y-axis
    ylabel('(m)','Fontsize',label_fontsize);
    % X-axis
    size_x = 6 + 1;
    Xt = 0:size_x;
    Xl = [0 size_x];
    set(gca,'XTick',Xt,'XLim',Xl);
    temp = cell(1,size_x+1);
    temp(2:end-1) = {'s1','s2','s3','s4','s5','s6'};
    set(gca,'XTickLabel',temp,'Fontsize',label_fontsize');
    box off;
end
l = legend(p,legend_case);
set(l,'Fontsize',label_fontsize)
sp = suptitle('Contact model parameters');
set(sp,'Fontsize',sup_fontsize);

%% Plot CPU times / Optimal cost
CPU_IPOPT = struct('m',[]);
CPU_NLP = struct('m',[]);
for k = 1:length(ww_3D)
    CPU_IPOPT(ww_3D(k)).m = Stats_3D(ww_3D(k)).m.t_proc_solver - ...
        Stats_3D(ww_3D(k)).m.t_proc_nlp_f - ...
        Stats_3D(ww_3D(k)).m.t_proc_nlp_g - ...
        Stats_3D(ww_3D(k)).m.t_proc_nlp_grad - ...
        Stats_3D(ww_3D(k)).m.t_proc_nlp_grad_f - ...
        Stats_3D(ww_3D(k)).m.t_proc_nlp_jac_g;
    CPU_NLP(ww_3D(k)).m = Stats_3D(ww_3D(k)).m.t_proc_solver - ...
        CPU_IPOPT(ww_3D(k)).m;
end
figure()
subplot(2,3,1)
p = gobjects(1,length(ww_3D));
for k = 1:length(ww_3D)
    p(k) = scatter(k,CPU_IPOPT(ww_3D(k)).m,...
        40,col(k,:),'filled');
    hold on;    
end
set(gca,'Fontsize',label_fontsize);
title('CPU time (IPOPT)','Fontsize',label_fontsize);
ylabel('(s)','Fontsize',label_fontsize);

subplot(2,3,2)
p = gobjects(1,length(ww_3D));
for k = 1:length(ww_3D)
    p(k) = scatter(k,CPU_NLP(ww_3D(k)).m,...
        40,col(k,:),'filled');
    hold on;    
end
set(gca,'Fontsize',label_fontsize);
title('CPU time (NLP)','Fontsize',label_fontsize);
ylabel('(s)','Fontsize',label_fontsize);

subplot(2,3,3)
p = gobjects(1,length(ww_3D));
CPU_tot = zeros(1,length(ww_3D));
for k = 1:length(ww_3D)
    CPU_tot(k) = (CPU_IPOPT(ww_3D(k)).m+CPU_NLP(ww_3D(k)).m);
    p(k) = scatter(k,CPU_tot(k),...
        40,col(k,:),'filled');
    hold on;    
end
set(gca,'Fontsize',label_fontsize);
title('CPU time (TOTAL)','Fontsize',label_fontsize);
ylabel('(s)','Fontsize',label_fontsize);

subplot(2,3,4)
p = gobjects(1,length(ww_3D));
for k = 1:length(ww_3D)
    p(k) = scatter(k,(Stats_3D(ww_3D(k)).m.iterations.obj(end)),...
        40,col(k,:),'filled');
    hold on;    
end
set(gca,'Fontsize',label_fontsize);
title('Optimal cost','Fontsize',label_fontsize);
ylabel('(-)','Fontsize',label_fontsize);
l = legend(p,legend_case);
set(l,'Fontsize',label_fontsize)

subplot(2,3,5)
p = gobjects(1,length(ww_3D));
for k = 1:length(ww_3D)
    p(k) = scatter(k,(Stats_3D(ww_3D(k)).m.iter_count),...
        40,col(k,:),'filled');
    hold on;    
end
set(gca,'Fontsize',label_fontsize);
title('# iterations','Fontsize',label_fontsize);
ylabel('(-)','Fontsize',label_fontsize);
l = legend(p,legend_case);
set(l,'Fontsize',label_fontsize)

subplot(2,3,6)
p = gobjects(1,length(ww_3D));
for k = 1:length(ww_3D)
    p(k) = scatter(k,(Stats_3D(ww_3D(k)).m.iterations.inf_du(end)),...
        40,col(k,:),'filled');
    hold on;    
end
set(gca,'Fontsize',label_fontsize);
title('Error NLP / Dual inf','Fontsize',label_fontsize);
ylabel('(-)','Fontsize',label_fontsize);
l = legend(p,legend_case);
set(l,'Fontsize',label_fontsize)

%% CPU time breakdown
yy = zeros(2,length(ww_3D));
for k = 1:length(ww_3D)
    yy(k,1) = Stats_3D(ww_3D(k)).m.t_proc_solver - ...
        Stats_3D(ww_3D(k)).m.t_proc_nlp_f - ...
        Stats_3D(ww_3D(k)).m.t_proc_nlp_g - ...
        Stats_3D(ww_3D(k)).m.t_proc_nlp_grad - ...
        Stats_3D(ww_3D(k)).m.t_proc_nlp_grad_f - ...
    Stats_3D(ww_3D(k)).m.t_proc_nlp_jac_g;
    yy(k,2) = Stats_3D(ww_3D(k)).m.t_proc_nlp_f;
    yy(k,3) = Stats_3D(ww_3D(k)).m.t_proc_nlp_g;
    yy(k,4) = Stats_3D(ww_3D(k)).m.t_proc_nlp_grad_f;
    yy(k,5) = Stats_3D(ww_3D(k)).m.t_proc_nlp_jac_g;
end
% Colors
color_all(1,:) = [244,194,13]/255;     % Yellow
color_all(2,:) = [60,186,84]/255;      % Green
color_all(3,:) = [0,0,0];              % Black
color_all(4,:) = [219,50,54]/255;      % Red
color_all(5,:) = [72,133,237]/255;     % Blue
figure()
h = bar(yy,'stacked');    
for k = 1:size(yy,2)
    set(h(k),'FaceColor',color_all(k,:));
end
set(gca, 'YScale', 'log')
l = legend('Solver (IPOPT)','Objective function evaluations',...
    'Constraint evaluations','Objective function gradient evaluations',...
    'Constraint Jacobian evaluations');
set(gca,'Fontsize',label_fontsize);   
set(l,'Fontsize',label_fontsize); 
set(l,'location','Northwest');
set(gca,'XTickLabel',legend_case,'Fontsize',label_fontsize');
ylabel('Computational time (s)','Fontsize',label_fontsize');
title('CPU time breakdown','Fontsize',label_fontsize');
box off;

%% Average CPU time over trials selected
CPU_time_all.all = zeros(1,length(ww_3D));
for k = 1:length(ww_3D)
    CPU_time_all.all(k) = Stats_3D(ww_3D(k)).m.t_proc_solver;
end
CPU_time_all.mean = mean(CPU_time_all.all);
CPU_time_all.std = std(CPU_time_all.all);
