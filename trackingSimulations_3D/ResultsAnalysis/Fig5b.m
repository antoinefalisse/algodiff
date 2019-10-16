% This script reproduces part of Fig 5 (see paper)
%
% Author: Antoine Falisse
% Date: 1/7/2019

clear all
close all
clc

%% Settings
% Case
ww_3D  = 2;
% Load pre-defined settings
pathmain = pwd;
[pathRepo_3D,~,~] = fileparts(pathmain);
pathSettings = [pathRepo_3D,'\Settings'];
addpath(genpath(pathSettings));
pathVariousFunctions = [pathRepo_3D,'\VariousFunctions'];
addpath(genpath(pathVariousFunctions));
% Fixed settings
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
% Loop over cases
settings_3D
for k = 1:length(ww_3D) 
    data_3D; 
end

%% Extract HS and TO
HS_r = struct('m',[]);
TO_l = struct('m',[]);
for k = 1:length(ww_3D) 
    threshold = 20;
    HS_r(ww_3D(k)).m = find(diff(GRFs_opt_3D(ww_3D(k)).m(:,2) > threshold)) +1;
    TO_l(ww_3D(k)).m = find(diff(GRFs_opt_3D(ww_3D(k)).m(:,5) < threshold)) +1;
end

%% Settings for plots
label_fontsize  = 14;
line_linewidth  = 3;
% Colors
color_all(1,:) = [0,0,0];
color_all(2,:) = [219,50,54]/255; % Red

%% Plot joint angles
pos_Qs = [1:3,7:9,13:15,19:21];
ylim_Qs = [-50,50;-20,20;-20,20;-50,50;-20,20;-20,20;-80,0;-30,30;-20,20;...
    -80,0;-30,30;-20,20];
idx_Qs = [10,11,12,7,8,9,14,16,18,13,15,17]; 
leg_Qs = {'r','r','r','l','l','l','r','r','r','l','l','l'};
NumTicks_Qs = 2;
RefData_str_tit = {'pelvis-tilt','pelvis-list','pelvis-rotation',...
    'pelvis-tx','pelvis-ty','pelvis-tz','Hip flex L',...
    'Hip add L','Hip rot L','Hip flex R',...
    'Hip add R','Hip rot R','Knee L','Knee R',...
    'Ankle L','Ankle R','Subtalar L','Subtalar R',...
    'lumbar-extension','lumbar-bending','lumbar-rotation',...
    'arm-flex-l','arm-add-l','arm-rot-l',...
    'arm-flex-r','arm-add-r','arm-rot-r',...
    'elbow-flex-l','elbow-flex-r'};
track_3D_corr = struct('m',[]);
figure()
for i = 1:length(idx_Qs)
    subplot(6,6,pos_Qs(i))
    % Experimental data
    Qs_toTrack_deg = Qs_toTrack_3D(ww_3D(k)).m;
    Qs_toTrack_deg(:,[2:4,8:end]) = Qs_toTrack_deg(:,[2:4,8:end])*180/pi;    
    plot(Qs_toTrack_deg(:,1),Qs_toTrack_deg(:,idx_Qs(i)+1),...
        'color',color_all(1,:),'linewidth',line_linewidth);  
    hold on
    % Simulation results
    for k = 1:length(ww_3D)
        plot(Qs_toTrack_3D(ww_3D(k)).m(:,1),...
            Qs_opt_3D(ww_3D(k)).m(:,idx_Qs(i)),'color',color_all(2,:),...
            'linestyle',':','linewidth',line_linewidth);
        hold on;   
        if strcmp(leg_Qs{i},'r')
            % heel-strike
            plot([Qs_toTrack_deg(HS_r(ww_3D(k)).m,1),...
                Qs_toTrack_deg(HS_r(ww_3D(k)).m,1)],...
                [ylim_Qs(i,1) ylim_Qs(i,2)],'k','linewidth',1);
        else
            % toe-off
            plot([Qs_toTrack_deg(TO_l(ww_3D(k)).m,1),...
                Qs_toTrack_deg(TO_l(ww_3D(k)).m,1)],...
                [ylim_Qs(i,1) ylim_Qs(i,2)],'k--','linewidth',1);
        end                
        [track_3D_corr(ww_3D(k)).m.r2.angles.all(i),...
            track_3D_corr(ww_3D(k)).m.rmse.angles.all(i)] = ...
            rsquare(Qs_toTrack_deg(:,idx_Qs(i)+1),...
                Qs_opt_3D(ww_3D(k)).m(:,idx_Qs(i)));        
    end    
    % Plot settings 
    set(gca,'Fontsize',label_fontsize);    
    title(RefData_str_tit{idx_Qs(i)},'Fontsize',label_fontsize);  
    % Y-axis
    ylim([ylim_Qs(i,1) ylim_Qs(i,2)]);
    L = get(gca,'YLim');
    set(gca,'YTick',linspace(L(1),L(2),NumTicks_Qs));       
    if i == 1 || i == 4 || i == 7 || i == 10
        ylabel('(°)','Fontsize',label_fontsize);
    end      
    % X-axis
    xlim([Qs_toTrack_deg(1,1),Qs_toTrack_deg(end,1)])
    set(gca,'XTick',[]);
    box off;
end 

%% Plot ground reaction forces
pos_GRF = [25:27,31:33];
ylim_GRF = [-50,50;0 150;-25 25;-50,50;0 150;-25 25];
NumTicks_GRF = 2;
temp = GRFs_toTrack_3D(ww_3D(1)).m(:,[3,6]);
temp(temp<0) = 0;
GRFs_toTrack_3D(ww_3D(1)).m(:,[3,6])= temp;
GRF_str = {'Fore-aft R','Vertical R','Lateral R',...
    'Fore-aft L','Vertical L','Lateral L'};
for i = 1:length(GRF_str)
    subplot(6,6,pos_GRF(i))
    % Experimental data
    plot(GRFs_toTrack_3D(ww_3D(k)).m(:,1),...
        GRFs_toTrack_3D(ww_3D(1)).m(:,i+1)./(body_weight/100),...
        'color',color_all(1,:),'linewidth',line_linewidth);
    hold on;
    % Simulation results
    for k = 1:length(ww_3D)
        plot(GRFs_toTrack_3D(ww_3D(k)).m(:,1),...
            GRFs_opt_3D(ww_3D(k)).m(:,i)./(body_weight/100),...
            'color',color_all(2,:),'linestyle',':','linewidth',line_linewidth);
        hold on;        
        if strcmp(leg_Qs{i},'r')
            % heel-strike
            plot([Qs_toTrack_deg(HS_r(ww_3D(k)).m,1),...
                Qs_toTrack_deg(HS_r(ww_3D(k)).m,1)],...
                [ylim_GRF(i,1) ylim_GRF(i,2)],'k','linewidth',1);
        else
            % toe-off
            plot([Qs_toTrack_deg(TO_l(ww_3D(k)).m,1),...
                Qs_toTrack_deg(TO_l(ww_3D(k)).m,1)],...
                [ylim_GRF(i,1) ylim_GRF(i,2)],'k--','linewidth',1);
        end
        [track_3D_corr(ww_3D(k)).m.r2.GRF.all(i),...
            track_3D_corr(ww_3D(k)).m.rmse.GRF.all(i)] = ...
            rsquare(GRFs_toTrack_3D(ww_3D(1)).m(:,i+1)./(body_weight/100),...
                GRFs_opt_3D(ww_3D(k)).m(:,i)./(body_weight/100));    
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
    set(gca,'XTick',[]);
    box off;   
end

%% Plot ground reaction torques
% Expressed with respect to ground frame origin
pos_GRM = [28:30,34:36];
ylim_GRM = [-20,200;-50 50;-250 250;-20,200;-50 50;-250 250];
for i = 1:length(GRF_str)
    subplot(6,6,pos_GRM(i))
    % Experimental data
    plot(GRMs_toTrack_3D(ww_3D(k)).m(:,1),GRMs_toTrack_3D(ww_3D(k)).m(:,i+1),...
        'color',color_all(1,:),'linewidth',line_linewidth);
    hold on;
    % Simulation results
    for k = 1:length(ww_3D)
        plot(GRMs_toTrack_3D(ww_3D(k)).m(:,1),GRMs_opt_3D(ww_3D(k)).m(:,i),...
            'color',color_all(2,:),'linestyle',':','linewidth',line_linewidth);
        hold on;        
        if strcmp(leg_Qs{i},'r')
            % heel-strike
            plot([Qs_toTrack_deg(HS_r(ww_3D(k)).m,1),...
                Qs_toTrack_deg(HS_r(ww_3D(k)).m,1)],...
                [ylim_GRM(i,1) ylim_GRM(i,2)],'k','linewidth',1);
        else
            % toe-off
            plot([Qs_toTrack_deg(TO_l(ww_3D(k)).m,1),...
                Qs_toTrack_deg(TO_l(ww_3D(k)).m,1)],...
                [ylim_GRM(i,1) ylim_GRM(i,2)],'k--','linewidth',1);
        end
        [track_3D_corr(ww_3D(k)).m.r2.GRM.all(i),...
            track_3D_corr(ww_3D(k)).m.rmse.GRM.all(i)] = ...
            rsquare(GRMs_toTrack_3D(ww_3D(k)).m(:,i+1),...
            GRMs_opt_3D(ww_3D(k)).m(:,i));  
    end
    % Plot settings 
    set(gca,'Fontsize',label_fontsize);    
    title(GRF_str{i},'Fontsize',label_fontsize);  
    % Y-axis
    ylim([ylim_GRM(i,1) ylim_GRM(i,2)]);
    L = get(gca,'YLim');
    set(gca,'YTick',linspace(L(1),L(2),NumTicks_GRF)); 
    ylabel('(Nm)','Fontsize',label_fontsize);
    % X-axis
    xlim([Qs_toTrack_deg(1,1),Qs_toTrack_deg(end,1)])
    set(gca,'XTick',[]);
    box off;    
end

%% Plot joint kinetics
idx_Qdots = [10,11,12,7,8,9,14,16,18,13,15,17]; 
ylim_Qdots = [-50,50;-100,100;-20,20;-50,50;-100,100;-20,20;-60,60;...
    -100,10;-20,20;-60,60;-100,10;-20,20];
pos_Qdots = [4:6,10:12,16:18,22:24];
for i = 1:length(idx_Qdots)
    subplot(6,6,pos_Qdots(i))
    % Experimental data
    plot(Ts_toTrack_3D(ww_3D(k)).m(:,1),...
        Ts_toTrack_3D(ww_3D(k)).m(:,idx_Qdots(i)+1),...
        'color',color_all(1,:),'linewidth',line_linewidth);
    hold on
    % Simulation results
    for k = 1:length(ww_3D)
        plot(Ts_toTrack_3D(ww_3D(k)).m(:,1),...
            Ts_opt_3D(ww_3D(k)).m(:,idx_Qdots(i)),...
            'color',color_all(2,:),'linestyle',':','linewidth',line_linewidth);
        hold on;        
        if strcmp(leg_Qs{i},'r')
            % heel-strike
            plot([Qs_toTrack_deg(HS_r(ww_3D(k)).m,1),...
                Qs_toTrack_deg(HS_r(ww_3D(k)).m,1)],...
                [ylim_Qdots(i,1) ylim_Qdots(i,2)],'k','linewidth',1);
        else
            % toe-off
            plot([Qs_toTrack_deg(TO_l(ww_3D(k)).m,1),...
                Qs_toTrack_deg(TO_l(ww_3D(k)).m,1)],...
                [ylim_Qdots(i,1) ylim_Qdots(i,2)],'k--','linewidth',1);
        end
        [track_3D_corr(ww_3D(k)).m.r2.T.all(i),...
            track_3D_corr(ww_3D(k)).m.rmse.T.all(i)] = ...
            rsquare(Ts_toTrack_3D(ww_3D(k)).m(:,idx_Qdots(i)+1),...
            Ts_opt_3D(ww_3D(k)).m(:,idx_Qdots(i)));  
    end
    % Plot settings 
    set(gca,'Fontsize',label_fontsize);    
    title(RefData_str_tit{idx_Qs(i)},'Fontsize',label_fontsize);  
    % Y-axis
    ylim([ylim_Qdots(i,1) ylim_Qdots(i,2)]);
    L = get(gca,'YLim');
    set(gca,'YTick',linspace(L(1),L(2),NumTicks_Qs));       
    if i == 1 || i == 4 || i == 7 || i == 10
        ylabel('(Nm)','Fontsize',label_fontsize);
    end      
    % X-axis
    xlim([Qs_toTrack_deg(1,1),Qs_toTrack_deg(end,1)])
    set(gca,'XTick',[]);
    box off;
end

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
muscleNames_tit = {'Glut med 1','Gluteus med','Glut med 3',...
    'Glut min 1','Glut min 2','Glut min 3','Semimem',...
    'Semiten','bifemlh','Bic fem sh','Sar','Adductor long',...
    'Add brev','Add mag 1','Add mag 2','Add mag 3','TFL',...
    'Pect','Grac','Glut max 1','Glut max 2','Glut max 3',......
    'Iliacus','Psoas','Quad fem','Gem','Peri',...
    'Rectus fem','Vas med','Vas int','Vastus lat','Med gas',...
    'Gastroc lat','Soleus','Tib post','Flex dig','Flex hal',...
    'Tibialis ant','Per brev','Per long','Per tert','Ext dig',...
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
pos_a_r = [4:6,10:12,16:2180];
idx_a = [2,12,24,28,31,10,33,34,38];
for i = 1:length(idx_a)
    subplot(6,6,pos_a_r(i))
    % Simulation results
    for k = 1:length(ww_3D)
        x = 1:(100-1)/(size(As_opt_3D(ww_3D(k)).m,1)-1):100;
        plot(Qs_toTrack_deg(:,1),As_opt_3D(ww_3D(k)).m(:,idx_a(i)+NMuscle/2),...
            'color',color_all(2,:),'linestyle',':','linewidth',line_linewidth);
        hold on;
    end
    % heel-strike
    plot([Qs_toTrack_deg(HS_r(ww_3D(k)).m,1),...
        Qs_toTrack_deg(HS_r(ww_3D(k)).m,1)],[0 1],'k','linewidth',1);

    if EMGcol_r(idx_a(i))
        % Normalize peak EMG to peak muscle activation
        a_peak = max(As_opt_3D(ww_3D(k)).m(:,idx_a(i)+NMuscle/2));
        emg_peak = max(EMGsel(:,strcmp(EMG.colheaders,EMGchannel_r{idx_a(i)})));
        norm_f = a_peak./emg_peak;       
        step = (EMGsel(end,1)-EMGsel(1,1))/(size(As_opt_3D(ww_3D(k)).m,1)-1);
        intervalInterp = EMGsel(1,1):step:EMGsel(end,1);        
        EMGselinterp = interp1(EMGsel(:,1),EMGsel(:,2:end),intervalInterp); 
        EMGselinterpInt = [intervalInterp',EMGselinterp];
        plot(Qs_toTrack_deg(:,1),EMGselinterpInt(:,strcmp(EMG.colheaders,...
            EMGchannel_r{idx_a(i)}))*norm_f,'k','linewidth',line_linewidth);     
    end
    % Plot settings
    set(gca,'Fontsize',label_fontsize)
    title(muscleNames_tit{idx_a(i)},'Fontsize',label_fontsize);    
    % X-axis
    xlim([Qs_toTrack_deg(1,1),Qs_toTrack_deg(end,1)])
    set(gca,'XTick',[]);
    % Y-axis
    ylim([0,1]);
    NumTicks = 2;
    LY = get(gca,'YLim');
    set(gca,'YTick',linspace(LY(1),LY(2),NumTicks))  
    if i == 1 || i == 4 || i == 7
        ylabel('(-)','Fontsize',label_fontsize)
    end 
    box off;
end

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
NMuscle = size(As_opt_3D(ww_3D(k)).m,2);
pos_a_l = [1:3,7:9,13:15];
for i = 1:length(idx_a)
    subplot(6,6,pos_a_l(i))
    % Simulation results
    for k = 1:length(ww_3D)
        x = 1:(100-1)/(size(As_opt_3D(ww_3D(k)).m,1)-1):100;
        plot(Qs_toTrack_deg(:,1),As_opt_3D(ww_3D(k)).m(:,idx_a(i)),...
            'color',color_all(2,:),'linestyle',':','linewidth',line_linewidth);
        hold on;
    end
    % toe-off
    plot([Qs_toTrack_deg(TO_l(ww_3D(k)).m,1),...
        Qs_toTrack_deg(TO_l(ww_3D(k)).m,1)],[0 1],'k--','linewidth',1);
    if EMGcol_l(idx_a(i))
        % Normalize peak EMG to peak muscle activation
        a_peak = max(As_opt_3D(ww_3D(k)).m(:,idx_a(i)+NMuscle/2));
        emg_peak = max(EMGsel(:,strcmp(EMG.colheaders,EMGchannel_l{idx_a(i)})));
        norm_f = a_peak./emg_peak;       
        step = (EMGsel(end,1)-EMGsel(1,1))/(size(As_opt_3D(ww_3D(k)).m,1)-1);
        intervalInterp = EMGsel(1,1):step:EMGsel(end,1);        
        EMGselinterp = interp1(EMGsel(:,1),EMGsel(:,2:end),intervalInterp); 
        EMGselinterpInt = [intervalInterp',EMGselinterp];
        plot(Qs_toTrack_deg(:,1),EMGselinterpInt(:,strcmp(EMG.colheaders,...
            EMGchannel_l{idx_a(i)}))*norm_f,'k','linewidth',line_linewidth);        
    end
    % Plot settings
    set(gca,'Fontsize',label_fontsize)
    title(muscleNames_tit{idx_a(i)},'Fontsize',label_fontsize);
    % X-axis
    xlim([Qs_toTrack_deg(1,1),Qs_toTrack_deg(end,1)])
    set(gca,'XTick',[]);
    ylim([0,1]);
    NumTicks = 2;
    LY = get(gca,'YLim');
    set(gca,'YTick',linspace(LY(1),LY(2),NumTicks))
    if i == 1 || i == 4 || i == 7
        ylabel('(-)','Fontsize',label_fontsize)
    end    
    box off;
end

%% Plot ground reaction forces
% Plotted a second time to have plots of size sizes; easy workaround.
pos_GRF = [25:27,31:33];
ylim_GRF = [-50,50;0 150;-25 25;-50,50;0 150;-25 25];
NumTicks_GRF = 2;
temp = GRFs_toTrack_3D(ww_3D(1)).m(:,[3,6]);
temp(temp<0) = 0;
GRFs_toTrack_3D(ww_3D(1)).m(:,[3,6])= temp;
GRF_str = {'Fore-aft R','Vertical R','Lateral R',...
    'Fore-aft L','Vertical L','Lateral L'};
for i = 1:length(GRF_str)
    subplot(6,6,pos_GRF(i))
    % Experimental data
    plot(GRFs_toTrack_3D(ww_3D(k)).m(:,1),...
        GRFs_toTrack_3D(ww_3D(1)).m(:,i+1)./(body_weight/100),...
        'color',color_all(1,:),'linewidth',line_linewidth);
    hold on;
    % Simulation results
    for k = 1:length(ww_3D)
        plot(GRFs_toTrack_3D(ww_3D(k)).m(:,1),...
            GRFs_opt_3D(ww_3D(k)).m(:,i)./(body_weight/100),...
            'color',color_all(2,:),'linestyle',':','linewidth',line_linewidth);
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
    set(gca,'XTick',[]);
    box off;   
end

%% Plot ground reaction moments
% Expressed with respect to ground frame origin
% Plotted a second time to have plots of size sizes; easy workaround.
pos_GRM = [28:30,34:36];
ylim_GRM = [-20,200;-50 50;-250 250;-20,200;-50 50;-250 250];
for i = 1:length(GRF_str)
    subplot(6,6,pos_GRM(i))
    % Experimental data
    plot(GRMs_toTrack_3D(ww_3D(k)).m(:,1),GRMs_toTrack_3D(ww_3D(k)).m(:,i+1),...
        'color',color_all(1,:),'linewidth',line_linewidth);
    hold on;
    % Simulation results
    for k = 1:length(ww_3D)
        plot(GRMs_toTrack_3D(ww_3D(k)).m(:,1),GRMs_opt_3D(ww_3D(k)).m(:,i),...
            'color',color_all(2,:),'linestyle',':','linewidth',line_linewidth);
        hold on;    
    end
    % Plot settings 
    set(gca,'Fontsize',label_fontsize);    
    title(GRF_str{i},'Fontsize',label_fontsize);  
    % Y-axis
    ylim([ylim_GRM(i,1) ylim_GRM(i,2)]);
    L = get(gca,'YLim');
    set(gca,'YTick',linspace(L(1),L(2),NumTicks_GRF));
    ylabel('(Nm)','Fontsize',label_fontsize);
    % X-axis
    xlim([Qs_toTrack_deg(1,1),Qs_toTrack_deg(end,1)])
    set(gca,'XTick',[]);
    box off;    
end

%% Analysis correlations
for k = 1:length(ww_3D)
    track_3D_corr(ww_3D(k)).m.r2.angles.mean = ...
        mean(track_3D_corr(ww_3D(k)).m.r2.angles.all);
    track_3D_corr(ww_3D(k)).m.r2.angles.std = ...
        std(track_3D_corr(ww_3D(k)).m.r2.angles.all);
    track_3D_corr(ww_3D(k)).m.r2.T.mean = ...
        mean(track_3D_corr(ww_3D(k)).m.r2.T.all);
    track_3D_corr(ww_3D(k)).m.r2.T.std = ...
        std(track_3D_corr(ww_3D(k)).m.r2.T.all);
    track_3D_corr(ww_3D(k)).m.r2.GRF.mean = ...
        mean(track_3D_corr(ww_3D(k)).m.r2.GRF.all);
    track_3D_corr(ww_3D(k)).m.r2.GRF.std = ...
        std(track_3D_corr(ww_3D(k)).m.r2.GRF.all);
    track_3D_corr(ww_3D(k)).m.r2.GRM.mean = ...
        mean(track_3D_corr(ww_3D(k)).m.r2.GRM.all);
    track_3D_corr(ww_3D(k)).m.r2.GRM.std = ...
        std(track_3D_corr(ww_3D(k)).m.r2.GRM.all);
    track_3D_corr(ww_3D(k)).m.r2.all.all = ...
        [track_3D_corr(ww_3D(k)).m.r2.angles.all,...
        track_3D_corr(ww_3D(k)).m.r2.T.all,...
        track_3D_corr(ww_3D(k)).m.r2.GRF.all,...
        track_3D_corr(ww_3D(k)).m.r2.GRM.all];
    track_3D_corr(ww_3D(k)).m.r2.all.mean = ...
        mean(track_3D_corr(ww_3D(k)).m.r2.all.all);
    track_3D_corr(ww_3D(k)).m.r2.all.std = ...
        std(track_3D_corr(ww_3D(k)).m.r2.all.all);
end
