% This script reproduces part of Fig 5 (see paper)
%
% Author: Antoine Falisse
% Date: 9/9/2019
%
clear all
close all
clc

%% Settings
% Case
ww_2D  = 2;
% Load pre-defined settings
pathmain = pwd;
[pathRepo_2D,~,~] = fileparts(pathmain);
pathSettings = [pathRepo_2D,'\Settings'];
addpath(genpath(pathSettings));
setup.ocp = 'PredSim_2D';
subject = 'subject1';
body_mass = 62;
body_weight = 62*9.81;

%% Load results
% Pre-allocation structures
Qs_opt_2D       = struct('m',[]);
Qdots_opt_2D    = struct('m',[]);
Acts_opt_2D   	= struct('m',[]);
GRFs_opt_2D 	= struct('m',[]);
Ts_opt_2D    	= struct('m',[]);
Stats_2D      	= struct('m',[]);
% Loop over cases
settings_2D
for k = 1:length(ww_2D)
    data_2D;
end

%% Helper names
joints_ref = {'hip_flexion','knee_angle','ankle_angle'};
joints_tit = {'Pelvis tilt','Pelvis tx','Pelvis ty','Hip flex',...
    'Hip flex','Knee','Knee','Ankle','Ankle','Lumbar extension'};
GRF_str = {'Fore-aft','Vertical'};
muscleNames = {'Hamstrings','Bic fem sh','Glut max','Iliopsoas',...
    'Rectus fem','Vasti','Gastroc','Soleus','Tibialis ant'};  

%% Load reference data
pathmain = pwd;
[pathrepo,~,~] = fileparts(pathmain);
pathReferenceData = [pathrepo,'\ExperimentalData'];
load([pathReferenceData,'\ExperimentalData.mat'],'ExperimentalData');

%% Settings for plots
label_fontsize  = 14;
line_linewidth  = 3;
color_all(1,:) = [219,50,54]/255; % red

%% Plot joint angles
Qref = ExperimentalData.Q;
idx_Qs = [5,7,9];
pos_Qs = 1:6:13;
ylim_Qs = [-50,50;-80,0;-30,30];
NumTicks_Qs = 2;
figure()
for i = 1:length(idx_Qs)
    subplot(6,6,pos_Qs(i))
    % Simulation results
    for k = 1:length(ww_2D)
        x = 1:(100-1)/(size(Qs_opt_2D(ww_2D(k)).m,1)-1):100;
        plot(x,Qs_opt_2D(ww_2D(k)).m(:,idx_Qs(i)),'color',color_all(1,:),...
            'linestyle',':','linewidth',line_linewidth);
        hold on;    
    end
    % Experimental data
    idx_jref = strcmp(Qref.(subject).Qs.colheaders,joints_ref{i});
    meanPlusSTD = Qref.(subject).Qs.mean(:,idx_jref) + ...
        2*Qref.(subject).Qs.std(:,idx_jref);
    meanMinusSTD = Qref.(subject).Qs.mean(:,idx_jref) - ...
        2*Qref.(subject).Qs.std(:,idx_jref);          
    stepQ = (size(Qs_opt_2D(ww_2D(k)).m,1)-1)/(size(meanPlusSTD,1)-1);
    intervalQ = 1:stepQ:size(Qs_opt_2D(ww_2D(k)).m,1);
    sampleQ = 1:size(Qs_opt_2D(ww_2D(k)).m,1);
    meanPlusSTD = interp1(intervalQ,meanPlusSTD,sampleQ);
    meanMinusSTD = interp1(intervalQ,meanMinusSTD,sampleQ);
    hold on
    te = fill([x fliplr(x)],[meanPlusSTD fliplr(meanMinusSTD)],'k');
    alpha(.25);
    % Title 
    set(gca,'Fontsize',label_fontsize);        
    title(joints_tit{idx_Qs(i)},'Fontsize',label_fontsize);
    % Y-axis
    ylim([ylim_Qs(i,1) ylim_Qs(i,2)]);
    L = get(gca,'YLim');
    set(gca,'YTick',linspace(L(1),L(2),NumTicks_Qs)); 
    ylabel('(°)','Fontsize',label_fontsize);
    % X-axis
    L = get(gca,'XLim'); 
    set(gca,'XTick',[]);
    box off;  
end 

%% Plot joint torques
IDref = ExperimentalData.Torques;
pos_Qdots = 2:6:14;
ylim_Qdots = [-80,80;-40,80;-120,40];
for i = 1:length(idx_Qs)
    subplot(6,6,pos_Qdots(i))
    % Simulation results
    for k = 1:length(ww_2D)
        x = 1:(100-1)/(size(Ts_opt_2D(ww_2D(k)).m,1)-1):100;
        plot(x,Ts_opt_2D(ww_2D(k)).m(:,idx_Qs(i))*body_mass,'color',...
            color_all(1,:),'linestyle',':','linewidth',line_linewidth);
        hold on;    
    end
    % Experimental data
    idx_jref = strcmp(IDref.(subject).colheaders,joints_ref{i});
    meanPlusSTD = IDref.(subject).mean(:,idx_jref) + ...
        2*IDref.(subject).std(:,idx_jref);
    meanMinusSTD = IDref.(subject).mean(:,idx_jref) - ...
        2*IDref.(subject).std(:,idx_jref);  
    stepID = (size(Ts_opt_2D(ww_2D(k)).m,1)-1)/(size(meanPlusSTD,1)-1);
    intervalID = 1:stepID:size(Ts_opt_2D(ww_2D(k)).m,1);
    sampleID = 1:size(Ts_opt_2D(ww_2D(k)).m,1);
    meanPlusSTD = interp1(intervalID,meanPlusSTD,sampleID);
    meanMinusSTD = interp1(intervalID,meanMinusSTD,sampleID); 
    hold on
    fill([x fliplr(x)],[meanPlusSTD fliplr(meanMinusSTD)],'k');
    alpha(.25);
    % Title 
    set(gca,'Fontsize',label_fontsize);   
    title(joints_tit{idx_Qs(i)},'Fontsize',label_fontsize);
    % Y-axis
    ylabel('(Nm)','Fontsize',label_fontsize);
    % X-axis    
    ylim([ylim_Qdots(i,1) ylim_Qdots(i,2)]);
    L = get(gca,'YLim');
    set(gca,'YTick',linspace(L(1),L(2),NumTicks_Qs));
    set(gca,'XTick',[]);
    box off;  
end 

%% Plot muscle activations
EMGref = ExperimentalData.EMG;
% Create a "fake" new channel for vastus intermedius that is (Vmed+Vlat)/2
EMGref.subject1.all(:,15,:) = EMGref.subject1.all(:,7,:) + ...
    (EMGref.subject1.all(:,8,:))./2;
EMGref.subject1.mean(:,15) = nanmean(EMGref.subject1.all(:,15,:),3);
EMGref.subject1.std(:,15) = nanstd(EMGref.subject1.all(:,15,:),[],3);
EMGref.subject1.colheaders{15} = 'Vastus-intermedius';
% Selecting muscles
EMGchannel = [1,1,9,99,10,15,12,6,2];
EMGcol = ones(1,length(muscleNames));
EMGcol(EMGchannel==99)=0;
EMGref.(subject).allnorm = ...
    NaN(2*N,size(EMGref.(subject).all,2),size(EMGref.(subject).all,3));
% Plot   
NMuscle = size(Acts_opt_2D(ww_2D(k)).m,2);
pos_a = [3:5,9:11,15:17];
for i = 1:size(Acts_opt_2D(ww_2D(k)).m,2)/2
    subplot(6,6,pos_a(i))
    p = gobjects(1,length(ww_2D));   
    % Simulation results
    for k = 1:length(ww_2D)
        x = 1:(100-1)/(size(Acts_opt_2D(ww_2D(k)).m,1)-1):100;
        p(k) = plot(x,Acts_opt_2D(ww_2D(k)).m(:,i+NMuscle/2),'color',...
            color_all(1,:),'linestyle',':','linewidth',line_linewidth);
        hold on;
    end
    if EMGcol(i)
        % Normalize peak EMG to peak muscle activation
        a_peak = max(Acts_opt_2D(ww_2D(k)).m(:,i+NMuscle/2));
        emg_peak = zeros(1,size(EMGref.(subject).all,3));
        for j = 1:size(EMGref.(subject).all,3)
            emg_peak(j) = nanmax(EMGref.(subject).all(:,EMGchannel(i),j),[],1);
        end
        norm_f = a_peak./emg_peak;           
        tempp(:,:) = EMGref.(subject).all(:,EMGchannel(i),:);
        intervalInterp = 1:(size(tempp,1)-1)/(2*N-1):size(tempp,1);
        temp = interp1(1:size(tempp,1),tempp,intervalInterp);        
        temp = temp.*repmat(norm_f,2*N,1);
        EMGref.(subject).allnorm(:,EMGchannel(i),:) = temp;
        EMGref.(subject).meannorm = nanmean(EMGref.(subject).allnorm,3);
        EMGref.(subject).stdnorm = nanstd(EMGref.(subject).allnorm,[],3);        
        meanPlusSTD = EMGref.(subject).meannorm(:,EMGchannel(i)) + ...
            2*EMGref.(subject).stdnorm(:,EMGchannel(i));
        meanMinusSTD = EMGref.(subject).meannorm(:,EMGchannel(i)) - ...
            2*EMGref.(subject).stdnorm(:,EMGchannel(i));
        stepa = (size(Acts_opt_2D(ww_2D(k)).m,1)-1)/(size(meanMinusSTD,1)-1);
        intervala = 1:stepa:size(Acts_opt_2D(ww_2D(k)).m,1);
        samplea = 1:size(Acts_opt_2D(ww_2D(k)).m,1);
        meanPlusSTD = interp1(intervala,meanPlusSTD,samplea);
        meanMinusSTD = interp1(intervala,meanMinusSTD,samplea);     
        hold on
        fill([x fliplr(x)],[meanPlusSTD fliplr(meanMinusSTD)],'k');
        alpha(.25);            
    end
    % Title
    set(gca,'Fontsize',label_fontsize)
    title(muscleNames{i},'Fontsize',label_fontsize);    
    % X-axis
    L = get(gca,'XLim');
    if i > 6
        set(gca,'XTick',linspace(L(1),L(2),2))
        xlabel('Gait cycle (%)','Fontsize',label_fontsize);
    else
        set(gca,'XTick',[]);
    end
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

%% Plot ground reaction forces
GRFref = ExperimentalData.GRFs;
pos_GRF = 6:6:12;
ylim_GRF = ([-50,50;0 200]);
for i = 1:length(GRF_str)
    subplot(6,6,pos_GRF(i))
    % Simulation results
    p = gobjects(1,length(ww_2D));
    for k = 1:length(ww_2D)
        x = 1:(100-1)/(size(GRFs_opt_2D(ww_2D(k)).m,1)-1):100;
        p(k) = plot(x,GRFs_opt_2D(ww_2D(k)).m(:,i),'color',color_all(1,:),...
            'linestyle',':','linewidth',line_linewidth);
        hold on;         
    end    
    % Experimental data
    meanPlusSTD = GRFref.(subject).mean(:,i) + 2*GRFref.(subject).std(:,i);    
    meanMinusSTD = GRFref.(subject).mean(:,i) - 2*GRFref.(subject).std(:,i);   
    stepGRF = (size(GRFs_opt_2D(ww_2D(k)).m,1)-1)/(size(meanPlusSTD,1)-1);
    intervalGRF = 1:stepGRF:size(GRFs_opt_2D(ww_2D(k)).m,1);
    sampleGRF = 1:size(GRFs_opt_2D(ww_2D(k)).m,1);
    meanPlusSTD = interp1(intervalGRF,meanPlusSTD,sampleGRF);
    meanMinusSTD = interp1(intervalGRF,meanMinusSTD,sampleGRF);
    hold on
    fill([x fliplr(x)],[meanPlusSTD fliplr(meanMinusSTD)],'k');     
    alpha(.25);
    % Plot settings 
    set(gca,'Fontsize',label_fontsize);
    title(GRF_str{i},'Fontsize',label_fontsize);
    % Y-axis        
    ylim([ylim_GRF(i,1) ylim_GRF(i,2)]);
    L = get(gca,'YLim');
    set(gca,'YTick',linspace(L(1),L(2),NumTicks_Qs));
    ylabel('(%BW)','Fontsize',label_fontsize);
    % X-axis
    if i == 2
    L = get(gca,'XLim');
    set(gca,'XTick',linspace(L(1),L(2),2))
    xlabel('Gait cycle (%)','Fontsize',label_fontsize);
    end
    set(gca,'XTick',[]);
    box off;  
end
