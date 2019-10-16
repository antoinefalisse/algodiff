% This script plots results users might be intersted in (e.g.,joint angles,
% velocities, torques, ground reaction forces, muscle activations, CPU times)
%
% Author: Antoine Falisse
% Date: 9/9/2019
%
clear all
close all
clc

%% User settings
% Select trials, for example
ww_2D  = [1,2,3];

%% Other settings
% Load pre-defined settings
pathmain = pwd;
[pathRepo_2D,~,~] = fileparts(pathmain);
pathSettings = [pathRepo_2D,'\Settings'];
addpath(genpath(pathSettings));
subject = 'subject1';
body_mass = 62;
body_weight = 62*9.81;
setup.ocp = 'PredSim_2D'; 

%% Load results
% Pre-allocation structures
Qs_opt_2D       = struct('m',[]);
Qdots_opt_2D  	= struct('m',[]);
Acts_opt_2D  	= struct('m',[]);
GRFs_opt_2D    	= struct('m',[]);
Ts_opt_2D    	= struct('m',[]);
Stats_2D        = struct('m',[]);
legend_case  	= cell(1,length(ww_2D));
settings_2D
% Loop over cases
for k = 1:length(ww_2D)
    data_2D;
    legend_case{k} = ['Case: ',num2str(ww_2D(k))];
end

%% Helper names
joints_ref = {'pelvis_tilt','hip_flexion','knee_angle','ankle_angle',...
    'lumbar_extension'};
joints_tit = {'Pelvis tilt','Pelvis tx','Pelvis ty','Hip flexion L',...
    'Hip flexion R','Knee L','Knee R','Ankle L','Ankle R','Lumbar extension'};
GRF_str = {'Fore-aft','Vertical'};
muscleNames = {'hamstrings','bifemsh','glut-max','iliopsoas',...
    'rect-fem','vasti','gastroc','soleus','tib-ant'};  

%% Load reference data
pathmain = pwd;
[pathrepo,~,~] = fileparts(pathmain);
pathReferenceData = [pathrepo,'\ExperimentalData'];
load([pathReferenceData,'\ExperimentalData.mat'],'ExperimentalData');

%% Common settings for plots
label_fontsize  = 16;
sup_fontsize  = 24;
line_linewidth  = 3;
col = hsv(length(ww_2D));

%% Plot joint angles
Qref = ExperimentalData.Q;
idx_Qs = [1,5,7,9,10];
NumTicks = 2;
figure()
for i = 1:length(idx_Qs)
    subplot(2,3,i)
    p = gobjects(1,length(ww_2D));
    % Simulation results
    for k = 1:length(ww_2D)
        x = 1:(100-1)/(size(Qs_opt_2D(ww_2D(k)).m,1)-1):100;
        p(k) = plot(x,Qs_opt_2D(ww_2D(k)).m(:,idx_Qs(i)),...
            'color',col(k,:),'linewidth',line_linewidth);
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
    fill([x fliplr(x)],[meanPlusSTD fliplr(meanMinusSTD)],'k');
    alpha(.25);
    % Title
    set(gca,'Fontsize',label_fontsize);        
    title(joints_tit{idx_Qs(i)},'Fontsize',label_fontsize);
    % Y-axis
    if i == 1 || i == 4
        ylabel('Angle (°)','Fontsize',label_fontsize);
    end
    % X-axis
    L = get(gca,'XLim');    
    if i > 2
        set(gca,'XTick',linspace(L(1),L(2),NumTicks))
        xlabel('Gait cycle (%)','Fontsize',label_fontsize);
    else
        set(gca,'XTick',[]);
    end
end 
l = legend(p,legend_case);
set(l,'Fontsize',label_fontsize)
sp = suptitle('Joint angles');
set(sp,'Fontsize',sup_fontsize);  

%% Plot joint velocities
figure()
for i = 1:length(idx_Qs)
    subplot(2,3,i)
    p = gobjects(1,length(ww_2D));
    % Simulation results
    for k = 1:length(ww_2D)
        x = 1:(100-1)/(size(Qdots_opt_2D(ww_2D(k)).m,1)-1):100;
        p(k) = plot(x,Qdots_opt_2D(ww_2D(k)).m(:,idx_Qs(i)),...
            'color',col(k,:),'linewidth',line_linewidth);
        hold on;    
    end
    % Experimental data
    idx_jref = strcmp(Qref.(subject).Qdots.colheaders,joints_ref{i});
    meanPlusSTD = Qref.(subject).Qdots.mean(:,idx_jref) + ...
        2*Qref.(subject).Qdots.std(:,idx_jref);
    meanMinusSTD = Qref.(subject).Qdots.mean(:,idx_jref) - ...
        2*Qref.(subject).Qdots.std(:,idx_jref);          
    stepQ = (size(Qdots_opt_2D(ww_2D(k)).m,1)-1)/(size(meanPlusSTD,1)-1);
    intervalQ = 1:stepQ:size(Qdots_opt_2D(ww_2D(k)).m,1);
    sampleQ = 1:size(Qdots_opt_2D(ww_2D(k)).m,1);
    meanPlusSTD = interp1(intervalQ,meanPlusSTD,sampleQ);
    meanMinusSTD = interp1(intervalQ,meanMinusSTD,sampleQ);
    hold on
    fill([x fliplr(x)],[meanPlusSTD fliplr(meanMinusSTD)],'k');
    alpha(.25);
    % Title 
    set(gca,'Fontsize',label_fontsize);        
    title(joints_tit{idx_Qs(i)},'Fontsize',label_fontsize);
    % Y-axis
    if i == 1 || i == 4 
        ylabel('Velocity (°/s)','Fontsize',label_fontsize);
    end
    % X-axis
    L = get(gca,'XLim');    
    if i > 2
        set(gca,'XTick',linspace(L(1),L(2),NumTicks))
        xlabel('Gait cycle (%)','Fontsize',label_fontsize);
    else
        set(gca,'XTick',[]);
    end
end 
l = legend(p,legend_case);
set(l,'Fontsize',label_fontsize)
sp = suptitle('Joint velocities');
set(sp,'Fontsize',sup_fontsize);  

%% Plot ground reaction forces
GRFref = ExperimentalData.GRFs;
ylim_GRF = ([-50,50;0 200]);
NumTicks_GRF = 2;
figure()
for i = 1:length(GRF_str)
    subplot(1,2,i)
    % Simulation results
    p = gobjects(1,length(ww_2D));
    for k = 1:length(ww_2D)
        x = 1:(100-1)/(size(GRFs_opt_2D(ww_2D(k)).m,1)-1):100;
        p(k) = plot(x,GRFs_opt_2D(ww_2D(k)).m(:,i),'color',...
            col(k,:),'linewidth',line_linewidth);
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
    % Title
    set(gca,'Fontsize',label_fontsize);
    title(GRF_str{i},'Fontsize',label_fontsize);
    % Y-axis    
    ylim([ylim_GRF(i,1) ylim_GRF(i,2)]);
    L = get(gca,'YLim');
    if i == 1
        set(gca,'YTick',[L(1),0,L(2)]);    
    else
        set(gca,'YTick',linspace(L(1),L(2),NumTicks_GRF));      
    end
    if i == 1
        ylabel('Force (%BW)','Fontsize',label_fontsize);
    end
    % X-axis
    L = get(gca,'XLim');
    set(gca,'XTick',linspace(L(1),L(2),NumTicks_GRF))
    xlabel('Gait cycle (%)','Fontsize',label_fontsize); 
end
l = legend(p,legend_case);
set(l,'Fontsize',16)
sp = suptitle('Ground reaction forces');
set(sp,'Fontsize',sup_fontsize);

%% Plot joint kinetics
IDref = ExperimentalData.Torques;
figure()
for i = 1:length(idx_Qs)-1
    subplot(2,2,i)
    % Simulation results
    p = gobjects(1,length(ww_2D));
    for k = 1:length(ww_2D)
        x = 1:(100-1)/(size(Ts_opt_2D(ww_2D(k)).m,1)-1):100;
        p(k) = plot(x,Ts_opt_2D(ww_2D(k)).m(:,idx_Qs(i+1))*body_mass,...
            'color',col(k,:),'linewidth',line_linewidth);
        hold on;    
    end
    % Experimental data
    idx_jref = strcmp(IDref.(subject).colheaders,joints_ref{i+1});
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
    title(joints_tit{idx_Qs(i+1)},'Fontsize',label_fontsize);
    % Y-axis
    if i == 1 || i == 3
        ylabel('Torque (Nm)','Fontsize',label_fontsize);
    end
    % X-axis    
    L = get(gca,'XLim');
    NumTicks = 2;
    if i > 2
        set(gca,'XTick',linspace(L(1),L(2),NumTicks))
        xlabel('Gait cycle (%)','Fontsize',label_fontsize);
    else
        set(gca,'XTick',[]);
    end
end 
l = legend(p,legend_case);
set(l,'Fontsize',16)
sp = suptitle('Joint torques');
set(sp,'Fontsize',sup_fontsize);

%% Plot joint powers
Pref = ExperimentalData.Powers;
figure()
for i = 1:length(idx_Qs)-1
    subplot(2,2,i)
    % Simulation results
    p = gobjects(1,length(ww_2D));
    for k = 1:length(ww_2D)
        x = 1:(100-1)/(size(Qdots_opt_2D(ww_2D(k)).m,1)-1):100;
        p(k) = plot(x,Qdots_opt_2D(ww_2D(k)).m(:,idx_Qs(i+1)).*pi/180.*...
            Ts_opt_2D(ww_2D(k)).m(:,idx_Qs(i+1))*body_mass,...
            'color',col(k,:),'linewidth',line_linewidth);
        hold on;    
    end
    % Experimental data
    idx_jref = strcmp(Pref.(subject).colheaders,joints_ref{i+1});
    meanPlusSTD = Pref.(subject).mean(:,idx_jref) + ...
        2*Pref.(subject).std(:,idx_jref);
    meanMinusSTD = Pref.(subject).mean(:,idx_jref) - ...
        2*Pref.(subject).std(:,idx_jref);          
    stepQ = (size(Qdots_opt_2D(ww_2D(k)).m,1)-1)/(size(meanPlusSTD,1)-1);
    intervalQ = 1:stepQ:size(Qdots_opt_2D(ww_2D(k)).m,1);
    sampleQ = 1:size(Qdots_opt_2D(ww_2D(k)).m,1);
    meanPlusSTD = interp1(intervalQ,meanPlusSTD,sampleQ);
    meanMinusSTD = interp1(intervalQ,meanMinusSTD,sampleQ);
    hold on
    fill([x fliplr(x)],[meanPlusSTD fliplr(meanMinusSTD)],'k');
    alpha(.25);
    % Title
    set(gca,'Fontsize',label_fontsize);   
    title(joints_tit{idx_Qs(i+1)},'Fontsize',label_fontsize);
    % Y-axis
    if i == 1 || i == 3
        ylabel('Power (W)','Fontsize',label_fontsize);
    end
    % X-axis    
    L = get(gca,'XLim');
    NumTicks = 2;
    if i > 2 
        set(gca,'XTick',linspace(L(1),L(2),NumTicks))
        xlabel('Gait cycle (%)','Fontsize',label_fontsize);
    else
        set(gca,'XTick',[]);
    end
end 
l = legend(p,legend_case);
set(l,'Fontsize',16)
sp = suptitle('Joint powers');
set(sp,'Fontsize',sup_fontsize);

%% Plot muscle activations (right)
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
figure()
for i = 1:size(Acts_opt_2D(ww_2D(k)).m,2)/2
    subplot(2,5,i)
    p = gobjects(1,length(ww_2D));   
    % Simulation results
    for k = 1:length(ww_2D)
        x = 1:(100-1)/(size(Acts_opt_2D(ww_2D(k)).m,1)-1):100;
        p(k) = plot(x,Acts_opt_2D(ww_2D(k)).m(:,i+NMuscle/2),'color',...
            col(k,:),'linewidth',line_linewidth);
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
    NumTicks = 3;
    if i > 5
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
    if i == 1 || i == 6
        ylabel('(-)','Fontsize',20);
    end    
end
l = legend(p,legend_case);
set(l,'Fontsize',label_fontsize)
sp = suptitle('Muscle activations: right');
set(sp,'Fontsize',sup_fontsize);

%% Plot muscle activations (left)
figure()
for i = 1:size(Acts_opt_2D(ww_2D(k)).m,2)/2
    subplot(2,5,i)
    p = gobjects(1,length(ww_2D));
    NMuscle = size(Acts_opt_2D(ww_2D(k)).m,2);
    for k = 1:length(ww_2D)
        x = 1:(100-1)/(size(Acts_opt_2D(ww_2D(k)).m,1)-1):100;
        p(k) = plot(x,Acts_opt_2D(ww_2D(k)).m(:,i),'color',col(k,:),...
            'linewidth',line_linewidth);
        hold on;
    end
    % Plot settings
    set(gca,'Fontsize',label_fontsize)
    title(muscleNames{i},'Fontsize',label_fontsize);    
    % X-axis
    L = get(gca,'XLim');
    NumTicks = 3;
    if i > 5
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
    if i == 1 || i == 6
        ylabel('(-)','Fontsize',20);
    end    
end
l = legend(p,legend_case);
set(l,'Fontsize',label_fontsize)
sp = suptitle('Muscle activations: left');
set(sp,'Fontsize',sup_fontsize);

%% Plot CPU times / Optimal cost
CPU_IPOPT = struct('m',[]);
CPU_NLP = struct('m',[]);
for k = 1:length(ww_2D)
    CPU_IPOPT(ww_2D(k)).m = Stats_2D(ww_2D(k)).m.t_proc_solver - ...
        Stats_2D(ww_2D(k)).m.t_proc_nlp_f - ...
        Stats_2D(ww_2D(k)).m.t_proc_nlp_g - ...
        Stats_2D(ww_2D(k)).m.t_proc_nlp_grad - ...
        Stats_2D(ww_2D(k)).m.t_proc_nlp_grad_f - ...
        Stats_2D(ww_2D(k)).m.t_proc_nlp_jac_g;
    CPU_NLP(ww_2D(k)).m = Stats_2D(ww_2D(k)).m.t_proc_solver - ...
        CPU_IPOPT(ww_2D(k)).m;
end
figure()
subplot(2,3,1)
p = gobjects(1,length(ww_2D));
for k = 1:length(ww_2D)
    p(k) = scatter(k,CPU_IPOPT(ww_2D(k)).m,40,col(k,:),'filled');
    hold on;    
end
set(gca,'Fontsize',label_fontsize);
title('CPU time (IPOPT)','Fontsize',label_fontsize);
ylabel('(s)','Fontsize',label_fontsize);

subplot(2,3,2)
p = gobjects(1,length(ww_2D));
for k = 1:length(ww_2D)
    p(k) = scatter(k,CPU_NLP(ww_2D(k)).m,40,col(k,:),'filled');
    hold on;    
end
set(gca,'Fontsize',label_fontsize);
title('CPU time (NLP)','Fontsize',label_fontsize);
ylabel('(s)','Fontsize',label_fontsize);

subplot(2,3,3)
p = gobjects(1,length(ww_2D));
for k = 1:length(ww_2D)
    p(k) = scatter(k,(CPU_IPOPT(ww_2D(k)).m+CPU_NLP(ww_2D(k)).m),...
        40,col(k,:),'filled');
    hold on;    
end
set(gca,'Fontsize',label_fontsize);
title('CPU time (TOTAL)','Fontsize',label_fontsize);
ylabel('(s)','Fontsize',label_fontsize);

subplot(2,3,4)
p = gobjects(1,length(ww_2D));
for k = 1:length(ww_2D)
    p(k) = scatter(k,(Stats_2D(ww_2D(k)).m.iterations.obj(end)),...
        40,col(k,:),'filled');
    hold on;    
end
set(gca,'Fontsize',label_fontsize);
title('Optimal cost','Fontsize',label_fontsize);
ylabel('(-)','Fontsize',label_fontsize);
l = legend(p,legend_case);
set(l,'Fontsize',label_fontsize)

subplot(2,3,5)
p = gobjects(1,length(ww_2D));
for k = 1:length(ww_2D)
    p(k) = scatter(k,(Stats_2D(ww_2D(k)).m.iter_count),...
        40,col(k,:),'filled');
    hold on;    
end
set(gca,'Fontsize',label_fontsize);
title('# iterations','Fontsize',label_fontsize);
ylabel('(-)','Fontsize',label_fontsize);
l = legend(p,legend_case);
set(l,'Fontsize',label_fontsize)

subplot(2,3,6)
p = gobjects(1,length(ww_2D));
for k = 1:length(ww_2D)
    p(k) = scatter(k,(Stats_2D(ww_2D(k)).m.iterations.inf_du(end)),...
        40,col(k,:),'filled');
    hold on;    
end
set(gca,'Fontsize',label_fontsize);
title('Error NLP / Dual inf','Fontsize',label_fontsize);
ylabel('(-)','Fontsize',label_fontsize);
l = legend(p,legend_case);
set(l,'Fontsize',label_fontsize)

%% CPU time breakdown
yy = zeros(2,length(ww_2D));
for k = 1:length(ww_2D)
    yy(k,1) = Stats_2D(ww_2D(k)).m.t_proc_solver - ...
        Stats_2D(ww_2D(k)).m.t_proc_nlp_f - ...
        Stats_2D(ww_2D(k)).m.t_proc_nlp_g - ...
        Stats_2D(ww_2D(k)).m.t_proc_nlp_grad - ...
        Stats_2D(ww_2D(k)).m.t_proc_nlp_grad_f - ...
        Stats_2D(ww_2D(k)).m.t_proc_nlp_jac_g;
    yy(k,2) = Stats_2D(ww_2D(k)).m.t_proc_nlp_f;
    yy(k,3) = Stats_2D(ww_2D(k)).m.t_proc_nlp_g;
    yy(k,4) = Stats_2D(ww_2D(k)).m.t_proc_nlp_grad_f;
    yy(k,5) = Stats_2D(ww_2D(k)).m.t_proc_nlp_jac_g;
end
% Colors
color_all(1,:) = [244,194,13]/255;
color_all(2,:) = [60,186,84]/255;      
color_all(3,:) = [0,0,0];              
color_all(4,:) = [219,50,54]/255;      
color_all(5,:) = [72,133,237]/255;     
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
CPU_time_all.all = zeros(1,length(ww_2D));
for k = 1:length(ww_2D)
    CPU_time_all.all(k) = Stats_2D(ww_2D(k)).m.t_proc_solver;
end
CPU_time_all.mean = mean(CPU_time_all.all);
CPU_time_all.std = std(CPU_time_all.all);
