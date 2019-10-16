% This script compares AD-Recorder with AD-ADOL-C with FD
% Author: Antoine Falisse
% Date: 1/7/2019

clear all
close all
clc

% Threshold used to discriminate between optimal solutions. We exclude the
% solution from an initial guess if it is larger than 
% threshold*(lowest solution across guesses).
threshold = 1.1;

%% Settings
% Select trials
% 1:3 => QR-guess (AD-Recorder, AD-ADOL-C, FD)
% 4:6 => DI-guess-walking (AD-Recorder, AD-ADOL-C, FD)
% 7:9 => DI-guess-running / DIm (AD-Recorder, AD-ADOL-C, FD)
% For tol=6
ww_2D  = [2,8,31,32];
% Load pre-defined settings
pathmain = pwd;
[pathMainRepo,~,~] = fileparts(pathmain);
pathRepo_2D = [pathMainRepo,'\predictiveSimulations_2D\'];
pathSettings_2D = [pathRepo_2D,'Settings'];
addpath(genpath(pathSettings_2D));
% Fixed settings
subject = 'subject1';
body_mass = 62;
body_weight = 62*9.81;
% Colors
color_all(1,:) = [244,194,13]/255;     % Yellow
color_all(2,:) = [219,50,54]/255;      % Red
color_all(4,:) = [60,186,84]/255;      % Green
color_all(3,:) = [0,0,0];              % Black
color_all(5,:) = [72,133,237]/255;     % Blue

%% Load results: predSim 2D
% Select setup 
setup.ocp = 'PredSim_2D'; 
settings_2D
% Pre-allocation structures
Qs_opt_2D              = struct('m',[]);
Qdots_opt_2D           = struct('m',[]);
Acts_opt_2D            = struct('m',[]);
GRFs_opt_2D            = struct('m',[]);
Ts_opt_2D              = struct('m',[]);
Stats_2D               = struct('m',[]);
% Loop over cases
for k = 1:length(ww_2D)
    data_2D;
end

%% Extract Results predSim 2D
t_proc_2D = zeros(length(ww_2D),5);
n_iter_2D = zeros(length(ww_2D),1);
fail_2D = 0;
obj_2D.all = zeros(length(ww_2D),1);
for k = 1:length(ww_2D)
    obj_2D.all(k) = Stats_2D(ww_2D(k)).m.iterations.obj(end);
    if Stats_2D(ww_2D(k)).m.success
        t_proc_2D(k,1)  = Stats_2D(ww_2D(k)).m.t_proc_solver - ...
            Stats_2D(ww_2D(k)).m.t_proc_nlp_f - ...
            Stats_2D(ww_2D(k)).m.t_proc_nlp_g - ...
            Stats_2D(ww_2D(k)).m.t_proc_nlp_grad - ...
            Stats_2D(ww_2D(k)).m.t_proc_nlp_grad_f - ...
            Stats_2D(ww_2D(k)).m.t_proc_nlp_jac_g;
        t_proc_2D(k,2)  = Stats_2D(ww_2D(k)).m.t_proc_nlp_f;
        t_proc_2D(k,3)  = Stats_2D(ww_2D(k)).m.t_proc_nlp_g;
        t_proc_2D(k,4)  = Stats_2D(ww_2D(k)).m.t_proc_nlp_grad_f;
        t_proc_2D(k,5)  = Stats_2D(ww_2D(k)).m.t_proc_nlp_jac_g;
        n_iter_2D(k)    = Stats_2D(ww_2D(k)).m.iter_count;  
    else
        % If the trial did not converge, then we assign NaN
        t_proc_2D(k,:) = NaN;
        n_iter_2D(k) = NaN;
        obj_2D.all(k) = NaN;
        fail_2D = fail_2D + 1;
        disp(['PredSim 2D: trial ',num2str(ww_2D(k)),' did not converge']);
    end       
end
t_proc_2D(:,end+1) = sum(t_proc_2D,2);

%% Analyze results
% No metabolic energy rate in the cost function

CPU_ratio.noME.all = t_proc_2D(2,:)./t_proc_2D(1,:);
CPU_ratio.ME.all = t_proc_2D(4,:)./t_proc_2D(3,:);

CPU_diff.noME.all = t_proc_2D(2,:)-t_proc_2D(1,:);
CPU_diff.ME.all = t_proc_2D(4,:)-t_proc_2D(3,:);

CPU_diff.noME.per = CPU_diff.noME.all(1:end-1)./CPU_diff.noME.all(end)*100;
CPU_diff.ME.per = CPU_diff.ME.all(1:end-1)./CPU_diff.ME.all(end)*100;
