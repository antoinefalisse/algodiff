% This script compares linear solvers
% Author: Antoine Falisse
% Date: 1/7/2019

clear all
close all
clc

% Threshold used to discriminate between optimal solutions. We exclude the
% solution from an initial guess if it is larger than 
% threshold*(lowest solution across guesses).
threshold = 1.01;

%% Settings
% Select trials
% For PredSim2D
% 1:6 => QR-guess (mumps,ma27,ma57,ma77,ma86,ma97)
% 7:12 => DI-guess-walking (mumps,ma27,ma57,ma77,ma86,ma97)
% 13:19 => DI-guess-running (mumps,ma27,ma57,ma77,ma86,ma97)
% For PredSim3D (MATLAB crashes for ma57 and 97 for TrackSim3D)
% 1:4 => QR-guess (mumps,ma27,ma77,ma86)
% 5:8 => DI-guess (mumps,ma27,ma77,ma86)
% 9:12 => DIm-guess (mumps,ma27,ma77,ma86)
% For tol=4
% ww_2D = [48:53,2,13:17,54:59];
% ww_3D = [1,24,26,27,2,8,10:11,13,29,31,32];
% For tol=6
ww_2D = [1,10,13,16,19,22,2,11,14,17,20,23,3,12,15,18,21,24];
ww_3D = [1,10,13,16,2,11,14,17,3,12,15,18];

% ww_3D = [5,34:36,6,37:39,23,40:42];
ww_pend = 2:10;
% Load pre-defined settings
pathmain = pwd;
[pathMainRepo,~,~] = fileparts(pathmain);
pathRepo_2D = [pathMainRepo,'\predictiveSimulations_2D\'];
pathSettings_2D = [pathRepo_2D,'Settings'];
addpath(genpath(pathSettings_2D));
pathRepo_3D = [pathMainRepo,'\trackingSimulations_3D\'];
pathSettings_3D = [pathRepo_3D,'Settings'];
addpath(genpath(pathSettings_3D));
pathResults_pend = [pathMainRepo,'\pendulumSimulations\Results\'];
% Fixed settings
subject = 'subject1';
body_mass = 62;
body_weight = 62*9.81;
% Colors
color_all(1,:) = [244,194,13]/255;     % Yellow
color_all(2,:) = [219,50,54]/255;      % Red
color_all(3,:) = [125,46,140]/255;         % Black
color_all(4,:) = [60,186,84]/255;      % Green
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

%% Load results: trackSim 3D
% Select setup 
setup.ocp = 'TrackSim_3D'; 
settings_3D
% Pre-allocation structures
Qs_opt_3D              = struct('m',[]);
Qdots_opt_3D           = struct('m',[]);
Acts_opt_3D            = struct('m',[]);
GRFs_opt_3D            = struct('m',[]);
Ts_opt_3D              = struct('m',[]);
Stats_3D               = struct('m',[]);
% Loop over cases
for k = 1:length(ww_3D)
    data_3D;
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
        t_proc_2D(k,:) = NaN;
        n_iter_2D(k) = NaN;
        obj_2D.all(k) = NaN;
        fail_2D = fail_2D + 1;
        disp(['PredSim 2D: trial ',num2str(ww_2D(k)),' did not converge']);
    end
end
% Assess convergence: we extract the optimal cost.
obj_2D.mumps = obj_2D.all(1:6:end,1);
obj_2D.ma27 = obj_2D.all(2:6:end,1);
obj_2D.ma57 = obj_2D.all(3:6:end,1);
obj_2D.ma77 = obj_2D.all(4:6:end,1);
obj_2D.ma86 = obj_2D.all(5:6:end,1);
obj_2D.ma97 = obj_2D.all(6:6:end,1);
% We discriminate between optimal solutions. We exclude the solution from an 
% initial guess if it is larger than threshold*(lowest solution across guesses).
min_obj_2D.mumps = min(obj_2D.mumps);
idx_obj_2D.mumps = obj_2D.mumps > (threshold*min_obj_2D.mumps);
min_obj_2D.ma27 = min(obj_2D.ma27);
idx_obj_2D.ma27 = obj_2D.ma27 > (threshold*min_obj_2D.ma27);
min_obj_2D.ma57 = min(obj_2D.ma57);
idx_obj_2D.ma57 = obj_2D.ma57 > (threshold*min_obj_2D.ma57);
min_obj_2D.ma77 = min(obj_2D.ma77);
idx_obj_2D.ma77 = obj_2D.ma77 > (threshold*min_obj_2D.ma77);
min_obj_2D.ma86 = min(obj_2D.ma86);
idx_obj_2D.ma86 = obj_2D.ma86 > (threshold*min_obj_2D.ma86);
min_obj_2D.ma97 = min(obj_2D.ma97);
idx_obj_2D.ma97 = obj_2D.ma97 > (threshold*min_obj_2D.ma97);
idx_obj_2D.all = [idx_obj_2D.mumps,idx_obj_2D.ma27,idx_obj_2D.ma57,...
    idx_obj_2D.ma77,idx_obj_2D.ma86,idx_obj_2D.ma97];
% We compare the lowest optimal solutions across cases and issue a warning
% if they differ across cases
min_mumps_ma27 = abs(min_obj_2D.mumps-min_obj_2D.ma27) < (1.02-threshold)*min(min_obj_2D.mumps,min_obj_2D.ma27);
min_mumps_ma57 = abs(min_obj_2D.mumps-min_obj_2D.ma57) < (1.02-threshold)*min(min_obj_2D.mumps,min_obj_2D.ma57);
min_mumps_ma77 = abs(min_obj_2D.mumps-min_obj_2D.ma77) < (1.02-threshold)*min(min_obj_2D.mumps,min_obj_2D.ma77);
min_mumps_ma86 = abs(min_obj_2D.mumps-min_obj_2D.ma86) < (1.02-threshold)*min(min_obj_2D.mumps,min_obj_2D.ma86);
min_mumps_ma97 = abs(min_obj_2D.mumps-min_obj_2D.ma97) < (1.02-threshold)*min(min_obj_2D.mumps,min_obj_2D.ma97);
if ~min_mumps_ma27
    disp('2D Pred Sim: mumps and ma27 have different lowest optimal cost')
end
if ~min_mumps_ma57
    disp('2D Pred Sim: mumps and ma57 have different lowest optimal cost')
end
if ~min_mumps_ma77
    disp('2D Pred Sim: mumps and ma77 have different lowest optimal cost')
end
if ~min_mumps_ma86
    disp('2D Pred Sim: mumps and ma86 have different lowest optimal cost')
end
if ~min_mumps_ma97
    disp('2D Pred Sim: mumps and ma97 have different lowest optimal cost')
end
% mumps
t_proc_all.pred2D.mumps.all = t_proc_2D(1:6:end,:);
t_proc_all.pred2D.mumps.all(idx_obj_2D.mumps,:) = NaN;
t_proc_all.pred2D.mumps.all(:,end+1) = sum(t_proc_all.pred2D.mumps.all,2);
t_proc_all.pred2D.mumps.mean = nanmean(t_proc_all.pred2D.mumps.all,1);
t_proc_all.pred2D.mumps.std = nanstd(t_proc_all.pred2D.mumps.all,[],1);
n_iter_all.pred2D.mumps.all = n_iter_2D(1:6:end,:);
n_iter_all.pred2D.mumps.all(idx_obj_2D.mumps,:) = NaN;
n_iter_all.pred2D.mumps.mean = nanmean(n_iter_all.pred2D.mumps.all,1);
n_iter_all.pred2D.mumps.std = nanstd(n_iter_all.pred2D.mumps.all,[],1);
t_iter_all.pred2D.mumps.all = t_proc_all.pred2D.mumps.all(:,end)./...
    n_iter_all.pred2D.mumps.all;
% ma27
t_proc_all.pred2D.ma27.all = t_proc_2D(2:6:end,:);
t_proc_all.pred2D.ma27.all(idx_obj_2D.ma27,:) = NaN;
t_proc_all.pred2D.ma27.all(:,end+1) = sum(t_proc_all.pred2D.ma27.all,2);
t_proc_all.pred2D.ma27.mean = nanmean(t_proc_all.pred2D.ma27.all,1);
t_proc_all.pred2D.ma27.std = nanstd(t_proc_all.pred2D.ma27.all,[],1);
n_iter_all.pred2D.ma27.all = n_iter_2D(2:6:end,:);
n_iter_all.pred2D.ma27.all(idx_obj_2D.ma27,:) = NaN;
n_iter_all.pred2D.ma27.mean = nanmean(n_iter_all.pred2D.ma27.all,1);
n_iter_all.pred2D.ma27.std = nanstd(n_iter_all.pred2D.ma27.all,[],1);
t_iter_all.pred2D.ma27.all = t_proc_all.pred2D.ma27.all(:,end)./...
    n_iter_all.pred2D.ma27.all;
% ma57
t_proc_all.pred2D.ma57.all = t_proc_2D(3:6:end,:);
t_proc_all.pred2D.ma57.all(idx_obj_2D.ma57,:) = NaN;
t_proc_all.pred2D.ma57.all(:,end+1) = sum(t_proc_all.pred2D.ma57.all,2);
t_proc_all.pred2D.ma57.mean = nanmean(t_proc_all.pred2D.ma57.all,1);
t_proc_all.pred2D.ma57.std = nanstd(t_proc_all.pred2D.ma57.all,[],1);
n_iter_all.pred2D.ma57.all = n_iter_2D(3:6:end,:);
n_iter_all.pred2D.ma57.all(idx_obj_2D.ma57,:) = NaN;
n_iter_all.pred2D.ma57.mean = nanmean(n_iter_all.pred2D.ma57.all,1);
n_iter_all.pred2D.ma57.std = nanstd(n_iter_all.pred2D.ma57.all,[],1);
t_iter_all.pred2D.ma57.all = t_proc_all.pred2D.ma57.all(:,end)./...
    n_iter_all.pred2D.ma57.all;
% ma77
t_proc_all.pred2D.ma77.all = t_proc_2D(4:6:end,:);
t_proc_all.pred2D.ma77.all(idx_obj_2D.ma77,:) = NaN;
t_proc_all.pred2D.ma77.all(:,end+1) = sum(t_proc_all.pred2D.ma77.all,2);
t_proc_all.pred2D.ma77.mean = nanmean(t_proc_all.pred2D.ma77.all,1);
t_proc_all.pred2D.ma77.std = nanstd(t_proc_all.pred2D.ma77.all,[],1);
n_iter_all.pred2D.ma77.all = n_iter_2D(4:6:end,:);
n_iter_all.pred2D.ma77.all(idx_obj_2D.ma77,:) = NaN;
n_iter_all.pred2D.ma77.mean = nanmean(n_iter_all.pred2D.ma77.all,1);
n_iter_all.pred2D.ma77.std = nanstd(n_iter_all.pred2D.ma77.all,[],1);
t_iter_all.pred2D.ma77.all = t_proc_all.pred2D.ma77.all(:,end)./...
    n_iter_all.pred2D.ma77.all;
% ma86
t_proc_all.pred2D.ma86.all = t_proc_2D(5:6:end,:);
t_proc_all.pred2D.ma86.all(idx_obj_2D.ma86,:) = NaN;
t_proc_all.pred2D.ma86.all(:,end+1) = sum(t_proc_all.pred2D.ma86.all,2);
t_proc_all.pred2D.ma86.mean = nanmean(t_proc_all.pred2D.ma86.all,1);
t_proc_all.pred2D.ma86.std = nanstd(t_proc_all.pred2D.ma86.all,[],1);
n_iter_all.pred2D.ma86.all = n_iter_2D(5:6:end,:);
n_iter_all.pred2D.ma86.all(idx_obj_2D.ma86,:) = NaN;
n_iter_all.pred2D.ma86.mean = nanmean(n_iter_all.pred2D.ma86.all,1);
n_iter_all.pred2D.ma86.std = nanstd(n_iter_all.pred2D.ma86.all,[],1);
t_iter_all.pred2D.ma86.all = t_proc_all.pred2D.ma86.all(:,end)./...
    n_iter_all.pred2D.ma86.all;
% ma97
t_proc_all.pred2D.ma97.all = t_proc_2D(6:6:end,:);
t_proc_all.pred2D.ma97.all(idx_obj_2D.ma97,:) = NaN;
t_proc_all.pred2D.ma97.all(:,end+1) = sum(t_proc_all.pred2D.ma97.all,2);
t_proc_all.pred2D.ma97.mean = nanmean(t_proc_all.pred2D.ma97.all,1);
t_proc_all.pred2D.ma97.std = nanstd(t_proc_all.pred2D.ma97.all,[],1);
n_iter_all.pred2D.ma97.all = n_iter_2D(6:6:end,:);
n_iter_all.pred2D.ma97.all(idx_obj_2D.ma97,:) = NaN;
n_iter_all.pred2D.ma97.mean = nanmean(n_iter_all.pred2D.ma97.all,1);
n_iter_all.pred2D.ma97.std = nanstd(n_iter_all.pred2D.ma97.all,[],1);
t_iter_all.pred2D.ma97.all = t_proc_all.pred2D.ma97.all(:,end)./...
    n_iter_all.pred2D.ma97.all;

%% Extract Results trackSim 3D
t_proc_3D = zeros(length(ww_3D),5);
n_iter_3D = zeros(length(ww_3D),1);
fail_3D = 0;
obj_3D.all = zeros(length(ww_3D),1);
for k = 1:length(ww_3D)
    obj_3D.all(k) = Stats_3D(ww_3D(k)).m.iterations.obj(end);
    if Stats_3D(ww_3D(k)).m.success
        t_proc_3D(k,1)  = Stats_3D(ww_3D(k)).m.t_proc_solver - ...
            Stats_3D(ww_3D(k)).m.t_proc_nlp_f - ...
            Stats_3D(ww_3D(k)).m.t_proc_nlp_g - ...
            Stats_3D(ww_3D(k)).m.t_proc_nlp_grad - ...
            Stats_3D(ww_3D(k)).m.t_proc_nlp_grad_f - ...
            Stats_3D(ww_3D(k)).m.t_proc_nlp_jac_g;
        t_proc_3D(k,2)  = Stats_3D(ww_3D(k)).m.t_proc_nlp_f;
        t_proc_3D(k,3)  = Stats_3D(ww_3D(k)).m.t_proc_nlp_g;
        t_proc_3D(k,4)  = Stats_3D(ww_3D(k)).m.t_proc_nlp_grad_f;
        t_proc_3D(k,5)  = Stats_3D(ww_3D(k)).m.t_proc_nlp_jac_g;
        n_iter_3D(k)    = Stats_3D(ww_3D(k)).m.iter_count;  
    else
        t_proc_3D(k,:) = NaN;
        n_iter_3D(k) = NaN;
        obj_3D.all(k) = NaN;
        fail_3D = fail_3D + 1;
        disp(['TrackSim 3D: trial ',num2str(ww_3D(k)),' did not converge']);
    end    
end
% Assess convergence: we extract the optimal cost.
obj_3D.mumps = obj_3D.all(1:4:end,1);
obj_3D.ma27 = obj_3D.all(2:4:end,1);
obj_3D.ma77 = obj_3D.all(3:4:end,1);
obj_3D.ma86 = obj_3D.all(4:4:end,1);
% We discriminate between optimal solutions. We exclude the solution from an 
% initial guess if it is larger than threshold*(lowest solution across guesses).
min_obj_3D.mumps = min(obj_3D.mumps);
idx_obj_3D.mumps = obj_3D.mumps > (threshold*min_obj_3D.mumps);
min_obj_3D.ma27 = min(obj_3D.ma27);
idx_obj_3D.ma27 = obj_3D.ma27 > (threshold*min_obj_3D.ma27);
min_obj_3D.ma77 = min(obj_3D.ma77);
idx_obj_3D.ma77 = obj_3D.ma77 > (threshold*min_obj_3D.ma77);
min_obj_3D.ma86 = min(obj_3D.ma86);
idx_obj_3D.ma86 = obj_3D.ma86 > (threshold*min_obj_3D.ma86);
idx_obj_3D.all = [idx_obj_3D.mumps,idx_obj_3D.ma27,...
    idx_obj_3D.ma77,idx_obj_3D.ma86];
% We compare the lowest optimal solutions across cases and issue a warning
% if they differ across cases
min_mumps_ma27 = abs(min_obj_3D.mumps-min_obj_3D.ma27) < (1.02-threshold)*min(min_obj_3D.mumps,min_obj_3D.ma27);
min_mumps_ma77 = abs(min_obj_3D.mumps-min_obj_3D.ma77) < (1.02-threshold)*min(min_obj_3D.mumps,min_obj_3D.ma77);
min_mumps_ma86 = abs(min_obj_3D.mumps-min_obj_3D.ma86) < (1.02-threshold)*min(min_obj_3D.mumps,min_obj_3D.ma86);
if ~min_mumps_ma27
    disp('3D Track Sim: mumps and ma27 have different lowest optimal cost')
end
if ~min_mumps_ma77
    disp('3D Track Sim: mumps and ma77 have different lowest optimal cost')
end
if ~min_mumps_ma86
    disp('3D Track Sim: mumps and ma86 have different lowest optimal cost')
end
% mumps
t_proc_all.track3D.mumps.all = t_proc_3D(1:4:end,:);
t_proc_all.track3D.mumps.all(idx_obj_3D.mumps,:) = NaN;
t_proc_all.track3D.mumps.all(:,end+1) = sum(t_proc_all.track3D.mumps.all,2);
t_proc_all.track3D.mumps.mean = nanmean(t_proc_all.track3D.mumps.all,1);
t_proc_all.track3D.mumps.std = nanstd(t_proc_all.track3D.mumps.all,[],1);
n_iter_all.track3D.mumps.all = n_iter_3D(1:4:end,:);
n_iter_all.track3D.mumps.all(idx_obj_3D.mumps,:) = NaN;
n_iter_all.track3D.mumps.mean = nanmean(n_iter_all.track3D.mumps.all,1);
n_iter_all.track3D.mumps.std = nanstd(n_iter_all.track3D.mumps.all,[],1);
t_iter_all.track3D.mumps.all = t_proc_all.track3D.mumps.all(:,end)./...
    n_iter_all.track3D.mumps.all;
% ma27
t_proc_all.track3D.ma27.all = t_proc_3D(2:4:end,:);
t_proc_all.track3D.ma27.all(idx_obj_3D.ma27,:) = NaN;
t_proc_all.track3D.ma27.all(:,end+1) = sum(t_proc_all.track3D.ma27.all,2);
t_proc_all.track3D.ma27.mean = nanmean(t_proc_all.track3D.ma27.all,1);
t_proc_all.track3D.ma27.std = nanstd(t_proc_all.track3D.ma27.all,[],1);
n_iter_all.track3D.ma27.all = n_iter_3D(2:4:end,:);
n_iter_all.track3D.ma27.all(idx_obj_3D.ma27,:) = NaN;
n_iter_all.track3D.ma27.mean = nanmean(n_iter_all.track3D.ma27.all,1);
n_iter_all.track3D.ma27.std = nanstd(n_iter_all.track3D.ma27.all,[],1);
t_iter_all.track3D.ma27.all = t_proc_all.track3D.ma27.all(:,end)./...
    n_iter_all.track3D.ma27.all;
% ma57
t_proc_all.track3D.ma57.all = NaN(size(t_proc_all.track3D.ma27.all));
n_iter_all.track3D.ma57.all = NaN(size(n_iter_all.track3D.ma27.all));
t_iter_all.track3D.ma57.all = NaN(size(n_iter_all.track3D.ma27.all));
% ma77
t_proc_all.track3D.ma77.all = t_proc_3D(3:4:end,:);
t_proc_all.track3D.ma77.all(idx_obj_3D.ma77,:) = NaN;
t_proc_all.track3D.ma77.all(:,end+1) = sum(t_proc_all.track3D.ma77.all,2);
t_proc_all.track3D.ma77.mean = nanmean(t_proc_all.track3D.ma77.all,1);
t_proc_all.track3D.ma77.std = nanstd(t_proc_all.track3D.ma77.all,[],1);
n_iter_all.track3D.ma77.all = n_iter_3D(3:4:end,:);
n_iter_all.track3D.ma77.all(idx_obj_3D.ma77,:) = NaN;
n_iter_all.track3D.ma77.mean = nanmean(n_iter_all.track3D.ma77.all,1);
n_iter_all.track3D.ma77.std = nanstd(n_iter_all.track3D.ma77.all,[],1);
t_iter_all.track3D.ma77.all = t_proc_all.track3D.ma77.all(:,end)./...
    n_iter_all.track3D.ma77.all;
% ma86
t_proc_all.track3D.ma86.all = t_proc_3D(4:4:end,:);
t_proc_all.track3D.ma86.all(idx_obj_3D.ma86,:) = NaN;
t_proc_all.track3D.ma86.all(:,end+1) = sum(t_proc_all.track3D.ma86.all,2);
t_proc_all.track3D.ma86.mean = nanmean(t_proc_all.track3D.ma86.all,1);
t_proc_all.track3D.ma86.std = nanstd(t_proc_all.track3D.ma86.all,[],1);
n_iter_all.track3D.ma86.all = n_iter_3D(4:4:end,:);
n_iter_all.track3D.ma86.all(idx_obj_3D.ma86,:) = NaN;
n_iter_all.track3D.ma86.mean = nanmean(n_iter_all.track3D.ma86.all,1);
n_iter_all.track3D.ma86.std = nanstd(n_iter_all.track3D.ma86.all,[],1);
t_iter_all.track3D.ma86.all = t_proc_all.track3D.ma86.all(:,end)./...
    n_iter_all.track3D.ma86.all;
% ma97
t_proc_all.track3D.ma97.all = NaN(size(t_proc_all.track3D.ma27.all));
n_iter_all.track3D.ma97.all = NaN(size(n_iter_all.track3D.ma27.all));
t_iter_all.track3D.ma97.all = NaN(size(n_iter_all.track3D.ma27.all));

%% Extract Results Pendulums
for i = 2:10
    % Add results from AD
    load([pathResults_pend,'\Pendulum_',num2str(i),'dofs\solution_AD'],...
        'solution');
    PendulumResultsAll.(['pendulum_',num2str(i),'dofs']).AD = solution;
    % Add results from FD
    load([pathResults_pend,'\Pendulum_',num2str(i),'dofs\solution_FD_F'],...
        'solution');
    PendulumResultsAll.(['pendulum_',num2str(i),'dofs']).FD = solution;
end
count = 1;
Options.Hessian='approx'; dim3=2;
tol_pend = 2;
der = {'AD'};
dim5=1; % Recorder
NIG = 10;
solvers={'mumps','ma27','ma57','ma77','ma86','ma97'};
NCases_pend = length(ww_pend)*length(solvers)*NIG;
t_proc_pend = zeros(NCases_pend,5);
n_iter_pend = zeros(NCases_pend,1);
obj_pend.all = zeros(NCases_pend,1);
fail_pend = 0;
for k = 2:length(ww_pend)+1 % loop over pendulum cases (eg 2dof)
    solution = PendulumResultsAll.(['pendulum_',num2str(k),'dofs']).(der{:});
    for linsol=solvers
        Options.solver=linsol{:};
        if strcmp(Options.solver,'mumps')
            dim4=1;
        elseif strcmp(Options.solver,'ma27')
            dim4=2;
        elseif strcmp(Options.solver,'ma57')
            dim4=3;
        elseif strcmp(Options.solver,'ma77')
            dim4=4;
        elseif strcmp(Options.solver,'ma86')
            dim4=5;
        elseif strcmp(Options.solver,'ma97')
            dim4=6;
        end
        for i_IG=1:NIG % loop over initial guesses            
            stats_temp = solution(i_IG,tol_pend,dim3,dim4,dim5).stats;          
            obj_pend.all(count) = stats_temp.iterations.obj(end);
            if stats_temp.success
                t_proc_pend(count,1) = stats_temp.t_proc_solver - ...
                    stats_temp.t_proc_nlp_f - ...
                    stats_temp.t_proc_nlp_g - ...
                    stats_temp.t_proc_nlp_grad - ...
                    stats_temp.t_proc_nlp_grad_f - ...
                    stats_temp.t_proc_nlp_jac_g;
                t_proc_pend(count,2) = stats_temp.t_proc_nlp_f;
                t_proc_pend(count,3) = stats_temp.t_proc_nlp_g;
                t_proc_pend(count,4) = stats_temp.t_proc_nlp_grad_f;
                t_proc_pend(count,5) = stats_temp.t_proc_nlp_jac_g;
                n_iter_pend(count)   = stats_temp.iter_count;    
            else
                t_proc_pend(count,:) = NaN;
                n_iter_pend(count) = NaN;
                obj_pend.all(count) = NaN;
                fail_pend = fail_pend + 1;
                disp(['Pendulum: trial ',num2str(count),' did not converge']);
            end                          
            count = count + 1;
        end
    end
end
% Assess convergence: we extract the optimal cost.
for k = 2:length(ww_pend)+1
    obj_pend.(['pendulum',num2str(k),'dof']).mumps = obj_pend.all((k-2)*(6*NIG)+1:(k-2)*(6*NIG)+NIG,1);
    obj_pend.(['pendulum',num2str(k),'dof']).ma27 = obj_pend.all((k-2)*(6*NIG)+1+NIG:(k-2)*(6*NIG)+2*NIG,1);
    obj_pend.(['pendulum',num2str(k),'dof']).ma57 = obj_pend.all((k-2)*(6*NIG)+1+2*NIG:(k-2)*(6*NIG)+3*NIG,1);
    obj_pend.(['pendulum',num2str(k),'dof']).ma77 = obj_pend.all((k-2)*(6*NIG)+1+3*NIG:(k-2)*(6*NIG)+4*NIG,1);
    obj_pend.(['pendulum',num2str(k),'dof']).ma86 = obj_pend.all((k-2)*(6*NIG)+1+4*NIG:(k-2)*(6*NIG)+5*NIG,1);
    obj_pend.(['pendulum',num2str(k),'dof']).ma97 = obj_pend.all((k-2)*(6*NIG)+1+5*NIG:(k-2)*(6*NIG)+6*NIG,1);
    % We discriminate between optimal solutions. We exclude the solution from an 
    % initial guess if it is larger than threshold*(lowest solution across guesses).
    min_obj_pend.mumps = min(obj_pend.(['pendulum',num2str(k),'dof']).mumps);
    idx_obj_pend.(['pendulum',num2str(k),'dof']).mumps = obj_pend.(['pendulum',num2str(k),'dof']).mumps > (threshold*min_obj_pend.mumps);
    min_obj_pend.ma27 = min(obj_pend.(['pendulum',num2str(k),'dof']).ma27);
    idx_obj_pend.(['pendulum',num2str(k),'dof']).ma27 = obj_pend.(['pendulum',num2str(k),'dof']).ma27 > (threshold*min_obj_pend.ma27);
    min_obj_pend.ma57 = min(obj_pend.(['pendulum',num2str(k),'dof']).ma57);
    idx_obj_pend.(['pendulum',num2str(k),'dof']).ma57 = obj_pend.(['pendulum',num2str(k),'dof']).ma57 > (threshold*min_obj_pend.ma57);
    min_obj_pend.ma77 = min(obj_pend.(['pendulum',num2str(k),'dof']).ma77);
    idx_obj_pend.(['pendulum',num2str(k),'dof']).ma77 = obj_pend.(['pendulum',num2str(k),'dof']).ma77 > (threshold*min_obj_pend.ma77);
    min_obj_pend.ma86 = min(obj_pend.(['pendulum',num2str(k),'dof']).ma86);
    idx_obj_pend.(['pendulum',num2str(k),'dof']).ma86 = obj_pend.(['pendulum',num2str(k),'dof']).ma86 > (threshold*min_obj_pend.ma86);
    min_obj_pend.ma97 = min(obj_pend.(['pendulum',num2str(k),'dof']).ma97);
    idx_obj_pend.(['pendulum',num2str(k),'dof']).ma97 = obj_pend.(['pendulum',num2str(k),'dof']).ma97 > (threshold*min_obj_pend.ma97);
    idx_obj_pend.(['pendulum',num2str(k),'dof']).all = [idx_obj_pend.(['pendulum',num2str(k),'dof']).mumps,...
        idx_obj_pend.(['pendulum',num2str(k),'dof']).ma27,idx_obj_pend.(['pendulum',num2str(k),'dof']).ma57,...
        idx_obj_pend.(['pendulum',num2str(k),'dof']).ma77,idx_obj_pend.(['pendulum',num2str(k),'dof']).ma86,...
        idx_obj_pend.(['pendulum',num2str(k),'dof']).ma97];
    % We compare the lowest optimal solutions across cases and issue a warning
    % if they differ across cases
    min_mumps_ma27 = abs(min_obj_pend.mumps-min_obj_pend.ma27) < (1.02-threshold)*min(min_obj_pend.mumps,min_obj_pend.ma27);
    min_mumps_ma57 = abs(min_obj_pend.mumps-min_obj_pend.ma57) < (1.02-threshold)*min(min_obj_pend.mumps,min_obj_pend.ma57);
    min_mumps_ma77 = abs(min_obj_pend.mumps-min_obj_pend.ma77) < (1.02-threshold)*min(min_obj_pend.mumps,min_obj_pend.ma77);
    min_mumps_ma86 = abs(min_obj_pend.mumps-min_obj_pend.ma86) < (1.02-threshold)*min(min_obj_pend.mumps,min_obj_pend.ma86);
    min_mumps_ma97 = abs(min_obj_pend.mumps-min_obj_pend.ma97) < (1.02-threshold)*min(min_obj_pend.mumps,min_obj_pend.ma97);
    if ~min_mumps_ma27
        disp(['Pendulum',num2str(k),'dof: mumps and ma27 have different lowest optimal cost'])
    end
    if ~min_mumps_ma57
        disp(['Pendulum',num2str(k),'dof: mumps and ma57 have different lowest optimal cost'])
    end
    if ~min_mumps_ma77
        disp(['Pendulum',num2str(k),'dof: mumps and ma77 have different lowest optimal cost'])
    end
    if ~min_mumps_ma86
        disp(['Pendulum',num2str(k),'dof: mumps and ma86 have different lowest optimal cost'])
    end
    if ~min_mumps_ma97
        disp(['Pendulum',num2str(k),'dof: mumps and ma97 have different lowest optimal cost'])
    end
    % Average across mumps cases
    t_proc_all.(['pendulum',num2str(k),'dof']).mumps.all = t_proc_pend((k-2)*(6*NIG)+1:(k-2)*(6*NIG)+NIG,:);
    t_proc_all.(['pendulum',num2str(k),'dof']).mumps.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).mumps,:) = NaN;
    t_proc_all.(['pendulum',num2str(k),'dof']).mumps.all(:,end+1) = sum(t_proc_all.(['pendulum',num2str(k),'dof']).mumps.all,2);
    t_proc_all.(['pendulum',num2str(k),'dof']).mumps.mean = nanmean(t_proc_all.(['pendulum',num2str(k),'dof']).mumps.all,1);
    t_proc_all.(['pendulum',num2str(k),'dof']).mumps.std = nanstd(t_proc_all.(['pendulum',num2str(k),'dof']).mumps.all,[],1);
    n_iter_all.(['pendulum',num2str(k),'dof']).mumps.all = n_iter_pend((k-2)*(6*NIG)+1:(k-2)*(6*NIG)+NIG,:);
    n_iter_all.(['pendulum',num2str(k),'dof']).mumps.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).mumps,:) = NaN;
    n_iter_all.(['pendulum',num2str(k),'dof']).mumps.mean = nanmean(n_iter_all.(['pendulum',num2str(k),'dof']).mumps.all,1);
    n_iter_all.(['pendulum',num2str(k),'dof']).mumps.std = nanstd(n_iter_all.(['pendulum',num2str(k),'dof']).mumps.all,[],1);
    t_iter_all.(['pendulum',num2str(k),'dof']).mumps.all = t_proc_all.(['pendulum',num2str(k),'dof']).mumps.all(:,end)./...
        n_iter_all.(['pendulum',num2str(k),'dof']).mumps.all;
    % Average across ma27 cases
    t_proc_all.(['pendulum',num2str(k),'dof']).ma27.all = t_proc_pend((k-2)*(6*NIG)+1+NIG:(k-2)*(6*NIG)+2*NIG,:);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma27.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma27,:) = NaN;
    t_proc_all.(['pendulum',num2str(k),'dof']).ma27.all(:,end+1) = sum(t_proc_all.(['pendulum',num2str(k),'dof']).ma27.all,2);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma27.mean = nanmean(t_proc_all.(['pendulum',num2str(k),'dof']).ma27.all,1);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma27.std = nanstd(t_proc_all.(['pendulum',num2str(k),'dof']).ma27.all,[],1);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma27.all = n_iter_pend((k-2)*(6*NIG)+1+NIG:(k-2)*(6*NIG)+2*NIG,:);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma27.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma27,:) = NaN;
    n_iter_all.(['pendulum',num2str(k),'dof']).ma27.mean = nanmean(n_iter_all.(['pendulum',num2str(k),'dof']).ma27.all,1);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma27.std = nanstd(n_iter_all.(['pendulum',num2str(k),'dof']).ma27.all,[],1);
    t_iter_all.(['pendulum',num2str(k),'dof']).ma27.all = t_proc_all.(['pendulum',num2str(k),'dof']).ma27.all(:,end)./...
        n_iter_all.(['pendulum',num2str(k),'dof']).ma27.all;
    % Average across ma57 cases
    t_proc_all.(['pendulum',num2str(k),'dof']).ma57.all = t_proc_pend((k-2)*(6*NIG)+1+2*NIG:(k-2)*(6*NIG)+3*NIG,:);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma57.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma57,:) = NaN;
    t_proc_all.(['pendulum',num2str(k),'dof']).ma57.all(:,end+1) = sum(t_proc_all.(['pendulum',num2str(k),'dof']).ma57.all,2);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma57.mean = nanmean(t_proc_all.(['pendulum',num2str(k),'dof']).ma57.all,1);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma57.std = nanstd(t_proc_all.(['pendulum',num2str(k),'dof']).ma57.all,[],1);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma57.all = n_iter_pend((k-2)*(6*NIG)+1+2*NIG:(k-2)*(6*NIG)+3*NIG,:);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma57.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma57,:) = NaN;
    n_iter_all.(['pendulum',num2str(k),'dof']).ma57.mean = nanmean(n_iter_all.(['pendulum',num2str(k),'dof']).ma57.all,1);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma57.std = nanstd(n_iter_all.(['pendulum',num2str(k),'dof']).ma57.all,[],1);
    t_iter_all.(['pendulum',num2str(k),'dof']).ma57.all = t_proc_all.(['pendulum',num2str(k),'dof']).ma57.all(:,end)./...
        n_iter_all.(['pendulum',num2str(k),'dof']).ma57.all;
    % Average across ma77 cases
    t_proc_all.(['pendulum',num2str(k),'dof']).ma77.all = t_proc_pend((k-2)*(6*NIG)+1+3*NIG:(k-2)*(6*NIG)+4*NIG,:);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma77.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma77,:) = NaN;
    t_proc_all.(['pendulum',num2str(k),'dof']).ma77.all(:,end+1) = sum(t_proc_all.(['pendulum',num2str(k),'dof']).ma77.all,2);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma77.mean = nanmean(t_proc_all.(['pendulum',num2str(k),'dof']).ma77.all,1);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma77.std = nanstd(t_proc_all.(['pendulum',num2str(k),'dof']).ma77.all,[],1);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma77.all = n_iter_pend((k-2)*(6*NIG)+1+3*NIG:(k-2)*(6*NIG)+4*NIG,:);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma77.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma77,:) = NaN;
    n_iter_all.(['pendulum',num2str(k),'dof']).ma77.mean = nanmean(n_iter_all.(['pendulum',num2str(k),'dof']).ma77.all,1);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma77.std = nanstd(n_iter_all.(['pendulum',num2str(k),'dof']).ma77.all,[],1);
    t_iter_all.(['pendulum',num2str(k),'dof']).ma77.all = t_proc_all.(['pendulum',num2str(k),'dof']).ma77.all(:,end)./...
        n_iter_all.(['pendulum',num2str(k),'dof']).ma77.all;
    % Average across ma86 cases
    t_proc_all.(['pendulum',num2str(k),'dof']).ma86.all = t_proc_pend((k-2)*(6*NIG)+1+4*NIG:(k-2)*(6*NIG)+5*NIG,:);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma86.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma86,:) = NaN;
    t_proc_all.(['pendulum',num2str(k),'dof']).ma86.all(:,end+1) = sum(t_proc_all.(['pendulum',num2str(k),'dof']).ma86.all,2);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma86.mean = nanmean(t_proc_all.(['pendulum',num2str(k),'dof']).ma86.all,1);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma86.std = nanstd(t_proc_all.(['pendulum',num2str(k),'dof']).ma86.all,[],1);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma86.all = n_iter_pend((k-2)*(6*NIG)+1+4*NIG:(k-2)*(6*NIG)+5*NIG,:);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma86.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma86,:) = NaN;
    n_iter_all.(['pendulum',num2str(k),'dof']).ma86.mean = nanmean(n_iter_all.(['pendulum',num2str(k),'dof']).ma86.all,1);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma86.std = nanstd(n_iter_all.(['pendulum',num2str(k),'dof']).ma86.all,[],1);
    t_iter_all.(['pendulum',num2str(k),'dof']).ma86.all = t_proc_all.(['pendulum',num2str(k),'dof']).ma86.all(:,end)./...
        n_iter_all.(['pendulum',num2str(k),'dof']).ma86.all;
    % Average across ma97 cases
    t_proc_all.(['pendulum',num2str(k),'dof']).ma97.all = t_proc_pend((k-2)*(6*NIG)+1+5*NIG:(k-2)*(6*NIG)+6*NIG,:);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma97.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma97,:) = NaN;
    t_proc_all.(['pendulum',num2str(k),'dof']).ma97.all(:,end+1) = sum(t_proc_all.(['pendulum',num2str(k),'dof']).ma97.all,2);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma97.mean = nanmean(t_proc_all.(['pendulum',num2str(k),'dof']).ma97.all,1);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma97.std = nanstd(t_proc_all.(['pendulum',num2str(k),'dof']).ma97.all,[],1);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma97.all = n_iter_pend((k-2)*(6*NIG)+1+5*NIG:(k-2)*(6*NIG)+6*NIG,:);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma97.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma97,:) = NaN;
    n_iter_all.(['pendulum',num2str(k),'dof']).ma97.mean = nanmean(n_iter_all.(['pendulum',num2str(k),'dof']).ma97.all,1);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma97.std = nanstd(n_iter_all.(['pendulum',num2str(k),'dof']).ma97.all,[],1);    
    t_iter_all.(['pendulum',num2str(k),'dof']).ma97.all = t_proc_all.(['pendulum',num2str(k),'dof']).ma97.all(:,end)./...
        n_iter_all.(['pendulum',num2str(k),'dof']).ma97.all;
end

%% Differences in CPU time between cases
% Combine results from PredSim 2D and TrackSim 3D and Pendulums
t_proc_all.pred2D_3D_pend.mumps.all = [t_proc_all.pred2D.mumps.all;t_proc_all.track3D.mumps.all];
t_proc_all.pred2D_3D_pend.ma27.all = [t_proc_all.pred2D.ma27.all;t_proc_all.track3D.ma27.all];
t_proc_all.pred2D_3D_pend.ma57.all = [t_proc_all.pred2D.ma57.all;t_proc_all.track3D.ma57.all];
t_proc_all.pred2D_3D_pend.ma77.all = [t_proc_all.pred2D.ma77.all;t_proc_all.track3D.ma77.all];
t_proc_all.pred2D_3D_pend.ma86.all = [t_proc_all.pred2D.ma86.all;t_proc_all.track3D.ma86.all];
t_proc_all.pred2D_3D_pend.ma97.all = [t_proc_all.pred2D.ma97.all;t_proc_all.track3D.ma97.all];
for k = 2:length(ww_pend)+1 
    t_proc_all.pred2D_3D_pend.mumps.all = [t_proc_all.pred2D_3D_pend.mumps.all;t_proc_all.(['pendulum',num2str(k),'dof']).mumps.all];
    t_proc_all.pred2D_3D_pend.ma27.all = [t_proc_all.pred2D_3D_pend.ma27.all;t_proc_all.(['pendulum',num2str(k),'dof']).ma27.all];
    t_proc_all.pred2D_3D_pend.ma57.all = [t_proc_all.pred2D_3D_pend.ma57.all;t_proc_all.(['pendulum',num2str(k),'dof']).ma57.all];
    t_proc_all.pred2D_3D_pend.ma77.all = [t_proc_all.pred2D_3D_pend.ma77.all;t_proc_all.(['pendulum',num2str(k),'dof']).ma77.all];
    t_proc_all.pred2D_3D_pend.ma86.all = [t_proc_all.pred2D_3D_pend.ma86.all;t_proc_all.(['pendulum',num2str(k),'dof']).ma86.all];
    t_proc_all.pred2D_3D_pend.ma97.all = [t_proc_all.pred2D_3D_pend.ma97.all;t_proc_all.(['pendulum',num2str(k),'dof']).ma97.all];
end
% Get number of discarded cases
nan_mumps = sum(isnan(t_proc_all.pred2D_3D_pend.mumps.all(:,1))); % 1 => QR-guess tracking
nan_ma27 = sum(isnan(t_proc_all.pred2D_3D_pend.ma27.all(:,1))); % 1 => QR-guess tracking
nan_ma57 = sum(isnan(t_proc_all.pred2D_3D_pend.ma57.all(:,1))); % 3 => no results for tracking
nan_ma77 = sum(isnan(t_proc_all.pred2D_3D_pend.ma77.all(:,1))); % 2 => QR-guess tracking and QR-guess predictive
nan_ma86 = sum(isnan(t_proc_all.pred2D_3D_pend.ma86.all(:,1))); % 1 => QR-guess tracking
nan_ma97 = sum(isnan(t_proc_all.pred2D_3D_pend.ma97.all(:,1))); % 3 => no results for tracking
% Calculate ratio between ma27 and mumps
% All: PredSim 2D, TrackSim 3D, and Pendulums
CPU_ratio.pred2D_3D_pend.ma27_mumps.all = t_proc_all.pred2D_3D_pend.ma27.all./t_proc_all.pred2D_3D_pend.mumps.all;
CPU_ratio.pred2D_3D_pend.ma27_mumps.mean = nanmean(CPU_ratio.pred2D_3D_pend.ma27_mumps.all,1);
CPU_ratio.pred2D_3D_pend.ma27_mumps.std = nanstd(CPU_ratio.pred2D_3D_pend.ma27_mumps.all,[],1);
% PredSim 2D
CPU_ratio.pred2D.ma27_mumps.all = t_proc_all.pred2D.ma27.all./t_proc_all.pred2D.mumps.all;
CPU_ratio.pred2D.ma27_mumps.mean = nanmean(CPU_ratio.pred2D.ma27_mumps.all,1);
CPU_ratio.pred2D.ma27_mumps.std = nanstd(CPU_ratio.pred2D.ma27_mumps.all,[],1);
CPU_ratio.pred2D.ma27_mumps_r.mean = round(nanmean(CPU_ratio.pred2D.ma27_mumps.all,1),1);
CPU_ratio.pred2D.ma27_mumps_r.std = round(nanstd(CPU_ratio.pred2D.ma27_mumps.all,[],1),1);
CPU_ratio.pred2D.mumps_ma27.mean = round(nanmean(1./CPU_ratio.pred2D.ma27_mumps.all,1),1); % inversion so mumps wrt ma27, + means mumps slower
CPU_ratio.pred2D.mumps_ma27.std = round(nanstd(1./CPU_ratio.pred2D.ma27_mumps.all,[],1),1); % inversion so mumps wrt ma27, + means mumps slower
% TrackSim 3D
CPU_ratio.track3D.ma27_mumps.all = t_proc_all.track3D.ma27.all./t_proc_all.track3D.mumps.all;
CPU_ratio.track3D.ma27_mumps.mean = nanmean(CPU_ratio.track3D.ma27_mumps.all,1); 
CPU_ratio.track3D.ma27_mumps.std = nanstd(CPU_ratio.track3D.ma27_mumps.all,[],1);
% Result table: TrackSim 3D
CPU_ratio.track3D.ma27_mumps_r.mean = round(nanmean(CPU_ratio.track3D.ma27_mumps.all,1),1); 
CPU_ratio.track3D.ma27_mumps_r.std = round(nanstd(CPU_ratio.track3D.ma27_mumps.all,[],1),1);
CPU_ratio.track3D.mumps_ma27.mean = round(nanmean(1./CPU_ratio.track3D.ma27_mumps.all,1),1); % inversion so mumps wrt ma27, + means mumps slower
CPU_ratio.track3D.mumps_ma27.std = round(nanstd(1./CPU_ratio.track3D.ma27_mumps.all,[],1),1); % inversion so mumps wrt ma27, + means mumps slower
% Pendulums
for k = 2:length(ww_pend)+1
    CPU_ratio.(['pendulum',num2str(k),'dof']).ma27_mumps.all = t_proc_all.(['pendulum',num2str(k),'dof']).ma27.all./t_proc_all.(['pendulum',num2str(k),'dof']).mumps.all;
    CPU_ratio.(['pendulum',num2str(k),'dof']).ma27_mumps.mean = nanmean(CPU_ratio.(['pendulum',num2str(k),'dof']).ma27_mumps.all,1);
    CPU_ratio.(['pendulum',num2str(k),'dof']).ma27_mumps.std = nanstd(CPU_ratio.(['pendulum',num2str(k),'dof']).ma27_mumps.all,[],1);
    CPU_ratio.pendulum_all.ma27_mumps.mean(k-1) = CPU_ratio.(['pendulum',num2str(k),'dof']).ma27_mumps.mean(end);
    CPU_ratio.pendulum_all.ma27_mumps.std(k-1) = CPU_ratio.(['pendulum',num2str(k),'dof']).ma27_mumps.std(end);
end
CPU_ratio.pred2D_3D_pend.ma27_mumps.all_mean = [CPU_ratio.pendulum_all.ma27_mumps.mean,CPU_ratio.pred2D.ma27_mumps.mean(end),CPU_ratio.track3D.ma27_mumps.mean(end)];
CPU_ratio.pred2D_3D_pend.ma27_mumps.all_std = [CPU_ratio.pendulum_all.ma27_mumps.std,CPU_ratio.pred2D.ma27_mumps.std(end),CPU_ratio.track3D.ma27_mumps.std(end)];
% Numbers for paper (mean of mean): I want to give the same weight to the
% different examples. Taking the mean over all trials would give more
% importance to the pendulum cases although in the end the average does
% not really change. The std (variability) should change.
CPU_ratio.pred2D_3D_pend.mumps_ma27.mean_mean = round(mean(1./CPU_ratio.pred2D_3D_pend.ma27_mumps.all_mean),1); % inversion so mumps wrt ma27, + means mumps slower
CPU_ratio.pred2D_3D_pend.mumps_ma27.mean_std = round(std(1./CPU_ratio.pred2D_3D_pend.ma27_mumps.all_mean),1); % inversion so mumps wrt ma27, + means mumps slower
% Here I only look at the pendulums, since different from walking simulations.
CPU_ratio.pendulum_all.ma27_mumps_r.mean_mean = round(mean(CPU_ratio.pendulum_all.ma27_mumps.mean),1); 
CPU_ratio.pendulum_all.ma27_mumps_r.mean_std = round(std(CPU_ratio.pendulum_all.ma27_mumps.mean),1); 
CPU_ratio.pendulum_all.mumps_ma27.mean_mean = round(mean(1./CPU_ratio.pendulum_all.ma27_mumps.mean),1); % inversion so mumps wrt ma27, + means mumps slower
CPU_ratio.pendulum_all.mumps_ma27.mean_std = round(std(1./CPU_ratio.pendulum_all.ma27_mumps.mean),1); % inversion so mumps wrt ma27, + means mumps slower
% Calculate ratio between ma57 and mumps
% All: PredSim 2D, TrackSim 3D, and Pendulums
CPU_ratio.pred2D_3D_pend.ma57_mumps.all = t_proc_all.pred2D_3D_pend.ma57.all./t_proc_all.pred2D_3D_pend.mumps.all;
CPU_ratio.pred2D_3D_pend.ma57_mumps.mean = nanmean(CPU_ratio.pred2D_3D_pend.ma57_mumps.all,1);
CPU_ratio.pred2D_3D_pend.ma57_mumps.std = nanstd(CPU_ratio.pred2D_3D_pend.ma57_mumps.all,[],1);
% PredSim 2D
CPU_ratio.pred2D.ma57_mumps.all = t_proc_all.pred2D.ma57.all./t_proc_all.pred2D.mumps.all;
CPU_ratio.pred2D.ma57_mumps.mean = nanmean(CPU_ratio.pred2D.ma57_mumps.all,1);
CPU_ratio.pred2D.ma57_mumps.std = nanstd(CPU_ratio.pred2D.ma57_mumps.all,[],1);
CPU_ratio.pred2D.ma57_mumps_r.mean = round(nanmean(CPU_ratio.pred2D.ma57_mumps.all,1),1);
CPU_ratio.pred2D.ma57_mumps_r.std = round(nanstd(CPU_ratio.pred2D.ma57_mumps.all,[],1),1);
CPU_ratio.pred2D.mumps_ma57.mean = round(nanmean(1./CPU_ratio.pred2D.ma57_mumps.all,1),1); % inversion so mumps wrt ma57, + means mumps slower
CPU_ratio.pred2D.mumps_ma57.std = round(nanstd(1./CPU_ratio.pred2D.ma57_mumps.all,[],1),1); % inversion so mumps wrt ma57, + means mumps slower
% TrackSim 3D
CPU_ratio.track3D.ma57_mumps.all = t_proc_all.track3D.ma57.all./t_proc_all.track3D.mumps.all;
CPU_ratio.track3D.ma57_mumps.mean = nanmean(CPU_ratio.track3D.ma57_mumps.all,1);
CPU_ratio.track3D.ma57_mumps.std = nanstd(CPU_ratio.track3D.ma57_mumps.all,[],1);
CPU_ratio.track3D.ma57_mumps_r.mean = round(nanmean(CPU_ratio.track3D.ma57_mumps.all,1),1);
CPU_ratio.track3D.ma57_mumps_r.std = round(nanstd(CPU_ratio.track3D.ma57_mumps.all,[],1),1);
CPU_ratio.track3D.mumps_ma57.mean = nanmean(1./CPU_ratio.track3D.ma57_mumps.all,1); % inversion so mumps wrt ma57, + means mumps slower
CPU_ratio.track3D.mumps_ma57.std = nanstd(1./CPU_ratio.track3D.ma57_mumps.all,[],1); % inversion so mumps wrt ma57, + means mumps slower
% Pendulums
for k = 2:length(ww_pend)+1
    CPU_ratio.(['pendulum',num2str(k),'dof']).ma57_mumps.all = t_proc_all.(['pendulum',num2str(k),'dof']).ma57.all./t_proc_all.(['pendulum',num2str(k),'dof']).mumps.all;
    CPU_ratio.(['pendulum',num2str(k),'dof']).ma57_mumps.mean = nanmean(CPU_ratio.(['pendulum',num2str(k),'dof']).ma57_mumps.all,1);
    CPU_ratio.(['pendulum',num2str(k),'dof']).ma57_mumps.std = nanstd(CPU_ratio.(['pendulum',num2str(k),'dof']).ma57_mumps.all,[],1);
    CPU_ratio.pendulum_all.ma57_mumps.mean(k-1) = CPU_ratio.(['pendulum',num2str(k),'dof']).ma57_mumps.mean(end);
    CPU_ratio.pendulum_all.ma57_mumps.std(k-1) = CPU_ratio.(['pendulum',num2str(k),'dof']).ma57_mumps.std(end);
end
CPU_ratio.pred2D_3D_pend.ma57_mumps.all_mean = [CPU_ratio.pendulum_all.ma57_mumps.mean,CPU_ratio.pred2D.ma57_mumps.mean(end),CPU_ratio.track3D.ma57_mumps.mean(end)];
CPU_ratio.pred2D_3D_pend.ma57_mumps.all_std = [CPU_ratio.pendulum_all.ma57_mumps.std,CPU_ratio.pred2D.ma57_mumps.std(end),CPU_ratio.track3D.ma57_mumps.std(end)];
% Numbers for paper (mean of mean): I want to give the same weight to the
% different examples.
% Here I only look at the pendulums, since different from walking simulations.
CPU_ratio.pendulum_all.ma57_mumps_r.mean_mean = round(mean(CPU_ratio.pendulum_all.ma57_mumps.mean),1); 
CPU_ratio.pendulum_all.ma57_mumps_r.mean_std = round(std(CPU_ratio.pendulum_all.ma57_mumps.mean),1); 
CPU_ratio.pendulum_all.mumps_ma57.mean_mean = round(mean(1./CPU_ratio.pendulum_all.ma57_mumps.mean),1); % inversion so mumps wrt ma57, + means mumps slower
CPU_ratio.pendulum_all.mumps_ma57.mean_std = round(std(1./CPU_ratio.pendulum_all.ma57_mumps.mean),1); % inversion so mumps wrt ma57, + means mumps slower
% Here I only look at the last two initial guesses of the 2D pred sim.
temp = CPU_ratio.pred2D.ma57_mumps.all(2:end,:);
CPU_ratio.pred2D_2IG.mumps_ma57.mean_mean = round(mean(1./temp(:,end)),1); % inversion so mumps wrt ma57, + means mumps slower
CPU_ratio.pred2D_2IG.mumps_ma57.mean_std = round(std(1./temp(:,end)),1); % inversion so mumps wrt ma57, + means mumps slower
% Calculate ratio between ma77 and mumps
% All: PredSim 2D, TrackSim 3D, and Pendulums
CPU_ratio.pred2D_3D_pend.ma77_mumps.all = t_proc_all.pred2D_3D_pend.ma77.all./t_proc_all.pred2D_3D_pend.mumps.all;
CPU_ratio.pred2D_3D_pend.ma77_mumps.mean = nanmean(CPU_ratio.pred2D_3D_pend.ma77_mumps.all,1);
CPU_ratio.pred2D_3D_pend.ma77_mumps.std = nanstd(CPU_ratio.pred2D_3D_pend.ma77_mumps.all,[],1);
% PredSim 2D
CPU_ratio.pred2D.ma77_mumps.all = t_proc_all.pred2D.ma77.all./t_proc_all.pred2D.mumps.all;
CPU_ratio.pred2D.ma77_mumps.mean = nanmean(CPU_ratio.pred2D.ma77_mumps.all,1);
CPU_ratio.pred2D.ma77_mumps.std = nanstd(CPU_ratio.pred2D.ma77_mumps.all,[],1);
CPU_ratio.pred2D.ma77_mumps_r.mean = round(nanmean(CPU_ratio.pred2D.ma77_mumps.all,1),1);
CPU_ratio.pred2D.ma77_mumps_r.std = round(nanstd(CPU_ratio.pred2D.ma77_mumps.all,[],1),1);
CPU_ratio.pred2D.mumps_ma77.mean = round(nanmean(1./CPU_ratio.pred2D.ma77_mumps.all,1),1); % inversion so mumps wrt ma77, + means mumps slower
CPU_ratio.pred2D.mumps_ma77.std = round(nanstd(1./CPU_ratio.pred2D.ma77_mumps.all,[],1),1); % inversion so mumps wrt ma77, + means mumps slower
% TrackSim 3D
CPU_ratio.track3D.ma77_mumps.all = t_proc_all.track3D.ma77.all./t_proc_all.track3D.mumps.all;
CPU_ratio.track3D.ma77_mumps.mean = nanmean(CPU_ratio.track3D.ma77_mumps.all,1);
CPU_ratio.track3D.ma77_mumps.std = nanstd(CPU_ratio.track3D.ma77_mumps.all,[],1);
% Result table: TrackSim 3D
CPU_ratio.track3D.ma77_mumps_r.mean = round(nanmean(CPU_ratio.track3D.ma77_mumps.all,1),1);
CPU_ratio.track3D.ma77_mumps_r.std = round(nanstd(CPU_ratio.track3D.ma77_mumps.all,[],1),1);
CPU_ratio.track3D.mumps_ma77.mean = round(nanmean(1./CPU_ratio.track3D.ma77_mumps.all,1),1); % inversion so mumps wrt ma77, + means mumps slower
CPU_ratio.track3D.mumps_ma77.std = round(nanstd(1./CPU_ratio.track3D.ma77_mumps.all,[],1),1); % inversion so mumps wrt ma77, + means mumps slower
% Pendulums
for k = 2:length(ww_pend)+1
    CPU_ratio.(['pendulum',num2str(k),'dof']).ma77_mumps.all = t_proc_all.(['pendulum',num2str(k),'dof']).ma77.all./t_proc_all.(['pendulum',num2str(k),'dof']).mumps.all;
    CPU_ratio.(['pendulum',num2str(k),'dof']).ma77_mumps.mean = nanmean(CPU_ratio.(['pendulum',num2str(k),'dof']).ma77_mumps.all,1);
    CPU_ratio.(['pendulum',num2str(k),'dof']).ma77_mumps.std = nanstd(CPU_ratio.(['pendulum',num2str(k),'dof']).ma77_mumps.all,[],1);
    CPU_ratio.pendulum_all.ma77_mumps.mean(k-1) = CPU_ratio.(['pendulum',num2str(k),'dof']).ma77_mumps.mean(end);
    CPU_ratio.pendulum_all.ma77_mumps.std(k-1) = CPU_ratio.(['pendulum',num2str(k),'dof']).ma77_mumps.std(end);
end
CPU_ratio.pred2D_3D_pend.ma77_mumps.all_mean = [CPU_ratio.pendulum_all.ma77_mumps.mean,CPU_ratio.pred2D.ma77_mumps.mean(end),CPU_ratio.track3D.ma77_mumps.mean(end)];
CPU_ratio.pred2D_3D_pend.ma77_mumps.all_std = [CPU_ratio.pendulum_all.ma77_mumps.std,CPU_ratio.pred2D.ma77_mumps.std(end),CPU_ratio.track3D.ma77_mumps.std(end)];
% Numbers for paper (mean of mean): I want to give the same weight to the
% different examples.
% Here I only look at the pendulums, since different from walking simulations.
CPU_ratio.pendulum_all.ma77_mumps.mean_mean = round(mean(CPU_ratio.pendulum_all.ma77_mumps.mean),1); 
CPU_ratio.pendulum_all.ma77_mumps.mean_std = round(std(CPU_ratio.pendulum_all.ma77_mumps.mean),1);
CPU_ratio.pendulum_all.mumps_ma77.mean_mean = round(mean(1./CPU_ratio.pendulum_all.ma77_mumps.mean),1); % inversion so mumps wrt ma77, + means mumps slower
CPU_ratio.pendulum_all.mumps_ma77.mean_std = round(std(1./CPU_ratio.pendulum_all.ma77_mumps.mean),1); % inversion so mumps wrt ma77, + means mumps slower
% Here I only look at the tracking simulations
temp = CPU_ratio.track3D.ma77_mumps.all;
CPU_ratio.track3D_2IG.mumps_ma77.mean_mean = round(nanmean(1./temp(:,end)),2); % inversion so mumps wrt ma77, + means mumps slower
CPU_ratio.track3D_2IG.mumps_ma77.mean_std = round(nanstd(1./temp(:,end)),2); % inversion so mumps wrt ma77, + means mumps slower
% Calculate ratio between ma86 and mumps
% All: PredSim 2D, TrackSim 3D, and Pendulums
CPU_ratio.pred2D_3D_pend.ma86_mumps.all = t_proc_all.pred2D_3D_pend.ma86.all./t_proc_all.pred2D_3D_pend.mumps.all;
CPU_ratio.pred2D_3D_pend.ma86_mumps.mean = nanmean(CPU_ratio.pred2D_3D_pend.ma86_mumps.all,1);
CPU_ratio.pred2D_3D_pend.ma86_mumps.std = nanstd(CPU_ratio.pred2D_3D_pend.ma86_mumps.all,[],1);
% PredSim 2D
CPU_ratio.pred2D.ma86_mumps.all = t_proc_all.pred2D.ma86.all./t_proc_all.pred2D.mumps.all;
CPU_ratio.pred2D.ma86_mumps.mean = nanmean(CPU_ratio.pred2D.ma86_mumps.all,1);
CPU_ratio.pred2D.ma86_mumps.std = nanstd(CPU_ratio.pred2D.ma86_mumps.all,[],1);
CPU_ratio.pred2D.ma86_mumps_r.mean = round(nanmean(CPU_ratio.pred2D.ma86_mumps.all,1),1);
CPU_ratio.pred2D.ma86_mumps_r.std = round(nanstd(CPU_ratio.pred2D.ma86_mumps.all,[],1),1);
CPU_ratio.pred2D.mumps_ma86.mean = round(nanmean(1./CPU_ratio.pred2D.ma86_mumps.all,1),1); % inversion so mumps wrt ma86, + means mumps slower
CPU_ratio.pred2D.mumps_ma86.std = round(nanstd(1./CPU_ratio.pred2D.ma86_mumps.all,[],1),1); % inversion so mumps wrt ma86, + means mumps slower
% TrackSim 3D
CPU_ratio.track3D.ma86_mumps.all = t_proc_all.track3D.ma86.all./t_proc_all.track3D.mumps.all;
CPU_ratio.track3D.ma86_mumps.mean = nanmean(CPU_ratio.track3D.ma86_mumps.all,1);
CPU_ratio.track3D.ma86_mumps.std = nanstd(CPU_ratio.track3D.ma86_mumps.all,[],1);
% Result table: TrackSim 3D
CPU_ratio.track3D.ma86_mumps_r.mean = round(nanmean(CPU_ratio.track3D.ma86_mumps.all,1),1);
CPU_ratio.track3D.ma86_mumps_r.std = round(nanstd(CPU_ratio.track3D.ma86_mumps.all,[],1),1);
CPU_ratio.track3D.mumps_ma86.mean = round(nanmean(1./CPU_ratio.track3D.ma86_mumps.all,1),1); % inversion so mumps wrt ma86, + means mumps slower
CPU_ratio.track3D.mumps_ma86.std = round(nanstd(1./CPU_ratio.track3D.ma86_mumps.all,[],1),1); % inversion so mumps wrt ma86, + means mumps slower
% Pendulums
for k = 2:length(ww_pend)+1
    CPU_ratio.(['pendulum',num2str(k),'dof']).ma86_mumps.all = t_proc_all.(['pendulum',num2str(k),'dof']).ma86.all./t_proc_all.(['pendulum',num2str(k),'dof']).mumps.all;
    CPU_ratio.(['pendulum',num2str(k),'dof']).ma86_mumps.mean = nanmean(CPU_ratio.(['pendulum',num2str(k),'dof']).ma86_mumps.all,1);
    CPU_ratio.(['pendulum',num2str(k),'dof']).ma86_mumps.std = nanstd(CPU_ratio.(['pendulum',num2str(k),'dof']).ma86_mumps.all,[],1);
    CPU_ratio.pendulum_all.ma86_mumps.mean(k-1) = CPU_ratio.(['pendulum',num2str(k),'dof']).ma86_mumps.mean(end);
    CPU_ratio.pendulum_all.ma86_mumps.std(k-1) = CPU_ratio.(['pendulum',num2str(k),'dof']).ma86_mumps.std(end);
end
CPU_ratio.pred2D_3D_pend.ma86_mumps.all_mean = [CPU_ratio.pendulum_all.ma86_mumps.mean,CPU_ratio.pred2D.ma86_mumps.mean(end),CPU_ratio.track3D.ma86_mumps.mean(end)];
CPU_ratio.pred2D_3D_pend.ma86_mumps.all_std = [CPU_ratio.pendulum_all.ma86_mumps.std,CPU_ratio.pred2D.ma86_mumps.std(end),CPU_ratio.track3D.ma86_mumps.std(end)];
% Numbers for paper (mean of mean): I want to give the same weight to the
% different examples.
% Here I only look at the pendulums, since different from walking simulations.
CPU_ratio.pendulum_all.ma86_mumps_r.mean_mean = round(mean(CPU_ratio.pendulum_all.ma86_mumps.mean),1); 
CPU_ratio.pendulum_all.ma86_mumps_r.mean_std = round(std(CPU_ratio.pendulum_all.ma86_mumps.mean),1);
CPU_ratio.pendulum_all.mumps_ma86.mean_mean = round(mean(1./CPU_ratio.pendulum_all.ma86_mumps.mean),1); % inversion so mumps wrt ma86, + means mumps slower
CPU_ratio.pendulum_all.mumps_ma86.mean_std = round(std(1./CPU_ratio.pendulum_all.ma86_mumps.mean),1); % inversion so mumps wrt ma86, + means mumps slower
% Pred2D and Track3D only
CPU_ratio.pred2D_3D.ma86_mumps.all = [CPU_ratio.pred2D.ma86_mumps.all;CPU_ratio.track3D.ma86_mumps.all];
CPU_ratio.pred2D_3D.ma86_mumps.mean = nanmean(CPU_ratio.pred2D_3D.ma86_mumps.all,1);
CPU_ratio.pred2D_3D.ma86_mumps.std = nanstd(CPU_ratio.pred2D_3D.ma86_mumps.all,[],1);
% Calculate ratio between ma97 and mumps
% All: PredSim 2D, TrackSim 3D, and Pendulums
CPU_ratio.pred2D_3D_pend.ma97_mumps.all = t_proc_all.pred2D_3D_pend.ma97.all./t_proc_all.pred2D_3D_pend.mumps.all;
CPU_ratio.pred2D_3D_pend.ma97_mumps.mean = nanmean(CPU_ratio.pred2D_3D_pend.ma97_mumps.all,1);
CPU_ratio.pred2D_3D_pend.ma97_mumps.std = nanstd(CPU_ratio.pred2D_3D_pend.ma97_mumps.all,[],1);
% PredSim 2D
CPU_ratio.pred2D.ma97_mumps.all = t_proc_all.pred2D.ma97.all./t_proc_all.pred2D.mumps.all;
CPU_ratio.pred2D.ma97_mumps.mean = nanmean(CPU_ratio.pred2D.ma97_mumps.all,1);
CPU_ratio.pred2D.ma97_mumps.std = nanstd(CPU_ratio.pred2D.ma97_mumps.all,[],1);
CPU_ratio.pred2D.ma97_mumps_r.mean = round(nanmean(CPU_ratio.pred2D.ma97_mumps.all,1),1);
CPU_ratio.pred2D.ma97_mumps_r.std = round(nanstd(CPU_ratio.pred2D.ma97_mumps.all,[],1),1);
CPU_ratio.pred2D.mumps_ma97.mean = round(nanmean(1./CPU_ratio.pred2D.ma97_mumps.all,1),1); % inversion so mumps wrt ma97, + means mumps slower
CPU_ratio.pred2D.mumps_ma97.std = round(nanstd(1./CPU_ratio.pred2D.ma97_mumps.all,[],1),1); % inversion so mumps wrt ma97, + means mumps slower
% TrackSim 3D
CPU_ratio.track3D.ma97_mumps.all = t_proc_all.track3D.ma97.all./t_proc_all.track3D.mumps.all;
CPU_ratio.track3D.ma97_mumps.mean = nanmean(CPU_ratio.track3D.ma97_mumps.all,1);
CPU_ratio.track3D.ma97_mumps.std = nanstd(CPU_ratio.track3D.ma97_mumps.all,[],1);
CPU_ratio.track3D.ma97_mumps_r.mean = round(nanmean(CPU_ratio.track3D.ma97_mumps.all,1),1);
CPU_ratio.track3D.ma97_mumps_r.std = round(nanstd(CPU_ratio.track3D.ma97_mumps.all,[],1),1);
CPU_ratio.track3D.mumps_ma97.mean = round(nanmean(1./CPU_ratio.track3D.ma97_mumps.all,1),1); % inversion so mumps wrt ma97, + means mumps slower
CPU_ratio.track3D.mumps_ma97.std = round(nanstd(1./CPU_ratio.track3D.ma97_mumps.all,[],1),1); % inversion so mumps wrt ma97, + means mumps slower
% Pendulums
for k = 2:length(ww_pend)+1
    CPU_ratio.(['pendulum',num2str(k),'dof']).ma97_mumps.all = t_proc_all.(['pendulum',num2str(k),'dof']).ma97.all./t_proc_all.(['pendulum',num2str(k),'dof']).mumps.all;
    CPU_ratio.(['pendulum',num2str(k),'dof']).ma97_mumps.mean = nanmean(CPU_ratio.(['pendulum',num2str(k),'dof']).ma97_mumps.all,1);
    CPU_ratio.(['pendulum',num2str(k),'dof']).ma97_mumps.std = nanstd(CPU_ratio.(['pendulum',num2str(k),'dof']).ma97_mumps.all,[],1);
    CPU_ratio.pendulum_all.ma97_mumps.mean(k-1) = CPU_ratio.(['pendulum',num2str(k),'dof']).ma97_mumps.mean(end);
    CPU_ratio.pendulum_all.ma97_mumps.std(k-1) = CPU_ratio.(['pendulum',num2str(k),'dof']).ma97_mumps.std(end);
end
CPU_ratio.pred2D_3D_pend.ma97_mumps.all_mean = [CPU_ratio.pendulum_all.ma97_mumps.mean,CPU_ratio.pred2D.ma97_mumps.mean(end),CPU_ratio.track3D.ma97_mumps.mean(end)];
CPU_ratio.pred2D_3D_pend.ma97_mumps.all_std = [CPU_ratio.pendulum_all.ma97_mumps.std,CPU_ratio.pred2D.ma97_mumps.std(end),CPU_ratio.track3D.ma97_mumps.std(end)];
% Here I only look at the pendulums, since different from walking simulations.
CPU_ratio.pendulum_all.ma97_mumps_r.mean_mean = round(mean(CPU_ratio.pendulum_all.ma97_mumps.mean),1); 
CPU_ratio.pendulum_all.ma97_mumps_r.mean_std = round(std(CPU_ratio.pendulum_all.ma97_mumps.mean),1);

CPU_ratio.pendulum_all.mumps_ma97.mean_mean = round(mean(1./CPU_ratio.pendulum_all.ma97_mumps.mean),1); % inversion so mumps wrt ma97, + means mumps slower
CPU_ratio.pendulum_all.mumps_ma97.mean_std = round(std(1./CPU_ratio.pendulum_all.ma97_mumps.mean),1); % inversion so mumps wrt ma97, + means mumps slower

%% Differences in number of iterations between cases
% Combine results from pred sim 2D and track sim 3D
n_iter_all.pred2D_3D_pend.mumps.all = [n_iter_all.pred2D.mumps.all;n_iter_all.track3D.mumps.all];
n_iter_all.pred2D_3D_pend.ma27.all = [n_iter_all.pred2D.ma27.all;n_iter_all.track3D.ma27.all];
n_iter_all.pred2D_3D_pend.ma57.all = [n_iter_all.pred2D.ma57.all;n_iter_all.track3D.ma57.all];
n_iter_all.pred2D_3D_pend.ma77.all = [n_iter_all.pred2D.ma77.all;n_iter_all.track3D.ma77.all];
n_iter_all.pred2D_3D_pend.ma86.all = [n_iter_all.pred2D.ma86.all;n_iter_all.track3D.ma86.all];
n_iter_all.pred2D_3D_pend.ma97.all = [n_iter_all.pred2D.ma97.all;n_iter_all.track3D.ma97.all];
for k = 2:length(ww_pend)+1 
    n_iter_all.pred2D_3D_pend.mumps.all = [n_iter_all.pred2D_3D_pend.mumps.all;n_iter_all.(['pendulum',num2str(k),'dof']).mumps.all];
    n_iter_all.pred2D_3D_pend.ma27.all = [n_iter_all.pred2D_3D_pend.ma27.all;n_iter_all.(['pendulum',num2str(k),'dof']).ma27.all];
    n_iter_all.pred2D_3D_pend.ma57.all = [n_iter_all.pred2D_3D_pend.ma57.all;n_iter_all.(['pendulum',num2str(k),'dof']).ma57.all];
    n_iter_all.pred2D_3D_pend.ma77.all = [n_iter_all.pred2D_3D_pend.ma77.all;n_iter_all.(['pendulum',num2str(k),'dof']).ma77.all];
    n_iter_all.pred2D_3D_pend.ma86.all = [n_iter_all.pred2D_3D_pend.ma86.all;n_iter_all.(['pendulum',num2str(k),'dof']).ma86.all];
    n_iter_all.pred2D_3D_pend.ma97.all = [n_iter_all.pred2D_3D_pend.ma97.all;n_iter_all.(['pendulum',num2str(k),'dof']).ma97.all];
end
% Calculate ratio between ma27 and mumps
% All: PredSim 2D, TrackSim 3D, and Pendulums
iter_ratio.pred2D_3D_pend.ma27_mumps.all = n_iter_all.pred2D_3D_pend.ma27.all./n_iter_all.pred2D_3D_pend.mumps.all;
iter_ratio.pred2D_3D_pend.ma27_mumps.mean = nanmean(iter_ratio.pred2D_3D_pend.ma27_mumps.all,1);
iter_ratio.pred2D_3D_pend.ma27_mumps.std = nanstd(iter_ratio.pred2D_3D_pend.ma27_mumps.all,[],1);
% PredSim 2D
iter_ratio.pred2D.ma27_mumps.all = n_iter_all.pred2D.ma27.all./n_iter_all.pred2D.mumps.all;
iter_ratio.pred2D.ma27_mumps.mean = nanmean(iter_ratio.pred2D.ma27_mumps.all,1);
iter_ratio.pred2D.ma27_mumps.std = nanstd(iter_ratio.pred2D.ma27_mumps.all,[],1);
iter_ratio.pred2D.ma27_mumps_r.mean = round(nanmean(iter_ratio.pred2D.ma27_mumps.all,1),1);
iter_ratio.pred2D.ma27_mumps_r.std = round(nanstd(iter_ratio.pred2D.ma27_mumps.all,[],1),1);
iter_ratio.pred2D.mumps_ma27.mean = round(nanmean(1./iter_ratio.pred2D.ma27_mumps.all,1),1);% inversion so mumps wrt ma27, + means more iterations for mumps
iter_ratio.pred2D.mumps_ma27.std = round(nanstd(1./iter_ratio.pred2D.ma27_mumps.all,[],1),1);% inversion so mumps wrt ma27, + means more iterations for mumps
% TrackSim 3D
iter_ratio.track3D.ma27_mumps.all = n_iter_all.track3D.ma27.all./n_iter_all.track3D.mumps.all;
iter_ratio.track3D.ma27_mumps.mean = nanmean(iter_ratio.track3D.ma27_mumps.all,1);
iter_ratio.track3D.ma27_mumps.std = nanstd(iter_ratio.track3D.ma27_mumps.all,[],1);
% Result table: TrackSim 3D
iter_ratio.track3D.ma27_mumps_r.mean = round(nanmean(iter_ratio.track3D.ma27_mumps.all,1),1);
iter_ratio.track3D.ma27_mumps_r.std = round(nanstd(iter_ratio.track3D.ma27_mumps.all,[],1),1);
iter_ratio.track3D.mumps_ma27.mean = round(nanmean(1./iter_ratio.track3D.ma27_mumps.all,1),1);% inversion so mumps wrt ma27, + means more iterations for mumps
iter_ratio.track3D.mumps_ma27.std = round(nanstd(1./iter_ratio.track3D.ma27_mumps.all,[],1),1);% inversion so mumps wrt ma27, + means more iterations for mumps
% Pendulums
for k = 2:length(ww_pend)+1
    iter_ratio.(['pendulum',num2str(k),'dof']).ma27_mumps.all = n_iter_all.(['pendulum',num2str(k),'dof']).ma27.all./n_iter_all.(['pendulum',num2str(k),'dof']).mumps.all;
    iter_ratio.(['pendulum',num2str(k),'dof']).ma27_mumps.mean = nanmean(iter_ratio.(['pendulum',num2str(k),'dof']).ma27_mumps.all,1);
    iter_ratio.(['pendulum',num2str(k),'dof']).ma27_mumps.std = nanstd(iter_ratio.(['pendulum',num2str(k),'dof']).ma27_mumps.all,[],1);
    iter_ratio.pendulum_all.ma27_mumps.mean(k-1) = iter_ratio.(['pendulum',num2str(k),'dof']).ma27_mumps.mean;
    iter_ratio.pendulum_all.ma27_mumps.std(k-1) = iter_ratio.(['pendulum',num2str(k),'dof']).ma27_mumps.std;
end
iter_ratio.pred2D_3D_pend.ma27_mumps.all_mean = [iter_ratio.pendulum_all.ma27_mumps.mean,iter_ratio.pred2D.ma27_mumps.mean,iter_ratio.track3D.ma27_mumps.mean];
iter_ratio.pred2D_3D_pend.ma27_mumps.all_std = [iter_ratio.pendulum_all.ma27_mumps.std,iter_ratio.pred2D.ma27_mumps.std,iter_ratio.track3D.ma27_mumps.std];
% Numbers for paper (mean of mean): I want to give the same weight to the
% different examples. Taking the mean over all trials would give more
% importance to the pendulum cases although in the end the average does
% not really change. The std (variability) should change.
iter_ratio.pred2D_3D_pend.mumps_ma27.mean_mean = round(mean(1./iter_ratio.pred2D_3D_pend.ma27_mumps.all_mean),2); % inversion so mumps wrt ma27, + means more iterations for mumps
iter_ratio.pred2D_3D_pend.mumps_ma27.mean_std = round(std(1./iter_ratio.pred2D_3D_pend.ma27_mumps.all_mean),2); % inversion so mumps wrt ma27, + means more iterations for mumps
% Pendulum only
iter_ratio.pendulum_all.ma27_mumps_r.mean_mean = round(mean(iter_ratio.pendulum_all.ma27_mumps.mean),1); % inversion so mumps wrt ma27, + means more iterations for mumps
iter_ratio.pendulum_all.ma27_mumps_r.mean_std = round(std(iter_ratio.pendulum_all.ma27_mumps.mean),1); % inversion so mumps wrt ma27, + means more iterations for mumps
iter_ratio.pendulum_all.mumps_ma27.mean_mean = round(mean(1./iter_ratio.pendulum_all.ma27_mumps.mean),1); % inversion so mumps wrt ma27, + means more iterations for mumps
iter_ratio.pendulum_all.mumps_ma27.mean_std = round(std(1./iter_ratio.pendulum_all.ma27_mumps.mean),1); % inversion so mumps wrt ma27, + means more iterations for mumps
% Calculate ratio between ma57 and mumps
% All: PredSim 2D, TrackSim 3D, and Pendulums
iter_ratio.pred2D_3D_pend.ma57_mumps.all = n_iter_all.pred2D_3D_pend.ma57.all./n_iter_all.pred2D_3D_pend.mumps.all;
iter_ratio.pred2D_3D_pend.ma57_mumps.mean = nanmean(iter_ratio.pred2D_3D_pend.ma57_mumps.all,1);
iter_ratio.pred2D_3D_pend.ma57_mumps.std = nanstd(iter_ratio.pred2D_3D_pend.ma57_mumps.all,[],1);
% PredSim 2D
iter_ratio.pred2D.ma57_mumps.all = n_iter_all.pred2D.ma57.all./n_iter_all.pred2D.mumps.all;
iter_ratio.pred2D.ma57_mumps.mean = nanmean(iter_ratio.pred2D.ma57_mumps.all,1);
iter_ratio.pred2D.ma57_mumps.std = nanstd(iter_ratio.pred2D.ma57_mumps.all,[],1);
iter_ratio.pred2D.ma57_mumps_r.mean = round(nanmean(iter_ratio.pred2D.ma57_mumps.all,1),1);
iter_ratio.pred2D.ma57_mumps_r.std = round(nanstd(iter_ratio.pred2D.ma57_mumps.all,[],1),1);
iter_ratio.pred2D.mumps_ma57.mean = round(nanmean(1./iter_ratio.pred2D.ma57_mumps.all,1),1);% inversion so mumps wrt ma57, + means more iterations for mumps
iter_ratio.pred2D.mumps_ma57.std = round(nanstd(1./iter_ratio.pred2D.ma57_mumps.all,[],1),1);% inversion so mumps wrt ma57, + means more iterations for mumps
% TrackSim 3D
iter_ratio.track3D.ma57_mumps.all = n_iter_all.track3D.ma57.all./n_iter_all.track3D.mumps.all;
iter_ratio.track3D.ma57_mumps.mean = nanmean(iter_ratio.track3D.ma57_mumps.all,1);
iter_ratio.track3D.ma57_mumps.std = nanstd(iter_ratio.track3D.ma57_mumps.all,[],1);
iter_ratio.track3D.ma57_mumps_r.mean = round(nanmean(iter_ratio.track3D.ma57_mumps.all,1),1);
iter_ratio.track3D.ma57_mumps_r.std = round(nanstd(iter_ratio.track3D.ma57_mumps.all,[],1),1);
iter_ratio.track3D.mumps_ma57.mean = round(nanmean(1./iter_ratio.track3D.ma57_mumps.all,1),1);% inversion so mumps wrt ma57, + means more iterations for mumps
iter_ratio.track3D.mumps_ma57.std = round(nanstd(1./iter_ratio.track3D.ma57_mumps.all,[],1),1);% inversion so mumps wrt ma57, + means more iterations for mumps
% Pendulums
for k = 2:length(ww_pend)+1
    iter_ratio.(['pendulum',num2str(k),'dof']).ma57_mumps.all = n_iter_all.(['pendulum',num2str(k),'dof']).ma57.all./n_iter_all.(['pendulum',num2str(k),'dof']).mumps.all;
    iter_ratio.(['pendulum',num2str(k),'dof']).ma57_mumps.mean = nanmean(iter_ratio.(['pendulum',num2str(k),'dof']).ma57_mumps.all,1);
    iter_ratio.(['pendulum',num2str(k),'dof']).ma57_mumps.std = nanstd(iter_ratio.(['pendulum',num2str(k),'dof']).ma57_mumps.all,[],1);
    iter_ratio.pendulum_all.ma57_mumps.mean(k-1) = iter_ratio.(['pendulum',num2str(k),'dof']).ma57_mumps.mean;
    iter_ratio.pendulum_all.ma57_mumps.std(k-1) = iter_ratio.(['pendulum',num2str(k),'dof']).ma57_mumps.std;
end
iter_ratio.pred2D_3D_pend.ma57_mumps.all_mean = [iter_ratio.pendulum_all.ma57_mumps.mean,iter_ratio.pred2D.ma57_mumps.mean,iter_ratio.track3D.ma57_mumps.mean];
iter_ratio.pred2D_3D_pend.ma57_mumps.all_std = [iter_ratio.pendulum_all.ma57_mumps.std,iter_ratio.pred2D.ma57_mumps.std,iter_ratio.track3D.ma57_mumps.std];
% Numbers for paper (mean of mean): I want to give the same weight to the
% different examples. Here I only look at the pendulums, since it is very
% different for the walking simulations.
iter_ratio.pendulum_all.ma57_mumps_r.mean_mean = round(mean(iter_ratio.pendulum_all.ma57_mumps.mean),1); 
iter_ratio.pendulum_all.ma57_mumps_r.mean_std = round(std(iter_ratio.pendulum_all.ma57_mumps.mean),1);
iter_ratio.pendulum_all.mumps_ma57.mean_mean = round(mean(1./iter_ratio.pendulum_all.ma57_mumps.mean),1); % inversion so mumps wrt ma57, + means more iterations for mumps
iter_ratio.pendulum_all.mumps_ma57.mean_std = round(std(1./iter_ratio.pendulum_all.ma57_mumps.mean),1); % inversion so mumps wrt ma57, + means more iterations for mumps
% Here I only look at two IG from 2D pred sim
temp = iter_ratio.pred2D.ma57_mumps.all(2:end,:);
iter_ratio.pred2D_2IG.ma57_mumps.mean_mean = round(mean(temp(:,end)),1); % no inversion so ma57 wrt mumps , + means more iterations for ma57
iter_ratio.pred2D_2IG.ma57_mumps.mean_std = round(std(temp(:,end)),1); % no inversion so ma57 wrt mumps , + means more iterations for ma57
% Calculate ratio between ma77 and mumps
% All: PredSim 2D, TrackSim 3D, and Pendulums
iter_ratio.pred2D_3D_pend.ma77_mumps.all = n_iter_all.pred2D_3D_pend.ma77.all./n_iter_all.pred2D_3D_pend.mumps.all;
iter_ratio.pred2D_3D_pend.ma77_mumps.mean = nanmean(iter_ratio.pred2D_3D_pend.ma77_mumps.all,1);
iter_ratio.pred2D_3D_pend.ma77_mumps.std = nanstd(iter_ratio.pred2D_3D_pend.ma77_mumps.all,[],1);
% PredSim 2D
iter_ratio.pred2D.ma77_mumps.all = n_iter_all.pred2D.ma77.all./n_iter_all.pred2D.mumps.all;
iter_ratio.pred2D.ma77_mumps.mean = nanmean(iter_ratio.pred2D.ma77_mumps.all,1);
iter_ratio.pred2D.ma77_mumps.std = nanstd(iter_ratio.pred2D.ma77_mumps.all,[],1);
iter_ratio.pred2D.ma77_mumps_r.mean = round(nanmean(iter_ratio.pred2D.ma77_mumps.all,1),1);
iter_ratio.pred2D.ma77_mumps_r.std = round(nanstd(iter_ratio.pred2D.ma77_mumps.all,[],1),1);
iter_ratio.pred2D.mumps_ma77.mean = round(nanmean(1./iter_ratio.pred2D.ma77_mumps.all,1),1);% inversion so mumps wrt ma77, + means more iterations for mumps
iter_ratio.pred2D.mumps_ma77.std = round(nanstd(1./iter_ratio.pred2D.ma77_mumps.all,[],1),1);% inversion so mumps wrt ma77, + means more iterations for mumps
% TrackSim 3D
iter_ratio.track3D.ma77_mumps.all = n_iter_all.track3D.ma77.all./n_iter_all.track3D.mumps.all;
iter_ratio.track3D.ma77_mumps.mean = nanmean(iter_ratio.track3D.ma77_mumps.all,1);
iter_ratio.track3D.ma77_mumps.std = nanstd(iter_ratio.track3D.ma77_mumps.all,[],1);
% Result table: TrackSim 3D
iter_ratio.track3D.ma77_mumps_r.mean = round(nanmean(iter_ratio.track3D.ma77_mumps.all,1),1);
iter_ratio.track3D.ma77_mumps_r.std = round(nanstd(iter_ratio.track3D.ma77_mumps.all,[],1),1);
iter_ratio.track3D.mumps_ma77.mean = round(nanmean(1./iter_ratio.track3D.ma77_mumps.all,1),1);% inversion so mumps wrt ma77, + means more iterations for mumps
iter_ratio.track3D.mumps_ma77.std = round(nanstd(1./iter_ratio.track3D.ma77_mumps.all,[],1),1);% inversion so mumps wrt ma77, + means more iterations for mumps
% Pendulums
for k = 2:length(ww_pend)+1
    iter_ratio.(['pendulum',num2str(k),'dof']).ma77_mumps.all = n_iter_all.(['pendulum',num2str(k),'dof']).ma77.all./n_iter_all.(['pendulum',num2str(k),'dof']).mumps.all;
    iter_ratio.(['pendulum',num2str(k),'dof']).ma77_mumps.mean = nanmean(iter_ratio.(['pendulum',num2str(k),'dof']).ma77_mumps.all,1);
    iter_ratio.(['pendulum',num2str(k),'dof']).ma77_mumps.std = nanstd(iter_ratio.(['pendulum',num2str(k),'dof']).ma77_mumps.all,[],1);
    iter_ratio.pendulum_all.ma77_mumps.mean(k-1) = iter_ratio.(['pendulum',num2str(k),'dof']).ma77_mumps.mean;
    iter_ratio.pendulum_all.ma77_mumps.std(k-1) = iter_ratio.(['pendulum',num2str(k),'dof']).ma77_mumps.std;
end
iter_ratio.pred2D_3D_pend.ma77_mumps.all_mean = [iter_ratio.pendulum_all.ma77_mumps.mean,iter_ratio.pred2D.ma77_mumps.mean,iter_ratio.track3D.ma77_mumps.mean];
iter_ratio.pred2D_3D_pend.ma77_mumps.all_std = [iter_ratio.pendulum_all.ma77_mumps.std,iter_ratio.pred2D.ma77_mumps.std,iter_ratio.track3D.ma77_mumps.std];
% Numbers for paper (mean of mean): I want to give the same weight to the
% different examples. 
% Here I only look at the pendulums, since different from walking simulations.
iter_ratio.pendulum_all.ma77_mumps.mean_mean = round(mean(iter_ratio.pendulum_all.ma77_mumps.mean),1); % no inversion so ma77 wrt mumps , + means more iterations for ma77
iter_ratio.pendulum_all.ma77_mumps.mean_std = round(std(iter_ratio.pendulum_all.ma77_mumps.mean),1); % no inversion so ma77 wrt mumps , + means more iterations for ma77
% Pendulum only
iter_ratio.pendulum_all.ma77_mumps_r.mean_mean = round(mean(iter_ratio.pendulum_all.ma77_mumps.mean),1); 
iter_ratio.pendulum_all.ma77_mumps_r.mean_std = round(std(iter_ratio.pendulum_all.ma77_mumps.mean),1);
iter_ratio.pendulum_all.mumps_ma77.mean_mean = round(mean(1./iter_ratio.pendulum_all.ma77_mumps.mean),1); % inversion so mumps wrt ma77, + means more iterations for mumps
iter_ratio.pendulum_all.mumps_ma77.mean_std = round(std(1./iter_ratio.pendulum_all.ma77_mumps.mean),1); % inversion so mumps wrt ma77, + means more iterations for mumps
% Here I only look at the tracking simulations
temp = iter_ratio.track3D.ma77_mumps.all;
iter_ratio.track3D_2IG.mumps_ma77.mean_mean = round(nanmean(1./temp(:,end)),1); % inversion so mumps wrt ma77, + means more iterations for mumps
iter_ratio.track3D_2IG.mumps_ma77.mean_std = round(nanstd(1./temp(:,end)),1); % inversion so mumps wrt ma77, + means more iterations for mumps
% Calculate ratio between ma86 and mumps
% All: PredSim 2D, TrackSim 3D, and Pendulums
iter_ratio.pred2D_3D_pend.ma86_mumps.all = n_iter_all.pred2D_3D_pend.ma86.all./n_iter_all.pred2D_3D_pend.mumps.all;
iter_ratio.pred2D_3D_pend.ma86_mumps.mean = nanmean(iter_ratio.pred2D_3D_pend.ma86_mumps.all,1);
iter_ratio.pred2D_3D_pend.ma86_mumps.std = nanstd(iter_ratio.pred2D_3D_pend.ma86_mumps.all,[],1);
% PredSim 2D
iter_ratio.pred2D.ma86_mumps.all = n_iter_all.pred2D.ma86.all./n_iter_all.pred2D.mumps.all;
iter_ratio.pred2D.ma86_mumps.mean = nanmean(iter_ratio.pred2D.ma86_mumps.all,1);
iter_ratio.pred2D.ma86_mumps.std = nanstd(iter_ratio.pred2D.ma86_mumps.all,[],1);
iter_ratio.pred2D.ma86_mumps_r.mean = round(nanmean(iter_ratio.pred2D.ma86_mumps.all,1),1);
iter_ratio.pred2D.ma86_mumps_r.std = round(nanstd(iter_ratio.pred2D.ma86_mumps.all,[],1),1);
iter_ratio.pred2D.mumps_ma86.mean = round(nanmean(1./iter_ratio.pred2D.ma86_mumps.all,1),1);% inversion so mumps wrt ma86, + means more iterations for mumps
iter_ratio.pred2D.mumps_ma86.std = round(nanstd(1./iter_ratio.pred2D.ma86_mumps.all,[],1),1);% inversion so mumps wrt ma86, + means more iterations for mumps
% TrackSim 3D
iter_ratio.track3D.ma86_mumps.all = n_iter_all.track3D.ma86.all./n_iter_all.track3D.mumps.all;
iter_ratio.track3D.ma86_mumps.mean = nanmean(iter_ratio.track3D.ma86_mumps.all,1);
iter_ratio.track3D.ma86_mumps.std = nanstd(iter_ratio.track3D.ma86_mumps.all,[],1);
% Result table: TrackSim 3D
iter_ratio.track3D.ma86_mumps_r.mean = round(nanmean(iter_ratio.track3D.ma86_mumps.all,1),1);
iter_ratio.track3D.ma86_mumps_r.std = round(nanstd(iter_ratio.track3D.ma86_mumps.all,[],1),1);
iter_ratio.track3D.mumps_ma86.mean = round(nanmean(1./iter_ratio.track3D.ma86_mumps.all,1),1);% inversion so mumps wrt ma86, + means more iterations for mumps
iter_ratio.track3D.mumps_ma86.std = round(nanstd(1./iter_ratio.track3D.ma86_mumps.all,[],1),1);% inversion so mumps wrt ma86, + means more iterations for mumps
% Pendulums
for k = 2:length(ww_pend)+1
    iter_ratio.(['pendulum',num2str(k),'dof']).ma86_mumps.all = n_iter_all.(['pendulum',num2str(k),'dof']).ma86.all./n_iter_all.(['pendulum',num2str(k),'dof']).mumps.all;
    iter_ratio.(['pendulum',num2str(k),'dof']).ma86_mumps.mean = nanmean(iter_ratio.(['pendulum',num2str(k),'dof']).ma86_mumps.all,1);
    iter_ratio.(['pendulum',num2str(k),'dof']).ma86_mumps.std = nanstd(iter_ratio.(['pendulum',num2str(k),'dof']).ma86_mumps.all,[],1);
    iter_ratio.pendulum_all.ma86_mumps.mean(k-1) = iter_ratio.(['pendulum',num2str(k),'dof']).ma86_mumps.mean;
    iter_ratio.pendulum_all.ma86_mumps.std(k-1) = iter_ratio.(['pendulum',num2str(k),'dof']).ma86_mumps.std;
end
iter_ratio.pred2D_3D_pend.ma86_mumps.all_mean = [iter_ratio.pendulum_all.ma86_mumps.mean,iter_ratio.pred2D.ma86_mumps.mean,iter_ratio.track3D.ma86_mumps.mean];
iter_ratio.pred2D_3D_pend.ma86_mumps.all_std = [iter_ratio.pendulum_all.ma86_mumps.std,iter_ratio.pred2D.ma86_mumps.std,iter_ratio.track3D.ma86_mumps.std];
% Pendulum only
iter_ratio.pendulum_all.ma86_mumps_r.mean_mean = round(mean(iter_ratio.pendulum_all.ma86_mumps.mean),1);
iter_ratio.pendulum_all.ma86_mumps_r.mean_std = round(std(iter_ratio.pendulum_all.ma86_mumps.mean),1);
iter_ratio.pendulum_all.mumps_ma86.mean_mean = round(mean(1./iter_ratio.pendulum_all.ma86_mumps.mean),1); % inversion so mumps wrt ma86, + means more iterations for mumps
iter_ratio.pendulum_all.mumps_ma86.mean_std = round(std(1./iter_ratio.pendulum_all.ma86_mumps.mean),1); % inversion so mumps wrt ma86, + means more iterations for mumps
% Pred2D and Track3D only
iter_ratio.pred2D_3D.ma86_mumps.all = [iter_ratio.pred2D.ma86_mumps.all;iter_ratio.track3D.ma86_mumps.all];
iter_ratio.pred2D_3D.ma86_mumps.mean = nanmean(iter_ratio.pred2D_3D.ma86_mumps.all,1); % no inversion so ma86 wrt mumps , + means more iterations for ma86
iter_ratio.pred2D_3D.ma86_mumps.std = nanstd(iter_ratio.pred2D_3D.ma86_mumps.all,[],1); % no inversion so ma86 wrt mumps , + means more iterations for ma86
% Calculate ratio between ma97 and mumps
% All: PredSim 2D, TrackSim 3D, and Pendulums
iter_ratio.pred2D_3D_pend.ma97_mumps.all = n_iter_all.pred2D_3D_pend.ma97.all./n_iter_all.pred2D_3D_pend.mumps.all;
iter_ratio.pred2D_3D_pend.ma97_mumps.mean = nanmean(iter_ratio.pred2D_3D_pend.ma97_mumps.all,1);
iter_ratio.pred2D_3D_pend.ma97_mumps.std = nanstd(iter_ratio.pred2D_3D_pend.ma97_mumps.all,[],1);
% PredSim 2D
iter_ratio.pred2D.ma97_mumps.all = n_iter_all.pred2D.ma97.all./n_iter_all.pred2D.mumps.all;
iter_ratio.pred2D.ma97_mumps.mean = nanmean(iter_ratio.pred2D.ma97_mumps.all,1);
iter_ratio.pred2D.ma97_mumps.std = nanstd(iter_ratio.pred2D.ma97_mumps.all,[],1);
iter_ratio.pred2D.ma97_mumps_r.mean = round(nanmean(iter_ratio.pred2D.ma97_mumps.all,1),1);
iter_ratio.pred2D.ma97_mumps_r.std = round(nanstd(iter_ratio.pred2D.ma97_mumps.all,[],1),1);
iter_ratio.pred2D.mumps_ma97.mean = round(nanmean(1./iter_ratio.pred2D.ma97_mumps.all,1),1);% inversion so mumps wrt ma97, + means more iterations for mumps
iter_ratio.pred2D.mumps_ma97.std = round(nanstd(1./iter_ratio.pred2D.ma97_mumps.all,[],1),1);% inversion so mumps wrt ma97, + means more iterations for mumps
% TrackSim 3D
iter_ratio.track3D.ma97_mumps.all = n_iter_all.track3D.ma97.all./n_iter_all.track3D.mumps.all;
iter_ratio.track3D.ma97_mumps.mean = nanmean(iter_ratio.track3D.ma97_mumps.all,1);
iter_ratio.track3D.ma97_mumps.std = nanstd(iter_ratio.track3D.ma97_mumps.all,[],1);
iter_ratio.track3D.ma97_mumps_r.mean = round(nanmean(iter_ratio.track3D.ma97_mumps.all,1),1);
iter_ratio.track3D.ma97_mumps_r.std = round(nanstd(iter_ratio.track3D.ma97_mumps.all,[],1),1);
iter_ratio.track3D.mumps_ma97.mean = round(nanmean(1./iter_ratio.track3D.ma97_mumps.all,1),1);% inversion so mumps wrt ma97, + means more iterations for mumps
iter_ratio.track3D.mumps_ma97.std = round(nanstd(1./iter_ratio.track3D.ma97_mumps.all,[],1),1);% inversion so mumps wrt ma97, + means more iterations for mumps
% Pendulums
for k = 2:length(ww_pend)+1
    iter_ratio.(['pendulum',num2str(k),'dof']).ma97_mumps.all = n_iter_all.(['pendulum',num2str(k),'dof']).ma97.all./n_iter_all.(['pendulum',num2str(k),'dof']).mumps.all;
    iter_ratio.(['pendulum',num2str(k),'dof']).ma97_mumps.mean = nanmean(iter_ratio.(['pendulum',num2str(k),'dof']).ma97_mumps.all,1);
    iter_ratio.(['pendulum',num2str(k),'dof']).ma97_mumps.std = nanstd(iter_ratio.(['pendulum',num2str(k),'dof']).ma97_mumps.all,[],1);
    iter_ratio.pendulum_all.ma97_mumps.mean(k-1) = iter_ratio.(['pendulum',num2str(k),'dof']).ma97_mumps.mean;
    iter_ratio.pendulum_all.ma97_mumps.std(k-1) = iter_ratio.(['pendulum',num2str(k),'dof']).ma97_mumps.std;
end
iter_ratio.pred2D_3D_pend.ma97_mumps.all_mean = [iter_ratio.pendulum_all.ma97_mumps.mean,iter_ratio.pred2D.ma97_mumps.mean,iter_ratio.track3D.ma97_mumps.mean];
iter_ratio.pred2D_3D_pend.ma97_mumps.all_std = [iter_ratio.pendulum_all.ma97_mumps.std,iter_ratio.pred2D.ma97_mumps.std,iter_ratio.track3D.ma97_mumps.std];
% Pendulum only
iter_ratio.pendulum_all.ma97_mumps_r.mean_mean = round(mean(iter_ratio.pendulum_all.ma97_mumps.mean),1); 
iter_ratio.pendulum_all.ma97_mumps_r.mean_std = round(std(iter_ratio.pendulum_all.ma97_mumps.mean),1);
iter_ratio.pendulum_all.mumps_ma97.mean_mean = round(mean(1./iter_ratio.pendulum_all.ma97_mumps.mean),1); % inversion so mumps wrt ma97, + means more iterations for mumps
iter_ratio.pendulum_all.mumps_ma97.mean_std = round(std(1./iter_ratio.pendulum_all.ma97_mumps.mean),1); % inversion so mumps wrt ma97, + means more iterations for mumps
% Avergae over all pendulums and all linear solvers
iter_ratio.pendulum_all.all_mumps.all = [iter_ratio.pendulum_all.ma27_mumps.mean;iter_ratio.pendulum_all.ma57_mumps.mean;iter_ratio.pendulum_all.ma77_mumps.mean;...
    iter_ratio.pendulum_all.ma86_mumps.mean;iter_ratio.pendulum_all.ma97_mumps.mean];
iter_ratio.pendulum_all.all_mumps.mean2 = mean2(iter_ratio.pendulum_all.all_mumps.all);
iter_ratio.pendulum_all.all_mumps.std2 = std2(iter_ratio.pendulum_all.all_mumps.all);

%% CPU time per iteration
CPU_iter_all.pred2D.mumps.all = t_proc_all.pred2D.mumps.all(:,end)./n_iter_all.pred2D.mumps.all;
CPU_iter_all.pred2D.mumps.mean = nanmean(CPU_iter_all.pred2D.mumps.all,1);
CPU_iter_all.pred2D.mumps.std = nanstd(CPU_iter_all.pred2D.mumps.all,[],1);
CPU_iter_all.pred2D.ma27.all = t_proc_all.pred2D.ma27.all(:,end)./n_iter_all.pred2D.ma27.all;
CPU_iter_all.pred2D.ma27.mean = nanmean(CPU_iter_all.pred2D.ma27.all,1);
CPU_iter_all.pred2D.ma27.std = nanstd(CPU_iter_all.pred2D.ma27.all,[],1);

CPU_iter_ratio.pred2D.ma27_mumps.all = CPU_iter_all.pred2D.ma27.all./CPU_iter_all.pred2D.mumps.all;
CPU_iter_ratio.pred2D.mumps_ma27.mean = round(nanmean(1./CPU_iter_ratio.pred2D.ma27_mumps.all,1),1); % inversion so mumps wrt ma27, + means mumps slower
CPU_iter_ratio.pred2D.mumps_ma27.std = round(nanstd(1./CPU_iter_ratio.pred2D.ma27_mumps.all,[],1),1); % inversion so mumps wrt ma27,  + means mumps slower

%% Difference in CPU time per iterations
% Combine results from PredSim 2D and TrackSim 3D and Pendulums
t_iter_all.pred2D_3D_pend.mumps.all = [t_iter_all.pred2D.mumps.all;t_iter_all.track3D.mumps.all];
t_iter_all.pred2D_3D_pend.ma27.all = [t_iter_all.pred2D.ma27.all;t_iter_all.track3D.ma27.all];
t_iter_all.pred2D_3D_pend.ma57.all = [t_iter_all.pred2D.ma57.all;t_iter_all.track3D.ma57.all];
t_iter_all.pred2D_3D_pend.ma77.all = [t_iter_all.pred2D.ma77.all;t_iter_all.track3D.ma77.all];
t_iter_all.pred2D_3D_pend.ma86.all = [t_iter_all.pred2D.ma86.all;t_iter_all.track3D.ma86.all];
t_iter_all.pred2D_3D_pend.ma97.all = [t_iter_all.pred2D.ma97.all;t_iter_all.track3D.ma97.all];
for k = 2:length(ww_pend)+1 
    t_iter_all.pred2D_3D_pend.mumps.all = [t_iter_all.pred2D_3D_pend.mumps.all;t_iter_all.(['pendulum',num2str(k),'dof']).mumps.all];
    t_iter_all.pred2D_3D_pend.ma27.all = [t_iter_all.pred2D_3D_pend.ma27.all;t_iter_all.(['pendulum',num2str(k),'dof']).ma27.all];
    t_iter_all.pred2D_3D_pend.ma57.all = [t_iter_all.pred2D_3D_pend.ma57.all;t_iter_all.(['pendulum',num2str(k),'dof']).ma57.all];
    t_iter_all.pred2D_3D_pend.ma77.all = [t_iter_all.pred2D_3D_pend.ma77.all;t_iter_all.(['pendulum',num2str(k),'dof']).ma77.all];
    t_iter_all.pred2D_3D_pend.ma86.all = [t_iter_all.pred2D_3D_pend.ma86.all;t_iter_all.(['pendulum',num2str(k),'dof']).ma86.all];
    t_iter_all.pred2D_3D_pend.ma97.all = [t_iter_all.pred2D_3D_pend.ma97.all;t_iter_all.(['pendulum',num2str(k),'dof']).ma97.all];
end
% Calculate ratio between ma27 and mumps
% All: PredSim 2D, TrackSim 3D, and Pendulums
t_iter_ratio.pred2D_3D_pend.ma27_mumps.all = t_iter_all.pred2D_3D_pend.ma27.all./t_iter_all.pred2D_3D_pend.mumps.all;
t_iter_ratio.pred2D_3D_pend.ma27_mumps.mean = nanmean(t_iter_ratio.pred2D_3D_pend.ma27_mumps.all,1);
t_iter_ratio.pred2D_3D_pend.ma27_mumps.std = nanstd(t_iter_ratio.pred2D_3D_pend.ma27_mumps.all,[],1);
% PredSim 2D
t_iter_ratio.pred2D.ma27_mumps.all = t_iter_all.pred2D.ma27.all./t_iter_all.pred2D.mumps.all;
t_iter_ratio.pred2D.ma27_mumps.mean = nanmean(t_iter_ratio.pred2D.ma27_mumps.all,1);
t_iter_ratio.pred2D.ma27_mumps.std = nanstd(t_iter_ratio.pred2D.ma27_mumps.all,[],1);
t_iter_ratio.pred2D.ma27_mumps_r.mean = round(nanmean(t_iter_ratio.pred2D.ma27_mumps.all,1),1);
t_iter_ratio.pred2D.ma27_mumps_r.std = round(nanstd(t_iter_ratio.pred2D.ma27_mumps.all,[],1),1);
t_iter_ratio.pred2D.mumps_ma27.mean = round(nanmean(1./t_iter_ratio.pred2D.ma27_mumps.all,1),1); % inversion so mumps wrt ma27, + means mumps slower
t_iter_ratio.pred2D.mumps_ma27.std = round(nanstd(1./t_iter_ratio.pred2D.ma27_mumps.all,[],1),1); % inversion so mumps wrt ma27, + means mumps slower
% TrackSim 3D
t_iter_ratio.track3D.ma27_mumps.all = t_iter_all.track3D.ma27.all./t_iter_all.track3D.mumps.all;
t_iter_ratio.track3D.ma27_mumps.mean = nanmean(t_iter_ratio.track3D.ma27_mumps.all,1); 
t_iter_ratio.track3D.ma27_mumps.std = nanstd(t_iter_ratio.track3D.ma27_mumps.all,[],1);
% Result table: TrackSim 3D
t_iter_ratio.track3D.ma27_mumps_r.mean = round(nanmean(t_iter_ratio.track3D.ma27_mumps.all,1),1); 
t_iter_ratio.track3D.ma27_mumps_r.std = round(nanstd(t_iter_ratio.track3D.ma27_mumps.all,[],1),1);
t_iter_ratio.track3D.mumps_ma27.mean = round(nanmean(1./t_iter_ratio.track3D.ma27_mumps.all,1),1); % inversion so mumps wrt ma27, + means mumps slower
t_iter_ratio.track3D.mumps_ma27.std = round(nanstd(1./t_iter_ratio.track3D.ma27_mumps.all,[],1),1); % inversion so mumps wrt ma27, + means mumps slower
% Pendulums
for k = 2:length(ww_pend)+1
    t_iter_ratio.(['pendulum',num2str(k),'dof']).ma27_mumps.all = t_iter_all.(['pendulum',num2str(k),'dof']).ma27.all./t_iter_all.(['pendulum',num2str(k),'dof']).mumps.all;
    t_iter_ratio.(['pendulum',num2str(k),'dof']).ma27_mumps.mean = nanmean(t_iter_ratio.(['pendulum',num2str(k),'dof']).ma27_mumps.all,1);
    t_iter_ratio.(['pendulum',num2str(k),'dof']).ma27_mumps.std = nanstd(t_iter_ratio.(['pendulum',num2str(k),'dof']).ma27_mumps.all,[],1);
    t_iter_ratio.pendulum_all.ma27_mumps.mean(k-1) = t_iter_ratio.(['pendulum',num2str(k),'dof']).ma27_mumps.mean(end);
    t_iter_ratio.pendulum_all.ma27_mumps.std(k-1) = t_iter_ratio.(['pendulum',num2str(k),'dof']).ma27_mumps.std(end);
end
t_iter_ratio.pred2D_3D_pend.ma27_mumps.all_mean = [t_iter_ratio.pendulum_all.ma27_mumps.mean,t_iter_ratio.pred2D.ma27_mumps.mean(end),t_iter_ratio.track3D.ma27_mumps.mean(end)];
t_iter_ratio.pred2D_3D_pend.ma27_mumps.all_std = [t_iter_ratio.pendulum_all.ma27_mumps.std,t_iter_ratio.pred2D.ma27_mumps.std(end),t_iter_ratio.track3D.ma27_mumps.std(end)];
% Numbers for paper (mean of mean): I want to give the same weight to the
% different examples. Taking the mean over all trials would give more
% importance to the pendulum cases although in the end the average does
% not really change. The std (variability) should change.
t_iter_ratio.pred2D_3D_pend.mumps_ma27.mean_mean = round(mean(1./t_iter_ratio.pred2D_3D_pend.ma27_mumps.all_mean),1); % inversion so mumps wrt ma27, + means mumps slower
t_iter_ratio.pred2D_3D_pend.mumps_ma27.mean_std = round(std(1./t_iter_ratio.pred2D_3D_pend.ma27_mumps.all_mean),1); % inversion so mumps wrt ma27, + means mumps slower
% Here I only look at the pendulums, since different from walking simulations.
t_iter_ratio.pendulum_all.ma27_mumps_r.mean_mean = round(mean(t_iter_ratio.pendulum_all.ma27_mumps.mean),1); 
t_iter_ratio.pendulum_all.ma27_mumps_r.mean_std = round(std(t_iter_ratio.pendulum_all.ma27_mumps.mean),1); 
t_iter_ratio.pendulum_all.mumps_ma27.mean_mean = round(mean(1./t_iter_ratio.pendulum_all.ma27_mumps.mean),1); % inversion so mumps wrt ma27, + means mumps slower
t_iter_ratio.pendulum_all.mumps_ma27.mean_std = round(std(1./t_iter_ratio.pendulum_all.ma27_mumps.mean),1); % inversion so mumps wrt ma27, + means mumps slower
% Calculate ratio between ma57 and mumps
% All: PredSim 2D, TrackSim 3D, and Pendulums
t_iter_ratio.pred2D_3D_pend.ma57_mumps.all = t_iter_all.pred2D_3D_pend.ma57.all./t_iter_all.pred2D_3D_pend.mumps.all;
t_iter_ratio.pred2D_3D_pend.ma57_mumps.mean = nanmean(t_iter_ratio.pred2D_3D_pend.ma57_mumps.all,1);
t_iter_ratio.pred2D_3D_pend.ma57_mumps.std = nanstd(t_iter_ratio.pred2D_3D_pend.ma57_mumps.all,[],1);
% PredSim 2D
t_iter_ratio.pred2D.ma57_mumps.all = t_iter_all.pred2D.ma57.all./t_iter_all.pred2D.mumps.all;
t_iter_ratio.pred2D.ma57_mumps.mean = nanmean(t_iter_ratio.pred2D.ma57_mumps.all,1);
t_iter_ratio.pred2D.ma57_mumps.std = nanstd(t_iter_ratio.pred2D.ma57_mumps.all,[],1);
t_iter_ratio.pred2D.ma57_mumps_r.mean = round(nanmean(t_iter_ratio.pred2D.ma57_mumps.all,1),1);
t_iter_ratio.pred2D.ma57_mumps_r.std = round(nanstd(t_iter_ratio.pred2D.ma57_mumps.all,[],1),1);
t_iter_ratio.pred2D.mumps_ma57.mean = round(nanmean(1./t_iter_ratio.pred2D.ma57_mumps.all,1),1); % inversion so mumps wrt ma57, + means mumps slower
t_iter_ratio.pred2D.mumps_ma57.std = round(nanstd(1./t_iter_ratio.pred2D.ma57_mumps.all,[],1),1); % inversion so mumps wrt ma57, + means mumps slower
% TrackSim 3D
t_iter_ratio.track3D.ma57_mumps.all = t_iter_all.track3D.ma57.all./t_iter_all.track3D.mumps.all;
t_iter_ratio.track3D.ma57_mumps.mean = nanmean(t_iter_ratio.track3D.ma57_mumps.all,1);
t_iter_ratio.track3D.ma57_mumps.std = nanstd(t_iter_ratio.track3D.ma57_mumps.all,[],1);
t_iter_ratio.track3D.ma57_mumps_r.mean = round(nanmean(t_iter_ratio.track3D.ma57_mumps.all,1),1);
t_iter_ratio.track3D.ma57_mumps_r.std = round(nanstd(t_iter_ratio.track3D.ma57_mumps.all,[],1),1);
t_iter_ratio.track3D.mumps_ma57.mean = nanmean(1./t_iter_ratio.track3D.ma57_mumps.all,1); % inversion so mumps wrt ma57, + means mumps slower
t_iter_ratio.track3D.mumps_ma57.std = nanstd(1./t_iter_ratio.track3D.ma57_mumps.all,[],1); % inversion so mumps wrt ma57, + means mumps slower
% Pendulums
for k = 2:length(ww_pend)+1
    t_iter_ratio.(['pendulum',num2str(k),'dof']).ma57_mumps.all = t_iter_all.(['pendulum',num2str(k),'dof']).ma57.all./t_iter_all.(['pendulum',num2str(k),'dof']).mumps.all;
    t_iter_ratio.(['pendulum',num2str(k),'dof']).ma57_mumps.mean = nanmean(t_iter_ratio.(['pendulum',num2str(k),'dof']).ma57_mumps.all,1);
    t_iter_ratio.(['pendulum',num2str(k),'dof']).ma57_mumps.std = nanstd(t_iter_ratio.(['pendulum',num2str(k),'dof']).ma57_mumps.all,[],1);
    t_iter_ratio.pendulum_all.ma57_mumps.mean(k-1) = t_iter_ratio.(['pendulum',num2str(k),'dof']).ma57_mumps.mean(end);
    t_iter_ratio.pendulum_all.ma57_mumps.std(k-1) = t_iter_ratio.(['pendulum',num2str(k),'dof']).ma57_mumps.std(end);
end
t_iter_ratio.pred2D_3D_pend.ma57_mumps.all_mean = [t_iter_ratio.pendulum_all.ma57_mumps.mean,t_iter_ratio.pred2D.ma57_mumps.mean(end),t_iter_ratio.track3D.ma57_mumps.mean(end)];
t_iter_ratio.pred2D_3D_pend.ma57_mumps.all_std = [t_iter_ratio.pendulum_all.ma57_mumps.std,t_iter_ratio.pred2D.ma57_mumps.std(end),t_iter_ratio.track3D.ma57_mumps.std(end)];
% Numbers for paper (mean of mean): I want to give the same weight to the
% different examples.
% Here I only look at the pendulums, since different from walking simulations.
t_iter_ratio.pendulum_all.ma57_mumps_r.mean_mean = round(mean(t_iter_ratio.pendulum_all.ma57_mumps.mean),1); 
t_iter_ratio.pendulum_all.ma57_mumps_r.mean_std = round(std(t_iter_ratio.pendulum_all.ma57_mumps.mean),1); 
t_iter_ratio.pendulum_all.mumps_ma57.mean_mean = round(mean(1./t_iter_ratio.pendulum_all.ma57_mumps.mean),1); % inversion so mumps wrt ma57, + means mumps slower
t_iter_ratio.pendulum_all.mumps_ma57.mean_std = round(std(1./t_iter_ratio.pendulum_all.ma57_mumps.mean),1); % inversion so mumps wrt ma57, + means mumps slower
% Here I only look at the last two initial guesses of the 2D pred sim.
temp = t_iter_ratio.pred2D.ma57_mumps.all(2:end,:);
t_iter_ratio.pred2D_2IG.mumps_ma57.mean_mean = round(mean(1./temp(:,end)),1); % inversion so mumps wrt ma57, + means mumps slower
t_iter_ratio.pred2D_2IG.mumps_ma57.mean_std = round(std(1./temp(:,end)),1); % inversion so mumps wrt ma57, + means mumps slower
% Calculate ratio between ma77 and mumps
% All: PredSim 2D, TrackSim 3D, and Pendulums
t_iter_ratio.pred2D_3D_pend.ma77_mumps.all = t_iter_all.pred2D_3D_pend.ma77.all./t_iter_all.pred2D_3D_pend.mumps.all;
t_iter_ratio.pred2D_3D_pend.ma77_mumps.mean = nanmean(t_iter_ratio.pred2D_3D_pend.ma77_mumps.all,1);
t_iter_ratio.pred2D_3D_pend.ma77_mumps.std = nanstd(t_iter_ratio.pred2D_3D_pend.ma77_mumps.all,[],1);
% PredSim 2D
t_iter_ratio.pred2D.ma77_mumps.all = t_iter_all.pred2D.ma77.all./t_iter_all.pred2D.mumps.all;
t_iter_ratio.pred2D.ma77_mumps.mean = nanmean(t_iter_ratio.pred2D.ma77_mumps.all,1);
t_iter_ratio.pred2D.ma77_mumps.std = nanstd(t_iter_ratio.pred2D.ma77_mumps.all,[],1);
t_iter_ratio.pred2D.ma77_mumps_r.mean = round(nanmean(t_iter_ratio.pred2D.ma77_mumps.all,1),1);
t_iter_ratio.pred2D.ma77_mumps_r.std = round(nanstd(t_iter_ratio.pred2D.ma77_mumps.all,[],1),1);
t_iter_ratio.pred2D.mumps_ma77.mean = round(nanmean(1./t_iter_ratio.pred2D.ma77_mumps.all,1),1); % inversion so mumps wrt ma77, + means mumps slower
t_iter_ratio.pred2D.mumps_ma77.std = round(nanstd(1./t_iter_ratio.pred2D.ma77_mumps.all,[],1),1); % inversion so mumps wrt ma77, + means mumps slower
% TrackSim 3D
t_iter_ratio.track3D.ma77_mumps.all = t_iter_all.track3D.ma77.all./t_iter_all.track3D.mumps.all;
t_iter_ratio.track3D.ma77_mumps.mean = nanmean(t_iter_ratio.track3D.ma77_mumps.all,1);
t_iter_ratio.track3D.ma77_mumps.std = nanstd(t_iter_ratio.track3D.ma77_mumps.all,[],1);
% Result table: TrackSim 3D
t_iter_ratio.track3D.ma77_mumps_r.mean = round(nanmean(t_iter_ratio.track3D.ma77_mumps.all,1),1);
t_iter_ratio.track3D.ma77_mumps_r.std = round(nanstd(t_iter_ratio.track3D.ma77_mumps.all,[],1),1);
t_iter_ratio.track3D.mumps_ma77.mean = round(nanmean(1./t_iter_ratio.track3D.ma77_mumps.all,1),1); % inversion so mumps wrt ma77, + means mumps slower
t_iter_ratio.track3D.mumps_ma77.std = round(nanstd(1./t_iter_ratio.track3D.ma77_mumps.all,[],1),1); % inversion so mumps wrt ma77, + means mumps slower
% Pendulums
for k = 2:length(ww_pend)+1
    t_iter_ratio.(['pendulum',num2str(k),'dof']).ma77_mumps.all = t_iter_all.(['pendulum',num2str(k),'dof']).ma77.all./t_iter_all.(['pendulum',num2str(k),'dof']).mumps.all;
    t_iter_ratio.(['pendulum',num2str(k),'dof']).ma77_mumps.mean = nanmean(t_iter_ratio.(['pendulum',num2str(k),'dof']).ma77_mumps.all,1);
    t_iter_ratio.(['pendulum',num2str(k),'dof']).ma77_mumps.std = nanstd(t_iter_ratio.(['pendulum',num2str(k),'dof']).ma77_mumps.all,[],1);
    t_iter_ratio.pendulum_all.ma77_mumps.mean(k-1) = t_iter_ratio.(['pendulum',num2str(k),'dof']).ma77_mumps.mean(end);
    t_iter_ratio.pendulum_all.ma77_mumps.std(k-1) = t_iter_ratio.(['pendulum',num2str(k),'dof']).ma77_mumps.std(end);
end
t_iter_ratio.pred2D_3D_pend.ma77_mumps.all_mean = [t_iter_ratio.pendulum_all.ma77_mumps.mean,t_iter_ratio.pred2D.ma77_mumps.mean(end),t_iter_ratio.track3D.ma77_mumps.mean(end)];
t_iter_ratio.pred2D_3D_pend.ma77_mumps.all_std = [t_iter_ratio.pendulum_all.ma77_mumps.std,t_iter_ratio.pred2D.ma77_mumps.std(end),t_iter_ratio.track3D.ma77_mumps.std(end)];
% Numbers for paper (mean of mean): I want to give the same weight to the
% different examples.
% Here I only look at the pendulums, since different from walking simulations.
t_iter_ratio.pendulum_all.ma77_mumps.mean_mean = round(mean(t_iter_ratio.pendulum_all.ma77_mumps.mean),1); 
t_iter_ratio.pendulum_all.ma77_mumps.mean_std = round(std(t_iter_ratio.pendulum_all.ma77_mumps.mean),1);
t_iter_ratio.pendulum_all.mumps_ma77.mean_mean = round(mean(1./t_iter_ratio.pendulum_all.ma77_mumps.mean),1); % inversion so mumps wrt ma77, + means mumps slower
t_iter_ratio.pendulum_all.mumps_ma77.mean_std = round(std(1./t_iter_ratio.pendulum_all.ma77_mumps.mean),1); % inversion so mumps wrt ma77, + means mumps slower
% Here I only look at the tracking simulations
temp = t_iter_ratio.track3D.ma77_mumps.all;
t_iter_ratio.track3D_2IG.mumps_ma77.mean_mean = round(nanmean(1./temp(:,end)),2); % inversion so mumps wrt ma77, + means mumps slower
t_iter_ratio.track3D_2IG.mumps_ma77.mean_std = round(nanstd(1./temp(:,end)),2); % inversion so mumps wrt ma77, + means mumps slower
% Calculate ratio between ma86 and mumps
% All: PredSim 2D, TrackSim 3D, and Pendulums
t_iter_ratio.pred2D_3D_pend.ma86_mumps.all = t_iter_all.pred2D_3D_pend.ma86.all./t_iter_all.pred2D_3D_pend.mumps.all;
t_iter_ratio.pred2D_3D_pend.ma86_mumps.mean = nanmean(t_iter_ratio.pred2D_3D_pend.ma86_mumps.all,1);
t_iter_ratio.pred2D_3D_pend.ma86_mumps.std = nanstd(t_iter_ratio.pred2D_3D_pend.ma86_mumps.all,[],1);
% PredSim 2D
t_iter_ratio.pred2D.ma86_mumps.all = t_iter_all.pred2D.ma86.all./t_iter_all.pred2D.mumps.all;
t_iter_ratio.pred2D.ma86_mumps.mean = nanmean(t_iter_ratio.pred2D.ma86_mumps.all,1);
t_iter_ratio.pred2D.ma86_mumps.std = nanstd(t_iter_ratio.pred2D.ma86_mumps.all,[],1);
t_iter_ratio.pred2D.ma86_mumps_r.mean = round(nanmean(t_iter_ratio.pred2D.ma86_mumps.all,1),1);
t_iter_ratio.pred2D.ma86_mumps_r.std = round(nanstd(t_iter_ratio.pred2D.ma86_mumps.all,[],1),1);
t_iter_ratio.pred2D.mumps_ma86.mean = round(nanmean(1./t_iter_ratio.pred2D.ma86_mumps.all,1),1); % inversion so mumps wrt ma86, + means mumps slower
t_iter_ratio.pred2D.mumps_ma86.std = round(nanstd(1./t_iter_ratio.pred2D.ma86_mumps.all,[],1),1); % inversion so mumps wrt ma86, + means mumps slower
% TrackSim 3D
t_iter_ratio.track3D.ma86_mumps.all = t_iter_all.track3D.ma86.all./t_iter_all.track3D.mumps.all;
t_iter_ratio.track3D.ma86_mumps.mean = nanmean(t_iter_ratio.track3D.ma86_mumps.all,1);
t_iter_ratio.track3D.ma86_mumps.std = nanstd(t_iter_ratio.track3D.ma86_mumps.all,[],1);
% Result table: TrackSim 3D
t_iter_ratio.track3D.ma86_mumps_r.mean = round(nanmean(t_iter_ratio.track3D.ma86_mumps.all,1),1);
t_iter_ratio.track3D.ma86_mumps_r.std = round(nanstd(t_iter_ratio.track3D.ma86_mumps.all,[],1),1);
t_iter_ratio.track3D.mumps_ma86.mean = round(nanmean(1./t_iter_ratio.track3D.ma86_mumps.all,1),1); % inversion so mumps wrt ma86, + means mumps slower
t_iter_ratio.track3D.mumps_ma86.std = round(nanstd(1./t_iter_ratio.track3D.ma86_mumps.all,[],1),1); % inversion so mumps wrt ma86, + means mumps slower
% Pendulums
for k = 2:length(ww_pend)+1
    t_iter_ratio.(['pendulum',num2str(k),'dof']).ma86_mumps.all = t_iter_all.(['pendulum',num2str(k),'dof']).ma86.all./t_iter_all.(['pendulum',num2str(k),'dof']).mumps.all;
    t_iter_ratio.(['pendulum',num2str(k),'dof']).ma86_mumps.mean = nanmean(t_iter_ratio.(['pendulum',num2str(k),'dof']).ma86_mumps.all,1);
    t_iter_ratio.(['pendulum',num2str(k),'dof']).ma86_mumps.std = nanstd(t_iter_ratio.(['pendulum',num2str(k),'dof']).ma86_mumps.all,[],1);
    t_iter_ratio.pendulum_all.ma86_mumps.mean(k-1) = t_iter_ratio.(['pendulum',num2str(k),'dof']).ma86_mumps.mean(end);
    t_iter_ratio.pendulum_all.ma86_mumps.std(k-1) = t_iter_ratio.(['pendulum',num2str(k),'dof']).ma86_mumps.std(end);
end
t_iter_ratio.pred2D_3D_pend.ma86_mumps.all_mean = [t_iter_ratio.pendulum_all.ma86_mumps.mean,t_iter_ratio.pred2D.ma86_mumps.mean(end),t_iter_ratio.track3D.ma86_mumps.mean(end)];
t_iter_ratio.pred2D_3D_pend.ma86_mumps.all_std = [t_iter_ratio.pendulum_all.ma86_mumps.std,t_iter_ratio.pred2D.ma86_mumps.std(end),t_iter_ratio.track3D.ma86_mumps.std(end)];
% Numbers for paper (mean of mean): I want to give the same weight to the
% different examples.
% Here I only look at the pendulums, since different from walking simulations.
t_iter_ratio.pendulum_all.ma86_mumps_r.mean_mean = round(mean(t_iter_ratio.pendulum_all.ma86_mumps.mean),1); 
t_iter_ratio.pendulum_all.ma86_mumps_r.mean_std = round(std(t_iter_ratio.pendulum_all.ma86_mumps.mean),1);
t_iter_ratio.pendulum_all.mumps_ma86.mean_mean = round(mean(1./t_iter_ratio.pendulum_all.ma86_mumps.mean),1); % inversion so mumps wrt ma86, + means mumps slower
t_iter_ratio.pendulum_all.mumps_ma86.mean_std = round(std(1./t_iter_ratio.pendulum_all.ma86_mumps.mean),1); % inversion so mumps wrt ma86, + means mumps slower
% Pred2D and Track3D only
t_iter_ratio.pred2D_3D.ma86_mumps.all = [t_iter_ratio.pred2D.ma86_mumps.all;t_iter_ratio.track3D.ma86_mumps.all];
t_iter_ratio.pred2D_3D.ma86_mumps.mean = nanmean(t_iter_ratio.pred2D_3D.ma86_mumps.all,1);
t_iter_ratio.pred2D_3D.ma86_mumps.std = nanstd(t_iter_ratio.pred2D_3D.ma86_mumps.all,[],1);
% Calculate ratio between ma97 and mumps
% All: PredSim 2D, TrackSim 3D, and Pendulums
t_iter_ratio.pred2D_3D_pend.ma97_mumps.all = t_iter_all.pred2D_3D_pend.ma97.all./t_iter_all.pred2D_3D_pend.mumps.all;
t_iter_ratio.pred2D_3D_pend.ma97_mumps.mean = nanmean(t_iter_ratio.pred2D_3D_pend.ma97_mumps.all,1);
t_iter_ratio.pred2D_3D_pend.ma97_mumps.std = nanstd(t_iter_ratio.pred2D_3D_pend.ma97_mumps.all,[],1);
% PredSim 2D
t_iter_ratio.pred2D.ma97_mumps.all = t_iter_all.pred2D.ma97.all./t_iter_all.pred2D.mumps.all;
t_iter_ratio.pred2D.ma97_mumps.mean = nanmean(t_iter_ratio.pred2D.ma97_mumps.all,1);
t_iter_ratio.pred2D.ma97_mumps.std = nanstd(t_iter_ratio.pred2D.ma97_mumps.all,[],1);
t_iter_ratio.pred2D.ma97_mumps_r.mean = round(nanmean(t_iter_ratio.pred2D.ma97_mumps.all,1),1);
t_iter_ratio.pred2D.ma97_mumps_r.std = round(nanstd(t_iter_ratio.pred2D.ma97_mumps.all,[],1),1);
t_iter_ratio.pred2D.mumps_ma97.mean = round(nanmean(1./t_iter_ratio.pred2D.ma97_mumps.all,1),1); % inversion so mumps wrt ma97, + means mumps slower
t_iter_ratio.pred2D.mumps_ma97.std = round(nanstd(1./t_iter_ratio.pred2D.ma97_mumps.all,[],1),1); % inversion so mumps wrt ma97, + means mumps slower
% TrackSim 3D
t_iter_ratio.track3D.ma97_mumps.all = t_iter_all.track3D.ma97.all./t_iter_all.track3D.mumps.all;
t_iter_ratio.track3D.ma97_mumps.mean = nanmean(t_iter_ratio.track3D.ma97_mumps.all,1);
t_iter_ratio.track3D.ma97_mumps.std = nanstd(t_iter_ratio.track3D.ma97_mumps.all,[],1);
t_iter_ratio.track3D.ma97_mumps_r.mean = round(nanmean(t_iter_ratio.track3D.ma97_mumps.all,1),1);
t_iter_ratio.track3D.ma97_mumps_r.std = round(nanstd(t_iter_ratio.track3D.ma97_mumps.all,[],1),1);
t_iter_ratio.track3D.mumps_ma97.mean = round(nanmean(1./t_iter_ratio.track3D.ma97_mumps.all,1),1); % inversion so mumps wrt ma97, + means mumps slower
t_iter_ratio.track3D.mumps_ma97.std = round(nanstd(1./t_iter_ratio.track3D.ma97_mumps.all,[],1),1); % inversion so mumps wrt ma97, + means mumps slower
% Pendulums
for k = 2:length(ww_pend)+1
    t_iter_ratio.(['pendulum',num2str(k),'dof']).ma97_mumps.all = t_iter_all.(['pendulum',num2str(k),'dof']).ma97.all./t_iter_all.(['pendulum',num2str(k),'dof']).mumps.all;
    t_iter_ratio.(['pendulum',num2str(k),'dof']).ma97_mumps.mean = nanmean(t_iter_ratio.(['pendulum',num2str(k),'dof']).ma97_mumps.all,1);
    t_iter_ratio.(['pendulum',num2str(k),'dof']).ma97_mumps.std = nanstd(t_iter_ratio.(['pendulum',num2str(k),'dof']).ma97_mumps.all,[],1);
    t_iter_ratio.pendulum_all.ma97_mumps.mean(k-1) = t_iter_ratio.(['pendulum',num2str(k),'dof']).ma97_mumps.mean(end);
    t_iter_ratio.pendulum_all.ma97_mumps.std(k-1) = t_iter_ratio.(['pendulum',num2str(k),'dof']).ma97_mumps.std(end);
end
t_iter_ratio.pred2D_3D_pend.ma97_mumps.all_mean = [t_iter_ratio.pendulum_all.ma97_mumps.mean,t_iter_ratio.pred2D.ma97_mumps.mean(end),t_iter_ratio.track3D.ma97_mumps.mean(end)];
t_iter_ratio.pred2D_3D_pend.ma97_mumps.all_std = [t_iter_ratio.pendulum_all.ma97_mumps.std,t_iter_ratio.pred2D.ma97_mumps.std(end),t_iter_ratio.track3D.ma97_mumps.std(end)];
% Here I only look at the pendulums, since different from walking simulations.
t_iter_ratio.pendulum_all.ma97_mumps_r.mean_mean = round(mean(t_iter_ratio.pendulum_all.ma97_mumps.mean),1); 
t_iter_ratio.pendulum_all.ma97_mumps_r.mean_std = round(std(t_iter_ratio.pendulum_all.ma97_mumps.mean),1);

t_iter_ratio.pendulum_all.mumps_ma97.mean_mean = round(mean(1./t_iter_ratio.pendulum_all.ma97_mumps.mean),1); % inversion so mumps wrt ma97, + means mumps slower
t_iter_ratio.pendulum_all.mumps_ma97.mean_std = round(std(1./t_iter_ratio.pendulum_all.ma97_mumps.mean),1); % inversion so mumps wrt ma97, + means mumps slower
%% Plots: 2 studied cases merged
label_fontsize = 18;
sup_fontsize  = 24;
line_linewidth  = 3;
ylim_CPU = [0 6];
NumTicks_CPU = 4;
ylim_iter = [0 6];
NumTicks_iter = 4;

figure()
subplot(3,1,1)
CPU_ratio_4plots.solvers.mean = zeros(length(ww_pend),5);
CPU_ratio_4plots.solvers.std = zeros(length(ww_pend),5);
for k = 2:length(ww_pend)+1
    CPU_ratio_4plots.solvers.mean(k-1,1) = CPU_ratio.(['pendulum',num2str(k),'dof']).ma27_mumps.mean(end);
    CPU_ratio_4plots.solvers.std(k-1,1) = CPU_ratio.(['pendulum',num2str(k),'dof']).ma27_mumps.std(end);
    CPU_ratio_4plots.solvers.mean(k-1,2) = CPU_ratio.(['pendulum',num2str(k),'dof']).ma57_mumps.mean(end);
    CPU_ratio_4plots.solvers.std(k-1,2) = CPU_ratio.(['pendulum',num2str(k),'dof']).ma57_mumps.std(end);
    CPU_ratio_4plots.solvers.mean(k-1,3) = CPU_ratio.(['pendulum',num2str(k),'dof']).ma77_mumps.mean(end);
    CPU_ratio_4plots.solvers.std(k-1,3) = CPU_ratio.(['pendulum',num2str(k),'dof']).ma77_mumps.std(end);
    CPU_ratio_4plots.solvers.mean(k-1,4) = CPU_ratio.(['pendulum',num2str(k),'dof']).ma86_mumps.mean(end);
    CPU_ratio_4plots.solvers.std(k-1,4) = CPU_ratio.(['pendulum',num2str(k),'dof']).ma86_mumps.std(end);
    CPU_ratio_4plots.solvers.mean(k-1,5) = CPU_ratio.(['pendulum',num2str(k),'dof']).ma97_mumps.mean(end);
    CPU_ratio_4plots.solvers.std(k-1,5) = CPU_ratio.(['pendulum',num2str(k),'dof']).ma97_mumps.std(end);
end
CPU_ratio_4plots.solvers.mean(k,1) = CPU_ratio.pred2D.ma27_mumps.mean(end);
CPU_ratio_4plots.solvers.mean(k+1,1) = CPU_ratio.track3D.ma27_mumps.mean(end);
CPU_ratio_4plots.solvers.std(k,1) = CPU_ratio.pred2D.ma27_mumps.std(end);
CPU_ratio_4plots.solvers.std(k+1,1) = CPU_ratio.track3D.ma27_mumps.std(end);
CPU_ratio_4plots.solvers.mean(k,2) = CPU_ratio.pred2D.ma57_mumps.mean(end);
CPU_ratio_4plots.solvers.mean(k+1,2) = CPU_ratio.track3D.ma57_mumps.mean(end);
CPU_ratio_4plots.solvers.std(k,2) = CPU_ratio.pred2D.ma57_mumps.std(end);
CPU_ratio_4plots.solvers.std(k+1,2) = CPU_ratio.track3D.ma57_mumps.std(end);
CPU_ratio_4plots.solvers.mean(k,3) = CPU_ratio.pred2D.ma77_mumps.mean(end);
CPU_ratio_4plots.solvers.mean(k+1,3) = CPU_ratio.track3D.ma77_mumps.mean(end);
CPU_ratio_4plots.solvers.std(k,3) = CPU_ratio.pred2D.ma77_mumps.std(end);
CPU_ratio_4plots.solvers.std(k+1,3) = CPU_ratio.track3D.ma77_mumps.std(end);
CPU_ratio_4plots.solvers.mean(k,4) = CPU_ratio.pred2D.ma86_mumps.mean(end);
CPU_ratio_4plots.solvers.mean(k+1,4) = CPU_ratio.track3D.ma86_mumps.mean(end);
CPU_ratio_4plots.solvers.std(k,4) = CPU_ratio.pred2D.ma86_mumps.std(end);
CPU_ratio_4plots.solvers.std(k+1,4) = CPU_ratio.track3D.ma86_mumps.std(end);
CPU_ratio_4plots.solvers.mean(k,5) = CPU_ratio.pred2D.ma97_mumps.mean(end);
CPU_ratio_4plots.solvers.mean(k+1,5) = CPU_ratio.track3D.ma97_mumps.mean(end);
CPU_ratio_4plots.solvers.std(k,5) = CPU_ratio.pred2D.ma97_mumps.std(end);
CPU_ratio_4plots.solvers.std(k+1,5) = CPU_ratio.track3D.ma97_mumps.std(end);
h5 = barwitherr(CPU_ratio_4plots.solvers.std,CPU_ratio_4plots.solvers.mean);
for i = 1:length(h5)
    set(h5(i),'FaceColor',color_all(i,:),'EdgeColor',color_all(i,:));
end
hold on
L = get(gca,'XLim');
plot([L(1) L(2)],[1 1],'k','linewidth',1)
set(gca,'Fontsize',label_fontsize);  
set(gca,'XTickLabel',{'','','','','','','','','','',''},'Fontsize',label_fontsize');
ylabel('CPU time','Fontsize',label_fontsize');
ylim([ylim_CPU(1) ylim_CPU(2)]);
L = get(gca,'YLim');
set(gca,'YTick',linspace(L(1),L(2),NumTicks_CPU));     
leg = legend(h5,'ma27 / mumps','ma57 / mumps','ma77 / mumps','ma86 / mumps','ma97 / mumps');
set(gca,'Fontsize',label_fontsize);  
set(leg,'Fontsize',label_fontsize); 
set(leg,'location','Northwest','orientation','horizontal');
box off;
subplot(3,1,2)
iter_ratio_4plots.solvers.mean = zeros(length(ww_pend),5);
iter_ratio_4plots.solvers.std = zeros(length(ww_pend),5);
for k = 2:length(ww_pend)+1
    iter_ratio_4plots.solvers.mean(k-1,1) = iter_ratio.(['pendulum',num2str(k),'dof']).ma27_mumps.mean(end);
    iter_ratio_4plots.solvers.std(k-1,1) = iter_ratio.(['pendulum',num2str(k),'dof']).ma27_mumps.std(end);
    iter_ratio_4plots.solvers.mean(k-1,2) = iter_ratio.(['pendulum',num2str(k),'dof']).ma57_mumps.mean(end);
    iter_ratio_4plots.solvers.std(k-1,2) = iter_ratio.(['pendulum',num2str(k),'dof']).ma57_mumps.std(end);
    iter_ratio_4plots.solvers.mean(k-1,3) = iter_ratio.(['pendulum',num2str(k),'dof']).ma77_mumps.mean(end);
    iter_ratio_4plots.solvers.std(k-1,3) = iter_ratio.(['pendulum',num2str(k),'dof']).ma77_mumps.std(end);
    iter_ratio_4plots.solvers.mean(k-1,4) = iter_ratio.(['pendulum',num2str(k),'dof']).ma86_mumps.mean(end);
    iter_ratio_4plots.solvers.std(k-1,4) = iter_ratio.(['pendulum',num2str(k),'dof']).ma86_mumps.std(end);
    iter_ratio_4plots.solvers.mean(k-1,5) = iter_ratio.(['pendulum',num2str(k),'dof']).ma97_mumps.mean(end);
    iter_ratio_4plots.solvers.std(k-1,5) = iter_ratio.(['pendulum',num2str(k),'dof']).ma97_mumps.std(end);
end
iter_ratio_4plots.solvers.mean(k,1) = iter_ratio.pred2D.ma27_mumps.mean(end);
iter_ratio_4plots.solvers.mean(k+1,1) = iter_ratio.track3D.ma27_mumps.mean(end);
iter_ratio_4plots.solvers.std(k,1) = iter_ratio.pred2D.ma27_mumps.std(end);
iter_ratio_4plots.solvers.std(k+1,1) = iter_ratio.track3D.ma27_mumps.std(end);
iter_ratio_4plots.solvers.mean(k,2) = iter_ratio.pred2D.ma57_mumps.mean(end);
iter_ratio_4plots.solvers.mean(k+1,2) = iter_ratio.track3D.ma57_mumps.mean(end);
iter_ratio_4plots.solvers.std(k,2) = iter_ratio.pred2D.ma57_mumps.std(end);
iter_ratio_4plots.solvers.std(k+1,2) = iter_ratio.track3D.ma57_mumps.std(end);
iter_ratio_4plots.solvers.mean(k,3) = iter_ratio.pred2D.ma77_mumps.mean(end);
iter_ratio_4plots.solvers.mean(k+1,3) = iter_ratio.track3D.ma77_mumps.mean(end);
iter_ratio_4plots.solvers.std(k,3) = iter_ratio.pred2D.ma77_mumps.std(end);
iter_ratio_4plots.solvers.std(k+1,3) = iter_ratio.track3D.ma77_mumps.std(end);
iter_ratio_4plots.solvers.mean(k,4) = iter_ratio.pred2D.ma86_mumps.mean(end);
iter_ratio_4plots.solvers.mean(k+1,4) = iter_ratio.track3D.ma86_mumps.mean(end);
iter_ratio_4plots.solvers.std(k,4) = iter_ratio.pred2D.ma86_mumps.std(end);
iter_ratio_4plots.solvers.std(k+1,4) = iter_ratio.track3D.ma86_mumps.std(end);
iter_ratio_4plots.solvers.mean(k,5) = iter_ratio.pred2D.ma97_mumps.mean(end);
iter_ratio_4plots.solvers.mean(k+1,5) = iter_ratio.track3D.ma97_mumps.mean(end);
iter_ratio_4plots.solvers.std(k,5) = iter_ratio.pred2D.ma97_mumps.std(end);
iter_ratio_4plots.solvers.std(k+1,5) = iter_ratio.track3D.ma97_mumps.std(end);
h6 = barwitherr(iter_ratio_4plots.solvers.std,iter_ratio_4plots.solvers.mean);
for i = 1:length(h5)
    set(h6(i),'FaceColor',color_all(i,:),'EdgeColor',color_all(i,:));
end
hold on;
L = get(gca,'XLim');
plot([L(1) L(2)],[1 1],'k','linewidth',1);
set(gca,'Fontsize',label_fontsize);  
set(gca,'XTickLabel',{'','','','','','','','','','',''},'Fontsize',label_fontsize');
% set(gca,'XTickLabel',{'P2','P3','P4','P5','P6','P7','P8','P9','P10','Pred','Track'},'Fontsize',label_fontsize');
ylabel('Iterations','Fontsize',label_fontsize');
ylim([ylim_iter(1) ylim_iter(2)]);
L = get(gca,'YLim');
set(gca,'YTick',linspace(L(1),L(2),NumTicks_iter));    
% leg = legend(h6,'ma27 vs mumps','ma57 vs mumps','ma77 vs mumps','ma86 vs mumps','ma97 vs mumps');
% set(gca,'Fontsize',label_fontsize);  
% set(leg,'Fontsize',label_fontsize); 
% set(leg,'location','Northwest');
box off;

%% Ranking

[Bsort,Isort] = sort(CPU_ratio_4plots.solvers.mean,2);


