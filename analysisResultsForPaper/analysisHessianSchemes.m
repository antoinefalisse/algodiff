% This script compares approximated and exact Hessian
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
% 1:2 => QR-guess-ma86 (approx,exact)
% 3:4 => DI(walking)-guess-ma86 (approx,exact)
% 5:6 => DI(running)-guess-ma86 (approx,exact)
% 7:8 => QR-guess-ma97 (approx,exact)
% 9:10 => DI(walking)-guess-ma97 (approx,exact)
% 11:12 => DI(running)-guess-ma97 (approx,exact)
ww_2D = [19,25,20,26,21,27,22,28,23,29,24,30];
ww_pend = 2:10;
% Load pre-defined settings
pathmain = pwd;
[pathMainRepo,~,~] = fileparts(pathmain);
pathRepo_2D = [pathMainRepo,'\predictiveSimulations_2D\'];
pathSettings_2D = [pathRepo_2D,'Settings'];
addpath(genpath(pathSettings_2D));
pathRepo_track3D = [pathMainRepo,'\trackingSimulations_3D\'];
pathSettings_track3D = [pathRepo_track3D,'Settings'];
addpath(genpath(pathSettings_track3D));
pathResults_pend = [pathMainRepo,'\pendulumSimulations\Results\'];
% Fixed settings
subject = 'subject1';
body_mass = 62;
body_weight = 62*9.81;
% Colors
color_all(1,:) = [244,194,13]/255;     % Yellow
color_all(2,:) = [60,186,84]/255;      % Green
color_all(3,:) = [0,0,0];              % Black
color_all(4,:) = [253,174,97]/255;      % Red
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
reg_2D.mean = zeros(length(ww_2D),1);
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
        reg_2D.mean(k)  = mean(Stats_2D(ww_2D(k)).m.iterations.regularization_size);
    else
        t_proc_2D(k,:) = NaN;
        n_iter_2D(k) = NaN;
        obj_2D.all(k) = NaN;
        reg_2D.mean(k) = NaN;
        fail_2D = fail_2D + 1;
        disp(['PredSim 2D: trial ',num2str(ww_2D(k)),' did not converge']);
    end
    
end
% Assess convergence: we extract the optimal cost.
obj_2D.ma86.approx = obj_2D.all(1:2:6,1);
obj_2D.ma86.exact = obj_2D.all(2:2:6,1);
obj_2D.ma97.approx = obj_2D.all(7:2:12,1);
obj_2D.ma97.exact = obj_2D.all(8:2:12,1);
% We discriminate between optimal solutions. We exclude the solution from an 
% initial guess if it is larger than threshold*(lowest solution across guesses).
min_obj_2D.ma86.approx = min(obj_2D.ma86.approx);
idx_obj_2D.ma86.approx = obj_2D.ma86.approx > (threshold*min_obj_2D.ma86.approx);
min_obj_2D.ma86.exact = min(obj_2D.ma86.exact);
idx_obj_2D.ma86.exact = obj_2D.ma86.exact > (threshold*min_obj_2D.ma86.exact);
min_obj_2D.ma97.approx = min(obj_2D.ma97.approx);
idx_obj_2D.ma97.approx = obj_2D.ma97.approx > (threshold*min_obj_2D.ma97.approx);
min_obj_2D.ma97.exact = min(obj_2D.ma97.exact);
idx_obj_2D.ma97.exact = obj_2D.ma97.exact > (threshold*min_obj_2D.ma97.exact);
idx_obj_2D.all = [idx_obj_2D.ma86.approx,idx_obj_2D.ma86.exact,...
    idx_obj_2D.ma97.approx,idx_obj_2D.ma97.exact];
% We compare the lowest optimal solutions across cases and issue a warning
% if they differ across cases
min_ma86_approx_exact = abs(min_obj_2D.ma86.approx-min_obj_2D.ma86.exact) < (1.02-threshold)*min(min_obj_2D.ma86.approx,min_obj_2D.ma86.exact);
min_ma97_approx_exact = abs(min_obj_2D.ma97.approx-min_obj_2D.ma97.exact) < (1.02-threshold)*min(min_obj_2D.ma97.approx,min_obj_2D.ma97.exact);
if ~min_ma86_approx_exact
    disp('2D Pred Sim: ma86 approx and exact have different lowest optimal cost')
end
if ~min_ma97_approx_exact
    disp('2D Pred Sim: ma97 approx and exact have different lowest optimal cost')
end
% ma86: approximated Hessian
t_proc_all.pred2D.ma86.approx.all = t_proc_2D(1:2:6,:);
t_proc_all.pred2D.ma86.approx.all(idx_obj_2D.ma86.approx,:) = NaN;
t_proc_all.pred2D.ma86.approx.all(:,end+1) = sum(t_proc_all.pred2D.ma86.approx.all,2);
t_proc_all.pred2D.ma86.approx.mean = nanmean(t_proc_all.pred2D.ma86.approx.all,1);
t_proc_all.pred2D.ma86.approx.std = nanstd(t_proc_all.pred2D.ma86.approx.all,[],1);
n_iter_all.pred2D.ma86.approx.all = n_iter_2D(1:2:6,:);
n_iter_all.pred2D.ma86.approx.all(idx_obj_2D.ma86.approx,:) = NaN;
n_iter_all.pred2D.ma86.approx.mean = nanmean(n_iter_all.pred2D.ma86.approx.all,1);
n_iter_all.pred2D.ma86.approx.std = nanstd(n_iter_all.pred2D.ma86.approx.all,[],1);
reg_all.pred2D.ma86.approx.all = reg_2D.mean(1:2:6,:);
reg_all.pred2D.ma86.approx.all(idx_obj_2D.ma86.approx,:) = NaN;
reg_all.pred2D.ma86.approx.mean = nanmean(reg_all.pred2D.ma86.approx.all,1);
reg_all.pred2D.ma86.approx.std = nanstd(reg_all.pred2D.ma86.approx.all,[],1);
% ma86: exact Hessian
t_proc_all.pred2D.ma86.exact.all = t_proc_2D(2:2:6,:);
t_proc_all.pred2D.ma86.exact.all(idx_obj_2D.ma86.exact,:) = NaN;
t_proc_all.pred2D.ma86.exact.all(:,end+1) = sum(t_proc_all.pred2D.ma86.exact.all,2);
t_proc_all.pred2D.ma86.exact.mean = nanmean(t_proc_all.pred2D.ma86.exact.all,1);
t_proc_all.pred2D.ma86.exact.std = nanstd(t_proc_all.pred2D.ma86.exact.all,[],1);
n_iter_all.pred2D.ma86.exact.all = n_iter_2D(2:2:6,:);
n_iter_all.pred2D.ma86.exact.all(idx_obj_2D.ma86.exact,:) = NaN;
n_iter_all.pred2D.ma86.exact.mean = nanmean(n_iter_all.pred2D.ma86.exact.all,1);
n_iter_all.pred2D.ma86.exact.std = nanstd(n_iter_all.pred2D.ma86.exact.all,[],1);
reg_all.pred2D.ma86.exact.all = reg_2D.mean(2:2:6,:);
reg_all.pred2D.ma86.exact.all(idx_obj_2D.ma86.exact,:) = NaN;
reg_all.pred2D.ma86.exact.mean = nanmean(reg_all.pred2D.ma86.exact.all,1);
reg_all.pred2D.ma86.exact.std = nanstd(reg_all.pred2D.ma86.exact.all,[],1);
% ma97: approximated Hessian
t_proc_all.pred2D.ma97.approx.all = t_proc_2D(7:2:12,:);
t_proc_all.pred2D.ma97.approx.all(idx_obj_2D.ma97.approx,:) = NaN;
t_proc_all.pred2D.ma97.approx.all(:,end+1) = sum(t_proc_all.pred2D.ma97.approx.all,2);
t_proc_all.pred2D.ma97.approx.mean = nanmean(t_proc_all.pred2D.ma97.approx.all,1);
t_proc_all.pred2D.ma97.approx.std = nanstd(t_proc_all.pred2D.ma97.approx.all,[],1);
n_iter_all.pred2D.ma97.approx.all = n_iter_2D(7:2:12,:);
n_iter_all.pred2D.ma97.approx.all(idx_obj_2D.ma97.approx,:) = NaN;
n_iter_all.pred2D.ma97.approx.mean = nanmean(n_iter_all.pred2D.ma97.approx.all,1);
n_iter_all.pred2D.ma97.approx.std = nanstd(n_iter_all.pred2D.ma97.approx.all,[],1);
reg_all.pred2D.ma97.approx.all = reg_2D.mean(7:2:12,:);
reg_all.pred2D.ma97.approx.all(idx_obj_2D.ma97.approx,:) = NaN;
reg_all.pred2D.ma97.approx.mean = nanmean(reg_all.pred2D.ma97.approx.all,1);
reg_all.pred2D.ma97.approx.std = nanstd(reg_all.pred2D.ma97.approx.all,[],1);
% ma97: exact Hessian
t_proc_all.pred2D.ma97.exact.all = t_proc_2D(8:2:12,:);
t_proc_all.pred2D.ma97.exact.all(idx_obj_2D.ma97.exact,:) = NaN;
t_proc_all.pred2D.ma97.exact.all(:,end+1) = sum(t_proc_all.pred2D.ma97.exact.all,2);
t_proc_all.pred2D.ma97.exact.mean = nanmean(t_proc_all.pred2D.ma97.exact.all,1);
t_proc_all.pred2D.ma97.exact.std = nanstd(t_proc_all.pred2D.ma97.exact.all,[],1);
n_iter_all.pred2D.ma97.exact.all = n_iter_2D(8:2:12,:);
n_iter_all.pred2D.ma97.exact.all(idx_obj_2D.ma97.exact,:) = NaN;
n_iter_all.pred2D.ma97.exact.mean = nanmean(n_iter_all.pred2D.ma97.exact.all,1);
n_iter_all.pred2D.ma97.exact.std = nanstd(n_iter_all.pred2D.ma97.exact.all,[],1);
reg_all.pred2D.ma97.exact.all = reg_2D.mean(8:2:12,:);
reg_all.pred2D.ma97.exact.all(idx_obj_2D.ma97.exact,:) = NaN;
reg_all.pred2D.ma97.exact.mean = nanmean(reg_all.pred2D.ma97.exact.all,1);
reg_all.pred2D.ma97.exact.std = nanstd(reg_all.pred2D.ma97.exact.all,[],1);

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
tol_pend = 2;
der = {'AD'};
dim5=1; % Recorder
NIG = 10;
solvers={'mumps','ma27','ma57','ma77','ma86','ma97'};
hessians = {'approx','exact'};
NCases_pend = length(ww_pend)*length(solvers)*length(hessians)*NIG;
t_proc_pend = zeros(NCases_pend,5);
n_iter_pend = zeros(NCases_pend,1);
obj_pend.all = zeros(NCases_pend,1);
reg_pend.mean = zeros(NCases_pend,1);
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
        for hess=hessians
            Options.Hessian=hess{:};
            if strcmp(Options.Hessian,'approx')
                dim3=2;
            elseif strcmp(Options.Hessian,'exact')
                dim3=1;
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
                    reg_pend.mean(count)  = mean(stats_temp.iterations.regularization_size);
                else
                    t_proc_pend(count,:) = NaN;
                    n_iter_pend(count) = NaN;
                    obj_pend.all(count) = NaN;
                    reg_pend.mean(count) = NaN;
                    fail_pend = fail_pend + 1;
                    disp(['Pendulum: trial ',num2str(count),' did not converge: ',...
                        'pendulum',num2str(k),'dof, ',Options.solver,' ',Options.Hessian,' ',num2str(i_IG)]);
                end                          
                count = count + 1;
            end
        end
    end
end
% Assess convergence: we extract the optimal cost.
for k = 2:length(ww_pend)+1
    obj_pend.(['pendulum',num2str(k),'dof']).mumps.approx   = obj_pend.all((k-2)*(12*NIG)+1+0*NIG:(k-2)*(12*NIG)+1*NIG,1);
    obj_pend.(['pendulum',num2str(k),'dof']).mumps.exact    = obj_pend.all((k-2)*(12*NIG)+1+1*NIG:(k-2)*(12*NIG)+2*NIG,1);
    obj_pend.(['pendulum',num2str(k),'dof']).ma27.approx    = obj_pend.all((k-2)*(12*NIG)+1+2*NIG:(k-2)*(12*NIG)+3*NIG,1);
    obj_pend.(['pendulum',num2str(k),'dof']).ma27.exact     = obj_pend.all((k-2)*(12*NIG)+1+3*NIG:(k-2)*(12*NIG)+4*NIG,1);
    obj_pend.(['pendulum',num2str(k),'dof']).ma57.approx    = obj_pend.all((k-2)*(12*NIG)+1+4*NIG:(k-2)*(12*NIG)+5*NIG,1);
    obj_pend.(['pendulum',num2str(k),'dof']).ma57.exact     = obj_pend.all((k-2)*(12*NIG)+1+5*NIG:(k-2)*(12*NIG)+6*NIG,1);
    obj_pend.(['pendulum',num2str(k),'dof']).ma77.approx    = obj_pend.all((k-2)*(12*NIG)+1+6*NIG:(k-2)*(12*NIG)+7*NIG,1);
    obj_pend.(['pendulum',num2str(k),'dof']).ma77.exact     = obj_pend.all((k-2)*(12*NIG)+1+7*NIG:(k-2)*(12*NIG)+8*NIG,1);
    obj_pend.(['pendulum',num2str(k),'dof']).ma86.approx    = obj_pend.all((k-2)*(12*NIG)+1+8*NIG:(k-2)*(12*NIG)+9*NIG,1);
    obj_pend.(['pendulum',num2str(k),'dof']).ma86.exact     = obj_pend.all((k-2)*(12*NIG)+1+9*NIG:(k-2)*(12*NIG)+10*NIG,1);
    obj_pend.(['pendulum',num2str(k),'dof']).ma97.approx    = obj_pend.all((k-2)*(12*NIG)+1+10*NIG:(k-2)*(12*NIG)+11*NIG,1);
    obj_pend.(['pendulum',num2str(k),'dof']).ma97.exact     = obj_pend.all((k-2)*(12*NIG)+1+11*NIG:(k-2)*(12*NIG)+12*NIG,1);
    % We discriminate between optimal solutions. We exclude the solution from an 
    % initial guess for the exact Hessian if it is different than threshold%*(solution from approx Hessian).   
    idx_obj_pend.(['pendulum',num2str(k),'dof']).mumps.approx_exact = abs(obj_pend.(['pendulum',num2str(k),'dof']).mumps.approx - ...
        obj_pend.(['pendulum',num2str(k),'dof']).mumps.exact) >  (1.02-threshold)*obj_pend.(['pendulum',num2str(k),'dof']).mumps.approx;
    temp = sum(idx_obj_pend.(['pendulum',num2str(k),'dof']).mumps.approx_exact);
    if sum(idx_obj_pend.(['pendulum',num2str(k),'dof']).mumps.approx_exact) > 0
        disp(['Pendulum',num2str(k),'dof: mumps approx and exact have ',num2str(temp),' different optimal cost'])
    end
    idx_obj_pend.(['pendulum',num2str(k),'dof']).ma27.approx_exact = abs(obj_pend.(['pendulum',num2str(k),'dof']).ma27.approx - ...
        obj_pend.(['pendulum',num2str(k),'dof']).ma27.exact) >  (1.02-threshold)*obj_pend.(['pendulum',num2str(k),'dof']).ma27.approx;
    temp = sum(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma27.approx_exact);
    if sum(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma27.approx_exact) > 0
        disp(['Pendulum',num2str(k),'dof: ma27 approx and exact have ',num2str(temp),' different optimal cost'])
    end
    idx_obj_pend.(['pendulum',num2str(k),'dof']).ma57.approx_exact = abs(obj_pend.(['pendulum',num2str(k),'dof']).ma57.approx - ...
        obj_pend.(['pendulum',num2str(k),'dof']).ma57.exact) >  (1.02-threshold)*obj_pend.(['pendulum',num2str(k),'dof']).ma57.approx;
    temp = sum(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma57.approx_exact);
    if sum(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma57.approx_exact) > 0
        disp(['Pendulum',num2str(k),'dof: ma57 approx and exact have ',num2str(temp),' different optimal cost'])
    end
    idx_obj_pend.(['pendulum',num2str(k),'dof']).ma77.approx_exact = abs(obj_pend.(['pendulum',num2str(k),'dof']).ma77.approx - ...
        obj_pend.(['pendulum',num2str(k),'dof']).ma77.exact) >  (1.02-threshold)*obj_pend.(['pendulum',num2str(k),'dof']).ma77.approx;
    temp = sum(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma77.approx_exact);
    if sum(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma77.approx_exact) > 0
        disp(['Pendulum',num2str(k),'dof: ma77 approx and exact have ',num2str(temp),' different optimal cost'])
    end
    idx_obj_pend.(['pendulum',num2str(k),'dof']).ma86.approx_exact = abs(obj_pend.(['pendulum',num2str(k),'dof']).ma86.approx - ...
        obj_pend.(['pendulum',num2str(k),'dof']).ma86.exact) >  (1.02-threshold)*obj_pend.(['pendulum',num2str(k),'dof']).ma86.approx;
    temp = sum(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma86.approx_exact);
    if sum(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma86.approx_exact) > 0
        disp(['Pendulum',num2str(k),'dof: ma86 approx and exact have ',num2str(temp),' different optimal cost'])
    end
    idx_obj_pend.(['pendulum',num2str(k),'dof']).ma97.approx_exact = abs(obj_pend.(['pendulum',num2str(k),'dof']).ma97.approx - ...
        obj_pend.(['pendulum',num2str(k),'dof']).ma97.exact) >  (1.02-threshold)*obj_pend.(['pendulum',num2str(k),'dof']).ma97.approx;
    temp = sum(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma97.approx_exact);
    if sum(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma97.approx_exact) > 0
        disp(['Pendulum',num2str(k),'dof: ma97 approx and exact have ',num2str(temp),' different optimal cost'])
    end
    % Average across mumps approx hessian cases
    t_proc_all.(['pendulum',num2str(k),'dof']).mumps.approx.all = t_proc_pend((k-2)*(12*NIG)+1+0*NIG:(k-2)*(12*NIG)+1*NIG,:);
    t_proc_all.(['pendulum',num2str(k),'dof']).mumps.approx.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).mumps.approx_exact,:) = NaN;
    t_proc_all.(['pendulum',num2str(k),'dof']).mumps.approx.all(:,end+1) = sum(t_proc_all.(['pendulum',num2str(k),'dof']).mumps.approx.all,2);
    t_proc_all.(['pendulum',num2str(k),'dof']).mumps.approx.mean = nanmean(t_proc_all.(['pendulum',num2str(k),'dof']).mumps.approx.all,1);
    t_proc_all.(['pendulum',num2str(k),'dof']).mumps.approx.std = nanstd(t_proc_all.(['pendulum',num2str(k),'dof']).mumps.approx.all,[],1);
    n_iter_all.(['pendulum',num2str(k),'dof']).mumps.approx.all = n_iter_pend((k-2)*(12*NIG)+1+0*NIG:(k-2)*(12*NIG)+1*NIG,:);
    n_iter_all.(['pendulum',num2str(k),'dof']).mumps.approx.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).mumps.approx_exact,:) = NaN;
    n_iter_all.(['pendulum',num2str(k),'dof']).mumps.approx.mean = nanmean(n_iter_all.(['pendulum',num2str(k),'dof']).mumps.approx.all,1);
    n_iter_all.(['pendulum',num2str(k),'dof']).mumps.approx.std = nanstd(n_iter_all.(['pendulum',num2str(k),'dof']).mumps.approx.all,[],1);
    reg_all.(['pendulum',num2str(k),'dof']).mumps.approx.all = reg_pend.mean((k-2)*(12*NIG)+1+0*NIG:(k-2)*(12*NIG)+1*NIG,:);
    reg_all.(['pendulum',num2str(k),'dof']).mumps.approx.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).mumps.approx_exact,:) = NaN;
    reg_all.(['pendulum',num2str(k),'dof']).mumps.approx.mean = nanmean(reg_all.(['pendulum',num2str(k),'dof']).mumps.approx.all,1);
    reg_all.(['pendulum',num2str(k),'dof']).mumps.approx.std = nanstd(reg_all.(['pendulum',num2str(k),'dof']).mumps.approx.all,[],1);
    % Average across mumps exact hessian cases
    t_proc_all.(['pendulum',num2str(k),'dof']).mumps.exact.all = t_proc_pend((k-2)*(12*NIG)+1+1*NIG:(k-2)*(12*NIG)+2*NIG,:);
    t_proc_all.(['pendulum',num2str(k),'dof']).mumps.exact.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).mumps.approx_exact,:) = NaN;
    t_proc_all.(['pendulum',num2str(k),'dof']).mumps.exact.all(:,end+1) = sum(t_proc_all.(['pendulum',num2str(k),'dof']).mumps.exact.all,2);
    t_proc_all.(['pendulum',num2str(k),'dof']).mumps.exact.mean = nanmean(t_proc_all.(['pendulum',num2str(k),'dof']).mumps.exact.all,1);
    t_proc_all.(['pendulum',num2str(k),'dof']).mumps.exact.std = nanstd(t_proc_all.(['pendulum',num2str(k),'dof']).mumps.exact.all,[],1);
    n_iter_all.(['pendulum',num2str(k),'dof']).mumps.exact.all = n_iter_pend((k-2)*(12*NIG)+1+1*NIG:(k-2)*(12*NIG)+2*NIG,:);
    n_iter_all.(['pendulum',num2str(k),'dof']).mumps.exact.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).mumps.approx_exact,:) = NaN;
    n_iter_all.(['pendulum',num2str(k),'dof']).mumps.exact.mean = nanmean(n_iter_all.(['pendulum',num2str(k),'dof']).mumps.exact.all,1);
    n_iter_all.(['pendulum',num2str(k),'dof']).mumps.exact.std = nanstd(n_iter_all.(['pendulum',num2str(k),'dof']).mumps.exact.all,[],1);
    reg_all.(['pendulum',num2str(k),'dof']).mumps.exact.all = reg_pend.mean((k-2)*(12*NIG)+1+1*NIG:(k-2)*(12*NIG)+2*NIG,:);
    reg_all.(['pendulum',num2str(k),'dof']).mumps.exact.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).mumps.approx_exact,:) = NaN;
    reg_all.(['pendulum',num2str(k),'dof']).mumps.exact.mean = nanmean(reg_all.(['pendulum',num2str(k),'dof']).mumps.exact.all,1);
    reg_all.(['pendulum',num2str(k),'dof']).mumps.exact.std = nanstd(reg_all.(['pendulum',num2str(k),'dof']).mumps.exact.all,[],1);
    % Average across ma27 approx hessian cases
    t_proc_all.(['pendulum',num2str(k),'dof']).ma27.approx.all = t_proc_pend((k-2)*(12*NIG)+1+2*NIG:(k-2)*(12*NIG)+3*NIG,:);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma27.approx.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma27.approx_exact,:) = NaN;
    t_proc_all.(['pendulum',num2str(k),'dof']).ma27.approx.all(:,end+1) = sum(t_proc_all.(['pendulum',num2str(k),'dof']).ma27.approx.all,2);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma27.approx.mean = nanmean(t_proc_all.(['pendulum',num2str(k),'dof']).ma27.approx.all,1);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma27.approx.std = nanstd(t_proc_all.(['pendulum',num2str(k),'dof']).ma27.approx.all,[],1);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma27.approx.all = n_iter_pend((k-2)*(12*NIG)+1+2*NIG:(k-2)*(12*NIG)+3*NIG,:);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma27.approx.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma27.approx_exact,:) = NaN;
    n_iter_all.(['pendulum',num2str(k),'dof']).ma27.approx.mean = nanmean(n_iter_all.(['pendulum',num2str(k),'dof']).ma27.approx.all,1);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma27.approx.std = nanstd(n_iter_all.(['pendulum',num2str(k),'dof']).ma27.approx.all,[],1);
    reg_all.(['pendulum',num2str(k),'dof']).ma27.approx.all = reg_pend.mean((k-2)*(12*NIG)+1+2*NIG:(k-2)*(12*NIG)+3*NIG,:);
    reg_all.(['pendulum',num2str(k),'dof']).ma27.approx.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma27.approx_exact,:) = NaN;
    reg_all.(['pendulum',num2str(k),'dof']).ma27.approx.mean = nanmean(reg_all.(['pendulum',num2str(k),'dof']).ma27.approx.all,1);
    reg_all.(['pendulum',num2str(k),'dof']).ma27.approx.std = nanstd(reg_all.(['pendulum',num2str(k),'dof']).ma27.approx.all,[],1);
    % Average across ma27 exact hessian cases
    t_proc_all.(['pendulum',num2str(k),'dof']).ma27.exact.all = t_proc_pend((k-2)*(12*NIG)+1+3*NIG:(k-2)*(12*NIG)+4*NIG,:);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma27.exact.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma27.approx_exact,:) = NaN;
    t_proc_all.(['pendulum',num2str(k),'dof']).ma27.exact.all(:,end+1) = sum(t_proc_all.(['pendulum',num2str(k),'dof']).ma27.exact.all,2);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma27.exact.mean = nanmean(t_proc_all.(['pendulum',num2str(k),'dof']).ma27.exact.all,1);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma27.exact.std = nanstd(t_proc_all.(['pendulum',num2str(k),'dof']).ma27.exact.all,[],1);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma27.exact.all = n_iter_pend((k-2)*(12*NIG)+1+3*NIG:(k-2)*(12*NIG)+4*NIG,:);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma27.exact.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma27.approx_exact,:) = NaN;
    n_iter_all.(['pendulum',num2str(k),'dof']).ma27.exact.mean = nanmean(n_iter_all.(['pendulum',num2str(k),'dof']).ma27.exact.all,1);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma27.exact.std = nanstd(n_iter_all.(['pendulum',num2str(k),'dof']).ma27.exact.all,[],1);
    reg_all.(['pendulum',num2str(k),'dof']).ma27.exact.all = reg_pend.mean((k-2)*(12*NIG)+1+3*NIG:(k-2)*(12*NIG)+4*NIG,:);
    reg_all.(['pendulum',num2str(k),'dof']).ma27.exact.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma27.approx_exact,:) = NaN;
    reg_all.(['pendulum',num2str(k),'dof']).ma27.exact.mean = nanmean(reg_all.(['pendulum',num2str(k),'dof']).ma27.exact.all,1);
    reg_all.(['pendulum',num2str(k),'dof']).ma27.exact.std = nanstd(reg_all.(['pendulum',num2str(k),'dof']).ma27.exact.all,[],1);
    % Average across ma57 approx hessian cases
    t_proc_all.(['pendulum',num2str(k),'dof']).ma57.approx.all = t_proc_pend((k-2)*(12*NIG)+1+4*NIG:(k-2)*(12*NIG)+5*NIG,:);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma57.approx.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma57.approx_exact,:) = NaN;
    t_proc_all.(['pendulum',num2str(k),'dof']).ma57.approx.all(:,end+1) = sum(t_proc_all.(['pendulum',num2str(k),'dof']).ma57.approx.all,2);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma57.approx.mean = nanmean(t_proc_all.(['pendulum',num2str(k),'dof']).ma57.approx.all,1);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma57.approx.std = nanstd(t_proc_all.(['pendulum',num2str(k),'dof']).ma57.approx.all,[],1);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma57.approx.all = n_iter_pend((k-2)*(12*NIG)+1+4*NIG:(k-2)*(12*NIG)+5*NIG,:);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma57.approx.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma57.approx_exact,:) = NaN;
    n_iter_all.(['pendulum',num2str(k),'dof']).ma57.approx.mean = nanmean(n_iter_all.(['pendulum',num2str(k),'dof']).ma57.approx.all,1);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma57.approx.std = nanstd(n_iter_all.(['pendulum',num2str(k),'dof']).ma57.approx.all,[],1);
    reg_all.(['pendulum',num2str(k),'dof']).ma57.approx.all = reg_pend.mean((k-2)*(12*NIG)+1+4*NIG:(k-2)*(12*NIG)+5*NIG,:);
    reg_all.(['pendulum',num2str(k),'dof']).ma57.approx.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma57.approx_exact,:) = NaN;
    reg_all.(['pendulum',num2str(k),'dof']).ma57.approx.mean = nanmean(reg_all.(['pendulum',num2str(k),'dof']).ma57.approx.all,1);
    reg_all.(['pendulum',num2str(k),'dof']).ma57.approx.std = nanstd(reg_all.(['pendulum',num2str(k),'dof']).ma57.approx.all,[],1);
    % Average across ma57 exact hessian cases
    t_proc_all.(['pendulum',num2str(k),'dof']).ma57.exact.all = t_proc_pend((k-2)*(12*NIG)+1+5*NIG:(k-2)*(12*NIG)+6*NIG,:);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma57.exact.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma57.approx_exact,:) = NaN;
    t_proc_all.(['pendulum',num2str(k),'dof']).ma57.exact.all(:,end+1) = sum(t_proc_all.(['pendulum',num2str(k),'dof']).ma57.exact.all,2);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma57.exact.mean = nanmean(t_proc_all.(['pendulum',num2str(k),'dof']).ma57.exact.all,1);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma57.exact.std = nanstd(t_proc_all.(['pendulum',num2str(k),'dof']).ma57.exact.all,[],1);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma57.exact.all = n_iter_pend((k-2)*(12*NIG)+1+5*NIG:(k-2)*(12*NIG)+6*NIG,:);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma57.exact.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma57.approx_exact,:) = NaN;
    n_iter_all.(['pendulum',num2str(k),'dof']).ma57.exact.mean = nanmean(n_iter_all.(['pendulum',num2str(k),'dof']).ma57.exact.all,1);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma57.exact.std = nanstd(n_iter_all.(['pendulum',num2str(k),'dof']).ma57.exact.all,[],1);
    reg_all.(['pendulum',num2str(k),'dof']).ma57.exact.all = reg_pend.mean((k-2)*(12*NIG)+1+5*NIG:(k-2)*(12*NIG)+6*NIG,:);
    reg_all.(['pendulum',num2str(k),'dof']).ma57.exact.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma57.approx_exact,:) = NaN;
    reg_all.(['pendulum',num2str(k),'dof']).ma57.exact.mean = nanmean(reg_all.(['pendulum',num2str(k),'dof']).ma57.exact.all,1);
    reg_all.(['pendulum',num2str(k),'dof']).ma57.exact.std = nanstd(reg_all.(['pendulum',num2str(k),'dof']).ma57.exact.all,[],1);
    % Average across ma77 approx hessian cases
    t_proc_all.(['pendulum',num2str(k),'dof']).ma77.approx.all = t_proc_pend((k-2)*(12*NIG)+1+6*NIG:(k-2)*(12*NIG)+7*NIG,:);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma77.approx.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma77.approx_exact,:) = NaN;
    t_proc_all.(['pendulum',num2str(k),'dof']).ma77.approx.all(:,end+1) = sum(t_proc_all.(['pendulum',num2str(k),'dof']).ma77.approx.all,2);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma77.approx.mean = nanmean(t_proc_all.(['pendulum',num2str(k),'dof']).ma77.approx.all,1);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma77.approx.std = nanstd(t_proc_all.(['pendulum',num2str(k),'dof']).ma77.approx.all,[],1);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma77.approx.all = n_iter_pend((k-2)*(12*NIG)+1+6*NIG:(k-2)*(12*NIG)+7*NIG,:);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma77.approx.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma77.approx_exact,:) = NaN;
    n_iter_all.(['pendulum',num2str(k),'dof']).ma77.approx.mean = nanmean(n_iter_all.(['pendulum',num2str(k),'dof']).ma77.approx.all,1);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma77.approx.std = nanstd(n_iter_all.(['pendulum',num2str(k),'dof']).ma77.approx.all,[],1);
    reg_all.(['pendulum',num2str(k),'dof']).ma77.approx.all = reg_pend.mean((k-2)*(12*NIG)+1+6*NIG:(k-2)*(12*NIG)+7*NIG,:);
    reg_all.(['pendulum',num2str(k),'dof']).ma77.approx.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma77.approx_exact,:) = NaN;
    reg_all.(['pendulum',num2str(k),'dof']).ma77.approx.mean = nanmean(reg_all.(['pendulum',num2str(k),'dof']).ma77.approx.all,1);
    reg_all.(['pendulum',num2str(k),'dof']).ma77.approx.std = nanstd(reg_all.(['pendulum',num2str(k),'dof']).ma77.approx.all,[],1);
    % Average across ma77 exact hessian cases
    t_proc_all.(['pendulum',num2str(k),'dof']).ma77.exact.all = t_proc_pend((k-2)*(12*NIG)+1+7*NIG:(k-2)*(12*NIG)+8*NIG,:);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma77.exact.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma77.approx_exact,:) = NaN;
    t_proc_all.(['pendulum',num2str(k),'dof']).ma77.exact.all(:,end+1) = sum(t_proc_all.(['pendulum',num2str(k),'dof']).ma77.exact.all,2);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma77.exact.mean = nanmean(t_proc_all.(['pendulum',num2str(k),'dof']).ma77.exact.all,1);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma77.exact.std = nanstd(t_proc_all.(['pendulum',num2str(k),'dof']).ma77.exact.all,[],1);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma77.exact.all = n_iter_pend((k-2)*(12*NIG)+1+7*NIG:(k-2)*(12*NIG)+8*NIG,:);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma77.exact.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma77.approx_exact,:) = NaN;
    n_iter_all.(['pendulum',num2str(k),'dof']).ma77.exact.mean = nanmean(n_iter_all.(['pendulum',num2str(k),'dof']).ma77.exact.all,1);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma77.exact.std = nanstd(n_iter_all.(['pendulum',num2str(k),'dof']).ma77.exact.all,[],1);
    reg_all.(['pendulum',num2str(k),'dof']).ma77.exact.all = reg_pend.mean((k-2)*(12*NIG)+1+7*NIG:(k-2)*(12*NIG)+8*NIG,:);
    reg_all.(['pendulum',num2str(k),'dof']).ma77.exact.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma77.approx_exact,:) = NaN;
    reg_all.(['pendulum',num2str(k),'dof']).ma77.exact.mean = nanmean(reg_all.(['pendulum',num2str(k),'dof']).ma77.exact.all,1);
    reg_all.(['pendulum',num2str(k),'dof']).ma77.exact.std = nanstd(reg_all.(['pendulum',num2str(k),'dof']).ma77.exact.all,[],1);
    % Average across ma86 approx hessian cases
    t_proc_all.(['pendulum',num2str(k),'dof']).ma86.approx.all = t_proc_pend((k-2)*(12*NIG)+1+8*NIG:(k-2)*(12*NIG)+9*NIG,:);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma86.approx.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma86.approx_exact,:) = NaN;
    t_proc_all.(['pendulum',num2str(k),'dof']).ma86.approx.all(:,end+1) = sum(t_proc_all.(['pendulum',num2str(k),'dof']).ma86.approx.all,2);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma86.approx.mean = nanmean(t_proc_all.(['pendulum',num2str(k),'dof']).ma86.approx.all,1);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma86.approx.std = nanstd(t_proc_all.(['pendulum',num2str(k),'dof']).ma86.approx.all,[],1);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma86.approx.all = n_iter_pend((k-2)*(12*NIG)+1+8*NIG:(k-2)*(12*NIG)+9*NIG,:);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma86.approx.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma86.approx_exact,:) = NaN;
    n_iter_all.(['pendulum',num2str(k),'dof']).ma86.approx.mean = nanmean(n_iter_all.(['pendulum',num2str(k),'dof']).ma86.approx.all,1);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma86.approx.std = nanstd(n_iter_all.(['pendulum',num2str(k),'dof']).ma86.approx.all,[],1);
    reg_all.(['pendulum',num2str(k),'dof']).ma86.approx.all = reg_pend.mean((k-2)*(12*NIG)+1+8*NIG:(k-2)*(12*NIG)+9*NIG,:);
    reg_all.(['pendulum',num2str(k),'dof']).ma86.approx.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma86.approx_exact,:) = NaN;
    reg_all.(['pendulum',num2str(k),'dof']).ma86.approx.mean = nanmean(reg_all.(['pendulum',num2str(k),'dof']).ma86.approx.all,1);
    reg_all.(['pendulum',num2str(k),'dof']).ma86.approx.std = nanstd(reg_all.(['pendulum',num2str(k),'dof']).ma86.approx.all,[],1);
    % Average across ma86 exact hessian cases
    t_proc_all.(['pendulum',num2str(k),'dof']).ma86.exact.all = t_proc_pend((k-2)*(12*NIG)+1+9*NIG:(k-2)*(12*NIG)+10*NIG,:);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma86.exact.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma86.approx_exact,:) = NaN;
    t_proc_all.(['pendulum',num2str(k),'dof']).ma86.exact.all(:,end+1) = sum(t_proc_all.(['pendulum',num2str(k),'dof']).ma86.exact.all,2);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma86.exact.mean = nanmean(t_proc_all.(['pendulum',num2str(k),'dof']).ma86.exact.all,1);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma86.exact.std = nanstd(t_proc_all.(['pendulum',num2str(k),'dof']).ma86.exact.all,[],1);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma86.exact.all = n_iter_pend((k-2)*(12*NIG)+1+9*NIG:(k-2)*(12*NIG)+10*NIG,:);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma86.exact.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma86.approx_exact,:) = NaN;
    n_iter_all.(['pendulum',num2str(k),'dof']).ma86.exact.mean = nanmean(n_iter_all.(['pendulum',num2str(k),'dof']).ma86.exact.all,1);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma86.exact.std = nanstd(n_iter_all.(['pendulum',num2str(k),'dof']).ma86.exact.all,[],1);
    reg_all.(['pendulum',num2str(k),'dof']).ma86.exact.all = reg_pend.mean((k-2)*(12*NIG)+1+9*NIG:(k-2)*(12*NIG)+10*NIG,:);
    reg_all.(['pendulum',num2str(k),'dof']).ma86.exact.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma86.approx_exact,:) = NaN;
    reg_all.(['pendulum',num2str(k),'dof']).ma86.exact.mean = nanmean(reg_all.(['pendulum',num2str(k),'dof']).ma86.exact.all,1);
    reg_all.(['pendulum',num2str(k),'dof']).ma86.exact.std = nanstd(reg_all.(['pendulum',num2str(k),'dof']).ma86.exact.all,[],1);
    % Average across ma97 approx hessian cases
    t_proc_all.(['pendulum',num2str(k),'dof']).ma97.approx.all = t_proc_pend((k-2)*(12*NIG)+1+10*NIG:(k-2)*(12*NIG)+11*NIG,:);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma97.approx.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma97.approx_exact,:) = NaN;
    t_proc_all.(['pendulum',num2str(k),'dof']).ma97.approx.all(:,end+1) = sum(t_proc_all.(['pendulum',num2str(k),'dof']).ma97.approx.all,2);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma97.approx.mean = nanmean(t_proc_all.(['pendulum',num2str(k),'dof']).ma97.approx.all,1);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma97.approx.std = nanstd(t_proc_all.(['pendulum',num2str(k),'dof']).ma97.approx.all,[],1);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma97.approx.all = n_iter_pend((k-2)*(12*NIG)+1+10*NIG:(k-2)*(12*NIG)+11*NIG,:);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma97.approx.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma97.approx_exact,:) = NaN;
    n_iter_all.(['pendulum',num2str(k),'dof']).ma97.approx.mean = nanmean(n_iter_all.(['pendulum',num2str(k),'dof']).ma97.approx.all,1);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma97.approx.std = nanstd(n_iter_all.(['pendulum',num2str(k),'dof']).ma97.approx.all,[],1);
    reg_all.(['pendulum',num2str(k),'dof']).ma97.approx.all = reg_pend.mean((k-2)*(12*NIG)+1+10*NIG:(k-2)*(12*NIG)+11*NIG,:);
    reg_all.(['pendulum',num2str(k),'dof']).ma97.approx.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma97.approx_exact,:) = NaN;
    reg_all.(['pendulum',num2str(k),'dof']).ma97.approx.mean = nanmean(reg_all.(['pendulum',num2str(k),'dof']).ma97.approx.all,1);
    reg_all.(['pendulum',num2str(k),'dof']).ma97.approx.std = nanstd(reg_all.(['pendulum',num2str(k),'dof']).ma97.approx.all,[],1);
    % Average across ma97 exact hessian cases
    t_proc_all.(['pendulum',num2str(k),'dof']).ma97.exact.all = t_proc_pend((k-2)*(12*NIG)+1+11*NIG:(k-2)*(12*NIG)+12*NIG,:);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma97.exact.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma97.approx_exact,:) = NaN;
    t_proc_all.(['pendulum',num2str(k),'dof']).ma97.exact.all(:,end+1) = sum(t_proc_all.(['pendulum',num2str(k),'dof']).ma97.exact.all,2);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma97.exact.mean = nanmean(t_proc_all.(['pendulum',num2str(k),'dof']).ma97.exact.all,1);
    t_proc_all.(['pendulum',num2str(k),'dof']).ma97.exact.std = nanstd(t_proc_all.(['pendulum',num2str(k),'dof']).ma97.exact.all,[],1);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma97.exact.all = n_iter_pend((k-2)*(12*NIG)+1+11*NIG:(k-2)*(12*NIG)+12*NIG,:);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma97.exact.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma97.approx_exact,:) = NaN;
    n_iter_all.(['pendulum',num2str(k),'dof']).ma97.exact.mean = nanmean(n_iter_all.(['pendulum',num2str(k),'dof']).ma97.exact.all,1);
    n_iter_all.(['pendulum',num2str(k),'dof']).ma97.exact.std = nanstd(n_iter_all.(['pendulum',num2str(k),'dof']).ma97.exact.all,[],1);
    reg_all.(['pendulum',num2str(k),'dof']).ma97.exact.all = reg_pend.mean((k-2)*(12*NIG)+1+11*NIG:(k-2)*(12*NIG)+12*NIG,:);
    reg_all.(['pendulum',num2str(k),'dof']).ma97.exact.all(idx_obj_pend.(['pendulum',num2str(k),'dof']).ma97.approx_exact,:) = NaN;
    reg_all.(['pendulum',num2str(k),'dof']).ma97.exact.mean = nanmean(reg_all.(['pendulum',num2str(k),'dof']).ma97.exact.all,1);
    reg_all.(['pendulum',num2str(k),'dof']).ma97.exact.std = nanstd(reg_all.(['pendulum',num2str(k),'dof']).ma97.exact.all,[],1);
end

%% Differences in CPU time between cases
% Combine results from PredSim 2D and TrackSim 3D and Pendulums
t_proc_all.pred2D_track3D_pend.mumps.approx.all = [];
t_proc_all.pred2D_track3D_pend.mumps.exact.all = [];
t_proc_all.pred2D_track3D_pend.ma27.approx.all = [];
t_proc_all.pred2D_track3D_pend.ma27.exact.all = [];
t_proc_all.pred2D_track3D_pend.ma57.approx.all = [];
t_proc_all.pred2D_track3D_pend.ma57.exact.all = [];
t_proc_all.pred2D_track3D_pend.ma77.approx.all = [];
t_proc_all.pred2D_track3D_pend.ma77.exact.all = [];
t_proc_all.pred2D_track3D_pend.ma86.approx.all = [t_proc_all.pred2D.ma86.approx.all];
t_proc_all.pred2D_track3D_pend.ma86.exact.all = [t_proc_all.pred2D.ma86.exact.all];
t_proc_all.pred2D_track3D_pend.ma97.approx.all = [t_proc_all.pred2D.ma97.approx.all];
t_proc_all.pred2D_track3D_pend.ma97.exact.all = [t_proc_all.pred2D.ma97.exact.all];
t_proc_all.pred2D_track3D_pend.all.approx.all = [t_proc_all.pred2D.ma86.approx.all;t_proc_all.pred2D.ma97.approx.all];
t_proc_all.pred2D_track3D_pend.all.exact.all = [t_proc_all.pred2D.ma86.exact.all;t_proc_all.pred2D.ma97.exact.all];
for k = 2:length(ww_pend)+1 
    t_proc_all.pred2D_track3D_pend.mumps.approx.all = [t_proc_all.pred2D_track3D_pend.mumps.approx.all;t_proc_all.(['pendulum',num2str(k),'dof']).mumps.approx.all];
    t_proc_all.pred2D_track3D_pend.ma27.approx.all = [t_proc_all.pred2D_track3D_pend.ma27.approx.all;t_proc_all.(['pendulum',num2str(k),'dof']).ma27.approx.all];
    t_proc_all.pred2D_track3D_pend.ma57.approx.all = [t_proc_all.pred2D_track3D_pend.ma57.approx.all;t_proc_all.(['pendulum',num2str(k),'dof']).ma57.approx.all];
    t_proc_all.pred2D_track3D_pend.ma77.approx.all = [t_proc_all.pred2D_track3D_pend.ma77.approx.all;t_proc_all.(['pendulum',num2str(k),'dof']).ma77.approx.all];
    t_proc_all.pred2D_track3D_pend.ma86.approx.all = [t_proc_all.pred2D_track3D_pend.ma86.approx.all;t_proc_all.(['pendulum',num2str(k),'dof']).ma86.approx.all];
    t_proc_all.pred2D_track3D_pend.ma97.approx.all = [t_proc_all.pred2D_track3D_pend.ma97.approx.all;t_proc_all.(['pendulum',num2str(k),'dof']).ma97.approx.all];    
    t_proc_all.pred2D_track3D_pend.mumps.exact.all = [t_proc_all.pred2D_track3D_pend.mumps.exact.all;t_proc_all.(['pendulum',num2str(k),'dof']).mumps.exact.all];
    t_proc_all.pred2D_track3D_pend.ma27.exact.all = [t_proc_all.pred2D_track3D_pend.ma27.exact.all;t_proc_all.(['pendulum',num2str(k),'dof']).ma27.exact.all];
    t_proc_all.pred2D_track3D_pend.ma57.exact.all = [t_proc_all.pred2D_track3D_pend.ma57.exact.all;t_proc_all.(['pendulum',num2str(k),'dof']).ma57.exact.all];
    t_proc_all.pred2D_track3D_pend.ma77.exact.all = [t_proc_all.pred2D_track3D_pend.ma77.exact.all;t_proc_all.(['pendulum',num2str(k),'dof']).ma77.exact.all];
    t_proc_all.pred2D_track3D_pend.ma86.exact.all = [t_proc_all.pred2D_track3D_pend.ma86.exact.all;t_proc_all.(['pendulum',num2str(k),'dof']).ma86.exact.all];
    t_proc_all.pred2D_track3D_pend.ma97.exact.all = [t_proc_all.pred2D_track3D_pend.ma97.exact.all;t_proc_all.(['pendulum',num2str(k),'dof']).ma97.exact.all];
    t_proc_all.pred2D_track3D_pend.all.approx.all = [t_proc_all.pred2D_track3D_pend.all.approx.all;t_proc_all.(['pendulum',num2str(k),'dof']).mumps.approx.all;...
        t_proc_all.(['pendulum',num2str(k),'dof']).ma27.approx.all;t_proc_all.(['pendulum',num2str(k),'dof']).ma57.approx.all;...
        t_proc_all.(['pendulum',num2str(k),'dof']).ma77.approx.all;t_proc_all.(['pendulum',num2str(k),'dof']).ma86.approx.all;...
        t_proc_all.(['pendulum',num2str(k),'dof']).ma97.approx.all];
    t_proc_all.pred2D_track3D_pend.all.exact.all = [t_proc_all.pred2D_track3D_pend.all.exact.all;t_proc_all.(['pendulum',num2str(k),'dof']).mumps.exact.all;...
        t_proc_all.(['pendulum',num2str(k),'dof']).ma27.exact.all;t_proc_all.(['pendulum',num2str(k),'dof']).ma57.exact.all;...
        t_proc_all.(['pendulum',num2str(k),'dof']).ma77.exact.all;t_proc_all.(['pendulum',num2str(k),'dof']).ma86.exact.all;...
        t_proc_all.(['pendulum',num2str(k),'dof']).ma97.exact.all];
end
% Analysis at the level of the solvers
% Calculate ratio between mumps.exact and mumps.approx
CPU_ratio.pred2D_track3D_pend.mumpse_mumpsa.all = t_proc_all.pred2D_track3D_pend.mumps.exact.all./t_proc_all.pred2D_track3D_pend.mumps.approx.all;
CPU_ratio.pred2D_track3D_pend.mumpse_mumpsa.mean = nanmean(CPU_ratio.pred2D_track3D_pend.mumpse_mumpsa.all,1);
CPU_ratio.pred2D_track3D_pend.mumpse_mumpsa.std = nanstd(CPU_ratio.pred2D_track3D_pend.mumpse_mumpsa.all,[],1);
% Calculate ratio between ma27.exact and ma27.approx
CPU_ratio.pred2D_track3D_pend.ma27e_ma27a.all = t_proc_all.pred2D_track3D_pend.ma27.exact.all./t_proc_all.pred2D_track3D_pend.ma27.approx.all;
CPU_ratio.pred2D_track3D_pend.ma27e_ma27a.mean = nanmean(CPU_ratio.pred2D_track3D_pend.ma27e_ma27a.all,1);
CPU_ratio.pred2D_track3D_pend.ma27e_ma27a.std = nanstd(CPU_ratio.pred2D_track3D_pend.ma27e_ma27a.all,[],1);
% Calculate ratio between ma57.exact and ma57.approx
CPU_ratio.pred2D_track3D_pend.ma57e_ma57a.all = t_proc_all.pred2D_track3D_pend.ma57.exact.all./t_proc_all.pred2D_track3D_pend.ma57.approx.all;
CPU_ratio.pred2D_track3D_pend.ma57e_ma57a.mean = nanmean(CPU_ratio.pred2D_track3D_pend.ma57e_ma57a.all,1);
CPU_ratio.pred2D_track3D_pend.ma57e_ma57a.std = nanstd(CPU_ratio.pred2D_track3D_pend.ma57e_ma57a.all,[],1);
% Calculate ratio between ma77.exact and ma77.approx
CPU_ratio.pred2D_track3D_pend.ma77e_ma77a.all = t_proc_all.pred2D_track3D_pend.ma77.exact.all./t_proc_all.pred2D_track3D_pend.ma77.approx.all;
CPU_ratio.pred2D_track3D_pend.ma77e_ma77a.mean = nanmean(CPU_ratio.pred2D_track3D_pend.ma77e_ma77a.all,1);
CPU_ratio.pred2D_track3D_pend.ma77e_ma77a.std = nanstd(CPU_ratio.pred2D_track3D_pend.ma77e_ma77a.all,[],1);
% Calculate ratio between ma86.exact and ma86.approx
CPU_ratio.pred2D_track3D_pend.ma86e_ma86a.all = t_proc_all.pred2D_track3D_pend.ma86.exact.all./t_proc_all.pred2D_track3D_pend.ma86.approx.all;
CPU_ratio.pred2D_track3D_pend.ma86e_ma86a.mean = nanmean(CPU_ratio.pred2D_track3D_pend.ma86e_ma86a.all,1);
CPU_ratio.pred2D_track3D_pend.ma86e_ma86a.std = nanstd(CPU_ratio.pred2D_track3D_pend.ma86e_ma86a.all,[],1);
% Calculate ratio between ma97.exact and ma97.approx
CPU_ratio.pred2D_track3D_pend.ma97e_ma97a.all = t_proc_all.pred2D_track3D_pend.ma97.exact.all./t_proc_all.pred2D_track3D_pend.ma97.approx.all;
CPU_ratio.pred2D_track3D_pend.ma97e_ma97a.mean = nanmean(CPU_ratio.pred2D_track3D_pend.ma97e_ma97a.all,1);
CPU_ratio.pred2D_track3D_pend.ma97e_ma97a.std = nanstd(CPU_ratio.pred2D_track3D_pend.ma97e_ma97a.all,[],1);
% Calculate ratio between all.exact and all.approx
CPU_ratio.pred2D_track3D_pend.maalle_maalla.all = t_proc_all.pred2D_track3D_pend.all.exact.all./t_proc_all.pred2D_track3D_pend.all.approx.all;
CPU_ratio.pred2D_track3D_pend.maalle_maalla.mean = nanmean(CPU_ratio.pred2D_track3D_pend.maalle_maalla.all,1);
CPU_ratio.pred2D_track3D_pend.maalle_maalla.std = nanstd(CPU_ratio.pred2D_track3D_pend.maalle_maalla.all,[],1);
% Analysis at the level of the problems
% PredSim 2D
% Calculate ratio between ma86.exact and ma86.approx
CPU_ratio.pred2D.ma86e_ma86a.all = t_proc_all.pred2D.ma86.exact.all./t_proc_all.pred2D.ma86.approx.all;
CPU_ratio.pred2D.ma86e_ma86a.mean = nanmean(CPU_ratio.pred2D.ma86e_ma86a.all,1);
CPU_ratio.pred2D.ma86e_ma86a.std = nanstd(CPU_ratio.pred2D.ma86e_ma86a.all,[],1);
% Calculate ratio between ma97.exact and ma97.approx
CPU_ratio.pred2D.ma97e_ma97a.all = t_proc_all.pred2D.ma97.exact.all./t_proc_all.pred2D.ma97.approx.all;
CPU_ratio.pred2D.ma97e_ma97a.mean = nanmean(CPU_ratio.pred2D.ma97e_ma97a.all,1);
CPU_ratio.pred2D.ma97e_ma97a.std = nanstd(CPU_ratio.pred2D.ma97e_ma97a.all,[],1);
% Calculate ratio between all.exact and all.approx
CPU_ratio.pred2D.alle_alla.all = [CPU_ratio.pred2D.ma86e_ma86a.all;CPU_ratio.pred2D.ma97e_ma97a.all];
CPU_ratio.pred2D.alle_alla.mean = nanmean(CPU_ratio.pred2D.alle_alla.all,1);
CPU_ratio.pred2D.alle_alla.std = nanstd(CPU_ratio.pred2D.alle_alla.all,[],1);
% Pendulums
CPU_ratio.allpendulum.alle_alla.all = [];
for k = 2:length(ww_pend)+1 
    % Calculate ratio between mumps.exact and mumps.approx
    CPU_ratio.(['pendulum',num2str(k),'dof']).mumpse_mumpsa.all = t_proc_all.(['pendulum',num2str(k),'dof']).mumps.exact.all./t_proc_all.(['pendulum',num2str(k),'dof']).mumps.approx.all;
    CPU_ratio.(['pendulum',num2str(k),'dof']).mumpse_mumpsa.mean = nanmean(CPU_ratio.(['pendulum',num2str(k),'dof']).mumpse_mumpsa.all,1);
    CPU_ratio.(['pendulum',num2str(k),'dof']).mumpse_mumpsa.std = nanstd(CPU_ratio.(['pendulum',num2str(k),'dof']).mumpse_mumpsa.all,[],1);
    % Calculate ratio between ma27.exact and ma27.approx
    CPU_ratio.(['pendulum',num2str(k),'dof']).ma27e_ma27a.all = t_proc_all.(['pendulum',num2str(k),'dof']).ma27.exact.all./t_proc_all.(['pendulum',num2str(k),'dof']).ma27.approx.all;
    CPU_ratio.(['pendulum',num2str(k),'dof']).ma27e_ma27a.mean = nanmean(CPU_ratio.(['pendulum',num2str(k),'dof']).ma27e_ma27a.all,1);
    CPU_ratio.(['pendulum',num2str(k),'dof']).ma27e_ma27a.std = nanstd(CPU_ratio.(['pendulum',num2str(k),'dof']).ma27e_ma27a.all,[],1);
    % Calculate ratio between ma57.exact and ma57.approx
    CPU_ratio.(['pendulum',num2str(k),'dof']).ma57e_ma57a.all = t_proc_all.(['pendulum',num2str(k),'dof']).ma57.exact.all./t_proc_all.(['pendulum',num2str(k),'dof']).ma57.approx.all;
    CPU_ratio.(['pendulum',num2str(k),'dof']).ma57e_ma57a.mean = nanmean(CPU_ratio.(['pendulum',num2str(k),'dof']).ma57e_ma57a.all,1);
    CPU_ratio.(['pendulum',num2str(k),'dof']).ma57e_ma57a.std = nanstd(CPU_ratio.(['pendulum',num2str(k),'dof']).ma57e_ma57a.all,[],1);
    % Calculate ratio between ma77.exact and ma77.approx
    CPU_ratio.(['pendulum',num2str(k),'dof']).ma77e_ma77a.all = t_proc_all.(['pendulum',num2str(k),'dof']).ma77.exact.all./t_proc_all.(['pendulum',num2str(k),'dof']).ma77.approx.all;
    CPU_ratio.(['pendulum',num2str(k),'dof']).ma77e_ma77a.mean = nanmean(CPU_ratio.(['pendulum',num2str(k),'dof']).ma77e_ma77a.all,1);
    CPU_ratio.(['pendulum',num2str(k),'dof']).ma77e_ma77a.std = nanstd(CPU_ratio.(['pendulum',num2str(k),'dof']).ma77e_ma77a.all,[],1);
    % Calculate ratio between ma86.exact and ma86.approx
    CPU_ratio.(['pendulum',num2str(k),'dof']).ma86e_ma86a.all = t_proc_all.(['pendulum',num2str(k),'dof']).ma86.exact.all./t_proc_all.(['pendulum',num2str(k),'dof']).ma86.approx.all;
    CPU_ratio.(['pendulum',num2str(k),'dof']).ma86e_ma86a.mean = nanmean(CPU_ratio.(['pendulum',num2str(k),'dof']).ma86e_ma86a.all,1);
    CPU_ratio.(['pendulum',num2str(k),'dof']).ma86e_ma86a.std = nanstd(CPU_ratio.(['pendulum',num2str(k),'dof']).ma86e_ma86a.all,[],1);
    % Calculate ratio between ma97.exact and ma97.approx
    CPU_ratio.(['pendulum',num2str(k),'dof']).ma97e_ma97a.all = t_proc_all.(['pendulum',num2str(k),'dof']).ma97.exact.all./t_proc_all.(['pendulum',num2str(k),'dof']).ma97.approx.all;
    CPU_ratio.(['pendulum',num2str(k),'dof']).ma97e_ma97a.mean = nanmean(CPU_ratio.(['pendulum',num2str(k),'dof']).ma97e_ma97a.all,1);
    CPU_ratio.(['pendulum',num2str(k),'dof']).ma97e_ma97a.std = nanstd(CPU_ratio.(['pendulum',num2str(k),'dof']).ma97e_ma97a.all,[],1);
    % Calculate ratio between all.exact and all.approx
    CPU_ratio.(['pendulum',num2str(k),'dof']).alle_alla.all = [CPU_ratio.(['pendulum',num2str(k),'dof']).mumpse_mumpsa.all;CPU_ratio.(['pendulum',num2str(k),'dof']).ma27e_ma27a.all;...
        CPU_ratio.(['pendulum',num2str(k),'dof']).ma57e_ma57a.all;CPU_ratio.(['pendulum',num2str(k),'dof']).ma77e_ma77a.all;...
        CPU_ratio.(['pendulum',num2str(k),'dof']).ma86e_ma86a.all;CPU_ratio.(['pendulum',num2str(k),'dof']).ma97e_ma97a.all];
    CPU_ratio.(['pendulum',num2str(k),'dof']).alle_alla.mean = nanmean(CPU_ratio.(['pendulum',num2str(k),'dof']).alle_alla.all,1);
    CPU_ratio.(['pendulum',num2str(k),'dof']).alle_alla.std = nanstd(CPU_ratio.(['pendulum',num2str(k),'dof']).alle_alla.all,[],1);   
    % Gather all solvers
    CPU_ratio.allpendulum.alle_alla.all = [CPU_ratio.allpendulum.alle_alla.all;CPU_ratio.(['pendulum',num2str(k),'dof']).alle_alla.all];
end
CPU_ratio.allpendulum.alle_alla.all = CPU_ratio.allpendulum.alle_alla.all(:,end);
CPU_ratio.allpendulum.alle_alla.mean = nanmean(1./CPU_ratio.allpendulum.alle_alla.all,1);
CPU_ratio.allpendulum.alle_alla.std = nanstd(1./CPU_ratio.allpendulum.alle_alla.all,[],1);


%% Differences in number of iterations between cases
% Combine results from PredSim 2D and TrackSim 3D and Pendulums
n_iter_all.pred2D_track3D_pend.mumps.approx.all = [];
n_iter_all.pred2D_track3D_pend.mumps.exact.all = [];
n_iter_all.pred2D_track3D_pend.ma27.approx.all = [];
n_iter_all.pred2D_track3D_pend.ma27.exact.all = [];
n_iter_all.pred2D_track3D_pend.ma57.approx.all = [];
n_iter_all.pred2D_track3D_pend.ma57.exact.all = [];
n_iter_all.pred2D_track3D_pend.ma77.approx.all = [];
n_iter_all.pred2D_track3D_pend.ma77.exact.all = [];
n_iter_all.pred2D_track3D_pend.ma86.approx.all = [n_iter_all.pred2D.ma86.approx.all];
n_iter_all.pred2D_track3D_pend.ma86.exact.all = [n_iter_all.pred2D.ma86.exact.all];
n_iter_all.pred2D_track3D_pend.ma97.approx.all = [n_iter_all.pred2D.ma97.approx.all];
n_iter_all.pred2D_track3D_pend.ma97.exact.all = [n_iter_all.pred2D.ma97.exact.all];
n_iter_all.pred2D_track3D_pend.all.approx.all = [n_iter_all.pred2D.ma86.approx.all;n_iter_all.pred2D.ma97.approx.all];
n_iter_all.pred2D_track3D_pend.all.exact.all = [n_iter_all.pred2D.ma86.exact.all;n_iter_all.pred2D.ma97.exact.all];
for k = 2:length(ww_pend)+1 
    n_iter_all.pred2D_track3D_pend.mumps.approx.all = [n_iter_all.pred2D_track3D_pend.mumps.approx.all;n_iter_all.(['pendulum',num2str(k),'dof']).mumps.approx.all];
    n_iter_all.pred2D_track3D_pend.ma27.approx.all = [n_iter_all.pred2D_track3D_pend.ma27.approx.all;n_iter_all.(['pendulum',num2str(k),'dof']).ma27.approx.all];
    n_iter_all.pred2D_track3D_pend.ma57.approx.all = [n_iter_all.pred2D_track3D_pend.ma57.approx.all;n_iter_all.(['pendulum',num2str(k),'dof']).ma57.approx.all];
    n_iter_all.pred2D_track3D_pend.ma77.approx.all = [n_iter_all.pred2D_track3D_pend.ma77.approx.all;n_iter_all.(['pendulum',num2str(k),'dof']).ma77.approx.all];
    n_iter_all.pred2D_track3D_pend.ma86.approx.all = [n_iter_all.pred2D_track3D_pend.ma86.approx.all;n_iter_all.(['pendulum',num2str(k),'dof']).ma86.approx.all];
    n_iter_all.pred2D_track3D_pend.ma97.approx.all = [n_iter_all.pred2D_track3D_pend.ma97.approx.all;n_iter_all.(['pendulum',num2str(k),'dof']).ma97.approx.all];    
    n_iter_all.pred2D_track3D_pend.mumps.exact.all = [n_iter_all.pred2D_track3D_pend.mumps.exact.all;n_iter_all.(['pendulum',num2str(k),'dof']).mumps.exact.all];
    n_iter_all.pred2D_track3D_pend.ma27.exact.all = [n_iter_all.pred2D_track3D_pend.ma27.exact.all;n_iter_all.(['pendulum',num2str(k),'dof']).ma27.exact.all];
    n_iter_all.pred2D_track3D_pend.ma57.exact.all = [n_iter_all.pred2D_track3D_pend.ma57.exact.all;n_iter_all.(['pendulum',num2str(k),'dof']).ma57.exact.all];
    n_iter_all.pred2D_track3D_pend.ma77.exact.all = [n_iter_all.pred2D_track3D_pend.ma77.exact.all;n_iter_all.(['pendulum',num2str(k),'dof']).ma77.exact.all];
    n_iter_all.pred2D_track3D_pend.ma86.exact.all = [n_iter_all.pred2D_track3D_pend.ma86.exact.all;n_iter_all.(['pendulum',num2str(k),'dof']).ma86.exact.all];
    n_iter_all.pred2D_track3D_pend.ma97.exact.all = [n_iter_all.pred2D_track3D_pend.ma97.exact.all;n_iter_all.(['pendulum',num2str(k),'dof']).ma97.exact.all];
    n_iter_all.pred2D_track3D_pend.all.approx.all = [n_iter_all.pred2D_track3D_pend.all.approx.all;n_iter_all.(['pendulum',num2str(k),'dof']).mumps.approx.all;...
        n_iter_all.(['pendulum',num2str(k),'dof']).ma27.approx.all;n_iter_all.(['pendulum',num2str(k),'dof']).ma57.approx.all;...
        n_iter_all.(['pendulum',num2str(k),'dof']).ma77.approx.all;n_iter_all.(['pendulum',num2str(k),'dof']).ma86.approx.all;...
        n_iter_all.(['pendulum',num2str(k),'dof']).ma97.approx.all];
    n_iter_all.pred2D_track3D_pend.all.exact.all = [n_iter_all.pred2D_track3D_pend.all.exact.all;n_iter_all.(['pendulum',num2str(k),'dof']).mumps.exact.all;...
        n_iter_all.(['pendulum',num2str(k),'dof']).ma27.exact.all;n_iter_all.(['pendulum',num2str(k),'dof']).ma57.exact.all;...
        n_iter_all.(['pendulum',num2str(k),'dof']).ma77.exact.all;n_iter_all.(['pendulum',num2str(k),'dof']).ma86.exact.all;...
        n_iter_all.(['pendulum',num2str(k),'dof']).ma97.exact.all];
end
% Analysis at the level of the solvers
% Calculate ratio between mumps.exact and mumps.approx
iter_ratio.pred2D_track3D_pend.mumpse_mumpsa.all = n_iter_all.pred2D_track3D_pend.mumps.exact.all./n_iter_all.pred2D_track3D_pend.mumps.approx.all;
iter_ratio.pred2D_track3D_pend.mumpse_mumpsa.mean = nanmean(iter_ratio.pred2D_track3D_pend.mumpse_mumpsa.all,1);
iter_ratio.pred2D_track3D_pend.mumpse_mumpsa.std = nanstd(iter_ratio.pred2D_track3D_pend.mumpse_mumpsa.all,[],1);
% Calculate ratio between ma27.exact and ma27.approx
iter_ratio.pred2D_track3D_pend.ma27e_ma27a.all = n_iter_all.pred2D_track3D_pend.ma27.exact.all./n_iter_all.pred2D_track3D_pend.ma27.approx.all;
iter_ratio.pred2D_track3D_pend.ma27e_ma27a.mean = nanmean(iter_ratio.pred2D_track3D_pend.ma27e_ma27a.all,1);
iter_ratio.pred2D_track3D_pend.ma27e_ma27a.std = nanstd(iter_ratio.pred2D_track3D_pend.ma27e_ma27a.all,[],1);
% Calculate ratio between ma57.exact and ma57.approx
iter_ratio.pred2D_track3D_pend.ma57e_ma57a.all = n_iter_all.pred2D_track3D_pend.ma57.exact.all./n_iter_all.pred2D_track3D_pend.ma57.approx.all;
iter_ratio.pred2D_track3D_pend.ma57e_ma57a.mean = nanmean(iter_ratio.pred2D_track3D_pend.ma57e_ma57a.all,1);
iter_ratio.pred2D_track3D_pend.ma57e_ma57a.std = nanstd(iter_ratio.pred2D_track3D_pend.ma57e_ma57a.all,[],1);
% Calculate ratio between ma77.exact and ma77.approx
iter_ratio.pred2D_track3D_pend.ma77e_ma77a.all = n_iter_all.pred2D_track3D_pend.ma77.exact.all./n_iter_all.pred2D_track3D_pend.ma77.approx.all;
iter_ratio.pred2D_track3D_pend.ma77e_ma77a.mean = nanmean(iter_ratio.pred2D_track3D_pend.ma77e_ma77a.all,1);
iter_ratio.pred2D_track3D_pend.ma77e_ma77a.std = nanstd(iter_ratio.pred2D_track3D_pend.ma77e_ma77a.all,[],1);
% Calculate ratio between ma86.exact and ma86.approx
iter_ratio.pred2D_track3D_pend.ma86e_ma86a.all = n_iter_all.pred2D_track3D_pend.ma86.exact.all./n_iter_all.pred2D_track3D_pend.ma86.approx.all;
iter_ratio.pred2D_track3D_pend.ma86e_ma86a.mean = nanmean(iter_ratio.pred2D_track3D_pend.ma86e_ma86a.all,1);
iter_ratio.pred2D_track3D_pend.ma86e_ma86a.std = nanstd(iter_ratio.pred2D_track3D_pend.ma86e_ma86a.all,[],1);
% Calculate ratio between ma97.exact and ma97.approx
iter_ratio.pred2D_track3D_pend.ma97e_ma97a.all = n_iter_all.pred2D_track3D_pend.ma97.exact.all./n_iter_all.pred2D_track3D_pend.ma97.approx.all;
iter_ratio.pred2D_track3D_pend.ma97e_ma97a.mean = nanmean(iter_ratio.pred2D_track3D_pend.ma97e_ma97a.all,1);
iter_ratio.pred2D_track3D_pend.ma97e_ma97a.std = nanstd(iter_ratio.pred2D_track3D_pend.ma97e_ma97a.all,[],1);
% Calculate ratio between all.exact and all.approx
iter_ratio.pred2D_track3D_pend.maalle_maalla.all = n_iter_all.pred2D_track3D_pend.all.exact.all./n_iter_all.pred2D_track3D_pend.all.approx.all;
iter_ratio.pred2D_track3D_pend.maalle_maalla.mean = nanmean(iter_ratio.pred2D_track3D_pend.maalle_maalla.all,1);
iter_ratio.pred2D_track3D_pend.maalle_maalla.std = nanstd(iter_ratio.pred2D_track3D_pend.maalle_maalla.all,[],1);
% Analysis at the level of the problems
% PredSim 2D
% Calculate ratio between ma86.exact and ma86.approx
iter_ratio.pred2D.ma86e_ma86a.all = n_iter_all.pred2D.ma86.exact.all./n_iter_all.pred2D.ma86.approx.all;
iter_ratio.pred2D.ma86e_ma86a.mean = nanmean(iter_ratio.pred2D.ma86e_ma86a.all,1);
iter_ratio.pred2D.ma86e_ma86a.std = nanstd(iter_ratio.pred2D.ma86e_ma86a.all,[],1);
% Calculate ratio between ma97.exact and ma97.approx
iter_ratio.pred2D.ma97e_ma97a.all = n_iter_all.pred2D.ma97.exact.all./n_iter_all.pred2D.ma97.approx.all;
iter_ratio.pred2D.ma97e_ma97a.mean = nanmean(iter_ratio.pred2D.ma97e_ma97a.all,1);
iter_ratio.pred2D.ma97e_ma97a.std = nanstd(iter_ratio.pred2D.ma97e_ma97a.all,[],1);
% Calculate ratio between all.exact and all.approx
iter_ratio.pred2D.alle_alla.all = [iter_ratio.pred2D.ma86e_ma86a.all;iter_ratio.pred2D.ma97e_ma97a.all];
iter_ratio.pred2D.alle_alla.mean = nanmean(iter_ratio.pred2D.alle_alla.all,1);
iter_ratio.pred2D.alle_alla.std = nanstd(iter_ratio.pred2D.alle_alla.all,[],1);
% Pendulums
iter_ratio.allpendulum.alle_alla.all = [];
for k = 2:length(ww_pend)+1 
    % Calculate ratio between mumps.exact and mumps.approx
    iter_ratio.(['pendulum',num2str(k),'dof']).mumpse_mumpsa.all = n_iter_all.(['pendulum',num2str(k),'dof']).mumps.exact.all./n_iter_all.(['pendulum',num2str(k),'dof']).mumps.approx.all;
    iter_ratio.(['pendulum',num2str(k),'dof']).mumpse_mumpsa.mean = nanmean(iter_ratio.(['pendulum',num2str(k),'dof']).mumpse_mumpsa.all,1);
    iter_ratio.(['pendulum',num2str(k),'dof']).mumpse_mumpsa.std = nanstd(iter_ratio.(['pendulum',num2str(k),'dof']).mumpse_mumpsa.all,[],1);
    % Calculate ratio between ma27.exact and ma27.approx
    iter_ratio.(['pendulum',num2str(k),'dof']).ma27e_ma27a.all = n_iter_all.(['pendulum',num2str(k),'dof']).ma27.exact.all./n_iter_all.(['pendulum',num2str(k),'dof']).ma27.approx.all;
    iter_ratio.(['pendulum',num2str(k),'dof']).ma27e_ma27a.mean = nanmean(iter_ratio.(['pendulum',num2str(k),'dof']).ma27e_ma27a.all,1);
    iter_ratio.(['pendulum',num2str(k),'dof']).ma27e_ma27a.std = nanstd(iter_ratio.(['pendulum',num2str(k),'dof']).ma27e_ma27a.all,[],1);
    % Calculate ratio between ma57.exact and ma57.approx
    iter_ratio.(['pendulum',num2str(k),'dof']).ma57e_ma57a.all = n_iter_all.(['pendulum',num2str(k),'dof']).ma57.exact.all./n_iter_all.(['pendulum',num2str(k),'dof']).ma57.approx.all;
    iter_ratio.(['pendulum',num2str(k),'dof']).ma57e_ma57a.mean = nanmean(iter_ratio.(['pendulum',num2str(k),'dof']).ma57e_ma57a.all,1);
    iter_ratio.(['pendulum',num2str(k),'dof']).ma57e_ma57a.std = nanstd(iter_ratio.(['pendulum',num2str(k),'dof']).ma57e_ma57a.all,[],1);
    % Calculate ratio between ma77.exact and ma77.approx
    iter_ratio.(['pendulum',num2str(k),'dof']).ma77e_ma77a.all = n_iter_all.(['pendulum',num2str(k),'dof']).ma77.exact.all./n_iter_all.(['pendulum',num2str(k),'dof']).ma77.approx.all;
    iter_ratio.(['pendulum',num2str(k),'dof']).ma77e_ma77a.mean = nanmean(iter_ratio.(['pendulum',num2str(k),'dof']).ma77e_ma77a.all,1);
    iter_ratio.(['pendulum',num2str(k),'dof']).ma77e_ma77a.std = nanstd(iter_ratio.(['pendulum',num2str(k),'dof']).ma77e_ma77a.all,[],1);
    % Calculate ratio between ma86.exact and ma86.approx
    iter_ratio.(['pendulum',num2str(k),'dof']).ma86e_ma86a.all = n_iter_all.(['pendulum',num2str(k),'dof']).ma86.exact.all./n_iter_all.(['pendulum',num2str(k),'dof']).ma86.approx.all;
    iter_ratio.(['pendulum',num2str(k),'dof']).ma86e_ma86a.mean = nanmean(iter_ratio.(['pendulum',num2str(k),'dof']).ma86e_ma86a.all,1);
    iter_ratio.(['pendulum',num2str(k),'dof']).ma86e_ma86a.std = nanstd(iter_ratio.(['pendulum',num2str(k),'dof']).ma86e_ma86a.all,[],1);
    % Calculate ratio between ma97.exact and ma97.approx
    iter_ratio.(['pendulum',num2str(k),'dof']).ma97e_ma97a.all = n_iter_all.(['pendulum',num2str(k),'dof']).ma97.exact.all./n_iter_all.(['pendulum',num2str(k),'dof']).ma97.approx.all;
    iter_ratio.(['pendulum',num2str(k),'dof']).ma97e_ma97a.mean = nanmean(iter_ratio.(['pendulum',num2str(k),'dof']).ma97e_ma97a.all,1);
    iter_ratio.(['pendulum',num2str(k),'dof']).ma97e_ma97a.std = nanstd(iter_ratio.(['pendulum',num2str(k),'dof']).ma97e_ma97a.all,[],1);
    % Calculate ratio between all.exact and all.approx
    iter_ratio.(['pendulum',num2str(k),'dof']).alle_alla.all = [iter_ratio.(['pendulum',num2str(k),'dof']).mumpse_mumpsa.all;iter_ratio.(['pendulum',num2str(k),'dof']).ma27e_ma27a.all;...
        iter_ratio.(['pendulum',num2str(k),'dof']).ma57e_ma57a.all;iter_ratio.(['pendulum',num2str(k),'dof']).ma77e_ma77a.all;...
        iter_ratio.(['pendulum',num2str(k),'dof']).ma86e_ma86a.all;iter_ratio.(['pendulum',num2str(k),'dof']).ma97e_ma97a.all];
    iter_ratio.(['pendulum',num2str(k),'dof']).alle_alla.mean = nanmean(iter_ratio.(['pendulum',num2str(k),'dof']).alle_alla.all,1);
    iter_ratio.(['pendulum',num2str(k),'dof']).alle_alla.std = nanstd(iter_ratio.(['pendulum',num2str(k),'dof']).alle_alla.all,[],1);    
    % Gather all solvers
    iter_ratio.allpendulum.alle_alla.all = [iter_ratio.allpendulum.alle_alla.all;iter_ratio.(['pendulum',num2str(k),'dof']).alle_alla.all];
end
iter_ratio.allpendulum.alle_alla.all = iter_ratio.allpendulum.alle_alla.all(:,end);
iter_ratio.allpendulum.alle_alla.mean = nanmean(1./iter_ratio.allpendulum.alle_alla.all,1);
iter_ratio.allpendulum.alle_alla.std = nanstd(1./iter_ratio.allpendulum.alle_alla.all,[],1);

%% Difference in regularization
% Pred Sim 2D
reg_all.pred2D.all.approx.all = [reg_all.pred2D.ma86.approx.all;reg_all.pred2D.ma97.approx.all];
reg_all.pred2D.all.approx.mean = nanmean(reg_all.pred2D.all.approx.all,1);
reg_all.pred2D.all.approx.std = nanstd(reg_all.pred2D.all.approx.all,[],1);
reg_all.pred2D.all.exact.all = [reg_all.pred2D.ma86.exact.all;reg_all.pred2D.ma97.exact.all];
reg_all.pred2D.all.exact.mean = nanmean(reg_all.pred2D.all.exact.all,1);
reg_all.pred2D.all.exact.std = nanstd(reg_all.pred2D.all.exact.all,[],1);
reg_ratio.pred2D.all.exact_approx.all = reg_all.pred2D.all.exact.all./reg_all.pred2D.all.approx.all;
reg_ratio.pred2D.all.exact_approx.mean = nanmean(reg_ratio.pred2D.all.exact_approx.all,1);
reg_ratio.pred2D.all.exact_approx.std = nanstd(reg_ratio.pred2D.all.exact_approx.all,[],1);
% Pendulums
for k = 2:length(ww_pend)+1 
    reg_all.(['pendulum',num2str(k),'dof']).all.approx.all = [reg_all.(['pendulum',num2str(k),'dof']).mumps.approx.all;...
        reg_all.(['pendulum',num2str(k),'dof']).ma27.approx.all;reg_all.(['pendulum',num2str(k),'dof']).ma57.approx.all;...
        reg_all.(['pendulum',num2str(k),'dof']).ma77.approx.all;reg_all.(['pendulum',num2str(k),'dof']).ma86.approx.all;...
        reg_all.(['pendulum',num2str(k),'dof']).ma97.approx.all];
    reg_all.(['pendulum',num2str(k),'dof']).all.approx.mean = nanmean(reg_all.(['pendulum',num2str(k),'dof']).all.approx.all,1);
    reg_all.(['pendulum',num2str(k),'dof']).all.approx.std = nanstd(reg_all.(['pendulum',num2str(k),'dof']).all.approx.all,[],1);
    
    reg_all.(['pendulum',num2str(k),'dof']).all.exact.all = [reg_all.(['pendulum',num2str(k),'dof']).mumps.exact.all;...
        reg_all.(['pendulum',num2str(k),'dof']).ma27.exact.all;reg_all.(['pendulum',num2str(k),'dof']).ma57.exact.all;...
        reg_all.(['pendulum',num2str(k),'dof']).ma77.exact.all;reg_all.(['pendulum',num2str(k),'dof']).ma86.exact.all;...
        reg_all.(['pendulum',num2str(k),'dof']).ma97.exact.all];
    reg_all.(['pendulum',num2str(k),'dof']).all.exact.mean = nanmean(reg_all.(['pendulum',num2str(k),'dof']).all.exact.all,1);
    reg_all.(['pendulum',num2str(k),'dof']).all.exact.std = nanstd(reg_all.(['pendulum',num2str(k),'dof']).all.exact.all,[],1);
    
    reg_ratio.(['pendulum',num2str(k),'dof']).all.exact_approx.all = reg_all.(['pendulum',num2str(k),'dof']).all.exact.all./reg_all.(['pendulum',num2str(k),'dof']).all.approx.all;
    reg_ratio.(['pendulum',num2str(k),'dof']).all.exact_approx.mean = nanmean(reg_ratio.(['pendulum',num2str(k),'dof']).all.exact_approx.all,1);
    reg_ratio.(['pendulum',num2str(k),'dof']).all.exact_approx.std = nanstd(reg_ratio.(['pendulum',num2str(k),'dof']).all.exact_approx.all,[],1);
end


%% Plots: 2 studied cases merged
label_fontsize  = 18;
sup_fontsize  = 24;
line_linewidth  = 3;
ylim_CPU = [0 8];
NumTicks_CPU = 3;
ylim_iter = [0 3];
NumTicks_iter = 4;
ylim_reg = [0 10^20];
NumTicks_reg = 5;

figure()
subplot(3,2,1)
CPU_ratio_4plots.approx_exact.mean = zeros(length(ww_pend),1);
CPU_ratio_4plots.approx_exact.std = zeros(length(ww_pend),1);
for k = 2:length(ww_pend)+1
    CPU_ratio_4plots.approx_exact.mean(k-1,1) = CPU_ratio.(['pendulum',num2str(k),'dof']).alle_alla.mean(end);
    CPU_ratio_4plots.approx_exact.std(k-1,1) = CPU_ratio.(['pendulum',num2str(k),'dof']).alle_alla.std(end);
end
CPU_ratio_4plots.approx_exact.mean(k,1) = CPU_ratio.pred2D.alle_alla.mean(end);
CPU_ratio_4plots.approx_exact.std(k,1) = CPU_ratio.pred2D.alle_alla.std(end);
h1 = barwitherr(CPU_ratio_4plots.approx_exact.std,CPU_ratio_4plots.approx_exact.mean);
set(h1(1),'FaceColor',color_all(4,:),'EdgeColor',color_all(4,:),'BarWidth',0.4);
hold on;
L = get(gca,'XLim');
plot([L(1) L(2)],[1 1],'k','linewidth',1);
set(gca,'Fontsize',label_fontsize);  
l = legend('Exact Hessian / Approximated Hessian');
set(gca,'Fontsize',label_fontsize);  
set(l,'Fontsize',label_fontsize); 
set(l,'location','Northwest');
set(gca,'XTickLabel',{'','','','','','','','','','',''},'Fontsize',label_fontsize');
ylabel('CPU time','Fontsize',label_fontsize');
ylim([ylim_CPU(1) ylim_CPU(2)]);
L = get(gca,'YLim');
set(gca,'YTick',linspace(L(1),L(2),NumTicks_CPU));
set(gca,'Fontsize',label_fontsize);  
box off;

subplot(3,2,3)
iter_ratio_4plots.approx_exact.mean = zeros(length(ww_pend),1);
iter_ratio_4plots.approx_exact.std = zeros(length(ww_pend),1);
for k = 2:length(ww_pend)+1
    iter_ratio_4plots.approx_exact.mean(k-1,1) = iter_ratio.(['pendulum',num2str(k),'dof']).alle_alla.mean(end);
    iter_ratio_4plots.approx_exact.std(k-1,1) = iter_ratio.(['pendulum',num2str(k),'dof']).alle_alla.std(end);
end
iter_ratio_4plots.approx_exact.mean(k,1) = iter_ratio.pred2D.alle_alla.mean(end);
iter_ratio_4plots.approx_exact.std(k,1) = iter_ratio.pred2D.alle_alla.std(end);
h2 = barwitherr(iter_ratio_4plots.approx_exact.std,iter_ratio_4plots.approx_exact.mean);
set(h2(1),'FaceColor',color_all(4,:),'EdgeColor',color_all(4,:),'BarWidth',0.4);
hold on;
L = get(gca,'XLim');
plot([L(1) L(2)],[1 1],'k','linewidth',1);
set(gca,'Fontsize',label_fontsize);  
set(gca,'XTickLabel',{'','','','','','','','','','',''},'Fontsize',label_fontsize');
ylabel('Iterations','Fontsize',label_fontsize');
ylim([ylim_iter(1) ylim_iter(2)]);
L = get(gca,'YLim');
set(gca,'YTick',linspace(L(1),L(2),NumTicks_iter));
box off;

% subplot(2,2,3)
% reg_4plots.approx_exact.mean = zeros(length(ww_pend),2);
% reg_4plots.approx_exact.std = zeros(length(ww_pend),2);
% for k = 2:length(ww_pend)+1
%     reg_4plots.approx_exact.mean(k-1,1) = reg_all.(['pendulum',num2str(k),'dof']).all.approx.mean;
%     reg_4plots.approx_exact.std(k-1,1) = reg_all.(['pendulum',num2str(k),'dof']).all.approx.std;
%     reg_4plots.approx_exact.mean(k-1,2) = reg_all.(['pendulum',num2str(k),'dof']).all.exact.mean;
%     reg_4plots.approx_exact.std(k-1,2) = reg_all.(['pendulum',num2str(k),'dof']).all.exact.std;
% end
% reg_4plots.approx_exact.mean(k,1) = reg_all.pred2D.all.approx.mean;
% reg_4plots.approx_exact.std(k,1) = reg_all.pred2D.all.approx.std;
% reg_4plots.approx_exact.mean(k,2) = reg_all.pred2D.all.exact.mean;
% reg_4plots.approx_exact.std(k,2) = reg_all.pred2D.all.exact.std;
% h3 = barwitherr(reg_4plots.approx_exact.std,reg_4plots.approx_exact.mean);
% set(gca,'Fontsize',label_fontsize);  
% set(gca,'XTickLabel',{'P2','P3','P4','P5','P6','P7','P8','P9','P10','Pred'},'Fontsize',label_fontsize');
% ylabel('Regularization size (mean)','Fontsize',label_fontsize');
% set(gca, 'YScale', 'log')
% % ylim([ylim_reg(1) inf]);
% % L = get(gca,'YLim');
% % set(gca,'YTick',linspace(L(1),L(2),NumTicks_reg));
% box off;
% 
% sp = suptitle('Exact versus approximated Hessian');
% set(sp,'Fontsize',label_fontsize);  

% %% Analyze regularization size
% % Pred Sim 2D
% % ma86: approx Hessian => trial 139
% % ma86: exact Hessian => trial 109
% % ma97: approx Hessian => trial 140
% % ma97: exact Hessian => trial 112
% kk = [3,4,9,10];
% figure()
% for k = 1:length(kk)
% subplot(2,2,k)
% plot(Stats_2D(ww_2D(kk(k))).m.iterations.regularization_size)
% set(gca,'Fontsize',label_fontsize);  
% if k == 1 || k == 3
%     ylabel('Regularization size','Fontsize',label_fontsize');
% end
% if k == 1
%     title('Approximated Hessian: sim1','Fontsize',label_fontsize');
% elseif k == 2
%     title('Exact Hessian: sim1','Fontsize',label_fontsize');
% elseif k == 3
%     title('Approximated Hessian: sim2','Fontsize',label_fontsize');
% elseif k == 4
%     title('Exact Hessian: sim1','Fontsize',label_fontsize');
% end
% end
% sp = suptitle('Regularization: 2D Predictive Walking Simulations');
% set(sp,'Fontsize',label_fontsize); 

