%%  Two-dimensional muscle-driven predictive simulations of human gaits
%
% Author: Antoine Falisse
% Date: 9/9/2019
%
% DEPENDENCY: please install CasADi (https://web.casadi.org/)
%
% This code was developed using MATLAB R2017b, you might run into some issues
% with other versions (or if you do not have all MATLAB toolboxes). Please
% let us know so that we can make the code better.
%
clear all;
clc
close all;

%% User inputs
% This script can be run to solve the optimal control problems but also to
% analyze and process the results. The user does not need to re-run the
% optimal control problems to analyze the results. Therefore, the user can
% select some settings beforehand through the variable num_set. For
% example, when num_set(1) is set to 0, the script will not run the
% optimal control problem. Here is a brief description of num_set:
% num_set(1): set to 1 to solve problem
% num_set(2): set to 1 to analyze results
% num_set(3): set to 1 to load results
% num_set(4): set to 1 to save results
% num_set(5): set to 1 to visualize guess-bounds 
% num_set(6): set to 1 to write .mot file

% num_set = [1,1,0,1,0,1]; % This configuration solves the problem
num_set = [0,1,1,1,0,1]; % This configuration analyzes the results

% The variable settings in the following section (loaded through
% settings_PredSim_2D) will set some parameters of the optimal control problems.
% Through the variable idx_ww, the user can select which row of parameters will
% be used.
% NOTE: at this stage, we only provide the AD-Recorder and FD cases,
% providing the libraries built with ADOL-C is harder (and less relevant, since
% less efficient). Therefore, the ADOL-C cases cannot be run (cases 4-6).
% Further, we are not allowed to share the HSL libraries. Therefore, only the
% cases with the mumps linear solver can be run (cases 1-3, 7-9, and 31-32).
idx_ww = 4; % Index row in matrix settings

%% Settings
import casadi.*
subject = 'subject1';

solveProblem    = num_set(1); % set to 1 to solve problem
analyseResults  = num_set(2); % set to 1 to analyze results
loadResults     = num_set(3); % set to 1 to load results
saveResults     = num_set(4); % set to 1 to save sens. results
checkBoundsIG   = num_set(5); % set to 1 to visualize guess-bounds 
writeIKmotion   = num_set(6); % set to 1 to write .mot file

pathmain = pwd;
[pathRepo,~,~] = fileparts(pathmain);
pathSettings = [pathRepo,'/Settings'];
addpath(genpath(pathSettings));
% Load settings
settings_2D

%% Select settings
for www = 1:length(idx_ww)
    
%% Set parameters based on settings    
ww = idx_ww(www);
% Variable parameters
deri        = settings(ww,1);   % derivative supplier identifier
hessi       = settings(ww,2);   % Hessian identifier
linsoli     = settings(ww,3);   % linear solver identifier
exp_E       = settings(ww,4);   % power metabolic energy rate
IGi         = settings(ww,5);   % initial guess identifier
N           = settings(ww,6);   % number of mesh intervals
% Fixed parameter
v_tgt       = 1.33;	% average speed
tol_ipopt   = 6;    % tolerance (means 1e-(tol_ipopt))
W.act       = 1;    % weight muscle activations
W.back      = 1;    % weight back excitations
W.acc       = 1;    % weight joint accelerations
exp_A       = 3;    % power muscle activations
W.mE        = 0;    % weight metabolic energy rate
if exp_E ~= 0
    W.mE = 0.0001;
end
W.u = 0.001;
% Derivative supplier
if deri == 1
    setup.derivatives = 'AD_Recorder'; % Algorithmic differentiation    
elseif deri == 2
    setup.derivatives = 'AD_ADOLC'; % Algorithmic differentiation 
elseif deri == 3
    setup.derivatives = 'FD'; % Finite differences
end
% Available linear solvers
linear_solvers = {'mumps','ma27','ma57','ma77','ma86','ma97'}; 
% The filename used to save the results depends on the settings 
savename = ['_c',num2str(ww)];

%% Load external functions
% The external function performs inverse dynamics through the
% OpenSim/Simbody C++ API. This external function is compiled as a dll from
% which we create a Function instance using CasADi in MATLAB. More details
% about the external function can be found in the documentation.
% We use different external functions, since we also want to access some 
% parameters of the model in a post-processing phase.
% AD-Recorder or FD
if deri == 1 || deri == 3
    pathexternal = [pathRepo,'/ExternalFunctions'];
% AD-ADOLC
elseif deri == 2
    disp('ADOL-C cases not available')
    break;
end
if analyseResults
    pathexternal1 = [pathRepo,'/ExternalFunctions'];
end 
% Load external functions
cd(pathexternal);
if ispc
switch setup.derivatives
    case 'AD_Recorder' % Algorithmic differentiation with Recorder
        if hessi == 1 % Approximated Hessian
            F = external('F','PredSim_2D.dll');  
        elseif hessi == 2 % Exact Hessian
            F = external('F','PredSim_2D_exactHessian.dll');
        end
        if analyseResults
            cd(pathexternal1);
            F1 = external('F','PredSim_2D_pp.dll'); 
        end
    case 'AD_ADOLC' % Algorithmic differentiation with ADOL-C
        if hessi == 1
            % ADOL-C cases not available
        end
        if analyseResults
            cd(pathexternal1);
            F1 = external('F','PredSim_2D_pp.dll'); 
        end
    case 'FD' % Finite differences
        if hessi == 1
            F = external('F','PredSim_2D.dll',struct('enable_fd',...
                true,'enable_forward',false,'enable_reverse',false,...
                'enable_jacobian',false,'fd_method','forward'));
        end
        if analyseResults
            cd(pathexternal1);
            F1 = external('F','PredSim_2D_pp.dll',struct(...
                'enable_fd',true,'enable_forward',false,...
                'enable_reverse',false,'enable_jacobian',false,...
                'fd_method','forward'));
        end
end
else
    disp('Currently no support for MAC or Linux, see documentation')
end
cd(pathmain);
% This is an example of how to call an external function with numerical values.
% vec1 = -ones(30,1);
% res1 = full(F(vec1));
% F_fwd = F.forward(1);
% res2 = full(F_fwd(vec1,res1,vec1+0.1));
% F_adj = F.reverse(1);
% res3 = full(F_adj(vec1,res1,res1+0.1));

%% Indices external function
% Indices of the elements in the external functions
% External function: F
% First, joint torques. 
jointi.pelvis.tilt  = 1; 
jointi.pelvis.tx    = 2;
jointi.pelvis.ty    = 3;
jointi.hip.l        = 4;
jointi.hip.r        = 5;
jointi.knee.l       = 6;
jointi.knee.r       = 7;
jointi.ankle.l      = 8;
jointi.ankle.r      = 9;
jointi.trunk.ext    = 10;
% Vectors of indices for later use
jointi.all          = jointi.pelvis.tilt:jointi.trunk.ext; % all 
jointi.gr_pelvis    = jointi.pelvis.tilt:jointi.pelvis.ty; % ground-pelvis
% Number of degrees of freedom for later use
nq.all              = length(jointi.all); % all
nq.abs              = length(jointi.gr_pelvis); % ground-pelvis
nq.leg              = 3; % #joints needed for polynomials
nq.trunk            = 1; % trunk
% External function: F1 (post-processing purpose only)
% Ground reaction forces (GRFs)
GRFi.r              = 11:12;
GRFi.l              = 13:14;
GRFi.all            = [GRFi.r,GRFi.l];
nGRF                = length(GRFi.all);

% Helper variable to impose periodic/symmetric constraints
% Qs and Qdots are inverted after a half gait cycle BUT
% Pelvis: pelvis tilt and pelvis ty should be equal
% Trunk: trunk ext should be equal
orderQsInv = [
    2*jointi.pelvis.tilt-1:2*jointi.pelvis.ty,...
    2*jointi.hip.r-1:2*jointi.hip.r,...
    2*jointi.hip.l-1:2*jointi.hip.l,...
    2*jointi.knee.r-1:2*jointi.knee.r,...
    2*jointi.knee.l-1:2*jointi.knee.l,...
    2*jointi.ankle.r-1:2*jointi.ankle.r,...
    2*jointi.ankle.l-1:2*jointi.ankle.l,...
    2*jointi.trunk.ext-1:2*jointi.trunk.ext];  

%% Model info
body_mass = 62;
body_weight = body_mass*9.81;

%% Collocation scheme
% We use a pseudospectral direct collocation method, i.e. we use Lagrange
% polynomials to approximate the state derivatives at the collocation
% points in each mesh interval. We use d=3 collocation points per mesh
% interval and Radau collocation points. 
pathCollocationScheme = [pathRepo,'/CollocationScheme'];
addpath(genpath(pathCollocationScheme));
d = 3; % degree of interpolating polynomial
method = 'radau'; % collocation method
[tau_root,C,D,B] = CollocationScheme(d,method);

%% Muscle-tendon parameters 
% Muscles from one leg
muscleNames = {'hamstrings_r','bifemsh_r','glut_max_r','iliopsoas_r',...
    'rect_fem_r','vasti_r','gastroc_r','soleus_r','tib_ant_r'};  
% Muscle indices for later use
pathmusclemodel = [pathRepo,'/MuscleModel'];
addpath(genpath(pathmusclemodel));    
musi = MuscleIndices_2D(muscleNames);
NMuscle = length(muscleNames)*2;
% Muscle-tendon parameters. Row 1: maximal isometric forces; Row 2: optimal
% fiber lengths; Row 3: tendon slack lengths; Row 4: optimal pennation 
% angles; Row 5: maximal contraction velocities
load([pathmusclemodel,'/MTparameters_',subject,'.mat']);
MTparameters_m = [MTparameters(:,musi),MTparameters(:,musi)];
% Indices of the muscles actuating the different joints for later use
pathpolynomial = [pathRepo,'/Polynomials'];
addpath(genpath(pathpolynomial));
tl = load([pathpolynomial,'/muscle_spanning_joint_INFO_',subject,'.mat']);
[Indmusi,mai] = MomentArmIndices_2D(muscleNames,...
    tl.muscle_spanning_joint_INFO);
% Parameters for activation dynamics
tact = 0.015; % Activation time constant
tdeact = 0.06; % Deactivation time constant
% Helper variable to impose periodic/symmetric constraints
orderMusInv = [NMuscle/2+1:NMuscle,1:NMuscle/2];

%% Metabolic energy model parameters
% We extract the specific tensions and slow twitch rations.
pathMetabolicEnergy = [pathRepo,'/MetabolicEnergy'];
addpath(genpath(pathMetabolicEnergy));
tension = getSpecificTensions_2D(muscleNames); 
tensions = [tension;tension];
pctst = getSlowTwitchRatios_2D(muscleNames); 
pctsts = [pctst;pctst];

%% CasADi functions
% We create several CasADi functions for later use
pathCasADiFunctions = [pathRepo,'/CasADiFunctions'];
addpath(genpath(pathCasADiFunctions));
% We load some variables for the polynomial approximations
load([pathpolynomial,'/muscle_spanning_joint_INFO_',subject,'.mat']);
load([pathpolynomial,'/MuscleInfo_',subject,'.mat']);
musi_pol = musi;
NMuscle_pol = NMuscle/2;
CasADiFunctions_2D

%% Experimental data
% We extract experimental data to set bounds and initial guesses if needed
pathData = [pathRepo,'/OpenSimModel/',subject];  
joints = {'pelvis_tilt','pelvis_tx','pelvis_ty','hip_flexion_l',...
        'hip_flexion_r','knee_angle_l','knee_angle_r','ankle_angle_l',...
        'ankle_angle_r','lumbar_extension'};
pathVariousFunctions = [pathRepo,'/VariousFunctions'];
addpath(genpath(pathVariousFunctions));
% Extract joint positions from average walking motion
motion_walk         = 'walking';
nametrial_walk.id   = ['motion_ave_',motion_walk];   
nametrial_walk.IK   = ['IK_',nametrial_walk.id];
pathIK_walk         = [pathData,'/IK/',nametrial_walk.IK,'.mot'];
Qs_walk             = getIK(pathIK_walk,joints);
time_IC             = [Qs_walk.time(1),Qs_walk.time(end)];
% Depending on the initial guess mode, we extract further experimental data
if IGi == 3
    % Extract joint positions from average running motion
    motion_run          = 'running';
    nametrial_run.id    = ['motion_ave_',motion_run];  
    nametrial_run.IK    = ['IK_',nametrial_run.id];
    pathIK_run          = [pathData,'/IK/',nametrial_run.IK,'.mot'];
    Qs_run              = getIK(pathIK_run,joints);
    time_IC_run         = [Qs_run.time(1),Qs_run.time(end)];
end

%% Bounds
pathBounds = [pathRepo,'/Bounds'];
addpath(genpath(pathBounds));
[bounds,scaling] = getBounds_2D(NMuscle,nq,jointi);

%% Initial guess
pathIG = [pathRepo,'/Guess'];
addpath(genpath(pathIG));
% The initial guess depends on the settings
if IGi == 1 % Quasi-random initial guess  
    guess = getGuess_2D_QR(N,nq,NMuscle,scaling,v_tgt,jointi);
elseif IGi == 2 % Data-informed initial guess   
    % Initial guess based on walking data    
    guess = getGuess_2D_DI(Qs_walk,nq,N,time_IC,NMuscle,jointi,scaling);
elseif IGi == 3 % Data-informed initial guess   
    % Initial guess based on running data   
    guess = getGuess_2D_DI(Qs_run,nq,N,time_IC_run,NMuscle,jointi,scaling);
end    
% This allows visualizing the initial guess and the bounds
if checkBoundsIG
    pathPlots = [pathRepo,'/Plots'];
    addpath(genpath(pathPlots));
    plot_BoundsVSInitialGuess_2D
end

%% Formulate the NLP
if solveProblem
    % Start with an empty NLP
    w   = {}; % design variables
    w0  = []; % initial guess for design variables
    lbw = []; % lower bounds for design variables
    ubw = []; % upper bounds for design variables
    J   = 0;  % initial value of cost function
    g   = {}; % constraints
    lbg = []; % lower bounds for constraints
    ubg = []; % upper bounds for constraints
    % Define static parameters
    % Final time
    tf              = MX.sym('tf',1);
    w               = [w {tf}];
    lbw             = [lbw; bounds.tf.lower];
    ubw             = [ubw; bounds.tf.upper];
    w0              = [w0;  guess.tf];    
    % TO DISCUSS
    % Define additional controls at first mesh point.
    % These controls are defined at that point only so that we can compute
    % muscle forces and joint torques. The reason for defining that here is
    % that it simplifies retrieving the results post-processing.
    % Time derivative of muscle-tendon forces (states)
    dFTtildek       = MX.sym(['dFTtilde_' num2str(1)], NMuscle);
    w               = [w {dFTtildek}];
    lbw             = [lbw; bounds.dFTtilde.lower'];
    ubw             = [ubw; bounds.dFTtilde.upper'];
    w0              = [w0; guess.dFTtilde(1,:)'];  
    % Time derivative of Qdots (states) 
    Ak              = MX.sym(['A_' num2str(1)], nq.all);
    w               = [w {Ak}];
    lbw             = [lbw; bounds.Qdotdots.lower'];
    ubw             = [ubw; bounds.Qdotdots.upper'];
    w0              = [w0; guess.Qdotdots(1,:)'];     
    % Define states at first mesh point
    % Muscle activations
    a0              = MX.sym('a0',NMuscle);
    w               = [w {a0}];
    lbw             = [lbw; bounds.a.lower'];
    ubw             = [ubw; bounds.a.upper'];
    w0              = [w0;  guess.a(1,:)'];
    % Muscle-tendon forces
    FTtilde0        = MX.sym('FTtilde0',NMuscle);
    w               = [w {FTtilde0}];
    lbw             = [lbw; bounds.FTtilde.lower'];
    ubw             = [ubw; bounds.FTtilde.upper'];
    w0              = [w0;  guess.FTtilde(1,:)'];
    % Qs and Qdots 
    X0              = MX.sym('X0',2*nq.all);
    w               = [w {X0}];    
    lbw             = [lbw; bounds.QsQdots_0.lower'];
    ubw             = [ubw; bounds.QsQdots_0.upper'];    
    w0              = [w0;  guess.QsQdots(1,:)'];
    % Back activations
    a_b0            = MX.sym('a_b0',nq.trunk);
    w               = [w {a_b0}];
    lbw             = [lbw; bounds.a_b.lower'];
    ubw             = [ubw; bounds.a_b.upper'];
    w0              = [w0;  guess.a_b(1,:)'];
    % We pre-allocate some of the states so that we can provide an
    % expression for the distance traveled   
    for k=0:N
        Xk{k+1,1} = MX.sym(['X_' num2str(k+1)], 2*nq.all);
    end 
    % "Lift" initial conditions
    ak          = a0;
    FTtildek    = FTtilde0;
    Xk{1,1}     = X0;
    a_bk        = a_b0; 
    % Provide expression for the distance traveled
    pelvis_tx0 = Xk{1,1}(2*jointi.pelvis.tx-1,1).*...
        scaling.QsQdots(2*jointi.pelvis.tx-1); % initial position pelvis_tx    
    pelvis_txf = Xk{N+1,1}(2*jointi.pelvis.tx-1,1).*...
        scaling.QsQdots(2*jointi.pelvis.tx-1); % final position pelvis_tx 
    dist_trav_tot = pelvis_txf-pelvis_tx0; % distance traveled 
    % Time step
    h = tf/N;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Loop over mesh points
    for k=0:N-1
        % Define controls at mesh point (piecewise-constant in interval) 
        % Time derivative of muscle activations (states)
        vAk                 = MX.sym(['vA_' num2str(k)], NMuscle);
        w                   = [w {vAk}];
        lbw                 = [lbw; bounds.vA.lower'];
        ubw                 = [ubw; bounds.vA.upper'];
        w0                  = [w0; guess.vA(k+1,:)'];  
        % Back excitations
        e_bk                = MX.sym(['e_b_' num2str(k)], nq.trunk);
        w                   = [w {e_bk}];
        lbw                 = [lbw; bounds.e_b.lower'];
        ubw                 = [ubw; bounds.e_b.upper'];
        w0                  = [w0; guess.e_b(k+1,:)'];
        % Define states at collocation points    
        % Muscle activations
        akj = {};
        for j=1:d
            akj{j}  = MX.sym(['	a_' num2str(k) '_' num2str(j)], NMuscle);
            w       = {w{:}, akj{j}};
            lbw     = [lbw; bounds.a.lower'];
            ubw     = [ubw; bounds.a.upper'];
            w0      = [w0;  guess.a(k+1,:)'];
        end   
        % Muscle-tendon forces
        FTtildekj = {};
        for j=1:d
            FTtildekj{j} = ...
                MX.sym(['FTtilde_' num2str(k) '_' num2str(j)], NMuscle);
            w            = {w{:}, FTtildekj{j}};
            lbw          = [lbw; bounds.FTtilde.lower'];
            ubw          = [ubw; bounds.FTtilde.upper'];
            w0           = [w0;  guess.FTtilde(k+1,:)'];
        end
        % Qs and Qdots       
        Xkj = {};
        for j=1:d
            Xkj{j} = MX.sym(['X_' num2str(k) '_' num2str(j)], 2*nq.all);
            w      = {w{:}, Xkj{j}};
            lbw    = [lbw; bounds.QsQdots.lower'];
            ubw    = [ubw; bounds.QsQdots.upper'];
            w0     = [w0;  guess.QsQdots(k+1,:)'];
        end   
        % Back activations
        a_bkj = {};
        for j=1:d
            a_bkj{j}= MX.sym(['	a_b_' num2str(k) '_' num2str(j)], nq.trunk);
            w       = {w{:}, a_bkj{j}};
            lbw     = [lbw; bounds.a_b.lower'];
            ubw     = [ubw; bounds.a_b.upper'];
            w0      = [w0;  guess.a_b(k+1,:)'];
        end   
        % Define additional controls at collocation points ("slack" variables)
        % Time derivative of muscle-tendon forces
        dFTtildekj = {};
        for j=1:d
            dFTtildekj{j}   = ...
                MX.sym(['dFTtilde_' num2str(k) '_' num2str(j)], NMuscle);
            w               = {w{:}, dFTtildekj{j}};
            lbw             = [lbw; bounds.dFTtilde.lower'];
            ubw             = [ubw; bounds.dFTtilde.upper'];
            w0              = [w0;  guess.dFTtilde(k+1,:)'];
        end
        % Time derivative of Qdots
        Akj = {};
        for j=1:d
            Akj{j} = MX.sym(['A_' num2str(k) '_' num2str(j)], nq.all);
            w      = {w{:}, Akj{j}};
            lbw    = [lbw; bounds.Qdotdots.lower'];
            ubw    = [ubw; bounds.Qdotdots.upper'];
            w0     = [w0;  guess.Qdotdots(k+1,:)'];
        end        
        % Unscale variables for later use
        Xk_nsc          = Xk{k+1,1}.*scaling.QsQdots';
        FTtildek_nsc    = FTtildek.*(scaling.FTtilde');        
        vAk_nsc         = vAk.*scaling.vA; 
        for j=1:d
            Xkj_nsc{j}          = Xkj{j}.*scaling.QsQdots';
            FTtildekj_nsc{j}    = FTtildekj{j}.*scaling.FTtilde';  
            dFTtildekj_nsc{j}   = dFTtildekj{j}.*scaling.dFTtilde'; 
            Akj_nsc{j}          = Akj{j}.*scaling.Qdotdots';
        end  
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Add the first mesh point, we impose the Hill-equilibrium using
        % the additional controls at that mesh points. Furthermore, we run
        % inverse dynamics and impose that the joint torques should be
        % driven by muscles. This is done separately for the first mesh
        % point, since that point is not "covered" by the collocation
        % points.
        if k == 0
            % Unscale variables for later use
            dFTtildek_nsc   = dFTtildek.*(scaling.dFTtilde');
            Ak_nsc          = Ak.*scaling.Qdotdots';        
            % Get muscle-tendon lengths, velocities, and moment arms
            % Left leg
            qin_l = [Xk_nsc(jointi.hip.l*2-1,1),...
                Xk_nsc(jointi.knee.l*2-1,1),Xk_nsc(jointi.ankle.l*2-1,1)];  
            qdotin_l = [Xk_nsc(jointi.hip.l*2,1),...
                Xk_nsc(jointi.knee.l*2,1),Xk_nsc(jointi.ankle.l*2,1)];  
            [lMTk_l,vMTk_l,MA_l] = f_lMT_vMT_dM(qin_l,qdotin_l);    
            MA_hip_l    =  MA_l(mai(1).mus.l',1);
            MA_knee_l   =  MA_l(mai(2).mus.l',2);
            MA_ankle_l  =  MA_l(mai(3).mus.l',3);    
            % Right leg
            qin_r = [Xk_nsc(jointi.hip.r*2-1,1),...
                Xk_nsc(jointi.knee.r*2-1,1),Xk_nsc(jointi.ankle.r*2-1,1)];  
            qdotin_r = [Xk_nsc(jointi.hip.r*2,1),...
                Xk_nsc(jointi.knee.r*2,1), Xk_nsc(jointi.ankle.r*2,1)];      
            [lMTk_r,vMTk_r,MA_r] = f_lMT_vMT_dM(qin_r,qdotin_r);  
            % Here we take the indices from left since the vector is 1:NMuscle/2
            MA_hip_r    = MA_r(mai(1).mus.l',1);
            MA_knee_r   = MA_r(mai(2).mus.l',2);
            MA_ankle_r  = MA_r(mai(3).mus.l',3);
            % Both legs
            lMTk_lr     = [lMTk_l;lMTk_r];
            vMTk_lr     = [vMTk_l;vMTk_r];   
            % Get muscle-tendon forces and derive Hill-equilibrium
            [Hilldiffk,FTk,~,~,~,~,~] = f_forceEquilibrium_FtildeState(...
                ak,FTtildek_nsc,dFTtildek_nsc,lMTk_lr,vMTk_lr,tensions); 
            % Call external function (run inverse dynamics)
            if deri == 2
                [Tk] = F(Xk_nsc,Ak_nsc); 
            else
                [Tk] = F([Xk_nsc;Ak_nsc]);      
            end
            % Add path constraints
            % Null pelvis residuals
            g           = {g{:},Tk(jointi.gr_pelvis,1)};
            lbg         = [lbg; zeros(nq.abs,1)];
            ubg         = [ubg; zeros(nq.abs,1)];    
            % Muscle-driven joint torques for the lower limbs
            % Hip flexion, left
            Ft_hip_l    = FTk(mai(1).mus.l',1);
            T_hip_l     = f_T4(MA_hip_l,Ft_hip_l);
            g           = {g{:},(Tk(jointi.hip.l,1)-(T_hip_l))};
            lbg         = [lbg; 0];
            ubg         = [ubg; 0];    
            % Hip flexion, right
            Ft_hip_r    = FTk(mai(1).mus.r',1);
            T_hip_r     = f_T4(MA_hip_r,Ft_hip_r);
            g           = {g{:},(Tk(jointi.hip.r,1)-(T_hip_r))};
            lbg         = [lbg; 0];
            ubg         = [ubg; 0];    
            % Knee, left
            Ft_knee_l   = FTk(mai(2).mus.l',1);
            T_knee_l    = f_T5(MA_knee_l,Ft_knee_l);
            g           = {g{:},(Tk(jointi.knee.l,1)-(T_knee_l))};
            lbg         = [lbg; 0];
            ubg         = [ubg; 0];    
            % Knee, right
            Ft_knee_r   = FTk(mai(2).mus.r',1);
            T_knee_r    = f_T5(MA_knee_r,Ft_knee_r);
            g           = {g{:},(Tk(jointi.knee.r,1)-(T_knee_r))};
            lbg         = [lbg; 0];
            ubg         = [ubg; 0];    
            % Ankle, left
            Ft_ankle_l  = FTk(mai(3).mus.l',1);
            T_ankle_l   = f_T3(MA_ankle_l,Ft_ankle_l);
            g           = {g{:},(Tk(jointi.ankle.l,1)-(T_ankle_l))};
            lbg         = [lbg; 0];
            ubg         = [ubg; 0];    
            % Ankle, right
            Ft_ankle_r  = FTk(mai(3).mus.r',1);
            T_ankle_r   = f_T3(MA_ankle_r,Ft_ankle_r);
            g           = {g{:},(Tk(jointi.ankle.r,1)-(T_ankle_r))};
            lbg         = [lbg; 0];
            ubg         = [ubg; 0];      
            % Torque-driven joint torque for the trunk
            % Trunk
            g           = {g{:},Tk(jointi.trunk.ext,1)./scaling.BackTau-a_bk};
            lbg         = [lbg; 0];
            ubg         = [ubg; 0];
            % Activation dynamics (implicit formulation)
            act1 = vAk_nsc + ak./(ones(size(ak,1),1)*tdeact);
            act2 = vAk_nsc + ak./(ones(size(ak,1),1)*tact);
            % act1
            g               = {g{:},act1};
            lbg             = [lbg; zeros(NMuscle,1)];
            ubg             = [ubg; inf*ones(NMuscle,1)]; 
            % act2
            g               = {g{:},act2};
            lbg             = [lbg; -inf*ones(NMuscle,1)];
            ubg             = [ubg; ones(NMuscle,1)./(ones(NMuscle,1)*tact)];        
            % Contraction dynamics (implicit formulation)
            g               = {g{:},Hilldiffk};
            lbg             = [lbg; zeros(NMuscle,1)];
            ubg             = [ubg; zeros(NMuscle,1)]; 
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Loop over collocation points
        Xk_nsc_end          = D(1)*Xk_nsc;
        FTtildek_nsc_end    = D(1)*FTtildek_nsc;
        ak_end              = D(1)*ak;
        a_bk_end            = D(1)*a_bk;
        for j=1:d   
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Get muscle-tendon lengths, velocities, and moment arms
            % Left leg
            qinkj_l = [Xkj_nsc{j}(jointi.hip.l*2-1,1),...
                Xkj_nsc{j}(jointi.knee.l*2-1,1),...
                Xkj_nsc{j}(jointi.ankle.l*2-1,1)];  
            qdotinkj_l = [Xkj_nsc{j}(jointi.hip.l*2,1),...
                Xkj_nsc{j}(jointi.knee.l*2,1),Xkj_nsc{j}(jointi.ankle.l*2,1)];  
            [lMTkj_l,vMTkj_l,MAkj_l] = f_lMT_vMT_dM(qinkj_l,qdotinkj_l);    
            MAkj_hip_l    =  MAkj_l(mai(1).mus.l',1);
            MAkj_knee_l   =  MAkj_l(mai(2).mus.l',2);
            MAkj_ankle_l  =  MAkj_l(mai(3).mus.l',3);    
            % Right leg
            qinkj_r = [Xkj_nsc{j}(jointi.hip.r*2-1,1),...
                Xkj_nsc{j}(jointi.knee.r*2-1,1),...
                Xkj_nsc{j}(jointi.ankle.r*2-1,1)];  
            qdotinkj_r = [Xkj_nsc{j}(jointi.hip.r*2,1),...
                Xkj_nsc{j}(jointi.knee.r*2,1), Xkj_nsc{j}(jointi.ankle.r*2,1)];      
            [lMTkj_r,vMTkj_r,MAkj_r] = f_lMT_vMT_dM(qinkj_r,qdotinkj_r);  
            % Here we take the indices from left since the vector is 1:NMuscle/2
            MAkj_hip_r    = MAkj_r(mai(1).mus.l',1);
            MAkj_knee_r   = MAkj_r(mai(2).mus.l',2);
            MAkj_ankle_r  = MAkj_r(mai(3).mus.l',3);
            % Both legs
            lMTkj_lr     = [lMTkj_l;lMTkj_r];
            vMTkj_lr     = [vMTkj_l;vMTkj_r]; 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Get muscle-tendon forces and derive Hill-equilibrium
            [Hilldiffkj,FTkj,Fcekj,Fpasskj,Fisokj,vMmaxkj,massMkj] = ...
                f_forceEquilibrium_FtildeState(akj{j},FTtildekj_nsc{j},...
                dFTtildekj_nsc{j},lMTkj_lr,vMTkj_lr,tensions);           
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Get metabolic energy rate if in the cost function   
            if W.mE ~= 0    
                % Get muscle fiber lengths
                [~,lMtildekj] = f_FiberLength_TendonForce(...
                    FTtildekj_nsc{j},lMTkj_lr); 
                % Get muscle fiber velocities
                [vMkj,~] = f_FiberVelocity_TendonForce(FTtildekj_nsc{j},...
                    dFTtildekj_nsc{j},lMTkj_lr,vMTkj_lr);
                % Get metabolic energy rate
                [e_totkj,~,~,~,~,~] = fgetMetabolicEnergySmooth2004all(...
                    akj{j},akj{j},lMtildekj,vMkj,Fcekj,Fpasskj,massMkj,...
                    pctsts,Fisokj,MTparameters_m(1,:)',body_mass,10);
            end  
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Collocation            
            % Expression for the state derivatives at the collocation point
            xp_nsc          = C(1,j+1)*Xk_nsc;
            FTtildep_nsc    = C(1,j+1)*FTtildek_nsc;
            ap              = C(1,j+1)*ak;
            a_bp            = C(1,j+1)*a_bk;
            for r=1:d
                xp_nsc       = xp_nsc + C(r+1,j+1)*Xkj_nsc{r};
                FTtildep_nsc = FTtildep_nsc + C(r+1,j+1)*FTtildekj_nsc{r};
                ap           = ap + C(r+1,j+1)*akj{r};
                a_bp         = a_bp + C(r+1,j+1)*a_bkj{r};
            end 
            % Append collocation equations
            % Dynamic constraints are scaled using the same scale
            % factors as was used to scale the states
            % Activation dynamics (implicit formulation)  
            g       = {g{:}, (h*vAk_nsc - ap)./scaling.a};
            lbg     = [lbg; zeros(NMuscle,1)];
            ubg     = [ubg; zeros(NMuscle,1)]; 
            % Contraction dynamics (implicit formulation)          
            g       = {g{:}, (h*dFTtildekj_nsc{j} - FTtildep_nsc)./...
                (scaling.FTtilde')};
            lbg     = [lbg; zeros(NMuscle,1)];
            ubg     = [ubg; zeros(NMuscle,1)];
            % Skeleton dynamics (implicit formulation)  
            xj_nsc  = [...
                Xkj_nsc{j}(2); Akj_nsc{j}(1); Xkj_nsc{j}(4); Akj_nsc{j}(2);...
                Xkj_nsc{j}(6); Akj_nsc{j}(3); Xkj_nsc{j}(8); Akj_nsc{j}(4);...
                Xkj_nsc{j}(10); Akj_nsc{j}(5); Xkj_nsc{j}(12); Akj_nsc{j}(6);...
                Xkj_nsc{j}(14); Akj_nsc{j}(7); Xkj_nsc{j}(16); Akj_nsc{j}(8);...
                Xkj_nsc{j}(18); Akj_nsc{j}(9); Xkj_nsc{j}(20); Akj_nsc{j}(10)];
            g       = {g{:}, (h*xj_nsc - xp_nsc)./(scaling.QsQdots')};
            lbg     = [lbg; zeros(2*nq.all,1)];
            ubg     = [ubg; zeros(2*nq.all,1)];   
            % Back activation dynamics (explicit formulation)  
            dadt    = f_BackActivationDynamics(e_bk,a_bkj{j});
            g       = {g{:}, (h*dadt - a_bp)./scaling.a_b};
            lbg     = [lbg; zeros(nq.trunk,1)];
            ubg     = [ubg; zeros(nq.trunk,1)]; 
            % Add contribution to the end state
            Xk_nsc_end = Xk_nsc_end + D(j+1)*Xkj_nsc{j};
            FTtildek_nsc_end = FTtildek_nsc_end + D(j+1)*FTtildekj_nsc{j};
            ak_end = ak_end + D(j+1)*akj{j};  
            a_bk_end = a_bk_end + D(j+1)*a_bkj{j};    
            % Add contribution to quadrature function
            if W.mE == 0
                J = J + 1/(dist_trav_tot)*(...
                    W.act*B(j+1)    *(f_sumsqr_exp(akj{j},exp_A))*h + ...
                    W.back*B(j+1)   *(sumsqr(e_bk))*h +... 
                    W.acc*B(j+1)    *(sumsqr(Akj{j}))*h + ...                          
                    W.u*B(j+1)      *(sumsqr(vAk))*h + ...
                    W.u*B(j+1)      *(sumsqr(dFTtildekj{j}))*h);  
            elseif W.act == 0
                J = J + 1/(dist_trav_tot)*(...
                    W.mE*B(j+1)     *(f_sumsqr_exp(e_totkj,exp_E))*h + ...
                    W.back*B(j+1)   *(sumsqr(e_bk))*h +... 
                    W.acc*B(j+1)    *(sumsqr(Akj{j}))*h + ...                          
                    W.u*B(j+1)      *(sumsqr(vAk))*h + ...
                    W.u*B(j+1)      *(sumsqr(dFTtildekj{j}))*h);  
            else
                J = J + 1/(dist_trav_tot)*(...
                    W.act*B(j+1)    *(f_sumsqr_exp(akj{j},exp_A))*h + ...
                    W.mE*B(j+1)     *(f_sumsqr_exp(e_totkj,exp_E))*h + ...
                    W.back*B(j+1)   *(sumsqr(e_bk))*h +... 
                    W.acc*B(j+1)    *(sumsqr(Akj{j}))*h + ...                          
                    W.u*B(j+1)      *(sumsqr(vAk))*h + ...
                    W.u*B(j+1)      *(sumsqr(dFTtildekj{j}))*h);  
            end  
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Call external function (run inverse dynamics)
            if deri == 2
                [Tkj] = F(Xkj_nsc{j},Akj_nsc{j}); 
            else
                [Tkj] = F([Xkj_nsc{j};Akj_nsc{j}]);      
            end
            % Add path constraints
            % Null pelvis residuals
            g           = {g{:},Tkj(jointi.gr_pelvis,1)};
            lbg         = [lbg; zeros(nq.abs,1)];
            ubg         = [ubg; zeros(nq.abs,1)];    
            % Muscle-driven joint torques for the lower limbs and the trunk
            % Hip flexion, left
            Ftkj_hip_l  = FTkj(mai(1).mus.l',1);
            Tkj_hip_l   = f_T4(MAkj_hip_l,Ftkj_hip_l);
            g           = {g{:},(Tkj(jointi.hip.l,1)-(Tkj_hip_l))};
            lbg         = [lbg; 0];
            ubg         = [ubg; 0];    
            % Hip flexion, right
            Ftkj_hip_r  = FTkj(mai(1).mus.r',1);
            Tkj_hip_r   = f_T4(MAkj_hip_r,Ftkj_hip_r);
            g           = {g{:},(Tkj(jointi.hip.r,1)-(Tkj_hip_r))};
            lbg         = [lbg; 0];
            ubg         = [ubg; 0];    
            % Knee, left
            Ftkj_knee_l = FTkj(mai(2).mus.l',1);
            Tkj_knee_l  = f_T5(MAkj_knee_l,Ftkj_knee_l);
            g           = {g{:},(Tkj(jointi.knee.l,1)-(Tkj_knee_l))};
            lbg         = [lbg; 0];
            ubg         = [ubg; 0];    
            % Knee, right
            Ftkj_knee_r = FTkj(mai(2).mus.r',1);
            Tkj_knee_r  = f_T5(MAkj_knee_r,Ftkj_knee_r);
            g           = {g{:},(Tkj(jointi.knee.r,1)-(Tkj_knee_r))};
            lbg         = [lbg; 0];
            ubg         = [ubg; 0];    
            % Ankle, left
            Ftkj_ankle_l    = FTkj(mai(3).mus.l',1);
            Tkj_ankle_l     = f_T3(MAkj_ankle_l,Ftkj_ankle_l);
            g               = {g{:},(Tkj(jointi.ankle.l,1)-(Tkj_ankle_l))};
            lbg             = [lbg; 0];
            ubg             = [ubg; 0];    
            % Ankle, right
            Ftkj_ankle_r    = FTkj(mai(3).mus.r',1);
            Tkj_ankle_r     = f_T3(MAkj_ankle_r,Ftkj_ankle_r);
            g               = {g{:},(Tkj(jointi.ankle.r,1)-(Tkj_ankle_r))};
            lbg             = [lbg; 0];
            ubg             = [ubg; 0];
            % Torque-driven joint torque for the trunk
            % Trunk
            g       = {g{:},Tkj(jointi.trunk.ext,1)./scaling.BackTau-a_bkj{j}};
            lbg     = [lbg; 0];
            ubg     = [ubg; 0];  
            % Contraction dynamics (implicit formulation)
            g               = {g{:},Hilldiffkj};
            lbg             = [lbg; zeros(NMuscle,1)];
            ubg             = [ubg; zeros(NMuscle,1)];              
            % Activation dynamics (implicit formulation)
            act1 = vAk_nsc + akj{j}./(ones(size(akj{j},1),1)*tdeact);
            act2 = vAk_nsc + akj{j}./(ones(size(akj{j},1),1)*tact);
            % act1
            g               = {g{:},act1};
            lbg             = [lbg; zeros(NMuscle,1)];
            ubg             = [ubg; inf*ones(NMuscle,1)]; 
            % act2
            g               = {g{:},act2};
            lbg             = [lbg; -inf*ones(NMuscle,1)];
            ubg             = [ubg; ones(NMuscle,1)./(ones(NMuscle,1)*tact)]; 
            
        end % End loop over collocation points           
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % New NLP variables for states at end of interval
        if k ~= N-1
            % Muscle activations
            ak              = MX.sym(['a_' num2str(k+1)], NMuscle);
            w               = {w{:}, ak};
            lbw             = [lbw; bounds.a.lower'];
            ubw             = [ubw; bounds.a.upper'];
            w0              = [w0;  guess.a(k+2,:)'];
            % Muscle-tendon forces
            FTtildek        = MX.sym(['FTtilde_' num2str(k+1)], NMuscle);
            w               = {w{:}, FTtildek};
            lbw             = [lbw; bounds.FTtilde.lower'];
            ubw             = [ubw; bounds.FTtilde.upper'];
            w0              = [w0;  guess.FTtilde(k+2,:)'];    
            % Qs and Qdots
            w               = {w{:}, Xk{k+2,1}};
            lbw             = [lbw; bounds.QsQdots.lower'];
            ubw             = [ubw; bounds.QsQdots.upper']; 
            w0              = [w0;  guess.QsQdots(k+2,:)'];
            % Back activations
            a_bk            = MX.sym(['a_b_' num2str(k+1)], nq.trunk);
            w               = {w{:}, a_bk};
            lbw             = [lbw; bounds.a_b.lower'];
            ubw             = [ubw; bounds.a_b.upper'];
            w0              = [w0;  guess.a_b(k+2,:)'];
        else % Periodicty 
            % Muscle activations
            ak              = MX.sym(['a_' num2str(k+1)], NMuscle);
            w               = {w{:}, ak};
            lbw             = [lbw; bounds.a.lower'];
            ubw             = [ubw; bounds.a.upper'];            
            w0              = [w0;  guess.a(1,orderMusInv)'];
            % Muscle-tendon forces
            FTtildek        = MX.sym(['FTtilde_' num2str(k+1)], NMuscle);
            w               = {w{:}, FTtildek};
            lbw             = [lbw; bounds.FTtilde.lower'];
            ubw             = [ubw; bounds.FTtilde.upper'];
            w0              = [w0;  guess.FTtilde(1,orderMusInv)'];    
            % Qs and Qdots
            w               = {w{:}, Xk{k+2,1}};
            lbw             = [lbw; bounds.QsQdots.lower'];
            ubw             = [ubw; bounds.QsQdots.upper'];
            % For "symmetric" joints, we invert right and left
            inv_X           = guess.QsQdots(1,orderQsInv);     
            dx = guess.QsQdots(end,2*jointi.pelvis.tx-1) - ...
                guess.QsQdots(end-1,2*jointi.pelvis.tx-1);
            inv_X(2*jointi.pelvis.tx-1) = ...
                guess.QsQdots(end,2*jointi.pelvis.tx-1) + dx;            
            w0                = [w0;  inv_X'];   
            % Back activations
            a_bk            = MX.sym(['a_b_' num2str(k+1)], nq.trunk);
            w               = {w{:}, a_bk};
            lbw             = [lbw; bounds.a_b.lower'];
            ubw             = [ubw; bounds.a_b.upper'];            
            w0              = [w0;  guess.a_b(1,:)'];
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Rescale variables to impose continuity constraints
        Xk_end = (Xk_nsc_end)./scaling.QsQdots';
        FTtildek_end = (FTtildek_nsc_end)./scaling.FTtilde';
        % Add continuity constraints (next interval starts with end values of 
        % states from previous interval)
        g   = {g{:}, Xk_end-Xk{k+2,1}, FTtildek_end-FTtildek, ...
            ak_end-ak, a_bk_end-a_bk};
        lbg = [lbg; zeros(2*nq.all + NMuscle + NMuscle + nq.trunk,1)];
        ubg = [ubg; zeros(2*nq.all + NMuscle + NMuscle + nq.trunk,1)];    
    end % End loop over mesh points
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Additional path constraints
    % Periodicity of the states
    % Qs and Qdots         
    QsInvA = [jointi.pelvis.tilt:2*jointi.pelvis.tilt,...
        2*jointi.pelvis.tx,2*jointi.pelvis.ty-1:2*jointi.trunk.ext]';
    QsInvB = [jointi.pelvis.tilt:2*jointi.pelvis.tilt,...
        2*jointi.pelvis.tx,2*jointi.pelvis.ty-1:2*jointi.pelvis.ty,...
        2*jointi.hip.r-1:2*jointi.hip.r,...
        2*jointi.hip.l-1:2*jointi.hip.l,...
        2*jointi.knee.r-1:2*jointi.knee.r,...
        2*jointi.knee.l-1:2*jointi.knee.l,...
        2*jointi.ankle.r-1:2*jointi.ankle.r,...
        2*jointi.ankle.l-1:2*jointi.ankle.l,...
        2*jointi.trunk.ext-1,2*jointi.trunk.ext]';             
    g   = {g{:}, Xk_end(QsInvA)-X0(QsInvB,1)};
    lbg = [lbg; zeros(length(QsInvB),1)];
    ubg = [ubg; zeros(length(QsInvB),1)];         
    % Muscle activations
    g   = {g{:}, ak_end-a0(orderMusInv,1)};
    lbg = [lbg; zeros(NMuscle,1)];
    ubg = [ubg; zeros(NMuscle,1)];
    % Muscle-tendon forces
    g   = {g{:}, FTtildek_end-FTtilde0(orderMusInv,1)};
    lbg = [lbg; zeros(NMuscle,1)];
    ubg = [ubg; zeros(NMuscle,1)];
    % Back activations
    g   = {g{:}, a_bk_end-a_b0(1,1)};
    lbg = [lbg; zeros(nq.trunk,1)];
    ubg = [ubg; zeros(nq.trunk,1)];
    % Average speed
    vel_aver_tot = dist_trav_tot/tf; 
    g   = {g{:}, vel_aver_tot - v_tgt};
    lbg = [lbg; 0];
    ubg = [ubg; 0];  
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Assert bounds / IG
    % Lower bounds smaller than upper bounds    
    assert_bw = isempty(find(lbw <= ubw == 0,1)); % Design variables
    if assert_bw 
        disp('Error in bounds design variable (lower larger than upper bounds)')
    end
    assert_bg = isempty(find(lbg <= ubg == 0,1)); % Constraints
    if assert_bg
        disp('Error in bounds constraint (lower larger than upper bounds)')
    end
    % Design variables between -1 and 1
    assert_bwl = isempty(find(lbw < -1 == 1,1));
    if assert_bwl 
        disp('WARNING: scaling (lower bounds smaller than -1)'); 
    end
    assert_bwu = isempty(find(1 < ubw == 1,1));   
    if assert_bwu 
        disp('WARNING: scaling (upper bounds larger than 1)'); 
    end
    % Initial guess within bounds
    assert_w0_ubw = isempty(find(w0 <= ubw == 0,1));
    if assert_bwu 
        disp('WARNING: initial guess (lower than lower bounds)'); 
    end
    assert_w0_lbw = isempty(find(lbw <= w0 == 0,1));
    if assert_bwu 
        disp('WARNING: initial guess (larger than upper bounds)'); 
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Create an NLP solver
    prob = struct('f', J, 'x', vertcat(w{:}), 'g', vertcat(g{:}));
    % Hessian approximation
    if hessi == 1
        options.ipopt.hessian_approximation = 'limited-memory';   
    elseif hessi == 2    
        options.ipopt.hessian_approximation = 'exact';  
    end
    options.ipopt.mu_strategy = 'adaptive';
    % Linear solver       
    setup.linear_solver = linear_solvers{linsoli};
    options.ipopt.linear_solver = setup.linear_solver;
    % Maximum number of iterations    
    options.ipopt.max_iter = 10000;
    % NLP error tolerance
    options.ipopt.tol = 1*10^(-tol_ipopt);
    switch setup.derivatives
        case 'FD' 
            options.common_options.helper_options = ...
                struct('enable_fd',true,'enable_forward',false,...
                'enable_reverse',false,'print_in',false,...
                'fd_method','forward');
    end
    solver = nlpsol('solver', 'ipopt', prob, options);   
    % Create and save diary
    p = mfilename('fullpath');
    [~,namescript,~] = fileparts(p);
    pathresults = [pathRepo,'/Results'];
    if ~(exist([pathresults,'/',namescript],'dir')==7)
        mkdir(pathresults,namescript);
    end
    if (exist([pathresults,'/',namescript,'/D',savename],'file')==2)
        delete ([pathresults,'/',namescript,'/D',savename])
    end 
    diary([pathresults,'/',namescript,'/D',savename]);  
    % Solve problem
    sol = solver('x0', w0, 'lbx', lbw, 'ubx', ubw, 'lbg', lbg, 'ubg', ubg);    
    diary off
    % Extract results
    w_opt = full(sol.x);
    g_opt = full(sol.g); 
    % Extract stats
    stats = solver.stats();
    % Create setup
    setup.tolerance.ipopt = tol_ipopt;
    setup.bounds = bounds;
    setup.scaling = scaling;
    setup.guess = guess;
    setup.lbw = lbw;
    setup.ubw = ubw;
    setup.hessian = hessi;
    setup.N = N;
    % Save results and setup
    save([pathresults,'/',namescript,'/w',savename],'w_opt');
    save([pathresults,'/',namescript,'/g',savename],'g_opt');
    save([pathresults,'/',namescript,'/s',savename],'setup');
    save([pathresults,'/',namescript,'/stats',savename],'stats');
end

%% Analyze results
if analyseResults
    %% Load results
    if loadResults
        p = mfilename('fullpath');
        [~,namescript,~] = fileparts(p);
        pathresults = [pathRepo,'/Results'];
        load([pathresults,'/',namescript,'/w',savename]);
        load([pathresults,'/',namescript,'/g',savename]);
        load([pathresults,'/',namescript,'/s',savename]);
        load([pathresults,'/',namescript,'/stats',savename]);
    end
    if deri == 1
        setup.derivatives = 'AD_Recorder'; % Algorithmic differentiation    
    elseif deri == 2
        setup.derivatives = 'AD_ADOLC'; % Algorithmic differentiation 
    elseif deri == 3
        setup.derivatives = 'FD'; % Finite differences
    end
    setup.hessian = hessi;
    save([pathresults,'/',namescript,'/s',savename],'setup');
    
    %% Extract results
    % All optimized design variables are saved in a single column vector      
    % Number of design variables    
    NControls = NMuscle+nq.trunk;
    NFirstMeshPointControls = NMuscle+nq.all;
    NSlackControls = NMuscle+nq.all;
    NStates = NMuscle+NMuscle+2*nq.all+nq.trunk;
    NParameters = 1;
    % Parameters and first mesh point controls
    NPFMPC = NParameters+NFirstMeshPointControls;
    % In the loop
    Nwl = NControls+d*(NStates)+d*NSlackControls+NStates;
    % In total
    Nw = NPFMPC+NStates+N*Nwl;
    % Before the variable corresponding to the first state at collocation point
    NwSCP = NPFMPC+NStates+NControls;
    % Before the variable corresponding to the first slack control at
    % collocation point
    NwSCCP = NPFMPC+NStates+NControls+d*NStates;    
    % Here we extract the results and re-organize them for analysis 
    % Static parameters
    tf_opt  = w_opt(1:NParameters);
    % Controls at first mesh points
    % Time derivative of muscle-tendon forces
    dFTtilde_opt = zeros(1,NMuscle);
    for i = 1:NMuscle
        dFTtilde_opt(:,i) = w_opt(NParameters+i);
    end
    % Time derivative of joint velocities
    qdotdot_opt = zeros(1,nq.all);
    for i = 1:nq.all
        qdotdot_opt(:,i) = w_opt(NParameters+NMuscle+i);
    end
    % States at mesh points
    % Muscle activations and muscle-tendon forces
    a_opt = zeros(N+1,NMuscle);
    FTtilde_opt = zeros(N+1,NMuscle);
    for i = 1:NMuscle
        a_opt(:,i) = w_opt(NPFMPC+i:Nwl:Nw);
        FTtilde_opt(:,i) = w_opt(NPFMPC+NMuscle+i:Nwl:Nw);
    end
    % Qs and Qdots
    q_opt = zeros(N+1,nq.all);
    qdot_opt = zeros(N+1,nq.all);
    count = 0;
    for i = 1:2:2*nq.all
        count = count +1;
        q_opt(:,count) = w_opt(NPFMPC+NMuscle+NMuscle+i:Nwl:Nw);
        qdot_opt(:,count) = w_opt(NPFMPC+NMuscle+NMuscle+i+1:Nwl:Nw);
    end
    % Arm activations
    a_b_opt = zeros(N+1,nq.trunk);
    for i = 1:nq.trunk
        a_b_opt(:,i) = w_opt(NPFMPC+NMuscle+NMuscle+2*nq.all+i:Nwl:Nw);
    end    
    % Controls at mesh points
    % Time derivative of muscle activations
    vA_opt = zeros(N,NMuscle);
    for i = 1:NMuscle
        vA_opt(:,i) = w_opt(NPFMPC+NStates+i:Nwl:Nw);
    end
    % Back excitations
    e_b_opt = zeros(N,nq.trunk);
    for i = 1:nq.trunk
        e_b_opt(:,i) = w_opt(NPFMPC+NStates+NMuscle+i:Nwl:Nw);
    end    
    % States at collocation points
    % Muscle activations
    a_opt_ext = zeros(N*(d+1)+1,NMuscle);
    a_opt_ext(1:(d+1):end,:) = a_opt;
    for nmi=1:NMuscle
        a_opt_ext(2:(d+1):end,nmi) = w_opt(NwSCP+nmi:Nwl:Nw);
        a_opt_ext(3:(d+1):end,nmi) = w_opt(NwSCP+NMuscle+nmi:Nwl:Nw);
        a_opt_ext(4:(d+1):end,nmi) = w_opt(NwSCP+NMuscle+NMuscle+nmi:Nwl:Nw);
    end  
    % Muscle-tendon forces
    FTtilde_opt_ext = zeros(N*(d+1)+1,NMuscle);
    FTtilde_opt_ext(1:(d+1):end,:) = FTtilde_opt;
    for nmi=1:NMuscle
        FTtilde_opt_ext(2:(d+1):end,nmi) = w_opt(NwSCP+d*NMuscle+nmi:Nwl:Nw);
        FTtilde_opt_ext(3:(d+1):end,nmi) = ...
            w_opt(NwSCP+d*NMuscle+NMuscle+nmi:Nwl:Nw);
        FTtilde_opt_ext(4:(d+1):end,nmi) = ...
            w_opt(NwSCP+d*NMuscle+NMuscle+NMuscle+nmi:Nwl:Nw);
    end
    % Qs and Qdots
    q_opt_ext = zeros(N*(d+1)+1,nq.all);
    q_opt_ext(1:(d+1):end,:) = q_opt;
    q_dot_opt_ext = zeros(N*(d+1)+1,nq.all);
    q_dot_opt_ext(1:(d+1):end,:) = qdot_opt;
    nqi_col = 1:2:2*nq.all;
    for nqi=1:nq.all
        nqi_q = nqi_col(nqi);
        q_opt_ext(2:(d+1):end,nqi) = w_opt(NwSCP+d*NMuscle+...
            d*NMuscle+nqi_q:Nwl:Nw);   
        q_opt_ext(3:(d+1):end,nqi) = w_opt(NwSCP+d*NMuscle+...
            d*NMuscle+2*nq.all+nqi_q:Nwl:Nw);  
        q_opt_ext(4:(d+1):end,nqi) = w_opt(NwSCP+d*NMuscle+...
            d*NMuscle+2*nq.all+2*nq.all+nqi_q:Nwl:Nw);  
        q_dot_opt_ext(2:(d+1):end,nqi) = w_opt(NwSCP+d*NMuscle+...
            d*NMuscle+nqi_q+1:Nwl:Nw);   
        q_dot_opt_ext(3:(d+1):end,nqi) = w_opt(NwSCP+d*NMuscle+...
            d*NMuscle+2*nq.all+nqi_q+1:Nwl:Nw);  
        q_dot_opt_ext(4:(d+1):end,nqi) = w_opt(NwSCP+d*NMuscle+...
            d*NMuscle+2*nq.all+2*nq.all+nqi_q+1:Nwl:Nw);
    end
    % Back activations
    a_b_opt_ext = zeros(N*(d+1)+1,nq.trunk);
    a_b_opt_ext(1:(d+1):end,:) = a_b_opt;
    for nmi=1:nq.trunk
        a_b_opt_ext(2:(d+1):end,nmi) = w_opt(NwSCP+d*NMuscle+...
            d*NMuscle+d*2*nq.all+nmi:Nwl:Nw);
        a_b_opt_ext(3:(d+1):end,nmi) = w_opt(NwSCP+d*NMuscle+...
            d*NMuscle+d*2*nq.all+nq.trunk+nmi:Nwl:Nw);
        a_b_opt_ext(4:(d+1):end,nmi) = w_opt(NwSCP+d*NMuscle+...
            d*NMuscle+d*2*nq.all+nq.trunk+nq.trunk+nmi:Nwl:Nw);
    end  
    % Slack controls at collocation points
    % Time derivative of muscle-tendon forces
    dFTtilde_opt_ext = zeros(N*d,NMuscle);
    for nmi=1:NMuscle
        dFTtilde_opt_ext(1:d:end,nmi) = w_opt(NwSCCP+nmi:Nwl:Nw);
        dFTtilde_opt_ext(2:d:end,nmi) = w_opt(NwSCCP+NMuscle+nmi:Nwl:Nw);
        dFTtilde_opt_ext(3:d:end,nmi) = ...
            w_opt(NwSCCP+NMuscle+NMuscle+nmi:Nwl:Nw);
    end
    % Time derivative of Qdots
    qdotdot_opt_ext=zeros(N*d,nq.all);
    for nmi=1:nq.all
        qdotdot_opt_ext(1:d:end,nmi) = w_opt(NwSCCP+d*NMuscle+nmi:Nwl:Nw);
        qdotdot_opt_ext(2:d:end,nmi) = ...
            w_opt(NwSCCP+d*NMuscle+nq.all+nmi:Nwl:Nw);
        qdotdot_opt_ext(3:d:end,nmi) = ...
            w_opt(NwSCCP+d*NMuscle+nq.all+nq.all+nmi:Nwl:Nw);
    end
    
    %% Unscale results
    % States at mesh points
    % Qs (1:N-1)
    q_opt_unsc.rad = ...
        q_opt(1:end-1,:).*repmat(scaling.Qs,size(q_opt(1:end-1,:),1),1); 
    % Convert in degrees
    q_opt_unsc.deg = q_opt_unsc.rad;
    dof_roti = [jointi.pelvis.tilt,jointi.hip.l:jointi.trunk.ext];
    q_opt_unsc.deg(:,dof_roti) = q_opt_unsc.deg(:,dof_roti).*180/pi;
    % Qs (1:N)
    q_opt_unsc_all.rad = q_opt.*repmat(scaling.Qs,size(q_opt,1),1); 
    % Convert in degrees
    q_opt_unsc_all.deg = q_opt_unsc_all.rad;
    q_opt_unsc_all.deg(:,dof_roti) = q_opt_unsc_all.deg(:,dof_roti).*180/pi;    
    % Qdots (1:N-1)
    qdot_opt_unsc.rad = qdot_opt(1:end-1,:).*repmat(...
        scaling.Qdots,size(qdot_opt(1:end-1,:),1),1);
    % Convert in degrees
    qdot_opt_unsc.deg = qdot_opt_unsc.rad;
    qdot_opt_unsc.deg(:,dof_roti) = qdot_opt_unsc.deg(:,dof_roti).*180/pi;    
    % Qdots (1:N)
    qdot_opt_unsc_all.rad = qdot_opt.*repmat(scaling.Qdots,size(qdot_opt,1),1); 
    % Muscle activations
    a_opt_unsc = a_opt(1:end-1,:).*repmat(...
        scaling.a,size(a_opt(1:end-1,:),1),size(a_opt,2));
    % Muscle-tendon forces
    FTtilde_opt_unsc = FTtilde_opt(1:end-1,:).*repmat(...
        scaling.FTtilde,size(FTtilde_opt(1:end-1,:),1),1);
    % Arm activations
    a_b_opt_unsc = a_b_opt(1:end-1,:).*repmat(...
        scaling.a_b,size(a_b_opt(1:end-1,:),1),size(a_b_opt,2));
    % Controls at mesh points
    % Time derivative of muscle activations (states)
    vA_opt_unsc = vA_opt.*repmat(scaling.vA,size(vA_opt,1),size(vA_opt,2));
    % Get muscle excitations from time derivative of muscle activations
    e_opt_unsc = computeExcitationRaasch(a_opt_unsc,vA_opt_unsc,...
        ones(1,NMuscle)*tdeact,ones(1,NMuscle)*tact);
    % Back excitations
    e_b_opt_unsc = e_b_opt.*repmat(scaling.e_b,size(e_b_opt,1),size(e_b_opt,2));    
    % States at mesh and collocation points
    % Qs
    q_opt_ext_unsc.rad = q_opt_ext.*repmat(scaling.Qs,size(q_opt_ext,1),1);   
    % Qdots
    qdot_opt_ext_unsc.rad = ...
        q_dot_opt_ext.*repmat(scaling.Qdots,size(q_dot_opt_ext,1),1);   
    % Slack controls at collocation points
    % Time derivative of Qdots
    qdotdot_opt_ext_unsc.rad = ...
        qdotdot_opt_ext.*repmat(scaling.Qdotdots,size(qdotdot_opt_ext,1),1);
    % Convert in degrees
    qdotdot_opt_ext_unsc.deg = qdotdot_opt_ext_unsc.rad;
    qdotdot_opt_ext_unsc.deg(:,dof_roti) = ...
        qdotdot_opt_ext_unsc.deg(:,dof_roti).*180/pi;  
    % Time derivative of muscle-tendon forces
    dFTtilde_opt_ext_unsc = dFTtilde_opt_ext.*repmat(...
        scaling.dFTtilde,size(dFTtilde_opt_ext,1),size(dFTtilde_opt_ext,2));
    % Controls at first mesh point (FMP)
    % Time derivative of Qdots
    qdotdot_opt_FMP_unsc.rad = ...
        qdotdot_opt.*repmat(scaling.Qdotdots,size(qdotdot_opt,1),1);
    % Convert in degrees
    qdotdot_opt_FMP_unsc.deg = qdotdot_opt_FMP_unsc.rad;
    qdotdot_opt_FMP_unsc.deg(:,dof_roti) = ...
        qdotdot_opt_FMP_unsc.deg(:,dof_roti).*180/pi;
    % Time derivative of muscle-tendon forces
    dFTtilde_opt_FMP_unsc = dFTtilde_opt.*repmat(...
        scaling.dFTtilde,size(dFTtilde_opt,1),size(dFTtilde_opt,2));
    
    %% Time grid    
    % Mesh points
    tgrid = linspace(0,tf_opt,N+1);
    dtime = zeros(1,d+1);
    for i=1:4
        dtime(i)=tau_root(i)*(tf_opt/N);
    end
    % Mesh points and collocation points
    tgrid_ext = zeros(1,(d+1)*N+1);
    for i=1:N
        tgrid_ext(((i-1)*4+1):1:i*4)=tgrid(i)+dtime;
    end
    tgrid_ext(end)=tf_opt;
    
    %% Joint torques and ground reaction forces at optimal solution
    Xk_Qs_Qdots_opt_ext = zeros(d*N+1,2*nq.all); 
    Xk_Qdotdots_opt_ext = zeros(d*N+1,nq.all); 
    a_b_opt_ext_atSlack = zeros(d*N+1,1); 
    % First mest point
    Xk_Qs_Qdots_opt_ext(1,1:2:end) = q_opt_ext_unsc.rad(1,:);
    Xk_Qs_Qdots_opt_ext(1,2:2:end) = qdot_opt_ext_unsc.rad(1,:);    
    Xk_Qdotdots_opt_ext(1,:) = qdotdot_opt_FMP_unsc.rad;        
    a_b_opt_ext_atSlack(1,:) = a_b_opt_ext(1,:);
    for n = 1:N
        ni = 2+(n-1)*(d+1);
        Xk_Qs_Qdots_opt_ext((n-1)*d+1+1:n*d+1,1:2:end) = ...
            q_opt_ext_unsc.rad(ni:ni+(d-1),:);
        Xk_Qs_Qdots_opt_ext((n-1)*d+1+1:n*d+1,2:2:end)  = ...
            qdot_opt_ext_unsc.rad(ni:ni+(d-1),:); 
        a_b_opt_ext_atSlack((n-1)*d+1+1:n*d+1,:) = a_b_opt_ext(ni:ni+(d-1),:); 
    end 
    Xk_Qdotdots_opt_ext(2:end,:) = qdotdot_opt_ext_unsc.rad;    
    out_res_opt_ext = zeros(d*N+1,nq.all+nGRF);
    for i = 1:d*N+1
        [res_ext] = F1([Xk_Qs_Qdots_opt_ext(i,:)';Xk_Qdotdots_opt_ext(i,:)']);
        out_res_opt_ext(i,:) = full(res_ext);    
    end
    GRF_opt_unsc_ext = out_res_opt_ext(:,GRFi.all);
    GRF_opt_unsc_temp = GRF_opt_unsc_ext(1:d:end,:);
    out_res_opt_temp = out_res_opt_ext(1:d:end,:);
    % TODO,  To make it consistent with previous code
    GRF_opt_unsc = GRF_opt_unsc_temp(1:end-1,:);
    out_res_opt = out_res_opt_temp(1:end-1,:);
    % assertBackTmax should be 0
    assertBackTmax_ext = ...
        max(max(abs(out_res_opt_ext(:,jointi.trunk.ext)/scaling.BackTau - ...
            (a_b_opt_ext_atSlack))));
    if assertBackTmax_ext > 1*10^(-tol_ipopt)
        disp('Issue when reconstructing residual forces')
    end 
       
    %% Assert average speed 
    dist_trav_opt = q_opt_ext(end,jointi.pelvis.tx)*...
        scaling.Qs(jointi.pelvis.tx) - q_opt_ext(1,jointi.pelvis.tx)*...
        scaling.Qs(jointi.pelvis.tx); % distance traveled    
    time_elaps_opt = tf_opt; % time elapsed
    % Average speed
    vel_aver_opt = dist_trav_opt/time_elaps_opt;  
    % assert_v_tg should be 0
    assert_v_tg = abs(vel_aver_opt-v_tgt);
    if assert_v_tg > 1*10^(-tol_ipopt)
        disp('Issue when reconstructing average speed')
    end 
   
    %% Reconstruct full gait cycle: starting with right heel strike
    % We reconstruct the full gait cycle from the simulated half gait cycle
    % Identify heel strike
    threshold = 20; % there is foot-ground contact above the threshold
    if exist('HS1','var')
        clear HS1
    end
    % TODO: might need to get adapted
    % Check if heel strike is on the right side  
    phase_tran_tgridi = find(GRF_opt_unsc(:,2)<threshold,1,'last');
    if ~isempty(phase_tran_tgridi)        
        if phase_tran_tgridi == N
            temp_idx = find(GRF_opt_unsc(:,2)>threshold,1,'first');
            if ~isempty(temp_idx)
                if temp_idx-1 ~= 0 && ...
                        find(GRF_opt_unsc(temp_idx-1,2)<threshold)
                    phase_tran_tgridi_t = temp_idx;             
                    IC1i = phase_tran_tgridi_t;
                    HS1 = 'r';
                end 
            else            
                IC1i = phase_tran_tgridi + 1; 
                HS1 = 'r';
            end
        else            
            IC1i = phase_tran_tgridi + 1; 
            HS1 = 'r';
        end        
    end
    if ~exist('HS1','var')
        % Check if heel strike is on the left side 
        phase_tran_tgridi = find(GRF_opt_unsc(:,4)<threshold,1,'last');       
        if phase_tran_tgridi == N
            temp_idx = find(GRF_opt_unsc(:,4)>threshold,1,'first');
            if ~isempty(temp_idx)  
                if temp_idx-1 ~= 0 && ...
                        find(GRF_opt_unsc(temp_idx-1,4)<threshold)
                    phase_tran_tgridi_t = temp_idx;             
                    IC1i = phase_tran_tgridi_t;
                    HS1 = 'l';
                else
                    IC1i = phase_tran_tgridi + 1; 
                    HS1 = 'l';
                end 
            else
                IC1i = phase_tran_tgridi + 1; 
                HS1 = 'l';
            end
        else            
            IC1i = phase_tran_tgridi + 1; 
            HS1 = 'l';
        end        
    end
    if isempty(phase_tran_tgridi)
        disp('No heel strike detected, consider increasing the threshold');
        continue;
    end

    % Joint kinematics
    % Helper variables to reconstruct full gait cycle assuming symmetry
    QsSymA = [jointi.pelvis.tilt,jointi.pelvis.ty,...
        jointi.hip.l:jointi.trunk.ext];
    QsSymB = [jointi.pelvis.tilt,jointi.pelvis.ty,...    
        jointi.hip.r,jointi.hip.l,jointi.knee.r,jointi.knee.l,...
        jointi.ankle.r,jointi.ankle.l,jointi.trunk.ext]; 
    QsSymA_ptx = [jointi.pelvis.tilt,jointi.pelvis.tx,...
        jointi.pelvis.ty,jointi.hip.l:jointi.trunk.ext];
    QsSymB_ptx = [jointi.pelvis.tilt,jointi.pelvis.tx,jointi.pelvis.ty,...    
        jointi.hip.r,jointi.hip.l,jointi.knee.r,jointi.knee.l,...
        jointi.ankle.r,jointi.ankle.l,jointi.trunk.ext]; 
    
    % Qs
    q_opt_GC = zeros(N*2,size(q_opt_unsc.deg,2));
    q_opt_GC(1:N-IC1i+1,:) = q_opt_unsc.deg(IC1i:end,:);   
    q_opt_GC(N-IC1i+2:N-IC1i+1+N,QsSymA) = q_opt_unsc.deg(1:end,QsSymB);
    q_opt_GC(N-IC1i+2:N-IC1i+1+N,jointi.pelvis.tx) = ...
        q_opt_unsc.deg(1:end,jointi.pelvis.tx) + ...
        q_opt_unsc_all.deg(end,jointi.pelvis.tx);        
    q_opt_GC(N-IC1i+2+N:2*N,:) = q_opt_unsc.deg(1:IC1i-1,:);    
    q_opt_GC(N-IC1i+2+N:2*N,jointi.pelvis.tx) = ...
        q_opt_unsc.deg(1:IC1i-1,jointi.pelvis.tx) + ...
        2*q_opt_unsc_all.deg(end,jointi.pelvis.tx);
    % If the first heel strike was on the left foot then we invert so that
    % we always start with the right foot, for analysis purpose
    if strcmp(HS1,'l')         
        q_opt_GC(:,QsSymA_ptx)  = q_opt_GC(:,QsSymB_ptx);
    end  
    temp_q_opt_GC_pelvis_tx = q_opt_GC(1,jointi.pelvis.tx);
    q_opt_GC(:,jointi.pelvis.tx) = q_opt_GC(:,jointi.pelvis.tx)-...
        temp_q_opt_GC_pelvis_tx;
    % For visualization in OpenSim GUI
    q_opt_GUI_GC = zeros(2*N,1+nq.all);
    q_opt_GUI_GC(1:N-IC1i+1,1) = tgrid(:,IC1i:end-1)';
    q_opt_GUI_GC(N-IC1i+2:N-IC1i+1+N,1)  = tgrid(:,1:end-1)' + tgrid(end);
    q_opt_GUI_GC(N-IC1i+2+N:2*N,1) = tgrid(:,1:IC1i-1)' + 2*tgrid(end);    
    q_opt_GUI_GC(:,2:end) = q_opt_GC;
    q_opt_GUI_GC(:,1) = q_opt_GUI_GC(:,1)-q_opt_GUI_GC(1,1);    
    % Create .mot file for OpenSim GUI
    if writeIKmotion
        pathOpenSim = [pathRepo,'/OpenSim'];
        addpath(genpath(pathOpenSim));
        JointAngle.labels = {'time','pelvis_tilt','pelvis_tx','pelvis_ty',...
        'hip_flexion_l','hip_flexion_r','knee_angle_l','knee_angle_r',...
        'ankle_angle_l','ankle_angle_r','lumbar_extension'};
        % Two gait cycles
        q_opt_GUI_GC_2 = [q_opt_GUI_GC;q_opt_GUI_GC];
        q_opt_GUI_GC_2(2*N+1:4*N,1) = q_opt_GUI_GC_2(2*N+1:4*N,1) + ...
            q_opt_GUI_GC_2(end,1)+q_opt_GUI_GC_2(end,1)-q_opt_GUI_GC_2(end-1,1);
        q_opt_GUI_GC_2(2*N+1:4*N,jointi.pelvis.tx+1) = ...
            q_opt_GUI_GC_2(2*N+1:4*N,jointi.pelvis.tx+1) + ...
            2*q_opt_unsc_all.deg(end,jointi.pelvis.tx);
        JointAngle.data = q_opt_GUI_GC_2;
        filenameJointAngles = [pathRepo,'/Results/',namescript,...
                '/IK',savename,'.mot'];
        write_motionFile(JointAngle, filenameJointAngles)
    end
    
    % Qdots
    qdot_opt_GC = zeros(N*2,size(q_opt,2));
    qdot_opt_GC(1:N-IC1i+1,:) = qdot_opt_unsc.deg(IC1i:end,:);
    qdot_opt_GC(N-IC1i+2:N-IC1i+1+N,QsSymA_ptx) = ...
        qdot_opt_unsc.deg(1:end,QsSymB_ptx);
    qdot_opt_GC(N-IC1i+2+N:2*N,:) = ...
        qdot_opt_unsc.deg(1:IC1i-1,:);
    % If the first heel strike was on the left foot then we invert so that
    % we always start with the right foot, for analysis purpose
    if strcmp(HS1,'l')
        qdot_opt_GC(:,QsSymA_ptx) = qdot_opt_GC(:,QsSymB_ptx);
    end
    
%     % Qdotdots
%     qdotdot_opt_GC = zeros(N*2,size(q_opt,2));
%     qdotdot_opt_GC(1:N-IC1i+1,:) = qdotdot_opt_FMP_unsc.deg(IC1i:end,:);
%     qdotdot_opt_GC(N-IC1i+2:N-IC1i+1+N,QsSymA_ptx) = ...
%         qdotdot_opt_FMP_unsc.deg(1:end,QsSymB_ptx);
%     qdotdot_opt_GC(N-IC1i+2+N:2*N,:) = qdotdot_opt_FMP_unsc.deg(1:IC1i-1,:);
%     % If the first heel strike was on the left foot then we invert so that
%     % we always start with the right foot, for analysis purpose
%     if strcmp(HS1,'l')
%         qdotdot_opt_GC(:,QsSymA_ptx) = qdotdot_opt_GC(:,QsSymB_ptx);
%     end

    % Ground reaction forces
    GRF_opt_GC = zeros(N*2,nGRF);
    GRF_opt_GC(1:N-IC1i+1,:) = GRF_opt_unsc(IC1i:end,:);
    GRF_opt_GC(N-IC1i+2:N-IC1i+1+N,:) = GRF_opt_unsc(1:end,[3:4,1:2]);
    GRF_opt_GC(N-IC1i+2+N:2*N,:) = GRF_opt_unsc(1:IC1i-1,:);
    GRF_opt_GC = GRF_opt_GC./(body_weight/100);
    % If the first heel strike was on the left foot then we invert so that
    % we always start with the right foot, for analysis purpose
    if strcmp(HS1,'l')
        GRF_opt_GC(:,[3:4,1:2]) = GRF_opt_GC(:,:);
    end  
    
    % Joint torques
    tau_opt_GC = zeros(N*2,size(q_opt,2));
    tau_opt_GC(1:N-IC1i+1,1:nq.all) = ...
        out_res_opt(IC1i:end,1:nq.all)./body_mass;
    tau_opt_GC(N-IC1i+2:N-IC1i+1+N,QsSymA_ptx) = ...
        out_res_opt(1:end,QsSymB_ptx)./body_mass;
    tau_opt_GC(N-IC1i+2+N:2*N,1:nq.all) = ...
        out_res_opt(1:IC1i-1,1:nq.all)./body_mass;
    % If the first heel strike was on the left foot then we invert so that
    % we always start with the right foot, for analysis purpose
    if strcmp(HS1,'l')
        tau_opt_GC(:,QsSymA_ptx) = tau_opt_GC(:,QsSymB_ptx);
    end

    % Muscle-Tendon Forces
    orderMusInv = [NMuscle/2+1:NMuscle,1:NMuscle/2];
    FTtilde_opt_GC = zeros(N*2,NMuscle);
    FTtilde_opt_GC(1:N-IC1i+1,:) = FTtilde_opt_unsc(IC1i:end,:);
    FTtilde_opt_GC(N-IC1i+2:N-IC1i+1+N,:) = ...
        FTtilde_opt_unsc(1:end,orderMusInv);
    FTtilde_opt_GC(N-IC1i+2+N:2*N,:) = FTtilde_opt_unsc(1:IC1i-1,:);
    % If the first heel strike was on the left foot then we invert so that
    % we always start with the right foot, for analysis purpose
    if strcmp(HS1,'l')
        FTtilde_opt_GC(:,:) = FTtilde_opt_GC(:,orderMusInv);
    end

    % Muscle activations
    a_opt_GC = zeros(N*2,NMuscle);
    a_opt_GC(1:N-IC1i+1,:) = a_opt_unsc(IC1i:end,:);
    a_opt_GC(N-IC1i+2:N-IC1i+1+N,:) = a_opt_unsc(1:end,orderMusInv);
    a_opt_GC(N-IC1i+2+N:2*N,:) = a_opt_unsc(1:IC1i-1,:);
    % If the first heel strike was on the left foot then we invert so that
    % we always start with the right foot, for analysis purpose
    if strcmp(HS1,'l')
        a_opt_GC(:,:) = a_opt_GC(:,orderMusInv);
    end

%     % Time derivative of muscle-tendon force
%     dFTtilde_opt_GC = zeros(N*2,NMuscle);
%     dFTtilde_opt_GC(1:N-IC1i+1,:) = dFTtilde_opt_unsc(IC1i:end,:);
%     dFTtilde_opt_GC(N-IC1i+2:N-IC1i+1+N,:) = ...
%         dFTtilde_opt_unsc(1:end,orderMusInv);
%     dFTtilde_opt_GC(N-IC1i+2+N:2*N,:) = dFTtilde_opt_unsc(1:IC1i-1,:);
%     % If the first heel strike was on the left foot then we invert so that
%     % we always start with the right foot, for analysis purpose
%     if strcmp(HS1,'l')
%         dFTtilde_opt_GC(:,:) = dFTtilde_opt_GC(:,orderMusInv);
%     end

    % Muscle excitations
    vA_opt_GC = zeros(N*2,NMuscle);
    vA_opt_GC(1:N-IC1i+1,:) = vA_opt_unsc(IC1i:end,:);
    vA_opt_GC(N-IC1i+2:N-IC1i+1+N,:) = vA_opt_unsc(1:end,orderMusInv);
    vA_opt_GC(N-IC1i+2+N:2*N,:) = vA_opt_unsc(1:IC1i-1,:);
    % If the first heel strike was on the left foot then we invert so that
    % we always start with the right foot, for analysis purpose
    if strcmp(HS1,'l')
        vA_opt_GC(:,:) = vA_opt_GC(:,orderMusInv);
    end
    e_opt_GC = computeExcitationRaasch(a_opt_GC,vA_opt_GC,...
        ones(1,NMuscle)*tdeact,ones(1,NMuscle)*tact);   
    
    % Back activations
    a_b_opt_GC = zeros(N*2,nq.trunk);
    a_b_opt_GC(1:N-IC1i+1,:) = a_b_opt_unsc(IC1i:end,:);
    a_b_opt_GC(N-IC1i+2:N-IC1i+1+N,:) = a_b_opt_unsc(1:end,:);
    a_b_opt_GC(N-IC1i+2+N:2*N,:) = a_b_opt_unsc(1:IC1i-1,:);
    % If the first heel strike was on the left foot then we invert so that
    % we always start with the right foot, for analysis purpose
    if strcmp(HS1,'l')
        a_b_opt_GC(:,:) = a_b_opt_GC(:,:);
    end

    % Back excitations
    e_b_opt_GC = zeros(N*2,nq.trunk);
    e_b_opt_GC(1:N-IC1i+1,:) = e_b_opt_unsc(IC1i:end,:);
    e_b_opt_GC(N-IC1i+2:N-IC1i+1+N,:) = e_b_opt_unsc(1:end,:);
    e_b_opt_GC(N-IC1i+2+N:2*N,:) = e_b_opt_unsc(1:IC1i-1,:);
    % If the first heel strike was on the left foot then we invert so that
    % we always start with the right foot, for analysis purpose
    if strcmp(HS1,'l')
        e_b_opt_GC(:,:) = e_b_opt_GC(:,:);
    end
  
    %% Save results  
    if saveResults
        hess_names = {'Approximated','Exact'};
        if (exist([pathresults,'/',namescript,'/Results_2D.mat'],'file')==2) 
            load([pathresults,'/',namescript,'/Results_2D.mat']);
        else
            Results_2D.(['Derivative_',setup.derivatives]). ...
                (['Hessian_',hess_names{hessi}]). ...
                (['LinearSolver_',linear_solvers{linsoli}]). ...
                (['MetabolicEnergyRate_',num2str(exp_E)]). ...
                (['InitialGuess_',num2str(IGi)]). ...
                (['MeshIntervals_',num2str(N)])= struct('q_opt',[]);
        end            
        % Put data into structure
        Results_2D.(['Derivative_',setup.derivatives]). ...
            (['Hessian_',hess_names{hessi}]). ...
            (['LinearSolver_',linear_solvers{linsoli}]). ...
            (['MetabolicEnergyRate_',num2str(exp_E)]). ...
            (['InitialGuess_',num2str(IGi)]). ...
            (['MeshIntervals_',num2str(N)]).q_opt_GC = q_opt_GC;
        Results_2D.(['Derivative_',setup.derivatives]). ...
            (['Hessian_',hess_names{hessi}]). ...
            (['LinearSolver_',linear_solvers{linsoli}]). ...
            (['MetabolicEnergyRate_',num2str(exp_E)]). ...
            (['InitialGuess_',num2str(IGi)]). ...
            (['MeshIntervals_',num2str(N)]).qdot_opt_GC = qdot_opt_GC;
%         Results_2D.(['Derivative_',setup.derivatives]). ...
%             (['Hessian_',hess_names{hessi}]). ...
%             (['LinearSolver_',linear_solvers{linsoli}]). ...
%             (['MetabolicEnergyRate_',num2str(exp_E)]). ...
%             (['InitialGuess_',num2str(IGi)]). ...
%             (['MeshIntervals_',num2str(N)]).qdotdot_opt_GC = qdotdot_opt_GC;
        Results_2D.(['Derivative_',setup.derivatives]). ...
            (['Hessian_',hess_names{hessi}]). ...
            (['LinearSolver_',linear_solvers{linsoli}]). ...
            (['MetabolicEnergyRate_',num2str(exp_E)]). ...
            (['InitialGuess_',num2str(IGi)]). ...
            (['MeshIntervals_',num2str(N)]).GRF_opt_GC = GRF_opt_GC;
        Results_2D.(['Derivative_',setup.derivatives]). ...
            (['Hessian_',hess_names{hessi}]). ...
            (['LinearSolver_',linear_solvers{linsoli}]). ...
            (['MetabolicEnergyRate_',num2str(exp_E)]). ...
            (['InitialGuess_',num2str(IGi)]). ...
            (['MeshIntervals_',num2str(N)]).tau_opt_GC = tau_opt_GC;
        Results_2D.(['Derivative_',setup.derivatives]). ...
            (['Hessian_',hess_names{hessi}]). ...
            (['LinearSolver_',linear_solvers{linsoli}]). ...
            (['MetabolicEnergyRate_',num2str(exp_E)]). ...
            (['InitialGuess_',num2str(IGi)]). ...
            (['MeshIntervals_',num2str(N)]).a_opt_GC = a_opt_GC;
        Results_2D.(['Derivative_',setup.derivatives]). ...
            (['Hessian_',hess_names{hessi}]). ...
            (['LinearSolver_',linear_solvers{linsoli}]). ...
            (['MetabolicEnergyRate_',num2str(exp_E)]). ...
            (['InitialGuess_',num2str(IGi)]). ...
            (['MeshIntervals_',num2str(N)]).FTtilde_opt_GC = FTtilde_opt_GC;
        Results_2D.(['Derivative_',setup.derivatives]). ...
            (['Hessian_',hess_names{hessi}]). ...
            (['LinearSolver_',linear_solvers{linsoli}]). ...
            (['MetabolicEnergyRate_',num2str(exp_E)]). ...
            (['InitialGuess_',num2str(IGi)]). ...
            (['MeshIntervals_',num2str(N)]).e_opt_GC = e_opt_GC;
%         Results_2D.(['Derivative_',setup.derivatives]). ...
%             (['Hessian_',hess_names{hessi}]). ...
%             (['LinearSolver_',linear_solvers{linsoli}]). ...
%             (['MetabolicEnergyRate_',num2str(exp_E)]). ...
%             (['InitialGuess_',num2str(IGi)]). ...
%             (['MeshIntervals_',num2str(N)]).dFTtilde_opt_GC = dFTtilde_opt_GC;
        Results_2D.(['Derivative_',setup.derivatives]). ...
            (['Hessian_',hess_names{hessi}]). ...
            (['LinearSolver_',linear_solvers{linsoli}]). ...
            (['MetabolicEnergyRate_',num2str(exp_E)]). ...
            (['InitialGuess_',num2str(IGi)]). ...
            (['MeshIntervals_',num2str(N)]).a_b_opt_GC = a_b_opt_GC;
        Results_2D.(['Derivative_',setup.derivatives]). ...
            (['Hessian_',hess_names{hessi}]). ...
            (['LinearSolver_',linear_solvers{linsoli}]). ...
            (['MetabolicEnergyRate_',num2str(exp_E)]). ...
            (['InitialGuess_',num2str(IGi)]). ...
            (['MeshIntervals_',num2str(N)]).stats = stats;
        % Save data
        save([pathresults,'/',namescript,'/Results_2D.mat'],'Results_2D');
    end % End SaveResults
    
end % End AnalyzeResults

end % End case
