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

num_set = [1,1,1,1,0,1]; % This configuration solves the problem
% num_set = [0,1,1,1,0,1]; % This configuration analyzes the results

% The variable settings in the following section (loaded through
% settings_PredSim_2D) will set some parameters of the optimal control problems.
% Through the variable idx_ww, the user can select which row of parameters will
% be used.
% NOTE: at this stage, we only provide the AD-Recorder and FD cases,
% providing the libraries built with ADOL-C is harder (and less relevant, since
% less efficient). Therefore, the ADOL-C cases cannot be run (cases 4-6).
% Further, we are not allowed to share the HSL libraries. Therefore, only the
% cases with the mumps linear solver can be run (cases 1-3, 7-9, and 31-32).
idx_ww = 1; % Index row in matrix settings

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
dof_roti            = [jointi.pelvis.tilt,jointi.hip.l:jointi.trunk.ext];
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
    guess = getGuess_2D_DI_opti_int(Qs_walk,nq,N,time_IC,NMuscle,jointi,scaling,d);
elseif IGi == 3 % Data-informed initial guess   
    % Initial guess based on running data   
    guess = getGuess_2D_DI_opti_int(Qs_run,nq,N,time_IC_run,NMuscle,jointi,scaling.d);
end    
% This allows visualizing the initial guess and the bounds
if checkBoundsIG
    pathPlots = [pathRepo,'/Plots'];
    addpath(genpath(pathPlots));
    plot_BoundsVSInitialGuess_2D
end

%% Formulate the NLP
if solveProblem
    % Create an opti instance
    opti = casadi.Opti();
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Define static parameters
    % Final time
    tf = opti.variable(1,N);
    opti.subject_to(bounds.tf.lower*ones(1,N) < tf < bounds.tf.upper*ones(1,N));
    opti.set_initial(tf, guess.tf*ones(1,N));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Define states 
    % Muscle activations at mesh points
    a = opti.variable(NMuscle,N+1);
    opti.subject_to(bounds.a.lower'*ones(1,N+1) < a < bounds.a.upper'*ones(1,N+1));
    opti.set_initial(a, guess.a');
    % Muscle activations at collocation points
    a_col = opti.variable(NMuscle,d*N);
    opti.subject_to(bounds.a.lower'*ones(1,d*N) < a_col < bounds.a.upper'*ones(1,d*N));
    opti.set_initial(a_col, guess.a_col'); 
    % Muscle-tendon forces at mesh points
    FTtilde = opti.variable(NMuscle,N+1);
    opti.subject_to(bounds.FTtilde.lower'*ones(1,N+1) < FTtilde < bounds.FTtilde.upper'*ones(1,N+1));
    opti.set_initial(FTtilde, guess.FTtilde');
    % Muscle-tendon forces at collocation points
    FTtilde_col = opti.variable(NMuscle,d*N);
    opti.subject_to(bounds.FTtilde.lower'*ones(1,d*N) < FTtilde_col < bounds.FTtilde.upper'*ones(1,d*N));
    opti.set_initial(FTtilde_col, guess.FTtilde_col'); 
    % Qs and Qdots at mesh points
    X = opti.variable(2*nq.all,N+1);
    opti.subject_to(bounds.QsQdots.lower'*ones(1,N+1) < X < bounds.QsQdots.upper'*ones(1,N+1));
    opti.set_initial(X, guess.QsQdots');
    % The initial pelvis_tx position is further constrained
    opti.subject_to(bounds.QsQdots_0.lower(2*jointi.pelvis.tx-1) < X(2*jointi.pelvis.tx-1,1) < bounds.QsQdots_0.upper(2*jointi.pelvis.tx-1)); 
    % Qs and Qdots at collocation points
    X_col = opti.variable(2*nq.all,d*N);
    opti.subject_to(bounds.QsQdots.lower'*ones(1,d*N) < X_col < bounds.QsQdots.upper'*ones(1,d*N));
    opti.set_initial(X_col, guess.QsQdots_col');
    % Back activations at mesh points
    a_b = opti.variable(nq.trunk,N+1);
    opti.subject_to(bounds.a_b.lower'*ones(1,N+1) < a_b < bounds.a_b.upper'*ones(1,N+1));
    opti.set_initial(a_b, guess.a_b');  
    % Back activations at collocation points
    a_b_col = opti.variable(nq.trunk,d*N);
    opti.subject_to(bounds.a_b.lower'*ones(1,d*N) < a_b_col < bounds.a_b.upper'*ones(1,d*N));
    opti.set_initial(a_b_col, guess.a_b_col');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Define controls
    % Time derivative of muscle activations (states) at mesh points
    vA = opti.variable(NMuscle, N);
    opti.subject_to(bounds.vA.lower'*ones(1,N) < vA < bounds.vA.upper'*ones(1,N));
    opti.set_initial(vA, guess.vA');   
    % Arm excitations
    e_b = opti.variable(nq.trunk, N);
    opti.subject_to(bounds.e_b.lower'*ones(1,N) < e_b < bounds.e_b.upper'*ones(1,N));
    opti.set_initial(e_b, guess.e_b');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Define "slack" controls
    % Time derivative of muscle-tendon forces (states) at collocation points
    dFTtilde_col = opti.variable(NMuscle, d*N);
    opti.subject_to(bounds.dFTtilde.lower'*ones(1,d*N) < dFTtilde_col < bounds.dFTtilde.upper'*ones(1,d*N));
    opti.set_initial(dFTtilde_col, guess.dFTtilde_col');
    % Time derivative of Qdots (states) at collocation points
    A_col = opti.variable(nq.all, d*N);
    opti.subject_to(bounds.Qdotdots.lower'*ones(1,d*N) < A_col < bounds.Qdotdots.upper'*ones(1,d*N));
    opti.set_initial(A_col, guess.Qdotdots_col');          
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Parallel formulation
    % Define CasADi variables for static parameters
    tfk = MX.sym('tfk');
    % Define CasADi variables for states
    ak = MX.sym('ak',NMuscle);
    aj = MX.sym('akmesh',NMuscle,d);
    akj = [ak aj];
    FTtildek = MX.sym('FTtildek',NMuscle); 
    FTtildej = MX.sym('FTtildej',NMuscle,d);
    FTtildekj = [FTtildek FTtildej];
    Xk = MX.sym('Xk_nsc',2*nq.all);
    Xj = MX.sym('Xj_nsc',2*nq.all,d);
    Xkj = [Xk Xj];
    a_bk = MX.sym('a_bk',nq.trunk);
    a_bj = MX.sym('a_bj',nq.trunk,d);
    a_bkj = [a_bk a_bj];
    % Define CasADi variables for controls
    vAk = MX.sym('vAk',NMuscle);
    e_bk = MX.sym('e_bk',nq.trunk);
    % Define CasADi variables for "slack" controls
    dFTtildej = MX.sym('dFTtildej',NMuscle,d);
    Aj = MX.sym('Aj',nq.all,d);  
    J = 0; % Initialize cost function
    eq_constr = {}; % Initialize equality constraint vector
    ineq_constr1 = {}; % Initialize inequality constraint vector 1
    ineq_constr2 = {}; % Initialize inequality constraint vector 2
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Time step
    hk = tfk/N;                
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for j=1:d % Loop over collocation points    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Unscale variables        
        Xkj_nsc = Xkj.*(scaling.QsQdots'*ones(1,size(Xkj,2)));
        FTtildekj_nsc = FTtildekj.*(scaling.FTtilde'*ones(1,size(FTtildekj,2)));
        dFTtildej_nsc = dFTtildej.*scaling.dFTtilde;
        Aj_nsc = Aj.*(scaling.Qdotdots'*ones(1,size(Aj,2)));  
        vAk_nsc = vAk.*scaling.vA;              
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Get muscle-tendon lengths, velocities, and moment arms
        % Left leg
        qinj_l = [Xkj_nsc(jointi.hip.l*2-1,j+1),...
            Xkj_nsc(jointi.knee.l*2-1,j+1),...
            Xkj_nsc(jointi.ankle.l*2-1,j+1)];  
        qdotinj_l = [Xkj_nsc(jointi.hip.l*2,j+1),...
            Xkj_nsc(jointi.knee.l*2,j+1),Xkj_nsc(jointi.ankle.l*2,j+1)];  
        [lMTj_l,vMTj_l,MAj_l] = f_lMT_vMT_dM(qinj_l,qdotinj_l);    
        MAj_hip_l    =  MAj_l(mai(1).mus.l',1);
        MAj_knee_l   =  MAj_l(mai(2).mus.l',2);
        MAj_ankle_l  =  MAj_l(mai(3).mus.l',3);    
        % Right leg
        qinj_r = [Xkj_nsc(jointi.hip.r*2-1,j+1),...
            Xkj_nsc(jointi.knee.r*2-1,j+1),...
            Xkj_nsc(jointi.ankle.r*2-1,j+1)];  
        qdotinj_r = [Xkj_nsc(jointi.hip.r*2,j+1),...
            Xkj_nsc(jointi.knee.r*2,j+1), Xkj_nsc(jointi.ankle.r*2,j+1)];      
        [lMTj_r,vMTj_r,MAj_r] = f_lMT_vMT_dM(qinj_r,qdotinj_r);  
        % Here we take the indices from left since the vector is 1:NMuscle/2
        MAj_hip_r    = MAj_r(mai(1).mus.l',1);
        MAj_knee_r   = MAj_r(mai(2).mus.l',2);
        MAj_ankle_r  = MAj_r(mai(3).mus.l',3);
        % Both legs
        lMTj_lr     = [lMTj_l;lMTj_r];
        vMTj_lr     = [vMTj_l;vMTj_r]; 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Get muscle-tendon forces and derive Hill-equilibrium
        [Hilldiffj,FTj,Fcej,Fpassj,Fisoj,~,massMj] = ...
            f_forceEquilibrium_FtildeState(akj(:,j+1),...
            FTtildekj_nsc(:,j+1),dFTtildej_nsc(:,j),lMTj_lr,vMTj_lr,...
            tensions);           
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Get metabolic energy rate if in the cost function   
        if W.mE ~= 0    
            % Get muscle fiber lengths
            [~,lMtildej] = f_FiberLength_TendonForce(...
                FTtildekj_nsc(:,j+1),lMTj_lr); 
            % Get muscle fiber velocities
            [vMj,~] = f_FiberVelocity_TendonForce(FTtildekj_nsc(:,j+1),...
                dFTtildej_nsc(:,j),lMTj_lr,vMTj_lr);
            % Get metabolic energy rate
            [e_totj,~,~,~,~,~] = fgetMetabolicEnergySmooth2004all(...
                akj(:,j+1),akj(:,j+1),lMtildej,vMj,Fcej,Fpassj,massMj,...
                pctsts,Fisoj,MTparameters_m(1,:)',body_mass,10);
        end  
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Collocation            
        % Expression for the state derivatives at the collocation point
        xp_nsc          = Xkj_nsc*C(:,j+1);
        FTtildep_nsc    = FTtildekj_nsc*C(:,j+1);
        ap              = akj*C(:,j+1);
        a_bp            = a_bkj*C(:,j+1);
        % Append collocation equations
        % Dynamic constraints are scaled using the same scale
        % factors as was used to scale the states
        % Activation dynamics (implicit formulation)  
        eq_constr{end+1} = (hk*vAk_nsc - ap)./scaling.a;
        % Contraction dynamics (implicit formulation)         
        eq_constr{end+1} = (hk*dFTtildej_nsc(:,j) - FTtildep_nsc)./scaling.FTtilde';
        % Skeleton dynamics (implicit formulation)  
        qdotj_nsc = Xkj_nsc(2:2:end,j+1); % velocity
        qp_nsc = xp_nsc(1:2:end); % position (state) derivative => velocity
        eq_constr{end+1} = (hk*qdotj_nsc - qp_nsc)./scaling.QsQdots(1:2:end)';
        qdotp_nsc = xp_nsc(2:2:end); % velocity (state) derivative => acceleration
        eq_constr{end+1} = (hk*Aj_nsc(:,j) - qdotp_nsc)./scaling.QsQdots(2:2:end)';
        % Back activation dynamics (explicit formulation)  
        dadtj    = f_BackActivationDynamics(e_bk,a_bkj(:,j+1)');
        eq_constr{end+1} = (hk*dadtj - a_bp)./scaling.a_b;
        % Add contribution to quadrature function
        if W.mE == 0
            J = J + (...
                W.act*B(j+1)    *(f_sumsqr_exp(akj(:,j+1)',exp_A))*hk + ...
                W.back*B(j+1)   *(sumsqr(e_bk))*hk +... 
                W.acc*B(j+1)    *(sumsqr(Aj(:,j)))*hk + ...                          
                W.u*B(j+1)      *(sumsqr(vAk))*hk + ...
                W.u*B(j+1)      *(sumsqr(dFTtildej(:,j)))*hk);  
        elseif W.act == 0
            J = J + (...
                W.mE*B(j+1)     *(f_sumsqr_exp(e_totj,exp_E))*hk + ...
                W.back*B(j+1)   *(sumsqr(e_bk))*hk +... 
                W.acc*B(j+1)    *(sumsqr(Aj(:,j)))*hk + ...                          
                W.u*B(j+1)      *(sumsqr(vAk))*hk + ...
                W.u*B(j+1)      *(sumsqr(dFTtildej(:,j)))*hk);  
        else
            J = J + (...
                W.act*B(j+1)    *(f_sumsqr_exp(akj(:,j+1)',exp_A))*hk + ...
                W.mE*B(j+1)     *(f_sumsqr_exp(e_totj,exp_E))*hk + ...
                W.back*B(j+1)   *(sumsqr(e_bk))*hk +... 
                W.acc*B(j+1)    *(sumsqr(Aj(:,j)))*hk + ...                          
                W.u*B(j+1)      *(sumsqr(vAk))*hk + ...
                W.u*B(j+1)      *(sumsqr(dFTtildej(:,j)))*hk);  
        end  
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Call external function (run inverse dynamics)
        if deri == 2
            [Tkj] = F(Xkj_nsc(:,j+1),Aj_nsc(:,j)); 
        else
            [Tkj] = F([Xkj_nsc(:,j+1);Aj_nsc(:,j)]);      
        end
        % Add path constraints
        % Null pelvis residuals
        eq_constr{end+1} = Tkj(jointi.gr_pelvis,1);
        % Muscle-driven joint torques for the lower limbs and the trunk
        % Hip flexion, left
        Ftkj_hip_l  = FTj(mai(1).mus.l',1);
        Tkj_hip_l   = f_T4(MAj_hip_l,Ftkj_hip_l);
        eq_constr{end+1} = (Tkj(jointi.hip.l,1)-(Tkj_hip_l));
        % Hip flexion, right
        Ftkj_hip_r  = FTj(mai(1).mus.r',1);
        Tkj_hip_r   = f_T4(MAj_hip_r,Ftkj_hip_r);
        eq_constr{end+1} = (Tkj(jointi.hip.r,1)-(Tkj_hip_r));
        % Knee, left
        Ftkj_knee_l = FTj(mai(2).mus.l',1);
        Tkj_knee_l  = f_T5(MAj_knee_l,Ftkj_knee_l);
        eq_constr{end+1} = (Tkj(jointi.knee.l,1)-(Tkj_knee_l));
        % Knee, right
        Ftkj_knee_r = FTj(mai(2).mus.r',1);
        Tkj_knee_r  = f_T5(MAj_knee_r,Ftkj_knee_r);
        eq_constr{end+1} = (Tkj(jointi.knee.r,1)-(Tkj_knee_r));
        % Ankle, left
        Ftkj_ankle_l    = FTj(mai(3).mus.l',1);
        Tkj_ankle_l     = f_T3(MAj_ankle_l,Ftkj_ankle_l);
        eq_constr{end+1} = (Tkj(jointi.ankle.l,1)-(Tkj_ankle_l));
        % Ankle, right
        Ftkj_ankle_r    = FTj(mai(3).mus.r',1);
        Tkj_ankle_r     = f_T3(MAj_ankle_r,Ftkj_ankle_r);
        eq_constr{end+1} = (Tkj(jointi.ankle.r,1)-(Tkj_ankle_r));
        % Torque-driven joint torque for the trunk
        % Trunk
        eq_constr{end+1} = Tkj(jointi.trunk.ext,1)./scaling.BackTau-a_bkj(:,j+1);
        % Contraction dynamics (implicit formulation)
        eq_constr{end+1} = Hilldiffj;
        % Activation dynamics (implicit formulation)
        act1 = vAk_nsc + akj(:,j+1)./(ones(size(akj(:,j+1),1),1)*tdeact);
        act2 = vAk_nsc + akj(:,j+1)./(ones(size(akj(:,j+1),1),1)*tact);    
        ineq_constr1{end+1} = act1;
        ineq_constr2{end+1} = act2;  
    end % End loop over collocation points
    eq_constr = vertcat(eq_constr{:});
    ineq_constr1 = vertcat(ineq_constr1{:});
    ineq_constr2 = vertcat(ineq_constr2{:});
    f_coll = Function('f_coll',{tfk,ak,aj,FTtildek,FTtildej,Xk,Xj,a_bk,a_bj,...
        vAk,e_bk,dFTtildej,Aj},{eq_constr,ineq_constr1, ineq_constr2, J});
    f_coll_map = f_coll.map(N,'openmp',8);
    [coll_eq_constr, coll_ineq_constr1, coll_ineq_constr2, Jall] = ...
        f_coll_map(tf, a(:,1:end-1), a_col, FTtilde(:,1:end-1), FTtilde_col, ...
        X(:,1:end-1), X_col, a_b(:,1:end-1), a_b_col, vA, e_b, dFTtilde_col, ...
        A_col);
    opti.subject_to(coll_eq_constr == 0);
    opti.subject_to(coll_ineq_constr1(:) >= 0);
    opti.subject_to(coll_ineq_constr2(:) <= 1/tact);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Loop over mesh points
    for k=1:N
        % Variables within current mesh interval
        % Static parameters
        tfk = tf(k);
        % States      
        akj = [a(:,k), a_col(:,(k-1)*d+1:k*d)]; 
        FTtildekj = [FTtilde(:,k), FTtilde_col(:,(k-1)*d+1:k*d)];
        Xkj = [X(:,k), X_col(:,(k-1)*d+1:k*d)];
        a_bkj = [a_b(:,k), a_b_col(:,(k-1)*d+1:k*d)];
        % Add equality constraints (next interval starts with end values of 
        % states from previous interval)
        opti.subject_to(a(:,k+1) == akj*D);
        opti.subject_to(FTtilde(:,k+1) == FTtildekj*D); % scaled
        opti.subject_to(X(:,k+1) == Xkj*D); % scaled
        opti.subject_to(a_b(:,k+1) == a_bkj*D);
        if k ~= N
            % Add equality constraints (final time is constant)
            opti.subject_to(tf(k+1) - tfk == 0);
        end
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
    opti.subject_to(X(QsInvA,end)-X(QsInvB,1) == 0);    
    % Muscle activations
    opti.subject_to(a(:,end) - a(orderMusInv,1) == 0);  
    % Muscle-tendon forces
    opti.subject_to(FTtilde(:,end) - FTtilde(orderMusInv,1) == 0); 
    % Back activations
    opti.subject_to(a_b(:,end) - a_b(:,1) == 0); 
    % Average speed
    % Provide expression for the distance traveled
    X_nsc = X.*(scaling.QsQdots'*ones(1,N+1));
    dist_trav_tot = X_nsc((2*jointi.pelvis.tx-1),end) - ...
        X_nsc((2*jointi.pelvis.tx-1),1);
    vel_aver_tot = dist_trav_tot/tf(1,1); 
    opti.subject_to(vel_aver_tot - v_tgt == 0);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    % Scale cost function      
    Jall_sc = sum(Jall)/dist_trav_tot;
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     % Plot Jacobian / Hessian
%     import casadi.*
%     % Jacobian
%     jac = jacobian(vertcat(g{:}),vertcat(w{:}));
%     spy(sparse(DM.ones(jac.sparsity())));
%     % Hessian
%     gtemp = vertcat(g{:});
%     lam = MX.sym('lam', gtemp.sparsity());
%     L = J + dot(lam, gtemp);
%     Hess = hessian(L, vertcat(w{:}));
%     spy(sparse(DM.ones(Hess.sparsity())));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Create an NLP solver
    opti.minimize(Jall_sc);
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
    % Opti does not use bounds on variables but constraints. This function
    % adjusts for that.
    [w_opt,stats] = solve_NLPSOL(opti,options);    
    diary off
    % Create setup
    setup.tolerance.ipopt = tol_ipopt;
    setup.bounds = bounds;
    setup.scaling = scaling;
    setup.guess = guess;
    setup.hessian = hessi;
    setup.N = N;
    % Save results and setup
    save([pathresults,'/',namescript,'/w',savename],'w_opt');
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
        load([pathresults,'/',namescript,'/stats',savename]);
    end
    NParameters = N;    
    tf_opt_all = w_opt(1:NParameters);
    tf_opt = tf_opt_all(1);    
    starti = NParameters+1;
    a_opt = reshape(w_opt(starti:starti+NMuscle*(N+1)-1),NMuscle,N+1)';
    starti = starti + NMuscle*(N+1);
    a_col_opt = reshape(w_opt(starti:starti+NMuscle*(d*N)-1),NMuscle,d*N)';
    starti = starti + NMuscle*(d*N);
    FTtilde_opt = reshape(w_opt(starti:starti+NMuscle*(N+1)-1),NMuscle,N+1)';
    starti = starti + NMuscle*(N+1);
    FTtilde_col_opt = reshape(w_opt(starti:starti+NMuscle*(d*N)-1),NMuscle,d*N)';
    starti = starti + NMuscle*(d*N);
    X_opt = reshape(w_opt(starti:starti+2*nq.all*(N+1)-1),2*nq.all,N+1)';
    starti = starti + 2*nq.all*(N+1);
    X_col_opt = reshape(w_opt(starti:starti+2*nq.all*(d*N)-1),2*nq.all,d*N)';
    starti = starti + 2*nq.all*(d*N);
    a_b_opt = reshape(w_opt(starti:starti+nq.trunk*(N+1)-1),nq.trunk,N+1)';
    starti = starti + nq.trunk*(N+1);
    a_b_col_opt = reshape(w_opt(starti:starti+nq.trunk*(d*N)-1),nq.trunk,d*N)';
    starti = starti + nq.trunk*(d*N);
    vA_opt = reshape(w_opt(starti:starti+NMuscle*N-1),NMuscle,N)';
    starti = starti + NMuscle*N;
    e_b_opt = reshape(w_opt(starti:starti+nq.trunk*N-1),nq.trunk,N)';
    starti = starti + nq.trunk*N;   
    dFTtilde_col_opt = reshape(w_opt(starti:starti+NMuscle*(d*N)-1),NMuscle,d*N)';
    starti = starti + NMuscle*(d*N);
    qdotdot_col_opt = reshape(w_opt(starti:starti+nq.all*(d*N)-1),nq.all,(d*N))';
    starti = starti + nq.all*(d*N);
    % Combine results at mesh and collocation points
    a_mesh_col_opt=zeros(N*(d+1)+1,NMuscle);
    a_mesh_col_opt(1:(d+1):end,:)= a_opt;
    FTtilde_mesh_col_opt=zeros(N*(d+1)+1,NMuscle);
    FTtilde_mesh_col_opt(1:(d+1):end,:)= FTtilde_opt;
    X_mesh_col_opt=zeros(N*(d+1)+1,2*nq.all);
    X_mesh_col_opt(1:(d+1):end,:)= X_opt;
    a_b_mesh_col_opt=zeros(N*(d+1)+1,nq.trunk);
    a_b_mesh_col_opt(1:(d+1):end,:)= a_b_opt;
    for k=1:N
        rangei = k*(d+1)-(d-1):k*(d+1);
        rangebi = (k-1)*d+1:k*d;
        a_mesh_col_opt(rangei,:) = a_col_opt(rangebi,:);
        FTtilde_mesh_col_opt(rangei,:) = FTtilde_col_opt(rangebi,:);
        X_mesh_col_opt(rangei,:) = X_col_opt(rangebi,:);
        a_b_mesh_col_opt(rangei,:) = a_b_col_opt(rangebi,:);
    end
    q_mesh_col_opt = X_mesh_col_opt(:,1:2:end);
    qdot_mesh_col_opt = X_mesh_col_opt(:,2:2:end); 
    q_opt = X_opt(:,1:2:end);
    qdot_opt = X_opt(:,2:2:end);
    q_col_opt = X_col_opt(:,1:2:end);
    qdot_col_opt = X_col_opt(:,2:2:end);        
    if deri == 1
        setup.derivatives = 'AD_Recorder'; % Algorithmic differentiation    
    elseif deri == 2
        setup.derivatives = 'AD_ADOLC'; % Algorithmic differentiation 
    elseif deri == 3
        setup.derivatives = 'FD'; % Finite differences
    end
    setup.hessian = hessi;
    
    %% Unscale results
    % States at mesh points
    % Qs (1:N-1)
    q_opt_unsc.rad = ...
        q_opt(1:end-1,:).*repmat(scaling.Qs,size(q_opt(1:end-1,:),1),1); 
    % Convert in degrees
    q_opt_unsc.deg = q_opt_unsc.rad;    
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
    % Muscle activations (1:N-1)
    a_opt_unsc = a_opt(1:end-1,:).*repmat(...
        scaling.a,size(a_opt(1:end-1,:),1),size(a_opt,2));
    % Muscle-tendon forces (1:N-1)
    FTtilde_opt_unsc = FTtilde_opt(1:end-1,:).*repmat(...
        scaling.FTtilde,size(FTtilde_opt(1:end-1,:),1),1);
    % Back activations (1:N-1)
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
    % States and slack controls at collocation points
    % Qs
    q_col_opt_unsc.rad = q_col_opt.*repmat(scaling.Qs,size(q_col_opt,1),1);   
    % Qdots
    qdot_col_opt_unsc.rad = ...
        qdot_col_opt.*repmat(scaling.Qdots,size(qdot_col_opt,1),1);
    % Time derivative of Qdots
    qdotdot_col_opt_unsc.rad = ...
        qdotdot_col_opt.*repmat(scaling.Qdotdots,size(qdotdot_col_opt,1),1);
    % Convert in degrees
    qdotdot_col_opt_unsc.deg = qdotdot_col_opt_unsc.rad;
    qdotdot_col_opt_unsc.deg(:,dof_roti) = ...
        qdotdot_col_opt_unsc.deg(:,dof_roti).*180/pi;  
    % Time derivative of muscle-tendon forces
    dFTtilde_opt_ext_unsc = dFTtilde_col_opt.*repmat(...
        scaling.dFTtilde,size(dFTtilde_col_opt,1),size(dFTtilde_col_opt,2));    
    
    %% Time grid    
    % Mesh points
    tgrid = linspace(0,tf_opt,N+1);
    dtime = zeros(1,d+1);
    for i=1:4
        dtime(i)=tau_root(i)*(tf_opt/N);
    end
    
    %% Joint torques and ground reaction forces at collocation points 
    X_col_opt_unsc = zeros(d*N,2*nq.all);
    X_col_opt_unsc(:,1:2:end) = q_col_opt_unsc.rad;
    X_col_opt_unsc(:,2:2:end) = qdot_col_opt_unsc.rad;
    out_res_opt_ext = zeros(d*N,nq.all+nGRF);    
    for i = 1:d*N
        [res_ext] = F1([X_col_opt_unsc(i,:)';qdotdot_col_opt_unsc.rad(i,:)']);
        out_res_opt_ext(i,:) = full(res_ext);    
    end
    % assertBackTmax should be 0
    assertBackTmax_ext = ...
        max(max(abs(out_res_opt_ext(:,jointi.trunk.ext)/scaling.BackTau - ...
            (a_b_col_opt))));
    if assertBackTmax_ext > 1*10^(-tol_ipopt)
        disp('Issue when reconstructing residual forces')
    end 
    GRF_col_opt_unsc = out_res_opt_ext(:,GRFi.all);
     % Joint torques and GRF at mesh points  (N-1), except #1
    GRF_opt = GRF_col_opt_unsc(d:d:end,:);
    out_res_opt = out_res_opt_ext(d:d:end,:);    
       
    %% Assert average speed 
    dist_trav_opt = q_opt_unsc_all.rad(end,jointi.pelvis.tx) - ...
        q_opt_unsc_all.rad(1,jointi.pelvis.tx); % distance traveled
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
    phase_tran_tgridi = find(GRF_opt(:,2)<threshold,1,'last');
    if ~isempty(phase_tran_tgridi)        
        if phase_tran_tgridi == N
            temp_idx = find(GRF_opt(:,2)>threshold,1,'first');
            if ~isempty(temp_idx)
                if temp_idx-1 ~= 0 && ...
                        find(GRF_opt(temp_idx-1,2)<threshold)
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
        phase_tran_tgridi = find(GRF_opt(:,4)<threshold,1,'last');       
        if phase_tran_tgridi == N
            temp_idx = find(GRF_opt(:,4)>threshold,1,'first');
            if ~isempty(temp_idx)  
                if temp_idx-1 ~= 0 && ...
                        find(GRF_opt(temp_idx-1,4)<threshold)
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
    
    % TODO: GRF_opt is at mesh points starting from k=2, we should thus
    % add 1 to IC1i
    IC1i = IC1i + 1;

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

%     % Ground reaction forces
%     GRF_opt_GC = zeros(N*2,nGRF);
%     GRF_opt_GC(1:N-IC1i+1,:) = GRF_opt(IC1i:end,:);
%     GRF_opt_GC(N-IC1i+2:N-IC1i+1+N,:) = GRF_opt(1:end,[3:4,1:2]);
%     GRF_opt_GC(N-IC1i+2+N:2*N,:) = GRF_opt(1:IC1i-1,:);
%     GRF_opt_GC = GRF_opt_GC./(body_weight/100);
%     % If the first heel strike was on the left foot then we invert so that
%     % we always start with the right foot, for analysis purpose
%     if strcmp(HS1,'l')
%         GRF_opt_GC(:,[3:4,1:2]) = GRF_opt_GC(:,:);
%     end  
    
%     % Joint torques
%     tau_opt_GC = zeros(N*2,size(q_opt,2));
%     tau_opt_GC(1:N-IC1i+1,1:nq.all) = ...
%         out_res_opt(IC1i:end,1:nq.all)./body_mass;
%     tau_opt_GC(N-IC1i+2:N-IC1i+1+N,QsSymA_ptx) = ...
%         out_res_opt(1:end,QsSymB_ptx)./body_mass;
%     tau_opt_GC(N-IC1i+2+N:2*N,1:nq.all) = ...
%         out_res_opt(1:IC1i-1,1:nq.all)./body_mass;
%     % If the first heel strike was on the left foot then we invert so that
%     % we always start with the right foot, for analysis purpose
%     if strcmp(HS1,'l')
%         tau_opt_GC(:,QsSymA_ptx) = tau_opt_GC(:,QsSymB_ptx);
%     end

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
%         Results_2D.(['Derivative_',setup.derivatives]). ...
%             (['Hessian_',hess_names{hessi}]). ...
%             (['LinearSolver_',linear_solvers{linsoli}]). ...
%             (['MetabolicEnergyRate_',num2str(exp_E)]). ...
%             (['InitialGuess_',num2str(IGi)]). ...
%             (['MeshIntervals_',num2str(N)]).GRF_opt_GC = GRF_opt_GC;
%         Results_2D.(['Derivative_',setup.derivatives]). ...
%             (['Hessian_',hess_names{hessi}]). ...
%             (['LinearSolver_',linear_solvers{linsoli}]). ...
%             (['MetabolicEnergyRate_',num2str(exp_E)]). ...
%             (['InitialGuess_',num2str(IGi)]). ...
%             (['MeshIntervals_',num2str(N)]).tau_opt_GC = tau_opt_GC;
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
