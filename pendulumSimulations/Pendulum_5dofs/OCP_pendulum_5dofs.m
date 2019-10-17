%%  Simulation of an inverted 5-dof pendulum subject to a perturbation.
% The optimal control problem identifies the joint torques necessary to restore
% the pendulum’ upright posture within one second while minimizing the actuator
% effort (i.e., squared joint torques) and satisfying the pendulum dynamics.
%
% See detailed formulation in Falisse A., Serrancolí G., Dembia C.L., Gillis J.,
% De Groote F. Algorithmic differentiation improves the computational efficiency
% of OpenSim-based trajectory optimization of human movement, (2019). PLOS ONE.
%
% Author: Gil Serrancoli
% Date: 9/9/2019
%
% DEPENDENCY: please install CasADi (https://web.casadi.org/)
%
clc
clearvars -except Options 
fclose all;
import casadi.*

pathmain=pwd;
[pathRepo,~,~] = fileparts(pathmain);

% Fixed settings
Options.tol=2; tol=2; % IPOPT tolerance is 10^(-6).
Options.N=25; % Number of mesh interval.
% Initial and final positions and velocities imposed through path constraints
% (1) or through variable bounds (0).
Options.constrIFpoint=1;
i_IG=Options.i_IG; % Initial guess case

% Variable settings
if strcmp(Options.type,'Recorder')
    pathexternal = [pathmain,'/ExternalFunctions'];
    cd(pathexternal);
elseif strcmp(Options.type,'ADOLC')
    disp('ADOL-C cases not available')
end
if ~strcmp(Options.solver,'mumps')
    disp('Only mumps is available as linear solver')
end
switch Options.derivatives
    case 'AD'
        if strcmp(Options.type,'Recorder')
            if strcmp(Options.Hessian,'exact')
                F = external('F','Pendulum_5dofs_exactHessian.dll'); 
            elseif strcmp(Options.Hessian,'approx')
                F = external('F','Pendulum_5dofs.dll'); 
            end
        else
            % ADOL-C cases not available
        end
    case 'FD_F'
        F = external('F','Pendulum_5dofs.dll',struct('enable_fd',true,...
        'enable_forward', false, 'enable_reverse', false, ...
        'enable_jacobian',false,'fd_method', 'forward'));
end
cd(pathmain);

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

%% Discretization
% Time horizon
T = 1;
% Number of mesh intervals
N = Options.N; 
% Step size
h = T/N;

%% Perturbation
load([pathRepo,'/platformPerturbation.mat']);
% Process the platform perturbation data
aPLAT = - data.analog.platemotion(534:534+2*1080,6);
atime = (0:1:2*1080)/1080;
tsim = (0:1/N:1)';
tau_root_dis = tau_root/N;
temp(1:4:(N*(d+1)+1)) = tsim;
for i = 1:length(tsim)-1
    temp((i-1)*(d+1)+2) = tsim(i)+tau_root_dis(2);  
    temp((i-1)*(d+1)+3) = tsim(i)+tau_root_dis(3);
    temp((i-1)*(d+1)+4) = tsim(i)+tau_root_dis(4);
end
pert = interp1(atime,aPLAT,temp);

%% Bounds and scaling
nq=5; % number of mobilities
scaling.time=1/5;
scaling.q=3;
scaling.uT=500;
scaling.qdot=scaling.q/scaling.time;
scaling.states=[scaling.q*ones(nq,1); scaling.qdot*ones(nq,1)]; 
scaling.ua=scaling.q/(scaling.time^2);
w_imp=1e-1;

bounds.q=[-pi/2 pi/2]./scaling.q;
bounds.qdot=[-20 20]/scaling.qdot;
bounds.ua=[-500 500]/scaling.ua;
bounds.uT=[-1000 1000]/scaling.uT;

%% Load initial guess
if N==25
    load([pathmain,'/initialGuesses.mat']);
end

%% Declare CasADi function
% Function for sum of squared values 
e = MX.sym('e',nq);
Jtemp = 0;
for i=1:length(e)
    Jtemp = Jtemp + e(i).^2;
end
f_J = Function('f_J',{e},{Jtemp});

%% Formulate the NLP
% Start with an empty NLP
w   = {};
w0  = [];
lbw = [];
ubw = [];
J   = 0;
g   = {};
lbg = [];
ubg = [];

% "Lift" initial conditions
% States
X0  = MX.sym('X0', 2*nq);
w   = {w{:}, X0};
if Options.constrIFpoint
    lbw = [lbw; repmat([bounds.q(1); bounds.qdot(1)],nq,1)];
    ubw = [ubw; repmat([bounds.q(2); bounds.qdot(2)],nq,1)];
    g = {g{:}, X0}; 
    lbg=[lbg; zeros(2*nq,1)];
    ubg=[ubg; zeros(2*nq,1)];
else
    lbw = [lbw; zeros(2*nq,1)];
    ubw = [ubw; zeros(2*nq,1)];
end
w0  = [w0;  [IG(i_IG).q(1,:)'; IG(i_IG).qdot(1,:)']./scaling.states];
Xk = X0;
Xk_nsc=Xk.*scaling.states;

% Loop over mesh points
for k=0:N-1
    % Controls at mesh points
    % Torques   
    Uk  = MX.sym(['U_' num2str(k)], nq);
    w   = {w{:}, Uk};
    lbw = [lbw; bounds.uT(1)*ones(nq,1)];
    ubw = [ubw; bounds.uT(2)*ones(nq,1)];
    w0  = [w0; zeros(nq,1)];
    % Accelerations
    Ak  = MX.sym(['A_' num2str(k)], nq);
    w   = {w{:}, Ak};
    lbw = [lbw; bounds.ua(1)*ones(nq,1)];
    ubw = [ubw; bounds.ua(2)*ones(nq,1)];
    w0  = [w0; IG(i_IG).qd2dot(k*(d+1)+1,:)'./scaling.ua];  
    Ak_nsc=Ak*scaling.ua;

    % States at collocation points
    Xkj = {};
    for j=1:d
        Xkj{j}  = MX.sym(['X_' num2str(k) '_' num2str(j)], 2*nq);
        w       = {w{:}, Xkj{j}};
        lbw     = [lbw; repmat([bounds.q(1); bounds.qdot(1)],nq,1)];
        ubw     = [ubw; repmat([bounds.q(2); bounds.qdot(2)],nq,1)];
        w0      = [w0; [IG(i_IG).q(k*(d+1)+1+j,:)'; ...
            IG(i_IG).qdot(k*(d+1)+1+j,:)']./scaling.states];   
        Xkj_nsc{j}=Xkj{j}.*scaling.states;
    end
    
    % Discretized perturbation
    pert_k = pert(k*(d+1)+1);   

    % Loop over collocation points
    Xk_end = D(1)*Xk_nsc;
    for j=1:d
       % Expression for the state derivative at the collocation point
       xp = C(1,j+1)*Xk_nsc;
       for r=1:d
           xp = xp + C(r+1,j+1)*Xkj_nsc{r};
       end
       % Append collocation equations (implicit formulation)
       fj=[Xkj_nsc{j}(2); Ak_nsc(1); Xkj_nsc{j}(4); Ak_nsc(2); ...
           Xkj_nsc{j}(6); Ak_nsc(3); Xkj_nsc{j}(8); Ak_nsc(4); ...
           Xkj_nsc{j}(10); Ak_nsc(5)]; 
       g = {g{:}, (h*fj - xp)./scaling.states};
       lbg = [lbg; zeros(2*nq,1)];
       ubg = [ubg; zeros(2*nq,1)];
       % Add contribution to the end state
       Xk_end = Xk_end + D(j+1)*Xkj_nsc{j};
       % Add contribution to quadrature function
       J = J + B(j+1)*f_J(Uk)*h + w_imp*B(j+1)*f_J(Ak)*h;
    end  
    
    % Extract the joint torques through the external function.
    if strcmp(Options.type,'Recorder')
        [qkj] = F([Xk_nsc;Ak_nsc;pert_k]);
    elseif strcmp(Options.type,'ADOLC')
        [qkj] = F(Xk_nsc,Ak_nsc,pert_k);
    end    
    
    % Add path constraints (implicit formulation)
    g = {g{:},Uk-qkj/scaling.uT};
    lbg = [lbg; zeros(nq,1)];
    ubg = [ubg; zeros(nq,1)];   
    
    % States at mesh points
    Xk  = MX.sym(['X_' num2str(k+1)], 2*nq);
    w   = {w{:}, Xk};
    if k == N-1
        if Options.constrIFpoint
            lbw = [lbw; repmat([bounds.q(1); bounds.qdot(1)],nq,1)];
            ubw = [ubw; repmat([bounds.q(2); bounds.qdot(2)],nq,1)];
            g = {g{:}, Xk}; 
            lbg=[lbg; zeros(2*nq,1)];
            ubg=[ubg; zeros(2*nq,1)];
        else
            lbw = [lbw; zeros(2*nq,1)];
            ubw = [ubw; zeros(2*nq,1)];  
        end 
    else
        lbw = [lbw; repmat([bounds.q(1); bounds.qdot(1)],nq,1)];
        ubw = [ubw; repmat([bounds.q(2); bounds.qdot(2)],nq,1)];
    end
    w0  = [w0; [IG(i_IG).q((k+1)*(d+1)+1,:)'; IG(i_IG).qdot((k+1)*(d+1)+1,:)']./scaling.states];
    Xk_nsc=Xk.*scaling.states;

    % Add equality constraints
    g = {g{:}, Xk_end-Xk_nsc};
    lbg = [lbg; zeros(2*nq,1)];
    ubg = [ubg; zeros(2*nq,1)];
end

%% Solve problem
% Create an NLP solver
prob = struct('f', J, 'x', vertcat(w{:}), 'g', vertcat(g{:}));
if strcmp(Options.Hessian,'approx')
    options.ipopt.hessian_approximation = 'limited-memory';
else
    options.ipopt.hessian_approximation='exact';
end
options.ipopt.mu_strategy = 'adaptive';
options.ipopt.linear_solver=Options.solver;
if strcmp(Options.derivatives,'FD_F') 
    options.common_options.helper_options = ...
            struct('enable_fd',true,'enable_forward',false,...
            'enable_reverse',false,'print_in',false,...
            'fd_method','forward');
end
if tol==1
    options.ipopt.tol=1e-4;
else
    options.ipopt.tol=1e-6;
end
solver = nlpsol('solver', 'ipopt', prob, options);

diary_file_name=['diary_file_' Options.derivatives '.txt'];
pathResults = [pathRepo,'/Results/Pendulum_5dofs/'];
if ~(exist(pathResults,'dir')==7)
    mkdir(pathResults);
end
diary([pathResults,diary_file_name]);

name_solution=['solution_' Options.derivatives];
if (exist([pathResults,'/',name_solution,'.mat'],'file')==2) 
    load([pathResults,'/',name_solution,'.mat']);
end

% Solve the NLP
disp(['initial guess ' num2str(i_IG)]);
sol = solver('x0', w0, 'lbx', lbw, 'ubx', ubw, 'lbg', lbg, 'ubg', ubg);
w_opt = full(sol.x);
stats=solver.stats;
if strcmp(Options.Hessian,'exact')
    dim3=1;
else
    dim3=2;
end
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
if strcmp(Options.type,'Recorder')
    dim5=1;
elseif strcmp(Options.type,'ADOLC')
    dim5=2;
end
solution(i_IG,tol,dim3,dim4,dim5).f=full(sol.f);
solution(i_IG,tol,dim3,dim4,dim5).g=full(sol.g);
solution(i_IG,tol,dim3,dim4,dim5).x=full(sol.x);
solution(i_IG,tol,dim3,dim4,dim5).lam_g=full(sol.lam_g);
solution(i_IG,tol,dim3,dim4,dim5).lam_x=full(sol.lam_x);
solution(i_IG,tol,dim3,dim4,dim5).stats=stats;
pathExtractSolution = [pathRepo,'/ExtractSolution'];
addpath(genpath(pathExtractSolution));
extract_solution;
solution(i_IG,tol,dim3,dim4,dim5).x_opt=x_opt;
solution(i_IG,tol,dim3,dim4,dim5).x_opt_ext=x_opt_ext;
solution(i_IG,tol,dim3,dim4,dim5).uT_opt=uT_opt;
solution(i_IG,tol,dim3,dim4,dim5).ua_opt=ua_opt;
save([pathResults,'/',name_solution,'.mat'],'solution');

diary off;

F.delete;
clear F;
