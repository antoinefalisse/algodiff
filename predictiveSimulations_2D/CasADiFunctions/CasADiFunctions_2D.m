% This script contains several CasADi-based functions that are used when solving
% the OCPs
%
% Author: Antoine Falisse
% Date: 9/9/2019
%
import casadi.*

%% Polynomial approximation
pathpolynomial = [pathRepo,'\Polynomials'];
addpath(genpath(pathpolynomial));
muscle_spanning_info_m = muscle_spanning_joint_INFO(musi_pol,:);
MuscleInfo_m.muscle    = MuscleInfo.muscle(musi_pol);                  
qin     = SX.sym('qin',1,nq.leg);
qdotin  = SX.sym('qdotin',1,nq.leg);
lMT     = SX(NMuscle_pol,1);
vMT     = SX(NMuscle_pol,1);
dM      = SX(NMuscle_pol,nq.leg);
for i=1:NMuscle_pol      
    index_dof_crossing  = find(muscle_spanning_info_m(i,:)==1);
    order               = MuscleInfo_m.muscle(i).order;
    [mat,diff_mat_q]    = n_art_mat_3_cas_SX(qin(1,index_dof_crossing),order);
    lMT(i,1)            = mat*MuscleInfo_m.muscle(i).coeff;
    vMT(i,1)            = 0;
    dM(i,1:nq.leg)      = 0;
    nr_dof_crossing     = length(index_dof_crossing); 
    for dof_nr = 1:nr_dof_crossing
        dM(i,index_dof_crossing(dof_nr)) = ...
            (-(diff_mat_q(:,dof_nr)))'*MuscleInfo_m.muscle(i).coeff;
        vMT(i,1) = vMT(i,1) + (-dM(i,index_dof_crossing(dof_nr))*...
            qdotin(1,index_dof_crossing(dof_nr)));
    end 
end
f_lMT_vMT_dM = Function('f_lMT_vMT_dM',{qin,qdotin},{lMT,vMT,dM});

%% Normalized sum of values to a certain power
etempNMuscle = SX.sym('etempNMuscle',NMuscle);
exp          = SX.sym('exp',1);
JtempNMuscle = 0;
for i=1:length(etempNMuscle)
    JtempNMuscle = JtempNMuscle + etempNMuscle(i).^exp;
end
f_sumsqr_exp=Function('f_sumsqr_exp',{etempNMuscle,exp},{JtempNMuscle});

%% Sum of products  
% Function for 3 elements 
ma_temp3 = SX.sym('ma_temp3',3);
ft_temp3 = SX.sym('ft_temp3',3);
J_sptemp3 = 0;
for i=1:length(ma_temp3)
    J_sptemp3 = J_sptemp3 + ma_temp3(i,1)*ft_temp3(i,1);    
end
f_T3 = Function('f_T3',{ma_temp3,ft_temp3},{J_sptemp3});
% Function for 4 elements 
ma_temp4 = SX.sym('ma_temp4',4);
ft_temp4 = SX.sym('ft_temp4',4);
J_sptemp4 = 0;
for i=1:length(ma_temp4)
    J_sptemp4 = J_sptemp4 + ma_temp4(i,1)*ft_temp4(i,1);    
end
f_T4 = Function('f_T4',{ma_temp4,ft_temp4},{J_sptemp4});
% Function for 5 elements 
ma_temp5 = SX.sym('ma_temp5',5);
ft_temp5 = SX.sym('ft_temp5',5);
J_sptemp5 = 0;
for i=1:length(ma_temp5)
    J_sptemp5 = J_sptemp5 + ma_temp5(i,1)*ft_temp5(i,1);    
end
f_T5 = Function('f_T5',{ma_temp5,ft_temp5},{J_sptemp5});

%% Muscle contraction dynamics
pathmusclemodel = [pathRepo,'\MuscleModel'];
addpath(genpath(pathmusclemodel));
% Function for Hill-equilibrium
FTtilde     = SX.sym('FTtilde',NMuscle); % Normalized tendon forces
a           = SX.sym('a',NMuscle); % Muscle activations
dFTtilde    = SX.sym('dFTtilde',NMuscle); % Time derivative tendon forces
lMT         = SX.sym('lMT',NMuscle); % Muscle-tendon lengths
vMT         = SX.sym('vMT',NMuscle); % Muscle-tendon velocities
tension_SX  = SX.sym('tension',NMuscle); % Tensions
Hilldiff    = SX(NMuscle,1); % Hill-equilibrium
FT          = SX(NMuscle,1); % Tendon forces
Fce         = SX(NMuscle,1); % Contractile element forces
Fpass       = SX(NMuscle,1); % Passive element forces
Fiso        = SX(NMuscle,1); % Normalized forces from force-length curve
vMmax       = SX(NMuscle,1); % Maximum contraction velocities
massM       = SX(NMuscle,1); % Muscle mass
% Parameters of force-length-velocity curves
load Fvparam
load Fpparam
load Faparam
for m = 1:NMuscle
    [Hilldiff(m),FT(m),Fce(m),Fpass(m),Fiso(m),vMmax(m),massM(m)] = ...
        ForceEquilibrium_FtildeState(a(m),FTtilde(m),dFTtilde(m),...
            lMT(m),vMT(m),MTparameters_m(:,m),Fvparam,Fpparam,Faparam,...
            tension_SX(m));
end
f_forceEquilibrium_FtildeState = ...
    Function('f_forceEquilibrium_FtildeState',{a,FTtilde,dFTtilde,...
        lMT,vMT,tension_SX},{Hilldiff,FT,Fce,Fpass,Fiso,vMmax,massM});
    
% Function to get (normalized) muscle fiber lengths
lM      = SX(NMuscle,1);
lMtilde = SX(NMuscle,1);
for m = 1:NMuscle
    [lM(m),lMtilde(m)] = FiberLength_TendonForce(FTtilde(m),...
        MTparameters_m(:,m),lMT(m));
end
f_FiberLength_TendonForce = Function('f_FiberLength_Ftilde',...
    {FTtilde,lMT},{lM,lMtilde});

% Function to get (normalized) muscle fiber velocities
vM      = SX(NMuscle,1);
vMtilde = SX(NMuscle,1);
for m = 1:NMuscle
    [vM(m),vMtilde(m)] = FiberVelocity_TendonForce(FTtilde(m),...
        dFTtilde(m),MTparameters_m(:,m),lMT(m),vMT(m));
end
f_FiberVelocity_TendonForce = Function('f_FiberVelocity_Ftilde',...
    {FTtilde,dFTtilde,lMT,vMT},{vM,vMtilde});

%% Back activation dynamics
e_b = SX.sym('e_b',nq.trunk); % back excitations
a_b = SX.sym('a_b',nq.trunk); % back activations
dadt = BackActivationDynamics(e_b,a_b);
f_BackActivationDynamics = ...
    Function('f_BackActivationDynamics',{e_b,a_b},{dadt});

%% Metabolic energy model
if W.mE ~= 0
    act_SX          = SX.sym('act_SX',NMuscle,1); % Muscle activations
    exc_SX          = SX.sym('exc_SX',NMuscle,1); % Muscle excitations
    lMtilde_SX      = SX.sym('lMtilde_SX',NMuscle,1); % N muscle fiber lengths
    vMtilde_SX      = SX.sym('vMtilde_SX',NMuscle,1); % N muscle fiber vel
    vM_SX           = SX.sym('vM_SX',NMuscle,1); % Muscle fiber velocities
    Fce_SX          = SX.sym('FT_SX',NMuscle,1); % Contractile element forces
    Fpass_SX        = SX.sym('FT_SX',NMuscle,1); % Passive element forces
    Fiso_SX         = SX.sym('Fiso_SX',NMuscle,1); % N forces (F-L curve)
    musclemass_SX   = SX.sym('musclemass_SX',NMuscle,1); % Muscle mass 
    vcemax_SX       = SX.sym('vcemax_SX',NMuscle,1); % Max contraction vel
    pctst_SX        = SX.sym('pctst_SX',NMuscle,1); % Slow twitch ratio 
    Fmax_SX         = SX.sym('Fmax_SX',NMuscle,1); % Max iso forces
    modelmass_SX    = SX.sym('modelmass_SX',1); % Model mass
    b_SX            = SX.sym('b_SX',1); % Parameter determining tanh smoothness
    % Bhargava et al. (2004)
    [energy_total_sm_SX,Adot_sm_SX,Mdot_sm_SX,Sdot_sm_SX,Wdot_sm_SX,...
        energy_model_sm_SX] = getMetabolicEnergySmooth2004all(exc_SX,act_SX,...
        lMtilde_SX,vM_SX,Fce_SX,Fpass_SX,musclemass_SX,pctst_SX,Fiso_SX,...
        Fmax_SX,modelmass_SX,b_SX);
    fgetMetabolicEnergySmooth2004all = ...
        Function('fgetMetabolicEnergySmooth2004all',...
        {exc_SX,act_SX,lMtilde_SX,vM_SX,Fce_SX,Fpass_SX,musclemass_SX,...
        pctst_SX,Fiso_SX,Fmax_SX,modelmass_SX,b_SX},{energy_total_sm_SX,...
        Adot_sm_SX,Mdot_sm_SX,Sdot_sm_SX,Wdot_sm_SX,energy_model_sm_SX});
end
