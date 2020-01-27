% PredSim_2D_opt_int_NLPSol
N = 50;
d = 3;
NMuscle = 18;
nq.all = 10;
nq.trunk = 1;
% States
Nkj = N+1 + d*N;
NStates = 2*NMuscle*Nkj + 2*nq.all*Nkj + nq.trunk*Nkj;
% Slack controls
Nj = d*N;
NSlackControls = NMuscle*Nj + nq.all*Nj;
% Controls
NControls = NMuscle*N + nq.trunk*N;
% Static parameters
NParameters = 1;
% Number design variables
NVariables = NStates + NSlackControls + NControls + NParameters;




