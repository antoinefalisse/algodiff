% This script provides an inital guess for the design variables. The guess is
% quasi-random (QR). We set constant values to the muscle variables, the arm
% variables and most joint variables. We only ensure that the distance traveled
% is not null. The model is moving forward at a constant speed and is standing
% on the ground.
%
% Author: Antoine Falisse
% Date: 12/19/2018
% 
function guess = getGuess_3D_QR(Qs,nq,N,NMuscle,jointi,time_opt,scaling)

%% Qs
% The model is moving forward but with a standing position (Qs=0)
guess.Qs = zeros(N,nq.all);
% Pelvis_tx
Qs_temp = Qs.allinterpfilt;
dist = Qs_temp(:,strcmp(Qs.colheaders(1,:),'pelvis_tx'));
v_tgt = (dist(end)-dist(1))/(time_opt(2)-time_opt(1));
guess.Qs(:,jointi.pelvis.tx) = linspace(0,(time_opt(2)-time_opt(1))*v_tgt,N)+dist(1);
% Pelvis_ty
guess.Qs(:,jointi.pelvis.ty) = 0.9385;
% Pelvis_tz
% We adjust pelvis tz so that it is within the bounds.
guess.Qs(:,jointi.pelvis.tz) = mean(Qs_temp(:,strcmp(Qs.colheaders(1,:),'pelvis_tz')));

%% Qdots: data-informed
guess.Qdots = zeros(N,nq.all);
% The model is moving forward with a constant speed
guess.Qdots(:,jointi.pelvis.tx) = v_tgt;
% Qs and Qdots are intertwined
guess.QsQdots = zeros(N,2*nq.all);
guess.QsQdots(:,1:2:end) = guess.Qs;
guess.QsQdots(:,2:2:end) = guess.Qdots;

%% Qdotdots
guess.Qdotdots = zeros(N,nq.all);

%% Muscle variables
guess.a = 0.1*ones(N,NMuscle);
guess.vA = 0.01*ones(N,NMuscle);
guess.FTtilde = 0.1*ones(N,NMuscle);
guess.dFTtilde = 0.01*ones(N,NMuscle);

%% Arm activations
guess.a_a = 0.1*ones(N,nq.arms);
guess.e_a = 0.1*ones(N,nq.arms);

%% Parameters contact model
% Original values
B_locSphere_s1_r    = [0.00190115788407966, -0.00382630379623308];
B_locSphere_s2_r    = [0.148386399942063, -0.028713422052654];
B_locSphere_s3_r    = [0.133001170607051, 0.0516362473449566];
B_locSphere_s4_r    = [0.06, -0.0187603084619177];    
B_locSphere_s5_r    = [0.0662346661991635, 0.0263641606741698];
B_locSphere_s6_r    = [0.045, 0.0618569567549652];
IG_rad              = 0.032*ones(1,6); 
guess.params = [B_locSphere_s1_r,B_locSphere_s2_r,...
    B_locSphere_s3_r,B_locSphere_s4_r,B_locSphere_s5_r,B_locSphere_s6_r,...
    IG_rad];

%% Scaling
guess.QsQdots   = guess.QsQdots./repmat(scaling.QsQdots,N,1);
guess.Qdotdots  = guess.Qdotdots./repmat(scaling.Qdotdots,N,1);
guess.a         = (guess.a)./repmat(scaling.a,N,size(guess.a,2));
guess.FTtilde   = (guess.FTtilde)./repmat(scaling.FTtilde,N,1);
guess.vA        = (guess.vA)./repmat(scaling.vA,N,size(guess.vA,2));
guess.dFTtilde  = (guess.dFTtilde)./repmat(scaling.dFTtilde,N,...
    size(guess.dFTtilde,2));
% no need to scale the IG for the arm activations / excitations
guess.params        = guess.params.*scaling.params.v + scaling.params.r;

end