% This generic script extracts optimal states and controls from the single
% column vector containing all optimal design variables.
%
% Author: Gil Serrancoli
% Date: 16/10/2019

% Joint positions and velocities at mesh points
for nqi=1:2*nq
    x_opt(:,nqi)=w_opt(nqi:(2*nq+nq+nq+2*nq*d):end);
end
% Joint positions and velocities at collocation points
x_opt_ext=zeros(N*(d+1)+1,nq*2);
x_opt_ext(1:(d+1):end,:)=x_opt;
for nqi=1:2*nq
    x_opt_ext(2:(d+1):end,nqi)=...
        w_opt((2*nq+nq+nq+nqi):(2*nq+nq+nq+2*nq*d):end);
    x_opt_ext(3:(d+1):end,nqi)=...
        w_opt((2*nq+nq+nq+2*nq+nqi):(2*nq+nq+nq+2*nq*d):end);
    x_opt_ext(4:(d+1):end,nqi)=...
        w_opt((2*nq+nq+nq+4*nq+nqi):(2*nq+nq+nq+2*nq*d):end);
end
% Joint torques at mesh points
for nqi=1:nq
    uT_opt(:,nqi)=w_opt((2*nq+nqi):(2*nq+nq+nq+2*nq*d):end);
end
% Joint accelerations at mesh points
for nqi=1:nq
    ua_opt(:,nqi)=w_opt((2*nq+nq+nqi):(2*nq+nq+nq+2*nq*d):end);
end
