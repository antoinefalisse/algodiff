% This script creates random initial guesses. To allow reproducibility, we
% saved the results in a .mat file.
%
% Author: Gil Serrancoli
% Date: 16/10/2019

ndof=9; % Number of degrees of freedom
d=3;    % Number of collocation points
N=50;   % Number of mesh intervals
T=1;    % Time horizon in s

main_folder=pwd;
[pathRepo,~,~] = fileparts(main_folder);
pathCollocationScheme = [pathRepo,'\CollocationScheme'];
addpath(genpath(pathCollocationScheme));
method = 'radau';
[tau_root,~,~,~] = CollocationScheme(d,method);

dtime = zeros(1,4);
for i=1:4
    dtime(i)=tau_root(i)*T/N;
end
tgrid_ext=[];
tgrid = linspace(0, T, N+1); % tgrid only contains mesh points
for i=1:N
    tgrid_ext(((i-1)*4+1):1:i*4)=tgrid(i)+dtime;
end
tgrid_ext(end+1)=T; %tgrid_ext contains mesh points and collocation points
tgrid2=linspace(0,T,4);

IG = struct('q',[]);
for i=1:10
    if i==1
        IG(i).q=zeros(N*(d+1)+1,ndof);
        IG(i).qdot=zeros(N*(d+1)+1,ndof);
        IG(i).qd2dot=zeros(N*(d+1)+1,ndof);
    else
        % random angles between -20 and 20 degrees
        qrand=rand(4,ndof)*40*pi/180-20*pi/180; 
        qrand_spline=spline(tgrid2,qrand');
        IG(i).q=ppval(qrand_spline,tgrid_ext)';
        qdotrand_spline=fnder(qrand_spline,1);
        IG(i).qdot=ppval(qdotrand_spline,tgrid_ext)';
        qd2dotrand_spline=fnder(qrand_spline,2);
        IG(i).qd2dot=ppval(qd2dotrand_spline,tgrid_ext)';
    end
end
