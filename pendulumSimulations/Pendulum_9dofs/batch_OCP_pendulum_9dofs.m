% This script loops over all cases with the 9-dof pendulum analyzed in the paper
%
% Author: Gil Serrancoli
% Date: 16/10/2019

%% Run cases with AD-Recorder, (AD-ADOLC), and FD, using mumps and an
% approximated Hessian of the Lagrangian
Options.solver='mumps'; % linear solver
Options.Hessian='approx'; % Hessian calculation scheme
for der={'AD','FD_F'}
    Options.derivatives=der{:};    
    if strcmp(Options.derivatives,'AD')
        % ADOL-C cases not available
        for type={'Recorder'} % other option is 'ADOLC' but not available
            Options.type=type;
            if strcmp(Options.type,'ADOLC')
                disp('ADOL-C cases not available')
                break;
            end           
            % 10 initial guesses
            for i_IG=1:10
                Options.i_IG=i_IG;
                OCP_pendulum_9dofs; 
            end
        end
    else
        Options.type='Recorder';
        % 10 initial guesses
        for i_IG=1:10
            Options.i_IG=i_IG;
            OCP_pendulum_9dofs; 
        end
    end
end

%% Run cases with AD-Recorder, (the different linear solvers), and the different
% Hessian calculation schemes (i.e., approximated or exact Hessian).
Options.derivatives='AD';
Options.type='Recorder';
% We are not allowed to share the linear solvers from the HSL collection
for linsol={'mumps'} % other options are 'ma27','ma57','ma77','ma86','ma97'
    Options.solver=linsol{:};
    if ~strcmp(Options.solver,'mumps')
        disp('Only mumps is available as linear solver')
        break;
    end
    for Hess={'approx','exact'}
        Options.Hessian=Hess;
        for i_IG=1:10
            Options.i_IG=i_IG;
            OCP_pendulum_9dofs; 
        end
    end
end
