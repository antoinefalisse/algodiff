% settings is a structure that allows easily switching between cases tested
% in the simulations
%
% Author: Antoine Falisse
% Date: 9/9/2019
%
% settings(1): derivative supplier identifier
%   1: Algorithmic differentiation / Recorder  
%   2: Algorithmic differentiation / ADOL-C  
%   3: Finite differences
% settings(2): Hessian identifier
%   1: Approximated Hessian
%   2: Exact Hessian
% settings(3):  linear solver identifier
%   1: mumps
%   2: ma27
%   3: ma57
%   4: ma77
%   5: ma86
%   6: ma97
% settings(4): metabolic energy rate
%   0: metabolic energy rate not included in the cost function
%   1: metabolic energy rate included in the cost function
% settings(5): initial guess identifier
%   1: quasi-random initial guess  
%   2: data-informed initial guess (data from a walking trial)
%   3: data-informed initial guess (data from a running trial)
settings = [
    % A. Impact of the derivative supplier
    % Recorder - Approximated Hessian - mumps
    1, 1, 1, 0, 2, 25; % 1
    1, 1, 1, 0, 2, 50; % 2
    1, 1, 1, 0, 2, 100; % 3    
    1, 1, 1, 0, 2, 200; % 4
    ];
