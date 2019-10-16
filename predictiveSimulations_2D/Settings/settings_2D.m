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
    1, 1, 1, 0, 1; % 1
    1, 1, 1, 0, 2; % 2
    1, 1, 1, 0, 3; % 3
    % ADOL-C - Approximated Hessian - mumps
    2, 1, 1, 0, 1; % 4
    2, 1, 1, 0, 2; % 5
    2, 1, 1, 0, 3; % 6
    % FD - Approximated Hessian - mumps
    3, 1, 1, 0, 1; % 7
    3, 1, 1, 0, 2; % 8
    3, 1, 1, 0, 3; % 9
    % B. Impact of the linear solver
    % Recorder - Approximated Hessian - ma27    
    1, 1, 2, 0, 1; % 10
    1, 1, 2, 0, 2; % 11
    1, 1, 2, 0, 3; % 12
    % Recorder - Approximated Hessian - ma57
    1, 1, 3, 0, 1; % 13
    1, 1, 3, 0, 2; % 14
    1, 1, 3, 0, 3; % 15
    % Recorder - Approximated Hessian - ma77
    1, 1, 4, 0, 1; % 16
    1, 1, 4, 0, 2; % 17
    1, 1, 4, 0, 3; % 18
    % Recorder - Approximated Hessian - ma86
    1, 1, 5, 0, 1; % 19
    1, 1, 5, 0, 2; % 20
    1, 1, 5, 0, 3; % 21
    % Recorder - Approximated Hessian - ma97
    1, 1, 6, 0, 1; % 22
    1, 1, 6, 0, 2; % 23
    1, 1, 6, 0, 3; % 24
    % C. Impact of the Hessian calculation scheme
    % Recorder - Exact Hessian - ma86
    1, 2, 5, 0, 1; % 25
    1, 2, 5, 0, 2; % 26
    1, 2, 5, 0, 3; % 27
    % Recorder - Exact Hessian - ma97
    1, 2, 6, 0, 1; % 28
    1, 2, 6, 0, 2; % 29
    1, 2, 6, 0, 3; % 30
    % D. Impact of having metabolic energy rate in cost function
    % Recorder - Approximated Hessian - mumps
    1, 1, 1, 1, 2; % 31
    % FD - Approximated Hessian - mumps
    3, 1, 1, 1, 2; % 32
    ];
