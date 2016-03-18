%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Rodrigo Pena,
% e-mail: rodrigo.pena@epfl.ch
% Date: 26 Nov 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script to find the optimal parameters, alpha and beta, for the
% optimization problem:
%
% argmin_(x, theta) E(x, theta) = ...
%       argmin_(x,theta) ||x||_1 + ...
%           (alpha/2)||A(theta)*x - b||_2^2 + ...
%           (beta/2)||(A(theta)*A(theta)' - I)*x||_2^2,
%
% with respect to the error measure given by sparse_error.m

function [parameters, sol_error, output] = find_optimal_parameters()
%% Setup
% Create parallel workers
if isempty(gcp)
    parpool;
end

%% Initialization
problem = struct('objective', @psobj, ...
    'x0', [4, 1], ...
    'Aineq', [], ...
    'bineq', [], ...
    'Aeq', [], ...
    'beq', [], ...
    'lb', [0, 0], ...
    'ub', [100, 100], ...
    'nonlcon', [], ...
    'options', [], ...
    'solver', 'patternsearch', ...
    'rngstate', []);

%% Run pattern search
tic;
[parameters, sol_error, ~, output] = patternsearch(problem);
finish = toc;

fprintf('Time to find optimal parameters: %1.4f s\n', finish);
fprintf('Optimal parameters:\n%1.4f,\n%1.4f\n', ...
    parameters(1), parameters(2));
fprintf('Solution error (hops) with these parameters:\n%1.4f\n', ...
    sol_error);

end

%% PSOBJ
% Creates a pattern search objective function to feed Matlab's
% patternsearch.m function
function y = psobj( parameters )
% Setup
param = struct( 'lambda', parameters(1), ...
    'alpha', parameters(2), ...
    'beta', 0e0, ...
    'MAX_ITER', 200, ...
    'TOL', 1e-6, ...
    'MAX_ITER_x', 150, ...
    'TOL_x', 1e-4, ...
    'MAX_ITER_t', 100, ...
    'TOL_t', 1e-5, ...
    'constraint_x', @(x) x, ...
    'constraint_t', @(t) t, ...
    'method', 'smooth-newton');

% Generate graph & sparse signal
[G, x_spikes, b] = snow_gis_reduce([], 1, 10);
M = ones(G.N, 1);

G = gsp_create_laplacian(G, 'normalized');
G = gsp_compute_fourier_basis(G);

% Filter assumption:
kernel = choose_kernel('heat');

% Initialization
x0 = zeros(G.N, 1);
theta0 = 7.24;
err_param.TOL = 0.01;

% Solve alternate optimization
x = alt_opt(G, b, kernel, M, theta0, x0, param);

% Compute solution error
y = hop_error(G, x_spikes, x, err_param);

end

