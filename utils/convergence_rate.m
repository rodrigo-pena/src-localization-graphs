function [ avg_energy, energy_cell, iter_vec] = convergence_rate(G, ...
    kernel, theta, N_SPIKES, N_TEST,  param)
% CONVERGENCE_RATE tests the convergence rate of the alternate optimization
% for the source localization problem on graphs. 
%
% It runs alt_opt.m Ntest times, picking randomly N_spikes sources on the 
% nodes of the graph G, and returns a matrix with the convergence errors 
% for each of the tests.
%
%
%   Input:
%         G         : Structure containing graph information (see GSPBox 
%                     docs)
%         kernel    : (Optional) Structure with spectral kernel function 
%                     handles:
%                     g:    Function handle specifying the diffusion 
%                           kernel in the Laplacian spectral domain. If 
%                           empty, g is assumed to be 
%                           g(e,theta) = exp(-theta * e).
%                     gp:   Function handle specifying the derivative of g 
%                           with respect to theta. If empty, gp is assumed
%                           to be (d/dtheta)exp(-theta * e).
%                     gpp:  Function handle specifying the second 
%                           derivative of g with respect to theta. If 
%                           empty, gpp is assumed to be 
%                           (d^2/dtheta^2)exp(-theta * e).
%         theta     : (Optional) Kernel diffusion parameter. Default: 2.        
%         N_SPIKES  : (Optional) Number of spikes to set as sources
%                     of diffusion on the graph. Default: 10
%         N_TEST    : (Optional) Number of realizations for each number of
%                     spike sources. Default: 100.
%       param     : Matlab structure with some additional parameters.
%           param.TOL       : Stop criterium, see alt_opt.m 
%                             (Default: 0).
%           param.MAX_ITER  : Stop criterium, see alt_opt.m  
%                             (Default: 1000).
%           param.verbose   : Level of verbose. (> 0) : Plot the result.
%                             (Default: 0).
%
%   Output:
%         avg_energy    : Average energy curve
%         energy_cell   : Cell with the energy curves for each of the
%                         N_TEST tests.
%         iter_vec      : Vector with the number of iterations until
%                         convergence for each of the N_TEST tests
%         
%   Example:
%          G = gsp_sensor(300);
%          G = gsp_compute_fourier_basis(G);
%          avg_energy = convergence_rate(G, [], [], [], [], [], [], 1);
%
%   See also: alt_opt.m
%
%   Requires: GSPBox (https://lts2.epfl.ch/gsp/)
%
%   References:
%
% Author: Rodrigo Pena
% Date: 11 Nov 2015
% Testing:

%% Parse input

% Graph
if ~isfield(G, 'U') || ~isfield(G, 'e')
    warning('Pre-compute the eigendecomposition of the graph Laplacian for speed.');
    G = gsp_compute_fourier_basis(G);
end

% Diffusion kernel
if (nargin < 2) || ~isfield(kernel, 'g') || ~isfield(kernel, 'gp') || ...
        ~isfield(kernel, 'gpp')
    kernel.g = @(e, theta) exp(-theta.*e);
    kernel.gp = @(e, theta) - e .* exp(-theta.*e);
    kernel.gpp = @(e, theta) (e.^2) .* exp(-theta.*e);
end

% Kernel parameter
if (nargin < 3) || isempty(theta)
    theta = 2;
end

% Max number of spikes
if (nargin < 4) || isempty(N_SPIKES)
    N_SPIKES = 10;
end

% Number of realizations
if (nargin < 5) || isempty(N_TEST)
    N_TEST = 100;
end

% param
if (nargin < 7) || isempty(param); param = []; end
if ~isfield(param, 'TOL'); param.TOL = 0; end
if ~isfield(param, 'MAX_ITER'); param.MAX_ITER = 1000; end
if ~isfield(param, 'verbose'); param.verbose = 0; end
verbose = param.verbose;
param.verbose = 0;

%% Initialization
avg_energy = zeros(1, MAX_ITER + 1);
energy_cell = cell(N_TEST, 1);
iter_vec = zeros(N_TEST,1);
A = G.U * diag(kernel.g(G.e, theta)) * G.U'; % Diffusion kernel
sparam.h_min = 1;
sparam.h_max = 1;
sparam.val_type = 'integer';

%% (Parallel) Iterate over the number of realizations
parfor j = 1:N_TEST
    
    % Pick randomly the spikes
    x_spikes = pick_spikes( G, N_SPIKES, sparam );
        
    % Diffuse the signal on the graph
    b = A * x_spikes;
    
    % Alternate optimization:
    [~, ~, n, ~, energy] = alt_opt(G, b, kernel, [], [], [], param);

    % Compute energy difference
    iter_vec(j) = n;
    difference = abs(energy - energy(end));
    energy_cell{j} = difference;
    
    % Add to compute the average:
    difference = padarray(difference, [0, MAX_ITER - n], 0, 'post');
    avg_energy = avg_energy + difference;
    
end

avg_energy = avg_energy./N_TEST;

%% Display results
if (nargout == 0) || (verbose > 0)
    for j = 1:N_TEST
        loglog(0:iter_vec(j), energy_cell{j});
        hold on
    end
    plt = loglog(0:MAX_ITER, avg_energy, '--r', 'LineWidth', 2);
    legend(plt, 'Average curve');
    hold off
    xlabel('Iteration number')
    ylabel('Energy E(x_n, \theta_n) - E(x_\infty, \theta_\infty)')
    t = 'G.N=';
    t = strcat(t, '', num2str(G.N),',');
    t = strcat(t, ' N_{SPIKES}= ', num2str(N_SPIKES),',');
    t = strcat(t, ' N_{TEST}= ', num2str(N_TEST));
    title(t);
end

end

