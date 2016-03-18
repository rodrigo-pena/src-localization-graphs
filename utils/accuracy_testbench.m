function [ errors, dim_strings ] = accuracy_testbench(G, kernel, param)
% ACCURACY_TESTS computes the error of the solution to the alternate 
% optimization for several experiment configurations.
%
% (1)                  argmin_(x, theta) E(x, theta) =
%   argmin_(x, theta) lambda||x||_1 + (alpha/2)||A(theta)*x - b||_2^2 + 
%               (beta/2)||(I - A(theta)'*A(theta))*x||_2^2
%
%   Usage:
%       [ errors ] = accuracy_testbench(G, kernel, param)
%
%   Input:
%       G         : Structure containing graph information. (see GSPBox 
%                   docs)
%       kernel    : Structure with spectral kernel function handles:
%                   g:    Function handle specifying the diffusion
%                         kernel in the Laplacian spectral domain. 
%                         (DEFAULT: g(e,theta) = exp(-theta*e)).                       
%                   gp:   Function handle specifying the derivative of g
%                         with respect to theta. 
%                         (DEFAULT: gp(e,theta) = -e.*exp(-theta*e)).
%                   gpp:  Function handle specifying the second
%                         derivative of g with respect to theta. 
%                         (DEFAULT: gpp(e,theta) = (e.^2).*exp(-theta*e)).
%       param     : Matlab structure specifying the experiment parameters.
%           param.n_test    : Number of realizations for each experiment 
%                             configuration. 
%                             (DEFAULT: 32)
%           param.thetas    : A vector with the kernel diffusion 
%                             parameters to use. 
%                             (DEFAULT: 2)     
%           param.n_spikes  : A vector with the number of spikes to draw
%                             on the nodes of the graph.
%                             (DEFAULT: 2)
%           param.distances : A vector with the hop distances to consider
%                             between spikes.
%                             (DEFAULT: [])
%           param.snrs      : A vector with SNR values (dB) of the 
%                             observations w.r.t. the added WG noise.
%                             (DEFAULT: Inf)
%           param.lambdas   : A vector with the l1-regularization
%                             parameters to consider (see alt_opt.m)
%                             (DEFAULT: 5e1)
%           param.alphas    : A vector with the l1-regularization
%                             parameters to consider (see alt_opt.m)
%                             (DEFAULT: 1e4)
%           param.betas     : A vector with the l1-regularization
%                             parameters to consider (see alt_opt.m)
%                             (DEFAULT: 0)
%           param.n_obs     : A vector with the numbers of observed nodes
%                             to consider
%                             (DEFAULT: G.N)
%           param.TOL       : Stop criterium (see alt_opt.m) 
%                             (Default: 1e-10).
%           param.MAX_ITER  : Stop criterium (see alt_opt.m)  
%                             (Default: 1000).
%
%   Output:
%       errors  : A squeezed version of a n_test-by-length(thetas)-by-
%                 length(n_spikes)-by-length(distances)-by-
%                 length(snrs)-by-length(lambdas)-by-
%                 length(alphas)-by-length(betas) matrix, that is, the 
%                 singleton dimensions are removed. The entries of this
%                 multidimensional array are the solution errors 
%                 corresponding to each experiment configuration.
%       dim_strings : A vector of strings with the names of the parameters 
%                     responsible for each of the dimensions of the errors
%                     multidimensional array, properly ordered.
%        
%   Example:
%
%   See also: alt_opt.m
%
%   Requires: GSPBox (https://lts2.epfl.ch/gsp/)
%             MatlabBGL (http://dgleich.github.io/matlab-bgl/)
%
%   References:
%
% Author: Rodrigo Pena
% Date: 17 Dec 2015
% Testing:

%% Parse input
if (nargin < 2)  || isempty(kernel); kernel = struct; end
if ~isfield(kernel, 'g'); kernel.g = @(e, theta) exp(-theta.*e); end

if isfield(G, 'U') && isfield(G, 'e');
    handle_flag = 0;
    A = @(t) G.U * bsxfun(@times, kernel.g(G.e, t), G.U');
else
    handle_flag = 1;
    A = @(x, t) gsp_filter(G, @(e) kernel.g(G.e, t), x);
end

if (nargin < 3); param = struct; end
if ~isfield(param, 'n_test') || isempty(param.n_test); 
    param.n_test = 32; end
if ~isfield(param, 'thetas') || isempty(param.thetas); 
    param.thetas = 2; end
if ~isfield(param, 'n_spikes') || isempty(param.n_spikes); 
    param.n_spikes = 2; end
if ~isfield(param, 'distances'); 
    param.distances = []; end
if ~isfield(param, 'snrs') || isempty(param.snrs); 
    param.snrs = Inf; end
if ~isfield(param, 'alphas') || isempty(param.alphas); 
    param.alphas = 1e4; end
if ~isfield(param, 'betas') || isempty(param.betas); 
    param.betas = 0; end
if ~isfield(param, 'lambdas') || isempty(param.lambdas); 
    param.lambdas = 2 * param.alpha; end
if ~isfield(param, 'n_obs') || isempty(param.n_obs); 
    param.n_obs = G.N; end
if ~isfield(param, 'TOL') || isempty(param.TOL); 
    param.TOL = 1e-10; end
if ~isfield(param, 'MAX_ITER') || isempty(param.MAX_ITER); 
    param.MAX_ITER = 1000; end
if ~isfield(param, 'TOL_x') || isempty(param.TOL_x);
    param.TOL_x = 1e-6; end
if ~isfield(param, 'MAX_ITER_x') || isempty(param.MAX_ITER_x);
    param.MAX_ITER_x = 250; end
if ~isfield(param, 'TOL_t') || isempty(param.TOL_t);
    param.TOL_t = 1e-3; end
if ~isfield(param, 'MAX_ITER_t') || isempty(param.MAX_ITER_t);
    param.MAX_ITER_t = 20; end
if ~isfield(param, 'method') || isempty(param.method);
    param.method = 'smooth-newton'; end
if ~isfield(param, 'constraint_x'); param.constraint_x = @(x) x; end
if ~isfield(param, 'constraint_t'); param.constraint_t = @(t) t; end

%% Initialization
% Create parallel workers
if isempty(gcp)
    parpool; 
end

% Extract param fields
n_test = param.n_test;
thetas = param.thetas;
n_spikes = param.n_spikes; 
distances = param.distances; 
snrs = param.snrs;
lambdas = param.lambdas; 
alphas = param.alphas; 
betas = param.betas;
n_obs = param.n_obs;

% Fill optparam fields (used in alt_opt.m)
optparam = struct;
optparam.TOL = param.TOL;
optparam.TOL_x = param.TOL_x;
optparam.TOL_t= param.TOL_t;
optparam.MAX_ITER = param.MAX_ITER;
optparam.MAX_ITER_x = param.MAX_ITER_x;
optparam.MAX_ITER_t= param.MAX_ITER_t;
optparam.constraint_x = param.constraint_x;
optparam.constraint_t= param.constraint_t;

% Get lengths of each parameter vector
l = get_lengths(param); 
l(1) = n_test;

% Get names of the non-singleton dimensions
dim_strings = get_non_singleton_labels(param, l);

% Make an index cell for each parameter vector
l = l(2:end-2); % Remove trivial n_test entry
ind_cell = make_ind_cell(l);
curr_idx = ones(length(l),1);

% Initialize errors array
errors = NaN([n_test, l']);

% Create configuration space
configurations = combvec(thetas, n_spikes, distances, snrs, ...
    lambdas, alphas, betas, n_obs);
[~, n_conf] = size(configurations);

x0 = zeros(G.N, 1);

%% Iterate over experiment configurations
for i = 1:n_conf
    theta = configurations(1, i);
    n_s = configurations(2, i);
    sparam.d_min = configurations(3, i);
    sparam.d_max = sparam.d_min;
    snr = configurations(4, i);
    optparam.lambda = configurations(5, i);
    optparam.alpha = configurations(6, i);
    optparam.beta = configurations(7, i);
    M = double(rand(G.N, 1) <= (configurations(8, i) ./ G.N));
    
    % (Parallel) Iterate over the number of realizations
    error_vec = zeros(n_test, 1);
    parfor j = 1:n_test
        % Pick the spikes
        x_spikes = pick_spikes(G, n_s, sparam);

        % Diffuse the signal on the graph
        if handle_flag
            b = A(x_spikes, theta);
        else
            b = A(theta) * x_spikes;
        end
        
        % Add noise
        b = awgn(b, snr, 'measured');
        
        % Apply observation mask
        b = M .* b;
        
        % Alternate optimization:
        x_sol = alt_opt(G, b, [], M, [], x0, optparam);
        
        % Compute error
        error_vec(j) = hop_error(G, x_spikes, x_sol);
    end
    
    % Save solution errors of the current configuration
    errors(:, curr_idx(1), curr_idx(2), curr_idx(3), curr_idx(4), ...
        curr_idx(5), curr_idx(6), curr_idx(7)) = error_vec;
    [curr_idx, ind_cell] = update_indices(ind_cell, l);
end

% Eliminate singleton dimensions
errors = squeeze(errors);

end

%% Auxiliary functions
function l = get_lengths(structure)
%GET_LENGTHS
    fields = fieldnames(structure);
    n_elements = numel(fields);
    l = zeros(n_elements, 1);
    for i = 1:n_elements
        l(i) = length(structure.(fields{i}));
    end
end

function ind_cell = make_ind_cell(l)
%MAKE_IND_CELL
    ind_cell = cell(1);
    for i = 1:length(l)
       ind_cell{1,i} = (1:l(i))'; 
    end
end

function [curr_idx, ind_cell] = update_indices(ind_cell, l)
%UPDATE_INDICES
    valid_indices = find(l > 1);
    
    curr_idx = zeros(length(l), 1);
    for i = 1:length(l)
        curr_idx(i) = ind_cell{i}(1);
    end
    
    ind_cell{valid_indices(1)} = circshift(ind_cell{valid_indices(1)}, -1);
    curr_idx(valid_indices(1)) = ind_cell{valid_indices(1)}(1);
    
    for i = 2:length(valid_indices)
        if ( sum(curr_idx(valid_indices(1:i - 1)) ~= 1) == 0 )
            ind_cell{valid_indices(i)} = ...
                circshift(ind_cell{valid_indices(i)}, -1);
        end
        curr_idx(valid_indices(i)) = ind_cell{valid_indices(i)}(1);
    end
end

function dim_strings = get_non_singleton_labels(param, l)
%GET_NON_SINGLETON_LABELS
    fields = fieldnames(param);
    valid_indices = find(l > 1);
    dim_strings = cell(1);
    for i = 1:length(valid_indices)
        dim_strings{i,1} = fields{valid_indices(i)};
    end
end