function [x, theta, g_est, energy_vec] = alt_opt(G, b, kernel, M, ...
    theta0, x0, param)
% ALT_OPT solves the alternate optimization for the source
% localization problem on graphs.
%
% (1)                  argmin_(x, theta) E(x, theta) =
%   argmin_(x, theta) lambda||x||_1 + (alpha/2)||A(theta)*x - b||_2^2 +
%               (beta/2)||(I - A(theta)'*A(theta))*x||_2^2
%
% It alternates between learning the sparse signal x, and the diffusion
% kernel parameter theta.
%
%   Usage:
%       [x, theta, g_est, energy_vec] = alt_opt(G, b, kernel, ...
%           theta0, x0, param);
%
%   Input:
%       G         : Structure containing graph information.
%           G.N : Number of nodes in the graph
%           G.L : The graph Laplacian
%       b         : Vector of length G.N corresponding to the observed
%                   diffused (and possibly noisy) signal on the vertices
%                   of the graph.
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
%                   (See also choose_kernel.m)
%       M       : A G.N-by-1 vector, representing the observation mask.
%                 (DEFAULT: ones(G.N, 1))
%       theta0    : Initialization of the kernel parameter to be learned.
%                   (DEFAULT: theta0 = start_theta(G, b))
%       x0        : Initialization of the sparse signal to be learned.
%                   (DEFAULT: x0 = gsp_filter(G, @(e) 1./kernel.g(e, theta0), b))
%       param     : Matlab structure with some additional parameters.
%           param.lambda    : Regularization parameter. See optimization
%                             problem (1).
%                             (DEFAULT: get_lambda(b)).
%           param.alpha     : Regularization parameter. See optimization
%                             problem (1).
%                             (DEFAULT: 1e4).
%           param.beta      : Regularization parameter. See optimization
%                             problem (1).
%                             (DEFAULT: 0).
%           param.TOL       : Stop criterium. When E(x(n), theta(n)) -
%                             E(x(n-1), theta(n-1)), where n is the
%                             iteration number, is less than TOL, we quit
%                             the iterative process.
%                             (DEFAULT: 1e-10).
%           param.MAX_ITER  : Stop criterium. When the number of iterations
%                             becomes greater than MAX_ITER, we quit the
%                             iterative process.
%                             (DEFAULT: 1000).
%           param.TOL_x     : Stop criterium for the optimization on x
%                             (see learn_sparse_signal_learning.m)
%                             (DEFAULT: 1e-10).
%           param.MAX_ITER_x: Stop criterium for the optimization on x
%                             (see learn_sparse_signal_learning.m)
%                             (DEFAULT: 1000).
%           param.TOL_t     : Stop criterium for the optimization on theta
%                             (see learn_param_kernel.m)
%                             (DEFAULT: 1e-10).
%           param.MAX_ITER_t: Stop criterium for the optimization
%                             on theta (see learn_param_kernel.m)
%                             (DEFAULT: 1000).
%           param.method    : Kernel learning solver method. See
%                             newton.m
%                             (DEFAULT: 'smooth-newton')
%           param.constraint_x : A function handle imposing some constraint
%                                on x (see FISTA.m)
%                                (Default: @(x) x).
%           param.constraint_t : A function handle imposing some constraint
%                                on theta (see newton.m)
%                                (Default: @(t) t).
%
%   Output:
%       x         : Vector of length G.N approximating the original sparse
%                   signal on the vertices of the graph.
%       theta     : Learnt kernel parameter
%       g_est     : Function handle of the estimated spectral kernel
%                   (i.e., g_est = @(e) kernel.g(e, theta))
%       energyVec : A 1-by-(n+1) vector with the energies
%                   E(x(n-1), theta(n-1)), where n is the iteration number.
%
%   Example:
%       [x, theta, n, g_est] = alt_opt(G, b, kernel);
%
%   Requires: GSPBox (https://lts2.epfl.ch/gsp/)
%
%   See also: learn_sparse_signal.m, learn_param_kernel.m
%
%   References:
%
% Author: Rodrigo Pena
% Date: 6 Nov 2015
% Testing: demo_alternate_optimization.m

%% Parse input
% G
assert(isfield(G, 'N') && isfield(G, 'L'), ...
    'G doesn''t contain the required fields.')
if isfield(G, 'U') && isfield(G, 'e');
    handle_flag = 0;
else
    handle_flag = 1;
end

% kernel
if (nargin < 3) || ~isfield(kernel, 'g') || ~isfield(kernel, 'gp') || ...
        ~isfield(kernel, 'gpp')
    kernel = choose_kernel('heat');
end

% M
if (nargin < 4) || isempty(M); M = ones(G.N, 1); end
assert(sum(size(M) ~= size(b)) == 0, 'M must be a G.N-by-1 vector');

% x0
if (nargin < 6); x0 = []; end
if ~isempty(x0); assert(length(x0) == length(b)); end

% theta0
if (nargin < 5) || isempty(theta0)
    [theta0, ~] = start_theta(G, b);
end

% param
if (nargin < 7); param = struct; end
if ~isfield(param, 'alpha') || isempty(param.alpha);
    param.alpha = 1e4; end
if ~isfield(param, 'beta') || isempty(param.beta);
    param.beta = 0; end
if ~isfield(param, 'lambda') || isempty(param.lambda);
    param.lambda = set_lambda(b, param.alpha); end
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

%% Functions in the minimization problem
if handle_flag
    A = @(x, t) gsp_filter(G, @(e) kernel.g(e, t), x);
    C = @(x, t) x - gsp_filter(G, @(e) kernel.g(e, t).^2, x);
else
    A = @(x, t) G.U * ( kernel.g(G.e, t) .* (G.U' * x) );
    C = @(x, t) x - G.U * ( (kernel.g(G.e, t).^2) .* (G.U' * x) );
end

g = @(x) (param.lambda) .* norm(x, 1);
f = @(x, t) (param.alpha ./ 2) .* norm( M .* A(x, t) - b, 2).^2 + ...
    (param.beta ./ 2) .* norm(C(x, t), 2).^2;

%% Initialization
n = 0; % Iteration number
theta = theta0;
if isempty(x0)
    x = gsp_filter(G, @(e) 1./kernel.g(e, theta), b);
else
    x = x0;
end

prev_energy = g(x) + f(x, theta);
energy_diff = Inf;
energy_vec = zeros(1, param.MAX_ITER);
energy_vec(1) = prev_energy;

xparam = struct('lambda', param.lambda, 'alpha', param.alpha, ...
    'beta', param.beta, 'MAX_ITER', param.MAX_ITER_x, ...
    'TOL', param.TOL_x, 'constraint', param.constraint_x);

thetaparam = struct('lambda', param.lambda, 'alpha', param.alpha, ...
    'beta', param.beta, 'MAX_ITER', param.MAX_ITER_t, ...
    'TOL', param.TOL_t, 'method', param.method, ...
    'constraint', param.constraint_t);

%% Iterative steps:
while (energy_diff > param.TOL) && (n < param.MAX_ITER)
    
    n = n + 1;
    
    % Solve for x:
    if handle_flag
        x = learn_sparse_signal(G, b, @(e) kernel.g(e, theta), M, x, xparam);
    else
        A_mat = G.U * bsxfun(@times, kernel.g(G.e, theta), G.U');
        x = learn_sparse_signal(G, b, A_mat, M, x, xparam);
    end

    % Solve for theta:
    theta = learn_param_kernel(G, b, x, kernel, M, theta, thetaparam);
      
    % Compute energy difference:
    curr_energy = g(x) + f(x, theta);
    energy_diff = abs(prev_energy - curr_energy);
    energy_vec(n + 1) = curr_energy;
    prev_energy = curr_energy;
end

%% Trim energy vector
energy_vec = energy_vec(1:(n + 1));

%% Estimated low-pass filter:
g_est = @(e) kernel.g(e, theta);

end

