function [x, energy_vec] = learn_sparse_signal(G, b, kernel, M, x0, param)
% LEARN_SPARSE_SIGNAL learns the sparse signal x on the nodes of a graph
% G from a diffused observation b, given a diffusion operator kernel, by
% solving a convex optimization problem:
%
% (1)                       argmin_(x) E(x) =
%       argmin_(x)  lambda||x||_1 + (alpha/2)||A*x - b||_2^2 +
%                   (beta/2)||(I - A*A')*x||_2^2
%
%   Usage:
%       [x, energy_vec] = learn_sparse_signal(G, b, kernel, x0, param)
%
%   Input:
%       G       : A Matlab structure containing graph information.
%           G.N : Number of nodes in the graph
%           G.L : The graph Laplacian
%       b       : Vector of size G.N corresponding to the observed
%                 diffused (and possibly noisy) signal on the vertices
%                 of the graph.
%       kernel  : A G.N-by-G.N diffusion matrix, OR a function handle with
%                 the expression for the spectral kernel as a function of
%                 the graph Laplacian eigenvalues (e.g.: @(e) exp(-2.0.*e))
%       M       : A G.N-by-1 vector, representing the observation mask.
%                 (DEFAULT: ones(G.N, 1))
%                 (DEFAULT: speye(G.N))
%       x0      : (Optional) Initialization of the signal to be learned.
%                 (DEFAULT: zeros(G.N, 1))
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
%           param.TOL       : Stop criterium. See FISTA.m
%                             (DEFAULT: 1e-10).
%           param.MAX_ITER  : Stop criterium. See FISTA.m
%                             (Default: 1000).
%           param.constraint: Function handle imposing some constraint on x
%                             (Default: @(x) x).
%
%   Output:
%         x             : The recovered sparse signal on the graph.
%         energy_vec    : A 1-by-(n+1) vector with the energies
%                         E(x(n-1)), where n is the iteration number.
%
%   Example:
%       x = learn_sparse_signal(G, b, x);
%
%   Requires: GSPBox (https://lts2.epfl.ch/gsp/)
%
%   See also: FISTA.m
%
%   References:

% Author: Rodrigo Pena
% Date: 23 Nov 2015
% Testing: demo_sparse_signal_learning.m

%% Parse Input
% G
assert(isfield(G, 'N') && isfield(G, 'L'), ...
    'G doesn''t contain the required fields.')

% b
assert(size(b,1) == 1 || size(b,2) == 1, 'b must be a vector');
assert(length(b) == G.N, 'The length of b must be G.N');

% kernel
if isa(kernel, 'function_handle')
    handle_flag = 1;
elseif isnumeric(kernel)
    assert(sum(size(kernel) ~= size(eye(G.N))) == 0, ...
        'If kernel is numeric, it should be a G.N-by-G.N matrix');
    handle_flag = 0;
else
    error('kernel must be a matrix, OR a function handle');
end

%M
if (nargin < 4) || isempty(M); M = ones(G.N, 1); end
assert(sum(size(M) ~= size(b)) == 0, 'M must be a G.N-by-1 vector');

% x0
if (nargin < 5) || isempty(x0) ; x0 = zeros(G.N, 1); end

% param
if (nargin < 6); param = struct; end
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
if ~isfield(param, 'method') || isempty(param.method);
    param.method = 'smooth-newton'; end
if ~isfield(param, 'constraint') || isempty(param.constraint);
    param.constraint = @(x) sign_constraint(b, x);
end
assert(isa(param.constraint, 'function_handle'), ...
    'The constraint must be a function handle on x');

%% Initialization
lambda = param.lambda;
alpha = param.alpha;
beta = param.beta;
g = [];
f = [];

%% Functions in the minimization problem
g.eval = @(x) lambda .* norm(x, 1);
g.prox = @(x, tau) shrinkage(x, lambda .* tau);

if handle_flag
    A = @(x) gsp_filter(G, kernel, x);
    C = @(x) x - gsp_filter(G, @(e) kernel(e).^2, x);
   
    f.L = estimate_lipschitz_constant(G, kernel, param);
    
else
    A = @(x) kernel * x;
    C = @(x) x - A(A(x));
    
    % FISTA is able to estimate the Lipschitz constant with backtracking)
    % The exact expression (below) takes considerable time to compute when
    % the kernel is a big matrix.
    % f.L = normest( alpha .* kernel^2 + beta .* (speye(G.N) - kernel^2) );
    f.L = [];

end

f.eval =  @(x) (alpha ./ 2) .* norm( (M .* A(x)) - b , 2).^2 + ...
    (beta ./ 2) .* norm(C(x), 2).^2;

f.grad = @(x) alpha .* A( M .* ( (M .* A(x)) - b ) ) + ...
    beta .* C(C(x));

%% Learn sparse signal
[x, energy_vec] = FISTA(g, f, G.N, x0, param);

% TODO: primal-dual doesn't work yet
% f.prox = @(x, tau) prox_f(x, tau, alpha, A, beta, C, b);
% [x, energy_vec] = primal_dual(g, f, speye(G.N), G.N, x0, param);

end

% function y = prox_f(x, tau, alpha, A, beta, C, b)
% if isa(A, 'function_handle') && isa(C, 'function_handle')
%     Acg = @(x) alpha.*A(A(x)) + beta.*C(C(x));
%     bcg = @(x, tau) tau.*alpha.*A(b) + sparse(x);
%     [y, ~, ~, ~, ~] = pcg(@(x) speye(size(A)) + tau.*Acg(x), ...
%         bcg(x, tau), 1e-10, 100, [], [], bcg(x, tau));
% else
%     Acg = @(tau) speye(size(A)) + tau.*(alpha.*A'*A + beta.*C'*C);
%     bcg = @(x, tau) tau.*alpha.*A'*b + sparse(x);
%     L = @(tau) ichol(sparse(Acg(tau)), struct('michol','on'));
%     [y, ~, ~, ~, ~] = pcg(Acg(tau), bcg(x, tau), 1e-10, 100, ...
%         L(tau), L(tau)', bcg(x, tau));
% end
% end