function [theta, g_est, energy_vec] = learn_param_kernel(G, b, x, kernel, ...
    M, theta0, param)
% LEARN_PARAM_KERNEL learns the parameter theta of a diffusion kernel given
% a graph G, the diffused observations b on the nodes, and the original
% signal x, by solving an optimization problem.
%
% (1)                   argmin_(theta) E(theta) =
%       argmin_(theta) lamdba||x||_1 + (alpha/2)||A(theta)*x - b||_2^2 +
%                      (beta/2)||(I - A(theta)*A(theta)')*x||_2^2
%
%   Usage:
%       [theta, g_est, energy_vec] = learn_param_kernel(G, b, x, ...
%           kernel, theta0, param);
%
%   Input:
%       G       : A Matlab structure containing graph information.
%           G.N : Number of nodes in the graph
%           G.L : The graph Laplacian
%       b       : Vector of size G.N corresponding to the observed
%                 diffused (and possibly noisy) signal on the vertices
%                 of the graph.
%       x       : Vector of size G.N corresponding to the recovered sparse
%                 signal on the vertices of the graph.
%       kernel  : A Matlab structure containing information about the
%                     spectral diffusion kernel. It must have as fields:
%           kernel.g        : A function handle with the expression for the
%                             spectral kernel g as a function of the graph
%                             Laplacian eigenvalues.
%                             (DEFAULT: @(e, theta) exp(-theta.*e))
%           kernel.gp     : A function handle with the expression for the
%                             derivative of g w.r.t. theta.
%                             (DEFAULT: @(e, theta) - e.*exp(-theta.*e))
%           kernel.gpp    : A function handle with the expression for the
%                             second derivative of g w.r.t. theta.
%                             (DEFAULT: @(e, theta) (e.^2).*exp(-theta.*e))
%       M       : A G.N-by-1 vector, representing the observation mask.
%                 (DEFAULT: ones(G.N, 1))
%       theta0  : Initialization of the parameter to be learned.
%                 (DEFAULT: theta0 = start_theta(G, b))
%       param     : Matlab structure with some additional parameters.
%           param.lambda    : Regularization parameter. See optimization
%                             problem (1).
%                             (DEFAULT: get_lambda(b)).
%           param.alpha     : Regularization parameter. See optimization
%                             problem (1).
%                             (DEFAULT: 1).
%           param.beta      : Regularization parameter. See optimization
%                             problem (1).
%                             (DEFAULT: 1).
%           param.TOL       : Stop criterium. See newton.m
%                             (DEFAULT: 1e-10).
%           param.MAX_ITER  : Stop criterium. See newton.m
%                             (DEFAULT: 1000).
%           param.method    : Solver method. See newton.m
%                             (DEFAULT: 'smooth-newton')
%
%   Output:
%         theta         : The argument that minimizes E(theta).
%         g_est         : Function handle of the estimated spectral
%                         diffusion kernel.
%         energy_vec    : A 1-by-(n+1) vector with the energies
%                         E(theta(n-1)), where n is the iteration number.
%
%   Example:
%       theta = learn_param_kernel(G, b, x);
%
%   Requires: GSPBox (https://lts2.epfl.ch/gsp/)
%
%   See also: learn_sparse_signal.m, newton.m
%
%   References:

% Author: Rodrigo Pena
% Date: 26 Oct 2015
% Testing: test_param_kernel_learning.m

%% Parse Input
% G
assert(isfield(G, 'N') && isfield(G, 'L'), ...
    'G doesn''t contain the required fields.')
if isfield(G, 'U') && isfield(G, 'e');
    handle_flag = 0;
else
    handle_flag = 1;
end

% b
assert(size(b,1) == 1 || size(b,2) == 1, 'b must be a vector');
assert(length(b) == G.N, 'The length of b must be G.N');

% x
assert(sum(size(x) ~= size(b)) == 0, 'x must be a G.N-by-1 vector');

% kernel
if (nargin < 4) || ~isfield(kernel, 'g') || ~isfield(kernel, 'gp') || ...
        ~isfield(kernel, 'gpp')
    kernel = choose_kernel('heat');
end

% M
if (nargin < 5) || isempty(M); M = ones(G.N, 1); end
assert(sum(size(M) ~= size(b)) == 0, 'M must be a G.N-by-1 vector');
 
% theta0
if (nargin < 6) || isempty(theta0); theta0 = start_theta(G, b); end

% param
if (nargin < 7); param = struct; end
if ~isfield(param, 'alpha') || isempty(param.alpha);
    param.alpha = 1e4; end
if ~isfield(param, 'beta') || isempty(param.beta);
    param.beta = 0; end
if ~isfield(param, 'lambda') || isempty(param.lambda);
    param.lambda = set_lambda(b, param.alpha); end
if ~isfield(param, 'TOL') || isempty(param.TOL);
    param.TOL = 1e-4; end
if ~isfield(param, 'MAX_ITER') || isempty(param.MAX_ITER);
    param.MAX_ITER = 500; end
if ~isfield(param, 'method') || isempty(param.method);
    param.method = 'smooth-newton'; end

%% Initialization
lambda = param.lambda;
alpha = param.alpha;
beta = param.beta;
energy_vec = [];
f = [];

%% Functions in the optimization problem
h = @(e, t) kernel.g(e,t).^2;
hp = @(e, t) 2 .* ( kernel.g(e,t) .* kernel.gp(e,t) );
hpp = @(e, t) 2 .* ( ( kernel.g(e,t) .* kernel.gpp(e,t) ) + kernel.gp(e,t).^2 );

if handle_flag
    Ax = @(t) gsp_filter(G, @(e) kernel.g(e, t), x);
    Apx = @(t) gsp_filter(G, @(e) kernel.gp(e, t), x);
    Appx = @(t) gsp_filter(G, @(e) kernel.gpp(e, t), x);
    
    Cx = @(t) x - gsp_filter(G, @(e) h(e, t), x);
    Cpx = @(t) gsp_filter(G, @(e) -hp(e, t), x);
    Cppx = @(t) gsp_filter(G, @(e) -hpp(e, t), x);
    
else
    Ax = @(t) G.U * (kernel.g(G.e, t) .* (G.U' * x));
    Apx = @(t) G.U * (kernel.gp(G.e, t) .* (G.U' * x));
    Appx = @(t) G.U * (kernel.gpp(G.e, t) .* (G.U' * x));
    
    Cx = @(t) x - G.U * (h(G.e, t) .* (G.U' * x));
    Cpx = @(t) - G.U * (hp(G.e, t) .* (G.U' * x));
    Cppx = @(t) - G.U * (hpp(G.e, t) .* (G.U' * x));
    
end

f.eval = @(t) lambda .* norm(x, 1) + ...
    (alpha ./ 2) .* norm(M .* Ax(t) - b,2).^2 + ...
    (beta ./ 2) .* norm(Cx(t), 2).^2;

f.grad = @(t) alpha .* (M .* Ax(t) - b)' * (M .* Apx(t)) + ...
    beta .* (Cx(t)' * Cpx(t));

f.hess = @(t) alpha .* (norm(M .* Apx(t), 2).^2 + ...
    (M .* Ax(t) - b)' * (M .* Appx(t))) + ...
    beta .* (norm(Cpx(t), 2).^2 + (Cx(t)' * Cppx(t)));

%% Learn kernel parameter
[theta, energy_vec] = newton(f, [], theta0, param);

% if param.MAX_ITER == 0
%     theta = theta0;
% else
%     theta = fminbnd(@f.eval, 0, 1000);
% end

g_est = @(e) kernel.g(e, theta);

end