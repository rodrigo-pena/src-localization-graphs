function L = estimate_lipschitz_constant( G, g, param )
% ESTIMATE_LIPSCHITZ_CONSTANT estimates the Lipschitz constant of f(x) = 
% (alpha/2)||A*x - b||_2^2 + (beta/2)||(I - A'*A)*x||_2^2, where x is a
% signal on the nodes of graph G, A = G.U * diag(g(G.e)) * G.U' is a
% diffusion matrix, and alpha and beta are fixed parameters

%% Parse input
assert(isfield(G, 'L') && isfield(G, 'N'), ...
    'G does not have the correct fields')
assert(isa(g, 'function_handle'), ...
    'g must be a function handle on the eigenvalues of the G.L');
assert(isfield(param, 'alpha') && isfield(param, 'beta'), ...
    'param.alpha and param.beta must be accessible');

%% Initialization
alpha = param.alpha;
beta = param.beta;

%% Get graph Laplacian eigenvalues
if isfield(G, 'e')
    e = G.e;
else
    if ~isfield(G, 'lmax')
        G = gsp_estimate_lmax(G);
    end
    e = linspace(0, G.lmax, G.N);
end

%% Estimate L = ||alpha*A^2 + beta*(I - A^2)^2||_2

% lambda: eigenvalues of A^2 = G.U * diag(g(G.e).^2) * G.U':
lambdas = sort(g(e).^2);
lambda_min = lambdas(1);
lambda_max = lambdas(end);

% phi: eigenvalues of alpha*A^2 + beta*(I - A^2)^2:
phi = @(l)  alpha .* l + beta .* (l.^2 - 2.*l + 1);

% Lipschitz constant L:
L = phi(lambda_min) .* (phi(lambda_min) > phi(lambda_max)) + ...
    phi(lambda_max) .* (phi(lambda_min) <= phi(lambda_max));

end

