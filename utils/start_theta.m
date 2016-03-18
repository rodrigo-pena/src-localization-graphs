function [theta0, x0] = start_theta(G, b, M, inv_kernel)
% START_THETA gives an automatic initial estimate of the diffusion 
% parameter theta based on the structure of the graph G and on the 
% observations b.
%
%   Usage:
%       theta0 = start_theta(G, b)
%
%   Input:
%       G           : Structure containing graph information (see GSPBox
%                     docs). G.W must be the graph's weighted adjacency
%                     matrix. G.N must be the number of nodes on the graph.
%                     Remark: works better if G has the fields G.U and G.e,
%                     i.e., the eigendecomposition of the graph Laplacian.
%       b           : N-by-1 vector representing the observed signal on the
%                     nodes of the graph G.
%       inv_kernel  : (Optional) A function handle representing the inverse
%                     spectral diffusion kernel with respect to theta, as a
%                     function of the eigenvalues of the graph and of the 
%                     "frequency-domain" impulse response of the diffusion
%                     kernel(i.e., h_hat = b_hat./x_hat).
%                     (DEFAULT: @(e, h) -(1./e).*log(abs(h)) )
%       M           : A G.N-by-1 vector, representing the observation mask.
%                     (DEFAULT: ones(G.N, 1))
%
%   Output:
%       theta0      : An initialization of the exponential diffusion kernel
%                     parameter
%   Example:
%       theta0 = start_theta( G, b );
%
%   See also: alt_opt.m
%
%   Requires:
%       GSPBox (https://lts2.epfl.ch/gsp/)
%
%   References:
%
% Author: Rodrigo Pena
% Date: 18 Nov 2015
% Testing:

%% Parse input
assert(isfield(G, 'N') && isfield(G, 'W'), 'G.N or G.W doesn''t exist');
if isfield(G, 'U') && isfield(G, 'e')
    fourier_flag = 1;
else
    fourier_flag = 0;
end

assert(size(b,1) == 1 || size(b,2) == 1, 'b must be a vector');
assert(length(b) == G.N, 'b must be of length G.N')

if nargin < 3 || isempty(M)
    M = ones(G.N, 1);
end
assert(size(M, 1) == 1 || size(M, 2) == 1, 'M must be a vector');
assert(length(M) == G.N, 'M must be of length G.N');

if nargin < 4 || isempty(inv_kernel)
    inv_kernel = @(e, h) -(1./e).*log(abs(h));
end
assert(isa(inv_kernel, 'function_handle'), ...
    'inv_kernel must be a function handle');

%% Initialization
observed_nodes = find(M ~= 0);
b = gsp_interpolate(G, b(observed_nodes), observed_nodes);

%% Compute theta0
if fourier_flag
    % Idea: try to estimate x0 by the regional maxima of b and compute theta
    % from there. Remark: It seems better to overestimate theta0!
    x0 = zeros(size(b));
    [val, ind] = sort(b, 1, 'descend');
    while val(1) > prctile(b, 90)
        d = bfs(G.W, ind(1));
        x0(ind(1)) = 1;
        % TODO: Check if this works:
        % x0(ind(1)) = sum(b(d(ind) <= 1));
        val = val(d(ind) > 1);
        ind = ind(d(ind) > 1);
    end
    b_hat = G.U'*b;
    x_hat = G.U'*x0;
    thetas = inv_kernel(G.e, b_hat./x_hat);
    theta0 = median(thetas);
else
    % Idea: try to estimate theta0 from the truncated Taylor expansion of
    % -4*t*theta/(theta^2 + t^2)^2, the derivative of 2*theta/(theta^2 + t^2)
    % Node with the maximum of b
    [~, n] = max(b);
    n = n(1);
    
    % Compute weighted distances from n to every other node
    W = G.W;
    W(W ~= 0) = -log(W(W ~= 0)./2);
    d = shortest_paths(W,n);
    
    % Compute hop distance from n to every other node
    hops = bfs(G.W,n);
    
    % Get (an approx. of) the mean first derivative of b around n
    first_deriv = abs(mean(b(n) - b(hops==1) ./ d(hops==1)));
    
    % Compute theta from the truncated Taylor expansion of
    % -4*t*theta/(theta^2 + t^2)^2, the derivative of 2*theta/(theta^2 + t^2)
    t = mean(d(hops==1));
    theta0 = (4*t./first_deriv)^(1./3);
end

end

