function error = sparse_error(G, x_ref, x_test, param)
% SPARSE_ERROR computes the error between two sparse signals on the nodes
% of a connected graph G = (V, E, W).
%
%   Usage:
%      error = sparse_error(G, x_ref, x_test, param)
%
%   Input:
%       G       : Structure containing graph information (see GSPBox docs).
%                 G.N is the number of nodes,  G.W is the weighted
%                 adjacency matrix of the graph.
%       x_ref   : A G.N-dimensional vector. The reference sparse signal on
%                 the nodes of G.
%       x_test  : A G.N-dimensional vector. A test sparse signal on the
%                 nodes of G.
%       param     : Matlab structure with some additional parameters.
%           param.TOL       : Sparsity tolerance on x_ref and x_test. 
%                             Entries less than TOL (in absolute value) in 
%                             these vectors are considered zero. 
%                             (DEFAULT: 0)
%           param.d_fun     : A function handle specifying a distance 
%                             measure between two nodes as a function of 
%                             the weight of edge that connects them.
%                             (DEFAULT: dij = real(-log(Wij))). 
%
%   Output:
%       error   : Error between x_ref and x_test.
%
%   Example:
%       G = gsp_sensor(300); % Requires GSPBox
%       x_ref = zeros(G.N, 1);
%       x_ref(1) = 1;
%       x_test = zeros(G.N, 1);
%       x_test(2) = 1;
%       error = sparse_error(G, x_ref, x_test);
%
%   Details: 
%       SPARSE_ERROR takes into account both the height difference and the 
%       weighted distance between the spikes in x_ref and x_test.
%       (Better seen in LaTeX)
%       \begin{equation}
%           e(x, y) = \underset{i \in \mathcal{A}}{\sum}\underset{j \in \mathcal{N}_{i}}
%           {\sum}\frac{
%           \left| x(i) - y(j) \right| 1_{\left\{d=0\right\}}(i, j) +
%           \left| y(j) \right| d(i, j)1_{\left\{d\neq0\right\}}(i, j)
%           }
%           {\left|\left|x\right|\right|_{1}},
%       \end{equation}
%       where $d(i, j)$ is the shortest-path distance between nodes $i$ and
%       $j$, and $\mathbb{1}_D(\omega)$ is the indicator function of a set 
%       D for the realization $\omega$. The set $\mathcal{A}$ indicates the
%       active nodes in the reference signal, and the set $\mathcal{N}_{i}$
%       indicates the nodes in the "influence zone" of node $i$ in the test
%       signal.
%
%   See also: hop_error.m
%
%   Requires: MatlabBGL (http://dgleich.github.io/matlab-bgl/)
%
%   References:
%
% Author: Rodrigo Pena
% Date: 16 Nov 2015
% Testing: demo_sparse_error.m

%% Parse input
% Graph
assert(isfield(G, 'N'), 'G has no field N (number of nodes)');
assert(isfield(G, 'W'), 'G has no field W (weighted adjacency matrix)');

% Reference signal
assert(size(x_ref,1) == 1 || size(x_ref,2) == 1, ...
    'x_ref must be a vector');
assert(length(x_ref) == G.N, ...
    'The length of x_ref must be equal to the number of nodes on the graph');

% Test signal
assert(size(x_test,1) == 1 || size(x_test,2) == 1, ...
    'x_test must be a vector');
assert(length(x_test) == G.N, ...
    'The length of x_test must be equal to the number of nodes on the graph');

% param
if (nargin < 4) || isempty(param); param = []; end
if ~isfield(param, 'TOL'); param.TOL = 0; end
if ~isfield(param, 'd_fun'); param.d_fun = @default_d_fun; end

%% Normalize both signals
if nnz(x_ref) ~= 0 && nnz(x_test) ~= 0
    x_test = x_test ./ max( abs(x_test) );
    x_ref = x_ref ./ max( abs(x_ref) );
end

%% Treshold to ensure we have sparse signals
x_ref( abs(x_ref) < param.TOL ) = 0;
x_test( abs(x_test) < param.TOL ) = 0;

%% Check if we still have non-zero elements
if nnz(x_ref) == 0
    if nnz(x_test) == 0
        error = 0;
        return;
    else
        tmp = x_ref;
        x_ref = x_test;
        x_test = tmp;
    end
end

%% Process weight matrix -> distance matrix
Dist = G.W;
Dist(Dist ~= 0) = param.d_fun(Dist(Dist ~= 0));

%% Initialization
active_nodes = find(x_ref);
n_spikes = length(active_nodes);
d_matrix = Inf(G.N, n_spikes);

%% Compute minimum distances from the active nodes to the other nodes
for n = 1:n_spikes
    d_matrix(:,n) = shortest_paths(Dist, active_nodes(n));
end
[d_vector, ind] = min(d_matrix, [], 2);

%% Keep only connected nodes
conn_nodes = find(d_vector ~= Inf);
x_ref = x_ref(conn_nodes);
active_nodes = find(x_ref);
x_test = x_test(conn_nodes);
ind = ind(conn_nodes);
d_vector = d_vector(conn_nodes);

%% Normalize the distances
d_vector = d_vector ./ max(d_vector);

%% Compute error
error = (d_vector == 0)' * abs(x_test - x_ref(active_nodes(ind))) + ...
        (d_vector ~= 0)' * (abs(x_test) .* d_vector);

error = error ./ norm(x_ref, 1);

end

%% Auxiliary functions
% Default distance function between two connected nodes:
function dist = default_d_fun(w)
    dist = real(-log(w));
end