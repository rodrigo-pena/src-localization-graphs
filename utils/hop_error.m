function error = hop_error(G, x_ref, x_test, param)
% HOP_ERROR computes the average hop distance between corresponding 
% spikes of two signals lying on the nodes of a graph G = (V, E, W).
%
%   Usage:
%      error = hop_error(G, x_ref, x_test, param)
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
%
%   Output:
%       error   : Average hop distance between corresponding spikes of 
%                 x_ref and x_test.
%
%   Example:
%       G = gsp_sensor(300); % Requires GSPBox
%       x_ref = zeros(G.N, 1);
%       x_ref(1) = 1;
%       x_test = zeros(G.N, 1);
%       x_test(2) = 1;
%       error = sparse_error(G, x_ref, x_test);
%
%   See also: sparse_error.m
%
%   Requires: MatlabBGL (http://dgleich.github.io/matlab-bgl/)
%
%   References:
%
% Author: Rodrigo Pena
% Date: 18 Feb 2016
% Testing: demo_sparse_error.m

%% Parse input
% Graph
assert(isfield(G, 'N'), 'G has no field N (number of nodes)');
assert(isfield(G, 'W'), 'G has no field W (adjacency matrix)');

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

%% Normalize both signals
if nnz(x_ref) ~= 0
    x_ref = x_ref ./ max( abs(x_ref) );
end

if nnz(x_test) ~= 0
    x_test = x_test ./ max( abs(x_test) );
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
        error = Inf;
        return;
    end
elseif nnz(x_test) == 0
    error = Inf;
    return
end

%% Initialization
active_nodes = find(x_ref); % Nodes with spikes in the reference signal
n_spikes = length(active_nodes);
hop_matrix = Inf(G.N, n_spikes);

%% Compute minimum hop distances from the active nodes to the other nodes
for n = 1:n_spikes
    hop_matrix(:, n) = bfs(G.W, active_nodes(n));
end

% Define the influence zone of each active node i as the set of nodes in G 
% which are closest to i than to any other active node.
hop_matrix(hop_matrix == -1) = Inf;
[hop_vector, ind] = min(hop_matrix, [], 2);

%% Keep only connected nodes
conn_nodes = find(hop_vector ~= Inf);
N = length(conn_nodes);

x_ref = x_ref(conn_nodes);
active_nodes = find(x_ref);
n_spikes = length(active_nodes);

x_test = x_test(conn_nodes);

ind = ind(conn_nodes);
hop_matrix = hop_matrix(conn_nodes, :);

%% Partition x_test values into the influence zones of each active node
x_test_per_zone = zeros(N, n_spikes);

for n = 1:n_spikes
    
    tmp = x_test;
    tmp(ind ~= n) = 0;
    
    if nnz(tmp) == 0; 
        mask = Inf(N, 1);
        mask(active_nodes) = 1;
        dist_active_nodes = hop_matrix(:, n) .* mask;
        [~, closest_active_nodes] = sort(dist_active_nodes);
        closest_active_nodes = closest_active_nodes(1:n_spikes);
        
        % Get the x_test values from the influence zone of the next 
        % closest active node
        count = 1;
        while (nnz(tmp) == 0) && (count <= n_spikes)
            count = count + 1;
            m = find(active_nodes == closest_active_nodes(count));
            if isempty(m) % Current active node is an isolated node
                tmp = zeros(size(x_test));
                break;
            else
                tmp = x_test;
                tmp(ind ~= m) = 0;
            end
        end
    end
    
    x_test_per_zone(:, n) = tmp; 
    
end

%% Compute mean hop error
hop_matrix(hop_matrix == Inf) = 0;
error = sum(hop_matrix .* abs(x_test_per_zone), 1) ./ sum(abs(x_test_per_zone), 1);
error = nanmean(error);

end
