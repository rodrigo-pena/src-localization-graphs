function [x, N_SPIKES] = pick_spikes(G, N_SPIKES, param)
% PICK_SPIKES randomly assigns spikes to the nodes of graph
%
%   Usage:
%       [x, N_SPIKES] = pick_spikes( G, N_SPIKES, param )
%
%   Input:
%       G           : A Matlab structure with graph information. It must
%                     have as fields:
%           G.N : Number of nodes in the graph
%           G.W : The graph's weighted adjacency matrix.
%       N_SPIKES    : Number of spikes to pick. Default: 1
%       param       : A Matlab structure with some additional parameters
%           param.h_min     : Minumum height of the spikes.
%                             (DEFAULT: 1)
%           param.h_max     : Maximum height of the spikes.
%                             (DEFAULT: 1)
%           param.d_min     : (> 0) Minimum hop distance between spikes.
%                             (DEFAULT: 1)
%           param.d_max     : (>= d_min) Maximum hop distance between 
%                             successively drawn spikes.
%                             (DEFAULT: Inf)
%           param.max_tries : Maximum number of tries to find a spike
%                             configuration respecting the distance
%                             constraints.
%                             (DEFAULT: 2)
%           param.val_type  : A string in {'integer', 'real'}, indicating
%                             if the spikes have integer or real heights,
%                             respectively.
%                             (DEFAULT: 'integer')
%           param.verbose   : Level of information display.
%               0   : No info
%               1   : Display warning on N_SPIKES
%                             (DEFAULT: 1)
%
%   Output:
%       x   : A N-by-1 sparse vector with N_SPIKES non-zero entries
%
%   Example:
%       G = gsp_sensor(N); % Requires GSPBox
%       x = pick_spikes(G, 10);
%
%   See also:
%
%   Requires: MatlabBGL (http://dgleich.github.io/matlab-bgl/)
%
%   References:
%
% Author: Rodrigo Pena
% Date: 18 Nov 2015
% Testing:

%% Parse input
% G
assert(isfield(G, 'N') && isfield(G, 'W'), ...
    'The graph G doesn''t have the required fields');

% N_SPIKES
if (nargin < 2) || isempty(N_SPIKES); N_SPIKES = 1; end
assert(N_SPIKES <= G.N, 'N_SPIKES must be <= N');

% param
if (nargin < 3); param = struct; end
if ~isfield(param, 'h_min') || isempty(param.h_min); param.h_min = 1; end
if ~isfield(param, 'h_max') || isempty(param.h_max); param.h_max = 1; end
if ~isfield(param, 'd_min') || isempty(param.d_min); param.d_min = 1; end
if ~isfield(param, 'd_max') || isempty(param.d_max); param.d_max = Inf; end
if ~isfield(param, 'n_tries') || isempty(param.n_tries);
    param.n_tries = 0; end
if ~isfield(param, 'max_tries') || isempty(param.max_tries);
    param.max_tries = 2; end
if ~isfield(param, 'val_type') || isempty(param.val_type); 
    param.val_type = 'integer'; end
assert(sum(size(param.h_min) ~= 1) == 0 && ...
    sum(size(param.h_max) ~= 1) == 0 && ...
    sum(size(param.d_min) ~= 1) == 0 && ...
    sum(size(param.d_max) ~= 1) == 0, ...
    'h_min, h_max, d_min, and d_max must be scalars');
assert(isnumeric(param.h_min) && isnumeric(param.h_max) && ...
    isnumeric(param.d_min) && isnumeric(param.d_max), ...
    'h_min, h_max, d_min, and d_max must be numeric');
assert(param.h_min <= param.h_max, ...
    'h_min must be less than or equal to h_max');
assert(param.d_min > 0, ...
    'd_min must be greater than zero');
assert(param.d_min <= param.d_max, ...
    'd_min must be less than or equal to d_max');
assert(strcmp(param.val_type, 'integer') || ...
    strcmp(param.val_type, 'real'), ...
    'val_type admits the values ''integer'' or ''real''.');

%% Initialization
x = zeros(G.N, 1);
nodes_with_spikes = zeros(N_SPIKES, 1);
h_min = param.h_min;
h_max = param.h_max;
d_min = param.d_min;
d_max = param.d_max;
val_type = param.val_type;
fail_flag = 0;

%% Pick spike heights
switch val_type
    case 'integer'
        spikes = randi([h_min, h_max], [N_SPIKES, 1]);
    case 'real'
        spikes = (h_max - h_min) .* rand([N_SPIKES, 1]) + h_min;
end

%% Compute the list of admissible nodes
% TODO: Consider edge weights.

% Get the first node
node_list = randperm(G.N);
current_node = node_list(1);
nodes_with_spikes(1) = current_node;
hops = bfs(G.W, current_node);
min_mask = (hops(node_list) >= d_min);
node_list = node_list(min_mask);

% Get the rest
for i = 2:N_SPIKES
    if ~isempty(node_list)
        max_mask = (hops(node_list) <= d_max);
        current_node = node_list(max_mask);
        if ~isempty(current_node)
            current_node = current_node(1);
            nodes_with_spikes(i) = current_node;
            hops = bfs(G.W, current_node);
            min_mask = (hops(node_list) >= d_min);
            node_list = node_list(min_mask);
        else
            fail_flag = 1;
            param.n_tries = param.n_tries + 1;
            break
        end
    else
        fail_flag = 1;
        param.n_tries = param.n_tries + 1;
        break
    end
end

%% Assign the spikes to the entries of x
if fail_flag
    if param.n_tries < param.max_tries
        x = pick_spikes(G, N_SPIKES, param);
    else
        warning(['pick_spikes.m was only able to pick ', num2str(i-1), ...
            ' spike(s) with the given distance constraint']);
        nodes_with_spikes = nodes_with_spikes(1:i-1);
        x(nodes_with_spikes) = spikes(1:i-1);
    end
else
    x(nodes_with_spikes) = spikes;
end

end

