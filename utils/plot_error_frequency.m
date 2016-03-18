function plot_error_frequency( E , nbins)
% PLOT_ERROR_MATRIX creates a 3D plot of the error matrix E.
%
%   Input:
%       E       : N-by-M error matrix.
%       nbins   : Number of bins to use on the histogram. Default:
%                 automatic (see function histogram.m).
%
%   Output:
%       fig_handle   : Plot figure handle
%
%   Example:
%       test_accuracy.m
%       plot_error_matrix(error_matrix)
%
%   See also: test_accuracy.m
%
%   Requires:
%
%   References:
%
% Author: Rodrigo Pena
% Date: 16 Nov 2015
% Testing:

%% Parse input
assert(size(E, 3) == 1, 'E must be a two-dimensional matrix');

% nbins
if (nargin < 2)
    nbins = [];
else
    round(nbins);
    assert(nbins > 0, 'nbins must be an integer > 0');
end

%% Initialization
[~, M] = size(E);

p_cell = cell(M,2);
for j = 1:M
    if ~isempty(nbins)
        [n, edges] = histcounts(E(:,j), nbins, 'Normalization', ...
            'probability', 'BinMethod', 'fd');
    else
        [n, edges] = histcounts(E(:,j), 'Normalization', 'probability', ...
            'BinMethod', 'fd');
    end
    p_cell{j}{1} = edges(1:end-1);
    p_cell{j}{2} = n;
end

%% Generate plot
for j = 1:M
    stem3(j*ones(1, length(p_cell{j}{1})), p_cell{j}{1}, p_cell{j}{2}, ...
        'filled');
    hold on
end
hold off
set(gca, 'XLim', [0 M+1]);
set(gca, 'XTick', 0:1:M+1);
view([115,20])
