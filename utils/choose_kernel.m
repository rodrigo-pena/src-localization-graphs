function kernel = choose_kernel(type_str)
%CHOOSE_KERNEL outputs a Matlab structure with kernel function handles as
%fields.
%
%   Usage:
%       kernel = choose_kernel(type_str)
%
%   Input:
%       type_str    : A string specifying the type of kernel to choose.
%           'heat'      : Heat diffusion kernel,
%                         g(e,t) = exp(-t.*e).
%           'voltage'   : Voltage propagation kernel, 
%                         g(e, t) = cos(t.*acos(1 - e)).
%           (DEFAULT: 'heat').
%
%   Output:
%       kernel  : A Matlab structure.
%           kernel.g    : A function handle on (e, t) specifying a spectral
%                         kernel with parameter t acting on e.
%           kernel.gp   : A function handle on (e, t) specifying the 
%                         derivative of kernel.g with respect to t.
%                         kernel with parameter t acting on e.
%           kernel.gpp  : A function handle on (e, t) specifying the 
%                         second derivative of kernel.g with respect to t.
%
%   Example:
%       kernel = choose_kernel(type_str)
%
%   Requires: 
%
% Author: Rodrigo Pena (rodrigo.pena@epfl.ch)
% Date: 15 Jan 2015

%% Parse input
if nargin < 1; type_str = []; end
if isempty(type_str); type_str = 'heat'; end
assert(isa(type_str, 'char'), 'type_str must be a string.');
assert(strcmp(type_str, 'heat') || strcmp(type_str, 'voltage'), ...
    'type_str only admits the following values: ''heat'', ''voltage''.');

%% Initialization
kernel = struct('g', [], 'gp', [], 'gpp', []);

%% Choose kernel functions
switch type_str
    case 'heat'
        kernel.g = @(e, t) exp(-t .* e);
        kernel.gp = @(e, t) - e .* exp(-t .* e);
        kernel.gpp = @(e, t) (e.^2) .* exp(-t .* e);    
        
    case 'voltage'
        kernel.g = @(e, t) cos(t .* acos(1 - e)); 
        kernel.gp = @(e, t) - acos(1 - e) .* sin(t .* acos(1 - e)); 
        kernel.gpp = @(e, t) - (acos(1 - e).^2) .* cos(t .* acos(1 - e));
end

end