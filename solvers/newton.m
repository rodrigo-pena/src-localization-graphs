function [t, energy] = newton(f, dims, t0, param)
% NEWTON solves argmin_t f(t) with Newton's method, where f: R^n -> R is 
% an at least twice differentiable function
%
%   Usage:
%       [t, energy] = newton(f, t0, param)
%
%   Input:
%       f       : A Matlab structure containing information about the
%                 differentiable function f. It must have as fields:
%           f.eval  : A function handle with the expression for f(t).
%           f.grad  : A function handle with the expression for the
%                     gradient of f w.r.t. t.
%           f.hess  : A function handle with the expression for the
%                     hessian of f w.r.t. t.
%       dims    : Vector with the dimensions of the argument to be learned
%                 (DEFAULT: size(t0))
%       t0      : Initialization of the argument to be learned.
%                 (DEFAULT: zeros(dims))
%       param     : Matlab structure with some additional parameters.
%           param.TOL       : Stop criterium. When norm(t(n) - t(n-1), 2), 
%                             where n is the iteration number, is less than
%                             TOL, we quit the iterative process. 
%                             (DEFAULT: 1e-10).
%           param.MAX_ITER  : Stop criterium. When the number of iterations
%                             becomes greater than MAX_ITER, we quit the 
%                             iterative process. 
%                             (DEFAULT: 1000).
%           param.method    : String specifying the method to use
%               'newton': Standard grandient descent with the Newton method
%               'smooth-newton': Slower, but adds a regularization term on 
%                                t that makes the solution more robust to 
%                                initialization changes.
%                             (DEFAULT: 'newton')
%           param.constraint: A function handle imposing some constraint
%                             on t.
%                             (Default: @(t) t).
%         
%   Output:
%         t         : Argument that minimizes f
%         energy    : A 1-by-(n+1) vector with the energies f(t(n-1)),
%                     where n is the iteration number
%
%   Example:
%       f.eval = @(t) norm(t,2).^2;
%       f.grad = @(t) 2.*t;
%       f.hess = @(t) 2;
%       t = newton(f, [], 100*randn(100,1));
%
%   See also: learn_param_kernel.m
%
%   References:

% Author: Rodrigo Pena
% Date: 15 Dec 2015
% Testing: demo_param_kernel_learning.m

%% Parameters
% f
assert(isfield(f, 'eval') && isfield(f, 'grad') && isfield(f, 'hess'), ...
    'f doesn''t have the correct fields.');
assert(isa(f.eval, 'function_handle'), ...
    'f.eval must be a function handle');
assert(isa(f.grad, 'function_handle'), ...
    'f.grad must be a function handle');
assert(isa(f.hess, 'function_handle'), ...
    'f.hess must be a function handle');

% N
if isempty(dims)
    assert(nargin > 2, 't0 must be provided if dims is not provided.');
    assert(~isempty(t0), 't0 must be provided if dims is not provided.');
    dims = size(t0);
else
    assert(isa(dims, 'numeric'), 'N must be numeric');
    dims = round(dims);
end

% t0
if (nargin < 3) || isempty(t0); t0 = zeros(dims); end
assert(sum(dims ~= size(t0)) == 0, 't0 must have size dims');

% param
if (nargin < 4); param = []; end
if ~isfield(param, 'TOL'); param.TOL = 1e-10; end
if ~isfield(param, 'MAX_ITER'); param.MAX_ITER = 1000; end
if ~isfield(param, 'method'); param.method = 'newton'; end
if ~isfield(param, 'constraint'); param.constraint = @(t) t; end
assert(isnumeric(param.TOL) && sum(size(param.TOL)~=1) == 0, ...
    'param.TOL must be a number.');
assert(isnumeric(param.MAX_ITER) && sum(size(param.MAX_ITER)~=1) == 0, ...
    'param.MAX_ITER must be a number.');
param.MAX_ITER = abs(round(param.MAX_ITER)); 
assert(strcmp(param.method, 'newton') || ...
    strcmp(param.method, 'smooth-newton'), ...
    'param.method only accepts ''newton'' or ''smooth-newton''.');
assert(isa(param.constraint, 'function_handle'), ...
    'param.constraint must be a function handle.');

%% Initialization
t = t0; 
n = 0; % Number of iterations
switch param.method
    case 'newton'
        epsilon = 0;
    case 'smooth-newton'
        epsilon = 0.5;
end

difference = Inf;
energy = zeros(1, param.MAX_ITER);
energy(1) = f.eval(t);

%% Iterative steps of Newton's Method
while (difference > param.TOL) && (n < param.MAX_ITER)
    
    epsilon = epsilon .* 2^(-n);
    n = n + 1;
    
    % Gradient
    Grad = f.grad(t) + 2 .* epsilon * t;
    
    % Hessian
    Hess = f.hess(t) + 2 .* epsilon;
    
    % Update t
    t_old = t;
    t = t_old - Hess\Grad;
    
    % Apply constraint
    t = param.constraint(t);
    
    % Compute energy
    energy(n + 1) = f.eval(t);
    
    % Update difference
    difference = norm(t - t_old, 2);
end

% Trim down the energy vector.
energy = energy(1:n+1);

end