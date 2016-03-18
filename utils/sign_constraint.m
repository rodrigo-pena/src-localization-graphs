function x = sign_constraint( b, x )
%SIGN_CONSTRAINT constraints the entries of x to have the same sign as the
% entries of b.
%
% For example, if the entries of b are all positive, the negative entries
% of x are set to zero.
%
%   Usage :
%       x = sign_constraint( b, x );
%
%   Input :
%       b   : An array.
%       x0  : An array with the same size as b. 
%
%   Output :
%       x   : Sign-constrained x.
%
%   Example:
%       
%       x = sign_constraint( b, x );
%          
%   See also:
%
%   References:
%
% Author: Rodrigo Pena
% Date: 27 Nov 2015
% Testing:

%% Parse input
assert(sum(size(b) ~= size(x)) == 0, ...
    'The arrays b and x must be of the same size.');

%% Sign constraint
x(b > 0) = x(b > 0) .* (sign(x(b > 0)) == 1);
x(b == 0) = x(b == 0) .* (sign(x(b == 0)) == 0);
x(b < 0) = x(b < 0) .* (sign(x(b < 0)) == -1);

end

