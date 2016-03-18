function lambda = set_lambda(b, alpha)
% GET_LAMBDA automatically estimates a good parameter lambda for the 
% optimization problem argmin_x lambda||x||_1 + (alpha/2)||Ax - b||_2^2 + 
% (beta/2)||(I - A'A)x||_2^2 

% right_val = abs(max(b(b~=0)));
% left_val = abs(min(b(b~=0)));
% lambda = 0.1 .* ( ...
%     right_val .* (right_val <= left_val) + ...
%     left_val .* ( right_val > left_val) ...
% );

% lambda = alpha .* sqrt((16 .* var(b) .* log(length(b))) / length(b));

lambda = alpha .* prctile(b, 75);

end