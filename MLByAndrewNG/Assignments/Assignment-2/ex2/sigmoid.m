function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

negZ = z .* -1;
exponentVal = e .^ negZ;
divisor = 1 .+ exponentVal;
g = 1 ./ divisor;



% =============================================================

end
