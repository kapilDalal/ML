function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = X * theta;
sig = sigmoid(z);
first = (-1 .* y) .* (log(sig));
second = (1 .- y).*(log(1 .- sig));
summationL = (1/m)*sum(first .- second);
thetaPenalised = theta(2:length(theta),1);
summationR = (lambda/(2*m))*sumsq(thetaPenalised);
totalSum = summationL + summationR;
J = totalSum;

g = sum((sig - y).* X) ./ m;
thetaPenalised = thetaPenalised';
lamVal = (lambda/m) .* thetaPenalised;
lamVal = [0,lamVal];
grad = g .+ lamVal;



% =============================================================

end
