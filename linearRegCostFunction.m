function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
theta2=theta;
theta2(1)=0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.

p=X*theta;
J=(sum((p-y).^2))/(2*m)+(theta2'*theta2*lambda/(2*m))

%grad=(predictions-y)'*X/m;
%save value for 1
grad=((p-y)'*X/m);
H=grad(1);
grad=((p-y)'*X/m)+(lambda*theta'/m);
%+lambda*theta/m;
grad(1)=H;









% =========================================================================

grad = grad(:);

end
