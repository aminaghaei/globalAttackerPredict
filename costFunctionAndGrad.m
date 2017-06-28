function [J, dJ] = costFunctionAndGrad(theta, X, y, lambda)
%   [J, dJ] = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of
%   using theta as the parameter for regularized logistic regression and
%   the gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size(theta); % number of features
J = 0;
dJ = zeros(n);

% Calculating the cost
h = X * theta;
g = sigmoid(h);

J = sum(-y .* log(g) - (1-y) .* log(1-g)) / m + ...
    (lambda / 2 / m) * sum(theta(2:end) .* theta(2:end));

% Calculating the derivative of the cost function
temp1 = X' * (g - y);
dJ(1) = temp1(1) / m;
dJ(2:end) = temp1(2:end) / m + (lambda / m) * theta(2:end);

end