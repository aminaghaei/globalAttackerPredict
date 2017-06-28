function [Xnorm, mu, sigma] = featureNormalize(X)

%   featureNormalize(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

szX = size(X, 2);
mu = zeros(1, szX);
sigma = zeros(1, szX);
Xnorm = zeros(size(X));

for i = 1:szX
    mu(1,i) = mean(X(:,i));
    sigma(1,i) = std(X(:,i));
    Xnorm(:,i) = (X(:,i) - mu(1,i)) /sigma(1,i);
end

end