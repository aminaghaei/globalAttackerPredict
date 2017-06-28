function out = mapPolynomialFeature(X, degree)
%
%   mapPolynomialFeature(X, degree) maps the input features X to polynomial features.
%
%   Returns a new feature array with more features, comprising of 
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%

if (degree >= 4)
    ME = MException('ML_GTA:mapPolynomialFeature', ...
        'Degree %d is not supported for the calculation of features.', degree);
    throw(ME);
end

m = size(X, 1);
n = size(X, 2); % number of original feature
N = 1 + n + n*(n+1)/2 * (degree >= 2) + n*(n+1)*(n+2)/6 * (degree >= 3); % number of features considering the polynoimals
out = [ones(m, 1), X];

if (degree >=2)
    for i = 1:n
        for j = i:n
            out(:,end+1) = X(:,i) .* X(:,j);
        end
    end
end

if (degree >=3)
    for i = 1:n
        for j = i:n
            for k = j:n
                out(:,end+1) = (X(:,i) .* X(:,j)) .* X(:,k);
            end
        end
    end
end

if (size(out,2) ~= N)
    ME = MException('ML_GTA:mapPolynomialFeature', ...
        'Error in mapPolynomialFeature() function');
    throw(ME);
end

end
