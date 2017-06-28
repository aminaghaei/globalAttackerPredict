function data = makeTrainTestData(data)

thresholdCV = 0.7;  % A real number between 0 and 1

m = size(data.X,1); % number of data
nX = size(data.X,2); % number of X features
nY = size(data.Y,2); % number of Y outputs

data.Xtrain = zeros(m,nX);
data.Ytrain = zeros(m,nY);
data.Xtest  = zeros(m,nX);
data.Ytest  = zeros(m,nY);
data.Xcv    = zeros(m,nX);
data.Ycv    = zeros(m,nY);

jTrain = 0;
jTest = 0;
jCV = 0;

for i = 1:m
    if (data.delete(i) == 1)
        continue;
    end
    
    if (data.Y(i,2) < 0 || data.Y(i,2) == 20202)
        jTest = jTest + 1;
        data.Xtest(jTest,:) = data.X(i,:);
        data.Ytest(jTest,:) = data.Y(i,:);
    elseif (rand < thresholdCV)
        jTrain = jTrain + 1;
        data.Xtrain(jTrain,:) = data.X(i,:);
        data.Ytrain(jTrain,:) = data.Y(i,:);
    else
        jCV = jCV + 1;
        data.Xcv(jCV,:) = data.X(i,:);
        data.Ycv(jCV,:) = data.Y(i,:);
    end
end

data.Xtrain(jTrain+1:end,:) = [];
data.Ytrain(jTrain+1:end,:) = [];
data.Xtest (jTest +1:end,:) = [];
data.Ytest (jTest +1:end,:) = [];
data.Xcv   (jCV   +1:end,:) = [];
data.Ycv   (jCV   +1:end,:) = [];

end