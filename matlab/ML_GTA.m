clc; clear;

lambda = 0;    % Regularization coefficients
degree = 2;    % Degree of the polynomials for features
minThresholdNumberOfGroups = 400;  % Discard any terror group which has commited less than this number during training (400 is equivalent to 30 groups)

%%
% Choose the feature IDs (from the following table) that you want your data to be trained with
%     Feature  | Year  Country  Suicide  Attack type  Target type  Weapon type  Ransom | GroupID (output)
%  ------------|-----------------------------------------------------------------------|----------------
%       ID     |  2       8       28         29          35            84         119  |       64
%  Unkown data |  -       -       -          9           20            13         -9   |    -9, 20202

featuresIDs = [2 8 28 29 35 84];  % The user only needs to change featuresIDs

% You don't need to touch the following
unkonwnData = ones(119,1) * -10000;
unkonwnData(29 ) =  9 ; 
unkonwnData(35 ) =  20; 
unkonwnData(84 ) =  13; 
unkonwnData(119) = -9 ;

%%
if exist('globalterrorismdb_0616dist.mat', 'file') == 2
    fprintf('Loading globalterrorismdb_0616dist.mat ...\n');
    d1 = load('globalterrorismdb_0616dist.mat');
    dataRaw = d1.data;
else
    fprintf('Loading globalterrorismdb_0616dist.xlsx and saving it for faster future loading ...\n');
    dataRaw = xlsread('globalterrorismdb_0616dist.xlsx');
    save('globalterrorismdb_0616dist.mat', 'data');
end

%%
fprintf('Create X and Y data ...\n');
for i = 1:length(featuresIDs)
    data.X(:,i) = dataRaw(:,featuresIDs(i));
end
data.Y(:,1) = dataRaw(:,59 );  % group name
data.Y(:,2) = dataRaw(:,64 );  % group name ID

%%
disp('Removing the data which have some missing information ...'); 
[data.delete] = removeMissingData(data.X, featuresIDs, unkonwnData);

%%
disp('Removing the data which we have little outout information about them ...'); 
data = removeDataWithLittleOutputInfo(data, minThresholdNumberOfGroups);

%%
disp('Normalizing the features to be zero-mean ...'); 
[data.X, data.mu, data.sigma] = featureNormalize(data.X);

%%
disp('Creating higher order features ...');
data.X = mapPolynomialFeature(data.X, degree);


%%
disp('Divide the data to train, cross validation and test data ...')
data = makeTrainTestData(data);

%%
% Create a vector with unique Terrorist Group IDs
% Also removing the first ID which is -9 which corresponds to unknown group ID
uniqueGroupIDs = unique(data.Ytrain(:,2));
% uniqueGroupIDs(1,:) = [];
nUniqueGroups = length(uniqueGroupIDs);

% Setting the options for the minimixation
options = optimset('GradObj', 'on', 'Algorithm', 'trust-region', 'MaxIter', 50);
nFeatures = size(data.Xtrain, 2);
all_theta = zeros(nUniqueGroups, nFeatures);

for i = 1:nUniqueGroups
    initial_theta = zeros(nFeatures, 1);
    classVec = (data.Ytrain(:,2) == uniqueGroupIDs(i));
    iniCost = costFunctionAndGrad(initial_theta, data.Xtrain, classVec, lambda);
    
    [theta, finalCost] = fminunc (@(t)(costFunctionAndGrad(t, data.Xtrain, classVec, lambda)), ...
                    initial_theta, options);
    all_theta(i,:) = theta(:);
    
    fprintf('    Class %d (out of %d) is trained. Cost function was decreased from %f to %f during minimization.\n', ...
        i, nUniqueGroups, iniCost, finalCost);
end

[mTrain, pTrain] = max(data.Xtrain * all_theta', [], 2);
[mCV   , pCV   ] = max(data.Xcv    * all_theta', [], 2);
errTrain = mean(double(uniqueGroupIDs(pTrain) == data.Ytrain(:,2))) * 100;
errCV    = mean(double(uniqueGroupIDs(pCV   ) == data.Ycv   (:,2))) * 100;
fprintf('\nTraining Set Accuracy        : %f\n', errTrain);
fprintf('Cross validation Set Accuracy: %f\n', errCV);
