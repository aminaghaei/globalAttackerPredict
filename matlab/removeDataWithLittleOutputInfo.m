function data = removeDataWithLittleOutputInfo(data, minThresholdNumberOfGroups)

iniData = size(data.X,1)-sum(data.delete);
fprintf('   Before removing the data with little information there are %d data.\n', iniData);

% Create a vector with unique Terrorist Group IDs
% Also removing the first ID which is -9 which corresponds to unknown group ID
[data.uniqueGroupIDs, ~, uniqueGroupIDsloc] = unique(data.Y(:,2));
data.nUniqueGroups = length(data.uniqueGroupIDs);

groupDataCount = zeros(data.nUniqueGroups,1);

for i = 1:size(data.Y,1)
    if data.delete(i) == 0
        j = uniqueGroupIDsloc(i);
        groupDataCount(j) = groupDataCount(j) + 1;
    end
end

for i = 1:size(data.Y,1)
    j = uniqueGroupIDsloc(i);
    if groupDataCount(j) < minThresholdNumberOfGroups
        data.delete(i) = 1;
    end
end

%
deleted = sum(data.delete);
perc = deleted / iniData * 100;
fprintf('   %d data are removed which account for %5.2f%% of inital data.\n', deleted, perc);

end
