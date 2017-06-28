function [delete] = removeMissingData(X, featuresIDs, unkonwnData)

nRows = size(X, 1);
nCols = size(X, 2);
delete = zeros(nRows, 1);

fprintf('   Before removing the missing data there are %d data.\n', nRows);

for i = 1:nRows
    for j = 1:nCols
        if ( isnan(X(i,j)) || (X(i,j) == unkonwnData(featuresIDs(j))))
            delete(i) = 1;
            break;
        end
    end
end

deleted = sum(delete);
perc = deleted / nRows * 100;

fprintf('   %d data are removed which account for %5.2f%% of inital data.\n', deleted, perc);

end
