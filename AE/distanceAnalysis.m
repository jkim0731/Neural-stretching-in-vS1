function [distance, varDistance] = distanceAnalysis(dataMatrix, comparisonVar, verbose)
    % dataMatrix: rows should be observations, columns variables / dims
    % comparisonVar: behavioral variable to plot against, should have same
    % number of entries as dataMatrix observations
    % verbose: true/false show figures

    % sort the behavioral variable for easy visualisation
    [sortedVar, sortingIdx] = sort(comparisonVar);

    % pairwise euclidean distance between observations
    distance = squareform(pdist(dataMatrix(sortingIdx, :)));

    % pairwise euclidean distance between behavioral var
    varDistance = squareform(pdist(sortedVar'));

    % plot figure if verbose=true
    if verbose
        % show distance heatmap
        figure; 
        subplot(1,2,1);
        imagesc(distance); set(gca, 'YDir', 'normal');

        % distance as a function of var difference
        subplot(1,2,2);
        boxplot(distance(:), varDistance(:));
    end
end

