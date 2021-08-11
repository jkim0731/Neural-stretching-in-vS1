function [cel] = cellFromUArray(uArray, fieldName)
    % a general function to return a trial repeating variable from a uArray
    % as a cell array
    nTrials = length(uArray.trials);
    cel = {};
    for i = 1:nTrials
        cel{i} = uArray.trials{i}.(fieldName);
    end
end

