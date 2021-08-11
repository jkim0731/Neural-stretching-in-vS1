function [vec] = vecFromUArray(uArray, fieldName)
    % a general function to return a trial repeating variable from a uArray
    % as a single vector
    nTrials = length(uArray.trials);
    for i = 1:nTrials
       vec(i) = uArray.trials{i}.(fieldName); 
    end
end

