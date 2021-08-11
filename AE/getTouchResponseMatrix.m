function [responseMatrix] = getTouchResponseMatrix(uArray, window)
    %% determine trials, cells in uArray, create empty response matrix to be filled
    nTrials = length(uArray.trials);
    nCells = size(uArray.cellNums, 2);
    responseMatrix = nan(nCells, nTrials);
    
    % for each trial in the uArray
    for i = 1:nTrials
        thisTrial = uArray.trials{i}; % reference to single trial
        % if there was a protraction touch in this trial
        if ~isempty(thisTrial.protractionTouchChunksByWhisking)
            
            % for each whisker touch in this trial, what vid frame did the
            % touch initiate?
            touchFrames = cellfun(@(v)v(1), thisTrial.protractionTouchChunksByWhisking);
            % convert the frame number to common time in seconds
            touchTimes = thisTrial.whiskerTime(touchFrames);
            % we only want touch times that occured before the answer lick
            if ~isempty(thisTrial.answerLickTime)
                touchTimes = touchTimes(touchTimes < thisTrial.answerLickTime);                
            else
                touchTimes = [];
            end
            
            % for each neuron recorded in this trial
            for n = 1:size(thisTrial.spk, 1)
                spksPerTouch = 0;
               
                % get the first number (plane number) of the neuron's ID
                % and wrap planes 5-8 back to 1-4 for index purposes
                tpmIdx = floor(thisTrial.neuindSession(n)/1000);
                if tpmIdx > 4
                    tpmIdx = tpmIdx - 4;
                end
                % get the common recording time of this neuron
                tIdx = thisTrial.tpmTime{tpmIdx};
               
                % for each touch time
                for t = touchTimes
                
                    % get the closest 2p frame for this touch time
                    frameAtTouch = find(tIdx >= t, 1);
                    % count the spikes for this neuron at touch frame +
                    % some window of frames after touch
                    spksPerTouch = spksPerTouch + sum(thisTrial.spk(n, (frameAtTouch):(frameAtTouch+window)));
                
                end
                
                % normalise number of spikes by number of touches
                spksPerTouch = spksPerTouch / length(touchTimes);
                
                % add result to responseMatrix
                responseMatrix(uArray.cellNums == thisTrial.neuindSession(n), i) = spksPerTouch;
                
            end
        end
    end
end

