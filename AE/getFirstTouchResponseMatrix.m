function [responseMatrix] = getFirstTouchResponseMatrix(uArray)
    nTrials = length(uArray.trials);
    nCells = size(uArray.cellNums, 2);
    responseMatrix = nan(nCells, nTrials);
    
    for i = 1:nTrials
       thisTrial = uArray.trials{i};
       if ~isempty(thisTrial.protractionTouchChunksByWhisking)
           
           firstTouchTime = thisTrial.whiskerTime(thisTrial.protractionTouchChunksByWhisking{1}(1));
           for n = 1:size(thisTrial.spk, 1)
               
               tpmIdx = floor(thisTrial.neuindSession(n)/1000);
               if tpmIdx > 4
                   tpmIdx = tpmIdx - 4;
               end
               tIdx = thisTrial.tpmTime{tpmIdx};
               frameAfterTouch = find(tIdx >= firstTouchTime, 1);
               spksAfterTouch = mean(thisTrial.spk(n, (frameAfterTouch):(frameAfterTouch+4)));
               
               responseMatrix(uArray.cellNums == thisTrial.neuindSession(n), i) = spksAfterTouch;
               
           end
           
       else
           firstTouchTime = nan;
       end
    end
end

