clear all; close all; clc;

%% load uArray
uArray = load('D:\2020 Neural stretching in S1\Data\UberJK025S19_NC.mat'); uArray = uArray.u;

%% format data
% for each trial, let's get spiking activity after the first pole touch for
% each neuron
nTrials = length(uArray.trials); % how many trials in this session
nCells  = size(uArray.cellNums, 2); % how many cells overall
responseMatrix = nan(nCells, nTrials); % blank matrix to store responses

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
