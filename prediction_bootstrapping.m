function [correctRate, chanceCR, predError, chancePE, confMat] = prediction_bootstrapping(ypair, outcome, varargin)
% Bootstrapping prediction-output data to match the number of samples
% across outcomes 
% 2021/01/06 JK

% Input:
%     ypair: n x 2 matrix (double). 1st column for prediction, 2nd column for output (correct answer)
%     outcome: p x 1 vector (double). A list of possible outcomes. 
%     varargin{1}: numSamplesBS (double). Number of samples to bootstrap in each outcome. Default: 100
%     varargin{2}: nIterBS (double). Number of iteration for bootstrapping. Default: 100
%     varargin{3}: nShuffle (double). Number of shuffling for calculating chance level.
% 
% Output:
%     correctRate: mean of the iterated bootstrapping correct rates
%     chanceCR: chance level correct rate, from the shuffling
%     predError: prediction error (mean of nIterBS iteration)
%     chancePE: chance level prediction error
%     confMat: confusion matrix

numSamplesBS = 100;
nIterBS = 100;
nShuffle = 100;

if nargin > 3
    numSamplesBS = varargin{1};
    if nargin > 4
        nIterBS = varargin{2};
        if naragin > 5
            nShuffle = varargin{3};
        end
    end
end
    

tempCR = zeros(nIterBS, 1);
tempChanceCR = zeros(nIterBS,1);
tempEA = zeros(nIterBS, 1);
tempChanceEA = zeros(nIterBS,1);
tempConfMat = zeros(length(outcome), length(outcome), nIterBS);
angleInds = cell(length(outcome),1);
for ai = 1 : length(outcome)
    angleInds{ai} = find(ypair(:,1)==outcome(ai));
end
for ii = 1 : nIterBS
    tempIterPair = zeros(numSamplesBS * length(outcome), 2);
    for ai = 1 : length(outcome)
        % bootstrapping
        tempInds = randi(length(angleInds{ai}),[numSamplesBS,1]);
        inds = angleInds{ai}(tempInds);
        tempIterPair( (ai-1)*numSamplesBS+1:ai*numSamplesBS, : ) = ypair(inds,:);
    end
    tempCR(ii) = length(find(tempIterPair(:,2) - tempIterPair(:,1)==0)) / (numSamplesBS * length(outcome));
    tempEA(ii) = mean(abs(tempIterPair(:,2) - tempIterPair(:,1)));

    tempTempCR = zeros(nShuffle,1);
    tempTempEA = zeros(nShuffle,1);
    for shuffi = 1 : nShuffle
        shuffledPair = [tempIterPair(:,1), tempIterPair(randperm(size(tempIterPair,1)),2)];
        tempTempCR(shuffi) = length(find(shuffledPair(:,2) - shuffledPair(:,1)==0)) / (numSamplesBS * length(outcome));
        tempTempEA(shuffi) = mean(abs(shuffledPair(:,2) - shuffledPair(:,1)));
    end
    tempChanceCR(ii) = mean(tempTempCR);
    tempChanceEA(ii) = mean(tempTempEA);

    tempConfMat(:,:,ii) = confusionmat(tempIterPair(:,1), tempIterPair(:,2))/numSamplesBS;
end
correctRate = mean(tempCR);
chanceCR = mean(tempChanceCR);
predError = mean(tempEA);
chancePE = mean(tempChanceEA);
confMat = mean(tempConfMat,3);
        