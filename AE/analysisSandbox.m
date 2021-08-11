clear all; close all; clc;

% uArray = load('D:\2020 Neural stretching in S1\Data\UberJK025S04_NC.mat'); uArray = uArray.u;
% uArray = load('D:\2020 Neural stretching in S1\Data\UberJK025S19_NC.mat'); uArray = uArray.u;

% uArray = load('D:\2020 Neural stretching in S1\Data\UberJK027S03_NC.mat'); uArray = uArray.u;
% uArray = load('D:\2020 Neural stretching in S1\Data\UberJK027S10_NC.mat'); uArray = uArray.u;

% uArray = load('D:\2020 Neural stretching in S1\Data\UberJK030S03_NC.mat'); uArray = uArray.u;
uArray = load('D:\2020 Neural stretching in S1\Data\UberJK030S21_NC.mat'); uArray = uArray.u;

%% get session-trial vars
% performance
performance = vecFromUArray(uArray, 'response');
performance(performance < 1) = 0;

% angle
angle = vecFromUArray(uArray, 'angle');
protractionKappaV = cellFromUArray(uArray, 'protractionTouchDKappaV');

% response matrix
responseMatrix = getFirstTouchResponseMatrix(uArray);

% layer indices
upperIdx = find(uArray.cellNums < 5000);
lowerIdx = find(uArray.cellNums >= 5000);

% seperate layers and strip nan
% splitResponse{1} = responseMatrix(upperIdx, :); splitResponse{2} = responseMatrix(lowerIdx, :);
trialIdx{1} = find(~isnan(responseMatrix(1, :))); trialIdx{2} = find(~isnan(responseMatrix(end, :)));
splitResponse{1} = responseMatrix(upperIdx, trialIdx{1}); splitResponse{2} = responseMatrix(lowerIdx, trialIdx{2});

% get the upper layer response matrix and strip nan
upperResponse = splitResponse{1};
angle = angle(trialIdx{1});
protractionKappaV = protractionKappaV(trialIdx{1});

% condition cell factors
protractionKappaV = cellfun(@mean, protractionKappaV);

%% plotting
% by trial
figure; subplot(3, 1, 1:2);
imagesc(upperResponse);
subplot(3, 1, 3);
plot(angle, 'k');

% angle sorted
[sortedAngle, sortIdx] = sort(angle);
figure; subplot(3, 1, 1:2);
imagesc(upperResponse(:, sortIdx));
subplot(3, 1, 3);
plot(sortedAngle);

%% dimensionality reduction
nDim = 2;
upperTsne = tsne(upperResponse', 'NumDimensions', nDim);
upperPCA = pca(upperResponse, 'NumComponents', nDim);

% plotting
cScheme = {[1 0 0], [0.7 0 0], [0.4 0 0], [0 0 0], [0 0 0.4], [0 0 0.7], [0 0 1]};
uniqueAngle = unique(angle);

% against angle class
figure; hold on;
for i = 1:size(upperTsne, 1)
   scatter3(angle(i), upperTsne(i, 1), upperTsne(i, 2), 'filled', 'MarkerFaceColor', cScheme{find(uniqueAngle == angle(i))}); 
end

figure; hold on;
for i = 1:size(upperTsne, 1)
   scatter3(angle(i), upperPCA(i, 1), upperPCA(i, 2), 'filled', 'MarkerFaceColor', cScheme{find(uniqueAngle == angle(i))}); 
end

% against a behavioral variable
figure; hold on;
for i = 1:size(upperTsne, 1)
   scatter3(protractionKappaV(i), upperTsne(i, 1), upperTsne(i, 2), 'filled', 'MarkerFaceColor', cScheme{find(uniqueAngle == angle(i))}); 
end

figure; hold on;
for i = 1:size(upperTsne, 1)
   scatter3(protractionKappaV(i), upperPCA(i, 1), upperPCA(i, 2), 'filled', 'MarkerFaceColor', cScheme{find(uniqueAngle == angle(i))}); 
end