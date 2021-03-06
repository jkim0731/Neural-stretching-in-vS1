clear all; close all; clc;
%% load the animal/session to analyse as uArray
% fname = 'Data/UberJK025S04_NC';
% fname = 'Data/UberJK025S19_NC';
% cname = 'Data/UberJK025_CellIDs';

% fname = 'Data/UberJK030S03_NC';
% fname = 'Data/UberJK030S21_NC';
% cname = 'Data/UberJK030_CellIDs';

% fname = 'Data/UberJK036S01_NC';
fname = 'Data/UberJK036S17_NC';
cname = 'Data/UberJK036_CellIDs';

% fname = 'Data/UberJK039S01_NC';
% fname = 'Data/UberJK039S23_NC';
% cname = 'Data/UberJK039_CellIDs';

% fname = 'Data/UberJK052S03_NC';
% fname = 'Data/UberJK052S21_NC';
% cname = 'Data/UberJK052_CellIDs';

uArray = load([fname '.mat']); uArray = uArray.u;
commonCells = load([cname '.mat']); commonCells = commonCells.commonCells;

[~, commonIdx] = intersect(uArray.cellNums, commonCells);

%% extract and format data from the uArray to use in further analysis
% first, the response matrix - average spikes/touch
responseMatrix = getTouchResponseMatrix(uArray, 1);
responseMatrix = responseMatrix(commonIdx, :);

% also extract any trial-by-trial variables we will need
angle = vecFromUArray(uArray, 'angle');
choice = vecFromUArray(uArray, 'response');
polePosition = vecFromUArray(uArray, 'position');
protractionKappaV = cellfun(@mean, cellFromUArray(uArray, 'protractionTouchDKappaV')); % also do mean over touches in each trial with cellfun
protractionSlideD = cellfun(@mean, cellFromUArray(uArray, 'protractionTouchSlideDistance'));

% trials recorded switching between layer volumes - use the cell IDs to
% determine which trials contain which layers
uArray.cellNums = uArray.cellNums(commonIdx);
layerIdx{1} = find(uArray.cellNums < 5000);
layerIdx{2} = find(uArray.cellNums >= 5000);

% some trials will have nan values due to absence of touch, we need an
% index of touch trials for each layer set
trialIdx{1} = find(~isnan(responseMatrix(1, :)));
trialIdx{2} = find(~isnan(responseMatrix(end, :)));

% split the responseMatrix according to touch trials and recorded layer
splitResponse{1} = responseMatrix(layerIdx{1}, trialIdx{1});
splitResponse{2} = responseMatrix(layerIdx{2}, trialIdx{2});

% since we discarded trials with no touch, we need to shape the behavioural
% variables too when we extract them
trialAngle{1} = angle(trialIdx{1}); trialAngle{2} = angle(trialIdx{2});
trialChoice{1} = choice(trialIdx{1}); trialChoice{2} = choice(trialIdx{2});
trialPoleP{1} = polePosition(trialIdx{1}); trialPoleP{2} = polePosition(trialIdx{2});
trialKappaV{1} = protractionKappaV(trialIdx{1}); trialKappaV{2} = protractionKappaV(trialIdx{2});
trialSlideD{1} = protractionSlideD(trialIdx{1}); trialSlideD{2} = protractionSlideD(trialIdx{2});

%% dimensionality reduction - all data
[coeff{1},score{1},latent{1},tsquared{1},explained{1},mu{1}] = pca(splitResponse{1}');
[coeff{2},score{2},latent{2},tsquared{2},explained{2},mu{2}] = pca(splitResponse{2}');

% variance explained from pca on each response part
figure; hold on;
plot(explained{1}); plot(explained{2});

% show the data re-plotted in PCA space along the top 3 components
% get unique angle classes and define a plot color scheme for them
cScheme = {[1 0 0], [0.7 0 0], [0.4 0 0], [0 0 0], [0 0 0.4], [0 0 0.7], [0 0 1]};
uniqueAngle = unique(angle);

figure;
% for each response split
for i = 1:size(score, 2)
   subplot(1,2,i); hold on;
   % for each trial in this split plot the d-reduced neural response, coded
   % by angle
   for j = 1:size(score{i}, 1)
       scatter3(score{i}(j, 1), score{i}(j, 2), score{i}(j, 3), 'filled', 'MarkerFaceColor', cScheme{find(uniqueAngle == trialAngle{i}(j))});
   end
end

[distance{1}, varDistance{1}] = distanceAnalysis(score{1}(:, 1:3), trialAngle{1}, true);
[distance{2}, varDistance{2}] = distanceAnalysis(score{2}(:, 1:3), trialAngle{2}, true);

% summary distance metric
figure; hold on;
histogram(distance{1}, 0:0.5:40); meanDistance(1) = mean(distance{1}(:));
histogram(distance{2}, 0:0.5:40); meanDistance(2) = mean(distance{2}(:));

% correlation of PCs with behavioral variables
trialVar = trialAngle;
nPCs = 3;
figure; c = 1;
for i = 1:length(score)
   for j = 1:nPCs
       subplot(length(score), nPCs, c);
       scatter(trialVar{i}, score{i}(:, j));
       [cc, p] = corrcoef(trialVar{i}, score{i}(:, j), 'Rows', 'complete');
       title(['cc: ' num2str(cc(1, 2)), ' p = ' num2str(p(1, 2))])
       
       c = c + 1;
   end
end

%% dimensionality reduction - leave one out
% choose split
splitID = 1;
response = splitResponse{splitID};
trialVar = trialAngle{splitID};

for n = 1:size(response, 1)
    % leave current neuron idx out of response data and run pca
    nIdx = 1:size(splitResponse{1}, 1); nIdx = nIdx(nIdx~=n);
    [cf,sc,la,tsq,ex,mu] = pca(response(nIdx, :)');
    [d, vD] = distanceAnalysis(sc(:, 1:3), trialVar, false);
    
    distanceMetricMean(n) = mean(d(:));
    distanceMetricSTD(n)  = std(d(:));
    cc = corrcoef(vD(:), d(:), 'Rows', 'Complete');
    trialVarCorr(n) = cc(1, 2);      
end

figure; 
subplot(2, 1, 1)
errorbar(1:size(response, 1), distanceMetricMean, distanceMetricSTD)
subplot(2, 1, 2)
plot(trialVarCorr);

figure;
scatter(trialVarCorr, coeff{splitID}(:, 1));
[cc, p] = corrcoef(trialVarCorr, coeff{splitID}(:, 1));
title(['cc: ' num2str(cc(1, 2)), ' p = ' num2str(p(1, 2))])

%% save structure for post-analysis
resultsStruct = struct();
resultsStruct.splitResponse = splitResponse;
resultsStruct.distance = distance;
resultsStruct.trialAngle = trialAngle;
resultsStruct.trialChoice = trialChoice;
resultsStruct.trialPoleP = trialPoleP;
resultsStruct.trialSlideD = trialSlideD;
resultsStruct.trialKappaV = trialKappaV;
resultsStruct.pca = struct();
resultsStruct.pca.score = score;
resultsStruct.pca.coeff = coeff;
resultsStruct.pca.explained = explained;

save([fname '_results.mat'], 'resultsStruct');

