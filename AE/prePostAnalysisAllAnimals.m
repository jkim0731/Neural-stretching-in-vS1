clear all; close all; clc;
%% load in data
% pre
dataPre =  {load('Data/UberJK025S04_NC_results').resultsStruct,...
            load('Data/UberJK030S03_NC_results').resultsStruct,...
            load('Data/UberJK036S01_NC_results').resultsStruct,...
            load('Data/UberJK039S01_NC_results').resultsStruct,...
            load('Data/UberJK052S03_NC_results').resultsStruct};
       
% post
dataPost = {load('Data/UberJK025S19_NC_results').resultsStruct,...
            load('Data/UberJK030S21_NC_results').resultsStruct,...
            load('Data/UberJK036S17_NC_results').resultsStruct,...
            load('Data/UberJK039S23_NC_results').resultsStruct,...
            load('Data/UberJK052S21_NC_results').resultsStruct};
        
nAnimals = length(dataPre);
layer = 1;

%% correlation distance vs. trial angle
allDistVAngle = {[], []};
allCbin = {cell(1, 7), cell(1, 7)};
for i = 1:nAnimals
   thisPre = dataPre{i};
   thisPost = dataPost{i};
      
   [dPre, vPre]   = distanceAnalysis(thisPre.pca.score{layer}(:, 1), thisPre.trialAngle{layer}, false);
   [cBinPre, classes] = classBin(dPre(:), vPre(:));
   
   [dPost, vPost] = distanceAnalysis(thisPost.pca.score{layer}(:, 1), thisPost.trialAngle{layer}, false);
   [cBinPost, classes] = classBin(dPost(:), vPost(:));
   
   allDistVAngle{1} = [allDistVAngle{1}; cellfun(@mean, cBinPre)];
   allDistVAngle{2} = [allDistVAngle{2}; cellfun(@mean, cBinPost)];
   
   for j = 1:7
      allCbin{1}{j} = [allCbin{1}{j}; cBinPre{j}];
      allCbin{2}{j} = [allCbin{2}{j}; cBinPost{j}];
   end
end

figure; hold on;
shadedErrorBar(classes, mean(allDistVAngle{1}), std(allDistVAngle{1})./sqrt(nAnimals), 'lineprops', 'k');
shadedErrorBar(classes, mean(allDistVAngle{2}), std(allDistVAngle{2})./sqrt(nAnimals), 'lineprops', 'r');
xticks(0:15:90)
xlim([-1 91])

%% distance vs. trial performance
figure; hold on;
means = [];
for i = 1:nAnimals
   thisPre = dataPre{i};
   thisPost = dataPost{i};
   
   r = thisPre.pca.score{layer}(:, 1:3); rD = vecnorm(r');
   c = thisPre.trialChoice{layer};
   
   tA = thisPre.trialAngle{layer};
   rD = rD(tA ~=90);
   c = c(tA ~=90);
   
   means = [means; [mean(rD(c == 0)), mean(rD(c == 1))]];
   plot([0, 1], [mean(rD(c == 0)), mean(rD(c == 1))], 'k')  
end
xlim([-0.2, 1.2]); ylim([3 16])

figure; hold on;
means = [];
for i = 1:nAnimals
   thisPre = dataPre{i};
   thisPost = dataPost{i};
   
   r = thisPost.pca.score{layer}(:, 1:3); rD = vecnorm(r');
   c = thisPost.trialChoice{layer};
   
   tA = thisPost.trialAngle{layer};
   rD = rD(tA ~=90);
   c = c(tA ~=90);
   
   means = [means; [mean(rD(c == 0)), mean(rD(c == 1))]];
   plot([0, 1], [mean(rD(c == 0)), mean(rD(c == 1))], 'r')  
end
xlim([-0.1, 1.1]); ylim([3 16])

