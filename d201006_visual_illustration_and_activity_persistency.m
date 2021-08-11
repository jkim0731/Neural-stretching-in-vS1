% Visual illustration of population response clustering (PRC)
% (1) 3D plot of PCA results from both naive and expert - representative examples
% (2) Distance plot across angle groups - representative examples
% (3) Quantification - Distance vs Angle difference
% (4) Quantification - using clustering index
% (5) Quantification - population decoding
% (6) Effect of whisker features (removing all other features)
% (7) Effect of each whisker feature
% New analyses
% (8) Transient VS persistent neurons
% (8-1) All active neurons
% (8-2) Angle-tuned neurons
% 

%% load data
clear
baseDir = 'D:\TPM\JK\suite2p\';
loadFn = 'pca_from_whisker_model';
load([baseDir, loadFn]);

mice = [25,27,30,36,39,52];
volumes = {'Upper', 'Lower'};
angles = 45:15:135;
angleColors = turbo(length(angles));

colorsTransient = [248 171 66; 40 170 225] / 255;
colorsPersistent = [1 0 0; 0 0 1];
%% (1) 3D plot of PCA results 
mi = 4;
vi = 1;
pcai = find([pcaResult.mouseInd] == mi & [pcaResult.volumeInd] == vi);

% plotting naive
figure, hold on
for ai = 1 : length(angles)
    angleInd = find(popAct.sortedAngleNaive{vi,mi}==angles(ai));
    scatter3(pcaResult(pcai).pcaCoordSpkNaive(angleInd,1), pcaResult(pcai).pcaCoordSpkNaive(angleInd,2), pcaResult(pcai).pcaCoordSpkNaive(angleInd,3), 10, angleColors(ai,:), 'filled')
end
xlabel('PC1'), ylabel('PC2'), zlabel('PC3')
title('Naive')
% plotting expert
figure, hold on
for ai = 1 : length(angles)
    angleInd = find(popAct.sortedAngleExpert{vi,mi}==angles(ai));
    scatter3(pcaResult(pcai).pcaCoordSpkExpert(angleInd,1), pcaResult(pcai).pcaCoordSpkExpert(angleInd,2), pcaResult(pcai).pcaCoordSpkExpert(angleInd,3), 10, angleColors(ai,:), 'filled')
end
xlabel('PC1'), ylabel('PC2'), zlabel('PC3')
title('Expert')

%%
saveDir = 'C:\Users\shires\Dropbox\Works\Presentation\Seminar\2020 SoonChunHyang Hospital\';
fn = 'pca036Uexpert.eps';
export_fig([saveDir, fn], '-depsc', '-painters', '-r600', '-transparent')
fix_eps_fonts([saveDir, fn])


%% (2) Distance plot across angle groups - representative examples
% with heatmap saturating at 5th and 95th percentiles
mi = 4;
vi = 1;
pcai = find([pcaResult.mouseInd] == mi & [pcaResult.volumeInd] == vi);
titleText = sprintf('JK%03d %s volume', mice(mi), volumes{vi});

figure('units','normalized','position',[0.2 0.3 0.75 0.5]),
% plotting naive
subplot(121), hold on
distIm = pdist2(pcaResult(pcai).pcaCoordSpkNaive(:,1:3), pcaResult(pcai).pcaCoordSpkNaive(:,1:3));
distDistr = triu(distIm,1);
lowClim = prctil(distDistr(:),5);
highClim = prctil(distDistr(:),95);

imagesc(distIm, [lowClim, highClim]), colormap hot
divisionInds = find(diff(popAct.sortedAngleNaive{vi,mi}));
for di = 1 : length(divisionInds)
    plot([-3, size(distIm,2)+3], [divisionInds(di)+1, divisionInds(di)+1], 'k-', 'linewidth',3)
    plot([divisionInds(di)+1, divisionInds(di)+1], [-3, size(distIm,2)+3], 'k-', 'linewidth',3)
end
axis square, axis off
xlim([-3 size(distIm,2)+3]), ylim([-3 size(distIm,1)+3])
xticks([]), yticks([])
title('Naive')

% plotting expert
subplot(122), hold on
distIm = pdist2(pcaResult(pcai).pcaCoordSpkExpert(:,1:3), pcaResult(pcai).pcaCoordSpkExpert(:,1:3));
distDistr = triu(distIm,1);
lowClim = prctil(distDistr(:),5);
highClim = prctil(distDistr(:),95);

imagesc(distIm, [lowClim, highClim]), colormap hot
divisionInds = find(diff(popAct.sortedAngleExpert{vi,mi}));
for di = 1 : length(divisionInds)
    plot([-3, size(distIm,2)+3], [divisionInds(di)+1, divisionInds(di)+1], 'k-', 'linewidth',3)
    plot([divisionInds(di)+1, divisionInds(di)+1], [-3, size(distIm,2)+3], 'k-', 'linewidth',3)
end
axis square, axis off
xlim([-3 size(distIm,2)+3]), ylim([-3 size(distIm,1)+3])
xticks([]), yticks([])
title('Expert')

sgtitle(titleText)

%% (3) Quantification - Distance vs Angle difference
% first, show examples
% then, averaged plot
angleDiffList =[0, unique(sort(pdist(angles')))];
mi = 4;
vi = 1;
pcai = find([pcaResult.mouseInd] == mi & [pcaResult.volumeInd] == vi);
titleText = sprintf('JK%03d %s volume', mice(mi), volumes{vi});

figure('units','normalized','position',[0.2 0.3 0.7 0.3])
% naive
naiveAngles = popAct.sortedAngleNaive{vi,mi};
distIm = pdist2(pcaResult(pcai).pcaCoordSpkNaive(:,1:3), pcaResult(pcai).pcaCoordSpkNaive(:,1:3));
distMat = distIm .* (tril(nan(size(distIm,1)))+1);

distDistrAngleNaive = cell(length(angleDiffList),1);
angleDiff = pdist2(naiveAngles, naiveAngles).*(tril(nan(length(naiveAngles)))+1);
for adi = 1 : length(angleDiffList)
    tempInd = find(angleDiff(:)==angleDiffList(adi));
    distDistrAngleNaive{adi} = distMat(tempInd);
end

subplot(131), hold on
scatterWidth = 0.3;
for adi = 1 : length(angleDiffList)
    xRand = rand(length(distDistrAngleNaive{adi}),1)*scatterWidth - scatterWidth/2;
    scatter(adi+xRand, distDistrAngleNaive{adi}, 0.1, colorsTransient(1,:), '.')
end
boxplot(distMat(:), angleDiff(:))
xlabel('DAngle'), ylabel('Distance')
set(gca, 'box', 'off')
title('Naive')
ylimAll = ylim();

% expert
% expert first, to get appropriate ylim
expertAngles = popAct.sortedAngleExpert{vi,mi};
distIm = pdist2(pcaResult(pcai).pcaCoordSpkExpert(:,1:3), pcaResult(pcai).pcaCoordSpkExpert(:,1:3));
distMat = distIm .* (tril(nan(size(distIm,1)))+1);

distDistrAngleExpert = cell(length(angleDiffList),1);
angleDiff = pdist2(expertAngles, expertAngles).*(tril(nan(length(expertAngles)))+1);
for adi = 1 : length(angleDiffList)
    tempInd = find(angleDiff(:)==angleDiffList(adi));
    distDistrAngleExpert{adi} = distMat(tempInd);
end

subplot(132), hold on
scatterWidth = 0.3;
for adi = 1 : length(angleDiffList)
    xRand = rand(length(distDistrAngleExpert{adi}),1)*scatterWidth - scatterWidth/2;
    scatter(adi+xRand, distDistrAngleExpert{adi}, 0.1, colorsTransient(2,:), '.')
end
boxplot(distMat(:), angleDiff(:))
xlabel('DAngle'), ylabel('Distance')
set(gca, 'box', 'off')
title('Expert')
ylim(ylimAll)





% Overlapping means
subplot(133), hold on
plot(angleDiffList, cellfun(@median, distDistrAngleNaive), '-', 'color', colorsTransient(1,:))
plot(angleDiffList, cellfun(@median, distDistrAngleExpert), '-', 'color', colorsTransient(2,:))
legend({'Naive', 'Expert'}, 'location', 'northwest')
xlabel('DAngle'), ylabel('Median distance')
ylim(ylimAll), xlim([-5 95]), xticks(0:15:90)

%% Averaging median curves
angleDiffList =[0, unique(sort(pdist(angles')))];
naiveMedians = zeros(length(pcaResult),length(angleDiffList));
expertMedians = zeros(length(pcaResult),length(angleDiffList));
for pi = 1 : length(pcaResult)
    mi = pcaResult(pi).mouseInd;
    vi = pcaResult(pi).volumeInd;
    
    % naive
    naiveAngles = popAct.sortedAngleNaive{vi,mi};
    distIm = pdist2(pcaResult(pi).pcaCoordSpkNaive(:,1:3), pcaResult(pi).pcaCoordSpkNaive(:,1:3));
    distMat = distIm .* (tril(nan(size(distIm,1)))+1);

    distDistrAngleNaive = cell(length(angleDiffList),1);
    angleDiff = pdist2(naiveAngles, naiveAngles).*(tril(nan(length(naiveAngles)))+1);
    for adi = 1 : length(angleDiffList)
        tempInd = find(angleDiff(:)==angleDiffList(adi));
        distDistrAngleNaive{adi} = distMat(tempInd);
    end
    naiveMedians(pi,:) = cellfun(@median, distDistrAngleNaive);
    
    %expert
    expertAngles = popAct.sortedAngleExpert{vi,mi};
    distIm = pdist2(pcaResult(pi).pcaCoordSpkExpert(:,1:3), pcaResult(pi).pcaCoordSpkExpert(:,1:3));
    distMat = distIm .* (tril(nan(size(distIm,1)))+1);

    distDistrAngleExpert = cell(length(angleDiffList),1);
    angleDiff = pdist2(expertAngles, expertAngles).*(tril(nan(length(expertAngles)))+1);
    for adi = 1 : length(angleDiffList)
        tempInd = find(angleDiff(:)==angleDiffList(adi));
        distDistrAngleExpert{adi} = distMat(tempInd);
    end
    expertMedians(pi,:) = cellfun(@median, distDistrAngleExpert);
end

figure, hold on
plot(angleDiffList, mean(naiveMedians), 'color', colorsTransient(1,:))
plot(angleDiffList, mean(expertMedians), 'color', colorsTransient(2,:))
legend({'Naive', 'Expert'}, 'location', 'northwest', 'autoupdate', false)
boundedline(angleDiffList, mean(naiveMedians), sem(naiveMedians), 'cmap', colorsTransient(1,:))
boundedline(angleDiffList, mean(expertMedians), sem(expertMedians), 'cmap', colorsTransient(2,:))
xlabel('DAngle')
xticks(angleDiffList), xlim([-5 95]), ylim([0 8])

ylabel('Median distance')



%% (4) Quantification - using clustering index
ciAll = zeros(length(pcaResult),2); % (:,1) naive, (:,2) expert
for pi = 1 : length(pcaResult)
    mi = pcaResult(pi).mouseInd;
    vi = pcaResult(pi).volumeInd;
    
    % naive
    naiveAngles = popAct.sortedAngleNaive{vi,mi};
    naiveCoord = pcaResult(pi).pcaCoordSpkNaive(:,1:3);
    ciAll(pi,1) = clustering_index(naiveCoord, naiveAngles);
    
    % expert
    expertAngles = popAct.sortedAngleExpert{vi,mi};
    expertCoord = pcaResult(pi).pcaCoordSpkExpert(:,1:3);
    ciAll(pi,2) = clustering_index(expertCoord, expertAngles);    
end

figure, hold on
for pi = 1 : length(pcaResult)
    plot(ciAll(pi,:), 'ko-')
end
errorbar(mean(ciAll), sem(ciAll), 'ro', 'lines', 'no')
xlim([0.5 2.5])
ylimCurr = ylim();
ylim([0, ylimCurr(2)])
ylabel('Clustering index')
xticks([1,2])
xticklabels({'Naive', 'Expert'})
xtickangle(45)
[~,p,m] = paired_test(ciAll(:,1), ciAll(:,2));
title(sprintf('p = %s; m = %s', num2str(p,3), m))



%% Clustering index from intermediate angles only








%% (5) Quantification - population decoding
% LDA, SVM, KNN
perfLDA = zeros(length(pcaResult),2);
perfLDAshuffle = zeros(length(pcaResult),2);
perfSVM = zeros(length(pcaResult),2);
perfSVMshuffle = zeros(length(pcaResult),2);
perfKNN = zeros(length(pcaResult),2);
perfKNNshuffle = zeros(length(pcaResult),2);
shuffleIterNum = 100;
for pi = 1 : length(pcaResult)
    fprintf('Processing plane #%d/%d\n', pi, length(pcaResult))
    mi = pcaResult(pi).mouseInd;
    vi = pcaResult(pi).volumeInd;
    
    % naive
    disp('Naive')
    naiveAngles = popAct.sortedAngleNaive{vi,mi};
%     naiveCoord = pcaResult(pi).pcaCoordSpkNaive(:,1:3);
    naiveCoord = popAct.spkNaive{vi,mi};
    
    [~, perfLDA(pi,1)] = angle_tuning_func_reorg_LDA([naiveCoord,naiveAngles], angles);
    [~, perfSVM(pi,1)] = angle_tuning_func_reorg_SVM([naiveCoord,naiveAngles], angles);
    [~, perfKNN(pi,1)] = angle_tuning_func_reorg_KNN([naiveCoord,naiveAngles], angles);
    naiveShuffle = zeros(shuffleIterNum,3);
    parfor shi = 1 : shuffleIterNum
        shuffledAngles = naiveAngles(randperm(length(naiveAngles)));
        tempShuffle = zeros(1,3);
        [~, tempShuffle(1)] = angle_tuning_func_reorg_LDA([naiveCoord,shuffledAngles], angles);
        [~, tempShuffle(2)] = angle_tuning_func_reorg_SVM([naiveCoord,shuffledAngles], angles);
        [~, tempShuffle(3)] = angle_tuning_func_reorg_KNN([naiveCoord,shuffledAngles], angles);
        naiveShuffle(shi,:) = tempShuffle;
    end
    perfLDAshuffle(pi,1) = median(naiveShuffle(:,1));
    perfSVMshuffle(pi,1) = median(naiveShuffle(:,2));
    perfKNNshuffle(pi,1) = median(naiveShuffle(:,3));
    
    
    % expert
    disp('Expert')
    expertAngles = popAct.sortedAngleExpert{vi,mi};
%     expertCoord = pcaResult(pi).pcaCoordSpkExpert(:,1:3);
    expertCoord = popAct.spkExpert{vi,mi};
    
    [~, perfLDA(pi,2)] = angle_tuning_func_reorg_LDA([expertCoord,expertAngles], angles);
    [~, perfSVM(pi,2)] = angle_tuning_func_reorg_SVM([expertCoord,expertAngles], angles);
    [~, perfKNN(pi,2)] = angle_tuning_func_reorg_KNN([expertCoord,expertAngles], angles);
    expertShuffle = zeros(shuffleIterNum,3);
    parfor shi = 1 : shuffleIterNum
        shuffledAngles = expertAngles(randperm(length(expertAngles)));
        tempShuffle = zeros(1,3);
        [~, tempShuffle(1)] = angle_tuning_func_reorg_LDA([expertCoord,shuffledAngles], angles);
        [~, tempShuffle(2)] = angle_tuning_func_reorg_SVM([expertCoord,shuffledAngles], angles);
        [~, tempShuffle(3)] = angle_tuning_func_reorg_KNN([expertCoord,shuffledAngles], angles);
        expertShuffle(shi,:) = tempShuffle;
    end
    perfLDAshuffle(pi,2) = median(expertShuffle(:,1));
    perfSVMshuffle(pi,2) = median(expertShuffle(:,2));
    perfKNNshuffle(pi,2) = median(expertShuffle(:,3));
end

%% Save data
% save('classifierPerformances_pca_3comps_spikes.mat','perf*', 'shuffleIterNum')
save('classifierPerformances_tuned.mat','perf*', 'shuffleIterNum')

%% Plot
% load('classifierPerformances_pca_3comps_spikes.mat','perf*', 'shuffleIterNum')
load('classifierPerformances_tuned.mat','perf*', 'shuffleIterNum')
figure
% LDA
subplot(131), hold on
for pi = 1 : length(pcaResult)
    plot(perfLDA(pi,:), 'ko-')
end
errorbar(mean(perfLDA), sem(perfLDA), 'ro', 'lines', 'no')
errorbar(mean(perfLDAshuffle), sem(perfLDAshuffle), 'o', 'lines', 'no', 'color', [0.6 0.6 0.6])
xlim([0.5 2.5]), ylim([0, 0.8])
ylabel('Classifier performance')
xticks([1,2])
xticklabels({'Naive', 'Expert'})
xtickangle(45)
[~,p,m] = paired_test(perfLDA(:,1), perfLDA(:,2));
title(sprintf('LDA\np = %s\nm = %s', num2str(p,3), m))

% SVM
subplot(132), hold on
for pi = 1 : length(pcaResult)
    plot(perfSVM(pi,:), 'ko-')
end
errorbar(mean(perfSVM), sem(perfSVM), 'ro', 'lines', 'no')
errorbar(mean(perfSVMshuffle), sem(perfSVMshuffle), 'o', 'lines', 'no', 'color', [0.6 0.6 0.6])
xlim([0.5 2.5]), ylim([0, 0.8])
ylabel('Classifier performance')
xticks([1,2])
xticklabels({'Naive', 'Expert'})
xtickangle(45)
[~,p,m] = paired_test(perfSVM(:,1), perfSVM(:,2));
title(sprintf('SVM\np = %s\nm = %s', num2str(p,3), m))

% LDA
subplot(133), hold on
for pi = 1 : length(pcaResult)
    plot(perfKNN(pi,:), 'ko-')
end
errorbar(mean(perfKNN), sem(perfKNN), 'ro', 'lines', 'no')
errorbar(mean(perfKNNshuffle), sem(perfKNNshuffle), 'o', 'lines', 'no', 'color', [0.6 0.6 0.6])
xlim([0.5 2.5]), ylim([0, 0.8])
ylabel('Classifier performance')
xticks([1,2])
xticklabels({'Naive', 'Expert'})
xtickangle(45)
[~,p,m] = paired_test(perfKNN(:,1), perfKNN(:,2));
title(sprintf('KNN\np = %s\nm = %s', num2str(p,3), m))




%% (6) Effect of whisker features (removing all other features)
% using clustering index
ciWhiskerOnly = zeros(length(pcaResult), 2);
for pi = 1 : length(pcaResult)
    ciWhiskerOnly(pi,1) = pcaResult(pi).CIWhiskerNaive{1}(3);
    ciWhiskerOnly(pi,2) = pcaResult(pi).CIWhiskerExpert{1}(3);
end

figure,
subplot(121), hold on
for pi = 1 : length(pcaResult)
    plot(ciAll(pi,:), 'ko-')
end
errorbar(mean(ciAll), sem(ciAll), 'ro', 'lines', 'no')
xlim([0.5 2.5])
ylim([0, 0.45]), yticks(0:0.1:0.4)
ylabel('Clustering index')
xticks([1,2])
xticklabels({'Naive', 'Expert'})
xtickangle(45)
[~,p,m] = paired_test(ciAll(:,1), ciAll(:,2));
title(sprintf('Inferred spikes\np = %s\nm = %s', num2str(p,3), m))

subplot(122), hold on
for pi = 1 : length(pcaResult)
    plot(ciWhiskerOnly(pi,:), 'o-', 'color', [0.6 0.6 0.6])
end
errorbar(mean(ciWhiskerOnly), sem(ciWhiskerOnly), 'ro', 'lines', 'no')
xlim([0.5 2.5])
ylim([0, 0.45]), yticks(0:0.1:0.4)
ylabel('Clustering index')
xticks([1,2])
xticklabels({'Naive', 'Expert'})
xtickangle(45)
[~,p,m] = paired_test(ciWhiskerOnly(:,1), ciWhiskerOnly(:,2));
title(sprintf('Whisker-only model\np = %s\nm = %s', num2str(p,3), m))




%% removing  each feature
% using clustering index
ciWoEachBehav = zeros(length(pcaResult), 5, 2);
ciFull = zeros(length(pcaResult),2);
for pi = 1 : length(pcaResult)
    for bi = 1 : 5
        ciWoEachBehav(pi,bi,1) = pcaResult(pi).CIBehavNaive{bi}(3);
        ciWoEachBehav(pi,bi,2) = pcaResult(pi).CIBehavExpert{bi}(3);
    end
    ciFull(pi,1) = pcaResult(pi).CIFullNaive{1}(3);
    ciFull(pi,2) = pcaResult(pi).CIFullExpert{1}(3);
end


ciDiffBehav = squeeze(ciWoEachBehav(:,:,2) - ciWoEachBehav(:,:,1));
ciDiffFull = ciFull(:,2) - ciFull(:,1);

ciImpactBehav = (ciDiffFull - ciDiffBehav) ./ ciDiffFull;

figure,
errorbar(mean(ciImpactBehav), sem(ciImpactBehav), 'ko', 'lines', 'no'), hold on
plot([0.5 5.5], [0 0], '--', 'color', [0.6 0.6 0.6])
xlim([0.5 5.5])
xticks([1:5]), xticklabels({'Whisker', 'Sound', 'Reward', 'Whisking', 'Licking'}), xtickangle(45)
ylabel('Impact on population response clustering')
set(gca, 'fontname', 'Arial', 'fontsize', 12, 'box', 'off')
%%
saveDir = 'C:\Users\shires\Dropbox\Works\Presentation\Seminar\2020 SoonChunHyang Hospital\';
fn = 'whisker-only model.eps';
export_fig([saveDir, fn], '-depsc', '-painters', '-r600', '-transparent')
fix_eps_fonts([saveDir, fn])

%%
outlierInd = find(ciWhiskerOnly(:,2)-ciWhiskerOnly(:,1)<0);

%% (7) Effect of each whisker feature (from whisker-only model)
% using clustering index
impactCI = zeros(length(pcaResult), 13);
for pi = 1 : length(pcaResult)
    for fi = 1 : 13
        impactCI(pi,fi) = pcaResult(pi).CIWhiskerExpert{fi}(3) - pcaResult(pi).CIWhiskerNaive{fi}(3);
    end
end

% whiskerTouchMat = [maxDthetaMat, maxDphiMat, maxDkappaHMat, maxDkappaVMat, maxSlideDistanceMat, maxDurationMat, ...    
%                     thetaAtTouchMat, phiAtTouchMat, kappaHAtTouchMat, kappaVAtTouchMat, arcLengthAtTouchMat, touchCountMat];

impactCI = impactCI(:,[1,8,9,10,11,12,13,2,3,4,5,7,6]);
impactRatio = -(impactCI(:,2:end) - impactCI(:,1)) ./ impactCI(:,1);

figure, hold on
errorbar([1:6, 8:13], mean(impactRatio), sem(impactRatio), 'ko', 'lines', 'no')
plot([0 14], [0 0], '--', 'color', [0.6 0.6 0.6])
xlim([0.5 13.5]), xticks([1:6, 8:13])
xticklabels({'Azimuthal angle', 'Vertical angle', 'Horizontal curvature','Vertical curvature','Arc length', 'Touch count', ...
    'Push angle', 'Vertical displacement', 'Horizontal bending', 'Vertical bending', 'Touch duration', 'Slide distance'})
xtickangle(45)
ylabel('Impact on DCI')




%%
impactCI = zeros(length(pcaResult), 13);
for pi = 1 : length(pcaResult)
    for fi = 1 : 13
        impactCI(pi,fi) = pcaResult(pi).CIWhiskerExpert{fi}(3) - pcaResult(pi).CIWhiskerNaive{fi}(3);
    end
end

% whiskerTouchMat = [maxDthetaMat, maxDphiMat, maxDkappaHMat, maxDkappaVMat, maxSlideDistanceMat, maxDurationMat, ...    
%                     thetaAtTouchMat, phiAtTouchMat, kappaHAtTouchMat, kappaVAtTouchMat, arcLengthAtTouchMat, touchCountMat];

impactCI = impactCI(:,[1,8,9,10,11,12,13,2,3,4,5,7,6]);
impactCI = impactCI(setdiff(1:size(impactCI,1),outlierInd),:);
impactRatio = -(impactCI(:,2:end) - impactCI(:,1)) ./ impactCI(:,1);

figure, hold on
errorbar([1:6, 8:13], mean(impactRatio), sem(impactRatio), 'ko', 'lines', 'no')
plot([0 14], [0 0], '--', 'color', [0.6 0.6 0.6])
xlim([0.5 13.5]), xticks([1:6, 8:13])
xticklabels({'Azimuthal angle', 'Vertical angle', 'Horizontal curvature','Vertical curvature','Arc length', 'Touch count', ...
    'Push angle', 'Vertical displacement', 'Horizontal bending', 'Vertical bending', 'Touch duration', 'Slide distance'})
xtickangle(45)
ylabel('Impact on DCI')




%%


clear
compName = getenv('computername');
if strcmp(compName, 'HNB228-JINHO')
    baseDir = 'D:\TPM\JK\suite2p\';
else
    baseDir = 'Y:\Whiskernas\JK\suite2p\';
end

saveFn = 'popActivityTuned.mat';
load([baseDir, saveFn], '*Naive', '*Expert', 'info')

load([baseDir, 'cellID_match_persAT_v9.mat'])
% cellIDpersAT
% (1) Persistently tuned
% (2) Transient & tuned
% (3) Persistently active and transiently tuned
% (4) Persistent & tuned (including (1) and (3))
mi = [1,1,2,3,3,4,4,5,5,6,6];

% %% PCA 
% Repeat 101 times in the higher tuned neuron session
% Across dim = [1:10,15:5:35]
% Find the median (51st) clustering index across # of dim, and plot them
% (Save all clustering indices only, not each pca)
% Show its variable explained across # of dim
% Show the map of the median PCA (2 dim)
% Plot averaged median clustering index and their corresponding variable
% explained across # of dim
% Show statistical test

numRepeat = 101;
medSort = 51;
numDims = 3;

numTunedNaive = numTunedNaive([1,2,4:12])';
numTunedExpert = numTunedExpert([1,2,4:12])';

popActNaive = popActNaive([1,2,4:12])';
popActExpert = popActExpert([1,2,4:12])';

sortedAngleNaive = sortedAngleNaive([1,2,4:12])';
sortedAngleExpert = sortedAngleExpert([1,2,4:12])';

sortedCidNaive = sortedCidNaive([1,2,4:12])';
sortedCidExpert = sortedCidExpert([1,2,4:12])';

sortedTouchNaive = sortedTouchNaive([1,2,4:12])';
sortedTouchExpert = sortedTouchExpert([1,2,4:12])';

numVol = 11;


numCellNaivePersAT = zeros(numVol,1);
numCellExpertPersAT = zeros(numVol,1);

numCellNaiveTransAT = zeros(numVol,1);
numCellExpertTransAT = zeros(numVol,1);

pcaNaivePersAT = cell(numVol,length(numDims));
pcaExpertPersAT = cell(numVol,length(numDims));

pcaNaiveTransAT = cell(numVol,length(numDims));
pcaExpertTransAT = cell(numVol,length(numDims));


varExpNaivePersAT = cell(numVol, length(numDims));
varExpExpertPersAT = cell(numVol, length(numDims));

varExpNaiveTransAT = cell(numVol, length(numDims));
varExpExpertTransAT = cell(numVol, length(numDims));



CIExpertPersAT = zeros(numVol, length(numDims));
CINaivePersAT = zeros(numVol, length(numDims));

CIExpertTransAT = zeros(numVol, length(numDims));
CINaiveTransAT = zeros(numVol, length(numDims));



CIExpertAllPersAT = cell(numVol, length(numDims));
CINaiveAllPersAT = cell(numVol, length(numDims));

CIExpertAllTransAT = cell(numVol, length(numDims));
CINaiveAllTransAT = cell(numVol, length(numDims));


startTime = tic;

for vi = 1 : numVol
    lapTime = toc(startTime);
    fprintf('Running vol %d/%d (%d:%d passed)\n', vi, numVol, floor(lapTime/60), floor(mod(lapTime,60)))
    % persistently angle-tuned neurons
    % number of neurons are the same
    
    naiveCellInd = find(ismember(sortedCidNaive{vi},cellIDpersATnaive{mi(vi),1})); % when using cellIDpersATnaive, there was number of cell error by 1)
    numCellNaivePersAT(vi) = length(naiveCellInd);
    for di = 1 : length(numDims)
        numDim = numDims(di);
        [~, pcaNaivePersAT{vi, di}, ~, ~, varExpNaivePersAT{vi, di}] = pca(popActNaive{vi}(:,naiveCellInd), 'NumComponents', numDim);
        CINaivePersAT(vi,di) = clustering_index(pcaNaivePersAT{vi,di}, sortedAngleNaive{vi});
    end
    
    expertCellInd = find(ismember(sortedCidExpert{vi}, cellIDpersATexpert{mi(vi),1}));
    numCellExpertPersAT(vi) = length(expertCellInd);
    for di = 1 : length(numDims)
        numDim = numDims(di);
        [~, pcaExpertPersAT{vi, di}, ~, ~, varExpExpertPersAT{vi, di}] = pca(popActExpert{vi}(:,expertCellInd), 'NumComponents', numDim);
        CIExpertPersAT(vi,di) = clustering_index(pcaExpertPersAT{vi,di}, sortedAngleExpert{vi});
    end
    
    % Transiently angle-tuned neurons
    naiveCellInd = find(ismember(sortedCidNaive{vi}, cellIDpersATnaive{mi(vi),2}));
    numCellNaiveTransAT(vi) = length(naiveCellInd);

    expertCellInd = find(ismember(sortedCidExpert{vi}, cellIDpersATexpert{mi(vi),2}));
    numCellExpertTransAT(vi) = length(expertCellInd);
    numNeuron = min(numCellNaiveTransAT(vi), numCellExpertTransAT(vi));
    if numCellNaiveTransAT(vi) > numNeuron % run repeats
        for di = 1 : length(numDims)
            tempPCA = cell(numRepeat,1);
            tempVarExp = cell(numRepeat,1);
            tempClusterInd = zeros(numRepeat,1);
            numDim = numDims(di);
            popActNaivePersAT = popActNaive{vi}(:,naiveCellInd);
            parfor ri = 1 : numRepeat
                tempInds = randperm(numCellNaiveTransAT(vi), numNeuron);
                [~, tempPCA{ri}, ~, ~, tempVarExp{ri}] = pca(popActNaivePersAT(:,tempInds),  'NumComponents', numDim);
                tempClusterInd(ri) = clustering_index(tempPCA{ri}, sortedAngleNaive{vi});
            end
            [~, sorti] = sort(tempClusterInd, 'descend');
            medInd = sorti(medSort);
            
            pcaNaiveTransAT{vi,di} = tempPCA{medInd};
            varExpNaiveTransAT{vi,di} = tempVarExp{medInd};
            CINaiveTransAT(vi,di) = tempClusterInd(medInd);
            
            CINaiveAllTransAT{vi, di} = tempClusterInd;
            
        end
        
        % in this case, expert session has the min # of neurons
        for di = 1 : length(numDims)
            numDim = numDims(di);
            [~, pcaExpertTransAT{vi, di}, ~, ~, varExpExpertTransAT{vi, di}] = pca(popActExpert{vi}(:,expertCellInd), 'NumComponents', numDim);
            CIExpertTransAT(vi,di) = clustering_index(pcaExpertTransAT{vi,di}, sortedAngleExpert{vi});
        end
    else
        for di = 1 : length(numDims)
            numDim = numDims(di);
            [~, pcaNaiveTransAT{vi, di}, ~, ~, varExpNaiveTransAT{vi, di}] = pca(popActNaive{vi}(:,naiveCellInd), 'NumComponents', numDim);
            CINaiveTransAT(vi,di) = clustering_index(pcaNaiveTransAT{vi,di}, sortedAngleNaive{vi});
        end
        
        % in this case, run repeats in the expert session
        for di = 1 : length(numDims)
            tempPCA = cell(numRepeat,1);
            tempVarExp = cell(numRepeat,1);
            tempClusterInd = zeros(numRepeat,1);
            numDim = numDims(di);
            popActExpertPersAT = popActExpert{vi}(:,expertCellInd);
            parfor ri = 1 : numRepeat
                tempInds = randperm(numCellExpertTransAT(vi), numNeuron);
                [~, tempPCA{ri}, ~, ~, tempVarExp{ri}] = pca(popActExpertPersAT(:,tempInds),  'NumComponents', numDim);
                tempClusterInd(ri) = clustering_index(tempPCA{ri}, sortedAngleExpert{vi});
            end
            [~, sorti] = sort(tempClusterInd, 'descend');
            medInd = sorti(medSort);
            
            pcaExpertTransAT{vi,di} = tempPCA{medInd};
            varExpExpertTransAT{vi,di} = tempVarExp{medInd};
            CIExpertTransAT(vi,di) = tempClusterInd(medInd);
            
            CIExpertAllTransAT{vi, di} = tempClusterInd;
        end
    end
end


% %% save the data
saveFn = 'pcaResultsTuned_pers_trans_AT.mat';
save([baseDir, saveFn], 'pca*', 'varExp*', 'CI*', 'numCell*')


%%
figure,
subplot(121), hold on
for pi = 1 : numVol
    plot([CINaivePersAT(pi), CIExpertPersAT(pi)], 'ko-')
end
errorbar(1, mean(CINaivePersAT), sem(CINaivePersAT), 'ro')
errorbar(2, mean(CIExpertPersAT), sem(CIExpertPersAT), 'ro')
xlim([0.5 2.5]), xticks([1,2]), xticklabels({'Naive', 'Expert'}), xtickangle(45)
ylim([0 0.4]), yticks(0:0.1:0.4)
ylabel('Clustering index')
[~,p,m] = paired_test(CINaivePersAT, CIExpertPersAT);
title(sprintf('Persistently angle-tuned\np = %s\n%s',num2str(p,3), m))

subplot(122), hold on
for pi = 1 : numVol
    plot([CINaiveTransAT(pi), CIExpertTransAT(pi)], 'ko-')
end
errorbar(1, mean(CINaiveTransAT), sem(CINaiveTransAT), 'ro')
errorbar(2, mean(CIExpertTransAT), sem(CIExpertTransAT), 'ro')
xlim([0.5 2.5]), xticks([1,2]), xticklabels({'Naive', 'Expert'}), xtickangle(45)
ylim([0 0.4]), yticks(0:0.1:0.4)
ylabel('Clustering index')
[~,p,m] = paired_test(CINaiveTransAT, CIExpertTransAT);
title(sprintf('Transient & angle-tuned\np = %s\n%s',num2str(p,3), m))

