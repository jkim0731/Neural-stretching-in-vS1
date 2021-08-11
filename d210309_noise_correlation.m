% Noise correlation
% First, quick and dirty test to see if there's any change in noise
% correlation between pairs of neurons across learning.
% Focus on the persistently tuned neurons first.
% Then, try figure out if that was the cause of increased clustering after
% learning (how?)
% Is there hub neurons, in terms of functional connectivity inferred by
% noise correlation?
% 2021/03/09 JK

%% (1) How does pairwise noise correlation change?
%% (1) - 1. Noise correlation distribution
% VS angle tuning (Delta_preferred angle, or angle tuning curve
% correlation)
% Should be calculated in different stimuli

clear
cd('C:\Users\jinho\Dropbox\Works\Projects\2020 Neural stretching in S1\Data')
numVol = 11;
angles = 45:15:135;
load('matchedPopResponse_201230')


%%
for vi = 2 : numVol
% for vi = 1
    patInd = intersect(naive(vi).indTuned, expert(vi).indTuned);
    tuningCurvesNaive = zeros(length(patInd), length(angles));
    tuningCurvesExpert = zeros(length(patInd), length(angles));
    tunedAnglesNaive = naive(vi).tunedAngle(find(ismember(naive(vi).indTuned, patInd)));
    tunedAnglesExpert = expert(vi).tunedAngle(find(ismember(expert(vi).indTuned, patInd)));
    
    ncTouchNaive = zeros(length(patInd), length(patInd), length(angles));
    ncTouchExpert = zeros(length(patInd), length(patInd), length(angles));
    
    ncTrialNaive = zeros(length(patInd), length(patInd), length(angles));
    ncTrialExpert = zeros(length(patInd), length(patInd), length(angles));
    
    for ai = 1 : length(angles)
        angle = angles(ai);
        trialInds = find(naive(vi).trialAngle == angle);
        tuningCurvesNaive(:,ai) = nanmean(naive(vi).touchBeforeAnswer(patInd, trialInds),2);
        ncTouchNaive(:,:,ai) = corrcoef(naive(vi).touchBeforeAnswer(patInd,trialInds)', 'Rows', 'pairwise');
        ncTrialNaive(:,:,ai) = corrcoef(naive(vi).poleBeforeAnswer(patInd,trialInds)');
        
        trialInds = find(expert(vi).trialAngle == angle);
        tuningCurvesExpert(:,ai) = nanmean(expert(vi).touchBeforeAnswer(patInd, trialInds),2);
        ncTouchExpert(:,:,ai) = corrcoef(expert(vi).touchBeforeAnswer(patInd,trialInds)', 'Rows', 'pairwise');
        ncTrialExpert(:,:,ai) = corrcoef(expert(vi).poleBeforeAnswer(patInd,trialInds)');
    end
    
    figure('unit', 'inch', 'pos', [2 3 13 5]) 
    for ai = 1:length(angles)
        [~,sortiNaive] = sort(tunedAnglesNaive);
        [~,sortiExpert] = sort(tunedAnglesExpert);
        naiveSortedNcTouchNaive = ncTouchNaive(sortiNaive, sortiNaive, ai);
        subplot(3,8,ai), imagesc(naiveSortedNcTouchNaive, [-1 1]), axis square
        title([num2str(angles(ai)), '\circ'])
        if ai == 1
            ylabel('Naive')
        end
        naiveSortedNcTouchExpert = ncTouchExpert(sortiNaive,sortiNaive,ai);
        subplot(3,8,ai+8), imagesc(naiveSortedNcTouchExpert, [-1 1]), axis square
        if ai == 1
            ylabel('Expert')
        end
        naiveSortedNcTouchDiff = naiveSortedNcTouchExpert - naiveSortedNcTouchNaive;
        subplot(3,8,ai+16), imagesc(naiveSortedNcTouchDiff, [-1 1]), axis square
        if ai == 1
            ylabel('Expert - Naive')
        end

    end
    subplot(3,8,8), imagesc(nanmean(ncTouchNaive,3), [-1 1]), axis square
    title('Average')
    subplot(3,8,16), imagesc(nanmean(ncTouchExpert,3), [-1 1]), axis square
    subplot(3,8,24), imagesc(nanmean(ncTouchExpert - ncTouchNaive,3), [-1 1]), axis square

    sgtitle(sprintf('Volume #%d', vi))
end
% 
% %% Does noise correlation change a lot by calculating from different stimuli?
% % Compare nanmean rho with each stimulus rho
% % Look at the spread
% % In each volume, and in congregate
% 
% % figure, 


% %%

% figure, 
% for ai = 1:length(angles)
%     [~,sortiNaive] = sort(tunedAnglesNaive);
%     [~,sortiExpert] = sort(tunedAnglesExpert);
%     naiveSortedNcTrialNaive = ncTrialNaive(sortiNaive, sortiNaive, ai);
%     subplot(3,7,ai), imagesc(naiveSortedNcTrialNaive, [-1 1])
%     naiveSortedNcTrialExpert = ncTrialExpert(sortiNaive,sortiNaive,ai);
%     subplot(3,7,ai+7), imagesc(naiveSortedNcTrialExpert, [-1 1])
%     naiveSortedNcTrialDiff = naiveSortedNcTrialExpert - naiveSortedNcTrialNaive;
%     subplot(3,7,ai+14), imagesc(naiveSortedNcTrialDiff, [-1 1])
% end


%% Distribution of noise correlation
% Across stimuli
% Across neurons tuned to certain angle
% To have statistical power, look at all tuned neurons at each session

load('popResponse_201228')
%% Show noise correlation distribution in images
figure('unit', 'inch', 'pos', [1 2 6 6])
ncWithinTuningStim = zeros(length(angle),length(angles), numVol); % columns: stimulation angle; rows: within tuning
for vi = 1:11
    
    % nc_withinTuning_stim(i,j) means within neurons that are tuned to ith
    % angle (angles(i)) at trials of jth object angle (angles(j))
    for ai = 1:length(angles)
        trialInds = find(naive(vi).trialAngle == angles(ai));
        rhomat = corrcoef(naive(vi).touchBeforeAnswer(naive(vi).indTuned,trialInds)', 'Rows', 'pairwise');
        for tai = 1:length(angles)
            tunedAngle = angles(tai);
            tunedInd = find(naive(vi).tunedAngle == tunedAngle);
            tempMat = make_diag_nan(rhomat(tunedInd, tunedInd));
            ncWithinTuningStim(tai,ai,vi) = nanmean(tempMat(:));

        end
    end

    subplot(3,4,vi), imagesc(ncWithinTuningStim(:,:,vi))
    axis equal, colorbar
    xticks([1,4,7]), xticklabels(num2cell(angles([1,4,7])))
    yticks([1,4,7]), yticklabels(num2cell(angles([1,4,7])))
    xlim([0.5 7.5]), ylim([0.5 7.5])
    title(sprintf('Volume #%d',vi))
    xlabel('Trial stimulus (\circ)')
    ylabel('Preferred angle')
    
end
subplot(3,4,12), imagesc(nanmean(ncWithinTuningStim,3))
axis equal, colorbar
xticks([1,4,7]), xticklabels(num2cell(angles([1,4,7])))
yticks([1,4,7]), yticklabels(num2cell(angles([1,4,7])))
xlim([0.5 7.5]), ylim([0.5 7.5])
title('Average')
xlabel('Trial stimulus (\circ)')
ylabel('Preferred angle')
    
sgtitle('Within tuning noice correlation (Naive)')
%%
figure('unit', 'inch', 'pos', [2 3 6 6])
ncWithinTuningStim = zeros(length(angle),length(angles), numVol); % columns: stimulation angle; rows: within tuning
for vi = 1:11

    % nc_withinTuning_stim(i,j) means within neurons that are tuned to ith
    % angle (angles(i)) at trials of jth object angle (angles(j))
    for ai = 1:length(angles)
        trialInds = find(expert(vi).trialAngle == angles(ai));
        rhomat = corrcoef(expert(vi).touchBeforeAnswer(expert(vi).indTuned,trialInds)', 'Rows', 'pairwise');
        for tai = 1:length(angles)
            tunedAngle = angles(tai);
            tunedInd = find(expert(vi).tunedAngle == tunedAngle);
            tempMat = make_diag_nan(rhomat(tunedInd, tunedInd));
            ncWithinTuningStim(tai,ai,vi) = nanmean(tempMat(:));
        end
    end

    subplot(3,4,vi), imagesc(ncWithinTuningStim(:,:,vi))
    axis equal, colorbar
    xticks([1,4,7]), xticklabels(num2cell(angles([1,4,7])))
    xlabel('Trial stimulus (\circ)')
    yticks([1,4,7]), yticklabels(num2cell(angles([1,4,7])))
    ylabel('Preferred angle')
    xlim([0.5 7.5]), ylim([0.5 7.5])
    title(sprintf('Volume #%d',vi))
end
subplot(3,4,12), imagesc(nanmean(ncWithinTuningStim,3))
axis equal, colorbar
xticks([1,4,7]), xticklabels(num2cell(angles([1,4,7])))
yticks([1,4,7]), yticklabels(num2cell(angles([1,4,7])))
xlim([0.5 7.5]), ylim([0.5 7.5])
title('Average')
xlabel('Trial stimulus (\circ)')
ylabel('Preferred angle')

sgtitle('Within tuning noice correlation (Expert)')

%% Quantify noise correlation distribution
ncWithinTuningStim = zeros(length(angle),length(angles), numVol, 2); % columns: stimulation angle; rows: within tuning
for vi = 1:11
    % nc_withinTuning_stim(i,j) means within neurons that are tuned to ith
    % angle (angles(i)) at trials of jth object angle (angles(j))
    for ai = 1:length(angles)
        trialInds = find(naive(vi).trialAngle == angles(ai));
        rhomat = corrcoef(naive(vi).touchBeforeAnswer(naive(vi).indTuned,trialInds)', 'Rows', 'pairwise');
        for tai = 1:length(angles)
            tunedAngle = angles(tai);
            tunedInd = find(naive(vi).tunedAngle == tunedAngle);
            tempMat = make_diag_nan(rhomat(tunedInd, tunedInd));
            ncWithinTuningStim(tai,ai,vi,1) = nanmean(tempMat(:));
        end
    end

    for ai = 1:length(angles)
        trialInds = find(expert(vi).trialAngle == angles(ai));
        rhomat = corrcoef(expert(vi).touchBeforeAnswer(expert(vi).indTuned,trialInds)', 'Rows', 'pairwise');
        for tai = 1:length(angles)
            tunedAngle = angles(tai);
            tunedInd = find(expert(vi).tunedAngle == tunedAngle);
            tempMat = make_diag_nan(rhomat(tunedInd, tunedInd));
            ncWithinTuningStim(tai,ai,vi,2) = nanmean(tempMat(:));
        end
    end
end

%% ANOVA in each tuning
close all
for ai = 1 : length(angles)
%     p = anova1(squeeze(ncWithinTuningStim(ai,:,:,1))');
    p = anova1(squeeze(ncWithinTuningStim(ai,:,:,2))');
    xticklabels(angles)
    xlabel('Trial stimulus (\circ)')
    ylabel('Noise correlation')
    title([sprintf('Within neurons tuned to %d', angles(ai)), '\circ ', sprintf('(ANOVA p = %.3f)',p)])
end

% In both naive and expert session, there is a significant trend that noise
% correlation within neurons that are tuned to an angle has higher noise
% correlation from trials of that specific angle

% Is this because of higher Fano factor?

%% Fano factor calculation
% variance / mean
fanoPerStim = zeros(length(angles),length(angles),numVol,2);
for vi = 1 : numVol
    for ai = 1 : length(angles)
        % naive
        trialInds = find(naive(vi).trialAngle == angles(ai));
        for tai = 1 : length(angles) % tuned angle index
            tunedInd = find(naive(vi).tunedAngle == angles(tai));
            tempMat = naive(vi).touchBeforeAnswer(naive(vi).indTuned(tunedInd),trialInds);
            fanoPerStim(tai,ai,vi,1) = nanmean(var(tempMat,0,2, 'omitnan') ./ nanmean(tempMat,2));
        end
        
        % naive
        trialInds = find(expert(vi).trialAngle == angles(ai));
        for tai = 1 : length(angles) % tuned angle index
            tunedInd = find(expert(vi).tunedAngle == angles(tai));
            tempMat = expert(vi).touchBeforeAnswer(expert(vi).indTuned(tunedInd),trialInds);
            fanoPerStim(tai,ai,vi,2) = nanmean(var(tempMat,0,2, 'omitnan') ./ nanmean(tempMat,2));
        end
        
    end
end

%%
figure, 
subplot(121), imagesc(nanmean(squeeze(fanoPerStim(:,:,:,1)),3)), axis equal, colorbar
xlim([0.5 7.5]), ylim([0.5 7.5])
xticks([1,4,7]), xticklabels(num2cell(angles([1,4,7])))
yticks([1,4,7]), yticklabels(num2cell(angles([1,4,7])))
ylabel('Preferred angle')
xlabel('Trial stimulus (\circ)')
title('Naive')
subplot(122), imagesc(nanmean(squeeze(fanoPerStim(:,:,:,2)),3)), axis equal, colorbar
xlim([0.5 7.5]), ylim([0.5 7.5])
xticks([1,4,7]), xticklabels(num2cell(angles([1,4,7])))
yticks([1,4,7]), yticklabels(num2cell(angles([1,4,7])))
xlabel('Trial stimulus (\circ)')
title('Expert')
sgtitle('Fano factor')

% Fano factor has similar trend in both naive and expert session.
% (45-tuned neurons have higher Fano factor at 45 degrees trials)


%% How do response rate, Fano factor, and noise correlation change across learning?
%% From here on, analysis is focused on persistently angle-tuned neurons.

load('matchedPopResponse_201230');
numVol = 11;
angles = 45:15:135;


%% First, look at them regardless of tuned angle.
rrAll = cell(numVol,2); % each cell from each volume. (:,1) value, (:,2) tuned angle
ffAll = cell(numVol,2); % Averaged across within-stimuli responses, i.e., calculate NC in each stimuli trials and then averaging them (/length(angles))
ncAll = cell(numVol,2); % (:,1) naive, (:,2) expert. Averaged across within-stimuli responses, i.e., calculate NC in each stimuli trials and then averaging them (/length(angles))

for vi = 1 : numVol
    ptInd = intersect(naive(vi).indTuned, expert(vi).indTuned); % sorted, and both indTuned are sorted
    naiveTunedAngle = naive(vi).tunedAngle(find(ismember(naive(vi).indTuned, ptInd)));
    expertTunedAngle = expert(vi).tunedAngle(find(ismember(expert(vi).indTuned, ptInd)));
    
    % Naive
    tempResponse = naive(vi).touchBeforeAnswer(ptInd,:);    
    rrAll{vi,1} = [nanmean(tempResponse,2), naiveTunedAngle];
    tempFF = zeros(length(ptInd),length(angles));
    tempNC = zeros(length(ptInd),length(ptInd),length(angles));
    for ai = 1 : length(angles)
        trialInd = find(naive(vi).trialAngle == angles(ai));
        tempMat = tempResponse(:,trialInd);
        tempFF(:,ai) = var(tempMat,0,2,'omitnan')./nanmean(tempMat,2);
        tempNC(:,:,ai) = corrcoef(tempMat', 'Rows', 'pairwise');        
    end
    ffAll{vi,1} = [nanmean(tempFF,2), naiveTunedAngle];
    ncAll{vi,1} = nanmean(tempNC,3);
    
    % Expert
    tempResponse = expert(vi).touchBeforeAnswer(ptInd,:);    
    rrAll{vi,2} = [nanmean(tempResponse,2), naiveTunedAngle];
    tempFF = zeros(length(ptInd),length(angles));
    tempNC = zeros(length(ptInd),length(ptInd),length(angles));
    for ai = 1 : length(angles)
        trialInd = find(expert(vi).trialAngle == angles(ai));
        tempMat = tempResponse(:,trialInd);
        tempFF(:,ai) = var(tempMat,0,2,'omitnan')./nanmean(tempMat,2);
        tempNC(:,:,ai) = corrcoef(tempMat', 'Rows', 'pairwise');        
    end
    ffAll{vi,2} = [nanmean(tempFF,2), naiveTunedAngle];
    ncAll{vi,2} = nanmean(tempNC,3);
end

% %%
% vi = 11;
% ptInd = intersect(naive(vi).indTuned, expert(vi).indTuned); % sorted, and both indTuned are sorted
% naiveTunedAngle = naive(vi).tunedAngle(find(ismember(naive(vi).indTuned, ptInd)));
% expertTunedAngle = expert(vi).tunedAngle(find(ismember(expert(vi).indTuned, ptInd)));
% 
% % Naive
% tempResponse = naive(vi).touchBeforeAnswer(ptInd,:);    
% tempFF = zeros(length(ptInd),length(angles));
% tempNC = zeros(length(ptInd),length(ptInd),length(angles));
% for ai = 1 : length(angles)
%     trialInd = find(naive(vi).trialAngle == angles(ai));
%     tempMat = tempResponse(:,trialInd);
%     tempFF(:,ai) = var(tempMat,0,2,'omitnan')./nanmean(tempMat,2);
%     tempNC(:,:,ai) = corrcoef(tempMat', 'Rows', 'pairwise');        
% end
%% Look at the distribution
%% Response rate
histGap = 0.06;
histRange = -1:histGap:1;
rrDiffHist = zeros(numVol, length(histRange)-1);
rrDiffMean = zeros(numVol,1);
for vi = 1 : numVol
    rrDiff = rrAll{vi,2}(:,1) - rrAll{vi,1}(:,1);
    rrDiffHist(vi,:) = histcounts(rrDiff, histRange, 'norm', 'probability');
    rrDiffMean(vi) = mean(rrDiff);
end
figure, hold on
for vi = 1 : numVol
    plot(histRange(2:end)-histGap, rrDiffHist(vi,:), 'color', [0.6 0.6 0.6])
end
errorbar(histRange(2:end)-histGap, mean(rrDiffHist), sem(rrDiffHist), 'r-')
xlabel('\DeltaResponse rate (Expert - Naive)')
ylabel('Proportion')
title(['Response rate (Mean \pm SEM: ', sprintf('%.3f', mean(rrDiffMean)), ' \pm ', sprintf('%.3f)',sem(rrDiffMean))])

%%
mean(cellfun(@(x) mean(x(:,1)), rrAll(:,1)))
sem(cellfun(@(x) mean(x(:,1)), rrAll(:,1)))

mean(cellfun(@(x) mean(x(:,1)), rrAll(:,2)))
sem(cellfun(@(x) mean(x(:,1)), rrAll(:,2)))

%% Fano factor
histGap = 0.1;
histRange = -1.5:histGap:1.5;
ffDiffHist = zeros(numVol, length(histRange)-1);
ffDiffMean = zeros(numVol,1);
for vi = 1 : numVol
    ffDiff = ffAll{vi,2}(:,1) - ffAll{vi,1}(:,1);
    ffDiffHist(vi,:) = histcounts(ffDiff, histRange, 'norm', 'probability');
    ffDiffMean(vi) = mean(ffDiff);
end
figure, hold on
for vi = 1 : numVol
    plot(histRange(2:end)-histGap, ffDiffHist(vi,:), 'color', [0.6 0.6 0.6])
end
errorbar(histRange(2:end)-histGap, mean(ffDiffHist), sem(ffDiffHist), 'r-')
xlabel('\DeltaFano factor (Expert - Naive)')
ylabel('Proportion')
title(['Fano factor (Mean \pm SEM: ', sprintf('%.3f', mean(ffDiffMean)), ' \pm ', sprintf('%.3f)',sem(ffDiffMean))])

%%
mean(cellfun(@(x) mean(x(:,1)), ffAll(:,1)))
sem(cellfun(@(x) mean(x(:,1)), ffAll(:,1)))

mean(cellfun(@(x) mean(x(:,1)), ffAll(:,2)))
sem(cellfun(@(x) mean(x(:,1)), ffAll(:,2)))
%% Noise correlation
histGap = 0.04;
histRange = -1:histGap:1;
ncDiffHist = zeros(numVol, length(histRange)-1);
ncDiffMean = zeros(numVol,1);
for vi = 1 : numVol
    diffMat = ncAll{vi,2} - ncAll{vi,1};
    ncDiff = diffMat(find(triu(ones(size(diffMat)),1)));
    ncDiffHist(vi,:) = histcounts(ncDiff, histRange, 'norm', 'probability');
    ncDiffMean(vi) = nanmean(ncDiff);
end
figure, hold on
for vi = 1 : numVol
    plot(histRange(2:end)-histGap, ncDiffHist(vi,:), 'color', [0.6 0.6 0.6])
end
errorbar(histRange(2:end)-histGap, mean(ncDiffHist), sem(ncDiffHist), 'r-')
xlabel('\DeltaNoise correlation (Expert - Naive)')
ylabel('Proportion')
title(['Noise correlation (Mean \pm SEM: ', sprintf('%.3f', mean(ncDiffMean)), ' \pm ', sprintf('%.3f)',sem(ncDiffMean))])

%%
mean(cellfun(@(x) nanmean(make_diag_nan(x), 'all'), ncAll(:,1)))
sem(cellfun(@(x) nanmean(make_diag_nan(x), 'all'), ncAll(:,1)))

mean(cellfun(@(x) nanmean(make_diag_nan(x), 'all'), ncAll(:,2)))
sem(cellfun(@(x) nanmean(make_diag_nan(x), 'all'), ncAll(:,2)))


%% Now, look at the tuned angle trials.
% For response rate and Fano factor, calculate at their most preferred
% angle (can be different across learning)
% For noise correlation, there are two calculations: one paired from naive,
% another paired from expert.

rrAll = cell(numVol,2); % each cell from each volume. (:,1) value, (:,2) tuned angle
ffAll = cell(numVol,2);
ncAllNaive = cell(numVol,length(angles),2); % (:,1) naive, (:,2) expert. Tuned angle determined from naive session
ncAllExpert = cell(numVol,length(angles),2); % (:,1) naive, (:,2) expert. Tuned angle determined from the expert session

for vi = 1 : numVol
    ptInd = intersect(naive(vi).indTuned, expert(vi).indTuned); % sorted, and both indTuned are sorted
    naiveTunedAngle = naive(vi).tunedAngle(find(ismember(naive(vi).indTuned, ptInd)));
    expertTunedAngle = expert(vi).tunedAngle(find(ismember(expert(vi).indTuned, ptInd)));
    
    tempNaiveResponse = naive(vi).touchBeforeAnswer(ptInd,:);
    tempExpertResponse = expert(vi).touchBeforeAnswer(ptInd,:);
    
    
    % Naive
    rr = zeros(length(ptInd),1);
    ff = zeros(length(ptInd),1);
    for ai = 1 : length(angles)
        trialInd = find(naive(vi).trialAngle == angles(ai));
        tunedInd = find(naiveTunedAngle == angles(ai));
        tempMat = tempNaiveResponse(tunedInd,trialInd);
        rr(tunedInd) = nanmean(tempMat,2);
        ff(tunedInd) = var(tempMat,0,2,'omitnan')./nanmean(tempMat,2);
        
        ncAllNaive{vi,ai,1} = corrcoef(tempMat', 'Rows', 'pairwise');
        
        expertTrialInd = find(expert(vi).trialAngle == angles(ai));
        tempMatExpert = tempExpertResponse(tunedInd, expertTrialInd);
        ncAllNaive{vi,ai,2} = corrcoef(tempMatExpert', 'Rows', 'pairwise');
    end
    rrAll{vi,1} = [rr, naiveTunedAngle];
    ffAll{vi,1} = [ff, naiveTunedAngle];
    
    % Expert
    rr = zeros(length(ptInd),1);
    ff = zeros(length(ptInd),1);
    for ai = 1 : length(angles)
        trialInd = find(expert(vi).trialAngle == angles(ai));
        tunedInd = find(expertTunedAngle == angles(ai));
        tempMat = tempExpertResponse(tunedInd,trialInd);
        rr(tunedInd) = nanmean(tempMat,2);
        ff(tunedInd) = var(tempMat,0,2,'omitnan')./nanmean(tempMat,2);
        
        ncAllExpert{vi,ai,2} = corrcoef(tempMat', 'Rows', 'pairwise');
        
        naiveTrialInd = find(naive(vi).trialAngle == angles(ai));
        tempMatNaive = tempNaiveResponse(tunedInd, naiveTrialInd);
        ncAllExpert{vi,ai,1} = corrcoef(tempMatNaive', 'Rows', 'pairwise');
    end
    rrAll{vi,2} = [rr, expertTunedAngle];
    ffAll{vi,2} = [ff, expertTunedAngle];
end


%% Look at the distribution
%% Response rate
histGap = 0.06;
histRange = -1:histGap:1;
rrDiffHist = zeros(numVol, length(histRange)-1);
rrDiffMean = zeros(numVol,1);
for vi = 1 : numVol
    rrDiff = rrAll{vi,2}(:,1) - rrAll{vi,1}(:,1);
    rrDiffHist(vi,:) = histcounts(rrDiff, histRange, 'norm', 'probability');
    rrDiffMean(vi) = mean(rrDiff);
end
figure, hold on
for vi = 1 : numVol
    plot(histRange(2:end)-histGap, rrDiffHist(vi,:), 'color', [0.6 0.6 0.6])
end
errorbar(histRange(2:end)-histGap, mean(rrDiffHist), sem(rrDiffHist), 'r-')
xlabel('\DeltaResponse rate (Expert - Naive)')
ylabel('Proportion')
[~,p] = ttest(rrDiffMean);
title(['Response rate (Mean \pm SEM: ', sprintf('%.3f', mean(rrDiffMean)), ' \pm ', sprintf('%.3f; p = %.3f)',sem(rrDiffMean), p)])
%%
mean(cellfun(@(x) mean(x(:,1)), rrAll(:,1)))
sem(cellfun(@(x) mean(x(:,1)), rrAll(:,1)))

mean(cellfun(@(x) mean(x(:,1)), rrAll(:,2)))
sem(cellfun(@(x) mean(x(:,1)), rrAll(:,2)))
%% Fano factor
histGap = 0.1;
histRange = -1.5:histGap:1.5;
ffDiffHist = zeros(numVol, length(histRange)-1);
ffDiffMean = zeros(numVol,1);
for vi = 1 : numVol
    ffDiff = ffAll{vi,2}(:,1) - ffAll{vi,1}(:,1);
    ffDiffHist(vi,:) = histcounts(ffDiff, histRange, 'norm', 'probability');
    ffDiffMean(vi) = mean(ffDiff);
end
figure, hold on
for vi = 1 : numVol
    plot(histRange(2:end)-histGap, ffDiffHist(vi,:), 'color', [0.6 0.6 0.6])
end
errorbar(histRange(2:end)-histGap, mean(ffDiffHist), sem(ffDiffHist), 'r-')
xlabel('\DeltaFano factor (Expert - Naive)')
ylabel('Proportion')
[~,p] = ttest(ffDiffMean);
title(['Fano factor (Mean \pm SEM: ', sprintf('%.3f', mean(ffDiffMean)), ' \pm ', sprintf('%.3f; p = %.3f)',sem(ffDiffMean), p)])

%%
mean(cellfun(@(x) mean(x(:,1)), ffAll(:,1)))
sem(cellfun(@(x) mean(x(:,1)), ffAll(:,1)))

mean(cellfun(@(x) mean(x(:,1)), ffAll(:,2)))
sem(cellfun(@(x) mean(x(:,1)), ffAll(:,2)))
%% Noise correlation from naive pairing
histGap = 0.08;
histRange = -1.5:histGap:1.5;
ncDiffHist = zeros(numVol, length(histRange)-1);
ncDiffMean = zeros(numVol,1);
for vi = 1 : numVol
    ncDiffCell = cell(length(angles),1);
    for ai = 1 : length(angles)
        diffMat = ncAllNaive{vi,ai,2} - ncAllNaive{vi,ai,1};
        ncDiffCell{ai} = diffMat(find(triu(ones(size(diffMat)),1)));
    end
    ncDiff = cell2mat(ncDiffCell);
    ncDiffHist(vi,:) = histcounts(ncDiff, histRange, 'norm', 'probability');
    ncDiffMean(vi) = nanmean(ncDiff);
end
figure, hold on
for vi = 1 : numVol
    plot(histRange(2:end)-histGap, ncDiffHist(vi,:), 'color', [0.6 0.6 0.6])
end
errorbar(histRange(2:end)-histGap, mean(ncDiffHist), sem(ncDiffHist), 'r-')
xlabel('\DeltaNoise correlation (Expert - Naive)')
ylabel('Proportion')
[~,p] = ttest(ncDiffMean);
title(['Noise correlation from pairing in Naive (Mean \pm SEM: ', sprintf('%.3f', mean(ncDiffMean)), ' \pm ', sprintf('%.3f; p = %.3f)',sem(ncDiffMean), p)])

%% Noise correlation from expert pairing
histGap = 0.08;
histRange = -1.5:histGap:1.5;
ncDiffHist = zeros(numVol, length(histRange)-1);
ncDiffMean = zeros(numVol,1);
for vi = 1 : numVol
    ncDiffCell = cell(length(angles),1);
    for ai = 1 : length(angles)
        diffMat = ncAllExpert{vi,ai,2} - ncAllExpert{vi,ai,1};
        ncDiffCell{ai} = diffMat(find(triu(ones(size(diffMat)),1)));
    end
    ncDiff = cell2mat(ncDiffCell);
    ncDiffHist(vi,:) = histcounts(ncDiff, histRange, 'norm', 'probability');
    ncDiffMean(vi) = nanmean(ncDiff);
end
figure, hold on
for vi = 1 : numVol
    plot(histRange(2:end)-histGap, ncDiffHist(vi,:), 'color', [0.6 0.6 0.6])
end
errorbar(histRange(2:end)-histGap, mean(ncDiffHist), sem(ncDiffHist), 'r-')
xlabel('\DeltaNoise correlation (Expert - Naive)')
ylabel('Proportion')
[~,p] = ttest(ncDiffMean);
title(['Noise correlation from pairing in Expert (Mean \pm SEM: ', sprintf('%.3f', mean(ncDiffMean)), ' \pm ', sprintf('%.3f; p = %.3f)',sem(ncDiffMean), p)])

%% For visual representation of noice correlation calculation
ncWithinTuningStimNaive = zeros(length(angles),length(angles), numVol); % columns: stimulation angle; rows: within tuning
ncWithinTuningStimExpert = zeros(length(angles),length(angles), numVol); % columns: stimulation angle; rows: within tuning
for vi = 1:11

    % nc_withinTuning_stim(i,j) means within neurons that are tuned to ith
    % angle (angles(i)) at trials of jth object angle (angles(j))
    for ai = 1:length(angles)
        trialInds = find(naive(vi).trialAngle == angles(ai));
        rhomat = corrcoef(naive(vi).touchBeforeAnswer(naive(vi).indTuned,trialInds)', 'Rows', 'pairwise');
        for tai = 1:length(angles)
            tunedAngle = angles(tai);
            tunedInd = find(naive(vi).tunedAngle == tunedAngle);
            tempMat = make_diag_nan(rhomat(tunedInd, tunedInd));
            ncWithinTuningStimNaive(tai,ai,vi) = nanmean(tempMat(:));
        end
    end
    
    for ai = 1:length(angles)
        trialInds = find(expert(vi).trialAngle == angles(ai));
        rhomat = corrcoef(expert(vi).touchBeforeAnswer(expert(vi).indTuned,trialInds)', 'Rows', 'pairwise');
        for tai = 1:length(angles)
            tunedAngle = angles(tai);
            tunedInd = find(expert(vi).tunedAngle == tunedAngle);
            tempMat = make_diag_nan(rhomat(tunedInd, tunedInd));
            ncWithinTuningStimExpert(tai,ai,vi) = nanmean(tempMat(:));
        end
    end
end
%%
figure,
subplot(1,3,1), imagesc(nanmean(ncWithinTuningStimNaive,3))
axis equal, colorbar
xticks([1,4,7]), xticklabels(num2cell(angles([1,4,7])))
yticks([1,4,7]), yticklabels(num2cell(angles([1,4,7])))
xlim([0.5 7.5]), ylim([0.5 7.5])
title('Naive average')
ylabel('Preferred angle')

subplot(1,3,2), imagesc(nanmean(ncWithinTuningStimExpert,3))
axis equal, colorbar
xticks([1,4,7]), xticklabels(num2cell(angles([1,4,7])))
yticks([1,4,7]), yticklabels(num2cell(angles([1,4,7])))
xlim([0.5 7.5]), ylim([0.5 7.5])
title('Expert average')
xlabel('Trial stimulus (\circ)')


subplot(1,3,3), imagesc(nanmean(ncWithinTuningStimExpert,3) - nanmean(ncWithinTuningStimNaive,3))
axis equal, colorbar, hold on
plot([0.5 0.5], [0.5 1.5], 'r-')
plot([0.5 1.5], [0.5 0.5], 'r-')
for ai = 1 : length(angles)
    plot([ai+0.5, ai+0.5], [ai-0.5, ai+1.5], 'r-')
    plot([ai-0.5, ai+1.5], [ai+0.5, ai+0.5], 'r-')
end
xticks([1,4,7]), xticklabels(num2cell(angles([1,4,7])))
yticks([1,4,7]), yticklabels(num2cell(angles([1,4,7])))
xlim([0.5 7.5]), ylim([0.5 7.5])
title('Expert - Naive, average')

sgtitle('Within tuning noice correlation')






%%
%% For presentation
%%
%% 2021/03/19

%% Showing heatmap of the data
load('matchedPopResponse_201230')
angles = 45:15:135;

%%
vi = 6;
testInd = intersect(matchedPR.naive(vi).indTuned, matchedPR.expert(vi).indTuned);
[sortedAnglesNaive, angleIndNaive] = sort(matchedPR.naive(vi).trialAngle);
[sortedAnglesExpert, angleIndExpert] = sort(matchedPR.expert(vi).trialAngle);
naiveTunedTestInd = find(ismember(matchedPR.naive(vi).indTuned, testInd));
[sortedTaNaive, taIndNaive] = sort(matchedPR.naive(vi).tunedAngle(naiveTunedTestInd));
expertTunedTestInd = find(ismember(matchedPR.expert(vi).indTuned, testInd));
[sortedTaExpert, taIndExpert] = sort(matchedPR.expert(vi).tunedAngle(expertTunedTestInd));

angleDividersNaive = find(diff(sortedAnglesNaive))+0.5;
angleDividersExpert = find(diff(sortedAnglesExpert))+0.5;
taDividersNaive = find(diff(sortedTaNaive))+0.5;
taDividersExpert = find(diff(sortedTaExpert))+0.5;

heatmapNaive = matchedPR.naive(vi).poleBeforeAnswer(testInd(taIndNaive),angleIndNaive);
heatmapExpert = matchedPR.expert(vi).poleBeforeAnswer(testInd(taIndExpert),angleIndExpert);
maxVal = max(max(heatmapNaive(:)), max(heatmapExpert(:)));
%
figure('unit', 'inch', 'pos', [1 1 13 5]),
subplot(121), imagesc(heatmapNaive, [0 maxVal]), colorbar, hold on

for di = 1 : length(angleDividersNaive)
    plot([angleDividersNaive(di), angleDividersNaive(di)], [0 length(taIndNaive)+1], 'w--')
end
for di = 1 : length(taDividersNaive)
    plot([0 length(angleIndNaive)+1], [taDividersNaive(di), taDividersNaive(di)], 'w--')
end
xtpos = [0; angleDividersNaive] + ([angleDividersNaive; length(angleIndNaive)] - [0; angleDividersNaive] )/2;
xticks(xtpos)
xticklabels(unique(sortedAnglesNaive))
xlabel('Trials (sorted by object angle; \circ)')

ytpos = [0; taDividersNaive] + ([taDividersNaive; length(taIndNaive)] - [0; taDividersNaive])/2;
yticks(ytpos)
yticklabels(unique(sortedTaNaive))
ylabel({'Persistently tuned neurons'; '(sorted by tuned angle; \circ)'})
title('Naive')
set(gca,'labelfontsizemultiplier',1.5, 'titlefontsizemultiplier',1.5)
%
subplot(122), imagesc(heatmapExpert, [0 maxVal]), colorbar, hold on
for di = 1 : length(angleDividersExpert)    
    plot([angleDividersExpert(di), angleDividersExpert(di)], [0 length(taIndExpert)+1], 'w--')
end
for di = 1 : length(taDividersExpert)
    plot([0 length(angleIndExpert)+1], [taDividersExpert(di), taDividersExpert(di)], 'w--')
end

xtpos = [0; angleDividersExpert] + ([angleDividersExpert; length(angleIndExpert)] - [0; angleDividersExpert] )/2;
xticks(xtpos)
xticklabels(unique(sortedAnglesExpert))
xlabel('Trials (sorted by object angle; \circ)')

ytpos = [0; taDividersExpert] + ([taDividersExpert; length(taIndExpert)] - [0; taDividersExpert])/2;
yticks(ytpos)
yticklabels(unique(sortedTaExpert))
ylabel({'Persistently tuned neurons'; '(sorted by tuned angle; \circ)'})
title('Expert')
set(gca,'labelfontsizemultiplier',1.5, 'titlefontsizemultiplier',1.5)

sgtitle(sprintf('Population activity - Volume #%d',vi))


%% Showing after decorrelation
decHmNaive = zeros(size(heatmapNaive));
decHmExpert = zeros(size(heatmapExpert));
for ai = 1 : length(angles)
    angleInd = find(sortedAnglesNaive == angles(ai));
    for ci = 1 : size(heatmapNaive,1)
        decHmNaive(ci,angleInd) = heatmapNaive(ci,angleInd(randperm(length(angleInd))));
    end
    
    angleInd = find(sortedAnglesExpert == angles(ai));
    for ci = 1 : size(heatmapExpert,1)
        decHmExpert(ci,angleInd) = heatmapExpert(ci,angleInd(randperm(length(angleInd))));
    end
end

%
figure('unit', 'inch', 'pos', [1 1 13 5]),
subplot(121), imagesc(decHmNaive, [0 maxVal]), colorbar, hold on

for di = 1 : length(angleDividersNaive)
    plot([angleDividersNaive(di), angleDividersNaive(di)], [0 length(taIndNaive)+1], 'w--')
end
for di = 1 : length(taDividersNaive)
    plot([0 length(angleIndNaive)+1], [taDividersNaive(di), taDividersNaive(di)], 'w--')
end
xtpos = [0; angleDividersNaive] + ([angleDividersNaive; length(angleIndNaive)] - [0; angleDividersNaive] )/2;
xticks(xtpos)
xticklabels(unique(sortedAnglesNaive))
xlabel('Trials (sorted by object angle; \circ)')

ytpos = [0; taDividersNaive] + ([taDividersNaive; length(taIndNaive)] - [0; taDividersNaive])/2;
yticks(ytpos)
yticklabels(unique(sortedTaNaive))
ylabel({'Persistently tuned neurons'; '(sorted by tuned angle; \circ)'})
title('Naive')
set(gca,'labelfontsizemultiplier',1.5, 'titlefontsizemultiplier',1.5)
%
subplot(122), imagesc(decHmExpert, [0 maxVal]), colorbar, hold on
for di = 1 : length(angleDividersExpert)    
    plot([angleDividersExpert(di), angleDividersExpert(di)], [0 length(taIndExpert)+1], 'w--')
end
for di = 1 : length(taDividersExpert)
    plot([0 length(angleIndExpert)+1], [taDividersExpert(di), taDividersExpert(di)], 'w--')
end

xtpos = [0; angleDividersExpert] + ([angleDividersExpert; length(angleIndExpert)] - [0; angleDividersExpert] )/2;
xticks(xtpos)
xticklabels(unique(sortedAnglesExpert))
xlabel('Trials (sorted by object angle; \circ)')

ytpos = [0; taDividersExpert] + ([taDividersExpert; length(taIndExpert)] - [0; taDividersExpert])/2;
yticks(ytpos)
yticklabels(unique(sortedTaExpert))
ylabel({'Persistently tuned neurons'; '(sorted by tuned angle; \circ)'})
title('Expert')
set(gca,'labelfontsizemultiplier',1.5, 'titlefontsizemultiplier',1.5)

sgtitle(sprintf('Decorrelated - Volume #%d',vi))
     


%% Showing shuffling
shuffHmNaive = zeros(size(heatmapNaive));
shuffHmExpert = zeros(size(heatmapExpert));
for ci = 1 : size(heatmapNaive,1)
    shuffHmNaive(ci,:) = heatmapNaive(ci,randperm(size(heatmapNaive,2)));
end
for ci = 1 : size(heatmapExpert,1)
    shuffHmExpert(ci,:) = heatmapExpert(ci,randperm(size(heatmapExpert,2)));
end

%
figure('unit', 'inch', 'pos', [1 1 13 5]),
subplot(121), imagesc(shuffHmNaive, [0 maxVal]), colorbar, hold on

for di = 1 : length(angleDividersNaive)
    plot([angleDividersNaive(di), angleDividersNaive(di)], [0 length(taIndNaive)+1], 'w--')
end
for di = 1 : length(taDividersNaive)
    plot([0 length(angleIndNaive)+1], [taDividersNaive(di), taDividersNaive(di)], 'w--')
end
xtpos = [0; angleDividersNaive] + ([angleDividersNaive; length(angleIndNaive)] - [0; angleDividersNaive] )/2;
xticks(xtpos)
xticklabels(unique(sortedAnglesNaive))
xlabel('Trials (sorted by object angle; \circ)')

ytpos = [0; taDividersNaive] + ([taDividersNaive; length(taIndNaive)] - [0; taDividersNaive])/2;
yticks(ytpos)
yticklabels(unique(sortedTaNaive))
ylabel({'Persistently tuned neurons'; '(sorted by tuned angle; \circ)'})
title('Naive')
set(gca,'labelfontsizemultiplier',1.5, 'titlefontsizemultiplier',1.5)
%
subplot(122), imagesc(shuffHmExpert, [0 maxVal]), colorbar, hold on
for di = 1 : length(angleDividersExpert)    
    plot([angleDividersExpert(di), angleDividersExpert(di)], [0 length(taIndExpert)+1], 'w--')
end
for di = 1 : length(taDividersExpert)
    plot([0 length(angleIndExpert)+1], [taDividersExpert(di), taDividersExpert(di)], 'w--')
end

xtpos = [0; angleDividersExpert] + ([angleDividersExpert; length(angleIndExpert)] - [0; angleDividersExpert] )/2;
xticks(xtpos)
xticklabels(unique(sortedAnglesExpert))
xlabel('Trials (sorted by object angle; \circ)')

ytpos = [0; taDividersExpert] + ([taDividersExpert; length(taIndExpert)] - [0; taDividersExpert])/2;
yticks(ytpos)
yticklabels(unique(sortedTaExpert))
ylabel({'Persistently tuned neurons'; '(sorted by tuned angle; \circ)'})
title('Expert')
set(gca,'labelfontsizemultiplier',1.5, 'titlefontsizemultiplier',1.5)

sgtitle(sprintf('Shuffled - Volume #%d',vi))

%% Tuning curve
ci = 1;
angleIndCell = cell(1,length(angles));
for ai = 1 : length(angles)
    angleIndCell{ai} = find(sortedAnglesNaive == angles(ai));
end
tuningCurve = cellfun(@(x) mean(heatmapNaive(ci,x)), angleIndCell);

figure
plot(angles, tuningCurve, 'k-'), hold on
for ai = 1 : length(angles)
    plot(ones(1,length(angleIndCell{ai}))*angles(ai), heatmapNaive(ci,angleIndCell{ai}), 'o', 'color', [0.6 0.6 0.6])
end
xticks(angles)
xlabel('Object angle (\circ)')
ylabel('Response (pole up before answer)')
title('Angle tuning curve')
set(gca,'labelfontsizemultiplier',1.5, 'titlefontsizemultiplier',1.5)



