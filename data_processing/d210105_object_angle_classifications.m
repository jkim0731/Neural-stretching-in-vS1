% Object angle classifiers using sensory inputs and population activities
% After dividing into 11 imaging volumes
% Re-iterating parts of previous code.

% First, start with the answer lick.
% Then, also look at the first lick.



%%
%%
%%
%% Before the answer lick
%%
%%
%%

%% Basic settings
clear
baseDir = 'C:\Users\shires\Dropbox\Works\Projects\2020 Neural stretching in S1\Data\';
matchedPR = load([baseDir, 'matchedPopResponse_201230'], 'naive', 'expert');
numVol = 11;
load([baseDir, 'objectAnglePrediction_answerLick'], 'Ypairs', 'objectAngleModel'); % This data is made from d200826_neural_stretching_confirm_and_control


%%
angles = 45:15:135;
numSamplesBS = 100;
nIterBS = 100;
nShuffle = 100;
correctRate = nan(12,2);
chanceCR = nan(12,2);
errorAngle = nan(12,2);
chanceEA = nan(12,2);
confMat = cell(12,2);
for gi = 1 : 12
    if gi ~= 3
        for si = 1 : 2
            tempPair = Ypairs{gi,si};
            tempCR = zeros(nIterBS, 1);
            tempChanceCR = zeros(nIterBS,1);
            tempEA = zeros(nIterBS, 1);
            tempChanceEA = zeros(nIterBS,1);
            tempConfMat = zeros(length(angles), length(angles), nIterBS);
            angleInds = cell(length(angles),1);
            for ai = 1 : length(angles)
                angleInds{ai} = find(tempPair(:,1)==angles(ai));
            end
            for ii = 1 : nIterBS
                tempIterPair = zeros(numSamplesBS * length(angles), 2);
                for ai = 1 : length(angles)
                    % bootstrapping
                    tempInds = randi(length(angleInds{ai}),[numSamplesBS,1]);
                    inds = angleInds{ai}(tempInds);
                    tempIterPair( (ai-1)*numSamplesBS+1:ai*numSamplesBS, : ) = tempPair(inds,:);
                end
                tempCR(ii) = length(find(tempIterPair(:,2) - tempIterPair(:,1)==0)) / (numSamplesBS * length(angles));
                tempEA(ii) = mean(abs(tempIterPair(:,2) - tempIterPair(:,1)));

                tempTempCR = zeros(nShuffle,1);
                tempTempEA = zeros(nShuffle,1);
                for shuffi = 1 : nShuffle
                    shuffledPair = [tempIterPair(:,1), tempIterPair(randperm(size(tempIterPair,1)),2)];
                    tempTempCR(shuffi) = length(find(shuffledPair(:,2) - shuffledPair(:,1)==0)) / (numSamplesBS * length(angles));
                    tempTempEA(shuffi) = mean(abs(shuffledPair(:,2) - shuffledPair(:,1)));
                end
                tempChanceCR(ii) = mean(tempTempCR);
                tempChanceEA(ii) = mean(tempTempEA);

                tempConfMat(:,:,ii) = confusionmat(tempIterPair(:,1), tempIterPair(:,2))/numSamplesBS;
            end
            correctRate(gi,si) = mean(tempCR);
            chanceCR(gi,si) = mean(tempChanceCR);
            errorAngle(gi,si) = mean(tempEA);
            chanceEA(gi,si) = mean(tempChanceEA);
            confMat{gi,si} = mean(tempConfMat,3);
        end
    end
end

%% Contingency tables - Naive
contMatNaive = nan(7,7,12);
for i = 1 : 12
    if i ~=3
        contMatNaive(:,:,i) = confMat{i,1};
    end
end

figure, imagesc(nanmean(contMatNaive,3),[0 0.85]), axis square, colorbar
yticklabels(angles)
xticklabels(angles)
ylabel('Data')
xlabel('Prediction')
title('Naive')

%% Contingency tables - Expert
contMatExpert = nan(7,7,12);
for i = 1 : 12
    if i ~=3
        contMatExpert(:,:,i) = confMat{i,2};
    end
end

figure, imagesc(nanmean(contMatExpert,3),[0 0.85]), axis square, colorbar
yticklabels(angles)
xticklabels(angles)
ylabel('Data')
xlabel('Prediction')
title('Expert')

%% Classification performance - correct rate

figure('units','norm','pos',[0.2 0.2 0.1 0.3]), hold on,
for i = 1 : 12
    plot(correctRate(i,:), 'ko-')
end
errorbar([1,2],nanmean(correctRate), sem(correctRate), 'ro', 'lines', 'no')
errorbar([1,2],nanmean(chanceCR), sem(chanceCR), 'o', 'lines', 'no', 'color', [0.6 0.6 0.6])

[~,p,m] = paired_test(diff(correctRate,1,2));

title(sprintf('p = %.3f; method = %s', p, m))
xlim([0.5 2.5]), xticks([1,2]), xticklabels({'Naive', 'Expert'})
ylim([0 1])
yticks([0:0.1:1])
ylabel('Correct rate')
set(gca,'fontsize',12, 'fontname', 'Arial')


%% For text
nanmean(correctRate)
sem(correctRate)

nanmean(chanceCR)
sem(chanceCR)


%% Classification performance - prediction error

figure('units','norm','pos',[0.2 0.2 0.1 0.3]), hold on,
for i = 1 : 12
    plot(errorAngle(i,:), 'ko-')
end
errorbar([1,2],nanmean(errorAngle), sem(errorAngle), 'ro', 'lines', 'no')
errorbar([1,2],nanmean(chanceEA), sem(chanceEA), 'o', 'lines', 'no', 'color', [0.6 0.6 0.6])

[~,p,m] = paired_test(diff(errorAngle,1,2));

title(sprintf('p = %.3f; method = %s', p, m))
xlim([0.5 2.5]), xticks([1,2]), xticklabels({'Naive', 'Expert'})
ylim([0 38])
yticks([0:5:35])
ylabel('Prediction error (\circ)')
set(gca,'fontsize',12, 'fontname', 'Arial')


%% for text
mean(errorAngle)
sem(errorAngle)

mean(chanceEA)
sem(chanceEA)






%%
%% First, retrieve data and save them
%%
% # whisks, # touches, touch duration,
% 12 sensory inputs

mice = [25,27,30,36,39,52];
sessions = {[4,19],[3,10],[3,21],[1,17],[1,23],[3,21]};
numMice = length(mice);

saveFn = 'rawWhiskerFeatures_answerLick.mat';
% This is the data to be saved. 3rd row will be removed (because it will be
% empty)
rawFeat = cell(12,2); % (:,1) naive, (:,2) expert. 
% In each cell, (:,1:12) 12 sensory features, (:,13) num whisk, (:,14) whisking amplitude, (:,15) object angle
% num Touch at (:,6), touch duration at (:,11)
for mi = 1 : numMice
    fprintf('Processing mouse #%d/%d.\n', mi, numMice)
    mouse = mice(mi);
    % naive
    fprintf('Naive session.\n')
    session = sessions{mi}(1);
    ufn = sprintf('UberJK%03dS%02d_NC',mouse, session);
    load(sprintf('%s%s',baseDir, ufn), 'u')
    
    answerLickTimeAll = cellfun(@(x) min([x.answerLickTime, x.poleDownOnsetTime]), u.trials, 'un', 0);
    touchTrialInd = find(cellfun(@(x) length(x.protractionTouchChunksByWhisking), u.trials));
    touchAnswerTrialInd = find(cellfun(@(x,y) x.whiskerTime(x.protractionTouchChunksByWhisking{1}(1)) < y, u.trials(touchTrialInd), answerLickTimeAll(touchTrialInd)));
    trialInd = touchTrialInd(touchAnswerTrialInd);
    upperInd = trialInd( find(cellfun(@(x) ismember(1, x.planes), u.trials(trialInd))) );
    lowerInd = trialInd( find(cellfun(@(x) ismember(5, x.planes), u.trials(trialInd))) );
    % upper
    if mi ~= 2 % disregard JK027 upper volume due to low # of trials in the expert session
        utrials = u.trials(upperInd);
        answerLickTime = answerLickTimeAll(upperInd);
        
        numTouchCell = cellfun(@(x,y) find(x.whiskerTime(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking)) < y), utrials, answerLickTime, 'uniformoutput', false);
        
        rawData = zeros(length(upperInd), 15); % 12 whisker inputs + num whisk, whisk amplitude, object angle
        rawData(:,1) = cellfun(@(x,y) mean(x.theta(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
        rawData(:,2) = cellfun(@(x,y) mean(x.phi(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
        rawData(:,3) = cellfun(@(x,y) mean(x.kappaH(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
        rawData(:,4) = cellfun(@(x,y) -mean(x.kappaV(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell); % inverse the sign
        rawData(:,5) = cellfun(@(x,y) mean(x.arcLength(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
        rawData(:,6) = cellfun(@length, numTouchCell);
        
        rawData(:,7) = cellfun(@(x,y) mean(x.protractionTouchDThetaByWhisking(y)), utrials, numTouchCell);
        rawData(:,8) = cellfun(@(x,y) mean(x.protractionTouchDPhiByWhisking(y)), utrials, numTouchCell);
        rawData(:,9) = cellfun(@(x,y) -mean(x.protractionTouchDKappaHByWhisking(y)), utrials, numTouchCell); % inverse the sign
        rawData(:,10) = cellfun(@(x,y) mean(x.protractionTouchDKappaVByWhisking(y)), utrials, numTouchCell);
        rawData(:,11) = cellfun(@(x,y) mean(x.protractionTouchDurationByWhisking(y)), utrials, numTouchCell);
        rawData(:,12) = cellfun(@(x,y) mean(x.protractionTouchSlideDistanceByWhisking(y)), utrials, numTouchCell);
        
        for ti = 1 : length(upperInd)
            tempT = utrials{ti};
            poleUpInd = find(tempT.whiskerTime > tempT.poleUpTime(1),1,'first');
            answerInd = find(tempT.whiskerTime < answerLickTime{ti},1,'last');
            theta = tempT.theta(poleUpInd:answerInd);
            [onsetFrame, ~, ~, whiskingAmp, ~] = jkWhiskerOnsetNAmplitude(theta);
            rawData(ti,13) = length(onsetFrame);
            rawData(ti,14) = mean(whiskingAmp(find(whiskingAmp>2.5)));
        end
        
        rawData(:,15) = cellfun(@(x) x.angle, utrials);
        rawFeat{(mi-1)*2+1,1} = rawData;
    end
    
    % lower
    utrials = u.trials(lowerInd);
    answerLickTime = answerLickTimeAll(lowerInd);

    numTouchCell = cellfun(@(x,y) find(x.whiskerTime(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking)) < y), utrials, answerLickTime, 'uniformoutput', false);

    rawData = zeros(length(lowerInd), 15); % 12 whisker inputs + num whisk, whisk amplitude, object angle
    rawData(:,1) = cellfun(@(x,y) mean(x.theta(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
    rawData(:,2) = cellfun(@(x,y) mean(x.phi(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
    rawData(:,3) = cellfun(@(x,y) mean(x.kappaH(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
    rawData(:,4) = cellfun(@(x,y) -mean(x.kappaV(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell); % inverse the sign
    rawData(:,5) = cellfun(@(x,y) mean(x.arcLength(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
    rawData(:,6) = cellfun(@length, numTouchCell);

    rawData(:,7) = cellfun(@(x,y) mean(x.protractionTouchDThetaByWhisking(y)), utrials, numTouchCell);
    rawData(:,8) = cellfun(@(x,y) mean(x.protractionTouchDPhiByWhisking(y)), utrials, numTouchCell);
    rawData(:,9) = cellfun(@(x,y) -mean(x.protractionTouchDKappaHByWhisking(y)), utrials, numTouchCell); % inverse the sign
    rawData(:,10) = cellfun(@(x,y) mean(x.protractionTouchDKappaVByWhisking(y)), utrials, numTouchCell);
    rawData(:,11) = cellfun(@(x,y) mean(x.protractionTouchDurationByWhisking(y)), utrials, numTouchCell);
    rawData(:,12) = cellfun(@(x,y) mean(x.protractionTouchSlideDistanceByWhisking(y)), utrials, numTouchCell);

    for ti = 1 : length(lowerInd)
        tempT = utrials{ti};
        poleUpInd = find(tempT.whiskerTime > tempT.poleUpTime(1),1,'first');
        answerInd = find(tempT.whiskerTime < answerLickTime{ti},1,'last');
        theta = tempT.theta(poleUpInd:answerInd);
        [onsetFrame, ~, ~, whiskingAmp, ~] = jkWhiskerOnsetNAmplitude(theta);
        rawData(ti,13) = length(onsetFrame);
        rawData(ti,14) = mean(whiskingAmp(find(whiskingAmp>2.5)));
    end

    rawData(:,15) = cellfun(@(x) x.angle, utrials);
    rawFeat{mi*2,1} = rawData;
    
        
    % expert
    fprintf('Expert session.\n')
    session = sessions{mi}(2);
    ufn = sprintf('UberJK%03dS%02d_NC',mouse, session);
    load(sprintf('%s%s',baseDir, ufn), 'u')
    answerLickTimeAll = cellfun(@(x) min([x.answerLickTime, x.poleDownOnsetTime]), u.trials, 'un', 0);
    touchTrialInd = find(cellfun(@(x) length(x.protractionTouchChunksByWhisking), u.trials));
    touchAnswerTrialInd = find(cellfun(@(x,y) x.whiskerTime(x.protractionTouchChunksByWhisking{1}(1)) < y, u.trials(touchTrialInd), answerLickTimeAll(touchTrialInd)));
    trialInd = touchTrialInd(touchAnswerTrialInd);
    upperInd = trialInd( find(cellfun(@(x) ismember(1, x.planes), u.trials(trialInd))) );
    lowerInd = trialInd( find(cellfun(@(x) ismember(5, x.planes), u.trials(trialInd))) );
    % upper
    if mi ~= 2 % disregard JK027 upper volume due to low # of trials in the expert session
        utrials = u.trials(upperInd);
        answerLickTime = answerLickTimeAll(upperInd);
        
        numTouchCell = cellfun(@(x,y) find(x.whiskerTime(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking)) < y), utrials, answerLickTime, 'uniformoutput', false);
        
        rawData = zeros(length(upperInd), 15); % 12 whisker inputs + num whisk, whisk amplitude, object angle
        rawData(:,1) = cellfun(@(x,y) mean(x.theta(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
        rawData(:,2) = cellfun(@(x,y) mean(x.phi(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
        rawData(:,3) = cellfun(@(x,y) mean(x.kappaH(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
        rawData(:,4) = cellfun(@(x,y) -mean(x.kappaV(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell); % inverse the sign
        rawData(:,5) = cellfun(@(x,y) mean(x.arcLength(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
        rawData(:,6) = cellfun(@length, numTouchCell);
        
        rawData(:,7) = cellfun(@(x,y) mean(x.protractionTouchDThetaByWhisking(y)), utrials, numTouchCell);
        rawData(:,8) = cellfun(@(x,y) mean(x.protractionTouchDPhiByWhisking(y)), utrials, numTouchCell);
        rawData(:,9) = cellfun(@(x,y) -mean(x.protractionTouchDKappaHByWhisking(y)), utrials, numTouchCell); % inverse the sign
        rawData(:,10) = cellfun(@(x,y) mean(x.protractionTouchDKappaVByWhisking(y)), utrials, numTouchCell);
        rawData(:,11) = cellfun(@(x,y) mean(x.protractionTouchDurationByWhisking(y)), utrials, numTouchCell);
        rawData(:,12) = cellfun(@(x,y) mean(x.protractionTouchSlideDistanceByWhisking(y)), utrials, numTouchCell);
        
        for ti = 1 : length(upperInd)
            tempT = utrials{ti};
            poleUpInd = find(tempT.whiskerTime > tempT.poleUpTime(1),1,'first');
            answerInd = find(tempT.whiskerTime < answerLickTime{ti},1,'last');
            theta = tempT.theta(poleUpInd:answerInd);
            [onsetFrame, ~, ~, whiskingAmp, ~] = jkWhiskerOnsetNAmplitude(theta);
            rawData(ti,13) = length(onsetFrame);
            rawData(ti,14) = mean(whiskingAmp(find(whiskingAmp>2.5)));
        end
        
        rawData(:,15) = cellfun(@(x) x.angle, utrials);
        rawFeat{(mi-1)*2+1,2} = rawData;
    end
    
    % lower
    utrials = u.trials(lowerInd);
    answerLickTime = answerLickTimeAll(lowerInd);

    numTouchCell = cellfun(@(x,y) find(x.whiskerTime(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking)) < y), utrials, answerLickTime, 'uniformoutput', false);

    rawData = zeros(length(lowerInd), 15); % 12 whisker inputs + num whisk, whisk amplitude, object angle
    rawData(:,1) = cellfun(@(x,y) mean(x.theta(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
    rawData(:,2) = cellfun(@(x,y) mean(x.phi(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
    rawData(:,3) = cellfun(@(x,y) mean(x.kappaH(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
    rawData(:,4) = cellfun(@(x,y) -mean(x.kappaV(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell); % inverse the sign
    rawData(:,5) = cellfun(@(x,y) mean(x.arcLength(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
    rawData(:,6) = cellfun(@length, numTouchCell);

    rawData(:,7) = cellfun(@(x,y) mean(x.protractionTouchDThetaByWhisking(y)), utrials, numTouchCell);
    rawData(:,8) = cellfun(@(x,y) mean(x.protractionTouchDPhiByWhisking(y)), utrials, numTouchCell);
    rawData(:,9) = cellfun(@(x,y) -mean(x.protractionTouchDKappaHByWhisking(y)), utrials, numTouchCell); % inverse the sign
    rawData(:,10) = cellfun(@(x,y) mean(x.protractionTouchDKappaVByWhisking(y)), utrials, numTouchCell);
    rawData(:,11) = cellfun(@(x,y) mean(x.protractionTouchDurationByWhisking(y)), utrials, numTouchCell);
    rawData(:,12) = cellfun(@(x,y) mean(x.protractionTouchSlideDistanceByWhisking(y)), utrials, numTouchCell);

    for ti = 1 : length(lowerInd)
        tempT = utrials{ti};
        poleUpInd = find(tempT.whiskerTime > tempT.poleUpTime(1),1,'first');
        answerInd = find(tempT.whiskerTime < answerLickTime{ti},1,'last');
        theta = tempT.theta(poleUpInd:answerInd);
        [onsetFrame, ~, ~, whiskingAmp, ~] = jkWhiskerOnsetNAmplitude(theta);
        rawData(ti,13) = length(onsetFrame);
        rawData(ti,14) = mean(whiskingAmp(find(whiskingAmp>2.5)));
    end

    rawData(:,15) = cellfun(@(x) x.angle, utrials);
    rawFeat{mi*2,2} = rawData;
end
rawFeat(3,:) = [];

save(saveFn,'rawFeat')




%%
%% Look at more basic whisking parameters
%%
% # of whisks, whisking amplitude, # of touches, mean touch duration

if ~exist('rawFeat', 'var')
    load('rawWhiskerFeatures_answerLick')
end
numVol = size(rawFeat,1);

figure,
subplot(141), hold on
tempMat = cellfun(@(x) mean(x(:,13)), rawFeat);
for i = 1 : numVol
    plot(tempMat(i,:), 'ko-')
end
errorbar(mean(tempMat), sem(tempMat), 'ro')
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
[~,p,m] = paired_test(tempMat(:,1), tempMat(:,2));
ylabel('Mean number of whisks')
title(sprintf('p = %.3f, m = %s', p, m))

subplot(142), hold on
tempMat = cellfun(@(x) nanmean(x(:,14)), rawFeat);
for i = 1 : numVol
    plot(tempMat(i,:), 'ko-')
end
errorbar(mean(tempMat), sem(tempMat), 'ro')
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
[~,p,m] = paired_test(tempMat(:,1), tempMat(:,2));
ylabel('Mean whisking amplitude (\circ)')
title(sprintf('p = %.3f, m = %s', p, m))

subplot(143), hold on
tempMat = cellfun(@(x) mean(x(:,6)), rawFeat);
for i = 1 : numVol
    plot(tempMat(i,:), 'ko-')
end
errorbar(mean(tempMat), sem(tempMat), 'ro')
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
[~,p,m] = paired_test(tempMat(:,1), tempMat(:,2));
ylabel('Mean number of touches')
title(sprintf('p = %.3f, m = %s', p, m))

subplot(144), hold on
tempMat = cellfun(@(x) mean(x(:,11)), rawFeat)*1000;
for i = 1 : numVol
    plot(tempMat(i,:), 'ko-')
end
errorbar(mean(tempMat), sem(tempMat), 'ro')
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
[~,p,m] = paired_test(tempMat(:,1), tempMat(:,2));
ylabel('Mean touch duration (ms)')
title(sprintf('p = %.3f, m = %s', p, m))
%%
%% Look at individual sensory inputs (trial-averaged) across angles
%%
% Similar to Fig 2C from Kim 2020 Neuron, but by 11 volumes (not 6 mice)

%% First, raw values
colorsTransient = [248 171 66; 40 170 225] / 255;
subplotPos = [1,2,5,6,9,10,3,4,7,8,11,12];
featNames = {'Azimuthal angle', 'Vertical angle', 'Horizontal curvature', 'Vertical curvature', 'Arc length', 'Touch count', ...
    'Push angle', 'Vertical displacement', 'Horizontal bending', 'Vertical bending', 'Touch duration', 'Slide distance'};
angles = 45:15:135;
figure('unit', 'inch', 'pos', [0 0 13 7])
for spi = 1 : 12
    subplot(3,4,subplotPos(spi)), hold on
    tempNaive = zeros(numVol, length(angles));
    tempExpert = zeros(numVol, length(angles));
    % naive
    for vi = 1 : numVol
        for ai = 1 : length(angles)
            tInd = find(rawFeat{vi,1}(:,15) == angles(ai));
            tempNaive(vi,ai) = nanmean(rawFeat{vi,1}(tInd,spi));
        end
    end
    % expert
    for vi = 1 : numVol
        for ai = 1 : length(angles)
            tInd = find(rawFeat{vi,2}(:,15) == angles(ai));
            tempExpert(vi,ai) = nanmean(rawFeat{vi,2}(tInd,spi));
        end
    end
    
    plot(angles, mean(tempNaive), 'color', colorsTransient(1,:))
    plot(angles, mean(tempExpert), 'color', colorsTransient(2,:))
    if spi == 8
        legend({'Naive', 'Expert'}, 'autoupdate', 'off', 'location', 'northwest')
    end
    
    boundedline(angles, mean(tempNaive), sem(tempNaive), 'cmap', colorsTransient(1,:))
    boundedline(angles, mean(tempExpert), sem(tempExpert), 'cmap', colorsTransient(2,:))
    xlim([40 140]), xticks(angles)
    if subplotPos(spi) < 9
        xticklabels('')
    end
    title(featNames{spi})
    
end

sgtitle('Raw values')

%% (2) across-session standardization 

figure('unit', 'inch', 'pos', [0 0 13 7])
for spi = 1 : 12
    subplot(3,4,subplotPos(spi)), hold on
    stdNaive = zeros(numVol, length(angles));
    stdExpert = zeros(numVol, length(angles));
    % naive
    for vi = 1 : numVol
        tempAll = [rawFeat{vi,1}(:,spi); rawFeat{vi,2}(:,spi)];
        tempStd = (tempAll - nanmean(tempAll)) / nanstd(tempAll);
        tempNaive = tempStd(1:size(rawFeat{vi,1},1));
        tempExpert = tempStd(size(rawFeat{vi,1},1)+1:end);
        for ai = 1 : length(angles)
            % naive
            tInd = find(rawFeat{vi,1}(:,15) == angles(ai));
            stdNaive(vi,ai) = nanmean(tempNaive(tInd));
            % expert
            tInd = find(rawFeat{vi,2}(:,15) == angles(ai));
            stdExpert(vi,ai) = nanmean(tempExpert(tInd));
        end
    end
    
    plot(angles, mean(stdNaive), 'color', colorsTransient(1,:))
    plot(angles, mean(stdExpert), 'color', colorsTransient(2,:))
    if spi == 8
        legend({'Naive', 'Expert'}, 'autoupdate', 'off', 'location', 'northwest')
    end
    
    boundedline(angles, mean(stdNaive), sem(stdNaive), 'cmap', colorsTransient(1,:))
    boundedline(angles, mean(stdExpert), sem(stdExpert), 'cmap', colorsTransient(2,:))
    xlim([40 140]), xticks(angles)
    if subplotPos(spi) < 9
        xticklabels('')
    end
    title(featNames{spi})
    
end

sgtitle('Across-session standardization')



%% (3) within-session standardization
figure('unit', 'inch', 'pos', [0 0 13 7])
for spi = 1 : 12
    subplot(3,4,subplotPos(spi)), hold on
    stdNaive = zeros(numVol, length(angles));
    stdExpert = zeros(numVol, length(angles));
    % naive
    for vi = 1 : numVol
        tempAll = [rawFeat{vi,1}(:,spi); rawFeat{vi,2}(:,spi)];
        tempStd = (tempAll - nanmean(tempAll)) / nanstd(tempAll);
        tempNaive = rawFeat{vi,1}(:,spi);
        tempNaive = (tempNaive - nanmean(tempNaive)) / nanstd(tempNaive);
        tempExpert = rawFeat{vi,2}(:,spi);
        tempExpert = (tempExpert - nanmean(tempExpert)) / nanstd(tempExpert);
        for ai = 1 : length(angles)
            % naive
            tInd = find(rawFeat{vi,1}(:,15) == angles(ai));
            stdNaive(vi,ai) = nanmean(tempNaive(tInd));
            % expert
            tInd = find(rawFeat{vi,2}(:,15) == angles(ai));
            stdExpert(vi,ai) = nanmean(tempExpert(tInd));
        end
    end
    
    plot(angles, mean(stdNaive), 'color', colorsTransient(1,:))
    plot(angles, mean(stdExpert), 'color', colorsTransient(2,:))
    if spi == 8
        legend({'Naive', 'Expert'}, 'autoupdate', 'off', 'location', 'northwest')
    end
    
    boundedline(angles, mean(stdNaive), sem(stdNaive), 'cmap', colorsTransient(1,:))
    boundedline(angles, mean(stdExpert), sem(stdExpert), 'cmap', colorsTransient(2,:))
    xlim([40 140]), xticks(angles)
    if subplotPos(spi) < 9
        xticklabels('')
    end
    ylim([-1.5 1.5])
    title(featNames{spi})
    
end

sgtitle('Within-session standardization')









%%
%% Other classifiers using sensory inputs
%%
% LDA, SVM, KNN, Random forest
if ~exist('rawFeat', 'var')
    load('rawWhiskerFeatures_answerLick')
end
numVol = size(rawFeat,1);
stdX = cellfun(@(x) (x(:,1:12) - nanmean(x(:,1:12)))./nanstd(x(:,1:12)), rawFeat, 'un', 0);
Y = cellfun(@(x) x(:,15), rawFeat, 'un', 0);
%% LDA

correctRate = zeros(numVol,2); %(:,1) naive, (:,2) expert
errorAngle = zeros(numVol,2); 
shuffleCR = zeros(numVol,2); 
shuffleEA = zeros(numVol,2); 
for vi = 1 : numVol
    for si = 1 : 2
        % 5-fold cross-validation
        numFold = 5;
        foldInds = cell(numFold,1);
        for fi = 1 : numFold
            foldInds{fi} = [];
        end
        tempX = stdX{vi,si};
        tempY = Y{vi,si};

        % stratification
        for ai = 1 : length(angles)
            aInd = find(tempY == angles(ai));
            randInd = aInd(randperm(length(aInd)));
            inds = round(linspace(0, length(randInd), numFold+1));
            for fi = 1 : length(inds)-1
                foldInds{fi} = [foldInds{fi}; randInd(inds(fi)+1:inds(fi+1))];
            end
        end

        foldCR = zeros(numFold,1);
        foldEA = zeros(numFold,1);
        shFoldCR = zeros(numFold,1);
        shFoldEA = zeros(numFold,1);
        numShuffle = 100;
        for fi = 1 : numFold
            trainFi = setdiff(1:numFold, fi); % training fold index
            trainInd = cell2mat(foldInds(trainFi));
            testInd = sort(foldInds{fi});

            trainX = tempX(trainInd,:);
            trainY = tempY(trainInd,:);

            testX = tempX(testInd,:);
            testY = tempY(testInd,:);

            mdl = fitcdiscr(trainX, trainY, 'Prior', 'uniform');
            label = predict(mdl, testX);
            foldCR(fi) = length(find((label-testY)==0)) / length(testY);
            foldEA(fi) = mean(abs(label-testY));

            % Shuffling
            tempPerf = zeros(numShuffle,1);
            tempAngle = zeros(numShuffle,1);
            for shi = 1 : numShuffle
                tempShY = testY(randperm(length(testY)));
                tempPerf(shi) = length(find((label-tempShY)==0)) / length(tempShY);
                tempAngle(shi) = mean(abs(label-tempShY));
            end
            shFoldCR(fi) = mean(tempPerf);
            shFoldEA(fi) = mean(tempAngle);
        end
        
        correctRate(vi,si) = mean(foldCR);
        errorAngle(vi,si) = mean(foldEA);
        shuffleCR(vi,si) = mean(shFoldCR);
        shuffleEA(vi,si) = mean(shFoldEA);
    end
end


figure, 
subplot(121), hold on
for vi = 1 : numVol
    plot(correctRate(vi,:), 'ko-')
end
errorbar(mean(correctRate), sem(correctRate), 'ro')
errorbar(mean(shuffleCR), sem(shuffleCR), 'o', 'color', [0.6 0.6 0.6])
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
ylabel('Correct Rate')
[~,p,m] = paired_test(correctRate(:,1), correctRate(:,2));
title(sprintf('p = %.3f (%s)', p, m))

subplot(122), hold on
for vi = 1 : numVol
    plot(errorAngle(vi,:), 'ko-')
end
errorbar(mean(errorAngle), sem(errorAngle), 'ro')
errorbar(mean(shuffleEA), sem(shuffleEA), 'o', 'color', [0.6 0.6 0.6])
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
ylabel('Error angle (\circ)')
[~,p,m] = paired_test(errorAngle(:,1), errorAngle(:,2));
title(sprintf('p = %.3f (%s)', p, m))


%% SVM

correctRate_LDA = zeros(numVol,2); %(:,1) naive, (:,2) expert
errorAngle_LDA = zeros(numVol,2);
shuffleCR_LDA = zeros(numVol,2);
shuffleEA_LDA = zeros(numVol,2);

correctRate_SVM = zeros(numVol,2); %(:,1) naive, (:,2) expert
errorAngle_SVM = zeros(numVol,2);
shuffleCR_SVM = zeros(numVol,2);
shuffleEA_SVM = zeros(numVol,2);

correctRate_KNN = zeros(numVol,2); %(:,1) naive, (:,2) expert
errorAngle_KNN = zeros(numVol,2);
shuffleCR_KNN = zeros(numVol,2);
shuffleEA_KNN = zeros(numVol,2);

for vi = 1 : numVol
    fprintf('Processing volume #%d/%d\n', vi, numVol)
    for si = 1 : 2
        if si == 1
            disp('Naive')
        else
            disp('Expert')
        end
        % 5-fold cross-validation
        numFold = 5;
        foldInds = cell(numFold,1);
        for fi = 1 : numFold
            foldInds{fi} = [];
        end
        tempX = stdX{vi,si};
        tempY = Y{vi,si};

        % stratification
        for ai = 1 : length(angles)
            aInd = find(tempY == angles(ai));
            randInd = aInd(randperm(length(aInd)));
            inds = round(linspace(0, length(randInd), numFold+1));
            for fi = 1 : length(inds)-1
                foldInds{fi} = [foldInds{fi}; randInd(inds(fi)+1:inds(fi+1))];
            end
        end

        % LDA
        disp('LDA')
        foldCR = zeros(numFold,1);
        foldEA = zeros(numFold,1);
        shFoldCR = zeros(numFold,1);
        shFoldEA = zeros(numFold,1);
        numShuffle = 100;
        for fi = 1 : numFold
            trainFi = setdiff(1:numFold, fi); % training fold index
            trainInd = cell2mat(foldInds(trainFi));
            testInd = sort(foldInds{fi});

            trainX = tempX(trainInd,:);
            trainY = tempY(trainInd,:);

            testX = tempX(testInd,:);
            testY = tempY(testInd,:);

            mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'discriminant', 'OptimizeHyperparameters', {'Delta', 'Gamma'}, 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
%             mdl = fitcdiscr(trainX, trainY, 'Prior', 'uniform');
            label = predict(mdl, testX);
            foldCR(fi) = length(find((label-testY)==0)) / length(testY);
            foldEA(fi) = mean(abs(label-testY));

            % Shuffling
            tempPerf = zeros(numShuffle,1);
            tempAngle = zeros(numShuffle,1);
            for shi = 1 : numShuffle
                tempShY = testY(randperm(length(testY)));
                tempPerf(shi) = length(find((label-tempShY)==0)) / length(tempShY);
                tempAngle(shi) = mean(abs(label-tempShY));
            end
            shFoldCR(fi) = mean(tempPerf);
            shFoldEA(fi) = mean(tempAngle);
        end
        
        correctRate_LDA(vi,si) = mean(foldCR);
        errorAngle_LDA(vi,si) = mean(foldEA);
        shuffleCR_LDA(vi,si) = mean(shFoldCR);
        shuffleEA_LDA(vi,si) = mean(shFoldEA);
        
        % SVM
        disp('SVM')
        foldCR = zeros(numFold,1);
        foldEA = zeros(numFold,1);
        shFoldCR = zeros(numFold,1);
        shFoldEA = zeros(numFold,1);
        numShuffle = 100;
        for fi = 1 : numFold
            trainFi = setdiff(1:numFold, fi); % training fold index
            trainInd = cell2mat(foldInds(trainFi));
            testInd = sort(foldInds{fi});

            trainX = tempX(trainInd,:);
            trainY = tempY(trainInd,:);

            testX = tempX(testInd,:);
            testY = tempY(testInd,:);

            mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'svm', 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
%             mdl = fitcdiscr(trainX, trainY, 'Prior', 'uniform');
            label = predict(mdl, testX);
            foldCR(fi) = length(find((label-testY)==0)) / length(testY);
            foldEA(fi) = mean(abs(label-testY));

            % Shuffling
            tempPerf = zeros(numShuffle,1);
            tempAngle = zeros(numShuffle,1);
            for shi = 1 : numShuffle
                tempShY = testY(randperm(length(testY)));
                tempPerf(shi) = length(find((label-tempShY)==0)) / length(tempShY);
                tempAngle(shi) = mean(abs(label-tempShY));
            end
            shFoldCR(fi) = mean(tempPerf);
            shFoldEA(fi) = mean(tempAngle);
        end
        
        correctRate_SVM(vi,si) = mean(foldCR);
        errorAngle_SVM(vi,si) = mean(foldEA);
        shuffleCR_SVM(vi,si) = mean(shFoldCR);
        shuffleEA_SVM(vi,si) = mean(shFoldEA);
        
        % KNN
        disp('KNN')
        foldCR = zeros(numFold,1);
        foldEA = zeros(numFold,1);
        shFoldCR = zeros(numFold,1);
        shFoldEA = zeros(numFold,1);
        numShuffle = 100;
        for fi = 1 : numFold
            trainFi = setdiff(1:numFold, fi); % training fold index
            trainInd = cell2mat(foldInds(trainFi));
            testInd = sort(foldInds{fi});

            trainX = tempX(trainInd,:);
            trainY = tempY(trainInd,:);

            testX = tempX(testInd,:);
            testY = tempY(testInd,:);

            mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'knn', 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
%             mdl = fitcdiscr(trainX, trainY, 'Prior', 'uniform');
            label = predict(mdl, testX);
            foldCR(fi) = length(find((label-testY)==0)) / length(testY);
            foldEA(fi) = mean(abs(label-testY));

            % Shuffling
            tempPerf = zeros(numShuffle,1);
            tempAngle = zeros(numShuffle,1);
            for shi = 1 : numShuffle
                tempShY = testY(randperm(length(testY)));
                tempPerf(shi) = length(find((label-tempShY)==0)) / length(tempShY);
                tempAngle(shi) = mean(abs(label-tempShY));
            end
            shFoldCR(fi) = mean(tempPerf);
            shFoldEA(fi) = mean(tempAngle);
        end
        
        correctRate_KNN(vi,si) = mean(foldCR);
        errorAngle_KNN(vi,si) = mean(foldEA);
        shuffleCR_KNN(vi,si) = mean(shFoldCR);
        shuffleEA_KNN(vi,si) = mean(shFoldEA);        
    end
end


figure, 
subplot(121), hold on
for vi = 1 : numVol
    plot(correctRate_LDA(vi,:), 'ko-')
end
errorbar(mean(correctRate_LDA), sem(correctRate_LDA), 'ro')
errorbar(mean(shuffleCR_LDA), sem(shuffleCR_LDA), 'o', 'color', [0.6 0.6 0.6])
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
ylabel('Correct Rate')
[~,p,m] = paired_test(correctRate_LDA(:,1), correctRate_LDA(:,2));
title(sprintf('p = %.3f (%s)', p, m))

subplot(122), hold on
for vi = 1 : numVol
    plot(errorAngle_LDA(vi,:), 'ko-')
end
errorbar(mean(errorAngle_LDA), sem(errorAngle_LDA), 'ro')
errorbar(mean(shuffleEA_LDA), sem(shuffleEA_LDA), 'o', 'color', [0.6 0.6 0.6])
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
ylabel('Error angle (\circ)')
[~,p,m] = paired_test(errorAngle_LDA(:,1), errorAngle_LDA(:,2));
title(sprintf('p = %.3f (%s)', p, m))
sgtitle('LDA')

figure, 
subplot(121), hold on
for vi = 1 : numVol
    plot(correctRate_SVM(vi,:), 'ko-')
end
errorbar(mean(correctRate_SVM), sem(correctRate_SVM), 'ro')
errorbar(mean(shuffleCR_SVM), sem(shuffleCR_SVM), 'o', 'color', [0.6 0.6 0.6])
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
ylabel('Correct Rate')
[~,p,m] = paired_test(correctRate_SVM(:,1), correctRate_SVM(:,2));
title(sprintf('p = %.3f (%s)', p, m))

subplot(122), hold on
for vi = 1 : numVol
    plot(errorAngle_SVM(vi,:), 'ko-')
end
errorbar(mean(errorAngle_SVM), sem(errorAngle_SVM), 'ro')
errorbar(mean(shuffleEA_SVM), sem(shuffleEA_SVM), 'o', 'color', [0.6 0.6 0.6])
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
ylabel('Error angle (\circ)')
[~,p,m] = paired_test(errorAngle_SVM(:,1), errorAngle_SVM(:,2));
title(sprintf('p = %.3f (%s)', p, m))
sgtitle('SVM')

figure, 
subplot(121), hold on
for vi = 1 : numVol
    plot(correctRate_KNN(vi,:), 'ko-')
end
errorbar(mean(correctRate_KNN), sem(correctRate_KNN), 'ro')
errorbar(mean(shuffleCR_KNN), sem(shuffleCR_KNN), 'o', 'color', [0.6 0.6 0.6])
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
ylabel('Correct Rate')
[~,p,m] = paired_test(correctRate_KNN(:,1), correctRate_KNN(:,2));
title(sprintf('p = %.3f (%s)', p, m))

subplot(122), hold on
for vi = 1 : numVol
    plot(errorAngle_KNN(vi,:), 'ko-')
end
errorbar(mean(errorAngle_KNN), sem(errorAngle_KNN), 'ro')
errorbar(mean(shuffleEA_KNN), sem(shuffleEA_KNN), 'o', 'color', [0.6 0.6 0.6])
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
ylabel('Error angle (\circ)')
[~,p,m] = paired_test(errorAngle_KNN(:,1), errorAngle_KNN(:,2));
title(sprintf('p = %.3f (%s)', p, m))
sgtitle('KNN')




%%
%% Instead of fold, have 70% train set, randomized 10 iterations
%%
% Due to low # of observations
if ~exist('rawFeat', 'var')
    load('rawWhiskerFeatures_answerLick')
end
numVol = size(rawFeat,1);
stdX = cellfun(@(x) (x(:,1:12) - nanmean(x(:,1:12)))./nanstd(x(:,1:12)), rawFeat, 'un', 0);
Y = cellfun(@(x) x(:,15), rawFeat, 'un', 0);

% 10 iteration, 70% train set
numIter = 100;
trainFrac = 0.7;
%% LDA

correctRate = zeros(numVol,2); %(:,1) naive, (:,2) expert
errorAngle = zeros(numVol,2); 
shuffleCR = zeros(numVol,2); 
shuffleEA = zeros(numVol,2); 
for vi = 1 : numVol
    for si = 1 : 2
        trainInds = cell(numIter,1);
        testInds = cell(numIter,1);
        for tri = 1 : numIter
            trainInds{tri} = [];
            testInds{tri} = [];
        end
        tempX = stdX{vi,si};
        tempY = Y{vi,si};

        % stratification
        for ai = 1 : length(angles)
            aInd = find(tempY == angles(ai));
            for tri = 1 : numIter
                randInd = aInd(randperm(length(aInd)));
                trainLength = round(length(aInd)*trainFrac);
                trainInds{tri} = [trainInds{tri}; randInd(1:trainLength)];
                testInds{tri} = [testInds{tri}; randInd(trainLength+1:end)];
            end
        end

        iterCR = zeros(numIter,1);
        iterEA = zeros(numIter,1);
        shIterCR = zeros(numIter,1);
        shIterEA = zeros(numIter,1);
        numShuffle = 100;
        for ii = 1 : numIter
            trainInd = cell2mat(trainInds(ii));
            testInd = sort(testInds{ii});

            trainX = tempX(trainInd,:);
            trainY = tempY(trainInd,:);

            testX = tempX(testInd,:);
            testY = tempY(testInd,:);

            mdl = fitcdiscr(trainX, trainY, 'Prior', 'uniform');
            label = predict(mdl, testX);
            iterCR(ii) = length(find((label-testY)==0)) / length(testY);
            iterEA(ii) = mean(abs(label-testY));

            % Shuffling
            tempPerf = zeros(numShuffle,1);
            tempAngle = zeros(numShuffle,1);
            for shi = 1 : numShuffle
                tempShY = testY(randperm(length(testY)));
                tempPerf(shi) = length(find((label-tempShY)==0)) / length(tempShY);
                tempAngle(shi) = mean(abs(label-tempShY));
            end
            shIterCR(ii) = mean(tempPerf);
            shIterEA(ii) = mean(tempAngle);
        end
        
        correctRate(vi,si) = mean(iterCR);
        errorAngle(vi,si) = mean(iterEA);
        shuffleCR(vi,si) = mean(shIterCR);
        shuffleEA(vi,si) = mean(shIterEA);
    end
end


figure, 
subplot(121), hold on
for vi = 1 : numVol
    plot(correctRate(vi,:), 'ko-')
end
errorbar(mean(correctRate), sem(correctRate), 'ro')
errorbar(mean(shuffleCR), sem(shuffleCR), 'o', 'color', [0.6 0.6 0.6])
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
ylabel('Correct Rate')
[~,p,m] = paired_test(correctRate(:,1), correctRate(:,2));
title(sprintf('p = %.3f (%s)', p, m))

subplot(122), hold on
for vi = 1 : numVol
    plot(errorAngle(vi,:), 'ko-')
end
errorbar(mean(errorAngle), sem(errorAngle), 'ro')
errorbar(mean(shuffleEA), sem(shuffleEA), 'o', 'color', [0.6 0.6 0.6])
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
ylabel('Error angle (\circ)')
[~,p,m] = paired_test(errorAngle(:,1), errorAngle(:,2));
title(sprintf('p = %.3f (%s)', p, m))


%% SVM

correctRate_LDA = zeros(numVol,2); %(:,1) naive, (:,2) expert
errorAngle_LDA = zeros(numVol,2);
shuffleCR_LDA = zeros(numVol,2);
shuffleEA_LDA = zeros(numVol,2);

correctRate_SVM = zeros(numVol,2); %(:,1) naive, (:,2) expert
errorAngle_SVM = zeros(numVol,2);
shuffleCR_SVM = zeros(numVol,2);
shuffleEA_SVM = zeros(numVol,2);

correctRate_KNN = zeros(numVol,2); %(:,1) naive, (:,2) expert
errorAngle_KNN = zeros(numVol,2);
shuffleCR_KNN = zeros(numVol,2);
shuffleEA_KNN = zeros(numVol,2);

for vi = 1 : numVol
    fprintf('Processing volume #%d/%d\n', vi, numVol)
    for si = 1 : 2
        if si == 1
            disp('Naive')
        else
            disp('Expert')
        end
        % 5-fold cross-validation
        numFold = 5;
        foldInds = cell(numFold,1);
        for fi = 1 : numFold
            foldInds{fi} = [];
        end
        tempX = stdX{vi,si};
        tempY = Y{vi,si};

        % stratification
        for ai = 1 : length(angles)
            aInd = find(tempY == angles(ai));
            randInd = aInd(randperm(length(aInd)));
            inds = round(linspace(0, length(randInd), numFold+1));
            for fi = 1 : length(inds)-1
                foldInds{fi} = [foldInds{fi}; randInd(inds(fi)+1:inds(fi+1))];
            end
        end

        % LDA
        disp('LDA')
        foldCR = zeros(numFold,1);
        foldEA = zeros(numFold,1);
        shFoldCR = zeros(numFold,1);
        shFoldEA = zeros(numFold,1);
        numShuffle = 100;
        for fi = 1 : numFold
            trainFi = setdiff(1:numFold, fi); % training fold index
            trainInd = cell2mat(foldInds(trainFi));
            testInd = sort(foldInds{fi});

            trainX = tempX(trainInd,:);
            trainY = tempY(trainInd,:);

            testX = tempX(testInd,:);
            testY = tempY(testInd,:);

            mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'discriminant', 'OptimizeHyperparameters', {'Delta', 'Gamma'}, 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
%             mdl = fitcdiscr(trainX, trainY, 'Prior', 'uniform');
            label = predict(mdl, testX);
            foldCR(fi) = length(find((label-testY)==0)) / length(testY);
            foldEA(fi) = mean(abs(label-testY));

            % Shuffling
            tempPerf = zeros(numShuffle,1);
            tempAngle = zeros(numShuffle,1);
            for shi = 1 : numShuffle
                tempShY = testY(randperm(length(testY)));
                tempPerf(shi) = length(find((label-tempShY)==0)) / length(tempShY);
                tempAngle(shi) = mean(abs(label-tempShY));
            end
            shFoldCR(fi) = mean(tempPerf);
            shFoldEA(fi) = mean(tempAngle);
        end
        
        correctRate_LDA(vi,si) = mean(foldCR);
        errorAngle_LDA(vi,si) = mean(foldEA);
        shuffleCR_LDA(vi,si) = mean(shFoldCR);
        shuffleEA_LDA(vi,si) = mean(shFoldEA);
        
        % SVM
        disp('SVM')
        foldCR = zeros(numFold,1);
        foldEA = zeros(numFold,1);
        shFoldCR = zeros(numFold,1);
        shFoldEA = zeros(numFold,1);
        numShuffle = 100;
        for fi = 1 : numFold
            trainFi = setdiff(1:numFold, fi); % training fold index
            trainInd = cell2mat(foldInds(trainFi));
            testInd = sort(foldInds{fi});

            trainX = tempX(trainInd,:);
            trainY = tempY(trainInd,:);

            testX = tempX(testInd,:);
            testY = tempY(testInd,:);

            mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'svm', 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
%             mdl = fitcdiscr(trainX, trainY, 'Prior', 'uniform');
            label = predict(mdl, testX);
            foldCR(fi) = length(find((label-testY)==0)) / length(testY);
            foldEA(fi) = mean(abs(label-testY));

            % Shuffling
            tempPerf = zeros(numShuffle,1);
            tempAngle = zeros(numShuffle,1);
            for shi = 1 : numShuffle
                tempShY = testY(randperm(length(testY)));
                tempPerf(shi) = length(find((label-tempShY)==0)) / length(tempShY);
                tempAngle(shi) = mean(abs(label-tempShY));
            end
            shFoldCR(fi) = mean(tempPerf);
            shFoldEA(fi) = mean(tempAngle);
        end
        
        correctRate_SVM(vi,si) = mean(foldCR);
        errorAngle_SVM(vi,si) = mean(foldEA);
        shuffleCR_SVM(vi,si) = mean(shFoldCR);
        shuffleEA_SVM(vi,si) = mean(shFoldEA);
        
        % KNN
        disp('KNN')
        foldCR = zeros(numFold,1);
        foldEA = zeros(numFold,1);
        shFoldCR = zeros(numFold,1);
        shFoldEA = zeros(numFold,1);
        numShuffle = 100;
        for fi = 1 : numFold
            trainFi = setdiff(1:numFold, fi); % training fold index
            trainInd = cell2mat(foldInds(trainFi));
            testInd = sort(foldInds{fi});

            trainX = tempX(trainInd,:);
            trainY = tempY(trainInd,:);

            testX = tempX(testInd,:);
            testY = tempY(testInd,:);

            mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'knn', 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
%             mdl = fitcdiscr(trainX, trainY, 'Prior', 'uniform');
            label = predict(mdl, testX);
            foldCR(fi) = length(find((label-testY)==0)) / length(testY);
            foldEA(fi) = mean(abs(label-testY));

            % Shuffling
            tempPerf = zeros(numShuffle,1);
            tempAngle = zeros(numShuffle,1);
            for shi = 1 : numShuffle
                tempShY = testY(randperm(length(testY)));
                tempPerf(shi) = length(find((label-tempShY)==0)) / length(tempShY);
                tempAngle(shi) = mean(abs(label-tempShY));
            end
            shFoldCR(fi) = mean(tempPerf);
            shFoldEA(fi) = mean(tempAngle);
        end
        
        correctRate_KNN(vi,si) = mean(foldCR);
        errorAngle_KNN(vi,si) = mean(foldEA);
        shuffleCR_KNN(vi,si) = mean(shFoldCR);
        shuffleEA_KNN(vi,si) = mean(shFoldEA);        
    end
end


figure, 
subplot(121), hold on
for vi = 1 : numVol
    plot(correctRate_LDA(vi,:), 'ko-')
end
errorbar(mean(correctRate_LDA), sem(correctRate_LDA), 'ro')
errorbar(mean(shuffleCR_LDA), sem(shuffleCR_LDA), 'o', 'color', [0.6 0.6 0.6])
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
ylabel('Correct Rate')
[~,p,m] = paired_test(correctRate_LDA(:,1), correctRate_LDA(:,2));
title(sprintf('p = %.3f (%s)', p, m))

subplot(122), hold on
for vi = 1 : numVol
    plot(errorAngle_LDA(vi,:), 'ko-')
end
errorbar(mean(errorAngle_LDA), sem(errorAngle_LDA), 'ro')
errorbar(mean(shuffleEA_LDA), sem(shuffleEA_LDA), 'o', 'color', [0.6 0.6 0.6])
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
ylabel('Error angle (\circ)')
[~,p,m] = paired_test(errorAngle_LDA(:,1), errorAngle_LDA(:,2));
title(sprintf('p = %.3f (%s)', p, m))
sgtitle('LDA')

figure, 
subplot(121), hold on
for vi = 1 : numVol
    plot(correctRate_SVM(vi,:), 'ko-')
end
errorbar(mean(correctRate_SVM), sem(correctRate_SVM), 'ro')
errorbar(mean(shuffleCR_SVM), sem(shuffleCR_SVM), 'o', 'color', [0.6 0.6 0.6])
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
ylabel('Correct Rate')
[~,p,m] = paired_test(correctRate_SVM(:,1), correctRate_SVM(:,2));
title(sprintf('p = %.3f (%s)', p, m))

subplot(122), hold on
for vi = 1 : numVol
    plot(errorAngle_SVM(vi,:), 'ko-')
end
errorbar(mean(errorAngle_SVM), sem(errorAngle_SVM), 'ro')
errorbar(mean(shuffleEA_SVM), sem(shuffleEA_SVM), 'o', 'color', [0.6 0.6 0.6])
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
ylabel('Error angle (\circ)')
[~,p,m] = paired_test(errorAngle_SVM(:,1), errorAngle_SVM(:,2));
title(sprintf('p = %.3f (%s)', p, m))
sgtitle('SVM')

figure, 
subplot(121), hold on
for vi = 1 : numVol
    plot(correctRate_KNN(vi,:), 'ko-')
end
errorbar(mean(correctRate_KNN), sem(correctRate_KNN), 'ro')
errorbar(mean(shuffleCR_KNN), sem(shuffleCR_KNN), 'o', 'color', [0.6 0.6 0.6])
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
ylabel('Correct Rate')
[~,p,m] = paired_test(correctRate_KNN(:,1), correctRate_KNN(:,2));
title(sprintf('p = %.3f (%s)', p, m))

subplot(122), hold on
for vi = 1 : numVol
    plot(errorAngle_KNN(vi,:), 'ko-')
end
errorbar(mean(errorAngle_KNN), sem(errorAngle_KNN), 'ro')
errorbar(mean(shuffleEA_KNN), sem(shuffleEA_KNN), 'o', 'color', [0.6 0.6 0.6])
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
ylabel('Error angle (\circ)')
[~,p,m] = paired_test(errorAngle_KNN(:,1), errorAngle_KNN(:,2));
title(sprintf('p = %.3f (%s)', p, m))
sgtitle('KNN')





%%
%%
%%
%%
%% Before the first lick
%%
%%
%%
%%
baseDir = 'C:\Users\shires\Dropbox\Works\Projects\2020 Neural stretching in S1\Data\';
saveFn = 'objectAnglePrediction_firstLick';

% object angle prediction from the model
% between 11 pairs of sessions
glmnetOpt = glmnetSet;
glmnetOpt.standardize = 0; %set to 0 b/c already standardized
glmnetOpt.alpha = 0.95;
glmnetOpt.xfoldCV = 5;

numIter = 10;
trainingProp = 0.7;

objectAngleModel = cell(12,2); % (:,1) naive, (:,2) expert. (3,:) will be empty.
Ypairs = cell(12,2);
for mi = 1 : numMice
    fprintf('Processing mouse #%d/%d.\n', mi, numMice)
    mouse = mice(mi);
    % naive
    fprintf('Naive session.\n')
    session = sessions{mi}(1);
    ufn = sprintf('UberJK%03dS%02d_NC',mouse, session);
    load(sprintf('%s%s',baseDir, ufn), 'u')
    
    poleUpTrialInd = find(cellfun(@(x) ~isempty(x.poleUpTime), u.trials));
    u.trials = u.trials(poleUpTrialInd);
    poleUpTimeAll = cellfun(@(x) x.poleUpTime(1), u.trials, 'un', 0);
    allLickTimeAll = cellfun(@(x) union(union(union(x.leftLickTime, x.rightLickTime), x.answerLickTime), x.poleDownOnsetTime), u.trials, 'un', 0);
    firstLickTimeAll = cellfun(@(x,y) x(find(x>y, 1, 'first')), allLickTimeAll, poleUpTimeAll, 'un', 0);
    touchTrialInd = find(cellfun(@(x) length(x.protractionTouchChunksByWhisking), u.trials));
    touchAnswerTrialInd = find(cellfun(@(x,y) x.whiskerTime(x.protractionTouchChunksByWhisking{1}(1)) < y, u.trials(touchTrialInd), firstLickTimeAll(touchTrialInd)));
    trialInd = touchTrialInd(touchAnswerTrialInd);
    upperInd = trialInd( find(cellfun(@(x) ismember(1, x.planes), u.trials(trialInd))) );
    lowerInd = trialInd( find(cellfun(@(x) ismember(5, x.planes), u.trials(trialInd))) );
    % upper
    if mi ~= 2 % disregard JK027 upper volume due to low # of trials in the expert session
        utrials = u.trials(upperInd);
        firstLickTime = firstLickTimeAll(upperInd);
        
        numTouchCell = cellfun(@(x,y) find(x.whiskerTime(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking)) < y), utrials, firstLickTime, 'uniformoutput', false);
        
        inputs = zeros(length(upperInd), 12); % 12 whisker inputs
        inputs(:,1) = cellfun(@(x,y) mean(x.theta(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
        inputs(:,2) = cellfun(@(x,y) mean(x.phi(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
        inputs(:,3) = cellfun(@(x,y) mean(x.kappaH(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
        inputs(:,4) = cellfun(@(x,y) mean(x.kappaV(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
        inputs(:,5) = cellfun(@(x,y) mean(x.arcLength(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
        inputs(:,6) = cellfun(@length, numTouchCell);
        
        inputs(:,7) = cellfun(@(x,y) mean(x.protractionTouchDThetaByWhisking(y)), utrials, numTouchCell);
        inputs(:,8) = cellfun(@(x,y) mean(x.protractionTouchDPhiByWhisking(y)), utrials, numTouchCell);
        inputs(:,9) = cellfun(@(x,y) mean(x.protractionTouchDKappaHByWhisking(y)), utrials, numTouchCell);
        inputs(:,10) = cellfun(@(x,y) mean(x.protractionTouchDKappaVByWhisking(y)), utrials, numTouchCell);
        inputs(:,11) = cellfun(@(x,y) mean(x.protractionTouchSlideDistanceByWhisking(y)), utrials, numTouchCell);
        inputs(:,12) = cellfun(@(x,y) mean(x.protractionTouchDurationByWhisking(y)), utrials, numTouchCell);
        
        stdInputs = (inputs - nanmean(inputs))./nanstd(inputs);
        
        output = cellfun(@(x) x.angle, utrials);
        
        % Divide into training (70%) and test (30%) set 
        % (70% of trial nums from the lowest trial num angle, to prevent
        % overly fitting to 90 degrees)
        % and run multinomial GLM fitting
        % Then, collect coefficients from 10 iterations
        fitCoeffs = zeros(13,7,numIter);
        correctRate = zeros(1,numIter);
        errorAngle = zeros(1,numIter);
        for ii = 1 : numIter
            % stratify by angles
            angles = unique(output);
            if length(angles) ~= 7
                error('Catch trials are included.')
            end
            
            numTrialAngle = zeros(length(angles),1);
            for ai = 1 : length(angles)
                numTrialAngle(ai) = length(find(output == angles(ai)));
            end
            minNumTrial = min(numTrialAngle);
            trainingNumAngle = round(trainingProp * minNumTrial);
            
            trainingInd = [];
            for ai = 1 : length(angles)
                tempInd = find(output == angles(ai));
                randInd = randperm(length(tempInd), trainingNumAngle);
                trainingInd = [trainingInd; tempInd(randInd)];
            end
            trainX = stdInputs(trainingInd,:);
            trainY = output(trainingInd);
            cv = cvglmnet(trainX, trainY, 'multinomial', glmnetOpt, [], glmnetOpt.xfoldCV);
            
            fitLambda = cv.lambda_1se;
            iLambda = find(cv.lambda == cv.lambda_1se);
            fitCoeffs(:,:,ii) = [cv.glmnet_fit.a0(:,iLambda)' ; cell2mat(cellfun(@(x) x(:,iLambda),cv.glmnet_fit.beta,'uniformoutput',false))];

            % Test set
            testInd = setdiff(1:length(output), trainingInd);
            testX = stdInputs(testInd,:);
            testY = output(testInd);
            predicts = cvglmnetPredict(cv,testX,fitLambda); %output as X*weights
            probability =  1 ./ (1+exp(predicts*-1)); % convert to probability by using mean function

            [~,predInd] = max(probability,[],2);
            pred = angles(predInd);
            
            % Goodness of fit metrics
            % correct rate and mean error in angle
            correctRate(ii) = mean(pred == testY);
            errorAngle(ii) = mean(abs(pred-testY));
        end
        objectAngleModel{mi*2-1,1}.fitCoeffs = fitCoeffs;
        objectAngleModel{mi*2-1,1}.correctRate = correctRate;
        objectAngleModel{mi*2-1,1}.errorAngle = errorAngle;
        
        meanCoeff = mean(fitCoeffs,3);
        dataX = [ones(size(stdInputs,1),1),stdInputs];
        probMat = dataX * meanCoeff;
        probY = exp(probMat) ./ sum(exp(probMat),2);

        [~, maxind] = max(probY,[],2);
        predY = angles(maxind);
        Ypairs{mi*2-1, 1} = [output, predY];
    end
    
    % lower
    utrials = u.trials(lowerInd);
    firstLickTime = firstLickTimeAll(lowerInd);

    numTouchCell = cellfun(@(x,y) find(x.whiskerTime(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking)) < y), utrials, firstLickTime, 'uniformoutput', false);

    inputs = zeros(length(lowerInd), 12); % 12 whisker inputs
    inputs(:,1) = cellfun(@(x,y) mean(x.theta(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
    inputs(:,2) = cellfun(@(x,y) mean(x.phi(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
    inputs(:,3) = cellfun(@(x,y) mean(x.kappaH(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
    inputs(:,4) = cellfun(@(x,y) mean(x.kappaV(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
    inputs(:,5) = cellfun(@(x,y) mean(x.arcLength(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
    inputs(:,6) = cellfun(@length, numTouchCell);

    inputs(:,7) = cellfun(@(x,y) mean(x.protractionTouchDThetaByWhisking(y)), utrials, numTouchCell);
    inputs(:,8) = cellfun(@(x,y) mean(x.protractionTouchDPhiByWhisking(y)), utrials, numTouchCell);
    inputs(:,9) = cellfun(@(x,y) mean(x.protractionTouchDKappaHByWhisking(y)), utrials, numTouchCell);
    inputs(:,10) = cellfun(@(x,y) mean(x.protractionTouchDKappaVByWhisking(y)), utrials, numTouchCell);
    inputs(:,11) = cellfun(@(x,y) mean(x.protractionTouchSlideDistanceByWhisking(y)), utrials, numTouchCell);
    inputs(:,12) = cellfun(@(x,y) mean(x.protractionTouchDurationByWhisking(y)), utrials, numTouchCell);

    stdInputs = (inputs - nanmean(inputs))./nanstd(inputs);

    output = cellfun(@(x) x.angle, utrials);

    % Divide into training (70%) and test (30%) set
    % and run multinomial GLM fitting
    % Then, collect coefficients from 10 iterations
    fitCoeffs = zeros(13,7,numIter);
    correctRate = zeros(1,numIter);
    errorAngle = zeros(1,numIter);
    for ii = 1 : numIter
        % stratify by angles
        angles = unique(output);
        if length(angles) ~= 7
            error('Catch trials are included.')
        end

        numTrialAngle = zeros(length(angles),1);
        for ai = 1 : length(angles)
            numTrialAngle(ai) = length(find(output == angles(ai)));
        end
        minNumTrial = min(numTrialAngle);
        trainingNumAngle = round(trainingProp * minNumTrial);

        trainingInd = [];
        for ai = 1 : length(angles)
            tempInd = find(output == angles(ai));
            randInd = randperm(length(tempInd), trainingNumAngle);
            trainingInd = [trainingInd; tempInd(randInd)];
        end
        trainX = stdInputs(trainingInd,:);
        trainY = output(trainingInd);
        cv = cvglmnet(trainX, trainY, 'multinomial', glmnetOpt, [], glmnetOpt.xfoldCV);

        fitLambda = cv.lambda_1se;
        iLambda = find(cv.lambda == cv.lambda_1se);
        fitCoeffs(:,:,ii) = [cv.glmnet_fit.a0(:,iLambda)' ; cell2mat(cellfun(@(x) x(:,iLambda),cv.glmnet_fit.beta,'uniformoutput',false))];

        % Test set
        testInd = setdiff(1:length(output), trainingInd);
        testX = stdInputs(testInd,:);
        testY = output(testInd);
        predicts = cvglmnetPredict(cv,testX,fitLambda); %output as X*weights
        probability =  1 ./ (1+exp(predicts*-1)); % convert to probability by using mean function

        [~,predInd] = max(probability,[],2);
        pred = angles(predInd);

        % Goodness of fit metrics
        % correct rate and mean error in angle
        correctRate(ii) = mean(pred == testY);
        errorAngle(ii) = mean(abs(pred-testY));
    end
    objectAngleModel{mi*2,1}.fitCoeffs = fitCoeffs;
    objectAngleModel{mi*2,1}.correctRate = correctRate;
    objectAngleModel{mi*2,1}.errorAngle = errorAngle;
    
    meanCoeff = mean(fitCoeffs,3);
    dataX = [ones(size(stdInputs,1),1),stdInputs];
    probMat = dataX * meanCoeff;
    probY = exp(probMat) ./ sum(exp(probMat),2);

    [~, maxind] = max(probY,[],2);
    predY = angles(maxind);
    Ypairs{mi*2, 1} = [output, predY];
        
    % expert
    fprintf('Expert session.\n')
    session = sessions{mi}(2);
    ufn = sprintf('UberJK%03dS%02d_NC',mouse, session);
    load(sprintf('%s%s',baseDir, ufn), 'u')
    
    poleUpTrialInd = find(cellfun(@(x) ~isempty(x.poleUpTime), u.trials));
    u.trials = u.trials(poleUpTrialInd);
    
    poleUpTimeAll = cellfun(@(x) x.poleUpTime(1), u.trials, 'un', 0);
    allLickTimeAll = cellfun(@(x) union(union(union(x.leftLickTime, x.rightLickTime), x.answerLickTime), x.poleDownOnsetTime), u.trials, 'un', 0);
    firstLickTimeAll = cellfun(@(x,y) x(find(x>y, 1, 'first')), allLickTimeAll, poleUpTimeAll, 'un', 0);
    
    touchTrialInd = find(cellfun(@(x) length(x.protractionTouchChunksByWhisking), u.trials));
    touchAnswerTrialInd = find(cellfun(@(x,y) x.whiskerTime(x.protractionTouchChunksByWhisking{1}(1)) < y, u.trials(touchTrialInd), firstLickTimeAll(touchTrialInd)));
    trialInd = touchTrialInd(touchAnswerTrialInd);
    upperInd = trialInd( find(cellfun(@(x) ismember(1, x.planes), u.trials(trialInd))) );
    lowerInd = trialInd( find(cellfun(@(x) ismember(5, x.planes), u.trials(trialInd))) );
    % upper
    if mi ~= 2 % disregard JK027 upper volume due to low # of trials in the expert session
        utrials = u.trials(upperInd);
        firstLickTime = firstLickTimeAll(upperInd);
        
        numTouchCell = cellfun(@(x,y) find(x.whiskerTime(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking)) < y), utrials, firstLickTime, 'uniformoutput', false);
        
        inputs = zeros(length(upperInd), 12); % 12 whisker inputs
        inputs(:,1) = cellfun(@(x,y) mean(x.theta(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
        inputs(:,2) = cellfun(@(x,y) mean(x.phi(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
        inputs(:,3) = cellfun(@(x,y) mean(x.kappaH(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
        inputs(:,4) = cellfun(@(x,y) mean(x.kappaV(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
        inputs(:,5) = cellfun(@(x,y) mean(x.arcLength(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
        inputs(:,6) = cellfun(@length, numTouchCell);
        
        inputs(:,7) = cellfun(@(x,y) mean(x.protractionTouchDThetaByWhisking(y)), utrials, numTouchCell);
        inputs(:,8) = cellfun(@(x,y) mean(x.protractionTouchDPhiByWhisking(y)), utrials, numTouchCell);
        inputs(:,9) = cellfun(@(x,y) mean(x.protractionTouchDKappaHByWhisking(y)), utrials, numTouchCell);
        inputs(:,10) = cellfun(@(x,y) mean(x.protractionTouchDKappaVByWhisking(y)), utrials, numTouchCell);
        inputs(:,11) = cellfun(@(x,y) mean(x.protractionTouchSlideDistanceByWhisking(y)), utrials, numTouchCell);
        inputs(:,12) = cellfun(@(x,y) mean(x.protractionTouchDurationByWhisking(y)), utrials, numTouchCell);
        
        stdInputs = (inputs - nanmean(inputs))./nanstd(inputs);
        
        output = cellfun(@(x) x.angle, utrials);
        
        % Divide into training (70%) and test (30%) set
        % and run multinomial GLM fitting
        % Then, collect coefficients from 10 iterations
        fitCoeffs = zeros(13,7,numIter);
        correctRate = zeros(1,numIter);
        errorAngle = zeros(1,numIter);
        for ii = 1 : numIter
            % stratify by angles
            angles = unique(output);
            if length(angles) ~= 7
                error('Catch trials are included.')
            end
            
            numTrialAngle = zeros(length(angles),1);
            for ai = 1 : length(angles)
                numTrialAngle(ai) = length(find(output == angles(ai)));
            end
            minNumTrial = min(numTrialAngle);
            trainingNumAngle = round(trainingProp * minNumTrial);
            
            trainingInd = [];
            for ai = 1 : length(angles)
                tempInd = find(output == angles(ai));
                randInd = randperm(length(tempInd), trainingNumAngle);
                trainingInd = [trainingInd; tempInd(randInd)];
            end
            trainX = stdInputs(trainingInd,:);
            trainY = output(trainingInd);
            cv = cvglmnet(trainX, trainY, 'multinomial', glmnetOpt, [], glmnetOpt.xfoldCV);
            
            fitLambda = cv.lambda_1se;
            iLambda = find(cv.lambda == cv.lambda_1se);
            fitCoeffs(:,:,ii) = [cv.glmnet_fit.a0(:,iLambda)' ; cell2mat(cellfun(@(x) x(:,iLambda),cv.glmnet_fit.beta,'uniformoutput',false))];

            % Test set
            testInd = setdiff(1:length(output), trainingInd);
            testX = stdInputs(testInd,:);
            testY = output(testInd);
            predicts = cvglmnetPredict(cv,testX,fitLambda); %output as X*weights
            probability =  1 ./ (1+exp(predicts*-1)); % convert to probability by using mean function

            [~,predInd] = max(probability,[],2);
            pred = angles(predInd);
            
            % Goodness of fit metrics
            % correct rate and mean error in angle
            correctRate(ii) = mean(pred == testY);
            errorAngle(ii) = mean(abs(pred-testY));
        end
        objectAngleModel{mi*2-1,2}.fitCoeffs = fitCoeffs;
        objectAngleModel{mi*2-1,2}.correctRate = correctRate;
        objectAngleModel{mi*2-1,2}.errorAngle = errorAngle;
        
        meanCoeff = mean(fitCoeffs,3);
        dataX = [ones(size(stdInputs,1),1),stdInputs];
        probMat = dataX * meanCoeff;
        probY = exp(probMat) ./ sum(exp(probMat),2);

        [~, maxind] = max(probY,[],2);
        predY = angles(maxind);
        Ypairs{mi*2-1, 2} = [output, predY];
    end
    
    % lower
    utrials = u.trials(lowerInd);
    firstLickTime = firstLickTimeAll(lowerInd);

    numTouchCell = cellfun(@(x,y) find(x.whiskerTime(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking)) < y), utrials, firstLickTime, 'uniformoutput', false);

    inputs = zeros(length(lowerInd), 12); % 12 whisker inputs
    inputs(:,1) = cellfun(@(x,y) mean(x.theta(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
    inputs(:,2) = cellfun(@(x,y) mean(x.phi(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
    inputs(:,3) = cellfun(@(x,y) mean(x.kappaH(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
    inputs(:,4) = cellfun(@(x,y) mean(x.kappaV(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
    inputs(:,5) = cellfun(@(x,y) mean(x.arcLength(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
    inputs(:,6) = cellfun(@length, numTouchCell);

    inputs(:,7) = cellfun(@(x,y) mean(x.protractionTouchDThetaByWhisking(y)), utrials, numTouchCell);
    inputs(:,8) = cellfun(@(x,y) mean(x.protractionTouchDPhiByWhisking(y)), utrials, numTouchCell);
    inputs(:,9) = cellfun(@(x,y) mean(x.protractionTouchDKappaHByWhisking(y)), utrials, numTouchCell);
    inputs(:,10) = cellfun(@(x,y) mean(x.protractionTouchDKappaVByWhisking(y)), utrials, numTouchCell);
    inputs(:,11) = cellfun(@(x,y) mean(x.protractionTouchSlideDistanceByWhisking(y)), utrials, numTouchCell);
    inputs(:,12) = cellfun(@(x,y) mean(x.protractionTouchDurationByWhisking(y)), utrials, numTouchCell);

    stdInputs = (inputs - nanmean(inputs))./nanstd(inputs);

    output = cellfun(@(x) x.angle, utrials);

    % Divide into training (70%) and test (30%) set
    % and run multinomial GLM fitting
    % Then, collect coefficients from 10 iterations
    fitCoeffs = zeros(13,7,numIter);
    correctRate = zeros(1,numIter);
    errorAngle = zeros(1,numIter);
    for ii = 1 : numIter
        % stratify by angles
        angles = unique(output);
        if length(angles) ~= 7
            error('Catch trials are included.')
        end

        numTrialAngle = zeros(length(angles),1);
        for ai = 1 : length(angles)
            numTrialAngle(ai) = length(find(output == angles(ai)));
        end
        minNumTrial = min(numTrialAngle);
        trainingNumAngle = round(trainingProp * minNumTrial);

        trainingInd = [];
        for ai = 1 : length(angles)
            tempInd = find(output == angles(ai));
            randInd = randperm(length(tempInd), trainingNumAngle);
            trainingInd = [trainingInd; tempInd(randInd)];
        end
        trainX = stdInputs(trainingInd,:);
        trainY = output(trainingInd);
        cv = cvglmnet(trainX, trainY, 'multinomial', glmnetOpt, [], glmnetOpt.xfoldCV);

        fitLambda = cv.lambda_1se;
        iLambda = find(cv.lambda == cv.lambda_1se);
        fitCoeffs(:,:,ii) = [cv.glmnet_fit.a0(:,iLambda)' ; cell2mat(cellfun(@(x) x(:,iLambda),cv.glmnet_fit.beta,'uniformoutput',false))];

        % Test set
        testInd = setdiff(1:length(output), trainingInd);
        testX = stdInputs(testInd,:);
        testY = output(testInd);
        predicts = cvglmnetPredict(cv,testX,fitLambda); %output as X*weights
        probability =  1 ./ (1+exp(predicts*-1)); % convert to probability by using mean function

        [~,predInd] = max(probability,[],2);
        pred = angles(predInd);

        % Goodness of fit metrics
        % correct rate and mean error in angle
        correctRate(ii) = mean(pred == testY);
        errorAngle(ii) = mean(abs(pred-testY));
    end
    objectAngleModel{mi*2,2}.fitCoeffs = fitCoeffs;
    objectAngleModel{mi*2,2}.correctRate = correctRate;
    objectAngleModel{mi*2,2}.errorAngle = errorAngle;
    
    meanCoeff = mean(fitCoeffs,3);
    dataX = [ones(size(stdInputs,1),1),stdInputs];
    probMat = dataX * meanCoeff;
    probY = exp(probMat) ./ sum(exp(probMat),2);

    [~, maxind] = max(probY,[],2);
    predY = angles(maxind);
    Ypairs{mi*2, 2} = [output, predY];
end

save([baseDir, saveFn], 'objectAngleModel', 'Ypairs')

%%

clear
baseDir = 'C:\Users\shires\Dropbox\Works\Projects\2020 Neural stretching in S1\Data\';
matchedPR = load([baseDir, 'matchedPopResponse_201230'], 'naive', 'expert');
numVol = 11;
load([baseDir, 'objectAnglePrediction_firstLick'], 'Ypairs', 'objectAngleModel'); % This data is made from d200826_neural_stretching_confirm_and_control


%%
angles = 45:15:135;
numSamplesBS = 100;
nIterBS = 100;
nShuffle = 100;
correctRate = nan(12,2);
chanceCR = nan(12,2);
errorAngle = nan(12,2);
chanceEA = nan(12,2);
confMat = cell(12,2);
for gi = 1 : 12
    if gi ~= 3
        for si = 1 : 2
            tempPair = Ypairs{gi,si};
            tempCR = zeros(nIterBS, 1);
            tempChanceCR = zeros(nIterBS,1);
            tempEA = zeros(nIterBS, 1);
            tempChanceEA = zeros(nIterBS,1);
            tempConfMat = zeros(length(angles), length(angles), nIterBS);
            angleInds = cell(length(angles),1);
            for ai = 1 : length(angles)
                angleInds{ai} = find(tempPair(:,1)==angles(ai));
            end
            for ii = 1 : nIterBS
                tempIterPair = zeros(numSamplesBS * length(angles), 2);
                for ai = 1 : length(angles)
                    % bootstrapping
                    tempInds = randi(length(angleInds{ai}),[numSamplesBS,1]);
                    inds = angleInds{ai}(tempInds);
                    tempIterPair( (ai-1)*numSamplesBS+1:ai*numSamplesBS, : ) = tempPair(inds,:);
                end
                tempCR(ii) = length(find(tempIterPair(:,2) - tempIterPair(:,1)==0)) / (numSamplesBS * length(angles));
                tempEA(ii) = mean(abs(tempIterPair(:,2) - tempIterPair(:,1)));

                tempTempCR = zeros(nShuffle,1);
                tempTempEA = zeros(nShuffle,1);
                for shuffi = 1 : nShuffle
                    shuffledPair = [tempIterPair(:,1), tempIterPair(randperm(size(tempIterPair,1)),2)];
                    tempTempCR(shuffi) = length(find(shuffledPair(:,2) - shuffledPair(:,1)==0)) / (numSamplesBS * length(angles));
                    tempTempEA(shuffi) = mean(abs(shuffledPair(:,2) - shuffledPair(:,1)));
                end
                tempChanceCR(ii) = mean(tempTempCR);
                tempChanceEA(ii) = mean(tempTempEA);

                tempConfMat(:,:,ii) = confusionmat(tempIterPair(:,1), tempIterPair(:,2))/numSamplesBS;
            end
            correctRate(gi,si) = mean(tempCR);
            chanceCR(gi,si) = mean(tempChanceCR);
            errorAngle(gi,si) = mean(tempEA);
            chanceEA(gi,si) = mean(tempChanceEA);
            confMat{gi,si} = mean(tempConfMat,3);
        end
    end
end

%% Contingency tables - Naive
contMatNaive = nan(7,7,12);
for i = 1 : 12
    if i ~=3
        contMatNaive(:,:,i) = confMat{i,1};
    end
end

figure, imagesc(nanmean(contMatNaive,3),[0 0.85]), axis square, colorbar
yticklabels(angles)
xticklabels(angles)
ylabel('Data')
xlabel('Prediction')
title('Naive')

%% Contingency tables - Expert
contMatExpert = nan(7,7,12);
for i = 1 : 12
    if i ~=3
        contMatExpert(:,:,i) = confMat{i,2};
    end
end

figure, imagesc(nanmean(contMatExpert,3),[0 0.85]), axis square, colorbar
yticklabels(angles)
xticklabels(angles)
ylabel('Data')
xlabel('Prediction')
title('Expert')

%% Classification performance - correct rate

figure('units','norm','pos',[0.2 0.2 0.1 0.3]), hold on,
for i = 1 : 12
    plot(correctRate(i,:), 'ko-')
end
errorbar([1,2],nanmean(correctRate), sem(correctRate), 'ro', 'lines', 'no')
errorbar([1,2],nanmean(chanceCR), sem(chanceCR), 'o', 'lines', 'no', 'color', [0.6 0.6 0.6])

[~,p,m] = paired_test(diff(correctRate,1,2));

title(sprintf('p = %.3f; method = %s', p, m))
xlim([0.5 2.5]), xticks([1,2]), xticklabels({'Naive', 'Expert'})
ylim([0 1])
yticks([0:0.1:1])
ylabel('Correct rate')
set(gca,'fontsize',12, 'fontname', 'Arial')


%% For text
nanmean(correctRate)
sem(correctRate)

nanmean(chanceCR)
sem(chanceCR)


%% Classification performance - prediction error

figure('units','norm','pos',[0.2 0.2 0.1 0.3]), hold on,
for i = 1 : 12
    plot(errorAngle(i,:), 'ko-')
end
errorbar([1,2],nanmean(errorAngle), sem(errorAngle), 'ro', 'lines', 'no')
errorbar([1,2],nanmean(chanceEA), sem(chanceEA), 'o', 'lines', 'no', 'color', [0.6 0.6 0.6])

[~,p,m] = paired_test(diff(errorAngle,1,2));

title(sprintf('p = %.3f; method = %s', p, m))
xlim([0.5 2.5]), xticks([1,2]), xticklabels({'Naive', 'Expert'})
ylim([0 38])
yticks([0:5:35])
ylabel('Prediction error (\circ)')
set(gca,'fontsize',12, 'fontname', 'Arial')


%% for text
mean(errorAngle)
sem(errorAngle)

mean(chanceEA)
sem(chanceEA)






%%
%% First, retrieve data and save them
%%
% # whisks, # touches, touch duration,
% 12 sensory inputs

mice = [25,27,30,36,39,52];
sessions = {[4,19],[3,10],[3,21],[1,17],[1,23],[3,21]};
numMice = length(mice);

saveFn = 'rawWhiskerFeatures_firstLick.mat';
% This is the data to be saved. 3rd row will be removed (because it will be
% empty)
rawFeat = cell(12,2); % (:,1) naive, (:,2) expert. 
% In each cell, (:,1:12) 12 sensory features, (:,13) num whisk, (:,14) whisking amplitude, (:,15) object angle
% num Touch at (:,6), touch duration at (:,11)
for mi = 1 : numMice
    fprintf('Processing mouse #%d/%d.\n', mi, numMice)
    mouse = mice(mi);
    % naive
    fprintf('Naive session.\n')
    session = sessions{mi}(1);
    ufn = sprintf('UberJK%03dS%02d_NC',mouse, session);
    load(sprintf('%s%s',baseDir, ufn), 'u')
    
    poleUpTrialInd = find(cellfun(@(x) ~isempty(x.poleUpTime), u.trials));
    u.trials = u.trial(poleUpTrialInd);
    
    poleUpTimeAll = cellfun(@(x) x.poleUpTime(1), u.trials, 'un', 0);
    allLickTimeAll = cellfun(@(x) union(union(union(x.leftLickTime, x.rightLickTime), x.answerLickTime), x.poleDownOnsetTime), u.trials, 'un', 0);
    firstLickTimeAll = cellfun(@(x,y) x(find(x>y, 1, 'first')), allLickTimeAll, poleUpTimeAll, 'un', 0);
    
    touchTrialInd = find(cellfun(@(x) length(x.protractionTouchChunksByWhisking), u.trials));
    touchAnswerTrialInd = find(cellfun(@(x,y) x.whiskerTime(x.protractionTouchChunksByWhisking{1}(1)) < y, u.trials(touchTrialInd), firstLickTimeAll(touchTrialInd)));
    trialInd = touchTrialInd(touchAnswerTrialInd);
    upperInd = trialInd( find(cellfun(@(x) ismember(1, x.planes), u.trials(trialInd))) );
    lowerInd = trialInd( find(cellfun(@(x) ismember(5, x.planes), u.trials(trialInd))) );
    % upper
    if mi ~= 2 % disregard JK027 upper volume due to low # of trials in the expert session
        utrials = u.trials(upperInd);
        firstLickTime = firstLickTimeAll(upperInd);
        
        numTouchCell = cellfun(@(x,y) find(x.whiskerTime(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking)) < y), utrials, firstLickTime, 'uniformoutput', false);
        
        rawData = zeros(length(upperInd), 15); % 12 whisker inputs + num whisk, whisk amplitude, object angle
        rawData(:,1) = cellfun(@(x,y) mean(x.theta(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
        rawData(:,2) = cellfun(@(x,y) mean(x.phi(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
        rawData(:,3) = cellfun(@(x,y) mean(x.kappaH(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
        rawData(:,4) = cellfun(@(x,y) -mean(x.kappaV(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell); % inverse the sign
        rawData(:,5) = cellfun(@(x,y) mean(x.arcLength(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
        rawData(:,6) = cellfun(@length, numTouchCell);
        
        rawData(:,7) = cellfun(@(x,y) mean(x.protractionTouchDThetaByWhisking(y)), utrials, numTouchCell);
        rawData(:,8) = cellfun(@(x,y) mean(x.protractionTouchDPhiByWhisking(y)), utrials, numTouchCell);
        rawData(:,9) = cellfun(@(x,y) -mean(x.protractionTouchDKappaHByWhisking(y)), utrials, numTouchCell); % inverse the sign
        rawData(:,10) = cellfun(@(x,y) mean(x.protractionTouchDKappaVByWhisking(y)), utrials, numTouchCell);
        rawData(:,11) = cellfun(@(x,y) mean(x.protractionTouchDurationByWhisking(y)), utrials, numTouchCell);
        rawData(:,12) = cellfun(@(x,y) mean(x.protractionTouchSlideDistanceByWhisking(y)), utrials, numTouchCell);
        
        for ti = 1 : length(upperInd)
            tempT = utrials{ti};
            poleUpInd = find(tempT.whiskerTime > tempT.poleUpTime(1),1,'first');
            lickInd = find(tempT.whiskerTime < firstLickTime{ti},1,'last');
            theta = tempT.theta(poleUpInd:lickInd);
            [onsetFrame, ~, ~, whiskingAmp, ~] = jkWhiskerOnsetNAmplitude(theta);
            rawData(ti,13) = length(onsetFrame);
            rawData(ti,14) = mean(whiskingAmp(find(whiskingAmp>2.5)));
        end
        
        rawData(:,15) = cellfun(@(x) x.angle, utrials);
        rawFeat{(mi-1)*2+1,1} = rawData;
    end
    
    % lower
    utrials = u.trials(lowerInd);
    firstLickTime = firstLickTimeAll(lowerInd);

    numTouchCell = cellfun(@(x,y) find(x.whiskerTime(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking)) < y), utrials, firstLickTime, 'uniformoutput', false);

    rawData = zeros(length(lowerInd), 15); % 12 whisker inputs + num whisk, whisk amplitude, object angle
    rawData(:,1) = cellfun(@(x,y) mean(x.theta(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
    rawData(:,2) = cellfun(@(x,y) mean(x.phi(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
    rawData(:,3) = cellfun(@(x,y) mean(x.kappaH(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
    rawData(:,4) = cellfun(@(x,y) -mean(x.kappaV(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell); % inverse the sign
    rawData(:,5) = cellfun(@(x,y) mean(x.arcLength(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
    rawData(:,6) = cellfun(@length, numTouchCell);

    rawData(:,7) = cellfun(@(x,y) mean(x.protractionTouchDThetaByWhisking(y)), utrials, numTouchCell);
    rawData(:,8) = cellfun(@(x,y) mean(x.protractionTouchDPhiByWhisking(y)), utrials, numTouchCell);
    rawData(:,9) = cellfun(@(x,y) -mean(x.protractionTouchDKappaHByWhisking(y)), utrials, numTouchCell); % inverse the sign
    rawData(:,10) = cellfun(@(x,y) mean(x.protractionTouchDKappaVByWhisking(y)), utrials, numTouchCell);
    rawData(:,11) = cellfun(@(x,y) mean(x.protractionTouchDurationByWhisking(y)), utrials, numTouchCell);
    rawData(:,12) = cellfun(@(x,y) mean(x.protractionTouchSlideDistanceByWhisking(y)), utrials, numTouchCell);

    for ti = 1 : length(lowerInd)
        tempT = utrials{ti};
        poleUpInd = find(tempT.whiskerTime > tempT.poleUpTime(1),1,'first');
        lickInd = find(tempT.whiskerTime < firstLickTime{ti},1,'last');
        theta = tempT.theta(poleUpInd:lickInd);
        [onsetFrame, ~, ~, whiskingAmp, ~] = jkWhiskerOnsetNAmplitude(theta);
        rawData(ti,13) = length(onsetFrame);
        rawData(ti,14) = mean(whiskingAmp(find(whiskingAmp>2.5)));
    end

    rawData(:,15) = cellfun(@(x) x.angle, utrials);
    rawFeat{mi*2,1} = rawData;
    
        
    % expert
    fprintf('Expert session.\n')
    session = sessions{mi}(2);
    ufn = sprintf('UberJK%03dS%02d_NC',mouse, session);
    load(sprintf('%s%s',baseDir, ufn), 'u')
    
    poleUpTrialInd = find(cellfun(@(x) ~isempty(x.poleUpTime), u.trials));
    u.trials = u.trial(poleUpTrialInd);
    
    poleUpTimeAll = cellfun(@(x) x.poleUpTime(1), u.trials, 'un', 0);
    allLickTimeAll = cellfun(@(x) union(union(union(x.leftLickTime, x.rightLickTime), x.answerLickTime), x.poleDownOnsetTime), u.trials, 'un', 0);
    firstLickTimeAll = cellfun(@(x,y) x(find(x>y, 1, 'first')), allLickTimeAll, poleUpTimeAll, 'un', 0);
    
    touchTrialInd = find(cellfun(@(x) length(x.protractionTouchChunksByWhisking), u.trials));
    touchAnswerTrialInd = find(cellfun(@(x,y) x.whiskerTime(x.protractionTouchChunksByWhisking{1}(1)) < y, u.trials(touchTrialInd), firstLickTimeAll(touchTrialInd)));
    trialInd = touchTrialInd(touchAnswerTrialInd);
    upperInd = trialInd( find(cellfun(@(x) ismember(1, x.planes), u.trials(trialInd))) );
    lowerInd = trialInd( find(cellfun(@(x) ismember(5, x.planes), u.trials(trialInd))) );
    % upper
    if mi ~= 2 % disregard JK027 upper volume due to low # of trials in the expert session
        utrials = u.trials(upperInd);
        firstLickTime = firstLickTimeAll(upperInd);
        
        numTouchCell = cellfun(@(x,y) find(x.whiskerTime(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking)) < y), utrials, firstLickTime, 'uniformoutput', false);
        
        rawData = zeros(length(upperInd), 15); % 12 whisker inputs + num whisk, whisk amplitude, object angle
        rawData(:,1) = cellfun(@(x,y) mean(x.theta(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
        rawData(:,2) = cellfun(@(x,y) mean(x.phi(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
        rawData(:,3) = cellfun(@(x,y) mean(x.kappaH(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
        rawData(:,4) = cellfun(@(x,y) -mean(x.kappaV(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell); % inverse the sign
        rawData(:,5) = cellfun(@(x,y) mean(x.arcLength(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
        rawData(:,6) = cellfun(@length, numTouchCell);
        
        rawData(:,7) = cellfun(@(x,y) mean(x.protractionTouchDThetaByWhisking(y)), utrials, numTouchCell);
        rawData(:,8) = cellfun(@(x,y) mean(x.protractionTouchDPhiByWhisking(y)), utrials, numTouchCell);
        rawData(:,9) = cellfun(@(x,y) -mean(x.protractionTouchDKappaHByWhisking(y)), utrials, numTouchCell); % inverse the sign
        rawData(:,10) = cellfun(@(x,y) mean(x.protractionTouchDKappaVByWhisking(y)), utrials, numTouchCell);
        rawData(:,11) = cellfun(@(x,y) mean(x.protractionTouchDurationByWhisking(y)), utrials, numTouchCell);
        rawData(:,12) = cellfun(@(x,y) mean(x.protractionTouchSlideDistanceByWhisking(y)), utrials, numTouchCell);
        
        for ti = 1 : length(upperInd)
            tempT = utrials{ti};
            poleUpInd = find(tempT.whiskerTime > tempT.poleUpTime(1),1,'first');
            lickInd = find(tempT.whiskerTime < firstLickTime{ti},1,'last');
            theta = tempT.theta(poleUpInd:lickInd);
            [onsetFrame, ~, ~, whiskingAmp, ~] = jkWhiskerOnsetNAmplitude(theta);
            rawData(ti,13) = length(onsetFrame);
            rawData(ti,14) = mean(whiskingAmp(find(whiskingAmp>2.5)));
        end
        
        rawData(:,15) = cellfun(@(x) x.angle, utrials);
        rawFeat{(mi-1)*2+1,2} = rawData;
    end
    
    % lower
    utrials = u.trials(lowerInd);
    firstLickTime = firstLickTimeAll(lowerInd);

    numTouchCell = cellfun(@(x,y) find(x.whiskerTime(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking)) < y), utrials, firstLickTime, 'uniformoutput', false);

    rawData = zeros(length(lowerInd), 15); % 12 whisker inputs + num whisk, whisk amplitude, object angle
    rawData(:,1) = cellfun(@(x,y) mean(x.theta(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
    rawData(:,2) = cellfun(@(x,y) mean(x.phi(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
    rawData(:,3) = cellfun(@(x,y) mean(x.kappaH(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
    rawData(:,4) = cellfun(@(x,y) -mean(x.kappaV(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell); % inverse the sign
    rawData(:,5) = cellfun(@(x,y) mean(x.arcLength(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking(y)))), utrials, numTouchCell);
    rawData(:,6) = cellfun(@length, numTouchCell);

    rawData(:,7) = cellfun(@(x,y) mean(x.protractionTouchDThetaByWhisking(y)), utrials, numTouchCell);
    rawData(:,8) = cellfun(@(x,y) mean(x.protractionTouchDPhiByWhisking(y)), utrials, numTouchCell);
    rawData(:,9) = cellfun(@(x,y) -mean(x.protractionTouchDKappaHByWhisking(y)), utrials, numTouchCell); % inverse the sign
    rawData(:,10) = cellfun(@(x,y) mean(x.protractionTouchDKappaVByWhisking(y)), utrials, numTouchCell);
    rawData(:,11) = cellfun(@(x,y) mean(x.protractionTouchDurationByWhisking(y)), utrials, numTouchCell);
    rawData(:,12) = cellfun(@(x,y) mean(x.protractionTouchSlideDistanceByWhisking(y)), utrials, numTouchCell);

    for ti = 1 : length(lowerInd)
        tempT = utrials{ti};
        poleUpInd = find(tempT.whiskerTime > tempT.poleUpTime(1),1,'first');
        lickInd = find(tempT.whiskerTime < firstLickTime{ti},1,'last');
        theta = tempT.theta(poleUpInd:lickInd);
        [onsetFrame, ~, ~, whiskingAmp, ~] = jkWhiskerOnsetNAmplitude(theta);
        rawData(ti,13) = length(onsetFrame);
        rawData(ti,14) = mean(whiskingAmp(find(whiskingAmp>2.5)));
    end

    rawData(:,15) = cellfun(@(x) x.angle, utrials);
    rawFeat{mi*2,2} = rawData;
end
rawFeat(3,:) = [];

save(saveFn,'rawFeat')




%%
%% Look at more basic whisking parameters
%%
% # of whisks, whisking amplitude, # of touches, mean touch duration

if ~exist('rawFeat', 'var')
    load('rawWhiskerFeatures_answerLick')
end
numVol = size(rawFeat,1);

figure,
subplot(141), hold on
tempMat = cellfun(@(x) mean(x(:,13)), rawFeat);
for i = 1 : numVol
    plot(tempMat(i,:), 'ko-')
end
errorbar(mean(tempMat), sem(tempMat), 'ro')
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
[~,p,m] = paired_test(tempMat(:,1), tempMat(:,2));
ylabel('Mean number of whisks')
title(sprintf('p = %.3f, m = %s', p, m))

subplot(142), hold on
tempMat = cellfun(@(x) nanmean(x(:,14)), rawFeat);
for i = 1 : numVol
    plot(tempMat(i,:), 'ko-')
end
errorbar(mean(tempMat), sem(tempMat), 'ro')
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
[~,p,m] = paired_test(tempMat(:,1), tempMat(:,2));
ylabel('Mean whisking amplitude (\circ)')
title(sprintf('p = %.3f, m = %s', p, m))

subplot(143), hold on
tempMat = cellfun(@(x) mean(x(:,6)), rawFeat);
for i = 1 : numVol
    plot(tempMat(i,:), 'ko-')
end
errorbar(mean(tempMat), sem(tempMat), 'ro')
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
[~,p,m] = paired_test(tempMat(:,1), tempMat(:,2));
ylabel('Mean number of touches')
title(sprintf('p = %.3f, m = %s', p, m))

subplot(144), hold on
tempMat = cellfun(@(x) mean(x(:,11)), rawFeat)*1000;
for i = 1 : numVol
    plot(tempMat(i,:), 'ko-')
end
errorbar(mean(tempMat), sem(tempMat), 'ro')
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
[~,p,m] = paired_test(tempMat(:,1), tempMat(:,2));
ylabel('Mean touch duration (ms)')
title(sprintf('p = %.3f, m = %s', p, m))
%%
%% Look at individual sensory inputs (trial-averaged) across angles
%%
% Similar to Fig 2C from Kim 2020 Neuron, but by 11 volumes (not 6 mice)

%% First, raw values
colorsTransient = [248 171 66; 40 170 225] / 255;
subplotPos = [1,2,5,6,9,10,3,4,7,8,11,12];
featNames = {'Azimuthal angle', 'Vertical angle', 'Horizontal curvature', 'Vertical curvature', 'Arc length', 'Touch count', ...
    'Push angle', 'Vertical displacement', 'Horizontal bending', 'Vertical bending', 'Touch duration', 'Slide distance'};
angles = 45:15:135;
figure('unit', 'inch', 'pos', [0 0 13 7])
for spi = 1 : 12
    subplot(3,4,subplotPos(spi)), hold on
    tempNaive = zeros(numVol, length(angles));
    tempExpert = zeros(numVol, length(angles));
    % naive
    for vi = 1 : numVol
        for ai = 1 : length(angles)
            tInd = find(rawFeat{vi,1}(:,15) == angles(ai));
            tempNaive(vi,ai) = nanmean(rawFeat{vi,1}(tInd,spi));
        end
    end
    % expert
    for vi = 1 : numVol
        for ai = 1 : length(angles)
            tInd = find(rawFeat{vi,2}(:,15) == angles(ai));
            tempExpert(vi,ai) = nanmean(rawFeat{vi,2}(tInd,spi));
        end
    end
    
    plot(angles, mean(tempNaive), 'color', colorsTransient(1,:))
    plot(angles, mean(tempExpert), 'color', colorsTransient(2,:))
    if spi == 8
        legend({'Naive', 'Expert'}, 'autoupdate', 'off', 'location', 'northwest')
    end
    
    boundedline(angles, mean(tempNaive), sem(tempNaive), 'cmap', colorsTransient(1,:))
    boundedline(angles, mean(tempExpert), sem(tempExpert), 'cmap', colorsTransient(2,:))
    xlim([40 140]), xticks(angles)
    if subplotPos(spi) < 9
        xticklabels('')
    end
    title(featNames{spi})
    
end

sgtitle('Raw values')

%% (2) across-session standardization 

figure('unit', 'inch', 'pos', [0 0 13 7])
for spi = 1 : 12
    subplot(3,4,subplotPos(spi)), hold on
    stdNaive = zeros(numVol, length(angles));
    stdExpert = zeros(numVol, length(angles));
    % naive
    for vi = 1 : numVol
        tempAll = [rawFeat{vi,1}(:,spi); rawFeat{vi,2}(:,spi)];
        tempStd = (tempAll - nanmean(tempAll)) / nanstd(tempAll);
        tempNaive = tempStd(1:size(rawFeat{vi,1},1));
        tempExpert = tempStd(size(rawFeat{vi,1},1)+1:end);
        for ai = 1 : length(angles)
            % naive
            tInd = find(rawFeat{vi,1}(:,15) == angles(ai));
            stdNaive(vi,ai) = nanmean(tempNaive(tInd));
            % expert
            tInd = find(rawFeat{vi,2}(:,15) == angles(ai));
            stdExpert(vi,ai) = nanmean(tempExpert(tInd));
        end
    end
    
    plot(angles, mean(stdNaive), 'color', colorsTransient(1,:))
    plot(angles, mean(stdExpert), 'color', colorsTransient(2,:))
    if spi == 8
        legend({'Naive', 'Expert'}, 'autoupdate', 'off', 'location', 'northwest')
    end
    
    boundedline(angles, mean(stdNaive), sem(stdNaive), 'cmap', colorsTransient(1,:))
    boundedline(angles, mean(stdExpert), sem(stdExpert), 'cmap', colorsTransient(2,:))
    xlim([40 140]), xticks(angles)
    if subplotPos(spi) < 9
        xticklabels('')
    end
    title(featNames{spi})
    
end

sgtitle('Across-session standardization')



%% (3) within-session standardization
figure('unit', 'inch', 'pos', [0 0 13 7])
for spi = 1 : 12
    subplot(3,4,subplotPos(spi)), hold on
    stdNaive = zeros(numVol, length(angles));
    stdExpert = zeros(numVol, length(angles));
    % naive
    for vi = 1 : numVol
        tempAll = [rawFeat{vi,1}(:,spi); rawFeat{vi,2}(:,spi)];
        tempStd = (tempAll - nanmean(tempAll)) / nanstd(tempAll);
        tempNaive = rawFeat{vi,1}(:,spi);
        tempNaive = (tempNaive - nanmean(tempNaive)) / nanstd(tempNaive);
        tempExpert = rawFeat{vi,2}(:,spi);
        tempExpert = (tempExpert - nanmean(tempExpert)) / nanstd(tempExpert);
        for ai = 1 : length(angles)
            % naive
            tInd = find(rawFeat{vi,1}(:,15) == angles(ai));
            stdNaive(vi,ai) = nanmean(tempNaive(tInd));
            % expert
            tInd = find(rawFeat{vi,2}(:,15) == angles(ai));
            stdExpert(vi,ai) = nanmean(tempExpert(tInd));
        end
    end
    
    plot(angles, mean(stdNaive), 'color', colorsTransient(1,:))
    plot(angles, mean(stdExpert), 'color', colorsTransient(2,:))
    if spi == 8
        legend({'Naive', 'Expert'}, 'autoupdate', 'off', 'location', 'northwest')
    end
    
    boundedline(angles, mean(stdNaive), sem(stdNaive), 'cmap', colorsTransient(1,:))
    boundedline(angles, mean(stdExpert), sem(stdExpert), 'cmap', colorsTransient(2,:))
    xlim([40 140]), xticks(angles)
    if subplotPos(spi) < 9
        xticklabels('')
    end
    ylim([-1.5 1.5])
    title(featNames{spi})
    
end

sgtitle('Within-session standardization')









%%
%% Other classifiers using sensory inputs
%%
% LDA, SVM, KNN, Random forest
if ~exist('rawFeat', 'var')
    load('rawWhiskerFeatures_answerLick')
end
numVol = size(rawFeat,1);
stdX = cellfun(@(x) (x(:,1:12) - nanmean(x(:,1:12)))./nanstd(x(:,1:12)), rawFeat, 'un', 0);
Y = cellfun(@(x) x(:,15), rawFeat, 'un', 0);
%% LDA

correctRate = zeros(numVol,2); %(:,1) naive, (:,2) expert
errorAngle = zeros(numVol,2); 
shuffleCR = zeros(numVol,2); 
shuffleEA = zeros(numVol,2); 
for vi = 1 : numVol
    for si = 1 : 2
        % 5-fold cross-validation
        numFold = 5;
        foldInds = cell(numFold,1);
        for fi = 1 : numFold
            foldInds{fi} = [];
        end
        tempX = stdX{vi,si};
        tempY = Y{vi,si};

        % stratification
        for ai = 1 : length(angles)
            aInd = find(tempY == angles(ai));
            randInd = aInd(randperm(length(aInd)));
            inds = round(linspace(0, length(randInd), numFold+1));
            for fi = 1 : length(inds)-1
                foldInds{fi} = [foldInds{fi}; randInd(inds(fi)+1:inds(fi+1))];
            end
        end

        foldCR = zeros(numFold,1);
        foldEA = zeros(numFold,1);
        shFoldCR = zeros(numFold,1);
        shFoldEA = zeros(numFold,1);
        numShuffle = 100;
        for fi = 1 : numFold
            trainFi = setdiff(1:numFold, fi); % training fold index
            trainInd = cell2mat(foldInds(trainFi));
            testInd = sort(foldInds{fi});

            trainX = tempX(trainInd,:);
            trainY = tempY(trainInd,:);

            testX = tempX(testInd,:);
            testY = tempY(testInd,:);

            mdl = fitcdiscr(trainX, trainY, 'Prior', 'uniform');
            label = predict(mdl, testX);
            foldCR(fi) = length(find((label-testY)==0)) / length(testY);
            foldEA(fi) = mean(abs(label-testY));

            % Shuffling
            tempPerf = zeros(numShuffle,1);
            tempAngle = zeros(numShuffle,1);
            for shi = 1 : numShuffle
                tempShY = testY(randperm(length(testY)));
                tempPerf(shi) = length(find((label-tempShY)==0)) / length(tempShY);
                tempAngle(shi) = mean(abs(label-tempShY));
            end
            shFoldCR(fi) = mean(tempPerf);
            shFoldEA(fi) = mean(tempAngle);
        end
        
        correctRate(vi,si) = mean(foldCR);
        errorAngle(vi,si) = mean(foldEA);
        shuffleCR(vi,si) = mean(shFoldCR);
        shuffleEA(vi,si) = mean(shFoldEA);
    end
end


figure, 
subplot(121), hold on
for vi = 1 : numVol
    plot(correctRate(vi,:), 'ko-')
end
errorbar(mean(correctRate), sem(correctRate), 'ro')
errorbar(mean(shuffleCR), sem(shuffleCR), 'o', 'color', [0.6 0.6 0.6])
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
ylabel('Correct Rate')
[~,p,m] = paired_test(correctRate(:,1), correctRate(:,2));
title(sprintf('p = %.3f (%s)', p, m))

subplot(122), hold on
for vi = 1 : numVol
    plot(errorAngle(vi,:), 'ko-')
end
errorbar(mean(errorAngle), sem(errorAngle), 'ro')
errorbar(mean(shuffleEA), sem(shuffleEA), 'o', 'color', [0.6 0.6 0.6])
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
ylabel('Error angle (\circ)')
[~,p,m] = paired_test(errorAngle(:,1), errorAngle(:,2));
title(sprintf('p = %.3f (%s)', p, m))


%% SVM

correctRate_LDA = zeros(numVol,2); %(:,1) naive, (:,2) expert
errorAngle_LDA = zeros(numVol,2);
shuffleCR_LDA = zeros(numVol,2);
shuffleEA_LDA = zeros(numVol,2);

correctRate_SVM = zeros(numVol,2); %(:,1) naive, (:,2) expert
errorAngle_SVM = zeros(numVol,2);
shuffleCR_SVM = zeros(numVol,2);
shuffleEA_SVM = zeros(numVol,2);

correctRate_KNN = zeros(numVol,2); %(:,1) naive, (:,2) expert
errorAngle_KNN = zeros(numVol,2);
shuffleCR_KNN = zeros(numVol,2);
shuffleEA_KNN = zeros(numVol,2);

for vi = 1 : numVol
    fprintf('Processing volume #%d/%d\n', vi, numVol)
    for si = 1 : 2
        if si == 1
            disp('Naive')
        else
            disp('Expert')
        end
        % 5-fold cross-validation
        numFold = 5;
        foldInds = cell(numFold,1);
        for fi = 1 : numFold
            foldInds{fi} = [];
        end
        tempX = stdX{vi,si};
        tempY = Y{vi,si};

        % stratification
        for ai = 1 : length(angles)
            aInd = find(tempY == angles(ai));
            randInd = aInd(randperm(length(aInd)));
            inds = round(linspace(0, length(randInd), numFold+1));
            for fi = 1 : length(inds)-1
                foldInds{fi} = [foldInds{fi}; randInd(inds(fi)+1:inds(fi+1))];
            end
        end

        % LDA
        disp('LDA')
        foldCR = zeros(numFold,1);
        foldEA = zeros(numFold,1);
        shFoldCR = zeros(numFold,1);
        shFoldEA = zeros(numFold,1);
        numShuffle = 100;
        for fi = 1 : numFold
            trainFi = setdiff(1:numFold, fi); % training fold index
            trainInd = cell2mat(foldInds(trainFi));
            testInd = sort(foldInds{fi});

            trainX = tempX(trainInd,:);
            trainY = tempY(trainInd,:);

            testX = tempX(testInd,:);
            testY = tempY(testInd,:);

            mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'discriminant', 'OptimizeHyperparameters', {'Delta', 'Gamma'}, 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
%             mdl = fitcdiscr(trainX, trainY, 'Prior', 'uniform');
            label = predict(mdl, testX);
            foldCR(fi) = length(find((label-testY)==0)) / length(testY);
            foldEA(fi) = mean(abs(label-testY));

            % Shuffling
            tempPerf = zeros(numShuffle,1);
            tempAngle = zeros(numShuffle,1);
            for shi = 1 : numShuffle
                tempShY = testY(randperm(length(testY)));
                tempPerf(shi) = length(find((label-tempShY)==0)) / length(tempShY);
                tempAngle(shi) = mean(abs(label-tempShY));
            end
            shFoldCR(fi) = mean(tempPerf);
            shFoldEA(fi) = mean(tempAngle);
        end
        
        correctRate_LDA(vi,si) = mean(foldCR);
        errorAngle_LDA(vi,si) = mean(foldEA);
        shuffleCR_LDA(vi,si) = mean(shFoldCR);
        shuffleEA_LDA(vi,si) = mean(shFoldEA);
        
        % SVM
        disp('SVM')
        foldCR = zeros(numFold,1);
        foldEA = zeros(numFold,1);
        shFoldCR = zeros(numFold,1);
        shFoldEA = zeros(numFold,1);
        numShuffle = 100;
        for fi = 1 : numFold
            trainFi = setdiff(1:numFold, fi); % training fold index
            trainInd = cell2mat(foldInds(trainFi));
            testInd = sort(foldInds{fi});

            trainX = tempX(trainInd,:);
            trainY = tempY(trainInd,:);

            testX = tempX(testInd,:);
            testY = tempY(testInd,:);

            mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'svm', 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
%             mdl = fitcdiscr(trainX, trainY, 'Prior', 'uniform');
            label = predict(mdl, testX);
            foldCR(fi) = length(find((label-testY)==0)) / length(testY);
            foldEA(fi) = mean(abs(label-testY));

            % Shuffling
            tempPerf = zeros(numShuffle,1);
            tempAngle = zeros(numShuffle,1);
            for shi = 1 : numShuffle
                tempShY = testY(randperm(length(testY)));
                tempPerf(shi) = length(find((label-tempShY)==0)) / length(tempShY);
                tempAngle(shi) = mean(abs(label-tempShY));
            end
            shFoldCR(fi) = mean(tempPerf);
            shFoldEA(fi) = mean(tempAngle);
        end
        
        correctRate_SVM(vi,si) = mean(foldCR);
        errorAngle_SVM(vi,si) = mean(foldEA);
        shuffleCR_SVM(vi,si) = mean(shFoldCR);
        shuffleEA_SVM(vi,si) = mean(shFoldEA);
        
        % KNN
        disp('KNN')
        foldCR = zeros(numFold,1);
        foldEA = zeros(numFold,1);
        shFoldCR = zeros(numFold,1);
        shFoldEA = zeros(numFold,1);
        numShuffle = 100;
        for fi = 1 : numFold
            trainFi = setdiff(1:numFold, fi); % training fold index
            trainInd = cell2mat(foldInds(trainFi));
            testInd = sort(foldInds{fi});

            trainX = tempX(trainInd,:);
            trainY = tempY(trainInd,:);

            testX = tempX(testInd,:);
            testY = tempY(testInd,:);

            mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'knn', 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
%             mdl = fitcdiscr(trainX, trainY, 'Prior', 'uniform');
            label = predict(mdl, testX);
            foldCR(fi) = length(find((label-testY)==0)) / length(testY);
            foldEA(fi) = mean(abs(label-testY));

            % Shuffling
            tempPerf = zeros(numShuffle,1);
            tempAngle = zeros(numShuffle,1);
            for shi = 1 : numShuffle
                tempShY = testY(randperm(length(testY)));
                tempPerf(shi) = length(find((label-tempShY)==0)) / length(tempShY);
                tempAngle(shi) = mean(abs(label-tempShY));
            end
            shFoldCR(fi) = mean(tempPerf);
            shFoldEA(fi) = mean(tempAngle);
        end
        
        correctRate_KNN(vi,si) = mean(foldCR);
        errorAngle_KNN(vi,si) = mean(foldEA);
        shuffleCR_KNN(vi,si) = mean(shFoldCR);
        shuffleEA_KNN(vi,si) = mean(shFoldEA);        
    end
end


figure, 
subplot(121), hold on
for vi = 1 : numVol
    plot(correctRate_LDA(vi,:), 'ko-')
end
errorbar(mean(correctRate_LDA), sem(correctRate_LDA), 'ro')
errorbar(mean(shuffleCR_LDA), sem(shuffleCR_LDA), 'o', 'color', [0.6 0.6 0.6])
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
ylabel('Correct Rate')
[~,p,m] = paired_test(correctRate_LDA(:,1), correctRate_LDA(:,2));
title(sprintf('p = %.3f (%s)', p, m))

subplot(122), hold on
for vi = 1 : numVol
    plot(errorAngle_LDA(vi,:), 'ko-')
end
errorbar(mean(errorAngle_LDA), sem(errorAngle_LDA), 'ro')
errorbar(mean(shuffleEA_LDA), sem(shuffleEA_LDA), 'o', 'color', [0.6 0.6 0.6])
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
ylabel('Error angle (\circ)')
[~,p,m] = paired_test(errorAngle_LDA(:,1), errorAngle_LDA(:,2));
title(sprintf('p = %.3f (%s)', p, m))
sgtitle('LDA')

figure, 
subplot(121), hold on
for vi = 1 : numVol
    plot(correctRate_SVM(vi,:), 'ko-')
end
errorbar(mean(correctRate_SVM), sem(correctRate_SVM), 'ro')
errorbar(mean(shuffleCR_SVM), sem(shuffleCR_SVM), 'o', 'color', [0.6 0.6 0.6])
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
ylabel('Correct Rate')
[~,p,m] = paired_test(correctRate_SVM(:,1), correctRate_SVM(:,2));
title(sprintf('p = %.3f (%s)', p, m))

subplot(122), hold on
for vi = 1 : numVol
    plot(errorAngle_SVM(vi,:), 'ko-')
end
errorbar(mean(errorAngle_SVM), sem(errorAngle_SVM), 'ro')
errorbar(mean(shuffleEA_SVM), sem(shuffleEA_SVM), 'o', 'color', [0.6 0.6 0.6])
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
ylabel('Error angle (\circ)')
[~,p,m] = paired_test(errorAngle_SVM(:,1), errorAngle_SVM(:,2));
title(sprintf('p = %.3f (%s)', p, m))
sgtitle('SVM')

figure, 
subplot(121), hold on
for vi = 1 : numVol
    plot(correctRate_KNN(vi,:), 'ko-')
end
errorbar(mean(correctRate_KNN), sem(correctRate_KNN), 'ro')
errorbar(mean(shuffleCR_KNN), sem(shuffleCR_KNN), 'o', 'color', [0.6 0.6 0.6])
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
ylabel('Correct Rate')
[~,p,m] = paired_test(correctRate_KNN(:,1), correctRate_KNN(:,2));
title(sprintf('p = %.3f (%s)', p, m))

subplot(122), hold on
for vi = 1 : numVol
    plot(errorAngle_KNN(vi,:), 'ko-')
end
errorbar(mean(errorAngle_KNN), sem(errorAngle_KNN), 'ro')
errorbar(mean(shuffleEA_KNN), sem(shuffleEA_KNN), 'o', 'color', [0.6 0.6 0.6])
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
ylabel('Error angle (\circ)')
[~,p,m] = paired_test(errorAngle_KNN(:,1), errorAngle_KNN(:,2));
title(sprintf('p = %.3f (%s)', p, m))
sgtitle('KNN')





















%%
% 
% %% 
% %% Object angle prediction using model performance from cross validation
% %%
% % load([baseDir, 'whiskerAngleData'], 'whiskerAngleData');
% angles = 45:15:135;
% perfLDA = zeros(numVol,2);
% perfLDAshuffle = zeros(numVol,2);
% perfSVM = zeros(numVol,2);
% perfSVMshuffle = zeros(numVol,2);
% perfKNN = zeros(numVol,2);
% perfKNNshuffle = zeros(numVol,2);
% perfRF = zeros(numVol,2);
% perfRFshuffle = zeros(numVol,2);
% shuffleIterNum = 100;
% for vi = 1 : numVol
%     fprintf('Processing volume #%d/%d\n', vi, numVol)
%     
%     % naive
%     disp('Naive')
%     data = whiskerAngleData{vi,1};
%     input = data(:,1:end-1);
%     output = data(:,end);
%     
%     [~, perfLDA(vi,1)] = angle_tuning_func_reorg_LDA(data, angles);
%     [~, perfSVM(vi,1)] = angle_tuning_func_reorg_SVM(data, angles);
%     [~, perfKNN(vi,1)] = angle_tuning_func_reorg_KNN(data, angles);
%     [~, perfRF(vi,1)] = angle_tuning_func_reorg_randForest(data, angles);
%     naiveShuffle = zeros(shuffleIterNum,4);
%     parfor shi = 1 : shuffleIterNum
%         shuffledOutput = output(randperm(length(output)));
%         tempShuffle = zeros(1,4);
%         [~, tempShuffle(1)] = angle_tuning_func_reorg_LDA([input,shuffledOutput], angles);
%         [~, tempShuffle(2)] = angle_tuning_func_reorg_SVM([input,shuffledOutput], angles);
%         [~, tempShuffle(3)] = angle_tuning_func_reorg_KNN([input,shuffledOutput], angles);
%         [~, tempShuffle(4)] = angle_tuning_func_reorg_randForest([input,shuffledOutput], angles);
%         naiveShuffle(shi,:) = tempShuffle;
%     end
%     perfLDAshuffle(vi,1) = median(naiveShuffle(:,1));
%     perfSVMshuffle(vi,1) = median(naiveShuffle(:,2));
%     perfKNNshuffle(vi,1) = median(naiveShuffle(:,3));
%     perfRFshuffle(vi,1) = median(naiveShuffle(:,4));
%     
%     % expert
%     disp('Expert')
%     data = whiskerAngleData{vi,2};
%     input = data(:,1:end-1);
%     output = data(:,end);
%     
%     [~, perfLDA(vi,2)] = angle_tuning_func_reorg_LDA(data, angles);
%     [~, perfSVM(vi,2)] = angle_tuning_func_reorg_SVM(data, angles);
%     [~, perfKNN(vi,2)] = angle_tuning_func_reorg_KNN(data, angles);
%     [~, perfRF(vi,2)] = angle_tuning_func_reorg_randForest(data, angles);
%     expertShuffle = zeros(shuffleIterNum,4);
%     parfor shi = 1 : shuffleIterNum
%         shuffledOutput = output(randperm(length(output)));
%         tempShuffle = zeros(1,4);
%         [~, tempShuffle(1)] = angle_tuning_func_reorg_LDA([input,shuffledOutput], angles);
%         [~, tempShuffle(2)] = angle_tuning_func_reorg_SVM([input,shuffledOutput], angles);
%         [~, tempShuffle(3)] = angle_tuning_func_reorg_KNN([input,shuffledOutput], angles);
%         [~, tempShuffle(4)] = angle_tuning_func_reorg_randForest([input,shuffledOutput], angles);
%         expertShuffle(shi,:) = tempShuffle;
%     end
%     perfLDAshuffle(vi,2) = median(expertShuffle(:,1));
%     perfSVMshuffle(vi,2) = median(expertShuffle(:,2));
%     perfKNNshuffle(vi,2) = median(expertShuffle(:,3));
%     perfRFshuffle(vi,2) = median(expertShuffle(:,4));
% end
% 
% 
% %%
% figure, 
% subplot(141), hold on
% for vi = 1 : numVol
%     plot([1,2], perfLDA(vi,:), 'ko-')
% end
% errorbar([1,2], mean(perfLDA), sem(perfLDA), 'ro', 'lines', 'no')
% errorbar([1,2], mean(perfLDAshuffle), sem(perfLDAshuffle), 'o', 'lines', 'no', 'color', [0.6 0.6 0.6])
% xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'})
% ylabel('Performance')
% [~,p,m] = paired_test(perfLDA(:,1), perfLDA(:,2));
% title({'LDA';sprintf('p = %.3f, %s', p, m)})
% 
% subplot(142), hold on
% for vi = 1 : numVol
%     plot([1,2], perfSVM(vi,:), 'ko-')
% end
% errorbar([1,2], mean(perfSVM), sem(perfSVM), 'ro', 'lines', 'no')
% errorbar([1,2], mean(perfSVMshuffle), sem(perfSVMshuffle), 'o', 'lines', 'no', 'color', [0.6 0.6 0.6])
% xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'})
% ylabel('Performance')
% [~,p,m] = paired_test(perfSVM(:,1), perfSVM(:,2));
% title({'SVM';sprintf('p = %.3f, %s', p, m)})
% 
% 
% subplot(143), hold on
% for vi = 1 : numVol
%     plot([1,2], perfKNN(vi,:), 'ko-')
% end
% errorbar([1,2], mean(perfKNN), sem(perfKNN), 'ro', 'lines', 'no')
% errorbar([1,2], mean(perfKNNshuffle), sem(perfKNNshuffle), 'o', 'lines', 'no', 'color', [0.6 0.6 0.6])
% xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'})
% ylabel('Performance')
% [~,p,m] = paired_test(perfKNN(:,1), perfKNN(:,2));
% title({'KNN';sprintf('p = %.3f, %s', p, m)})
% 
% 
% 
% subplot(144), hold on
% for vi = 1 : numVol
%     plot([1,2], perfRF(vi,:), 'ko-')
% end
% errorbar([1,2], mean(perfRF), sem(perfRF), 'ro', 'lines', 'no')
% errorbar([1,2], mean(perfRFshuffle), sem(perfRFshuffle), 'o', 'lines', 'no', 'color', [0.6 0.6 0.6])
% xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'})
% ylabel('Performance')
% [~,p,m] = paired_test(perfRF(:,1), perfRF(:,2));
% title({'Random forest';sprintf('p = %.3f, %s', p, m)})
% 
% 
% 
% 
% 
% 
% 
% %%
% %% Performance calculation using the model and bootstrapping to match the # of samples per angle
% %%
% % load([baseDir, 'whiskerAngleData'], 'whiskerAngleData');
% angles = 45:15:135;
% crLDA = zeros(numVol,2);
% crLDAshuffle = zeros(numVol,2);
% peLDA = zeros(numVol,2);
% peLDAshuffle = zeros(numVol,2);
% confMatLDAAll = cell(numVol,2);
% 
% crSVM = zeros(numVol,2);
% crSVMshuffle = zeros(numVol,2);
% peSVM = zeros(numVol,2);
% peSVMshuffle = zeros(numVol,2);
% confMatSVMAll = cell(numVol,2);
% 
% crKNN = zeros(numVol,2);
% crKNNshuffle = zeros(numVol,2);
% peKNN = zeros(numVol,2);
% peKNNshuffle = zeros(numVol,2);
% confMatKNNAll = cell(numVol,2);
% 
% crRF = zeros(numVol,2);
% crRFshuffle = zeros(numVol,2);
% peRF = zeros(numVol,2);
% peRFshuffle = zeros(numVol,2);
% confMatRFAll = cell(numVol,2);
% 
% for vi = 1 : numVol
%     fprintf('Processing volume #%d/%d\n', vi, numVol)
%     
%     % naive
%     disp('Naive')
%     data = whiskerAngleData{vi,1};
%     input = data(:,1:end-1);
%     output = data(:,end);
%     
%     model = angle_tuning_func_reorg_LDA(data, angles);
%     pred = model.predictFcn(input);
%     ypair = [pred, output];
%     [crLDA(vi,1), crLDAshuffle(vi,1), peLDA(vi,1), peLDAshuffle(vi,1), confMatLDAAll{vi,1}] = prediction_bootstrapping(ypair, angles);
%     
%     model = angle_tuning_func_reorg_SVM(data, angles);
%     pred = model.predictFcn(input);
%     ypair = [pred, output];
%     [crSVM(vi,1), crSVMshuffle(vi,1), peSVM(vi,1), peSVMshuffle(vi,1), confMatSVMAll{vi,1}] = prediction_bootstrapping(ypair, angles);
%     
%     model = angle_tuning_func_reorg_KNN(data, angles);
%     pred = model.predictFcn(input);
%     ypair = [pred, output];
%     [crKNN(vi,1), crKNNshuffle(vi,1), peKNN(vi,1), peKNNshuffle(vi,1), confMatKNNAll{vi,1}] = prediction_bootstrapping(ypair, angles);
%     
%     model = angle_tuning_func_reorg_RF(data, angles);
%     pred = model.predictFcn(input);
%     ypair = [pred, output];
%     [crRF(vi,1), crRFshuffle(vi,1), peRF(vi,1), peRFshuffle(vi,1), confMatRFAll{vi,1}] = prediction_bootstrapping(ypair, angles);
%     
%     % expert
%     disp('Expert')
%     data = whiskerAngleData{vi,2};
%     input = data(:,1:end-1);
%     output = data(:,end);
%     
%     model = angle_tuning_func_reorg_LDA(data, angles);
%     pred = model.predictFcn(input);
%     ypair = [pred, output];
%     [crLDA(vi,2), crLDAshuffle(vi,2), peLDA(vi,2), peLDAshuffle(vi,2), confMatLDAAll{vi,2}] = prediction_bootstrapping(ypair, angles);
%     
%     model = angle_tuning_func_reorg_SVM(data, angles);
%     pred = model.predictFcn(input);
%     ypair = [pred, output];
%     [crSVM(vi,2), crSVMshuffle(vi,2), peSVM(vi,2), peSVMshuffle(vi,2), confMatSVMAll{vi,2}] = prediction_bootstrapping(ypair, angles);
%     
%     model = angle_tuning_func_reorg_KNN(data, angles);
%     pred = model.predictFcn(input);
%     ypair = [pred, output];
%     [crKNN(vi,2), crKNNshuffle(vi,2), peKNN(vi,2), peKNNshuffle(vi,2), confMatKNNAll{vi,2}] = prediction_bootstrapping(ypair, angles);
%     
%     model = angle_tuning_func_reorg_RF(data, angles);
%     pred = model.predictFcn(input);
%     ypair = [pred, output];
%     [crRF(vi,2), crRFshuffle(vi,2), peRF(vi,2), peRFshuffle(vi,2), confMatRFAll{vi,2}] = prediction_bootstrapping(ypair, angles);
%     
% end
% 
% 
% %%
% figure, 
% subplot(141), hold on
% for vi = 1 : numVol
%     plot([1,2], crLDA(vi,:), 'ko-')
% end
% errorbar([1,2], mean(crLDA), sem(crLDA), 'ro', 'lines', 'no')
% errorbar([1,2], mean(crLDAshuffle), sem(crLDAshuffle), 'o', 'lines', 'no', 'color', [0.6 0.6 0.6])
% xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'})
% ylabel('Performance')
% [~,p,m] = paired_test(crLDA(:,1), crLDA(:,2));
% title({'LDA';sprintf('p = %.3f, %s', p, m)})
% 
% 
% subplot(142), hold on
% for vi = 1 : numVol
%     plot([1,2], crSVM(vi,:), 'ko-')
% end
% errorbar([1,2], mean(crSVM), sem(crSVM), 'ro', 'lines', 'no')
% errorbar([1,2], mean(crSVMshuffle), sem(crSVMshuffle), 'o', 'lines', 'no', 'color', [0.6 0.6 0.6])
% xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'})
% ylabel('Performance')
% [~,p,m] = paired_test(crSVM(:,1), crSVM(:,2));
% title({'SVM';sprintf('p = %.3f, %s', p, m)})
% 
% 
% subplot(143), hold on
% for vi = 1 : numVol
%     plot([1,2], crKNN(vi,:), 'ko-')
% end
% errorbar([1,2], mean(crKNN), sem(crKNN), 'ro', 'lines', 'no')
% errorbar([1,2], mean(crKNNshuffle), sem(crKNNshuffle), 'o', 'lines', 'no', 'color', [0.6 0.6 0.6])
% xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'})
% ylabel('Performance')
% [~,p,m] = paired_test(crKNN(:,1), crKNN(:,2));
% title({'KNN';sprintf('p = %.3f, %s', p, m)})
% 
% 
% subplot(144), hold on
% for vi = 1 : numVol
%     plot([1,2], crRF(vi,:), 'ko-')
% end
% errorbar([1,2], mean(crRF), sem(crRF), 'ro', 'lines', 'no')
% errorbar([1,2], mean(crRFshuffle), sem(crRFshuffle), 'o', 'lines', 'no', 'color', [0.6 0.6 0.6])
% xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'})
% ylabel('Performance')
% [~,p,m] = paired_test(crRF(:,1), crRF(:,2));
% title({'Random forest';sprintf('p = %.3f, %s', p, m)})
% 
% 
% 
% 
% 
% 
% 
% %%
% %% Performance calculation using simply the model prediction
% %%
% % load([baseDir, 'whiskerAngleData'], 'whiskerAngleData');
% angles = 45:15:135;
% crLDA = zeros(numVol,2);
% crLDAshuffle = zeros(numVol,2);
% peLDA = zeros(numVol,2);
% peLDAshuffle = zeros(numVol,2);
% confMatLDAAll = cell(numVol,2);
% 
% crSVM = zeros(numVol,2);
% crSVMshuffle = zeros(numVol,2);
% peSVM = zeros(numVol,2);
% peSVMshuffle = zeros(numVol,2);
% confMatSVMAll = cell(numVol,2);
% 
% crKNN = zeros(numVol,2);
% crKNNshuffle = zeros(numVol,2);
% peKNN = zeros(numVol,2);
% peKNNshuffle = zeros(numVol,2);
% confMatKNNAll = cell(numVol,2);
% 
% crRF = zeros(numVol,2);
% crRFshuffle = zeros(numVol,2);
% peRF = zeros(numVol,2);
% peRFshuffle = zeros(numVol,2);
% confMatRFAll = cell(numVol,2);
% 
% for vi = 1 : numVol
%     fprintf('Processing volume #%d/%d\n', vi, numVol)
%     
%     % naive
%     disp('Naive')
%     data = whiskerAngleData{vi,1};
%     input = data(:,1:end-1);
%     output = data(:,end);
%     
%     model = angle_tuning_func_reorg_LDA(data, angles);
%     pred = model.predictFcn(input);
%     crLDA(vi,1) = length(find((pred - output)==0)) / length(pred);
%     peLDA(vi,1) = mean(abs(pred-output));
%     
%     model = angle_tuning_func_reorg_SVM(data, angles);
%     pred = model.predictFcn(input);
%     crSVM(vi,1) = length(find((pred - output)==0)) / length(pred);
%     
%     model = angle_tuning_func_reorg_KNN(data, angles);
%     pred = model.predictFcn(input);
%     crKNN(vi,1) = length(find((pred - output)==0)) / length(pred);
%     
%     model = angle_tuning_func_reorg_RF(data, angles);
%     pred = model.predictFcn(input);
%     crRF(vi,1) = length(find((pred - output)==0)) / length(pred);
%     
%     % expert
%     disp('Expert')
%     data = whiskerAngleData{vi,2};
%     input = data(:,1:end-1);
%     output = data(:,end);
%     
%     model = angle_tuning_func_reorg_LDA(data, angles);
%     pred = model.predictFcn(input);
%     crLDA(vi,2) = length(find((pred - output)==0)) / length(pred);
%     
%     model = angle_tuning_func_reorg_SVM(data, angles);
%     pred = model.predictFcn(input);
%     crSVM(vi,2) = length(find((pred - output)==0)) / length(pred);
%     
%     model = angle_tuning_func_reorg_KNN(data, angles);
%     pred = model.predictFcn(input);
%     crKNN(vi,2) = length(find((pred - output)==0)) / length(pred);
%     
%     model = angle_tuning_func_reorg_RF(data, angles);
%     pred = model.predictFcn(input);
%     crRF(vi,2) = length(find((pred - output)==0)) / length(pred);
%     
% end
% 
% 
% %%
% figure, 
% subplot(141), hold on
% for vi = 1 : numVol
%     plot([1,2], crLDA(vi,:), 'ko-')
% end
% errorbar([1,2], mean(crLDA), sem(crLDA), 'ro', 'lines', 'no')
% % errorbar([1,2], mean(crLDAshuffle), sem(crLDAshuffle), 'o', 'lines', 'no', 'color', [0.6 0.6 0.6])
% xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'})
% ylabel('Performance')
% [~,p,m] = paired_test(crLDA(:,1), crLDA(:,2));
% title({'LDA';sprintf('p = %.3f, %s', p, m)})
% 
% 
% subplot(142), hold on
% for vi = 1 : numVol
%     plot([1,2], crSVM(vi,:), 'ko-')
% end
% errorbar([1,2], mean(crSVM), sem(crSVM), 'ro', 'lines', 'no')
% % errorbar([1,2], mean(crSVMshuffle), sem(crSVMshuffle), 'o', 'lines', 'no', 'color', [0.6 0.6 0.6])
% xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'})
% ylabel('Performance')
% [~,p,m] = paired_test(crSVM(:,1), crSVM(:,2));
% title({'SVM';sprintf('p = %.3f, %s', p, m)})
% 
% 
% subplot(143), hold on
% for vi = 1 : numVol
%     plot([1,2], crKNN(vi,:), 'ko-')
% end
% errorbar([1,2], mean(crKNN), sem(crKNN), 'ro', 'lines', 'no')
% % errorbar([1,2], mean(crKNNshuffle), sem(crKNNshuffle), 'o', 'lines', 'no', 'color', [0.6 0.6 0.6])
% xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'})
% ylabel('Performance')
% [~,p,m] = paired_test(crKNN(:,1), crKNN(:,2));
% title({'KNN';sprintf('p = %.3f, %s', p, m)})
% 
% 
% subplot(144), hold on
% for vi = 1 : numVol
%     plot([1,2], crRF(vi,:), 'ko-')
% end
% errorbar([1,2], mean(crRF), sem(crRF), 'ro', 'lines', 'no')
% % errorbar([1,2], mean(crRFshuffle), sem(crRFshuffle), 'o', 'lines', 'no', 'color', [0.6 0.6 0.6])
% xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'})
% ylabel('Performance')
% [~,p,m] = paired_test(crRF(:,1), crRF(:,2));
% title({'Random forest';sprintf('p = %.3f, %s', p, m)})








%%
%% Angle prediction using population activity
%%
% perfLDA = zeros(numVol,2);
% perfLDAshuffle = zeros(numVol,2);
% perfSVM = zeros(numVol,2);
% perfSVMshuffle = zeros(numVol,2);
% perfKNN = zeros(numVol,2);
% perfKNNshuffle = zeros(numVol,2);
% shuffleIterNum = 100;
% for vi = 1 : numVol
%     fprintf('Processing plane #%d/%d\n', pi, length(pcaResult))
%     
%     % naive
%     disp('Naive')
%     naiveAngles = matchedPR.naive(vi).trialAngle';
%     naivePopAct = matchedPR.naive(vi).poleBeforeAnswer;
% %     naivePopAct = matchedPR.naive(vi).poleBeforeFirstLick;
%     
%     [~, perfLDA(pi,1)] = angle_tuning_func_reorg_LDA([naivePopAct,naiveAngles], angles);
%     [~, perfSVM(pi,1)] = angle_tuning_func_reorg_SVM([naivePopAct,naiveAngles], angles);
%     [~, perfKNN(pi,1)] = angle_tuning_func_reorg_KNN([naivePopAct,naiveAngles], angles);
%     naiveShuffle = zeros(shuffleIterNum,3);
%     parfor shi = 1 : shuffleIterNum
%         shuffledAngles = naiveAngles(randperm(length(naiveAngles)));
%         tempShuffle = zeros(1,3);
%         [~, tempShuffle(1)] = angle_tuning_func_reorg_LDA([naiveCoord,shuffledAngles], angles);
%         [~, tempShuffle(2)] = angle_tuning_func_reorg_SVM([naiveCoord,shuffledAngles], angles);
%         [~, tempShuffle(3)] = angle_tuning_func_reorg_KNN([naiveCoord,shuffledAngles], angles);
%         naiveShuffle(shi,:) = tempShuffle;
%     end
%     perfLDAshuffle(pi,1) = median(naiveShuffle(:,1));
%     perfSVMshuffle(pi,1) = median(naiveShuffle(:,2));
%     perfKNNshuffle(pi,1) = median(naiveShuffle(:,3));
%     
%     
%     % expert
%     disp('Expert')
%     expertAngles = popAct.sortedAngleExpert{vi,mi};
% %     expertCoord = pcaResult(pi).pcaCoordSpkExpert(:,1:3);
%     expertCoord = popAct.spkExpert{vi,mi};
%     
%     [~, perfLDA(pi,2)] = angle_tuning_func_reorg_LDA([expertCoord,expertAngles], angles);
%     [~, perfSVM(pi,2)] = angle_tuning_func_reorg_SVM([expertCoord,expertAngles], angles);
%     [~, perfKNN(pi,2)] = angle_tuning_func_reorg_KNN([expertCoord,expertAngles], angles);
%     expertShuffle = zeros(shuffleIterNum,3);
%     parfor shi = 1 : shuffleIterNum
%         shuffledAngles = expertAngles(randperm(length(expertAngles)));
%         tempShuffle = zeros(1,3);
%         [~, tempShuffle(1)] = angle_tuning_func_reorg_LDA([expertCoord,shuffledAngles], angles);
%         [~, tempShuffle(2)] = angle_tuning_func_reorg_SVM([expertCoord,shuffledAngles], angles);
%         [~, tempShuffle(3)] = angle_tuning_func_reorg_KNN([expertCoord,shuffledAngles], angles);
%         expertShuffle(shi,:) = tempShuffle;
%     end
%     perfLDAshuffle(pi,2) = median(expertShuffle(:,1));
%     perfSVMshuffle(pi,2) = median(expertShuffle(:,2));
%     perfKNNshuffle(pi,2) = median(expertShuffle(:,3));
% end
% 
