% Population decoding of object angles (7-angle sessions) from touch responses
% First, use trial-averaged touch responses before answer lick time (or pole down onset time) from ALL active neurons in L2/3, 
% before and after learning

% save data and analysis results in Y NAS

compName = getenv('computername');
if strcmp(compName, 'HNB228-JINHO')
    baseDir = 'D:\TPM\JK\suite2p\';
else
    baseDir = 'Y:\Whiskernas\JK\suite2p\';
end

mice = [25,27,30,36,39,52];
numMice = length(mice);
sessions = {[4,19],[3,10],[3,21],[1,17],[1,23],[3,21]};

cIdThresh = 5000;
cellDepthThresh = 350;
% touchFrameDelays = [0,1,2]'; % this is important. without 2 delays (i.e., [0,1]'), there's no tuned pattern
touchFrameDelays = 0;
numDim = 2;
showPopActivity = 0;

popActNaive = cell(2,numMice);
popActExpert = cell(2,numMice);

trialNumsNaive = cell(2,numMice);
trialNumsExpert = cell(2,numMice);
sortedAngleNaive = cell(2,numMice);
sortedAngleExpert = cell(2,numMice);
sortedTouchNaive = cell(2,numMice);
sortedTouchExpert = cell(2,numMice);
sortedCidNaive = cell(2,numMice);
sortedCidExpert = cell(2,numMice);

tsneNaive = cell(2,numMice);
tsneExpert = cell(2,numMice);

pcaNaive = cell(2,numMice);
pcaExpert = cell(2,numMice);

for mi = 1 : numMice
% for mi = 1
    mouse = mice(mi);
    for si = 1 : 2
%     for si = 1
        session = sessions{mi}(si);
        fprintf('Processing JK%03d S%02d\n', mouse, session)
        load(sprintf('%s%03d\\UberJK%03dS%02d_NC',baseDir, mouse, mouse, session), 'u')
        load(sprintf('%s%03d\\JK%03dS%02dangle_tuning_lasso_preAnswer_perTouch_spkOnly_NC', baseDir, mouse, mouse, session), 'spk')
        touchTrialsInd = find(cellfun(@(x) ~isempty(x.protractionTouchOnsetFramesByWhisking{1}), u.trials)); % all touches, not just before answer or pole down
        upperPlaneTouchTrialsInd = intersect(find(cellfun(@(x) ismember(1, x.planes), u.trials)), touchTrialsInd);
        lowerPlaneTouchTrialsInd = intersect(find(cellfun(@(x) ismember(5, x.planes), u.trials)), touchTrialsInd);
        L23CellInd = find(u.cellDepths <= cellDepthThresh);
        upperPlaneCID = u.cellNums(intersect(find(u.cellNums < cIdThresh), L23CellInd));        
        lowerPlaneCID = u.cellNums(intersect(find(u.cellNums > cIdThresh), L23CellInd));
        upperPlaneCind = find(ismember(u.trials{upperPlaneTouchTrialsInd(1)}.neuindSession, upperPlaneCID));
        lowerPlaneCind = find(ismember(u.trials{lowerPlaneTouchTrialsInd(1)}.neuindSession, lowerPlaneCID));
        
        sessionUpperTouchNan = nan(length(upperPlaneTouchTrialsInd), length(upperPlaneCID)); % nan rows after allocation will be removed
        sessionUpperNopoleNan = nan(length(upperPlaneTouchTrialsInd), length(upperPlaneCID));
        sessionUpperPoleNan = nan(length(upperPlaneTouchTrialsInd), length(upperPlaneCID));
        sessionUpperAllNan = nan(length(upperPlaneTouchTrialsInd), length(upperPlaneCID));
        sessionUpperAngleNan = nan(length(upperPlaneTouchTrialsInd),1);
        sessionUpperTrialNumNan = nan(length(upperPlaneTouchTrialsInd),1);
        
        sessionLowerTouchNan = nan(length(lowerPlaneTouchTrialsInd), length(lowerPlaneCID)); % nan rows after allocation will be removed
        sessionLowerNopoleNan = nan(length(lowerPlaneTouchTrialsInd), length(lowerPlaneCID));
        sessionLowerPoleNan = nan(length(lowerPlaneTouchTrialsInd), length(lowerPlaneCID));
        sessionLowerAllNan = nan(length(lowerPlaneTouchTrialsInd), length(lowerPlaneCID));
        sessionLowerAngleNan = nan(length(lowerPlaneTouchTrialsInd),1);
        sessionLowerTrialNumNan = nan(length(lowerPlaneTouchTrialsInd),1);
        
        %% first, upper plane L2/3
        if mouse ~= 27 % I don't have JK027 upper plane
            disp('Upper plane')
            for ti = 1 : length(upperPlaneTouchTrialsInd)
                trialInd = upperPlaneTouchTrialsInd(ti);
                uTrial = u.trials{trialInd};
                if ~isempty(uTrial.answerLickTime)
                    lastTime = uTrial.answerLickTime;
                else
                    lastTime = uTrial.poleDownOnsetTime;
                end
                lastTouchFrameCell = cellfun(@(x,y) find(x(y) < lastTime, 1, 'last'), uTrial.tpmTime, uTrial.protractionTouchOnsetFramesByWhisking, 'un', 0); % minimum # of tpm touch frames before answer or pole-down from each planes
                if min(cellfun(@length, lastTouchFrameCell)) > 0 % if all cell is not empty                
                    lastTouchFrame = min(cellfun(@max, lastTouchFrameCell));
                    if ~isempty(lastTouchFrame)                
                        trialTouchFramesBinaryAllNeurons = zeros(length(upperPlaneCID), size(uTrial.spk,2), 'logical');
                        trialSpontFramesBinaryAllNeurons = ones(length(upperPlaneCID), size(uTrial.spk,2), 'logical');
                        for pi = 1 : 4
                            planeCind = find(floor(upperPlaneCID/1000) == pi);
                            planeTouchFrames = unique(repmat(uTrial.protractionTouchOnsetFramesByWhisking{pi}(1:lastTouchFrame),size(touchFrameDelays,1), size(touchFrameDelays,2)) + touchFrameDelays);
                            trialTouchFramesBinaryAllNeurons(planeCind, planeTouchFrames) = 1;

                            planePoleUpFrame = find(uTrial.tpmTime{pi} > uTrial.poleUpOnsetTime, 1, 'first');
                            planePoleDownFrame = find(uTrial.tpmTime{pi} > uTrial.poleDownOnsetTime, 1, 'first');
                            trialSpontFramesBinaryAllNeurons(planeCind, planePoleUpFrame:planePoleDownFrame) = 0;
                        end
                        trialPopTouch = sum(uTrial.spk(upperPlaneCind,:) .* trialTouchFramesBinaryAllNeurons, 2) / lastTouchFrame;
                        sessionUpperTouchNan(ti,:) = trialPopTouch';
                        trialPopNopole = mean(uTrial.spk(upperPlaneCind,:) .* trialSpontFramesBinaryAllNeurons, 2);
                        sessionUpperNopoleNan(ti,:) = trialPopNopole';
                        trialPopPole = mean(uTrial.spk(upperPlaneCind,:) .* (1-trialSpontFramesBinaryAllNeurons), 2);
                        sessionUpperPoleNan(ti,:) = trialPopPole';
                        sessionUpperAllNan(ti,:) = mean(uTrial.spk(upperPlaneCind,:),2);
                        sessionUpperAngleNan(ti,:) = uTrial.angle;
                        sessionUpperTrialNumNan(ti,:) = uTrial.trialNum;
                    end
                end
            end
            noNanInd = find(~isnan(sessionUpperTouchNan(:,1)));
            sessionUpperTouch = sessionUpperTouchNan(noNanInd,:);
            sessionUpperNopole = sessionUpperNopoleNan(noNanInd,:);
            sessionUpperPole = sessionUpperPoleNan(noNanInd,:);
            sessionUpperAll = sessionUpperAllNan(noNanInd,:);
            sessionUpperAngle = sessionUpperAngleNan(noNanInd,:);
            sessionUpperTrialNum = sessionUpperTrialNumNan(noNanInd,:);

            %% Saving (and showing) population activity
    %         figure, subplot(411), imagesc(sessionUpperTouch), title('Touch frames, pre-answer'), subplot(413), imagesc(sessionUpperNopole), title('No pole period'), 
    %         subplot(412), imagesc(sessionUpperPole), title('Pole period'), subplot(414), imagesc(sessionUpperAll), title('All frames')

            [sortedAngle, sorti] = sort(sessionUpperAngle, 'ascend');
            sortedTrialNum = sessionUpperTrialNum(sorti);
            sessionUpperTouchSorted = sessionUpperTouch(sorti,:);
            sessionUpperNopoleSorted = sessionUpperNopole(sorti,:);
            sessionUpperPoleSorted = sessionUpperPole(sorti,:);
            sessionUpperAllSorted = sessionUpperAll(sorti,:);

    %         figure, subplot(411), imagesc(sessionUpperTouchSorted), title('Touch frames, pre-answer (sorted by angle(Y))'), subplot(413), imagesc(sessionUpperNopoleSorted), title('No pole period (sorted by angle(Y))'), 
    %         subplot(412), imagesc(sessionUpperPoleSorted), title('Pole period (sorted by angle(Y))'), subplot(414), imagesc(sessionUpperAllSorted), title('All frames (sorted by angle(Y))')

    %         %% sorting by tuned angle and activity level
            tunedAngle = zeros(length(upperPlaneCID),1); % if non-touch, 0, if non-selective touch, 1; otherwise, tuned angle (typically 45:15:135)
            tunedAngle(find(ismember(upperPlaneCID, spk.touchID))) = 1;
            angles = setdiff(unique(spk.tunedAngle),0); % all angles, sorted. Typically, 45:15:135.
            for ai = 1 : length(angles)
                tempAngle = angles(ai);
                indSpkAngle = find(spk.tunedAngle == tempAngle);
                cIDangle = spk.touchID(indSpkAngle);
                tunedAngle(find(ismember(upperPlaneCID, cIDangle))) = tempAngle;
            end

            [~, activitySorti] = sort(mean(sessionUpperTouch), 'descend');
            tunedAngleSortedByActivity = tunedAngle(activitySorti);
            [tunedAngleSorted, tunedAngleSorti] = sort(tunedAngleSortedByActivity, 'ascend');
            finalSorti = activitySorti(tunedAngleSorti);
            cIDsorted = upperPlaneCID(finalSorti);

            sessionUpperTouchSortedSorted = sessionUpperTouchSorted(:, finalSorti);
            sessionUpperNopoleSortedSorted = sessionUpperNopoleSorted(:, finalSorti);
            sessionUpperPoleSortedSorted = sessionUpperPoleSorted(:, finalSorti);
            sessionUpperAllSortedSorted = sessionUpperAllSorted(:, finalSorti);

            if showPopActivity
                figure, subplot(411), imagesc(sessionUpperTouchSortedSorted), title('Touch frames, pre-answer (sorted by angle(Y), activity(X), and tuned angle(X))'),
                subplot(413), imagesc(sessionUpperNopoleSortedSorted), title('No pole period (sorted by angle(Y), activity(X), and tuned angle(X))'),
                subplot(412), imagesc(sessionUpperPoleSortedSorted), title('Pole period (sorted by angle(Y), activity(X), and tuned angle(X))'),
                subplot(414), imagesc(sessionUpperAllSortedSorted), title('All frames (sorted by angle(Y), activity(X), and tuned angle(X))')
            end

    % %         %% Normalized in each neurons
    %         sessionUpperTouchSortedSortedNormalized = (sessionUpperTouchSortedSorted - min(sessionUpperTouchSortedSorted)) ./ (max(sessionUpperTouchSortedSorted) - min(sessionUpperTouchSortedSorted));
    %         sessionUpperNopoleSortedSortedNormalized = (sessionUpperNopoleSortedSorted - min(sessionUpperNopoleSortedSorted)) ./ (max(sessionUpperNopoleSortedSorted) - min(sessionUpperNopoleSortedSorted));
    %         sessionUpperPoleSortedSortedNormalized = (sessionUpperPoleSortedSorted - min(sessionUpperPoleSortedSorted)) ./ (max(sessionUpperPoleSortedSorted) - min(sessionUpperPoleSortedSorted));
    %         sessionUpperAllSortedSortedNormalized = (sessionUpperAllSortedSorted - min(sessionUpperAllSortedSorted)) ./ (max(sessionUpperAllSortedSorted) - min(sessionUpperAllSortedSorted));
    %         
    %         figure, subplot(411), imagesc(sessionUpperTouchSortedSortedNormalized), title('Touch frames, pre-answer (sorted by angle(Y), activity(X), and tuned angle(X), normalized in each neuron)'),
    %         subplot(413), imagesc(sessionUpperNopoleSortedSortedNormalized), title('No pole period (sorted by angle(Y), activity(X), and tuned angle(X), normalized in each neuron)'),
    %         subplot(412), imagesc(sessionUpperPoleSortedSortedNormalized), title('Pole period (sorted by angle(Y), activity(X), and tuned angle(X), normalized in each neuron)'),
    %         subplot(414), imagesc(sessionUpperAllSortedSortedNormalized), title('All frames (sorted by angle(Y), activity(X), and tuned angle(X), normalized in each neuron)')
            if si == 1
                popActNaive{1,mi} = sessionUpperTouchSortedSorted;
                trialNumsNaive{1,mi} = sortedTrialNum;
                sortedAngleNaive{1,mi} = sortedAngle;
                sortedTouchNaive{1,mi} = tunedAngleSorted;
                sortedCidNaive{1,mi} = cIDsorted;
                
            elseif si == 2
                popActExpert{1,mi} = sessionUpperTouchSortedSorted;
                trialNumsExpert{1,mi} = sortedTrialNum;
                sortedAngleExpert{1,mi} = sortedAngle;
                sortedTouchExpert{1,mi} = tunedAngleSorted;
                sortedCidExpert{1,mi} = cIDsorted;
            else
                error('Session index (si) is wrong')
            end


            %% Demensionality reduction
            if si == 1
                tsneNaive{1,mi} = tsne(sessionUpperTouchSortedSorted, 'NumDimensions', numDim);
                pcaNaive{1,mi} = pca(sessionUpperTouchSortedSorted', 'NumComponents', numDim);
            elseif si == 2
                tsneExpert{1,mi} = tsne(sessionUpperTouchSortedSorted, 'NumDimensions', numDim);
                pcaExpert{1,mi} = pca(sessionUpperTouchSortedSorted', 'NumComponents', numDim);
            else
                error('Session index (si) is wrong')
            end
        end
        
        %% then, lower plane L2/3
        disp('Lower plane')
        for ti = 1 : length(lowerPlaneTouchTrialsInd)
            trialInd = lowerPlaneTouchTrialsInd(ti);
            uTrial = u.trials{trialInd};
            if ~isempty(uTrial.answerLickTime)
                lastTime = uTrial.answerLickTime;
            else
                lastTime = uTrial.poleDownOnsetTime;
            end
            lastTouchFrameCell = cellfun(@(x,y) find(x(y) < lastTime, 1, 'last'), uTrial.tpmTime, uTrial.protractionTouchOnsetFramesByWhisking, 'un', 0); % minimum # of tpm touch frames before answer or pole-down from each planes
            if min(cellfun(@length, lastTouchFrameCell)) > 0 % if all cell is not empty                
                lastTouchFrame = min(cellfun(@max, lastTouchFrameCell));
                if ~isempty(lastTouchFrame)                
                    trialTouchFramesBinaryAllNeurons = zeros(length(lowerPlaneCID), size(uTrial.spk,2), 'logical');
                    trialSpontFramesBinaryAllNeurons = ones(length(lowerPlaneCID), size(uTrial.spk,2), 'logical');
                    for pi = 1 : 4
                        planeCind = find(floor(lowerPlaneCID/1000) == pi+4);
                        planeTouchFrames = unique(repmat(uTrial.protractionTouchOnsetFramesByWhisking{pi}(1:lastTouchFrame),size(touchFrameDelays,1), size(touchFrameDelays,2)) + touchFrameDelays);
                        trialTouchFramesBinaryAllNeurons(planeCind, planeTouchFrames) = 1;
                        
                        planePoleUpFrame = find(uTrial.tpmTime{pi} > uTrial.poleUpOnsetTime, 1, 'first');
                        planePoleDownFrame = find(uTrial.tpmTime{pi} > uTrial.poleDownOnsetTime, 1, 'first');
                        trialSpontFramesBinaryAllNeurons(planeCind, planePoleUpFrame:planePoleDownFrame) = 0;
                    end
                    trialPopTouch = sum(uTrial.spk(lowerPlaneCind,:) .* trialTouchFramesBinaryAllNeurons, 2) / lastTouchFrame;
                    sessionLowerTouchNan(ti,:) = trialPopTouch';
                    trialPopNopole = mean(uTrial.spk(lowerPlaneCind,:) .* trialSpontFramesBinaryAllNeurons, 2);
                    sessionLowerNopoleNan(ti,:) = trialPopNopole';
                    trialPopPole = mean(uTrial.spk(lowerPlaneCind,:) .* (1-trialSpontFramesBinaryAllNeurons), 2);
                    sessionLowerPoleNan(ti,:) = trialPopPole';
                    sessionLowerAllNan(ti,:) = mean(uTrial.spk(lowerPlaneCind,:),2);
                    sessionLowerAngleNan(ti,:) = uTrial.angle;
                    sessionLowerTrialNumNan(ti,:) = uTrial.trialNum;
                end
            end
        end
        noNanInd = find(~isnan(sessionLowerTouchNan(:,1)));
        sessionLowerTouch = sessionLowerTouchNan(noNanInd,:);
        sessionLowerNopole = sessionLowerNopoleNan(noNanInd,:);
        sessionLowerPole = sessionLowerPoleNan(noNanInd,:);
        sessionLowerAll = sessionLowerAllNan(noNanInd,:);
        sessionLowerAngle = sessionLowerAngleNan(noNanInd,:);
        sessionLowerTrialNum = sessionLowerTrialNumNan(noNanInd,:);
        
        %% Saving (and showing) population activity
%         figure, subplot(411), imagesc(sessionLowerTouch), title('Touch frames, pre-answer'), subplot(413), imagesc(sessionLowerNopole), title('No pole period'), 
%         subplot(412), imagesc(sessionLowerPole), title('Pole period'), subplot(414), imagesc(sessionLowerAll), title('All frames')

        [sortedAngle, sorti] = sort(sessionLowerAngle, 'ascend');
        sortedTrialNum = sessionLowerTrialNum(sorti);
        sessionLowerTouchSorted = sessionLowerTouch(sorti,:);
        sessionLowerNopoleSorted = sessionLowerNopole(sorti,:);
        sessionLowerPoleSorted = sessionLowerPole(sorti,:);
        sessionLowerAllSorted = sessionLowerAll(sorti,:);
        
%         figure, subplot(411), imagesc(sessionLowerTouchSorted), title('Touch frames, pre-answer (sorted by angle(Y))'), subplot(413), imagesc(sessionLowerNopoleSorted), title('No pole period (sorted by angle(Y))'), 
%         subplot(412), imagesc(sessionLowerPoleSorted), title('Pole period (sorted by angle(Y))'), subplot(414), imagesc(sessionLowerAllSorted), title('All frames (sorted by angle(Y))')
        
%         %% sorting by tuned angle and activity level
        tunedAngle = zeros(length(lowerPlaneCID),1); % if non-touch, 0, if non-selective touch, 1; otherwise, tuned angle (typically 45:15:135)
        tunedAngle(find(ismember(lowerPlaneCID, spk.touchID))) = 1;
        angles = setdiff(unique(spk.tunedAngle),0); % all angles, sorted. Typically, 45:15:135.
        for ai = 1 : length(angles)
            tempAngle = angles(ai);
            indSpkAngle = find(spk.tunedAngle == tempAngle);
            cIDangle = spk.touchID(indSpkAngle);
            tunedAngle(find(ismember(lowerPlaneCID, cIDangle))) = tempAngle;
        end

        [~, activitySorti] = sort(mean(sessionLowerTouch), 'descend');
        tunedAngleSortedByActivity = tunedAngle(activitySorti);
        [tunedAngleSorted, tunedAngleSorti] = sort(tunedAngleSortedByActivity, 'ascend');
        finalSorti = activitySorti(tunedAngleSorti);
        cIDsorted = lowerPlaneCID(finalSorti);
        
        sessionLowerTouchSortedSorted = sessionLowerTouchSorted(:, finalSorti);
        sessionLowerNopoleSortedSorted = sessionLowerNopoleSorted(:, finalSorti);
        sessionLowerPoleSortedSorted = sessionLowerPoleSorted(:, finalSorti);
        sessionLowerAllSortedSorted = sessionLowerAllSorted(:, finalSorti);

        if showPopActivity
            figure, subplot(411), imagesc(sessionLowerTouchSortedSorted), title('Touch frames, pre-answer (sorted by angle(Y), activity(X), and tuned angle(X))'),
            subplot(413), imagesc(sessionLowerNopoleSortedSorted), title('No pole period (sorted by angle(Y), activity(X), and tuned angle(X))'),
            subplot(412), imagesc(sessionLowerPoleSortedSorted), title('Pole period (sorted by angle(Y), activity(X), and tuned angle(X))'),
            subplot(414), imagesc(sessionLowerAllSortedSorted), title('All frames (sorted by angle(Y), activity(X), and tuned angle(X))')
        end
        
% %         %% Normalized in each neurons
%         sessionLowerTouchSortedSortedNormalized = (sessionLowerTouchSortedSorted - min(sessionLowerTouchSortedSorted)) ./ (max(sessionLowerTouchSortedSorted) - min(sessionLowerTouchSortedSorted));
%         sessionLowerNopoleSortedSortedNormalized = (sessionLowerNopoleSortedSorted - min(sessionLowerNopoleSortedSorted)) ./ (max(sessionLowerNopoleSortedSorted) - min(sessionLowerNopoleSortedSorted));
%         sessionLowerPoleSortedSortedNormalized = (sessionLowerPoleSortedSorted - min(sessionLowerPoleSortedSorted)) ./ (max(sessionLowerPoleSortedSorted) - min(sessionLowerPoleSortedSorted));
%         sessionLowerAllSortedSortedNormalized = (sessionLowerAllSortedSorted - min(sessionLowerAllSortedSorted)) ./ (max(sessionLowerAllSortedSorted) - min(sessionLowerAllSortedSorted));
%         
%         figure, subplot(411), imagesc(sessionLowerTouchSortedSortedNormalized), title('Touch frames, pre-answer (sorted by angle(Y), activity(X), and tuned angle(X), normalized in each neuron)'),
%         subplot(413), imagesc(sessionLowerNopoleSortedSortedNormalized), title('No pole period (sorted by angle(Y), activity(X), and tuned angle(X), normalized in each neuron)'),
%         subplot(412), imagesc(sessionLowerPoleSortedSortedNormalized), title('Pole period (sorted by angle(Y), activity(X), and tuned angle(X), normalized in each neuron)'),
%         subplot(414), imagesc(sessionLowerAllSortedSortedNormalized), title('All frames (sorted by angle(Y), activity(X), and tuned angle(X), normalized in each neuron)')        
        
        if si == 1
            popActNaive{2,mi} = sessionLowerTouchSortedSorted;
            trialNumsNaive{2,mi} = sortedTrialNum;
            sortedAngleNaive{2,mi} = sortedAngle;
            sortedTouchNaive{2,mi} = tunedAngleSorted;
            sortedCidNaive{2,mi} = cIDsorted;
        elseif si == 2
            popActExpert{2,mi} = sessionLowerTouchSortedSorted;
            trialNumsExpert{2,mi} = sortedTrialNum;
            sortedAngleExpert{2,mi} = sortedAngle;
            sortedTouchExpert{2,mi} = tunedAngleSorted;
            sortedCidExpert{2,mi} = cIDsorted;
        else
            error('Session index (si) is wrong')
        end
        
        %% Dimensionality reduction
        if si == 1
            tsneNaive{2,mi} = tsne(sessionLowerTouchSortedSorted, 'NumDimensions', numDim);
            pcaNaive{2,mi} = pca(sessionLowerTouchSortedSorted', 'NumComponents', numDim);
        elseif si == 2
            tsneExpert{2,mi} = tsne(sessionLowerTouchSortedSorted, 'NumDimensions', numDim);
            pcaExpert{2,mi} = pca(sessionLowerTouchSortedSorted', 'NumComponents', numDim);
        else
            error('Session index (si) is wrong')
        end

    end
end


%% save processed data
info = struct;
info.mice = mice;
info.sessions = sessions;
info.numDim = numDim;
info.cellDepthThresh = cellDepthThresh;
info.touchFrameDelays = touchFrameDelays;
info.angles = angles;
save('Y:\Whiskernas\JK\suite2p\pop_decoding_7angles_dim2_noDelayFrames.mat', '*Naive', '*Expert', 'info')


% info = struct;
% info.mice = mice;
% info.sessions = sessions;
% info.numDim = numDim;
% info.cellDepthThresh = cellDepthThresh;
% info.touchFrameDelays = touchFrameDelays;
% info.angles = angles;
% save('Y:\Whiskernas\JK\suite2p\pop_decoding_7angles_dim2.mat', '*Naive', '*Expert', 'info')

% %% save processed data
% info = struct;
% info.mice = mice;
% info.sessions = sessions;
% info.numDim = numDim;
% info.cellDepthThresh = cellDepthThresh;
% info.touchFrameDelays = touchFrameDelays;
% info.angles = angles;
% save('Y:\Whiskernas\JK\suite2p\pop_decoding_7angles.mat', '*Naive', '*Expert', 'info')

%%

clear

load('Y:\Whiskernas\JK\suite2p\pop_decoding_7angles.mat', '*Naive', '*Expert', 'info')



%% 3D visualization
vi = 1; % volume index
mi = 3; % mouse index
angles = 45:15:135;
colors = jet(7);

allColorNaive = zeros(length(sortedAngleNaive{vi,mi}),3);
for ai = 1 : length(angles)
    indAngle = find(sortedAngleNaive{vi,mi} == angles(ai));
    allColorNaive(indAngle,:) = repmat(colors(ai,:), length(indAngle),1);
end
allColorExpert = zeros(length(sortedAngleExpert{vi,mi}),3);
for ai = 1 : length(angles)
    indAngle = find(sortedAngleExpert{vi,mi} == angles(ai));
    allColorExpert(indAngle,:) = repmat(colors(ai,:), length(indAngle),1);
end

figure('units', 'normalized', 'outerposition', [0.3 0.3 0.3 0.3]), 
subplot(121),
scatter3(tsneNaive{vi,mi}(:,1),tsneNaive{vi,mi}(:,2),tsneNaive{vi,mi}(:,3), 10, allColorNaive)
title(sprintf('JK%03d v%d tSNE - Naive', info.mice(mi), vi))
subplot(122)
scatter3(tsneExpert{vi,mi}(:,1),tsneExpert{vi,mi}(:,2),tsneExpert{vi,mi}(:,3), 10, allColorExpert)
title('Expert')

figure('units', 'normalized', 'outerposition', [0.3 0.3 0.3 0.3]),  
subplot(121)
scatter3(pcaNaive{vi,mi}(:,1),pcaNaive{vi,mi}(:,2),pcaNaive{vi,mi}(:,3), 10, allColorNaive)
title(sprintf('JK%03d v%d PCA - Naive', info.mice(mi), vi))
subplot(122)
scatter3(pcaExpert{vi,mi}(:,1),pcaExpert{vi,mi}(:,2),pcaExpert{vi,mi}(:,3), 10, allColorExpert)
title('Expert')

%% Pairwise distance visualization

figure('units', 'normalized', 'outerposition', [0.3 0.3 0.3 0.3]), 
subplot(121)
imagesc(squareform(pdist(tsneNaive{vi,mi}))), axis square
title(sprintf('JK%03d v%d tSNE - Naive', info.mice(mi), vi))
subplot(122)
imagesc(squareform(pdist(tsneExpert{vi,mi}))), axis square
title('Expert')

figure('units', 'normalized', 'outerposition', [0.2 0.4 0.3 0.3])
subplot(121)
imagesc(squareform(pdist(pcaNaive{vi,mi}))), axis square
title(sprintf('JK%03d v%d PCA - Naive', info.mice(mi), vi))
subplot(122)
imagesc(squareform(pdist(pcaExpert{vi,mi}))), axis square
title('Expert')



%% remove outliers (trials) from PCA analysis
stdThresh = 3;
pcaN = pcaNaive{vi,mi};
m = mean(pcaN);
sd = std(pcaN);
outlierInd = union(union(find(abs(pcaN(:,1) - m(:,1)) > stdThresh * sd(:,1)), find(abs(pcaN(:,2) - m(:,2)) > stdThresh * sd(:,2))), find(abs(pcaN(:,3) - m(:,3)) > stdThresh * sd(:,3)));
pcaN(outlierInd,:) = [];

pcaDist = squareform(pdist(pcaN));
figure, imagesc(pcaDist)

pcaN = pcaExpert{vi,mi};
m = mean(pcaN);
sd = std(pcaN);
outlierInd = union(union(find(abs(pcaN(:,1) - m(:,1)) > stdThresh * sd(:,1)), find(abs(pcaN(:,2) - m(:,2)) > stdThresh * sd(:,2))), find(abs(pcaN(:,3) - m(:,3)) > stdThresh * sd(:,3)));
pcaN(outlierInd,:) = [];

pcaDist = squareform(pdist(pcaN));
figure, imagesc(pcaDist)


%% Visualizing again, with comparison between tsne and outlier-removed pca
vi = 2; % volume index
mi = 6; % mouse index
angles = 45:15:135;
colors = jet(7);

allColorNaive = zeros(length(sortedAngleNaive{vi,mi}),3);
for ai = 1 : length(angles)
    indAngle = find(sortedAngleNaive{vi,mi} == angles(ai));
    allColorNaive(indAngle,:) = repmat(colors(ai,:), length(indAngle),1);
end
allColorExpert = zeros(length(sortedAngleExpert{vi,mi}),3);
for ai = 1 : length(angles)
    indAngle = find(sortedAngleExpert{vi,mi} == angles(ai));
    allColorExpert(indAngle,:) = repmat(colors(ai,:), length(indAngle),1);
end

figure('units', 'normalized', 'outerposition', [0.2 0.2 0.25 0.25]), 
subplot(121),
scatter3(tsneNaive{vi,mi}(:,1),tsneNaive{vi,mi}(:,2),tsneNaive{vi,mi}(:,3), 10, allColorNaive)
title(sprintf('JK%03d v%d tSNE - Naive', info.mice(mi), vi))
subplot(122)
scatter3(tsneExpert{vi,mi}(:,1),tsneExpert{vi,mi}(:,2),tsneExpert{vi,mi}(:,3), 10, allColorExpert)
title('Expert')

% pca removing outliers

figure('units', 'normalized', 'outerposition', [0.25 0.25 0.25 0.25]),  
subplot(121)
stdThresh = 3;
pcaN = pcaNaive{vi,mi};
m = mean(pcaN);
sd = std(pcaN);
outlierInd = union(union(find(abs(pcaN(:,1) - m(:,1)) > stdThresh * sd(:,1)), find(abs(pcaN(:,2) - m(:,2)) > stdThresh * sd(:,2))), find(abs(pcaN(:,3) - m(:,3)) > stdThresh * sd(:,3)));
pcaN(outlierInd,:) = [];
tempColor = allColorNaive;
tempColor(outlierInd,:) = [];
scatter3(pcaN(:,1),pcaN(:,2),pcaN(:,3), 10, tempColor)
title(sprintf('JK%03d v%d PCA - Naive', info.mice(mi), vi))

subplot(122)
pcaE = pcaExpert{vi,mi};
m = mean(pcaE);
sd = std(pcaE);
outlierInd = union(union(find(abs(pcaE(:,1) - m(:,1)) > stdThresh * sd(:,1)), find(abs(pcaE(:,2) - m(:,2)) > stdThresh * sd(:,2))), find(abs(pcaE(:,3) - m(:,3)) > stdThresh * sd(:,3)));
pcaE(outlierInd,:) = [];
tempColor = allColorExpert;
tempColor(outlierInd,:) = [];
scatter3(pcaE(:,1),pcaE(:,2),pcaE(:,3), 10, tempColor)
title('Expert')


figure('units', 'normalized', 'outerposition', [0.3 0.3 0.25 0.25]), 
subplot(121)
imagesc(squareform(pdist(tsneNaive{vi,mi}))), axis square
title(sprintf('JK%03d v%d tSNE - Naive', info.mice(mi), vi))
subplot(122)
imagesc(squareform(pdist(tsneExpert{vi,mi}))), axis square
title('Expert')

figure('units', 'normalized', 'outerposition', [0.35 0.35 0.25 0.25])
subplot(121)
imagesc(squareform(pdist(pcaN))), axis square
title(sprintf('JK%03d v%d PCA - Naive', info.mice(mi), vi))
subplot(122)
imagesc(squareform(pdist(pcaE))), axis square
title('Expert')


%% Clustering quantification (other than decoding)
% mean( mean(within-group distance) / mean(between-group distance) )
% (1) averaging from all individual trials

mice = info.mice;

pcaDistCluster = nan(size(pcaNaive,1) * size(pcaNaive,2), 2); % (:,1) naive, (:,2) expert
tsneDistCluster = nan(size(tsneNaive,1) * size(tsneNaive,2), 2); % (:,1) naive, (:,2) expert

for mi = 1 : length(mice)
    for vi = 1 : 2
        ind = (mi-1) * 2 + vi;
        if ~isempty(pcaNaive{vi,mi})
            % tsne distance grouping quantification
            % naive
            sortedAngle = sortedAngleNaive{vi,mi};
            tempdist = squareform(pdist(tsneNaive{vi,mi}));
            betweenGroupBin = squareform(pdist(sortedAngle)>0);
            withinGroupBin = squareform(pdist(sortedAngle)==0);
            tsneDistCluster(ind,1) = mean(   ( (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) - (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) )  ...
                ./ ( (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) + (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) )   );
            
            % expert
            sortedAngle = sortedAngleExpert{vi,mi};
            tempdist = squareform(pdist(tsneExpert{vi,mi}));
            betweenGroupBin = squareform(pdist(sortedAngle)>0);
            withinGroupBin = squareform(pdist(sortedAngle)==0);
            tsneDistCluster(ind,2) = mean(   ( (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) - (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) )  ...
                ./ ( (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) + (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) )   );
            
            % pca distance grouping quantification
            % naive
            sortedAngle = sortedAngleNaive{vi,mi};
            tempdist = squareform(pdist(pcaNaive{vi,mi}));
            betweenGroupBin = squareform(pdist(sortedAngle)>0);
            withinGroupBin = squareform(pdist(sortedAngle)==0);
            pcaDistCluster(ind,1) = mean(   ( (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) - (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) )  ...
                ./ ( (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) + (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) )   );
            
            % expert
            sortedAngle = sortedAngleExpert{vi,mi};
            tempdist = squareform(pdist(pcaExpert{vi,mi}));
            betweenGroupBin = squareform(pdist(sortedAngle)>0);
            withinGroupBin = squareform(pdist(sortedAngle)==0);
            pcaDistCluster(ind,2) = mean(   ( (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) - (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) )  ...
                ./ ( (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) + (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) )   );
        else
            removeInd = ind;
        end
    end
end

pcaDistCluster(removeInd,:) = [];
tsneDistCluster(removeInd,:) = [];

figure('units','norm','outerpos',[0.3 0.3 0.4 0.3]),
subplot(121), hold on
for i = 1 : size(tsneDistCluster,1)
    plot([1,2], tsneDistCluster(i,:), 'ko-')
end
errorbar(1, mean(tsneDistCluster(:,1)), sem(tsneDistCluster(:,1)), 'r')
errorbar(2, mean(tsneDistCluster(:,2)), sem(tsneDistCluster(:,2)), 'r')
[~, p] = ttest(tsneDistCluster(:,1), tsneDistCluster(:,2));
xlim([0.5 2.5])
xticks([1, 2])
xticklabels({'Naive', 'Expert'})
title({'t-SNE clustering index'; sprintf('p = %.3f', p)})

subplot(122), hold on
for i = 1 : size(pcaDistCluster,1)
    plot([1,2], pcaDistCluster(i,:), 'ko-')
end
errorbar(1, mean(pcaDistCluster(:,1)), sem(pcaDistCluster(:,1)), 'r')
errorbar(2, mean(pcaDistCluster(:,2)), sem(pcaDistCluster(:,2)), 'r')
[~, p] = ttest(pcaDistCluster(:,1), pcaDistCluster(:,2));
xlim([0.5 2.5])
xticks([1, 2])
xticklabels({'Naive', 'Expert'})
title({'PCA clustering index'; sprintf('p = %.3f', p)})

%%
% (2) averaging after same-angle averaging

mice = info.mice;
angles = info.angles;

pcaDistCluster = nan(size(pcaNaive,1) * size(pcaNaive,2), 2); % (:,1) naive, (:,2) expert
tsneDistCluster = nan(size(tsneNaive,1) * size(tsneNaive,2), 2); % (:,1) naive, (:,2) expert

for mi = 1 : length(mice)
    for vi = 1 : 2
        ind = (mi-1) * 2 + vi;
        if ~isempty(pcaNaive{vi,mi})
            % tsne distance grouping quantification
            % naive
            sortedAngle = sortedAngleNaive{vi,mi};
            tempdist = squareform(pdist(tsneNaive{vi,mi}));
            betweenGroupBin = squareform(pdist(sortedAngle)>0);
            withinGroupBin = squareform(pdist(sortedAngle)==0);
            tempCIAll = ( sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2) - sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2) )   ...
                ./  ( sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2) + sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2) );
            tempCI = zeros(length(angles),1);
            for ai = 1 : length(angles)
                angleInd = find(sortedAngle == angles(ai));
                tempCI(ai) = mean(tempCIAll(angleInd));
            end
            tsneDistCluster(ind,1) = mean(tempCI);
            
            % expert
            sortedAngle = sortedAngleExpert{vi,mi};
            tempdist = squareform(pdist(tsneExpert{vi,mi}));
            betweenGroupBin = squareform(pdist(sortedAngle)>0);
            withinGroupBin = squareform(pdist(sortedAngle)==0);
            tempCIAll = ( sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2) - sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2) )   ...
                ./  ( sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2) + sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2) );
            tempCI = zeros(length(angles),1);
            for ai = 1 : length(angles)
                angleInd = find(sortedAngle == angles(ai));
                tempCI(ai) = mean(tempCIAll(angleInd));
            end
            tsneDistCluster(ind,2) = mean(tempCI);
            
            % pca distance grouping quantification
            % naive
            sortedAngle = sortedAngleNaive{vi,mi};
            tempdist = squareform(pdist(pcaNaive{vi,mi}));
            betweenGroupBin = squareform(pdist(sortedAngle)>0);
            withinGroupBin = squareform(pdist(sortedAngle)==0);
            tempCIAll = ( sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2) - sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2) )   ...
                ./  ( sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2) + sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2) );
            tempCI = zeros(length(angles),1);
            for ai = 1 : length(angles)
                angleInd = find(sortedAngle == angles(ai));
                tempCI(ai) = mean(tempCIAll(angleInd));
            end
            pcaDistCluster(ind,1) = mean(tempCI);
            
            % expert
            sortedAngle = sortedAngleExpert{vi,mi};
            tempdist = squareform(pdist(pcaExpert{vi,mi}));
            betweenGroupBin = squareform(pdist(sortedAngle)>0);
            withinGroupBin = squareform(pdist(sortedAngle)==0);
            tempCIAll = ( sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2) - sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2) )   ...
                ./  ( sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2) + sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2) );
            tempCI = zeros(length(angles),1);
            for ai = 1 : length(angles)
                angleInd = find(sortedAngle == angles(ai));
                tempCI(ai) = mean(tempCIAll(angleInd));
            end
            pcaDistCluster(ind,2) = mean(tempCI);
        else
            removeInd = ind;
        end
    end
end

pcaDistCluster(removeInd,:) = [];
tsneDistCluster(removeInd,:) = [];

figure('units','norm','outerpos',[0.3 0.3 0.4 0.3]),
subplot(121), hold on
for i = 1 : size(tsneDistCluster,1)
    plot([1,2], tsneDistCluster(i,:), 'ko-')
end
errorbar(1, mean(tsneDistCluster(:,1)), sem(tsneDistCluster(:,1)), 'r')
errorbar(2, mean(tsneDistCluster(:,2)), sem(tsneDistCluster(:,2)), 'r')
[~, p] = ttest(tsneDistCluster(:,1), tsneDistCluster(:,2));
xlim([0.5 2.5])
xticks([1, 2])
xticklabels({'Naive', 'Expert'})
title({'t-SNE clustering index'; sprintf('p = %.3f', p)})

subplot(122), hold on
for i = 1 : size(pcaDistCluster,1)
    plot([1,2], pcaDistCluster(i,:), 'ko-')
end
errorbar(1, mean(pcaDistCluster(:,1)), sem(pcaDistCluster(:,1)), 'r')
errorbar(2, mean(pcaDistCluster(:,2)), sem(pcaDistCluster(:,2)), 'r')
[~, p] = ttest(pcaDistCluster(:,1), pcaDistCluster(:,2));
xlim([0.5 2.5])
xticks([1, 2])
xticklabels({'Naive', 'Expert'})
title({'PCA clustering index'; sprintf('p = %.3f', p)})



%% How about removing outliers in PCA?
% using method (1) 

mice = info.mice;

pcaDistCluster = nan(size(pcaNaive,1) * size(pcaNaive,2), 2); % (:,1) naive, (:,2) expert
tsneDistCluster = nan(size(tsneNaive,1) * size(tsneNaive,2), 2); % (:,1) naive, (:,2) expert

for mi = 1 : length(mice)
    for vi = 1 : 2
        ind = (mi-1) * 2 + vi;
        if ~isempty(pcaNaive{vi,mi})
            % tsne distance grouping quantification
            % naive
            sortedAngle = sortedAngleNaive{vi,mi};
            tempdist = squareform(pdist(tsneNaive{vi,mi}));
            betweenGroupBin = squareform(pdist(sortedAngle)>0);
            withinGroupBin = squareform(pdist(sortedAngle)==0);
            tsneDistCluster(ind,1) = mean(   ( (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) - (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) )  ...
                ./ ( (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) + (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) )   );
            
            % expert
            sortedAngle = sortedAngleExpert{vi,mi};
            tempdist = squareform(pdist(tsneExpert{vi,mi}));
            betweenGroupBin = squareform(pdist(sortedAngle)>0);
            withinGroupBin = squareform(pdist(sortedAngle)==0);
            tsneDistCluster(ind,2) = mean(   ( (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) - (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) )  ...
                ./ ( (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) + (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) )   );
            
            % pca distance grouping quantification
            % naive
            pcaN = pcaNaive{vi,mi};
            m = mean(pcaN);
            sd = std(pcaN);
            outlierInd = union(union(find(abs(pcaN(:,1) - m(:,1)) > stdThresh * sd(:,1)), find(abs(pcaN(:,2) - m(:,2)) > stdThresh * sd(:,2))), find(abs(pcaN(:,3) - m(:,3)) > stdThresh * sd(:,3)));
            pcaN(outlierInd,:) = [];
            tempdist = squareform(pdist(pcaN));
            
            sortedAngle = sortedAngleNaive{vi,mi};
            sortedAngle(outlierInd,:) = [];
            
            betweenGroupBin = squareform(pdist(sortedAngle)>0);
            withinGroupBin = squareform(pdist(sortedAngle)==0);
            pcaDistCluster(ind,1) = mean(   ( (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) - (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) )  ...
                ./ ( (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) + (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) )   );
            
            % expert
            pcaE = pcaExpert{vi,mi};
            m = mean(pcaE);
            sd = std(pcaE);
            outlierInd = union(union(find(abs(pcaE(:,1) - m(:,1)) > stdThresh * sd(:,1)), find(abs(pcaE(:,2) - m(:,2)) > stdThresh * sd(:,2))), find(abs(pcaE(:,3) - m(:,3)) > stdThresh * sd(:,3)));
            pcaE(outlierInd,:) = [];
            tempdist = squareform(pdist(pcaE));
            
            sortedAngle = sortedAngleExpert{vi,mi};
            sortedAngle(outlierInd,:) = [];
            
            betweenGroupBin = squareform(pdist(sortedAngle)>0);
            withinGroupBin = squareform(pdist(sortedAngle)==0);
            pcaDistCluster(ind,2) = mean(   ( (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) - (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) )  ...
                ./ ( (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) + (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) )   );
        else
            removeInd = ind;
        end
    end
end

pcaDistCluster(removeInd,:) = [];
tsneDistCluster(removeInd,:) = [];

figure('units','norm','outerpos',[0.3 0.3 0.4 0.3]),
subplot(121), hold on
for i = 1 : size(tsneDistCluster,1)
    plot([1,2], tsneDistCluster(i,:), 'ko-')
end
errorbar(1, mean(tsneDistCluster(:,1)), sem(tsneDistCluster(:,1)), 'r')
errorbar(2, mean(tsneDistCluster(:,2)), sem(tsneDistCluster(:,2)), 'r')
[~, p] = ttest(tsneDistCluster(:,1), tsneDistCluster(:,2));
xlim([0.5 2.5])
xticks([1, 2])
xticklabels({'Naive', 'Expert'})
title({'t-SNE clustering index'; sprintf('p = %.3f', p)})

subplot(122), hold on
for i = 1 : size(pcaDistCluster,1)
    plot([1,2], pcaDistCluster(i,:), 'ko-')
end
errorbar(1, mean(pcaDistCluster(:,1)), sem(pcaDistCluster(:,1)), 'r')
errorbar(2, mean(pcaDistCluster(:,2)), sem(pcaDistCluster(:,2)), 'r')
[~, p] = ttest(pcaDistCluster(:,1), pcaDistCluster(:,2));
xlim([0.5 2.5])
xticks([1, 2])
xticklabels({'Naive', 'Expert'})
title({'PCA clustering index'; sprintf('p = %.3f', p)})



%% Dimensionality analysis
mice = info.mice;
dims = [1:10, 15:5:50, 60:10:100];
pcaCI = nan(size(pcaNaive,1) * size(pcaNaive,2), 2, length(dims)+1); % (:,1,:) naive, (:,2,:) expert
tsneCI = nan(size(tsneNaive,1) * size(tsneNaive,2), 2, length(dims)+1); % (:,1,:) naive, (:,2,:) expert

tsneLoss = nan(size(tsneNaive,1) * size(tsneNaive,2), 2, length(dims+1));
pcaVE = nan(size(tsneNaive,1) * size(tsneNaive,2), 2, length(dims+1));

procDur = zeros(length(dims)+1,1);

%%
% parfor di = 1 : length(dims)
% parfor di = 11:14
%     di = diInds(dii);
for di = 10
    numDim = dims(di);
    
    fprintf('Processing dimension %03d\n', numDim);
    tic
    parpcaCI = nan(size(pcaNaive,1) * size(pcaNaive,2), 2); % (:,1) naive, (:,2) expert
    partsneCI = nan(size(tsneNaive,1) * size(tsneNaive,2), 2); % (:,1) naive, (:,2) expert

    partsneLoss = nan(size(tsneNaive,1) * size(tsneNaive,2), 2);
    parpcaVE = nan(size(tsneNaive,1) * size(tsneNaive,2), 2);

    for mi = 1 : length(mice)
        for vi = 1 : 2
            ind = (mi-1) * 2 + vi;
            response = popActNaive{vi,mi};
            if ~isempty(response)
%                 [tsneResult, partsneLoss(ind,1)] = tsne(response, 'NumDimensions', numDim);
                [pcaResult, ~, ~, ~, explained] = pca(response', 'NumComponents', numDim);
                parpcaVE(ind,1) = sum(explained(1:numDim));
                
                sortedAngle = sortedAngleNaive{vi,mi};
%                 tempdist = squareform(pdist(tsneResult));
                betweenGroupBin = squareform(pdist(sortedAngle)>0);
                withinGroupBin = squareform(pdist(sortedAngle)==0);
%                 partsneCI(ind,1) = mean(   ( (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) - (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) )  ...
%                     ./ ( (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) + (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) )   );

                tempdist = squareform(pdist(pcaResult));
                parpcaCI(ind,1) = mean(   ( (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) - (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) )  ...
                    ./ ( (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) + (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) )   );
                
                
                response = popActExpert{vi,mi};
%                 [tsneResult, partsneLoss(ind,2)] = tsne(response, 'NumDimensions', numDim);
                [pcaResult, ~,~,~, explained] = pca(response', 'NumComponents', numDim);
                parpcaVE(ind,2) = sum(explained(1:numDim));

                sortedAngle = sortedAngleExpert{vi,mi};
%                 tempdist = squareform(pdist(tsneResult));
                betweenGroupBin = squareform(pdist(sortedAngle)>0);
                withinGroupBin = squareform(pdist(sortedAngle)==0);
%                 partsneCI(ind,2) = mean(   ( (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) - (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) )  ...
%                     ./ ( (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) + (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) )   );

                tempdist = squareform(pdist(pcaResult));
                parpcaCI(ind,2) = mean(   ( (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) - (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) )  ...
                    ./ ( (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) + (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) )   );
                
            else
                removeInd = ind;
            end
        end
    end
    
    pcaCI(:,:,di) = parpcaCI;
%     tsneCI(:,:,di) = partsneCI;

%     tsneLoss(:,:,di) = partsneLoss;
    pcaVE(:,:,di) = parpcaVE;
    procDur(di) = toc
    
end
%%
% for using all active neurons
fprintf('Processing using all neurons\n');
tic
for mi = 1 : length(mice)
    for vi = 1 : 2
        ind = (mi-1) * 2 + vi;
        response = popActNaive{vi,mi};
        if ~isempty(response)
            tsneResult = response;
            pcaResult = response;

            sortedAngle = sortedAngleNaive{vi,mi};
            tempdist = squareform(pdist(tsneResult));
            betweenGroupBin = squareform(pdist(sortedAngle)>0);
            withinGroupBin = squareform(pdist(sortedAngle)==0);
            tsneCI(ind,1, end) = mean(   ( (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) - (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) )  ...
                ./ ( (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) + (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) )   );

            tempdist = squareform(pdist(pcaResult));
            pcaCI(ind,1, end) = mean(   ( (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) - (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) )  ...
                ./ ( (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) + (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) )   );


            response = popActExpert{vi,mi};
            tsneResult = response;
            pcaResult = response;

            sortedAngle = sortedAngleExpert{vi,mi};
            tempdist = squareform(pdist(tsneResult));
            betweenGroupBin = squareform(pdist(sortedAngle)>0);
            withinGroupBin = squareform(pdist(sortedAngle)==0);
            tsneCI(ind,2, end) = mean(   ( (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) - (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) )  ...
                ./ ( (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) + (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) )   );

            tempdist = squareform(pdist(pcaResult));
            pcaCI(ind,2, end) = mean(   ( (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) - (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) )  ...
                ./ ( (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) + (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) )   );

        else
            removeInd = ind;
        end
    end
end
procDur(end) = toc

%%
data2 = struct;
data2.pcaCI = pcaCI;
data2.tsneCI = tsneCI;
data2.tsneLoss = tsneLoss;
data2.pcaVE = pcaVE;
data2.procDur = procDur;
save('Y:\Whiskernas\JK\suite2p\population_response_dimension_analysis_temp2.mat', 'data2')

%%
data3 = struct;
data3.pcaCI = pcaCI;
data3.tsneCI = tsneCI;
data3.tsneLoss = tsneLoss;
data3.pcaVE = pcaVE;
data3.procDur = procDur;
save('Y:\Whiskernas\JK\suite2p\population_response_dimension_analysis_temp3.mat', 'data3')


%%
clear
load('Y:\Whiskernas\JK\suite2p\population_response_dimension_analysis_temp2.mat', 'data2')
load('Y:\Whiskernas\JK\suite2p\population_response_dimension_analysis_temp3.mat', 'data3')

%%
dims = [1:10, 15:5:50, 60:10:100];
pcaCI = data2.pcaCI;
pcaCI(:,:,5:9) = data3.pcaCI(:,:,5:9);

pcaVE = data2.pcaVE;
pcaVE(:,:,5:9) = data3.pcaVE(:,:,5:9);

tsneCI = data3.tsneCI;
tsneCI(:,:,10) = nan;
tsneCI(:,:,[15:19,24]) = data2.tsneCI(:,:,[15:19,24]);

tsneLoss = data3.tsneLoss;
tsneLoss(:,:,10) = nan;
tsneLoss(:,:,15:19) = data2.tsneLoss(:,:,15:19);

%%
save('Y:\Whiskernas\JK\suite2p\population_response_dimension_analysis.mat', 'pcaCI', 'pcaVE', 'tsneCI', 'tsneLoss', 'dims')

%%
clear
load('Y:\Whiskernas\JK\suite2p\population_response_dimension_analysis.mat', 'pcaCI', 'pcaVE', 'tsneCI', 'tsneLoss', 'dims')
%%

removeInd = 3;


tsneCI(removeInd,:,:) = [];
pcaCI(removeInd,:,:) = [];

tsneVals = squeeze(tsneCI(:,2,:) - tsneCI(:,1,:));
pcaVals = squeeze(pcaCI(:,2,:) - pcaCI(:,1,:));

tsneLoss(removeInd,:,:) = [];
pcaVE(removeInd,:,:) = [];

%%
figure, hold on
for i = 1 : size(tsneVals)
    plot([dims,200], tsneVals(i,:), '-', 'color', [0.6 0.6 0.6])
end
plot([dims,200], mean(tsneVals), 'r-')
plot(200, mean(tsneVals(:,end)), 'r.')
xlabel('# of dimension')
ylabel('CI difference (Expert - Naive)')
title('t-SNE')
%%
figure, hold on
for i = 1 : size(pcaVals)
    plot([dims,200], pcaVals(i,:), '-', 'color', [0.6 0.6 0.6])
end
plot([dims,200], mean(pcaVals), 'r-')
plot(200, mean(pcaVals(:,end)), 'r.')
xlabel('# of dimension')
ylabel('CI difference (Expert - Naive)')
title('PCA')


%% use first 9 dimensions only.

%%
figure, hold on
for i = 1 : size(tsneVals)
    plot(dims(1:9), tsneVals(i,1:9), '-', 'color', [0.6 0.6 0.6])
end
plot(dims(1:9), mean(tsneVals(:,1:9)), 'r-')
xlabel('# of dimension')
ylabel('CI difference (Expert - Naive)')
title('t-SNE')
%%
figure, hold on
for i = 1 : size(pcaVals)
    plot(dims(1:9), pcaVals(i,1:9), '-', 'color', [0.6 0.6 0.6])
end
plot(dims(1:9), mean(pcaVals(:,1:9)), 'r-')

xlabel('# of dimension')
ylabel('CI difference (Expert - Naive)')
title('PCA')
%%

meanTsneCI = squeeze(mean(tsneCI(:,:,1:9)));
figure, hold on
plot(dims(1:9), meanTsneCI(1,:), 'b-')
plot(dims(1:9), meanTsneCI(2,:), 'r-')
ylabel('Mean CI values')
legend({'Naive', 'Expert'})
xlabel('# of dimension')
title('t-SNE')

%%

meanPcaCI = squeeze(mean(pcaCI(:,:,1:9)));
figure, hold on
plot(dims(1:9), meanPcaCI(1,:), 'b-')
plot(dims(1:9), meanPcaCI(2,:), 'r-')
ylabel('Mean CI values')
legend({'Naive', 'Expert'})
xlabel('# of dimension')
title('PCA')


%%
meanTsneLoss = squeeze(mean(tsneLoss(:,:,1:9)));
figure, hold on
plot(dims(1:9), meanTsneLoss(1,:), 'b-')
plot(dims(1:9), meanTsneLoss(2,:), 'r-')
ylabel('Mean loss')
legend({'Naive', 'Expert'})
xlabel('# of dimension')
title('t-SNE')

%%
meanPcaVE = squeeze(mean(pcaVE(:,:,1:9)));
figure, hold on
plot(dims(1:9), meanPcaVE(1,:), 'b-')
plot(dims(1:9), meanPcaVE(2,:), 'r-')
ylabel('Mean variance explained')
legend({'Naive', 'Expert'})
xlabel('# of dimension')
title('PCA')

%%
figure, hold on
plot(dims, squeeze(mean(pcaVE(:,1,:))), 'b-')
plot(dims, squeeze(mean(pcaVE(:,2,:))), 'r-')
ylabel('Mean variance explained')
legend({'Naive', 'Expert'})
xlabel('# of dimension')
title('PCA')


%% Try visualizing PCA 2 dim
clear
load('Y:\Whiskernas\JK\suite2p\pop_decoding_7angles_dim2.mat', '*Naive', '*Expert', 'info')

%%
vi = 1; % volume index
mi = 5; % mouse index
angles = 45:15:135;
colors = jet(7);

allColorNaive = zeros(length(sortedAngleNaive{vi,mi}),3);
for ai = 1 : length(angles)
    indAngle = find(sortedAngleNaive{vi,mi} == angles(ai));
    allColorNaive(indAngle,:) = repmat(colors(ai,:), length(indAngle),1);
end
allColorExpert = zeros(length(sortedAngleExpert{vi,mi}),3);
for ai = 1 : length(angles)
    indAngle = find(sortedAngleExpert{vi,mi} == angles(ai));
    allColorExpert(indAngle,:) = repmat(colors(ai,:), length(indAngle),1);
end

figure('units', 'normalized', 'outerposition', [0.3 0.3 0.25 0.25]), 
subplot(121),
scatter(tsneNaive{vi,mi}(:,1),tsneNaive{vi,mi}(:,2), 10, allColorNaive)
title(sprintf('JK%03d v%d tSNE - Naive', info.mice(mi), vi))
subplot(122)
scatter(tsneExpert{vi,mi}(:,1),tsneExpert{vi,mi}(:,2), 10, allColorExpert)
title('Expert')

figure('units', 'normalized', 'outerposition', [0.25 0.25 0.25 0.25]),  
subplot(121)
stdThresh = 2;
pcaN = pcaNaive{vi,mi};
m = mean(pcaN);
sd = std(pcaN);
outlierInd = union(find(abs(pcaN(:,1) - m(:,1)) > stdThresh * sd(:,1)), find(abs(pcaN(:,2) - m(:,2)) > stdThresh * sd(:,2)));
pcaN(outlierInd,:) = [];
tempColor = allColorNaive;
tempColor(outlierInd,:) = [];
scatter(pcaN(:,1),pcaN(:,2), 10, tempColor)
title(sprintf('JK%03d v%d PCA - Naive', info.mice(mi), vi))

subplot(122)
pcaE = pcaExpert{vi,mi};
m = mean(pcaE);
sd = std(pcaE);
outlierInd = union(find(abs(pcaE(:,1) - m(:,1)) > stdThresh * sd(:,1)), find(abs(pcaE(:,2) - m(:,2)) > stdThresh * sd(:,2)));
pcaE(outlierInd,:) = [];
tempColor = allColorExpert;
tempColor(outlierInd,:) = [];
scatter(pcaE(:,1),pcaE(:,2), 10, tempColor)
title('Expert')


figure('units', 'normalized', 'outerposition', [0.3 0.3 0.25 0.25]), 
subplot(121)
imagesc(squareform(pdist(tsneNaive{vi,mi}))), axis square
title(sprintf('JK%03d v%d tSNE - Naive', info.mice(mi), vi))
subplot(122)
imagesc(squareform(pdist(tsneExpert{vi,mi}))), axis square
title('Expert')

figure('units', 'normalized', 'outerposition', [0.35 0.35 0.25 0.25])
subplot(121)
imagesc(squareform(pdist(pcaN))), axis square
title(sprintf('JK%03d v%d PCA - Naive', info.mice(mi), vi))
subplot(122)
imagesc(squareform(pdist(pcaE))), axis square
title('Expert')


%% Comparing CI before and after learning

mice = info.mice;

pcaDistCluster = nan(size(pcaNaive,1) * size(pcaNaive,2), 2); % (:,1) naive, (:,2) expert
tsneDistCluster = nan(size(tsneNaive,1) * size(tsneNaive,2), 2); % (:,1) naive, (:,2) expert

for mi = 1 : length(mice)
    for vi = 1 : 2
        ind = (mi-1) * 2 + vi;
        if ~isempty(pcaNaive{vi,mi})
            % tsne distance grouping quantification
            % naive
            sortedAngle = sortedAngleNaive{vi,mi};
            tempdist = squareform(pdist(tsneNaive{vi,mi}));
            betweenGroupBin = squareform(pdist(sortedAngle)>0);
            withinGroupBin = squareform(pdist(sortedAngle)==0);
            tsneDistCluster(ind,1) = mean(   ( (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) - (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) )  ...
                ./ ( (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) + (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) )   );
            
            % expert
            sortedAngle = sortedAngleExpert{vi,mi};
            tempdist = squareform(pdist(tsneExpert{vi,mi}));
            betweenGroupBin = squareform(pdist(sortedAngle)>0);
            withinGroupBin = squareform(pdist(sortedAngle)==0);
            tsneDistCluster(ind,2) = mean(   ( (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) - (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) )  ...
                ./ ( (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) + (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) )   );
            
            % pca distance grouping quantification
            % naive
            sortedAngle = sortedAngleNaive{vi,mi};
            tempdist = squareform(pdist(pcaNaive{vi,mi}));
            betweenGroupBin = squareform(pdist(sortedAngle)>0);
            withinGroupBin = squareform(pdist(sortedAngle)==0);
            pcaDistCluster(ind,1) = mean(   ( (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) - (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) )  ...
                ./ ( (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) + (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) )   );
            
            % expert
            sortedAngle = sortedAngleExpert{vi,mi};
            tempdist = squareform(pdist(pcaExpert{vi,mi}));
            betweenGroupBin = squareform(pdist(sortedAngle)>0);
            withinGroupBin = squareform(pdist(sortedAngle)==0);
            pcaDistCluster(ind,2) = mean(   ( (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) - (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) )  ...
                ./ ( (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) + (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) )   );
        else
            removeInd = ind;
        end
    end
end

pcaDistCluster(removeInd,:) = [];
tsneDistCluster(removeInd,:) = [];

figure('units','norm','outerpos',[0.3 0.3 0.4 0.3]),
subplot(121), hold on
for i = 1 : size(tsneDistCluster,1)
    plot([1,2], tsneDistCluster(i,:), 'ko-')
end
errorbar(1, mean(tsneDistCluster(:,1)), sem(tsneDistCluster(:,1)), 'r')
errorbar(2, mean(tsneDistCluster(:,2)), sem(tsneDistCluster(:,2)), 'r')
[~, p] = ttest(tsneDistCluster(:,1), tsneDistCluster(:,2));
xlim([0.5 2.5])
xticks([1, 2])
xticklabels({'Naive', 'Expert'})
title({'t-SNE clustering index'; sprintf('p = %.3f', p)})

subplot(122), hold on
for i = 1 : size(pcaDistCluster,1)
    plot([1,2], pcaDistCluster(i,:), 'ko-')
end
errorbar(1, mean(pcaDistCluster(:,1)), sem(pcaDistCluster(:,1)), 'r')
errorbar(2, mean(pcaDistCluster(:,2)), sem(pcaDistCluster(:,2)), 'r')
[~, p] = ttest(pcaDistCluster(:,1), pcaDistCluster(:,2));
xlim([0.5 2.5])
xticks([1, 2])
xticklabels({'Naive', 'Expert'})
title({'PCA clustering index'; sprintf('p = %.3f', p)})



%% clustering index change from 0 touch frame delays
clear
load('Y:\Whiskernas\JK\suite2p\pop_decoding_7angles_dim2_noDelayFrames.mat', '*Naive', '*Expert', 'info')

mice = info.mice;

pcaDistCluster = nan(size(pcaNaive,1) * size(pcaNaive,2), 2); % (:,1) naive, (:,2) expert
tsneDistCluster = nan(size(tsneNaive,1) * size(tsneNaive,2), 2); % (:,1) naive, (:,2) expert

for mi = 1 : length(mice)
    for vi = 1 : 2
        ind = (mi-1) * 2 + vi;
        if ~isempty(pcaNaive{vi,mi})
            % tsne distance grouping quantification
            % naive
            sortedAngle = sortedAngleNaive{vi,mi};
            tempdist = squareform(pdist(tsneNaive{vi,mi}));
            betweenGroupBin = squareform(pdist(sortedAngle)>0);
            withinGroupBin = squareform(pdist(sortedAngle)==0);
            tsneDistCluster(ind,1) = mean(   ( (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) - (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) )  ...
                ./ ( (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) + (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) )   );
            
            % expert
            sortedAngle = sortedAngleExpert{vi,mi};
            tempdist = squareform(pdist(tsneExpert{vi,mi}));
            betweenGroupBin = squareform(pdist(sortedAngle)>0);
            withinGroupBin = squareform(pdist(sortedAngle)==0);
            tsneDistCluster(ind,2) = mean(   ( (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) - (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) )  ...
                ./ ( (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) + (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) )   );
            
            % pca distance grouping quantification
            % naive
            sortedAngle = sortedAngleNaive{vi,mi};
            tempdist = squareform(pdist(pcaNaive{vi,mi}));
            betweenGroupBin = squareform(pdist(sortedAngle)>0);
            withinGroupBin = squareform(pdist(sortedAngle)==0);
            pcaDistCluster(ind,1) = mean(   ( (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) - (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) )  ...
                ./ ( (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) + (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) )   );
            
            % expert
            sortedAngle = sortedAngleExpert{vi,mi};
            tempdist = squareform(pdist(pcaExpert{vi,mi}));
            betweenGroupBin = squareform(pdist(sortedAngle)>0);
            withinGroupBin = squareform(pdist(sortedAngle)==0);
            pcaDistCluster(ind,2) = mean(   ( (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) - (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) )  ...
                ./ ( (sum(tempdist.*withinGroupBin, 2) ./ sum(withinGroupBin, 2)) + (sum(tempdist.*betweenGroupBin, 2) ./ sum(betweenGroupBin, 2)) )   );
        else
            removeInd = ind;
        end
    end
end

pcaDistCluster(removeInd,:) = [];
tsneDistCluster(removeInd,:) = [];

figure('units','norm','outerpos',[0.3 0.3 0.4 0.3]),
subplot(121), hold on
for i = 1 : size(tsneDistCluster,1)
    plot([1,2], tsneDistCluster(i,:), 'ko-')
end
errorbar(1, mean(tsneDistCluster(:,1)), sem(tsneDistCluster(:,1)), 'r')
errorbar(2, mean(tsneDistCluster(:,2)), sem(tsneDistCluster(:,2)), 'r')
[~, p] = ttest(tsneDistCluster(:,1), tsneDistCluster(:,2));
xlim([0.5 2.5])
xticks([1, 2])
xticklabels({'Naive', 'Expert'})
title({'t-SNE clustering index'; sprintf('p = %.3f', p)})

subplot(122), hold on
for i = 1 : size(pcaDistCluster,1)
    plot([1,2], pcaDistCluster(i,:), 'ko-')
end
errorbar(1, mean(pcaDistCluster(:,1)), sem(pcaDistCluster(:,1)), 'r')
errorbar(2, mean(pcaDistCluster(:,2)), sem(pcaDistCluster(:,2)), 'r')
[~, p] = ttest(pcaDistCluster(:,1), pcaDistCluster(:,2));
xlim([0.5 2.5])
xticks([1, 2])
xticklabels({'Naive', 'Expert'})
title({'PCA clustering index'; sprintf('p = %.3f', p)})
