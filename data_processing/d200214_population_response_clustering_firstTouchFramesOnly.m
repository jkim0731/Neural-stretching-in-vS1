% One way of chekcing if increased population response clustering is confounded by other behavioral events
% Use responses from first touch only.
% - The touch should have lasted more than 10 ms (i.e., first >10 ms touch)
% - No frame delays


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
touchDurThresh = 0.01; % touch duration threshold. in s

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
                
                % only if there is long enough protraction touch 
                if ~isempty(find(uTrial.protractionTouchDurationByWhisking > touchDurThresh,1))
                    
                    % index of first qualified touch
                    touchInd = find(uTrial.protractionTouchDurationByWhisking > touchDurThresh, 1, 'first'); 
                    
                    if ~isempty(uTrial.answerLickTime)
                        lastTime = uTrial.answerLickTime;
                    else
                        lastTime = uTrial.poleDownOnsetTime;
                    end

                    % only if the first qualified touch ended before the last time
                    if uTrial.whiskerTime(uTrial.protractionTouchChunksByWhisking{touchInd}(end)) < lastTime
                        lastFrames = zeros(4,1);
                        firstTouchFrames = zeros(4,1);
                        for pi = 1 : 4
                            lastFrames(pi) = find(uTrial.tpmTime{pi} > lastTime, 1, 'first');
                            firstTouchFrames(pi) = find(uTrial.tpmTime{pi} > uTrial.whiskerTime(uTrial.protractionTouchChunksByWhisking{touchInd}(1)), 1, 'first');
                        end
                        
                        % only if first touch frames are before last frames, in all planes
                        if isempty(find(firstTouchFrames - lastFrames >= 0,1))                        
                            trialTouchFramesBinaryAllNeurons = zeros(length(upperPlaneCID), size(uTrial.spk,2), 'logical');
                            trialSpontFramesBinaryAllNeurons = ones(length(upperPlaneCID), size(uTrial.spk,2), 'logical');
                            for pi = 1 : 4
                                planeCind = find(floor(upperPlaneCID/1000) == pi);
                                planeTouchFrames = firstTouchFrames(pi);
                                trialTouchFramesBinaryAllNeurons(planeCind, planeTouchFrames) = 1;

                                planePoleUpFrame = find(uTrial.tpmTime{pi} > uTrial.poleUpOnsetTime, 1, 'first');
                                planePoleDownFrame = find(uTrial.tpmTime{pi} > uTrial.poleDownOnsetTime, 1, 'first');
                                trialSpontFramesBinaryAllNeurons(planeCind, planePoleUpFrame:planePoleDownFrame) = 0;
                            end
                            trialPopTouch = sum(uTrial.spk(upperPlaneCind,:) .* trialTouchFramesBinaryAllNeurons, 2);
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
            end
            noNanInd = find(~isnan(sessionUpperTouchNan(:,1)));
            sessionUpperTouch = sessionUpperTouchNan(noNanInd,:);
            sessionUpperNopole = sessionUpperNopoleNan(noNanInd,:);
            sessionUpperPole = sessionUpperPoleNan(noNanInd,:);
            sessionUpperAll = sessionUpperAllNan(noNanInd,:);
            sessionUpperAngle = sessionUpperAngleNan(noNanInd,:);
            sessionUpperTrialNum = sessionUpperTrialNumNan(noNanInd,:);

            %% Saving (and showing) population activity
            [sortedAngle, sorti] = sort(sessionUpperAngle, 'ascend');
            sortedTrialNum = sessionUpperTrialNum(sorti);
            sessionUpperTouchSorted = sessionUpperTouch(sorti,:);
            sessionUpperNopoleSorted = sessionUpperNopole(sorti,:);
            sessionUpperPoleSorted = sessionUpperPole(sorti,:);
            sessionUpperAllSorted = sessionUpperAll(sorti,:);

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
                figure, subplot(411), imagesc(sessionUpperTouchSortedSorted), title('First touch frames, pre-answer (sorted by angle(Y), activity(X), and tuned angle(X))'),
                subplot(413), imagesc(sessionUpperNopoleSortedSorted), title('No pole period (sorted by angle(Y), activity(X), and tuned angle(X))'),
                subplot(412), imagesc(sessionUpperPoleSortedSorted), title('Pole period (sorted by angle(Y), activity(X), and tuned angle(X))'),
                subplot(414), imagesc(sessionUpperAllSortedSorted), title('All frames (sorted by angle(Y), activity(X), and tuned angle(X))')
            end

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

            % only if there is long enough protraction touch 
            if ~isempty(find(uTrial.protractionTouchDurationByWhisking > touchDurThresh, 1))

                % index of first qualified touch
                touchInd = find(uTrial.protractionTouchDurationByWhisking > touchDurThresh, 1, 'first'); 

                if ~isempty(uTrial.answerLickTime)
                    lastTime = uTrial.answerLickTime;
                else
                    lastTime = uTrial.poleDownOnsetTime;
                end

                % only if the first qualified touch ended before the last time
                if uTrial.whiskerTime(uTrial.protractionTouchChunksByWhisking{touchInd}(end)) < lastTime
                    lastFrames = zeros(4,1);
                    firstTouchFrames = zeros(4,1);
                    for pi = 1 : 4
                        lastFrames(pi) = find(uTrial.tpmTime{pi} > lastTime, 1, 'first');
                        firstTouchFrames(pi) = find(uTrial.tpmTime{pi} > uTrial.whiskerTime(uTrial.protractionTouchChunksByWhisking{touchInd}(1)), 1, 'first');
                    end

                    % only if first touch frames are before last frames, in all planes
                    if isempty(find(firstTouchFrames - lastFrames >= 0, 1))                        
                        trialTouchFramesBinaryAllNeurons = zeros(length(lowerPlaneCID), size(uTrial.spk,2), 'logical');
                        trialSpontFramesBinaryAllNeurons = ones(length(lowerPlaneCID), size(uTrial.spk,2), 'logical');
                        for pi = 1 : 4
% This is the line that is different from 'Upper plane'                            
                            planeCind = find(floor(lowerPlaneCID/1000) == pi+4);
                            planeTouchFrames = firstTouchFrames(pi);
                            trialTouchFramesBinaryAllNeurons(planeCind, planeTouchFrames) = 1;

                            planePoleUpFrame = find(uTrial.tpmTime{pi} > uTrial.poleUpOnsetTime, 1, 'first');
                            planePoleDownFrame = find(uTrial.tpmTime{pi} > uTrial.poleDownOnsetTime, 1, 'first');
                            trialSpontFramesBinaryAllNeurons(planeCind, planePoleUpFrame:planePoleDownFrame) = 0;
                        end
                        trialPopTouch = sum(uTrial.spk(lowerPlaneCind,:) .* trialTouchFramesBinaryAllNeurons, 2);
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
save('Y:\Whiskernas\JK\suite2p\pop_decoding_7angles_dim2_firstTouch.mat', '*Naive', '*Expert', 'info')




%% Clustering index comparison

clear
load('Y:\Whiskernas\JK\suite2p\pop_decoding_7angles_dim2_firstTouch.mat', '*Naive', '*Expert', 'info')

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


























%% Before first lick

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
touchDurThresh = 0.01; % touch duration threshold. in s

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
                
                % only if there is long enough protraction touch 
                if ~isempty(find(uTrial.protractionTouchDurationByWhisking > touchDurThresh,1))
                    
                    % index of first qualified touch
                    touchInd = find(uTrial.protractionTouchDurationByWhisking > touchDurThresh, 1, 'first'); 
                    
                    allLickTime = union(uTrial.leftLickTime, uTrial.rightLickTime);
                    firstLickTimeAfterPoleUp = allLickTime(find(allLickTime > uTrial.poleUpOnsetTime, 1, 'first'));
                    lastTime = uTrial.poleDownOnsetTime;
                    if ~isempty(firstLickTimeAfterPoleUp)
                        if firstLickTimeAfterPoleUp < lastTime
                            lastTime = firstLickTimeAfterPoleUp;
                        end
                    end

                    % only if the first qualified touch ended before the last time
                    if uTrial.whiskerTime(uTrial.protractionTouchChunksByWhisking{touchInd}(end)) < lastTime
                        lastFrames = zeros(4,1);
                        firstTouchFrames = zeros(4,1);
                        for pi = 1 : 4
                            lastFrames(pi) = find(uTrial.tpmTime{pi} > lastTime, 1, 'first');
                            firstTouchFrames(pi) = find(uTrial.tpmTime{pi} > uTrial.whiskerTime(uTrial.protractionTouchChunksByWhisking{touchInd}(1)), 1, 'first');
                        end
                        
                        % only if first touch frames are before last frames, in all planes
                        if isempty(find(firstTouchFrames - lastFrames >= 0,1))                        
                            trialTouchFramesBinaryAllNeurons = zeros(length(upperPlaneCID), size(uTrial.spk,2), 'logical');
                            trialSpontFramesBinaryAllNeurons = ones(length(upperPlaneCID), size(uTrial.spk,2), 'logical');
                            for pi = 1 : 4
                                planeCind = find(floor(upperPlaneCID/1000) == pi);
                                planeTouchFrames = firstTouchFrames(pi);
                                trialTouchFramesBinaryAllNeurons(planeCind, planeTouchFrames) = 1;

                                planePoleUpFrame = find(uTrial.tpmTime{pi} > uTrial.poleUpOnsetTime, 1, 'first');
                                planePoleDownFrame = find(uTrial.tpmTime{pi} > uTrial.poleDownOnsetTime, 1, 'first');
                                trialSpontFramesBinaryAllNeurons(planeCind, planePoleUpFrame:planePoleDownFrame) = 0;
                            end
                            trialPopTouch = sum(uTrial.spk(upperPlaneCind,:) .* trialTouchFramesBinaryAllNeurons, 2);
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
            end
            noNanInd = find(~isnan(sessionUpperTouchNan(:,1)));
            sessionUpperTouch = sessionUpperTouchNan(noNanInd,:);
            sessionUpperNopole = sessionUpperNopoleNan(noNanInd,:);
            sessionUpperPole = sessionUpperPoleNan(noNanInd,:);
            sessionUpperAll = sessionUpperAllNan(noNanInd,:);
            sessionUpperAngle = sessionUpperAngleNan(noNanInd,:);
            sessionUpperTrialNum = sessionUpperTrialNumNan(noNanInd,:);

            %% Saving (and showing) population activity
            [sortedAngle, sorti] = sort(sessionUpperAngle, 'ascend');
            sortedTrialNum = sessionUpperTrialNum(sorti);
            sessionUpperTouchSorted = sessionUpperTouch(sorti,:);
            sessionUpperNopoleSorted = sessionUpperNopole(sorti,:);
            sessionUpperPoleSorted = sessionUpperPole(sorti,:);
            sessionUpperAllSorted = sessionUpperAll(sorti,:);

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
                figure, subplot(411), imagesc(sessionUpperTouchSortedSorted), title('First touch frames, pre-answer (sorted by angle(Y), activity(X), and tuned angle(X))'),
                subplot(413), imagesc(sessionUpperNopoleSortedSorted), title('No pole period (sorted by angle(Y), activity(X), and tuned angle(X))'),
                subplot(412), imagesc(sessionUpperPoleSortedSorted), title('Pole period (sorted by angle(Y), activity(X), and tuned angle(X))'),
                subplot(414), imagesc(sessionUpperAllSortedSorted), title('All frames (sorted by angle(Y), activity(X), and tuned angle(X))')
            end

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

            % only if there is long enough protraction touch 
            if ~isempty(find(uTrial.protractionTouchDurationByWhisking > touchDurThresh, 1))

                % index of first qualified touch
                touchInd = find(uTrial.protractionTouchDurationByWhisking > touchDurThresh, 1, 'first'); 

                allLickTime = union(uTrial.leftLickTime, uTrial.rightLickTime);
                firstLickTimeAfterPoleUp = allLickTime(find(allLickTime > uTrial.poleUpOnsetTime, 1, 'first'));
                lastTime = uTrial.poleDownOnsetTime;
                if ~isempty(firstLickTimeAfterPoleUp)
                    if firstLickTimeAfterPoleUp < lastTime
                        lastTime = firstLickTimeAfterPoleUp;
                    end
                end

                % only if the first qualified touch ended before the last time
                if uTrial.whiskerTime(uTrial.protractionTouchChunksByWhisking{touchInd}(end)) < lastTime
                    lastFrames = zeros(4,1);
                    firstTouchFrames = zeros(4,1);
                    for pi = 1 : 4
                        lastFrames(pi) = find(uTrial.tpmTime{pi} > lastTime, 1, 'first');
                        firstTouchFrames(pi) = find(uTrial.tpmTime{pi} > uTrial.whiskerTime(uTrial.protractionTouchChunksByWhisking{touchInd}(1)), 1, 'first');
                    end

                    % only if first touch frames are before last frames, in all planes
                    if isempty(find(firstTouchFrames - lastFrames >= 0, 1))                        
                        trialTouchFramesBinaryAllNeurons = zeros(length(lowerPlaneCID), size(uTrial.spk,2), 'logical');
                        trialSpontFramesBinaryAllNeurons = ones(length(lowerPlaneCID), size(uTrial.spk,2), 'logical');
                        for pi = 1 : 4
% This is the line that is different from 'Upper plane'                            
                            planeCind = find(floor(lowerPlaneCID/1000) == pi+4);
                            planeTouchFrames = firstTouchFrames(pi);
                            trialTouchFramesBinaryAllNeurons(planeCind, planeTouchFrames) = 1;

                            planePoleUpFrame = find(uTrial.tpmTime{pi} > uTrial.poleUpOnsetTime, 1, 'first');
                            planePoleDownFrame = find(uTrial.tpmTime{pi} > uTrial.poleDownOnsetTime, 1, 'first');
                            trialSpontFramesBinaryAllNeurons(planeCind, planePoleUpFrame:planePoleDownFrame) = 0;
                        end
                        trialPopTouch = sum(uTrial.spk(lowerPlaneCind,:) .* trialTouchFramesBinaryAllNeurons, 2);
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
save('Y:\Whiskernas\JK\suite2p\pop_decoding_7angles_dim2_firstTouch_beforeFirstLick.mat', '*Naive', '*Expert', 'info')




%% Clustering index comparison

clear
load('Y:\Whiskernas\JK\suite2p\pop_decoding_7angles_dim2_firstTouch_beforeFirstLick.mat', '*Naive', '*Expert', 'info')

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