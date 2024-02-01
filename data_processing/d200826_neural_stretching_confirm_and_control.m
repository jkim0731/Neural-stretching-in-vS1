% Population decoding of object angles (7-angle sessions) from touch responses
% Trial-averaged touch responses before the first lick, from angle-tuned neurons, in L2-4
% before and after learning
% Match the # of tuned neurons before and after, by randomly selecting fewer # of
% neurons from the session that has higher # of tuned neurons, repeat and average.

% Run PCA with [1:10,15:5:35] dims, save data
% Then, calculate clustering index

compName = getenv('computername');
if strcmp(compName, 'HNB228-JINHO')
    baseDir = 'D:\TPM\JK\suite2p\';
else
    baseDir = 'Y:\Whiskernas\JK\suite2p\';
end

saveFn = 'popActivityTuned.mat';

mice = [25,27,30,36,39,52];
numMice = length(mice);
sessions = {[4,19],[3,10],[3,21],[1,17],[1,23],[3,21]};

cIdThresh = 5000;
% touchFrameDelays = [0,1,2]'; % this is important. without 2 delays (i.e., [0,1]'), there's no tuned pattern
touchFrameDelays = [0,1]';

popActNaive = cell(2,numMice);
popActExpert = cell(2,numMice);
numTunedNaive = zeros(2,numMice);
numTunedExpert = zeros(2,numMice);

trialNumsNaive = cell(2,numMice);
trialNumsExpert = cell(2,numMice);
sortedAngleNaive = cell(2,numMice);
sortedAngleExpert = cell(2,numMice);
sortedTouchNaive = cell(2,numMice);
sortedTouchExpert = cell(2,numMice);
sortedCidNaive = cell(2,numMice);
sortedCidExpert = cell(2,numMice);

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

        tunedCellID = spk.touchID(find(spk.tuned));
        upperPlaneCID = intersect(u.cellNums(find(u.cellNums < cIdThresh)), tunedCellID);
        lowerPlaneCID = intersect(u.cellNums(find(u.cellNums > cIdThresh)), tunedCellID);
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
        
        %% first, upper volume
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
            [sortedAngle, sorti] = sort(sessionUpperAngle, 'ascend');
            sortedTrialNum = sessionUpperTrialNum(sorti);
            sessionUpperTouchSorted = sessionUpperTouch(sorti,:);
            sessionUpperNopoleSorted = sessionUpperNopole(sorti,:);
            sessionUpperPoleSorted = sessionUpperPole(sorti,:);
            sessionUpperAllSorted = sessionUpperAll(sorti,:);

            %% sorting by tuned angle and activity level
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

            if si == 1
                popActNaive{1,mi} = sessionUpperTouchSortedSorted;
                trialNumsNaive{1,mi} = sortedTrialNum;
                sortedAngleNaive{1,mi} = sortedAngle;
                sortedTouchNaive{1,mi} = tunedAngleSorted;
                sortedCidNaive{1,mi} = cIDsorted;
                numTunedNaive(1,mi) = length(cIDsorted);                
            elseif si == 2
                popActExpert{1,mi} = sessionUpperTouchSortedSorted;
                trialNumsExpert{1,mi} = sortedTrialNum;
                sortedAngleExpert{1,mi} = sortedAngle;
                sortedTouchExpert{1,mi} = tunedAngleSorted;
                sortedCidExpert{1,mi} = cIDsorted;
                numTunedExpert(1,mi) = length(cIDsorted);
                
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

        [sortedAngle, sorti] = sort(sessionLowerAngle, 'ascend');
        sortedTrialNum = sessionLowerTrialNum(sorti);
        sessionLowerTouchSorted = sessionLowerTouch(sorti,:);
        sessionLowerNopoleSorted = sessionLowerNopole(sorti,:);
        sessionLowerPoleSorted = sessionLowerPole(sorti,:);
        sessionLowerAllSorted = sessionLowerAll(sorti,:);

        %% sorting by tuned angle and activity level
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
        
        if si == 1
            popActNaive{2,mi} = sessionLowerTouchSortedSorted;
            trialNumsNaive{2,mi} = sortedTrialNum;
            sortedAngleNaive{2,mi} = sortedAngle;
            sortedTouchNaive{2,mi} = tunedAngleSorted;
            sortedCidNaive{2,mi} = cIDsorted;
            numTunedNaive(2,mi) = length(cIDsorted);
        elseif si == 2
            popActExpert{2,mi} = sessionLowerTouchSortedSorted;
            trialNumsExpert{2,mi} = sortedTrialNum;
            sortedAngleExpert{2,mi} = sortedAngle;
            sortedTouchExpert{2,mi} = tunedAngleSorted;
            sortedCidExpert{2,mi} = cIDsorted;
            numTunedExpert(2,mi) = length(cIDsorted);
        else
            error('Session index (si) is wrong')
        end

    end
end


%% save processed data
info = struct;
info.mice = mice;
info.sessions = sessions;
info.touchFrameDelays = touchFrameDelays;
info.angles = angles;
save([baseDir, saveFn], '*Naive', '*Expert', 'info')



%% Run PCA and compare 

clear
compName = getenv('computername');
if strcmp(compName, 'HNB228-JINHO')
    baseDir = 'D:\TPM\JK\suite2p\';
else
    baseDir = 'Y:\Whiskernas\JK\suite2p\';
end

saveFn = 'popActivityTuned.mat';
load([baseDir, saveFn], '*Naive', '*Expert', 'info')


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
numDims = [1:10,15:5:35];

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

pcaNaive = cell(numVol,length(numDims));
pcaExpert = cell(numVol,length(numDims));

varExpSpkNaive = cell(numVol, length(numDims));
varExpExpert = cell(numVol, length(numDims));

CIExpert = zeros(numVol, length(numDims));
CISpkNaive = zeros(numVol, length(numDims));

CIExpertAll = cell(numVol, length(numDims));
CINaiveAll = cell(numVol, length(numDims));

startTime = tic;

for vi = 1 : numVol
    lapTime = toc(startTime);
    fprintf('Running vol %d/%d (%d:%d passed)\n', vi, numVol, floor(lapTime/60), floor(mod(lapTime,60)))
    numNeuron = min(numTunedNaive(vi), numTunedExpert(vi));
    if numTunedNaive(vi) > numNeuron % run repeats
        for di = 1 : length(numDims)
            tempPCA = cell(numRepeat,1);
            tempVarExp = cell(numRepeat,1);
            tempClusterInd = zeros(numRepeat,1);
            numDim = numDims(di);
            for ri = 1 : numRepeat
                tempInds = randperm(numTunedNaive(vi), numNeuron);
                [~, tempPCA{ri}, ~, ~, tempVarExp{ri}] = pca(popActNaive{vi}(:,tempInds), 'NumComponents', numDim);
                tempClusterInd(ri) = clustering_index(tempPCA{ri}, sortedAngleNaive{vi});
            end
            [~, sorti] = sort(tempClusterInd, 'descend');
            medInd = sorti(medSort);
            
            pcaNaive{vi,di} = tempPCA{medInd};
            varExpSpkNaive{vi,di} = tempVarExp{medInd};
            CISpkNaive(vi,di) = tempClusterInd(medInd);
            
            CINaiveAll{vi, di} = tempClusterInd;
            
        end
        
        % in this case, expert session has the min # of neurons
        for di = 1 : length(numDims)
            numDim = numDims(di);
            [~, pcaExpert{vi, di}, ~, ~, varExpExpert{vi, di}] = pca(popActExpert{vi}, 'NumComponents', numDim);
            CIExpert(vi,di) = clustering_index(pcaExpert{vi,di}, sortedAngleExpert{vi});
        end
    else
        for di = 1 : length(numDims)
            numDim = numDims(di);
            [~, pcaNaive{vi, di}, ~, ~, varExpSpkNaive{vi, di}] = pca(popActNaive{vi}, 'NumComponents', numDim);
            CISpkNaive(vi,di) = clustering_index(pcaNaive{vi,di}, sortedAngleNaive{vi});
        end
        
        % in this case, run repeats in the expert session
        for di = 1 : length(numDims)
            tempPCA = cell(numRepeat,1);
            tempVarExp = cell(numRepeat,1);
            tempClusterInd = zeros(numRepeat,1);
            numDim = numDims(di);
            for ri = 1 : numRepeat
                tempInds = randperm(numTunedExpert(vi), numNeuron);
                [~, tempPCA{ri}, ~, ~, tempVarExp{ri}] = pca(popActExpert{vi}(:,tempInds),  'NumComponents', numDim);
                tempClusterInd(ri) = clustering_index(tempPCA{ri}, sortedAngleExpert{vi});
            end
            [~, sorti] = sort(tempClusterInd, 'descend');
            medInd = sorti(medSort);
            
            pcaExpert{vi,di} = tempPCA{medInd};
            varExpExpert{vi,di} = tempVarExp{medInd};
            CIExpert(vi,di) = tempClusterInd(medInd);
            
            CIExpertAll{vi, di} = tempClusterInd;
        end
    end
end


% %% save the data
saveFn = 'pcaResultsTuned.mat';
save([baseDir, saveFn], 'pca*', 'varExp*', 'CI*')

%%
clear
compName = getenv('computername');
if strcmp(compName, 'HNB228-JINHO')
    baseDir = 'D:\TPM\JK\suite2p\';
else
    baseDir = 'Y:\Whiskernas\JK\suite2p\';
end

saveFn = 'pcaResultsTuned.mat';
load([baseDir, saveFn], 'pca*', 'varExp*', 'CI*')

%%

colorsTransient = [248 171 66; 40 170 225] / 255;
colorsPersistent = [1 0 0; 0 0 1];
numDims = [1:10,15:5:35];


figure, hold on
plot(numDims, mean(CINaive), '-', 'color', colorsTransient(1,:))
plot(numDims, mean(CIExpert), '-', 'color', colorsTransient(2,:))
legend({'Naive', 'Expert'}, 'autoupdate', false)
boundedline(numDims, mean(CINaive), sem(CINaive), 'cmap', colorsTransient(1,:))
boundedline(numDims, mean(CIExpert), sem(CIExpert), 'cmap', colorsTransient(2,:))
plot(numDims, mean(CINaive), '-', 'color', colorsTransient(1,:))
plot(numDims, mean(CIExpert), '-', 'color', colorsTransient(2,:))
xlabel('# of components')
ylabel('Clustering index')

%%
veNaive = cell2mat(cellfun(@(x) cumsum(x(1:max(numDims))), varExpNaive(:,1), 'un', 0)')';
veExpert = cell2mat(cellfun(@(x) cumsum(x(1:max(numDims))), varExpExpert(:,1), 'un', 0)')';
figure, hold on
plot(1:max(numDims), mean(veNaive), '-', 'color', colorsTransient(1,:))
plot(1:max(numDims), mean(veExpert), '-', 'color', colorsTransient(2,:))
legend({'Naive', 'Expert'}, 'autoupdate', false)
boundedline(1:max(numDims), mean(veNaive), sem(veNaive), 'cmap', colorsTransient(1,:))
boundedline(1:max(numDims), mean(veExpert), sem(veExpert), 'cmap', colorsTransient(2,:))
plot(1:max(numDims), mean(veNaive), '-', 'color', colorsTransient(1,:))
plot(1:max(numDims), mean(veExpert), '-', 'color', colorsTransient(2,:))
xlabel('# of components')
ylabel('Variance explained (%)')
ylim([0 100])

%%
CIdiff = CIExpert - CISpkNaive;
figure, 
boundedline(numDims, mean(CIdiff), sem(CIdiff), 'k')
xlabel('# of components')
ylabel('\DeltaClustering index')






%% Using reconstructions from whisker feature encodings
% use 'wkv_angle_tuning_model_v9.mat'

clear
baseDir = 'D:\TPM\JK\suite2p\';
load([baseDir, 'wkv_angle_tuning_model_v9.mat'], 'expert', 'naive')
naive = naive([1:4,7,9]);

saveFn = 'pca_from_whisker_model.mat';

angles = 45:15:135;
numDims = [1:9,10:5:35];
numRepeat = 101;
medSort = 51;

popAct.spkNaive = cell(2,6); % for confirmation of the calculation
popAct.spkExpert = cell(2,6);
popAct.fullModelNaive = cell(2,6, 13); % full whisker model, with each whisker feature removed (leave-one-out)
popAct.fullModelExpert = cell(2,6, 13);
popAct.woModelNaive = cell(2,6, 13); % whisker-only model, with each whisker feature removed (leave-one-out)
popAct.woModelExpert = cell(2,6, 13);
popAct.fullBehavNaive = cell(2,6, 5); % full whisker model where each behavior category removed (leave-one-out only)
popAct.fullBehavExpert = cell(2,6, 5);
popAct.sortedAngleNaive = cell(2,6);
popAct.sortedAngleExpert = cell(2,6);

pcaResult = struct; % save results from each # of components
% This includes pca loading, pca coordinates, angles, variance explained, and clustering index

resultInd = 0;
for mi = 1 : 6 % mouse index
    fprintf('Processing mouse %d/6\n', mi)
    fprintf('Upper volume.\n')
    % Upper plane
    % Naive
    if mi ~= 2 % I don't have upper plane from JK027 (2nd mouse)
        ci = intersect(find(naive(mi).cIDAll < 5000), find(naive(mi).tuned)); % using angle-tuned neurons only
        tempAngleLengths = [0,cumsum(cellfun(@length, naive(mi).spikeAngleAll{ci(1)}))];
        tempTrialNum = tempAngleLengths(end);
        tempSortedAngle = zeros(tempTrialNum,1);
        for ai = 1 : length(tempAngleLengths)-1
            tempSortedAngle(tempAngleLengths(ai)+1:tempAngleLengths(ai+1)) = angles(ai);
        end
        popAct.sortedAngleNaive{1,mi} = tempSortedAngle;
        naiveNumCell = length(ci);

        tempPopAct = zeros(tempTrialNum, length(ci));
        for ai = 1 : length(tempAngleLengths)-1
            tempPopAct(tempAngleLengths(ai)+1:tempAngleLengths(ai+1), :) = cell2mat(cellfun(@(x) x{ai}, naive(mi).spikeAngleAll(ci), 'un', 0)');
        end
        popAct.spkNaive{1,mi} = tempPopAct;

        for fi = 1 : 13
            tempPopAct = zeros(tempTrialNum, length(ci));
            for ai = 1 : length(tempAngleLengths)-1
                tempPopAct(tempAngleLengths(ai)+1:tempAngleLengths(ai+1), :) = cell2mat(cellfun(@(x) x{fi,ai}, naive(mi).fullModelAngleAll(ci), 'un', 0)');
            end
            popAct.fullModelNaive{1,mi,fi} = tempPopAct;

            tempPopAct = zeros(tempTrialNum, length(ci));
            for ai = 1 : length(tempAngleLengths)-1
                tempPopAct(tempAngleLengths(ai)+1:tempAngleLengths(ai+1), :) = cell2mat(cellfun(@(x) x{fi,ai}, naive(mi).whiskerOnlyAngleAll(ci), 'un', 0)');
            end
            popAct.woModelNaive{1,mi,fi} = tempPopAct;
        end
        
        for bi = 1 : 5
            tempPopAct = zeros(tempTrialNum, length(ci));
            for ai = 1 : length(tempAngleLengths)-1
                tempPopAct(tempAngleLengths(ai)+1:tempAngleLengths(ai+1), :) = cell2mat(cellfun(@(x) x{bi,ai}, naive(mi).fullBehaviorAngleAll(ci), 'un', 0)');
            end
            popAct.fullBehavNaive{1,mi,bi} = tempPopAct;
        end
        
        
        % Expert
        ci = intersect(find(expert(mi).cIDAll < 5000), find(expert(mi).tuned));
        tempAngleLengths = [0,cumsum(cellfun(@length, expert(mi).spikeAngleAll{ci(1)}))];
        tempTrialNum = tempAngleLengths(end);
        tempSortedAngle = zeros(tempTrialNum,1);
        for ai = 1 : length(tempAngleLengths)-1
            tempSortedAngle(tempAngleLengths(ai)+1:tempAngleLengths(ai+1)) = angles(ai);
        end
        popAct.sortedAngleExpert{1,mi} = tempSortedAngle;
        expertNumCell = length(ci);

        tempPopAct = zeros(tempTrialNum, length(ci));
        for ai = 1 : length(tempAngleLengths)-1
            tempPopAct(tempAngleLengths(ai)+1:tempAngleLengths(ai+1), :) = cell2mat(cellfun(@(x) x{ai}, expert(mi).spikeAngleAll(ci), 'un', 0)');
        end
        popAct.spkExpert{1,mi} = tempPopAct;

        for fi = 1 : 13
            tempPopAct = zeros(tempTrialNum, length(ci));
            for ai = 1 : length(tempAngleLengths)-1
                tempPopAct(tempAngleLengths(ai)+1:tempAngleLengths(ai+1), :) = cell2mat(cellfun(@(x) x{fi,ai}, expert(mi).fullModelAngleAll(ci), 'un', 0)');
            end
            popAct.fullModelExpert{1,mi,fi} = tempPopAct;

            tempPopAct = zeros(tempTrialNum, length(ci));
            for ai = 1 : length(tempAngleLengths)-1
                tempPopAct(tempAngleLengths(ai)+1:tempAngleLengths(ai+1), :) = cell2mat(cellfun(@(x) x{fi,ai}, expert(mi).whiskerOnlyAngleAll(ci), 'un', 0)');
            end
            popAct.woModelExpert{1,mi,fi} = tempPopAct;
        end
        
        for bi = 1 : 5
            tempPopAct = zeros(tempTrialNum, length(ci));
            for ai = 1 : length(tempAngleLengths)-1
                tempPopAct(tempAngleLengths(ai)+1:tempAngleLengths(ai+1), :) = cell2mat(cellfun(@(x) x{bi,ai}, expert(mi).fullBehaviorAngleAll(ci), 'un', 0)');
            end
            popAct.fullBehavExpert{1,mi,bi} = tempPopAct;
        end
        
        fprintf('Running PCA from upper volume.\n')
        % Result variables initialization before running PCA
%         pcaCoordSpkNaive = cell(1,1);
%         varExpSpkNaive = zeros(1,length(numDims));
        CISpkNaive = zeros(length(numDims),1);
        
        pcaCoordFullNaive = cell(1,13);
        varExpFullNaive = cell(1,13);
        CIFullNaive = cell(1,13);
        
        pcaCoordWhiskerNaive = cell(1,13);
        varExpWhiskerNaive = cell(1,13);
        CIWhiskerNaive = cell(1,13);
        
        pcaCoordBehavNaive = cell(1,5);
        varExpBehavNaive = cell(1,5);
        CIBehavNaive = cell(1,5);
        
%         pcaCoordSpkExpert = cell(1,1);
%         varExpSpkExpert = zeros(1,length(numDims));
        CISpkExpert = zeros(length(numDims),1);
        
        pcaCoordFullExpert = cell(1,13);
        varExpFullExpert = cell(1,13);
        CIFullExpert = cell(1,13);
        
        pcaCoordWhiskerExpert = cell(1,13);
        varExpWhiskerExpert = cell(1,13);
        CIWhiskerExpert = cell(1,13);

        pcaCoordBehavExpert = cell(1,5);
        varExpBehavExpert = cell(1,5);
        CIBehavExpert = cell(1,5);
        
        % Run PCA and collect results
        if naiveNumCell < expertNumCell
            % From spikes
            [~, pcaCoordSpkNaive, ~, ~, varExpSpkNaive] = pca(popAct.spkNaive{1,mi});
            for di = 1 : length(numDims)
                numDim = numDims(di);
                CISpkNaive(di) = clustering_index(pcaCoordSpkNaive(:,1:numDim), popAct.sortedAngleNaive{1,mi});
            end
            
            % From full models
            for fi = 1 : 13
                [~, pcaCoordFullNaive{fi}, ~, ~, varExpFullNaive{fi}] = pca(popAct.fullModelNaive{1,mi,fi});
                CIFullNaive{fi} = zeros(length(numDims),1);
                for di = 1 : length(numDims)
                    numDim = numDims(di);
                    CIFullNaive{fi}(di) = clustering_index(pcaCoordFullNaive{fi}(:,1:numDim), popAct.sortedAngleNaive{1,mi});
                end
            end
            
            % From whisker-only models
            for fi = 1 : 13
                [~, pcaCoordWhiskerNaive{fi}, ~, ~, varExpWhiskerNaive{fi}] = pca(popAct.woModelNaive{1,mi,fi});
                CIWhiskerNaive{fi} = zeros(length(numDims),1);
                for di = 1 : length(numDims)
                    numDim = numDims(di);
                    CIWhiskerNaive{fi}(di) = clustering_index(pcaCoordWhiskerNaive{fi}(:,1:numDim), popAct.sortedAngleNaive{1,mi});
                end
            end
            
            % From full models, removing each behavior category
            for bi = 1 : 5
                [~, pcaCoordBehavNaive{bi}, ~, ~, varExpBehavNaive{bi}] = pca(popAct.fullBehavNaive{1,mi,bi});
                CIBehavNaive{bi} = zeros(length(numDims),1);
                for di = 1 : length(numDims)
                    numDim = numDims(di);
                    CIBehavNaive{bi}(di) = clustering_index(pcaCoordBehavNaive{bi}(:,1:numDim), popAct.sortedAngleNaive{1,mi});
                end
            end
            
            
            % In this case, run repeats in the expert session
            % with the random selection of number of angle-tuned neurons in
            % the naive session
            
            % Choose the median from 3 components
            
            % From spikes
            tempCoord = cell(numRepeat,1);
            tempVarExp = cell(numRepeat,1);
            tempClusterInd = zeros(numRepeat,1); % from 3 components
            for ri = 1 : numRepeat
                tempInds = randperm(expertNumCell, naiveNumCell);
                [~, tempCoord{ri}, ~, ~, tempVarExp{ri}] = pca(popAct.spkExpert{1,mi}(:,tempInds));
                tempClusterInd(ri) = clustering_index(tempCoord{ri}(:,1:3), popAct.sortedAngleExpert{1,mi});
            end
            
            [~, sorti] = sort(tempClusterInd, 'descend');
            medInd = sorti(medSort);
            
            pcaCoordSpkExpert = tempCoord{medInd};
            varExpSpkExpert = tempVarExp{medInd};
            
            for di = 1 : length(numDims)
                numDim = numDims(di);
                CISpkExpert(di) = clustering_index(pcaCoordSpkExpert(:,1:numDim), popAct.sortedAngleExpert{1,mi});
            end
            
            % From full models
            for fi = 1 : 13
                tempCoord = cell(numRepeat,1);
                tempVarExp = cell(numRepeat,1);
                tempClusterInd = zeros(numRepeat,1); % from 3 components
                for ri = 1 : numRepeat
                    tempInds = randperm(expertNumCell, naiveNumCell);
                    [~, tempCoord{ri}, ~, ~, tempVarExp{ri}] = pca(popAct.fullModelExpert{1,mi,fi}(:,tempInds));
                    tempClusterInd(ri) = clustering_index(tempCoord{ri}(:,1:3), popAct.sortedAngleExpert{1,mi});
                end

                [~, sorti] = sort(tempClusterInd, 'descend');
                medInd = sorti(medSort);

                pcaCoordFullExpert{fi} = tempCoord{medInd};
                varExpFullExpert{fi} = tempVarExp{medInd};

                CIFullExpert{fi} = zeros(length(numDims),1);
                for di = 1 : length(numDims)
                    numDim = numDims(di);
                    CIFullExpert{fi}(di) = clustering_index(pcaCoordFullExpert{fi}(:,1:numDim), popAct.sortedAngleExpert{1,mi});
                end
            end
            
            % From whisker-only models
            for fi = 1 : 13
                tempCoord = cell(numRepeat,1);
                tempVarExp = cell(numRepeat,1);
                tempClusterInd = zeros(numRepeat,1); % from 3 components
                for ri = 1 : numRepeat
                    tempInds = randperm(expertNumCell, naiveNumCell);
                    [~, tempCoord{ri}, ~, ~, tempVarExp{ri}] = pca(popAct.woModelExpert{1,mi,fi}(:,tempInds));
                    tempClusterInd(ri) = clustering_index(tempCoord{ri}(:,1:3), popAct.sortedAngleExpert{1,mi});
                end

                [~, sorti] = sort(tempClusterInd, 'descend');
                medInd = sorti(medSort);

                pcaCoordWhiskerExpert{fi} = tempCoord{medInd};
                varExpWhiskerExpert{fi} = tempVarExp{medInd};

                CIWhiskerExpert{fi} = zeros(length(numDims),1);
                for di = 1 : length(numDims)
                    numDim = numDims(di);
                    CIWhiskerExpert{fi}(di) = clustering_index(pcaCoordWhiskerExpert{fi}(:,1:numDim), popAct.sortedAngleExpert{1,mi});
                end
            end
            
            % From full models, removing each behavior category
            for bi = 1 : 5
                tempCoord = cell(numRepeat,1);
                tempVarExp = cell(numRepeat,1);
                tempClusterInd = zeros(numRepeat,1); % from 3 components
                for ri = 1 : numRepeat
                    tempInds = randperm(expertNumCell, naiveNumCell);
                    [~, tempCoord{ri}, ~, ~, tempVarExp{ri}] = pca(popAct.fullBehavExpert{1,mi,bi}(:,tempInds));
                    tempClusterInd(ri) = clustering_index(tempCoord{ri}(:,1:3), popAct.sortedAngleExpert{1,mi});
                end

                [~, sorti] = sort(tempClusterInd, 'descend');
                medInd = sorti(medSort);

                pcaCoordBehavExpert{bi} = tempCoord{medInd};
                varExpBehavExpert{bi} = tempVarExp{medInd};

                CIBehavExpert{bi} = zeros(length(numDims),1);
                for di = 1 : length(numDims)
                    numDim = numDims(di);
                    CIBehavExpert{bi}(di) = clustering_index(pcaCoordBehavExpert{bi}(:,1:numDim), popAct.sortedAngleExpert{1,mi});
                end
            end
        else
            % In this case, run repeats in the naive session
            % with the random selection of number of angle-tuned neurons in
            % the expert session
            
            % From spikes
            tempCoord = cell(numRepeat,1);
            tempVarExp = cell(numRepeat,1);
            tempClusterInd = zeros(numRepeat,1); % from 3 components
            for ri = 1 : numRepeat
                tempInds = randperm(naiveNumCell, expertNumCell);
                [~, tempCoord{ri}, ~, ~, tempVarExp{ri}] = pca(popAct.spkNaive{1,mi}(:,tempInds));
                tempClusterInd(ri) = clustering_index(tempCoord{ri}(:,1:3), popAct.sortedAngleNaive{1,mi});
            end
            
            [~, sorti] = sort(tempClusterInd, 'descend');
            medInd = sorti(medSort);
            
            pcaCoordSpkNaive = tempCoord{medInd};
            varExpSpkNaive = tempVarExp{medInd};
            
            for di = 1 : length(numDims)
                numDim = numDims(di);
                CISpkNaive(di) = clustering_index(pcaCoordSpkNaive(:,1:numDim), popAct.sortedAngleNaive{1,mi});
            end
            
            % From full models
            for fi = 1 : 13
                tempCoord = cell(numRepeat,1);
                tempVarExp = cell(numRepeat,1);
                tempClusterInd = zeros(numRepeat,1); % from 3 components
                for ri = 1 : numRepeat
                    tempInds = randperm(naiveNumCell, expertNumCell);
                    [~, tempCoord{ri}, ~, ~, tempVarExp{ri}] = pca(popAct.fullModelNaive{1,mi,fi}(:,tempInds));
                    tempClusterInd(ri) = clustering_index(tempCoord{ri}(:,1:3), popAct.sortedAngleNaive{1,mi});
                end

                [~, sorti] = sort(tempClusterInd, 'descend');
                medInd = sorti(medSort);

                pcaCoordFullNaive{fi} = tempCoord{medInd};
                varExpFullNaive{fi} = tempVarExp{medInd};

                CIFullNaive{fi} = zeros(length(numDims),1);
                for di = 1 : length(numDims)
                    numDim = numDims(di);
                    CIFullNaive{fi}(di) = clustering_index(pcaCoordFullNaive{fi}(:,1:numDim), popAct.sortedAngleNaive{1,mi});
                end
            end
            
            % From whisker-only models
            for fi = 1 : 13
                tempCoord = cell(numRepeat,1);
                tempVarExp = cell(numRepeat,1);
                tempClusterInd = zeros(numRepeat,1); % from 3 components
                for ri = 1 : numRepeat
                    tempInds = randperm(naiveNumCell, expertNumCell);
                    [~, tempCoord{ri}, ~, ~, tempVarExp{ri}] = pca(popAct.woModelNaive{1,mi,fi}(:,tempInds));
                    tempClusterInd(ri) = clustering_index(tempCoord{ri}(:,1:3), popAct.sortedAngleNaive{1,mi});
                end

                [~, sorti] = sort(tempClusterInd, 'descend');
                medInd = sorti(medSort);

                pcaCoordWhiskerNaive{fi} = tempCoord{medInd};
                varExpWhiskerNaive{fi} = tempVarExp{medInd};

                CIWhiskerNaive{fi} = zeros(length(numDims),1);
                for di = 1 : length(numDims)
                    numDim = numDims(di);
                    CIWhiskerNaive{fi}(di) = clustering_index(pcaCoordWhiskerNaive{fi}(:,1:numDim), popAct.sortedAngleNaive{1,mi});
                end
            end
            
            % From full models, removing each behavior category
            for bi = 1 : 5
                tempCoord = cell(numRepeat,1);
                tempVarExp = cell(numRepeat,1);
                tempClusterInd = zeros(numRepeat,1); % from 3 components
                for ri = 1 : numRepeat
                    tempInds = randperm(naiveNumCell, expertNumCell);
                    [~, tempCoord{ri}, ~, ~, tempVarExp{ri}] = pca(popAct.fullBehavNaive{1,mi,bi}(:,tempInds));
                    tempClusterInd(ri) = clustering_index(tempCoord{ri}(:,1:3), popAct.sortedAngleNaive{1,mi});
                end

                [~, sorti] = sort(tempClusterInd, 'descend');
                medInd = sorti(medSort);

                pcaCoordBehavNaive{bi} = tempCoord{medInd};
                varExpBehavNaive{bi} = tempVarExp{medInd};

                CIBehavNaive{bi} = zeros(length(numDims),1);
                for di = 1 : length(numDims)
                    numDim = numDims(di);
                    CIBehavNaive{bi}(di) = clustering_index(pcaCoordBehavNaive{bi}(:,1:numDim), popAct.sortedAngleNaive{1,mi});
                end
            end
            
            % Now experts
            % From spikes
            [~, pcaCoordSpkExpert, ~, ~, varExpSpkExpert] = pca(popAct.spkExpert{1,mi});
            for di = 1 : length(numDims)
                numDim = numDims(di);
                CISpkExpert(di) = clustering_index(pcaCoordSpkExpert(:,1:numDim), popAct.sortedAngleExpert{1,mi});
            end
            
            % From full models
            for fi = 1 : 13
                [~, pcaCoordFullExpert{fi}, ~, ~, varExpFullExpert{fi}] = pca(popAct.fullModelExpert{1,mi,fi});
                CIFullExpert{fi} = zeros(length(numDims),1);
                for di = 1 : length(numDims)
                    numDim = numDims(di);
                    CIFullExpert{fi}(di) = clustering_index(pcaCoordFullExpert{fi}(:,1:numDim), popAct.sortedAngleExpert{1,mi});
                end
            end
            
            % From whisker-only models
            for fi = 1 : 13
                [~, pcaCoordWhiskerExpert{fi}, ~, ~, varExpWhiskerExpert{fi}] = pca(popAct.woModelExpert{1,mi,fi});
                CIWhiskerExpert{fi} = zeros(length(numDims),1);
                for di = 1 : length(numDims)
                    numDim = numDims(di);
                    CIWhiskerExpert{fi}(di) = clustering_index(pcaCoordWhiskerExpert{fi}(:,1:numDim), popAct.sortedAngleExpert{1,mi});
                end
            end
            
            % From full models, removing each behavior category
            for bi = 1 : 5
                [~, pcaCoordBehavExpert{bi}, ~, ~, varExpBehavExpert{bi}] = pca(popAct.fullBehavExpert{1,mi,bi});
                CIBehavExpert{bi} = zeros(length(numDims),1);
                for di = 1 : length(numDims)
                    numDim = numDims(di);
                    CIBehavExpert{bi}(di) = clustering_index(pcaCoordBehavExpert{bi}(:,1:numDim), popAct.sortedAngleExpert{1,mi});
                end
            end
        end
        
        % Collect results
        resultInd = resultInd + 1;
        pcaResult(resultInd).mouseInd = mi;
        pcaResult(resultInd).volumeInd = 1;
        
        pcaResult(resultInd).pcaCoordSpkNaive = pcaCoordSpkNaive;
        pcaResult(resultInd).pcaCoordFullNaive = pcaCoordFullNaive;
        pcaResult(resultInd).pcaCoordWhiskerNaive = pcaCoordWhiskerNaive;
        pcaResult(resultInd).pcaCoordBehavNaive = pcaCoordBehavNaive;

        pcaResult(resultInd).pcaCoordSpkExpert = pcaCoordSpkExpert;
        pcaResult(resultInd).pcaCoordFullExpert = pcaCoordFullExpert;
        pcaResult(resultInd).pcaCoordWhiskerExpert = pcaCoordWhiskerExpert;
        pcaResult(resultInd).pcaCoordBehavExpert = pcaCoordBehavExpert;


        pcaResult(resultInd).varExpSpkNaive = varExpSpkNaive;
        pcaResult(resultInd).varExpFullNaive = varExpFullNaive;
        pcaResult(resultInd).varExpWhiskerNaive = varExpWhiskerNaive;
        pcaResult(resultInd).varExpBehavNaive = varExpBehavNaive;

        pcaResult(resultInd).varExpSpkExpert = varExpSpkExpert;
        pcaResult(resultInd).varExpFullExpert = varExpFullExpert;
        pcaResult(resultInd).varExpWhiskerExpert = varExpWhiskerExpert;
        pcaResult(resultInd).varExpBehavExpert = varExpBehavExpert;


        pcaResult(resultInd).CISpkNaive = CISpkNaive;
        pcaResult(resultInd).CIFullNaive = CIFullNaive;
        pcaResult(resultInd).CIWhiskerNaive = CIWhiskerNaive;
        pcaResult(resultInd).CIBehavNaive = CIBehavNaive;

        pcaResult(resultInd).CISpkExpert = CISpkExpert;
        pcaResult(resultInd).CIFullExpert = CIFullExpert;
        pcaResult(resultInd).CIWhiskerExpert = CIWhiskerExpert;
        pcaResult(resultInd).CIBehavExpert = CIBehavExpert;
    end
    
    fprintf('Lower volume.\n')
    % Lower volume
    % Naive
    ci = intersect(find(naive(mi).cIDAll > 5000), find(naive(mi).tuned));
    tempAngleLengths = [0,cumsum(cellfun(@length, naive(mi).spikeAngleAll{ci(1)}))];
    tempTrialNum = tempAngleLengths(end);
    tempSortedAngle = zeros(tempTrialNum,1);
    for ai = 1 : length(tempAngleLengths)-1
        tempSortedAngle(tempAngleLengths(ai)+1:tempAngleLengths(ai+1)) = angles(ai);
    end
    popAct.sortedAngleNaive{2,mi} = tempSortedAngle;
    naiveNumCell = length(ci);
    
    tempPopAct = zeros(tempTrialNum, length(ci));
    for ai = 1 : length(tempAngleLengths)-1
        tempPopAct(tempAngleLengths(ai)+1:tempAngleLengths(ai+1), :) = cell2mat(cellfun(@(x) x{ai}, naive(mi).spikeAngleAll(ci), 'un', 0)');
    end
    popAct.spkNaive{2,mi} = tempPopAct;
    
    for fi = 1 : 13
        tempPopAct = zeros(tempTrialNum, length(ci));
        for ai = 1 : length(tempAngleLengths)-1
            tempPopAct(tempAngleLengths(ai)+1:tempAngleLengths(ai+1), :) = cell2mat(cellfun(@(x) x{fi,ai}, naive(mi).fullModelAngleAll(ci), 'un', 0)');
        end
        popAct.fullModelNaive{2,mi,fi} = tempPopAct;
        
        tempPopAct = zeros(tempTrialNum, length(ci));
        for ai = 1 : length(tempAngleLengths)-1
            tempPopAct(tempAngleLengths(ai)+1:tempAngleLengths(ai+1), :) = cell2mat(cellfun(@(x) x{fi,ai}, naive(mi).whiskerOnlyAngleAll(ci), 'un', 0)');
        end
        popAct.woModelNaive{2,mi,fi} = tempPopAct;
    end
    
    for bi = 1 : 5
        tempPopAct = zeros(tempTrialNum, length(ci));
        for ai = 1 : length(tempAngleLengths)-1
            tempPopAct(tempAngleLengths(ai)+1:tempAngleLengths(ai+1), :) = cell2mat(cellfun(@(x) x{bi,ai}, naive(mi).fullBehaviorAngleAll(ci), 'un', 0)');
        end
        popAct.fullBehavNaive{2,mi,bi} = tempPopAct;
    end
    
    % Expert
    ci = intersect(find(expert(mi).cIDAll > 5000), find(expert(mi).tuned));
    tempAngleLengths = [0,cumsum(cellfun(@length, expert(mi).spikeAngleAll{ci(1)}))];
    tempTrialNum = tempAngleLengths(end);
    tempSortedAngle = zeros(tempTrialNum,1);
    for ai = 1 : length(tempAngleLengths)-1
        tempSortedAngle(tempAngleLengths(ai)+1:tempAngleLengths(ai+1)) = angles(ai);
    end
    popAct.sortedAngleExpert{2,mi} = tempSortedAngle;
    expertNumCell = length(ci);
    
    tempPopAct = zeros(tempTrialNum, length(ci));
    for ai = 1 : length(tempAngleLengths)-1
        tempPopAct(tempAngleLengths(ai)+1:tempAngleLengths(ai+1), :) = cell2mat(cellfun(@(x) x{ai}, expert(mi).spikeAngleAll(ci), 'un', 0)');
    end
    popAct.spkExpert{2,mi} = tempPopAct;
    
    for fi = 1 : 13
        tempPopAct = zeros(tempTrialNum, length(ci));
        for ai = 1 : length(tempAngleLengths)-1
            tempPopAct(tempAngleLengths(ai)+1:tempAngleLengths(ai+1), :) = cell2mat(cellfun(@(x) x{fi,ai}, expert(mi).fullModelAngleAll(ci), 'un', 0)');
        end
        popAct.fullModelExpert{2,mi,fi} = tempPopAct;
        
        tempPopAct = zeros(tempTrialNum, length(ci));
        for ai = 1 : length(tempAngleLengths)-1
            tempPopAct(tempAngleLengths(ai)+1:tempAngleLengths(ai+1), :) = cell2mat(cellfun(@(x) x{fi,ai}, expert(mi).whiskerOnlyAngleAll(ci), 'un', 0)');
        end
        popAct.woModelExpert{2,mi,fi} = tempPopAct;
    end
    
    for bi = 1 : 5
        tempPopAct = zeros(tempTrialNum, length(ci));
        for ai = 1 : length(tempAngleLengths)-1
            tempPopAct(tempAngleLengths(ai)+1:tempAngleLengths(ai+1), :) = cell2mat(cellfun(@(x) x{bi,ai}, expert(mi).fullBehaviorAngleAll(ci), 'un', 0)');
        end
        popAct.fullBehavExpert{2,mi,bi} = tempPopAct;
    end
    
    fprintf('Running PCA from lower volume.\n')
    % Result variables initialization before running PCA
%         pcaCoordSpkNaive = cell(1,1);
%         varExpSpkNaive = zeros(1,length(numDims));
    CISpkNaive = zeros(length(numDims),1);

    pcaCoordFullNaive = cell(1,13);
    varExpFullNaive = cell(1,13);
    CIFullNaive = cell(1,13);

    pcaCoordWhiskerNaive = cell(1,13);
    varExpWhiskerNaive = cell(1,13);
    CIWhiskerNaive = cell(1,13);

    pcaCoordBehavNaive = cell(1,5);
    varExpBehavNaive = cell(1,5);
    CIBehavNaive = cell(1,5);

%         pcaCoordSpkExpert = cell(1,1);
%         varExpSpkExpert = zeros(1,length(numDims));
    CISpkExpert = zeros(length(numDims),1);

    pcaCoordFullExpert = cell(1,13);
    varExpFullExpert = cell(1,13);
    CIFullExpert = cell(1,13);

    pcaCoordWhiskerExpert = cell(1,13);
    varExpWhiskerExpert = cell(1,13);
    CIWhiskerExpert = cell(1,13);
    
    pcaCoordBehavExpert = cell(1,5);
    varExpBehavExpert = cell(1,5);
    CIBehavExpert = cell(1,5);

    % Run PCA and collect results
    if naiveNumCell < expertNumCell
        % From spikes
        [~, pcaCoordSpkNaive, ~, ~, varExpSpkNaive] = pca(popAct.spkNaive{2,mi});
        for di = 1 : length(numDims)
            numDim = numDims(di);
            CISpkNaive(di) = clustering_index(pcaCoordSpkNaive(:,1:numDim), popAct.sortedAngleNaive{2,mi});
        end

        % From full models
        for fi = 1 : 13
            [~, pcaCoordFullNaive{fi}, ~, ~, varExpFullNaive{fi}] = pca(popAct.fullModelNaive{2,mi,fi});
            CIFullNaive{fi} = zeros(length(numDims),1);
            for di = 1 : length(numDims)
                numDim = numDims(di);
                CIFullNaive{fi}(di) = clustering_index(pcaCoordFullNaive{fi}(:,1:numDim), popAct.sortedAngleNaive{2,mi});
            end
        end

        % From whisker-only models
        for fi = 1 : 13
            [~, pcaCoordWhiskerNaive{fi}, ~, ~, varExpWhiskerNaive{fi}] = pca(popAct.woModelNaive{2,mi,fi});
            CIWhiskerNaive{fi} = zeros(length(numDims),1);
            for di = 1 : length(numDims)
                numDim = numDims(di);
                CIWhiskerNaive{fi}(di) = clustering_index(pcaCoordWhiskerNaive{fi}(:,1:numDim), popAct.sortedAngleNaive{2,mi});
            end
        end
        
        % From full models, removing each behavior category
        for bi = 1 : 5
            [~, pcaCoordBehavNaive{bi}, ~, ~, varExpBehavNaive{bi}] = pca(popAct.fullBehavNaive{2,mi,bi});
            CIBehavNaive{bi} = zeros(length(numDims),1);
            for di = 1 : length(numDims)
                numDim = numDims(di);
                CIBehavNaive{bi}(di) = clustering_index(pcaCoordBehavNaive{bi}(:,1:numDim), popAct.sortedAngleNaive{2,mi});
            end
        end

        % In this case, run repeats in the expert session
        % with the random selection of number of angle-tuned neurons in
        % the naive session

        % Choose the median from 3 components

        % From spikes
        tempCoord = cell(numRepeat,1);
        tempVarExp = cell(numRepeat,1);
        tempClusterInd = zeros(numRepeat,1); % from 3 components
        for ri = 1 : numRepeat
            tempInds = randperm(expertNumCell, naiveNumCell);
            [~, tempCoord{ri}, ~, ~, tempVarExp{ri}] = pca(popAct.spkExpert{2,mi}(:,tempInds));
            tempClusterInd(ri) = clustering_index(tempCoord{ri}(:,1:3), popAct.sortedAngleExpert{2,mi});
        end

        [~, sorti] = sort(tempClusterInd, 'descend');
        medInd = sorti(medSort);

        pcaCoordSpkExpert = tempCoord{medInd};
        varExpSpkExpert = tempVarExp{medInd};

        for di = 1 : length(numDims)
            numDim = numDims(di);
            CISpkExpert(di) = clustering_index(pcaCoordSpkExpert(:,1:numDim), popAct.sortedAngleExpert{2,mi});
        end

        % From full models
        for fi = 1 : 13
            tempCoord = cell(numRepeat,1);
            tempVarExp = cell(numRepeat,1);
            tempClusterInd = zeros(numRepeat,1); % from 3 components
            for ri = 1 : numRepeat
                tempInds = randperm(expertNumCell, naiveNumCell);
                [~, tempCoord{ri}, ~, ~, tempVarExp{ri}] = pca(popAct.fullModelExpert{2,mi,fi}(:,tempInds));
                tempClusterInd(ri) = clustering_index(tempCoord{ri}(:,1:3), popAct.sortedAngleExpert{2,mi});
            end

            [~, sorti] = sort(tempClusterInd, 'descend');
            medInd = sorti(medSort);

            pcaCoordFullExpert{fi} = tempCoord{medInd};
            varExpFullExpert{fi} = tempVarExp{medInd};

            CIFullExpert{fi} = zeros(length(numDims),1);
            for di = 1 : length(numDims)
                numDim = numDims(di);
                CIFullExpert{fi}(di) = clustering_index(pcaCoordFullExpert{fi}(:,1:numDim), popAct.sortedAngleExpert{2,mi});
            end
        end

        % From whisker-only models
        for fi = 1 : 13
           tempCoord = cell(numRepeat,1);
            tempVarExp = cell(numRepeat,1);
            tempClusterInd = zeros(numRepeat,1); % from 3 components
            for ri = 1 : numRepeat
                tempInds = randperm(expertNumCell, naiveNumCell);
                [~, tempCoord{ri}, ~, ~, tempVarExp{ri}] = pca(popAct.woModelExpert{2,mi,fi}(:,tempInds));
                tempClusterInd(ri) = clustering_index(tempCoord{ri}(:,1:3), popAct.sortedAngleExpert{2,mi});
            end

            [~, sorti] = sort(tempClusterInd, 'descend');
            medInd = sorti(medSort);

            pcaCoordWhiskerExpert{fi} = tempCoord{medInd};
            varExpWhiskerExpert{fi} = tempVarExp{medInd};

            CIWhiskerExpert{fi} = zeros(length(numDims),1);
            for di = 1 : length(numDims)
                numDim = numDims(di);
                CIWhiskerExpert{fi}(di) = clustering_index(pcaCoordWhiskerExpert{fi}(:,1:numDim), popAct.sortedAngleExpert{2,mi});
            end
        end
        
        % From full models, removing each behavior category
        for bi = 1 : 5
            tempCoord = cell(numRepeat,1);
            tempVarExp = cell(numRepeat,1);
            tempClusterInd = zeros(numRepeat,1); % from 3 components
            for ri = 1 : numRepeat
                tempInds = randperm(expertNumCell, naiveNumCell);
                [~, tempCoord{ri}, ~, ~, tempVarExp{ri}] = pca(popAct.fullBehavExpert{2,mi,bi}(:,tempInds));
                tempClusterInd(ri) = clustering_index(tempCoord{ri}(:,1:3), popAct.sortedAngleExpert{2,mi});
            end

            [~, sorti] = sort(tempClusterInd, 'descend');
            medInd = sorti(medSort);

            pcaCoordBehavExpert{bi} = tempCoord{medInd};
            varExpBehavExpert{bi} = tempVarExp{medInd};

            CIBehavExpert{bi} = zeros(length(numDims),1);
            for di = 1 : length(numDims)
                numDim = numDims(di);
                CIBehavExpert{bi}(di) = clustering_index(pcaCoordBehavExpert{bi}(:,1:numDim), popAct.sortedAngleExpert{2,mi});
            end
        end
        
    else
        % In this case, run repeats in the naive session
        % with the random selection of number of angle-tuned neurons in
        % the expert session

        % From spikes
        tempCoord = cell(numRepeat,1);
        tempVarExp = cell(numRepeat,1);
        tempClusterInd = zeros(numRepeat,1); % from 3 components
        for ri = 1 : numRepeat
            tempInds = randperm(naiveNumCell, expertNumCell);
            [~, tempCoord{ri}, ~, ~, tempVarExp{ri}] = pca(popAct.spkNaive{2,mi}(:,tempInds));
            tempClusterInd(ri) = clustering_index(tempCoord{ri}(:,1:3), popAct.sortedAngleNaive{2,mi});
        end

        [~, sorti] = sort(tempClusterInd, 'descend');
        medInd = sorti(medSort);

        pcaCoordSpkNaive = tempCoord{medInd};
        varExpSpkNaive = tempVarExp{medInd};

        for di = 1 : length(numDims)
            numDim = numDims(di);
            CISpkNaive(di) = clustering_index(pcaCoordSpkNaive(:,1:numDim), popAct.sortedAngleNaive{2,mi});
        end

        % From full models
        for fi = 1 : 13
            tempCoord = cell(numRepeat,1);
            tempVarExp = cell(numRepeat,1);
            tempClusterInd = zeros(numRepeat,1); % from 3 components
            for ri = 1 : numRepeat
                tempInds = randperm(naiveNumCell, expertNumCell);
                [~, tempCoord{ri}, ~, ~, tempVarExp{ri}] = pca(popAct.fullModelNaive{2,mi,fi}(:,tempInds));
                tempClusterInd(ri) = clustering_index(tempCoord{ri}(:,1:3), popAct.sortedAngleNaive{2,mi});
            end

            [~, sorti] = sort(tempClusterInd, 'descend');
            medInd = sorti(medSort);

            pcaCoordFullNaive{fi} = tempCoord{medInd};
            varExpFullNaive{fi} = tempVarExp{medInd};

            CIFullNaive{fi} = zeros(length(numDims),1);
            for di = 1 : length(numDims)
                numDim = numDims(di);
                CIFullNaive{fi}(di) = clustering_index(pcaCoordFullNaive{fi}(:,1:numDim), popAct.sortedAngleNaive{2,mi});
            end
        end

        % From whisker-only models
        for fi = 1 : 13
            tempCoord = cell(numRepeat,1);
            tempVarExp = cell(numRepeat,1);
            tempClusterInd = zeros(numRepeat,1); % from 3 components
            for ri = 1 : numRepeat
                tempInds = randperm(naiveNumCell, expertNumCell);
                [~, tempCoord{ri}, ~, ~, tempVarExp{ri}] = pca(popAct.woModelNaive{2,mi,fi}(:,tempInds));
                tempClusterInd(ri) = clustering_index(tempCoord{ri}(:,1:3), popAct.sortedAngleNaive{2,mi});
            end

            [~, sorti] = sort(tempClusterInd, 'descend');
            medInd = sorti(medSort);

            pcaCoordWhiskerNaive{fi} = tempCoord{medInd};
            varExpWhiskerNaive{fi} = tempVarExp{medInd};

            CIWhiskerNaive{fi} = zeros(length(numDims),1);
            for di = 1 : length(numDims)
                numDim = numDims(di);
                CIWhiskerNaive{fi}(di) = clustering_index(pcaCoordWhiskerNaive{fi}(:,1:numDim), popAct.sortedAngleNaive{2,mi});
            end
        end
        
        % From full models, removing each behavior category
        for bi = 1 : 5
            tempCoord = cell(numRepeat,1);
            tempVarExp = cell(numRepeat,1);
            tempClusterInd = zeros(numRepeat,1); % from 3 components
            for ri = 1 : numRepeat
                tempInds = randperm(naiveNumCell, expertNumCell);
                [~, tempCoord{ri}, ~, ~, tempVarExp{ri}] = pca(popAct.fullBehavNaive{2,mi,bi}(:,tempInds));
                tempClusterInd(ri) = clustering_index(tempCoord{ri}(:,1:3), popAct.sortedAngleNaive{2,mi});
            end

            [~, sorti] = sort(tempClusterInd, 'descend');
            medInd = sorti(medSort);

            pcaCoordBehavNaive{bi} = tempCoord{medInd};
            varExpBehavNaive{bi} = tempVarExp{medInd};

            CIBehavNaive{bi} = zeros(length(numDims),1);
            for di = 1 : length(numDims)
                numDim = numDims(di);
                CIBehavNaive{bi}(di) = clustering_index(pcaCoordBehavNaive{bi}(:,1:numDim), popAct.sortedAngleNaive{2,mi});
            end
        end

        % Now experts
        % From spikes
        [~, pcaCoordSpkExpert, ~, ~, varExpSpkExpert] = pca(popAct.spkExpert{2,mi});
        for di = 1 : length(numDims)
            numDim = numDims(di);
            CISpkExpert(di) = clustering_index(pcaCoordSpkExpert(:,1:numDim), popAct.sortedAngleExpert{2,mi});
        end

        % From full models
        for fi = 1 : 13
            [~, pcaCoordFullExpert{fi}, ~, ~, varExpFullExpert{fi}] = pca(popAct.fullModelExpert{2,mi,fi});
            CIFullExpert{fi} = zeros(length(numDims),1);
            for di = 1 : length(numDims)
                numDim = numDims(di);
                CIFullExpert{fi}(di) = clustering_index(pcaCoordFullExpert{fi}(:,1:numDim), popAct.sortedAngleExpert{2,mi});
            end
        end

        % From whisker-only models
        for fi = 1 : 13
            [~, pcaCoordWhiskerExpert{fi}, ~, ~, varExpWhiskerExpert{fi}] = pca(popAct.woModelExpert{2,mi,fi});
            CIWhiskerExpert{fi} = zeros(length(numDims),1);
            for di = 1 : length(numDims)
                numDim = numDims(di);
                CIWhiskerExpert{fi}(di) = clustering_index(pcaCoordWhiskerExpert{fi}(:,1:numDim), popAct.sortedAngleExpert{2,mi});
            end
        end
        
        % From full models, removing each behavior category
        for bi = 1 : 5
            [~, pcaCoordBehavExpert{bi}, ~, ~, varExpBehavExpert{bi}] = pca(popAct.fullBehavExpert{2,mi,bi});
            CIBehavExpert{bi} = zeros(length(numDims),1);
            for di = 1 : length(numDims)
                numDim = numDims(di);
                CIBehavExpert{bi}(di) = clustering_index(pcaCoordBehavExpert{bi}(:,1:numDim), popAct.sortedAngleExpert{2,mi});
            end
        end
    end

    % Collect results
    resultInd = resultInd + 1;
    pcaResult(resultInd).mouseInd = mi;
    pcaResult(resultInd).volumeInd = 2;

    pcaResult(resultInd).pcaCoordSpkNaive = pcaCoordSpkNaive;
    pcaResult(resultInd).pcaCoordFullNaive = pcaCoordFullNaive;
    pcaResult(resultInd).pcaCoordWhiskerNaive = pcaCoordWhiskerNaive;
    pcaResult(resultInd).pcaCoordBehavNaive = pcaCoordBehavNaive;

    pcaResult(resultInd).pcaCoordSpkExpert = pcaCoordSpkExpert;
    pcaResult(resultInd).pcaCoordFullExpert = pcaCoordFullExpert;
    pcaResult(resultInd).pcaCoordWhiskerExpert = pcaCoordWhiskerExpert;
    pcaResult(resultInd).pcaCoordBehavExpert = pcaCoordBehavExpert;

    
    pcaResult(resultInd).varExpSpkNaive = varExpSpkNaive;
    pcaResult(resultInd).varExpFullNaive = varExpFullNaive;
    pcaResult(resultInd).varExpWhiskerNaive = varExpWhiskerNaive;
    pcaResult(resultInd).varExpBehavNaive = varExpBehavNaive;

    pcaResult(resultInd).varExpSpkExpert = varExpSpkExpert;
    pcaResult(resultInd).varExpFullExpert = varExpFullExpert;
    pcaResult(resultInd).varExpWhiskerExpert = varExpWhiskerExpert;
    pcaResult(resultInd).varExpBehavExpert = varExpBehavExpert;
    

    pcaResult(resultInd).CISpkNaive = CISpkNaive;
    pcaResult(resultInd).CIFullNaive = CIFullNaive;
    pcaResult(resultInd).CIWhiskerNaive = CIWhiskerNaive;
    pcaResult(resultInd).CIBehavNaive = CIBehavNaive;

    pcaResult(resultInd).CISpkExpert = CISpkExpert;
    pcaResult(resultInd).CIFullExpert = CIFullExpert;
    pcaResult(resultInd).CIWhiskerExpert = CIWhiskerExpert;
    pcaResult(resultInd).CIBehavExpert = CIBehavExpert;
end

info.angles = angles;
info.numDims = numDims;
info.numRepeat = numRepeat;
info.medSort = medSort;


save([baseDir, saveFn], 'pcaResult', 'popAct', 'info')



%%
%% Analyze the results
%%
clear
load('pca_from_whisker_model.mat')

numDims = info.numDims;
colorsTransient = [248 171 66; 40 170 225] / 255;
%% Neural stretching in whisker models

ciSpkNaive = zeros(length(pcaResult), length(numDims)); % for sanity check
ciSpkExpert = zeros(length(pcaResult), length(numDims));
ciFullNaive = zeros(length(pcaResult), length(numDims));
ciFullExpert = zeros(length(pcaResult), length(numDims));
ciWhiskerNaive = zeros(length(pcaResult), length(numDims));
ciWhiskerExpert = zeros(length(pcaResult), length(numDims));
for vi = 1 : length(pcaResult)
    ciSpkNaive(vi,:) = pcaResult(vi).CISpkNaive';
    ciSpkExpert(vi,:) = pcaResult(vi).CISpkExpert';
    
    ciFullNaive(vi,:) = pcaResult(vi).CIFullNaive{1}';
    ciFullExpert(vi,:) = pcaResult(vi).CIFullExpert{1}';
    
    ciWhiskerNaive(vi,:) = pcaResult(vi).CIWhiskerNaive{1}';
    ciWhiskerExpert(vi,:) = pcaResult(vi).CIWhiskerExpert{1}';
end


%%
figure, 
subplot(131), hold on
plot(numDims, mean(ciSpkNaive), 'color', colorsTransient(1,:))
plot(numDims, mean(ciSpkExpert), 'color', colorsTransient(2,:))
legend({'Naive', 'Expert'}, 'autoupdate', false)
boundedline(numDims, mean(ciSpkNaive), sem(ciSpkNaive), 'cmap', colorsTransient(1,:))
boundedline(numDims, mean(ciSpkExpert), sem(ciSpkExpert), 'cmap', colorsTransient(2,:))
ylabel('Clustering index')
ylim([0 0.45]), yticks(0:0.1:0.4)
xlabel('# of components')
title('Inferred spikes')

subplot(132), hold on
plot(numDims, mean(ciFullNaive), 'color', colorsTransient(1,:))
plot(numDims, mean(ciFullExpert), 'color', colorsTransient(2,:))
boundedline(numDims, mean(ciFullNaive), sem(ciFullNaive), 'cmap', colorsTransient(1,:))
boundedline(numDims, mean(ciFullExpert), sem(ciFullExpert), 'cmap', colorsTransient(2,:))
ylim([0 0.45]), yticks(0:0.1:0.4)
xlabel('# of components')
title('Full whisker model')

subplot(133), hold on
plot(numDims, mean(ciWhiskerNaive), 'color', colorsTransient(1,:))
plot(numDims, mean(ciWhiskerExpert), 'color', colorsTransient(2,:))
boundedline(numDims, mean(ciWhiskerNaive), sem(ciWhiskerNaive), 'cmap', colorsTransient(1,:))
boundedline(numDims, mean(ciWhiskerExpert), sem(ciWhiskerExpert), 'cmap', colorsTransient(2,:))
ylim([0 0.45]), yticks(0:0.1:0.4)
xlabel('# of components')
title('Whisker-only model')

% There still is increased clustering index when using whisker-only model.
% But, why does the value converge more quickly than using spikes?

%% Look at % variance explained (%VE)

veSpkNaive = zeros(length(pcaResult), max(numDims)); % for sanity check
veSpkExpert = zeros(length(pcaResult), max(numDims));
veFullNaive = zeros(length(pcaResult), max(numDims));
veFullExpert = zeros(length(pcaResult), max(numDims));
veWhiskerNaive = zeros(length(pcaResult), max(numDims));
veWhiskerExpert = zeros(length(pcaResult), max(numDims));
for vi = 1 : length(pcaResult)
    veSpkNaive(vi,:) = cumsum(pcaResult(vi).varExpSpkNaive(1:max(numDims))');
    veSpkExpert(vi,:) = cumsum(pcaResult(vi).varExpSpkExpert(1:max(numDims))');
    
    veFullNaive(vi,:) = cumsum(pcaResult(vi).varExpFullNaive{1}(1:max(numDims))');
    veFullExpert(vi,:) = cumsum(pcaResult(vi).varExpFullExpert{1}(1:max(numDims))');
    
    veWhiskerNaive(vi,:) = cumsum(pcaResult(vi).varExpWhiskerNaive{1}(1:max(numDims))');
    veWhiskerExpert(vi,:) = cumsum(pcaResult(vi).varExpWhiskerExpert{1}(1:max(numDims))');
end

%%
figure, 
subplot(131), hold on
plot(1:max(numDims), mean(veSpkNaive), 'color', colorsTransient(1,:))
plot(1:max(numDims), mean(veSpkExpert), 'color', colorsTransient(2,:))
legend({'Naive', 'Expert'}, 'autoupdate', false)
boundedline(1:max(numDims), mean(veSpkNaive), sem(veSpkNaive), 'cmap', colorsTransient(1,:))
boundedline(1:max(numDims), mean(veSpkExpert), sem(veSpkExpert), 'cmap', colorsTransient(2,:))
ylabel('% Variance explained')
ylim([0 100])
xlabel('# of components')
title('Spikes')

subplot(132), hold on
plot(1:max(numDims), mean(veFullNaive), 'color', colorsTransient(1,:))
plot(1:max(numDims), mean(veFullExpert), 'color', colorsTransient(2,:))
boundedline(1:max(numDims), mean(veFullNaive), sem(veFullNaive), 'cmap', colorsTransient(1,:))
boundedline(1:max(numDims), mean(veFullExpert), sem(veFullExpert), 'cmap', colorsTransient(2,:))
ylim([0 100])
xlabel('# of components')
title('Full whisker model')

subplot(133), hold on
plot(1:max(numDims), mean(veWhiskerNaive), 'color', colorsTransient(1,:))
plot(1:max(numDims), mean(veWhiskerExpert), 'color', colorsTransient(2,:))
boundedline(1:max(numDims), mean(veWhiskerNaive), sem(veWhiskerNaive), 'cmap', colorsTransient(1,:))
boundedline(1:max(numDims), mean(veWhiskerExpert), sem(veWhiskerExpert), 'cmap', colorsTransient(2,:))
ylim([0 100])
xlabel('# of components')
title('Whisker-only model')


%%
%% Compare between data and models
%%
numDims = info.numDims;
ciDiffSpike = zeros(length(pcaResult), length(numDims));
ciDiffFull = zeros(length(pcaResult), length(numDims));
ciDiffWhisker = zeros(length(pcaResult), length(numDims));
for vi = 1 : length(pcaResult)
    ciDiffSpike(vi,:) = pcaResult(vi).CISpkExpert - pcaResult(vi).CISpkNaive;
    ciDiffFull(vi,:) = pcaResult(vi).CIFullExpert{1} - pcaResult(vi).CIFullNaive{1};
    ciDiffWhisker(vi,:) = pcaResult(vi).CIWhiskerExpert{1} - pcaResult(vi).CIWhiskerNaive{1};
end

figure, hold on
plot(numDims, mean(ciDiffSpike), 'k-')
plot(numDims, mean(ciDiffFull), 'c-')
plot(numDims, mean(ciDiffWhisker), 'b-')
legend({'Inferred spike', 'Full model', 'Whisker model'}, 'autoupdate', false)
boundedline(numDims, mean(ciDiffSpike), sem(ciDiffSpike), 'k-')
boundedline(numDims, mean(ciDiffFull), sem(ciDiffFull),'c-')
boundedline(numDims, mean(ciDiffWhisker), sem(ciDiffWhisker),'b-')
plot(numDims, mean(ciDiffSpike), 'k-')
plot(numDims, mean(ciDiffFull), 'c-')
plot(numDims, mean(ciDiffWhisker), 'b-')

xlabel('# of components')
ylabel('\DeltaClustering index')
ylim([0 0.24])




%%
%% How are they fit across training?
%%
% Is full model or whisker model better fit to angle-tuned neurons 
% during expert sessions, compared to naive sessions?
% Compare deviance explained in angle-tuned neurons

baseDir = 'D:\TPM\JK\suite2p\';
load([baseDir, 'wkv_angle_tuning_model_v9.mat'], 'expert', 'naive')
naive = naive([1:4,7,9]);

%%
numMice = length(naive);
numVol = length(pcaResult);
colorsTransient = [248 171 66; 40 170 225] / 255;

deRange = [0:0.01:0.6,1.0];
deCumFullNaive = nan(numMice,length(deRange)-1);
deCumFullExpert = nan(numVol,length(deRange)-1);
deCumWhiskerNaive = nan(numVol,length(deRange)-1);
deCumWhiskerExpert = nan(numVol,length(deRange)-1);
for mi = 1 : numMice
    % upper volume
    if mi ~=2
        naiveInd = intersect(find(naive(mi).tuned), find(naive(mi).cIDAll < 5000));
        expertInd = intersect(find(expert(mi).tuned), find(expert(mi).cIDAll < 5000));
        deCumFullNaive(mi*2-1,:) = histcounts(naive(mi).deFull( naiveInd ), deRange, 'norm', 'cdf');
        deCumFullExpert(mi*2-1,:) = histcounts(expert(mi).deFull( expertInd ), deRange, 'norm', 'cdf');
        deCumWhiskerNaive(mi*2-1,:) = histcounts(naive(mi).deWhiskerOnly( naiveInd ), deRange, 'norm', 'cdf');
        deCumWhiskerExpert(mi*2-1,:) = histcounts(expert(mi).deWhiskerOnly( expertInd ), deRange, 'norm', 'cdf');
    end
    
    % lower volume
    naiveInd = intersect(find(naive(mi).tuned), find(naive(mi).cIDAll > 5000));
    expertInd = intersect(find(expert(mi).tuned), find(expert(mi).cIDAll > 5000));
    deCumFullNaive(mi*2,:) = histcounts(naive(mi).deFull( naiveInd ), deRange, 'norm', 'cdf');
    deCumFullExpert(mi*2,:) = histcounts(expert(mi).deFull( expertInd ), deRange, 'norm', 'cdf');
    deCumWhiskerNaive(mi*2,:) = histcounts(naive(mi).deWhiskerOnly( naiveInd ), deRange, 'norm', 'cdf');
    deCumWhiskerExpert(mi*2,:) = histcounts(expert(mi).deWhiskerOnly( expertInd ), deRange, 'norm', 'cdf');
end

figure, 
subplot(121), hold on
plot(deRange(1:end-1), nanmean(deCumFullNaive), 'color', colorsTransient(1,:))
plot(deRange(1:end-1), nanmean(deCumFullExpert), 'color', colorsTransient(2,:))
legend({'Naive', 'Expert'}, 'autoupdate', false)
boundedline(deRange(1:end-1), nanmean(deCumFullNaive), sem(deCumFullNaive), 'cmap', colorsTransient(1,:))
boundedline(deRange(1:end-1), nanmean(deCumFullExpert), sem(deCumFullExpert), 'cmap', colorsTransient(2,:))
plot(deRange(1:end-1), nanmean(deCumFullNaive), 'color', colorsTransient(1,:))
xlabel('Goodness-of-fit')
ylabel('Cumulative proportion')
title('Full model - angle-tuned neurons')

subplot(122), hold on
plot(deRange(1:end-1), nanmean(deCumWhiskerNaive), 'color', colorsTransient(1,:))
plot(deRange(1:end-1), nanmean(deCumWhiskerExpert), 'color', colorsTransient(2,:))
legend({'Naive', 'Expert'}, 'autoupdate', false)
boundedline(deRange(1:end-1), nanmean(deCumWhiskerNaive), sem(deCumWhiskerNaive), 'cmap', colorsTransient(1,:))
boundedline(deRange(1:end-1), nanmean(deCumWhiskerExpert), sem(deCumWhiskerExpert), 'cmap', colorsTransient(2,:))
plot(deRange(1:end-1), nanmean(deCumWhiskerNaive), 'color', colorsTransient(1,:))
xlabel('Goodness-of-fit')
ylabel('Cumulative proportion')
title('Whisker-only model - angle-tuned neurons')



%%
%% Compare mean and median between two groups, between 
%%
deFullMean = nan(numVol, 2); % (:,1) naive, (:,2) expert
deFullMedian = nan(numVol, 2);
deWhiskerMean = nan(numVol, 2); % (:,1) naive, (:,2) expert
deWhiskerMedian = nan(numVol, 2);

for mi = 1 : numMice
    % upper volume
    if mi ~= 2
        naiveInd = intersect(find(naive(mi).tuned), find(naive(mi).cIDAll < 5000));
        expertInd = intersect(find(expert(mi).tuned), find(expert(mi).cIDAll < 5000));
        
        deFullMean(mi*2-1,1) = mean(naive(mi).deFull( intersect(naiveInd, find(isfinite(naive(mi).deFull))) ));
        deFullMean(mi*2-1,2) = mean(expert(mi).deFull( intersect(expertInd, find(isfinite(expert(mi).deFull))) ));
        deFullMedian(mi*2-1,1) = median(naive(mi).deFull( intersect(naiveInd, find(isfinite(naive(mi).deFull))) ));
        deFullMedian(mi*2-1,2) = median(expert(mi).deFull( intersect(expertInd, find(isfinite(expert(mi).deFull))) ));
        
        deWhiskerMean(mi*2-1,1) = mean(naive(mi).deWhiskerOnly( intersect(naiveInd, find(isfinite(naive(mi).deWhiskerOnly))) ));
        deWhiskerMean(mi*2-1,2) = mean(expert(mi).deWhiskerOnly( intersect(expertInd, find(isfinite(expert(mi).deWhiskerOnly))) ));
        deWhiskerMedian(mi*2-1,1) = median(naive(mi).deWhiskerOnly( intersect(naiveInd, find(isfinite(naive(mi).deWhiskerOnly))) ));
        deWhiskerMedian(mi*2-1,2) = median(expert(mi).deWhiskerOnly( intersect(expertInd, find(isfinite(expert(mi).deWhiskerOnly))) ));
    end
    
    % lower volume
    naiveInd = intersect(find(naive(mi).tuned), find(naive(mi).cIDAll > 5000));
    expertInd = intersect(find(expert(mi).tuned), find(expert(mi).cIDAll > 5000));
    
    deFullMean(mi*2,1) = mean(naive(mi).deFull( intersect(naiveInd, find(isfinite(naive(mi).deFull))) ));
    deFullMean(mi*2,2) = mean(expert(mi).deFull( intersect(expertInd, find(isfinite(expert(mi).deFull))) ));
    deFullMedian(mi*2,1) = median(naive(mi).deFull( intersect(naiveInd, find(isfinite(naive(mi).deFull))) ));
    deFullMedian(mi*2,2) = median(expert(mi).deFull( intersect(expertInd, find(isfinite(expert(mi).deFull))) ));
    
    deWhiskerMean(mi*2,1) = mean(naive(mi).deWhiskerOnly( intersect(naiveInd, find(isfinite(naive(mi).deWhiskerOnly))) ));
    deWhiskerMean(mi*2,2) = mean(expert(mi).deWhiskerOnly( intersect(expertInd, find(isfinite(expert(mi).deWhiskerOnly))) ));
    deWhiskerMedian(mi*2,1) = median(naive(mi).deWhiskerOnly( intersect(naiveInd, find(isfinite(naive(mi).deWhiskerOnly))) ));
    deWhiskerMedian(mi*2,2) = median(expert(mi).deWhiskerOnly( intersect(expertInd, find(isfinite(expert(mi).deWhiskerOnly))) ));
end

figure, 
subplot(121), hold on
for vi = 1 : numVol
    plot(deFullMean(vi,:), 'ko-')
end
errorbar(nanmean(deFullMean), sem(deFullMean), 'ro', 'lines', 'no')
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), ylabel('Mean goodness-of-fit'), xtickangle(45)
ylim([0 0.5])
[~,p,m] = paired_test(deFullMean(:,1), deFullMean(:,2));
title(sprintf('p = %.3f; method = %s', p, m))

subplot(122), hold on
for vi = 1 : numVol
    plot(deFullMedian(vi,:), 'ko-')
end
errorbar(nanmean(deFullMedian), sem(deFullMedian), 'ro', 'lines', 'no')
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), ylabel('Median goodness-of-fit'), xtickangle(45)
ylim([0 0.5])
[~,p,m] = paired_test(deFullMedian(:,1), deFullMedian(:,2));
title(sprintf('p = %.3f; method = %s', p, m))

sgtitle('Full model')


figure, 
subplot(121), hold on
for vi = 1 : numVol
    plot(deWhiskerMean(vi,:), 'ko-')
end
errorbar(nanmean(deWhiskerMean), sem(deWhiskerMean), 'ro', 'lines', 'no')
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), ylabel('Mean goodness-of-fit'), xtickangle(45)
ylim([0 0.5])
[~,p,m] = paired_test(deWhiskerMean(:,1), deWhiskerMean(:,2));
title(sprintf('p = %.3f; method = %s', p, m))

subplot(122), hold on
for vi = 1 : numVol
    plot(deWhiskerMedian(vi,:), 'ko-')
end
errorbar(nanmean(deWhiskerMedian), sem(deWhiskerMedian), 'ro', 'lines', 'no')
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), ylabel('Median goodness-of-fit'), xtickangle(45)
ylim([0 0.5])
[~,p,m] = paired_test(deWhiskerMedian(:,1), deWhiskerMedian(:,2));
title(sprintf('p = %.3f; method = %s', p, m))

sgtitle('Whisker-only model')


%%
%% Sensory input stability
%%

% # of touches, # of whisks, and mean duration of touch
% between 11 pairs of sessions

uDir = 'D:\TPM\JK\suite2p\';
mice = [25,27,30,36,39,52];
numMice = length(mice);
sessions = {[4,19],[3,10],[3,21],[1,17],[1,23],[3,21]};

%%
numTouches = nan(numMice * 2, 2); % # of touches before the answer lick (or before pole down, if not answered). (:,1) naive, (:,2) expert
numWhisks = nan(numMice * 2, 2); % # of whisks before the answer lick (or before pole down, if not answered). (:,1) naive, (:,2) expert
meanDuration = nan(numMice * 2, 2); % Mean duration of touch before the answer lick (or before pole down, if not answered). (:,1) naive, (:,2) expert. in ms.

for mi = 1 : numMice
    fprintf('Processing mouse #%d/%d.\n', mi, numMice)
    mouse = mice(mi);
    
    % naive
    fprintf('Naive session.\n')
    session = sessions{mi}(1);
    ufn = sprintf('UberJK%03dS%02d_NC',mouse, session);
    load(sprintf('%s%03d\\%s',uDir, mouse, ufn), 'u')
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
        
        numTouches((mi-1)*2+1, 1) = mean(cellfun(@length, numTouchCell));
        numWhisks((mi-1)*2+1, 1) = mean(cellfun(@(x,y) length( find(x.whiskerTime(jkWhiskerOnsetNAmplitude(x.theta)) < y) ), utrials, answerLickTime));
        meanDuration((mi-1)*2+1, 1) = mean(cellfun(@(x,y) mean(x.protractionTouchDurationByWhisking(y)), utrials, numTouchCell)) * 1000;  
    end
    % lower
    utrials = u.trials(lowerInd);
    answerLickTime = answerLickTimeAll(lowerInd);
        
    numTouchCell = cellfun(@(x,y) find(x.whiskerTime(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking)) < y), utrials, answerLickTime, 'uniformoutput', false);

    numTouches(mi*2, 1) = mean(cellfun(@length, numTouchCell));
    numWhisks(mi*2, 1) = mean(cellfun(@(x,y) length( find(x.whiskerTime(jkWhiskerOnsetNAmplitude(x.theta)) < y) ), utrials, answerLickTime));
    meanDuration(mi*2, 1) = mean(cellfun(@(x,y) mean(x.protractionTouchDurationByWhisking(y)), utrials, numTouchCell)) * 1000;
    
    % expert
    fprintf('Expert session.\n')
    session = sessions{mi}(2);
    ufn = sprintf('UberJK%03dS%02d_NC',mouse, session);
    load(sprintf('%s%03d\\%s',uDir, mouse, ufn), 'u')
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
        
        numTouches((mi-1)*2+1, 2) = mean(cellfun(@length, numTouchCell));
        numWhisks((mi-1)*2+1, 2) = mean(cellfun(@(x,y) length( find(x.whiskerTime(jkWhiskerOnsetNAmplitude(x.theta)) < y) ), utrials, answerLickTime));
        meanDuration((mi-1)*2+1, 2) = mean(cellfun(@(x,y) mean(x.protractionTouchDurationByWhisking(y)), utrials, numTouchCell)) * 1000;
    end
    % lower
    utrials = u.trials(lowerInd);
    answerLickTime = answerLickTimeAll(lowerInd);
        
    numTouchCell = cellfun(@(x,y) find(x.whiskerTime(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking)) < y), utrials, answerLickTime, 'uniformoutput', false);

    numTouches(mi*2, 2) = mean(cellfun(@length, numTouchCell));
    numWhisks(mi*2, 2) = mean(cellfun(@(x,y) length( find(x.whiskerTime(jkWhiskerOnsetNAmplitude(x.theta)) < y) ), utrials, answerLickTime));
    meanDuration(mi*2, 2) = mean(cellfun(@(x,y) mean(x.protractionTouchDurationByWhisking(y)), utrials, numTouchCell)) * 1000;
end

%%
figure,
subplot(131), hold on
for i = 1 : 12
    plot(numTouches(i,:), 'ko-')
end
errorbar(nanmean(numTouches), sem(numTouches), 'ro', 'lines', 'no')
xlim([0.5 2.5]), xticks([1,2]), xticklabels({'Naive', 'Expert'}), xtickangle(45)
ylimVal = ylim(); ylim([0 ylimVal(2)])
ylabel('Number of touches')
[~, p, m] = paired_test(numTouches(:,1), numTouches(:,2));
title(sprintf('p = %.3f; method = %s',p, m))

subplot(132), hold on
for i = 1 : 12
    plot(numWhisks(i,:), 'ko-')
end
errorbar(nanmean(numWhisks), sem(numWhisks), 'ro', 'lines', 'no')
xlim([0.5 2.5]), xticks([1,2]), xticklabels({'Naive', 'Expert'}), xtickangle(45)
ylimVal = ylim(); ylim([0 ylimVal(2)])
ylabel('Number of whisks')
[~, p, m] = paired_test(numWhisks(:,1), numWhisks(:,2));
title(sprintf('p = %.3f; method = %s',p, m))

subplot(133), hold on
for i = 1 : 12
    plot(meanDuration(i,:), 'ko-')
end
errorbar(nanmean(meanDuration), sem(meanDuration), 'ro', 'lines', 'no')
xlim([0.5 2.5]), xticks([1,2]), xticklabels({'Naive', 'Expert'}), xtickangle(45)
ylimVal = ylim(); ylim([0 ylimVal(2)])
ylabel('Mean touch duration (ms)')
[~, p, m] = paired_test(meanDuration(:,1), meanDuration(:,2));
title(sprintf('p = %.3f; method = %s',p, m))





%% Multinomial GLM for object angle prediction using whisker inputs
%% For population decoding of object angle across learning
%% Divide by imaging volumes

baseDir = 'D:\TPM\JK\suite2p\';
saveFn = 'objectAnglePrediction_answerLick';

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
    load(sprintf('%s%03d\\%s',uDir, mouse, ufn), 'u')
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
    answerLickTime = answerLickTimeAll(lowerInd);

    numTouchCell = cellfun(@(x,y) find(x.whiskerTime(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking)) < y), utrials, answerLickTime, 'uniformoutput', false);

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
    load(sprintf('%s%03d\\%s',uDir, mouse, ufn), 'u')
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
    answerLickTime = answerLickTimeAll(lowerInd);

    numTouchCell = cellfun(@(x,y) find(x.whiskerTime(cellfun(@(z) z(1), x.protractionTouchChunksByWhisking)) < y), utrials, answerLickTime, 'uniformoutput', false);

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


%% Bootsrap to have 100 samples in each real angle (total 700 samples)
% And then calculate the performance (correct rate & error angle)
% Iterate 100 times and then take the mean
% Also calculate chance levels by shuffling matches 
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
%% The effect of sensory input encoding in neural stretching
%% (from full whisker model)
%%
clear
load('pca_from_whisker_model.mat')

numDims = info.numDims;
colorsTransient = [248 171 66; 40 170 225] / 255;

%% First, see difference in clustering index across components
% from full model
diffCI = zeros(length(pcaResult),length(numDims));
for vi = 1 : length(pcaResult)
    diffCI(vi,:) = pcaResult(vi).CIFullExpert{1} - pcaResult(vi).CIFullNaive{1};
end

figure, boundedline(numDims, mean(diffCI), sem(diffCI), 'k-')
xlabel('# of components')
ylabel('\DeltaClustering index')
title('Full model')

%% Compare removing each sensory input
% in raw value
compNum = 2; % at 2-component
compInd = find(numDims == compNum);

behavImpactCI = zeros(length(pcaResult),5); % 5 behavior categories00
for vi = 1 : length(pcaResult)
    tempDiffCI = pcaResult(vi).CIFullExpert{1}(compInd) - pcaResult(vi).CIFullNaive{1}(compInd);
    for bi = 1 : 5
        behavDiffCI = pcaResult(vi).CIBehavExpert{bi}(compInd) - pcaResult(vi).CIBehavNaive{bi}(compInd);
        behavImpactCI(vi,bi) = tempDiffCI - behavDiffCI;
    end
end

figure, hold on
errorbar(1:5, mean(behavImpactCI), sem(behavImpactCI), 'ko', 'lines', 'no')

xticks(1:5), xticklabels({'Whisker', 'Sound', 'Reward', 'Whisking', 'Licking'})
xtickangle(45)
xlim([0.5 5.5])
ylabel('Impact on CI')
legend(sprintf('# of components = %d',compNum))

t = array2table([[1:11]',behavImpactCI], 'VariableNames', {'imVol','v1','v2','v3','v4','v5'});
behav = table([1:5]', 'VariableNames', {'behaviors'});
rm = fitrm(t, 'v1-v5~imVol', 'WithinDesign', behav);
mt = mauchly(rm);
raResult = ranova(rm);
if mt.pValue(1) <= 0.05
    p = raResult.pValueGG(1);
else
    p = raResult.pValue(1);
end

% multiple comparison
% multcompare(rm,'behaviors')

title(sprintf('Repeated ANOVA p = %.3f',p))

[~, p, m] = paired_test(behavImpactCI);

%% Compare removing each sensory input
% in proportion
compNum = 3; % at 2-component
compInd = find(numDims == compNum);

behavImpactCI = zeros(length(pcaResult),5); % 5 behavior categories00
for vi = 1 : length(pcaResult)
    tempDiffCI = pcaResult(vi).CIFullExpert{1}(compInd) - pcaResult(vi).CIFullNaive{1}(compInd);
    for bi = 1 : 5
        behavDiffCI = pcaResult(vi).CIBehavExpert{bi}(compInd) - pcaResult(vi).CIBehavNaive{bi}(compInd);
        behavImpactCI(vi,bi) = (tempDiffCI - behavDiffCI) / tempDiffCI * 100;
    end
end

figure, hold on
errorbar(1:5, mean(behavImpactCI), sem(behavImpactCI), 'ko', 'lines', 'no')

xticks(1:5), xticklabels({'Whisker', 'Sound', 'Reward', 'Whisking', 'Licking'})
xtickangle(45)
xlim([0.5 5.5])
ylabel('Impact on CI (%)')
legend(sprintf('# of components = %d',compNum))

t = array2table([[1:11]',behavImpactCI], 'VariableNames', {'imVol','v1','v2','v3','v4','v5'});
behav = table([1:5]', 'VariableNames', {'behaviors'});
rm = fitrm(t, 'v1-v5~imVol', 'WithinDesign', behav);
mt = mauchly(rm);
raResult = ranova(rm);
if mt.pValue(1) <= 0.05
    p = raResult.pValueGG(1);
else
    p = raResult.pValue(1);
end

% multiple comparison
% multcompare(rm,'behaviors')

title(sprintf('Repeated ANOVA p = %.3f',p))

[~, p, m] = paired_test(behavImpactCI);



%%
saveDir = 'C:\Users\jinho\Dropbox\Works\grant proposal\2021 Simons BTI\Figures\';
fn = 'behavior_effect_CI.eps';
export_fig([saveDir, fn], '-depsc', '-painters', '-r600', '-transparent')
fix_eps_fonts([saveDir, fn])




%%
%% Neural stretching without a behavioral category
%%

%% without licking
woLickInd = 5;
woLickNaive = zeros(11,length(numDims));
woLickExpert = zeros(11,length(numDims));
for vi = 1 : 11
    woLickNaive(vi,:) = pcaResult(vi).CIBehavNaive{woLickInd};
    woLickExpert(vi,:) = pcaResult(vi).CIBehavExpert{woLickInd};
end

woLickDiff = woLickExpert - woLickNaive;
figure, 
subplot(121), hold on
plot(numDims, mean(woLickNaive), 'color', colorsTransient(1,:));
plot(numDims, mean(woLickExpert), 'color', colorsTransient(2,:));
legend({'Naive', 'Expert'}, 'autoupdate', false)
boundedline(numDims, mean(woLickNaive), sem(woLickNaive), 'cmap', colorsTransient(1,:));
boundedline(numDims, mean(woLickExpert), sem(woLickExpert), 'cmap', colorsTransient(2,:));
xlabel('# of components')
ylabel('Clustering index')
ylim([0 0.22])
subplot(122), hold on
plot(numDims, mean(diffCI), 'k')
plot(numDims, mean(woLickDiff), 'r')
legend({'Full model', 'Without Licking'}, 'autoupdate', false)
boundedline(numDims, mean(diffCI), sem(diffCI), 'k')
boundedline(numDims, mean(woLickDiff), sem(woLickDiff), 'r')
xlabel('# of components')
ylabel('\DeltaClustering index')
ylim([0 0.16])
sgtitle('Without Licking')



%% without whisker input
woWIind = 1;
woWInaive = zeros(11,length(numDims));
woWIexpert = zeros(11,length(numDims));
for vi = 1 : 11
    woWInaive(vi,:) = pcaResult(vi).CIBehavNaive{woWIind};
    woWIexpert(vi,:) = pcaResult(vi).CIBehavExpert{woWIind};
end

woWIdiff = woWIexpert - woWInaive;
figure, 
subplot(121), hold on
plot(numDims, mean(woWInaive), 'color', colorsTransient(1,:));
plot(numDims, mean(woWIexpert), 'color', colorsTransient(2,:));
legend({'Naive', 'Expert'}, 'autoupdate', false)
boundedline(numDims, mean(woWInaive), sem(woWInaive), 'cmap', colorsTransient(1,:));
boundedline(numDims, mean(woWIexpert), sem(woWIexpert), 'cmap', colorsTransient(2,:));
xlabel('# of components')
ylabel('Clustering index')
ylim([0 0.22])
subplot(122), hold on
plot(numDims, mean(diffCI), 'k')
plot(numDims, mean(woWIdiff), 'r')
legend({'Full model', 'Without whisker input'}, 'autoupdate', false)
boundedline(numDims, mean(diffCI), sem(diffCI), 'k')
boundedline(numDims, mean(woWIdiff), sem(woWIdiff), 'r')
xlabel('# of components')
ylabel('\DeltaClustering index')
ylim([0 0.16])
sgtitle('Without whisker input')




%%
%% The effect of each sensory input encoding in neural stretching
%% (from whisker-only model)
%%

clear
load('pca_from_whisker_model.mat')

numDims = info.numDims;
colorsTransient = [248 171 66; 40 170 225] / 255;

%% First, see difference in clustering index across components
% from whisker-only model
diffCI = zeros(length(pcaResult),length(numDims));
for vi = 1 : length(pcaResult)
    diffCI(vi,:) = pcaResult(vi).CIWhiskerExpert{1} - pcaResult(vi).CIWhiskerNaive{1};
end

figure, boundedline(numDims, mean(diffCI), sem(diffCI), 'k-')
xlabel('# of components')
ylabel('\DeltaClustering index')
title('Whisker-only model')

% 2-component gives the best difference across learning

%% Compare removing each sensory input
% in raw value
compNum = 3; % at 2-component
compInd = find(numDims == compNum);

whiskerImpactCI = zeros(length(pcaResult),12); % 12 features
for vi = 1 : length(pcaResult)
    tempDiffCI = pcaResult(vi).CIWhiskerExpert{1}(compInd) - pcaResult(vi).CIWhiskerNaive{1}(compInd);
    for fi = 1 : 12
        featureDiffCI = pcaResult(vi).CIWhiskerExpert{fi+1}(compInd) - pcaResult(vi).CIWhiskerNaive{fi+1}(compInd);
        whiskerImpactCI(vi,fi) = tempDiffCI - featureDiffCI;
    end
end


% whiskerTouchMat = [maxDthetaMat, maxDphiMat, maxDkappaHMat, maxDkappaVMat, maxSlideDistanceMat, maxDurationMat, ...    
%                     thetaAtTouchMat, phiAtTouchMat, kappaHAtTouchMat, kappaVAtTouchMat, arcLengthAtTouchMat, touchCountMat];

xpos = [8,9,10,11,13,12, 1,2,3,4,5,6];
figure, hold on
errorbar(xpos, mean(whiskerImpactCI), sem(whiskerImpactCI), 'ko', 'lines', 'no')

xticks([1:6, 8:13]), xticklabels({'\theta', '\phi', '\kappa_H', '\kappa_V', 'arc length', 'touch count', ...
    '\Delta\theta', '\Delta\phi', 'max\Delta\kappa_H', 'max\Delta\kappa_V', 'touch duration', 'slide distance'})
xtickangle(45)
ylabel('Impact on CI')
legend(sprintf('# of components = %d',compNum))

t = array2table([[1:11]',whiskerImpactCI], 'VariableNames', {'imVol','v1','v2','v3','v4','v5','v6','v7','v8','v9','v10','v11','v12'});
feat = table([1:12]', 'VariableNames', {'features'});
rm = fitrm(t, 'v1-v12~imVol', 'WithinDesign', feat);
mt = mauchly(rm);
raResult = ranova(rm);
if mt.pValue(1) <= 0.05
    p = raResult.pValueGG(1);
else
    p = raResult.pValue(1);
end

% multiple comparison
% multcompare(rm,'features')

title(sprintf('Repeated ANOVA p = %.3f',p))

[~, p, m] = paired_test(whiskerImpactCI);


%% Compare removing each sensory input
% in relative change (% change)
compNum = 3; % at 2-component
compInd = find(numDims == compNum);

whiskerImpactCI = zeros(length(pcaResult),12); % 12 features
for vi = 1 : length(pcaResult)
    tempDiffCI = pcaResult(vi).CIWhiskerExpert{1}(compInd) - pcaResult(vi).CIWhiskerNaive{1}(compInd);
    for fi = 1 : 12
        featureDiffCI = pcaResult(vi).CIWhiskerExpert{fi+1}(compInd) - pcaResult(vi).CIWhiskerNaive{fi+1}(compInd);
        whiskerImpactCI(vi,fi) = (tempDiffCI - featureDiffCI)/tempDiffCI * 100;
    end
end


% whiskerTouchMat = [maxDthetaMat, maxDphiMat, maxDkappaHMat, maxDkappaVMat, maxSlideDistanceMat, maxDurationMat, ...    
%                     thetaAtTouchMat, phiAtTouchMat, kappaHAtTouchMat, kappaVAtTouchMat, arcLengthAtTouchMat, touchCountMat];

xpos = [8,9,10,11,13,12, 1,2,3,4,5,6];
figure, hold on
errorbar(xpos, mean(whiskerImpactCI), sem(whiskerImpactCI), 'ko', 'lines', 'no')

xticks([1:6, 8:13]), xticklabels({'\theta', '\phi', '\kappa_H', '\kappa_V', 'arc length', 'touch count', ...
    '\Delta\theta', '\Delta\phi', 'max\Delta\kappa_H', 'max\Delta\kappa_V', 'touch duration', 'slide distance'})
xtickangle(45)
ylabel('Impact on CI (%)')
legend(sprintf('# of components = %d',compNum))

t = array2table([[1:11]',whiskerImpactCI], 'VariableNames', {'imVol','v1','v2','v3','v4','v5','v6','v7','v8','v9','v10','v11','v12'});
feat = table([1:12]', 'VariableNames', {'features'});
rm = fitrm(t, 'v1-v12~imVol', 'WithinDesign', feat);
mt = mauchly(rm);
raResult = ranova(rm);
if mt.pValue(1) <= 0.05
    p = raResult.pValueGG(1);
else
    p = raResult.pValue(1);
end

% multiple comparison
% multcompare(rm,'features')

title(sprintf('Repeated ANOVA p = %.3f',p))

[~, p, m] = paired_test(whiskerImpactCI);


%% Compare removing each sensory input
% in relative change (% change)
% only in during touch features
compNum = 2; % at 2-component
compInd = find(numDims == compNum);

whiskerImpactCI = zeros(length(pcaResult),6); % 6 features
for vi = 1 : length(pcaResult)
    tempDiffCI = pcaResult(vi).CIWhiskerExpert{1}(compInd) - pcaResult(vi).CIWhiskerNaive{1}(compInd);
    for fi = 1 : 6
        featureDiffCI = pcaResult(vi).CIWhiskerExpert{fi+1}(compInd) - pcaResult(vi).CIWhiskerNaive{fi+1}(compInd);
        whiskerImpactCI(vi,fi) = (tempDiffCI - featureDiffCI)/tempDiffCI;
    end
end


% whiskerTouchMat = [maxDthetaMat, maxDphiMat, maxDkappaHMat, maxDkappaVMat, maxSlideDistanceMat, maxDurationMat, ...    
%                     thetaAtTouchMat, phiAtTouchMat, kappaHAtTouchMat, kappaVAtTouchMat, arcLengthAtTouchMat, touchCountMat];

xpos = [1,2,3,4,6,5];
figure, hold on
errorbar(xpos, mean(whiskerImpactCI), sem(whiskerImpactCI), 'ko', 'lines', 'no')

xticks(1:6), xticklabels({'\Delta\theta', '\Delta\phi', 'max\Delta\kappa_H', 'max\Delta\kappa_V', 'touch duration', 'slide distance'})
xtickangle(45)
ylabel('Impact on CI')
legend('# of components = 2')

t = array2table([[1:11]',whiskerImpactCI], 'VariableNames', {'imVol','v1','v2','v3','v4','v5','v6'});
feat = table([1:6]', 'VariableNames', {'features'});
rm = fitrm(t, 'v1-v6~imVol', 'WithinDesign', feat);
mt = mauchly(rm);
raResult = ranova(rm);
if mt.pValue(1) <= 0.05
    p = raResult.pValueGG(1);
else
    p = raResult.pValue(1);
end

% multiple comparison
% multcompare(rm,'features')

title(sprintf('Repeated ANOVA p = %.3f',p))



%%
saveDir = 'C:\Users\jinho\Dropbox\Works\grant proposal\2021 Simons BTI\Figures\';
fn = 'whisker_feature_effect_CI.eps';
export_fig([saveDir, fn], '-depsc', '-painters', '-r600', '-transparent')
fix_eps_fonts([saveDir, fn])




%%
%% Neural stretching without vertical bending
%%
woVBind = 5;
woVBnaive = zeros(11,length(numDims));
woVBexpert = zeros(11,length(numDims));
for vi = 1 : 11
    woVBnaive(vi,:) = pcaResult(vi).CIWhiskerNaive{woVBind};
    woVBexpert(vi,:) = pcaResult(vi).CIWhiskerExpert{woVBind};
end

woVBdiff = woVBexpert - woVBnaive;
figure, 
subplot(121), hold on
plot(numDims, mean(woVBnaive), 'color', colorsTransient(1,:));
plot(numDims, mean(woVBexpert), 'color', colorsTransient(2,:));
legend({'Naive', 'Expert'}, 'autoupdate', false)
boundedline(numDims, mean(woVBnaive), sem(woVBnaive), 'cmap', colorsTransient(1,:));
boundedline(numDims, mean(woVBexpert), sem(woVBexpert), 'cmap', colorsTransient(2,:));
xlabel('# of components')
ylabel('Clustering index')
ylim([0 0.22])
subplot(122), hold on
plot(numDims, mean(diffCI), 'k')
plot(numDims, mean(woVBdiff), 'r')
legend({'Whisker-only model', 'Without vertical bending'}, 'autoupdate', false)
boundedline(numDims, mean(diffCI), sem(diffCI), 'k')
boundedline(numDims, mean(woVBdiff), sem(woVBdiff), 'r')
xlabel('# of components')
ylabel('\DeltaClustering index')
ylim([0 0.16])
sgtitle('Without vertical bending')






