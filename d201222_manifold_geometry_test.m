% 2020/12/22 JK
% Further analysis on 7-angle sessions for neural stretching
% Answering some of the questions, focusing on the geometry of the manifold:
% (1) How does it look when including different types of neurons in response to touch?
% - tuned neurons, touch neurons, non-touch neurons
% - tuning type (selective, broad, complex, non-selective)
% - Shape index: ICA after PCA? ICA instead of PCA?
% - Select representative population from this analysis (I can always
% revisit and test different populations on other Q's later on)
% (2) How does it look when matching cell identities across learning (including silent neurons)?
% (3) Comparing between correct and wrong trials
% (4) From the models (Touch models or whisker models? Try both, first)


%% basic settings
clear
baseDir = 'C:\Users\shires\Dropbox\Works\Projects\2020 Neural stretching in S1\Data\';
load([baseDir, 'cellMatching_beforeNafter.mat'], 'match')
tune = load([baseDir, 'angle_tuning_summary_preAnswer_perTouch_NC.mat'], 'naive', 'expert');
learnerInd = [1:4,7,9];
tune.naive = tune.naive(learnerInd);
jk027inds = find(tune.naive(2).touchID > 5000);

fn = fieldnames(tune.naive);
for i = 1 : length(fn)
    tune.naive(2).(fn{i}) = tune.naive(2).(fn{i})(jk027inds);
end

angles = 45:15:135;

mice = [25,27,30,36,39,52];
sessions = {[4,19],[3,10],[3,21],[1,17],[1,23],[3,21]};

touchFrameBuffer = [0;1]; 

%% (1) How does it look when including different types of neurons in response to touch?
% - Tuned, touch but not tuned, non-touch
% - First, save all responses and neuron ID's, along with the response
% types (non-touch, non-specific touch, angle-tuned touch), angle
% selectivity, tuning type, event rate, object angle, licking response, 
% trial outcome, and session performance (correct rate).
% - Summarize responses in different time points, such as before pole up,
% during pole up before the answer lick, during pole up after the answer
% lick, after pole down, touch response before the first lick, touch
% response after the answer lick.

% All responses and activities are in Hz (by applying  '*u.frameRate' and
% dividing by the number of corresponding frames)
saveFn = 'popResponse_201228.mat';
naive = struct;
expert = struct;
i = 0;
for mi = 1 : 6 % mouse index 1:6
    mouse = mice(mi);

    for vi = 1 : 2 % volume index 1:2, except for mi == 2 (where vi = 2)
        if ~(mi==2 && vi==1)
            i = i + 1;
            %% Naive
            % Neuron ID in the naive session
            idTunedNaive = tune.naive(mi).touchID(tune.naive(mi).tuned==1);
            idNottunedNaive = tune.naive(mi).touchID(tune.naive(mi).tuned==0);
            idNontouchNaive = setdiff(match{mi,1}, tune.naive(mi).touchID);

            % Loading the naive session
            load(sprintf('%sUberJK%03dS%02d_NC',baseDir,mouse,sessions{mi}(1)), 'u')

            % Dividing into imaged volumes
            if vi == 1
                allIDNaive = u.cellNums(u.cellNums<5000);
            else
                allIDNaive = u.cellNums(u.cellNums>5000);
            end

            % Assigning neuronal response type
            indTunedNaive = find(ismember(allIDNaive, idTunedNaive));
            indNottunedNaive = find(ismember(allIDNaive, idNottunedNaive));
            indNontouchNaive = find(ismember(allIDNaive, idNontouchNaive));
            tunedAngleNaive = zeros(length(indTunedNaive),1);
            angleSelectivityNaive = zeros(length(indTunedNaive),1);
            selectiveTunedNaive = zeros(length(indTunedNaive),1, 'logical');
            broadTunedNaive = zeros(length(indTunedNaive),1, 'logical');
            complexTunedNaive = zeros(length(indTunedNaive),1, 'logical');
            for ci = 1 : length(indTunedNaive)
                tempID = allIDNaive(indTunedNaive(ci));
                tempInd = find(tune.naive(mi).touchID == tempID);
                tunedAngleNaive(ci) = tune.naive(mi).tunedAngle( tempInd );
                angleSelectivityNaive(ci) = tune.naive(mi).sharpness( tempInd );
                if tune.naive(mi).unimodalSingle(tempInd)
                    selectiveTunedNaive(ci) = 1;
                elseif tune.naive(mi).unimodalBroad(tempInd)
                    broadTunedNaive(ci) = 1;
                elseif tune.naive(mi).multimodal(tempInd)
                    complexTunedNaive(ci) = 1;
                else
                    error('Tuning type undefined')
                end 
            end

            % Identifying u.trials indices to investigate
            volTrialInd = u.planeTrialInds{vi};
            touchTrialInd = find(cellfun(@(x) length(x.protractionTouchChunksByWhisking), u.trials));
            answerTrialInd = find(cellfun(@(x) ~isempty(x.answerLickTime), u.trials));
            trialInd = intersect(intersect(volTrialInd, touchTrialInd), answerTrialInd);

            % Assigning event rate (from corresponding trials only)
            allSpk = cell2mat(cellfun(@(x) x.spk, u.trials(trialInd)', 'un', 0));
            eventRateNaive = mean(allSpk,2) * u.frameRate;

            % Assigning touch frames based on the imaged plane
            cellPlaneInds = (mod(floor(u.trials{trialInd(1)}.neuindSession/1000)-1,4)+1); % this is same across the whole session

            % Activities of all neurons
            beforePoleUpNaive = zeros(length(allIDNaive), length(trialInd));
            poleBeforeFirstLickNaive = zeros(length(allIDNaive), length(trialInd));
            poleBeforeAnswerNaive = zeros(length(allIDNaive), length(trialInd));
            poleAfterAnswerNaive = zeros(length(allIDNaive), length(trialInd));
            afterPoleDownNaive = zeros(length(allIDNaive), length(trialInd));

            touchBeforeFirstLickNaive = zeros(length(allIDNaive),length(trialInd));
            touchBeforeAnswerNaive = zeros(length(allIDNaive),length(trialInd));
            touchAfterAnswerNaive = zeros(length(allIDNaive),length(trialInd));
            touchResponseNaive = zeros(length(allIDNaive),length(trialInd));
            
            numTouchFrameBeforeFirstLick = zeros(length(allIDNaive),length(trialInd));
            numTouchFrameBeforeAnswer = zeros(length(allIDNaive),length(trialInd));
            numTouchFrameAfterAnswer = zeros(length(allIDNaive),length(trialInd));
            numTouchFrame = zeros(length(allIDNaive),length(trialInd));
            
            numTouchBeforeFirstLick = zeros(1,length(trialInd));
            numTouchBeforeAnswer = zeros(1,length(trialInd));
            numTouchAfterAnswer = zeros(1,length(trialInd));
            numTouch = zeros(1,length(trialInd));
            % At each trial, 
            for ti = 1 : length(trialInd)
                tempTrial = u.trials{trialInd(ti)};

                % Make masks for each situation
                beforePoleUpMask = zeros(size(tempTrial.spk), 'logical');
                poleBeforeFirstLickMask = zeros(size(tempTrial.spk), 'logical');
                poleBeforeAnswerMask = zeros(size(tempTrial.spk), 'logical');
                poleAfterAnswerMask = zeros(size(tempTrial.spk), 'logical');
                afterPoleDownMask = zeros(size(tempTrial.spk), 'logical');

                touchBeforeFirstLickMask = zeros(size(tempTrial.spk), 'logical');
                touchBeforeAnswerMask = zeros(size(tempTrial.spk), 'logical');
                touchAfterAnswerMask = zeros(size(tempTrial.spk), 'logical');

                touchMask = zeros(size(tempTrial.spk), 'logical');
                for pi = 1 : 4
                    ci = find(cellPlaneInds == pi);
                    tempTouchFrame = unique(tempTrial.protractionTouchOnsetFramesByWhisking{pi}+touchFrameBuffer);
                    touchMask(ci,tempTouchFrame) = 1;

                    poleUpStartFrame = find(tempTrial.tpmTime{pi} < tempTrial.poleMovingTime(1), 1, 'last');
                    poleUpEndFrame = find(tempTrial.tpmTime{pi} > tempTrial.poleUpTime(1), 1, 'first');
                    poleDownStartFrame = find(tempTrial.tpmTime{pi} < tempTrial.poleUpTime(end), 1, 'last');
                    poleDownEndFrame = find(tempTrial.tpmTime{pi} > tempTrial.poleMovingTime(end), 1, 'first') + max(touchFrameBuffer);
                    allLickTimes = union(union(tempTrial.leftLickTime, tempTrial.rightLickTime), tempTrial.answerLickTime);
                    firstLickTime = allLickTimes(find(allLickTimes > tempTrial.poleUpTime(1), 1, 'first'));
                    firstLickFrame = find(tempTrial.tpmTime{pi} < firstLickTime, 1, 'last');
                    answerFrame = find(tempTrial.tpmTime{pi} < tempTrial.answerLickTime, 1, 'last'); % To consider preparatory time

                    beforePoleUpMask(ci, 1 : poleUpStartFrame-1) = 1;
                    poleBeforeFirstLickMask(ci, poleUpEndFrame : firstLickFrame) = 1;
                    poleBeforeAnswerMask(ci, poleUpEndFrame : answerFrame) = 1;
                    poleAfterAnswerMask(ci, answerFrame+1 : poleDownStartFrame) = 1;
                    afterPoleDownMask(ci, poleDownEndFrame : end) = 1;
                    
                    touchBeforeFirstLickMask(ci, :) = touchMask(ci,:) .* poleBeforeFirstLickMask(ci,:);
                    touchBeforeAnswerMask(ci, :) = touchMask(ci,:) .* poleBeforeAnswerMask(ci,:);
                    touchAfterAnswerMask(ci, :) = touchMask(ci,:) .* poleAfterAnswerMask(ci,:);
                end

                numTouchBeforeFirstLick(ti) = length(find(cellfun(@(x) tempTrial.whiskerTime(x(1)), tempTrial.protractionTouchChunksByWhisking) < firstLickTime));
                numTouchBeforeAnswer(ti) = length(find(cellfun(@(x) tempTrial.whiskerTime(x(1)), tempTrial.protractionTouchChunksByWhisking) < tempTrial.answerLickTime));
                numTouchAfterAnswer(ti) = length(find(cellfun(@(x) tempTrial.whiskerTime(x(1)), tempTrial.protractionTouchChunksByWhisking) >= tempTrial.answerLickTime));
                numTouch(ti) = length(tempTrial.protractionTouchChunksByWhisking);

                beforePoleUpNaive(:,ti) = sum(tempTrial.spk .* beforePoleUpMask, 2) ./ sum(beforePoleUpMask,2) * u.frameRate;
                poleBeforeFirstLickNaive(:,ti) = sum(tempTrial.spk .* poleBeforeFirstLickMask, 2) ./ sum(poleBeforeFirstLickMask,2)  * u.frameRate;
                poleBeforeAnswerNaive(:,ti) = sum(tempTrial.spk .* poleBeforeAnswerMask, 2) ./ sum(poleBeforeAnswerMask,2)  * u.frameRate;
                poleAfterAnswerNaive(:,ti) = sum(tempTrial.spk .* poleAfterAnswerMask, 2) ./ sum(poleAfterAnswerMask,2)  * u.frameRate;
                afterPoleDownNaive(:,ti) = sum(tempTrial.spk .* afterPoleDownMask, 2) ./ sum(afterPoleDownMask,2)  * u.frameRate;

                touchBeforeFirstLickNaive(:,ti) = sum(tempTrial.spk .* touchBeforeFirstLickMask,2) / numTouchBeforeFirstLick(ti); % some can be NaN
                touchBeforeAnswerNaive(:,ti) = sum(tempTrial.spk .* touchBeforeAnswerMask,2) / numTouchBeforeAnswer(ti); % some can be NaN
                touchAfterAnswerNaive(:,ti) = sum(tempTrial.spk .* touchAfterAnswerMask,2) / numTouchAfterAnswer(ti); % some can be NaN
                touchResponseNaive(:,ti) = sum(tempTrial.spk .* touchMask,2) / numTouch(ti);
                
                numTouchFrameBeforeFirstLick(:,ti) = sum(touchBeforeFirstLickMask,2);
                numTouchFrameBeforeAnswer(:,ti) = sum(touchBeforeAnswerMask,2);
                numTouchFrameAfterAnswer(:,ti) = sum(touchAfterAnswerMask,2);
                numTouchFrame(:,ti) = sum(touchMask,2);
                
            end
            % Change Inf to NaN (for pca)
            beforePoleUpNaive(find(isinf(beforePoleUpNaive))) = NaN;
            poleBeforeFirstLickNaive(find(isinf(poleBeforeFirstLickNaive))) = NaN;
            poleBeforeAnswerNaive(find(isinf(poleBeforeAnswerNaive))) = NaN;
            poleAfterAnswerNaive(find(isinf(poleAfterAnswerNaive))) = NaN;
            afterPoleDownNaive(find(isinf(afterPoleDownNaive))) = NaN;
            
            touchBeforeFirstLickNaive(find(isinf(touchBeforeFirstLickNaive))) = NaN;
            touchBeforeAnswerNaive(find(isinf(touchBeforeAnswerNaive))) = NaN;
            touchAfterAnswerNaive(find(isinf(touchAfterAnswerNaive))) = NaN;
            touchResponseNaive(find(isinf(touchResponseNaive))) = NaN;
            
            % Trial and session info
            trialAngleNaive = cellfun(@(x) x.angle, u.trials(trialInd));
            trialAnswerNaive = cellfun(@(x) x.choice, u.trials(trialInd), 'un', 0);
            trialOutcomeNaive = cellfun(@(x) x.response, u.trials(trialInd)); % 0 for wrong, 1 for correct
            non90Ind = find(trialAngleNaive ~= 90);
            sessionPerformanceNaive = mean(trialOutcomeNaive(non90Ind));

            % Data gathering
            % Neuron ID and type
            naive(i).allID = allIDNaive;
            naive(i).indTuned = indTunedNaive;
            naive(i).indNottuned = indNottunedNaive;
            naive(i).indNontouch = indNontouchNaive;
            naive(i).eventRate = eventRateNaive;

            % Tuning properties
            naive(i).tunedAngle = tunedAngleNaive;
            naive(i).angleSelectivity = angleSelectivityNaive;
            naive(i).selectiveTuned = selectiveTunedNaive;
            naive(i).broadTuned = broadTunedNaive;
            naive(i).complexTuned = complexTunedNaive;

            % Population response
            naive(i).beforePoleUp = beforePoleUpNaive;
            naive(i).poleBeforeFirstLick = poleBeforeFirstLickNaive;
            naive(i).poleBeforeAnswer = poleBeforeAnswerNaive;
            naive(i).poleAfterAnswer = poleAfterAnswerNaive;
            naive(i).afterPoleDown = afterPoleDownNaive;
            
            naive(i).touchBeforeFirstLick = touchBeforeFirstLickNaive;
            naive(i).touchBeforeAnswer = touchBeforeAnswerNaive;
            naive(i).touchAfterAnswer = touchAfterAnswerNaive;
            naive(i).touchResponse = touchResponseNaive;
            
            naive(i).numTouchFrameBeforeFirstLick = numTouchFrameBeforeFirstLick;
            naive(i).numTouchFrameBeforeAnswer = numTouchFrameBeforeAnswer;
            naive(i).numTouchFrameAfterAnswer = numTouchFrameAfterAnswer;
            naive(i).numTouchFrame = numTouchFrame;

            % Trial parameters
            naive(i).trialAngle = trialAngleNaive;
            naive(i).trialAnswer = trialAnswerNaive;
            naive(i).trialOutcome = trialOutcomeNaive;
            naive(i).sessionPerformance = sessionPerformanceNaive;
            naive(i).frameRate = u.frameRate;
            naive(i).numTouchBeforeFirstLick = numTouchBeforeFirstLick;
            naive(i).numTouchBeforeAnswer = numTouchBeforeAnswer;
            naive(i).numTouchAfterAnswer = numTouchAfterAnswer;
            naive(i).numTouch = numTouch;

            %% Expert
            % Neuron ID in the expert session
            idTunedExpert = tune.expert(mi).touchID(tune.expert(mi).tuned==1);
            idNottunedExpert = tune.expert(mi).touchID(tune.expert(mi).tuned==0);
            idNontouchExpert = setdiff(match{mi,3}, tune.expert(mi).touchID);

            % Loading the expert session
            load(sprintf('%sUberJK%03dS%02d_NC',baseDir,mouse,sessions{mi}(2)), 'u')

            % Dividing into imaged volumes
            if vi == 1
                allIDExpert = u.cellNums(u.cellNums<5000);
            else
                allIDExpert = u.cellNums(u.cellNums>5000);
            end

            % Assigning neuronal response type
            indTunedExpert = find(ismember(allIDExpert, idTunedExpert));
            indNottunedExpert = find(ismember(allIDExpert, idNottunedExpert));
            indNontouchExpert = find(ismember(allIDExpert, idNontouchExpert));
            tunedAngleExpert = zeros(length(indTunedExpert),1);
            angleSelectivityExpert = zeros(length(indTunedExpert),1);
            selectiveTunedExpert = zeros(length(indTunedExpert),1, 'logical');
            broadTunedExpert = zeros(length(indTunedExpert),1, 'logical');
            complexTunedExpert = zeros(length(indTunedExpert),1, 'logical');
            for ci = 1 : length(indTunedExpert)
                tempID = allIDExpert(indTunedExpert(ci));
                tempInd = find(tune.expert(mi).touchID == tempID);
                tunedAngleExpert(ci) = tune.expert(mi).tunedAngle( tempInd );
                angleSelectivityExpert(ci) = tune.expert(mi).sharpness( tempInd );
                if tune.expert(mi).unimodalSingle(tempInd)
                    selectiveTunedExpert(ci) = 1;
                elseif tune.expert(mi).unimodalBroad(tempInd)
                    broadTunedExpert(ci) = 1;
                elseif tune.expert(mi).multimodal(tempInd)
                    complexTunedExpert(ci) = 1;
                else
                    error('Tuning type undefined')
                end
            end

            % Identifying u.trials indices to investigate
            volTrialInd = u.planeTrialInds{vi};
            touchTrialInd = find(cellfun(@(x) length(x.protractionTouchChunksByWhisking), u.trials));
            answerTrialInd = find(cellfun(@(x) ~isempty(x.answerLickTime), u.trials));
            trialInd = intersect(intersect(volTrialInd, touchTrialInd), answerTrialInd);

            % Assigning event rate (from corresponding trials only)
            allSpk = cell2mat(cellfun(@(x) x.spk, u.trials(trialInd)', 'un', 0));
            eventRateExpert = mean(allSpk,2);

            % Assigning touch frames based on the imaged plane
            cellPlaneInds = (mod(floor(u.trials{trialInd(1)}.neuindSession/1000)-1,4)+1); % this is same across the whole session

            % Responses of all neurons
            beforePoleUpExpert = zeros(length(allIDExpert), length(trialInd));
            poleBeforeFirstLickExpert = zeros(length(allIDExpert), length(trialInd));
            poleBeforeAnswerExpert = zeros(length(allIDExpert), length(trialInd));
            poleAfterAnswerExpert = zeros(length(allIDExpert), length(trialInd));
            afterPoleDownExpert = zeros(length(allIDExpert), length(trialInd));

            touchBeforeFirstLickExpert = zeros(length(allIDExpert),length(trialInd));
            touchBeforeAnswerExpert = zeros(length(allIDExpert),length(trialInd));
            touchAfterAnswerExpert = zeros(length(allIDExpert),length(trialInd));
            touchResponseExpert = zeros(length(allIDExpert),length(trialInd));
            
            numTouchFrameBeforeFirstLick = zeros(length(allIDExpert),length(trialInd));
            numTouchFrameBeforeAnswer = zeros(length(allIDExpert),length(trialInd));
            numTouchFrameAfterAnswer = zeros(length(allIDExpert),length(trialInd));
            numTouchFrame = zeros(length(allIDExpert),length(trialInd));
            numTouchBeforeAnswer = zeros(1,length(trialInd));
            numTouchAfterAnswer = zeros(1,length(trialInd));
            numTouch = zeros(1,length(trialInd));
            % At each trial, 
            for ti = 1 : length(trialInd)
                tempTrial = u.trials{trialInd(ti)};

                % Make masks for each situation
                beforePoleUpMask = zeros(size(tempTrial.spk), 'logical');
                poleBeforeFirstLickMask = zeros(size(tempTrial.spk), 'logical');
                poleBeforeAnswerMask = zeros(size(tempTrial.spk), 'logical');
                poleAfterAnswerMask = zeros(size(tempTrial.spk), 'logical');
                afterPoleDownMask = zeros(size(tempTrial.spk), 'logical');

                touchBeforeFirstLickMask = zeros(size(tempTrial.spk), 'logical');
                touchBeforeAnswerMask = zeros(size(tempTrial.spk), 'logical');
                touchAfterAnswerMask = zeros(size(tempTrial.spk), 'logical');

                touchMask = zeros(size(tempTrial.spk), 'logical');
                for pi = 1 : 4
                    ci = find(cellPlaneInds == pi);
                    tempTouchFrame = unique(tempTrial.protractionTouchOnsetFramesByWhisking{pi}+touchFrameBuffer);
                    touchMask(ci,tempTouchFrame) = 1;

                    poleUpStartFrame = find(tempTrial.tpmTime{pi} < tempTrial.poleMovingTime(1), 1, 'last');
                    poleUpEndFrame = find(tempTrial.tpmTime{pi} > tempTrial.poleUpTime(1), 1, 'first');
                    poleDownStartFrame = find(tempTrial.tpmTime{pi} < tempTrial.poleUpTime(end), 1, 'last');
                    poleDownEndFrame = find(tempTrial.tpmTime{pi} > tempTrial.poleMovingTime(end), 1, 'first') + max(touchFrameBuffer);
                    allLickTimes = union(union(tempTrial.leftLickTime, tempTrial.rightLickTime), tempTrial.answerLickTime);
                    firstLickTime = allLickTimes(find(allLickTimes > tempTrial.poleUpTime(1), 1, 'first'));
                    firstLickFrame = find(tempTrial.tpmTime{pi} < firstLickTime, 1, 'last');
                    answerFrame = find(tempTrial.tpmTime{pi} < tempTrial.answerLickTime, 1, 'last'); % To consider preparatory time

                    beforePoleUpMask(ci, 1 : poleUpStartFrame-1) = 1;
                    poleBeforeFirstLickMask(ci, poleUpEndFrame : firstLickFrame) = 1;
                    poleBeforeAnswerMask(ci, poleUpEndFrame : answerFrame) = 1;
                    poleAfterAnswerMask(ci, answerFrame+1 : poleDownStartFrame) = 1;
                    afterPoleDownMask(ci, poleDownEndFrame : end) = 1;
                    
                    touchBeforeFirstLickMask(ci, :) = touchMask(ci,:) .* poleBeforeFirstLickMask(ci,:);
                    touchBeforeAnswerMask(ci, :) = touchMask(ci,:) .* poleBeforeAnswerMask(ci,:);
                    touchAfterAnswerMask(ci, :) = touchMask(ci,:) .* poleAfterAnswerMask(ci,:);
                end
                
                numTouchBeforeFirstLick(ti) = length(find(cellfun(@(x) tempTrial.whiskerTime(x(1)), tempTrial.protractionTouchChunksByWhisking) < firstLickTime));
                numTouchBeforeAnswer(ti) = length(find(cellfun(@(x) tempTrial.whiskerTime(x(1)), tempTrial.protractionTouchChunksByWhisking) < tempTrial.answerLickTime));
                numTouchAfterAnswer(ti) = length(find(cellfun(@(x) tempTrial.whiskerTime(x(1)), tempTrial.protractionTouchChunksByWhisking) >= tempTrial.answerLickTime));
                numTouch(ti) = length(tempTrial.protractionTouchChunksByWhisking);
                
                beforePoleUpExpert(:,ti) = sum(tempTrial.spk .* beforePoleUpMask, 2) ./ sum(beforePoleUpMask,2)  * u.frameRate;
                poleBeforeFirstLickExpert(:,ti) = sum(tempTrial.spk .* poleBeforeFirstLickMask, 2) ./ sum(poleBeforeFirstLickMask,2)  * u.frameRate;
                poleBeforeAnswerExpert(:,ti) = sum(tempTrial.spk .* poleBeforeAnswerMask, 2) ./ sum(poleBeforeAnswerMask,2)  * u.frameRate;
                poleAfterAnswerExpert(:,ti) = sum(tempTrial.spk .* poleAfterAnswerMask, 2) ./ sum(poleAfterAnswerMask,2)  * u.frameRate;
                afterPoleDownExpert(:,ti) = sum(tempTrial.spk .* afterPoleDownMask, 2) ./ sum(afterPoleDownMask,2)  * u.frameRate;

                touchBeforeFirstLickExpert(:,ti) = sum(tempTrial.spk .* touchBeforeFirstLickMask,2) / numTouchBeforeFirstLick(ti); % some can be NaN
                touchBeforeAnswerExpert(:,ti) = sum(tempTrial.spk .* touchBeforeAnswerMask,2) / numTouchBeforeAnswer(ti); % some can be NaN
                touchAfterAnswerExpert(:,ti) = sum(tempTrial.spk .* touchAfterAnswerMask,2) / numTouchAfterAnswer(ti); % some can be NaN
                touchResponseExpert(:,ti) = sum(tempTrial.spk .* touchMask,2) / numTouch(ti);
                
                numTouchFrameBeforeFirstLick(:,ti) = sum(touchBeforeFirstLickMask,2);
                numTouchFrameBeforeAnswer(:,ti) = sum(touchBeforeAnswerMask,2);
                numTouchFrameAfterAnswer(:,ti) = sum(touchAfterAnswerMask,2);
                numTouchFrame(:,ti) = sum(touchMask,2);
            end
            % Change Inf to NaN (for pca)
            beforePoleUpExpert(find(isinf(beforePoleUpExpert))) = NaN;
            poleBeforeFirstLickExpert(find(isinf(poleBeforeFirstLickExpert))) = NaN;
            poleBeforeAnswerExpert(find(isinf(poleBeforeAnswerExpert))) = NaN;
            poleAfterAnswerExpert(find(isinf(poleAfterAnswerExpert))) = NaN;
            afterPoleDownExpert(find(isinf(afterPoleDownExpert))) = NaN;
            touchBeforeFirstLickExpert(find(isinf(touchBeforeFirstLickExpert))) = NaN;
            touchBeforeAnswerExpert(find(isinf(touchBeforeAnswerExpert))) = NaN;
            touchAfterAnswerExpert(find(isinf(touchAfterAnswerExpert))) = NaN;
            touchResponseExpert(find(isinf(touchResponseExpert))) = NaN;
            
            % Trial and session info
            trialAngleExpert = cellfun(@(x) x.angle, u.trials(trialInd));
            trialAnswerExpert = cellfun(@(x) x.choice, u.trials(trialInd), 'un', 0);
            trialOutcomeExpert = cellfun(@(x) x.response, u.trials(trialInd)); % 0 for wrong, 1 for correct
            non90Ind = find(trialAngleExpert ~= 90);
            sessionPerformanceExpert = mean(trialOutcomeExpert(non90Ind));

            % Data gathering
            % Neuron ID and type
            expert(i).allID = allIDExpert;
            expert(i).indTuned = indTunedExpert;
            expert(i).indNottuned = indNottunedExpert;
            expert(i).indNontouch = indNontouchExpert;
            expert(i).eventRate = eventRateExpert;

            % Tuning properties
            expert(i).tunedAngle = tunedAngleExpert;
            expert(i).angleSelectivity = angleSelectivityExpert;
            expert(i).selectiveTuned = selectiveTunedExpert;
            expert(i).broadTuned = broadTunedExpert;
            expert(i).complexTuned = complexTunedExpert;

            % Population response
            expert(i).beforePoleUp = beforePoleUpExpert;
            expert(i).poleBeforeFirstLick = poleBeforeFirstLickExpert;
            expert(i).poleBeforeAnswer = poleBeforeAnswerExpert;
            expert(i).poleAfterAnswer = poleAfterAnswerExpert;
            expert(i).afterPoleDown = afterPoleDownExpert;
            
            expert(i).touchBeforeFirstLick = touchBeforeFirstLickExpert;
            expert(i).touchBeforeAnswer = touchBeforeAnswerExpert;
            expert(i).touchAfterAnswer = touchAfterAnswerExpert;
            expert(i).touchResponse = touchResponseExpert;
            
            expert(i).numTouchFrameBeforeFirstLick = numTouchFrameBeforeFirstLick;
            expert(i).numTouchFrameBeforeAnswer = numTouchFrameBeforeAnswer;
            expert(i).numTouchFrameAfterAnswer = numTouchFrameAfterAnswer;
            expert(i).numTouchFrame = numTouchFrame;

            % Trial parameters
            expert(i).trialAngle = trialAngleExpert;
            expert(i).trialAnswer = trialAnswerExpert;
            expert(i).trialOutcome = trialOutcomeExpert;
            expert(i).sessionPerformance = sessionPerformanceExpert;
            expert(i).frameRate = u.frameRate;
            expert(i).numTouchBeforeFirstLick = numTouchBeforeFirstLick;
            expert(i).numTouchBeforeAnswer = numTouchBeforeAnswer;
            expert(i).numTouchAfterAnswer = numTouchAfterAnswer;
            expert(i).numTouch = numTouch;
        end
    end
end

save([baseDir, saveFn], 'naive', 'expert')





















%%
%% (1) How does it look when including different types of neurons in response to touch?
%%


clear
baseDir = 'C:\Users\shires\Dropbox\Works\Projects\2020 Neural stretching in S1\Data\';
load([baseDir, 'cellMatching_beforeNafter.mat'], 'match')
popres = load([baseDir, 'popResponse_201228.mat'], 'naive', 'expert');

numVol = length(popres.naive); % # of imaged volumes
%% (1)-1 Each response types
%% First, calculate clustering index in different subgroups 
%% First, from touch response before the answer lick

%% Not considering # of neurons
numDim = 3;
clusterIndNaive = zeros(numVol,6); % angle-tuned, not-tuned touch, non-touch, all touch, all active, all except angle-tuned neurons
clusterIndExpert = zeros(numVol,6);
pcaCoordNaive = cell(numVol,6);
pcaCoordExpert = cell(numVol,6);
veNaive = cell(numVol,6);
veExpert = cell(numVol,6);
for vi = 1 : numVol
    [~, pcaCoordNaive{vi,1},~,~,veNaive{vi,1}] = pca(popres.naive(vi).touchBeforeAnswer(popres.naive(vi).indTuned,:)');
    [~, pcaCoordNaive{vi,2},~,~,veNaive{vi,2}] = pca(popres.naive(vi).touchBeforeAnswer(popres.naive(vi).indNottuned,:)');
    [~, pcaCoordNaive{vi,3},~,~,veNaive{vi,3}] = pca(popres.naive(vi).touchBeforeAnswer(popres.naive(vi).indNontouch,:)');
    [~, pcaCoordNaive{vi,4},~,~,veNaive{vi,4}] = pca(popres.naive(vi).touchBeforeAnswer(union(popres.naive(vi).indTuned, popres.naive(vi).indNottuned),:)');
    [~, pcaCoordNaive{vi,5},~,~,veNaive{vi,5}] = pca(popres.naive(vi).touchBeforeAnswer');
    [~, pcaCoordNaive{vi,6},~,~,veNaive{vi,6}] = pca(popres.naive(vi).touchBeforeAnswer(union(popres.naive(vi).indNottuned, popres.naive(vi).indNontouch),:)');
    
    for ii = 1 : 6
        clusterIndNaive(vi,ii) = clustering_index(pcaCoordNaive{vi,ii}(:,1:numDim), popres.naive(vi).trialAngle);
    end
    
    [~, pcaCoordExpert{vi,1},~,~,veExpert{vi,1}] = pca(popres.expert(vi).touchBeforeAnswer(popres.expert(vi).indTuned,:)');
    [~, pcaCoordExpert{vi,2},~,~,veExpert{vi,2}] = pca(popres.expert(vi).touchBeforeAnswer(popres.expert(vi).indNottuned,:)');
    [~, pcaCoordExpert{vi,3},~,~,veExpert{vi,3}] = pca(popres.expert(vi).touchBeforeAnswer(popres.expert(vi).indNontouch,:)');
    [~, pcaCoordExpert{vi,4},~,~,veExpert{vi,4}] = pca(popres.expert(vi).touchBeforeAnswer(union(popres.expert(vi).indTuned, popres.expert(vi).indNottuned),:)');
    [~, pcaCoordExpert{vi,5},~,~,veExpert{vi,5}] = pca(popres.expert(vi).touchBeforeAnswer');
    [~, pcaCoordExpert{vi,6},~,~,veExpert{vi,6}] = pca(popres.expert(vi).touchBeforeAnswer(union(popres.expert(vi).indNottuned, popres.expert(vi).indNontouch),:)');
    
    for ii = 1 : 6
        clusterIndExpert(vi,ii) = clustering_index(pcaCoordExpert{vi,ii}(:,1:numDim), popres.expert(vi).trialAngle);
    end
end

%% show the result
% Clustering index is high when angle-tuned neurons are included.
% Clustering index is low when angle-tuned neurons are excluded.
% -> Response of angle-tuned neurons are necessary and sufficient to show
% object-angle related clustering.
% -> So, focus on angle-tuned neurons only.
figure, hold on
errorbar([1:6]-0.1, mean(clusterIndNaive), sem(clusterIndNaive), 'ko', 'lines', 'no')
errorbar([1:6]+0.1, mean(clusterIndExpert), sem(clusterIndExpert), 'ro', 'lines', 'no')

ylabel('Clustering index')
xlim([0.5 6.5])
xticklabels({'Angle-tuned (1)', 'Not-tuned touch (2)', 'Non-touch (3)', 'Touch (1+2)', 'All active (1+2+3)', 'All - angle-tuned (2+3)'})
xtickangle(45)
legend({'Naive', 'Expert'})

title({'Touch frames before the answer lick'; '(# of neurons NOT controlled)'})



%% Addendum 
% From neurons that are tuned to intermediate angles (60:15:105 degrees)
% From angle-tuned neurons. Not considering # of neurons
clusterIndIA = zeros(numVol,2); %(:,1) naive, (:,2) expert. (IA: Intermediate Angle)
for vi = 1 : numVol
    intermediateInd = intersect(find(popres.naive(vi).tunedAngle > 45), find(popres.naive(vi).tunedAngle < 120));
    naiveInd = popres.naive(vi).indTuned(intermediateInd);
    [~, pcaCoord] = pca(popres.naive(vi).touchBeforeAnswer(naiveInd,:)');
    clusterIndIA(vi,1) = clustering_index(pcaCoord(:,1:numDim), popres.naive(vi).trialAngle);
    
    intermediateInd = intersect(find(popres.expert(vi).tunedAngle > 45), find(popres.expert(vi).tunedAngle < 120));
    expertInd = popres.expert(vi).indTuned(intermediateInd);
    [~, pcaCoord] = pca(popres.expert(vi).touchBeforeAnswer(expertInd ,:)');
    clusterIndIA(vi,2) = clustering_index(pcaCoord(:,1:numDim), popres.expert(vi).trialAngle);
end
%%
figure, hold on
for vi = 1 : numVol
    plot(clusterIndIA(vi,:), 'ko-')
end
errorbar(mean(clusterIndIA), sem(clusterIndIA), 'ro', 'lines','no')

ylabel('Clustering index')
xlim([0.5 2.5]), xticks(1:2), 
xticklabels({'Naive', 'Expert'})
% xtickangle(45)
[~,p,m] = paired_test(clusterIndIA(:,1), clusterIndIA(:,2));
title({'From neurons prefering intermediate angles (60-105\circ)'; sprintf('p = %.5f (%s)', p, m)})


%% considering # of neurons
% It takes about an hour (pure guestimation for now)
numDim = 3;
randNum = 1001;
clusterIndNaive = zeros(numVol,6); % angle-tuned, not-tuned touch, non-touch, all touch, all active, all except angle-tuned neurons
clusterIndExpert = zeros(numVol,6);
pcaCoordNaive = cell(numVol,6);
pcaCoordExpert = cell(numVol,6);
veNaive = cell(numVol,6);
veExpert = cell(numVol,6);
tic
for vi = 1 : numVol
    fprintf('Processing volume #%d/%d\n', vi, numVol)
%     if length(popres.naive(vi).indTuned) > length(popres.expert(vi).indTuned)
%         numCell = length(popres.expert(vi).indTuned);
%         [~, pcaCoordExpert{vi,1},~,~,veExpert{vi,1}] = pca(popres.expert(vi).touchBeforeAnswer(popres.expert(vi).indTuned,:)');
%         
%         tempCoord = cell(randNum,1);
%         tempVE = cell(randNum,1);
%         tempCI = zeros(randNum,1);
%         for ri = 1 : randNum
%             tempInd = randperm(length(popres.naive(vi).indTuned), numCell);
%             [~,tempCoord{ri},~,~,tempVE{ri}] = pca(popres.naive(vi).touchBeforeAnswer(popres.naive(vi).indTuned(tempInd),:)');
%             tempCI(ri) = clustering_index(tempCoord{ri}(:,1:numDim), popres.naive(vi).trialAngle);
%         end
%         medInd = find(tempCI == median(tempCI));
%         pcaCoordNaive{vi,1} = tempCoord{medInd};
%         veNaive{vi,1} = tempVE{medInd};
%     else
%         numCell = length(popres.naive(vi).indTuned);
%         [~, pcaCoordNaive{vi,1},~,~,veNaive{vi,1}] = pca(popres.naive(vi).touchBeforeAnswer(popres.naive(vi).indTuned,:)');
%         
%         tempCoord = cell(randNum,1);
%         tempVE = cell(randNum,1);
%         tempCI = zeros(randNum,1);
%         for ri = 1 : randNum
%             tempInd = randperm(length(popres.expert(vi).indTuned), numCell);
%             [~,tempCoord{ri},~,~,tempVE{ri}] = pca(popres.expert(vi).touchBeforeAnswer(popres.expert(vi).indTuned(tempInd),:)');
%             tempCI(ri) = clustering_index(tempCoord{ri}(:,1:numDim), popres.expert(vi).trialAngle);
%         end
%         medInd = find(tempCI == median(tempCI));
%         pcaCoordExpert{vi,1} = tempCoord{medInd};
%         veExpert{vi,1} = tempVE{medInd};
%     end
    
    fprintf('1/6, ')
    [pcaCoordNaive{vi,1}, veNaive{vi,1}, pcaCoordExpert{vi,1}, veExpert{vi,1}] = pca_num_matching(popres.naive(vi).touchBeforeAnswer(popres.naive(vi).indTuned,:)', ...
        popres.expert(vi).touchBeforeAnswer(popres.expert(vi).indTuned,:)', popres.naive(vi).trialAngle, popres.expert(vi).trialAngle, randNum, numDim);
    fprintf('2/6, ')
    [pcaCoordNaive{vi,2}, veNaive{vi,2}, pcaCoordExpert{vi,2}, veExpert{vi,2}] = pca_num_matching(popres.naive(vi).touchBeforeAnswer(popres.naive(vi).indNottuned,:)', ...
        popres.expert(vi).touchBeforeAnswer(popres.expert(vi).indNottuned,:)', popres.naive(vi).trialAngle, popres.expert(vi).trialAngle, randNum, numDim);
    fprintf('3/6, ')
    [pcaCoordNaive{vi,3}, veNaive{vi,3}, pcaCoordExpert{vi,3}, veExpert{vi,3}] = pca_num_matching(popres.naive(vi).touchBeforeAnswer(popres.naive(vi).indNontouch,:)', ...
        popres.expert(vi).touchBeforeAnswer(popres.expert(vi).indNontouch,:)', popres.naive(vi).trialAngle, popres.expert(vi).trialAngle, randNum, numDim);
    fprintf('4/6, ')
    [pcaCoordNaive{vi,4}, veNaive{vi,4}, pcaCoordExpert{vi,4}, veExpert{vi,4}] = pca_num_matching(popres.naive(vi).touchBeforeAnswer(union(popres.naive(vi).indTuned, popres.naive(vi).indNottuned),:)', ...
        popres.expert(vi).touchBeforeAnswer(union(popres.expert(vi).indTuned, popres.expert(vi).indNottuned),:)', popres.naive(vi).trialAngle, popres.expert(vi).trialAngle, randNum, numDim);
    fprintf('5/6, ')
    [pcaCoordNaive{vi,5}, veNaive{vi,5}, pcaCoordExpert{vi,5}, veExpert{vi,5}] = pca_num_matching(popres.naive(vi).touchBeforeAnswer', ...
        popres.expert(vi).touchBeforeAnswer', popres.naive(vi).trialAngle, popres.expert(vi).trialAngle, randNum, numDim);
    fprintf('6/6, ')
    [pcaCoordNaive{vi,6}, veNaive{vi,6}, pcaCoordExpert{vi,6}, veExpert{vi,6}] = pca_num_matching(popres.naive(vi).touchBeforeAnswer(union(popres.naive(vi).indNottuned, popres.naive(vi).indNontouch),:)', ...
        popres.expert(vi).touchBeforeAnswer(union(popres.expert(vi).indNottuned, popres.expert(vi).indNontouch),:)', popres.naive(vi).trialAngle, popres.expert(vi).trialAngle, randNum, numDim);

    for ii = 1 : 6
        clusterIndNaive(vi,ii) = clustering_index(pcaCoordNaive{vi,ii}(:,1:numDim), popres.naive(vi).trialAngle);
        clusterIndExpert(vi,ii) = clustering_index(pcaCoordExpert{vi,ii}(:,1:numDim), popres.expert(vi).trialAngle);
    end
    fprintf('Volume #%d/%d done.\n', vi, numVol)
end
fprintf('Total time: %d s. \n', round(toc))

figure, hold on
errorbar([1:6]-0.1, mean(clusterIndNaive), sem(clusterIndNaive), 'ko', 'lines', 'no')
errorbar([1:6]+0.1, mean(clusterIndExpert), sem(clusterIndExpert), 'ro', 'lines', 'no')

ylabel('Clustering index')
xlim([0.5 6.5])
xticklabels({'Angle-tuned (1)', 'Not-tuned touch (2)', 'Non-touch (3)', 'Touch (1+2)', 'All active (1+2+3)', 'All - angle-tuned (2+3)'})
xtickangle(45)
legend({'Naive', 'Expert'})

title({'Touch frames before the answer lick'; '(# of neurons matched between Naive and Expert)'})

%%
saveFn = 'pca_num_match_response_category';
save([baseDir, saveFn], 'pcaCoordNaive', 'pcaCoordExpert', 'veNaive', 'veExpert', 'clusterIndNaive', 'clusterIndExpert')

%% Result 
% Matching # of neurons does not change the result significantly.
% So, just use the whole neurons.
% For now, just focus on the angle-tuned neurons without matching the # of
% neurons (for the sake of time).


%% Consider different types of angle tuning.
%% First, tuning type, then, angle selectivity (1/3 grouping)

%% Tuning type - specific, broad, complex
% Question: Do complex-tuned neurons help increasing the clustering index?

numDim = 3;
clusterIndNaive = zeros(numVol,7); % specific, broad, complex, -specific, -broad, -complex, all angle-tuned
clusterIndExpert = zeros(numVol,7);
pcaCoordNaive = cell(numVol,7);
pcaCoordExpert = cell(numVol,7);
veNaive = cell(numVol,7);
veExpert = cell(numVol,7);
for vi = 1 : numVol
    tempNaive = popres.naive(vi);
    [~, pcaCoordNaive{vi,1},~,~,veNaive{vi,1}] = pca(tempNaive.touchBeforeAnswer(tempNaive.indTuned(find(tempNaive.selectiveTuned)),:)');
    [~, pcaCoordNaive{vi,2},~,~,veNaive{vi,2}] = pca(tempNaive.touchBeforeAnswer(tempNaive.indTuned(find(tempNaive.broadTuned)),:)');
    [~, pcaCoordNaive{vi,3},~,~,veNaive{vi,3}] = pca(tempNaive.touchBeforeAnswer(tempNaive.indTuned(find(tempNaive.complexTuned)),:)');
    [~, pcaCoordNaive{vi,4},~,~,veNaive{vi,4}] = pca(tempNaive.touchBeforeAnswer(tempNaive.indTuned(find(tempNaive.selectiveTuned==0)),:)');
    [~, pcaCoordNaive{vi,5},~,~,veNaive{vi,5}] = pca(tempNaive.touchBeforeAnswer(tempNaive.indTuned(find(tempNaive.broadTuned==0)),:)');
    [~, pcaCoordNaive{vi,6},~,~,veNaive{vi,6}] = pca(tempNaive.touchBeforeAnswer(tempNaive.indTuned(find(tempNaive.complexTuned==0)),:)');
    [~, pcaCoordNaive{vi,7},~,~,veNaive{vi,7}] = pca(tempNaive.touchBeforeAnswer(tempNaive.indTuned,:)');
    
    for ii = 1 : 7
        clusterIndNaive(vi,ii) = clustering_index(pcaCoordNaive{vi,ii}(:,1:numDim), tempNaive.trialAngle);
    end
    
    tempExpert = popres.expert(vi);
    [~, pcaCoordExpert{vi,1},~,~,veExpert{vi,1}] = pca(tempExpert.touchBeforeAnswer(tempExpert.indTuned(find(tempExpert.selectiveTuned)),:)');
    [~, pcaCoordExpert{vi,2},~,~,veExpert{vi,2}] = pca(tempExpert.touchBeforeAnswer(tempExpert.indTuned(find(tempExpert.broadTuned)),:)');
    [~, pcaCoordExpert{vi,3},~,~,veExpert{vi,3}] = pca(tempExpert.touchBeforeAnswer(tempExpert.indTuned(find(tempExpert.complexTuned)),:)');
    [~, pcaCoordExpert{vi,4},~,~,veExpert{vi,4}] = pca(tempExpert.touchBeforeAnswer(tempExpert.indTuned(find(tempExpert.selectiveTuned==0)),:)');
    [~, pcaCoordExpert{vi,5},~,~,veExpert{vi,5}] = pca(tempExpert.touchBeforeAnswer(tempExpert.indTuned(find(tempExpert.broadTuned==0)),:)');
    [~, pcaCoordExpert{vi,6},~,~,veExpert{vi,6}] = pca(tempExpert.touchBeforeAnswer(tempExpert.indTuned(find(tempExpert.complexTuned==0)),:)');
    [~, pcaCoordExpert{vi,7},~,~,veExpert{vi,7}] = pca(tempExpert.touchBeforeAnswer(tempExpert.indTuned,:)');
    
    for ii = 1 : 7
        clusterIndExpert(vi,ii) = clustering_index(pcaCoordExpert{vi,ii}(:,1:numDim), tempExpert.trialAngle);
    end
end

%%
figure, hold on
errorbar([1:7]-0.1, mean(clusterIndNaive), sem(clusterIndNaive), 'ko', 'lines', 'no')
errorbar([1:7]+0.1, mean(clusterIndExpert), sem(clusterIndExpert), 'ro', 'lines', 'no')

ylabel('Clustering index')
xlim([0.5 7.5])
xticklabels({'Specific (1)', 'Broad (2)', 'Complex (3)', '- Specific (2+3)', '- Broad (1+3)', '- Complex (2+3)', 'All angle-tuned (1+2+3)'})
xtickangle(45)
legend({'Naive', 'Expert'})

title({'Touch frames before the answer lick'; '(# of neurons NOT controlled)'})


%% Result
% Clustering index: specific >= broad >> complex
% Increase in clustering index does not seem to depend on a single tuning type



%% Angle selectivity
% Do neurons with higher angle selectivity affect either clustering index
% or increase of it more than those with lower angle selectivity?
% Divide groups by 3 groups (low, middle, high angle selectivity)

numDim = 3;
clusterIndNaive = zeros(numVol,4); % all, high 3rd, middle 3rd, low 3rd
clusterIndExpert = zeros(numVol,4);
pcaCoordNaive = cell(numVol,4);
pcaCoordExpert = cell(numVol,4);
veNaive = cell(numVol,4);
veExpert = cell(numVol,4);
for vi = 1 : numVol
    tempNaive = popres.naive(vi);
    [~, pcaCoordNaive{vi,1},~,~,veNaive{vi,1}] = pca(tempNaive.touchBeforeAnswer(tempNaive.indTuned,:)');
    [~, pcaCoordNaive{vi,2},~,~,veNaive{vi,2}] = pca(tempNaive.touchBeforeAnswer(tempNaive.indTuned(find(tempNaive.angleSelectivity >= prctile(tempNaive.angleSelectivity, 66.7))),:)');
    [~, pcaCoordNaive{vi,3},~,~,veNaive{vi,3}] = pca(tempNaive.touchBeforeAnswer(tempNaive.indTuned( intersect(find(tempNaive.angleSelectivity >= prctile(tempNaive.angleSelectivity, 33.3)), find(tempNaive.angleSelectivity < prctile(tempNaive.angleSelectivity, 66.7))) ),:)');
    [~, pcaCoordNaive{vi,4},~,~,veNaive{vi,4}] = pca(tempNaive.touchBeforeAnswer(tempNaive.indTuned(find(tempNaive.angleSelectivity < prctile(tempNaive.angleSelectivity, 33.3))),:)');
    
    for ii = 1 : 4
        clusterIndNaive(vi,ii) = clustering_index(pcaCoordNaive{vi,ii}(:,1:numDim), tempNaive.trialAngle);
    end
    
    tempExpert = popres.expert(vi);
    [~, pcaCoordExpert{vi,1},~,~,veExpert{vi,1}] = pca(tempExpert.touchBeforeAnswer(tempExpert.indTuned,:)');
    [~, pcaCoordExpert{vi,2},~,~,veExpert{vi,2}] = pca(tempExpert.touchBeforeAnswer(tempExpert.indTuned(find(tempExpert.angleSelectivity >= prctile(tempExpert.angleSelectivity, 66.7))),:)');
    [~, pcaCoordExpert{vi,3},~,~,veExpert{vi,3}] = pca(tempExpert.touchBeforeAnswer(tempExpert.indTuned( intersect(find(tempExpert.angleSelectivity >= prctile(tempExpert.angleSelectivity, 33.3)), find(tempExpert.angleSelectivity < prctile(tempExpert.angleSelectivity, 66.7))) ),:)');
    [~, pcaCoordExpert{vi,4},~,~,veExpert{vi,4}] = pca(tempExpert.touchBeforeAnswer(tempExpert.indTuned(find(tempExpert.angleSelectivity < prctile(tempExpert.angleSelectivity, 33.3))),:)');
    
    for ii = 1 : 4
        clusterIndExpert(vi,ii) = clustering_index(pcaCoordExpert{vi,ii}(:,1:numDim), tempExpert.trialAngle);
    end
end

%%
figure, hold on
errorbar([1:4]-0.1, mean(clusterIndNaive), sem(clusterIndNaive), 'ko', 'lines', 'no')
errorbar([1:4]+0.1, mean(clusterIndExpert), sem(clusterIndExpert), 'ro', 'lines', 'no')

ylabel('Clustering index')
xlim([0.5 4.5])
xticks(1:4)
xticklabels({'All angle-tuned', 'Top 3rd', 'Middle 3rd', 'Bottom 3rd'})
xtickangle(45)
xlabel('Angle selectivity groups')
legend({'Naive', 'Expert'})

title({'Touch frames before the answer lick'; '(# of neurons NOT controlled)'})


%% Result
% Clustering index is correlated with the angle selectivity of the population, 
% but the increase in clustering index is not



%% How about other time windows?
% Before pole up, pole up before the answer lick, pole up after the answer
% lick, after pole down, touch before the answer lick, touch after the
% answer lick, all touches
% Using angle tuned neurons only

numDim = 3;
clusterIndNaive = zeros(numVol,9); % (:,1) before pole-up, (:,2) pole-up before the first lick lick, (:,3) pole-up before the answer lick, (:,4) pole-up after the answer lick, (:,5) after pole down, 
% (:,6) touch response before the first lick, (:,7) touch response before the answer lick, (:,8) touch response after the answer lick, (:,9) all touch response
clusterIndExpert = zeros(numVol,9);
pcaCoordNaive = cell(numVol,9);
pcaCoordExpert = cell(numVol,9);
veNaive = cell(numVol,9);
veExpert = cell(numVol,9);
for vi = 1 : numVol
    [~, pcaCoordNaive{vi,1},~,~,veNaive{vi,1}] = pca(popres.naive(vi).beforePoleUp(popres.naive(vi).indTuned,:)');
    [~, pcaCoordNaive{vi,2},~,~,veNaive{vi,2}] = pca(popres.naive(vi).poleBeforeFirstLick(popres.naive(vi).indTuned,:)');
    [~, pcaCoordNaive{vi,3},~,~,veNaive{vi,3}] = pca(popres.naive(vi).poleBeforeAnswer(popres.naive(vi).indTuned,:)');
    [~, pcaCoordNaive{vi,4},~,~,veNaive{vi,4}] = pca(popres.naive(vi).poleAfterAnswer(popres.naive(vi).indTuned,:)');
    [~, pcaCoordNaive{vi,5},~,~,veNaive{vi,5}] = pca(popres.naive(vi).afterPoleDown(popres.naive(vi).indTuned,:)');
    
    [~, pcaCoordNaive{vi,6},~,~,veNaive{vi,6}] = pca(popres.naive(vi).touchBeforeFirstLick(popres.naive(vi).indTuned,:)');
    [~, pcaCoordNaive{vi,7},~,~,veNaive{vi,7}] = pca(popres.naive(vi).touchBeforeAnswer(popres.naive(vi).indTuned,:)');
    [~, pcaCoordNaive{vi,8},~,~,veNaive{vi,8}] = pca(popres.naive(vi).touchAfterAnswer(popres.naive(vi).indTuned,:)');
    [~, pcaCoordNaive{vi,9},~,~,veNaive{vi,9}] = pca(popres.naive(vi).touchResponse(popres.naive(vi).indTuned,:)');
    
    for ii = 1 : 9
        clusterIndNaive(vi,ii) = clustering_index(pcaCoordNaive{vi,ii}(:,1:numDim), popres.naive(vi).trialAngle);
    end
    
    [~, pcaCoordExpert{vi,1},~,~,veExpert{vi,1}] = pca(popres.expert(vi).beforePoleUp(popres.expert(vi).indTuned,:)');
    [~, pcaCoordExpert{vi,2},~,~,veExpert{vi,2}] = pca(popres.expert(vi).poleBeforeFirstLick(popres.expert(vi).indTuned,:)');
    [~, pcaCoordExpert{vi,3},~,~,veExpert{vi,3}] = pca(popres.expert(vi).poleBeforeAnswer(popres.expert(vi).indTuned,:)');
    [~, pcaCoordExpert{vi,4},~,~,veExpert{vi,4}] = pca(popres.expert(vi).poleAfterAnswer(popres.expert(vi).indTuned,:)');
    [~, pcaCoordExpert{vi,5},~,~,veExpert{vi,5}] = pca(popres.expert(vi).afterPoleDown(popres.expert(vi).indTuned,:)');
    
    [~, pcaCoordExpert{vi,6},~,~,veExpert{vi,6}] = pca(popres.expert(vi).touchBeforeFirstLick(popres.expert(vi).indTuned,:)');
    [~, pcaCoordExpert{vi,7},~,~,veExpert{vi,7}] = pca(popres.expert(vi).touchBeforeAnswer(popres.expert(vi).indTuned,:)');
    [~, pcaCoordExpert{vi,8},~,~,veExpert{vi,8}] = pca(popres.expert(vi).touchAfterAnswer(popres.expert(vi).indTuned,:)');
    [~, pcaCoordExpert{vi,9},~,~,veExpert{vi,9}] = pca(popres.expert(vi).touchResponse(popres.expert(vi).indTuned,:)');
    
    for ii = 1 : 9
        clusterIndExpert(vi,ii) = clustering_index(pcaCoordExpert{vi,ii}(:,1:numDim), popres.expert(vi).trialAngle);
    end
end

figure, hold on
errorbar([1:9]-0.1, mean(clusterIndNaive), sem(clusterIndNaive), 'ko', 'lines', 'no')
errorbar([1:9]+0.1, mean(clusterIndExpert), sem(clusterIndExpert), 'ro', 'lines', 'no')

ylabel('Clustering index')
xlim([0.5 9.5])
xticklabels({'Before pole-up', 'Pole-up before first lick', 'Pole-up before answer', 'Pole-up after answer', 'After pole-down', 'Touch response - before first lick', 'Touch response - before answer', 'Touch response - after answer', 'Touch response - All'})
xtickangle(45)
legend({'Naive', 'Expert'})

title({'From angle-tuned neurons'; '(# of neurons NOT controlled)'})

%% Result
% No clustering from baseline (control)
% Increased clustering after the answer lick, especially from the Naïve sessions.
% Differences in the clustering still remain after pole-down.
% 
% How do they look like?


%% Now, let's look at the manifolds.
%% First, from non-touch neurons (since their clustering index seemed to increase)

%% Not considering # of neurons
numDim = 3;
clusterIndNaive = zeros(numVol,6); % angle-tuned, not-tuned touch, non-touch, all touch, all active, all except angle-tuned neurons
clusterIndExpert = zeros(numVol,6);
pcaCoordNaive = cell(numVol,6);
pcaCoordExpert = cell(numVol,6);
veNaive = cell(numVol,6);
veExpert = cell(numVol,6);
for vi = 1 : numVol
    [~, pcaCoordNaive{vi,1},~,~,veNaive{vi,1}] = pca(popres.naive(vi).touchBeforeAnswer(popres.naive(vi).indTuned,:)');
    [~, pcaCoordNaive{vi,2},~,~,veNaive{vi,2}] = pca(popres.naive(vi).touchBeforeAnswer(popres.naive(vi).indNottuned,:)');
    [~, pcaCoordNaive{vi,3},~,~,veNaive{vi,3}] = pca(popres.naive(vi).touchBeforeAnswer(popres.naive(vi).indNontouch,:)');
    [~, pcaCoordNaive{vi,4},~,~,veNaive{vi,4}] = pca(popres.naive(vi).touchBeforeAnswer(union(popres.naive(vi).indTuned, popres.naive(vi).indNottuned),:)');
    [~, pcaCoordNaive{vi,5},~,~,veNaive{vi,5}] = pca(popres.naive(vi).touchBeforeAnswer');
    [~, pcaCoordNaive{vi,6},~,~,veNaive{vi,6}] = pca(popres.naive(vi).touchBeforeAnswer(union(popres.naive(vi).indNottuned, popres.naive(vi).indNontouch),:)');
    
    for ii = 1 : 6
        clusterIndNaive(vi,ii) = clustering_index(pcaCoordNaive{vi,ii}(:,1:numDim), popres.naive(vi).trialAngle);
    end
    
    [~, pcaCoordExpert{vi,1},~,~,veExpert{vi,1}] = pca(popres.expert(vi).touchBeforeAnswer(popres.expert(vi).indTuned,:)');
    [~, pcaCoordExpert{vi,2},~,~,veExpert{vi,2}] = pca(popres.expert(vi).touchBeforeAnswer(popres.expert(vi).indNottuned,:)');
    [~, pcaCoordExpert{vi,3},~,~,veExpert{vi,3}] = pca(popres.expert(vi).touchBeforeAnswer(popres.expert(vi).indNontouch,:)');
    [~, pcaCoordExpert{vi,4},~,~,veExpert{vi,4}] = pca(popres.expert(vi).touchBeforeAnswer(union(popres.expert(vi).indTuned, popres.expert(vi).indNottuned),:)');
    [~, pcaCoordExpert{vi,5},~,~,veExpert{vi,5}] = pca(popres.expert(vi).touchBeforeAnswer');
    [~, pcaCoordExpert{vi,6},~,~,veExpert{vi,6}] = pca(popres.expert(vi).touchBeforeAnswer(union(popres.expert(vi).indNottuned, popres.expert(vi).indNontouch),:)');
    
    for ii = 1 : 6
        clusterIndExpert(vi,ii) = clustering_index(pcaCoordExpert{vi,ii}(:,1:numDim), popres.expert(vi).trialAngle);
    end
end

%%
angles = 45:15:135;
colors = turbo(7);
vi = 1;
groupi = 3;
groupNames = {'Angle-tuned', 'Not-tuned touch', 'Non-touch', 'Touch', 'All', 'All - angle-tuned'};
figure, 
subplot(121), hold on
for ai = 1 : length(angles)
    tempAngle = angles(ai);
    tempi = find(popres.naive(vi).trialAngle == tempAngle);
    scatter3(pcaCoordNaive{vi,groupi}(tempi,1), pcaCoordNaive{vi,groupi}(tempi,2), pcaCoordNaive{vi,groupi}(tempi,3), 10, colors(ai,:), 'filled')
end
axis square
title('Naive')

subplot(122), hold on
for ai = 1 : length(angles)
    tempAngle = angles(ai);
    tempi = find(popres.expert(vi).trialAngle == tempAngle);
    scatter3(pcaCoordExpert{vi,groupi}(tempi,1), pcaCoordExpert{vi,groupi}(tempi,2), pcaCoordExpert{vi,groupi}(tempi,3), 10, colors(ai,:), 'filled')
end
axis square
title('Expert')
sgtitle(groupNames{groupi})

%% Result
% Even though clustering index seems to increase after learning in non-touch population, 
% qualitatively there is no change in manifold or clustering in this population after learning

%% Then, from different time window
%% Before pole-up, touch before answer, touch after answer, after pole-down

numDim = 3;
clusterIndNaive = zeros(numVol,7); % (:,1) before pole-up, (:,2) pole-up before the answer lick, (:,3) pole-up after the answer lick, (:,4) after pole down, (:,5) touch response before the answer lick, (:,6) touch response after the answer lick, (:,7) all touch response
clusterIndExpert = zeros(numVol,7);
pcaCoordNaive = cell(numVol,7);
pcaCoordExpert = cell(numVol,7);
veNaive = cell(numVol,7);
veExpert = cell(numVol,7);
for vi = 1 : numVol
    [~, pcaCoordNaive{vi,1},~,~,veNaive{vi,1}] = pca(popres.naive(vi).beforePoleUp(popres.naive(vi).indTuned,:)');
    [~, pcaCoordNaive{vi,2},~,~,veNaive{vi,2}] = pca(popres.naive(vi).poleBeforeAnswer(popres.naive(vi).indTuned,:)');
    [~, pcaCoordNaive{vi,3},~,~,veNaive{vi,3}] = pca(popres.naive(vi).poleAfterAnswer(popres.naive(vi).indTuned,:)');
    [~, pcaCoordNaive{vi,4},~,~,veNaive{vi,4}] = pca(popres.naive(vi).afterPoleDown(popres.naive(vi).indTuned,:)');
    [~, pcaCoordNaive{vi,5},~,~,veNaive{vi,5}] = pca(popres.naive(vi).touchBeforeAnswer(popres.naive(vi).indTuned,:)');
    [~, pcaCoordNaive{vi,6},~,~,veNaive{vi,6}] = pca(popres.naive(vi).touchAfterAnswer(popres.naive(vi).indTuned,:)');
    [~, pcaCoordNaive{vi,7},~,~,veNaive{vi,7}] = pca(popres.naive(vi).touchResponse(popres.naive(vi).indTuned,:)');
    
    for ii = 1 : 7
        clusterIndNaive(vi,ii) = clustering_index(pcaCoordNaive{vi,ii}(:,1:numDim), popres.naive(vi).trialAngle);
    end
    
    [~, pcaCoordExpert{vi,1},~,~,veExpert{vi,1}] = pca(popres.expert(vi).beforePoleUp(popres.expert(vi).indTuned,:)');
    [~, pcaCoordExpert{vi,2},~,~,veExpert{vi,2}] = pca(popres.expert(vi).poleBeforeAnswer(popres.expert(vi).indTuned,:)');
    [~, pcaCoordExpert{vi,3},~,~,veExpert{vi,3}] = pca(popres.expert(vi).poleAfterAnswer(popres.expert(vi).indTuned,:)');
    [~, pcaCoordExpert{vi,4},~,~,veExpert{vi,4}] = pca(popres.expert(vi).afterPoleDown(popres.expert(vi).indTuned,:)');
    [~, pcaCoordExpert{vi,5},~,~,veExpert{vi,5}] = pca(popres.expert(vi).touchBeforeAnswer(popres.expert(vi).indTuned,:)');
    [~, pcaCoordExpert{vi,6},~,~,veExpert{vi,6}] = pca(popres.expert(vi).touchAfterAnswer(popres.expert(vi).indTuned,:)');
    [~, pcaCoordExpert{vi,7},~,~,veExpert{vi,7}] = pca(popres.expert(vi).touchResponse(popres.expert(vi).indTuned,:)');
    
    for ii = 1 : 7
        clusterIndExpert(vi,ii) = clustering_index(pcaCoordExpert{vi,ii}(:,1:numDim), popres.expert(vi).trialAngle);
    end
end
%%
angles = 45:15:135;
colors = turbo(7);
vi = 10;
for groupi = [1,2,3,4]
    groupNames = {'Before pole-up', 'Pole-up before answer', 'Pole-up after answer', 'After pole-down', 'Touch response - before answer', 'Touch response - after answer', 'Touch response - All'};
    figure, 
    subplot(121), hold on
    for ai = 1 : length(angles)
        tempAngle = angles(ai);
        tempi = find(popres.naive(vi).trialAngle == tempAngle);
        scatter3(pcaCoordNaive{vi,groupi}(tempi,1), pcaCoordNaive{vi,groupi}(tempi,2), pcaCoordNaive{vi,groupi}(tempi,3), 10, colors(ai,:), 'filled')
    end
    xlabel('PC1'), ylabel('PC2'), zlabel('PC3')
    axis square
    title('Naive')

    subplot(122), hold on
    for ai = 1 : length(angles)
        tempAngle = angles(ai);
        tempi = find(popres.expert(vi).trialAngle == tempAngle);
        scatter3(pcaCoordExpert{vi,groupi}(tempi,1), pcaCoordExpert{vi,groupi}(tempi,2), pcaCoordExpert{vi,groupi}(tempi,3), 10, colors(ai,:), 'filled')
    end
    xlabel('PC1'), ylabel('PC2'), zlabel('PC3')
    axis square
    title('Expert')
    sgtitle(groupNames{groupi})
end

%% Result
% There IS a structure even before pole up, though there is no clustering.
% The structure seems to be maintained after pole down.
% But this needs to be confirmed by following up in the same space.





















%%
%% (2) Matching neuronal identity, and along temporal structure of the task
%%

%% First, try with all population.
%% Then, consider angle-tuned neurons only (in either session or both)

clear
baseDir = 'C:\Users\shires\Dropbox\Works\Projects\2020 Neural stretching in S1\Data\';
load([baseDir, 'cellMatching_beforeNafter.mat'], 'match')
popres = load([baseDir, 'popResponse_201228.mat'], 'naive', 'expert');

numVol = length(popres.naive); % # of imaged volumes
%% All population
% match neuronal identity, and save this data

saveFn = 'matchedPopResponse_201230';
% There was an error about tuned index matching
% Important change!!
% Fixing error of tuned index matching
% Added checking all neuron order and tuned neuron order in the next
% section
% 2021/03/03 JK
%%
%%
%%
for vi = 1 : numVol
    % Re-ID'ing
    % Based on naive ID, then add silent neurons from expert ID (matching the total # of active neurons)
    % Reassign expert ID's based on this new naive ID's.
    tempNaive = popres.naive(vi);
    tempExpert = popres.expert(vi);
    if vi < 3
        matchi = 1;
    elseif vi == 3
        matchi = 2;
    else
        matchi = floor(vi/2) + 1;
    end
    
    planes = unique(floor(tempNaive.allID/1000)); % either 1:4 or 5:8, same between naive and expert
    newNaiveIDcell = cell(length(planes),1);
    newExpertIndCell = cell(length(planes),1); % index matching to new naive ID's for new expert ID's (e.g., if expertID(1) is the same neuron to naiveID(5), then expertInd(1) = 5)
    for pi = 1 : length(planes)
        tempNaiveID = match{matchi,1}(match{matchi,1} > planes(pi)*1000 & match{matchi,1} < (planes(pi)+1)*1000);
        tempMaxID = max(tempNaiveID);
        tempNaiveInd = find(ismember(match{matchi,1}, tempNaiveID));
        tempPersExpertID = setdiff(match{matchi,2}(tempNaiveInd), 0);
        tempExpertID = match{matchi,3}(match{matchi,3} > planes(pi)*1000 & match{matchi,3} < (planes(pi)+1)*1000);
        naiveSilentExpertID = setdiff(tempExpertID, tempPersExpertID);
        naiveSilentID = tempMaxID + 1 : tempMaxID + length(naiveSilentExpertID);
        newNaiveIDcell{pi} = [tempNaiveID'; naiveSilentID'];
        
        expertInd = zeros(length(newNaiveIDcell{pi}),1); % index matching to new naive ID's for new expert ID's (e.g., if expertID(1) is the same neuron to naiveID(5), then expertInd(1) = 5)
%         expertSilentInd = find(match{matchi,2}(tempNaiveInd)==0);
        naiveSilentExpertInd = find(ismember(tempExpertID, naiveSilentExpertID));
        
        [indNaive, locExpert] = ismember(match{matchi,2}(tempNaiveInd), tempExpertID);
        expertInd(find(indNaive)) = locExpert(find(indNaive)); % persistent neurons
        expertInd(find(indNaive==0)) = length(tempExpertID)+1 : length(expertInd); % silent at the expert session
        expertInd(length(tempNaiveID)+1 : length(newNaiveIDcell{pi})) = naiveSilentExpertInd; % silent at the naive session
        if length(expertInd) ~= length(unique(expertInd))
            error('Expert indices have redunduncy.')
        elseif ~isempty(find(expertInd==0))
            error('Expert indices have 0.')
        end
        if pi == 1
            newExpertIndCell{pi} = expertInd;
        else
            newExpertIndCell{pi} = max(newExpertIndCell{pi-1}) + expertInd;
        end
    end
    newNaiveID = cell2mat(newNaiveIDcell);
    newExpertInd = cell2mat(newExpertIndCell);
    
    
    % Assigning ID's and indices for new structs
    newNaive = struct;
    newExpert = struct;
    newNaive.allID = newNaiveID;
    newNaive.allInd = 1:length(newNaiveID);
    newExpert.allID = newNaiveID;
    newExpert.allInd = newExpertInd;
    newExpert.matchedID = newNaiveID(newExpertInd); % everything goes with matched ID for expert sessions. These IDs are re-sorted to match with newNaive
    % matchedIDs are "expert session" ID's with the matched indices to those of new naive session. 
    % E.g., newNaive.allID(17) is the same neuron as newExpert.matchedID(17), even though the number does not match.
    
    newNaive.indTuned = find(ismember(newNaive.allID, tempNaive.allID(tempNaive.indTuned)));
    newNaive.indNottuned = find(ismember(newNaive.allID, tempNaive.allID(tempNaive.indNottuned)));
    newNaive.indNontouch = find(ismember(newNaive.allID, tempNaive.allID(tempNaive.indNontouch)));
    newNaive.indSilent = setdiff(1:length(newNaive.allID), union(union(newNaive.indTuned, newNaive.indNottuned), newNaive.indNontouch)); % Sorted
    % Check index of silent neurons
    if any(newNaive.allID(newNaive.indSilent) - setdiff(newNaive.allID, tempNaive.allID))
        error('Naive silent neuron indexing error')
    end
     
    newExpert.indTuned = find(ismember(newExpert.matchedID, tempExpert.allID(tempExpert.indTuned)));
    newExpert.indNottuned = find(ismember(newExpert.matchedID, tempExpert.allID(tempExpert.indNottuned)));
    newExpert.indNontouch = find(ismember(newExpert.matchedID, tempExpert.allID(tempExpert.indNontouch)));
    newExpert.indSilent = setdiff(1:length(newExpert.matchedID), union(union(newExpert.indTuned, newExpert.indNottuned), newExpert.indNontouch)); % Sorted
    
    % Check index of silent neurons
    if ~isempty(setdiff(newExpert.matchedID(newExpert.indSilent), setdiff(newExpert.allID, tempExpert.allID))) % setdiff instead of any because matchedID is NOT sorted (while indSilent is sorted)
        error('Expert silent neuron indexing error')
    end
    
    % Copying some of non-changing data (indices related to angle-tuning)
    % BUT! Matching the order! 2021/03/03 JK
    % Only affects expert
    [~, indTunedTemp] = ismember(tempExpert.allID(tempExpert.indTuned), newExpert.matchedID);
    [~, ind] = sort(indTunedTemp(find(indTunedTemp)));
    statFn = {'tunedAngle', 'angleSelectivity', 'selectiveTuned', 'broadTuned', 'complexTuned'};
    for fieldi = 1 : length(statFn)
        newNaive.(statFn{fieldi}) = tempNaive.(statFn{fieldi});
        newExpert.(statFn{fieldi}) = tempExpert.(statFn{fieldi})(ind);
    end
    
    % Re-assigning data matrix
    dataName = {'eventRate', 'beforePoleUp', 'poleBeforeAnswer', 'poleBeforeFirstLick', 'poleAfterAnswer', 'afterPoleDown', 'touchBeforeFirstLick', 'touchBeforeAnswer', 'touchAfterAnswer', 'touchResponse', ...
         'numTouchFrameBeforeFirstLick', 'numTouchFrameBeforeAnswer', 'numTouchFrameAfterAnswer', 'numTouchFrame'};
    [~,newLocNaive] = ismember(tempNaive.allID, newNaive.allID); % newLocNaive: index in newNaive
    [~,newLocExpert] = ismember(tempExpert.allID, newExpert.matchedID); % newLocExpert: index in newExpert
    for fieldi = 1 : length(dataName)
        % Naive
        tempData = tempNaive.(dataName{fieldi});
        newData = zeros(length(newNaive.allID), size(tempData,2), 'like', tempData);
        % Data are padded with zero, now fill up the ones from active neurons
        newData(newLocNaive,:) = tempData;
        newNaive.(dataName{fieldi}) = newData;
        
        % Expert
        tempData = tempExpert.(dataName{fieldi});
        newData = zeros(length(newExpert.allID), size(tempData,2), 'like', tempData);
        % Data are padded with zero, now fill up the ones from active neurons
        newData(newLocExpert,:) = tempData;
        newExpert.(dataName{fieldi}) = newData;
    end
    
    % Filling 0's with previous chunk
    % Fill with the same plane cells
    dataName = {'numTouchFrameBeforeFirstLick', 'numTouchFrameBeforeAnswer', 'numTouchFrameAfterAnswer', 'numTouchFrame'};
    for fieldi = 1 : length(dataName)
        % Naive
        tempData = newNaive.(dataName{fieldi});
        planes = floor(newNaive.allID/1000);
        uniquePlanes = unique(planes);
        stTind = find(max(tempData) > 3, 1, 'first'); % Standard trial ind, which has max > 3 frames for the first time. Selected 3 just to be sure
        stData = tempData(:,stTind); % Standard data, a column vector
        for pi = 1 : length(uniquePlanes)
            planeIndAll = find(planes == uniquePlanes(pi));
            numTouchList = unique(stData(planeIndAll));
            if length(numTouchList) ~= 2
                error('# touch frame list should have exactly 2 members.')
            elseif ~ismember(0, numTouchList)
                error('There must be a 0 touch frame.')
            end
            zeroInd = find(stData(planeIndAll)==0);
            maxInd = find(stData(planeIndAll)==numTouchList(find(numTouchList)),1,'first'); % can be any.
            tempData(planeIndAll(zeroInd),:) = repmat((tempData(planeIndAll(maxInd),:)), [length(zeroInd),1]);
        end
        newNaive.(dataName{fieldi}) = tempData;
        
        % Expert
        tempData = newExpert.(dataName{fieldi});
        planes = floor(newExpert.allID/1000);
        uniquePlanes = unique(planes);
        stTind = find(max(tempData) > 3, 1, 'first'); % Standard trial ind, which has max > 3 frames for the first time. Selected 3 just to be sure
        stData = tempData(:,stTind); % Standard data, a column vector
        for pi = 1 : length(uniquePlanes)
            planeIndAll = find(planes == uniquePlanes(pi));
            numTouchList = unique(stData(planeIndAll));
            if length(numTouchList) ~= 2
                error('# touch frame list should have exactly 2 members.')
            elseif ~ismember(0, numTouchList)
                error('There must be a 0 touch frame.')
            end
            zeroInd = find(stData(planeIndAll)==0);
            maxInd = find(stData(planeIndAll)==numTouchList(find(numTouchList)),1,'first'); % can be any.
            tempData(planeIndAll(zeroInd),:) = repmat((tempData(planeIndAll(maxInd),:)), [length(zeroInd),1]);
        end
        newExpert.(dataName{fieldi}) = tempData;
    end
    
    % Copy some more static data (trial information)
    statFn = {'trialAngle', 'trialAnswer', 'trialOutcome', 'sessionPerformance', 'frameRate', 'numTouchBeforeFirstLick', 'numTouchBeforeAnswer', 'numTouchAfterAnswer', 'numTouch'};
    for fieldi = 1 : length(statFn)
        newNaive.(statFn{fieldi}) = tempNaive.(statFn{fieldi});
        newExpert.(statFn{fieldi}) = tempExpert.(statFn{fieldi});
    end
    naive(vi) = newNaive;
    expert(vi) = newExpert;
end

save([baseDir, saveFn], 'naive', 'expert')





%% Confirmation for newNaiveID and newExpertInd
% using match
matchi = 6;
for eci = 1 : length(newNaiveID)
    nID = newNaiveID(eci); 
    eID = newNaiveID(newExpertInd(eci));

    if floor(eID/1000) ~= floor(nID/1000)
        error('Different plane')
    else
        if ismember(nID, match{matchi,1}) % active at naive
            eMatchID = match{matchi,2}(find(match{matchi,1} == nID));
            if eMatchID % active at expert, so, persistent neuron
                if eID == eMatchID
%                     disp('Correct')
                else
                    error('Expert ID does not match')
                end
            else % silent at expert
                if ismember(eID, match{matchi,3})
                    error('Expert activeness is different')
                else
%                     disp('No error (silent at Expert)')
                end
            end
        else % silent at naive
            if ismember(eID, match{matchi,2}) % having a match with Naive active
                error('Naive activeness is different')
            elseif ismember(eID, match{matchi,3})
%                 disp('No error (silent at Naive)')
            else
                error('Silent at Naive cannot be silent at Expert')
            end
        end
    end
end

disp('Correct')


%% Confirmation for matched population response
% Check IDs (tuned, nottuned, nontouch)
% Check activity, event rate sort (all-neuron order), angle selectivity
% sort (angle-tuned neuron order)

clear
baseDir = 'C:\Users\shires\Dropbox\Works\Projects\2020 Neural stretching in S1\Data\';
matchedPR = load([baseDir, 'matchedPopResponse_201230'], 'naive', 'expert');
popres = load([baseDir, 'popResponse_201228'], 'naive', 'expert');
numVol = 11;
for vi = 1 : numVol
    % Naive
    % Checking ID's
    if ~isempty(setdiff(popres.naive(vi).allID(popres.naive(vi).indTuned), matchedPR.naive(vi).allID(matchedPR.naive(vi).indTuned)))
        error('Naive indTuned ID match error at volume #%d', vi)
    end
    if ~isempty(setdiff(popres.naive(vi).allID(popres.naive(vi).indNottuned), matchedPR.naive(vi).allID(matchedPR.naive(vi).indNottuned)))
        error('Naive indNottuned ID match error at volume #%d', vi)
    end
    if ~isempty(setdiff(popres.naive(vi).allID(popres.naive(vi).indNontouch), matchedPR.naive(vi).allID(matchedPR.naive(vi).indNontouch)))
        error('Naive indNontouch ID match error at volume #%d', vi)
    end
    
    % Checking activities
    matchingInd = find(ismember(matchedPR.naive(vi).allID, popres.naive(vi).allID));
    if any(popres.naive(vi).beforePoleUp - matchedPR.naive(vi).beforePoleUp(matchingInd,:))
        error('Naive before pole up activity match error at volume #%d', vi)
    end
    if any(popres.naive(vi).touchBeforeAnswer - matchedPR.naive(vi).touchBeforeAnswer(matchingInd,:))
        error('Naive touch response before the answer lick match error at volume #%d', vi)
    end 
    if any(popres.naive(vi).touchAfterAnswer - matchedPR.naive(vi).touchAfterAnswer(matchingInd,:))
        error('Naive touch response after the answer lick  match error at volume #%d', vi)
    end
    if any(popres.naive(vi).afterPoleDown - matchedPR.naive(vi).afterPoleDown(matchingInd,:))
        error('Naive after pole down activity match error at volume #%d', vi)
    end
    
    % Checking all order
    % Use event rate to compare
    for testi = 1 : length(popres.naive(vi).allID)
        naiveID = popres.naive(vi).allID(testi);
        matchNaiveInd = find(matchedPR.naive(vi).allID == naiveID);
        matchER = matchedPR.naive(vi).eventRate(matchNaiveInd);
        naiveER = popres.naive(vi).eventRate(testi);
        if matchER - naiveER ~= 0
            error('Naive all order error at volume #%d', vi)
        end
    end
    
    % Checking tuned order
    % Use angle selectivity to compare
    for testi = 1 : length(popres.naive(vi).indTuned)
        naiveID = popres.naive(vi).allID(popres.naive(vi).indTuned(testi));
        matchNaiveInd = find(matchedPR.naive(vi).allID == naiveID);
        matchNaiveTunedInd = find(matchedPR.naive(vi).indTuned == matchNaiveInd);
        matchAS = matchedPR.naive(vi).angleSelectivity(matchNaiveTunedInd);
        naiveAS = popres.naive(vi).angleSelectivity(testi);
        if matchAS - naiveAS ~= 0
            error('Naive tuned order error at volume #%d', vi)
        end
    end
    
    % Expert
    % Checking ID's
    if ~isempty(setdiff(popres.expert(vi).allID(popres.expert(vi).indTuned), matchedPR.expert(vi).matchedID(matchedPR.expert(vi).indTuned)))
        error('Expert indTuned ID match error at volume #%d', vi)
    end
    if ~isempty(setdiff(popres.expert(vi).allID(popres.expert(vi).indNottuned), matchedPR.expert(vi).matchedID(matchedPR.expert(vi).indNottuned)))
        error('Expert indNottuned ID match error at volume #%d', vi)
    end
    if ~isempty(setdiff(popres.expert(vi).allID(popres.expert(vi).indNontouch), matchedPR.expert(vi).matchedID(matchedPR.expert(vi).indNontouch)))
        error('Expert indNontouch ID match error at volume #%d', vi)
    end
    
    % Checking activities
    [matchingInd, matchedLoc] = ismember(matchedPR.expert(vi).matchedID, popres.expert(vi).allID);
    matchingInd = find(matchingInd);
    matchedLoc = matchedLoc(matchingInd);
    if any(popres.expert(vi).beforePoleUp(matchedLoc,:) - matchedPR.expert(vi).beforePoleUp(matchingInd,:))
        error('Expert before pole up activity match error at volume #%d', vi)
    end
    if any(popres.expert(vi).touchBeforeAnswer(matchedLoc,:) - matchedPR.expert(vi).touchBeforeAnswer(matchingInd,:))
        error('Expert touch response before the answer lick match error at volume #%d', vi)
    end 
    if any(popres.expert(vi).touchAfterAnswer(matchedLoc,:) - matchedPR.expert(vi).touchAfterAnswer(matchingInd,:))
        error('Expert touch response after the answer lick  match error at volume #%d', vi)
    end
    if any(popres.expert(vi).afterPoleDown(matchedLoc,:) - matchedPR.expert(vi).afterPoleDown(matchingInd,:))
        error('Expert after pole down activity match error at volume #%d', vi)
    end
    
    % Checking all order
    % Use event rate to compare
    for testi = 1 : length(popres.expert(vi).allID)
        expertID = popres.expert(vi).allID(testi);
        matchExpertInd = find(matchedPR.expert(vi).matchedID == expertID);
        matchER = matchedPR.expert(vi).eventRate(matchExpertInd);
        expertER = popres.expert(vi).eventRate(testi);
        if matchER - expertER ~= 0
            error('Expert all order error at volume #%d', vi)
        end
    end
    
    % Checking tuned order
    % Use angle selectivity to compare
    for testi = 1 : length(popres.expert(vi).indTuned)
        expertID = popres.expert(vi).allID(popres.expert(vi).indTuned(testi));
        matchExpertInd = find(matchedPR.expert(vi).matchedID == expertID);
        matchExpertTunedInd = find(matchedPR.expert(vi).indTuned == matchExpertInd);
        matchAS = matchedPR.expert(vi).angleSelectivity(matchExpertTunedInd);
        expertAS = popres.expert(vi).angleSelectivity(testi);
        if matchAS - expertAS ~= 0
            error('Expert tuned order error at volume #%d', vi)
        end
    end
    
end

disp('No error')




%%
vi = 1;
testi = 105;
expertID = popres.expert(vi).allID(testi);
matchExpertInd = find(matchedPR.expert(vi).matchedID == expertID);
matchER = matchedPR.expert(vi).eventRate(matchExpertInd);
expertER = popres.expert(vi).eventRate(testi);
matchER - expertER

%%
vi = 1 ;
testi = 15;
expertID = popres.expert(vi).allID(popres.expert(vi).indTuned(testi));
matchExpertInd = find(matchedPR.expert(vi).matchedID == expertID);
matchExpertTunedInd = find(matchedPR.expert(vi).indTuned == matchExpertInd);
matchAS = matchedPR.expert(vi).angleSelectivity(matchExpertTunedInd);
expertAS = popres.expert(vi).angleSelectivity(testi);
matchAS - expertAS


%%
vi = 1;
testi = 105;
naiveID = popres.naive(vi).allID(testi);
matchNaiveInd = find(matchedPR.naive(vi).allID == naiveID);
matchER = matchedPR.naive(vi).eventRate(matchNaiveInd);
naiveER = popres.naive(vi).eventRate(testi);
matchER - naiveER


%%
vi = 1 ;
testi = 30;
naiveID = popres.naive(vi).allID(popres.naive(vi).indTuned(testi));
matchNaiveInd = find(matchedPR.naive(vi).allID == naiveID);
matchNaiveTunedInd = find(matchedPR.naive(vi).indTuned == matchNaiveInd);
matchAS = matchedPR.naive(vi).angleSelectivity(matchNaiveTunedInd);
naiveAS = popres.naive(vi).angleSelectivity(testi);
matchAS - naiveAS








%%
vi = 1 ;
testi = 100;
expertID = popres.expert(vi).allID(popres.expert(vi).indTuned(testi));
matchExpertInd = find(expert(vi).matchedID == expertID);
matchExpertTunedInd = find(expert(vi).indTuned == matchExpertInd);
matchAS = expert(vi).angleSelectivity(matchExpertTunedInd);
expertAS = popres.expert(vi).angleSelectivity(testi);
matchAS - expertAS







%%
%% Matched PCA 
%%
% First, using all neurons
% Combine 'beforePoleUp', 'touchBeforeAnswer', 'touchAfterAnswer', 'afterPoleDown'
% Combine Naive and Expert
clear
baseDir = 'C:\Users\shires\Dropbox\Works\Projects\2020 Neural stretching in S1\Data\';
% baseDir = 'C:\Users\jinho\Dropbox\Works\Projects\2020 Neural stretching in S1\Data\';
matchedPR = load([baseDir, 'matchedPopResponse_201230'], 'naive', 'expert');
numVol = 11;
numDim = 3;
angles = 45:15:135;
colors = turbo(length(angles));
pcaCoord = cell(numVol,1);
varExp = cell(numVol,1);


%% First, see if different subgroups defined by activeness affects clustering index
% Not by combining both sessions (treat naive and expert sessions
% individually)
% During touch response before the answer lick
% Divide by all active, persistent, and transient neurons
ciNaive = zeros(numVol,3); % (:,1) all, (:,2) persistently active, (:,3) transiently active
ciExpert = zeros(numVol,3); % (:,1) all, (:,2) persistently active, (:,3) transiently active
for vi = 1 : numVol
    tempNaive = matchedPR.naive(vi);
    tempExpert = matchedPR.expert(vi);
    indNaive = setdiff(tempNaive.allInd, tempNaive.indSilent);
    indExpert = setdiff(tempExpert.allInd, tempExpert.indSilent);
    indPers = intersect(indNaive, indExpert);
    % Naive
    % All
    [~, pcaScore] = pca( tempNaive.touchBeforeAnswer(indNaive, :)' );
    ciNaive(vi,1) = clustering_index(pcaScore(:,1:3), tempNaive.trialAngle);
    % Persistently Active
    [~, pcaScore] = pca( tempNaive.touchBeforeAnswer(indPers, :)' );
    ciNaive(vi,2) = clustering_index(pcaScore(:,1:3), tempNaive.trialAngle);
    % Transiently Active
    [~, pcaScore] = pca( tempNaive.touchBeforeAnswer(tempExpert.indSilent, :)' );
    ciNaive(vi,3) = clustering_index(pcaScore(:,1:3), tempNaive.trialAngle);
    
    % Expert
    % All
    [~, pcaScore] = pca( tempExpert.touchBeforeAnswer(indExpert, :)' );
    ciExpert(vi,1) = clustering_index(pcaScore(:,1:3), tempExpert.trialAngle);
    % Persistently Active
    [~, pcaScore] = pca( tempExpert.touchBeforeAnswer(indPers, :)' );
    ciExpert(vi,2) = clustering_index(pcaScore(:,1:3), tempExpert.trialAngle);
    % Transiently Active
    [~, pcaScore] = pca( tempExpert.touchBeforeAnswer(tempNaive.indSilent, :)' );
    ciExpert(vi,3) = clustering_index(pcaScore(:,1:3), tempExpert.trialAngle);
end

%
figure, hold on
errorbar(mean(ciNaive), sem(ciNaive), 'ko', 'lines', 'no')
errorbar(mean(ciExpert), sem(ciExpert), 'ro', 'lines', 'no')
legend({'Naive', 'Expert'})
xlim([0.5 3.5])
xticks(1:3)
xticklabels({'All', 'Persistent', 'Transient'})
ylabel('Clustering index')
ylim([0 0.3])
title('Touch response before the answer lick')






%% Repeat on average response before the answer lick
ciNaive = zeros(numVol,3); % (:,1) all, (:,2) persistently active, (:,3) transiently active
ciExpert = zeros(numVol,3); % (:,1) all, (:,2) persistently active, (:,3) transiently active
for vi = 1 : numVol
    tempNaive = matchedPR.naive(vi);
    tempExpert = matchedPR.expert(vi);
    indNaive = setdiff(tempNaive.allInd, tempNaive.indSilent);
    indExpert = setdiff(tempExpert.allInd, tempExpert.indSilent);
    indPers = intersect(indNaive, indExpert);
    % Naive
    % All
    [~, pcaScore] = pca( tempNaive.poleBeforeAnswer(indNaive, :)' );
    ciNaive(vi,1) = clustering_index(pcaScore(:,1:3), tempNaive.trialAngle);
    % Persistently Active
    [~, pcaScore] = pca( tempNaive.poleBeforeAnswer(indPers, :)' );
    ciNaive(vi,2) = clustering_index(pcaScore(:,1:3), tempNaive.trialAngle);
    % Transiently Active
    [~, pcaScore] = pca( tempNaive.poleBeforeAnswer(tempExpert.indSilent, :)' );
    ciNaive(vi,3) = clustering_index(pcaScore(:,1:3), tempNaive.trialAngle);
    
    % Expert
    % All
    [~, pcaScore] = pca( tempExpert.poleBeforeAnswer(indExpert, :)' );
    ciExpert(vi,1) = clustering_index(pcaScore(:,1:3), tempExpert.trialAngle);
    % Persistently Active
    [~, pcaScore] = pca( tempExpert.poleBeforeAnswer(indPers, :)' );
    ciExpert(vi,2) = clustering_index(pcaScore(:,1:3), tempExpert.trialAngle);
    % Transiently Active
    [~, pcaScore] = pca( tempExpert.poleBeforeAnswer(tempNaive.indSilent, :)' );
    ciExpert(vi,3) = clustering_index(pcaScore(:,1:3), tempExpert.trialAngle);
end

%
figure, hold on
errorbar(mean(ciNaive), sem(ciNaive), 'ko', 'lines', 'no')
errorbar(mean(ciExpert), sem(ciExpert), 'ro', 'lines', 'no')
legend({'Naive', 'Expert'})
xlim([0.5 3.5])
xticks(1:3)
xticklabels({'All', 'Persistent', 'Transient'})
ylabel('Clustering index')
ylim([0 0.3])
title('Average activity before the answer lick, after the pole up')




%%
%% 3D plots from all active neurons
%%
% for vi = 1 : numVol
for vi = 1
%     allPopAct = [matchedPR.naive(vi).beforePoleUp, matchedPR.naive(vi).touchBeforeAnswer, matchedPR.naive(vi).touchAfterAnswer, matchedPR.naive(vi).afterPoleDown, ...
%         matchedPR.expert(vi).beforePoleUp, matchedPR.expert(vi).touchBeforeAnswer, matchedPR.expert(vi).touchAfterAnswer, matchedPR.expert(vi).afterPoleDown];
    allPopAct = [matchedPR.naive(vi).beforePoleUp, matchedPR.naive(vi).poleBeforeAnswer, matchedPR.naive(vi).poleAfterAnswer, matchedPR.naive(vi).afterPoleDown, ...
        matchedPR.expert(vi).beforePoleUp, matchedPR.expert(vi).poleBeforeAnswer, matchedPR.expert(vi).poleAfterAnswer, matchedPR.expert(vi).afterPoleDown];
    [~, pcaCoord{vi},~,~,varExp{vi}] = pca(allPopAct');
    
    angleNaive = matchedPR.naive(vi).trialAngle;
    angleExpert = matchedPR.expert(vi).trialAngle;
    numNaiveTrial = length(angleNaive);
    numExpertTrial = length(angleExpert);
    
    %% plotting all time point
%     figure, 
%     % Plot Naive
%     subplot(121), hold on
%     for ai = 1 : length(angles)
% %     for ai = 1
%         angleInd = find(angleNaive == angles(ai));
%         for aii = 1 : length(angleInd)
% %         for aii = 1
%             temporalInd = [angleInd(aii), angleInd(aii) + numNaiveTrial, angleInd(aii) + numNaiveTrial*2, angleInd(aii) + numNaiveTrial*3];
% %             plot3(pcaCoord{vi}(temporalInd,1), pcaCoord{vi}(temporalInd,2), pcaCoord{vi}(temporalInd,3), '.-', 'color', colors(ai,:))
%             for tempori = 1 : length(temporalInd)
%                 scatter3(pcaCoord{vi}(temporalInd(tempori),1), pcaCoord{vi}(temporalInd(tempori),2), pcaCoord{vi}(temporalInd(tempori),3), 20, colors(ai,:), 'o', 'filled', 'markerfacealpha', 0.25*tempori)
%             end
%         end
%     end
%     title('Naive')
%     
%     % Plot Expert
%     subplot(122), hold on
%     for ai = 1 : length(angles)
% %     for ai = 1
%         angleInd = find(angleExpert == angles(ai));
%         for aii = 1 : length(angleInd)
% %         for aii = 1
%             temporalInd = [angleInd(aii), angleInd(aii) + numExpertTrial, angleInd(aii) + numExpertTrial*2, angleInd(aii) + numExpertTrial*3]+numNaiveTrial*4;
% %             plot3(pcaCoord{vi}(temporalInd,1), pcaCoord{vi}(temporalInd,2), pcaCoord{vi}(temporalInd,3), '.-', 'color', colors(ai,:))
%             for tempori = 1 : length(temporalInd)
%                 scatter3(pcaCoord{vi}(temporalInd(tempori),1), pcaCoord{vi}(temporalInd(tempori),2), pcaCoord{vi}(temporalInd(tempori),3), 20, colors(ai,:), 'o', 'filled', 'markerfacealpha', 0.25*tempori)
%             end
%         end
%     end
%     title('Expert')
%     
%     sgtitle(sprintf('Volume #%d', vi))
%     
%     
%     %% Plotting touch response before the answer lick
%     %% Naive (open) and expert (filled) overlaid
%     figure, hold on
%     for ai = 1 : length(angles)
%         angleIndNaive = find(angleNaive == angles(ai));
%         scatter3(pcaCoord{vi}(numNaiveTrial+angleIndNaive,1), pcaCoord{vi}(numNaiveTrial+angleIndNaive,2), pcaCoord{vi}(numNaiveTrial+angleIndNaive,3), 20, colors(ai,:), 'o')
%         
%         angleIndExpert = find(angleExpert == angles(ai));
%         scatter3(pcaCoord{vi}(numNaiveTrial*4 + numExpertTrial + angleIndExpert,1), pcaCoord{vi}(numNaiveTrial*4 + numExpertTrial + angleIndExpert,2), pcaCoord{vi}(numNaiveTrial*4 + numExpertTrial + angleIndExpert,3), 20, colors(ai,:), 'o', 'filled')
%     end
    
    %% Plotting activity before the answer lick
    %% Naive (open) and expert (filled) overlaid
    figure, hold on
    for ai = 1 : length(angles)
        angleIndNaive = find(angleNaive == angles(ai));
        scatter3(pcaCoord{vi}(numNaiveTrial+angleIndNaive,1), pcaCoord{vi}(numNaiveTrial+angleIndNaive,2), pcaCoord{vi}(numNaiveTrial+angleIndNaive,3), 20, colors(ai,:), 'o')
        
        angleIndExpert = find(angleExpert == angles(ai));
        scatter3(pcaCoord{vi}(numNaiveTrial*4 + numExpertTrial + angleIndExpert,1), pcaCoord{vi}(numNaiveTrial*4 + numExpertTrial + angleIndExpert,2), pcaCoord{vi}(numNaiveTrial*4 + numExpertTrial + angleIndExpert,3), 20, colors(ai,:), 'o', 'filled')
    end
        %% Plotting before pole up activity
    %% Naive (open) and expert (filled) overlaid
%     figure, hold on
    for ai = 1 : length(angles)
        angleIndNaive = find(angleNaive == angles(ai));
        scatter3(pcaCoord{vi}(angleIndNaive,1), pcaCoord{vi}(angleIndNaive,2), pcaCoord{vi}(angleIndNaive,3), 20, [0 0 0], 'o')
        
        angleIndExpert = find(angleExpert == angles(ai));
        scatter3(pcaCoord{vi}(numNaiveTrial*4 + angleIndExpert,1), pcaCoord{vi}(numNaiveTrial*4 + angleIndExpert,2), pcaCoord{vi}(numNaiveTrial*4 + angleIndExpert,3), 20, [0 0 0], 'o', 'filled')
    end
    xlabel('PC1'), ylabel('PC2'), zlabel('PC3')
end



%% From persistent neurons only
for vi = 6
    naiveSilentInd = matchedPR.naive(vi).indSilent;
    expertSilentInd = matchedPR.expert(vi).indSilent;
    persistentInd = setdiff(1:length(matchedPR.naive(vi).allID), union(naiveSilentInd, expertSilentInd));
%     allPopAct = [matchedPR.naive(vi).beforePoleUp, matchedPR.naive(vi).touchBeforeAnswer, matchedPR.naive(vi).touchAfterAnswer, matchedPR.naive(vi).afterPoleDown, ...
%         matchedPR.expert(vi).beforePoleUp, matchedPR.expert(vi).touchBeforeAnswer, matchedPR.expert(vi).touchAfterAnswer, matchedPR.expert(vi).afterPoleDown];
    allPopAct = [matchedPR.naive(vi).beforePoleUp, matchedPR.naive(vi).poleBeforeAnswer, matchedPR.naive(vi).poleAfterAnswer, matchedPR.naive(vi).afterPoleDown, ...
        matchedPR.expert(vi).beforePoleUp, matchedPR.expert(vi).poleBeforeAnswer, matchedPR.expert(vi).poleAfterAnswer, matchedPR.expert(vi).afterPoleDown];
    [~, pcaCoord{vi},~,~,varExp{vi}] = pca(allPopAct(persistentInd,:)');
    
    angleNaive = matchedPR.naive(vi).trialAngle;
    angleExpert = matchedPR.expert(vi).trialAngle;
    numNaiveTrial = length(angleNaive);
    numExpertTrial = length(angleExpert);
    
    %% plotting all time point
%     figure, 
%     % Plot Naive
%     subplot(121), hold on
%     for ai = 1 : length(angles)
% %     for ai = 1
%         angleInd = find(angleNaive == angles(ai));
%         for aii = 1 : length(angleInd)
% %         for aii = 1
%             temporalInd = [angleInd(aii), angleInd(aii) + numNaiveTrial, angleInd(aii) + numNaiveTrial*2, angleInd(aii) + numNaiveTrial*3];
% %             plot3(pcaCoord{vi}(temporalInd,1), pcaCoord{vi}(temporalInd,2), pcaCoord{vi}(temporalInd,3), '.-', 'color', colors(ai,:))
%             for tempori = 1 : length(temporalInd)
%                 scatter3(pcaCoord{vi}(temporalInd(tempori),1), pcaCoord{vi}(temporalInd(tempori),2), pcaCoord{vi}(temporalInd(tempori),3), 20, colors(ai,:), 'o', 'filled', 'markerfacealpha', 0.25*tempori)
%             end
%         end
%     end
%     title('Naive')
%     
%     % Plot Expert
%     subplot(122), hold on
%     for ai = 1 : length(angles)
% %     for ai = 1
%         angleInd = find(angleExpert == angles(ai));
%         for aii = 1 : length(angleInd)
% %         for aii = 1
%             temporalInd = [angleInd(aii), angleInd(aii) + numExpertTrial, angleInd(aii) + numExpertTrial*2, angleInd(aii) + numExpertTrial*3]+numNaiveTrial*4;
% %             plot3(pcaCoord{vi}(temporalInd,1), pcaCoord{vi}(temporalInd,2), pcaCoord{vi}(temporalInd,3), '.-', 'color', colors(ai,:))
%             for tempori = 1 : length(temporalInd)
%                 scatter3(pcaCoord{vi}(temporalInd(tempori),1), pcaCoord{vi}(temporalInd(tempori),2), pcaCoord{vi}(temporalInd(tempori),3), 20, colors(ai,:), 'o', 'filled', 'markerfacealpha', 0.25*tempori)
%             end
%         end
%     end
%     title('Expert')
%     
%     sgtitle(sprintf('Volume #%d', vi))
%     
%     
    %% Plotting activity before the answer lick
    %% Naive (open circle) and expert (filled triangle) overlaid
    figure, hold on
    for ai = 1 : length(angles)
        angleIndNaive = find(angleNaive == angles(ai));
        scatter3(pcaCoord{vi}(numNaiveTrial+angleIndNaive,1), pcaCoord{vi}(numNaiveTrial+angleIndNaive,2), pcaCoord{vi}(numNaiveTrial+angleIndNaive,3), 20, colors(ai,:), 'o')
        
        angleIndExpert = find(angleExpert == angles(ai));
        scatter3(pcaCoord{vi}(numNaiveTrial*4 + numExpertTrial + angleIndExpert,1), pcaCoord{vi}(numNaiveTrial*4 + numExpertTrial + angleIndExpert,2), pcaCoord{vi}(numNaiveTrial*4 + numExpertTrial + angleIndExpert,3), 40, colors(ai,:), '*')
    end
    
    %% Plotting before pole up activity
    %% Naive (open) and expert (filled) overlaid
%     figure, hold on
    for ai = 1 : length(angles)
        angleIndNaive = find(angleNaive == angles(ai));
        scatter3(pcaCoord{vi}(angleIndNaive,1), pcaCoord{vi}(angleIndNaive,2), pcaCoord{vi}(angleIndNaive,3), 20, [0 0 0], 'o')
        
        angleIndExpert = find(angleExpert == angles(ai));
        scatter3(pcaCoord{vi}(numNaiveTrial*4 + angleIndExpert,1), pcaCoord{vi}(numNaiveTrial*4 + angleIndExpert,2), pcaCoord{vi}(numNaiveTrial*4 + angleIndExpert,3), 40, [0 0 0], '*')
    end
    
    %% Plotting AFTER pole down
    %% Naive (open) and expert (filled) overlaid
%     figure, hold on
%     for ai = 1 : length(angles)
%         angleIndNaive = find(angleNaive == angles(ai));
%         scatter3(pcaCoord{vi}(numNaiveTrial*3 + angleIndNaive,1), pcaCoord{vi}(numNaiveTrial*3 + angleIndNaive,2), pcaCoord{vi}(numNaiveTrial*3 + angleIndNaive,3), 20, colors(ai,:), 'o')
%         
%         angleIndExpert = find(angleExpert == angles(ai));
%         scatter3(pcaCoord{vi}(numNaiveTrial*4 + numExpertTrial*3 + angleIndExpert,1), pcaCoord{vi}(numNaiveTrial*4 + numExpertTrial*3 + angleIndExpert,2), pcaCoord{vi}(numNaiveTrial*4 + numExpertTrial*3 + angleIndExpert,3), 20, colors(ai,:), 'o', 'filled')
%     end
%     
    xlabel('PC1'), ylabel('PC2'), zlabel('PC3')
end




%% From persistently angle-tuned neurons only
for vi = 6
    naiveTunedInd = matchedPR.naive(vi).indTuned;
    expertTunedInd = matchedPR.expert(vi).indTuned;
    persTunedInd = intersect(naiveTunedInd, expertTunedInd);
%     allPopAct = [matchedPR.naive(vi).beforePoleUp, matchedPR.naive(vi).touchBeforeAnswer, matchedPR.naive(vi).touchAfterAnswer, matchedPR.naive(vi).afterPoleDown, ...
%         matchedPR.expert(vi).beforePoleUp, matchedPR.expert(vi).touchBeforeAnswer, matchedPR.expert(vi).touchAfterAnswer, matchedPR.expert(vi).afterPoleDown];
    allPopAct = [matchedPR.naive(vi).beforePoleUp, matchedPR.naive(vi).poleBeforeAnswer, matchedPR.naive(vi).poleAfterAnswer, matchedPR.naive(vi).afterPoleDown, ...
        matchedPR.expert(vi).beforePoleUp, matchedPR.expert(vi).poleBeforeAnswer, matchedPR.expert(vi).poleAfterAnswer, matchedPR.expert(vi).afterPoleDown];
    [~, pcaCoord{vi},~,~,varExp{vi}] = pca(allPopAct(persTunedInd,:)');
    
    angleNaive = matchedPR.naive(vi).trialAngle;
    angleExpert = matchedPR.expert(vi).trialAngle;
    numNaiveTrial = length(angleNaive);
    numExpertTrial = length(angleExpert);
    
    %% plotting all time point
%     figure, 
%     % Plot Naive
%     subplot(121), hold on
%     for ai = 1 : length(angles)
% %     for ai = 1
%         angleInd = find(angleNaive == angles(ai));
%         for aii = 1 : length(angleInd)
% %         for aii = 1
%             temporalInd = [angleInd(aii), angleInd(aii) + numNaiveTrial, angleInd(aii) + numNaiveTrial*2, angleInd(aii) + numNaiveTrial*3];
% %             plot3(pcaCoord{vi}(temporalInd,1), pcaCoord{vi}(temporalInd,2), pcaCoord{vi}(temporalInd,3), '.-', 'color', colors(ai,:))
%             for tempori = 1 : length(temporalInd)
%                 scatter3(pcaCoord{vi}(temporalInd(tempori),1), pcaCoord{vi}(temporalInd(tempori),2), pcaCoord{vi}(temporalInd(tempori),3), 20, colors(ai,:), 'o', 'filled', 'markerfacealpha', 0.25*tempori)
%             end
%         end
%     end
%     title('Naive')
%     
%     % Plot Expert
%     subplot(122), hold on
%     for ai = 1 : length(angles)
% %     for ai = 1
%         angleInd = find(angleExpert == angles(ai));
%         for aii = 1 : length(angleInd)
% %         for aii = 1
%             temporalInd = [angleInd(aii), angleInd(aii) + numExpertTrial, angleInd(aii) + numExpertTrial*2, angleInd(aii) + numExpertTrial*3]+numNaiveTrial*4;
% %             plot3(pcaCoord{vi}(temporalInd,1), pcaCoord{vi}(temporalInd,2), pcaCoord{vi}(temporalInd,3), '.-', 'color', colors(ai,:))
%             for tempori = 1 : length(temporalInd)
%                 scatter3(pcaCoord{vi}(temporalInd(tempori),1), pcaCoord{vi}(temporalInd(tempori),2), pcaCoord{vi}(temporalInd(tempori),3), 20, colors(ai,:), 'o', 'filled', 'markerfacealpha', 0.25*tempori)
%             end
%         end
%     end
%     title('Expert')
%     
%     sgtitle(sprintf('Volume #%d', vi))
%     
%     
    %% Plotting touch response before the answer lick
    %% Naive (open) and expert (filled) overlaid
    figure, hold on
    for ai = 1 : length(angles)
        angleIndNaive = find(angleNaive == angles(ai));
        scatter3(pcaCoord{vi}(numNaiveTrial+angleIndNaive,1), pcaCoord{vi}(numNaiveTrial+angleIndNaive,2), pcaCoord{vi}(numNaiveTrial+angleIndNaive,3), 20, colors(ai,:), 'o')
        
        angleIndExpert = find(angleExpert == angles(ai));
        scatter3(pcaCoord{vi}(numNaiveTrial*4 + numExpertTrial + angleIndExpert,1), pcaCoord{vi}(numNaiveTrial*4 + numExpertTrial + angleIndExpert,2), pcaCoord{vi}(numNaiveTrial*4 + numExpertTrial + angleIndExpert,3), 20, colors(ai,:), 'o', 'filled')
    end
    
    %% Plotting before pole up activity
    %% Naive (open) and expert (filled) overlaid
%     figure, hold on
    for ai = 1 : length(angles)
        angleIndNaive = find(angleNaive == angles(ai));
        scatter3(pcaCoord{vi}(angleIndNaive,1), pcaCoord{vi}(angleIndNaive,2), pcaCoord{vi}(angleIndNaive,3), 20, [0 0 0], 'o')
        
        angleIndExpert = find(angleExpert == angles(ai));
        scatter3(pcaCoord{vi}(numNaiveTrial*4 + angleIndExpert,1), pcaCoord{vi}(numNaiveTrial*4 + angleIndExpert,2), pcaCoord{vi}(numNaiveTrial*4 + angleIndExpert,3), 20, [0 0 0], 'o', 'filled')
    end    
    
    xlabel('PC1'), ylabel('PC2'), zlabel('PC3')
end

%%
saveDir = 'C:\Users\shires\Dropbox\Works\Presentation\Conference\2020 SfN\';
fn = 'persAT_vi4.eps';
export_fig([saveDir, fn], '-depsc', '-painters', '-r600', '-transparent')
fix_eps_fonts([saveDir, fn])




%% Making videos
fn = 'pca_vi04.mp4';
v = VideoWriter(fn, 'MPEG-4');
v.FrameRate = 30;
v.Quality = 100;
open(v)
az = 0; el = 90; % start
for el = 90:-1:-90
    view(az,el)
    xlim([-10 50]), ylim([-30 30]), zlim([-20 30])
    drawnow;
    F = getframe(gcf);
    writeVideo(v,F.cdata)
end
for el = -90:0
    view(az,el)
    xlim([-10 50]), ylim([-30 30]), zlim([-20 30])
    drawnow;
    F = getframe(gcf);
    writeVideo(v,F.cdata)
end
for az = 0:180
    view(az,el)
    xlim([-10 50]), ylim([-30 30]), zlim([-20 30])
    drawnow;
    F = getframe(gcf);
    writeVideo(v,F.cdata)
end
for el = 0 : 82
   az = -180 + el * (-114+180)/82;
   view(az,el)
    xlim([-10 50]), ylim([-30 30]), zlim([-20 30])
    drawnow;
    F = getframe(gcf);
    writeVideo(v,F.cdata)
end
for az = -114:-90
    el = 82 + -(az+114) * 8 / (90-114);
    view(az,el)
    xlim([-10 50]), ylim([-30 30]), zlim([-20 30])
    drawnow;
    F = getframe(gcf);
    writeVideo(v,F.cdata)
end
close(v)


%% Making videos
fn = 'pca_vi06.mp4';
v = VideoWriter(fn, 'MPEG-4');
v.FrameRate = 30;
v.Quality = 100;
open(v)
az = 0; el = 90; % start
for el = 90:-1:-90
    view(az,el)
    xlim([-20 60]), ylim([-40 50]), zlim([-30 30])
    drawnow;
    F = getframe(gcf);
    writeVideo(v,F.cdata)
end
for el = -90:0
    view(az,el)
    xlim([-20 60]), ylim([-40 50]), zlim([-30 30])
    drawnow;
    F = getframe(gcf);
    writeVideo(v,F.cdata)
end
for az = 0:180
    view(az,el)
    xlim([-20 60]), ylim([-40 50]), zlim([-30 30])
    drawnow;
    F = getframe(gcf);
    writeVideo(v,F.cdata)
end
for el = 0 : 90
    az = 180-el;
    view(az,el)
    xlim([-20 60]), ylim([-40 50]), zlim([-30 30])
    drawnow;
    F = getframe(gcf);
    writeVideo(v,F.cdata)
end
% for az = -114:-90
%     el = 82 + -(az+114) * 8 / (90-114);
%     view(az,el)
%     xlim([-20 60]), ylim([-40 50]), zlim([-30 30])
%     drawnow;
%     F = getframe(gcf);
%     writeVideo(v,F.cdata)
% end
close(v)


%% Sanity check
vi = 8;
naiveSilentInd = matchedPR.naive(vi).indSilent;
expertSilentInd = matchedPR.expert(vi).indSilent;
persSilentInd = intersect(naiveSilentInd, expertSilentInd);


%% From transient neurons only
dims = [1:3];
for vi = 1
    naiveSilentInd = matchedPR.naive(vi).indSilent;
    expertSilentInd = matchedPR.expert(vi).indSilent;
    transientInd = union(naiveSilentInd, expertSilentInd);
%     allPopAct = [matchedPR.naive(vi).beforePoleUp, matchedPR.naive(vi).touchBeforeAnswer, matchedPR.naive(vi).touchAfterAnswer, matchedPR.naive(vi).afterPoleDown, ...
%         matchedPR.expert(vi).beforePoleUp, matchedPR.expert(vi).touchBeforeAnswer, matchedPR.expert(vi).touchAfterAnswer, matchedPR.expert(vi).afterPoleDown];
    allPopAct = [matchedPR.naive(vi).beforePoleUp, matchedPR.naive(vi).poleBeforeAnswer, matchedPR.naive(vi).poleAfterAnswer, matchedPR.naive(vi).afterPoleDown, ...
        matchedPR.expert(vi).beforePoleUp, matchedPR.expert(vi).poleBeforeAnswer, matchedPR.expert(vi).poleAfterAnswer, matchedPR.expert(vi).afterPoleDown];
    [~, pcaCoord{vi},~,~,varExp{vi}] = pca(allPopAct(transientInd,:)');
    
    angleNaive = matchedPR.naive(vi).trialAngle;
    angleExpert = matchedPR.expert(vi).trialAngle;
    numNaiveTrial = length(angleNaive);
    numExpertTrial = length(angleExpert);

    %% Plotting touch response before the answer lick
    %% Naive (open) and expert (filled) overlaid
    figure, hold on
    for ai = 1 : length(angles)
        angleIndNaive = find(angleNaive == angles(ai));
        scatter3(pcaCoord{vi}(numNaiveTrial+angleIndNaive,dims(1)), pcaCoord{vi}(numNaiveTrial+angleIndNaive,dims(2)), pcaCoord{vi}(numNaiveTrial+angleIndNaive,dims(3)), 20, colors(ai,:), 'o')
        
        angleIndExpert = find(angleExpert == angles(ai));
        scatter3(pcaCoord{vi}(numNaiveTrial*4 + numExpertTrial + angleIndExpert, dims(1)), pcaCoord{vi}(numNaiveTrial*4 + numExpertTrial + angleIndExpert, dims(2)), ...
            pcaCoord{vi}(numNaiveTrial*4 + numExpertTrial + angleIndExpert, dims(3)), 20, colors(ai,:), 'o', 'filled')
    end
    
    %% Plotting before pole up activity
    %% Naive (open) and expert (filled) overlaid
%     figure, hold on
    for ai = 1 : length(angles)
        angleIndNaive = find(angleNaive == angles(ai));
        scatter3(pcaCoord{vi}(angleIndNaive, dims(1)), pcaCoord{vi}(angleIndNaive, dims(2)), pcaCoord{vi}(angleIndNaive, dims(3)), 20, [0 0 0], 'o')
        
        angleIndExpert = find(angleExpert == angles(ai));
        scatter3(pcaCoord{vi}(numNaiveTrial*4 + angleIndExpert, dims(1)), pcaCoord{vi}(numNaiveTrial*4 + angleIndExpert, dims(2)), pcaCoord{vi}(numNaiveTrial*4 + angleIndExpert, dims(3)), 20, [0 0 0], 'o', 'filled')
    end    
    
    xlabel('PC1'), ylabel('PC2'), zlabel('PC3')
end




%% From all angle-tuned neurons only
dims = [1:3];
for vi = 4
    naiveTunedInd = matchedPR.naive(vi).indTuned;
    expertTunedInd = matchedPR.expert(vi).indTuned;
    allTunedInd = union(naiveTunedInd, expertTunedInd);
    allPopAct = [matchedPR.naive(vi).beforePoleUp, matchedPR.naive(vi).touchBeforeAnswer, matchedPR.naive(vi).touchAfterAnswer, matchedPR.naive(vi).afterPoleDown, ...
        matchedPR.expert(vi).beforePoleUp, matchedPR.expert(vi).touchBeforeAnswer, matchedPR.expert(vi).touchAfterAnswer, matchedPR.expert(vi).afterPoleDown];
    [~, pcaCoord{vi},~,~,varExp{vi}] = pca(allPopAct(allTunedInd,:)');
    
    angleNaive = matchedPR.naive(vi).trialAngle;
    angleExpert = matchedPR.expert(vi).trialAngle;
    numNaiveTrial = length(angleNaive);
    numExpertTrial = length(angleExpert);

    %% Plotting touch response before the answer lick
    %% Naive (open) and expert (filled) overlaid
    figure, hold on
    for ai = 1 : length(angles)
        angleIndNaive = find(angleNaive == angles(ai));
        scatter3(pcaCoord{vi}(numNaiveTrial+angleIndNaive,dims(1)), pcaCoord{vi}(numNaiveTrial+angleIndNaive,dims(2)), pcaCoord{vi}(numNaiveTrial+angleIndNaive,dims(3)), 20, colors(ai,:), 'o')
        
        angleIndExpert = find(angleExpert == angles(ai));
        scatter3(pcaCoord{vi}(numNaiveTrial*4 + numExpertTrial + angleIndExpert, dims(1)), pcaCoord{vi}(numNaiveTrial*4 + numExpertTrial + angleIndExpert, dims(2)), ...
            pcaCoord{vi}(numNaiveTrial*4 + numExpertTrial + angleIndExpert, dims(3)), 20, colors(ai,:), 'o', 'filled')
    end
    
    %% Plotting before pole up activity
    %% Naive (open) and expert (filled) overlaid
%     figure, hold on
    for ai = 1 : length(angles)
        angleIndNaive = find(angleNaive == angles(ai));
        scatter3(pcaCoord{vi}(angleIndNaive, dims(1)), pcaCoord{vi}(angleIndNaive, dims(2)), pcaCoord{vi}(angleIndNaive, dims(3)), 20, [0 0 0], 'o')
        
        angleIndExpert = find(angleExpert == angles(ai));
        scatter3(pcaCoord{vi}(numNaiveTrial*4 + angleIndExpert, dims(1)), pcaCoord{vi}(numNaiveTrial*4 + angleIndExpert, dims(2)), pcaCoord{vi}(numNaiveTrial*4 + angleIndExpert, dims(3)), 20, [0 0 0], 'o', 'filled')
    end    
    
    xlabel('PC1'), ylabel('PC2'), zlabel('PC3')
end



%% From persistently active, angle-tuned neurons only
dims = [1:3];
for vi = 4
    naiveTunedInd = matchedPR.naive(vi).indTuned;
    expertTunedInd = matchedPR.expert(vi).indTuned;
    persInd = intersect( setdiff(matchedPR.naive(vi).allInd, matchedPR.naive(vi).indSilent), setdiff(matchedPR.expert(vi).allInd, matchedPR.expert(vi).indSilent) );
    activeTunedInd = intersect( persInd, union(naiveTunedInd, expertTunedInd) );
%     allPopAct = [matchedPR.naive(vi).beforePoleUp, matchedPR.naive(vi).touchBeforeAnswer, matchedPR.naive(vi).touchAfterAnswer, matchedPR.naive(vi).afterPoleDown, ...
%         matchedPR.expert(vi).beforePoleUp, matchedPR.expert(vi).touchBeforeAnswer, matchedPR.expert(vi).touchAfterAnswer, matchedPR.expert(vi).afterPoleDown];
    allPopAct = [matchedPR.naive(vi).beforePoleUp, matchedPR.naive(vi).poleBeforeAnswer, matchedPR.naive(vi).poleAfterAnswer, matchedPR.naive(vi).afterPoleDown, ...
        matchedPR.expert(vi).beforePoleUp, matchedPR.expert(vi).poleBeforeAnswer, matchedPR.expert(vi).poleAfterAnswer, matchedPR.expert(vi).afterPoleDown];
    [~, pcaCoord{vi},~,~,varExp{vi}] = pca(allPopAct(activeTunedInd,:)');
    
    angleNaive = matchedPR.naive(vi).trialAngle;
    angleExpert = matchedPR.expert(vi).trialAngle;
    numNaiveTrial = length(angleNaive);
    numExpertTrial = length(angleExpert);

    %% Plotting touch response before the answer lick
    %% Naive (open) and expert (filled) overlaid
    figure, hold on
    for ai = 1 : length(angles)
        angleIndNaive = find(angleNaive == angles(ai));
        scatter3(pcaCoord{vi}(numNaiveTrial+angleIndNaive,dims(1)), pcaCoord{vi}(numNaiveTrial+angleIndNaive,dims(2)), pcaCoord{vi}(numNaiveTrial+angleIndNaive,dims(3)), 20, colors(ai,:), 'o')
        
        angleIndExpert = find(angleExpert == angles(ai));
        scatter3(pcaCoord{vi}(numNaiveTrial*4 + numExpertTrial + angleIndExpert, dims(1)), pcaCoord{vi}(numNaiveTrial*4 + numExpertTrial + angleIndExpert, dims(2)), ...
            pcaCoord{vi}(numNaiveTrial*4 + numExpertTrial + angleIndExpert, dims(3)), 20, colors(ai,:), 'o', 'filled')
    end
    
    %% Plotting before pole up activity
    %% Naive (open) and expert (filled) overlaid
%     figure, hold on
    for ai = 1 : length(angles)
        angleIndNaive = find(angleNaive == angles(ai));
        scatter3(pcaCoord{vi}(angleIndNaive, dims(1)), pcaCoord{vi}(angleIndNaive, dims(2)), pcaCoord{vi}(angleIndNaive, dims(3)), 20, [0 0 0], 'o')
        
        angleIndExpert = find(angleExpert == angles(ai));
        scatter3(pcaCoord{vi}(numNaiveTrial*4 + angleIndExpert, dims(1)), pcaCoord{vi}(numNaiveTrial*4 + angleIndExpert, dims(2)), pcaCoord{vi}(numNaiveTrial*4 + angleIndExpert, dims(3)), 20, [0 0 0], 'o', 'filled')
    end    
    
    xlabel('PC1'), ylabel('PC2'), zlabel('PC3')
end

%%
%%
%%









%%
%% How does clustering look like between persistently active angle-tuned neurons, VS persistently angle-tuned neurons?
%% When the activity was matched throughout the trial and across learning
%%

ciNaivePole = zeros(numVol, 2); % (:,1) persistent & angle-tuned, (:,2) persistentLY angle-tuned
ciExpertPole = zeros(numVol, 2);
ciNaiveBaseline = zeros(numVol, 2); % (:,1) persistent & angle-tuned, (:,2) persistentLY angle-tuned
ciExpertBaseline = zeros(numVol, 2);

for vi = 1 : numVol
    naiveTunedInd = matchedPR.naive(vi).indTuned;
    expertTunedInd = matchedPR.expert(vi).indTuned;
    persInd = intersect( setdiff(matchedPR.naive(vi).allInd, matchedPR.naive(vi).indSilent), setdiff(matchedPR.expert(vi).allInd, matchedPR.expert(vi).indSilent) );
    activeTunedInd = intersect( persInd, union(naiveTunedInd, expertTunedInd) );
    persTunedInd = intersect(naiveTunedInd, expertTunedInd);
    
    allPopAct = [matchedPR.naive(vi).beforePoleUp, matchedPR.naive(vi).poleBeforeAnswer, matchedPR.naive(vi).poleAfterAnswer, matchedPR.naive(vi).afterPoleDown, ...
        matchedPR.expert(vi).beforePoleUp, matchedPR.expert(vi).poleBeforeAnswer, matchedPR.expert(vi).poleAfterAnswer, matchedPR.expert(vi).afterPoleDown];
    
    angleNaive = matchedPR.naive(vi).trialAngle;
    angleExpert = matchedPR.expert(vi).trialAngle;
    numNaiveTrial = length(angleNaive);
    numExpertTrial = length(angleExpert);
    
    [~, pcaCoord] = pca(allPopAct(activeTunedInd,:)');
    ciNaivePole(vi,1) = clustering_index(  pcaCoord(numNaiveTrial+1:numNaiveTrial*2,1:3), angleNaive  );
    ciNaiveBaseline(vi,1) = clustering_index(pcaCoord(1:numNaiveTrial,1:3), angleNaive);
    
    ciExpertPole(vi,1) = clustering_index(  pcaCoord( numNaiveTrial*4 + numExpertTrial + 1 : numNaiveTrial*4 + numExpertTrial*2 , 1:3 ), angleExpert  );
    ciExpertBaseline(vi,1) = clustering_index(pcaCoord(numNaiveTrial*4+1 : numNaiveTrial*4 + numExpertTrial, 1:3), angleExpert);
    
    
    
    [~, pcaCoord] = pca(allPopAct(persTunedInd,:)');
    ciNaivePole(vi,2) = clustering_index(pcaCoord(numNaiveTrial+1:numNaiveTrial*2,1:3), angleNaive);
    ciNaiveBaseline(vi,2) = clustering_index(pcaCoord(1:numNaiveTrial,1:3), angleNaive);
    
    ciExpertPole(vi,2) = clustering_index(  pcaCoord( numNaiveTrial*4 + numExpertTrial + 1 : numNaiveTrial*4 + numExpertTrial*2 , 1:3 ), angleExpert  );
    ciExpertBaseline(vi,2) = clustering_index(pcaCoord(numNaiveTrial*4+1 : numNaiveTrial*4 + numExpertTrial, 1:3), angleExpert);
end
%%
figure, hold on
errorbar([1:2]-0.1, mean(ciNaiveBaseline), sem(ciNaiveBaseline), 'ko', 'lines', 'no')
errorbar([1:2]+0.1, mean(ciExpertBaseline), sem(ciExpertBaseline), 'ro', 'lines', 'no')

legend({'Naive', 'Expert'}, 'autoupdate', 'off')

errorbar([4:5]-0.1, mean(ciNaivePole), sem(ciNaivePole), 'ko', 'lines', 'no')
errorbar([4:5]+0.1, mean(ciExpertPole), sem(ciExpertPole), 'ro', 'lines', 'no')

xlim([0.5 5.5])
xticks([1,2,4,5])
xticklabels({'Baseline (pers & tuned)', 'Baseline (persistentLY tuned)', 'Before answer (pers & tuned)', 'Before answer (persistentLY tuned)'})
xtickangle(45)
ylabel('Clustering index')
yticks(0:0.1:0.3)



%%
colorsTransient = [248 171 66; 40 170 225] / 255;
figure('unit', 'inch', 'position', [2 2 4 6]), hold on
errorbar(1-0.2, mean(ciNaiveBaseline(:,2)), sem(ciNaiveBaseline(:,2)), 'o', 'color', colorsTransient(1,:))
errorbar(1+0.2, mean(ciExpertBaseline(:,2)), sem(ciExpertBaseline(:,2)), 'o', 'color', colorsTransient(2,:))
legend({'Naive', 'Expert'}, 'autoupdate', 'off')
errorbar(2-0.2, mean(ciNaivePole(:,2)), sem(ciNaivePole(:,2)), 'o', 'color', colorsTransient(1,:))
errorbar(2+0.2, mean(ciExpertPole(:,2)), sem(ciExpertPole(:,2)), 'o', 'color', colorsTransient(2,:))
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Baseline', 'Sampling period'}), xtickangle(45)
yticks(0:0.1:0.3), ylabel('Clustering index')
for vi = 1: numVol
    plot([1.8 2.2], [ciNaivePole(vi,2), ciExpertPole(vi,2)], 'k-')
    plot([0.8 1.2], [ciNaiveBaseline(vi,2), ciExpertBaseline(vi,2)], 'k-')
end
errorbar(1-0.2, mean(ciNaiveBaseline(:,2)), sem(ciNaiveBaseline(:,2)), 'o', 'color', colorsTransient(1,:))
errorbar(1+0.2, mean(ciExpertBaseline(:,2)), sem(ciExpertBaseline(:,2)), 'o', 'color', colorsTransient(2,:))
errorbar(2-0.2, mean(ciNaivePole(:,2)), sem(ciNaivePole(:,2)), 'o', 'color', colorsTransient(1,:))
errorbar(2+0.2, mean(ciExpertPole(:,2)), sem(ciExpertPole(:,2)), 'o', 'color', colorsTransient(2,:))
%%
[~, p] = paired_test(ciNaivePole(:,2), ciExpertPole(:,2))


%%
saveDir = 'C:\Users\shires\Dropbox\Works\Presentation\Conference\2020 SfN\';
fn = 'ci_comparison.eps';
export_fig([saveDir, fn], '-depsc', '-painters', '-r600', '-transparent')
fix_eps_fonts([saveDir, fn])



%%
figure, hold on
for i = 1 : numVol
    plot([1,2],[ciNaive(i,2), ciExpert(i,2)], 'k-')
    if i == 8 || i == 9
        scatter(1,ciNaive(i,2), 'ko', 'filled')
        scatter(2,ciExpert(i,2), 'ro', 'filled')
    else
        scatter(1,ciNaive(i,2), 'ko')
        scatter(2,ciExpert(i,2), 'ro')
    end
end

xlim([0.5 2.5])
xticks(1:2)
xticklabels({'Naive', 'Expert'})
ylabel('Clutering index')
yticks([0:0.1:0.3])



%%
%% Change in angle-tunedness
%%

atChange = zeros(numVol, 1); % persistently tuned / tuned at any session
for vi = 1 : numVol
%     persInd = intersect( setdiff(matchedPR.naive(vi).allInd, matchedPR.naive(vi).indSilent), setdiff(matchedPR.expert(vi).allInd, matchedPR.expert(vi).indSilent) );
    allTunedInd = union( matchedPR.naive(vi).indTuned, matchedPR.expert(vi).indTuned );
    persTunedInd = intersect( matchedPR.naive(vi).indTuned, matchedPR.expert(vi).indTuned );
    if ~isempty(setdiff(persTunedInd, allTunedInd))
        error('Something''s wrong')
    end
    atChange(vi) = length(persTunedInd) / length(allTunedInd);
end

figure, hold on
plot([atChange(1), atChange(2)], 'ko-')
plot(2, atChange(3), 'ko')
for i = 4:2:10
    plot([atChange(i), atChange(i+1)], 'ko-')
end
plot([atChange(1), atChange(2)], 'k-')

xlim([0.5 2.5])
xticks(1:2)
xticklabels({'Upper volume', 'Lower volume'})
ylabel('Angle-tuning consistency')
yticks([0:0.1:0.5])





%%
%% Controls - Check activity rate and angle selectivity between Naive and Expert sessions
%% Plus # of touch and # of touch frames, those across angles (before answer lick)
% Just focus on persistent & angle-tuned neurons
ar = zeros(numVol, 2); % (:,1) Naive, (:,2) Expert
arBase = zeros(numVol, 2); % Activity rate from the baseline
arTouch = zeros(numVol, 2); % Activity rate from before the answer lick after pole up
arAfterAnswer = zeros(numVol, 2);
arAfterPole = zeros(numVol, 2);

as = zeros(numVol, 2);
asExtreme = zeros(numVol, 2); % Angle selectivity from angle-tuned neurons preferring 45, 120, and 135 degrees
asInter = zeros(numVol, 2); % Angle selectivity from angle-tuned neurons preferring 60, 75, 90, and 105 degrees

numTouch = zeros(numVol, length(angles)+1, 2); % (:,:,1) naive, (:,:,2) expert / (:,1,:) from all trials
numTF = zeros(numVol, length(angles), 2); % (:,:,1) naive, (:,:,2) expert
numTouchFstat = zeros(numVol, 2); % F-statistic (same as anova p-value): between-group variability / within-group variability, compared to its distribution
numTFFstat = zeros(numVol, 2);

extremeAngles = [45,120,135];
interAngles = 60:15:105;
for vi = 1 : numVol
    indPers = intersect( setdiff(matchedPR.naive(vi).allInd, matchedPR.naive(vi).indSilent), setdiff(matchedPR.expert(vi).allInd, matchedPR.expert(vi).indSilent) );
    indTuned = union( matchedPR.naive(vi).indTuned, matchedPR.expert(vi).indTuned );
    indActiveTuned = intersect(indPers, indTuned);
    indPersTuned = intersect( matchedPR.naive(vi).indTuned, matchedPR.expert(vi).indTuned );
    ar(vi,1) = mean(matchedPR.naive(vi).eventRate(indPersTuned));
    ar(vi,2) = mean(matchedPR.expert(vi).eventRate(indPersTuned));

    if ~isempty((find(matchedPR.expert(vi).eventRate(indPersTuned)==0)))
        error('Silent neuron detected.')
    end
    
    arBase(vi,1) = mean(mean(matchedPR.naive(vi).beforePoleUp(indPersTuned,:),2));
    arBase(vi,2) = mean(mean(matchedPR.expert(vi).beforePoleUp(indPersTuned,:),2));
    
    arTouch(vi,1) = mean(mean(matchedPR.naive(vi).poleBeforeAnswer(indPersTuned,:),2));
    arTouch(vi,2) = mean(mean(matchedPR.expert(vi).poleBeforeAnswer(indPersTuned,:),2));
    
    arAfterAnswer(vi,1) = mean(nanmean(matchedPR.naive(vi).poleAfterAnswer(indPersTuned,:),2));
    arAfterAnswer(vi,2) = mean(nanmean(matchedPR.expert(vi).poleAfterAnswer(indPersTuned,:),2));
    
    arAfterPole(vi,1) = mean(nanmean(matchedPR.naive(vi).afterPoleDown(indPersTuned,:),2));
    arAfterPole(vi,2) = mean(nanmean(matchedPR.expert(vi).afterPoleDown(indPersTuned,:),2));
    
    
    numTouch(vi,1,1) = mean(matchedPR.naive(vi).numTouchBeforeAnswer);
    numTouch(vi,1,2) = mean(matchedPR.expert(vi).numTouchBeforeAnswer);
    numTF(vi,1,1) = mean(matchedPR.naive(vi).numTouchFrameBeforeAnswer(1,:));
    numTF(vi,1,2) = mean(matchedPR.expert(vi).numTouchFrameBeforeAnswer(1,:));
    
    for ai = 1 : length(angles)
        % Naive
        angleInd = find(matchedPR.naive(vi).trialAngle == angles(ai));
        numTouch(vi,ai+1,1) = mean(matchedPR.naive(vi).numTouchBeforeAnswer(angleInd));
        numTF(vi,ai+1,1) = mean(matchedPR.naive(vi).numTouchFrameBeforeAnswer(1,angleInd));
        
        % Expert
        angleInd = find(matchedPR.expert(vi).trialAngle == angles(ai));
        numTouch(vi,ai+1,2) = mean(matchedPR.expert(vi).numTouchBeforeAnswer(angleInd));
        numTF(vi,ai+1,2) = mean(matchedPR.expert(vi).numTouchFrameBeforeAnswer(1,angleInd));
    end
    
    numTouchFstat(vi,1) = anova1(matchedPR.naive(vi).numTouchBeforeAnswer, matchedPR.naive(vi).trialAngle, 'off');
    numTouchFstat(vi,2) = anova1(matchedPR.expert(vi).numTouchBeforeAnswer, matchedPR.expert(vi).trialAngle, 'off');
    numTFFstat(vi,1) = anova1(matchedPR.naive(vi).numTouchFrameBeforeAnswer(1,:), matchedPR.naive(vi).trialAngle, 'off');
    numTFFstat(vi,2) = anova1(matchedPR.expert(vi).numTouchFrameBeforeAnswer(1,:), matchedPR.expert(vi).trialAngle, 'off');
    
    
    indNaive = find(ismember(matchedPR.naive(vi).indTuned, indPersTuned)); % index of angle-tuned neurons
    indExpert = find(ismember(matchedPR.expert(vi).indTuned, indPersTuned));
    as(vi,1) = mean(matchedPR.naive(vi).angleSelectivity(indNaive)); % because angle selectivity is calculated only within angle-tuned neurons
    as(vi,2) = mean(matchedPR.expert(vi).angleSelectivity(indExpert));
    
    indNaiveExtremeAll =  intersect(   indPersTuned,    matchedPR.naive(vi).indTuned(  find(ismember(matchedPR.naive(vi).tunedAngle, extremeAngles))  )    ); % index of all neurons
    indNaiveExtremeTuned = find(ismember(matchedPR.naive(vi).indTuned, indNaiveExtremeAll)); % index of angle-tuned neurons
    indNaiveInterAll =  intersect(   indPersTuned,    matchedPR.naive(vi).indTuned(  find(ismember(matchedPR.naive(vi).tunedAngle, interAngles))  )     ); % index of all neurons
    indNaiveInterTuned = find(ismember(matchedPR.naive(vi).indTuned, indNaiveInterAll)); % index of angle-tuned neurons
    
    indExpertExtremeAll =  intersect(   indPersTuned,    matchedPR.expert(vi).indTuned(  find(ismember(matchedPR.expert(vi).tunedAngle, extremeAngles))  )    ); % index of all neurons
    indExpertExtremeTuned = find(ismember(matchedPR.expert(vi).indTuned, indExpertExtremeAll)); % index of angle-tuned neurons
    indExpertInterAll =  intersect(   indPersTuned,    matchedPR.expert(vi).indTuned(  find(ismember(matchedPR.expert(vi).tunedAngle, interAngles))  )     ); % index of all neurons
    indExpertInterTuned = find(ismember(matchedPR.expert(vi).indTuned, indExpertInterAll)); % index of angle-tuned neurons
    
    
    
    asExtreme(vi,1) = mean(matchedPR.naive(vi).angleSelectivity(indNaiveExtremeTuned));
    asExtreme(vi,2) = mean(matchedPR.expert(vi).angleSelectivity(indExpertExtremeTuned));
    
    asInter(vi,1) = mean(matchedPR.naive(vi).angleSelectivity(indNaiveInterTuned));
    asInter(vi,2) = mean(matchedPR.expert(vi).angleSelectivity(indExpertInterTuned));
    
end

%
tempNaive = [ar(:,1), arBase(:,1), arTouch(:,1), arAfterAnswer(:,1), arAfterPole(:,1), as(:,1), asExtreme(:,1), asInter(:,1)];
tempExpert = [ar(:,2), arBase(:,2), arTouch(:,2), arAfterAnswer(:,2), arAfterPole(:,2), as(:,2), asExtreme(:,2), asInter(:,2)];

%%
figure, hold on
errorbar([1:5]-0.1, mean(tempNaive(:,1:5)), sem(tempNaive(:,1:5)), 'ko', 'lines', 'no')
errorbar([1:5]+0.1, mean(tempExpert(:,1:5)), sem(tempExpert(:,1:5)), 'ro', 'lines', 'no')

xlim([0.5 5.5]), xticks(1:5), xticklabels({'Session-wide', 'Before pole up', 'Pole up before answer', 'Pole up after answer', 'After pole down'})
xtickangle(45)
ylabel('Mean event rate (Hz)')
title('Persistently angle-tuned neurons')

%%
figure, hold on
errorbar([1:3]-0.1, mean(tempNaive(:,6:8)), sem(tempNaive(:,6:8)), 'ko', 'lines', 'no')
errorbar([1:3]+0.1, mean(tempExpert(:,6:8)), sem(tempExpert(:,6:8)), 'ro', 'lines', 'no')

xlim([0.5 3.5]), xticks(1:3), xticklabels({'Angle selectivity', 'Angle selectivity - extreme angles', 'Angle selectivity - intermediate angles'})
xtickangle(45)
ylabel('Angle selectivity')
title('Persistently angle-tuned neurons')


%%
figure, hold on
errorbar([1,3:9]-0.1, mean(squeeze(numTouch(:,:,1))), sem(squeeze(numTouch(:,:,1))), 'ko', 'lines', 'no')
errorbar([1,3:9]+0.1, mean(squeeze(numTouch(:,:,2))), sem(squeeze(numTouch(:,:,2))), 'ro', 'lines', 'no')
xticks([1,3:9]), xlim([0.5 9.5])
xticklabels({'All', '45', '60', '75', '90', '105', '120', '135'})
ylabel('Number of touches'), ylim([0 8])

figure, hold on
errorbar([1,3:9]-0.1, mean(squeeze(numTF(:,:,1))), sem(squeeze(numTF(:,:,1))), 'ko', 'lines', 'no')
errorbar([1,3:9]+0.1, mean(squeeze(numTF(:,:,2))), sem(squeeze(numTF(:,:,2))), 'ro', 'lines', 'no')
xticks([1,3:9]), xlim([0.5 9.5])
xticklabels({'All', '45', '60', '75', '90', '105', '120', '135'})
ylabel('Number of touch frames'), ylim([0 6])

%%
figure,
subplot(121), hold on
errorbar(mean(numTouchFstat), sem(numTouchFstat), 'ko', 'lines', 'no')
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
ylabel('Anova p-value number of touch'), ylim([0 0.2]), yticks(0:0.05:0.2)

subplot(122), hold on
errorbar(mean(numTFFstat), sem(numTFFstat), 'ko', 'lines', 'no')
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
ylabel('Anova p-value number of touch'), ylim([0 0.2]), yticks(0:0.05:0.2)









%%
%% Control for activity rate
%%
% Randomly select neurons to have similar activity rate distribution
% (during sampling).
% Repeat 1001 times and select the group with median clustering index
% difference.
% Target persistently angle-tuned neurons.
clear
% baseDir = 'C:\Users\shires\Dropbox\Works\Projects\2020 Neural stretching in S1\Data\';
baseDir = 'C:\Users\jinho\Dropbox\Works\Projects\2020 Neural stretching in S1\Data\';
matchedPR = load([baseDir, 'matchedPopResponse_201230'], 'naive', 'expert');
numVol = 11;
numDim = 3;
angles = 45:15:135;
colors = turbo(length(angles));
numRepeat = 1001;

%% 
% Takes about 3 min

sampleInd = cell(numVol,1);
pcaCoord = cell(numVol,1);
varExp = cell(numVol,1);
clusteringInd = zeros(numVol,2); % (:,1) naive, (:,2) expert

for vi = 1 : numVol
    fprintf('Processing #%d/%d...\n', vi, numVol)
% for vi = 1
    indPersTuned = intersect( matchedPR.naive(vi).indTuned, matchedPR.expert(vi).indTuned );
    
    allAct = [matchedPR.naive(vi).beforePoleUp, matchedPR.naive(vi).poleBeforeAnswer, matchedPR.naive(vi).poleAfterAnswer, matchedPR.naive(vi).afterPoleDown, ...
        matchedPR.expert(vi).beforePoleUp, matchedPR.expert(vi).poleBeforeAnswer, matchedPR.expert(vi).poleAfterAnswer, matchedPR.expert(vi).afterPoleDown];
    naiveAngles = matchedPR.naive(vi).trialAngle;
    numNaiveTrials = length(naiveAngles);
    expertAngles = matchedPR.expert(vi).trialAngle;
    numExpertTrials = length(expertAngles);
    
    naiveAct = mean(matchedPR.naive(vi).poleBeforeAnswer(indPersTuned,:),2);
    expertAct = mean(matchedPR.expert(vi).poleBeforeAnswer(indPersTuned,:),2);
    
    diffAct = naiveAct-expertAct;
    
    if max(diffAct) <= 0
        error('All activity rate goes up after learning.')
    end
    if min(diffAct) >= 0
        error('All activity rate goes down after learning.')
    end
    maxVal = floor(max(diffAct)*10) / 10;
    minVal = ceil(min(diffAct)*10) / 10;
    if abs(maxVal) < abs(minVal)
        minVal = -maxVal;
    else
        maxVal = -minVal;
    end
    distBin = maxVal/10;
%     distBin = 0.5;
    
    binsLow = 0 : -distBin : minVal;
    binsHigh = 0 : distBin : maxVal;
    
    pcaCoordRepeat = cell(numRepeat,1); 
    varExpRepeat = cell(numRepeat,1);
    ciDiff = zeros(numRepeat,1);
    allIndCell = cell(numRepeat,1);
    parfor ri = 1 : numRepeat
        flag = 1;
        count = 0;
        while(flag)
            count = count + 1;
            if count > 100
                error('Distribution not matched after 100 repeats')
            end
            
            indCell = cell(length(binsLow)-1,1);
            for bi = 1 : length(binsLow)-1
                sampleLow = find(diffAct <= binsLow(bi) & diffAct > binsLow(bi+1));
                sampleHigh = find(diffAct >= binsHigh(bi) & diffAct < binsHigh(bi+1));
                numSampleLow = length(sampleLow);
                numSampleHigh = length(sampleHigh);
                numSample = min(numSampleLow, numSampleHigh);
                if numSampleLow < numSampleHigh
                    indCell{bi} = [sampleLow; sampleHigh(randperm(numSampleHigh, numSampleLow))];
                else
                    indCell{bi} = [sampleLow(randperm(numSampleLow, numSampleHigh)); sampleHigh];
                end
            end
            allInd = cell2mat(indCell);
            allIndCell{ri} = allInd;
            newDiff = naiveAct(allInd) - expertAct(allInd);
            if kstest2(naiveAct(allInd), expertAct(allInd)) == 0 % do not reject that naiveAct and expertAct comes from the same population)
                flag = 0; % to escape from the while loop
                [~,pcaCoordRepeat{ri}, ~, ~, varExpRepeat{ri}] = pca(allAct(indPersTuned(allInd),:)');
                ciDiff(ri) = clustering_index(pcaCoordRepeat{ri}(numNaiveTrials+1:numNaiveTrials*2, 1:3), naiveAngles) - ...
                    clustering_index(pcaCoordRepeat{ri}(numNaiveTrials*4 + numExpertTrials + 1 : numNaiveTrials*4 + numExpertTrials*2, 1:3), expertAngles);
            end
        end
    end
    rind = find(ciDiff == median(ciDiff),1,'first');
    sampleInd{vi} = indPersTuned(allIndCell{rind});
    pcaCoord{vi} = pcaCoordRepeat{rind};
    varExp{vi} = varExpRepeat{rind};
    clusteringInd(vi,1) = clustering_index(pcaCoord{vi}(numNaiveTrials+1:numNaiveTrials*2, 1:3), naiveAngles);
    clusteringInd(vi,2) = clustering_index(pcaCoord{vi}(numNaiveTrials*4 + numExpertTrials + 1 : numNaiveTrials*4 + numExpertTrials*2, 1:3), expertAngles);
end

%%
colorsTransient = [248 171 66; 40 170 225] / 255;
histRange = 0:0.1:7;
rrHistNaive = zeros(numVol,length(histRange)-1);
rrHistExpert = zeros(numVol,length(histRange)-1);
for vi = 1 : numVol
    rrNaive = mean(matchedPR.naive(vi).poleBeforeAnswer(sampleInd{vi},:),2);
    rrExpert = mean(matchedPR.expert(vi).poleBeforeAnswer(sampleInd{vi},:),2);
    rrHistNaive(vi,:) = histcounts(rrNaive, histRange, 'norm', 'cdf');
    rrHistExpert(vi,:) = histcounts(rrExpert, histRange, 'norm', 'cdf');
end

figure, hold on
plot(histRange(1:end-1), mean(rrHistNaive), 'color', colorsTransient(1,:))
plot(histRange(1:end-1), mean(rrHistExpert), 'color', colorsTransient(2,:))
legend({'Naive', 'Expert'}, 'autoupdate', false)
boundedline(histRange(1:end-1), mean(rrHistNaive), sem(rrHistNaive), 'cmap', colorsTransient(1,:))
boundedline(histRange(1:end-1), mean(rrHistExpert), sem(rrHistExpert), 'cmap', colorsTransient(2,:))
xlabel('Response rate (Hz)')
ylabel('Cumulative probability')



%%
numSamples = cellfun(@length, sampleInd);
mean(numSamples), sem(numSamples)

figure, hold on
errorbar(1,mean(clusteringInd(:,1)), sem(clusteringInd(:,1)), 'ko', 'lines', 'no')
errorbar(2,mean(clusteringInd(:,2)), sem(clusteringInd(:,2)), 'ro', 'lines', 'no')
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
ylabel('Clustering index')
ylim([0 0.25]), yticks(0:0.05:0.25)
title(sprintf('Response rate controlled, mean(n) = %.1f +/- %.1f', mean(numSamples), sem(numSamples)))



%%
save('responseRateMatching_210323','sampleInd','pcaCoord','varExp','clusteringInd')


%% Comparing LDA
crRrMatched = zeros(numVol,2);
eaRrMatched = zeros(numVol,2);
crRrMatchedShuff = zeros(numVol,2);
eaRrMatchedShuff = zeros(numVol,2);

for vi = 1 : 11
    % Naive
    data = matchedPR.naive(vi).poleBeforeAnswer(sampleInd{vi},:);
    sessionAngles = matchedPR.naive(vi).trialAngle;
    numFold = 10;
    trainFrac = 0.7;

    cr = zeros(numFold,1);
    ea = zeros(numFold,1);
    crShuff = zeros(numFold,1);
    eaShuff = zeros(numFold,1);
    for fi = 1 : numFold
        % Stratify training data
        trainInd = [];
        for ai = 1 : length(angles)
            angleInd = find(sessionAngles == angles(ai));
            trainInd = [trainInd; angleInd(randperm(length(angleInd), round(length(angleInd)*trainFrac)))];
        end
        testInd = setdiff(1:length(sessionAngles), trainInd);
        trainX = data(:,trainInd)';
        trainY = sessionAngles(trainInd);
        testX = data(:,testInd)';
        testY = sessionAngles(testInd);
        mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'discriminant');
        label = predict(mdl, testX);
        cr(fi) = length(find((label-testY)==0)) / length(testY);
        ea(fi) = mean(abs(label-testY));
        
        crShuff(fi) = length(find(label-testY(randperm(length(testY)))==0)) / length(testY);
        eaShuff(fi) = mean(abs(label-testY(randperm(length(testY)))));
    end
    crRrMatched(vi,1) = mean(cr);
    eaRrMatched(vi,1) = mean(ea);
    crRrMatchedShuff(vi,1) = mean(crShuff);
    eaRrMatchedShuff(vi,1) = mean(eaShuff);
    
    % Expert
    data = matchedPR.expert(vi).poleBeforeAnswer(sampleInd{vi},:);
    sessionAngles = matchedPR.expert(vi).trialAngle;
    numFold = 10;
    trainFrac = 0.7;

    cr = zeros(numFold,1);
    ea = zeros(numFold,1);
    crShuff = zeros(numFold,1);
    eaShuff = zeros(numFold,1);
    for fi = 1 : numFold
        % Stratify training data
        trainInd = [];
        for ai = 1 : length(angles)
            angleInd = find(sessionAngles == angles(ai));
            trainInd = [trainInd; angleInd(randperm(length(angleInd), round(length(angleInd)*trainFrac)))];
        end
        testInd = setdiff(1:length(sessionAngles), trainInd);
        trainX = data(:,trainInd)';
        trainY = sessionAngles(trainInd);
        testX = data(:,testInd)';
        testY = sessionAngles(testInd);
        mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'discriminant');
        label = predict(mdl, testX);
        cr(fi) = length(find((label-testY)==0)) / length(testY);
        ea(fi) = mean(abs(label-testY));
        
        crShuff(fi) = length(find(label-testY(randperm(length(testY)))==0)) / length(testY);
        eaShuff(fi) = mean(abs(label-testY(randperm(length(testY)))));
    end
    crRrMatched(vi,2) = mean(cr);
    eaRrMatched(vi,2) = mean(ea);
    crRrMatchedShuff(vi,2) = mean(crShuff);
    eaRrMatchedShuff(vi,2) = mean(eaShuff);
end


%%
figure, hold on
for vi = 1 : 11
    plot([1,2], crRrMatched(vi,:), 'ko-')
end
errorbar([1,2], mean(crRrMatched), sem(crRrMatched), 'ro', 'lines', 'no')
errorbar([1,2], mean(crRrMatchedShuff), sem(crRrMatchedShuff), 'o', 'color', [0.6 0.6 0.6], 'lines', 'no')
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
[~,p,m] = paired_test(crRrMatched(:,1), crRrMatched(:,2));
ylabel('Correct Rate'), ylim([0 0.6])
title(sprintf('Naive VS Expert: p = %.3f (%s)', p, m))



%%

saveDir = 'C:\Users\jinho\Dropbox\Works\grant proposal\2021 NARSAD\';
fn = 'LDA_response_rate_matched.eps';
export_fig([saveDir, fn], '-depsc', '-painters', '-r600', '-transparent')
fix_eps_fonts([saveDir, fn])



%%
%% Control angle selectivity
%%
% Takes about 3 min

sampleInd = cell(numVol,1);
pcaCoord = cell(numVol,1);
varExp = cell(numVol,1);
clusteringInd = zeros(numVol,2); % (:,1) naive, (:,2) expert

for vi = 1 : numVol
    fprintf('Processing #%d/%d...\n', vi, numVol)
% for vi = 1
    indPersTuned = intersect( matchedPR.naive(vi).indTuned, matchedPR.expert(vi).indTuned );
    indNaive = find(ismember(matchedPR.naive(vi).indTuned, indPersTuned)); % ind for tuned
    indExpert = find(ismember(matchedPR.expert(vi).indTuned, indPersTuned)); % ind for tuned
    
    allAct = [matchedPR.naive(vi).beforePoleUp, matchedPR.naive(vi).poleBeforeAnswer, matchedPR.naive(vi).poleAfterAnswer, matchedPR.naive(vi).afterPoleDown, ...
        matchedPR.expert(vi).beforePoleUp, matchedPR.expert(vi).poleBeforeAnswer, matchedPR.expert(vi).poleAfterAnswer, matchedPR.expert(vi).afterPoleDown];
    naiveAngles = matchedPR.naive(vi).trialAngle;
    numNaiveTrials = length(naiveAngles);
    expertAngles = matchedPR.expert(vi).trialAngle;
    numExpertTrials = length(expertAngles);
    
    % These should be from matched neurons
    naiveAS = matchedPR.naive(vi).angleSelectivity(indNaive);
    expertAS = matchedPR.expert(vi).angleSelectivity(indExpert);
    
    diffAS = naiveAS-expertAS;
    
    if max(diffAS) <= 0
        error('All activity rate goes up after learning.')
    end
    if min(diffAS) >= 0
        error('All activity rate goes down after learning.')
    end
    maxVal = floor(max(diffAS)*10) / 10;
    minVal = ceil(min(diffAS)*10) / 10;
    if abs(maxVal) < abs(minVal)
        minVal = -maxVal;
    else
        maxVal = -minVal;
    end
    distBin = maxVal/10;
%     distBin = 0.5;
    
    binsLow = 0 : -distBin : minVal;
    binsHigh = 0 : distBin : maxVal;
    
    pcaCoordRepeat = cell(numRepeat,1); 
    varExpRepeat = cell(numRepeat,1);
    ciDiff = zeros(numRepeat,1);
    allIndCell = cell(numRepeat,1);
    parfor ri = 1 : numRepeat
        flag = 1;
        count = 0;
        while(flag)
            count = count + 1;
            if count > 100
                error('Distribution not matched after 100 repeats')
            end
            
            indCell = cell(length(binsLow)-1,1);
            for bi = 1 : length(binsLow)-1
                sampleLow = find(diffAS <= binsLow(bi) & diffAS > binsLow(bi+1));
                sampleHigh = find(diffAS >= binsHigh(bi) & diffAS < binsHigh(bi+1));
                numSampleLow = length(sampleLow);
                numSampleHigh = length(sampleHigh);
                numSample = min(numSampleLow, numSampleHigh);
                if numSampleLow < numSampleHigh
                    indCell{bi} = [sampleLow; sampleHigh(randperm(numSampleHigh, numSampleLow))];
                else
                    indCell{bi} = [sampleLow(randperm(numSampleLow, numSampleHigh)); sampleHigh];
                end
            end
            allInd = cell2mat(indCell);
            allIndCell{ri} = allInd;
            newDiff = naiveAS(allInd) - naiveAS(allInd);
            if kstest2(naiveAS(allInd), naiveAS(allInd)) == 0 % do not reject that naiveAct and expertAct comes from the same population)
                flag = 0; % to escape from the while loop
                [~,pcaCoordRepeat{ri}, ~, ~, varExpRepeat{ri}] = pca(allAct(indPersTuned(allInd),:)');
                ciDiff(ri) = clustering_index(pcaCoordRepeat{ri}(numNaiveTrials+1:numNaiveTrials*2, 1:3), naiveAngles) - ...
                    clustering_index(pcaCoordRepeat{ri}(numNaiveTrials*4 + numExpertTrials + 1 : numNaiveTrials*4 + numExpertTrials*2, 1:3), expertAngles);
            end
        end
    end
    rind = find(ciDiff == median(ciDiff),1,'first');
    sampleInd{vi} = indPersTuned(allIndCell{rind});
    pcaCoord{vi} = pcaCoordRepeat{rind};
    varExp{vi} = varExpRepeat{rind};
    clusteringInd(vi,1) = clustering_index(pcaCoord{vi}(numNaiveTrials+1:numNaiveTrials*2, 1:3), naiveAngles);
    clusteringInd(vi,2) = clustering_index(pcaCoord{vi}(numNaiveTrials*4 + numExpertTrials + 1 : numNaiveTrials*4 + numExpertTrials*2, 1:3), expertAngles);
end


%%
colorsTransient = [248 171 66; 40 170 225] / 255;
histRange = 0:0.05:2.05;
asHistNaive = zeros(numVol,length(histRange)-1);
asHistExpert = zeros(numVol,length(histRange)-1);
for vi = 1 : numVol
    asNaive = matchedPR.naive(vi).angleSelectivity(ismember(matchedPR.naive(vi).indTuned, sampleInd{vi}));
    asExpert = matchedPR.expert(vi).angleSelectivity(ismember(matchedPR.expert(vi).indTuned, sampleInd{vi}));
    asHistNaive(vi,:) = histcounts(asNaive, histRange, 'norm', 'cdf');
    asHistExpert(vi,:) = histcounts(asExpert, histRange, 'norm', 'cdf');
end

figure, hold on
plot(histRange(1:end-1), mean(asHistNaive), 'color', colorsTransient(1,:))
plot(histRange(1:end-1), mean(asHistExpert), 'color', colorsTransient(2,:))
legend({'Naive', 'Expert'}, 'autoupdate', false)
boundedline(histRange(1:end-1), mean(asHistNaive), sem(asHistNaive), 'cmap', colorsTransient(1,:))
boundedline(histRange(1:end-1), mean(asHistExpert), sem(asHistExpert), 'cmap', colorsTransient(2,:))
xlabel('Angle selectivity')
ylabel('Cumulative probability')




%%
numSamples = cellfun(@length, sampleInd);
mean(numSamples), sem(numSamples)

figure, hold on
errorbar(1,mean(clusteringInd(:,1)), sem(clusteringInd(:,1)), 'ko', 'lines', 'no')
errorbar(2,mean(clusteringInd(:,2)), sem(clusteringInd(:,2)), 'ro', 'lines', 'no')
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
ylabel('Clustering index')
ylim([0 0.25]), yticks(0:0.05:0.25)
title(sprintf('From angle selectivity-matched population, mean(n) = %.1f +/- %.1f', mean(numSamples), sem(numSamples)))





%%
%% Does co-activity matter?
%%
% Shuffle responses within same angle trials to maintain angle-tuning
% curve and angle selectivity while destroying neuron-to-neuron correlation

% Repeat 1001 times and select the repetition with median clustering index
% difference.
% Target persistently angle-tuned neurons.
clear
baseDir = 'C:\Users\shires\Dropbox\Works\Projects\2020 Neural stretching in S1\Data\';
matchedPR = load([baseDir, 'matchedPopResponse_201230'], 'naive', 'expert');
numVol = 11;
numDim = 3;
angles = 45:15:135;
colors = turbo(length(angles));
numRepeat = 1001;

%%
% Takes about 3 min
pcaCoord = cell(numVol,1);
varExp = cell(numVol,1);
clusteringInd = zeros(numVol,2); % (:,1) naive, (:,2) expert
for vi = 1 : numVol
    fprintf('Processing #%d/%d...\n', vi, numVol)
% for vi = 1
    indPersTuned = intersect( matchedPR.naive(vi).indTuned, matchedPR.expert(vi).indTuned );
%     allAct = [matchedPR.naive(vi).beforePoleUp, matchedPR.naive(vi).poleBeforeAnswer, matchedPR.naive(vi).poleAfterAnswer, matchedPR.naive(vi).afterPoleDown, ...
%         matchedPR.expert(vi).beforePoleUp, matchedPR.expert(vi).poleBeforeAnswer, matchedPR.expert(vi).poleAfterAnswer, matchedPR.expert(vi).afterPoleDown];
%     subAct = allAct(indPersTuned,:);
    naiveAngles = matchedPR.naive(vi).trialAngle;
    expertAngles = matchedPR.expert(vi).trialAngle;
    numTrialNaive = length(naiveAngles);
    numTrialExpert = length(expertAngles);
    naiveAngleInds = cell(length(angles),1);
    expertAngleInds = cell(length(angles),1);
    for ai = 1 : length(angles)
        naiveAngleInds{ai} = find(naiveAngles == angles(ai));
        expertAngleInds{ai} = find(expertAngles == angles(ai));
    end
    % Sorting trials based on sorted angles
    naiveTrialInds = cell2mat(naiveAngleInds);
    expertTrialInds = cell2mat(expertAngleInds);
    
    naiveAngleSorted = naiveAngles(naiveTrialInds);
    expertAngleSorted = expertAngles(expertTrialInds);
    naiveSubActCell = cell(1,4);
    naiveSubActCell{1} = matchedPR.naive(vi).beforePoleUp(indPersTuned,:);
    naiveSubActCell{2} = matchedPR.naive(vi).poleBeforeAnswer(indPersTuned,:);
    naiveSubActCell{3} = matchedPR.naive(vi).poleAfterAnswer(indPersTuned,:);
    naiveSubActCell{4} = matchedPR.naive(vi).afterPoleDown(indPersTuned,:);
    expertSubActCell = cell(1,4);
    expertSubActCell{1} = matchedPR.expert(vi).beforePoleUp(indPersTuned,:);
    expertSubActCell{2} = matchedPR.expert(vi).poleBeforeAnswer(indPersTuned,:);
    expertSubActCell{3} = matchedPR.expert(vi).poleAfterAnswer(indPersTuned,:);
    expertSubActCell{4} = matchedPR.expert(vi).afterPoleDown(indPersTuned,:);
    
    pcaRepeat = cell(numRepeat,1);
    veRepeat = cell(numRepeat,1);
    ciRepeat1 = zeros(numRepeat,1);
    ciRepeat2 = zeros(numRepeat,1);
    parfor ri = 1 : numRepeat
        % Shuffle naive
        tempNaiveActCell = cell(1,4);
        for celli = 1 : 4
            tempNaiveActCell{celli} = zeros(size(naiveSubActCell{celli}));
        end
        for ci = 1 : length(indPersTuned)
            repeatInd = cell2mat(cellfun(@(x) x(randperm(length(x))), naiveAngleInds, 'un', 0));
            for celli = 1 : 4
                tempNaiveActCell{celli}(ci,:) = naiveSubActCell{celli}(ci,repeatInd);
            end
        end
        
        % Shuffle expert
        tempExpertActCell = cell(1,4);
        for celli = 1 : 4
            tempExpertActCell{celli} = zeros(size(expertSubActCell{celli}));
        end
        for ci = 1 : length(indPersTuned)
            repeatInd = cell2mat(cellfun(@(x) x(randperm(length(x))), expertAngleInds, 'un', 0));
            for celli = 1 : 4
                tempExpertActCell{celli}(ci,:) = expertSubActCell{celli}(ci,repeatInd);
            end
        end
        
        % Combine activities
        allAct = [cell2mat(tempNaiveActCell), cell2mat(tempExpertActCell)];
        [~, pcaRepeat{ri}, ~, ~, veRepeat{ri}] = pca(allAct');
        ciRepeat1(ri) = clustering_index(pcaRepeat{ri}(numTrialNaive+1:numTrialNaive*2, 1:3), naiveAngleSorted);
        ciRepeat2(ri) = clustering_index(pcaRepeat{ri}(numTrialNaive*4 + numTrialExpert + 1 : numTrialNaive*4 + numTrialExpert*2, 1:3), expertAngleSorted);
    end
    ciDiff = abs(ciRepeat2 - mean(ciRepeat2)) + abs(ciRepeat1 - mean(ciRepeat1));
    rind = find(ciDiff == min(ciDiff),1,'first');
    pcaCoord{vi} = pcaRepeat{rind};
    varExp{vi} = veRepeat{rind};
    clusteringInd(vi,:) = [ciRepeat1(rind), ciRepeat2(rind)];
end

%

figure, hold on
errorbar(1,mean(clusteringInd(:,1)), sem(clusteringInd(:,1)), 'ko', 'lines', 'no')
errorbar(2,mean(clusteringInd(:,2)), sem(clusteringInd(:,2)), 'ro', 'lines', 'no')
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
ylabel('Clustering index')
ylim([0 0.5]), yticks(0:0.1:0.5)
title(sprintf('From decorrelated sample'))



%% How do they look like?
% all angle-sorted
dims = [1:3];
vi = 4;

naiveInd = cell(length(angles),1);
expertInd = cell(length(angles),1);
for ai = 1 : length(angles)
    if ai == 1
        naiveInd{ai} = 1 : length(find(matchedPR.naive(vi).trialAngle==angles(ai)));
        expertInd{ai} = 1 : length(find(matchedPR.expert(vi).trialAngle==angles(ai)));
    else
        prevLength = sum(cellfun(@length, naiveInd(1:ai)));
        naiveInd{ai} = prevLength + 1 : prevLength + length(find(matchedPR.naive(vi).trialAngle==angles(ai)));
        prevLength = sum(cellfun(@length, expertInd(1:ai)));
        expertInd{ai} = prevLength + 1 : prevLength + length(find(matchedPR.expert(vi).trialAngle==angles(ai)));
    end
end

numNaiveTrial = naiveInd{end}(end);
numExpertTrial = expertInd{end}(end);
figure, hold on
for ai = 1 : length(angles)
    scatter3(pcaCoord{vi}(numNaiveTrial+naiveInd{ai},dims(1)), pcaCoord{vi}(numNaiveTrial+naiveInd{ai},dims(2)), pcaCoord{vi}(numNaiveTrial+naiveInd{ai},dims(3)), 20, colors(ai,:), 'o')
    scatter3(pcaCoord{vi}(numNaiveTrial*4 + numExpertTrial + expertInd{ai}, dims(1)), pcaCoord{vi}(numNaiveTrial*4 + numExpertTrial + expertInd{ai}, dims(2)), ...
        pcaCoord{vi}(numNaiveTrial*4 + numExpertTrial + expertInd{ai}, dims(3)), 20, colors(ai,:), 'o', 'filled')
    scatter3(pcaCoord{vi}(naiveInd{ai}, dims(1)), pcaCoord{vi}(naiveInd{ai}, dims(2)), pcaCoord{vi}(naiveInd{ai}, dims(3)), 20, [0 0 0], 'o')
    scatter3(pcaCoord{vi}(numNaiveTrial*4 + expertInd{ai}, dims(1)), pcaCoord{vi}(numNaiveTrial*4 + expertInd{ai}, dims(2)), pcaCoord{vi}(numNaiveTrial*4 + expertInd{ai}, dims(3)), 20, [0 0 0], 'o', 'filled')

end

xlabel('PC1'), ylabel('PC2'), zlabel('PC3')






%%
%% What if I shuffle within each session? (naive and expert session sepratately, and compare the clustering index between original and shuffled data)
%%

pcaCoord = cell(numVol,2);
varExp = cell(numVol,2);
clusteringInd = cell(numVol,2); % (:,1) naive, (:,2) expert
for vi = 1 : numVol
    fprintf('Processing #%d/%d...\n', vi, numVol)
% for vi = 1
    indPersTuned = intersect( matchedPR.naive(vi).indTuned, matchedPR.expert(vi).indTuned );
%     allAct = [matchedPR.naive(vi).beforePoleUp, matchedPR.naive(vi).poleBeforeAnswer, matchedPR.naive(vi).poleAfterAnswer, matchedPR.naive(vi).afterPoleDown, ...
%         matchedPR.expert(vi).beforePoleUp, matchedPR.expert(vi).poleBeforeAnswer, matchedPR.expert(vi).poleAfterAnswer, matchedPR.expert(vi).afterPoleDown];
%     subAct = allAct(indPersTuned,:);
    naiveAngles = matchedPR.naive(vi).trialAngle;
    expertAngles = matchedPR.expert(vi).trialAngle;
    numTrialNaive = length(naiveAngles);
    numTrialExpert = length(expertAngles);
    naiveAngleInds = cell(length(angles),1);
    expertAngleInds = cell(length(angles),1);
    for ai = 1 : length(angles)
        naiveAngleInds{ai} = find(naiveAngles == angles(ai));
        expertAngleInds{ai} = find(expertAngles == angles(ai));
    end
    % Sorting trials based on sorted angles
    naiveTrialInds = cell2mat(naiveAngleInds);
    expertTrialInds = cell2mat(expertAngleInds);
    
    naiveAngleSorted = naiveAngles(naiveTrialInds);
    expertAngleSorted = expertAngles(expertTrialInds);
    naiveSubActCell = cell(1,4);
    naiveSubActCell{1} = matchedPR.naive(vi).beforePoleUp(indPersTuned,:);
    naiveSubActCell{2} = matchedPR.naive(vi).poleBeforeAnswer(indPersTuned,:);
    naiveSubActCell{3} = matchedPR.naive(vi).poleAfterAnswer(indPersTuned,:);
    naiveSubActCell{4} = matchedPR.naive(vi).afterPoleDown(indPersTuned,:);
    expertSubActCell = cell(1,4);
    expertSubActCell{1} = matchedPR.expert(vi).beforePoleUp(indPersTuned,:);
    expertSubActCell{2} = matchedPR.expert(vi).poleBeforeAnswer(indPersTuned,:);
    expertSubActCell{3} = matchedPR.expert(vi).poleAfterAnswer(indPersTuned,:);
    expertSubActCell{4} = matchedPR.expert(vi).afterPoleDown(indPersTuned,:);
    
    pcaRepeatExpert = cell(numRepeat,1);
    pcaRepeatNaive = cell(numRepeat,1);
    veRepeatExpert = cell(numRepeat,1);
    veRepeatNaive = cell(numRepeat,1);
    ciExpert = cell(numRepeat,1);
    ciNaive = cell(numRepeat,1);
    parfor ri = 1 : numRepeat
        % Shuffle naive
        tempNaiveActCell = cell(1,4);
        for celli = 1 : 4
            tempNaiveActCell{celli} = zeros(size(naiveSubActCell{celli}));
        end
        for ci = 1 : length(indPersTuned)
            repeatInd = cell2mat(cellfun(@(x) x(randperm(length(x))), naiveAngleInds, 'un', 0));
            for celli = 1 : 4
                tempNaiveActCell{celli}(ci,:) = naiveSubActCell{celli}(ci,repeatInd);
            end
        end
        naiveAct = [cell2mat(cellfun(@(x) x(:,naiveTrialInds), naiveSubActCell, 'un', 0)), cell2mat(tempNaiveActCell)];
        [~, pcaRepeatNaive{ri}, ~, ~, veRepeatNaive{ri}] = pca(naiveAct');
        ciNaive{ri} = [clustering_index(pcaRepeatNaive{ri}(numTrialNaive+1:numTrialNaive*2,1:3), naiveAngleSorted), ...
            clustering_index(pcaRepeatNaive{ri}(numTrialNaive*5+1:numTrialNaive*6,1:3), naiveAngleSorted)];
        
        % Shuffle expert
        tempExpertActCell = cell(1,4);
        for celli = 1 : 4
            tempExpertActCell{celli} = zeros(size(expertSubActCell{celli}));
        end
        for ci = 1 : length(indPersTuned)
            repeatInd = cell2mat(cellfun(@(x) x(randperm(length(x))), expertAngleInds, 'un', 0));
            for celli = 1 : 4
                tempExpertActCell{celli}(ci,:) = expertSubActCell{celli}(ci,repeatInd);
            end
        end
        expertAct = [cell2mat(cellfun(@(x) x(:,expertTrialInds), expertSubActCell, 'un', 0)), cell2mat(tempExpertActCell)];
        [~, pcaRepeatExpert{ri}, ~, ~, veRepeatNaive{ri}] = pca(expertAct');
        ciExpert{ri} = [clustering_index(pcaRepeatExpert{ri}(numTrialExpert+1:numTrialExpert*2,1:3), expertAngleSorted), ...
            clustering_index(pcaRepeatExpert{ri}(numTrialExpert*5+1:numTrialExpert*6,1:3), expertAngleSorted)];
    end
    ciNaive = cell2mat(ciNaive);
    clusteringInd{vi,1} = ciNaive;
    ciDiff = abs(ciNaive(:,1) - mean(ciNaive(:,1))) + abs(ciNaive(:,2) - mean(ciNaive(:,2)));
    rind = find(ciDiff == min(ciDiff),1,'first');
    pcaCoord{vi,1} = pcaRepeatNaive{rind};
    varExp{vi,1} = veRepeatNaive{rind};
    
    
    ciExpert = cell2mat(ciExpert);
    clusteringInd{vi,2} = ciExpert;
    ciDiff = abs(ciExpert(:,1) - mean(ciExpert(:,1))) + abs(ciExpert(:,2) - mean(ciExpert(:,2)));
    rind = find(ciDiff == min(ciDiff),1,'first');
    pcaCoord{vi,2} = pcaRepeatExpert{rind};
    varExp{vi,2} = veRepeatExpert{rind};
    
end

naiveShuff = cellfun(@(x) mean(x(:,2)-x(:,1)), clusteringInd(:,1));
expertShuff = cellfun(@(x) mean(x(:,2)-x(:,1)), clusteringInd(:,2));

figure, hold on
errorbar(1,mean(naiveShuff), sem(naiveShuff), 'ko')
errorbar(2,mean(expertShuff), sem(expertShuff), 'ro')
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'})
ylabel('\DeltaClustering index (Shuffle - data)'), ylim([0 0.25]), yticks(0:0.1:0.2)





