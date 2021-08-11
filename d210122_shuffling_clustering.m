% Shuffling leads to higher clustering.
% Is this real?
% How can this happen?

% Accidentally, when comparing between persistently angle-tuned neurons,
% when PCA was run separately, I observed that clustering index does NOT
% increase after learning. - Is this really because of combined PCA or some
% other mistake? Look into this first (#0, 2021/02/19 JK)


%% (0) Combined vs separate PCA between persistently angle-tuned neurons across learning

clear
cd('C:\Users\shires\Dropbox\Works\Projects\2020 Neural stretching in S1\Data')
numVol = 11;
angles = 45:15:135;
load('matchedPopResponse_201230')

CIcomb = zeros(numVol, 2);
CIsep = zeros(numVol, 2);
for vi = 1 : numVol
    testInd = intersect(naive(vi).indTuned, expert(vi).indTuned);
    paNaive = naive(vi).poleBeforeAnswer(testInd,:); % population activity from Naive session
    paExpert = expert(vi).poleBeforeAnswer(testInd,:); % population activity from Expert session
    [~,pcaNaive] = pca(paNaive');
    CIsep(vi,1) = clustering_index(pcaNaive(:,1:3), naive(vi).trialAngle);
    [~,pcaExpert] = pca(paExpert');
    CIsep(vi,2) = clustering_index(pcaExpert(:,1:3), expert(vi).trialAngle);
    [~,pcaComb] = pca([paNaive, paExpert]');
    CIcomb(vi,1) = clustering_index(pcaComb(1:size(paNaive,2),1:3), naive(vi).trialAngle);
    CIcomb(vi,2) = clustering_index(pcaComb(size(paNaive,2)+1:end,1:3), expert(vi).trialAngle);
end

figure, hold on
offset = 0.1;
errorbar(1-offset, mean(CIsep(:,1)), sem(CIsep(:,1)), 'ko')
errorbar(1+offset, mean(CIsep(:,2)), sem(CIsep(:,2)), 'ro')
legend({'Naive', 'Expert'}, 'autoupdate', false)
errorbar(2-offset, mean(CIcomb(:,1)), sem(CIcomb(:,1)), 'ko')
errorbar(2+offset, mean(CIcomb(:,2)), sem(CIcomb(:,2)), 'ro')
xticks(1:2)
xticklabels({'Separate PCA', 'Combined PCA'})
ylabel('Clustering index')
ylim([0 0.3])

%% Result
% Well, it doesn't matter, and in both cases clustering increases!

% Compared with previous results (the one that clustering index does not
% increase), naive values are exactly the same. Expert clustering indices
% went down at this previous analysis. What is going on??

% It was a variable name problem....


%% TEsting with copying the code, except for shuffling.
load('matchedPopResponse_201230')

popActDataGroup = cell(numVol,2); % (:,1) naive, (:,2) expert
pcaDataGroup = cell(numVol,2);
clustIndDataGroup = cell(numVol,2);
crDataGroup = cell(numVol,2);
eaDataGroup = cell(numVol,2);
for vi = 1 : numVol
    fprintf('Processing volume #%d/%d\n', vi, numVol);
    % Take persistently angle-tuned neurons only
    patInds = intersect(naive(vi).indTuned, expert(vi).indTuned);
    for si = 1 : 2
        if si == 1
            sessionAngles = naive(vi).trialAngle;
            popActData = naive(vi).poleBeforeAnswer(patInds,:);
            disp('Naive session')
        else
            sessionAngles = expert(vi).trialAngle;
            popActData = expert(vi).poleBeforeAnswer(patInds,:);
            disp('Expert session')
        end
        % Run PCA
        [~, pcaData] = pca(popActData');

        % Clustering Index
        clustIndData = clustering_index(pcaData(:,1:3), sessionAngles);

        %% Classification
        % Takes about 10 min for 10-fold, 1000 shuffling WITHOUT regularization
        % With regularization, it will take i-don't-know-how-long
        numFold = 10;
        trainFrac = 0.7;

        crData = zeros(numFold,4); % cr: Correct rate. (1,1) LDA, (1,2) SVM, (1,3) KNN, (1,4) Random Forest
        eaData = zeros(numFold,4); % ea: Error angle. (1,1) LDA, (1,2) SVM, (1,3) KNN, (1,4) Random Forest
        for fi = 1 : numFold
            fprintf('Processing fold #%d/%d\n', fi, numFold)
            % Stratify training data
            trainInd = [];
            for ai = 1 : length(angles)
                angleInd = find(sessionAngles == angles(ai));
                trainInd = [trainInd; angleInd(randperm(length(angleInd), round(length(angleInd)*trainFrac)))];
            end
            testInd = setdiff(1:length(sessionAngles), trainInd);

            % Classify data
            trainX = popActData(:,trainInd)';
            trainY = sessionAngles(trainInd);
            testX = popActData(:,testInd)';
            testY = sessionAngles(testInd);

            % LDA
        %     mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'discriminant', 'OptimizeHyperparameters', {'Delta', 'Gamma'}, 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
            mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'discriminant');
            label = predict(mdl, testX);
            crData(fi,1) = length(find((label-testY)==0)) / length(testY);
            eaData(fi,1) = mean(abs(label-testY));
            % SVM
        %     mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'svm', 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
            mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'svm');
            label = predict(mdl, testX);
            crData(fi,2) = length(find((label-testY)==0)) / length(testY);
            eaData(fi,2) = mean(abs(label-testY));
            % KNN
        %     mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'knn', 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
            mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'knn');
            label = predict(mdl, testX);
            crData(fi,3) = length(find((label-testY)==0)) / length(testY);
            eaData(fi,3) = mean(abs(label-testY));
            % Random Forest
            mdl = TreeBagger(500, trainX, trainY, 'Method', 'classification', 'Prior', 'Uniform');
            label = cellfun(@str2double, predict(mdl, testX));
            crData(fi,4) = length(find((label-testY)==0)) / length(testY);
            eaData(fi,4) = mean(abs(label-testY));

        end
       
        popActDataGroup{vi,si} = popActData;
        pcaDataGroup{vi,si} = pcaData;
        clustIndDataGroup{vi,si} = clustIndData;
        crDataGroup{vi,si} = crData;
        eaDataGroup{vi,si} = eaData;
    end
end


% Clustering index
data = cell2mat(clustIndDataGroup);

figure, hold on
errorbar([1,2], mean(data), sem(data), 'ko', 'lines', 'no')
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
[~,p,m] = paired_test(data(:,1), data(:,2));
ylabel('Clustering index'), ylim([0 0.8])
title(sprintf('Naive VS Expert: p = %.3f (%s)', p, m))

%% Error angle
data = cell2mat(eaDataGroup);

figure, hold on
errorbar([1,2], mean(data), sem(data), 'ko', 'lines', 'no')
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
[~,p,m] = paired_test(data(:,1), data(:,2));
ylabel('Error angle'), ylim([0 0.8])
title(sprintf('Naive VS Expert: p = %.3f (%s)', p, m))





%% First, confirm that this is real.
% How? Shuffle activities in multiple different methods.
% Check activity shuffling.




%% (1) Repeat previous method. 
% Confirm activity afterwards (individual neuron level, individual trial
% level)

clear
cd('C:\Users\shires\Dropbox\Works\Projects\2020 Neural stretching in S1\Data')
numVol = 11;
angles = 45:15:135;

%% Look at each individual imaged volume
% Takes a little more than 24 hr
% Look at matched ANGLE-TUNED neurons first.
% And then test with all persistently active neurons.
% And then with all active neurons.
load('matchedPopResponse_201230')

popActDataGroup = cell(numVol,2); % (:,1) naive, (:,2) expert
popActShuffAngleGroup = cell(numVol,2); % shuffling within same angle trials
popActShuffAllGroup = cell(numVol,2); % shuffling from all trials regardless of angles

pcaDataGroup = cell(numVol,2);
pcaShuffAngleGroup = cell(numVol,2);
pcaShuffAllGroup = cell(numVol,2);

loadingDataGroup = cell(numVol,2);
loadingShuffAngleGroup = cell(numVol,2);
loadingShuffAllGroup = cell(numVol,2);

veDataGroup = cell(numVol, 2); % variance explained
veShuffAngleGroup = cell(numVol, 2);
veShuffAllGroup = cell(numVol, 2);

clustIndDataGroup = cell(numVol,2);
clustIndShuffAngleGroup = cell(numVol,2);
clustIndShuffAllGroup = cell(numVol,2);

crDataGroup = cell(numVol,2);
crShuffAngleGroup = cell(numVol,2);
crShuffAllGroup = cell(numVol,2);

eaDataGroup = cell(numVol,2);
eaShuffAngleGroup = cell(numVol,2);
eaShuffAllGroup = cell(numVol,2);
for vi = 1 : numVol
    fprintf('Processing volume #%d/%d\n', vi, numVol);
% vi  = 1;
    % Take persistently angle-tuned neurons only
    patInd = intersect(naive(vi).indTuned, expert(vi).indTuned);
    for si = 1 : 2
        if si == 1
            sessionAngles = naive(vi).trialAngle;
            popActData = naive(vi).poleBeforeAnswer(patInd,:);
            disp('Naive session')
        else
            sessionAngles = expert(vi).trialAngle;
            popActData = expert(vi).poleBeforeAnswer(patInd,:);
            disp('Expert session')
        end
%         naiveAngles = naive(vi).trialAngle;
        % First, look at naive session only. 
        % Does clustering index increase?
        % How about classification from population activity?
        numShuff = 1000;
        
        numCell = size(popActData,1);
        popActShuffAngle = cell(numShuff,1);
        popActShuffAll = cell(numShuff,1);
        parfor shi = 1 : numShuff
            popActShuffAngle{shi} = zeros(size(popActData));
            popActShuffAll{shi} = zeros(size(popActData));
            for ci = 1 : numCell
                for ai = 1 : length(angles)
                    angle = angles(ai);
                    angleInd = find(sessionAngles == angle);
                    tempDataAngle = popActData(ci,angleInd);
                    popActShuffAngle{shi}(ci,angleInd) = tempDataAngle(randperm(length(tempDataAngle)));
                end
                tempDataAll = popActData(ci,:);
                popActShuffAll{shi}(ci,:) = tempDataAll(randperm(length(tempDataAll)));
            end
        end

        % Run PCA
        [loadingData, pcaData, ~, ~, veData] = pca(popActData');
        pcaShuffAngle = cell(numShuff,1);
        loadingShuffAngle = cell(numShuff,1);
        veShuffleAngle = cell(numShuff,1);
        pcaShuffAll = cell(numShuff,1);
        loadingShuffAll = cell(numShuff,1);
        veShuffleAll = cell(numShuff,1);
        parfor shi = 1 : numShuff
            [loadingShuffAngle{shi}, pcaShuffAngle{shi}, ~, ~, veShuffAngle{shi}] = pca(popActShuffAngle{shi}');
            [loadingShuffAll{shi}, pcaShuffAll{shi}, ~, ~, veShuffAll{shi}] = pca(popActShuffAll{shi}');
        end

        % Clustering Index
        clustIndData = clustering_index(pcaData(:,1:3), sessionAngles);
        clustIndShuffAngle = cell(numShuff,1);
        clustIndShuffAll = cell(numShuff,1);
        parfor shi = 1 : numShuff
            clustIndShuffAngle{shi} = clustering_index(pcaShuffAngle{shi}(:,1:3), sessionAngles);
            clustIndShuffAll{shi} = clustering_index(pcaShuffAll{shi}(:,1:3), sessionAngles);
        end

        %% Classification
        % Takes about 10 min for 10-fold, 1000 shuffling WITHOUT regularization
        % With regularization, it will take i-don't-know-how-long
        numFold = 10;
        trainFrac = 0.7;

        crData = zeros(numFold,4); % cr: Correct rate. (1,1) LDA, (1,2) SVM, (1,3) KNN, (1,4) Random Forest
        eaData = zeros(numFold,4); % ea: Error angle. (1,1) LDA, (1,2) SVM, (1,3) KNN, (1,4) Random Forest
        
        crShuffAngle = zeros(numShuff,numFold, 4);
        eaShuffAngle = zeros(numShuff,numFold, 4);
        
        crShuffAll = zeros(numShuff,numFold, 4);
        eaShuffAll = zeros(numShuff,numFold, 4);
        
        for fi = 1 : numFold
            fprintf('Processing fold #%d/%d\n', fi, numFold)
            % Stratify training data
            trainInd = [];
            for ai = 1 : length(angles)
                angleInd = find(sessionAngles == angles(ai));
                trainInd = [trainInd; angleInd(randperm(length(angleInd), round(length(angleInd)*trainFrac)))];
            end
            testInd = setdiff(1:length(sessionAngles), trainInd);

            % Classify data
            trainX = popActData(:,trainInd)';
            trainY = sessionAngles(trainInd);
            testX = popActData(:,testInd)';
            testY = sessionAngles(testInd);

            % LDA
        %     mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'discriminant', 'OptimizeHyperparameters', {'Delta', 'Gamma'}, 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
            mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'discriminant');
            label = predict(mdl, testX);
            crData(fi,1) = length(find((label-testY)==0)) / length(testY);
            eaData(fi,1) = mean(abs(label-testY));
            % SVM
        %     mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'svm', 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
            mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'svm');
            label = predict(mdl, testX);
            crData(fi,2) = length(find((label-testY)==0)) / length(testY);
            eaData(fi,2) = mean(abs(label-testY));
            % KNN
        %     mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'knn', 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
            mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'knn');
            label = predict(mdl, testX);
            crData(fi,3) = length(find((label-testY)==0)) / length(testY);
            eaData(fi,3) = mean(abs(label-testY));
            % Random Forest
            mdl = TreeBagger(500, trainX, trainY, 'Method', 'classification', 'Prior', 'Uniform');
            label = cellfun(@str2double, predict(mdl, testX));
            crData(fi,4) = length(find((label-testY)==0)) / length(testY);
            eaData(fi,4) = mean(abs(label-testY));

            % Classify within-angle shuffled data
            % Indices are going to be the same
            % Y's are going to be the same (trainY and testY)
            shLDAcr = zeros(numShuff,1);
            shLDAea = zeros(numShuff,1);
            shSVMcr = zeros(numShuff,1);
            shSVMea = zeros(numShuff,1);
            shKNNcr = zeros(numShuff,1);
            shKNNea = zeros(numShuff,1);
            shRFcr = zeros(numShuff,1);
            shRFea = zeros(numShuff,1);
            parfor shi = 1 : numShuff
                trainX = popActShuffAngle{shi}(:,trainInd)';
                testX = popActShuffAngle{shi}(:,testInd)';

                % LDA
        %         mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'discriminant', 'OptimizeHyperparameters', {'Delta', 'Gamma'}, 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
                mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'discriminant');
                label = predict(mdl, testX);
                shLDAcr(shi) = length(find((label-testY)==0)) / length(testY);
                shLDAea(shi) = mean(abs(label-testY));
                % SVM
        %         mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'svm', 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
                mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'svm');
                label = predict(mdl, testX);
                shSVMcr(shi) = length(find((label-testY)==0)) / length(testY);
                shSVMea(shi) = mean(abs(label-testY));
                % KNN
        %         mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'knn', 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
                mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'knn');
                label = predict(mdl, testX);
                shKNNcr(shi) = length(find((label-testY)==0)) / length(testY);
                shKNNea(shi) = mean(abs(label-testY));
                % Random Forest
                mdl = TreeBagger(500, trainX, trainY, 'Method', 'classification', 'Prior', 'Uniform');
                label = cellfun(@str2double, predict(mdl, testX));
                shRFcr(shi) = length(find((label-testY)==0)) / length(testY);
                shRFea(shi) = mean(abs(label-testY));
            end
            crShuffAngle(:,fi,1) = shLDAcr;
            crShuffAngle(:,fi,2) = shSVMcr;
            crShuffAngle(:,fi,3) = shKNNcr;
            crShuffAngle(:,fi,4) = shRFcr;

            eaShuffAngle(:,fi,1) = shLDAea;
            eaShuffAngle(:,fi,2) = shSVMea;
            eaShuffAngle(:,fi,3) = shKNNea;
            eaShuffAngle(:,fi,4) = shRFea;
            
            % Classify all-shuffled data
            % Indices are going to be the same
            % Y's are going to be the same (trainY and testY)
            shLDAcr = zeros(numShuff,1);
            shLDAea = zeros(numShuff,1);
            shSVMcr = zeros(numShuff,1);
            shSVMea = zeros(numShuff,1);
            shKNNcr = zeros(numShuff,1);
            shKNNea = zeros(numShuff,1);
            shRFcr = zeros(numShuff,1);
            shRFea = zeros(numShuff,1);
            parfor shi = 1 : numShuff
                trainX = popActShuffAll{shi}(:,trainInd)';
                testX = popActShuffAll{shi}(:,testInd)';

                % LDA
        %         mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'discriminant', 'OptimizeHyperparameters', {'Delta', 'Gamma'}, 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
                mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'discriminant');
                label = predict(mdl, testX);
                shLDAcr(shi) = length(find((label-testY)==0)) / length(testY);
                shLDAea(shi) = mean(abs(label-testY));
                % SVM
        %         mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'svm', 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
                mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'svm');
                label = predict(mdl, testX);
                shSVMcr(shi) = length(find((label-testY)==0)) / length(testY);
                shSVMea(shi) = mean(abs(label-testY));
                % KNN
        %         mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'knn', 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
                mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'knn');
                label = predict(mdl, testX);
                shKNNcr(shi) = length(find((label-testY)==0)) / length(testY);
                shKNNea(shi) = mean(abs(label-testY));
                % Random Forest
                mdl = TreeBagger(500, trainX, trainY, 'Method', 'classification', 'Prior', 'Uniform');
                label = cellfun(@str2double, predict(mdl, testX));
                shRFcr(shi) = length(find((label-testY)==0)) / length(testY);
                shRFea(shi) = mean(abs(label-testY));
            end
            crShuffAll(:,fi,1) = shLDAcr;
            crShuffAll(:,fi,2) = shSVMcr;
            crShuffAll(:,fi,3) = shKNNcr;
            crShuffAll(:,fi,4) = shRFcr;

            eaShuffAll(:,fi,1) = shLDAea;
            eaShuffAll(:,fi,2) = shSVMea;
            eaShuffAll(:,fi,3) = shKNNea;
            eaShuffAll(:,fi,4) = shRFea;

        end
        % mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'discriminant', 'OptimizeHyperparameters', {'Delta', 'Gamma'}, 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
        % mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'svm', 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
        % mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'knn', 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
        % mdl = TreeBagger(500, trainX, trainY, 'Method', 'classification', 'Prior', 'Uniform');
        % label = predict(mdl, testX);
        % foldCR(fi) = length(find((label-testY)==0)) / length(testY);
        % foldEA(fi) = mean(abs(label-testY));
        
        popActDataGroup{vi,si} = popActData;
        popActShuffAngleGroup{vi,si} = popActShuffAngle;
        popActShuffAllGroup{vi,si} = popActShuffAll;
        
        pcaDataGroup{vi,si} = pcaData;
        pcaShuffAngleGroup{vi,si} = pcaShuffAngle;
        pcaShuffAllGroup{vi,si} = pcaShuffAll;
        
        loadingDataGroup{vi,si} = loadingData;
        loadingShuffAngleGroup{vi,si} = loadingShuffAngle;
        loadingShuffAllGroup{vi,si} = loadingShuffAll;
        
        veDataGroup{vi,si} = veData;
        veShuffAngleGroup{vi,si} = veShuffAngle;
        veShuffAllGroup{vi,si} = veShuffAll;
        
        
        clustIndDataGroup{vi,si} = clustIndData;
        clustIndShuffAngleGroup{vi,si} = clustIndShuffAngle;
        clustIndShuffAllGroup{vi,si} = clustIndShuffAll;
        
        crDataGroup{vi,si} = crData;
        crShuffAngleGroup{vi,si} = crShuffAngle;
        crShuffAllGroup{vi,si} = crShuffAll;
        
        eaDataGroup{vi,si} = eaData;
        eaShuffAngleGroup{vi,si} = eaShuffAngle;
        eaShuffAllGroup{vi,si} = eaShuffAll;
    end
end

save('matchedTunedPopActShuffle_210122','*Group')


 
%% From persistently active neurons
% Takes a little more than 24 hr
load('matchedPopResponse_201230')

popActDataGroup = cell(numVol,2); % (:,1) naive, (:,2) expert
popActShuffAngleGroup = cell(numVol,2); % shuffling within same angle trials
popActShuffAllGroup = cell(numVol,2); % shuffling from all trials regardless of angles

pcaDataGroup = cell(numVol,2);
pcaShuffAngleGroup = cell(numVol,2);
pcaShuffAllGroup = cell(numVol,2);

loadingDataGroup = cell(numVol,2);
loadingShuffAngleGroup = cell(numVol,2);
loadingShuffAllGroup = cell(numVol,2);

veDataGroup = cell(numVol, 2); % variance explained
veShuffAngleGroup = cell(numVol, 2);
veShuffAllGroup = cell(numVol, 2);

clustIndDataGroup = cell(numVol,2);
clustIndShuffAngleGroup = cell(numVol,2);
clustIndShuffAllGroup = cell(numVol,2);

crDataGroup = cell(numVol,2);
crShuffAngleGroup = cell(numVol,2);
crShuffAllGroup = cell(numVol,2);

eaDataGroup = cell(numVol,2);
eaShuffAngleGroup = cell(numVol,2);
eaShuffAllGroup = cell(numVol,2);
for vi = 1 : numVol
    fprintf('Processing volume #%d/%d\n', vi, numVol);
    % Take persistently active neurons only (all matched neurons)
    paInd = setdiff(naive(vi).allInd, union(naive(vi).indSilent, expert(vi).indSilent));
    for si = 1 : 2
        if si == 1
            sessionAngles = naive(vi).trialAngle;
            popActData = naive(vi).poleBeforeAnswer(paInd,:);
            disp('Naive session')
        else
            sessionAngles = expert(vi).trialAngle;
            popActData = expert(vi).poleBeforeAnswer(paInd,:);
            disp('Expert session')
        end
%         naiveAngles = naive(vi).trialAngle;
        % First, look at naive session only. 
        % Does clustering index increase?
        % How about classification from population activity?
        numShuff = 1000;
        
        numCell = size(popActData,1);
        popActShuffAngle = cell(numShuff,1);
        popActShuffAll = cell(numShuff,1);
        parfor shi = 1 : numShuff
            popActShuffAngle{shi} = zeros(size(popActData));
            popActShuffAll{shi} = zeros(size(popActData));
            for ci = 1 : numCell
                for ai = 1 : length(angles)
                    angle = angles(ai);
                    angleInd = find(sessionAngles == angle);
                    tempDataAngle = popActData(ci,angleInd);
                    popActShuffAngle{shi}(ci,angleInd) = tempDataAngle(randperm(length(tempDataAngle)));
                end
                tempDataAll = popActData(ci,:);
                popActShuffAll{shi}(ci,:) = tempDataAll(randperm(length(tempDataAll)));
            end
        end

        % Run PCA
        [loadingData, pcaData, ~, ~, veData] = pca(popActData');
        pcaShuffAngle = cell(numShuff,1);
        loadingShuffAngle = cell(numShuff,1);
        veShuffleAngle = cell(numShuff,1);
        pcaShuffAll = cell(numShuff,1);
        loadingShuffAll = cell(numShuff,1);
        veShuffleAll = cell(numShuff,1);
        parfor shi = 1 : numShuff
            [loadingShuffAngle{shi}, pcaShuffAngle{shi}, ~, ~, veShuffAngle{shi}] = pca(popActShuffAngle{shi}');
            [loadingShuffAll{shi}, pcaShuffAll{shi}, ~, ~, veShuffAll{shi}] = pca(popActShuffAll{shi}');
        end

        % Clustering Index
        clustIndData = clustering_index(pcaData(:,1:3), sessionAngles);
        clustIndShuffAngle = cell(numShuff,1);
        clustIndShuffAll = cell(numShuff,1);
        parfor shi = 1 : numShuff
            clustIndShuffAngle{shi} = clustering_index(pcaShuffAngle{shi}(:,1:3), sessionAngles);
            clustIndShuffAll{shi} = clustering_index(pcaShuffAll{shi}(:,1:3), sessionAngles);
        end

        %% Classification
        % Takes about 10 min for 10-fold, 1000 shuffling WITHOUT regularization
        % With regularization, it will take i-don't-know-how-long
        numFold = 10;
        trainFrac = 0.7;

        crData = zeros(numFold,4); % cr: Correct rate. (1,1) LDA, (1,2) SVM, (1,3) KNN, (1,4) Random Forest
        eaData = zeros(numFold,4); % ea: Error angle. (1,1) LDA, (1,2) SVM, (1,3) KNN, (1,4) Random Forest
        
        crShuffAngle = zeros(numShuff,numFold, 4);
        eaShuffAngle = zeros(numShuff,numFold, 4);
        
        crShuffAll = zeros(numShuff,numFold, 4);
        eaShuffAll = zeros(numShuff,numFold, 4);
        
        for fi = 1 : numFold
            fprintf('Processing fold #%d/%d\n', fi, numFold)
            % Stratify training data
            trainInd = [];
            for ai = 1 : length(angles)
                angleInd = find(sessionAngles == angles(ai));
                trainInd = [trainInd; angleInd(randperm(length(angleInd), round(length(angleInd)*trainFrac)))];
            end
            testInd = setdiff(1:length(sessionAngles), trainInd);

            % Classify data
            trainX = popActData(:,trainInd)';
            trainY = sessionAngles(trainInd);
            testX = popActData(:,testInd)';
            testY = sessionAngles(testInd);

            % LDA
        %     mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'discriminant', 'OptimizeHyperparameters', {'Delta', 'Gamma'}, 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
            mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'discriminant');
            label = predict(mdl, testX);
            crData(fi,1) = length(find((label-testY)==0)) / length(testY);
            eaData(fi,1) = mean(abs(label-testY));
            % SVM
        %     mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'svm', 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
            mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'svm');
            label = predict(mdl, testX);
            crData(fi,2) = length(find((label-testY)==0)) / length(testY);
            eaData(fi,2) = mean(abs(label-testY));
            % KNN
        %     mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'knn', 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
            mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'knn');
            label = predict(mdl, testX);
            crData(fi,3) = length(find((label-testY)==0)) / length(testY);
            eaData(fi,3) = mean(abs(label-testY));
            % Random Forest
            mdl = TreeBagger(500, trainX, trainY, 'Method', 'classification', 'Prior', 'Uniform');
            label = cellfun(@str2double, predict(mdl, testX));
            crData(fi,4) = length(find((label-testY)==0)) / length(testY);
            eaData(fi,4) = mean(abs(label-testY));

            % Classify within-angle shuffled data
            % Indices are going to be the same
            % Y's are going to be the same (trainY and testY)
            shLDAcr = zeros(numShuff,1);
            shLDAea = zeros(numShuff,1);
            shSVMcr = zeros(numShuff,1);
            shSVMea = zeros(numShuff,1);
            shKNNcr = zeros(numShuff,1);
            shKNNea = zeros(numShuff,1);
            shRFcr = zeros(numShuff,1);
            shRFea = zeros(numShuff,1);
            parfor shi = 1 : numShuff
                trainX = popActShuffAngle{shi}(:,trainInd)';
                testX = popActShuffAngle{shi}(:,testInd)';

                % LDA
        %         mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'discriminant', 'OptimizeHyperparameters', {'Delta', 'Gamma'}, 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
                mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'discriminant');
                label = predict(mdl, testX);
                shLDAcr(shi) = length(find((label-testY)==0)) / length(testY);
                shLDAea(shi) = mean(abs(label-testY));
                % SVM
        %         mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'svm', 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
                mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'svm');
                label = predict(mdl, testX);
                shSVMcr(shi) = length(find((label-testY)==0)) / length(testY);
                shSVMea(shi) = mean(abs(label-testY));
                % KNN
        %         mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'knn', 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
                mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'knn');
                label = predict(mdl, testX);
                shKNNcr(shi) = length(find((label-testY)==0)) / length(testY);
                shKNNea(shi) = mean(abs(label-testY));
                % Random Forest
                mdl = TreeBagger(500, trainX, trainY, 'Method', 'classification', 'Prior', 'Uniform');
                label = cellfun(@str2double, predict(mdl, testX));
                shRFcr(shi) = length(find((label-testY)==0)) / length(testY);
                shRFea(shi) = mean(abs(label-testY));
            end
            crShuffAngle(:,fi,1) = shLDAcr;
            crShuffAngle(:,fi,2) = shSVMcr;
            crShuffAngle(:,fi,3) = shKNNcr;
            crShuffAngle(:,fi,4) = shRFcr;

            eaShuffAngle(:,fi,1) = shLDAea;
            eaShuffAngle(:,fi,2) = shSVMea;
            eaShuffAngle(:,fi,3) = shKNNea;
            eaShuffAngle(:,fi,4) = shRFea;
            
            % Classify all-shuffled data
            % Indices are going to be the same
            % Y's are going to be the same (trainY and testY)
            shLDAcr = zeros(numShuff,1);
            shLDAea = zeros(numShuff,1);
            shSVMcr = zeros(numShuff,1);
            shSVMea = zeros(numShuff,1);
            shKNNcr = zeros(numShuff,1);
            shKNNea = zeros(numShuff,1);
            shRFcr = zeros(numShuff,1);
            shRFea = zeros(numShuff,1);
            parfor shi = 1 : numShuff
                trainX = popActShuffAll{shi}(:,trainInd)';
                testX = popActShuffAll{shi}(:,testInd)';

                % LDA
        %         mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'discriminant', 'OptimizeHyperparameters', {'Delta', 'Gamma'}, 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
                mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'discriminant');
                label = predict(mdl, testX);
                shLDAcr(shi) = length(find((label-testY)==0)) / length(testY);
                shLDAea(shi) = mean(abs(label-testY));
                % SVM
        %         mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'svm', 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
                mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'svm');
                label = predict(mdl, testX);
                shSVMcr(shi) = length(find((label-testY)==0)) / length(testY);
                shSVMea(shi) = mean(abs(label-testY));
                % KNN
        %         mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'knn', 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
                mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'knn');
                label = predict(mdl, testX);
                shKNNcr(shi) = length(find((label-testY)==0)) / length(testY);
                shKNNea(shi) = mean(abs(label-testY));
                % Random Forest
                mdl = TreeBagger(500, trainX, trainY, 'Method', 'classification', 'Prior', 'Uniform');
                label = cellfun(@str2double, predict(mdl, testX));
                shRFcr(shi) = length(find((label-testY)==0)) / length(testY);
                shRFea(shi) = mean(abs(label-testY));
            end
            crShuffAll(:,fi,1) = shLDAcr;
            crShuffAll(:,fi,2) = shSVMcr;
            crShuffAll(:,fi,3) = shKNNcr;
            crShuffAll(:,fi,4) = shRFcr;

            eaShuffAll(:,fi,1) = shLDAea;
            eaShuffAll(:,fi,2) = shSVMea;
            eaShuffAll(:,fi,3) = shKNNea;
            eaShuffAll(:,fi,4) = shRFea;

        end
        % mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'discriminant', 'OptimizeHyperparameters', {'Delta', 'Gamma'}, 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
        % mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'svm', 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
        % mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'knn', 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
        % mdl = TreeBagger(500, trainX, trainY, 'Method', 'classification', 'Prior', 'Uniform');
        % label = predict(mdl, testX);
        % foldCR(fi) = length(find((label-testY)==0)) / length(testY);
        % foldEA(fi) = mean(abs(label-testY));
        
        popActDataGroup{vi,si} = popActData;
        popActShuffAngleGroup{vi,si} = popActShuffAngle;
        popActShuffAllGroup{vi,si} = popActShuffAll;
        
        pcaDataGroup{vi,si} = pcaData;
        pcaShuffAngleGroup{vi,si} = pcaShuffAngle;
        pcaShuffAllGroup{vi,si} = pcaShuffAll;
        
        loadingDataGroup{vi,si} = loadingData;
        loadingShuffAngleGroup{vi,si} = loadingShuffAngle;
        loadingShuffAllGroup{vi,si} = loadingShuffAll;
        
        veDataGroup{vi,si} = veData;
        veShuffAngleGroup{vi,si} = veShuffAngle;
        veShuffAllGroup{vi,si} = veShuffAll;
        
        
        clustIndDataGroup{vi,si} = clustIndData;
        clustIndShuffAngleGroup{vi,si} = clustIndShuffAngle;
        clustIndShuffAllGroup{vi,si} = clustIndShuffAll;
        
        crDataGroup{vi,si} = crData;
        crShuffAngleGroup{vi,si} = crShuffAngle;
        crShuffAllGroup{vi,si} = crShuffAll;
        
        eaDataGroup{vi,si} = eaData;
        eaShuffAngleGroup{vi,si} = eaShuffAngle;
        eaShuffAllGroup{vi,si} = eaShuffAll;
    end
end
save('matchedPersPopActShuffle_210122','*Group')



%% From all neurons
% Takes a little more than 24 hr
load('matchedPopResponse_201230')

popActDataGroup = cell(numVol,2); % (:,1) naive, (:,2) expert
popActShuffAngleGroup = cell(numVol,2); % shuffling within same angle trials
popActShuffAllGroup = cell(numVol,2); % shuffling from all trials regardless of angles

pcaDataGroup = cell(numVol,2);
pcaShuffAngleGroup = cell(numVol,2);
pcaShuffAllGroup = cell(numVol,2);

loadingDataGroup = cell(numVol,2);
loadingShuffAngleGroup = cell(numVol,2);
loadingShuffAllGroup = cell(numVol,2);

veDataGroup = cell(numVol, 2); % variance explained
veShuffAngleGroup = cell(numVol, 2);
veShuffAllGroup = cell(numVol, 2);

clustIndDataGroup = cell(numVol,2);
clustIndShuffAngleGroup = cell(numVol,2);
clustIndShuffAllGroup = cell(numVol,2);

crDataGroup = cell(numVol,2);
crShuffAngleGroup = cell(numVol,2);
crShuffAllGroup = cell(numVol,2);

eaDataGroup = cell(numVol,2);
eaShuffAngleGroup = cell(numVol,2);
eaShuffAllGroup = cell(numVol,2);
for vi = 1 : numVol
    fprintf('Processing volume #%d/%d\n', vi, numVol);
    % Take all neurons
    for si = 1 : 2
        if si == 1
            sessionAngles = naive(vi).trialAngle;
            popActData = naive(vi).poleBeforeAnswer;
            disp('Naive session')
        else
            sessionAngles = expert(vi).trialAngle;
            popActData = expert(vi).poleBeforeAnswer;
            disp('Expert session')
        end
%         naiveAngles = naive(vi).trialAngle;
        % First, look at naive session only. 
        % Does clustering index increase?
        % How about classification from population activity?
        numShuff = 1000;
        
        numCell = size(popActData,1);
        popActShuffAngle = cell(numShuff,1);
        popActShuffAll = cell(numShuff,1);
        parfor shi = 1 : numShuff
            popActShuffAngle{shi} = zeros(size(popActData));
            popActShuffAll{shi} = zeros(size(popActData));
            for ci = 1 : numCell
                for ai = 1 : length(angles)
                    angle = angles(ai);
                    angleInd = find(sessionAngles == angle);
                    tempDataAngle = popActData(ci,angleInd);
                    popActShuffAngle{shi}(ci,angleInd) = tempDataAngle(randperm(length(tempDataAngle)));
                end
                tempDataAll = popActData(ci,:);
                popActShuffAll{shi}(ci,:) = tempDataAll(randperm(length(tempDataAll)));
            end
        end

        % Run PCA
        [loadingData, pcaData, ~, ~, veData] = pca(popActData');
        pcaShuffAngle = cell(numShuff,1);
        loadingShuffAngle = cell(numShuff,1);
        veShuffleAngle = cell(numShuff,1);
        pcaShuffAll = cell(numShuff,1);
        loadingShuffAll = cell(numShuff,1);
        veShuffleAll = cell(numShuff,1);
        parfor shi = 1 : numShuff
            [loadingShuffAngle{shi}, pcaShuffAngle{shi}, ~, ~, veShuffAngle{shi}] = pca(popActShuffAngle{shi}');
            [loadingShuffAll{shi}, pcaShuffAll{shi}, ~, ~, veShuffAll{shi}] = pca(popActShuffAll{shi}');
        end

        % Clustering Index
        clustIndData = clustering_index(pcaData(:,1:3), sessionAngles);
        clustIndShuffAngle = cell(numShuff,1);
        clustIndShuffAll = cell(numShuff,1);
        parfor shi = 1 : numShuff
            clustIndShuffAngle{shi} = clustering_index(pcaShuffAngle{shi}(:,1:3), sessionAngles);
            clustIndShuffAll{shi} = clustering_index(pcaShuffAll{shi}(:,1:3), sessionAngles);
        end

        %% Classification
        % Takes about 10 min for 10-fold, 1000 shuffling WITHOUT regularization
        % With regularization, it will take i-don't-know-how-long
        numFold = 10;
        trainFrac = 0.7;

        crData = zeros(numFold,4); % cr: Correct rate. (1,1) LDA, (1,2) SVM, (1,3) KNN, (1,4) Random Forest
        eaData = zeros(numFold,4); % ea: Error angle. (1,1) LDA, (1,2) SVM, (1,3) KNN, (1,4) Random Forest
        
        crShuffAngle = zeros(numShuff,numFold, 4);
        eaShuffAngle = zeros(numShuff,numFold, 4);
        
        crShuffAll = zeros(numShuff,numFold, 4);
        eaShuffAll = zeros(numShuff,numFold, 4);
        
        for fi = 1 : numFold
            fprintf('Processing fold #%d/%d\n', fi, numFold)
            % Stratify training data
            trainInd = [];
            for ai = 1 : length(angles)
                angleInd = find(sessionAngles == angles(ai));
                trainInd = [trainInd; angleInd(randperm(length(angleInd), round(length(angleInd)*trainFrac)))];
            end
            testInd = setdiff(1:length(sessionAngles), trainInd);

            % Classify data
            trainX = popActData(:,trainInd)';
            trainY = sessionAngles(trainInd);
            testX = popActData(:,testInd)';
            testY = sessionAngles(testInd);

            % LDA
        %     mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'discriminant', 'OptimizeHyperparameters', {'Delta', 'Gamma'}, 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
            mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'discriminant');
            label = predict(mdl, testX);
            crData(fi,1) = length(find((label-testY)==0)) / length(testY);
            eaData(fi,1) = mean(abs(label-testY));
            % SVM
        %     mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'svm', 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
            mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'svm');
            label = predict(mdl, testX);
            crData(fi,2) = length(find((label-testY)==0)) / length(testY);
            eaData(fi,2) = mean(abs(label-testY));
            % KNN
        %     mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'knn', 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
            mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'knn');
            label = predict(mdl, testX);
            crData(fi,3) = length(find((label-testY)==0)) / length(testY);
            eaData(fi,3) = mean(abs(label-testY));
            % Random Forest
            mdl = TreeBagger(500, trainX, trainY, 'Method', 'classification', 'Prior', 'Uniform');
            label = cellfun(@str2double, predict(mdl, testX));
            crData(fi,4) = length(find((label-testY)==0)) / length(testY);
            eaData(fi,4) = mean(abs(label-testY));

            % Classify within-angle shuffled data
            % Indices are going to be the same
            % Y's are going to be the same (trainY and testY)
            shLDAcr = zeros(numShuff,1);
            shLDAea = zeros(numShuff,1);
            shSVMcr = zeros(numShuff,1);
            shSVMea = zeros(numShuff,1);
            shKNNcr = zeros(numShuff,1);
            shKNNea = zeros(numShuff,1);
            shRFcr = zeros(numShuff,1);
            shRFea = zeros(numShuff,1);
            parfor shi = 1 : numShuff
                trainX = popActShuffAngle{shi}(:,trainInd)';
                testX = popActShuffAngle{shi}(:,testInd)';

                % LDA
        %         mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'discriminant', 'OptimizeHyperparameters', {'Delta', 'Gamma'}, 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
                mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'discriminant');
                label = predict(mdl, testX);
                shLDAcr(shi) = length(find((label-testY)==0)) / length(testY);
                shLDAea(shi) = mean(abs(label-testY));
                % SVM
        %         mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'svm', 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
                mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'svm');
                label = predict(mdl, testX);
                shSVMcr(shi) = length(find((label-testY)==0)) / length(testY);
                shSVMea(shi) = mean(abs(label-testY));
                % KNN
        %         mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'knn', 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
                mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'knn');
                label = predict(mdl, testX);
                shKNNcr(shi) = length(find((label-testY)==0)) / length(testY);
                shKNNea(shi) = mean(abs(label-testY));
                % Random Forest
                mdl = TreeBagger(500, trainX, trainY, 'Method', 'classification', 'Prior', 'Uniform');
                label = cellfun(@str2double, predict(mdl, testX));
                shRFcr(shi) = length(find((label-testY)==0)) / length(testY);
                shRFea(shi) = mean(abs(label-testY));
            end
            crShuffAngle(:,fi,1) = shLDAcr;
            crShuffAngle(:,fi,2) = shSVMcr;
            crShuffAngle(:,fi,3) = shKNNcr;
            crShuffAngle(:,fi,4) = shRFcr;

            eaShuffAngle(:,fi,1) = shLDAea;
            eaShuffAngle(:,fi,2) = shSVMea;
            eaShuffAngle(:,fi,3) = shKNNea;
            eaShuffAngle(:,fi,4) = shRFea;
            
            % Classify all-shuffled data
            % Indices are going to be the same
            % Y's are going to be the same (trainY and testY)
            shLDAcr = zeros(numShuff,1);
            shLDAea = zeros(numShuff,1);
            shSVMcr = zeros(numShuff,1);
            shSVMea = zeros(numShuff,1);
            shKNNcr = zeros(numShuff,1);
            shKNNea = zeros(numShuff,1);
            shRFcr = zeros(numShuff,1);
            shRFea = zeros(numShuff,1);
            parfor shi = 1 : numShuff
                trainX = popActShuffAll{shi}(:,trainInd)';
                testX = popActShuffAll{shi}(:,testInd)';

                % LDA
        %         mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'discriminant', 'OptimizeHyperparameters', {'Delta', 'Gamma'}, 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
                mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'discriminant');
                label = predict(mdl, testX);
                shLDAcr(shi) = length(find((label-testY)==0)) / length(testY);
                shLDAea(shi) = mean(abs(label-testY));
                % SVM
        %         mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'svm', 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
                mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'svm');
                label = predict(mdl, testX);
                shSVMcr(shi) = length(find((label-testY)==0)) / length(testY);
                shSVMea(shi) = mean(abs(label-testY));
                % KNN
        %         mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'knn', 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
                mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'knn');
                label = predict(mdl, testX);
                shKNNcr(shi) = length(find((label-testY)==0)) / length(testY);
                shKNNea(shi) = mean(abs(label-testY));
                % Random Forest
                mdl = TreeBagger(500, trainX, trainY, 'Method', 'classification', 'Prior', 'Uniform');
                label = cellfun(@str2double, predict(mdl, testX));
                shRFcr(shi) = length(find((label-testY)==0)) / length(testY);
                shRFea(shi) = mean(abs(label-testY));
            end
            crShuffAll(:,fi,1) = shLDAcr;
            crShuffAll(:,fi,2) = shSVMcr;
            crShuffAll(:,fi,3) = shKNNcr;
            crShuffAll(:,fi,4) = shRFcr;

            eaShuffAll(:,fi,1) = shLDAea;
            eaShuffAll(:,fi,2) = shSVMea;
            eaShuffAll(:,fi,3) = shKNNea;
            eaShuffAll(:,fi,4) = shRFea;

        end
        % mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'discriminant', 'OptimizeHyperparameters', {'Delta', 'Gamma'}, 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
        % mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'svm', 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
        % mdl = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'knn', 'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));
        % mdl = TreeBagger(500, trainX, trainY, 'Method', 'classification', 'Prior', 'Uniform');
        % label = predict(mdl, testX);
        % foldCR(fi) = length(find((label-testY)==0)) / length(testY);
        % foldEA(fi) = mean(abs(label-testY));
        
        popActDataGroup{vi,si} = popActData;
        popActShuffAngleGroup{vi,si} = popActShuffAngle;
        popActShuffAllGroup{vi,si} = popActShuffAll;
        
        pcaDataGroup{vi,si} = pcaData;
        pcaShuffAngleGroup{vi,si} = pcaShuffAngle;
        pcaShuffAllGroup{vi,si} = pcaShuffAll;
        
        loadingDataGroup{vi,si} = loadingData;
        loadingShuffAngleGroup{vi,si} = loadingShuffAngle;
        loadingShuffAllGroup{vi,si} = loadingShuffAll;
        
        veDataGroup{vi,si} = veData;
        veShuffAngleGroup{vi,si} = veShuffAngle;
        veShuffAllGroup{vi,si} = veShuffAll;
        
        
        clustIndDataGroup{vi,si} = clustIndData;
        clustIndShuffAngleGroup{vi,si} = clustIndShuffAngle;
        clustIndShuffAllGroup{vi,si} = clustIndShuffAll;
        
        crDataGroup{vi,si} = crData;
        crShuffAngleGroup{vi,si} = crShuffAngle;
        crShuffAllGroup{vi,si} = crShuffAll;
        
        eaDataGroup{vi,si} = eaData;
        eaShuffAngleGroup{vi,si} = eaShuffAngle;
        eaShuffAllGroup{vi,si} = eaShuffAll;
    end
end

save('matchedAllPopActShuffle_210122','*Group')



%% First of all, look at classifier performance
load('matchedTunedPopActShuffle_210122','clustInd*Group', 'cr*Group', 'ea*Group')
% load('matchedPersPopActShuffle_210122','clustInd*Group', 'cr*Group', 'ea*Group')
% load('matchedAllPopActShuffle_210122','clustInd*Group', 'cr*Group', 'ea*Group')
% Compare between naive and expert, compare between data and shuffle

%% Clustering index
shuffAngle = cellfun(@(x) mean(cell2mat(x)), clustIndShuffAngleGroup);
shuffAll = cellfun(@(x) mean(cell2mat(x)), clustIndShuffAllGroup);
data = cell2mat(clustIndDataGroup);

figure, hold on
errorbar([1,2], mean(data), sem(data), 'ko', 'lines', 'no')
errorbar([1,2], mean(shuffAngle), sem(shuffAngle), 'ro', 'lines', 'no')
errorbar([1,2], mean(shuffAll), sem(shuffAll), 'o', 'color', [0.6 0.6 0.6], 'lines', 'no')
legend({'Data', 'Shuffle-Angle', 'Shuffle-All'}, 'location', 'northwest')
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
[~,p1,m1] = paired_test(shuffAngle(:,1), data(:,1));
[~,p2,m2] = paired_test(shuffAngle(:,2), data(:,2));
[~,p,m] = paired_test(data(:,1), data(:,2));
ylabel('Clustering index'), ylim([0 0.5]), yticks(0:0.1:0.5)

title({sprintf('Naive VS Expert: p = %.3f (%s)', p, m); sprintf('Naive data VS shuffle: p = %.3f (%s)', p1, m1); ...
    sprintf('Expert data VS shuffle: p = %.3f (%s)', p2, m2)})


%% Correct rate
dataGroup = cellfun(@mean, crDataGroup, 'un', 0);
shuffAngleGroup = cellfun(@(x) mean(mean(x)), crShuffAngleGroup, 'un', 0);
shuffAllGroup = cellfun(@(x) mean(mean(x)), crShuffAllGroup, 'un', 0);

methods = {'LDA', 'SVM', 'KNN', 'Random forest'};
figure('unit', 'inch', 'pos', [0 0 13 7])   
for spi = 1 : 4
    subplot(1,4,spi), hold on
    
    data = cellfun(@(x) x(spi), dataGroup);
    shuffAngle = cellfun(@(x) x(spi), shuffAngleGroup);
    shuffAll = cellfun(@(x) x(spi), shuffAllGroup);
    errorbar([1,2], mean(shuffAngle), sem(shuffAngle), 'ro', 'lines', 'no')
    errorbar([1,2], mean(shuffAll), sem(shuffAll), 'o', 'color', [0.6 0.6 0.6], 'lines', 'no')
    errorbar([1,2], mean(data), sem(data), 'ko', 'lines', 'no')
    xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
    [~,p1,m1] = paired_test(shuffAngle(:,1), data(:,1));
    [~,p2,m2] = paired_test(shuffAngle(:,2), data(:,2));
    [~,p,m] = paired_test(data(:,1), data(:,2));
    [~,ps,ms] = paired_test(shuffAngle(:,1), shuffAngle(:,2));
    ylabel('Correct rate'), ylim([0 0.8])
    
    title({methods{spi}; sprintf('Naive VS Expert: p = %.3f (%s)', p, m); sprintf('Naive data VS shuffle: p = %.3f (%s)', p1, m1); ...
        sprintf('Expert data VS shuffle: p = %.3f (%s)', p2, m2); 'Naive shuffle VS expert shuffle:'; sprintf( 'p = %.3f (%s)', ps, ms)})
end


%%

saveDir = 'C:\Users\jinho\Dropbox\Works\grant proposal\2021 NARSAD\';
fn = 'classfiers_shuffled_persTuned.eps';
export_fig([saveDir, fn], '-depsc', '-painters', '-r600', '-transparent')
fix_eps_fonts([saveDir, fn])


%% Just LDA
dataGroup = cellfun(@(x) mean(x(:,1)), crDataGroup);
dataShuff = cellfun(@(x) mean(x(:,1)), crShuffAllGroup);
figure, hold on
for vi = 1 : 11
    plot([1,2], dataGroup(vi,:), 'ko-')
end
errorbar([1,2], mean(dataGroup), sem(dataGroup), 'ro', 'lines', 'no')
errorbar([1,2], mean(dataShuff), sem(dataShuff), 'o', 'color', [0.6 0.6 0.6], 'lines', 'no')
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
[~,p,m] = paired_test(dataGroup(:,1), dataGroup(:,2));
ylabel('Correct Rate'), ylim([0 0.7])
title(sprintf('Naive VS Expert: p = %.3f (%s)', p, m))


%%

saveDir = 'C:\Users\jinho\Dropbox\Works\grant proposal\2021 NARSAD\';
fn = 'LDA_persTuned.eps';
export_fig([saveDir, fn], '-depsc', '-painters', '-r600', '-transparent')
fix_eps_fonts([saveDir, fn])







%%
% Error angle
shuffAngleGroup = cellfun(@(x) mean(mean(x)), eaShuffAngleGroup, 'un', 0);
shuffAllGroup = cellfun(@(x) mean(mean(x)), eaShuffAllGroup, 'un', 0);
dataGroup = cellfun(@mean, eaDataGroup, 'un', 0);
methods = {'LDA', 'SVM', 'KNN', 'Random forest'};
figure('unit', 'inch', 'pos', [0 0 13 7])   
for spi = 1 : 4
    subplot(1,4,spi), hold on
    shuffAngle = cellfun(@(x) x(spi), shuffAngleGroup);
    shuffAll = cellfun(@(x) x(spi), shuffAllGroup);
    data = cellfun(@(x) x(spi), dataGroup);
    errorbar([1,2], mean(shuffAngle), sem(shuffAngle), 'ro', 'lines', 'no')
    errorbar([1,2], mean(shuffAll), sem(shuffAll), 'o', 'color', [0.6 0.6 0.6], 'lines', 'no')
    errorbar([1,2], mean(data), sem(data), 'ko', 'lines', 'no')
    xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'}), xtickangle(45)
    [~,p1,m1] = paired_test(shuffAngle(:,1), data(:,1));
    [~,p2,m2] = paired_test(shuffAngle(:,2), data(:,2));
    [~,p,m] = paired_test(data(:,1), data(:,2));
    ylabel('Error angle'), ylim([0 40])
    
    title({methods{spi}; sprintf('Naive VS Expert: p = %.3f (%s)', p, m); sprintf('Naive data VS shuffle: p = %.3f (%s)', p1, m1); ...
        sprintf('Expert data VS shuffle: p = %.3f (%s)', p2, m2)})
end



%% How do loadings look like?
clear
load('matchedTunedPopActShuffle_210122','*Group')
numVol = 11;
angles = 45:15:135;

%% Distribution of loadings
vi = 1;
si = 2;
loading = loadingDataGroup{vi,si};

absLoading = abs(loading);
numBin = 20;

sigPrct = 80;
sigInd = find(absLoading(:,1) > prctile(absLoading(:,1),sigPrct));

figure('unit', 'inch', 'pos', [1 1 3.5 7]), 
for pcai = 1 : 3
    histRange = linspace(min(absLoading(:,pcai)), max(absLoading(:,pcai)), numBin+1);
    if pcai == 1
        allDist = histcounts(absLoading(:,pcai), histRange, 'norm', 'count');
        sigDist = histcounts(absLoading(sigInd,pcai), histRange, 'norm', 'count');
        subplot(3,1,pcai), hold on
        bar(histRange(2:end), allDist)
        bar(histRange(2:end), sigDist)
        legend({'All', sprintf('Top %d percentile',sigPrct)}, 'box', 'off')
        title(sprintf('|Loading| from PC#%d',pcai))
        ylabel('Counts')
    else
        allDist = histcounts(absLoading(:,pcai), histRange, 'norm', 'cdf');
        sigDist = histcounts(absLoading(sigInd,pcai), histRange, 'norm', 'cdf');
        subplot(3,1,pcai), hold on
        p1 = plot(histRange(2:end), allDist);
        p2 = plot(histRange(2:end), sigDist);
        p3 = plot(histRange(2:end), allDist - sigDist, 'color', [0.6 0.6 0.6]);
        title(sprintf('|Loading| distribution on PC#%d',pcai))
        ylabel('Cumulative proportion')
        ylim([0 1])
    end
    
    if pcai == 2
        legend(p3, sprintf('All - Top %d percentile', sigPrct), 'box', 'off')
    end
    if pcai == 3
        xlabel('|Loading|')
    end
end
sgtitle(sprintf('Volume #%d, session #%d', vi, si))


%% From all volumes
allDist2 = zeros(numVol, numBin, 2); % (:,:,1) Naive, (:,:,2) Expert
allDist3 = zeros(numVol, numBin, 2);
sigDist2 = zeros(numVol, numBin, 2);
sigDist3 = zeros(numVol, numBin, 2);

maxVal = max(max(cellfun(@(x) max(max(abs(x(:,1:3)))), loadingDataGroup)));
histRange = linspace(0, maxVal, numBin+1);
plotRange = histRange(2:end);
for vi = 1 : numVol
    for si = 1 : 2
        loading = loadingDataGroup{vi,si};
        absLoading = abs(loading);
        sigInd = find(absLoading(:,1) > prctile(absLoading(:,1),sigPrct));
        
        allDist2(vi,:,si) = histcounts(absLoading(:,2), histRange, 'norm', 'cdf');
        sigDist2(vi,:,si) = histcounts(absLoading(sigInd,2), histRange, 'norm', 'cdf');
        allDist3(vi,:,si) = histcounts(absLoading(:,3), histRange, 'norm', 'cdf');
        sigDist3(vi,:,si) = histcounts(absLoading(sigInd,3), histRange, 'norm', 'cdf');
        
    end
end

%
figure, 
defaultColors = get(gca,'colororder');

subplot(2,2,1), hold on
tempAll = squeeze(allDist2(:,:,1));
tempTop = squeeze(sigDist2(:,:,1));
tempDiff = tempAll - tempTop;
plot(plotRange, mean(tempAll), 'color', defaultColors(1,:))
plot(plotRange, mean(tempTop), 'color', defaultColors(2,:))
plot(plotRange, mean(tempDiff), 'color', [0.6 0.6 0.6])
legend({'All', sprintf('Top %d percentile',sigPrct), sprintf('All - top %d percentile', sigPrct)}, 'box', 'off', 'autoupdate', 0)
boundedline(plotRange, mean(tempAll), sem(tempAll), 'cmap', defaultColors(1,:))
boundedline(plotRange, mean(tempTop), sem(tempTop), 'cmap', defaultColors(2,:))
boundedline(plotRange, mean(tempDiff), sem(tempDiff), 'cmap', [0.6 0.6 0.6])
title('|Loading| on PC2 (Naive)')
ylabel('Cumulative proportion')

subplot(2,2,3), hold on
tempAll = squeeze(allDist3(:,:,1));
tempTop = squeeze(sigDist3(:,:,1));
tempDiff = tempAll - tempTop;
boundedline(plotRange, mean(tempAll), sem(tempAll), 'cmap', defaultColors(1,:))
boundedline(plotRange, mean(tempTop), sem(tempTop), 'cmap', defaultColors(2,:))
boundedline(plotRange, mean(tempDiff), sem(tempDiff), 'cmap', [0.6 0.6 0.6])
title('|Loading| on PC3 (Naive)')
ylabel('Cumulative proportion')
xlabel('|Loading|')


subplot(2,2,2), hold on
tempAll = squeeze(allDist2(:,:,2));
tempTop = squeeze(sigDist2(:,:,2));
tempDiff = tempAll - tempTop;
boundedline(plotRange, mean(tempAll), sem(tempAll), 'cmap', defaultColors(1,:))
boundedline(plotRange, mean(tempTop), sem(tempTop), 'cmap', defaultColors(2,:))
boundedline(plotRange, mean(tempDiff), sem(tempDiff), 'cmap', [0.6 0.6 0.6])
title('|Loading| on PC2 (Expert)')

subplot(2,2,4), hold on
tempAll = squeeze(allDist3(:,:,2));
tempTop = squeeze(sigDist3(:,:,2));
tempDiff = tempAll - tempTop;
boundedline(plotRange, mean(tempAll), sem(tempAll), 'cmap', defaultColors(1,:))
boundedline(plotRange, mean(tempTop), sem(tempTop), 'cmap', defaultColors(2,:))
boundedline(plotRange, mean(tempDiff), sem(tempDiff), 'cmap', [0.6 0.6 0.6])
title('|Loading| on PC3 (Expert)')
xlabel('|Loading|')


%% Neurons with Top 20% PC1 loading does not have distinct loading on other PCs


%% what is correlated with PC loading?
% angle selectivity (as), event rate (er)
clear
load('matchedPopResponse_201230')
load('matchedTunedPopActShuffle_210122','loadingDataGroup')
numVol = length(naive);

angleSelectivity = cell(numVol,2);
eventRate = cell(numVol,2);
for vi = 1 : numVol
    % Take persistently angle-tuned neurons only
    patInd = intersect(naive(vi).indTuned, expert(vi).indTuned); % sorted
    patID = naive(vi).allID(patInd); % sorted
    for si = 1 : 2
        if si == 1
            angleSelectivity{vi,si} = naive(vi).angleSelectivity(find(ismember(naive(vi).indTuned, patInd)));
            eventRate{vi,si} = naive(vi).eventRate(find(ismember(naive(vi).allID, patID)));
        else
            angleSelectivity{vi,si} = expert(vi).angleSelectivity(find(ismember(expert(vi).indTuned, patInd)));
            eventRate{vi,si} = expert(vi).eventRate(find(ismember(expert(vi).allID, patID)));
        end
    end
end

%%
asCorr = zeros(numVol,3,2);
erCorr = zeros(numVol,3,2);
for vi = 1 : numVol
    for si = 1 : 2
        for pcai = 1 : 3
            asCorr(vi,pcai,si) = abs(corr(loadingDataGroup{vi,si}(:,pcai), angleSelectivity{vi,si}));
            erCorr(vi,pcai,si) = abs(corr(loadingDataGroup{vi,si}(:,pcai), eventRate{vi,si}));
        end
    end
end

figure
for i = 1 : 3
    subplot(2,3,i), hold on
    for vi = 1 : 11
        plot([1,2], [asCorr(vi,i,1), asCorr(vi,i,2)], 'k-')
    end
    for si = 1 : 2
        errorbar(si, mean(asCorr(:,i,si)), sem(asCorr(:,i,si)), 'ro')
    end
    
    if i == 1
        ylabel('|Correlation| with angle selectivity')
    end
    xticks(1:2), xticklabels(''), xlim([0.5 2.5])
    ylim([0 1])
    [~,p,m] = paired_test(squeeze(asCorr(:,i,1)), squeeze(asCorr(:,i,2)));
    title({sprintf('Loading PC%d',i); sprintf('p = %.3f (%s)',p,m)})
    
    subplot(2,3,3+i), hold on
    for vi = 1 : 11
        plot([1,2], [erCorr(vi,i,1), erCorr(vi,i,2)], 'k-')
    end
    for si = 1 : 2
        errorbar(si, mean(erCorr(:,i,si)), sem(erCorr(:,i,si)), 'ro')
    end
    if i == 1
        ylabel('|Correlation| with event rate')
    end
    xticks(1:2), xticklabels({'Naive', 'Expert'}), xlim([0.5 2.5])
    ylim([0 1])
    [~,p,m] = paired_test(squeeze(erCorr(:,i,1)), squeeze(erCorr(:,i,2)));
    title(sprintf('p = %.3f (%s)',p,m))
end


%%
%% In naïve, loading in PC1 is correlated with angle selectivity and event rate.
%% In expert, these correlations decrease, and the correlation with angle selectivity is abolished.
% Are angle selectivity and event rate correlated in this subset of neurons?

aserCorr = zeros(numVol,2);
for vi = 1 : numVol
    for si = 1 : 2
        aserCorr(vi,si) = corr(angleSelectivity{vi,si}, eventRate{vi,si});
    end
end

figure, hold on
for vi = 1 : numVol
    plot(aserCorr(vi,:), 'k-')
end
errorbar(1, mean(aserCorr(:,1)), sem(aserCorr(:,1)), 'ro')
errorbar(2, mean(aserCorr(:,2)), sem(aserCorr(:,2)), 'ro')
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'})
[~,p,m] = paired_test(aserCorr(:,1), aserCorr(:,2));
title(sprintf('p = %.3f (%s)',p,m))
ylabel('Correlation'), ylim([0 1])

%% Confirm from separate population activity data

load('popResponse_201228.mat')
load('cellID_match_persAT_v9')

angleSelectivity2 = cell(11,2);
eventRate2 = cell(11,2);
for vi = 1:11
    if vi < 3
        mi = 1;
    elseif vi == 3
        mi = 2;
    else
        mi = floor(vi/2)+1;
    end
    
    IDPersTunedNaive = intersect(match{mi,1}(find(match{mi,2})), naive(vi).allID(naive(vi).indTuned));
    IDPersTunedExpert = intersect(match{mi,2}, expert(vi).allID(expert(vi).indTuned));
    
    indPersTunedNaive = find(ismember(match{mi,1}, IDPersTunedNaive));
    IDptnExpert = match{mi,2}(indPersTunedNaive);
    
    indPersTunedExpert = find(ismember(match{mi,2}, IDPersTunedExpert));
    IDpteNaive = match{mi,1}(indPersTunedExpert);
    
    
    finalNaiveTunedInd = find(ismember(naive(vi).allID(naive(vi).indTuned), IDpteNaive));
    finalExpertTunedInd = find(ismember(expert(vi).allID(expert(vi).indTuned), IDptnExpert));
    
    angleSelectivity2{vi,1} = naive(vi).angleSelectivity(finalNaiveTunedInd);
    angleSelectivity2{vi,2} = expert(vi).angleSelectivity(finalExpertTunedInd);
    
    
    finalNaiveInd = find(ismember(naive(vi).allID, intersect(IDpteNaive, IDPersTunedNaive)));
    finalExpertInd = find(ismember(expert(vi).allID, intersect(IDptnExpert, IDPersTunedExpert)));
    
    eventRate2{vi,1} = naive(vi).eventRate(finalNaiveInd);
    eventRate2{vi,2} = expert(vi).eventRate(finalExpertInd);
end

aserCorr = zeros(11,2);
for vi = 1 : 11
    for si = 1 : 2
        aserCorr(vi,si) = corr(angleSelectivity2{vi,si}, eventRate2{vi,si});
    end
end

figure, hold on
for vi = 1 : 11
    plot(aserCorr(vi,:), 'k-')
end
errorbar(1, mean(aserCorr(:,1)), sem(aserCorr(:,1)), 'ro')
errorbar(2, mean(aserCorr(:,2)), sem(aserCorr(:,2)), 'ro')
xlim([0.5 2.5]), xticks(1:2), xticklabels({'Naive', 'Expert'})
[~,p,m] = paired_test(aserCorr(:,1), aserCorr(:,2));
title(sprintf('p = %.3f (%s)',p,m))
ylabel('Correlation')





%% How does trial-averaged activity correlation between neurons look like?
% Covariance matrix, after sorting neurons to tuned angle
clear
load('matchedPopResponse_201230')
load('matchedTunedPopActShuffle_210122','popAct*Group')
angles = 45:15:135;
%%
for vi = 1 : 11
    patInd = intersect(naive(vi).indTuned, expert(vi).indTuned);
    naiveInd = find(ismember(naive(vi).indTuned, expert(vi).indTuned));
    expertInd = find(ismember(expert(vi).indTuned, naive(vi).indTuned));

    tunedAngle = naive(vi).tunedAngle(naiveInd);
    [taNaive,ind] = sort(tunedAngle);
    popAct = popActDataGroup{vi,1}(ind,:);
    naiveCov = cov(popAct');

    numShuff = length(popActShuffAngleGroup{vi,1});
    naiveShuffCovAll = zeros([size(naiveCov),numShuff]);
    for shi = 1 : numShuff
        naiveShuffCovAll(:,:,shi) = cov(popActShuffAngleGroup{vi,1}{shi}(ind,:)');
    end
    naiveShuffCov = mean(naiveShuffCovAll,3);

    tunedAngle = expert(vi).tunedAngle(expertInd);
    [taExpert,ind] = sort(tunedAngle);
    popAct = popActDataGroup{vi,2}(ind,:);
    expertCov = cov(popAct');

    expertShuffCovAll = zeros([size(expertCov),numShuff]);
    for shi = 1 : numShuff
        expertShuffCovAll(:,:,shi) = cov(popActShuffAngleGroup{vi,2}{shi}(ind,:)');
    end
    expertShuffCov = mean(expertShuffCovAll,3);

    % naive
    figure,
    subplot(221), imagesc(naiveCov), axis square, colorbar, hold on

    num45 = length(find(taNaive==45));
    start90 = length(find(taNaive < 90));
    end90 = length(find(taNaive <= 90));
    num135 = length(find(taNaive==135));
    maxnum = size(naiveCov,1);
    plot([0,maxnum+1], [num45+0.5, num45+0.5], 'w--')
    plot([num45+0.5, num45+0.5], [0,maxnum+1], 'w--')

    plot([0,maxnum+1], [start90+0.5, start90+0.5], 'w--')
    plot([start90+0.5, start90+0.5], [0,maxnum+1], 'w--')

    plot([0,maxnum+1], [end90+0.5, end90+0.5], 'w--')
    plot([end90+0.5, end90+0.5], [0,maxnum+1], 'w--')

    plot([0,maxnum+1], [maxnum-num135+0.5, maxnum-num135+0.5], 'w--')
    plot([maxnum-num135+0.5, maxnum-num135+0.5], [0,maxnum+1], 'w--')

    ticks = [num45/2, start90+(end90-start90)/2, maxnum-num135+num135/2];
    yticks(ticks)
    yticklabels({'45', '90', '135'})
    xticks(ticks)
    xticklabels({'45', '90', '135'})

    title('Naive Data')

    % naive shuffle
    subplot(223), imagesc(naiveShuffCov), axis square, colorbar, hold on

    plot([0,maxnum+1], [num45+0.5, num45+0.5], 'w--')
    plot([num45+0.5, num45+0.5], [0,maxnum+1], 'w--')

    plot([0,maxnum+1], [start90+0.5, start90+0.5], 'w--')
    plot([start90+0.5, start90+0.5], [0,maxnum+1], 'w--')

    plot([0,maxnum+1], [end90+0.5, end90+0.5], 'w--')
    plot([end90+0.5, end90+0.5], [0,maxnum+1], 'w--')

    plot([0,maxnum+1], [maxnum-num135+0.5, maxnum-num135+0.5], 'w--')
    plot([maxnum-num135+0.5, maxnum-num135+0.5], [0,maxnum+1], 'w--')

    ticks = [num45/2, start90+(end90-start90)/2, maxnum-num135+num135/2];
    yticks(ticks)
    yticklabels({'45', '90', '135'})
    xticks(ticks)
    xticklabels({'45', '90', '135'})

    title('Naive Shuffle')

    % expert
    subplot(222), imagesc(expertCov), axis square, colorbar, hold on

    num45 = length(find(taExpert==45));
    start90 = length(find(taExpert < 90));
    end90 = length(find(taExpert <= 90));
    num135 = length(find(taExpert==135));
    maxnum = size(expertCov,1);
    plot([0,maxnum+1], [num45+0.5, num45+0.5], 'w--')
    plot([num45+0.5, num45+0.5], [0,maxnum+1], 'w--')

    plot([0,maxnum+1], [start90+0.5, start90+0.5], 'w--')
    plot([start90+0.5, start90+0.5], [0,maxnum+1], 'w--')

    plot([0,maxnum+1], [end90+0.5, end90+0.5], 'w--')
    plot([end90+0.5, end90+0.5], [0,maxnum+1], 'w--')

    plot([0,maxnum+1], [maxnum-num135+0.5, maxnum-num135+0.5], 'w--')
    plot([maxnum-num135+0.5, maxnum-num135+0.5], [0,maxnum+1], 'w--')

    ticks = [num45/2, start90+(end90-start90)/2, maxnum-num135+num135/2];
    yticks(ticks)
    yticklabels({'45', '90', '135'})
    xticks(ticks)
    xticklabels({'45', '90', '135'})

    title('Expert Data')

    % expert shuffle
    subplot(224), imagesc(expertShuffCov), axis square, colorbar, hold on

    plot([0,maxnum+1], [num45+0.5, num45+0.5], 'w--')
    plot([num45+0.5, num45+0.5], [0,maxnum+1], 'w--')

    plot([0,maxnum+1], [start90+0.5, start90+0.5], 'w--')
    plot([start90+0.5, start90+0.5], [0,maxnum+1], 'w--')

    plot([0,maxnum+1], [end90+0.5, end90+0.5], 'w--')
    plot([end90+0.5, end90+0.5], [0,maxnum+1], 'w--')

    plot([0,maxnum+1], [maxnum-num135+0.5, maxnum-num135+0.5], 'w--')
    plot([maxnum-num135+0.5, maxnum-num135+0.5], [0,maxnum+1], 'w--')

    ticks = [num45/2, start90+(end90-start90)/2, maxnum-num135+num135/2];
    yticks(ticks)
    yticklabels({'45', '90', '135'})
    xticks(ticks)
    xticklabels({'45', '90', '135'})

    title('Expert Shuffle')

    sgtitle(sprintf('Volume #%02d', vi))
end



%%
% for vi = 1 : 11
for vi = 1
    patInd = intersect(naive(vi).indTuned, expert(vi).indTuned);
    naiveInd = find(ismember(naive(vi).indTuned, expert(vi).indTuned));
    expertInd = find(ismember(expert(vi).indTuned, naive(vi).indTuned));

    tunedAngle = naive(vi).tunedAngle(naiveInd);
    [taNaive,ind] = sort(tunedAngle);
    popAct = popActDataGroup{vi,1}(ind,:);
    naiveCorr = make_diag_nan(corrcoef(popAct'));

    numShuff = length(popActShuffAngleGroup{vi,1});
    naiveAngleShuffCorrAll = zeros([size(naiveCorr),numShuff]);
    naiveAllShuffCorrAll = zeros([size(naiveCorr),numShuff]);
    for shi = 1 : numShuff
        naiveAngleShuffCorrAll(:,:,shi) = corrcoef(popActShuffAngleGroup{vi,1}{shi}(ind,:)');
        naiveAllShuffCorrAll(:,:,shi) = corrcoef(popActShuffAllGroup{vi,1}{shi}(ind,:)');
    end
    naiveAngleShuffCorr = make_diag_nan(mean(naiveAngleShuffCorrAll,3));
    naiveAllShuffCorr = make_diag_nan(mean(naiveAllShuffCorrAll,3));

    tunedAngle = expert(vi).tunedAngle(expertInd);
    [taExpert,ind] = sort(tunedAngle);
    popAct = popActDataGroup{vi,2}(ind,:);
    expertCorr = make_diag_nan(corrcoef(popAct'));

    expertAngleShuffCorrAll = zeros([size(expertCorr),numShuff]);
    expertAllShuffCorrAll = zeros([size(expertCorr),numShuff]);
    for shi = 1 : numShuff
        expertAngleShuffCorrAll(:,:,shi) = corrcoef(popActShuffAngleGroup{vi,2}{shi}(ind,:)');
        expertAllShuffCorrAll(:,:,shi) = corrcoef(popActShuffAllGroup{vi,2}{shi}(ind,:)');
    end
    expertAngleShuffCorr = make_diag_nan(mean(expertAngleShuffCorrAll,3));
    expertAllShuffCorr = make_diag_nan(mean(expertAllShuffCorrAll,3));
    

    % naive
    figure('unit', 'inch','position',[2 3 12 6]),
    subplot(231), imagesc(naiveCorr), axis square, colorbar, hold on

    num45 = length(find(taNaive==45));
    start90 = length(find(taNaive < 90));
    end90 = length(find(taNaive <= 90));
    num135 = length(find(taNaive==135));
    maxnum = size(naiveCorr,1);
    plot([0,maxnum+1], [num45+0.5, num45+0.5], 'w--')
    plot([num45+0.5, num45+0.5], [0,maxnum+1], 'w--')

    plot([0,maxnum+1], [start90+0.5, start90+0.5], 'w--')
    plot([start90+0.5, start90+0.5], [0,maxnum+1], 'w--')

    plot([0,maxnum+1], [end90+0.5, end90+0.5], 'w--')
    plot([end90+0.5, end90+0.5], [0,maxnum+1], 'w--')

    plot([0,maxnum+1], [maxnum-num135+0.5, maxnum-num135+0.5], 'w--')
    plot([maxnum-num135+0.5, maxnum-num135+0.5], [0,maxnum+1], 'w--')

    ticks = [num45/2, start90+(end90-start90)/2, maxnum-num135+num135/2];
    yticks(ticks)
    yticklabels({'45', '90', '135'})
    xticks(ticks)
    xticklabels({'45', '90', '135'})

    title('Naive Data')

    % naive angle shuffle
    subplot(232), imagesc(naiveAngleShuffCorr), axis square, colorbar, hold on

    plot([0,maxnum+1], [num45+0.5, num45+0.5], 'w--')
    plot([num45+0.5, num45+0.5], [0,maxnum+1], 'w--')

    plot([0,maxnum+1], [start90+0.5, start90+0.5], 'w--')
    plot([start90+0.5, start90+0.5], [0,maxnum+1], 'w--')

    plot([0,maxnum+1], [end90+0.5, end90+0.5], 'w--')
    plot([end90+0.5, end90+0.5], [0,maxnum+1], 'w--')

    plot([0,maxnum+1], [maxnum-num135+0.5, maxnum-num135+0.5], 'w--')
    plot([maxnum-num135+0.5, maxnum-num135+0.5], [0,maxnum+1], 'w--')

    ticks = [num45/2, start90+(end90-start90)/2, maxnum-num135+num135/2];
    yticks(ticks)
    yticklabels({'45', '90', '135'})
    xticks(ticks)
    xticklabels({'45', '90', '135'})

    title('Naive Shuffle - Angle')
    
    % naive all shuffle
    subplot(233), imagesc(naiveAllShuffCorr), axis square, colorbar, hold on

    plot([0,maxnum+1], [num45+0.5, num45+0.5], 'w--')
    plot([num45+0.5, num45+0.5], [0,maxnum+1], 'w--')

    plot([0,maxnum+1], [start90+0.5, start90+0.5], 'w--')
    plot([start90+0.5, start90+0.5], [0,maxnum+1], 'w--')

    plot([0,maxnum+1], [end90+0.5, end90+0.5], 'w--')
    plot([end90+0.5, end90+0.5], [0,maxnum+1], 'w--')

    plot([0,maxnum+1], [maxnum-num135+0.5, maxnum-num135+0.5], 'w--')
    plot([maxnum-num135+0.5, maxnum-num135+0.5], [0,maxnum+1], 'w--')

    ticks = [num45/2, start90+(end90-start90)/2, maxnum-num135+num135/2];
    yticks(ticks)
    yticklabels({'45', '90', '135'})
    xticks(ticks)
    xticklabels({'45', '90', '135'})

    title('Naive Shuffle - All')


    % expert
    subplot(234), imagesc(expertCorr), axis square, colorbar, hold on

    num45 = length(find(taExpert==45));
    start90 = length(find(taExpert < 90));
    end90 = length(find(taExpert <= 90));
    num135 = length(find(taExpert==135));
    maxnum = size(expertCorr,1);
    plot([0,maxnum+1], [num45+0.5, num45+0.5], 'w--')
    plot([num45+0.5, num45+0.5], [0,maxnum+1], 'w--')

    plot([0,maxnum+1], [start90+0.5, start90+0.5], 'w--')
    plot([start90+0.5, start90+0.5], [0,maxnum+1], 'w--')

    plot([0,maxnum+1], [end90+0.5, end90+0.5], 'w--')
    plot([end90+0.5, end90+0.5], [0,maxnum+1], 'w--')

    plot([0,maxnum+1], [maxnum-num135+0.5, maxnum-num135+0.5], 'w--')
    plot([maxnum-num135+0.5, maxnum-num135+0.5], [0,maxnum+1], 'w--')

    ticks = [num45/2, start90+(end90-start90)/2, maxnum-num135+num135/2];
    yticks(ticks)
    yticklabels({'45', '90', '135'})
    xticks(ticks)
    xticklabels({'45', '90', '135'})

    title('Expert Data')

    % expert angle shuffle
    subplot(235), imagesc(expertAngleShuffCorr), axis square, colorbar, hold on

    plot([0,maxnum+1], [num45+0.5, num45+0.5], 'w--')
    plot([num45+0.5, num45+0.5], [0,maxnum+1], 'w--')

    plot([0,maxnum+1], [start90+0.5, start90+0.5], 'w--')
    plot([start90+0.5, start90+0.5], [0,maxnum+1], 'w--')

    plot([0,maxnum+1], [end90+0.5, end90+0.5], 'w--')
    plot([end90+0.5, end90+0.5], [0,maxnum+1], 'w--')

    plot([0,maxnum+1], [maxnum-num135+0.5, maxnum-num135+0.5], 'w--')
    plot([maxnum-num135+0.5, maxnum-num135+0.5], [0,maxnum+1], 'w--')

    ticks = [num45/2, start90+(end90-start90)/2, maxnum-num135+num135/2];
    yticks(ticks)
    yticklabels({'45', '90', '135'})
    xticks(ticks)
    xticklabels({'45', '90', '135'})

    title('Expert Shuffle - Angle')
    
    % expert all shuffle
    subplot(236), imagesc(expertAllShuffCorr), axis square, colorbar, hold on

    plot([0,maxnum+1], [num45+0.5, num45+0.5], 'w--')
    plot([num45+0.5, num45+0.5], [0,maxnum+1], 'w--')

    plot([0,maxnum+1], [start90+0.5, start90+0.5], 'w--')
    plot([start90+0.5, start90+0.5], [0,maxnum+1], 'w--')

    plot([0,maxnum+1], [end90+0.5, end90+0.5], 'w--')
    plot([end90+0.5, end90+0.5], [0,maxnum+1], 'w--')

    plot([0,maxnum+1], [maxnum-num135+0.5, maxnum-num135+0.5], 'w--')
    plot([maxnum-num135+0.5, maxnum-num135+0.5], [0,maxnum+1], 'w--')

    ticks = [num45/2, start90+(end90-start90)/2, maxnum-num135+num135/2];
    yticks(ticks)
    yticklabels({'45', '90', '135'})
    xticks(ticks)
    xticklabels({'45', '90', '135'})

    title('Expert Shuffle - All')
    
    
    

    sgtitle(sprintf('Volume #%02d', vi))
end




%% How does trial-averaged activity correlation between trials (population vectors) look like?
% for vi = 1 : 11
for vi = 7
    % Naive
    trialAngle = naive(vi).trialAngle;
    [taNaive,ind] = sort(trialAngle);
    popAct = popActDataGroup{vi,1}(:,ind);
    naiveCorr = make_diag_nan(corrcoef(popAct));

    numShuff = length(popActShuffAngleGroup{vi,1});
    naiveAngleShuffCorrAll = zeros([size(naiveCorr),numShuff]);
    naiveAllShuffCorrAll = zeros([size(naiveCorr),numShuff]);
    for shi = 1 : numShuff
        naiveAngleShuffCorrAll(:,:,shi) = corrcoef(popActShuffAngleGroup{vi,1}{shi}(:,ind));
        naiveAllShuffCorrAll(:,:,shi) = corrcoef(popActShuffAllGroup{vi,1}{shi}(:,ind));
    end
    naiveAngleShuffCorr = make_diag_nan(mean(naiveAngleShuffCorrAll,3));
    naiveAllShuffCorr = make_diag_nan(mean(naiveAllShuffCorrAll,3));
    
    % Expert
    trialAngle = expert(vi).trialAngle;
    [taExpert,ind] = sort(trialAngle);
    popAct = popActDataGroup{vi,2}(:,ind);
    expertCorr = make_diag_nan(corrcoef(popAct));

    numShuff = length(popActShuffAngleGroup{vi,2});
    expertAngleShuffCorrAll = zeros([size(expertCorr),numShuff]);
    expertAllShuffCorrAll = zeros([size(expertCorr),numShuff]);
    for shi = 1 : numShuff
        expertAngleShuffCorrAll(:,:,shi) = make_diag_nan(corrcoef(popActShuffAngleGroup{vi,2}{shi}(:,ind)));
        expertAllShuffCorrAll(:,:,shi) = make_diag_nan(corrcoef(popActShuffAllGroup{vi,2}{shi}(:,ind)));
    end
    expertAngleShuffCorr = mean(expertAngleShuffCorrAll,3);
    expertAllShuffCorr = mean(expertAllShuffCorrAll,3);

    % Plot
    % Naive data
    figure('unit','inch','position',[2 3 12*0.9 6*0.9]),
    subplot(231), imagesc(naiveCorr), axis square, colorbar, hold on

    maxnum = size(naiveCorr,1);
    dividers = zeros(length(angles),1);
    for ai = 1 : length(angles)
        dividers(ai) = length(find(taNaive <= angles(ai)));
    end
    
    for ai = 1 : length(angles)-1
        plot([0 maxnum+1], [dividers(ai)+0.5, dividers(ai)+0.5], 'w--')
        plot([dividers(ai)+0.5, dividers(ai)+0.5], [0 maxnum+1], 'w--')
    end
    ticks = [dividers(1)/2, dividers(3)+(dividers(4)-dividers(3))/2, maxnum-(dividers(7)-dividers(6))/2];
    yticks(ticks)
    yticklabels({'45', '90', '135'})
    xticks(ticks)
    xticklabels({'45', '90', '135'})

    title('Naive Data')

    % Naive shuffle - angle
    subplot(232), imagesc(naiveAngleShuffCorr), axis square, colorbar, hold on

    for ai = 1 : length(angles)-1
        plot([0 maxnum+1], [dividers(ai)+0.5, dividers(ai)+0.5], 'w--')
        plot([dividers(ai)+0.5, dividers(ai)+0.5], [0 maxnum+1], 'w--')
    end
    ticks = [dividers(1)/2, dividers(3)+(dividers(4)-dividers(3))/2, maxnum-(dividers(7)-dividers(6))/2];
    yticks(ticks)
    yticklabels({'45', '90', '135'})
    xticks(ticks)
    xticklabels({'45', '90', '135'})

    title('Naive Shuffle - Angle')
    
    % Naive shuffle - angle
    subplot(233), imagesc(naiveAllShuffCorr), axis square, colorbar, hold on

    for ai = 1 : length(angles)-1
        plot([0 maxnum+1], [dividers(ai)+0.5, dividers(ai)+0.5], 'w--')
        plot([dividers(ai)+0.5, dividers(ai)+0.5], [0 maxnum+1], 'w--')
    end
    ticks = [dividers(1)/2, dividers(3)+(dividers(4)-dividers(3))/2, maxnum-(dividers(7)-dividers(6))/2];
    yticks(ticks)
    yticklabels({'45', '90', '135'})
    xticks(ticks)
    xticklabels({'45', '90', '135'})

    title('Naive Shuffle - All')
    

    % Expert data
    subplot(234), imagesc(expertCorr), axis square, colorbar, hold on

    maxnum = size(expertCorr,1);
    dividers = zeros(length(angles),1);
    for ai = 1 : length(angles)
        dividers(ai) = length(find(taExpert <= angles(ai)));
    end
    
    for ai = 1 : length(angles)-1
        plot([0 maxnum+1], [dividers(ai)+0.5, dividers(ai)+0.5], 'w--')
        plot([dividers(ai)+0.5, dividers(ai)+0.5], [0 maxnum+1], 'w--')
    end
    ticks = [dividers(1)/2, dividers(3)+(dividers(4)-dividers(3))/2, maxnum-(dividers(7)-dividers(6))/2];
    yticks(ticks)
    yticklabels({'45', '90', '135'})
    xticks(ticks)
    xticklabels({'45', '90', '135'})

    title('Expert Data')

    % Expert shuffle - angle
    subplot(235), imagesc(expertAngleShuffCorr), axis square, colorbar, hold on

    for ai = 1 : length(angles)-1
        plot([0 maxnum+1], [dividers(ai)+0.5, dividers(ai)+0.5], 'w--')
        plot([dividers(ai)+0.5, dividers(ai)+0.5], [0 maxnum+1], 'w--')
    end
    ticks = [dividers(1)/2, dividers(3)+(dividers(4)-dividers(3))/2, maxnum-(dividers(7)-dividers(6))/2];
    yticks(ticks)
    yticklabels({'45', '90', '135'})
    xticks(ticks)
    xticklabels({'45', '90', '135'})

    title('Expert Shuffle - Angle')

    % Expert shuffle - all
    subplot(236), imagesc(expertAllShuffCorr), axis square, colorbar, hold on

    for ai = 1 : length(angles)-1
        plot([0 maxnum+1], [dividers(ai)+0.5, dividers(ai)+0.5], 'w--')
        plot([dividers(ai)+0.5, dividers(ai)+0.5], [0 maxnum+1], 'w--')
    end
    ticks = [dividers(1)/2, dividers(3)+(dividers(4)-dividers(3))/2, maxnum-(dividers(7)-dividers(6))/2];
    yticks(ticks)
    yticklabels({'45', '90', '135'})
    xticks(ticks)
    xticklabels({'45', '90', '135'})

    title('Expert Shuffle - All')

    
    
    sgtitle(sprintf('Volume #%02d', vi))
end





%% Classifier dimension (hyperplane)
clear
angles = 45:15:135;
trainFrac = 0.7;
% load('matchedTunedPopActShuffle_210122', 'popAct*Group')
load('matchedPopResponse_201230')
vi = 1;
sessionAngles = naive(vi).trialAngle;
popActData = naive(vi).poleBeforeAnswer;
trainInd = [];
for ai = 1 : length(angles)
    angleInd = find(sessionAngles == angles(ai));
    trainInd = [trainInd; angleInd(randperm(length(angleInd), round(length(angleInd)*trainFrac)))];
end
testInd = setdiff(1:length(sessionAngles), trainInd);

% Classify data
trainX = popActData(:,trainInd)';
trainY = sessionAngles(trainInd);
testX = popActData(:,testInd)';
testY = sessionAngles(testInd);

% LDA
lda = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', 'discriminant');

% SVM
t = templateSVM('SaveSupportVectors',true);
svm = fitcecoc(trainX, trainY, 'Prior', 'uniform', 'Learners', t);
