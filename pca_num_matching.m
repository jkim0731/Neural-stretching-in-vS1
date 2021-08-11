function [pcaCoordNaive, veNaive, pcaCoordExpert, veExpert] = pca_num_matching(popresNaive, popresExpert, angleNaive, angleExpert, numRepeat, numDim)
% Run PCA with matching the number of neurons (variables).
% Mainly for comparing between naive and expert sessions.
% When matching the number of neurons, consider clustering index (requires
% clustering_index function). Run 
% 2020/12/28 JK
%
% Inputs:
%     popresNaive: n x p data matrix, population response in the naive session
%     popresExpert: n x p data matrix, population response in the expert session
%     angleNaive: p x 1 vector, object angle matching to the naive session trials
%     angleExpert: p x 1 vector, object angle matching to the expert session trials
%     numRepeat: number of repeat for calculating median clustering index
%     numDim: number of PCA dimension to calculate the clutering index
% 
% Outputs:
%     pcaCoordNaive: pca coordinate for the naive session (score)
%     veNaive: variance explained for the naive session
%     pcaCoordNaive: pca coordinate for the expert session (score)
%     veExpert: variance explained for the expert session

numCellNaive = size(popresNaive,2);
numCellExpert = size(popresExpert,2);

if numCellNaive > numCellExpert
    [~, pcaCoordExpert,~,~,veExpert] = pca(popresExpert);

    tempCoord = cell(numRepeat,1);
    tempVE = cell(numRepeat,1);
    tempCI = zeros(numRepeat,1);
    for ri = 1 : numRepeat
        tempInd = randperm(numCellNaive, numCellExpert);
        [~,tempCoord{ri},~,~,tempVE{ri}] = pca(popresNaive(:,tempInd));
        tempCI(ri) = clustering_index(tempCoord{ri}(:,1:numDim), angleNaive);
    end
    medInd = find(tempCI == median(tempCI));
    pcaCoordNaive = tempCoord{medInd};
    veNaive = tempVE{medInd};
elseif numCellNaive < numCellExpert
    [~, pcaCoordNaive,~,~,veNaive] = pca(popresNaive);

    tempCoord = cell(numRepeat,1);
    tempVE = cell(numRepeat,1);
    tempCI = zeros(numRepeat,1);
    for ri = 1 : numRepeat
        tempInd = randperm(numCellExpert, numCellNaive);
        [~,tempCoord{ri},~,~,tempVE{ri}] = pca(popresExpert(:,tempInd));
        tempCI(ri) = clustering_index(tempCoord{ri}(:,1:numDim), angleExpert);
    end
    medInd = find(tempCI == median(tempCI));
    pcaCoordExpert = tempCoord{medInd};
    veExpert = tempVE{medInd};
else % numCellNaive == numCellExpert
    [~, pcaCoordNaive,~,~,veNaive] = pca(popresNaive);
    [~, pcaCoordExpert,~,~,veExpert] = pca(popresExpert);
end