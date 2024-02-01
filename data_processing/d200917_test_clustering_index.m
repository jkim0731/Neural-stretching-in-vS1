% Try comparing different methods of calculating clustering index
% Look at their relationship with NoC (# of components)
% 1. Euclidean distance
% 2. Euclidean distance rank from all pairs
% 3. Euclidean distance rank from each row
% 4. Manhattan distance
% 5. Manhattan distance rank
% 6. Manhattan distance rank from each row

% Using clustering_index function (updated on 2020/09/17)

% 2020/09/17 JK

%% load data
clear
baseDir = 'D:\TPM\JK\suite2p\';
loadFn = 'pca_from_whisker_model';
load([baseDir, loadFn]);

%% Look at the distribution of pair-wise distances (or ranks)
% To see what is the high-dimensional problem in calculating clustering index
%
% Compare between within and between distances
% From a well-separated session (JK030 upper after training; volume index = 4)
% Compare between 6 different methods
% Across different # of components (NoC)

volInd = 4;
mi = 3; vi = 1; % for angle 
numComp = 3;
data = pcaResult(volInd).pcaCoordSpkExpert(:,1:numComp);

distVal = zeros(size(data,1)*size(data,2),6); % (:,1-6) corresponds to 6 methods of distance calculation
edistAll = pdist2(data,data,'eucledian');
diagNan = 1-eye(size(data,1));
diagNan(find(diagNan==0)) = Nan;
edistAll = edistAll .* diagNan;

edistAll = triu(edistAll);
edistAll(find(edistAll==0)) = nan; % Nan lower triangle, including diagonals (because they are 0 in distance)
[~,allSorti] = sort(allDist(:),'ascend'); % Sort once, collect the indices
[~,allSortii] = sort(allSorti,'ascend'); % Sort the indices
distRankUp = reshape(allSortii, size(allDist)); % Upper triangular matrix of the distance rank
distRankUp(find(isnan(distRankUp))) = 0; 
edistRankAll = distRankUp + distRankUp';
edistRankAll(find(edistRankAll==0)) = nan; % Diagonals are now nan again

edistRow

mdistAll = pdist2(data,data,'cityblock');



