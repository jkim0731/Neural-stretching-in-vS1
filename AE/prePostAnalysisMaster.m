clear all; close all; clc;
%% load the session pair to compare;
% data{1} = load('Data/UberJK025S04_NC_results').resultsStruct; data{2} = load('Data/UberJK025S19_NC_results').resultsStruct;
% data{1} = load('Data/UberJK030S03_NC_results').resultsStruct; data{2} = load('Data/UberJK030S21_NC_results').resultsStruct;
data{1} = load('Data/UberJK036S01_NC_results').resultsStruct; data{2} = load('Data/UberJK036S17_NC_results').resultsStruct;
% data{1} = load('Data/UberJK039S01_NC_results').resultsStruct; data{2} = load('Data/UberJK039S23_NC_results').resultsStruct;
% data{1} = load('Data/UberJK052S03_NC_results').resultsStruct; data{2} = load('Data/UberJK052S21_NC_results').resultsStruct;

%% look at PCA space response differences for both layers
cScheme = {[1 0 0], [0.7 0 0], [0.4 0 0], [0 0 0], [0 0 0.4], [0 0 0.7], [0 0 1]};
uniqueAngle = [45, 60, 75, 90, 105, 120, 135];

figure;
c = 1;
for i = 1:2
   for j = 1:2
      subplot(2,2,c); hold on
      
      for k = 1:size(data{j}.pca.score{i}, 2)
          scatter3(data{j}.pca.score{i}(k, 1),...
                   data{j}.pca.score{i}(k, 2),...
                   data{j}.pca.score{i}(k, 3),...
                   'filled', 'MarkerFaceColor', cScheme{find(uniqueAngle == data{j}.trialAngle{i}(k))});
      end
      
      c = c + 1;
      xlim([-20 20])
      ylim([-20 20])
   end
end

%% look at distance map difference for both layers
figure;

% upper layer
overallMax = max(cellfun(@max, cellfun(@max, {data{1}.distance{1}, data{2}.distance{1}}, 'UniformOutput', false)));
subplot(2,2,1); hold on;
imagesc(data{1}.distance{1}); set(gca, 'YDir', 'normal'); caxis([0 overallMax]); colormap('hot');
a = sort(data{1}.trialAngle{1}); idx = find(diff(a)>1);
for i = 1:length(idx)
   plot([0 length(a)], [idx(i), idx(i)], 'w--', 'LineWidth', 1.5); 
   plot([idx(i), idx(i)], [0 length(a)], 'w--', 'LineWidth', 1.5); 
end
xlim([0 length(a)]); ylim([0 length(a)]);

subplot(2,2,2); hold on;
imagesc(data{2}.distance{1}); set(gca, 'YDir', 'normal'); caxis([0 overallMax]); colormap('hot');
a = sort(data{2}.trialAngle{1}); idx = find(diff(a)>1);
for i = 1:length(idx)
   plot([0 length(a)], [idx(i), idx(i)], 'w--', 'LineWidth', 1.5); 
   plot([idx(i), idx(i)], [0 length(a)], 'w--', 'LineWidth', 1.5); 
end
xlim([0 length(a)]); ylim([0 length(a)]);

% lower layer
overallMax = max(cellfun(@max, cellfun(@max, {data{1}.distance{2}, data{2}.distance{2}}, 'UniformOutput', false)));
subplot(2,2,3); hold on;
imagesc(data{1}.distance{2}); set(gca, 'YDir', 'normal'); caxis([0 overallMax]); colormap('hot');
a = sort(data{1}.trialAngle{2}); idx = find(diff(a)>1);
for i = 1:length(idx)
   plot([0 length(a)], [idx(i), idx(i)], 'w--', 'LineWidth', 1.5); 
   plot([idx(i), idx(i)], [0 length(a)], 'w--', 'LineWidth', 1.5); 
end
xlim([0 length(a)]); ylim([0 length(a)]);

subplot(2,2,4); hold on;
imagesc(data{2}.distance{2}); set(gca, 'YDir', 'normal'); caxis([0 overallMax]); colormap('hot');
a = sort(data{2}.trialAngle{2}); idx = find(diff(a)>1);
for i = 1:length(idx)
   plot([0 length(a)], [idx(i), idx(i)], 'w--', 'LineWidth', 1.5); 
   plot([idx(i), idx(i)], [0 length(a)], 'w--', 'LineWidth', 1.5); 
end
xlim([0 length(a)]); ylim([0 length(a)]);

%% correlation of distance with trial angle
figure;
trialVar = 'trialAngle';

% upper layer
subplot(2,2,1); hold on;
[d, v] = distanceAnalysis(data{1}.pca.score{1}(:, 1:3), data{1}.(trialVar){1}, false);
% scatter(v(:) + randn(1,length(v(:)))', d(:), 'k.'); ylim([0 40]);
scatter(v(1:10:end) + randn(1,length(v(1:10:end))), d(1:10:end), 'k.'); ylim([0 40]);
[cBin, classes] = classBin(d(:), v(:)); plot(classes, cellfun(@mean, cBin), 'b', 'LineWidth', 2);
[rho, pVal] = corr(v(:), d(:),'Type','Spearman');

subplot(2,2,2); hold on;
[d, v] = distanceAnalysis(data{2}.pca.score{1}(:, 1:3), data{2}.(trialVar){1}, false);
% scatter(v(:) + randn(1,length(v(:)))', d(:), 'k.'); ylim([0 40]);
scatter(v(1:10:end) + randn(1,length(v(1:10:end))), d(1:10:end), 'k.'); ylim([0 40]);
[cBin, classes] = classBin(d(:), v(:)); plot(classes, cellfun(@mean, cBin), 'b', 'LineWidth', 2);
[rho, pVal] = corr(v(:), d(:),'Type','Spearman');

% lower layer
subplot(2,2,3); hold on;
[d, v] = distanceAnalysis(data{1}.pca.score{2}(:, 1:3), data{1}.(trialVar){2}, false);
% scatter(v(:) + randn(1,length(v(:)))', d(:), 'k.'); ylim([0 40]);
scatter(v(1:10:end) + randn(1,length(v(1:10:end))), d(1:10:end), 'k.'); ylim([0 40]);
[cBin, classes] = classBin(d(:), v(:)); plot(classes, cellfun(@mean, cBin), 'b', 'LineWidth', 2);
[rho, pVal] = corr(v(:), d(:),'Type','Spearman');

subplot(2,2,4); hold on;
[d, v] = distanceAnalysis(data{2}.pca.score{2}(:, 1:3), data{2}.(trialVar){2}, false);
% scatter(v(:) + randn(1,length(v(:)))', d(:), 'k.'); ylim([0 40]);
scatter(v(1:10:end) + randn(1,length(v(1:10:end))), d(1:10:end), 'k.'); ylim([0 40]);
[cBin, classes] = classBin(d(:), v(:)); plot(classes, cellfun(@mean, cBin), 'b', 'LineWidth', 2);
[rho, pVal] = corr(v(:), d(:),'Type','Spearman');

%% distance vs. trial performance
figure;

% upper layer
subplot(2,2,1); hold on;
r = data{1}.pca.score{1}(:, 1:3); nR = vecnorm(r'); c = data{1}.trialChoice{1};
% tA = data{1}.trialAngle{1}; nR = nR(tA ~= 90); c = c(tA ~=90);
histogram(nR(c==0 | c==-1), 0:2:30, 'FaceColor', 'm', 'Normalization', 'probability');
histogram(nR(c==1), 0:2:30, 'FaceColor', 'c', 'Normalization', 'probability');
ylim([0 0.4]);

subplot(2,2,2); hold on;
r = data{2}.pca.score{1}(:, 1:3); nR = vecnorm(r'); c = data{2}.trialChoice{1};
% tA = data{2}.trialAngle{1}; nR = nR(tA ~= 90); c = c(tA ~=90);
histogram(nR(c==0 | c==-1), 0:2:30, 'FaceColor', 'm', 'Normalization', 'probability');
histogram(nR(c==1), 0:2:30, 'FaceColor', 'c', 'Normalization', 'probability');
ylim([0 0.4]);

% lower layer
subplot(2,2,3); hold on;
r = data{1}.pca.score{2}(:, 1:3); nR = vecnorm(r'); c = data{1}.trialChoice{2};
tA = data{1}.trialAngle{2}; nR = nR(tA ~= 90); c = c(tA ~=90);
histogram(nR(c==0 | c==-1), 0:2:30, 'FaceColor', 'm', 'Normalization', 'probability');
histogram(nR(c==1), 0:2:30, 'FaceColor', 'c', 'Normalization', 'probability');
ylim([0 0.6]);

subplot(2,2,4); hold on;
r = data{2}.pca.score{2}(:, 1:3); nR = vecnorm(r'); c = data{2}.trialChoice{2};
tA = data{2}.trialAngle{2}; nR = nR(tA ~= 90); c = c(tA ~=90);
histogram(nR(c==0 | c==-1), 0:2:30, 'FaceColor', 'm', 'Normalization', 'probability');
histogram(nR(c==1), 0:2:30, 'FaceColor', 'c', 'Normalization', 'probability');
ylim([0 0.6]);

%% look at distance distribution difference for both layers
figure;
subplot(2,1,1); hold on;
histogram(data{1}.distance{1}(:), 0:1:40, 'FaceColor', 'k', 'Normalization', 'probability'); 
histogram(data{2}.distance{1}(:), 0:1:40, 'FaceColor', 'r', 'Normalization', 'probability');
[h, p] = ttest2(data{1}.distance{1}(:), data{2}.distance{1}(:)); title(p);

subplot(2,1,2); hold on;
histogram(data{1}.distance{2}(:), 0:1:40, 'FaceColor', 'k', 'Normalization', 'probability'); 
histogram(data{2}.distance{2}(:), 0:1:40, 'FaceColor', 'r', 'Normalization', 'probability');
[h, p] = ttest2(data{1}.distance{2}(:), data{2}.distance{2}(:)); title(p);