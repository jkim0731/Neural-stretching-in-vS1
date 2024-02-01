%% Angle selectivity and encoding of behavioral inputs
%% Transient VS persistent neurons

%% Purpose:

% Is there a difference between persistent and transient neurons,
% In terms of angle selectivity
% In terms of tuned angle response
% In terms of behavioral encoding?

% Compare between Transient & angle-tuned VS Persistent & angle-tuned
% Also report persistently angle-tuned neurons and persistently
% touch-responsive neurons

% Add before answer and after answer
% Compare between imaging volumes

%% Loading
baseDir = 'C:\Users\shires\Dropbox\Works\Projects\2020 Neural stretching in S1\Data\';
load([baseDir, 'matchedPopResponse_201230'])
numVol = 11;
colorsTransient = [248 171 66; 40 170 225] / 255;
colorsPersistent = [1 0 0; 0 0 1];

%% Angle selectivity
angleSelectivityBeforeAnswer = zeros(numVol,2,11); % (:,1,:) Naive, (:,2,:) Expert, 
% (:,:,1) Transient angle-tuned
% (:,:,2) Persistent angle-tuned
% (:,:,3) Transient touch
% (:,:,4) Persistent touch
% (:,:,5) Transient non-touch
% (:,:,6) Persistent non-touch
% (:,:,7) Transiently active
% (:,:,8) Persistently active

% (:,:,9) Persistently angle-tuned
% (:,:,10) Persistently touch
% (:,:,11) Persistently non-touch

angleSelectivityAfterAnswer = zeros(numVol,2,11);

for vi = 1 : numVol
    naiveTransientInd = expert(vi).indSilent;
    persistentInd = setdiff(1:length(naive(vi).allID), union(naive(vi).indSilent, expert(vi).indSilent));
    expertTransientInd = naive(vi).indSilent;
    
    % Before Answer Lick
    % Naive
    % 1. Transient angle-tuned
    atc = angle_tuning_curve(naive(vi).touchBeforeAnswer( intersect(naiveTransientInd, naive(vi).indTuned), :), naive(vi).trialAngle);
    angleSelectivityBeforeAnswer(vi,1,1) = mean(angle_selectivity(atc));
    % 2. Persistent angle-tuned
    atc = angle_tuning_curve(naive(vi).touchBeforeAnswer( intersect(persistentInd, naive(vi).indTuned), :), naive(vi).trialAngle);
    angleSelectivityBeforeAnswer(vi,1,2) = mean(angle_selectivity(atc));
    % 3. Transient touch
    atc = angle_tuning_curve(naive(vi).touchBeforeAnswer( intersect(naiveTransientInd, union(naive(vi).indTuned, naive(vi).indNottuned)), :), naive(vi).trialAngle);
    angleSelectivityBeforeAnswer(vi,1,3) = mean(angle_selectivity(atc));
    % 4. Persistent touch
    atc = angle_tuning_curve(naive(vi).touchBeforeAnswer( intersect(persistentInd, union(naive(vi).indTuned, naive(vi).indNottuned)), :), naive(vi).trialAngle);
    angleSelectivityBeforeAnswer(vi,1,4) = mean(angle_selectivity(atc));
    % 5. Transient non-touch
    atc = angle_tuning_curve(naive(vi).touchBeforeAnswer( intersect(naiveTransientInd, naive(vi).indNontouch), :), naive(vi).trialAngle);
    angleSelectivityBeforeAnswer(vi,1,5) = mean(angle_selectivity(atc));
    % 6. Persistent non-touch
    atc = angle_tuning_curve(naive(vi).touchBeforeAnswer( intersect(persistentInd, naive(vi).indNontouch), :), naive(vi).trialAngle);
    angleSelectivityBeforeAnswer(vi,1,6) = mean(angle_selectivity(atc));
    % 7. Transiently active
    atc = angle_tuning_curve(naive(vi).touchBeforeAnswer( naiveTransientInd, :), naive(vi).trialAngle);
    angleSelectivityBeforeAnswer(vi,1,7) = mean(angle_selectivity(atc));
    % 8. Persistently active
    atc = angle_tuning_curve(naive(vi).touchBeforeAnswer( persistentInd, :), naive(vi).trialAngle);
    angleSelectivityBeforeAnswer(vi,1,8) = mean(angle_selectivity(atc));
    
    % 9. Persistently angle-tuned
    atc = angle_tuning_curve(naive(vi).touchBeforeAnswer( intersect(naive(vi).indTuned, expert(vi).indTuned), :), naive(vi).trialAngle);
    angleSelectivityBeforeAnswer(vi,1,9) = mean(angle_selectivity(atc));
    % 10. Persistently touch
    atc = angle_tuning_curve(naive(vi).touchBeforeAnswer( intersect(union(naive(vi).indTuned, naive(vi).indNottuned), union(expert(vi).indTuned, expert(vi).indNottuned)), :), naive(vi).trialAngle);
    angleSelectivityBeforeAnswer(vi,1,10) = mean(angle_selectivity(atc));
    % 11. Persistently non-touch
    atc = angle_tuning_curve(naive(vi).touchBeforeAnswer( intersect(naive(vi).indNontouch, expert(vi).indNontouch), :), naive(vi).trialAngle);
    angleSelectivityBeforeAnswer(vi,1,11) = mean(angle_selectivity(atc));
    
    % Expert
    % 1. Transient angle-tuned
    atc = angle_tuning_curve(expert(vi).touchBeforeAnswer( intersect(expertTransientInd, expert(vi).indTuned), :), expert(vi).trialAngle);
    angleSelectivityBeforeAnswer(vi,2,1) = mean(angle_selectivity(atc));
    % 2. Persistent angle-tuned
    atc = angle_tuning_curve(expert(vi).touchBeforeAnswer( intersect(persistentInd, expert(vi).indTuned), :), expert(vi).trialAngle);
    angleSelectivityBeforeAnswer(vi,2,2) = mean(angle_selectivity(atc));
    % 3. Transient touch
    atc = angle_tuning_curve(expert(vi).touchBeforeAnswer( intersect(expertTransientInd, union(expert(vi).indTuned, expert(vi).indNottuned)), :), expert(vi).trialAngle);
    angleSelectivityBeforeAnswer(vi,2,3) = mean(angle_selectivity(atc));
    % 4. Persistent touch
    atc = angle_tuning_curve(expert(vi).touchBeforeAnswer( intersect(persistentInd, union(expert(vi).indTuned, expert(vi).indNottuned)), :), expert(vi).trialAngle);
    angleSelectivityBeforeAnswer(vi,2,4) = mean(angle_selectivity(atc));
    % 5. Transient non-touch
    atc = angle_tuning_curve(expert(vi).touchBeforeAnswer( intersect(expertTransientInd, expert(vi).indNontouch), :), expert(vi).trialAngle);
    angleSelectivityBeforeAnswer(vi,2,5) = mean(angle_selectivity(atc));
    % 6. Persistent non-touch
    atc = angle_tuning_curve(expert(vi).touchBeforeAnswer( intersect(persistentInd, expert(vi).indNontouch), :), expert(vi).trialAngle);
    angleSelectivityBeforeAnswer(vi,2,6) = mean(angle_selectivity(atc));
    % 7. Transient non-touch
    atc = angle_tuning_curve(expert(vi).touchBeforeAnswer( expertTransientInd, :), expert(vi).trialAngle);
    angleSelectivityBeforeAnswer(vi,2,7) = mean(angle_selectivity(atc));
    % 8. Persistent non-touch
    atc = angle_tuning_curve(expert(vi).touchBeforeAnswer( persistentInd, :), expert(vi).trialAngle);
    angleSelectivityBeforeAnswer(vi,2,8) = mean(angle_selectivity(atc));
    
    % 9. Persistently angle-tuned
    atc = angle_tuning_curve(expert(vi).touchBeforeAnswer( intersect(naive(vi).indTuned, expert(vi).indTuned), :), expert(vi).trialAngle);
    angleSelectivityBeforeAnswer(vi,2,9) = mean(angle_selectivity(atc));
    % 10. Persistently touch
    atc = angle_tuning_curve(expert(vi).touchBeforeAnswer( intersect(union(naive(vi).indTuned, naive(vi).indNottuned), union(expert(vi).indTuned, expert(vi).indNottuned)), :), expert(vi).trialAngle);
    angleSelectivityBeforeAnswer(vi,2,10) = mean(angle_selectivity(atc));
    % 11. Persistently non-touch
    atc = angle_tuning_curve(expert(vi).touchBeforeAnswer( intersect(naive(vi).indNontouch, expert(vi).indNontouch), :), expert(vi).trialAngle);
    angleSelectivityBeforeAnswer(vi,2,11) = mean(angle_selectivity(atc));
    
    
    
    % After Answer Lick
    % Naive
    % 1. Transient angle-tuned
    atc = angle_tuning_curve(naive(vi).touchAfterAnswer( intersect(naiveTransientInd, naive(vi).indTuned), :), naive(vi).trialAngle);
    angleSelectivityAfterAnswer(vi,1,1) = mean(angle_selectivity(atc));
    % 2. Persistent angle-tuned
    atc = angle_tuning_curve(naive(vi).touchAfterAnswer( intersect(persistentInd, naive(vi).indTuned), :), naive(vi).trialAngle);
    angleSelectivityAfterAnswer(vi,1,2) = mean(angle_selectivity(atc));
    % 3. Transient touch
    atc = angle_tuning_curve(naive(vi).touchAfterAnswer( intersect(naiveTransientInd, union(naive(vi).indTuned, naive(vi).indNottuned)), :), naive(vi).trialAngle);
    angleSelectivityAfterAnswer(vi,1,3) = mean(angle_selectivity(atc));
    % 4. Persistent touch
    atc = angle_tuning_curve(naive(vi).touchAfterAnswer( intersect(persistentInd, union(naive(vi).indTuned, naive(vi).indNottuned)), :), naive(vi).trialAngle);
    angleSelectivityAfterAnswer(vi,1,4) = mean(angle_selectivity(atc));
    % 5. Transient non-touch
    atc = angle_tuning_curve(naive(vi).touchAfterAnswer( intersect(naiveTransientInd, naive(vi).indNontouch), :), naive(vi).trialAngle);
    angleSelectivityAfterAnswer(vi,1,5) = mean(angle_selectivity(atc));
    % 6. Persistent non-touch
    atc = angle_tuning_curve(naive(vi).touchAfterAnswer( intersect(persistentInd, naive(vi).indNontouch), :), naive(vi).trialAngle);
    angleSelectivityAfterAnswer(vi,1,6) = mean(angle_selectivity(atc));
    % 7. Transiently active
    atc = angle_tuning_curve(naive(vi).touchAfterAnswer( naiveTransientInd, :), naive(vi).trialAngle);
    angleSelectivityAfterAnswer(vi,1,7) = mean(angle_selectivity(atc));
    % 8. Persistently active
    atc = angle_tuning_curve(naive(vi).touchAfterAnswer( persistentInd, :), naive(vi).trialAngle);
    angleSelectivityAfterAnswer(vi,1,8) = mean(angle_selectivity(atc));
    
    % 9. Persistently angle-tuned
    atc = angle_tuning_curve(naive(vi).touchAfterAnswer( intersect(naive(vi).indTuned, expert(vi).indTuned), :), naive(vi).trialAngle);
    angleSelectivityAfterAnswer(vi,1,9) = mean(angle_selectivity(atc));
    % 10. Persistently touch
    atc = angle_tuning_curve(naive(vi).touchAfterAnswer( intersect(union(naive(vi).indTuned, naive(vi).indNottuned), union(expert(vi).indTuned, expert(vi).indNottuned)), :), naive(vi).trialAngle);
    angleSelectivityAfterAnswer(vi,1,10) = mean(angle_selectivity(atc));
    % 11. Persistently non-touch
    atc = angle_tuning_curve(naive(vi).touchAfterAnswer( intersect(naive(vi).indNontouch, expert(vi).indNontouch), :), naive(vi).trialAngle);
    angleSelectivityAfterAnswer(vi,1,11) = mean(angle_selectivity(atc));
    
    % Expert
    % 1. Transient angle-tuned
    atc = angle_tuning_curve(expert(vi).touchAfterAnswer( intersect(expertTransientInd, expert(vi).indTuned), :), expert(vi).trialAngle);
    angleSelectivityAfterAnswer(vi,2,1) = mean(angle_selectivity(atc));
    % 2. Persistent angle-tuned
    atc = angle_tuning_curve(expert(vi).touchAfterAnswer( intersect(persistentInd, expert(vi).indTuned), :), expert(vi).trialAngle);
    angleSelectivityAfterAnswer(vi,2,2) = mean(angle_selectivity(atc));
    % 3. Transient touch
    atc = angle_tuning_curve(expert(vi).touchAfterAnswer( intersect(expertTransientInd, union(expert(vi).indTuned, expert(vi).indNottuned)), :), expert(vi).trialAngle);
    angleSelectivityAfterAnswer(vi,2,3) = mean(angle_selectivity(atc));
    % 4. Persistent touch
    atc = angle_tuning_curve(expert(vi).touchAfterAnswer( intersect(persistentInd, union(expert(vi).indTuned, expert(vi).indNottuned)), :), expert(vi).trialAngle);
    angleSelectivityAfterAnswer(vi,2,4) = mean(angle_selectivity(atc));
    % 5. Transient non-touch
    atc = angle_tuning_curve(expert(vi).touchAfterAnswer( intersect(expertTransientInd, expert(vi).indNontouch), :), expert(vi).trialAngle);
    angleSelectivityAfterAnswer(vi,2,5) = mean(angle_selectivity(atc));
    % 6. Persistent non-touch
    atc = angle_tuning_curve(expert(vi).touchAfterAnswer( intersect(persistentInd, expert(vi).indNontouch), :), expert(vi).trialAngle);
    angleSelectivityAfterAnswer(vi,2,6) = mean(angle_selectivity(atc));
    % 7. Transiently active
    atc = angle_tuning_curve(expert(vi).touchAfterAnswer( expertTransientInd, :), expert(vi).trialAngle);
    angleSelectivityAfterAnswer(vi,2,7) = mean(angle_selectivity(atc));
    % 8. Persistently active
    atc = angle_tuning_curve(expert(vi).touchAfterAnswer( persistentInd, :), expert(vi).trialAngle);
    angleSelectivityAfterAnswer(vi,2,8) = mean(angle_selectivity(atc));
    
    % 9. Persistently angle-tuned
    atc = angle_tuning_curve(expert(vi).touchAfterAnswer( intersect(naive(vi).indTuned, expert(vi).indTuned), :), expert(vi).trialAngle);
    angleSelectivityAfterAnswer(vi,2,9) = mean(angle_selectivity(atc));
    % 10. Persistently touch
    atc = angle_tuning_curve(expert(vi).touchAfterAnswer( intersect(union(naive(vi).indTuned, naive(vi).indNottuned), union(expert(vi).indTuned, expert(vi).indNottuned)), :), expert(vi).trialAngle);
    angleSelectivityAfterAnswer(vi,2,10) = mean(angle_selectivity(atc));
    % 11. Persistently non-touch
    atc = angle_tuning_curve(expert(vi).touchAfterAnswer( intersect(naive(vi).indNontouch, expert(vi).indNontouch), :), expert(vi).trialAngle);
    angleSelectivityAfterAnswer(vi,2,11) = mean(angle_selectivity(atc));
end

%% Draw figures
% Transient, before answer lick
offset = 0.1;
figure, hold on
p = zeros(4,1);
m = cell(4,1);

for posi = 1 : 4
    errorbar(posi-offset, mean(angleSelectivityBeforeAnswer(:,1,(posi-1)*2+1)), sem(angleSelectivityBeforeAnswer(:,1,(posi-1)*2+1)), 'o', 'lineWidth', 1, 'Color', colorsTransient(1,:))
    errorbar(posi+offset, mean(angleSelectivityBeforeAnswer(:,2,(posi-1)*2+1)), sem(angleSelectivityBeforeAnswer(:,2,(posi-1)*2+1)), 'o', 'lineWidth', 1, 'Color', colorsTransient(2,:))
    if posi == 1
        legend({'Naive', 'Expert'}, 'autoupdate', false)
    end
    for vi = 1 : numVol
        plot([posi-offset, posi+offset], angleSelectivityBeforeAnswer(vi,:,(posi-1)*2+1), 'color', [0.6 0.6 0.6])
    end
    [~, p(posi), m{posi}] = paired_test(angleSelectivityBeforeAnswer(:,1,(posi-1)*2+1), angleSelectivityBeforeAnswer(:,2,(posi-1)*2+1));
end

ylabel('Angle selectivity')
xticks(1:4), xticklabels({'Angle-tuned', 'Touch', 'Non-touch', 'All'})
title('Transient (before answer lick)')
% maxVal = max(max(max(angleSelectivityBeforeAnswer)));
% ylim([0 maxVal+0.05])
for i = 1 : 4
    maxVal = max(max(angleSelectivityBeforeAnswer(:,:,(i-1)*2+1)));
    text(i,maxVal+0.05, sprintf('p = %.3f\n(%s)', p(i), m{i}), 'HorizontalAlignment', 'center', 'FontSize', 8)
end

%% Persistent, before answer lick
offset = 0.1;
figure, hold on
p = zeros(4,1);
m = cell(4,1);

for posi = 1 : 4
    errorbar(posi-offset, mean(angleSelectivityBeforeAnswer(:,1,posi*2)), sem(angleSelectivityBeforeAnswer(:,1,posi*2)), 'o', 'lineWidth', 1, 'Color', colorsPersistent(1,:))
    errorbar(posi+offset, mean(angleSelectivityBeforeAnswer(:,2,posi*2)), sem(angleSelectivityBeforeAnswer(:,2,posi*2)), 'o', 'lineWidth', 1, 'Color', colorsPersistent(2,:))
    if posi == 1
        legend({'Naive', 'Expert'}, 'autoupdate', false)
    end
    for vi = 1 : numVol
        plot([posi-offset, posi+offset], angleSelectivityBeforeAnswer(vi,:,posi*2), 'color', [0.6 0.6 0.6])
    end
    [~, p(posi), m{posi}] = paired_test(angleSelectivityBeforeAnswer(:,1,posi*2), angleSelectivityBeforeAnswer(:,2,posi*2));
end

ylabel('Angle selectivity')
xticks(1:4), xticklabels({'Angle-tuned', 'Touch', 'Non-touch', 'All'})
title('Persistent (before answer lick)')
% maxVal = max(max(max(angleSelectivityBeforeAnswer)));
% ylim([0 maxVal+0.05])
for i = 1 : 4
    maxVal = max(max(angleSelectivityBeforeAnswer(:,:,i*2)));
    text(i,maxVal+0.05, sprintf('p = %.3f\n(%s)', p(i), m{i}), 'HorizontalAlignment', 'center', 'FontSize', 8)
end

%% Persistently, before answer lick

offset = 0.1;
figure, hold on
p = zeros(3,1);
m = cell(3,1);

for posi = 1 : 3
    errorbar(posi-offset, mean(angleSelectivityBeforeAnswer(:,1,posi+8)), sem(angleSelectivityBeforeAnswer(:,1,posi+8)), 'o', 'lineWidth', 1, 'Color', colorsPersistent(1,:))
    errorbar(posi+offset, mean(angleSelectivityBeforeAnswer(:,2,posi+8)), sem(angleSelectivityBeforeAnswer(:,2,posi+8)), 'o', 'lineWidth', 1, 'Color', colorsPersistent(2,:))
    if posi == 1
        legend({'Naive', 'Expert'}, 'autoupdate', false)
    end
    for vi = 1 : numVol
        plot([posi-offset, posi+offset], angleSelectivityBeforeAnswer(vi,:,posi+8), 'color', [0.6 0.6 0.6])
    end
    [~, p(posi), m{posi}] = paired_test(angleSelectivityBeforeAnswer(:,1,posi+8), angleSelectivityBeforeAnswer(:,2,posi+8));
end

ylabel('Angle selectivity')
xticks(1:3), xticklabels({'Angle-tuned', 'Touch', 'Non-touch'})
title('Persistently (before answer lick)')
% maxVal = max(max(max(angleSelectivityBeforeAnswer)));
% ylim([0 maxVal+0.05])
for i = 1 : 3
    maxVal = max(max(angleSelectivityBeforeAnswer(:,:,i+8)));
    text(i,maxVal+0.05, sprintf('p = %.3f\n(%s)', p(i), m{i}), 'HorizontalAlignment', 'center', 'FontSize', 8)
end

