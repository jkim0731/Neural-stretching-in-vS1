% To see how many active neurons are added throughout each session
% 11 Volumes, each naive and expert

mice = [25,27,30,36,39,52];
sessions = {[4,19], [3,10], [3,21], [1,17], [1,23], [3,21]};

tActPropCell = cell(12,2); % (:,1) naive, (:,2) expert
for mi = 1 : length(mice)
    mouse = mice(mi);
    for si = 1 : 2
        session = sessions{mi}(si);
        load(sprintf('UberJK%03dS%02d_NC',mouse, session), 'u');
        % Upper layer
        if mi ~= 2 % Not for JK027
            tInd = find(cellfun(@(x) ismember(1,x.planes), u.trials));
            activity = cell2mat(cellfun(@(x) x.spk, u.trials(tInd)', 'un', 0));
            tempProp = ones(1,size(activity,2));
            numCell = size(activity,1);
            fi = 1;
            while ~isempty(activity)
                actInd = find(activity(:,fi));
                if fi == 1
                    tempProp(fi) = length(actInd)/numCell;
                else
                    tempProp(fi) = tempProp(fi-1) + length(actInd)/numCell;
                end
                activity(actInd,:) = [];
                fi = fi + 1;
            end
            tempTime = [1:size(activity,2)] / u.frameRate;
            tActPropCell{(mi-1)*2 + 1,si} = [tempProp; tempTime];
        end
        
        % Lower layer
        tInd = find(cellfun(@(x) ismember(5,x.planes), u.trials));
        activity = cell2mat(cellfun(@(x) x.spk, u.trials(tInd)', 'un', 0));
        tempProp = ones(1,size(activity,2));
        numCell = size(activity,1);
        fi = 1;
        while ~isempty(activity)
            actInd = find(activity(:,fi));
            if fi == 1
                tempProp(fi) = length(actInd)/numCell;
            else
                tempProp(fi) = tempProp(fi-1) + length(actInd)/numCell;
            end
            activity(actInd,:) = [];
            fi = fi + 1;
        end
        tempTime = [1:size(activity,2)] / u.frameRate;
        tActPropCell{mi*2,si} = [tempProp; tempTime];
    end
end

tActPropCell(3,:) = [];

%%
cellfun(@(x) x(2,end), tActPropCell)

%%
cellfun(@(x) x(2,find(abs(x(1,:)-1) <0.00001)), tActPropCell)

%%
tRange = 0:10:1210;
tProp = cell(11,2);
for i = 1 : 11
    for j = 1 : 2
        tempT = nan(1,length(tRange));
        tempT(1) = 0;
        for ri = 2 : length(tRange)
            ind = find(tActPropCell{i,j}(2,:) < tRange(ri), 1, 'last');
            if ~isempty(ind)
                tempT(ri) = tActPropCell{i,j}(1,ind);
            end
        end
        tProp{i,j} = tempT;
    end
end

%%
colorsTransient = [248 171 66; 40 170 225] / 255;
figure, hold on
plot(tRange, nanmean(cell2mat(tProp(:,1))), 'color', colorsTransient(1,:))
plot(tRange, nanmean(cell2mat(tProp(:,2))), 'color', colorsTransient(2,:))

legend({'Naive', 'Expert'}, 'autoupdate', 'off')

boundedline(tRange, nanmean(cell2mat(tProp(:,1))), sem(cell2mat(tProp(:,1))), 'cmap', colorsTransient(1,:))
boundedline(tRange, nanmean(cell2mat(tProp(:,2))), sem(cell2mat(tProp(:,2))), 'cmap', colorsTransient(2,:))

xlabel('Time (s)')
ylabel('Active neuron proportion')
xlim([0 200])
plot([0 200], [0.99 0.99], '--', 'color', [0.6 0.6 0.6])




%%
%% Dividing into first half and last half
%%

mice = [25,27,30,36,39,52];
sessions = {[4,19], [3,10], [3,21], [1,17], [1,23], [3,21]};

numLostSH = zeros(12,2); % number of lost neurons at the second half (:,1) naive, (:,2) expert 
numNewSH = zeros(12,2); % number of new neurons at the second half (:,1) naive, (:,2) expert 
numAll = zeros(12,2);
for mi = 1 : length(mice)
    mouse = mice(mi);
    for si = 1 : 2
        session = sessions{mi}(si);
        load(sprintf('UberJK%03dS%02d_NC',mouse, session), 'u');
        % Upper layer
        if mi ~= 2 % Not for JK027
            tInd = find(cellfun(@(x) ismember(1,x.planes), u.trials));
            divPoint = floor(length(tInd)/2);
            activityFH = cell2mat(cellfun(@(x) x.spk, u.trials(tInd(1:divPoint))', 'un', 0));
            activitySH = cell2mat(cellfun(@(x) x.spk, u.trials(tInd(divPoint+1:end))', 'un', 0));
            
            silentFH = find(sum(activityFH,2)==0);
            silentSH = find(sum(activitySH,2)==0);
            if ~isempty(intersect(silentFH, silentSH))
                error('Silent neuron')
            end
            
            numLostSH((mi-1)*2 + 1,si) = length(silentSH);
            numNewSH((mi-1)*2 + 1,si) = length(silentFH);
            numAll((mi-1)*2+1, si) = size(activityFH,1);
        end
        
        % Lower layer
        tInd = find(cellfun(@(x) ismember(5,x.planes), u.trials));
        divPoint = floor(length(tInd)/2);
        activityFH = cell2mat(cellfun(@(x) x.spk, u.trials(tInd(1:divPoint))', 'un', 0));
        activitySH = cell2mat(cellfun(@(x) x.spk, u.trials(tInd(divPoint+1:end))', 'un', 0));

        silentFH = find(sum(activityFH,2)==0);
        silentSH = find(sum(activitySH,2)==0);
        if ~isempty(intersect(silentFH, silentSH))
            error('Silent neuron')
        end

        numLostSH(mi*2,si) = length(silentSH);
        numNewSH(mi*2,si) = length(silentFH);
        numAll(mi*2, si) = size(activityFH,1);
    end
end

numLostSH(3,:) = [];
numNewSH(3,:) = [];
numAll(3,:) = [];
%%
colorsTransient = [248 171 66; 40 170 225] / 255;
figure, hold on
tempNaive = [numLostSH(:,1), numNewSH(:,1)];
tempExpert = [numLostSH(:,2), numNewSH(:,2)];
errorbar([1:2], mean(tempNaive), sem(tempNaive), 'o-', 'color', colorsTransient(1,:))
errorbar([1:2], mean(tempExpert), sem(tempExpert), 'o-', 'color', colorsTransient(2,:))
legend({'Naive', 'Expert'})
xlim([0.5 2.5]), xticks(1:2), xticklabels({'First Half', 'Second Half'}), xtickangle(45)
ylabel('Number of unique neurons')


%%
colorsTransient = [248 171 66; 40 170 225] / 255;
figure, hold on
tempNaive = [numLostSH(:,1), numNewSH(:,1)]./numAll(:,1);
tempExpert = [numLostSH(:,2), numNewSH(:,2)]./numAll(:,2);
errorbar([1:2], mean(tempNaive), sem(tempNaive), 'o-', 'color', colorsTransient(1,:))
errorbar([1:2], mean(tempExpert), sem(tempExpert), 'o-', 'color', colorsTransient(2,:))
legend({'Naive', 'Expert'})
xlim([0.5 2.5]), xticks(1:2), xticklabels({'First Half', 'Second Half'}), xtickangle(45)
ylabel('Proportion of unique neurons')
