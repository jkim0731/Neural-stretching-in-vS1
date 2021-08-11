clear all; close all; clc;

%%% Generate a uArray overview %%%

%% load uArray
uArray = load('D:\2020 Neural stretching in S1\Data\UberJK025S19_NC.mat'); uArray = uArray.u;

%% show reference images
nRefImg = length(uArray.mimg);
figure;
for i = 1:nRefImg
   subplot(1,nRefImg,i); imagesc(uArray.mimg{i}); 
end

%% show a trial
% behavioral
trialIdx = 1;
trial = uArray.trials{trialIdx};
figure; subplot(2,1,1); hold on;
plot(trial.whiskerTime, trial.theta, 'k');
plot([trial.poleUpOnsetTime trial.poleDownOnsetTime], [0, 0], 'b', 'LineWidth', 2);

% spiking
neuIdx = [1, 48, 115, 194, 274, 398, 462, 465]; c = 0;
subplot(2,1,2); hold on;
for n = neuIdx
   disp(n);
   tIdx = trial.tpmTime{floor(trial.neuindSession(n)/1000)};
   plot(tIdx, trial.spk(n,:) + c, 'k');
   
   c = c + max(trial.spk(n,:)) + 0.5;
end

%% get performance
performance = vecFromUArray(uArray, 'response');
performance(performance < 1) = 0;
figure;
plot(movmean(performance, 100));