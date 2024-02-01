% Checking error sessions manually.
% Most of them are due to frame order error during merging two or more .sbx
% files. See if this is due to the error in the sbx file or during running
% jksuite2p.
% Also, in some cases there was an error in allocating volumes. Fix this
% MANUALLY.
% 2020/11/24 JK

baseDir = 'E:\';
mouse = 30;
session = 14;

fn0 = sprintf('%03d_%03d_000',mouse,session);
fn1 = sprintf('%03d_%03d_001',mouse,session);
info0 = load([fn0,'.trials'],'-mat');
info1 = load([fn1,'.trials'],'-mat');

frame0 = squeeze(jksbxread(fn0,info0.frame_to_use{5}(10),1));
frame1 = squeeze(jksbxread(fn1,info1.frame_to_use{5}(2),1));

imshowpair(mat2gray(frame0), mat2gray(frame1), 'montage')



%% Block (volume) assignment error in JK039 S901
%%
fn = '039_901_000';
im1 = jksbxread(fn,trials(33).frames(1)+3,4);
figure, montage(squeeze(im1(1,:,:,:)))

%%
im1 = jksbxread(fn,trials(43).frames(1)+4,4);
figure, montage(squeeze(im1(1,:,:,:)))

%%
im2 = jksbxread(fn,trials(53).frames(1)+5,4);
figure, montage(squeeze(im2(1,:,:,:)))

%%
% Change the first chunk to volume 1
clear
load('039_901_000')
newMessage = info.messages;
newMessage(2:33) = info.messages(1:32);
newMessage{1} = info.messages{33};
info.messages = newMessage;
save('039_901_000.mat', 'info')
%% 
%% DON'T FORGET TO SAVE THIS TO HDD!!
%%


%% Possibly similar error at JK038 S901
load('038_901_000.trials', '-mat')
fn = '038_901_000';
%% Upper volume
im1 = jksbxread(fn,trials(8).frames(1)+3,4);
figure, montage(squeeze(im1(1,:,:,:)))
title('Upper volume')
%% Lower volume
im2 = jksbxread(fn,trials(18).frames(1)+5,4);
figure, montage(squeeze(im2(1,:,:,:)))
title('Lower volume')

%% Test volume
tInd = 41;
im = jksbxread(fn,trials(tInd).frames(1)+5,4);
figure, montage(squeeze(im(1,:,:,:)))
title(sprintf('Trial Index = %d', tInd))


%%
% Change the first chunk to volume 1
clear
load('038_901_000')
newMessage = info.messages;
newMessage(2:8) = info.messages(1:7);
newMessage{1} = info.messages{8};
%%
info.messages = newMessage;
save('038_901_000.mat', 'info')


%%
%% DON'T FORGET TO SAVE THIS TO HDD!!
%%
