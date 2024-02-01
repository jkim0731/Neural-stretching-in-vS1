% 2021/11/23
% Check images from JK039 9998_1 (piezo pre-learning)
% There was evidence of saturated frames from data.bin (after suite2p processing)

%% BS
sbxDir = 'J:\';
mouse = 39;
sessionHeader = '9998_1';
%% Get filenames for the corresponding sessions
fnlist = dir(sprintf('%s%03d\\%03d_%s*.sbx', sbxDir, mouse, mouse, sessionHeader));
fnInfo = [fnlist(1).folder, filesep, fnlist(1).name(1:end-4), '.mat'];
load(fnInfo, 'info');
mimgAll = zeros(info.sz(1), info.sz(2)-99, length(fnlist));
clear info

maxidx = sbx_maxidx([fnlist(1).folder, filesep, fnlist(1).name(1:end-4)]);
imgs = jksbxread([fnlist(1).folder, filesep, fnlist(1).name(1:end-4)], 0, maxidx);
imgs = squeeze(imgs(1,:,:,:));
mimg = mean(imgs(:,100:end,:),3);
mimgAll(:,:,1) = mimg;
%%
for fi = 9 : length(fnlist)
    maxidx = uint64(sbx_maxidx([fnlist(fi).folder, filesep, fnlist(fi).name(1:end-4)]));
    imgs = jksbxread([fnlist(fi).folder, filesep, fnlist(fi).name(1:end-4)], 0, maxidx);
    imgs = squeeze(imgs(1,:,:,:));
    mimg = mean(imgs(:,100:end,:),3);
    mimgAll(:,:,fi) = mimg;
end

%%
imshow3D(mimgAll)
%%
figure, imshow(mat2gray(mimgAll(:,:,2)))
%%
maxidx = uint64(sbx_maxidx('039_9998_111'));
img = jksbxread('039_9998_111',0,maxidx);
img = squeeze(img(1,:,100:end,:));
imshow3D(img)

%%
figure, imshow(mat2gray(squeeze(mimgAll(:,:,4))))
%%
fnInfo = [fnlist(1).folder, filesep, fnlist(1).name(1:end-4), '.mat'];
load(fnInfo, 'info');
lastImgs = zeros(info.sz(1), info.sz(2)-99, length(fnlist));
for fi = 1 : length(fnlist)
    maxidx = uint64(sbx_maxidx([fnlist(fi).folder, filesep, fnlist(fi).name(1:end-4)]));
    img = jksbxread([fnlist(fi).folder, filesep, fnlist(fi).name(1:end-4)], maxidx, 1);
    lastImgs(:,:,fi) = squeeze(img(1,:,100:end));
end
imshow3D(lastImgs)