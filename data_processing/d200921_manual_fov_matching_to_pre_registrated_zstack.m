% FOV matching
% using zstackReg (made by using zstack_registration.m)
% Each mouse individually.
% Have an option to split FOV
% First, register with mean subvolume around the estimated depth
% Then, calculate correlation with each frame of the subvolume.
% Estimate peak with 2nd polynomial fit.

% Polynomial fit is sensitive to x-y data pairs (center position, # of data
% points, etc). Instead, use smoothing (window of PSF) and spline
% interpolation. 
% Compare correlation values to the correlation between naive and expert
% mean images. If the latter is higher, than two planes are at the
% identical plane.


% Just look at the top plane.
% JK027 (mi==2) has only bottom plane.
% Record correlation curves and best estimate manually, with corresponding
% parameters.

%%
clear
baseDir = 'D:\TPM\JK\suite2p\';
mice = [25,27,30,36,...
    37,38,39,41,...
    52,53,54,56];
numMice = length(mice);
zstackFilenum = [1000, 1000, 2000, 997, ...
    998, 998, 998, 998, ...
    995, 990, 995, 994];
    % after awake / after awake/ after anesthetized / after anesthetized / 
    % before anesthetized / before anesthetized / before anesthetized / before anesthetized /
    % before anesthetized / after ? / after awake / after ?
    % All in  2 um resolution (z-axis)
sessions = {[0:6,13:19,22:25],[0:7,10,14,23:25],[901,902,1:5,12:15,17:25],[901,1:11,13:21],...
    [901,902,1:24],[901,1:22,24:31],[901,1:28],[901,1:19,21:30],...
    [0:29],[0:3,5:15,17:21],[0:14,16:26],[0:13]};

[opt__1,metric] = imregconfig('monomodal');
opt__2 = opt__1;
opt__2.GradientMagnitudeTolerance = 1e-5;
opt__2.MinimumStepLength = 1e-6;
opt__2.MaximumStepLength = 1e-2;
opt__2.RelaxationFactor = 0.75;
opt__2.MaximumIterations = 200;
%%
opt__3 = opt__1;
opt__3.GradientMagnitudeTolerance = 1e-20;
opt__3.MinimumStepLength = 1e-10;
opt__3.MaximumStepLength = 1e-5;
% opt__3.RelaxationFactor = 0.5;
opt__3.MaximumIterations = 500;

%% load data for each mouse

mi = 6;
mouse = mice(mi);

% load registered z-stack
fn = sprintf('%s%03d\\zstackReg_%03d_%d',baseDir,mouse,mouse,zstackFilenum(mi));
load(fn, 'zstackDepths', 'zstackReg')


%% load data for each session, each plane
close all

si = 1;

planei = 5; % 1 for upper volume, 5 for lower volume

% regMeth = 'similarity';
regMeth = 'rigid';

histPlane = 1; % adapthisteq for imaging plane. 0 or 1
histStack = 0; % adapthisteq for z-stack mean image. 0 or 1
histMatch = 1; % imhistmatch for imaging plane to z-stack mean image. 0 or 1

objz = 90; % z value during imaging, in microns
% depthEst = objz / cosd(35); % estimated depth of the imaging plane
depthEst = 290; % or use manual correction

session = sessions{mi}(si);

% load mean image
regfn = sprintf('%s%03d\\regops_%03d_%03d',baseDir,mouse,mouse,session);
load(regfn, 'ops1');
mimg = mat2gray(ops1{planei}.mimg1);


% Basic setting for each plane
svThickness = 160; % in micron
offsetDepth = 0; % in micron
imMargin = 50; % for z-stack


if mi < 6
    xReg = 500; % for mimg of each session
    yReg = 350; % for mimg of each session
elseif mi == 6
    if planei == 1 && si <15
        xReg = 500; % for mimg of each session
        yReg = 350; % for mimg of each session
    else
        xReg = 600; % for mimg of each session
        yReg = 450; % for mimg of each session
    end
elseif mi < 9
    xReg = 600; % for mimg of each session
    yReg = 450; % for mimg of each session
else
    xReg = 600; % for mimg of each session
    yReg = 350; % for mimg of each session
end

split = [2, 2]; % Number of division in rows and columns (in this order)
numSplit = prod(split);

planeDepth = depthEst + offsetDepth; 
svtopDepth = planeDepth - svThickness/2;
[~, svtopPlaneInd] = min(abs(zstackDepths - svtopDepth));
svbottomDepth = planeDepth + svThickness/2;
[~, svbottomPlaneInd] = min(abs(zstackDepths - svbottomDepth));

if histPlane == 1
    mimg = adapthisteq(mimg);
end
mimg = mimg(round(size(mimg,1)/2 - yReg/2) : round(size(mimg,1)/2 + yReg/2), ...
    round(size(mimg,2)/2 - xReg/2) : round(size(mimg,2)/2 + xReg/2));

svRaw = zstackReg(imMargin:end-imMargin,imMargin:end-imMargin,svbottomPlaneInd:svtopPlaneInd);
if mi == 3 && planei < 5
    svRaw = svRaw(120:end-50, 150:end-80, :);
end
if mi == 3 && planei >= 5
    svRaw = svRaw(120:end-30, 130:end-150, :);
end
if mi == 4
    svRaw = svRaw(100:end-50, 120:end-70, :);
    if planei >= 5 && si >= 20
        svRaw = svRaw(80:end, 100:end-100, :);
        mimg = mimg(50:end-50, 80:end-50);
    end
end
if mi == 5 
    if si == 1
        svRaw = svRaw(300:end, 150:end-150, :);
    elseif si >= 2
        svRaw = svRaw(250:end-50, 180:end-150, :);
    end
end
if mi == 6
%     svRaw = svRaw(200:end-30, 120:end-100,:);
    svRaw = svRaw(130:end, 70:end-50,:);
end

if mi == 7
    svRaw = svRaw(150:end-80, 70:end-70, :);
    mimg = mimg(1:end-100, :);
end

if mi == 8
    if planei <5
        svRaw = svRaw(150:end-70,80:end-50, :);
    else 
        svRaw = svRaw(150:end-70,80:end-50, :);
    end
end

if mi == 9
    if planei < 5
        svRaw = svRaw(130:end, 30:end-30, :);
    else 
        svRaw = svRaw(130:end, 50:end-150, :);
    end
end

if mi == 10
    if planei < 5
        svRaw = svRaw(100:end, 180:end-30, :);
    else 
        svRaw = svRaw(100:end, 180:end-30, :);
    end
end

if mi == 11
    if planei < 5
        svRaw = svRaw(70:end-20, 50:end-30, :);
    else 
%         svRaw = svRaw(100:end, 180:end-30, :);
    end
end

if mi == 12
    if planei < 5
        svRaw = svRaw(30:end, 30:end-100, :);
        mimg = mimg(:,1:end-40);
    else 
        svRaw = svRaw(30:end, 30:end-160, :);
        mimg = mimg(:,1:end-100);
    end
end

if histStack == 0
    svMean = (mean(svRaw,3));
else
    svMean = adapthisteq(mean(svRaw,3));
end

if histMatch == 1
    mimg = imhistmatch(mimg, svMean); % option
end
% %%
% figure,
% subplot(121), imshow(mimg), title('Session image'), axis image
% subplot(122), imshow(svMean), title('Mean subvolume'), axis image

% %% First, make a general registration 
% %% and then calculate loose boundary for each split FOV
tformOpt = opt__2;
tform = imregtform(mimg, svMean, regMeth, tformOpt, metric);
moved = imwarp(mimg, tform, 'OutputView', imref2d(size(svMean)));

splitIm = cell(numSplit,1); % Split image of the session mean image
splitFOVreg = cell(numSplit,1); % Registered FOV of split images. For test
splitFOVzstack = cell(numSplit,1); % FOV of template z-stack planes for each split.
                                            % This is going to be the template to match later.
y = size(mimg,1);
rows = split(1);
x = size(mimg,2);
cols = split(2);
for ri = 1 : rows
    for ci = 1 : cols
        tempRows = (ri-1)*round(y/rows)+1 : min(ri*round(y/rows), y);
        tempCols = (ci-1)*round(x/cols)+1 : min(ci*round(x/cols), x);
        tempMat = zeros(y,x);
        tempMat(tempRows, tempCols) = 1;
        splitMat = mimg .* tempMat;
        splitImTemp = zeros(length(tempRows), length(tempCols));
        nzInd = find(tempMat);
        splitImTemp(:) = splitMat(nzInd);
        splitIm{(ri-1)*split(2)+ci} = splitImTemp;
        splitReg = imwarp(tempMat, tform, 'OutputView', imref2d(size(svMean)));
        splitFOVreg{(ri-1)*split(2)+ci} = splitReg;
        tempReg = zeros(size(splitReg));
        tempinds = find(splitReg > 0.8); % 0.8 as an arbitrary threshold
        tempReg(tempinds) = 1;
        ymin = find(sum(tempReg,2),1,'first');
        ymax = find(sum(tempReg,2),1,'last');
        xmin = find(sum(tempReg),1,'first');
        xmax = find(sum(tempReg),1,'last');
        % Dilate into a rectangle 
        % Factor of 0.2 to each side. 0.2 is arbitrarily set.
        newymin = max(round(ymin-(ymax-ymin)*0.2),1);
        newymax = min(round(ymax+(ymax-ymin)*0.2),size(splitReg,1));
        newxmin = max(round(xmin-(xmax-xmin)*0.2),1);
        newxmax = min(round(xmax+(xmax-xmin)*0.2),size(splitReg,2));
        tempMat = zeros(size(splitReg));
        tempMat(newymin:newymax, newxmin:newxmax) = 1;
        splitFOVzstack{(ri-1)*split(2)+ci} = tempMat;
    end
end

% %% confirm the quality of registration visually
figure, 
subplot(211), imshowpair(moved, svMean, 'montage')
title(sprintf('histPlane = %d; histStack = %d; histMatch = %d', histPlane, histStack, histMatch))
subplot(212), imshowpair(moved, svMean) 
if tformOpt.GradientMagnitudeTolerance == opt__2.GradientMagnitudeTolerance
    title('tform opt2')
else
    title('tform opt1')
end
sgtitle(sprintf('JK%03d S%03d plane%02d',mouse, session, planei))

% %% Show correlation & estimate the peak
% % Polynomial fit is sensitive to x-y pair. Don't use it.
% % Instead, use smoothing with window similar to SPF and spline
% % interpolation in 1 um resolution.
smoothWindow = 10;

corrVals = zeros(size(svRaw,3),1);
nzind = find(moved);
for i = 1 : size(svRaw,3)
    currIm = squeeze(svRaw(:,:,i));
    nzindSv = find(currIm);
    nzind = intersect(nzind, nzindSv);
    corrVals(i) = corr(moved(nzind), currIm(nzind));
end

x = zstackDepths(svbottomPlaneInd:svtopPlaneInd);
xx = round(zstackDepths(svtopPlaneInd)):round(zstackDepths(svbottomPlaneInd));
s = smooth(corrVals,smoothWindow);
sp = spline(x,s, xx);
[~, maxind] = max(sp);
maxD = xx(maxind);

figure, hold on
plot(zstackDepths(svbottomPlaneInd:svtopPlaneInd),corrVals, 'color', [0.6 0.6 0.6])
plot(xx, sp, 'r')
legend({'Raw', 'Smooth'}, 'location', 'northwest')
title(sprintf('JK%03d S%03d plane #%d, max %d um (smoothing %d\\mum)',mouse, session, planei, round(maxD), smoothWindow*2))
xlabel('Depth (um)')
ylabel('Correlation')









%%
%% In case of split FOV
%%

% options
histFixed = 1; % 0 or 1. 

% confirm splitFOVzstack visually
% sumRegIm = zeros(size(svMean));
% sumFOVim = zeros(size(svMean));
% for i = 1 : length(splitFOVzstack)
%     sumRegIm = sumRegIm + splitFOVreg{i}*i;
%     sumFOVim = sumFOVim + splitFOVzstack{i}*i;
% end

% figure, 
% subplot(121), imagesc(sumRegIm), axis image
% subplot(122), imagesc(sumFOVim), axis image
% %% Register to mean subvolume in each split FOV

tformRefNaive = cell(1,numSplit);
splitImNaive = cell(1,numSplit);
sumTest = zeros(size(svMean));
for si = 1 : numSplit
    ymin = find(sum(splitFOVzstack{si},2),1,'first');
    ymax = find(sum(splitFOVzstack{si},2),1,'last');
    xmin = find(sum(splitFOVzstack{si}),1,'first');
    xmax = find(sum(splitFOVzstack{si}),1,'last');
    fixed = zeros(ymax-ymin+1, xmax-xmin+1);
    nzInd = find(splitFOVzstack{si});
    fixed(:) = svMean(nzInd);
    if histFixed ==1
        fixed = adapthisteq(fixed); % option
    end
    moving = adapthisteq(splitIm{si});
%     moving = splitIm{si};
%     moving = imhistmatch(splitIm{si}, fixed);

    tformRefNaive{si} = imregtform(moving, fixed, 'similarity', tformOpt, metric);
    tempTestIm = imwarp(splitIm{si}, tformRefNaive{si}, 'OutputView', imref2d(size(fixed)));
    testIm = zeros(size(svMean));
    testIm(ymin:ymax, xmin:xmax) = tempTestIm;
    splitImNaive{si} = testIm;
    sumTest = sumTest + testIm;
end
% %%
% QC
figure, imshowpair(imhistmatch(sumTest,svMean),svMean)
title(sprintf('JK%03d S%03d plane #%d; histFixed = %d ',mouse, session, planei, histFixed))


%% Calculate correaltion in each subvolume
% And estimate peak in each subvolume using smoothing and spline
corrSplit = zeros(size(svRaw,3),numSplit);
for si = 1 : numSplit
    nzInd = find(splitImNaive{si});
    pixVal = splitImNaive{si}(nzInd);
    for pi = 1 : size(svRaw,3)
        planeIm = squeeze(svRaw(:,:,pi));
        planePixVal = planeIm(nzInd);
        corrSplit(pi,si) = corr(pixVal,planePixVal);
    end
end

figure, hold on
for si = 1 : numSplit
    plot(zstackDepths(svbottomPlaneInd:svtopPlaneInd), corrSplit(:,si), 'color', [0.5 0.5 0.5])
end
pmean = plot(zstackDepths(svbottomPlaneInd:svtopPlaneInd), mean(corrSplit,2), 'k');


x = zstackDepths(svbottomPlaneInd:svtopPlaneInd);
xx = round(zstackDepths(svtopPlaneInd)):round(zstackDepths(svbottomPlaneInd));
s = smooth(mean(corrSplit,2),smoothWindow);
sp = spline(x,s, xx);
[~, maxind] = max(sp);
maxD = xx(maxind);

ps = plot(xx, sp, 'r');

legend([pmean, ps], {'Mean', 'Fit'})
title(sprintf('JK%03d S%03d plane #%d, max %d um',mouse, session, planei, round(maxD)))
xlabel('Depth (\mum)')
ylabel('Correlation')

% %%
% %% In case of a sub-split error, select split indices
splitInd = [1,2];
corrSplitSelectNaive = zeros(size(svRaw,3), length(splitInd));
for spi = 1 : length(splitInd)
    corrSplitSelectNaive(:,spi) = corrSplit(:,splitInd(spi));
end

figure, hold on
for spi = 1 : length(splitInd)
    plot(zstackDepths(svbottomPlaneInd:svtopPlaneInd), corrSplitSelectNaive(:,spi), 'color', [0.5 0.5 0.5])
end
pmean = plot(zstackDepths(svbottomPlaneInd:svtopPlaneInd), mean(corrSplitSelectNaive,2), 'k');


x = zstackDepths(svbottomPlaneInd:svtopPlaneInd);
xx = round(zstackDepths(svtopPlaneInd)):round(zstackDepths(svbottomPlaneInd));
s = smooth(mean(corrSplitSelectNaive,2),3);
sp = spline(x,s, xx);
[~, maxind] = max(sp);
maxD = xx(maxind);

ps = plot(xx, sp, 'r');

legend([pmean, ps], {'Mean', 'Fit'})
title(sprintf('JK%03d S%03d plane #%d, max %d um; splits = %s',mouse, session,planei, round(maxD), num2str(splitInd)))
xlabel('Depth (um)')
ylabel('Correlation')








% %% Trying nonrigid registration
% 
% D = imregdemons(mimg, svMean, [500,400,300]);
% % %%
% moved = imwarp(mimg, D);
% 
% figure, 
% subplot(211), imshowpair(moved, svMean, 'montage')
% title(sprintf('histPlane = %d; histStack = %d; histMatch = %d', histPlane, histStack, histMatch))
% subplot(212), imshowpair(moved, svMean) 
% if tformOpt.GradientMagnitudeTolerance == opt__2.GradientMagnitudeTolerance
%     title('tform opt2')
% else
%     title('tform opt1')
% end
% sgtitle(sprintf('JK%03d S%03d plane%02d',mouse, session, planei))
% 
% 
% %%
% D = imregdemons(moved, svMean, [20 20 20]);
% % %%
% movedNR = imwarp(moved, D);
% 
% figure, 
% subplot(211), imshowpair(movedNR, svMean, 'montage')
% title(sprintf('histPlane = %d; histStack = %d; histMatch = %d', histPlane, histStack, histMatch))
% subplot(212), imshowpair(movedNR, svMean) 
% if tformOpt.GradientMagnitudeTolerance == opt__2.GradientMagnitudeTolerance
%     title('tform opt2')
% else
%     title('tform opt1')
% end
% sgtitle(sprintf('JK%03d S%03d plane%02d',mouse, session, planei))















































































%%
%%
%% Spontaneous and passive deflection sessions
%%
%%
% Mean images are saved in a mat file, processed from python


%%
clear
baseDir = 'D:\TPM\JK\suite2p\';
mice = [25,27,30,36,...
    37,38,39,41,...
    52,53,54,56];
numMice = length(mice);
zstackFilenum = [1000, 1000, 2000, 997, ...
    998, 998, 998, 998, ...
    995, 990, 995, 994];
    % after awake / after awake/ after anesthetized / after anesthetized / 
    % before anesthetized / before anesthetized / before anesthetized / before anesthetized /
    % before anesthetized / after ? / after awake / after ?
    % All in  2 um resolution (z-axis)
h5Dir = 'D:\TPM\JK\h5\';
sessionNames = [5554,5555,9998,9999];

[opt__1,metric] = imregconfig('monomodal');
opt__2 = opt__1;
opt__2.GradientMagnitudeTolerance = 1e-5;
opt__2.MinimumStepLength = 1e-6;
opt__2.MaximumStepLength = 1e-2;
opt__2.RelaxationFactor = 0.75;
opt__2.MaximumIterations = 200;
%%
opt__3 = opt__1;
opt__3.GradientMagnitudeTolerance = 1e-20;
opt__3.MinimumStepLength = 1e-10;
opt__3.MaximumStepLength = 1e-5;
% opt__3.RelaxationFactor = 0.5;
opt__3.MaximumIterations = 500;

%% load data for each mouse

mi = 4;
mouse = mice(mi);

% load registered z-stack
fn = sprintf('%s%03d\\zstackReg_%03d_%d',baseDir,mouse,mouse,zstackFilenum(mi));
load(fn, 'zstackDepths', 'zstackReg')


%% load data for each session, each plane


sni = 2;
planei = 1; % 1 for upper volume, 5 for lower volume
sessions = dir(sprintf('%s%03d\\plane_%d\\%03d_%d_*_plane_%d_mimg.mat',h5Dir, mouse, planei, mouse, sessionNames(sni), planei));
length(sessions)

%%
close all
si = 2;

% regMeth = 'similarity';
regMeth = 'rigid';
tformOpt = opt__2;

histPlane = 1; % adapthisteq for imaging plane. 0 or 1
histStack = 0; % adapthisteq for z-stack mean image. 0 or 1
histMatch = 1; % imhistmatch for imaging plane to z-stack mean image. 0 or 1

objz = 90; % z value during imaging, in microns
% depthEst = objz / cosd(35); % estimated depth of the imaging plane
depthEst = 132; % or use manual correction

sessionTemp = strsplit(sessions(si).name, '_');
if floor(sessionNames(sni)/1000) == 5
    titleText = [sessionTemp{1}, ' spont ', sessionTemp{3}, ' plane #', sessionTemp{5}, ' (', regMeth, ')'];
elseif floor(sessionNames(sni)/1000) == 9
    titleText = [sessionTemp{1}, ' passive ', sessionTemp{3}, ' plane #', sessionTemp{5}, ' (', regMeth, ')'];
else
    error('Something''s wrong.')
end
% load mean image

load([sessions(si).folder, filesep, sessions(si).name], 'mimg');
mimg = mat2gray(mimg);


% Basic setting for each plane
svThickness = 150; % in micron
offsetDepth = 0; % in micron
imMargin = 50; % for z-stack

% xReg = size(mimg,2);
% yReg = size(mimg,1);
if mi < 6
    xReg = 600; % for mimg of each session
    yReg = 350; % for mimg of each session
elseif mi == 6
    if planei == 1 && si <15
        xReg = 500; % for mimg of each session
        yReg = 350; % for mimg of each session
    else
        xReg = 600; % for mimg of each session
        yReg = 450; % for mimg of each session
    end
elseif mi < 9
    xReg = 600; % for mimg of each session
    yReg = 450; % for mimg of each session
else
    xReg = 600; % for mimg of each session
    yReg = 350; % for mimg of each session
end

split = [2, 2]; % Number of division in rows and columns (in this order)
numSplit = prod(split);

planeDepth = depthEst + offsetDepth; 
svtopDepth = planeDepth - svThickness/2;
[~, svtopPlaneInd] = min(abs(zstackDepths - svtopDepth));
svbottomDepth = planeDepth + svThickness/2;
[~, svbottomPlaneInd] = min(abs(zstackDepths - svbottomDepth));

if histPlane == 1
    mimg = adapthisteq(mimg);
end
mimg = mimg(round(size(mimg,1)/2 - yReg/2) : round(size(mimg,1)/2 + yReg/2), ...
    round(size(mimg,2)/2 - xReg/2) : round(size(mimg,2)/2 + xReg/2));

svRaw = zstackReg(imMargin+1:end-imMargin,imMargin+1:end-imMargin,svbottomPlaneInd:svtopPlaneInd);
if mi == 2
    svRaw = svRaw(100:end, 70:end-80, :);
    mimg = mimg(1:end-50, :);
end
if mi == 3 && planei < 5
    svRaw = svRaw(120:end-50, 150:end-80, :);
end
if mi == 3 && planei >= 5
    svRaw = svRaw(120:end-30, 130:end-150, :);
end
if mi == 4
    svRaw = svRaw(100:end-50, 120:end-70, :);
    if planei >= 5 && si >= 20
        svRaw = svRaw(80:end, 100:end-100, :);
        mimg = mimg(50:end-50, 80:end-50);
    end
end
if mi == 5 
    if si == 1
        svRaw = svRaw(300:end, 150:end-150, :);
    elseif si >= 2
        svRaw = svRaw(250:end-50, 180:end-150, :);
    end
end
if mi == 6
%     svRaw = svRaw(200:end-30, 120:end-100,:);
    svRaw = svRaw(130:end, 70:end-50,:);
end

if mi == 7
    svRaw = svRaw(150:end-80, 70:end-70, :);
    mimg = mimg(1:end-100, :);
end

if mi == 8
    if planei <5
        svRaw = svRaw(150:end-70,80:end-50, :);
    else 
        svRaw = svRaw(150:end-70,80:end-50, :);
    end
    
end

if mi == 9
    if planei < 5
        svRaw = svRaw(130:end, 30:end-30, :);
    else 
        svRaw = svRaw(130:end, 50:end-150, :);
    end
end

if mi == 10
    if planei < 5
        svRaw = svRaw(100:end, 180:end-30, :);
    else 
        svRaw = svRaw(100:end, 180:end-30, :);
    end
end

if mi == 11
    if planei < 5
        svRaw = svRaw(70:end-20, 50:end-30, :);
    else 
%         svRaw = svRaw(100:end, 180:end-30, :);
    end
end

if mi == 12
    if planei < 5
        svRaw = svRaw(30:end, 30:end-100, :);
        mimg = mimg(:,1:end-40);
    else 
        svRaw = svRaw(30:end, 30:end-160, :);
        mimg = mimg(:,1:end-100);
    end
end

if histStack == 0
    svMean = (mean(svRaw,3));
else
    svMean = adapthisteq(mean(svRaw,3));
end

if histMatch == 1
    mimg = imhistmatch(mimg, svMean); % option
end
% %%
% figure,
% subplot(121), imshow(mimg), title('Session image'), axis image
% subplot(122), imshow(svMean), title('Mean subvolume'), axis image

% %% First, make a general registration 
% %% and then calculate loose boundary for each split FOV

tform = imregtform(mimg, svMean, regMeth, tformOpt, metric);
moved = imwarp(mimg, tform, 'OutputView', imref2d(size(svMean)));

splitIm = cell(numSplit,1); % Split image of the session mean image
splitFOVreg = cell(numSplit,1); % Registered FOV of split images. For test
splitFOVzstack = cell(numSplit,1); % FOV of template z-stack planes for each split.
                                            % This is going to be the template to match later.
y = size(mimg,1);
rows = split(1);
x = size(mimg,2);
cols = split(2);
for ri = 1 : rows
    for ci = 1 : cols
        tempRows = (ri-1)*round(y/rows)+1 : min(ri*round(y/rows), y);
        tempCols = (ci-1)*round(x/cols)+1 : min(ci*round(x/cols), x);
        tempMat = zeros(y,x);
        tempMat(tempRows, tempCols) = 1;
        splitMat = mimg .* tempMat;
        splitImTemp = zeros(length(tempRows), length(tempCols));
        nzInd = find(tempMat);
        splitImTemp(:) = splitMat(nzInd);
        splitIm{(ri-1)*split(2)+ci} = splitImTemp;
        splitReg = imwarp(tempMat, tform, 'OutputView', imref2d(size(svMean)));
        splitFOVreg{(ri-1)*split(2)+ci} = splitReg;
        tempReg = zeros(size(splitReg));
        tempinds = find(splitReg > 0.8); % 0.8 as an arbitrary threshold
        tempReg(tempinds) = 1;
        ymin = find(sum(tempReg,2),1,'first');
        ymax = find(sum(tempReg,2),1,'last');
        xmin = find(sum(tempReg),1,'first');
        xmax = find(sum(tempReg),1,'last');
        % Dilate into a rectangle 
        % Factor of 0.2 to each side. 0.2 is arbitrarily set.
        newymin = max(round(ymin-(ymax-ymin)*0.2),1);
        newymax = min(round(ymax+(ymax-ymin)*0.2),size(splitReg,1));
        newxmin = max(round(xmin-(xmax-xmin)*0.2),1);
        newxmax = min(round(xmax+(xmax-xmin)*0.2),size(splitReg,2));
        tempMat = zeros(size(splitReg));
        tempMat(newymin:newymax, newxmin:newxmax) = 1;
        splitFOVzstack{(ri-1)*split(2)+ci} = tempMat;
    end
end

% %% confirm the quality of registration visually
figure, 
subplot(211), imshowpair(moved, svMean, 'montage')
title(sprintf('histPlane = %d; histStack = %d; histMatch = %d', histPlane, histStack, histMatch))
subplot(212), imshowpair(moved, svMean) 
if tformOpt.GradientMagnitudeTolerance == opt__2.GradientMagnitudeTolerance
    title('tform opt2')
else
    title('tform opt1')
end
sgtitle(titleText)

% %% Show correlation & estimate the peak
% % Polynomial fit is sensitive to x-y pair. Don't use it.
% % Instead, use smoothing with window similar to SPF and spline
% % interpolation in 1 um resolution.
smoothWindow = 10;

corrVals = zeros(size(svRaw,3),1);
nzind = find(moved);
for i = 1 : size(svRaw,3)
    currIm = squeeze(svRaw(:,:,i));
    nzindSv = find(currIm);
    nzind = intersect(nzind, nzindSv);
    corrVals(i) = corr(moved(nzind), currIm(nzind));
end

x = zstackDepths(svbottomPlaneInd:svtopPlaneInd);
xx = round(zstackDepths(svtopPlaneInd)):round(zstackDepths(svbottomPlaneInd));
s = smooth(corrVals,smoothWindow);
sp = spline(x,s, xx);
[~, maxind] = max(sp);
maxD = xx(maxind);

figure, hold on
plot(zstackDepths(svbottomPlaneInd:svtopPlaneInd),corrVals, 'color', [0.6 0.6 0.6])
plot(xx, sp, 'r')
legend({'Raw', 'Smooth'}, 'location', 'northwest')
title(sprintf('%s, max %d um (smoothing %d\\mum)',titleText, round(maxD), smoothWindow*2))
xlabel('Depth (um)')
ylabel('Correlation')









%%
%% In case of split FOV
%%

% options
histFixed = 1; % 0 or 1. 

% confirm splitFOVzstack visually
% sumRegIm = zeros(size(svMean));
% sumFOVim = zeros(size(svMean));
% for i = 1 : length(splitFOVzstack)
%     sumRegIm = sumRegIm + splitFOVreg{i}*i;
%     sumFOVim = sumFOVim + splitFOVzstack{i}*i;
% end

% figure, 
% subplot(121), imagesc(sumRegIm), axis image
% subplot(122), imagesc(sumFOVim), axis image
% %% Register to mean subvolume in each split FOV

tformRefNaive = cell(1,numSplit);
splitImNaive = cell(1,numSplit);
sumTest = zeros(size(svMean));
for si = 1 : numSplit
    ymin = find(sum(splitFOVzstack{si},2),1,'first');
    ymax = find(sum(splitFOVzstack{si},2),1,'last');
    xmin = find(sum(splitFOVzstack{si}),1,'first');
    xmax = find(sum(splitFOVzstack{si}),1,'last');
    fixed = zeros(ymax-ymin+1, xmax-xmin+1);
    nzInd = find(splitFOVzstack{si});
    fixed(:) = svMean(nzInd);
    if histFixed ==1
        fixed = adapthisteq(fixed); % option
    end
    moving = adapthisteq(splitIm{si});
%     moving = splitIm{si};
%     moving = imhistmatch(splitIm{si}, fixed);

    tformRefNaive{si} = imregtform(moving, fixed, regMeth, tformOpt, metric);
    tempTestIm = imwarp(splitIm{si}, tformRefNaive{si}, 'OutputView', imref2d(size(fixed)));
    testIm = zeros(size(svMean));
    testIm(ymin:ymax, xmin:xmax) = tempTestIm;
    splitImNaive{si} = testIm;
    sumTest = sumTest + testIm;
end
% %%
% QC
figure, imshowpair(imhistmatch(sumTest,svMean),svMean)
title(sprintf('%s; histFixed = %d ',titleText, histFixed))


%% Calculate correaltion in each subvolume
% And estimate peak in each subvolume using smoothing and spline
corrSplit = zeros(size(svRaw,3),numSplit);
for si = 1 : numSplit
    nzInd = find(splitImNaive{si});
    pixVal = splitImNaive{si}(nzInd);
    for pi = 1 : size(svRaw,3)
        planeIm = squeeze(svRaw(:,:,pi));
        planePixVal = planeIm(nzInd);
        corrSplit(pi,si) = corr(pixVal,planePixVal);
    end
end

figure, hold on
for si = 1 : numSplit
    plot(zstackDepths(svbottomPlaneInd:svtopPlaneInd), corrSplit(:,si), 'color', [0.5 0.5 0.5])
end
pmean = plot(zstackDepths(svbottomPlaneInd:svtopPlaneInd), mean(corrSplit,2), 'k');


x = zstackDepths(svbottomPlaneInd:svtopPlaneInd);
xx = round(zstackDepths(svtopPlaneInd)):round(zstackDepths(svbottomPlaneInd));
s = smooth(mean(corrSplit,2),smoothWindow);
sp = spline(x,s, xx);
[~, maxind] = max(sp);
maxD = xx(maxind);

ps = plot(xx, sp, 'r');

legend([pmean, ps], {'Mean', 'Fit'})
title(sprintf('%s, max %d um',titleText, round(maxD)))
xlabel('Depth (\mum)')
ylabel('Correlation')

%%
% %% In case of a sub-split error, select split indices
splitInd = [2,3,4];
corrSplitSelectNaive = zeros(size(svRaw,3), length(splitInd));
for spi = 1 : length(splitInd)
    corrSplitSelectNaive(:,spi) = corrSplit(:,splitInd(spi));
end

figure, hold on
for spi = 1 : length(splitInd)
    plot(zstackDepths(svbottomPlaneInd:svtopPlaneInd), corrSplitSelectNaive(:,spi), 'color', [0.5 0.5 0.5])
end
pmean = plot(zstackDepths(svbottomPlaneInd:svtopPlaneInd), mean(corrSplitSelectNaive,2), 'k');


x = zstackDepths(svbottomPlaneInd:svtopPlaneInd);
xx = round(zstackDepths(svtopPlaneInd)):round(zstackDepths(svbottomPlaneInd));
s = smooth(mean(corrSplitSelectNaive,2),3);
sp = spline(x,s, xx);
[~, maxind] = max(sp);
maxD = xx(maxind);

ps = plot(xx, sp, 'r');

legend([pmean, ps], {'Mean', 'Fit'})
title(sprintf('%s, max %d um; splits = %s', titleText, round(maxD), num2str(splitInd)))
xlabel('Depth (um)')
ylabel('Correlation')






