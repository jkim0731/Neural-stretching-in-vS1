% Test jksbxreadframes

baseDir = 'P:\038\';
fn = '038_017_000';
t = importdata([fn,'.trials']);
testFrames = t.frame_to_use{1}(1:4000);





tic
z = sbxread(fn,0,1);
imRef = zeros([size(z,2), size(z,3),length(testFrames)]);
for i = 1 : length(testFrames)
    z = sbxread(fn, testFrames(i),1);
    imRef(:,:,i) = squeeze(z(1,:,:,:));
end
toc


tic
imTest = squeeze(jksbxreadframes(fn,testFrames,1));
toc


% z = sbxread(fname,0,1);
% z = zeros([size(z,2) size(z,3) length(frames)]);

% for i = 1:length(frames)
%     temp = sbxread(fn,frames(i),1);
%     z(:,:,i) = squeeze(temp(1,:,:));
% end

%%
baseDir = 'D:\TPM\JK\h5\038\';
fn = '038_016_000';
plane = 6;
data = h5read([baseDir,'plane_', num2str(plane), filesep, fn, '_plane_', num2str(plane),'.h5'], '/data');

%%
t = importdata([fn,'.trials']);
framei = 2081;
frame = t.frame_to_use{plane}(framei);

refx = sbxread(fn,frame,1);
ref = squeeze(refx(1,:,100:end-10,:));

test = data(:,:,framei);

compare(test,ref)