% 2021/11/23
% Fix problems with spontaneous sessions of JK036, 038, 039
% These have wrong allocation of frames into planes.
% Planes 2 and 4 (6 and 8) are OK, but 1 and 3 (5 and 7) are swapped.
% The order was wrong: Instead of 4->3->2->1, they were 1->2->3->4.
% 
% First, check info. Maybe the info was saved in a wrong way.
% Then, try running jksbxsplittiral_4h5c again.
%     
%% BS
sbxDir = 'J:\';
mouse = 39;
tempDir = 'C:\JK\temp\';

%% Check info file
sessionName = '025_000';
info1 = load(sprintf('%s%03d\\%03d_%s.mat',sbxDir, mouse, mouse, sessionName));
%%
sessionName = '016_000';
info2 = load(sprintf('%s%03d\\%03d_%s.mat',sbxDir, mouse, mouse, sessionName));

% Doesn't seem there is any wrong with info file.
% Try running jksbxsplittiral_4h5c again.

%% running jksbxsplittiral_4h5c again.
sessionName = '025_000';
fn = sprintf('%s%03d\\%03d_%s',sbxDir, mouse, mouse, sessionName);
laserOnFrames = laser_on_frames_4h5c(fn);
jksbxsplittrial_4h5c(fn, laserOnFrames)

%% Check the result
trials = load('039_025_000.trials', '-mat');

% Correct. Just run them all again.



%% 
sbxDir = 'J:\';
mouse = 36;
sessionNames = {'5554_001','5554_010', '5554_011', '5554_101', '5554_110', '5554_111', ...
    '5555_001','5555_010', '5555_011', '5555_101', '5555_110', '5555_111'};
for si = 1 : length(sessionNames)
    sname = sessionNames{si};
    fn = sprintf('%s%03d\\%03d_%s',sbxDir, mouse, mouse, sname);
    laserOnFrames = laser_on_frames_4h5c(fn);
    jksbxsplittrial_4h5c(fn, laserOnFrames)
end

%% 
