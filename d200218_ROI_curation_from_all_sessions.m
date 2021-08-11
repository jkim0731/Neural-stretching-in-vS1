%% 
% ROI curation from all sessions
% Use all frames from all comparing sessions together

%% 

baseDir = 'D:\TPM\JK\suite2p\';
mice = [25,27,30,36,37,38,39,41,52,53,54,56];
% mainSessions = [4,3,3,1,7,2,1,3,3,3,3,3];
curationSessions = {[2:3,5:18], [1:2,4:7], [1:2,4:7,9:20], [2:16], [1:6,8:10,12:24], [1,3:22,24:31], [2:21], [1,2,4:19,21:30], [1,2,5:20], [1,2,5:15,17:21], [1,2,5:14,16:26], [1,2,6:13]};
planes = 1 : 8;

