% Testing w3 and wf difference in arcLength
% Examples from 221024_build_dataset.ipynb
% 2023/01/06 JK

w_dir = 'D:\WhiskerVideo\';
mouse = 25;
session = 16;

session_dir = sprintf('%sJK%03dS%02d', w_dir, mouse, session);

trial_nums_error = [3,11,13,138,252,289,290,330,452];
trial_nums_correct = trial_nums_error -1;

time_diff = zeros(length(trial_nums_error),1);
arcLength_diff = zeros(length(trial_nums_error),1);
for ti = 1:length(trial_nums_error)
    tnum = trial_nums_error(ti);
    fn_w3 = [num2str(tnum), '_W3_2pad.mat'];
    fn_wf = [num2str(tnum), '_WF_2pad.mat'];
    load([session_dir, '\', fn_w3], 'w3')
    load([session_dir, '\', fn_wf], 'wf')
    time_diff(ti) = length(wf.time) - length(w3.time);
    arcLength_diff(ti) = length(wf.arcLength) - length(w3.lengthAlongWhisker);
end
print(time_diff)
print(arcLength_diff)