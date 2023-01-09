% Save behavior files (e.g., SoloData/JK0xx/behavior_JK0xx.mat) to .h5
% files (e.g., SoloData/JK0xx/behavior_JK0xx_h5.mat) using -v7.3
% From each mouse.
% Collate all sessions.
% Data to save:
%     b{x}.mouseName
%     b{x}.sessionName
%     b{x}.sessionType
%     b{x}.taskTarget
%     b{x}.distractor
%     b{x}{y}.trialNum
%     b{x}{y}.extraITIOnErrorSetting
%     b{x}{y}.motorDistance
%     b{x}{y}.motorApPosition
%     b{x}{y}.servoAngle
%     b{x}{y}.beamBreakTimesLeft
%     b{x}{y}.beamBreakTimesRight
%     b{x}{y}.rewardTimeLeft
%     b{x}{y}.rewardTimeRight
%     b{x}{y}.poleUpOnsetTime
%     b{x}{y}.poleDownOnsetTime
%     b{x}{y}.samplingPeriodTime
%     b{x}{y}.answerPeriodTime
%     b{x}{y}.answerLickTime
%     b{x}{y}.drinkingTime

b_dir = 'D:\SoloData\';
mice = [25,27,30,36,37,38,39,41,52,53,54,56];
%%
for mi = 1:length(mice)
    mouse = mice(mi);
    b_file = sprintf('%sJK%03d\\behavior_JK%03d.mat',b_dir,mouse,mouse);
    b = load(b_file, 'b');
    %%
    h5_file = sprintf('%sJK%03d\\behavior_JK%03d.h5',b_dir,mouse,mouse);

    %%
    session_attr = {'sessionName', 'sessionType', 'taskTarget', 'distractor'};
    trial_attr = {'trialNum', 'trialType', 'choice', 'extraITIOnErrorSetting', 'motorDistance', 'motorApPosition', ...
        'servoAngle', 'beamBreakTimesLeft', 'beamBreakTimesRight', 'rewardTimeLeft', 'rewardTimeRight', ...
        'poleUpOnsetTime', 'poleDownOnsetTime', 'samplingPeriodTime', 'answerPeriodTime', ...
        'answerLickTime', 'drinkingTime'};

    %%
    hb = {};
    hb.mouse_name = b.b{1}.mouseName;
    hb.sessions = {};
    for i = 1 : length(b.b)
        for i_att = 1 : length(session_attr)
            hb.sessions{i}.(session_attr{i_att}) = b.b{i}.(session_attr{i_att});
        end
        hb.sessions{i}.trials = {};
        for j = 1 : length(b.b{i}.trials)
            for j_att = 1 : length(trial_attr)
                hb.sessions{i}.trials{j}.(trial_attr{j_att}) = b.b{i}.trials{j}.(trial_attr{j_att});
            end
        end
    end

    %%
    save_fn = sprintf('%sJK%03d\\behavior_JK%03d_h5.mat',b_dir,mouse,mouse);
    save(save_fn, 'hb', '-v7.3')
end