h5_dir = 'H:\';
mice = [25,27,30];
num_nan_tn_list = [];
tn_list = {};
for mi=1:length(mice)
    mouse = mice(mi);
    mouse_dir = sprintf('%s%03d\\',h5_dir,mouse);
    trial_list = dir(sprintf('%s*.trials',mouse_dir));
    for ti = 1:length(trial_list)
        trial_fn = trial_list(ti).name;
        trial_cell = strsplit(trial_fn, '_');
        session_num = str2double(trial_cell{2});
        if session_num < 100
            tn_list = [tn_list, trial_fn];
            trials = load([mouse_dir, trial_fn], '-mat');
            trial_nums = zeros(length(trials.trials),1);
            for i =1:length(trials.trials)
                trial_nums(i) = trials.trials(i).trialnum;
            end
            num_nan_tn = length(find(isnan(trial_nums)));
            num_nan_tn_list = [num_nan_tn_list, num_nan_tn];
        end
    end
end
            
%%
h5_dir = 'F:\';
mice = [36,39,52];
num_nan_tn_list = [];
tn_list = {};
for mi=1:length(mice)
    mouse = mice(mi);
    mouse_dir = sprintf('%s%03d\\',h5_dir,mouse);
    trial_list = dir(sprintf('%s*.trials',mouse_dir));
    for ti = 1:length(trial_list)
        trial_fn = trial_list(ti).name;
        trial_cell = strsplit(trial_fn, '_');
        session_num = str2double(trial_cell{2});
        if session_num < 100
            tn_list = [tn_list, trial_fn];
            trials = load([mouse_dir, trial_fn], '-mat');
            trial_nums = zeros(length(trials.trials),1);
            for i =1:length(trials.trials)
                trial_nums(i) = trials.trials(i).trialnum;
            end
            num_nan_tn = length(find(isnan(trial_nums)));
            num_nan_tn_list = [num_nan_tn_list, num_nan_tn];
        end
    end
end