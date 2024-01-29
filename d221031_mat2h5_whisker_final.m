% Save whisker final files to h5-readable files.
% (\WhiskerVideo\JK0xxSyy\zzz_WF_2pad.mat -> \WhiskerVideo\JK0xxSyy\zzz_WF_h5.mat)
% Add whisking amplitude, midpoint, phase, and whiskingStartFrames

% 12 mice takes about an hour.

%%
%% Single session version
%%

w_dir = 'E:\TPM\JK\WhiskerVideo\';
mouse = 36;
session = 1;
session_attr = {'mouseName', 'sessionName'};
trial_attr = {'trialNum', 'poleAngle', 'polePosition', 'poleDistance', 'time', 'poleUpFrames', 'poleMovingFrames', ...
    'protractionTFchunks', 'protractionTFchunksByWhisking', 'retractionTFchunks', 'retractionTFchunksByWhisking', ...
    'protractionTouchDuration', 'protractionTouchDurationByWhisking', 'retractionTouchDuration', 'retractionTouchDurationByWhisking', ...
    'protractionSlide', 'protractionSlideByWhisking', 'retractionSlide', 'retractionSlideByWhisking',...
    'theta', 'phi', 'kappaH', 'kappaV', 'arcLength'};

whiskingAmpThreshold = 2.5; % in degrees

session_dir = sprintf('JK%03dS%02d', mouse, session);
sprintf('Processing %s',session_dir);

save_fn = [session_dir, '_whisker_final_h5.mat'];
save_fp = [w_dir, session_dir, '/', save_fn];

wf_list_pre = dir([w_dir, session_dir, '/', '*_WF_2pad.mat']);

wf_list_pre = dir([w_dir, session_dir, '/', '*_WF_2pad.mat']);
if isempty(wf_list_pre) == 0 % only when there are WF files
    % sort WF filename
    wf_num = zeros(length(wf_list_pre),1);
    for i = 1 : length(wf_list_pre)
        tnum_str = strsplit(wf_list_pre(i).name,'_');
        wf_num(i) = str2double(tnum_str{1});
    end
    wf_num = sort(wf_num);
    % Gather data
    for ti = 1 : length(wf_num)
        tnum = wf_num(ti);
        wfn = [num2str(tnum), '_WF_2pad.mat'];
        data = load([w_dir, session_dir, '/', wfn], 'wf');
        if ti == 1
            hw.mouse_name = data.wf.mouseName;
            hw.session_name = data.wf.sessionName;
        end
        for j = 1 : length(trial_attr)
            hw.trials{ti}.(trial_attr{j}) = data.wf.(trial_attr{j});
            [~, amplitude, ~, midpoint, ~, ~, phase, ~] = jkWhiskerDecomposition(data.wf.theta);
            potentialWhiskingStartInds = [1;find([0;diff(phase)]< -pi); length(phase)+1];
            % pi (~3.41) is defined as the threshold by observing the histogram of diff(phase)
            whiskingStartFrames= [];
            if ~isempty(potentialWhiskingStartInds)
                for i = 1 : length(potentialWhiskingStartInds)-1
                    if max(amplitude(potentialWhiskingStartInds(i):potentialWhiskingStartInds(i+1)-1)) > whiskingAmpThreshold
                        whiskingStartFrames = [whiskingStartFrames, potentialWhiskingStartInds(i)];
                    end
                end
            end
            hw.trials{ti}.amplitude = amplitude;
            hw.trials{ti}.midpoint = midpoint;
            hw.trials{ti}.phase = phase;
            hw.trials{ti}.whiskingStartFrames = whiskingStartFrames;
        end
    end
    % Save data
    save(save_fp, 'hw', '-v7.3')
end


%%
%% Running multiple sessions
%%


w_dir = 'D:\WhiskerVideo\';
mice = [25,27,30,36,37,38,39,41,52,53,54,56];
% mice = [52];
session_attr = {'mouseName', 'sessionName'};
trial_attr = {'trialNum', 'poleAngle', 'polePosition', 'poleDistance', 'time', 'poleUpFrames', 'poleMovingFrames', ...
    'protractionTFchunks', 'protractionTFchunksByWhisking', 'retractionTFchunks', 'retractionTFchunksByWhisking', ...
    'protractionTouchDuration', 'protractionTouchDurationByWhisking', 'retractionTouchDuration', 'retractionTouchDurationByWhisking', ...
    'protractionSlide', 'protractionSlideByWhisking', 'retractionSlide', 'retractionSlideByWhisking',...
    'theta', 'phi', 'kappaH', 'kappaV', 'arcLength'};

whiskingAmpThreshold = 2.5; % in degrees

for mi = 1 : length(mice)
    mouse = mice(mi);
    sprintf('Processing JK%03d (%d/%d)', mouse, mi, length(mice))
    sessions_dn = dir(sprintf('%sJK%03dS*',w_dir,mouse));
    parfor si = 1 : length(sessions_dn)
%     for si = 32
        session_dir = sessions_dn(si).name;
        sprintf('Processing %s',session_dir);
        hw = {};
        hw.trials = {};        
        if isfolder(sprintf('%s%s',w_dir,session_dir))
            % Check if processed already
            save_fn = [session_dir, '_whisker_final_h5.mat'];
            save_fp = [w_dir, session_dir, '/', save_fn];
%             if isfile(save_fp) == 0
                wf_list_pre = dir([w_dir, session_dir, '/', '*_WF_2pad.mat']);
                if isempty(wf_list_pre) == 0 % only when there are WF files
                    % sort WF filename
                    wf_num = zeros(length(wf_list_pre),1);
                    for i = 1 : length(wf_list_pre)
                        tnum_str = strsplit(wf_list_pre(i).name,'_');
                        wf_num(i) = str2double(tnum_str{1});
                    end
                    wf_num = sort(wf_num);
                    % Gather data
                    for ti = 1 : length(wf_num)
                        tnum = wf_num(ti);
                        wfn = [num2str(tnum), '_WF_2pad.mat'];
                        data = load([w_dir, session_dir, '/', wfn], 'wf');
                        if ti == 1
                            hw.mouse_name = data.wf.mouseName;
                            hw.session_name = data.wf.sessionName;
                        end
                        for j = 1 : length(trial_attr)
                            hw.trials{ti}.(trial_attr{j}) = data.wf.(trial_attr{j});
                            [~, amplitude, ~, midpoint, ~, ~, phase, ~] = jkWhiskerDecomposition(data.wf.theta);
                            potentialWhiskingStartInds = [1;find([0;diff(phase)]< -pi); length(phase)+1];
                            % pi (~3.41) is defined as the threshold by observing the histogram of diff(phase)
                            whiskingStartFrames= [];
                            if ~isempty(potentialWhiskingStartInds)
                                for i = 1 : length(potentialWhiskingStartInds)-1
                                    if max(amplitude(potentialWhiskingStartInds(i):potentialWhiskingStartInds(i+1)-1)) > whiskingAmpThreshold
                                        whiskingStartFrames = [whiskingStartFrames, potentialWhiskingStartInds(i)];
                                    end
                                end
                            end
                            hw.trials{ti}.amplitude = amplitude;
                            hw.trials{ti}.midpoint = midpoint;
                            hw.trials{ti}.phase = phase;
                            hw.trials{ti}.whiskingStartFrames = whiskingStartFrames;
                        end
                    end
                    % Save data
                    save_h5_parfor(save_fp, hw);
                end
%             end
        end
        
    end
end

function save_h5_parfor(file_path, hw)    
    save(file_path, 'hw', '-v7.3')
end
