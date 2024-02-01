
mice = [25, 27, 30, 36, 39, 52];
sessions_all_mice = {[1,2,3,4,5,6,7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    [1,2,3,4,5,6,7,8,9,10,12,13,15,16],
    [1,2,3,5,6,7,9,11,12,14,15,17,18,19,20,21],
    [1,2,3,4,5,6,7,8,9,10,13,14,15,16,17],
    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22,23],
    [1,3,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21]};
errors = {};
for mi = 1 : length(mice)
    mouse = mice(mi);
    sessions = sessions_all_mice{mi};
    for si = 1 : length(sessions)
        session = sessions(si);
        mouseName = sprintf('JK%03d', mouse);
        sessionName = sprintf('S%02d', session);
        
        save_fn = sprintf('%s%s_touch_qc.mat',mouseName, sessionName);
        save_path = sprintf('%sTouchQC\\%s', wDir, save_fn);
        if exist(save_path, 'file') == 2
            continue
        else
            try
                wDir = 'E:\TPM\JK\WhiskerVideo\';
                sdir = [wDir,mouseName,sessionName];
                
                cd(sdir)
                hpFn = [mouseName, sessionName, '_touch_hp.mat'];
                load(hpFn)
                
                wsa = Whisker.WhiskerSignalTrialArray_2pad(sdir);
        
                scatter2d_points = cell(1,length(servo_distance_pair));
                touchhp_2d = cell(1,length(servo_distance_pair));
                scatter3d_points = cell(1,length(servo_distance_pair));
        
                for sdi = 1 : length(servo_distance_pair)
               
                    angle = servo_distance_pair{sdi}(1);
                    dist = servo_distance_pair{sdi}(2);
                    tinds = find(cellfun(@(x) x.angle == angle && x.radialDistance == dist, wsa.trials));
                    points = [];
                    for j = 1 : length(tinds)
                        frames = wsa.trials{tinds(j)}.poleUpFrames;
                        points = [points, [wsa.trials{tinds(j)}.whiskerEdgeCoord(frames,1)'; wsa.trials{tinds(j)}.whiskerEdgeCoord(frames,2)'; ones(1,length(frames))*wsa.trials{tinds(j)}.apUpPosition]];
                    end
                
                    if psi1(sdi) > 90
                        A = viewmtx(psi1(sdi),-90+psi2(sdi));
                    else
                        A = viewmtx(psi1(sdi),90-psi2(sdi));
                    end
                    points_4d = [points; ones(1,size(points,2))];
                    points_2d = A*points_4d;
                    points_2d = unique(round(points_2d(1:2,:)',2),'rows');
                
                    th_4d1 = [touch_hp{sdi}(1,:) + hp_peaks{sdi}(1)     + 0;    touch_hp{sdi}(2:3,:);ones(1,size(touch_hp{sdi},2))];
                    th_2d1 = A*th_4d1;
                    th_2d1 = unique(th_2d1(1:2,:)','rows');
                    th_4d2 = [touch_hp{sdi}(1,:) + hp_peaks{sdi}(2)     + 0;    touch_hp{sdi}(2:3,:);ones(1,size(touch_hp{sdi},2))];
                    th_2d2 = A*th_4d2;
                    th_2d2 = unique(th_2d2(1:2,:)','rows');
        
        
                    scatter2d_points{sdi} = points_2d;
                    touchhp_2d{sdi} = {th_2d1, th_2d2};
                    % scatter(points_2d(:,1),points_2d(:,2),'k.'), hold on, scatter(th_2d1(:,1), th_2d1(:,2),'r.'), scatter(th_2d2(:,1), th_2d2(:,2),'r.')
                    
                    scatter3d_points{sdi} = points;
        
                    % plot3(points(1,:), points(2,:), points(3,:), 'k.'), hold on
                    % plot3(touch_hp{sdi}(1,:) + hp_peaks{sdi}(1), touch_hp{sdi}(2,:), touch_hp{sdi}(3,:), 'r-')
                    % plot3(touch_hp{sdi}(1,:) + hp_peaks{sdi}(2), touch_hp{sdi}(2,:), touch_hp{sdi}(3,:), 'r-')
                    
                end
            catch
                errors = [errors, sprintf('%s%s',mouseName, sessionName)];
            end
            
            save(save_path, 'scatter2d_points', 'touchhp_2d', 'scatter3d_points', 'touch_hp', 'hp_peaks')
        end
    end
end

%%
%% Look at one error session, JK036S02
%%
mouse = 36;
session = 2;
mouseName = sprintf('JK%03d', mouse);
sessionName = sprintf('S%02d', session);

save_fn = sprintf('%s%s_touch_qc.mat',mouseName, sessionName);
save_path = sprintf('%sTouchQC\\%s', wDir, save_fn);

wDir = 'E:\TPM\JK\WhiskerVideo\';
sdir = [wDir,mouseName,sessionName];

cd(sdir)
hpFn = [mouseName, sessionName, '_touch_hp.mat'];
load(hpFn)

wsa = Whisker.WhiskerSignalTrialArray_2pad(sdir);

scatter2d_points = cell(1,length(servo_distance_pair));
touchhp_2d = cell(1,length(servo_distance_pair));
scatter3d_points = cell(1,length(servo_distance_pair));
%%
% servo_distance_pair has 4 values.
% including 60 and 90 degrees
% Behavior note says toss the first 2 trials, 
% because they have 60 and 90 degrees.
% wsa starts with trial 4, so this must have been tossed already.
angles = [];
for i = 1:length(wsa.trials)
    angles = [angles, wsa.trials{i}.angle];
end

unique(angles)
% returns 45 and 135.
% so just remove those 2 from servo_distance_pair.
% and save it again.
% also remove those from hp_peaks, num_points_in_hp, steps_hp, touch_hp
% in the same index
% they were blank
%%
take_inds = [1,4];
touch_hp = touch_hp(take_inds);
hp_peaks = hp_peaks(take_inds);
num_points_in_hp = num_points_in_hp(take_inds);
steps_hp = steps_hp(take_inds);
servo_distance_pair = servo_distance_pair(take_inds);
thPolygon = thPolygon(take_inds);
%%
save(hpFn, 'touch_hp', 'hp_peaks', 'thPolygon', 'num_points_in_hp', 'steps_hp', 'servo_distance_pair', 'psi1', 'psi2')


%%
%% Another error session JK036 S01 -> S81 for the first 245 trials (trialNum 2-246)
mouse = 36;
session = 81;
mouseName = sprintf('JK%03d', mouse);
sessionName = sprintf('S%02d', session);

save_fn = sprintf('%s%s_touch_qc.mat',mouseName, sessionName);
save_path = sprintf('%sTouchQC\\%s', wDir, save_fn);

wDir = 'E:\TPM\JK\WhiskerVideo\';
sdir = [wDir,mouseName,sessionName];

cd(sdir)
hpFn = [mouseName, sessionName, '_touch_hp.mat'];
load(hpFn)

wsa = Whisker.WhiskerSignalTrialArray_2pad(sdir);

scatter2d_points = cell(1,length(servo_distance_pair));
touchhp_2d = cell(1,length(servo_distance_pair));
scatter3d_points = cell(1,length(servo_distance_pair));