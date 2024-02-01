qc_dir = 'E:\TPM\JK\WhiskerVideo\TouchQC\';
mice = [25, 27, 30, 36, 39, 52];
sessions_all_mice = {[1,2,3,4,5,6,7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    [1,2,3,4,5,6,7,8,9,10,12,13,15,16],
    [1,2,3,5,6,7,9,11,12,14,15,17,18,19,20,21],
    [1,2,3,4,5,6,7,8,9,10,13,14,15,16,17],
    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22,23],
    [1,3,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21]};
for mi = 1 : length(mice)
    mouse = mice(mi);
    sessions = sessions_all_mice{mi};
    for si = 1 : length(sessions)
        session = sessions(si);
        mouseName = sprintf('JK%03d', mouse);
        sessionName = sprintf('S%02d', session);

        load_fn = [qc_dir, sprintf('%s%s_touch_qc.mat',mouseName,sessionName)];
        load(load_fn) % hp_peaks, scatter2d_points, scatter3d_points, touch_hp, touchhp_2d

        figure('units','normalized','position', [0 0 1 1])
        for sdi = 1:2
            subplot(1,2,sdi)
            angle = servo_distance_pair{sdi}(1);
            dist = servo_distance_pair{sdi}(2);
            points_2d = scatter2d_points{sdi};
            th_2d1 = touchhp_2d{sdi}{1};
            th_2d2 = touchhp_2d{sdi}{2};
            
            scatter(points_2d(:,1),points_2d(:,2),'k.'), hold on, scatter(th_2d1(:,1), th_2d1(:,2),'r.'), scatter(th_2d2(:,1), th_2d2(:,2),'r.')
            title(['Angle = ', num2str(angle), ', Dist = ', num2str(dist)])
        end
        suptitle([mouseName, ' ' , sessionName])
        save_fn = [qc_dir, mouseName, sessionName, 'touch_QC_2D.png'];
        saveas(gcf, save_fn)
        close;
    end
end

%%
qc_dir = 'E:\TPM\JK\WhiskerVideo\TouchQC\';
mouse = 36;
session = 2;
mouseName = sprintf('JK%03d', mouse);
sessionName = sprintf('S%02d', session);

load_fn = [qc_dir, sprintf('%s%s_touch_qc.mat',mouseName,sessionName)];
load(load_fn) % hp_peaks, scatter2d_points, scatter3d_points, touch_hp, touchhp_2d

sdi = 2;
points_3d = scatter3d_points{sdi};
plot3(points_3d(1,:), points_3d(2,:), points_3d(3,:), 'k.'), hold on
plot3(touch_hp{sdi}(1,:) + hp_peaks{sdi}(1), touch_hp{sdi}(2,:), touch_hp{sdi}(3,:), 'r-')
plot3(touch_hp{sdi}(1,:) + hp_peaks{sdi}(2), touch_hp{sdi}(2,:), touch_hp{sdi}(3,:), 'r-')


%% JK036 S81 (first 245 trials of S01)

mouse = 36;
session = 81;
mouseName = sprintf('JK%03d', mouse);
sessionName = sprintf('S%02d', session);

save_fn = sprintf('%s%s_touch_qc.mat',mouseName, sessionName);

wDir = 'E:\TPM\JK\WhiskerVideo\';
save_path = sprintf('%sTouchQC\\%s', wDir, save_fn);

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
    scatter(points_2d(:,1),points_2d(:,2),'k.'), hold on, scatter(th_2d1(:,1), th_2d1(:,2),'r.'), scatter(th_2d2(:,1), th_2d2(:,2),'r.')
    
    scatter3d_points{sdi} = points;

    plot3(points(1,:), points(2,:), points(3,:), 'k.'), hold on
    plot3(touch_hp{sdi}(1,:) + hp_peaks{sdi}(1), touch_hp{sdi}(2,:), touch_hp{sdi}(3,:), 'r-')
    plot3(touch_hp{sdi}(1,:) + hp_peaks{sdi}(2), touch_hp{sdi}(2,:), touch_hp{sdi}(3,:), 'r-')
    
end


%%
save(save_path, 'scatter2d_points', 'touchhp_2d', 'scatter3d_points', 'touch_hp', 'hp_peaks')

%%
sdi = 7;
points_3d = scatter3d_points{sdi};
plot3(points_3d(1,:), points_3d(2,:), points_3d(3,:), 'k.'), hold on
plot3(touch_hp{sdi}(1,:) + hp_peaks{sdi}(1), touch_hp{sdi}(2,:), touch_hp{sdi}(3,:), 'r-')
plot3(touch_hp{sdi}(1,:) + hp_peaks{sdi}(2), touch_hp{sdi}(2,:), touch_hp{sdi}(3,:), 'r-')
