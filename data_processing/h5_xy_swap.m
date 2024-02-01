% h5 file x and y dimension change
% 2021/02/02 JK
clear

baseDir = 'D:\TPM\JK\h5\';
mice = [25,27,30,36,37,38,39,41,52,53,54,56];
% for mi = 1 : length(mice)
for mi = 9:length(mice)
    mouse = mice(mi);
    for pi = 1 : 8
        tempDir = sprintf('%s%03d\\plane_%d\\',baseDir, mouse, pi);
        tempH5 = dir(sprintf('%s%03d_*_plane_%d.h5',tempDir, mouse, pi));
        for fi = 1 : length(tempH5)
            newFn = [tempDir, 'new_', tempH5(fi).name];
            if ~exist(newFn,'file')
                data = h5read([tempH5(fi).folder, filesep, tempH5(fi).name], '/data');
                newData = permute(data, [2 1 3]);

                h5create(newFn, '/data', size(newData), 'Datatype', 'uint16')
                h5write(newFn, '/data', newData);
            end
        end
        % for sub directories
        % move them back to upper directory
        tempDn = dir(sprintf('%s%03d_*_plane_%d',tempDir, mouse, pi));
        for di = 1 : length(tempDn)
            if tempDn(di).isdir
                tempSubDir = [tempDir, tempDn(di).name, '\'];
                h5files = dir(sprintf('%s*.h5',tempSubDir));
                for fi = 1 : length(h5files)
                    data = h5read([h5files(fi).folder, filesep, h5files(fi).name], '/data');
                    newData = permute(data, [2 1 3]);
                    newFn = [tempDir, 'new_', h5files(fi).name];
                    h5create(newFn, '/data', size(newData), 'Datatype', 'uint16')
                    h5write(newFn, '/data', newData);
                end
            end
        end
    end
end

% %%
% clear
% tempFn = '025_001_000_plane_1.h5';
% data = h5read(tempFn, '/data');
% 
% newData = permute(data,[2 1 3]);
% %%
% newFn = ['new_', tempFn];
% h5create(newFn, '/data', size(newData), 'Datatype', 'uint16')
% h5write(newFn, '/data', newData)
% h5disp(newFn)












%% QC

% numRandFile = 6;
% numRandFrame = 100;
% 
% for mi = 1 : 7
%     mouse = mice(mi);
%     fprintf('Mouse %03d\n', mouse)
%     for pi = 1 : 8
%         fprintf('Plane %d\n', pi)
%         tempDir = sprintf('%s%03d\\plane_%d\\',baseDir, mouse, pi);
%         oldflist = dir(sprintf('%s%03d_*.h5', tempDir, mouse));
%         try
%             testFileInd = randperm(length(oldflist), numRandFile);
%         catch
%             testFileInd = 1:length(oldflist);
%         end
%         for fi = 1 : length(testFileInd)
%             tfi = testFileInd(fi);
%             oldData = h5read([oldflist(tfi).folder, filesep, oldflist(tfi).name], '/data');
%             newData = h5read([oldflist(tfi).folder, filesep, 'new_', oldflist(tfi).name], '/data');
%             if size(oldData,1) ~= size(newData,2) || size(oldData,2) ~= size(newData,1) || size(oldData,3) ~= size(newData,3)
%                 error(sprintf('Size mismatch, %03d plane #d %s', mouse, pi, oldflist(tfi).name))
%             end
%             testInd = randperm(size(oldData,3), numRandFrame);
%             for ti = 1 : length(testInd)
%                 if squeeze(oldData(:,:,testInd(ti)))' ~= squeeze(newData(:,:,testInd(ti)))
%                     error(sprintf('Image mistmatch, %03d plane #d %s', mouse, pi, oldflist(tfi).name))
%                 end
%             end
%         end
%     end
% end
% 
% disp('No error')