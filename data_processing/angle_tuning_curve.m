function atc = angle_tuning_curve(response,angles)
% Calculate angle tuning curve from response matrix and angle vector
% Inputs:
%     response (n x t matrix): response matrix of n neurons from t trials
%     angles (1 x t vector): a vector of trial angles
% Output:
%     atc (n x length(unique(angles)) matrix): angle tuning curve

if length(size(response)) > 2
    error('Response matrix should be in 2 dim')
end

if size(angles,1) ~= 1
    angles = angles';
    if size(angles,1) ~= 1
        error('Angles should be 1 x t vector')
    end
end

if size(response,2) ~= size(angles,2)
    error('Number of trials should match')
end

nNeuron = size(response,1);
angleList = unique(angles);
nAngle = length(angleList);

atc = zeros(nNeuron, nAngle);
for ai = 1 : length(angleList)
    angle = angleList(ai);
    atc(:,ai) = nanmean(response(:,find(angles == angle)),2);
end