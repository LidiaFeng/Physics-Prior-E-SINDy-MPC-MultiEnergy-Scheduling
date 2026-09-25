function [T, D] = horizon_maps(H, Nc)
% T: hold mapping from control horizon to prediction horizon (control input remains constant after Nc)
% D: control increment difference matrix
T = zeros(H, Nc);
for k = 1:H
    T(k, min(k, Nc)) = 1;
end
D = eye(Nc) - diag(ones(Nc-1,1), -1);
D(1,1) = 0;    % The first-step increment is relative to the previous actual value and handled separately in the objective.
end
