function [G, R] = weak_rows(S, K, stride, usePrior)
% Weak-form transformation (W-SINDy, Messenger & Bortz).
% Extract sliding windows of length K along each continuous trajectory.
% The compact-support test function is psi(s)=(1-s^2)^3.
% Integration by parts converts ∫psi*dx/dt to -∫psi'*x, avoiding numerical differentiation of noisy states.
% Return weak-form dictionary rows G and response rows R for subsequent STRidge.

if nargin < 2, K = 9; end
if nargin < 3, stride = 3; end
if nargin < 4, usePrior = true; end

% Test function and its derivative (for internal variable s ∈ [-1,1])
s = linspace(-1, 1, K)';
psi  = (1 - s.^2).^3;
dpsi = -6 * s .* (1 - s.^2).^2;

% Weak-form operation on state trajectories (current states Y and concurrent exogenous inputs U)
Theta = build_library(S.Y, S.U, usePrior);
nLib = size(Theta, 2);

% Identify all continuous segments (same group with consecutive indices)
brk = [1; find(diff(S.group) ~= 0 | diff(S.seq) ~= 1) + 1; S.n + 1];

Gcell = {};
Rcell = {};
for b = 1:numel(brk)-1
    i1 = brk(b); i2 = brk(b+1) - 1;
    len = i2 - i1 + 1;
    if len < K, continue; end
    for w0 = i1 : stride : i2 - K + 1
        idx = w0 : w0 + K - 1;
        dtw = S.dt(idx);
        T = sum(dtw);                        % Window duration (hours)
        % ∫ psi * Theta dt, trapezoidal approximation
        wq = trap_w(dtw);
        Gcell{end+1,1} = (psi .* wq)' * Theta(idx,:); %#ok<AGROW>
        % -∫ psi' * x dt; derivative of psi w.r.t. real time scaled by 2/T
        Rcell{end+1,1} = -((dpsi * 2 / T) .* wq)' * S.Y(idx,:); %#ok<AGROW>
    end
end

if isempty(Gcell)
    G = zeros(0, nLib); R = zeros(0, 4);
else
    G = cell2mat(Gcell);
    R = cell2mat(Rcell);
end
end

function w = trap_w(dtw)
% Trapezoidal integration weights
K = numel(dtw);
w = zeros(K,1);
w(1) = dtw(2)/2; w(K) = dtw(K)/2;
for k = 2:K-1
    w(k) = (dtw(k) + dtw(k+1))/2;
end
end
