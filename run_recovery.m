function tbl = run_recovery(yearDs, nSeeds)
% Parameter recovery experiment: construct a synthetic system with known ground-truth sparse equations to examine whether the three methods can recover coefficients.
% Dictionary and unified coordinate system consistent with main experiments; two physical-prior columns removed,
% as they are exact linear combinations of other basis terms. Retaining them would lead to non-unique ground-truth coefficients and undefined recovery error.
% Excitation: short trajectories with multiple random initial states. Transient responses from random initial conditions explore the state space.
% Fine-step simulation followed by hourly sampling to validate the weak-form continuity assumption.
% Two noise levels: SNR = 40 dB and 30 dB; results averaged over nSeeds random realizations.
% Metrics: support-set F1 score for structural recovery, relative coefficient error for numerical recovery.

if nargin < 2, nSeeds = 5; end
nSeg = 250; segLen = 30; sub = 20;
nTot = nSeg * segLen;
hsub = 1 / sub;

[XiTrue, names] = truth_system(); %#ok<ASGLU>

% Ground-truth system simulation: random initial states, driven by true exogenous series, integrated with sub-step resolution.
rng(20260899, 'twister');
X = zeros(nTot, 4); U = zeros(nTot, 5); grp = zeros(nTot, 1);
row = 0;
for s = 1:nSeg
    off = randi(numel(yearDs.load) - segLen - 1);
    ang = 2*pi*(yearDs.hour(off+(1:segLen)) + yearDs.minute(off+(1:segLen))/60) / 24;
    Us = [yearDs.load(off+(1:segLen)), yearDs.wind(off+(1:segLen)), ...
        yearDs.pv(off+(1:segLen)), sin(ang), cos(ang)];
    x = [10800*rand, 1200*rand, 950*rand, 12000*rand];
    for k = 1:segLen
        row = row + 1;
        X(row,:) = x; U(row,:) = Us(k,:); grp(row) = s;
        uNext = Us(min(k+1, segLen), :);
        for q = 1:sub
            Th = build_library(x, uNext, false);
            x = x + hsub * (Th * XiTrue);
        end
    end
end

Sbase = struct('X', [X(1,:); X(1:end-1,:)], 'U', U, 'Y', X, ...
    'dt', ones(nTot,1), 'group', grp, 'seq', (1:nTot)', ...
    'first', [true; diff(grp) ~= 0], 'n', nTot);

methods = {'W-SINDy', 'SINDy', 'PI-E-SINDy'};
snrList = [40, 30];
rows = cell(numel(snrList)*3, 6);
r = 0;
for iSnr = 1:numel(snrList)
    snr = snrList(iSnr);
    f1 = zeros(nSeeds, 3); ce = zeros(nSeeds, 3);
    for seed = 1:nSeeds
        rng(700 + seed, 'twister');
        Xn = X + randn(nTot,4) .* (std(X, 0, 1) * 10^(-snr/20));
        S = Sbase;
        S.Y = Xn;
        S.X = [Xn(1,:); Xn(1:end-1,:)];
        ff = find(S.first);
        S.X(ff,:) = Xn(ff,:);
        for m = 1:3
            mdl = fit_surrogate(S, methods{m}, struct('usePrior', false));
            f1(seed,m) = support_f1(mdl.Xi, XiTrue);
            ce(seed,m) = norm(mdl.Xi - XiTrue,'fro') / norm(XiTrue,'fro');
        end
    end
    for m = 1:3
        r = r + 1;
        rows(r,:) = {sprintf('SNR %d dB', snr), methods{m}, ...
            mean(f1(:,m)), std(f1(:,m)), mean(ce(:,m)), std(ce(:,m))};
    end
end

tbl = cell2table(rows, 'VariableNames', {'Noise level','Method','Support set F1_mean', ...
    'Support set F1_std','Relative coefficient error_mean','Relative coefficient error_std'});
end

function [Xi, names] = truth_system()
% Ground-truth equations (dictionary under unified coordinate system with prior columns removed; rows = candidate basis terms, columns = state equations).
% Time constants relaxed to 4–8 h, ensuring hourly sampling is sufficiently dense relative to system dynamics
% to validate numerical integration under weak formulation.
[~, names] = build_library(zeros(1,4), zeros(1,5), false);
p = numel(names);
Xi = zeros(p, 4);
    function put(eq, term, val)
        k = find(strcmp(names, term), 1);
        Xi(k, eq) = val;
    end
% Hydropower: relaxation towards net load.
put(1,'hyd',-1485); put(1,'net',1500);
% Pumped-storage generation: track net load, suppressed by hydro-net load coupling.
put(2,'gen',-180);  put(2,'net',120);  put(2,'hyd*net',-90);
% Pumping mode: modulated by load and net load.
put(3,'pump',-106.9); put(3,'load',28.5); put(3,'net',-23.75);
% Net load: driven by source-load profiles plus diurnal component.
put(4,'net',-1050); put(4,'load',900); put(4,'wind',-750);
put(4,'pv',-600);   put(4,'sin',150);
end

function f1 = support_f1(Xi, XiTrue)
est = abs(Xi) > 1e-10;
tru = abs(XiTrue) > 1e-10;
tp = nnz(est & tru);
prec = tp / max(nnz(est), 1);
rec  = tp / max(nnz(tru), 1);
f1 = 2*prec*rec / max(prec + rec, eps);
end
