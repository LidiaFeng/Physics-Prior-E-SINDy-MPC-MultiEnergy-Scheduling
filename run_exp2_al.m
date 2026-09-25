function [tbl, curves] = run_exp2_al(yearDs, poolDs, testDs, nSeeds)
% Active-learning augmentation experiment (Random / Uncertainty-AL /
% Risk-aware AL). Candidate labels remain hidden until queried. Selection
% uses input features and current ensemble uncertainty. Models are retrained
% and candidates reranked after every query batch. Results report the mean
% and 95% confidence interval over nSeeds runs. Labeling cost is expressed
% in person-hours according to scenario type.

if nargin < 4, nSeeds = 20; end
budgets = [0 24 48 72 96];
batch = 24;
nRound = numel(budgets) - 1;
alNames = {'Random', 'Uncertainty-AL', 'Risk-aware AL'};

% Per-sample labeling cost (person-hours) for four extreme scenarios:
% 1 PV drop, 2 wind ramp, 3 load surge, 4 net-load transition.
costPerGroup = [1.0, 1.2, 1.1, 1.5];

Sbase = sample_pairs(yearDs);
Spool = sample_pairs(poolDs);
Stest = sample_pairs(testDs);
[Gb, Rb] = weak_rows(Sbase, 9, 3);      % Compute baseline weak-form rows once.
fitOpts = struct('nBoot', 12, 'baseGR', {{Gb, Rb}});

% Feature-based risk score (independent of labels) and the top-48 high-risk
% candidate set used to evaluate coverage.
riskPool = physical_risk(Spool);
[~, riskOrd] = sort(riskPool, 'descend');
topRisk = false(Spool.n, 1);
topRisk(riskOrd(1:min(48, Spool.n))) = true;

nPool = Spool.n;
poolCost = costPerGroup(max(min(round(Spool.group), 4), 1))';

rmseMat = zeros(nSeeds, 3, nRound+1);
covMat  = zeros(nSeeds, 3, nRound+1);
balMat  = zeros(nSeeds, 3, nRound+1);
costMat = zeros(nSeeds, 3, nRound+1);
timeMat = zeros(nSeeds, 3);

for seed = 1:nSeeds
    rng(20260810 + seed, 'twister');
    % Zero-budget baseline shared by all strategies before any query.
    mdl0 = fit_surrogate(Sbase, 'PI-E-SINDy', fitOpts);
    [rmse0, bal0] = eval_model(mdl0, Stest);
    for a = 1:3
        queried = false(nPool, 1);
        mdl = mdl0;
        rmseMat(seed,a,1) = rmse0; balMat(seed,a,1) = bal0;
        covMat(seed,a,1) = 0; costMat(seed,a,1) = 0;
        tTrain = 0;
        for r = 1:nRound
            % Rescore the remaining candidates with the current model.
            remain = find(~queried);
            switch a
                case 1   % Random: uniform random sampling.
                    pickLocal = randperm(numel(remain), min(batch, numel(remain)));
                case 2   % Uncertainty-AL: ensemble predictive disagreement.
                    u = ensemble_std(mdl, subset_samples(Spool, remain));
                    [~, ord] = sort(u, 'descend');
                    pickLocal = ord(1:min(batch, numel(remain)));
                case 3   % Risk-aware AL: uncertainty, physical risk, and coverage.
                    u = norm01(ensemble_std(mdl, subset_samples(Spool, remain)));
                    rk = norm01(riskPool(remain));
                    cv = coverage_score(Spool.group(remain), Spool.group(queried));
                    score = 0.30*u + 0.55*rk + 0.15*cv;
                    [~, ord] = sort(score, 'descend');
                    pickLocal = ord(1:min(batch, numel(remain)));
            end
            queried(remain(pickLocal)) = true;

            % Retrain after obtaining the newly queried labels.
            t0 = tic;
            opt = fitOpts;
            opt.extraS = subset_samples(Spool, find(queried));
            mdl = fit_surrogate(Sbase, 'PI-E-SINDy', opt);
            tTrain = tTrain + toc(t0);

            [rmseMat(seed,a,r+1), balMat(seed,a,r+1)] = eval_model(mdl, Stest);
            covMat(seed,a,r+1) = 100 * nnz(topRisk & queried) / nnz(topRisk);
            costMat(seed,a,r+1) = sum(poolCost(queried));
        end
        timeMat(seed,a) = tTrain;
    end
end

% Aggregate curves using means and 95% confidence-interval half-widths.
curves = struct('budgets', budgets, 'names', {alNames});
curves.rmseMean = squeeze(mean(rmseMat, 1));
curves.rmseCI   = ci_half(rmseMat);
curves.covMean  = squeeze(mean(covMat, 1));
curves.covCI    = ci_half(covMat);
curves.costMean = squeeze(mean(costMat, 1));
curves.balMean  = squeeze(mean(balMat, 1));
curves.nSeeds   = nSeeds;

rows = cell(3, 8);
for a = 1:3
    rows(a,:) = {alNames{a}, curves.rmseMean(a,end), ...
        curves.rmseCI(a,end), curves.covMean(a,end), ...
        curves.costMean(a,end), budgets(end), ...
        curves.balMean(a,end), mean(timeMat(:,a))};
end
tbl = cell2table(rows, 'VariableNames', {'Sampling strategy','Final RMSE_MW','RMSE 95CI half-width', ...
    'High-risk coverage_pct','Labeling cost_person-hours','Queried samples','Power-balance residual_MW','Mean training time_s'});
end

function [rmse, bal] = eval_model(mdl, Stest)
% Test labels are used only for evaluation.
Yhat = predict_surrogate(mdl, Stest);
rmse = sqrt(mean((Yhat(:,1) - Stest.Y(:,1)).^2));
bal = mean(abs(Yhat(:,1) + Yhat(:,2) - Yhat(:,3) - Yhat(:,4)));
end

function r = physical_risk(S)
% Feature-level physical risk: net-load ramp plus hydropower and storage margins.
netNow = S.U(:,1) - S.U(:,2) - S.U(:,3);
ramp = abs(netNow - S.X(:,4));
hydMargin = min(S.X(:,1), 10800 - S.X(:,1));
psMargin = min(1200 - S.X(:,2), 950 - S.X(:,3));
r = 0.55*norm01(ramp) + 0.25*(1 - norm01(max(hydMargin,0))) + 0.20*(1 - norm01(max(psMargin,0)));
end

function c = coverage_score(remainGroup, queriedGroup)
% Give higher scores to underrepresented queried classes to improve coverage.
c = zeros(size(remainGroup));
ug = unique(remainGroup);
nq = max(numel(queriedGroup), 1);
for k = 1:numel(ug)
    share = nnz(queriedGroup == ug(k)) / nq;
    c(remainGroup == ug(k)) = 1 - share;
end
c = norm01(c);
end

function Ssub = subset_samples(S, idx)
Ssub = S;
f = {'X','U','Y','dt','group','seq','first'};
for k = 1:numel(f)
    Ssub.(f{k}) = S.(f{k})(idx, :);
end
Ssub.n = numel(idx);
end

function x = norm01(x)
lo = min(x); hi = max(x);
x = (x - lo) / max(hi - lo, eps);
end

function h = ci_half(M)
% 95% confidence-interval half-width using the t distribution.
% M dimensions: seeds x strategies x budgets.
n = size(M,1);
se = squeeze(std(M, 0, 1)) / sqrt(n);
h = t975(n-1) * se;
end

function t = t975(df)
% Lookup for the 0.975 t quantile without requiring Statistics Toolbox.
tab = [12.706 4.303 3.182 2.776 2.571 2.447 2.365 2.306 2.262 2.228 ...
       2.201 2.179 2.160 2.145 2.131 2.120 2.110 2.101 2.093 2.086 ...
       2.080 2.074 2.069 2.064 2.060 2.056 2.052 2.048 2.045 2.042];
if df <= 30
    t = tab(max(df,1));
else
    t = 1.960 + 2.4/df;   % Large-degrees-of-freedom asymptotic correction.
end
end
