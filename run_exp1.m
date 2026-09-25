function [tbl, predStore, baseModels] = run_exp1(trainSets, testSets, cleanYear, stageNames)
% Experiment 1: Identification accuracy and generalization of three surrogate models across three scenarios.
% Coefficient drift: relative deviation of scenario-specific model coefficients against the full-year baseline of the same method, computed under the same coordinate system.

methods = {'W-SINDy', 'SINDy', 'PI-E-SINDy'};

% Full-year baseline models for each method. Reset random seed before each fitting to ensure identical models on the same dataset;
% Bootstrap randomness excluded from drift calculation.
Syear = sample_pairs(cleanYear);
baseModels = cell(1,3);
for m = 1:3
    rng(20260810, 'twister');
    baseModels{m} = fit_surrogate(Syear, methods{m});
end

rows = cell(9, 8);
predStore = cell(3, 3);
r = 0;
for s = 1:numel(trainSets)
    Str = sample_pairs(trainSets{s});
    Ste = sample_pairs(testSets{s});

    % Inputs in low-sample high-noise scenario are contaminated; ground truth for evaluation is replaced by clean full-year series aligned by time index.
    Ytruth = Ste.Y;
    if s == 1
        [ok, loc] = ismember(Ste.seq, cleanYear.seq);
        cleanY = [cleanYear.hydro, cleanYear.gen, cleanYear.pump, cleanYear.net];
        Ytruth(ok,:) = cleanY(loc(ok),:);
    end

    for m = 1:3
        r = r + 1;
        rng(20260810, 'twister');
        t0 = tic;
        mdl = fit_surrogate(Str, methods{m});
        tFit = toc(t0);
        Yhat = predict_surrogate(mdl, Ste);
        predStore{s,m} = struct('truth', Ytruth, 'pred', Yhat, ...
            'group', Ste.group, 'hour', Ste.X(:,1)*0 + (1:Ste.n)', 'dt', Ste.dt);

        nnzEq = sum(abs(mdl.Xi) > 1e-10, 1);
        lbl = sprintf('%d/%d/%d/%d (total %d)', nnzEq, sum(nnzEq));
        rho = mean_corr(Ytruth, Yhat);
        bal = mean(abs(Yhat(:,1) + Yhat(:,2) - Yhat(:,3) - Yhat(:,4)));
        drift = norm(mdl.Xi - baseModels{m}.Xi, 'fro') / max(norm(baseModels{m}.Xi, 'fro'), eps);
        if isempty(mdl.inc), incPct = NaN;
        else, incPct = 100 * mean(mdl.inc(abs(mdl.Xi) > 1e-10)); end
        rows(r,:) = {stageNames{s}, methods{m}, lbl, incPct, rho, bal, drift, tFit};
    end
end

tbl = cell2table(rows, 'VariableNames', {'Scenario','Method','Sparse Dictionary Terms','Coefficient Inclusion Probability_pct', ...
    'Temporal correlation coefficient','Power balance residual_MW','Coefficient drift_vs full-year baseline','Training time_s'});
end

function r = mean_corr(Y, P)
v = zeros(1, size(Y,2));
for j = 1:size(Y,2)
    c = corrcoef(Y(:,j), P(:,j));
    if numel(c) >= 4 && isfinite(c(1,2)), v(j) = c(1,2); end
end
r = mean(v);
end
