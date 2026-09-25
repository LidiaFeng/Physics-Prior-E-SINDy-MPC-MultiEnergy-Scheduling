function mdl = fit_surrogate(S, method, opts)
% Unified interface for the three surrogate models. The dictionary, threshold and ridge coefficient are identical across all methods.
% The only difference lies in the regression scheme: pointwise-differentiated SINDy (Brunton 2016), weak-form integral regression for W-SINDy (Messenger & Bortz),
% and PI-E-SINDy which further incorporates Bootstrap-STRidge ensemble and probability-informed pruning on top of the weak form.
% All coefficients lie in the unified coordinate system and can be directly compared to compute drift.

if nargin < 3, opts = struct(); end
lambda  = getdef(opts, 'lambda', 0.05);
ridge   = getdef(opts, 'ridge', 1e-6);
K       = getdef(opts, 'K', 9);
stride  = getdef(opts, 'stride', 3);
nBoot   = getdef(opts, 'nBoot', 80);
pTh     = getdef(opts, 'pTh', 0.75);
extraS  = getdef(opts, 'extraS', []);   % Discrete samples augmented via active learning (one-step difference rows)
baseGR  = getdef(opts, 'baseGR', []);   % Precomputed weak-form rows {G,R} to save time during repeated training
usePrior = getdef(opts, 'usePrior', true);  % Set to false when removing degenerate prior columns in parameter recovery experiments

switch method
    case 'SINDy'
        use = ~S.first;                 % The first point of each scenario has no predecessor within the group and cannot form a difference sample.
        D = (S.Y(use,:) - S.X(use,:)) ./ S.dt(use);
        A = build_library(S.X(use,:), S.U(use,:), usePrior);
        [A, D] = append_extra(A, D, extraS, usePrior);
        Xi = stridge(A, D, lambda, ridge);
        boots = []; inc = [];

    case 'W-SINDy'
        [G, R] = get_rows(S, K, stride, baseGR, usePrior);
        [G, R] = append_extra(G, R, extraS, usePrior);
        Xi = stridge(G, R, lambda, ridge);
        boots = []; inc = [];

    case 'PI-E-SINDy'
        % Weak-form physics prior embedding: weak-form integral rows are treated as constraint rows and jointly regressed with pointwise difference rows
        % (equal weights after row magnitude alignment), followed by Bootstrap-STRidge ensemble.
        use = ~S.first;
        Dp = (S.Y(use,:) - S.X(use,:)) ./ S.dt(use);
        Ap = build_library(S.X(use,:), S.U(use,:), usePrior);
        [Gw, Rw] = get_rows(S, K, stride, baseGR, usePrior);
        rowScale = sqrt(mean(Ap(:).^2)) / max(sqrt(mean(Gw(:).^2)), eps);
        G = [Ap; rowScale * Gw];
        R = [Dp; rowScale * Rw];
        [G, R] = append_extra(G, R, extraS, usePrior);
        n = size(G,1); p = size(G,2); q = size(R,2);
        boots = zeros(p, q, nBoot);
        for b = 1:nBoot
            rows = randi(n, n, 1);      % Bootstrap row resampling
            boots(:,:,b) = stridge(G(rows,:), R(rows,:), lambda, ridge);
        end
        inc = mean(abs(boots) > 1e-10, 3);          % Inclusion probability of candidate terms
        support = inc >= pTh;
        Xi = zeros(p, q);
        for j = 1:q
            s = support(:,j);
            if ~any(s)
                [~, ix] = max(inc(:,j)); s(ix) = true;
            end
            % Ridge regression refitting using all weak-form rows on the support set (columns normalized consistently with STRidge)
            cs = sqrt(mean(G(:,s).^2,1)); cs(cs<1e-12) = 1;
            Gs = G(:,s) ./ cs;
            w = (Gs'*Gs + ridge*eye(nnz(s))) \ (Gs'*R(:,j));
            Xi(s,j) = w ./ cs';
        end

    otherwise
        error('Unknown method: %s', method);
end

[~, names] = build_library(S.X(1,:), S.U(1,:), usePrior);
mdl = struct('Xi', Xi, 'method', method, 'lambda', lambda, ...
    'inc', inc, 'names', {names}, 'nBoot', nBoot, 'usePrior', usePrior);
mdl.boots = boots;
end

function [G, R] = get_rows(S, K, stride, baseGR, usePrior)
if isempty(baseGR)
    [G, R] = weak_rows(S, K, stride, usePrior);
else
    G = baseGR{1}; R = baseGR{2};
end
end

function [A, D] = append_extra(A, D, extraS, usePrior)
% Samples acquired via active learning are fed into the regression system as one-step difference rows.
if isempty(extraS) || extraS.n == 0, return; end
use = ~extraS.first;
if ~any(use), return; end
De = (extraS.Y(use,:) - extraS.X(use,:)) ./ extraS.dt(use);
Ae = build_library(extraS.X(use,:), extraS.U(use,:), usePrior);
A = [A; Ae];
D = [D; De];
end

function v = getdef(s, f, d)
if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
