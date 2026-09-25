function u = ensemble_std(mdl, S)
% Prediction discrepancy of the integrated submodels in PI-E-SINDy (uncertainty score for active learning).
% Perform one-step prediction for each Bootstrap submodel, and compute the root mean square of the cross-model standard deviation over states.

if isempty(mdl.boots)
    u = zeros(S.n, 1);
    return;
end
usePrior = ~isfield(mdl, 'usePrior') || mdl.usePrior;
Theta = build_library(S.X, S.U, usePrior);
B = size(mdl.boots, 3);
sc = lib_scales();
pred = zeros(S.n, 4, B);
for b = 1:B
    pred(:,:,b) = S.X + S.dt .* (Theta * mdl.boots(:,:,b));
end
% Each state is normalized by its range before aggregation to avoid net load dominance.
v = var(pred, 0, 3) ./ (sc(1:4).^2);
u = sqrt(mean(v, 2));
end
