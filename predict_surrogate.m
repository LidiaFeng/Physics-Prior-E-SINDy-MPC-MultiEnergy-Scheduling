function Yhat = predict_surrogate(mdl, S)
% One-step prediction: x(t+1) = x(t) + dt * Theta(x(t), u(t+1)) * Xi
% The three methods share the same prediction formula and feasible region clipping.
% PI-E-SINDy additionally performs its built-in physics-consistent projection
% (predictor-side formulation of weak-form energy balance prior), using only exogenous variables.

usePrior = ~isfield(mdl, 'usePrior') || mdl.usePrior;
Theta = build_library(S.X, S.U, usePrior);
Yhat = S.X + S.dt .* (Theta * mdl.Xi);

if strcmp(mdl.method, 'PI-E-SINDy')
    % Physics-consistent projection: after fusing net load prediction and source-load balance values, the power balance residual
    % allocated to three controllable outputs with fixed ratios. Uniform gains for all scenarios; only exogenous variables.
    alpha = 0.50;    % Total gain for balance correction
    beta  = 0.15;    % Net load physics fusion weight
    share = [0.80, 0.10, 0.10];   % Allocation ratios: hydropower / pumped-storage generation / pumping
    netPhys = min(max(S.U(:,1) - S.U(:,2) - S.U(:,3), 0), 12000);
    Yhat(:,4) = (1-beta)*Yhat(:,4) + beta*netPhys;
    imb = Yhat(:,1) + Yhat(:,2) - Yhat(:,3) - Yhat(:,4);
    Yhat(:,1) = Yhat(:,1) - alpha*share(1)*imb;
    Yhat(:,2) = Yhat(:,2) - alpha*share(2)*imb;
    Yhat(:,3) = Yhat(:,3) + alpha*share(3)*imb;
end

Yhat = clip_feasible(Yhat);
end

function Y = clip_feasible(Y)
% Unified feasible region clipping: equipment physical bounds plus mutual exclusivity between pumped-storage generation and pumping.
Y(:,1) = min(max(Y(:,1), 0), 10800);
Y(:,2) = min(max(Y(:,2), 0), 1200);
Y(:,3) = min(max(Y(:,3), 0), 950);
Y(:,4) = min(max(Y(:,4), 0), 12000);
both = Y(:,2) > 0 & Y(:,3) > 0;
keepGen = Y(:,2) >= Y(:,3);
Y(both &  keepGen, 3) = 0;
Y(both & ~keepGen, 2) = 0;
end
