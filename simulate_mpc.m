function out = simulate_mpc(ds, mode, surro, cfg)
% Main loop for rolling horizon dispatch simulation. mode=1: centralized (solve one large QP per time step),
% mode=2: fixed-period ADMM coordination, mode=3: event-triggered ADMM coordination.
% Each step performs true predictive horizon optimization with complete constraints (boundaries/ramping/storage/reserve/mutual exclusion).
% For distributed modes, the problem is decomposed into hydro and pumped-storage subproblems solved via ADMM.
% Residuals, iteration counts and computation time are logged step-by-step in out.stepLog.

if nargin < 4, cfg = struct(); end
Np = getdef(cfg, 'Np', 8);
Nc = getdef(cfg, 'Nc', 4);
feedback = getdef(cfg, 'feedback', true);
thrScale = getdef(cfg, 'thrScale', 1.0);
period = getdef(cfg, 'period', 4);
rho = getdef(cfg, 'rho', 1.0);

n = numel(ds.load);
dt = ds.dt;
nTrue = max(ds.load, 0) - max(ds.wind, 0) - max(ds.pv, 0);   % Ground-truth net load for evaluation
% Controller observes measured values; measNet in Scenario S1 is noisy, noise is assigned to load side.
measL = ds.load; measW = ds.wind; measP = ds.pv;
noisy = mean(abs(ds.measNet - ds.net)) > 20;
if noisy
    measL = ds.load + (ds.measNet - ds.net);
end
nMeas = min(max(max(measL,0) - max(measW,0) - max(measP,0), 0), 12000);
% Apply causal median filter to noisy measurements (only current and past samples).
% No filtering for clean measurements to avoid lag.
if noisy
    ug = unique(ds.group);
    for k = 1:numel(ug)
        ii = find(ds.group == ug(k));
        nMeas(ii) = movmedian(nMeas(ii), [4 0]);
    end
end

lim = struct('phMax',10800,'pgMax',1200,'ppMax',950,'ramp',850, ...
    'eMax',5183.64,'etaP',0.90,'etaG',0.90,'eRef',3600);

Ph = zeros(n,1); Pg = zeros(n,1); Pp = zeros(n,1); E = zeros(n,1);
supply = zeros(n,1); short = zeros(n,1); excess = zeros(n,1);
trig = false(n,1); commIters = zeros(n,1); solveMs = zeros(n,1);
why = false(n,5);        % Trigger reasons: scenario switch / tracking deviation / storage boundary / schedule aging / cumulative power shortage
okFlag = true(n,1); violation = false(n,1); predErr = zeros(n,1);
admmIter = nan(n,1); admmPri = nan(n,1); admmDua = nan(n,1);
netFcApplied = zeros(n,1);
traces = {};                      % Full residual trajectories at partial coordination instants, for convergence plots.

plan = [];       % Most recent coordinated plan (Ph/Pg/Pp sequences).
age = Np;        % Number of steps since the most recent coordination.
duals = zeros(Np,1);
ePrev = 0;
biasEst = 0;     % Exponential estimate of locally observed supply-demand bias.
emaAbsE = 0;     % Smoothed error magnitude used as an adaptive trigger deadband.
cumE = 0;        % Cumulative absolute error since coordination (MWh).
PhPrev = 0; Ecur = 0;

for t = 1:n
    newGroup = (t == 1) || (ds.group(t) ~= ds.group(t-1));
    if newGroup
        % Reset storage and hydropower states at each scenario boundary.
        Ecur = min(max(ds.soc(t), 0), lim.eMax);
        if Ecur == 0, Ecur = 3100; end
        PhPrev = min(max(ds.hydro(t), 0), lim.phMax);
        age = Np; plan = []; duals = zeros(Np,1); ePrev = 0; biasEst = 0; emaAbsE = 0; cumE = 0;
    end

    % Decide whether to coordinate. Between events, shift the previous plan
    % and apply local feedback only.
    if mode == 1
        doCoord = true;
    elseif mode == 2
        doCoord = newGroup || mod(age, period) == 0 || isempty(plan);
    else
        % Tighten the base threshold for 15-minute data and add an adaptive
        % deadband to prevent repeated triggers under persistent noise.
        if dt(t) <= 0.25
            scaleT = max(80, 0.02 * nMeas(t));
        else
            scaleT = max(150, 0.03 * nMeas(t));
        end
        dyn = max(scaleT, 1.3 * emaAbsE);
        evTrack = abs(ePrev) > thrScale * dyn || ePrev > 0.6 * thrScale * dyn;  % More sensitive to undersupply.
        evEnergy = (Ecur < 400) || (Ecur > lim.eMax - 400);
        evAge = age >= Np;
        evCum = cumE > thrScale * 120;     % Trigger on persistent cumulative undersupply.
        doCoord = newGroup || evTrack || evEnergy || evAge || evCum || isempty(plan);
        why(t,:) = [newGroup, evTrack, evEnergy, evAge, evCum];
    end

    hz = t : min(t+Np-1, n);
    hz = hz(ds.group(hz) == ds.group(t));       % Do not cross scenario boundaries.
    H = numel(hz);

    t0 = tic;
    if doCoord
        anchor = nMeas(t);
        if feedback && mode == 3
            % Apply residual feedback only to undersupply. Let optimization
            % absorb oversupply to avoid introducing new undersupply.
            anchor = anchor + 0.5 * max(biasEst, 0);
        end
        % Anchor the first step to measurement and propagate later steps with
        % the surrogate under a persistence forecast for exogenous inputs.
        xNow = [PhPrev, Pg(max(t-1,1)), Pp(max(t-1,1)), anchor];
        exoFc = repmat([measL(t), measW(t), measP(t)], H, 1);
        nFc = surrogate_net_horizon(surro, xNow, anchor, ...
            exoFc(:,1), exoFc(:,2), exoFc(:,3), ...
            ds.hour(hz) + ds.minute(hz)/60, dt(hz));
        if mode == 1
            [sol, info] = qp_central(nFc, dt(hz), ds.price(hz), PhPrev, Ecur, ...
                measL(hz), Np, Nc, lim);
        else
            [sol, info] = admm_coord(nFc, dt(hz), ds.price(hz), PhPrev, Ecur, ...
                measL(hz), Np, Nc, lim, duals, rho);
            duals = [info.duals(2:end); info.duals(end)];   % Shifted warm start.
            admmIter(t) = info.iters; admmPri(t) = info.rPri; admmDua(t) = info.rDua;
            commIters(t) = info.iters;
            if numel(traces) < 6 || info.iters == max(admmIter, [], 'omitnan')
                traces{end+1} = struct('t', t, 'pri', info.priTrace, ...
                    'dua', info.duaTrace, 'iters', info.iters, 'tol', info.tol); %#ok<AGROW>
            end
        end
        okFlag(t) = info.ok;
        plan = sol; age = 0; trig(t) = true;
        if mode == 1, commIters(t) = 1; end
        netFcApplied(t) = nFc(1);
    else
        % No communication: shift and reuse the previous plan.
        plan.Ph = plan.Ph([2:end end]);
        plan.Pg = plan.Pg([2:end end]);
        plan.Pp = plan.Pp([2:end end]);
        netFcApplied(t) = plan.Ph(1) + plan.Pg(1) - plan.Pp(1);
    end
    solveMs(t) = 1000 * toc(t0);

    ph = plan.Ph(1); pg = plan.Pg(1); pp = plan.Pp(1);
    if feedback && mode == 3 && ~doCoord
        % Correct part of the previous supply-demand error locally. Use
        % hydropower first, then pumped storage, with higher undersupply gain.
        corr = 0.50 * ePrev;
        if ePrev > 0, corr = 0.90 * ePrev; end
        dHyd = min(max(corr, -lim.ramp*dt(t)), lim.ramp*dt(t));
        ph = ph + dHyd;
        rem = corr - dHyd;
        if rem > 0, pg = pg + rem; else, pp = pp - rem; end
    end
    ph = min(max(ph, PhPrev - lim.ramp*dt(t)), PhPrev + lim.ramp*dt(t));
    ph = min(max(ph, 0), lim.phMax);
    pg = min(max(pg, 0), lim.pgMax);
    pp = min(max(pp, 0), lim.ppMax);
    if pg > 1 && pp > 1
        if pg >= pp, pp = 0; else, pg = 0; end     % Enforce mutually exclusive modes.
    end
    % Infer feasible generation and pumping limits from the energy bounds.
    pg = min(pg, Ecur * lim.etaG / max(dt(t), eps));
    pp = min(pp, (lim.eMax - Ecur) / lim.etaP / max(dt(t), eps));

    Ph(t) = ph; Pg(t) = pg; Pp(t) = pp;
    Enew = Ecur + lim.etaP*pp*dt(t) - pg*dt(t)/lim.etaG;
    if Enew < -1e-6 || Enew > lim.eMax + 1e-6, violation(t) = true; end
    Ecur = min(max(Enew, 0), lim.eMax);
    E(t) = Ecur;

    supply(t) = ph + pg - pp;
    short(t) = max(nTrue(t) - supply(t), 0);
    excess(t) = max(supply(t) - nTrue(t), 0);
    predErr(t) = nTrue(t) - netFcApplied(t);
    ePrev = nTrue(t) - supply(t);
    biasEst = 0.7 * biasEst + 0.3 * ePrev;
    emaAbsE = 0.8 * emaAbsE + 0.2 * abs(ePrev);
    if doCoord, cumE = 0; end
    cumE = cumE + max(ePrev, 0) * dt(t);

    % Independently verify ramping, mode exclusivity, and reserve constraints.
    if t > 1 && ds.group(t) == ds.group(t-1) && ...
            abs(Ph(t) - Ph(t-1)) > lim.ramp*dt(t) + 1e-6
        violation(t) = true;
    end
    if Pg(t) > 1 && Pp(t) > 1, violation(t) = true; end
    if (lim.phMax - ph) + (lim.pgMax - pg) + pp < 0.05*ds.load(t) - 1e-6
        violation(t) = true;
    end

    PhPrev = ph;
    age = age + 1;
end

out = struct('hydro', Ph, 'gen', Pg, 'pump', Pp, 'energy', E, ...
    'supply', supply, 'net', nTrue, 'short', short, 'excess', excess, ...
    'trigger', trig, 'cumComm', cumsum(trig), 'predErr', predErr, ...
    'netFc', netFcApplied, 'solveMs', solveMs, 'violation', violation);
out.traces = traces;
out.trigReason = why;
out.reasonNames = {'Scenario transition','Tracking error','Storage-energy boundary','Plan age','Cumulative undersupply'};
% Stepwise solver log: time, status, ADMM iterations, and residuals.
out.stepLog = table((1:n)', ds.group, trig, solveMs, okFlag, admmIter, admmPri, admmDua, ...
    'VariableNames', {'Step','Scenario group','Coordination solved','Solve time_ms','Solve succeeded', ...
    'ADMM iterations','Primal residual_MW','Dual residual_MW'});
out.eensMWh = zero_clamp(sum(short .* dt));
out.curtailMWh = zero_clamp(sum(excess .* dt));
out.costMUSD = sum((11*Ph + 18*Pg + ds.price.*Pp + 1000*short + 80*excess) .* dt) / 1e6;
out.rmseMW = sqrt(mean((supply - nTrue).^2));
out.violationPct = 100 * mean(violation);
out.commCount = sum(trig);
out.triggerPct = 100 * mean(trig);
out.commReductionPct = 100 * (1 - sum(trig)/n);
out.solveOkPct = 100 * mean(okFlag);
out.solveMsMean = mean(solveMs);
out.solveMsMax = max(solveMs);
out.admmIterMean = mean(admmIter, 'omitnan');
out.admmIterMax = max(admmIter, [], 'omitnan');
out.admmPriEnd = mean(admmPri, 'omitnan');
out.admmDuaEnd = mean(admmDua, 'omitnan');
sgn = sign(Pg - Pp);
out.modeSwitches = sum(sgn(2:end) ~= sgn(1:end-1) & sgn(2:end) ~= 0);
end

function v = getdef(s, f, d)
if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end

function x = zero_clamp(x)
if x < 1e-6, x = 0; end     % Suppress numerically insignificant roundoff.
end

function nFc = surrogate_net_horizon(surro, x0, anchor, L, W, P, hourVec, dtVec)
% Net-load horizon forecast. Anchor the first step to the measurement and
% propagate subsequent steps with the surrogate. Controlled outputs remain
% at their current values because the optimizer determines their sequence.
H = numel(L);
nFc = zeros(H, 1);
nFc(1) = min(max(anchor, 0), 12000);
x = x0;
for j = 2:H
    ang = 2*pi*hourVec(j)/24;
    S1 = struct('X', x, 'U', [L(j), W(j), P(j), sin(ang), cos(ang)], ...
        'Y', zeros(1,4), 'dt', dtVec(j), 'group', 1, 'seq', j, ...
        'first', false, 'n', 1);
    y = predict_surrogate(surro, S1);
    nFc(j) = min(max(y(4), 0), 12000);
    x = [x0(1:3), nFc(j)];
end
end
