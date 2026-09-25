function [tbl, simStore, ablTbl, ablStore, sensTbl, sensStore, stepLog] = run_exp3_mpc(dsList, stageNames, surro)
% MPC group experiments: three schemes including Centralized, Periodic-DMPC and ET-DMPC.
% Ablation study on residual feedback and trigger threshold sweeping for ET-DMPC.
% Solver wall-clock time is measured; step-wise logs are aggregated and saved separately in stepLog.

mpcNames = {'Centralized MPC', 'Periodic-DMPC', 'ET-DMPC'};
nS = numel(dsList);

% Warm-up run for quadprog to exclude JIT loading overhead from the first timing measurement.
quadprog(eye(2), [0;0], [], [], [], [], [0;0], [1;1], [], ...
    optimoptions('quadprog', 'Display', 'off'));

rows = cell(nS*3, 17);
simStore = cell(nS, 3);
logs = {};
r = 0;
for s = 1:nS
    for m = 1:3
        r = r + 1;
        sim = simulate_mpc(dsList{s}, m, surro, struct());
        simStore{s,m} = sim;
        rows(r,:) = {stageNames{s}, mpcNames{m}, sim.costMUSD, sim.curtailMWh, ...
            sim.eensMWh, sim.rmseMW, sim.violationPct, sim.commCount, ...
            sim.triggerPct, sim.commReductionPct, sim.solveMsMean, sim.solveMsMax, ...
            sim.solveOkPct, sim.admmIterMean, sim.admmIterMax, sim.admmPriEnd, sim.admmDuaEnd};
        L = sim.stepLog;
        L = addvars(L, repmat(stageNames(s), height(L), 1), ...
            repmat(mpcNames(m), height(L), 1), 'Before', 1, ...
            'NewVariableNames', {'Scenario', 'Method'});
        logs{end+1} = L; %#ok<AGROW>
    end
end
tbl = cell2table(rows, 'VariableNames', {'Scenario','Method','Total cost_Million USD','Curtailment_MWh', ...
    'ENS_MWh','RMSE_MW','Constraint violation_pct','Communication count','Trigger rate_pct','Communication reduction_pct', ...
    'Mean solve time_ms','Max solve time_ms','Solve success rate_pct','Mean ADMM iterations','Max ADMM iterations', ...
    'Primal residual_converged mean','Dual residual_converged mean'});
stepLog = vertcat(logs{:});

% Residual feedback ablation: ET-DMPC with / without feedback.
ablStore = cell(nS, 2);
ablRows = cell(nS*2, 7);
q = 0;
for s = 1:nS
    with = simStore{s,3};
    without = simulate_mpc(dsList{s}, 3, surro, struct('feedback', false));
    ablStore{s,1} = with; ablStore{s,2} = without;
    gain = 100 * (without.eensMWh - with.eensMWh) / max(without.eensMWh, eps);
    pair = {with, 'With', gain; without, 'Without', 0};
    for k = 1:2
        q = q + 1;
        z = pair{k,1};
        ablRows(q,:) = {stageNames{s}, pair{k,2}, z.eensMWh, z.rmseMW, ...
            z.triggerPct, z.commReductionPct, pair{k,3}};
    end
end
ablTbl = cell2table(ablRows, 'VariableNames', ...
    {'Scenario','Residual feedback','ENS_MWh','RMSE_MW','Trigger rate_pct','Communication reduction_pct','ENS improvement_pct'});

% Trigger threshold sweep, tested on the 15-min extreme scenario.
thrList = [0.7, 0.8, 1.0, 1.3];
sensStore = cell(numel(thrList), 1);
sensRows = cell(numel(thrList), 6);
for k = 1:numel(thrList)
    z = simulate_mpc(dsList{end}, 3, surro, struct('thrScale', thrList(k)));
    sensStore{k} = z;
    sensRows(k,:) = {thrList(k), z.commCount, z.rmseMW, z.eensMWh, ...
        z.triggerPct, z.solveMsMean};
end
sensTbl = cell2table(sensRows, 'VariableNames', ...
    {'Threshold multiplier','Communication count','RMSE_MW','ENS_MWh','Trigger rate_pct','Mean solve time_ms'});
end
