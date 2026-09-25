function run_all(quickMode)
% Main experimental entry point. Run three groups of experiments and export all tables and figures.
% Experiment 1: surrogate modeling via SINDy / W-SINDy / PI-E-SINDy, plus parameter recovery with known ground truth.
% Experiment 2: active learning using Random / Uncertainty-AL / Risk-aware AL; multiple seeds with confidence intervals.
% Experiment 3: rolling dispatch with Centralized / Periodic-DMPC / ET-DMPC, ADMM-based distributed coordination.
% Run run_all() for final results; use run_all(true) for quick self-check after code modifications with reduced seeds.
% Requires MATLAB R2024b or later and the Optimization Toolbox.

if nargin < 1, quickMode = false; end
tAll = tic;
rng(20260810, 'twister');

here = fileparts(mfilename('fullpath'));
dataRoot = fullfile(here, 'data');
resDir = fullfile(here, 'results');
figDir = fullfile(here, 'figures');
if ~exist(resDir, 'dir'), mkdir(resDir); end
if ~exist(figDir, 'dir'), mkdir(figDir); end

exp1Dir = fullfile(dataRoot, 'identification');
exp2Dir = fullfile(dataRoot, 'mpc');

fprintf('Read data...\n');
E1S1tr = load_dataset(fullfile(exp1Dir, 'low_data_high_noise', 'train.csv'), 'Window index');
E1S1te = load_dataset(fullfile(exp1Dir, 'low_data_high_noise', 'test.csv'), 'Window index');
E1S2tr = load_dataset(fullfile(exp1Dir, 'typical_day_transfer', 'train.csv'), '');
E1S2te = load_dataset(fullfile(exp1Dir, 'typical_day_transfer', 'test.csv'), 'Scenario index');
E1S3tr = load_dataset(fullfile(exp1Dir, 'extreme_extrapolation', 'train.csv'), '');
E1S3pool = load_dataset(fullfile(exp1Dir, 'extreme_extrapolation', 'active_learning_pool.csv'), 'Scenario index');
E1S3te = load_dataset(fullfile(exp1Dir, 'extreme_extrapolation', 'test.csv'), 'Scenario index');

E2S1 = load_dataset(fullfile(exp2Dir, 'low_data_high_noise.csv'), 'Window index');
E2S2 = load_dataset(fullfile(exp2Dir, 'typical_day_transfer.csv'), 'Scenario index');
E2S3 = load_dataset(fullfile(exp2Dir, 'extreme_15min.csv'), 'Scenario index');
E2S1 = attach_clean_truth(E2S1, E1S3tr);

stageNames = {'Low data, high noise', 'Typical-day transfer', '15-min extreme extrapolation'};

% Experiment 1
fprintf('Experiment 1: surrogate model identification (SINDy / W-SINDy / PI-E-SINDy)...\n');
[modelTbl, predStore, baseModels] = run_exp1({E1S1tr, E1S2tr, E1S3tr}, ...
    {E1S1te, E1S2te, E1S3te}, E1S3tr, stageNames);
disp(modelTbl);
writetable(modelTbl, fullfile(resDir, 'table_5_2_surrogate_identification.csv'));

fprintf('Experiment 1 supplement: sparse-equation parameter recovery...\n');
recSeeds = 5; if quickMode, recSeeds = 2; end
recoveryTbl = run_recovery(E1S3tr, recSeeds);
disp(recoveryTbl);
writetable(recoveryTbl, fullfile(resDir, 'table_5_3_parameter_recovery.csv'));

% Experiment 2
nSeeds = 20; if quickMode, nSeeds = 3; end
fprintf('Experiment 2: active learning with %d random seeds...\n', nSeeds);
[alTbl, alCurves] = run_exp2_al(E1S3tr, E1S3pool, E1S3te, nSeeds);
disp(alTbl);
writetable(alTbl, fullfile(resDir, 'table_5_4_active_learning.csv'));

% Experiment 3
fprintf('Experiment 3: receding-horizon dispatch (Centralized / Periodic-DMPC / ET-DMPC)...\n');
surro = baseModels{3};       % Use the annual PI-E-SINDy baseline model in MPC.
[mpcTbl, simStore, ablTbl, ablStore, sensTbl, sensStore, mpcStepLog] = ...
    run_exp3_mpc({E2S1, E2S2, E2S3}, stageNames, surro);
disp(mpcTbl);
disp(ablTbl);
disp(sensTbl);
writetable(mpcTbl, fullfile(resDir, 'table_5_5_mpc_performance.csv'));
writetable(ablTbl, fullfile(resDir, 'table_5_6_residual_feedback_ablation.csv'));
writetable(sensTbl, fullfile(resDir, 'table_5_7_trigger_threshold_sensitivity.csv'));
writetable(mpcStepLog, fullfile(resDir, 'mpc_step_log.csv'));

% Export a consolidated workbook.
xlsx = fullfile(resDir, 'experiment_results.xlsx');
if exist(xlsx, 'file'), delete(xlsx); end
writetable(modelTbl, xlsx, 'Sheet', 'Surrogate identification');
writetable(recoveryTbl, xlsx, 'Sheet', 'Parameter recovery');
writetable(alTbl, xlsx, 'Sheet', 'Active learning');
writetable(mpcTbl, xlsx, 'Sheet', 'MPC performance');
writetable(ablTbl, xlsx, 'Sheet', 'Feedback ablation');
writetable(sensTbl, xlsx, 'Sheet', 'Threshold sensitivity');

save(fullfile(resDir, 'complete_results.mat'), 'modelTbl', 'recoveryTbl', 'alTbl', ...
    'alCurves', 'mpcTbl', 'ablTbl', 'sensTbl', 'predStore', 'simStore', ...
    'ablStore', 'sensStore', '-v7');

% Generate figures.
if exist('make_figures.m', 'file')
    fprintf('Generating figures...\n');
    make_figures(figDir, predStore, alCurves, simStore, ablStore, sensTbl, E2S3, mpcTbl);
end

fprintf('All experiments completed in %.1f minutes.\n', toc(tAll)/60);
end

function noisy = attach_clean_truth(noisy, clean)
% Map the low-data, high-noise scenario to the clean annual sequence for
% evaluation. Retain the noisy measurement in measNet for the controller.
[ok, loc] = ismember(noisy.seq, clean.seq);
f = {'load','wind','pv','hydro','gen','pump','soc','net','price'};
for k = 1:numel(f)
    v = noisy.(f{k});
    v(ok) = clean.(f{k})(loc(ok));
    noisy.(f{k}) = v;
end
end
