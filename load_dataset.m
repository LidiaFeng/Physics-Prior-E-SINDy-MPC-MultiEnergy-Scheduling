function ds = load_dataset(csvPath, groupVar)
% Read CSV files of pure numerical scenarios and organize into a unified structure.
% groupVar specifies the column name for scenario/window grouping; leave empty for continuous full-length data.

T = readtable(csvPath, 'VariableNamingRule', 'preserve', 'Encoding', 'UTF-8');
n = height(T);

ds.seq   = pick(T, 'Time index', (1:n)');
if isempty(groupVar)
    ds.group = ones(n,1);
else
    ds.group = pick(T, groupVar, ones(n,1));
end
ds.dt    = pick(T, 'Time resolution_min', 60*ones(n,1)) / 60;   % hours
ds.load  = pick(T, 'Load power_MW', zeros(n,1));
ds.wind  = pick(T, 'Wind power_MW', zeros(n,1));
ds.pv    = pick(T, 'PV power_MW', zeros(n,1));
ds.hydro = pick(T, 'Hydropower output_MW', zeros(n,1));
ds.gen   = pick(T, 'Pumped-storage generation_MW', zeros(n,1));
ds.pump  = pick(T, 'Pumped-storage pumping_MW', zeros(n,1));
ds.soc   = pick(T, 'Stored energy_MWh', 3600*ones(n,1));
ds.net   = pick(T, 'Net load_MW', ds.load - ds.wind - ds.pv);
ds.price = pick(T, 'Yalong River proxy tariff_CNY_per_MWh', 450*ones(n,1)) / 7.2;  % converted to USD
ds.hour  = pick(T, 'Hour', zeros(n,1));
ds.minute = pick(T, 'Minute', zeros(n,1));

% measNet retains raw (possibly noisy) net load measurements; the net field can later be replaced with clean ground truth.
ds.measNet = ds.net;
end

function x = pick(T, name, fallback)
if any(strcmp(T.Properties.VariableNames, name))
    x = double(T.(name));
else
    x = fallback;
end
end
