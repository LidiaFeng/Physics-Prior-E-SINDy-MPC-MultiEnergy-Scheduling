function sc = lib_scales()
% Fixed physical ranges (MW), shared across all methods and datasets to form the unified coordinate system.
% Order: [hydropower, pumped-storage generation, pumping, net load, load, wind power, photovoltaic, sin, cos]
sc = [10800, 1200, 950, 12000, 12000, 1200, 3000, 1, 1];
end
