%% Mesh convergence analysis for FEBio meniscus stress/strain results
% Produces 4 plots:
% 1) Combined stress distributions vs normalized element rank
% 2) Combined strain distributions vs normalized element rank
% 3) Mean stress shown as horizontal lines vs normalized element rank
% 4) Mean strain shown as horizontal lines vs normalized element rank

clear; clc; close all;

%% Select main folder
% This folder must contain:
% size 1, size 0.5, size 0.25, size 0.1, size 0.05
base_folder = uigetdir('', 'Select the folder that contains the size folders');

%% Element sizes
element_sizes = [1, 0.5, 0.25, 0.1, 0.05];

%% Choose meniscus
% 'mm' = medial meniscus
% 'lm' = lateral meniscus
meniscus = 'lm';   % change to 'lm' for lateral meniscus

%% Initialize arrays
mean_stress = zeros(size(element_sizes));
mean_strain = zeros(size(element_sizes));
p95_stress  = zeros(size(element_sizes));
p95_strain  = zeros(size(element_sizes));

all_stress_data = cell(size(element_sizes));
all_strain_data = cell(size(element_sizes));

%% Read files
for i = 1:length(element_sizes)

    size_str = num2str(element_sizes(i));
    current_folder = fullfile(base_folder, ['size ', size_str]);

    stress_pattern = fullfile(current_folder, [meniscus, '_stress*']);
    strain_pattern = fullfile(current_folder, [meniscus, '_strain*']);

    stress_match = dir(stress_pattern);
    strain_match = dir(strain_pattern);

    if isempty(stress_match)
        error('No stress file found in folder: %s', current_folder);
    end

    if isempty(strain_match)
        error('No strain file found in folder: %s', current_folder);
    end

    stress_file = fullfile(current_folder, stress_match(1).name);
    strain_file = fullfile(current_folder, strain_match(1).name);

    fprintf('Reading:\n%s\n%s\n\n', stress_file, strain_file);

    stress_data = read_febio_element_data(stress_file);
    strain_data = read_febio_element_data(strain_file);

    all_stress_data{i} = stress_data;
    all_strain_data{i} = strain_data;

    mean_stress(i) = mean(stress_data);
    mean_strain(i) = mean(strain_data);

    p95_stress(i) = percentile_value(stress_data, 95);
    p95_strain(i) = percentile_value(strain_data, 95);

end

%% Save table
results = table(element_sizes', mean_stress', mean_strain', p95_stress', p95_strain', ...
    'VariableNames', {'ElementSize','MeanStress','MeanStrain','P95Stress','P95Strain'});
disp(results);
writetable(results, ['mesh_convergence_results_', meniscus, '.csv']);

%% Plot settings
colors = lines(length(element_sizes));

% Use this x-axis for "distribution-style" plots
x_rank_plot = linspace(90, 100, 1000);

%% -------------------------------------------------
%% Plot 1: Combined STRESS distributions (zoomed 90-100%)
fig1 = figure;
hold on;

for i = 1:length(element_sizes)

    sorted_stress = sort(all_stress_data{i});
    x_rank = linspace(0, 100, length(sorted_stress));

    plot(x_rank, sorted_stress, ...
        'LineWidth', 1.5, ...
        'Color', colors(i,:));

end

grid on;
xlim([95 100])
ylim([0 0.6])
xlabel('Normalized element rank [%]');
ylabel('Stress');
title(['Combined stress distributions - ', upper(meniscus)]);
legend(compose('size = %g', element_sizes), 'Location', 'best');
xlim([90 100]);
hold off;

%% -------------------------------------------------
%% Plot 2: Combined STRAIN distributions (zoomed 90-100%)
fig2 = figure;
hold on;

for i = 1:length(element_sizes)

    sorted_strain = sort(all_strain_data{i});
    x_rank = linspace(0, 100, length(sorted_strain));

    plot(x_rank, sorted_strain, ...
        'LineWidth', 1.5, ...
        'Color', colors(i,:));

end

grid on;
xlim([95 100])
ylim([0 0.035])
xlabel('Normalized element rank [%]');
ylabel('Strain');
title(['Combined strain distributions - ', upper(meniscus)]);
legend(compose('size = %g', element_sizes), 'Location', 'best');
xlim([90 100]);
hold off;

%% -------------------------------------------------

%% Save figures
saveas(fig1, ['combined_stress_distribution_zoom_', meniscus, '.png']);
saveas(fig2, ['combined_strain_distribution_zoom_', meniscus, '.png']);


%% ---------------- Local functions ----------------

function values = read_febio_element_data(filename)

    fid = fopen(filename, 'r');

    if fid == -1
        error('Could not open file: %s', filename);
    end

    values = [];

    while ~feof(fid)

        line = strtrim(fgetl(fid));

        if isempty(line)
            continue;
        end

        if startsWith(line, '*')
            continue;
        end

        nums = sscanf(line, '%f');

        if length(nums) >= 2
            values(end+1,1) = nums(2);
        end

    end

    fclose(fid);

end

function p = percentile_value(data, percentile)

    data = sort(data(:));
    n = length(data);

    if n == 0
        error('Empty data array.');
    end

    index = ceil((percentile / 100) * n);
    index = max(1, min(index, n));

    p = data(index);

end