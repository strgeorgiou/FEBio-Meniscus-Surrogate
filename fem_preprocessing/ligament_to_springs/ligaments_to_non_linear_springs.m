% MATLAB script to replace ligaments with non linear springs

% Clear workspace and command window
clear; clc;

% Define constants
k = ;  % Slope constant 
epsilon_l = ; % Threshold strain value
Lr =; % Reference length of the spring 
epsilon_r =; % Reference strain of the ligament
L0 = Lr / (epsilon_r + 1); % Zero-load length of the spring



% Define a range of L values for plotting
L_values = linspace(0, 3000, 1000); % Adjusted range
f_values = zeros(size(L_values)); % Initialize f values
epsilon_values = (L_values - L0) / (L0); % Compute epsilon for each L

% Calculate f(epsilon) for each L
for i = 1:length(L_values)
    epsilon = epsilon_values(i);
    if epsilon >= 0 && epsilon <= 2 * epsilon_l
        f_values(i) = 0.25 * k * epsilon^2 / epsilon_l;
    elseif epsilon > 2 * epsilon_l
        f_values(i) = k * (epsilon - epsilon_l);
    else
        f_values(i) = 0;
    end
end

% Export the results to a CSV file
results = table(epsilon_values', f_values', 'VariableNames', {'strain', 'force'});
writetable(results, 'f_epsilon_resultsM0.csv');


fprintf('Results have been exported to "f_epsilon_results.csv".\n');

% Plot the results
figure;
plot(epsilon_values, f_values, 'b-', 'LineWidth', 1.5);
grid on;
xlabel('Strain');
ylabel('Force');
title('Plot of Force vs. Strain');