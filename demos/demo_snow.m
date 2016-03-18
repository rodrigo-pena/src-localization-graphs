%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Rodrigo Pena, rodrigo.pena@epfl.ch
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Script to test the alternate optimization problem with John Snow's GIS
% data about a cholera outbreak in London in 1854.

%% Setup
% Parameters of the solver (see alt_opt.m)
param = struct( 'lambda', [], ...
    'alpha', 1e1, ...
    'beta', 0e0, ...
    'MAX_ITER', 200, ...
    'TOL', 1e-8, ...
    'MAX_ITER_x', 150, ...
    'TOL_x', 1e-5, ...
    'MAX_ITER_t', 0, ...
    'TOL_t', [], ...
    'constraint_x', @(x) x, ...
    'constraint_t', @(t) t, ...
    'method', 'smooth-newton');

% Level of information display
verbose = 3;

flag = nan;
while (flag ~= 0) && (flag ~= 1)
    flag = input('Discard info from node with max death count?\n[(yes=1)/(no=0)]: ');
    assert(isnumeric(flag) && ~isempty(flag), ...
        'The answer must be a number');
end

%% Generate graph & signals
if flag
    [G, x_spikes, b, ~] = snow_gis_reduce([], 4, 70, 20);
    G = gsp_create_laplacian(G, 'normalized');
    G = gsp_compute_fourier_basis(G);
    
    b(b == max(b)) = 0;
    M = ones(size(b));
    b = gsp_interpolate(G, b(b ~= 0), find(b ~= 0));
else
    [G, x_spikes, b, M] = snow_gis_reduce([], 4, 38, 20);
    G = gsp_create_laplacian(G, 'normalized');
    G = gsp_compute_fourier_basis(G);
end

% Our filter assumption:
kernel = choose_kernel('heat');

%% Initialization
x0 = mean(b) + std(b) .* rand(G.N, 1);
theta0 = 6.5;

%% Solve alternate optimization
tic;
[x, theta, g_est, energy_vec] = alt_opt(G, b, kernel, M, theta0, x0, param);
finish = toc;

n = length(energy_vec);
diff_vec = abs(energy_vec(1:end-1) - energy_vec(end));

fprintf('Time to solve optimization problem: %1.4f s\n', finish);

err_param = struct('TOL', 0.01);

err_sparse = sparse_error(G, x_spikes, x, err_param);
fprintf('Sparse error: %1.4f\n', err_sparse);

err_hop = hop_error(G, x_spikes, x, err_param);
fprintf('Hop error: %1.4f\n', err_hop);

%% Figures
if verbose > 0
    figure;
    subplot(221);
    plot_snow_gis(G, x_spikes);
    title('Infected pump');
    set(gca, 'FontName', 'Times New Roman')
    set(gca, 'FontSize', 14)
    
    subplot(222);
    plot_snow_gis(G, b);
    title('Observed death count');
    set(gca, 'FontName', 'Times New Roman')
    set(gca, 'FontSize', 14)
    
    subplot(223);
    plot_snow_gis(G, x);
    title('Solution');
    set(gca, 'FontName', 'Times New Roman')
    set(gca, 'FontSize', 14)
    
    set(gcf, 'Position', [500, 500, 1280, 960])
end

if verbose > 1
    figure;
    gsp_plot_filter(G, g_est);
    xlabel('\lambda')
    title('Estimated diffusion filter');
    set(gca, 'FontName', 'Times New Roman')
    set(gca, 'Fontsize', 14)
    set(gcf, 'Position', [500, 500, 1080, 720])
end

if verbose > 2
    figure;
    subplot(121)
    loglog(1:n, energy_vec)
    title('E(x_k, \theta_k)')
    xlabel('Iteration number (k)');
    set(gca, 'FontName', 'Times New Roman')
    set(gca, 'FontSize', 14)
    grid on
    
    subplot(122)
    loglog(1:n-1, diff_vec)
    title('|E(x_k, \theta_k) - E(x_\infty, \theta_\infty)|')
    xlabel('Iteration number (k)');
    set(gca, 'FontName', 'Times New Roman')
    set(gca, 'FontSize', 14)
    grid on
    
    set(gcf, 'Position', [500, 500, 1080, 720])
    
    pFit = polyfit(log(1:n-1),log(diff_vec),1);
    slope = pFit(1);
    fprintf('Slope of Energy Difference plot: %1.2f\n', slope);
end
