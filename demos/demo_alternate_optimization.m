%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Rodrigo Pena, rodrigo.pena@epfl.ch
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Script to test the alternate optimization problem

%% Setup
% Graph & signal parameters
N = 250; % Number of nodes
N_SPIKES = 2; % Number of spikes
sparam = struct('h_min', 1, ...
    'h_max', 1, ...
    'd_min', [], ...
    'd_max', [], ...
    'val_type', 'integer');

% True diffusion kernel parameter
theta_gt = 7;

M = ones(N, 1); % Observation mask
% M = double(rand(N,1) < .3);

% Parameters of the solver (see learn_sparse_signal.m)
param = struct( 'lambda', 5e-2, ...
    'alpha', 1e3, ...
    'beta', 0e0, ...
    'MAX_ITER', 200, ...
    'TOL', 1e-8, ...
    'MAX_ITER_x', 200, ...
    'TOL_x', 1e-5, ...
    'MAX_ITER_t', 10, ...
    'TOL_t', [], ...
    'constraint_x', @(x) x, ...
    'constraint_t', @(t) t, ...
    'method', 'smooth-newton');

snr = Inf;

% Level of information display
verbose = 3;

%% Generate graph & signals
gparam.distribute = 1;
gparam.nnparam.k = 6;
G = gsp_sensor(N, gparam);

% Compute Graph Fourier basis
G = gsp_create_laplacian(G, 'normalized');
G = gsp_compute_fourier_basis(G);

% Sparse original signal
x_spikes = pick_spikes(G, N_SPIKES, sparam);

% Our filter assumption:
kernel = choose_kernel('heat');

% Ground-truth low-pass filter:
g_gt = @(e) kernel.g(e, theta_gt);

% Diffused observations
b = gsp_filter(G, g_gt, x_spikes); % Diffuse
b = abs(awgn(b, snr, 'measured')); % Add noise
b = M .* b;

%% Initialization
x0 = mean(b) + std(b) .* rand(G.N, 1);
theta0 = start_theta(G, b);

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
    G.plotting.vertex_size = 200;
    
    figure;
    subplot(221);
    gsp_plot_signal(G, x_spikes);
    title('Signal of spikes');
    colorbar;
    set(gca, 'FontName', 'Times New Roman')
    set(gca, 'FontSize', 14)
    
    subplot(222);
    gsp_plot_signal(G, b);
    title('Noisy diffused signal of spikes');
    colorbar;
    set(gca, 'FontName', 'Times New Roman')
    set(gca, 'FontSize', 14)
    
    subplot(223);
    gsp_plot_signal(G, x);
    title('Solution x');
    colorbar;
    set(gca, 'FontName', 'Times New Roman')
    set(gca, 'FontSize', 14)
    
    set(gcf, 'Position', [0, 0, 1280, 960])
end

if verbose > 1   
    figure;
    subplot(211)
    gsp_plot_filter(G, g_gt);
    title('True filter');
    xlabel('\lambda');
    set(gca, 'FontName', 'Times New Roman')
    set(gca, 'Fontsize', 14)
    
    subplot(212)
    gsp_plot_filter(G, g_est);
    title('Estimated filter');
    xlabel('\lambda');
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
