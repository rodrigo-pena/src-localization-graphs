%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Rodrigo Pena, rodrigo.pena@epfl.ch
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Script to test the sparse signal recovery on the nodes of a graph G,
% given the diffused observations and the diffusion kernel
%
% Requires: GSPBox (https://lts2.epfl.ch/gsp/)

%% Setup
% Graph & signal parameters
N = 250; % Number of nodes
N_SPIKES = 2; % Number of spikes
sparam = struct('h_min', 1, ...
                'h_max', 1, ...
                'd_min', 4, ...
                'val_type', 'integer');

% True diffusion kernel parameter
theta_gt = 20; 

M = ones(N, 1); % Observation mask
% M = double(rand(N,1) < .3);

% Parameters of the solver (see learn_sparse_signal.m)
param = struct( 'lambda', 1e-1, ...
                'alpha', 1e4, ...
                'beta', 0e0, ...
                'MAX_ITER', 1000, ...
                'TOL', 1e-8, ...
                'constraint', @(x) x);
            
snr = Inf;

% Level of information display
verbose = 3;

%% Generate graph & signals
gparam.distribute = 1;
gparam.nnparam.k = 9;
G = gsp_sensor(N, gparam);

G = gsp_create_laplacian(G, 'normalized');
G = gsp_compute_fourier_basis(G);

x_spikes = pick_spikes(G, N_SPIKES, sparam);

kernel = choose_kernel('heat');

b = gsp_filter(G, @(e) kernel.g(e, theta_gt), x_spikes);
b = abs(awgn(b, snr, 'measured')); % Add noise
b = M .* b;

%% Initialization
x0 = mean(b) + std(b) .* rand(G.N, 1);

%% Solve optimization problem
tic;
[x, energy_vec] = learn_sparse_signal(G, b, @(e) kernel.g(e, theta_gt), M, x0, param);
finish = toc;

fprintf('Time to solve optimization problem: %1.4f s\n', finish);

err_param = struct('TOL', 0.01);

err_sparse = sparse_error(G, x_spikes, x, err_param);
fprintf('Sparse error: %1.4f\n', err_sparse);

err_hop = hop_error(G, x_spikes, x, err_param);
fprintf('Hop error: %1.4f\n', err_hop);

n = length(energy_vec);
diff_vec = abs(energy_vec(1:end-1) - energy_vec(end));

%% Figures
if verbose > 0
    G.plotting.vertex_size = 200;
    
    figure(1);
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
    figure(2);
    gsp_plot_filter(G, @(e) kernel.g(e, theta_gt));
    xlabel('\lambda');
    title('Spectral diffusion kernel');
    set(gca, 'FontName', 'Times New Roman')
    set(gca, 'Fontsize', 14)
    set(gcf, 'Position', [500, 500, 1080, 720])
end

if verbose > 2
    figure(3)
    subplot(121)
    loglog(1:n, energy_vec)
    title('E(x_k)')
    xlabel('Iteration number (k)');
    set(gca, 'FontName', 'Times New Roman')
    set(gca, 'FontSize', 14)
    grid on
    
    subplot(122)
    loglog(1:n-1, diff_vec)
    title('|E(x_k) - E(x_\infty)|')
    xlabel('Iteration number (k)');
    set(gca, 'FontName', 'Times New Roman')
    set(gca, 'FontSize', 14)
    grid on
    
    set(gcf, 'Position', [1000, 1000, 1080, 720])
    
    % Slope of the LogLog plot
    pFit = polyfit(log(1:n-1),log(diff_vec),1);
    slope = pFit(1);
    fprintf('Slope of Energy Difference plot: %1.2f\n', slope);
end