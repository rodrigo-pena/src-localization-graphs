
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Rodrigo Pena, rodrigo.pena@epfl.ch
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Script to test the optimization problems 
% (1) argmin_theta (||x||_1 + (alpha/2)||A(theta)*x - b||_2^2) = 
%       argmin_theta E(theta)
% and
% (2) argmin_theta ||x||_1 + (alpha/2)||A(theta)*x - b||_2^2 + ...
%                  (beta/2)||(A(theta)*A(theta)'-I)*x||_2^2 = 
%       argmin_theta E(theta)

%% Setup
% Graph & signal parameters
N = 250; % Number of nodes
N_SPIKES = 2; % Number of spikes
sparam = struct('h_min', 1, ...
                'h_max', 1, ...
                'd_min', 4, ...
                'val_type', 'integer');

% True diffusion kernel parameter
theta_gt = 50; 

% Observation mask
M = ones(N, 1);
% M = double(rand(N,1) < .9);

% Parameters of the solver (see learn_param_kernel.m)
param = struct( 'lambda', 1e-1, ...
                'alpha', 1e4, ...
                'beta', 0e0, ...
                'MAX_ITER', 1000, ...
                'TOL', 1e-8, ...
                'method', 'smooth-newton');
            
snr = Inf;
            
% Level of information display
verbose = 3;

%% Generate graph & signals
gparam.distribute = 1;
gparam.nnparam.k = 9;
G = gsp_sensor(N, gparam);

G = gsp_create_laplacian(G, 'normalized');
G = gsp_compute_fourier_basis(G);

x = pick_spikes(G, N_SPIKES, sparam);

kernel = choose_kernel('heat');
g_gt = @(e) kernel.g(e, theta_gt); % Ground-truth filter

b = gsp_filter(G, g_gt, x);
b = abs(awgn(b, snr, 'measured')); % Add noise
b = M .* b;

%% Initialization
% (Tested with theta_gt = 10.0)
% Newton: -25.5*theta_gt <= theta0 < 1.52*theta_gt
% Smooth Newton: -27*theta_gt <= theta0 < 53*theta_gt

%theta0 = start_theta(G, b);
theta0 = 0;

%% Solve the kernel learning problem assuming we know x_spikes:
tic;
[theta, g_est, energy_vec] = learn_param_kernel(G, b, x, kernel, ...
    [], theta0, param);
finish = toc;

fprintf('Time to solve optimization: %1.4f s\n', finish);
fprintf('Estimated theta: %1.2f\n', theta);

n = length(energy_vec);
diff_vec = abs(energy_vec(1:end-1) - energy_vec(end));

%% Display results
if verbose > 0
    G.plotting.vertex_size = 200;
    
    figure(1);
    subplot(221);
    gsp_plot_signal(G, x);
    title('Original sparse signal');
    colorbar;
    set(gca, 'FontName', 'Times New Roman')
    set(gca, 'FontSize', 14)
    
    subplot(222);
    gsp_plot_signal(G, b);
    title('Observed diffused signal');
    colorbar;
    set(gca, 'FontName', 'Times New Roman')
    set(gca, 'FontSize', 14)
    
    subplot(223);
    gsp_plot_signal(G, gsp_filter(G, g_est, x));
    title('Diffused signal with the learnt kernel');
    colorbar;
    set(gca, 'FontName', 'Times New Roman')
    set(gca, 'FontSize', 14)
    
    set(gcf, 'Position', [0, 0, 1280, 960])
end

if verbose > 1
    figure(2)
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
    figure(3)
    subplot(121)
    loglog(1:n, energy_vec)
    title('E(\theta_k)')
    xlabel('Iteration number (k)');
    set(gca, 'FontName', 'Times New Roman')
    set(gca, 'FontSize', 14)
    grid on
    
    subplot(122)
    loglog(1:n-1, diff_vec)
    title('|E(\theta_k) - E(\theta_\infty)|')
    xlabel('Iteration number (k)');
    set(gca, 'FontName', 'Times New Roman')
    set(gca, 'FontSize', 14)
    grid on

    % Slope of the LogLog plot
    pFit = polyfit(log(1:n-1),log(diff_vec),1);
    slope = pFit(1);
    fprintf('Slope of Energy Difference plot: %1.2f\n', slope);
end