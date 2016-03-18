%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test sparse_error
% Requires: MatlabBGL (http://dgleich.github.io/matlab-bgl/)
%           GSPBox (https://lts2.epfl.ch/gsp/)
%
% Author: Rodrigo C. G. Pena (rodrigo.pena@epfl.ch)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Setup
close all
clc

gsp_reset_seed(0);

% Parameters
N = 200; % Number of nodes
theta = 1.0;

%% Create graph
gparam.distribute = 1;
G = gsp_sensor(N, gparam);
G = gsp_create_laplacian(G, 'normalized');
G = gsp_compute_fourier_basis(G);

%% Create signals
x_ref = zeros(G.N,1);
x_ref([30, 50, 70]) = 1;

% x0 = zeros(G.N,1);
x0 = gsp_filter(G, @(e) exp(-theta .* e), x_ref);

x1 = zeros(G.N,1);
x1([20, 40, 60]) = 1;

x2 = zeros(G.N,1);
x2([25, 45, 65]) = 1;

%% Compute errors
error = Inf(3, 3);

error(1,1) = norm(x_ref - x0, 1)./norm(x_ref, 1);
error(1,2) = norm(x_ref - x1, 1)./norm(x_ref, 1);
error(1,3) = norm(x_ref - x2, 1)./norm(x_ref, 1);

error(2,1) = sparse_error(G, x_ref, x0);
error(2,2) = sparse_error(G, x_ref, x1);
error(2,3) = sparse_error(G, x_ref, x2);

error(3,1) = hop_error(G, x_ref, x0);
error(3,2) = hop_error(G, x_ref, x1);
error(3,3) = hop_error(G, x_ref, x2);

%% Plot signals
figure(1);
subplot(221)
gsp_plot_signal(G, x_ref);
title('x_{ref}')
subplot(222)
gsp_plot_signal(G, x0);
title(['naive-error(x_{ref}, x_0) = ', num2str(error(1,1))])
subplot(223)
gsp_plot_signal(G, x1);
title(['naive-error(x_{ref}, x_1) = ', num2str(error(1,2))])
subplot(224)
gsp_plot_signal(G, x2);
title(['naive-error(x_{ref}, x_2) = ', num2str(error(1,3))])

figure(2);
subplot(221)
gsp_plot_signal(G, x_ref);
title('x_{ref}')
subplot(222)
gsp_plot_signal(G, x0);
title(['sparse-error(x_{ref}, x_0) = ', num2str(error(2,1))])
subplot(223)
gsp_plot_signal(G, x1);
title(['sparse-error(x_{ref}, x_1) = ', num2str(error(2,2))])
subplot(224)
gsp_plot_signal(G, x2);
title(['sparse-error(x_{ref}, x_2) = ', num2str(error(2,3))])

figure(3);
subplot(221)
gsp_plot_signal(G, x_ref);
title('x_{ref}')
subplot(222)
gsp_plot_signal(G, x0);
title(['hop-error(x_{ref}, x_0) = ', num2str(error(3,1))])
subplot(223)
gsp_plot_signal(G, x1);
title(['hop-error(x_{ref}, x_1) = ', num2str(error(3,2))])
subplot(224)
gsp_plot_signal(G, x2);
title(['hop-error(x_{ref}, x_2) = ', num2str(error(3,3))])
