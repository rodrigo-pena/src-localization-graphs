%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Rodrigo Pena, rodrigo.pena@epfl.ch
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Script to test the influence of the number of neighbors of each node in
% the Snow GIS reduced graph on the accuracy of the solution of alt_opt.m

%% Setup

verbose = 1;

% Parameters of the solver (see learn_sparse_signal.m)
param = struct( 'lambda', [], ...
    'alpha', 10, ...
    'beta', 0e0, ...
    'MAX_ITER', 200, ...
    'TOL', 1e-6, ...
    'MAX_ITER_x', 150, ...
    'TOL_x', 1e-4, ...
    'MAX_ITER_t', 0, ...
    'TOL_t', 1e-5, ...
    'constraint_x', @(x) x, ...
    'constraint_t', @(t) t, ...
    'method', 'smooth-newton');

max_num_neighbors = 258;
err_param = struct('TOL', 0.01);
err_sparse = NaN(1, max_num_neighbors - 1);
err_hop = NaN(1, max_num_neighbors - 1);
err_sparse_max_removed = NaN(1, max_num_neighbors - 1);
err_hop_max_removed = NaN(1, max_num_neighbors - 1);

% Our filter assumption:
kernel = choose_kernel('heat');

% Open video object
if verbose > 1
    writerObj = VideoWriter([pwd, '/results/snow_gis/neighbours.avi']);
    writerObj.FrameRate = 2;
    writerObj.Quality = 100;
    open(writerObj);
    fig_handle = figure;
    set(fig_handle, 'Position', [100, 100, 1080, 720]);
    set(gca,'nextplot','replacechildren');
    set(gcf,'color', 'white');
    axis tight
end

%% Vary the number of neighbours
for flag_max_removed = [false, true]
    for i = 2:max_num_neighbors
        [G, x_spikes, b, M] = snow_gis_reduce([], 4, i, 20);
        G = gsp_create_laplacian(G, 'normalized');
        G = gsp_compute_fourier_basis(G);
        
        if flag_max_removed
            [~, ind] = max(b);
            b(ind) = 0;
            M = double(b ~= 0);
        end       
        
        % Initialization points
        x0 = zeros(G.N, 1);
        theta0 = 6.5;
        
        % Solve alternate optimization
        [x, ~, g_est] = alt_opt(G, b, kernel, M, theta0, x0, param);
        
        % Compute errors
        if flag_max_removed
            err_sparse_max_removed(i - 1) = sparse_error(G, x_spikes, ...
                x, err_param);
            err_hop_max_removed(i - 1) = hop_error(G, x_spikes, ...
                x, err_param);
        else
            err_sparse(i - 1) = sparse_error(G, x_spikes, x, err_param);
            err_hop(i - 1) = hop_error(G, x_spikes, x, err_param);
        end
        
        % Record frames
        if verbose > 1
            gsp_plot_filter(G, g_est);
            drawnow;
            frame = getframe(gcf);
            writeVideo(writerObj,frame);
        end
    end
end
%% Figures
if verbose > 0
    figure;
    plot(2:max_num_neighbors, err_hop, 'LineWidth', 2);
    hold on
    plot(2:max_num_neighbors, err_hop_max_removed, 'LineWidth', 2);
    set(gcf, 'Position', [100, 100, 1080, 720]);
    set(gca, 'FontName', 'Times New Roman');
    set(gca, 'FontSize', 14);
    xlabel('# neighbors', 'FontSize', 20)
    ylabel('Average hop error', 'FontSize', 20)
    h_l = legend('b as is', 'b with max removed');
    set(h_l, 'FontSize', 14)
end
