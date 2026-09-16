function [best_x, best_f, convergence, diversity] = SMTS(obj_func, dim, bounds, pop_size, max_evals, k_neighbors)
% SMTS - Starling Murmuration Topological Search Optimizer (MATLAB Implementation)
%
% Grounded in the empirical physics of European starlings (Cavagna et al., 2010; Attanasi et al., 2014):
% 1. Density-invariant topological k-NN interaction graph (k in [6, 7]).
% 2. Undamped kinematic spin-wave velocity alignment consensus.
% 3. Gradient-free topological simplex regression operator.
% 4. Adaptive phase-shift exploration-to-exploitation scheduling.
% 5. Adaptive stochastic interior boundary rebound.
%
% Syntax:
%   [best_x, best_f, convergence, diversity] = SMTS(obj_func, dim, bounds, pop_size, max_evals, k_neighbors)
%
% Inputs:
%   obj_func    - Function handle for objective: f = obj_func(x) where x is 1xD row vector
%   dim         - Problem dimensionality (D)
%   bounds      - [lb, ub] 1x2 scalar bounds or 2xD matrix [lb; ub]
%   pop_size    - Swarm size N (default: 50)
%   max_evals   - Evaluation budget (default: 1000 * dim)
%   k_neighbors - Topological neighborhood size (default: 7, biologically optimal)
%
% Outputs:
%   best_x      - Global best candidate solution found (1xD)
%   best_f      - Global best objective function value (scalar)
%   convergence - Vector of best_f values tracked over generations
%   diversity   - Vector of population spatial diversity metrics over time

    % Parameter defaults
    if nargin < 4 || isempty(pop_size), pop_size = 50; end
    if nargin < 5 || isempty(max_evals), max_evals = 1000 * dim; end
    if nargin < 6 || isempty(k_neighbors), k_neighbors = 7; end

    % Parse bounds
    if isvector(bounds) && length(bounds) == 2
        lb = bounds(1) * ones(1, dim);
        ub = bounds(2) * ones(1, dim);
    else
        lb = bounds(1, :);
        ub = bounds(2, :);
    end
    domain_span = ub - lb;
    k_eff = min(k_neighbors, pop_size - 1);

    % Dynamic parameter bounds
    w_max = 0.9;
    w_min = 0.4;
    c1_init = 1.2;
    c2_init = 1.0;
    c3_init = 1.2;
    c4_final = 1.5;
    fluctuation_amp = 0.05;

    % 1. Population and Velocity Initialization
    X = repmat(lb, pop_size, 1) + rand(pop_size, dim) .* repmat(domain_span, pop_size, 1);
    V = -0.1 * repmat(domain_span, pop_size, 1) + 0.2 * rand(pop_size, dim) .* repmat(domain_span, pop_size, 1);

    % Initial evaluations
    fitness = zeros(pop_size, 1);
    evals = 0;
    for i = 1:pop_size
        if evals >= max_evals, break; end
        fitness(i) = obj_func(X(i, :));
        evals = evals + 1;
    end

    % Personal bests and Global best
    P = X;
    P_fit = fitness;
    [best_f, best_idx] = min(P_fit);
    best_x = P(best_idx, :);

    convergence = [best_f];
    centroid = mean(X, 1);
    init_div = mean(sqrt(sum((X - repmat(centroid, pop_size, 1)).^2, 2))) / (norm(domain_span) + 1e-12);
    diversity = [init_div];

    % Optimization loop
    while evals < max_evals
        tau = min(1.0, evals / max_evals);

        % Parameter schedules
        w = w_max - (w_max - w_min) * tau;
        c1 = c1_init * (1.0 - 0.5 * tau);
        c2 = c2_init * (1.0 - tau);
        c3 = c3_init * (1.0 - 0.5 * tau);
        c4 = c4_final * (tau^2);

        % 2. Dynamic Topological k-NN Lookup
        % Vectorized Euclidean pairwise distance matrix
        D_mat = zeros(pop_size, pop_size);
        for i = 1:pop_size
            diff_i = X - repmat(X(i, :), pop_size, 1);
            D_mat(i, :) = sqrt(sum(diff_i.^2, 2))';
            D_mat(i, i) = Inf; % Exclude self-loop
        end

        % Sort to get k nearest neighbors
        [sorted_dists, sorted_indices] = sort(D_mat, 2, 'ascend');
        neighbor_indices = sorted_indices(:, 1:k_eff);
        neighbor_dists = sorted_dists(:, 1:k_eff);

        % 3. Kinematic Spin-Wave Consensus
        a_spin = zeros(pop_size, dim);
        for i = 1:pop_size
            nbrs = neighbor_indices(i, :);
            mean_v_nbr = mean(V(nbrs, :), 1);
            a_spin(i, :) = mean_v_nbr - V(i, :);
        end

        % 4. Topological Simplex Gradient Descent
        topo_grad = zeros(pop_size, dim);
        for i = 1:pop_size
            nbrs = neighbor_indices(i, :);
            f_i = fitness(i);
            f_nbrs = fitness(nbrs);
            delta_f = f_i - f_nbrs; % Positive when neighbor is better
            dist_sq = (neighbor_dists(i, :).^2)' + 1e-12;

            disp_mat = X(nbrs, :) - repmat(X(i, :), k_eff, 1);
            weight_diff = max(-1e4, min(1e4, delta_f ./ dist_sq));

            raw_grad = sum(repmat(weight_diff, 1, dim) .* disp_mat, 1);
            norm_g = norm(raw_grad) + 1e-12;
            mean_d = mean(neighbor_dists(i, :));
            topo_grad(i, :) = (raw_grad / norm_g) * mean_d;
        end

        % 5. Local Topological Elite Attraction
        a_elite = zeros(pop_size, dim);
        for i = 1:pop_size
            closed_nbrs = [neighbor_indices(i, :), i];
            [~, min_sub_idx] = min(P_fit(closed_nbrs));
            local_best_idx = closed_nbrs(min_sub_idx);
            a_elite(i, :) = P(local_best_idx, :) - X(i, :);
        end

        % 6. Scale-Free Cauchy Thermal Fluctuations
        mean_dists_all = mean(neighbor_dists, 2);
        % Cauchy random numbers: tan(pi * (rand - 0.5))
        cauchy_noise = tan(pi * (rand(pop_size, dim) - 0.5));
        cauchy_noise = max(-5.0, min(5.0, cauchy_noise));
        fluctuations = fluctuation_amp * ((1.0 - tau)^2) .* repmat(mean_dists_all, 1, dim) .* cauchy_noise;

        % 7. Velocity Update
        r1 = rand(pop_size, dim);
        r2 = rand(pop_size, dim);
        r3 = rand(pop_size, dim);
        r4 = rand(pop_size, dim);

        V = w * V + ...
            c1 * r1 .* a_spin + ...
            c2 * r2 .* topo_grad + ...
            c3 * r3 .* a_elite + ...
            c4 * r4 .* (repmat(best_x, pop_size, 1) - X) + ...
            fluctuations;

        % Velocity clamping (20% of domain span)
        max_vel = 0.2 * domain_span;
        V = max(-repmat(max_vel, pop_size, 1), min(repmat(max_vel, pop_size, 1), V));

        % 8. Position Update
        X = X + V;

        % 9. Adaptive Stochastic Boundary Rebound
        lb_mat = repmat(lb, pop_size, 1);
        ub_mat = repmat(ub, pop_size, 1);

        lower_mask = X < lb_mat;
        upper_mask = X > ub_mat;

        if any(lower_mask(:))
            r_rebound = 0.5 * rand(pop_size, dim);
            X(lower_mask) = lb_mat(lower_mask) + r_rebound(lower_mask) .* (max(P(lower_mask), lb_mat(lower_mask)) - lb_mat(lower_mask) + 1e-4);
            V(lower_mask) = -0.5 * V(lower_mask);
        end

        if any(upper_mask(:))
            r_rebound = 0.5 * rand(pop_size, dim);
            X(upper_mask) = ub_mat(upper_mask) - r_rebound(upper_mask) .* (ub_mat(upper_mask) - min(P(upper_mask), ub_mat(upper_mask)) + 1e-4);
            V(upper_mask) = -0.5 * V(upper_mask);
        end

        % Clip to ensure hard domain feasibility
        X = max(lb_mat, min(ub_mat, X));

        % 10. Evaluation & Elite Updates
        for i = 1:pop_size
            if evals >= max_evals, break; end
            f_val = obj_func(X(i, :));
            evals = evals + 1;
            fitness(i) = f_val;

            if f_val < P_fit(i)
                P_fit(i) = f_val;
                P(i, :) = X(i, :);
                if f_val < best_f
                    best_f = f_val;
                    best_x = X(i, :);
                end
            end
        end

        convergence(end + 1) = best_f; %#ok<AGROW>
        centroid = mean(X, 1);
        div_t = mean(sqrt(sum((X - repmat(centroid, pop_size, 1)).^2, 2))) / (norm(domain_span) + 1e-12);
        diversity(end + 1) = div_t; %#ok<AGROW>
    end
end
