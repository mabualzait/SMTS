% RUN_BENCHMARKS - Comparative benchmark script (SMTS vs Standard PSO) in MATLAB
fprintf('========================================================================\n');
fprintf('Benchmarking SMTS vs Standard PSO on Multi-Dimensional Landscapes\n');
fprintf('========================================================================\n\n');

funcs = benchmark_functions();
num_funcs = length(funcs);
dim = 30;
pop_size = 50;
max_evals = 30000;
k_neighbors = 7;
runs = 10;

results_smts = zeros(num_funcs, runs);
results_pso = zeros(num_funcs, runs);

for f_idx = 1:num_funcs
    f_info = funcs{f_idx};
    fprintf('Testing %-12s (Dim=%d, MaxEvals=%d)...\n', f_info.name, dim, max_evals);

    for r = 1:runs
        % Run SMTS
        [~, b_f_smts, ~, ~] = SMTS(f_info.func, dim, f_info.bounds, pop_size, max_evals, k_neighbors);
        results_smts(f_idx, r) = b_f_smts;

        % Standard PSO Implementation for comparison
        lb = f_info.bounds(1) * ones(1, dim);
        ub = f_info.bounds(2) * ones(1, dim);
        span = ub - lb;
        X_pso = repmat(lb, pop_size, 1) + rand(pop_size, dim) .* repmat(span, pop_size, 1);
        V_pso = -0.1 * repmat(span, pop_size, 1) + 0.2 * rand(pop_size, dim) .* repmat(span, pop_size, 1);
        fit_pso = zeros(pop_size, 1);
        for i = 1:pop_size
            fit_pso(i) = f_info.func(X_pso(i, :));
        end
        P_pso = X_pso;
        P_fit_pso = fit_pso;
        [gbest_f_pso, g_idx] = min(P_fit_pso);
        gbest_x_pso = P_pso(g_idx, :);

        evals_pso = pop_size;
        while evals_pso < max_evals
            tau = evals_pso / max_evals;
            w = 0.9 - 0.5 * tau;
            c1 = 2.0; c2 = 2.0;

            r1 = rand(pop_size, dim);
            r2 = rand(pop_size, dim);
            V_pso = w * V_pso + c1 * r1 .* (P_pso - X_pso) + c2 * r2 .* (repmat(gbest_x_pso, pop_size, 1) - X_pso);
            V_pso = max(-0.2 * repmat(span, pop_size, 1), min(0.2 * repmat(span, pop_size, 1), V_pso));
            X_pso = max(repmat(lb, pop_size, 1), min(repmat(ub, pop_size, 1), X_pso + V_pso));

            for i = 1:pop_size
                if evals_pso >= max_evals, break; end
                f_val = f_info.func(X_pso(i, :));
                evals_pso = evals_pso + 1;
                if f_val < P_fit_pso(i)
                    P_fit_pso(i) = f_val;
                    P_pso(i, :) = X_pso(i, :);
                    if f_val < gbest_f_pso
                        gbest_f_pso = f_val;
                        gbest_x_pso = X_pso(i, :);
                    end
                end
            end
        end
        results_pso(f_idx, r) = gbest_f_pso;
    end
end

% Summary Table
fprintf('\n========================================================================\n');
fprintf('%-15s | %-22s | %-22s\n', 'Function', 'SMTS Mean ± Std', 'PSO Mean ± Std');
fprintf('------------------------------------------------------------------------\n');
for f_idx = 1:num_funcs
    m_smts = mean(results_smts(f_idx, :));
    s_smts = std(results_smts(f_idx, :));
    m_pso = mean(results_pso(f_idx, :));
    s_pso = std(results_pso(f_idx, :));
    fprintf('%-15s | %10.4e ± %9.4e | %10.4e ± %9.4e\n', ...
        funcs{f_idx}.name, m_smts, s_smts, m_pso, s_pso);
end
fprintf('========================================================================\n');
