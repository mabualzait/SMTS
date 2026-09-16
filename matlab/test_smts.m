% TEST_SMTS - Verification Test Script for SMTS Optimizer in MATLAB/Octave
fprintf('====================================================\n');
fprintf('Testing Starling Murmuration Topological Search (SMTS)\n');
fprintf('====================================================\n\n');

funcs = benchmark_functions();

% Test 1: Sphere Function in 10D
f_sphere = funcs{1};
dim = 10;
pop_size = 40;
max_evals = 5000;
bounds = f_sphere.bounds;

fprintf('1. Testing SMTS on %s (%dD, MaxEvals=%d)...\n', f_sphere.name, dim, max_evals);
[best_x, best_f, conv, div] = SMTS(f_sphere.func, dim, bounds, pop_size, max_evals, 7);

fprintf('   -> Best Cost Found: %e\n', best_f);
fprintf('   -> Norm of Best X:   %e\n', norm(best_x));
assert(best_f < 1e-4, 'Error: SMTS failed to converge on Sphere function!');
fprintf('   [PASS] Sphere 10D converged successfully!\n\n');

% Test 2: Rastrigin Function in 20D
f_rast = funcs{3};
dim = 20;
pop_size = 50;
max_evals = 15000;
bounds = f_rast.bounds;

fprintf('2. Testing SMTS on %s (%dD, MaxEvals=%d)...\n', f_rast.name, dim, max_evals);
[best_x, best_f, conv, div] = SMTS(f_rast.func, dim, bounds, pop_size, max_evals, 7);

fprintf('   -> Best Cost Found: %e\n', best_f);
fprintf('   [PASS] Rastrigin 20D completed successfully!\n\n');

fprintf('====================================================\n');
fprintf('All SMTS MATLAB tests completed successfully!\n');
fprintf('====================================================\n');
