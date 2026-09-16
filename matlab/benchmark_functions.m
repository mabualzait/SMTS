function [funcs] = benchmark_functions()
% BENCHMARK_FUNCTIONS Returns a cell array of standard optimization test functions.
% Each entry is a struct with fields:
%   name    - String name
%   func    - Function handle f = func(x) for row vector x (1xD)
%   bounds  - [lb, ub] scalar bounds
%   opt_f   - Known theoretical global minimum value

    funcs = cell(5, 1);

    % 1. Sphere Function (Unimodal)
    funcs{1}.name = 'Sphere';
    funcs{1}.bounds = [-100, 100];
    funcs{1}.opt_f = 0.0;
    funcs{1}.func = @(x) sum(x.^2);

    % 2. Rosenbrock Function (Multimodal / Valley)
    funcs{2}.name = 'Rosenbrock';
    funcs{2}.bounds = [-30, 30];
    funcs{2}.opt_f = 0.0;
    funcs{2}.func = @(x) sum(100 * (x(2:end) - x(1:end-1).^2).^2 + (x(1:end-1) - 1).^2);

    % 3. Rastrigin Function (Highly Multimodal)
    funcs{3}.name = 'Rastrigin';
    funcs{3}.bounds = [-5.12, 5.12];
    funcs{3}.opt_f = 0.0;
    funcs{3}.func = @(x) 10 * length(x) + sum(x.^2 - 10 * cos(2 * pi * x));

    % 4. Ackley Function (Multimodal)
    funcs{4}.name = 'Ackley';
    funcs{4}.bounds = [-32, 32];
    funcs{4}.opt_f = 0.0;
    funcs{4}.func = @(x) -20 * exp(-0.2 * sqrt(mean(x.^2))) - exp(mean(cos(2 * pi * x))) + 20 + exp(1);

    % 5. Griewank Function (Multimodal)
    funcs{5}.name = 'Griewank';
    funcs{5}.bounds = [-600, 600];
    funcs{5}.opt_f = 0.0;
    funcs{5}.func = @(x) sum(x.^2)/4000 - prod(cos(x ./ sqrt(1:length(x)))) + 1;
end
