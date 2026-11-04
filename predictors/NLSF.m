classdef NLSF
%NLSF Nonlinear least-squares fit of physical model parameters.
%   Fits bounded parameters of a SystemBase model by minimizing the
%   multi-step prediction error with LSQNONLIN. If the model has hidden
%   states (outputDim < stateDim), the initial state of every trajectory
%   is estimated jointly with the parameters, starting from a guess
%   derived from the first measurement.
%
%   obj = NLSF(system, bounds, Y1, U1, Y2, U2, ..., horizon=Inf,
%              substeps=4, memory=2*ceil(nx/ny), initialState=[],
%              outputScaling=Scale(...),
%              inputScaling=Scale(...), optimOptions=optimoptions("lsqnonlin"))
%     system        SystemBase instance (required). Unbounded parameters
%                   stay fixed; noise is ignored.
%     bounds        Struct of [lower, upper] per fitted parameter, named
%                   like the property of system (required). The initial
%                   guess is the system's value clipped to the bounds.
%     Yi, Ui        ny-by-N_i measured outputs and nu-by-N_i inputs;
%                   Yi(:,k+1) follows holding Ui(:,k) for Ts. At least one
%                   pair. From cells: pairs = [Y(:)'; U(:)']; then
%                   NLSF(system, bounds, pairs{:}, ...)
%     horizon       Max. prediction steps per trajectory, rolled out once
%                   from its first sample (default Inf).
%     substeps      Fixed RK4 steps per Ts (default 4). Keeps the cost
%                   smooth for finite differences; increase if stiff.
%     memory        Past samples used by forecast to estimate the state
%                   (default 2*ceil(nx/ny), at least ceil(nx/ny)).
%     initialState  Function handle y -> x0 mapping measured outputs
%                   (physical units, ny-by-K) to an initial state guess
%                   (nx-by-K). Default: y itself if ny = nx, else y
%                   repeated nx/ny times (e.g. sensors tracking heaters).
%     outputScaling Scale from physical outputs to Yi (default: none).
%     inputScaling  Scale from physical inputs to Ui (default: none).
%     optimOptions  LSQNONLIN options (default: Display="off").
%
%   Parameters are optimized normalized to [0, 1]; initial states are
%   optimized in physical units. Simulation is in
%   physical coordinates; data, cost and forecasts are in the (possibly
%   scaled) data coordinates. The cost sums squared errors over all
%   trajectories, simulated in parallel.
%
%   Properties (read-only):
%     system               Fitted noise-free system with step Ts.
%     parameters           Struct of the estimated parameters.
%     outputScaling        Output scaling, or empty.
%     inputScaling         Input scaling, or empty.
%     inputDim, outputDim  Number of inputs and outputs.
%     memory               Past samples used by forecast.
%
%   Y = forecast(obj, uPast, yPast, uFuture)
%     Estimates the state at the start of the last `memory` samples by
%     least squares on yPast (initial guess from initialState, parameters
%     fixed), then simulates under the nu-by-T inputs uFuture and returns
%     the ny-by-T outputs, in data coordinates. uPast and yPast need at
%     least `memory` columns ending at the same step.
%
%   Requires the Optimization Toolbox.
%
%   See also SystemBase, LSQNONLIN, OPTIMOPTIONS.

    properties (SetAccess=immutable)
        system (1,1);
        parameters (1,1) struct;
        outputScaling Scale {mustBeScalarOrEmpty} = Scale.empty;
        inputScaling Scale {mustBeScalarOrEmpty} = Scale.empty;
        outputDim (1,1) {mustBeInteger, mustBePositive} = 1;
        inputDim (1,1) {mustBeInteger, mustBePositive} = 1;
        memory (1,1) {mustBeInteger, mustBePositive} = 1;
        substeps (1,1) {mustBeInteger, mustBePositive} = 4;
        initialState (1,1) function_handle = @(y) y;
    end

    methods
        function obj = NLSF(system, bounds, Y, U, options)
            arguments (Input)
                system (1,1) SystemBase;
                bounds (1,1) struct;
            end
            arguments (Input, Repeating)
                Y (:,:) {mustBeNumeric, NLSF.mustMatch(system, "outputDim", Y)}
                U (:,:) {mustBeNumeric, NLSF.mustMatch(system, "inputDim", U), NLSF.mustBeEqualSize(Y, U)}
            end
            arguments (Input)
                options.horizon (1,1) {mustBePositive} = Inf;
                options.substeps (1,1) {mustBeInteger, mustBePositive} = 4;
                options.memory {mustBeScalarOrEmpty, mustBeInteger, mustBePositive} = [];
                options.initialState {mustBeScalarOrEmpty} = [];
                options.outputScaling Scale {mustBeScalarOrEmpty} = Scale.empty;
                options.inputScaling Scale {mustBeScalarOrEmpty} = Scale.empty;
                options.optimOptions = optimoptions("lsqnonlin", Display="off");
            end
            if isempty(Y)
                error( ...
                    "NLSF:NoData", ...
                    "Provide at least one (Y, U) trajectory pair." ...
                );
            end
            [ny, nu] = deal(size(Y{1}, 1), size(U{1}, 1));
            [ys, us] = deal(options.outputScaling, options.inputScaling);

            % Full parameter struct of the system
            p = NLSF.systemProperties(system);
            [names, lb, ub] = NLSF.parseBounds(bounds, p);
            theta0 = arrayfun(@(n) p.(n), names);
            z0 = (clip(theta0, lb, ub) - lb) ./ (ub - lb);

            nx = system.stateDim;
            if isempty(options.memory)
                options.memory = 2 * ceil(nx / ny);
            end
            if options.memory * ny < nx
                error( ...
                    "NLSF:MemoryTooShort", ...
                    "memory must be at least %d to make the state observable from the outputs.", ...
                    ceil(nx / ny) ...
                );
            end
            init = options.initialState;
            if isempty(init)
                init = NLSF.defaultInitialState(system);
            end

            % Rollout data per trajectory, initial states from measurement
            data = NLSF.rolloutData(Y, U, options.horizon, ys, us, init);
            data.substeps = options.substeps;
            M = size(data.x0, 2);

            % Decision variables: normalized parameters, then all x0(:)
            nTheta = numel(names);
            zInit = [z0(:); data.x0(:)];
            zLb = [zeros(nTheta, 1); -inf(nx * M, 1)];
            zUb = [ones(nTheta, 1); inf(nx * M, 1)];
            optimOptions = optimoptions( ...
                options.optimOptions, ...
                JacobPattern=NLSF.jacobianPattern(data, nTheta, nx) ...
            );

            z = lsqnonlin( ...
                @(z) NLSF.residual(z, system, p, names, lb, ub, ys, data), ...
                zInit, zLb, zUb, optimOptions ...
            );
            theta = lb + z(1:nTheta)' .* (ub - lb);

            % Final noise-free system with the estimated parameters
            p = NLSF.merge(p, names, theta);
            p.Ts = system.Ts;
            p.processNoiseCov = [];
            p.measurementNoiseCov = [];
            args = namedargs2cell(p);
            obj.system = feval(class(system), args{:});

            obj.parameters = cell2struct(num2cell(theta(:)), cellstr(names), 1);
            obj.outputScaling = ys;
            obj.inputScaling = us;
            obj.outputDim = ny;
            obj.inputDim = nu;
            obj.memory = options.memory;
            obj.substeps = options.substeps;
            obj.initialState = init;
        end

        function Y = forecast(obj, uPast, yPast, uFuture)
            arguments
                obj (1,1) NLSF;
                uPast (:,:) {NLSF.mustMatch(obj, "inputDim", uPast)};
                yPast (:,:) {NLSF.mustMatch(obj, "outputDim", yPast)};
                uFuture (:,:) {NLSF.mustMatch(obj, "inputDim", uFuture)};
            end
            if size(yPast, 2) < obj.memory || size(uPast, 2) < obj.memory
                error( ...
                    "NLSF:InsufficientHistory", ...
                    "uPast and yPast need at least %d columns.", ...
                    obj.memory ...
                );
            end
            sys = obj.system;
            nu = obj.inputDim;
            m = obj.memory;
            yw = yPast(:, end-m+1:end);
            uw = NLSF.unscale(obj.inputScaling, uPast(:, end-m+1:end-1));

            % Estimate the state at the start of the window
            xGuess = obj.initialState(NLSF.unscale(obj.outputScaling, yw(:, 1)));
            res = @(x) NLSF.windowResidual(x, obj, uw, yw);
            x = lsqnonlin(res, xGuess, [], [], optimoptions("lsqnonlin", Display="off"));

            % Propagate to the end of the window, then forecast
            if m > 1
                X = NLSF.rollout(sys, sys, x, reshape(uw, nu, 1, []), obj.substeps);
                x = X(:, end);
            end
            ut = NLSF.unscale(obj.inputScaling, [uPast(:, end), uFuture(:, 1:end-1)]);
            X = NLSF.rollout(sys, sys, x, reshape(ut, nu, 1, []), obj.substeps);
            Y = NLSF.scale(obj.outputScaling, sys.output(X(:, :)));
        end
    end

    methods (Static, Hidden, Access=protected)
        function e = residual(z, system, p, names, lb, ub, ys, data)
            nTheta = numel(names);
            p = NLSF.merge(p, names, lb + z(1:nTheta)' .* (ub - lb));
            x0 = reshape(z(nTheta+1:end), size(data.x0));
            X = NLSF.rollout(system, p, x0, data.U, data.substeps);
            e0 = NLSF.scale(ys, system.output(x0)) - data.Y0;
            e = NLSF.scale(ys, system.output(X(:, :))) - data.Y;
            e = [e0(:); e(~isnan(data.Y))];
        end

        function e = windowResidual(x, obj, uw, yw)
            sys = obj.system;
            Xw = x;
            if size(uw, 2) > 0
                X = NLSF.rollout(sys, sys, x, reshape(uw, obj.inputDim, 1, []), obj.substeps);
                Xw = [x, X(:, :)];
            end
            e = NLSF.scale(obj.outputScaling, sys.output(Xw)) - yw;
            e = e(:);
        end

        function init = defaultInitialState(sys)
            [nx, ny] = deal(sys.stateDim, sys.outputDim);
            if mod(nx, ny) ~= 0
                error( ...
                    "NLSF:InitialStateRequired", ...
                    "stateDim is not a multiple of outputDim; provide the initialState option." ...
                );
            end
            init = @(y) repmat(y, nx / ny, 1);
        end

        function J = jacobianPattern(data, nTheta, nx)
            % Initial states of a trajectory only affect its own residuals
            [ny, M] = size(data.Y0);
            H = size(data.U, 3);
            valid = ~isnan(data.Y);
            trajOfColumn = repmat(mod(0:M*H-1, M) + 1, ny, 1);
            traj = [reshape(repmat(1:M, ny, 1), [], 1); trajOfColumn(valid)];
            nRows = numel(traj);
            cols = reshape(((traj(:) - 1) * nx + (1:nx))', [], 1);
            x0Part = sparse(repelem((1:nRows)', nx), cols, 1, nRows, nx * M);
            J = [sparse(ones(nRows, nTheta)), x0Part];
        end

        function data = rolloutData(Y, U, h, ys, us, init)
            N = cellfun(@(y) size(y, 2), Y(:)');
            if any(N < 2)
                error( ...
                    "NLSF:TrajectoryTooShort", ...
                    "Every trajectory needs at least 2 samples." ...
                );
            end
            [ny, nu] = deal(size(Y{1}, 1), size(U{1}, 1));
            steps = min(N - 1, h);
            H = max(steps);
            M = numel(Y);

            data.Y0 = zeros(ny, M);
            data.U = zeros(nu, M, H);
            Yw = nan(ny, M, H);
            for i = 1:M
                k = 1:steps(i);
                data.Y0(:, i) = Y{i}(:, 1);
                data.U(:, i, k) = reshape(NLSF.unscale(us, U{i}(:, k)), nu, 1, []);
                Yw(:, i, k) = reshape(Y{i}(:, k + 1), ny, 1, []);
            end
            data.Y = Yw(:, :);
            data.x0 = init(NLSF.unscale(ys, data.Y0));
        end

        function X = rollout(sys, p, X0, U, substeps)
            H = size(U, 3);
            X = nan([size(X0), H]);
            x = X0;
            dt = sys.Ts / substeps;
            f = @(x, u) sys.dynamics(x, u, p);
            for k = 1:H
                u = U(:, :, k);
                for s = 1:substeps
                    k1 = f(x, u);
                    k2 = f(x + dt / 2 * k1, u);
                    k3 = f(x + dt / 2 * k2, u);
                    k4 = f(x + dt * k3, u);
                    x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
                end
                X(:, :, k) = x;
            end
        end

        function p = systemProperties(sys)
            mc = metaclass(sys);
            props = mc.PropertyList(~[mc.PropertyList.Constant] & ~[mc.PropertyList.Dependent]);
            p = struct();
            for name = string({props.Name})
                p.(name) = sys.(name);
            end
        end

        function [names, lb, ub] = parseBounds(bounds, p)
            names = string(fieldnames(bounds))';
            if isempty(names)
                error( ...
                    "NLSF:InvalidBounds", ...
                    "Provide at least one parameter bound." ...
                );
            end
            unknown = names(~isfield(p, names));
            if ~isempty(unknown)
                error( ...
                    "NLSF:InvalidBounds", ...
                    "Unknown parameter(s): %s.", ...
                    strjoin(unknown, ", ") ...
                );
            end
            b = arrayfun(@(n) bounds.(n), names, UniformOutput=false);
            if ~all(cellfun(@(v) isnumeric(v) && isreal(v) && numel(v) == 2 && v(1) < v(2), b))
                error( ...
                    "NLSF:InvalidBounds", ...
                    "Each bound must be a real [lower, upper] pair " + ...
                    "with lower < upper." ...
                );
            end
            b = vertcat(b{:});
            [lb, ub] = deal(b(:, 1)', b(:, 2)');
        end

        function p = merge(p, names, theta)
            for i = 1:numel(names)
                p.(names(i)) = theta(i);
            end
        end

        function x = scale(s, x)
            if ~isempty(s)
                x = s.scale(x);
            end
        end

        function x = unscale(s, x)
            if ~isempty(s)
                x = s.unscale(x);
            end
        end

        function mustBeEqualSize(Y, U)
            if size(Y, 2) ~= size(U, 2)
                error( ...
                    "NLSF:DimensionMismatch", ...
                    "Y and U must have trajectories of equal length." ...
                );
            end
        end

        function mustMatch(obj, dim, val)
            if size(val, 1) ~= obj.(dim)
                error( ...
                    "NLSF:DimensionMismatch", ...
                    "Value must have %d row(s) to match the " + ...
                    "%s dimension.", ...
                    obj.(dim), erase(dim, "Dim") ...
                );
            end
        end
    end
end
