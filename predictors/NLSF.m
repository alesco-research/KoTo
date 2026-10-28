classdef NLSF
%NLSF Nonlinear least-squares fit of physical model parameters.
%   Fits bounded parameters of a SystemBase model by minimizing the
%   multi-step prediction error with LSQNONLIN.
%
%   obj = NLSF(system, bounds, Y1, U1, Y2, U2, ..., horizon=Inf,
%              substeps=4, outputScaling=Scale(...),
%              inputScaling=Scale(...), optimOptions=optimoptions("lsqnonlin"))
%     system        SystemBase instance (required). Unbounded parameters
%                   stay fixed; noise is ignored.
%     bounds        Struct of [lower, upper] per fitted parameter, named
%                   like the property of system (required). The initial
%                   guess is the system's value clipped to the bounds.
%     Yi, Ui        ny-by-N_i full-state outputs and nu-by-N_i inputs;
%                   Yi(:,k+1) follows holding Ui(:,k) for Ts. At least one
%                   pair. From cells: pairs = [Y(:)'; U(:)']; then
%                   NLSF(system, bounds, pairs{:}, ...)
%     horizon       Max. prediction steps per trajectory, rolled out once
%                   from its first sample (default Inf).
%     substeps      Fixed RK4 steps per Ts (default 4). Keeps the cost
%                   smooth for finite differences; increase if stiff.
%     outputScaling Scale from physical states to Yi (default: none).
%     inputScaling  Scale from physical inputs to Ui (default: none).
%     optimOptions  LSQNONLIN options (default: Display="off").
%
%   Parameters are optimized normalized to [0, 1]. Simulation is in
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
%     memory               Past samples used by forecast (always 1).
%
%   Y = forecast(obj, uPast, yPast, uFuture)
%     Simulates the fitted system from yPast(:,end) under the nu-by-T
%     inputs uFuture and returns the ny-by-T outputs, in data coordinates.
%     uPast and yPast need at least `memory` columns ending at the same
%     step.
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
    end

    methods
        function obj = NLSF(system, bounds, Y, U, options)
            arguments (Input)
                system (1,1) SystemBase;
                bounds (1,1) struct;
            end
            arguments (Input, Repeating)
                Y (:,:) {mustBeNumeric, NLSF.mustMatch(system, "stateDim", Y)}
                U (:,:) {mustBeNumeric, NLSF.mustMatch(system, "inputDim", U), NLSF.mustBeEqualSize(Y, U)}
            end
            arguments (Input)
                options.horizon (1,1) {mustBePositive} = Inf;
                options.substeps (1,1) {mustBeInteger, mustBePositive} = 4;
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

            % Rollout data per trajectory
            data = NLSF.rolloutData(Y, U, options.horizon, ys, us);
            data.substeps = options.substeps;

            z = lsqnonlin( ...
                @(z) NLSF.residual(z, system, p, names, lb, ub, ys, data), ...
                z0, zeros(size(z0)), ones(size(z0)), options.optimOptions ...
            );
            theta = lb + z .* (ub - lb);

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
        end

        function Y = forecast(obj, uPast, yPast, uFuture)
            arguments
                obj (1,1) NLSF;
                uPast (:,:) {NLSF.mustMatch(obj, "inputDim", uPast)};
                yPast (:,:) {NLSF.mustMatch(obj, "outputDim", yPast)};
                uFuture (:,:) {NLSF.mustMatch(obj, "inputDim", uFuture)};
            end
            x0 = NLSF.unscale(obj.outputScaling, yPast(:, end));
            ut = NLSF.unscale(obj.inputScaling, [uPast(:, end), uFuture(:, 1:end-1)]);
            X = obj.system.simulate(x0, ut);
            Y = NLSF.scale(obj.outputScaling, X(:, 2:end));
        end
    end

    methods (Static, Hidden, Access=protected)
        function e = residual(z, system, p, names, lb, ub, ys, data)
            p = NLSF.merge(p, names, lb + z .* (ub - lb));
            X = NLSF.rollout(system, p, data.x0, data.U, data.substeps);
            e = NLSF.scale(ys, X(:, :)) - data.Y;
            e = e(~isnan(data.Y));
        end

        function data = rolloutData(Y, U, h, ys, us)
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

            data.x0 = zeros(ny, M);
            data.U = zeros(nu, M, H);
            Yw = nan(ny, M, H);
            for i = 1:M
                k = 1:steps(i);
                data.x0(:, i) = NLSF.unscale(ys, Y{i}(:, 1));
                data.U(:, i, k) = reshape(NLSF.unscale(us, U{i}(:, k)), nu, 1, []);
                Yw(:, i, k) = reshape(Y{i}(:, k + 1), ny, 1, []);
            end
            data.Y = Yw(:, :);
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
