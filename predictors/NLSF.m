classdef NLSF
%NLSF Nonlinear least-squares fit of physical model parameters.
%   Estimates bounded parameters of a SystemBase model by minimizing the
%   multi-step prediction error on input-output data with LSQNONLIN and
%   provides warm-started forecasting with the fitted model.
%
%   obj = NLSF(system, bounds, Y1, U1, Y2, U2, ..., horizon=Inf,
%              substeps=4, outputScaling=Scale(...),
%              inputScaling=Scale(...), optimOptions=optimoptions("lsqnonlin"))
%     system        SystemBase instance to be fitted (required). Parameters
%                   not listed in bounds are fixed to its values. Noise is
%                   ignored.
%     bounds        Struct with one [lower, upper] field per fitted
%                   parameter, named like the parameter property of system
%                   (required). The initial guess is the value in system,
%                   clipped to the bounds.
%     Yi            ny-by-N_i output trajectory. The full state is assumed
%                   to be measured, i.e. ny = system.stateDim.
%     Ui            nu-by-N_i input trajectory matching Yi. Yi(:,k+1) is
%                   the output after holding Ui(:,k) for Ts seconds. Pairs
%                   (Yi, Ui) are passed as repeated arguments; at least one
%                   is required. From cell arrays: pairs = [Y(:)'; U(:)'];
%                   NLSF(system, bounds, pairs{:}, ...)
%     horizon       Maximum number of prediction steps per trajectory
%                   (default Inf). Each trajectory is rolled out once from
%                   its first sample over min(N_i - 1, horizon) steps.
%     substeps      Number of fixed RK4 steps per sampling interval Ts
%                   (default 4). A fixed step makes the cost smooth in the
%                   parameters, which keeps finite-difference gradients
%                   accurate. Increase it for stiff or fast dynamics.
%     outputScaling Scale mapping physical outputs (states) to the
%                   coordinates of Yi (default: none).
%     inputScaling  Scale mapping physical inputs to the coordinates of Ui
%                   (default: none).
%     optimOptions  Options passed to LSQNONLIN (default: Display="off").
%
%   The parameters are optimized after scaling them to [0, 1] with the
%   bounds. The system is simulated in physical coordinates. If scalings
%   are given, data, cost and forecasts are in scaled coordinates;
%   otherwise everything is left unscaled. The cost is the sum of squared
%   prediction errors over all trajectories, which are simulated in
%   parallel.
%
%   Properties (read-only):
%     system                    Instance of the class of the passed system
%                               with the estimated parameters, step Ts and
%                               no process or measurement noise.
%     parameters                Struct of the estimated parameters.
%     outputScaling             Output scaling, or empty if none.
%     inputScaling              Input scaling, or empty if none.
%     inputDim, outputDim       Number of inputs and outputs.
%     memory                    Past samples used by forecast (always 1).
%
%   Y = forecast(obj, uPast, yPast, uFuture)
%     Warm-started multi-step output prediction.
%       uPast, yPast  Past samples (columns), at least `memory` each, ending
%                     at the same time step. Only the last is used.
%       uFuture       nu-by-T future inputs.
%     Arguments and outputs are in the coordinates of the training data.
%     Calls simulate of the fitted system from yPast(:,end). Returns an
%     ny-by-T matrix of predicted outputs for uFuture.
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
            X = NLSF.rollout(system, p, data.x0, data.U, data.steps, data.substeps);
            e = NLSF.scale(ys, X(:, data.active)) - data.Y;
            e = e(:);
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
            data.steps = steps;
            data.active = (1:H) <= steps(:);
            data.Y = Yw(:, data.active);
        end

        function X = rollout(sys, p, X0, U, steps, substeps)
            H = size(U, 3);
            X = nan([size(X0), H]);
            x = X0;
            dt = sys.Ts / substeps;
            f = @(x, u) sys.dynamics(x, u, p);
            for k = 1:H
                active = k <= steps(:)';
                u = U(:, active, k);
                xa = x(:, active);
                for s = 1:substeps
                    k1 = f(xa, u);
                    k2 = f(xa + dt / 2 * k1, u);
                    k3 = f(xa + dt / 2 * k2, u);
                    k4 = f(xa + dt * k3, u);
                    xa = xa + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
                end
                x(:, active) = xa;
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
