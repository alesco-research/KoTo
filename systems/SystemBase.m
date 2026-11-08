classdef SystemBase
    %SYSTEMBASE Abstract base class for nonlinear systems simulated via ODE45.
    %   Models dx/dt = f(x,u) with a zero-order hold on the input and
    %   optional additive Gaussian process and measurement noise. Subclasses
    %   define the constant properties stateDim and outputDim, the property
    %   inputDim (constant, or dependent on the configuration) and
    %   implement dynamics(obj,x,u,params), which returns dx/dt.
    %
    %   The optional struct PARAMS overrides the model parameters stored in
    %   the object without creating a new instance (e.g. for parameter
    %   estimation). Its fields are named like the parameter properties and
    %   must contain every parameter used by the dynamics. If omitted, the
    %   object's own parameters are used.
    %
    %   Subclasses are responsible for their own constructors. The public
    %   properties (Ts, odeopts, noise covariances) can be accepted as
    %   name-value arguments (e.g. via options.?SubclassName).
    %
    %   X = SIMULATE(S,X0,U) simulates from initial state X0 (stateDim-by-1)
    %   under inputs U (inputDim-by-T) and returns the true state trajectory
    %   X (stateDim-by-(T+1)), where column t+1 is the state after holding
    %   U(:,t) for Ts seconds. If processNoiseCov is set, noise is added to
    %   the state after each step and propagates through the dynamics. X0
    %   is treated as exact.
    %
    %   [X,Y] = SIMULATE(S,X0,U) also returns the measured trajectory Y
    %   (same size as X). Y equals X plus measurement noise in columns 2 to
    %   T+1; column 1 is the exact initial state. Measurement noise does not
    %   propagate into X. If measurementNoiseCov is empty, Y equals X.
    %
    %   Noise is drawn from MATLAB's global random stream; call RNG
    %   beforehand for reproducible results.
    %
    %   SystemBase properties:
    %      stateDim            - Number of states (abstract, constant)
    %      inputDim            - Number of inputs (abstract, constant or dependent)
    %      Ts                  - Simulation time step (default: 0.005)
    %      odeopts             - Options passed to ODE45
    %      processNoiseCov     - Process noise covariance per step (default: [], none)
    %      measurementNoiseCov - Measurement noise covariance (default: [], none)
    %
    %   Both covariances must be stateDim-by-stateDim, real, finite,
    %   symmetric and positive semidefinite (or empty for no noise).
    %   Off-diagonal entries produce correlated noise across states.
    %
    %   SystemBase methods:
    %      simulate   - Simulate the system over an input sequence
    %      dynamics   - State derivative f(x,u,params) (abstract)
    %
    %   See also ODE45, ODESET, RNG.

    properties (Abstract, Constant)
        stateDim (1,1) {mustBeInteger, mustBePositive};
        outputDim (1,1) {mustBeInteger, mustBePositive};
    end

    properties (Abstract)
        inputDim (1,1) {mustBeInteger, mustBePositive};
    end

    properties
        Ts (1,1) double {mustBeNumeric,mustBePositive} = 0.005;
        odeopts (1,1) struct = odeset('RelTol',1e-8,'AbsTol',1e-10)
        processNoiseCov {mustBeNumeric, mustBeMatrix} = []
        measurementNoiseCov {mustBeNumeric, mustBeMatrix} = []
    end

    methods (Abstract)
        dx = dynamics(obj, x, u, params)
    end

    methods
        function [X, Y] = simulate(obj, x0, u)
            arguments
                obj (1,1) SystemBase;
                x0 (:,1) double {mustBeNumeric, SystemBase.mustMatch(obj, "stateDim", x0)};
                u (:,:) double {mustBeNumeric, SystemBase.mustMatch(obj, "inputDim", u)};
            end
            T = size(u, 2);
            X = zeros(obj.stateDim, T+1);
            X(:, 1) = x0(:);

            Lq = SystemBase.covFactor(obj.processNoiseCov);
            Lr = SystemBase.covFactor(obj.measurementNoiseCov);

            for t = 1:T
                [~, xSim] = ode45(@(~, x) obj.dynamics(x, u(:, t)), [0, obj.Ts], X(:, t), obj.odeopts);
                X(:, t+1) = xSim(end, :)';
                if ~isempty(Lq)
                    X(:, t+1) = X(:, t+1) + Lq * randn(obj.stateDim, 1);
                end
            end
            if nargout > 1
                Y = obj.output(X);
                if ~isempty(Lr)
                    Y(:, 2:end) = Y(:, 2:end) + Lr * randn(obj.outputDim, T);
                end
            end
        end

        function Y = output(obj, X)
            arguments
                obj (1,1) SystemBase; %#ok<INUSA>
                X (:,:) {mustBeNumeric};
            end
            Y = X;
        end

        function obj = set.processNoiseCov(obj, val)
            SystemBase.validateCovariance(obj, val, "processNoiseCov");
            obj.processNoiseCov = val;
        end

        function obj = set.measurementNoiseCov(obj, val)
            SystemBase.validateCovariance(obj, val, "measurementNoiseCov");
            obj.measurementNoiseCov = val;
        end
    end

    methods (Static, Hidden, Access=protected)
        function mustMatch(obj, dim, val)
            if size(val, 1) ~= obj.(dim)
                error( ...
                    "SystemBase:DimensionMismatch", ...
                    "Value must have %d row(s) to match the " + ...
                    "%s dimension.", ...
                    obj.(dim), erase(dim, "Dim") ...
                );
            end
        end

        function validateCovariance(obj, cov, name)
            if isempty(cov); return; end

            dim = obj.stateDim;
            if name == "measurementNoiseCov"
                dim = obj.outputDim;
            end
            
            msg = "";
            tol = 1e-10 * norm(cov, "fro");
            if ~isreal(cov) || ~all(isfinite(cov), "all")
                msg = "real and finite";
            elseif ~isequal(size(cov), dim * [1,1])
                msg = sprintf("%d-by-%d" , dim * [1,1]);
            elseif norm(cov - cov.', "fro") > tol
                msg = "symmetric";
            elseif min(eig((cov + cov')/2, "vector")) < -tol
                msg = "positive semidefinite";
            end
            if msg ~= ""
                error( ...
                    "SystemBase:InvalidCovariance", ...
                    "%s must be %s.", ...
                    name, msg ...
                );
            end
        end

        function L = covFactor(cov)
            L = [];
            if ~isempty(cov)
                cov = (cov + cov')/2;
                [V, D] = eig(cov);
                L = V * diag(sqrt(max(diag(D), 0)));
            end
        end
    end
end