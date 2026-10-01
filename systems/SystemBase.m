classdef SystemBase
    %SystemBase Abstract base class for nonlinear systems simulated via ODE45.
    %   Models dx/dt = f(x,u) with a zero-order hold on the input. Subclasses
    %   define the constant properties stateDim and inputDim and implement
    %   the method dynamics(obj,x,u), which returns dx/dt.
    %
    %   Subclasses are responsible for their own constructors. Ts and
    %   odeopts are public properties, so a subclass can accept them as
    %   name-value arguments (e.g. via options.?SubclassName).
    %
    %   X = SIMULATE(S,X0,U) simulates from initial state X0 (stateDim-by-1)
    %   under inputs U (inputDim-by-T) and returns the trajectory X
    %   (stateDim-by-(T+1)), where column t+1 is the state after holding
    %   U(:,t) for Ts seconds.
    %
    %   SystemBase properties:
    %      stateDim   - Number of states (abstract, constant)
    %      inputDim   - Number of inputs (abstract, constant)
    %      Ts         - Simulation time step (default: 0.005)
    %      odeopts    - Options passed to ODE45
    %
    %   SystemBase methods:
    %      simulate   - Simulate the system over an input sequence
    %      dynamics   - State derivative f(x,u) (abstract)
    %
    %   See also ODE45, ODESET.

    properties (Abstract, Constant)
        stateDim (1,1) {mustBeInteger, mustBePositive};
        inputDim (1,1) {mustBeInteger, mustBePositive};
    end

    properties
        Ts (1,1) double {mustBeNumeric,mustBePositive} = 0.005;
        odeopts (1,1) struct = odeset('RelTol',1e-8,'AbsTol',1e-10)
    end

    methods (Abstract)
        dx = dynamics(obj, x, u)
    end

    methods
        function X = simulate(obj, x0, u)
            arguments
                obj (1,1) SystemBase;
                x0 (:,1) double {mustBeNumeric, SystemBase.mustMatchStateDim(obj, x0)};
                u (:,:) double {mustBeNumeric, SystemBase.mustMatchInputDim(obj, u)};
            end
            T = size(u, 2);
            X = zeros(obj.stateDim, T+1);
            X(:, 1) = x0(:);
            for t = 1:T
                [~, xSim] = ode45(@(~, x) obj.dynamics(x, u(:, t)), [0, obj.Ts], X(:, t), obj.odeopts);
                X(:, t+1) = xSim(end, :)';
            end
        end
    end

    methods (Static, Hidden, Access=protected)
        function mustMatchStateDim(obj, val)
            if size(val, 1) ~= obj.stateDim
                error( ...
                    "SystemBase:DimensionMismatch", ...
                    "Value must have %d row(s).", ...
                    obj.stateDim ...
                );
            end
        end

        function mustMatchInputDim(obj, val)
            if size(val, 1) ~= obj.inputDim
                error( ...
                    "SystemBase:DimensionMismatch", ...
                    "Value must have %d row(s).", ...
                    obj.inputDim ...
                );
            end
        end
    end
end