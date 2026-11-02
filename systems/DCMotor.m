classdef DCMotor < SystemBase
    %DCMOTOR Nonlinear DC motor model simulated via ODE45.
    %   M = DCMOTOR() creates a DCMotor object with default parameters
    %   and simulation settings.
    %
    %   M = DCMOTOR(Name=Value) overrides one or more of the following
    %   properties at construction:
    %
    %      Ra         - Armature resistance (default: 12.345)
    %      La         - Armature inductance (default: 0.314)
    %      km         - Motor constant (default: 0.253)
    %      ua         - Armature voltage (default: 60)
    %      B          - Viscous friction coefficient (default: 0.00732)
    %      tau_l      - Load torque (default: 1.47)
    %      J          - Moment of inertia (default: 0.00441)
    %      inputType  - Input nonlinearity, "non-monotone" (default) or
    %                   "saturation"
    %      Ts         - Simulation time step (default: 0.005)
    %      odeopts    - Options passed to ODE45
    %
    %   The motor parameters and inputType can only be set at
    %   construction. Ts and odeopts are inherited from SystemBase.
    %
    %   X = SIMULATE(M,X0,U) simulates the motor from initial state X0
    %   (2-by-1) under input sequence U (1-by-T) and returns the state
    %   trajectory X (2-by-(T+1)).
    %
    %   DCMotor methods:
    %      simulate   - Simulate the motor over an input sequence
    %
    %   See also SystemBase, ODE45, ODESET.

    properties (Constant)
        stateDim = 2;
        inputDim = 1;
    end

    properties (GetAccess=public, SetAccess=immutable)
        Ra (1,1) {mustBeNumeric, mustBePositive} = 12.345;  % [ohm] resistance
        La (1,1) {mustBeNumeric, mustBePositive} = 0.314;   % [H] inductance
        km (1,1) {mustBeNumeric, mustBePositive} = 0.253;   % [Nm/A] motor constant
        ua (1,1) {mustBeNumeric, mustBePositive} = 60;      % [V] armature voltage
        B  (1,1) {mustBeNumeric, mustBePositive} = 0.00732; % [Nm/rad/s] viscous friction coefficient
        tau_l (1,1) {mustBeNumeric} = 1.47;                 % [Nm] load torque
        J  (1,1) {mustBeNumeric, mustBePositive} = 0.00441; % [kg*m^2] moment of inertia
        inputType (1,1) string {mustBeMember(inputType,["saturation","non-monotone"])} = "non-monotone";
    end

    methods
        function obj = DCMotor(options)
            % Explicit name-value arguments; those not given keep their defaults.
            arguments
                options.Ra (1,1) {mustBeNumeric, mustBePositive};
                options.La (1,1) {mustBeNumeric, mustBePositive};
                options.km (1,1) {mustBeNumeric, mustBePositive};
                options.ua (1,1) {mustBeNumeric, mustBePositive};
                options.B (1,1) {mustBeNumeric, mustBePositive};
                options.tau_l (1,1) {mustBeNumeric};
                options.J (1,1) {mustBeNumeric, mustBePositive};
                options.inputType (1,1) string;
                options.Ts (1,1) {mustBePositive};
                options.odeopts (1,1) struct;
                options.processNoiseCov {mustBeNumeric};
                options.measurementNoiseCov {mustBeNumeric};
            end
            for f = string(fieldnames(options))'
                obj.(f) = options.(f);
            end
        end

        function dx = dynamics(obj, x, u, p)
            if nargin < 4
                p = obj;
            end
            fu = obj.inputNonlinearity(u);
            dx = [
                -(p.Ra / p.La) .* x(1,:) - (p.km / p.La) .* x(2,:) .* fu + p.ua / p.La;
                -(p.B  / p.J ) .* x(2,:) + (p.km / p.J ) .* x(1,:) .* fu - p.tau_l / p.J
            ];
        end
    end

    methods (Hidden, Access=private)
        function u_out = inputNonlinearity(obj, u)
            switch obj.inputType
                case "saturation"
                    u_out = 2 * tanh(u);
                case "non-monotone"
                    u_out = 2 * tanh(u .* cos(u));
            end
        end
    end
end