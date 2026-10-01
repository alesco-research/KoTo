classdef TCLab < SystemBase
    %TCLAB Nonlinear two-heater temperature control lab model (ODE45).
    %   M = TCLAB(Name=Value) creates a TCLab object. Optional properties,
    %   settable only at construction:
    %
    %      Ta         - Ambient temperature [K] (default: 296.15)
    %      U          - Heat transfer coeff. to ambient [W/m^2-K] (default: 5)
    %      Us         - Heat transfer coeff. between heaters [W/m^2-K] (default: 24)
    %      m          - Mass [kg] (default: 4E-3)
    %      Cp         - Heat capacity [J/kg-K] (default: 500)
    %      A          - Surface area not between heaters [m^2] (default: 1E-3)
    %      As         - Surface area between heaters [m^2] (default: 2E-4)
    %      Q1max      - Max power of heater 1 [W] (default: 1)
    %      Q2max      - Max power of heater 2 [W] (default: 0.75)
    %      emissivity - Emissivity [-] (default: 0.9)
    %      Ts         - Time step [s] (default: 0.5)
    %      odeopts    - Options passed to ODE45
    %
    %   State x = [T1; T2] in K. Input u = [u1; u2] are heater duty
    %   cycles, clipped to [0,1].
    %
    %   X = SIMULATE(M,X0,U) returns the 2-by-(T+1) state trajectory for
    %   initial state X0 (2-by-1) and inputs U (2-by-T).
    %
    %   DX = DYNAMICS(M,X,U) returns dx/dt in K/s. X, U and DX are 2-by-N.
    %
    %   See also SystemBase, ODE45.

    properties (Constant)
        stateDim = 2;
        inputDim = 2;
        sigma = 5.67E-8;    % [W/m^2-K^4] Stefan-Boltzman constant
    end

    properties (GetAccess=public, SetAccess=immutable)
        Ta (1,1) {mustBeNumeric, mustBePositive} = 296.15;  % [K] ambient temperature
        U  (1,1) {mustBeNumeric, mustBePositive} = 5;       % [W/m^2-K] heat transfer coefficient
        Us (1,1) {mustBeNumeric, mustBePositive} = 24.0;    % [W/m^2-K] heat transfer coefficient between heaters
        m  (1,1) {mustBeNumeric, mustBePositive} = 4E-3;    % [kg] mass
        Cp (1,1) {mustBeNumeric, mustBePositive} = 500;     % [J/kg-K] heat capacity
        A  (1,1) {mustBeNumeric, mustBePositive} = 1E-3;    % [m^2] area not between heaters
        As (1,1) {mustBeNumeric, mustBePositive} = 2E-4;    % [m^2] surface area between heaters
        Q1max (1,1) {mustBeNumeric, mustBePositive} = 1;    % [W] max power heater 1
        Q2max (1,1) {mustBeNumeric, mustBePositive} = 0.75; % [W] max power heater 2
        emissivity (1,1) {mustBeNumeric, mustBePositive} = 0.9;
    end

    methods
        function obj = TCLab(options)
            arguments
                options.?TCLab;
                options.Ts {mustBePositive} = 0.5;
            end
            for f = string(fieldnames(options))'
                obj.(f) = options.(f);
            end
        end

        function dx = dynamics(obj, x, u)
            % Heat transfer between heaters
            convection = obj.Us * obj.As * diff(x, 1, 1);
            radiation  = obj.emissivity * obj.sigma * obj.As * diff(x.^4, 1, 1);

            % Energy balance
            dx = [ ...
                obj.U * obj.A * (obj.Ta - x(1,:)) + ...
                obj.emissivity * obj.sigma * obj.A * (obj.Ta^4 - x(1,:).^4) + ...
                convection + radiation + obj.Q1max * clip(u(1,:), 0, 1);
                obj.U * obj.A * (obj.Ta - x(2,:)) + ...
                obj.emissivity * obj.sigma * obj.A * (obj.Ta^4 - x(2,:).^4) - ...
                convection - radiation + obj.Q2max * clip(u(2,:), 0, 1) ...
            ] / (obj.m * obj.Cp);
        end
    end
end