classdef TCLab < SystemBase
    %TCLAB Simulated TC-Lab with optional cooling fan and sensor dynamics (ODE45).
    %   Two heaters, driven by u1 and u2, exchange heat with each other
    %   (convection and radiation through the shared surface As) and with
    %   the ambient. Each temperature sensor is a first-order lag of its
    %   heater temperature (time constant tauSensor), giving 4 states.
    %
    %   With fanSwitch="on", input u3 drives a cooling fan that increases
    %   the heat transfer coefficient to ambient linearly with fan speed,
    %
    %       U(u3) = U + dU * u3 / 100,   u3 in % (clipped to [0, 100]),
    %
    %   so the fan input multiplies the temperature difference to ambient:
    %   the dynamics contain the product of state and input (bilinear
    %   coupling), whereas the heater inputs enter additively. With
    %   fanSwitch="off", there is no fan input and U is constant.
    %
    %   M = TCLAB(Name=Value) sets the properties below at construction.
    %
    %   State  x = [T1; T2; S1; S2] in °C (heater and sensor temperatures).
    %   Output y = x(3:4) (measured sensor temperatures), see OUTPUT.
    %   Input  u = [u1; u2; u3] if fanSwitch="on", else u = [u1; u2]:
    %   heater powers u1, u2 in % and fan speed u3 in %, each clipped to
    %   [0, 100]. inputDim (3 or 2) follows fanSwitch.
    %
    %   Properties (defaults):
    %      Ta 23 °C, U 5 W/m^2-K, dU 150 W/m^2-K (full fan: U -> 155 W/m^2-K),
    %      Us 24 W/m^2-K, m 4E-3 kg, Cp 500 J/kg-K, A 1E-3 m^2,
    %      As 2E-4 m^2, Q1max 2 W, Q2max 1.5 W, emissivity 0.9,
    %      tauSensor 20 s, fanSwitch "off", Ts 1 s.
    %
    %   See also SystemBase, generateTCLabFanData.

    properties (Constant)
        stateDim = 4;
        outputDim = 2;
        sigma = 5.67E-8;    % [W/m^2-K^4] Stefan-Boltzmann constant
        T0 = 273.15;        % [K]
    end

    properties (Dependent)
        inputDim
    end

    properties (GetAccess=public, SetAccess=immutable)
        Ta (1,1) {mustBeNumeric} = 23;                       % [°C] ambient temperature
        U  (1,1) {mustBeNumeric, mustBePositive} = 5;        % [W/m^2-K] heat transfer coefficient, fan off
        dU (1,1) {mustBeNumeric, mustBeNonnegative} = 150;   % [W/m^2-K] increase at full fan speed
        Us (1,1) {mustBeNumeric, mustBePositive} = 24.0;     % [W/m^2-K] heat transfer coefficient between heaters
        m  (1,1) {mustBeNumeric, mustBePositive} = 4E-3;     % [kg] mass
        Cp (1,1) {mustBeNumeric, mustBePositive} = 500;      % [J/kg-K] heat capacity
        A  (1,1) {mustBeNumeric, mustBePositive} = 1E-3;     % [m^2] area not between heaters
        As (1,1) {mustBeNumeric, mustBePositive} = 2E-4;     % [m^2] surface area between heaters
        Q1max (1,1) {mustBeNumeric, mustBePositive} = 2;     % [W] max power heater 1
        Q2max (1,1) {mustBeNumeric, mustBePositive} = 1.5;  % [W] max power heater 2
        tauSensor (1,1) {mustBeNumeric, mustBePositive} = 20;% [s] sensor time constant
        emissivity (1,1) {mustBeNumeric, mustBePositive} = 0.9;
        fanSwitch (1,1) string {mustBeMember(fanSwitch, ["on", "off"])} = "off";
    end

    methods
        function obj = TCLab(options)
            arguments
                options.Ta (1,1) {mustBeNumeric};
                options.U (1,1) {mustBeNumeric, mustBePositive};
                options.dU (1,1) {mustBeNumeric, mustBeNonnegative};
                options.Us (1,1) {mustBeNumeric, mustBePositive};
                options.m (1,1) {mustBeNumeric, mustBePositive};
                options.Cp (1,1) {mustBeNumeric, mustBePositive};
                options.A (1,1) {mustBeNumeric, mustBePositive};
                options.As (1,1) {mustBeNumeric, mustBePositive};
                options.Q1max (1,1) {mustBeNumeric, mustBePositive};
                options.Q2max (1,1) {mustBeNumeric, mustBePositive};
                options.emissivity (1,1) {mustBeNumeric, mustBePositive};
                options.tauSensor (1,1) {mustBeNumeric, mustBePositive};
                options.Ts (1,1) {mustBePositive} = 1;
                options.odeopts (1,1) struct;
                options.processNoiseCov {mustBeNumeric};
                options.measurementNoiseCov {mustBeNumeric};
                options.fanSwitch (1,1) string {mustBeMember(options.fanSwitch, ["on", "off"])} = "off";
            end
            for f = string(fieldnames(options))'
                obj.(f) = options.(f);
            end
        end

        function dx = dynamics(obj, x, u, p)
            if nargin < 4
                p = obj;
            end
            T = x(1:2, :); % heater temperatures
            S = x(3:4, :); % sensor temperatures
            Q = [p.Q1max; p.Q2max] .* clip(u(1:2, :), 0, 100) / 100;

            Uc = p.U;
            if p.fanSwitch == "on"
                Uc = p.U + p.dU * clip(u(3, :), 0, 100) / 100;
            end

            % Heat transfer between heaters
            convection = p.Us * p.As * (T(2, :) - T(1, :));
            radiation = p.emissivity * obj.sigma * p.As * ((obj.T0 + T(2, :)).^4 - (obj.T0 + T(1, :)).^4);

            % Losses to ambient (fan-dependent convection + radiation)
            ambient = Uc .* p.A .* (p.Ta - T) + ...
                p.emissivity * obj.sigma * p.A * ((obj.T0 + p.Ta)^4 - (obj.T0 + T).^4);

            dx = [
                (ambient(1, :) + convection + radiation + Q(1, :)) / (p.m * p.Cp);
                (ambient(2, :) - convection - radiation + Q(2, :)) / (p.m * p.Cp);
                (T - S) / p.tauSensor
            ];
        end

        function Y = output(obj, X)
            arguments
                obj (1,1) TCLab; %#ok<INUSA>
                X (4,:) {mustBeNumeric};
            end
            Y = X(3:4, :);
        end

        function n = get.inputDim(obj)
            n = 2 + (obj.fanSwitch == "on");
        end
    end
end
