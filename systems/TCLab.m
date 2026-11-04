classdef TCLab < SystemBase
    %TCLABFAN Simulated TC-Lab with a cooling fan and sensor dynamics (ODE45).
    %   Hypothetical variant of the TC-Lab used to study state-input
    %   coupling. Heater 1 is driven by u1; heater 2 is unpowered and is
    %   heated through the heat sink only. Input u2 drives a cooling fan
    %   that amplifies the heat transfer coefficient to ambient by the
    %   factor fanGain when switched on,
    %
    %       U(u2) = U * (1 + fanGain * u2),   u2 in {0, 1},
    %
    %   so the fan input multiplies the temperature difference to ambient:
    %   the dynamics contain the product of state and input (bilinear
    %   coupling), unlike the TC-Lab, whose heater inputs enter additively.
    %   Each temperature sensor is a first-order lag of its heater
    %   temperature (time constant tauSensor), giving 4 states.
    %
    %   M = TCLABFAN(Name=Value) sets the properties below at construction.
    %
    %   State  x = [T1; T2; S1; S2] in °C (heater and sensor temperatures).
    %   Output y = x(3:4) (measured sensor temperatures), see OUTPUT.
    %   Input  u = [u1; u2], u1 in % (clipped to [0, 100]), u2 the fan
    %   switch (off if u2 <= 0.5, on otherwise).
    %
    %   Properties (defaults):
    %      Ta 23 °C, U 5 W/m^2-K, fanGain 11 (fan on: U -> 55 W/m^2-K),
    %      Us 24 W/m^2-K, m 4E-3 kg, Cp 500 J/kg-K, A 1E-3 m^2,
    %      As 2E-4 m^2, Qmax 0.8 W (heater 1), emissivity 0.9,
    %      tauSensor 15 s, Ts 1 s.
    %
    %   See also TCLab, SystemBase, generateTCLabFanData.

    properties (Constant)
        stateDim = 4;
        inputDim = 3;
        outputDim = 2;
        sigma = 5.67E-8;    % [W/m^2-K^4] Stefan-Boltzmann constant
        T0 = 273.15;        % [K]
    end

    properties (GetAccess=public, SetAccess=immutable)
        Ta (1,1) {mustBeNumeric} = 23;                       % [°C] ambient temperature
        U  (1,1) {mustBeNumeric, mustBePositive} = 5;        % [W/m^2-K] heat transfer coefficient, fan off
        dU (1,1) {mustBeNumeric, mustBeNonnegative} = 50;    % [W/m^2-K] increase at full fan speed
        fanGain (1,1) {mustBeNumeric, mustBeNonnegative} = 11; % [-] amplification of U when the fan is on
        Us (1,1) {mustBeNumeric, mustBePositive} = 24.0;     % [W/m^2-K] heat transfer coefficient between heaters
        m  (1,1) {mustBeNumeric, mustBePositive} = 4E-3;     % [kg] mass
        Cp (1,1) {mustBeNumeric, mustBePositive} = 500;      % [J/kg-K] heat capacity
        A  (1,1) {mustBeNumeric, mustBePositive} = 1E-3;     % [m^2] area not between heaters
        As (1,1) {mustBeNumeric, mustBePositive} = 2E-4;     % [m^2] surface area between heaters
        Q1max (1,1) {mustBeNumeric, mustBePositive} = 1;     % [W] max power heater 1
        Q2max (1,1) {mustBeNumeric, mustBePositive} = 0.75;  % [W] max power heater 2
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
                options.fanGain (1,1) {mustBeNumeric, mustBeNonnegative};
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
                Uc = p.U + p.dU * clip(u(2, :), 0, 100) / 100;
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
    end
end
