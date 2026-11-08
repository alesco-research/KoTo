function [A, B] = linearizeTCLab(params)
    %LINEARIZETCLAB Linearized heater subsystem of TCLab around ambient.
    %   [A, B] = linearizeTCLab()
    %   [A, B] = linearizeTCLab(Name=Value)
    %
    %   Linearizes the two-heater energy balance (convection, radiation and
    %   conduction between heaters) around the ambient temperature Ta. The
    %   resulting continuous-time model is dx/dt = A*x + B*u, where x are
    %   the heater temperatures relative to Ta and u are the heater inputs
    %   in percent.
    %
    %   Name-Value Arguments:
    %     Ta          Ambient temperature [°C] (default: 23).
    %     U           Heat transfer coeff. to ambient [W/m^2-K] (default: 4.5).
    %     Us          Heat transfer coeff. between heaters [W/m^2-K]
    %                 (default: 24).
    %     m           Mass [kg] (default: 4E-3).
    %     Cp          Heat capacity [J/kg-K] (default: 500).
    %     A           Surface area not between heaters [m^2] (default: 1E-3).
    %     As          Surface area between heaters [m^2] (default: 2E-4).
    %     Q1max       Max. power of heater 1 [W] (default: 1).
    %     Q2max       Max. power of heater 2 [W] (default: 0.75).
    %     emissivity  Emissivity [-] (default: 0.9).
    %
    %   Outputs:
    %     A  2-by-2 continuous-time system matrix [1/s].
    %     B  2-by-2 input matrix [K/(s*%)].

    arguments
        params.Ta (1,1) {mustBeNumeric} = 23;                      % [°C] ambient temperature
        params.U  (1,1) {mustBeNumeric, mustBePositive} = 4.5;     % [W/m^2-K] heat transfer coefficient
        params.Us (1,1) {mustBeNumeric, mustBePositive} = 24.0;    % [W/m^2-K] heat transfer coefficient between heaters
        params.m  (1,1) {mustBeNumeric, mustBePositive} = 4E-3;    % [kg] mass
        params.Cp (1,1) {mustBeNumeric, mustBePositive} = 500;     % [J/kg-K] heat capacity
        params.A  (1,1) {mustBeNumeric, mustBePositive} = 1E-3;    % [m^2] area not between heaters
        params.As (1,1) {mustBeNumeric, mustBePositive} = 2E-4;    % [m^2] surface area between heaters
        params.Q1max (1,1) {mustBeNumeric, mustBePositive} = 1;    % [W] max power heater 1
        params.Q2max (1,1) {mustBeNumeric, mustBePositive} = 0.75; % [W] max power heater 2
        params.emissivity (1,1) {mustBeNumeric, mustBePositive} = 0.9;
    end
    T0 = 273.15;
    sigma = 5.67E-8;
    mcp = params.m * params.Cp;
    rad_self = 4 * params.emissivity * sigma * params.A  * (params.Ta + T0)^3;
    rad_cross  = 4 * params.emissivity * sigma * params.As * (params.Ta + T0)^3;
    conv_self = params.U  * params.A;
    cond_cross = params.Us * params.As;

    a_self  = (conv_self + rad_self + cond_cross + rad_cross) / mcp;
    a_cross = (cond_cross + rad_cross) / mcp;

    A = [-a_self,  a_cross;
          a_cross, -a_self];
    B = [params.Q1max/mcp/100, 0;
        0, params.Q2max/mcp/100];
end