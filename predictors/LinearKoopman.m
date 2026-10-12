classdef LinearKoopman < KoopmanBase
    %LINEARKOOPMAN Linear model y = A*z + B*u fitted by ridge regression.
    %
    %   obj = LinearKoopman(Name=Value, ...)
    %     lifted features.
    %     useMultiStepWindows (true)  Pair selection mode:
    %       true : requires horizon, maxNumberOfWindows;
    %              optional maxNumberOfRawWindows. Expressiv windows are 
    %              chosen by k-means on [start; mid; end] states.
    %       false: requires maxNumberOfPairs (random one-step pairs).
    %
    %   y = predict(obj, z, u)
    %     Requires a fitted model (Koopman:OperatorUndefined otherwise).
    %     z, u must match stateDim and inputDim.

    properties (SetAccess=immutable)
        A (:,:) {mustBeNumeric} = [];
        B (:,:) {mustBeNumeric} = [];
    end

    methods
        function obj = LinearKoopman(Y, Z, U, options)
            arguments
                Y (:,:) {mustBeNumeric};
                Z (:,:) {mustBeNumeric, KoopmanBase.mustBeEqualSize(Y, Z)};
                U (:,:) {mustBeNumeric, KoopmanBase.mustBeEqualSize(Y, U)};
                options.gamma (1,1) {mustBeNumeric, mustBeNonnegative} = 0;
            end
            [nz, nu] = deal(size(Z, 1), size(U, 1));

            Phi = [Z; U];
            Theta = (Y * Phi') / (Phi * Phi' + options.gamma * eye(nz + nu));

            obj.stateDim = nz;
            obj.inputDim = nu;
            obj.A = Theta(:, 1:nz);
            obj.B = Theta(:, nz + 1:end);
        end

        function y = predict(obj, z, u)
            arguments
                obj (1,1) LinearKoopman;
                z (:,:) {KoopmanBase.mustMatchStateDim(obj, z)};
                u (:,:) {KoopmanBase.mustMatchInputDim(obj, u)};
            end
            y = obj.A * z + obj.B * u;
        end

        function [normA, normB] = getOperatorNorm(obj)
            arguments
                obj (1,1) LinearKoopman;
            end
            normA = norm(obj.A, "fro");
            normB = norm(obj.B, "fro");
            fprintf( ...
                "Linear:\t ||A||_F = %.4e,   ||B||_F = %.4e\n", normA, normB ...
            );
        end
    end
end
