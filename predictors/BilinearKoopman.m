classdef BilinearKoopman < KoopmanBase
    %BILINEARKOOPMAN Bilinear model y = A*z + B*u + N*kron(u,z), ridge-fitted.
    %
    %   obj = BilinearKoopman(Name=Value, ...)
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
        N (:,:) {mustBeNumeric} = [];
    end

    methods
        function obj = BilinearKoopman(Y, Z, U, options)
            arguments
                Y (:,:) {mustBeNumeric};
                Z (:,:) {mustBeNumeric, KoopmanBase.mustBeEqualSize(Y, Z)};
                U (:,:) {mustBeNumeric, KoopmanBase.mustBeEqualSize(Y, U)};
                options.gamma (1,1) {mustBeNumeric, mustBeNonnegative} = 0;
            end
            [nz, nu] = deal(size(Z, 1), size(U, 1));

            Phi = [Z; U; KoopmanBase.columnKron(U, Z)];
            Theta = (Y * Phi') / (Phi * Phi' + options.gamma * eye(size(Phi, 1)));

            obj.stateDim = nz;
            obj.inputDim = nu;
            obj.A = Theta(:, 1:nz);
            obj.B = Theta(:, nz + 1 : nz + nu);
            obj.N = Theta(:, nz + nu + 1 : end);
        end

        function y = predict(obj, z, u)
            arguments
                obj (1,1) BilinearKoopman;
                z (:,:) {KoopmanBase.mustMatchStateDim(obj, z)};
                u (:,:) {KoopmanBase.mustMatchInputDim(obj, u)};
            end
            y = obj.A * z + obj.B * u + obj.N * KoopmanBase.columnKron(u, z);
        end

        function [normA, normB, normN] = getOperatorNorm(obj)
            arguments
                obj (1,1) BilinearKoopman;
            end
            normA = norm(obj.A, "fro");
            normB = norm(obj.B, "fro");
            normN = norm(obj.N, "fro");
            fprintf( ...
                "Bilinear:" + ...
                "||A||_F = %.4e,   ||B||_F = %.4e,   ||N||_F = %.4e\n", ...
                normA, normB, normN ...
            );
        end
    end
end