classdef LinearKoopman < KoopmanBase
    %LINEARKOOPMAN Linear EDMDc predictor y = A*z + B*u.
    %
    %   obj = LinearKoopman(Y, Z, U, gamma=0)
    %     Y, Z   nz-by-P successor and current lifted states.
    %     U      nu-by-P raw inputs, Y(:,k) follows Z(:,k) under U(:,k).
    %     gamma  Ridge regularization weight (default 0).
    %     Fits [A, B] by ridge regression on the regressors [Z; U].
    %
    %   y = predict(obj, z, u)
    %     Single-step prediction; z, u must match stateDim and inputDim.
    %
    %   [normA, normB] = getOperatorNorm(obj)
    %     Frobenius norms of A and B (also printed).
    %
    %   See also KoopmanBase, getSingleStepPairs.

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
