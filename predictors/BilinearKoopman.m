classdef BilinearKoopman < KoopmanBase
    %BILINEARKOOPMAN Bilinear EDMDc predictor y = A*z + B*u + N*kron(u,z).
    %
    %   obj = BilinearKoopman(Y, Z, U, gamma=0)
    %     Y, Z   nz-by-P successor and current lifted states.
    %     U      nu-by-P raw inputs, Y(:,k) follows Z(:,k) under U(:,k).
    %     gamma  Ridge regularization weight (default 0).
    %     Fits [A, B, N] by ridge regression on the regressors
    %     [Z; U; kron(U,Z)] (column-wise Kronecker product).
    %
    %   y = predict(obj, z, u)
    %     Single-step prediction; z, u must match stateDim and inputDim.
    %
    %   [normA, normB, normN] = getOperatorNorm(obj)
    %     Frobenius norms of A, B and N (also printed).
    %
    %   See also KoopmanBase, getSingleStepPairs.

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
                "Bilinear:\t " + ...
                "||A||_F = %.4e,   ||B||_F = %.4e,   ||N||_F = %.4e\n", ...
                normA, normB, normN ...
            );
        end
    end
end