classdef GeKoKoopman < KoopmanBase
    %GEKOKOOPMAN Generalized Koopman (GeKo) predictor y = K*kron(z,v).
    %
    %   obj = GeKoKoopman(Y, Z, V, gamma=0)
    %     Y, Z   nz-by-P successor and current lifted states.
    %     V      nv-by-P lifted inputs, Y(:,k) follows Z(:,k) under V(:,k).
    %     gamma  Ridge regularization weight (default 0).
    %     Fits K by ridge regression on the regressors kron(Z,V)
    %     (column-wise Kronecker product). For V = [1; U] and a state
    %     lifting that contains a constant, this equals BilinearKoopman.
    %
    %   y = predict(obj, z, v)
    %     Single-step prediction; z, v must match stateDim and inputDim.
    %
    %   normK = getOperatorNorm(obj)
    %     Frobenius norm of K (also printed).
    %
    %   See also KoopmanBase, getSingleStepPairs.

    properties (SetAccess=immutable)
        K (:,:) {mustBeNumeric} = [];
    end

    methods
        function obj = GeKoKoopman(Y, Z, V, options)
            arguments
                Y (:,:) {mustBeNumeric};
                Z (:,:) {mustBeNumeric, KoopmanBase.mustBeEqualSize(Y, Z)};
                V (:,:) {mustBeNumeric, KoopmanBase.mustBeEqualSize(Y, V)};
                options.gamma (1,1) {mustBeNumeric, mustBeNonnegative} = 0;
            end
            [obj.stateDim, obj.inputDim] = deal(size(Z, 1), size(V, 1));

            Phi = KoopmanBase.columnKron(Z, V);
            obj.K = (Y * Phi') / (Phi * Phi' + options.gamma * eye(size(Phi, 1)));
        end

        function y = predict(obj, z, v)
            arguments
                obj (1,1) GeKoKoopman;
                z (:,:) {KoopmanBase.mustMatchStateDim(obj, z)};
                v (:,:) {KoopmanBase.mustMatchInputDim(obj, v)};
            end
            y = obj.K * KoopmanBase.columnKron(z, v);
        end

        function [normK] = getOperatorNorm(obj)
            arguments
                obj (1,1) GeKoKoopman;
            end
            normK = norm(obj.K, "fro");
            fprintf("GeKo:\t ||K||_F = %.4e\n", normK);
        end
    end
end