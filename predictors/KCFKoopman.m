classdef KCFKoopman < KoopmanBase
    %KCFKOOPMAN Koopman control family (KCF) predictor in input-state
    %   separable form, y = A11*z + A12*kron(v(vKeep), z(couplingIdxs)).
    %
    %   obj = KCFKoopman(Y, Z, V, couplingIdxs=1:nz, gamma=0)
    %     Y, Z          nz-by-P successor and current lifted states.
    %     V             nv-by-P lifted inputs, Y(:,k) follows Z(:,k) under V(:,k).
    %     couplingIdxs  Rows of z coupled with the input, i.e. the
    %                   selection z^c = S*z (default: all rows).
    %     gamma         Ridge regularization weight (default 0).
    %     Rows of V that are constant in the training data (e.g. the
    %     constant of a polynomial input lifting) are dropped from the
    %     coupling term; vKeep marks the retained rows.
    %
    %   y = predict(obj, z, v)
    %     Single-step prediction; z, v must match stateDim and inputDim,
    %     and the dropped rows of v must stay constant.
    %
    %   [normA11, normA12] = getOperatorNorm(obj)
    %     Frobenius norms of A11 and A12 (also printed).
    %
    %   See also KoopmanBase, getSingleStepPairs.

    properties (SetAccess=immutable)
        A11 (:,:) {mustBeNumeric} = [];
        A12 (:,:) {mustBeNumeric} = [];
        couplingIdxs (1,:) {mustBeInteger} = [];
        vKeep (:,1) logical = true(0,1);
    end

    methods
        function obj = KCFKoopman(Y, Z, V, options)
            arguments
                Y (:,:) {mustBeNumeric};
                Z (:,:) {mustBeNumeric, KoopmanBase.mustBeEqualSize(Y, Z)};
                V (:,:) {mustBeNumeric, KoopmanBase.mustBeEqualSize(Y, V)};
                options.gamma (1,1) {mustBeNumeric, mustBeNonnegative} = 0;
                options.couplingIdxs (1,:) {mustBeInteger, mustBePositive} = 1:size(Z, 1);
            end
            nz = size(Z, 1);
            ci = options.couplingIdxs;
            if max(ci) > nz
                error( ...
                    "KoopmanBase:InvalidCouplingIdxs", ...
                    "max(couplingIdxs) (%d) exceeds the lifted state dimension %d.", ...
                    max(ci), nz ...
                )
            end
            [obj.stateDim, obj.inputDim] = deal(size(Z, 1), size(V, 1));

            % Drop constant inputs below threshold variance
            obj.vKeep = var(V, 0, 2) > 1E-10;

            Phi   = [Z; KoopmanBase.columnKron(V(obj.vKeep, :), Z(ci, :))];
            Theta = (Y * Phi') / (Phi * Phi' + options.gamma * eye(size(Phi, 1)));

            obj.A11 = Theta(:, 1:nz);
            obj.A12 = Theta(:, nz + 1:end);
            obj.couplingIdxs = ci;
        end

        function y = predict(obj, z, v)
            arguments
                obj (1,1) KCFKoopman;
                z (:,:) {KoopmanBase.mustMatchStateDim(obj, z)};
                v (:,:) {KCFKoopman.mustBeValidInput(obj, v)};
            end
            y = obj.A11 * z + obj.A12 * KoopmanBase.columnKron(v(obj.vKeep, :), z(obj.couplingIdxs, :));
        end

        function [normA11, normA12] = getOperatorNorm(obj)
            arguments
                obj (1,1) KCFKoopman;
            end
            normA11 = norm(obj.A11, "fro");
            normA12 = norm(obj.A12, "fro");
            fprintf( ...
                "KCF:\t ||A11||_F = %.4e, ||A12||_F = %.4e\n", ...
                normA11, normA12 ...
            );
        end
    end

    methods (Static, Hidden, Access=protected)
        function mustBeValidInput(obj, v)
            KoopmanBase.mustMatchInputDim(obj, v)
            constantInputs = var(v(~obj.vKeep,:), 0, 2) <= 1E-10;
            if ~all(constantInputs)
                constantIdxs = find(~obj.vKeep);
                error( ...
                    "KoopmanBase:InvalidInput", ...
                    "Input lifting(s) %s were constant during " + ...
                    "training but vary in the supplied data.", ...
                    mat2str(constantIdxs(~constantInputs)) ...
                )
            end
        end
    end
end