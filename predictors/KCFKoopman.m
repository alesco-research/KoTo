classdef KCFKoopman < KoopmanBase
    %KCFKOOPMAN Input-state separable KCF (Haseli & Cortes 2025):
    %   y = A11*z + A12*kron(v_eff, z(coupleIdx)). 
    %
    %   obj = KCFKoopman(Name=Value, ...)
    %     lifted features.
    %     useMultiStepWindows (true)  Pair selection mode:
    %       true : requires horizon, maxNumberOfWindows;
    %              optional maxNumberOfRawWindows. Expressiv windows are 
    %              chosen by k-means on [start; mid; end] states.
    %       false: requires maxNumberOfPairs (random one-step pairs).
    %
    %   learn(obj, Z, V, gamma=0, coupleIdx=1:stateDim)
    %     gamma:     nonnegative ridge weight (uniform).
    %     coupleIdx: lifted state dimensions to which the inputs couple to.
    %   Drops constant inputs v with variance < 1e-10.
    %
    %   y = predict(obj, z, v)
    %     Requires a fitted model (Koopman:OperatorUndefined otherwise).
    %     z, v must match stateDim and inputDim.

    properties (SetAccess=protected)
        A11 (:,:) {mustBeNumeric} = [];
        A12 (:,:) {mustBeNumeric} = [];
        couplingIdxs (1,:) {mustBeInteger} = [];
        vKeep (:,1) logical = true(0,1);
    end

    methods
        function obj = KCFKoopman(Y, Z, V, options)
            arguments
                Y (:,:) {mustBeNumeric};
                Z (:,:) {mustBeNumeric};
                V (:,:) {mustBeNumeric};
                options.gamma (1,1) {mustBeNumeric, mustBeNonnegative} = 0;
                options.couplingIdxs (1,:) {mustBeInteger, mustBePositive} = 1:size(Z, 1);
            end
            nz = size(Z, 1);
            ci = options.couplingIdxs;
            if max(ci) > nz
                error( ...
                    "Koopman:InvalidCoupleIdx", ...
                    "coupleIdx max (%d) exceeds lift dimension %d.", ...
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
                obj (1,1) KCFKoopman {KoopmanBase.mustBeFitted(obj)};
                z (:,:) {KoopmanBase.mustMatchStateDim(obj, z)};
                v (:,:) {KoopmanBase.mustMatchInputDim(obj, v)};
            end
            y = obj.A11 * z + obj.A12 * KoopmanBase.columnKron(v(obj.vKeep, :), z(obj.couplingIdxs, :));
        end

        function [normA11, normA12] = getOperatorNorm(obj)
            arguments
                obj (1,1) KCFKoopman {KoopmanBase.mustBeFitted(obj)};
            end
            normA11 = norm(obj.A11, "fro");
            normA12 = norm(obj.A12, "fro");
            fprintf( ...
                "KCF:\t ||A11||_F = %.4e, ||A12||_F = %.4e\n", ...
                normA11, normA12 ...
            );
        end
    end
end