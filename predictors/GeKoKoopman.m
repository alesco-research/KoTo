classdef GeKoKoopman < KoopmanBase
    %GeKoKoopman Model y = K*kron(z,v), ridge-fitted.
    %
    %   obj = GeKoKoopman(Name=Value, ...)
    %     lifted features.
    %     useMultiStepWindows (true)  Pair selection mode:
    %       true : requires horizon, maxNumberOfWindows;
    %              optional maxNumberOfRawWindows. Expressiv windows are 
    %              chosen by k-means on [start; mid; end] states.
    %       false: requires maxNumberOfPairs (random one-step pairs).
    %
    %   learn(obj, Z, V, gamma=0)
    %     gamma: nonnegative ridge regularization weight.
    %
    %   y = predict(obj, z, v)
    %     Requires a fitted model (Koopman:OperatorUndefined otherwise).
    %     z, v must match stateDim and inputDim.

    properties (SetAccess=protected)
        K (:,:) {mustBeNumeric} = [];
    end

    methods
        function obj = GeKoKoopman(Y, Z, V, options)
            arguments
                Y (:,:) {mustBeNumeric};
                Z (:,:) {mustBeNumeric};
                V (:,:) {mustBeNumeric};
                options.gamma (1,1) {mustBeNumeric, mustBeNonnegative} = 0;
            end
            [obj.stateDim, obj.inputDim] = deal(size(Z, 1), size(V, 1));

            Phi = KoopmanBase.columnKron(Z, V);
            obj.K = (Y * Phi') / (Phi * Phi' + options.gamma * eye(size(Phi, 1)));
        end

        function y = predict(obj, z, v)
            arguments
                obj (1,1) GeKoKoopman {KoopmanBase.mustBeFitted(obj)};
                z (:,:) {KoopmanBase.mustMatchStateDim(obj, z)};
                v (:,:) {KoopmanBase.mustMatchInputDim(obj, v)};
            end
            y = obj.K * KoopmanBase.columnKron(z, v);
        end

        function [normK] = getOperatorNorm(obj)
            arguments
                obj (1,1) GeKoKoopman {KoopmanBase.mustBeFitted(obj)};
            end
            normK = norm(obj.K, "fro");
            fprintf("Geko:\t ||K||_F = %.4e\n", normK);
        end
    end
end