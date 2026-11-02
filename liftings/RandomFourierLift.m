classdef RandomFourierLift < LiftBase
    %RANDOMFOURIERLIFT Random Fourier feature (RFF) lift of the Gaussian kernel.
    %   obj = RANDOMFOURIERLIFT(dim, numberOfFeatures, sigma) draws
    %   numberOfFeatures random frequencies omega_i ~ N(0, I/sigma^2) and
    %   phases b_i ~ U[0, 2*pi), and lifts X (dim x nSamples) to
    %
    %       Z = [1; X; sqrt(2/D) * cos(Omega*X + b)],   D = numberOfFeatures,
    %
    %   so that phi(x)'*phi(y) approximates exp(-||x-y||^2 / (2 sigma^2))
    %   (Rahimi & Recht, 2007). The state is included (includeState = true),
    %   hence the projection back to X is exact.
    %
    %   Frequencies and phases are drawn from MATLAB's global random
    %   stream; call RNG beforehand for reproducible results.
    %
    %   Properties (read-only):
    %     numberOfFeatures - Number of random Fourier features D
    %     sigma            - Kernel bandwidth (in scaled state units)
    %     frequencies      - Omega, D-by-dim
    %     phases           - b, D-by-1

    properties (SetAccess=protected)
        dim = 1;
        featureDim = 1;
        includeState = true;
        numberOfFeatures (1,1) {mustBeInteger, mustBePositive} = 1;
        sigma (1,1) {mustBePositive} = 1;
        frequencies (:,:) {mustBeNumeric} = [];
        phases (:,1) {mustBeNumeric} = [];
    end

    methods
        function obj = RandomFourierLift(dim, numberOfFeatures, sigma)
            arguments
                dim (1,1) {mustBeInteger, mustBePositive};
                numberOfFeatures (1,1) {mustBeInteger, mustBePositive};
                sigma (1,1) {mustBePositive};
            end
            obj.dim = dim;
            obj.numberOfFeatures = numberOfFeatures;
            obj.sigma = sigma;
            obj.frequencies = randn(numberOfFeatures, dim) / sigma;
            obj.phases = 2 * pi * rand(numberOfFeatures, 1);
            obj.featureDim = 1 + dim + numberOfFeatures;
        end

        function Z = lift(obj, X)
            arguments
                obj (1,1) RandomFourierLift;
                X (:,:) {mustBeNumeric, LiftBase.mustMatchDim(obj, X)};
            end
            Z = [
                ones(1, size(X, 2));
                X;
                sqrt(2 / obj.numberOfFeatures) * cos(obj.frequencies * X + obj.phases)
            ];
        end
    end
end
