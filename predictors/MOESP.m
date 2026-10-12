classdef MOESP
%MOESP Subspace identification of a linear state-space model.
%   Identifies (A, B, C, D) from input-output data with the MOESP
%   algorithm and provides one-step prediction and warm-started forecasting.
%
%   obj = MOESP(Y, U, L, gamma=0.85, stateDim=[])
%     Y         ny-by-N output data (samples as columns).
%     U         nu-by-N input data (samples as columns).
%     L         Block-Hankel depth. Must be larger than the state dimension.
%     gamma     Cumulative singular-value threshold in (0, 1) used to select
%               the model order (default 0.85).
%     stateDim  Fixes the model order and overrides gamma.
%
%   Properties (read-only):
%     A, B, C, D                State-space matrices.
%     inputDim, outputDim       Number of inputs and outputs.
%     stateDim                  Identified model order.
%     sigma                     Singular values used for order selection.
%     memory                    Number of past samples required by forecast
%                               (equal to stateDim).
%     O, G                      Observability matrix and Toeplitz matrix of
%                               Markov parameters over the memory window,
%                               used to estimate the initial state.
%
%   [x, y] = predict(obj, x0, u)
%     One-step prediction: x = A*x0 + B*u, y = C*x0 + D*u.
%
%   Y = forecast(obj, uPast, yPast, uFuture)
%     Warm-started multi-step output prediction.
%       uPast, yPast  Last `memory` input and output samples (columns).
%       uFuture       nu-by-T future inputs.
%     The state at the start of the window is estimated by least squares
%     from (uPast, yPast), then simulated through the window and the
%     future inputs. Returns an ny-by-(memory+T) matrix Y, where the first
%     `memory` columns (the warm-up window) are NaN and columns
%     memory+1:end hold the predicted outputs for uFuture.

    properties (SetAccess=immutable)
        outputDim (1,1) {mustBeInteger, mustBePositive};
        stateDim {mustBeInteger, mustBePositive};
        inputDim {mustBeInteger, mustBePositive};
        A (:,:) {mustBeNumeric, mustBeMatrix};
        B (:,:) {mustBeNumeric, mustBeMatrix};
        C (:,:) {mustBeNumeric, mustBeMatrix};
        D (:,:) {mustBeNumeric, mustBeMatrix};
        O (:,:) {mustBeNumeric, mustBeMatrix};
        G (:,:) {mustBeNumeric, mustBeMatrix};
        sigma (:,1) {mustBeNumeric, mustBeNonnegative};
        memory {mustBeInteger, mustBePositive};
    end

    methods
        function obj = MOESP(Y, U, L, options)
            arguments
                Y (:,:) {mustBeNumeric}
                U (:,:) {mustBeNumeric, MOESP.mustBeEqualSize(Y, U)}
                L (1,1) {mustBeInteger, mustBePositive}
                options.gamma (1,1) {mustBeNumeric, mustBeBetween(options.gamma, 0, 1)} = 0.85;
                options.stateDim {mustBeInteger, MOESP.mustBeValidLength(L, options.stateDim)} = [];
            end
            [ny, nu] = deal(size(Y, 1), size(U, 1));

            % Get scaled hankel matrices
            N = size(Y, 2) - L + 1;
            Hy = MOESP.hankel(Y, L) ./ sqrt(N);
            Hu = MOESP.hankel(U, L) ./ sqrt(N);

            Hyp = Hy - (Hy / Hu) * Hu;
            [Un, obj.sigma] = svd(Hyp, "econ", "vector");

            nx = options.stateDim;
            if isempty(nx)
                nx = find(cumsum(obj.sigma) / sum(obj.sigma) > options.gamma, 1);
                MOESP.mustBeValidLength(L, nx)
            end

            obsL = Un(:, 1:nx) * diag(sqrt(obj.sigma(1:nx)));

            % Compute A and B
            obj.C = obsL(1:ny, :);
            obj.A = pinv(obsL(1:ny * (L - 1),:)) * obsL(ny + 1:L * ny,:);
            
            % Compute B and D
            U2t = Un(:, nx + 1:end)';
            M1 = U2t * (Hy / Hu);
            m = ny * L - nx;
            M = zeros(m * L, nu);
            R = zeros(m * L, ny + nx);
            for k = 1:L
                M((k-1) * m + 1:k * m,:) = M1(:,(k - 1) * nu + 1:k * nu);
                R((k-1) * m + 1:k * m,:) = [
                    U2t(:, (k - 1) * ny + 1:k * ny), ...
                    U2t(:, k * ny + 1:end) * obsL(1:end - k * ny,:)
                ];
            end
            DB = R \ M;
            obj.D = DB(1:ny, :);
            obj.B = DB(ny + 1:end, :);
            obj.outputDim = ny;
            obj.stateDim = nx;
            obj.inputDim = nu;
            obj.memory = nx;
            [obj.O, obj.G] = obj.observabilityAndToeplitz(obj.memory);
        end

        function [x, y] = predict(obj, x0, u)
            arguments
                obj (1,1) MOESP;
                x0 (:,:) {MOESP.mustMatch(obj, "stateDim", x0)};
                u (:,:) {MOESP.mustMatch(obj, "inputDim", u)};
            end
            x = obj.A * x0 + obj.B * u;
            y = obj.C * x0 + obj.D * u;
        end

        function Y = forecast(obj, uPast, yPast, uFuture)
            arguments
                obj (1,1) MOESP;
                uPast (:,:) {MOESP.mustMatch(obj, "inputDim", uPast), MOESP.mustMatchMemory(obj, uPast)};
                yPast (:,:) {MOESP.mustMatch(obj, "outputDim", yPast), MOESP.mustMatchMemory(obj, yPast)};
                uFuture (:,:) {MOESP.mustMatch(obj, "inputDim", uFuture)};
            end
            [m, T] = deal(size(uPast, 2), size(uFuture, 2));
            
            x = obj.O \ (yPast(:) - obj.G * uPast(:));

            u = [uPast, uFuture];
            Y = zeros(obj.outputDim, m + T);
            for t = 1:(m + T)
                [x, Y(:, t)] = obj.predict(x, u(:, t));
            end
            Y(:, 1:m) = nan;
        end
    end

    methods (Access=protected)
        function [O, G] = observabilityAndToeplitz(obj, m)
            ny = obj.outputDim;
            nx = obj.stateDim;
            nu = obj.inputDim;

            O = zeros(m * ny, nx);      % observability matrix
            G = zeros(m * ny, m * nu);  % toeplitz matrix
            H = zeros(ny, nu, m);
            H(:, :, 1) = obj.D;

            CAk = obj.C;
            for k = 1:m
                O((k - 1) * ny + (1:ny), :) = CAk;
                if k < m
                    H(:, :, k + 1) = CAk * obj.B;
                end
                CAk = CAk * obj.A;
            end

            for i = 1:m
                for j = 1:i
                    G((i - 1) * ny + (1:ny), (j - 1) * nu + (1:nu)) = H(:, :, i - j + 1);
                end
            end
        end
    end

    methods (Static, Hidden, Access=protected)
        function h = hankel(X, L)
            [d, T] = size(X);
            cols = T - L + 1;

            idx = (0:L-1)' + (1:cols);
            h = reshape(X(:, idx(:)), L * d, cols);
        end

        function mustBeValidLength(L, stateDim)
            if L <= stateDim
                error( ...
                    "MOESP:HankelMatrixTooShort", ...
                    "L %d was chosen to small. Must be larger than the " + ...
                    "state dimension %d.", ...
                    L, stateDim ...    
                )
            end
        end

        function mustBeEqualSize(Z, W)
            if size(Z, 2) ~= size(W, 2)
                error( ...
                    "MOESP:DimensionMismatch", ...
                    "Inputs must have the same number of samples. " + ...
                    "Received inputs of dimension %d and %d instead.", ...
                    size(Z, 2), ...
                    size(W, 2) ...
            );
            end
        end

        function mustMatch(obj, dim, val)
            if size(val, 1) ~= obj.(dim)
                error( ...
                    "MOESP:DimensionMismatch", ...
                    "Value must have %d row(s) to match the " + ... 
                    "%s dimension.", ...
                    obj.(dim), dim ...
                );
            end
        end

        function mustMatchMemory(obj, val)
            if size(val, 2) ~= obj.memory
                error( ...
                    "MOESP:MemoryMismatch", ...
                    "Must provide %d past inputs and outputs to match the " + ... 
                    "memory structure of the predictor.", ...
                    obj.memory ...
                );
            end
        end
    end
end