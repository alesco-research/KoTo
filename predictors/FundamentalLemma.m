classdef FundamentalLemma
%FUNDAMENTALLEMMA Data-driven predictor based on Willems' fundamental lemma.
%   Represents the finite-length trajectories of an LTI system by the
%   column span of block-Hankel matrices of recorded input-output data and
%   provides warm-started forecasting without identifying a model.
%
%   obj = FundamentalLemma(Y1, U1, Y2, U2, ..., memory=10, horizon=50, gamma=0)
%     Yi        ny-by-N_i output trajectory.
%     Ui        nu-by-N_i input trajectory matching Yi. Pairs (Yi, Ui) are
%               passed as repeated arguments; at least one is required.
%               From cell arrays: pairs = [Y(:)'; U(:)'];
%               FundamentalLemma(pairs{:})
%     memory    Number of past samples fixing the initial condition. Must
%               be at least the lag of the system (default 10).
%     horizon   Number of future samples predicted per step (default 50).
%               The Hankel depth is L = memory + horizon, and every
%               trajectory needs at least L samples.
%     gamma    Tikhonov regularization of the Hankel coefficients. Zero
%               gives the minimum-norm solution (default 0).
%
%   Hankel matrices are built per trajectory and concatenated column-wise,
%   so trajectories may differ in length and start from different states.
%   For exact LTI data the lemma requires the input to be persistently
%   exciting of order L + n; a warning is issued if the input Hankel matrix
%   does not have full row rank, which is a necessary condition.
%
%   Properties (read-only):
%     inputDim, outputDim       Number of inputs and outputs.
%     memory, horizon           Past and future window lengths.
%     gamma                    Regularization weight.
%     K                         Predictor matrix mapping
%                               [uPast(:); yPast(:); uFuture(:)] to the
%                               predicted outputs yFuture(:) over horizon.
%
%   Y = forecast(obj, uPast, yPast, uFuture)
%     Warm-started multi-step output prediction.
%       uPast, yPast  Past samples (columns), at least `memory` each, ending
%                     at the same time step. Only the last `memory` are used.
%       uFuture       nu-by-T future inputs.
%     Predictions are made in blocks of `horizon` samples, each block
%     warm-started with the last `memory` inputs and predicted outputs. The
%     last block is padded by holding the final input. Returns an ny-by-T
%     matrix of predicted outputs for uFuture.

    properties (SetAccess=immutable)
        outputDim {mustBeInteger, mustBePositive} = [];
        inputDim {mustBeInteger, mustBePositive} = [];
        memory {mustBeInteger, mustBePositive};
        horizon {mustBeInteger, mustBePositive};
        gamma (1,1) {mustBeNumeric, mustBeNonnegative};
        K (:,:) {mustBeNumeric, mustBeMatrix};
    end

    methods
        function obj = FundamentalLemma(Y, U, options)
            arguments (Input, Repeating)
                Y (:,:) {mustBeNumeric}
                U (:,:) {mustBeNumeric, FundamentalLemma.mustBeEqualSize(Y, U)}
            end
            arguments (Input)
                options.memory (1,1) {mustBeInteger, mustBePositive} = 10;
                options.horizon (1,1) {mustBeInteger, mustBePositive} = 50;
                options.gamma (1,1) {mustBeNumeric, mustBeNonnegative} = 0;
            end
            if isempty(Y)
                error( ...
                    "FundamentalLemma:NoData", ...
                    "Provide at least one (Y, U) trajectory pair." ...
                );
            end
            [ny, nu] = deal(size(Y{1}, 1), size(U{1}, 1));
            [m, h] = deal(options.memory, options.horizon);
            L = m + h;

            % Per-trajectory Hankel matrices, concatenated column-wise
            Hy = cellfun(@(y) FundamentalLemma.hankel(y, L), Y, UniformOutput=false);
            Hu = cellfun(@(u) FundamentalLemma.hankel(u, L), U, UniformOutput=false);
            Hy = [Hy{:}];
            Hu = [Hu{:}];

            if rank(Hu) < nu * L
                warning( ...
                    "FundamentalLemma:NotPersistentlyExciting", ...
                    "Input Hankel matrix of depth L = %d is rank " + ...
                    "deficient. The input is not persistently exciting.", ...
                    L ...
                );
            end

            % Scale by the total number of columns
            N = size(Hy, 2);
            Hy = Hy ./ sqrt(N);
            Hu = Hu ./ sqrt(N);

            % Split into past and future
            Up = Hu(1:nu * m, :);
            Uf = Hu(nu * m + 1:end, :);
            Yp = Hy(1:ny * m, :);
            Yf = Hy(ny * m + 1:end, :);

            % yFuture = Yf * g with [Up; Yp; Uf] * g = [uPast; yPast; uFuture]
            H = [Up; Yp; Uf];
            obj.K = (Yf * H') / (H * H' + options.gamma * eye(size(H, 1)));

            obj.outputDim = ny;
            obj.inputDim = nu;
            obj.memory = m;
            obj.horizon = h;
            obj.gamma = options.gamma;
        end

        function Y = forecast(obj, uPast, yPast, uFuture)
            arguments
                obj (1,1) FundamentalLemma;
                uPast (:,:) {FundamentalLemma.mustMatch(obj, "inputDim", uPast), FundamentalLemma.mustMatchMemory(obj, uPast)};
                yPast (:,:) {FundamentalLemma.mustMatch(obj, "outputDim", yPast), FundamentalLemma.mustMatchMemory(obj, yPast)};
                uFuture (:,:) {FundamentalLemma.mustMatch(obj, "inputDim", uFuture)};
            end
            [m, h] = deal(obj.memory, obj.horizon);
            yPast = yPast(:,1 + end - m:end);
            uPast = uPast(:,1 + end - m:end);
            T = size(uFuture, 2);

            % Pad input to ceiled value
            numberOfBlocks = ceil(T / h);
            uFuture = paddata( ...
                uFuture, numberOfBlocks * h, Dimension=2, Pattern="edge" ...
            );

            Y = zeros(obj.outputDim, numberOfBlocks * h);
            for k = 1:numberOfBlocks
                idx = (k - 1) * h + (1:h);
                uBlock = uFuture(:, idx);
                yBlock = reshape( ...
                    obj.K * [uPast(:); yPast(:); uBlock(:)], ...
                    obj.outputDim, h ...
                );
                Y(:, idx) = yBlock;

                % Shift past
                uTemp = [uPast, uBlock];
                yTemp = [yPast, yBlock];
                uPast = uTemp(:, end - m + 1:end);
                yPast = yTemp(:, end - m + 1:end);
            end
            Y(:, T+1:end) = [];
        end
    end

    methods (Static, Hidden, Access=protected)
        function h = hankel(X, L)
            [d, T] = size(X);
            cols = T - L + 1;

            if cols < 1
                error( ...
                    "FundamentalLemma:TrajectoryTooShort", ...
                    "Trajectory has %d samples but needs at least L = %d.", ...
                    T, L ...
                );
            end

            idx = (0:L-1)' + (1:cols);
            h = reshape(X(:, idx(:)), L * d, cols);
        end

        function mustBeEqualSize(Y, U)
            if size(Y, 2) ~= size(U, 2)
                error( ...
                    "FundamentalLemma:DimensionMismatch", ...
                    "Y and U must have trajectories of equal length." ...
                );
            end
        end

        function mustMatch(obj, dim, val)
            if size(val, 1) ~= obj.(dim)
                error( ...
                    "FundamentalLemma:DimensionMismatch", ...
                    "Value must have %d row(s) to match the " + ...
                    "%s dimension.", ...
                    obj.(dim), erase(dim, "Dim") ...
                );
            end
        end

        function mustMatchMemory(obj, val)
            if size(val, 2) < obj.memory
                error( ...
                    "FundamentalLemma:MemoryMismatch", ...
                    "Must provide %d past inputs and outputs to match the " + ...
                    "memory structure of the predictor.", ...
                    obj.memory ...
                );
            end
        end
    end
end
