classdef DelayEmbedding
    %DELAYEMBEDDING Time-delay (Hankel) embedding of multivariate time series.
    %   obj = DELAYEMBEDDING(dim, degree) stacks degree+1 copies of the
    %   data, each delayed by one sample. degree=0 returns the data as-is
    %   (no delays added).
    %
    %   obj = DELAYEMBEDDING(dim, degree, stride) delays successive copies
    %   by stride samples.
    %
    %   E = embed(obj, X)
    %       X is dim-by-T (columns are observations, rows are dimensions).
    %       E is ((degree+1)*dim)-by-(T-memory) for degree>0, or
    %       dim-by-T for degree=0. The column at time t is
    %           [X(:,t); X(:,t-stride); ...; X(:,t-degree*stride)],
    %       for t = memory+1, ..., T. The newest sample is the top block,
    %       so the current state is always E(1:dim,:). X must have at
    %       least memory+1 columns. Embed each experiment separately,
    %       never across experiment boundaries.
    %
    %   X = unembed(obj, E)
    %       Return the current-sample block E(1:dim,:).
    %
    %   X = unembed(obj, E, true)
    %       Also reconstruct and prepend the first memory samples from
    %       the lagged blocks of E, recovering all T columns exactly (no
    %       interpolation). Requires degree>0 and at least stride columns
    %       in E.
    %
    %   Example:
    %       X  = randn(3, 50);
    %       de = DelayEmbedding(3, 3, 2);
    %       E  = embed(de, X);               % 12-by-44
    %       Y  = unembed(de, E, true);       % equals X
    %
    %   Properties (read-only):
    %      degree     - Number of delays added (0 = no delays)
    %      stride     - Delay between copies in samples (default 1)
    %      memory     - Number of past samples required, stride*degree
    %      dim        - Input dimensionality
    %      featureDim - Output row count, dim*(degree+1)

    properties (GetAccess=public, SetAccess=immutable)
        memory (1,1) {mustBeInteger, mustBeNonnegative}
        stride (1,1) {mustBeInteger, mustBePositive} = 1;
        degree (1,1) {mustBeInteger, mustBeNonnegative}
        dim (1,1) {mustBeInteger, mustBePositive} = 1;
        featureDim (1,1) {mustBeInteger, mustBePositive} = 1;
    end

    methods
        function obj = DelayEmbedding(dim, degree, stride)
            arguments
                dim (1,1) {mustBeInteger, mustBePositive}
                degree (1,1) {mustBeInteger, mustBeNonnegative}
                stride (1,1) {mustBeInteger, mustBePositive} = 1;
            end
            obj.dim = dim;
            obj.degree = degree;
            obj.stride = stride;
            obj.memory = stride * degree;
            obj.featureDim = dim * (degree + 1);
        end

        function embedding = embed(obj, data)
            arguments
                obj  (1,1) DelayEmbedding
                data (:,:) {DelayEmbedding.mustBeEmbeddable(obj, data)}
            end
            T = size(data, 2);

            embedding = zeros((obj.degree + 1) * obj.dim, T - obj.memory);
            for i = 0:obj.degree
                cols = (obj.memory + 1 : T) - i * obj.stride;
                embedding(i * obj.dim + (1:obj.dim), :) = data(:, cols);
            end
        end

        function data = unembed(obj, embedding, reconstruct)
            arguments
                obj (1,1) DelayEmbedding
                embedding (:,:) {DelayEmbedding.mustBeEmbedding(obj, embedding)}
                reconstruct (1,1) logical = false;
            end
            data = embedding(1:obj.dim, :);

            if reconstruct && obj.degree > 0
                DelayEmbedding.mustBeReconstructable(obj, embedding);
                H = embedding(obj.dim + 1:end, 1:obj.stride);
                H = reshape(H, obj.dim, obj.degree, obj.stride);
                H = flip(H, 2);
                H = permute(H, [1 3 2]);
                hist = reshape(H, obj.dim, []);
                data = [hist, data];
            end
        end
    end

    methods (Access=private, Hidden, Static)
        function mustBeEmbeddable(obj, data)
            mustBeNumeric(data);
            if size(data, 1) ~= obj.dim
                error( ...
                    "DelayEmbedding:DimensionMismatch", ...
                    "Data has %d rows but dim=%d was specified.", ...
                    size(data,1), obj.dim);
            end
            if size(data, 2) < obj.memory + 1
                error( ...
                    "DelayEmbedding:TooFewData", ...
                    "Data must have at least memory+1 = %d column(s).", ...
                    obj.memory + 1);
            end
        end

        function mustBeEmbedding(obj, embedding)
            mustBeNumeric(embedding);
            expected = (obj.degree + 1) * obj.dim;
            if size(embedding, 1) ~= expected
                error( ...
                    "DelayEmbedding:DimensionMismatch", ...
                    "Embedding must have (degree+1)*dim = %d rows, got %d.", ...
                    expected, size(embedding,1));
            end
        end

        function mustBeReconstructable(obj, embedding)
            if size(embedding, 2) < obj.stride
                error( ...
                    "DelayEmbedding:TooFewColumns", ...
                    "Need at least stride=%d column(s) to reconstruct history.", ...
                    obj.stride);
            end
        end
    end
end