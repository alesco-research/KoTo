function B = paddata(A, m, options)
%PADDATA Compatibility shim for MATLAB releases before R2023b.
%   B = paddata(A, m, Dimension=d, FillValue=v) pads A at the end of
%   dimension d to length m with the constant v (default 0).
%   B = paddata(A, m, Dimension=d, Pattern="edge") repeats the last slice.
%   Only trailing padding is supported (the default of the built-in).
%   If size(A, d) >= m, A is returned unchanged.
%   Loaded by initWorkspace only when the built-in is missing.
    arguments
        A
        m (1,1) {mustBeInteger, mustBeNonnegative}
        options.Dimension (1,1) {mustBeInteger, mustBePositive} = find(size(A) ~= 1, 1)
        options.FillValue (1,1) = 0
        options.Pattern (1,1) string = "constant"
    end
    if ~any(options.Pattern == ["constant", "edge"])
        error("paddata shim: only Pattern=""constant"" or ""edge"" is supported.");
    end
    d = options.Dimension;
    n = size(A, d);
    if n >= m
        B = A;
        return
    end
    sz = size(A);
    sz(end+1:d) = 1;
    padSize = sz;
    padSize(d) = m - n;
    if options.Pattern == "edge" && n > 0
        idx = repmat({':'}, 1, numel(sz));
        idx{d} = n;
        reps = ones(1, numel(sz));
        reps(d) = m - n;
        P = repmat(A(idx{:}), reps);
    else
        P = repmat(cast(options.FillValue, "like", A), padSize);
    end
    B = cat(d, A, P);
end
