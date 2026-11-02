function mustBeBetween(A, lowerBound, upperBound, varargin)
%MUSTBEBETWEEN Compatibility shim for MATLAB releases before R2025a.
%   mustBeBetween(A, lower, upper) errors unless lower <= A <= upper.
%   Optional bound flags are passed to mustBeInRange (R2020b+); the
%   flags "open"/"closed"/"openleft"/"openright" are mapped as well.
%   Loaded by initWorkspace only when the built-in is missing.
    map = struct(open="exclusive", closed="inclusive", ...
        openleft="exclude-lower", openright="exclude-upper");
    flags = string(varargin);
    for k = 1:numel(flags)
        if isfield(map, lower(flags(k)))
            flags(k) = map.(lower(flags(k)));
        end
    end
    try
        flagArgs = cellstr(flags);
        mustBeInRange(A, lowerBound, upperBound, flagArgs{:});
    catch e
        throwAsCaller(MException("MATLAB:validators:mustBeBetween", "%s", e.message));
    end
end
