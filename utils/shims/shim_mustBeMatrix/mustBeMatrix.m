function mustBeMatrix(A)
%MUSTBEMATRIX Compatibility shim for MATLAB releases before R2024b.
%   Loaded by initWorkspace only when the built-in is missing.
    if ~ismatrix(A)
        throwAsCaller(MException("MATLAB:validators:mustBeMatrix", ...
            "Value must be a matrix."));
    end
end
