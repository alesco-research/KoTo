function y = clip(x, lower, upper)
%CLIP Compatibility shim for MATLAB releases before R2024a.
%   y = clip(x, lower, upper) limits x to [lower, upper]; NaN is kept.
%   Loaded by initWorkspace only when the built-in is missing.
    y = min(max(x, lower), upper);
    y(isnan(x)) = NaN;
end
