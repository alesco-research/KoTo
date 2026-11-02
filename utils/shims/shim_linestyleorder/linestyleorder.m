function linestyleorder(target, styles)
%LINESTYLEORDER Compatibility shim for MATLAB releases before R2023a.
%   Sets the line style order of a figure (its existing axes and the
%   default for new axes) or of a single axes. The style names
%   "mixedstyles"/"mixedmarkers" map to {'-', '--', ':', '-.'}.
%   Loaded by initWorkspace only when the built-in is missing.
    if (isstring(styles) || ischar(styles)) && any(strcmpi(string(styles), ["mixedstyles", "mixedmarkers"]))
        styles = {'-', '--', ':', '-.'};
    end
    if isa(target, "matlab.ui.Figure")
        set(target, "DefaultAxesLineStyleOrder", styles);
        axs = findall(target, "Type", "axes");
        if ~isempty(axs)
            set(axs, "LineStyleOrder", styles);
        end
    else
        set(target, "LineStyleOrder", styles);
    end
end
