function styles = lineStyles(name)
%LINESTYLES Return a named line style order, with a fallback for old releases.
%   styles = lineStyles("mixedstyles") returns the name on R2023a and newer
%   and the cell {'-', '--', ':', '-.'} otherwise.
    arguments
        name (1,1) string = "mixedstyles"
    end
    if isMATLABReleaseOlderThan("R2023a")
        styles = {'-', '--', ':', '-.'};
    else
        styles = name;
    end
end
