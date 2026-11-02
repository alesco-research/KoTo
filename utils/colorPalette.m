function colors = colorPalette(name)
%COLORPALETTE Return a named color palette, with a fallback for old releases.
%   colors = colorPalette("gem12") returns "gem12" on R2023b and newer
%   (where colororder accepts palette names) and lines(12) otherwise.
    arguments
        name (1,1) string = "gem12"
    end
    if isMATLABReleaseOlderThan("R2023b")
        n = str2double(extract(name, digitsPattern));
        if isempty(n) || isnan(n); n = 7; end
        colors = lines(n);
    else
        colors = name;
    end
end
