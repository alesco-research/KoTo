function initWorkspace()
% initWorkspace  Set up the MATLAB workspace relative to this file.
%   Adds all project folders to the path. On MATLAB releases older than
%   R2026a, missing functions are added from the compat folder.
%   Missing toolboxes are reported but only needed by some scripts.
    thisfile = which(mfilename);
    parentFolder = fileparts(thisfile);

    % Remove other copies from the path
    copies = string(which("initWorkspace", "-all"));
    for c = copies(:)'
        otherFolder = fileparts(c);
        if ~strcmp(otherFolder, parentFolder) && isfolder(otherFolder)
            warning("off", "MATLAB:rmpath:DirNotFound");
            rmpath(genpath(otherFolder));
            fprintf('### Removed another copy of the code from the path: %s\n', otherFolder);
        end
    end

    cd(parentFolder);
    addpath(genpath(parentFolder));

    % Compatibility shims: only add those whose built-in is missing
    compatFolder = fullfile(parentFolder, "utils", "shims");
    rmpath(genpath(compatFolder));
    shims = ["mustBeBetween", "mustBeMatrix", "clip", "paddata", "linestyleorder"];
    added = strings(1, 0);
    for s = shims
        if ~exist(char(s)) %#ok<EXIST>
            addpath(fullfile(compatFolder, "shim_" + s));
            added(end+1) = s; %#ok<AGROW>
        end
    end
    fprintf('### Path has been set for this MATLAB session (%s).\n', version('-release'));
    if ~isempty(added)
        fprintf('### Compatibility shims loaded for: %s\n', strjoin(added, ', '));
    end

    % Toolboxes used by some scripts only
    toolboxes = [ ...
        "lsqnonlin", "Optimization Toolbox", "identifyTCLab (NLSF model)"; ...
        "kmeans", "Statistics and Machine Learning Toolbox", "kernel lifts (rbf/rq/matern) and multi-step windows"];
    for k = 1:size(toolboxes, 1)
        if ~exist(char(toolboxes(k, 1))) %#ok<EXIST>
            fprintf('### Note: %s not found (%s) - needed for %s.\n', ...
                toolboxes(k, 1), toolboxes(k, 2), toolboxes(k, 3));
        end
    end
    if isMATLABReleaseOlderThan("R2022b")
        warning("initWorkspace:OldRelease", ...
            "This code uses argument validation features from R2022b, " + ...
            "release %s may not be supported.", version("-release"));
    end
    fprintf('\n');
end
