function initWorkspace()
% initWorkspace  Set up the MATLAB workspace relative to this file.
    thisfile = which(mfilename);
    parentFolder = fileparts(thisfile);
    cd(parentFolder);
    addpath(genpath(parentFolder));
    fprintf('### Path has been set for this MATLAB session.\n\n');
end