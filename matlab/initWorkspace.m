function initWorkspace()
% initWorkspace  Set up the MATLAB workspace relative to this file.
    thisfile = which(mfilename);
    parentFolder = fileparts(thisfile);
    grandParentFolder = fileparts(parentFolder);
    cd(grandParentFolder);
    addpath(genpath(grandParentFolder));
    fprintf('### Path has been set for this MATLAB session.\n\n');
end