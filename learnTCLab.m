%% Learn Koopman Operator for a DC-tcLab
clc, clearvars, close all;

%% Set parameters
rng(42); % for reproducibility
numberOfSamples = 2000;
nuberOfTrajectorySamples = 6;

%% Instantiate model, scaling, lifting and predictor
tcLab = TCLab();
stateScaling = Scale([276, 343; 276, 343]); % x1 = [276, 343] K, x2 = [276, 343] K
inputScaling = Scale([0, 1; 0, 1]);         % u1 = [0, 1] -, u2 = [0, 1] -

kernelType = "poly";
switch kernelType
    case "rbf"
        stateLifting = @(centers) GaussianLift(tcLab.stateDim, 10, centers);
        numberOfCenters = 300;
    case "rq"
        stateLifting = @(centers) RationalQuadraticLift(tcLab.stateDim, 10, 2.0, centers);
        numberOfCenters = 300;
    case "matern"
        stateLifting = @(centers) Matern52Lift(tcLab.stateDim, 10, centers);
        numberOfCenters = 300;
    case "poly"
        stateLifting = @(~) PolynomialLift(tcLab.stateDim, 3);
end
inputLifting = PolynomialLift(tcLab.inputDim, 5);

switch "multi"
    case "one"
        getPairs = @(Z, U, V) getOneStepPairs(Z, U, V, 20*800);
    case "multi"
        getPairs = @(Z, U, V) getMultiStepWindows(Z, U, V, horizon=20, maxNumberOfWindows=800, maxNumberOfRawWindows=6000);
end

% KCF coupling paremeters
kcfCouplingDegree = 2;  % <= polynomial lift degree
kcfCouplingRatio = 2;   % > 1

savePath = fullfile( ...
    fileparts(mfilename("fullpath")), "./tcLab_" + kernelType + ".mat" ...
);

%% Sample tcLab
fprintf('\n[1] Generating training trajectories...\n');
XTrajs = cell(nuberOfTrajectorySamples,1);
UTrajs = cell(nuberOfTrajectorySamples,1);

xRange = [276, 343; 276, 343]; % physical operating ranges
t  = (0:numberOfSamples-1)' * tcLab.Ts; % time vector
freqs  = [150, 350, 55, 95, 155, 265]; % driving input frequencies

for l = 1:nuberOfTrajectorySamples
    x0 = xRange(:,1) + diff(xRange, 1, 2) .* rand(2,1);
    ut = [
        1 + sum(sin(freqs .* t + 2 * pi * rand(1,6)), 2)';
        1 + sum(sin(freqs .* t + 2 * pi * rand(1,6)), 2)'
    ] * 0.25;
    ut = clip(ut, 0, 1);

    xt = tcLab.simulate(x0, ut);

    XTrajs{l} = stateScaling.scale(xt);
    UTrajs{l} = inputScaling.scale(ut);    
end

% Stack trajectories
X = cell2mat(cellfun(@(x) x(:, 1:end-1), XTrajs, UniformOutput=false)');
U = cell2mat(UTrajs');

fprintf( ...
    '\tNumber of trajectories: %d, total number of samples: %d\n', ...
    nuberOfTrajectorySamples, ...
    size(X, 2) ...
);

%% Lift data
fprintf('\n[2] Lifting training data...\n');
centers = [];
if any(kernelType == ["rbf", "rq", "matern"])
    [~, centers] = kmeans(X', numberOfCenters, MaxIter=300, Replicates=3);
end
stateLifting = stateLifting(centers');

ZTrajs = cell(nuberOfTrajectorySamples, 1);
VTrajs = cell(nuberOfTrajectorySamples, 1);
for l = 1:nuberOfTrajectorySamples
    ZTrajs{l} = stateLifting.lift(XTrajs{l});
    VTrajs{l} = inputLifting.lift(UTrajs{l});
end

% Stack trajectories
Z = cell2mat(cellfun(@(z) z(:, 1:end-1), ZTrajs, UniformOutput=false)');
V = cell2mat(VTrajs');

reprojectionError = stateLifting.fitProjection(X, gamma=1e-4);
fprintf('\tReprojection Error: %.4f\n', reprojectionError);

%% Fit Koopman controllers
fprintf('\n[3] Fit Koopman controllers...\n');
[Ypairs, Zpairs, Upairs, Vpairs] = getPairs(ZTrajs, UTrajs, VTrajs);

linearPredictor = LinearKoopman(Ypairs, Zpairs, Upairs, gamma=1E-4);
bilinearPredictor = BilinearKoopman(Ypairs, Zpairs, Upairs, gamma=1E-4);
geKoPredictor = GeKoKoopman(Ypairs, Zpairs, Vpairs, gamma=1E-4);

if kernelType == "poly"
    % Get polynominal coupling
    kcfCouplingIdxs = 1:nchoosek(tcLab.stateDim + kcfCouplingDegree, kcfCouplingDegree);
else
    % Get variance sorted coupling from coupling ratio
    if stateLifting.includeState
        [~, ordering] = sort(var(Z(1 + stateLifting.dim:end, :), 0, 2), "descend");
        kcfOrdering = [1, 2, 3, 3 + ordering];
    else
        [~, kcfOrdering] = sort(var(Z, 0, 2), 'descend');
    end
    effectiveInputs = sum(var(V, 0, 2) > 1E-10);
    numberOfCouplings = round((kcfCouplingRatio - 1) * stateLifting.featureDim / effectiveInputs);
    numberOfCouplings = clip(numberOfCouplings, 1, stateLifting.featureDim);
    kcfCouplingIdxs = sort(kcfOrdering(1:numberOfCouplings))';
end

kcfPredictor = KCFKoopman(Ypairs, Zpairs, Vpairs, couplingIdxs=kcfCouplingIdxs, gamma=1E-4);

%% Evaluate operator norms
linearPredictor.getOperatorNorm();
bilinearPredictor.getOperatorNorm();
geKoPredictor.getOperatorNorm();
kcfPredictor.getOperatorNorm();

%% Save trained models
fprintf('\n[10] Saving trained models...\n');
save(savePath, ...
    "tcLab", ...
    "stateScaling", "inputScaling", ...
    "stateLifting", "inputLifting", ...
    "linearPredictor", "bilinearPredictor", "geKoPredictor", "kcfPredictor" ...
);
fprintf('\tSaved to %s\n', savePath);