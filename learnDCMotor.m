%% Learn Koopman Operator for a DC-Motor
% clc, clearvars, close all;

%% Set parameters
rng(42); % for reproducibility
numberOfSamples = 2000;
nuberOfTrajectorySamples = 6;

%% Instantiate model, scaling, lifting and predictor
motor = DCMotor();
stateScaling = Scale([0, 10; 0, 250]); % x1 = [0, 10] A, x2 = [0, 250] rad/s
inputScaling = Scale([0, 4]);          % u = [0, 4] wihtout units

kernelType = "poly";
switch kernelType
    case "rbf"
        stateLifting = @(centers) GaussianLift(motor.stateDim, 0.5, centers);
        numberOfCenters = 300;
    case "rq"
        stateLifting = @(centers) RationalQuadraticLift(motor.stateDim, 0.5, 2.0, centers);
        numberOfCenters = 300;
    case "matern"
        stateLifting = @(centers) Matern52Lift(motor.stateDim, 0.5, centers);
        numberOfCenters = 300;
    case "poly"
        stateLifting = @(~) PolynomialLift(motor.stateDim, 3);
end
inputLifting = ChebyshevLift(8);

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
    fileparts(mfilename("fullpath")), "./dcMotor_" + kernelType + ".mat" ...
);

%% Sample motor
fprintf("\n[1] Generating training trajectories...\n");
XTrajs = cell(nuberOfTrajectorySamples,1);
UTrajs = cell(nuberOfTrajectorySamples,1);

xRange = [-5, 5; -80, 20]; % physical operating ranges
t  = (0:numberOfSamples-1)' * motor.Ts; % time vector
freqs  = [1.5, 3.5, 5.5, 9.5, 15.5, 26.5]; % driving input frequencies

for l = 1:nuberOfTrajectorySamples
    x0 = xRange(:,1) + diff(xRange, 1, 2) .* rand(2,1);
    ut = sum(0.8 * sin(freqs .* t + 2 * pi * rand(1,6)), 2)';
    ut = clip(ut, -4, 4);

    xt = motor.simulate(x0, ut);

    XTrajs{l} = stateScaling.scale(xt);
    UTrajs{l} = inputScaling.scale(ut);    
end

% Stack trajectories
X = cell2mat(cellfun(@(x) x(:, 1:end-1), XTrajs, UniformOutput=false)');
U = cell2mat(UTrajs');

fprintf( ...
    "\tNumber of trajectories: %d, total number of samples: %d\n", ...
    nuberOfTrajectorySamples, ...
    size(X, 2) ...
);

%% Lift data
fprintf("\n[2] Lifting training data...\n");
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
fprintf("\tReprojection Error: %.4f\n", reprojectionError);

%% Fit Koopman controllers
fprintf("\n[3] Fit Koopman controllers...\n");
[Ypairs, Zpairs, Upairs, Vpairs] = getPairs(ZTrajs, UTrajs, VTrajs);

predictors = cell(4, 1);
predictors{1} = LinearKoopman(Ypairs, Zpairs, Upairs, gamma=1E-4);
predictors{2} = BilinearKoopman(Ypairs, Zpairs, Upairs, gamma=1E-4);
predictors{3} = GeKoKoopman(Ypairs, Zpairs, Vpairs, gamma=1E-4);

if kernelType == "poly"
    % Get polynominal coupling
    kcfCouplingIdxs = 1:nchoosek(motor.stateDim + kcfCouplingDegree, kcfCouplingDegree);
else
    % Get variance sorted coupling from coupling ratio
    if stateLifting.includeState
        [~, ordering] = sort(var(Z(1 + stateLifting.dim:end, :), 0, 2), "descend");
        kcfOrdering = [1, 2, 3, 3 + ordering];
    else
        [~, kcfOrdering] = sort(var(Z, 0, 2), "descend");
    end
    effectiveInputs = sum(var(V, 0, 2) > 1E-10);
    numberOfCouplings = round((kcfCouplingRatio - 1) * stateLifting.featureDim / effectiveInputs);
    numberOfCouplings = clip(numberOfCouplings, 1, stateLifting.featureDim);
    kcfCouplingIdxs = sort(kcfOrdering(1:numberOfCouplings))';
end

predictors{4} = KCFKoopman(Ypairs, Zpairs, Vpairs, couplingIdxs=kcfCouplingIdxs, gamma=1E-4);

if exist("M_ms", "var")
    if all([
        all(predictors{1}.A == M_ms{1}.A, "all");
        all(predictors{1}.B == M_ms{1}.B, "all");
        all(predictors{2}.A == M_ms{2}.A, "all");
        all(predictors{2}.B == M_ms{2}.B, "all");
        all(predictors{2}.N == M_ms{2}.N, "all");
        all(predictors{3}.K == M_ms{3}.K, "all");
        all(predictors{4}.A11 == M_ms{4}.A11, "all");
        all(predictors{4}.A12 == M_ms{4}.A12, "all");
    ])
        fprintf("\tOperators match Lazar's operators.\n\n");
    else
        fprintf("\tOperators do NOT match Lazar's operators.\n\n");
    end
end

%% Evaluate operator norms
fprintf("\t%s", evalc("predictors{1}.getOperatorNorm();"))
fprintf("\t%s", evalc("predictors{2}.getOperatorNorm();"))
fprintf("\t%s", evalc("predictors{3}.getOperatorNorm();"))
fprintf("\t%s", evalc("predictors{4}.getOperatorNorm();"))

%% Evaluate prediction error
fprintf("\n[4] Evaluating prediction error...\n");
testHorizon = 300;
numberOfTestRuns = 10;
testInitialState = [3; -40];
randperm(size(Z, 2), size(Z, 2)); % match the random number generator state

testTime = (0:testHorizon - 1) * motor.Ts;
testFrequencies = [2, 5, 11.5, 23.5];
testPhases = 2 * pi * rand(1, numel(testFrequencies));
testInput = sum(0.8 * sin(testFrequencies .* testTime' + testPhases), 2)';
testInput = clip(testInput, -4, 4);

rngState = rng; rng(2025, "twister");
testInitialStates = [
    testInitialState, xRange(:, 1) + diff(xRange, 1, 2) .* rand(numberOfTestRuns - 1, 2)';
];
rng(rngState);

predictorNames = ["Linear"; "Bilinear"; "GeKo"; "KCF"];
testInputScaled = inputScaling.scale(testInput);
testInputLifted = inputLifting.lift(testInputScaled);
testInputs = {testInputScaled, testInputScaled, testInputLifted, testInputLifted};

meanStateErrors = zeros(numberOfTestRuns, numel(predictors));
for i = 1:numberOfTestRuns
    testStates = motor.simulate(testInitialStates(:, i), testInput);
    
    testStatesScaled = stateScaling.scale(testStates);
    trueLiftedStates = stateLifting.lift(testStatesScaled);

    for j = 1:numel(predictors)
        predictedLiftedStates = predictors{j}.rollout( ...
            trueLiftedStates(:, 1), testInputs{j} ...
        );
        predictedStates = stateLifting.project(predictedLiftedStates);
        stateErrors = sqrt(sum((predictedStates - testStatesScaled).^2, 1));
        meanStateErrors(i, j) = mean(stateErrors);
    end
end

fprintf("\tPer-trajectory mean error (scaled state)\n\tNr.");
fprintf("  %12s", predictorNames);
fprintf("\n");
for i = 1:numberOfTestRuns
    fprintf("\t%-3d", i);
    fprintf("  %12.4e", meanStateErrors(i, :));
    fprintf("\n");
end

%% Save trained models
fprintf("\n[6] Saving trained models...\n");
save(savePath, ...
    "motor", ...
    "stateScaling", "inputScaling", ...
    "stateLifting", "inputLifting", ...
    "predictors" ...
);
fprintf("\tSaved to %s\n", savePath);