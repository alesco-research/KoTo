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
fprintf("\n[1] Generating training trajectories...\n");
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
    kcfCouplingIdxs = 1:nchoosek(tcLab.stateDim + kcfCouplingDegree, kcfCouplingDegree);
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

%% Evaluate operator norms
fprintf("\t%s", evalc("predictors{1}.getOperatorNorm();"))
fprintf("\t%s", evalc("predictors{2}.getOperatorNorm();"))
fprintf("\t%s", evalc("predictors{3}.getOperatorNorm();"))
fprintf("\t%s", evalc("predictors{4}.getOperatorNorm();"))

%% Evaluate prediction error
fprintf("\n[4] Evaluating prediction error...\n");
testHorizon = 300;
numberOfTestRuns = 10;
testInitialState = [276; 343];

testTime = (0:testHorizon) * tcLab.Ts;
testFrequencies = [100, 125, 257, 90];
testPhases = 2 * pi * rand(1, numel(testFrequencies));
testInput = [
        1 + sum(sin(testFrequencies .* testTime(1:end - 1)' + testPhases), 2)';
        1 + sum(sin(testFrequencies .* testTime(1:end - 1)' + testPhases), 2)'
] * 0.25;
testInput = clip(testInput, 0, 1);

testInitialStates = [
    testInitialState, xRange(:, 1) + diff(xRange, 1, 2) .* rand(2, numberOfTestRuns - 1);
];

predictorNames = ["Linear"; "Bilinear"; "GeKo"; "KCF"];
testInputScaled = inputScaling.scale(testInput);
testInputLifted = inputLifting.lift(testInputScaled);
testInputs = {testInputScaled, testInputScaled, testInputLifted, testInputLifted};

testStates = zeros(numberOfTestRuns, tcLab.stateDim, testHorizon + 1);
predictedStates = zeros(numberOfTestRuns, tcLab.stateDim, testHorizon + 1, numel(predictors));
liftedStateError = zeros(numberOfTestRuns, testHorizon + 1, numel(predictors));
scaledStateError = zeros(numberOfTestRuns, testHorizon + 1, numel(predictors));
for i = 1:numberOfTestRuns
    testStates(i, :, :) = tcLab.simulate(testInitialStates(:, i), testInput);

    testStatesScaled = stateScaling.scale(testStates(i, :, :));
    trueLiftedStates = stateLifting.lift(testStatesScaled);

    for j = 1:numel(predictors)
        predictedLiftedStates = predictors{j}.rollout( ...
            trueLiftedStates(:, 1), testInputs{j} ...
        );
        predictedStatesScaled = stateLifting.project(predictedLiftedStates);
        predictedStates(i, :, :, j) = stateScaling.unscale(predictedStatesScaled);

        liftedStateError(i, :, j) = sqrt( ...
            sum((predictedLiftedStates - trueLiftedStates).^2, 1) ...
        );
        scaledStateError(i, :, j) = sqrt( ...
            sum((predictedStatesScaled - testStatesScaled).^2, 1) ...
        );
    end
end

meanLiftedStateErrors = mean(liftedStateError, 2);
meanScaledStateErrors = mean(scaledStateError, 2);

fprintf("\tPer-trajectory mean error (scaled state)\n\tNr.");
fprintf("  %12s", predictorNames);
fprintf("\n");
for i = 1:numberOfTestRuns
    fprintf("\t%-3d", i);
    fprintf("  %12.4e", meanScaledStateErrors(i, :));
    fprintf("\n");
end

%% Plot
fprintf("\n[5] Plotting...\n");
export = false;
margin = 0;
width = 12;
height = 8;
tickFontSize = 8;
labelFontSize = 10;
legendFontSize = 10;
markers = {'o', 's', '^', 'd'};
markerIndices = unique(round(linspace(1, numel(testTime), 12)));
colorscheme = [
    0.00 0.45 0.74;
    0.00 0.60 0.20;
    0.85 0.10 0.10;
    0.75 0.00 0.75
];

figure(Name="Feature-space Rollout Error"), hold on, grid on
h = plot( ...
    testTime, squeeze(liftedStateError(1, :, :)), ...
    LineWidth=2, MarkerSize=5, MarkerIndices=markerIndices ...
);
set(gca,'YScale','log');
set(h, {'Marker'}, markers(:));
xlabel("Time [s]", Interpreter="latex")
ylabel("$\left\|\hat{\bf{z}} - \bf{z}^\star\right\|$", Interpreter="latex")
legend(predictorNames, Location="best", Interpreter="latex")
colororder(colorscheme)
if export
    exportPlot(gcf(), 'data/feature_space_error.pdf', ...
        width=width, ...
        height=height, ...
        margin=margin, ...
        tickFontSize=tickFontSize, ...
        labelFontSize=labelFontSize ...
    );
end

figure(Name="State-space Rollout Error"), hold on, grid on
h = plot( ...
    testTime, squeeze(scaledStateError(1, :, :)), ...
    LineWidth=2, MarkerSize=5, MarkerIndices=markerIndices ...
);
set(gca,'YScale','log');
set(h, {'Marker'}, markers(:));
xlabel("Time [s]", Interpreter="latex")
ylabel("$\left\|\hat{\bf{x}} - \bf{x}^\star\right\|$", Interpreter="latex")
legend(predictorNames, Location="best", Interpreter="latex")
colororder(colorscheme)
if export
    exportPlot(gcf(), 'data/state_space_error.pdf', ...
        width=width, ...
        height=height, ...
        margin=margin, ...
        tickFontSize=tickFontSize, ...
        labelFontSize=labelFontSize ...
    );
end

figure(Name="State Trajectories")
subplot(2, 1, 1), hold on, grid on
plot(testTime, squeeze(testStates(1, 1, :)) - 273.15, "k-", LineWidth=2.5)
set(gca, "ColorOrderIndex", 1); colororder(colorscheme)
h = plot( ...
    testTime, squeeze(predictedStates(1, 1, :, :)) - 273.15, ...
    LineWidth=1.5, MarkerSize=5, MarkerIndices=markerIndices ...
);
set(h, {'Marker'}, markers(:));
ylabel("$x_1$ $[^\circ$C]", Interpreter="latex")
legend(["True"; predictorNames], Location="best", Interpreter="latex")

subplot(2, 1, 2), hold on, grid on
plot(testTime, squeeze(testStates(1, 2, :)) - 273.15, "k-", LineWidth=2.5)
set(gca, "ColorOrderIndex", 1); colororder(colorscheme)
h = plot( ...
    testTime, squeeze(predictedStates(1, 2, :, :)) - 273.15, ...
    LineWidth=1.5, MarkerSize=5, MarkerIndices=markerIndices ...
);
set(h, {'Marker'}, markers(:));
xlabel("Time [s]", Interpreter="latex")
ylabel("$x_2$ $[^\circ$C]", Interpreter="latex")
legend(["True"; predictorNames], Location="best", Interpreter="latex")
if export
    exportPlot(gcf(), 'data/state_trajectories.pdf', ...
        width=width, ...
        height=height, ...
        margin=margin, ...
        tickFontSize=tickFontSize, ...
        labelFontSize=labelFontSize ...
    );
end

figure(Name="Phase Portrait"), hold on, grid on
plot(squeeze(testStates(1, 1, :)) - 273.15, squeeze(testStates(1, 2, :)) - 273.15, "k-", LineWidth=2.5)
set(gca, "ColorOrderIndex", 1); colororder(colorscheme)
for i = 1:numel(predictors)
    plot(squeeze(predictedStates(1, 1, :, i)) - 273.15, ...
        squeeze(predictedStates(1, 2, :, i)) - 273.15, ...
        LineWidth=1.5, MarkerSize=5, Marker=markers{i}, MarkerIndices=markerIndices);
end
plot(testStates(1, 1, 1) - 273.15, testStates(1, 2, 2) - 273.15, "ks", MarkerSize=10, MarkerFaceColor="y")
xlabel("$x_1$ $[^\circ$C]", Interpreter="latex")
ylabel("$x_2$ $[^\circ$C]", Interpreter="latex")
legend(["True"; predictorNames; "$x_0$"], Location="best", Interpreter="latex")
if export
    exportPlot(gcf(), 'data/phase_portrait.pdf', ...
        width=width, ...
        height=height, ...
        margin=margin, ...
        tickFontSize=tickFontSize, ...
        labelFontSize=labelFontSize ...
    );
end

%% Save trained models
fprintf("\n[10] Saving trained models...\n");
save(savePath, ...
    "tcLab", ...
    "stateScaling", "inputScaling", ...
    "stateLifting", "inputLifting", ...
    "predictors" ...
);
fprintf("\tSaved to %s\n", savePath);