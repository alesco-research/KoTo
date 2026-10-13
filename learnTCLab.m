%% Learn Koopman Operator for a TC-Lab
clc, clearvars, close all;
initWorkspace;
%#ok<*UNRCH>

%% Set parameters
rng(42); % for reproducibility
savePlots = false;
saveModels = false;
plotTrainData = true;
plotTestData = true;

%% Instantiate model, scaling, lifting and predictor
[stateDim, inputDim] = deal(2, 2);
stateScaling = Scale([23, 50; 23, 50]); % [x1] = °C, [x2] = °C
inputScaling = Scale([0, 50; 0, 50]);   % [u1] = %,  [u2] = %

kernelType = "poly";
switch kernelType
    case "rbf"
        stateLifting = @(centers) GaussianLift(stateDim, 0.5, centers);
        numberOfCenters = 300;
    case "rq"
        stateLifting = @(centers) RationalQuadraticLift(stateDim, 1, 5, centers);
        numberOfCenters = 300;
    case "matern"
        stateLifting = @(centers) Matern52Lift(stateDim, 1, centers);
        numberOfCenters = 300;
    case "poly"
        stateLifting = @(~) PolynomialLift(stateDim, 3);
end
inputLifting = PolynomialLift(inputDim, 3);

switch "multi"
    case "one"
        getPairs = @(Z, U, V) getOneStepPairs(Z, U, V, maxNumberOfPairs=1000);
    case "multi"
        getPairs = @(Z, U, V) getMultiStepWindows(Z, U, V, horizon=20, maxNumberOfWindows=200);
end

% KCF coupling parameters
kcfCouplingRatio = 2;   % > 1
kcfCouplingDegree = 2;  % <= polynomial lift degree

scriptDir = fileparts(which("learnTCLab"));
dataDir = fullfile(scriptDir, "data");

trainPath = fullfile(dataDir, "TCLabTrain1.mat");
testPath = fullfile(dataDir, "TCLabTest1.mat");
savePath = fullfile(dataDir, "TCLab" + upper(kernelType) + ".mat");

%% Load data
fprintf("\n[1] Load training trajectories...\n");
load(trainPath, "XTrain", "UTrain", "Ts")
numberOfTrajectorySamples = numel(XTrain);
TsTrain = Ts;

% Scale trajectories
XTrainScaled = cellfun(@stateScaling.scale, XTrain, UniformOutput=false);
UTrainScaled = cellfun(@inputScaling.scale, UTrain, UniformOutput=false);

% Stack trajectories
X = cell2mat(cellfun(@(x) x(:, 1:end-1), XTrainScaled, UniformOutput=false)');
U = cell2mat(UTrainScaled');

fprintf( ...
    "\tNumber of trajectories: %d, total number of samples: %d\n", ...
    numberOfTrajectorySamples, ...
    size(X, 2) ...
);

if plotTrainData
    plotTrajectories( ...
        XTrain, UTrain, Ts, name="Train Trajectories", ...
        stateLabels={
            ["Temp. Heater 1", "$[^\circ$C]"];
            ["Temp. Heater 2", "$[^\circ$C]"]
        }, ...
        inputLabels={
            ["Input Heater 1",  "[\%]"];
            ["Input Heater 2",  "[\%]"]
        } ...
    );
end

%% Lift data
fprintf("\n[2] Lifting training data...\n");
centers = [];
if any(kernelType == ["rbf", "rq", "matern"])
    [~, centers] = kmeans(X', numberOfCenters, MaxIter=300, Replicates=3);
end
stateLifting = stateLifting(centers');

% Lift trajectories
ZTrain = cellfun(@stateLifting.lift, XTrainScaled, UniformOutput=false);
VTrain = cellfun(@inputLifting.lift, UTrainScaled, UniformOutput=false);

% Stack trajectories
Z = cell2mat(cellfun(@(z) z(:, 1:end-1), ZTrain, UniformOutput=false)');
V = cell2mat(VTrain');

reprojectionError = stateLifting.fitProjection(X, gamma=1e-4);
fprintf("\tReprojection Error: %.4f\n", reprojectionError);

%% Fit Koopman controllers
fprintf("\n[3] Fit Koopman controllers...\n");
[Ypairs, Zpairs, Upairs, Vpairs] = getPairs(ZTrain, UTrainScaled, VTrain);
pairs = [
    cellfun(@(x) x(:,1:end-1), XTrainScaled(:)', UniformOutput=false);
    UTrainScaled(:)'
];

memoryPredictors = cell(1, 1);
memoryPredictorNames = "MOESP";
memoryPredictors{1} = MOESP(pairs{:}, L=100, stateDim = 4);

predictors = cell(4, 1);
predictorNames = ["Linear"; "Bilinear"; "GeKo"; "KCF"];
predictors{1} = LinearKoopman(Ypairs, Zpairs, Upairs, gamma=1E-1);
predictors{2} = BilinearKoopman(Ypairs, Zpairs, Upairs, gamma=1E-1);
predictors{3} = GeKoKoopman(Ypairs, Zpairs, Vpairs, gamma=1E-1);

if kernelType == "poly"
    % Get polynomial coupling
    kcfCouplingIdxs = 1:nchoosek(stateDim + kcfCouplingDegree, kcfCouplingDegree);
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

predictors{4} = KCFKoopman(Ypairs, Zpairs, Vpairs, couplingIdxs=kcfCouplingIdxs, gamma=1E-1);

%% Evaluate operator norms
fprintf("\t%s", evalc("predictors{1}.getOperatorNorm();"))
fprintf("\t%s", evalc("predictors{2}.getOperatorNorm();"))
fprintf("\t%s", evalc("predictors{3}.getOperatorNorm();"))
fprintf("\t%s", evalc("predictors{4}.getOperatorNorm();"))

%% Load test data
fprintf("\n[4] Load test data...\n");
load(testPath, "XTest", "UTest", "Ts")
numberOfTestRuns = numel(XTest);

if TsTrain ~= Ts
    error("Train and test sampling time mismatch.")
end

% Scale trajectories
XTestScaled = cellfun(@stateScaling.scale, XTest, UniformOutput=false)';
UTestScaled = cellfun(@inputScaling.scale, UTest, UniformOutput=false)';

% Lift trajectories
ZTest = cellfun(@stateLifting.lift, XTestScaled, UniformOutput=false);
VTest = cellfun(@inputLifting.lift, UTestScaled, UniformOutput=false);

fprintf("\tNumber of test trajectories: %d.", numberOfTestRuns);
if plotTestData
    plotTrajectories( ...
        XTest, UTest, Ts, name="Test Trajectories", ...
        stateLabels={
            ["Temp. Heater 1", "$[^\circ$C]"];
            ["Temp. Heater 2", "$[^\circ$C]"]
        }, ...
        inputLabels={
            ["Input Heater 1",  "[\%]"];
            ["Input Heater 2",  "[\%]"]
        } ...
    );
end

%% Evaluate prediction error
fprintf("\n[5] Evaluating prediction error...\n");
testHorizon = size(UTest{1}, 2);
testTime = (0:testHorizon) * Ts;
numberOfPredictors = numel(memoryPredictors) + numel(predictors);

testInputs = {UTestScaled, UTestScaled, VTest, VTest};

XPredicted = zeros(numberOfTestRuns, stateDim, testHorizon + 1, numberOfPredictors);
liftedStateError = zeros(numberOfTestRuns, testHorizon + 1, numel(predictors));
scaledStateError = zeros(numberOfTestRuns, testHorizon + 1, numberOfPredictors);
for i = 1:numberOfTestRuns
    % State-full predictors
    for j = 1:numel(memoryPredictors)
        m = memoryPredictors{j}.memory;
        uWarm = UTestScaled{i}(:, 1:m);
        yWarm = XTestScaled{i}(:, 1:m);
        uFuture = UTestScaled{i}(:, m+1:end);

        XPredictedScaled = nan(stateDim, testHorizon + 1);
        XPredictedScaled(:, m+1:end-1) = memoryPredictors{j}.forecast(uWarm, yWarm, uFuture);
        
        XPredicted(i, :, :, j) = stateScaling.unscale(XPredictedScaled);
        scaledStateError(i, :, j) = sqrt( ...
            sum((XPredictedScaled - XTestScaled{i}).^2, 1, "omitnan") ...
        );
    end
    % Regular predictors
    for k = 1:numel(predictors)
        ZPredicted = predictors{k}.rollout(ZTest{i}(:, 1), testInputs{k}{i});
        XPredictedScaled = stateLifting.project(ZPredicted);
        XPredicted(i, :, :, k+j) = stateScaling.unscale(XPredictedScaled);

        liftedStateError(i, :, k) = sqrt(sum((ZPredicted - ZTest{i}).^2, 1));
        scaledStateError(i, :, k+j) = sqrt( ...
            sum((XPredictedScaled - XTestScaled{i}).^2, 1) ...
        );
    end
end

meanLiftedStateErrors = mean(liftedStateError, 2);
meanScaledStateErrors = mean(scaledStateError, 2);

fprintf("\tPer-trajectory mean error (scaled state)\n\tNr.");
fprintf("  %12s", [memoryPredictorNames(:); predictorNames(:)]);
fprintf("\n");
for i = 1:numberOfTestRuns
    fprintf("\t%-3d", i);
    fprintf("  %12.4e", meanScaledStateErrors(i, :));
    fprintf("\n");
end

fprintf("\n[5] Plotting...\n");
margin = 0;
width = 12;
height = 8;
tickFontSize = 8;
labelFontSize = 10;
legendFontSize = 10;
markerIndices = unique(round(linspace(1, numel(testTime), 12)));
colorscheme = [
    0.00 0.45 0.74;
    0.00 0.60 0.20;
    0.85 0.10 0.10;
    0.75 0.00 0.75;
    0.00 1.00 1.00
];
set(groot, defaultTextInterpreter="latex");
set(groot, defaultLegendInterpreter="latex");
set(groot, defaultAxesTickLabelInterpreter="latex");

figure(Name="Feature-space Rollout Error"), hold on, grid on, colororder(colorscheme)
h = plot( ...
    testTime, squeeze(liftedStateError(1, :, :)), ...
    LineWidth=2, MarkerSize=5, MarkerIndices=markerIndices ...
);
set(gca,'YScale','log');
xlabel("Time [s]")
ylabel("$\left\|\hat{\bf{z}} - \bf{z}^\star\right\|$")
legend(predictorNames, Location="best")
if savePlots
    exportPlot(gcf(), 'data/feature_space_error.pdf', ...
        width=width, ...
        height=height, ...
        margin=margin, ...
        tickFontSize=tickFontSize, ...
        labelFontSize=labelFontSize ...
    );
end

figure(Name="State-space Rollout Error"), hold on, grid on, colororder(colorscheme)
h = plot( ...
    testTime, squeeze(scaledStateError(1, :, :)), ...
    LineWidth=2, MarkerSize=5, MarkerIndices=markerIndices ...
);
set(gca,'YScale','log');
xlabel("Time [s]")
ylabel("$\left\|\hat{\bf{x}} - \bf{x}^\star\right\|$")
legend([memoryPredictorNames(:); predictorNames(:)], Location="best")
if savePlots
    exportPlot(gcf(), 'data/state_space_error.pdf', ...
        width=width, ...
        height=height, ...
        margin=margin, ...
        tickFontSize=tickFontSize, ...
        labelFontSize=labelFontSize ...
    );
end

figure(Name="State Trajectories")
subplot(2, 1, 1), hold on, grid on, colororder(colorscheme)
h = plot( ...
    testTime, squeeze(XPredicted(1, 1, :, :)) - 273.15, ...
    LineWidth=1.5, MarkerSize=5, MarkerIndices=markerIndices ...
);
plot(testTime, squeeze(XTest{1}(1, :)) - 273.15, "k--", LineWidth=1.5)
ylabel("$x_1$ $[^\circ$C]")
legend([memoryPredictorNames(:); predictorNames(:); "True"], Location="best")

subplot(2, 1, 2), hold on, grid on, colororder(colorscheme)
h = plot( ...
    testTime, squeeze(XPredicted(1, 2, :, :)) - 273.15, ...
    LineWidth=1.5, MarkerSize=5, MarkerIndices=markerIndices ...
);
plot(testTime, squeeze(XTest{1}(2, :)) - 273.15, "k--", LineWidth=1.5)
xlabel("Time [s]")
ylabel("$x_2$ $[^\circ$C]")
legend([memoryPredictorNames(:); predictorNames(:); "True"], Location="best")
if savePlots
    exportPlot(gcf(), 'data/state_trajectories.pdf', ...
        width=width, ...
        height=height, ...
        margin=margin, ...
        tickFontSize=tickFontSize, ...
        labelFontSize=labelFontSize ...
    );
end

figure(Name="Phase Portrait"), hold on, grid on, colororder(colorscheme)
for i = 1:numberOfPredictors
    plot(squeeze(XPredicted(1, 1, :, i)) - 273.15, ...
        squeeze(XPredicted(1, 2, :, i)) - 273.15, ...
        LineWidth=1.5, MarkerSize=5, MarkerIndices=markerIndices);
end
plot(squeeze(XTest{1}(1, :)) - 273.15, squeeze(XTest{1}(2, :)) - 273.15, "k--", LineWidth=1.5)
plot(XTest{1}(1, 1) - 273.15, XTest{1}(2, 1) - 273.15, "ko", MarkerSize=10, MarkerFaceColor="y")
xlabel("$x_1$ $[^\circ$C]")
ylabel("$x_2$ $[^\circ$C]")
legend([memoryPredictorNames(:); predictorNames(:); "True"; "$x_0$"], Location="best")
if savePlots
    exportPlot(gcf(), 'data/phase_portrait.pdf', ...
        width=width, ...
        height=height, ...
        margin=margin, ...
        tickFontSize=tickFontSize, ...
        labelFontSize=labelFontSize ...
    );
end

%% Save trained models
if saveModels
    fprintf("\n[7] Saving trained models...\n");
    save(savePath, ...
        "stateScaling", "inputScaling", ...
        "stateLifting", "inputLifting", ...
        "predictors" ...
    );
    fprintf("\tSaved to %s\n", savePath);
end