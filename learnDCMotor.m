%% Learn Koopman Operator for a DC-Motor
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
motor = DCMotor();
stateScaling = Scale([0, 10; 0, 250]); % [x1 = A, [x2] = rad/s
inputScaling = Scale([0, 4]);          % [u] = -

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
        getPairs = @(Z, U, V) getOneStepPairs(Z, U, V, maxNumberOfPairs=20*800);
    case "multi"
        getPairs = @(Z, U, V) getMultiStepWindows(Z, U, V, horizon=20, maxNumberOfWindows=800, maxNumberOfRawWindows=6000);
end

% KCF coupling parameters
kcfCouplingRatio = 2;   % > 1
kcfCouplingDegree = 2;  % <= polynomial lift degree

savePath = fullfile( ...
    fileparts(which("learnDCMotor")), "./data/dcMotor" + upper(kernelType) + ".mat" ...
);

%% Sample motor
fprintf("\n[1] Generating training trajectories...\n");

trainHorizon = 2000;
numberOfTrajectorySamples = 6;
XTrain = cell(numberOfTrajectorySamples,1);
UTrain = cell(numberOfTrajectorySamples,1);

xRange = [-5, 5; -80, 20]; % physical operating ranges
trainTime = (0:trainHorizon-1)' * motor.Ts; % time vector
trainFreqs = [1.5, 3.5, 5.5, 9.5, 15.5, 26.5]; % driving input frequencies

for l = 1:numberOfTrajectorySamples
    x0 = xRange(:,1) + diff(xRange, 1, 2) .* rand(2,1);
    uTrain = sum(0.8 * sin(trainFreqs .* trainTime + 2 * pi * rand(1,6)), 2)';
    UTrain{l} = clip(uTrain, -4, 4);
    XTrain{l} = motor.simulate(x0, UTrain{l});
end

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
        XTrain, UTrain, name="Train Trajectories", ...
        stateLabels={
            ["Current $I$", "[A]"];
            ["Angular Velocity $\omega$", "[rad / s]"]
        }, ...
        inputLabels={["Input ", "[\%]"]} ...
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

predictors = cell(4, 1);
predictorNames = ["Linear"; "Bilinear"; "GeKo"; "KCF"];
predictors{1} = LinearKoopman(Ypairs, Zpairs, Upairs, gamma=1E-4);
predictors{2} = BilinearKoopman(Ypairs, Zpairs, Upairs, gamma=1E-4);
predictors{3} = GeKoKoopman(Ypairs, Zpairs, Vpairs, gamma=1E-4);

if kernelType == "poly"
    % Get polynomial coupling
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
randperm(size(Z, 2), size(Z, 2)); % to reproduce Lazar's rng state

testTime = (0:testHorizon) * motor.Ts;
testFreqs = [2, 5, 11.5, 23.5];
testPhases = 2 * pi * rand(1, numel(testFreqs));
UTest = sum(0.8 * sin(testFreqs .* testTime(1:end - 1)' + testPhases), 2)';
UTest = clip(UTest, -4, 4);

UTestScaled = inputScaling.scale(UTest);
VTest = inputLifting.lift(UTestScaled);
testInputs = {UTestScaled, UTestScaled, VTest, VTest};

rngState = rng; rng(2025, "twister");
x0Test = [
    [3; -40], xRange(:, 1) + diff(xRange, 1, 2) .* rand(numberOfTestRuns - 1, 2)';
]; rng(rngState);

XTest = cell(numberOfTestRuns, 1);
XPredicted = zeros(numberOfTestRuns, motor.stateDim, testHorizon + 1, numel(predictors));
liftedStateError = zeros(numberOfTestRuns, testHorizon + 1, numel(predictors));
scaledStateError = zeros(numberOfTestRuns, testHorizon + 1, numel(predictors));
for i = 1:numberOfTestRuns
    XTest{i} = motor.simulate(x0Test(:, i), UTest);

    XTestScaled = stateScaling.scale(XTest{i});
    ZTest = stateLifting.lift(XTestScaled);

    for j = 1:numel(predictors)
        ZPredicted = predictors{j}.rollout(ZTest(:, 1), testInputs{j});
        XPredictedScaled = stateLifting.project(ZPredicted);
        XPredicted(i, :, :, j) = stateScaling.unscale(XPredictedScaled);

        liftedStateError(i, :, j) = sqrt(sum((ZPredicted - ZTest).^2, 1));
        scaledStateError(i, :, j) = sqrt( ...
            sum((XPredictedScaled - XTestScaled).^2, 1) ...
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

if plotTestData
    plotTrajectories( ...
        XTest, cellfun(@(x) UTest, XTest, UniformOutput=false), ...
        name="Test Trajectories", ...
        stateLabels={
            ["Current $I$", "[A]"];
            ["Angular Velocity $\omega$", "[rad / s]"]
        }, ...
        inputLabels={["Input ", "[\%]"]} ...
    );
end

%% Plot
fprintf("\n[5] Plotting...\n");
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
set(groot, defaultTextInterpreter="latex");
set(groot, defaultLegendInterpreter="latex");
set(groot, defaultAxesTickLabelInterpreter="latex");

figure(Name="Feature-space Rollout Error"), hold on, grid on, colororder(colorscheme)
h = plot( ...
    testTime, squeeze(liftedStateError(1, :, :)), ...
    LineWidth=2, MarkerSize=5, MarkerIndices=markerIndices ...
);
set(gca,'YScale','log');
set(h, {'Marker'}, markers(:));
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
set(h, {'Marker'}, markers(:));
xlabel("Time [s]")
ylabel("$\left\|\hat{\bf{x}} - \bf{x}^\star\right\|$")
legend(predictorNames, Location="best")
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
    testTime, squeeze(XPredicted(1, 1, :, :)), ...
    LineWidth=1.5, MarkerSize=5, MarkerIndices=markerIndices ...
);
set(h, {'Marker'}, markers(:));
plot(testTime, squeeze(XTest{1}(1, :)), "k--", LineWidth=1.5)
ylabel("$x_1$ [A]")
legend([predictorNames(:); "True"], Location="best")

subplot(2, 1, 2), hold on, grid on, colororder(colorscheme)
h = plot( ...
    testTime, squeeze(XPredicted(1, 2, :, :)), ...
    LineWidth=1.5, MarkerSize=5, MarkerIndices=markerIndices ...
);
set(h, {'Marker'}, markers(:));
plot(testTime, squeeze(XTest{1}(2, :)), "k--", LineWidth=1.5)
xlabel("Time [s]")
ylabel("$x_2$ [rad / s]")
legend([predictorNames(:); "True"], Location="best")
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
for i = 1:numel(predictors)
    plot(squeeze(XPredicted(1, 1, :, i)), squeeze(XPredicted(1, 2, :, i)), ...
        LineWidth=1.5, MarkerSize=5, Marker=markers{i}, MarkerIndices=markerIndices);
end
plot(squeeze(XTest{1}(1, :)), squeeze(XTest{1}(2, :)), "k--", LineWidth=1.5)
plot(XTest{1}(1, 1), XTest{1}(2, 1), "ko", MarkerSize=10, MarkerFaceColor="y")
xlabel("$x_1$ [A]")
ylabel("$x_2$ [rad / s]")
legend([predictorNames(:); "True"; "$x_0$"], Location="best")
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
    fprintf("\n[6] Saving trained models...\n");
    save(savePath, ...
        "motor", ...
        "stateScaling", "inputScaling", ...
        "stateLifting", "inputLifting", ...
        "predictors" ...
    );
    fprintf("\tSaved to %s\n", savePath);
end