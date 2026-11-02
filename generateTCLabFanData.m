function generateTCLabFanData(options)
%GENERATETCLABFANDATA Simulate training and test data for the TC-Lab with fan.
%   generateTCLabFanData() simulates TCLabFan with
%     - the measured input sequences and initial temperatures of
%       TCLabTrain01 and TCLabTest01 (same excitation as the real data), and
%     - numberOfExtraRuns additional normalised multisine experiments of
%       extraRunLength samples (initial temperatures 23-33 °C),
%   adds Gaussian measurement noise (noiseStd) and quantises the measured
%   temperatures (quantization, ~10-bit ADC + TMP36 sensor), and saves
%   data/TCLabFanTrain.mat (XTrain, UTrain, Ts) and data/TCLabFanTest.mat
%   (XTest, UTest, Ts) in the same format as the measured data.
%
%   generateTCLabFanData(Name=Value) changes the defaults:
%     numberOfExtraRuns (30), extraRunLength (400), noiseStd (0.05 °C),
%     quantization (0.32 °C), seed (42), dataDir (./data).
    arguments
        options.numberOfExtraRuns (1,1) {mustBeInteger, mustBeNonnegative} = 30;
        options.extraRunLength (1,1) {mustBeInteger, mustBePositive} = 400;
        options.noiseStd (1,1) {mustBeNonnegative} = 0.05;
        options.quantization (1,1) {mustBeNonnegative} = 0.32;
        options.seed (1,1) {mustBeInteger, mustBeNonnegative} = 42;
        options.dataDir (1,1) string = fullfile(fileparts(which("generateTCLabFanData")), "data");
    end
    rng(options.seed);
    plant = TCLabFan();
    train = load(fullfile(options.dataDir, "TCLabTrain01.mat"), "XTrain", "UTrain", "Ts");
    test = load(fullfile(options.dataDir, "TCLabTest01.mat"), "XTest", "UTest", "Ts");
    Ts = train.Ts;
    if plant.Ts ~= Ts
        error("Sampling time of TCLabFan (%g s) and data (%g s) differ.", plant.Ts, Ts);
    end

    % Training inputs: measured sequences + extra multisine experiments
    UTrain = train.UTrain(:);
    x0Train = cellfun(@(x) x(:, 1), train.XTrain(:), UniformOutput=false);
    periods = 10:15:150; % multisine periods [samples]
    time = (0:options.extraRunLength - 1)';
    for l = 1:options.numberOfExtraRuns
        u = zeros(2, options.extraRunLength);
        for j = 1:2 % normalised multisine in [0, 100] (no saturation)
            s = sum(sin(2 * pi * (time ./ periods + rand(1, numel(periods)))), 2)';
            u(j, :) = 50 * (1 + s / max(abs(s)));
        end
        UTrain{end+1, 1} = clip(u, 0, 100); %#ok<AGROW>
        x0Train{end+1, 1} = 23 + 10 * rand(2, 1); %#ok<AGROW>
    end
    UTest = test.UTest(:);
    x0Test = cellfun(@(x) x(:, 1), test.XTest(:), UniformOutput=false);

    fprintf("\tSimulating %d training and %d test experiments of the TC-Lab with fan...\n", ...
        numel(UTrain), numel(UTest));
    XTrain = simulateAll(plant, x0Train, UTrain, options);
    XTest = simulateAll(plant, x0Test, UTest, options);

    save(fullfile(options.dataDir, "TCLabFanTrain.mat"), "XTrain", "UTrain", "Ts");
    save(fullfile(options.dataDir, "TCLabFanTest.mat"), "XTest", "UTest", "Ts");
    fprintf("\tSaved TCLabFanTrain.mat and TCLabFanTest.mat to %s\n", options.dataDir);
end

function Y = simulateAll(plant, x0, U, options)
    Y = cell(numel(U), 1);
    for l = 1:numel(U)
        X = plant.simulate([x0{l}; x0{l}], U{l}); % heaters and sensors start equal
        y = plant.output(X) + options.noiseStd * randn(2, size(X, 2));
        if options.quantization > 0
            y = round(y / options.quantization) * options.quantization;
        end
        Y{l} = y;
    end
end
