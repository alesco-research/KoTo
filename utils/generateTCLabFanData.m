function [XTrain, UTrain, XTest, UTest, Ts] = generateTCLabFanData(options)
%GENERATETCLABFANDATA Simulate training and test data for the TC-Lab with fan.
%   [XTrain, UTrain, XTest, UTest, Ts] = generateTCLabFanData() simulates
%   TCLab (fanSwitch="on") for numberOfTrainRuns training and
%   numberOfTestRuns test experiments of runLength samples each. Every
%   experiment is excited by independent multisines of raised cosines
%   (periods 10:15:145 s, random magnitudes and phases) on all inputs:
%   the heater magnitudes sum to 120 % (clipped to [0, 100] %), and with
%   probability 0.2 each, heater 1 or heater 2 is off for the whole run;
%   the fan magnitudes sum to 2, so u3 lies in [0, 2]. Experiments start
%   from random initial temperatures T1, T2 ~ U(20, 30) °C (hence
%   |T1 - T2| <= 10 °C). Gaussian measurement noise (noiseStd) is added and
%   the measured temperatures are quantised (quantization, ~10-bit ADC +
%   TMP36 sensor). The trajectories are returned as column cell arrays
%   (X: 2 x N, U: 3 x N-1); Ts is the sampling time of the plant [s].
%
%   generateTCLabFanData(Name=Value) changes the defaults:
%     numberOfTrainRuns (30), numberOfTestRuns (6), runLength (400),
%     noiseStd (0.05 °C), quantization (0.32 °C), seed (42).
    arguments
        options.numberOfTrainRuns (1,1) {mustBeInteger, mustBePositive} = 30;
        options.numberOfTestRuns (1,1) {mustBeInteger, mustBePositive} = 6;
        options.runLength (1,1) {mustBeInteger, mustBePositive} = 400;
        options.noiseCov (:,:) {mustBeNonnegative} = 0.05 * eye(2);
        options.seed (1,1) {mustBeInteger, mustBeNonnegative} = 42;
        options.Ts (1,1) {mustBePositive} = 1;
    end
    rng(options.seed);
    Ts = options.Ts;
    plant = TCLab(Ts=Ts, measurementNoiseCov=options.noiseCov, fanSwitch="on");

    [x0Train, UTrain] = sampleExperiments(options.numberOfTrainRuns, options.runLength, Ts);
    [x0Test, UTest] = sampleExperiments(options.numberOfTestRuns, options.runLength, Ts);

    XTrain = simulateAll(plant, x0Train, UTrain);
    XTest = simulateAll(plant, x0Test, UTest);
end

function [x0, U] = sampleExperiments(numberOfRuns, runLength, Ts)
    periods = (10:15:145)';
    time = (0:runLength - 1) * Ts;
    x0 = cell(numberOfRuns, 1);
    U = cell(numberOfRuns, 1);
    for l = 1:numberOfRuns
        switchOff = rand();
        magnitudes = rand(3, numel(periods));
        magnitudes = 120 * magnitudes ./ sum(magnitudes, 2);
        magnitudes(1, :) = magnitudes(1, :) * (switchOff < 0.8);
        magnitudes(2, :) = magnitudes(2, :) * (switchOff > 0.2);
        
        u = zeros(3, runLength);
        for j = 1:3
            u(j, :) = multisine(magnitudes(j, :), periods, time) + 10 * (0.5 - rand(1, runLength));
        end

        U{l} = clip(u, 0, 100);
        x0{l} = 20 + 10 * rand(2, 1);
    end
end

function u = multisine(magnitudes, periods, time)
    phases = 2 * pi * rand(numel(periods), 1);
    waves = 1 + cos(2 * pi * time ./ periods + phases);
    u = magnitudes * waves / 2;
end

function Y = simulateAll(plant, x0, U)
    Y = cell(numel(U), 1);
    for l = 1:numel(U)
         % Start heaters and sensors at equal temperature
        [~, Y{l}] = plant.simulate([x0{l}; x0{l}], U{l});
    end
end
