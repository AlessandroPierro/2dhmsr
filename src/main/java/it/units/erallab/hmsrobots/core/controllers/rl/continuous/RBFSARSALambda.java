package it.units.erallab.hmsrobots.core.controllers.rl.continuous;

import it.units.erallab.hmsrobots.core.controllers.rl.discrete.DiscreteRL;
import it.units.erallab.hmsrobots.core.controllers.rl.discrete.converters.BinaryOutputConverter;
import it.units.erallab.hmsrobots.core.snapshots.QTableAgentState;
import it.units.erallab.hmsrobots.core.snapshots.Snapshot;
import it.units.erallab.hmsrobots.core.snapshots.Snapshottable;

import java.util.Arrays;
import java.util.Random;

public class RBFSARSALambda implements ContinuousRL, Snapshottable {

    private final double discountFactor;
    private final double lambda;
    private final int stateDim;
    private final int actionDim;
    private final Random random;
    private double[][] weights;
    private final double[] eTraces;
    private double epsilon;
    private double learningRate;
    private boolean learn = true;
    private int episodeNumber = 0;
    private double sigma = 0.15;

    private int previousAction;
    private double[] previousState;
    private double previousQ;
    private boolean initialized = false;

    private final DiscreteRL.OutputConverter outputConverter;

    public RBFSARSALambda(
            double lambda, double discountFactor, int stateDim, int actionDim, double initMean, double initStd, int seed
    ) {
        this.discountFactor = discountFactor;
        this.lambda = lambda;
        this.stateDim = stateDim;
        this.actionDim = actionDim;
        this.weights = new double[actionDim][stateDim + 1];
        this.eTraces = new double[stateDim + 1];
        Arrays.fill(eTraces, 0d);
        this.random = new java.util.Random(seed);
        for (int i = 0; i < actionDim; i++) {
            for (int j = 0; j < stateDim + 1; j++) {
                weights[i][j] = random.nextGaussian() * initStd + initMean;
            }
        }

        outputConverter = new BinaryOutputConverter(4, 0.5);
        previousState = new double[stateDim];
    }

    public RBFSARSALambda(
            double lambda, double discountFactor, int stateDim, int actionDim, double initMean, double initStd, int seed, double[][] weights
    ) {
        this(lambda, discountFactor, stateDim, actionDim, initMean, initStd, seed);
        this.weights = Arrays.stream(weights).map(double[]::clone).toArray(double[][]::new);
    }

    @Override
    public double[] apply(double t, double[] newState, double reward) {
        epsilon = explorationRateSchedule(t);
        learningRate = learningRateSchedule(t);

        int newAction = 0;
        if (random.nextDouble() < epsilon) {
            newAction = random.nextInt(actionDim);
        } else {
            double Q_value = Double.MIN_VALUE;
            for (int i = 0; i < actionDim; i++) {
                double Q = 0;
                for (int j = 0; j < stateDim + 1; j++) {
                    Q += weights[i][j] * (j == stateDim ? 1 : newState[j]);
                }
                if (Q > Q_value) {
                    newAction = i;
                    Q_value = Q;
                }
            }
        }

        if (initialized && learn) {
            double Q = 0d;
            double Q_new = 0d;

            for (int i = 0; i < stateDim + 1; i++) {
                Q += weights[previousAction][i] * (i == stateDim ? 1 : previousState[i]);
                Q_new += weights[newAction][i] * (i == stateDim ? 1 : newState[i]);
                eTraces[i] = (i == stateDim ? 1 : previousState[i]) + discountFactor * lambda * eTraces[i];
            }

            double delta = reward + discountFactor * Q_new - Q;
            double norm = 0d;

            for (int i = 0; i < stateDim + 1; i++) {
                weights[previousAction][i] += learningRate * delta * eTraces[i];
                norm += Math.pow(weights[previousAction][i], 2);
            }

            norm = Math.sqrt(norm);

            for (int i = 0; i < stateDim + 1; i++) {
                weights[previousAction][i] /= norm;
            }
        }

        initialized = true;
        if (stateDim >= 0) System.arraycopy(newState, 0, previousState, 0, stateDim);
        previousAction = newAction;
        return outputConverter.apply(newAction);
    }

    @Override
    public void reinitialize() {

    }

    private double explorationRateSchedule(double t) {
        return learn ? 1.0 / episodeNumber : 0.05;
    }

    @Override
    public int getInputDimension() {
        return stateDim;
    }

    @Override
    public int getOutputDimension() {
        return (int) (Math.log(actionDim) / Math.log(2));
    }

    @Override
    public Snapshot getSnapshot() {
        return null;
    }

    private double learningRateSchedule(double t) {
        return 0.1 / episodeNumber;
    }

    @Override
    public void reset() {
        episodeNumber++;
        initialized = false;
        Arrays.fill(eTraces, 0d);
    }

    public void startLearning() {
        learn = true;
    }

    public void stopLearning() {
        learn = false;
        System.out.print("weights: ");
        System.out.println(Arrays.deepToString(weights));
    }
}
