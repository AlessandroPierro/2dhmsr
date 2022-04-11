package it.units.erallab.hmsrobots.core.controllers.rl.continuous;

import it.units.erallab.hmsrobots.core.snapshots.QTableAgentState;
import it.units.erallab.hmsrobots.core.snapshots.Snapshot;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

public class GaussianPG implements ContinuousRL, Serializable {

    private final double discountFactor;
    private final int stateDim;
    private final int actionDim;

    private final Random random;
    private final double[][][] weights;
    private static final double sigmaPolicy = 0.2;
    private static final double sigmaFeatures = 0.15;

    private static final int featureDim = 3;
    private double learningRate;
    private double G = 0d;
    private boolean initialized = false;
    private boolean learn = true;
    private double[] previousAction;
    private double[][][] previousFeatures;
    private int step = 0;

    public GaussianPG(double discountFactor, int stateDim, int actionDim) {
        this.discountFactor = discountFactor;
        this.stateDim = stateDim;
        this.actionDim = actionDim;

        // TODO : remove random from here
        this.random = new Random();

        this.weights = new double[actionDim][stateDim][featureDim];
        for (int i = 0; i < actionDim; i++) {
            for (int j = 0; j < stateDim; j++) {
                for (int k = 0; k < featureDim; k++) {
                    weights[i][j][k] = -0.5;
                }
            }
        }

        this.previousAction = new double[actionDim];
        this.previousFeatures = new double[actionDim][stateDim][featureDim];
        this.learningRate = 0.1;
    }

    @Override
    public int getInputDimension() {
        return 0;
    }

    @Override
    public int getOutputDimension() {
        return 0;
    }

    @Override
    public void reset() {
        initialized = false;
    }

    @Override
    public double[] apply(double t, double[] newState, double reward) {
        G = reward + discountFactor * G;
        if (initialized && learn) {
            updateWeights();
        }
        computeFeatures(newState);
        sampleAction(previousFeatures);
        initialized = true;
        step += 1;
        return previousAction;
    }

    @Override
    public Snapshot getSnapshot() {
        QTableAgentState content = new QTableAgentState(
                new double[2][2],
                stateDim,
                actionDim,
                0d,
                discountFactor,
                0
        );

        return new Snapshot(content, this.getClass());
    }

    private void updateWeights() {
        double partialUpdate = learningRate * Math.pow(discountFactor, step) * G / Math.pow(sigmaPolicy, 2);
        for (int i = 0; i < actionDim; i++) {
            double mean = 0;
            for (int j = 0; j < stateDim; j++) {
                for (int k = 0; k < featureDim; k++) {
                    mean += weights[i][j][k] * previousFeatures[i][j][k];
                }
            }
            double norm = 0d;
            for (int j = 0; j < stateDim; j++) {
                for (int k = 0; k < featureDim; k++) {
                    double x = (1 + 2 * previousAction[i]) / (1 - 2 * previousAction[i]);
                    weights[i][j][k] += partialUpdate * (0.5 * Math.log(x) - mean) * previousFeatures[i][j][k];
                    norm += Math.pow(weights[i][j][k], 2);
                }
            }

            norm = Math.sqrt(norm);
            for (int j = 0; j < stateDim; j++) {
                for (int k = 0; k < featureDim; k++) {
                    weights[i][j][k] /= norm;
                }
            }
        }
    }

    private void computeFeatures(double[] state) {
        for (int i = 0; i < actionDim; i++) {
            for (int j = 0; j < stateDim; j++) {
                previousFeatures[i][j][0] = Math.exp(-Math.pow(state[j] - 0d, 2) / (2 * Math.pow(sigmaFeatures, 2)));
                previousFeatures[i][j][1] = Math.exp(-Math.pow(state[j] - 1d, 2) / (2 * Math.pow(sigmaFeatures, 2)));
                previousFeatures[i][j][2] = 1d;
            }
        }
    }

    private void sampleAction(double[][][] features) {
        for (int i = 0; i < actionDim; i++) {
            double mean = 0;
            for (int j = 0; j < stateDim; j++) {
                for (int k = 0; k < featureDim; k++) {
                    mean += weights[i][j][k] * features[i][j][k];
                }
            }
            System.out.println("mean : " + mean);
            double action = mean + random.nextGaussian() * sigmaPolicy;
            double aProb = Math.exp(-Math.pow(action - mean, 2) / (2 * Math.pow(sigmaPolicy, 2)));
            System.out.println("aProb : " + aProb);
            if (random.nextDouble() < aProb) {
                previousAction[i] = Math.tanh(action) * 0.5;
            } else {
                System.out.println("greedy");
                previousAction[i] = Math.tanh(mean) * 0.5;
            }
            System.out.println("action : " + previousAction[i]);
        }
    }

}
