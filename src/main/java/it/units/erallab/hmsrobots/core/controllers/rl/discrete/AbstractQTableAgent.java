package it.units.erallab.hmsrobots.core.controllers.rl.discrete;

import it.units.erallab.hmsrobots.core.snapshots.QTableAgentState;
import it.units.erallab.hmsrobots.core.snapshots.Snapshot;
import it.units.erallab.hmsrobots.core.snapshots.Snapshottable;
import org.apache.commons.math3.linear.RealMatrix;

import java.io.Serializable;
import java.util.Random;
import java.util.random.RandomGenerator;

import static org.apache.commons.math3.linear.MatrixUtils.createRealMatrix;

public abstract class AbstractQTableAgent implements DiscreteRL, Snapshottable, Serializable {
    protected final double discountFactor;
    protected final int stateDim;
    protected final int actionDim;
    protected final RandomGenerator random;
    protected final double meanQ;
    protected final double stdQ;

    protected RealMatrix qTable;
    protected int previousState;
    protected int previousAction;
    protected double previousReward;

    protected double epsilon;
    protected double learningRate;

    // Internals
    protected boolean learn = true;
    protected int episodeNumber;

    public AbstractQTableAgent(double discountFactor, int stateDim, int actionDim, double meanQ, double stdQ, int seed) {
        this.discountFactor = discountFactor;
        this.random = new Random(seed);
        this.meanQ = meanQ;
        this.stdQ = stdQ;
        this.stateDim = stateDim;
        this.actionDim = actionDim;
        this.qTable = createRealMatrix(stateDim, actionDim);

        for (int i = 0; i < stateDim; i++) {
            for (int j = 0; j < actionDim; j++) {
                qTable.setEntry(i, j, random.nextGaussian() * stdQ + meanQ);
            }
        }

        this.episodeNumber = 0;
    }

    @Override
    public int apply(double t, int newState, double reward) {
        epsilon = explorationRateSchedule(t);
        learningRate = learningRateSchedule(t);
        int action = selectEpsGreedyAction(newState);
        if (previousAction != Integer.MIN_VALUE && learn) {
            updateQTable(newState, reward, action);
        }
        previousState = newState;
        previousAction = action;
        previousReward = reward;
        return action;
    }

    private double explorationRateSchedule(double t) {
        return 1d / (1d + episodeNumber);
    }

    private double learningRateSchedule(double t) {
        return 0.05;
    }

    public int getInputDimension() {
        return stateDim;
    }

    public int getOutputDimension() {
        return actionDim;
    }

    protected int getMaxAction(int state) {
        int maxAction = 0;
        for (int action = 1; action < actionDim; action++) {
            maxAction = qTable.getEntry(state, action) > qTable.getEntry(state, maxAction) ? action : maxAction;
        }
        return maxAction;
    }

    protected double getMaxQ(int state) {
        return qTable.getEntry(state, getMaxAction(state));
    }

    @Override
    public Snapshot getSnapshot() {
        QTableAgentState content = new QTableAgentState(qTable.getData(), stateDim, actionDim, learningRate, epsilon, previousState, previousAction, previousReward);

        return new Snapshot(content, this.getClass());
    }

    @Override
    public void reset() {
        previousAction = Integer.MIN_VALUE;
        previousState = Integer.MIN_VALUE;
        episodeNumber++;
    }

    protected void updateQTable(int newState, double reward, int action) {
    }

    protected int selectEpsGreedyAction(int state) {
        return random.nextDouble() < epsilon ? random.nextInt(actionDim) : getMaxAction(state);
    }

    @Override
    public void reinitialize() {
        episodeNumber = 0;
        for (int i = 0; i < stateDim; i++) {
            for (int j = 0; j < actionDim; j++) {
                qTable.setEntry(i, j, random.nextGaussian() * stdQ + meanQ);
            }
        }
        previousAction = Integer.MIN_VALUE;
        previousState = Integer.MIN_VALUE;
    }

}
