package it.units.erallab.hmsrobots.core.controllers.rl.discrete;

import it.units.erallab.hmsrobots.core.snapshots.QTableAgentState;
import it.units.erallab.hmsrobots.core.snapshots.Snapshot;
import it.units.erallab.hmsrobots.core.snapshots.Snapshottable;

import java.util.Arrays;
import java.util.Random;
import java.util.random.RandomGenerator;

public abstract class AbstractQTableAgent implements DiscreteRL, Snapshottable {
  protected final double discountFactor;
  protected final int stateDim;
  protected final int actionDim;
  protected final RandomGenerator random;

  protected double[][] qTable;
  protected int previousState;
  protected int previousAction;

  protected double epsilon;
  protected double learningRate;

  protected boolean learn = true;
  protected int episodeNumber = 0;

  public AbstractQTableAgent(
      double discountFactor,
      int seed,
      int stateDim,
      int actionDim
  ) {
    this.discountFactor = discountFactor;
    this.random = new Random(seed);
    this.stateDim = stateDim;
    this.actionDim = actionDim;
    this.qTable = new double[stateDim][actionDim];
    for (int i = 0; i < stateDim; i++) {
      for (int j = 0; j < actionDim; j++) {
        qTable[i][j] = 0d;
      }
    }
  }

  public AbstractQTableAgent(
      double discountFactor,
      int seed,
      int stateDim,
      int actionDim,
      double[][] qTable
  ) {
    this(discountFactor, seed, stateDim, actionDim);
    this.qTable = copy(qTable);
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
    return action;
  }

  private double explorationRateSchedule(double t) {
    return 1d / (1d + episodeNumber);
  }

  private double learningRateSchedule(double t) {
    return 1d / (1d + episodeNumber);
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
      maxAction = qTable[state][action] > qTable[state][maxAction] ? action : maxAction;
    }
    return maxAction;
  }

  protected double getMaxQ(int state) {
    return qTable[state][getMaxAction(state)];
  }

  @Override
  public Snapshot getSnapshot() {
    QTableAgentState content = new QTableAgentState(
        copy(qTable),
        stateDim,
        actionDim,
        0d,
        discountFactor,
        previousState
    );

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

  protected double[][] copy(double[][] matrix) {
    return Arrays.stream(matrix).map(double[]::clone).toArray(double[][]::new);
  }

  protected int selectEpsGreedyAction(int state) {
    return random.nextDouble() < epsilon ? random.nextInt(actionDim) : getMaxAction(state);
  }

  public void stopLearning() {
    learn = false;
  }

  public void startLearning() {
    learn = true;
  }

}
