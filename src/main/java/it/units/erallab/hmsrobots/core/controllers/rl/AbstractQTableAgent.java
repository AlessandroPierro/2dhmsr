package it.units.erallab.hmsrobots.core.controllers.rl;

import it.units.erallab.hmsrobots.core.snapshots.QTableAgentState;
import it.units.erallab.hmsrobots.core.snapshots.Snapshot;

import java.util.function.Supplier;
import java.util.random.RandomGenerator;

public abstract class AbstractQTableAgent implements DiscreteRL {
  protected final double learningRateDecay;
  protected final double explorationRateDecay;
  protected final double discountFactor;
  protected final RandomGenerator random;
  protected final int inputDimension;
  protected final int outputDimension;
  protected final double[][] qTable;
  protected final boolean episodic;
  protected double learningRate;
  protected double explorationRate;
  protected boolean initialized;
  protected int previousState;
  protected int action;

  public AbstractQTableAgent(
      double learningRate,
      double explorationRate,
      double learningRateDecay,
      double explorationRateDecay,
      double discountFactor, RandomGenerator random,
      Supplier<Double> initializer,
      int inputDimension,
      int outputDimension,
      boolean episodic
  ) {
    this.learningRate = learningRate;
    this.explorationRate = explorationRate;
    this.learningRateDecay = learningRateDecay;
    this.explorationRateDecay = explorationRateDecay;
    this.discountFactor = discountFactor;
    this.random = random;
    this.inputDimension = inputDimension;
    this.outputDimension = outputDimension;
    this.qTable = new double[inputDimension][outputDimension];

    for (int i = 0; i < inputDimension; i++) {
      for (int j = 0; j < outputDimension; j++) {
        qTable[i][j] = initializer.get();
      }
    }

    this.episodic = episodic;
  }

  @Override
  public int apply(double t, int input, double r) {
    if (initialized) {
      updateQTable(previousState, action, input, r);
    } else {
      initialized = true;
    }

    action = random.nextDouble() < explorationRate ? random.nextInt(outputDimension) : getMaxAction(input);
    previousState = input;

    if (!episodic) {
      learningRate = learningRate * learningRateDecay;
      explorationRate = explorationRate * explorationRateDecay;
    }

    return action;
  }

  public int getInputDimension() {
    return inputDimension;
  }

  public int getOutputDimension() {
    return outputDimension;
  }

  protected int getMaxAction(int state) {
    double maxQ = Double.NEGATIVE_INFINITY;
    int maxAction = 0;
    for (int action = 0; action < outputDimension; action++) {
      if (qTable[state][action] > maxQ) {
        maxQ = qTable[state][action];
        maxAction = action;
      }
    }
    return maxAction;
  }

  protected double getMaxQ(int state) {
    double maxQ = Double.NEGATIVE_INFINITY;
    for (int action = 0; action < outputDimension; action++) {
      maxQ = Math.max(maxQ, qTable[state][action]);
    }
    return maxQ;
  }

  @Override
  public Snapshot getSnapshot() {
    QTableAgentState content = new QTableAgentState(
        qTable,
        inputDimension,
        outputDimension,
        learningRate,
        explorationRate,
        learningRateDecay,
        explorationRateDecay,
        discountFactor
    );
    return new Snapshot(content, this.getClass());
  }

  @Override
  public void reset() {
    initialized = false;
    learningRate = learningRate * learningRateDecay;
    explorationRate = explorationRate * explorationRateDecay;
  }

  protected void updateQTable(int previousState, int action, int newState, double r) {}

  public void stopExploration() {
    this.explorationRate = 0.0;
  }
}
