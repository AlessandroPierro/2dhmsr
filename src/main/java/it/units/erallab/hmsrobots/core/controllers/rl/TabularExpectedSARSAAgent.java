package it.units.erallab.hmsrobots.core.controllers.rl;

import it.units.erallab.hmsrobots.core.controllers.Resettable;

import java.util.function.Supplier;
import java.util.random.RandomGenerator;

public class TabularExpectedSARSAAgent implements DiscreteRL, Resettable {
  private final double learningRateDecay;
  private final double explorationRateDecay;
  private final double discountFactor;
  private final RandomGenerator random;
  private final int inputDimension;
  private final int outputDimension;
  private final double[][] qTable;
  private final boolean episodic;
  private double learningRate;
  private double explorationRate;
  private boolean initialized = false;
  private int previousState;
  private int action;

  public TabularExpectedSARSAAgent(
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

  private int getMaxAction(int state) {
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

  private double getMaxQ(int state) {
    double maxQ = Double.NEGATIVE_INFINITY;
    for (int action = 0; action < outputDimension; action++) {
      maxQ = Math.max(maxQ, qTable[state][action]);
    }
    return maxQ;
  }

  @Override
  public void reset() {
    initialized = false;
    learningRate = learningRate * learningRateDecay;
    explorationRate = explorationRate * explorationRateDecay;
  }

  private void updateQTable(int previousState, int action, int newState, double r) {
    double q = qTable[previousState][action];
    int maxQAction = getMaxAction(newState);
    double expectedSARSA = 0.0;
    for (int possibleAction = 0; possibleAction < outputDimension; possibleAction++) {
      expectedSARSA += qTable[newState][possibleAction] * (possibleAction == maxQAction ? 0.9 : (0.1 / (outputDimension - 1)));
    }
    qTable[previousState][action] = q + learningRate * (r + discountFactor * expectedSARSA - q);
  }

  public void setExplorationRate(double explorationRate) {
    this.explorationRate = explorationRate;
  }
}
