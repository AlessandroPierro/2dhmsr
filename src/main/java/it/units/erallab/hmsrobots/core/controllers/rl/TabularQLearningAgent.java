package it.units.erallab.hmsrobots.core.controllers.rl;

import it.units.erallab.hmsrobots.core.controllers.Resettable;

import java.util.function.Supplier;
import java.util.random.RandomGenerator;

public class TabularQLearningAgent implements DiscreteRL, Resettable {
  private double learningRate;
  private double explorationRate;
  private final double learningRateDecay;
  private final double explorationRateDecay;
  private final double discountFactor;
  private final RandomGenerator random;

  private final int inputDimension;
  private final int outputDimension;

  private final double[][] qTable;
  private boolean initialized;
  private int previousState;
  private int action;

  public TabularQLearningAgent(
      double learningRate,
      double explorationRate,
      double learningRateDecay,
      double explorationRateDecay,
      double discountFactor, RandomGenerator random,
      Supplier<Double> initializer,
      int inputDimension,
      int outputDimension
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
    learningRate = learningRate * learningRateDecay;
    explorationRate = explorationRate * explorationRateDecay;
    return action;
  }

  public int getInputDimension() {
    return inputDimension;
  }

  public int getOutputDimension() {
    return outputDimension;
  }

  @Override
  public void reset() {
    initialized = false;
  }

  private void updateQTable(int previousState, int action, int newState, double r) {
    double q = qTable[previousState][action];
    double maxQ = getMaxQ(newState);
    qTable[previousState][action] = q + learningRate * (r + discountFactor * maxQ - q);
  }

  private double getMaxQ(int state) {
    double maxQ = Double.NEGATIVE_INFINITY;
    for (int action = 0; action < outputDimension; action++) {
      maxQ = Math.max(maxQ, qTable[state][action]);
    }
    return maxQ;
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
}
