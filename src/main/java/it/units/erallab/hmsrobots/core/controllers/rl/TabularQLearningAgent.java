package it.units.erallab.hmsrobots.core.controllers.rl;

import java.util.function.Supplier;
import java.util.random.RandomGenerator;

public class TabularQLearningAgent implements DiscreteRL {
  private double learningRate;
  private double explorationRate;
  private final double learningRateDecay;
  private final double explorationRateDecay;
  private final RandomGenerator rg;

  private final int inputDimension;
  private final int outputDimension;

  private final double[][] qTable;

  private int previousState;
  private int action;

  public TabularQLearningAgent(
      double learningRate,
      double explorationRate,
      double learningRateDecay,
      double explorationRateDecay,
      RandomGenerator rg,
      Supplier<Double> initializer,
      int inputDimension,
      int outputDimension
  ) {
    this.learningRate = learningRate;
    this.explorationRate = explorationRate;
    this.learningRateDecay = learningRateDecay;
    this.explorationRateDecay = explorationRateDecay;
    this.rg = rg;
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
    updateQTable(previousState, action, input, r);
    double random = rg.nextDouble();
    if (random < explorationRate) {
      action = rg.nextInt(outputDimension);
    } else {
      action = getMaxAction(input);
    }
    previousState = input;
    learningRate = learningRate * learningRateDecay;
    explorationRate = explorationRate * explorationRateDecay;
    return action;
  }

  @Override
  public int getInputDimension() {
    return inputDimension;
  }

  @Override
  public int getOutputDimension() {
    return outputDimension;
  }

  private void updateQTable(int previous_state, int action, int new_state, double r) {
    double q = qTable[previous_state][action];
    double maxQ = getMaxQ(new_state);
    qTable[previous_state][action] = q + learningRate * (r + maxQ - q);
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
