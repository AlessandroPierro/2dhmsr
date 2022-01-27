package it.units.erallab.hmsrobots.core.controllers.rl;

import java.util.random.RandomGenerator;

public class QLearningAgent implements DiscreteRL {
  private double learningRate;
  private double explorationRate;
  private final double learningRateDecay;
  private final double explorationRateDecay;
  private final RandomGenerator rg;

  private double[][] qTable;

  private int previous_state;
  private int action;

  private final InputConverter inputConverter;
  private final OutputConverter outputConverter;

  public QLearningAgent(
      double learningRate,
      double explorationRate,
      double learningRateDecay,
      double explorationRateDecay,
      RandomGenerator rg,
      InputConverter inputConverter,
      OutputConverter outputConverter
  ) {
    this.learningRate = learningRate;
    this.explorationRate = explorationRate;
    this.learningRateDecay = learningRateDecay;
    this.explorationRateDecay = explorationRateDecay;
    this.rg = rg;
    this.inputConverter = inputConverter;
    this.outputConverter = outputConverter;

    qTable = new double[this.inputConverter.getInputDimension()][this.outputConverter.getOutputDimension()];
  }

  @Override
  public int apply(double t, int input, double r) {
    updateQTable(previous_state, action, input, r);
    double random = rg.nextDouble();
    if (random < explorationRate) {
      action = rg.nextInt(outputConverter.getOutputDimension());
    } else {
      action = getMaxAction(input);
    }
    previous_state = input;
    learningRate = learningRate * learningRateDecay;
    explorationRate = explorationRate * explorationRateDecay;
    return action;
  }

  @Override
  public ContinuousRL with(
      InputConverter inputConverter, OutputConverter outputConverter
  ) {
    return DiscreteRL.super.with(inputConverter, outputConverter);
  }

  private void updateQTable(int previous_state, int action, int new_state, double r) {
    double q = qTable[previous_state][action];
    double maxQ = getMaxQ(new_state);
    qTable[previous_state][action] = q + learningRate * (r + maxQ - q);
  }

  private double getMaxQ(int state) {
    double maxQ = Double.NEGATIVE_INFINITY;
    for (int action = 0; action < outputConverter.getOutputDimension(); action++) {
      maxQ = Math.max(maxQ, qTable[state][action]);
    }
    return maxQ;
  }

  private int getMaxAction(int state) {
    double maxQ = Double.NEGATIVE_INFINITY;
    int maxAction = 0;
    for (int action = 0; action < outputConverter.getOutputDimension(); action++) {
      if (qTable[state][action] > maxQ) {
        maxQ = qTable[state][action];
        maxAction = action;
      }
    }
    return maxAction;
  }
}
