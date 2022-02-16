package it.units.erallab.hmsrobots.core.controllers.rl;

import java.util.function.Supplier;
import java.util.random.RandomGenerator;

public class TabularQLearningAgent extends AbstractQTableAgent {

  public TabularQLearningAgent(
      double learningRate,
      double explorationRate,
      double learningRateDecay,
      double explorationRateDecay,
      double discountFactor,
      RandomGenerator random,
      Supplier<Double> initializer,
      int inputDimension,
      int outputDimension,
      boolean episodic
  ) {
    super(
        learningRate,
        explorationRate,
        learningRateDecay,
        explorationRateDecay,
        discountFactor,
        random,
        initializer,
        inputDimension,
        outputDimension,
        episodic
    );
  }

  @Override
  protected void updateQTable(int previousState, int action, int newState, double r) {
    double q = qTable[previousState][action];
    double maxQ = getMaxQ(newState);
    qTable[previousState][action] = q + learningRate * (r + discountFactor * maxQ - q);
  }
}
