package it.units.erallab.hmsrobots.core.controllers.rl;

import java.util.function.Supplier;

public class TabularQLearningAgent extends AbstractQTableAgent {

  public TabularQLearningAgent(
      double learningRate,
      double explorationRate,
      double learningRateDecay,
      double explorationRateDecay,
      double discountFactor,
      int seed,
      Supplier<Double> initializer,
      int inputDimension,
      int outputDimension,
      boolean episodic,
      int stateSpaceDimenstion,
      int actionSpaceDimension
  ) {
    super(
        learningRate,
        explorationRate,
        learningRateDecay,
        explorationRateDecay,
        discountFactor,
        seed,
        initializer,
        inputDimension,
        outputDimension,
        episodic,
        stateSpaceDimenstion,
        actionSpaceDimension
    );
  }

  @Override
  protected void updateQTable(int previousState, int action, int newState, double r) {
    double q = qTable[previousState][action];
    double maxQ = getMaxQ(newState);
    qTable[previousState][action] = q + learningRate * (r + discountFactor * maxQ - q);
  }
}
