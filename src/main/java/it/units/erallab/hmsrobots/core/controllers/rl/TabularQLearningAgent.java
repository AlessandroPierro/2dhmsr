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
      boolean episodic,
      int stateSpaceDimension,
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
        episodic,
        stateSpaceDimension,
        actionSpaceDimension
    );
  }

  @Override
  protected void updateQTable(int previousState, int action, double reward, int newState) {
    double q = qTable[previousState][action];
    double maxQ = getMaxQ(newState);
    qTable[previousState][action] = q + learningRate * (reward + discountFactor * maxQ - q);
  }
}
