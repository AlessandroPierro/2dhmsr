package it.units.erallab.hmsrobots.core.controllers.rl.discrete;

import java.util.function.Supplier;

public class QLearningAgent extends AbstractQTableAgent {

  public QLearningAgent(
      double learningRateDecay,
      double explorationRateDecay,
      double discountFactor,
      int seed,
      Supplier<Double> initializer,
      int stateSpaceDimension,
      int actionSpaceDimension
  ) {
    super(
        learningRateDecay,
        explorationRateDecay,
        discountFactor,
        seed,
        initializer,
        stateSpaceDimension,
        actionSpaceDimension
    );
  }

  @Override
  protected void updateQTable(int previousState, int action, double reward, int newState, double[][] qTable) {
    double q = qTable[previousState][action];
    double maxQ = getMaxQ(newState, qTable);
    qTable[previousState][action] = q + learningRates[previousState] * (reward + discountFactor * maxQ - q);
  }
}
