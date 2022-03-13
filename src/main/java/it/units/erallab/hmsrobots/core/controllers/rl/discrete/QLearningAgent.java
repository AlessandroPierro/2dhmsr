package it.units.erallab.hmsrobots.core.controllers.rl.discrete;

public class QLearningAgent extends AbstractQTableAgent {

  public QLearningAgent(
      double learningRateDecay,
      double explorationRateDecay,
      double discountFactor,
      int seed,
      int stateSpaceDimension,
      int actionSpaceDimension
  ) {
    super(
        learningRateDecay,
        explorationRateDecay,
        discountFactor,
        seed,
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
