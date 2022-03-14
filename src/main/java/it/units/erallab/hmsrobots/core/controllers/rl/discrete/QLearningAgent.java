package it.units.erallab.hmsrobots.core.controllers.rl.discrete;

import com.fasterxml.jackson.annotation.JsonProperty;

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

  public QLearningAgent(
      @JsonProperty("learningRateDecay") double learningRateDecay,
      @JsonProperty("explorationRateDecay") double explorationRateDecay,
      @JsonProperty("discountFactor") double discountFactor,
      @JsonProperty("qTable") double[][] qTable,
      @JsonProperty("seed") int seed,
      @JsonProperty("stateSpaceDimension") int stateSpaceDimension,
      @JsonProperty("actionSpaceDimension") int actionSpaceDimension,
      @JsonProperty("nLearningVisits") int[] nLearningVisits,
      @JsonProperty("nExplorationVisits") int[] nExplorationVisits,
      @JsonProperty("learningRates") double[] learningRates,
      @JsonProperty("explorationRates") double[] explorationRates
  ) {
    super(
        learningRateDecay,
        explorationRateDecay,
        discountFactor,
        qTable,
        seed,
        stateSpaceDimension,
        actionSpaceDimension,
        nLearningVisits,
        nExplorationVisits,
        learningRates,
        explorationRates
    );
  }

  @Override
  protected void updateQTable(int previousState, int action, double reward, int newState, double[][] qTable) {
    double q = qTable[previousState][action];
    double maxQ = getMaxQ(newState, qTable);
    qTable[previousState][action] = q + learningRates[previousState] * (reward + discountFactor * maxQ - q);
  }
}
