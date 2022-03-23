package it.units.erallab.hmsrobots.core.controllers.rl.discrete;

import com.fasterxml.jackson.annotation.JsonProperty;

public class QLearningAgent extends AbstractQTableAgent {

  public QLearningAgent(
      double learningRateDecay,
      double discountFactor,
      int seed,
      int stateSpaceDimension,
      int actionSpaceDimension
  ) {
    super(
        learningRateDecay,
        discountFactor,
        seed,
        stateSpaceDimension,
        actionSpaceDimension
    );
  }

  public QLearningAgent(
      @JsonProperty("learningRateDecay") double learningRateDecay,
      @JsonProperty("discountFactor") double discountFactor,
      @JsonProperty("qTable") double[][] qTable,
      @JsonProperty("seed") int seed,
      @JsonProperty("stateSpaceDimension") int stateSpaceDimension,
      @JsonProperty("actionSpaceDimension") int actionSpaceDimension,
      @JsonProperty("nLearningVisits") int[] nLearningVisits,
      @JsonProperty("learningRates") double[] learningRates
  ) {
    super(
        learningRateDecay,
        discountFactor,
        qTable,
        seed,
        stateSpaceDimension,
        actionSpaceDimension,
        nLearningVisits,
        learningRates
    );
  }

  @Override
  protected void updateQTable(int previousState, int action, double reward, int newState, double[][] qTable) {
    double q = qTable[previousState][action];
    double maxQ = getMaxQ(newState, qTable);
    qTable[previousState][action] = q + learningRates[previousState] * (reward + discountFactor * maxQ - q);
  }
}
