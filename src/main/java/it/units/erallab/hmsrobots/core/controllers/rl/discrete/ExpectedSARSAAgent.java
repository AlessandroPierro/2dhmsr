package it.units.erallab.hmsrobots.core.controllers.rl.discrete;

import com.fasterxml.jackson.annotation.JsonProperty;

public class ExpectedSARSAAgent extends AbstractQTableAgent {

  public ExpectedSARSAAgent(
      double learningRateDecay,
      double explorationRateDecay,
      double discountFactor, int seed,
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

  public ExpectedSARSAAgent(
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
    int maxQAction = getMaxAction(newState, qTable);
    double expectedSARSA = 0d;
    for (int possibleAction = 0; possibleAction < actionSpaceDimension; possibleAction++) {
      expectedSARSA += qTable[newState][possibleAction] * (explorationRates[previousState] / actionSpaceDimension + (possibleAction == maxQAction ? 1 - explorationRates[previousState] : 0d));
    }
    qTable[previousState][action] = q + learningRates[previousState] * (reward + discountFactor * expectedSARSA - q);
  }
}
