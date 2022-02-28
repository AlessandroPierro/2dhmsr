package it.units.erallab.hmsrobots.core.controllers.rl;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.function.Supplier;

public class TabularExpectedSARSAAgent extends AbstractQTableAgent {

  public TabularExpectedSARSAAgent(
      double learningRate,
      double explorationRate,
      double learningRateDecay,
      double explorationRateDecay,
      double discountFactor,
      int seed,
      Supplier<Double> initializer,
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
        stateSpaceDimension,
        actionSpaceDimension
    );
  }

  public TabularExpectedSARSAAgent(
      @JsonProperty("learningRate") double learningRate,
      @JsonProperty("explorationRate") double explorationRate,
      @JsonProperty("learningRateDecay") double learningRateDecay,
      @JsonProperty("explorationRateDecay") double explorationRateDecay,
      @JsonProperty("discountFactor") double discountFactor,
      @JsonProperty("qTable") double[][] qTable,
      @JsonProperty("seed") int seed,
      @JsonProperty("stateSpaceDimension") int stateSpaceDimension,
      @JsonProperty("actionSpaceDimension") int actionSpaceDimension
  ) {
    super(
        learningRate,
        explorationRate,
        learningRateDecay,
        explorationRateDecay,
        discountFactor,
        qTable,
        seed,
        stateSpaceDimension,
        actionSpaceDimension
    );
  }

  @Override
  protected void updateQTable(int previousState, int action, double reward, int newState, double[][] qTable) {
    double q = qTable[previousState][action];
    int maxQAction = getMaxAction(newState, qTable);
    double expectedSARSA = 0.0;
    for (int possibleAction = 0; possibleAction < actionSpaceDimension; possibleAction++) {
      expectedSARSA += qTable[newState][possibleAction] * (explorationRate / actionSpaceDimension + (possibleAction == maxQAction ? 1 - explorationRate : 0.0));
    }
    qTable[previousState][action] = q + learningRate * (reward + discountFactor * expectedSARSA - q);
  }
}
