package it.units.erallab.hmsrobots.core.controllers.rl.tabular;

import com.fasterxml.jackson.annotation.JsonProperty;
import it.units.erallab.hmsrobots.core.controllers.rl.RLUtils;

public class TabularQLearning extends AbstractQTableAgent {

  public TabularQLearning(
      double discountFactor,
      int stateDim,
      int actionDim,
      double meanQ,
      double stdQ,
      int seed,
      RLUtils.InputConverter inputConverter,
      RLUtils.OutputConverter outputConverter
  ) {
    super(discountFactor, stateDim, actionDim, meanQ, stdQ, seed, inputConverter, outputConverter);
  }

  public TabularQLearning(
      @JsonProperty("discountFactor") double discountFactor,
      @JsonProperty("stateDim") int stateDim, @JsonProperty("actionDim") int actionDim,
      @JsonProperty("qTable") double[][] qTable, @JsonProperty("seed") int seed,
      @JsonProperty("episodeNumber") int episodeNumber,
      @JsonProperty("inputConverter") RLUtils.InputConverter inputConverter,
      @JsonProperty("outputConverter") RLUtils.OutputConverter outputConverter
  ) {
    super(discountFactor, stateDim, actionDim, qTable, seed, episodeNumber, inputConverter, outputConverter);
  }

  @Override
  protected void updateQTable(int newState, double reward, int newAction) {
    double delta = reward + discountFactor * getMaxQ(newState) - qTable[previousState][previousAction];
    qTable[previousState][previousAction] += learningRate * delta;
  }
}
