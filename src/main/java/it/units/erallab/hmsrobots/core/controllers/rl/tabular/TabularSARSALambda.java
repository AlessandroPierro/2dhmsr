package it.units.erallab.hmsrobots.core.controllers.rl.tabular;

import com.fasterxml.jackson.annotation.JsonProperty;
import it.units.erallab.hmsrobots.core.controllers.rl.RLUtils;

public class TabularSARSALambda extends AbstractQTableAgent {
  @JsonProperty
  protected final double lambda;

  protected transient double[][] eTraces;

  public TabularSARSALambda(
      double discountFactor,
      int stateDim,
      int actionDim,
      double meanQ,
      double stdQ,
      int seed,
      RLUtils.InputConverter inputConverter,
      RLUtils.OutputConverter outputConverter,
      double lambda
  ) {
    super(discountFactor, stateDim, actionDim, meanQ, stdQ, seed, inputConverter, outputConverter);
    this.lambda = lambda;
  }

  public TabularSARSALambda(
      @JsonProperty("discountFactor") double discountFactor,
      @JsonProperty("stateDim") int stateDim,
      @JsonProperty("actionDim") int actionDim,
      @JsonProperty("qTable") double[][] qTable,
      @JsonProperty("seed") int seed,
      @JsonProperty("episodeNumber") int episodeNumber,
      @JsonProperty("inputConverter") RLUtils.InputConverter inputConverter,
      @JsonProperty("outputConverter") RLUtils.OutputConverter outputConverter,
      @JsonProperty("lambda") double lambda
  ) {
    super(discountFactor, stateDim, actionDim, qTable, seed, episodeNumber, inputConverter, outputConverter);
    this.lambda = lambda;
  }

  @Override
  public void reset() {
    super.reset();
    eTraces = new double[stateDim][actionDim];
  }

  @Override
  protected void updateQTable(int newState, double reward, int newAction) {
    double delta = reward + discountFactor * qTable[newState][newAction] - qTable[previousState][previousAction];
    eTraces[previousState][previousAction] = 1;
    for (int i = 0; i < stateDim; i++) {
      for (int j = 0; j < actionDim; j++) {
        qTable[i][j] += learningRate * delta * eTraces[i][j];
        eTraces[i][j] *= discountFactor * lambda;
      }
    }
  }

}
