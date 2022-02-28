package it.units.erallab.hmsrobots.core.controllers.rl.discrete.converters;

import it.units.erallab.hmsrobots.core.controllers.rl.discrete.DiscreteRL;

import java.util.function.Function;

public class BinaryInputConverter implements DiscreteRL.InputConverter {

  private final int inputDimension;
  private final double[] splitValues;

  public BinaryInputConverter(
      int inputDimension,
      double[] splitValues
  ) {
    this.inputDimension = inputDimension;
    this.splitValues = splitValues;
  }

  @Override
  public Integer apply(double[] doubles) {
    StringBuilder result = new StringBuilder();
    for (int i = 0; i < inputDimension; i++) {
      result.append(doubles[i] > splitValues[i] ? "1" : "0");
    }
    return Integer.parseInt(result.toString(), 2);
  }

  @Override
  public <V> Function<V, Integer> compose(Function<? super V, ? extends double[]> before) {
    return DiscreteRL.InputConverter.super.compose(before);
  }

  @Override
  public <V> Function<double[], V> andThen(Function<? super Integer, ? extends V> after) {
    return DiscreteRL.InputConverter.super.andThen(after);
  }

  @Override
  public int getInputDimension() {
    return inputDimension;
  }

  @Override
  public int getOutputDimension() {
    return 1;
  }
}