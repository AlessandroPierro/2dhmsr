package it.units.erallab.hmsrobots.core.controllers.rl.discrete;

import it.units.erallab.hmsrobots.core.controllers.rl.discrete.DiscreteRL;

import java.util.function.Function;

public class BinaryInputConverter implements DiscreteRL.InputConverter {

  private final int inputDimension;
  private final double cutoff;

  public BinaryInputConverter(
      int inputDimension,
      double cutoff
  ) {
    this.inputDimension = inputDimension;
    this.cutoff = cutoff;
  }

  @Override
  public Integer apply(double[] doubles) {
    StringBuilder result = new StringBuilder();
    for (double d : doubles) {
      result.append(d > cutoff ? "1" : "0");
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