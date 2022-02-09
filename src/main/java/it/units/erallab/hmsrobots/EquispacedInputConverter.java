package it.units.erallab.hmsrobots;

import it.units.erallab.hmsrobots.core.controllers.rl.DiscreteRL;

import java.util.function.Function;

public class EquispacedInputConverter implements DiscreteRL.InputConverter {

  private final int inputDimension;
  private final double[] upperBound;
  private final double[] lowerBound;
  private final int[] sizes;
  private final int[] transformationIndexes;

  EquispacedInputConverter(
      int inputDimension,
      double[] upperBound,
      double[] lowerBound,
      int[] sizes
  ) {
    if (upperBound.length != inputDimension) {
      throw new IllegalArgumentException("upperBound array dimension must be equal to input dimension");
    }
    if (lowerBound.length != inputDimension) {
      throw new IllegalArgumentException("lowerBound array dimension must be equal to input dimension");
    }
    if (sizes.length != inputDimension) {
      throw new IllegalArgumentException("sizes array dimension must be equal to input dimension");
    }

    this.inputDimension = inputDimension;
    this.upperBound = upperBound;
    this.lowerBound = lowerBound;
    this.sizes = sizes;

    this.transformationIndexes = inverseCumulativeProduct(sizes);
  }

  @Override
  public Integer apply(double[] doubles) {
    int result = 0;
    for (int i = 0; i < inputDimension; i++) {
      // TODO Do we want this behavior?
      //if (doubles[i] < lowerBound[i] || doubles[i] > upperBound[i]) {
      //  // throw exception and print value
      //  throw new IllegalArgumentException("Input value out of bounds. Value: " + doubles[i]);
      //} else {
      int index = (int) Math.floor((doubles[i] - lowerBound[i]) / (upperBound[i] - lowerBound[i]) * sizes[i]);
      result += (index * transformationIndexes[i]);
      //}
    }

    return result;
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

  private int[] inverseCumulativeProduct(int[] a) {
    if (a.length == 0) {
      throw new IllegalArgumentException("inverseCumulativeProduct - Input array can't have 0 length!");
    }
    int[] result = new int[a.length];
    result[result.length - 1] = 1;
    for (int i = result.length - 2; i >= 0; i--) {
      result[i] = result[i + 1] * a[i + 1];
    }

    return result;
  }
}