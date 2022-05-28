package it.units.erallab.hmsrobots.core.controllers.rl.discrete.converters;

import it.units.erallab.hmsrobots.core.controllers.rl.discrete.DiscreteRL;

import java.io.Serializable;
import java.util.Arrays;

public class InputConverter implements DiscreteRL.InputConverter, Serializable {

  private final int[] nSplits;
  private final int inputDimension;

  public InputConverter(
      int[] nSplits
  ) {
    this.nSplits = nSplits;
    this.inputDimension = nSplits.length;
  }

  @Override
  public Integer apply(double[] doubles) {
    if (doubles.length != inputDimension) {
      throw new IllegalArgumentException("Input dimension mismatch");
    }
    int[] ints = new int[inputDimension];
    for (int i = 0; i < inputDimension; i++) {
      doubles[i] = Math.max(0d, Math.min(1d, doubles[i]));
      int temp = 0;
      while (!(doubles[i] >= temp / (double) nSplits[i] && doubles[i] <= (temp + 1) / (double) nSplits[i])) {
        temp++;
      }
      ints[i] = temp;
    }
    int index = 0;
    int prod = Arrays.stream(nSplits).reduce(1, (a, b) -> a * b);
    for (int i = 0; i < inputDimension; i++) {
      index += ints[i] * prod / nSplits[i];
      prod /= nSplits[i];
    }
    return index;
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