package it.units.erallab.hmsrobots.core.controllers.rl.discrete.converters;

import it.units.erallab.hmsrobots.core.controllers.rl.discrete.DiscreteRL;

import java.io.Serializable;

public class BinaryInputConverter implements DiscreteRL.InputConverter, Serializable {

  private final int inputDimension;
  private final double splitValue;

  public BinaryInputConverter(
      int inputDimension,
      double splitValue
  ) {
    this.inputDimension = inputDimension;
    this.splitValue = splitValue;
  }

  public BinaryInputConverter(
      int inputDimension
  ) {
    this(inputDimension, 0.5);
  }

  @Override
  public Integer apply(double[] doubles) {
    int input = 0;
    for (int i = 0; i < inputDimension; i++) {
      input += doubles[i] < splitValue ? 0 : (int) Math.pow(2, i);
    }
    return input;
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