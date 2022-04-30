package it.units.erallab.hmsrobots.core.controllers.rl.discrete.converters;

import it.units.erallab.hmsrobots.core.controllers.rl.discrete.DiscreteRL;

import java.io.Serializable;

public class BinaryOutputConverter implements DiscreteRL.OutputConverter, Serializable {

  private final int outputDimension;
  private final double force;

  public BinaryOutputConverter(
      int outputDimension, double force
  ) {
    this.outputDimension = outputDimension;
    this.force = Math.abs(force);
  }

  public BinaryOutputConverter(
      int outputDimension
  ) {
    this(outputDimension, 0.5);
  }

  @Override
  public double[] apply(Integer integer) {
    double[] output = new double[outputDimension];
    for (int i = 0; i < outputDimension; i++) {
      output[i] = integer >= (int) Math.pow(2, outputDimension - 1 - i) ? force : -force;
      integer %= (int) Math.pow(2, outputDimension - 1 - i);
    }
    return output;
  }

  @Override
  public int getInputDimension() {
    return 1;
  }

  @Override
  public int getOutputDimension() {
    return outputDimension;
  }
}
