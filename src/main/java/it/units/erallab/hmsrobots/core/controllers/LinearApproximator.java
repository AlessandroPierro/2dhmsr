package it.units.erallab.hmsrobots.core.controllers;

public class LinearApproximator implements DifferentiableRealFunction {

  private final double[] coefficients;

  public LinearApproximator(double[] coefficients) {
    this.coefficients = coefficients;
  }

  @Override
  public double[] apply(double[] input) {
    if (input.length != coefficients.length) {
      throw new IllegalArgumentException("LinearApproximator - Input dimension mismatch");
    }

    double output = 0;
    for (int i = 0; i < coefficients.length; i++) {
      output += coefficients[i] * input[i];
    }
    return new double[]{output};
  }

  public double[] getCoefficients() {
    return coefficients;
  }

  @Override
  public int getInputDimension() {
    return coefficients.length;
  }

  @Override
  public int getOutputDimension() {
    return 1;
  }

  @Override
  public double[] gradient(double[] x) {
    if (x.length != coefficients.length) {
      throw new IllegalArgumentException("LinearApproximator - Input dimension mismatch");
    }
    return x;
  }

}
