package it.units.erallab.hmsrobots.core.controllers;

public interface DifferentiableRealFunction extends RealFunction {

    double[] gradient(double[] x);

}
