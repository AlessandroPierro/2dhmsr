package it.units.erallab.hmsrobots.core.controllers;

/**
 * @author eric on 2021/03/09 for 2dhmsr
 */
public interface TimedRealFunction extends IOSized {
  double[] apply(double t, double[] input);
}
