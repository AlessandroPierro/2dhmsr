package it.units.erallab.hmsrobots.core.controllers.rl;

import it.units.erallab.hmsrobots.core.controllers.IOSized;

public interface ContinuousRL extends IOSized {
  double[] apply(double t, double[] input, double r);


}
