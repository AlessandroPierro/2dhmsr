package it.units.erallab.hmsrobots.core.controllers.rl;

import it.units.erallab.hmsrobots.core.controllers.IOSized;
import it.units.erallab.hmsrobots.core.controllers.Resettable;
import it.units.erallab.hmsrobots.core.snapshots.Snapshottable;

public interface ContinuousRL extends IOSized, Snapshottable, Resettable {
  double[] apply(double t, double[] input, double r);

}
