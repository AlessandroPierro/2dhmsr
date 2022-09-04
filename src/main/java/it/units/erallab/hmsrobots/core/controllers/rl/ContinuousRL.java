package it.units.erallab.hmsrobots.core.controllers.rl;

import it.units.erallab.hmsrobots.core.controllers.Resettable;
import it.units.erallab.hmsrobots.core.snapshots.Snapshottable;

import java.io.Serializable;
import java.util.function.BiFunction;

public abstract class ContinuousRL implements Resettable, Serializable, Snapshottable, BiFunction<double[], Double, double[]> {

}
