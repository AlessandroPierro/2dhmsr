package it.units.erallab.hmsrobots.core.controllers.dqn;

import it.units.erallab.hmsrobots.util.Grid;

public record RLControllerState(double[] observation, double reward, double[] action, Grid<Double> control) {

}

