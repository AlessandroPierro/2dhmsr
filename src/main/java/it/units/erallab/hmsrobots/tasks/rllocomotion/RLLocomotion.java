/*
 * Copyright (C) 2021 Eric Medvet <eric.medvet@gmail.com> (as Eric Medvet <eric.medvet@gmail.com>)
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package it.units.erallab.hmsrobots.tasks.rllocomotion;

import it.units.erallab.hmsrobots.StarterRL;
import it.units.erallab.hmsrobots.core.controllers.CompositeController;
import it.units.erallab.hmsrobots.core.controllers.rl.RLController;
import it.units.erallab.hmsrobots.core.controllers.rl.RLListener;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.core.snapshots.SnapshotListener;
import it.units.erallab.hmsrobots.tasks.AbstractTask;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import org.dyn4j.dynamics.Settings;

import java.io.File;
import java.util.Map;
import java.util.function.Predicate;
import java.util.function.ToDoubleFunction;
import java.util.stream.Collectors;

public class RLLocomotion extends AbstractTask<ToDoubleFunction<Grid<Voxel>>, RLOutcome> {

    private final double maxTime;
    private final int nAgents;
    private final Robot robot;
    private final boolean detailedLog;

    public RLLocomotion(double maxTime, int nAgents, Robot robot, boolean detailedLog) {
        super(new Settings());
        this.maxTime = maxTime;
        this.nAgents = nAgents;
        // TODO : Serialize/deserialize robot
        this.robot = robot;
        this.detailedLog = detailedLog;
    }

    Predicate<Map<Double, Outcome.Observation>> makeStoppingCriterion(double remainingTime) {
        return map -> {
            if (map.isEmpty()) {
                return false;
            }
            double lastTime = map.keySet().stream().max(Double::compareTo).orElse(-1d);
            if (lastTime > remainingTime) {
                return true;
            }
            map = map.entrySet().stream().filter(e -> e.getKey() >= lastTime - 5).collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
            return map.values().stream().map(obs -> obs.voxelPolies().stream().filter(e -> e.key().y() == 0 && e.value() != null).map(Grid.Entry::value).findFirst().get().getAngle()).map(angle -> angle < -Math.PI / 2d || angle > Math.PI / 2d).reduce(true, (a, b) -> a && b);
        };
    }

    @Override
    public RLOutcome apply(ToDoubleFunction<Grid<Voxel>> rewardFunction, SnapshotListener listener) {
        double maxVelocity = Double.MIN_VALUE;
        double meanVelocity = 0d;
        double minVelocity = Double.MAX_VALUE;
        if (robot.getController() instanceof CompositeController cc) {
            if (cc.getInnermostController() instanceof RLController rlController) {
                rlController.setRewardFunction(rewardFunction);
                for (int i = 0; i < nAgents; i++) {
                    int counter = 0;
                    // TODO : remove this
                    System.out.println("Agent " + i);
                    rlController.getRL().reinitialize();
                    double usedTime = 0;
                    while (usedTime < maxTime - 0.5) {
                        counter++;
                        final double remainingTime = maxTime - usedTime;
                        // TODO : remove this
                        System.out.println("New episode" + " remaining time: " + remainingTime);
                        Predicate<Map<Double, Outcome.Observation>> earlyStopping = makeStoppingCriterion(200d);
                        Locomotion locomotion = new Locomotion(earlyStopping, StarterRL.getTerrain(), 50000, new Settings());
                        RLListener innerListener = detailedLog ? new RLListener() : null;
                        Outcome outcome = locomotion.apply(robot, listener);
                        if (detailedLog) {
                            File file = new File("logs/rl/rl_" + i + "_" + counter + ".csv");
                            innerListener.toFile(file);
                        }
                        usedTime += outcome.getTime();
                    }
                    // TODO : remove this
                    System.out.println("Agent " + i + " finished after " + counter + " episodes");

                    Locomotion locomotion = new Locomotion(60, StarterRL.getTerrain(), 50000, new Settings());
                    Outcome outcome = locomotion.apply(robot);
                    double velocity = outcome.getDistance() / outcome.getTime();
                    meanVelocity = meanVelocity + velocity;
                    maxVelocity = Math.max(maxVelocity, velocity);
                    minVelocity = Math.min(minVelocity, velocity);
                }
                // TODO : remove testing
                //Locomotion locomotion = new Locomotion(60, StarterRL.getTerrain(), 5000, new Settings());
                //GridOnlineViewer.run(locomotion, Grid.create(1, 1, new NamedValue<>("SARSA(Lambda)", robot)), Drawers::basicWithMiniWorldAndRL);

            }
        }

        System.out.println("Mean velocity: " + meanVelocity / nAgents);
        System.out.println("Max velocity: " + maxVelocity);
        System.out.println("Min velocity: " + minVelocity);
        return new RLOutcome(minVelocity, maxVelocity, meanVelocity / nAgents);
    }

    @Override
    public RLOutcome apply(ToDoubleFunction<Grid<Voxel>> solution) {
        return apply(solution, null);
    }

}
