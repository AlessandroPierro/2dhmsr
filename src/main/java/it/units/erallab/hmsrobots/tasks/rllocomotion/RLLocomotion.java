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
import it.units.erallab.hmsrobots.core.controllers.rl.RewardFunction;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.snapshots.SnapshotListener;
import it.units.erallab.hmsrobots.tasks.AbstractTask;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.hmsrobots.viewers.GridFileWriter;
import it.units.erallab.hmsrobots.viewers.NamedValue;
import it.units.erallab.hmsrobots.viewers.VideoUtils;
import it.units.erallab.hmsrobots.viewers.drawers.Drawers;
import org.dyn4j.dynamics.Settings;

import java.io.File;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Collectors;

public class RLLocomotion extends AbstractTask<RewardFunction, RLEnsembleOutcome> {

  private final double maxTime;
  private final double maxEpisodeTime;
  private final int nAgents;
  private final Robot robot;

  private static final double VALIDATION_TIME = 30d;

  public RLLocomotion(double maxTime, double maxEpisodeTime, int nAgents, Robot robot) {
    super(new Settings());
    this.maxTime = maxTime;
    this.maxEpisodeTime = maxEpisodeTime;
    this.nAgents = nAgents;
    this.robot = robot;
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
      map = map.entrySet()
          .stream()
          .filter(e -> e.getKey() >= lastTime - 5)
          .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
      return map.values()
          .stream()
          .map(obs -> obs.voxelPolies()
              .stream()
              .filter(e -> e.key().y() == 0 && e.value() != null)
              .map(Grid.Entry::value)
              .findFirst()
              .get()
              .getAngle())
          .map(angle -> angle < -Math.PI / 2d || angle > Math.PI / 2d)
          .reduce(true, (a, b) -> a && b);
    };
  }

  @Override
  public RLEnsembleOutcome apply(RewardFunction rewardFunction, SnapshotListener listener) {
    Set<RLEnsembleOutcome.RLOutcome> outcomes = new HashSet<>();
    if (robot.getController() instanceof CompositeController cc) {
      if (cc.getInnermostController() instanceof RLController rlController) {
        rlController.setRewardFunction(rewardFunction);
        for (int i = 0; i < nAgents; i++) {
          double usedTime = 0;
          RLListener innerListener = new RLListener();
          rlController.getRL().reinitialize();
          while (usedTime < maxTime - 0.5) {
            Predicate<Map<Double, Outcome.Observation>> earlyStopping = makeStoppingCriterion(Math.min(
                maxEpisodeTime,
                maxTime - usedTime
            ));
            Locomotion locomotion = new Locomotion(earlyStopping, StarterRL.getTerrain(), 50000, new Settings());
            Outcome outcome = locomotion.apply(robot, innerListener);
            usedTime += outcome.getTime();
          }
          Locomotion locomotion = new Locomotion(VALIDATION_TIME, StarterRL.getTerrain(), 50000, new Settings());
          Outcome outcome = locomotion.apply(robot);


          Locomotion locomotionTest = new Locomotion(60, StarterRL.getTerrain(), 10000, new Settings());

          GridFileWriter.save(
              locomotionTest,
              Grid.create(1, 1, new NamedValue<>("robot", robot)),
              600, 600, 1, 24,
              VideoUtils.EncoderFacility.JCODEC,
              new File("ciaociao.mp4"),
              Drawers::basicWithMiniWorld
          );


          //locomotion = new Locomotion(70, StarterRL.getTerrain(), 50000, new Settings());
          //RLListener ll = new RLListener();
          //locomotion.apply(robot, ll);
          //File file = new File("validation_data.csv");
          //ll.toFile(file);
          double avgVelocity = outcome.getDistance() / outcome.getTime();
          outcomes.add(new RLEnsembleOutcome.RLOutcome(
              innerListener.extractVelocities().stream().toList(),
              innerListener.extractRewards().stream().toList(),
              avgVelocity
          ));
        }
      }
    }
    return new RLEnsembleOutcome(outcomes);
  }

  @Override
  public RLEnsembleOutcome apply(RewardFunction solution) {
    return apply(solution, null);
  }

}
