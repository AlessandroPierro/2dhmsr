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
package it.units.erallab.hmsrobots.core.sensors;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import it.units.erallab.hmsrobots.util.DoubleRange;

public class VelocityX extends AbstractSensor {

  @JsonProperty
  private final double maxVelocityNorm;

  @JsonCreator
  public VelocityX(
      @JsonProperty("maxVelocityNorm") double maxVelocityNorm
  ) {
    super(new DoubleRange[]{DoubleRange.of(-maxVelocityNorm, maxVelocityNorm)});
    this.maxVelocityNorm = maxVelocityNorm;
  }

  @Override
  public double[] sense(double t) {
    return new double[]{voxel.getLinearVelocity().x()};
  }

  @Override
  public String toString() {
    return "VelocityX{" +
        ", maxVelocityNorm=" + maxVelocityNorm +
        '}';
  }
}
