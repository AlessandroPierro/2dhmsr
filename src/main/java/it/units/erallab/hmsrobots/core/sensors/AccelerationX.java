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
import it.units.erallab.hmsrobots.util.DoubleRange;

public class AccelerationX extends AbstractSensor {

    private static final AbstractSensor sensor = new VelocityX(4d);

    @JsonCreator
    public AccelerationX(
    ) {
        super(new DoubleRange[]{new DoubleRange(0, 1)});
    }

    @Override
    public double[] sense(double t) {
        sensor.setVoxel(voxel);
        return sensor.sense(t);
    }

    @Override
    public String toString() {
        return "AccelerationX{}";
    }
}
