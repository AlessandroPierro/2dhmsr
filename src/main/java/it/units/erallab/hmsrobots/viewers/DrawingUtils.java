/*
 * Copyright (c) "Eric Medvet" 2021.
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

package it.units.erallab.hmsrobots.viewers;

import it.units.erallab.hmsrobots.core.geometry.BoundingBox;
import it.units.erallab.hmsrobots.core.geometry.Point2;
import it.units.erallab.hmsrobots.core.geometry.Poly;
import it.units.erallab.hmsrobots.core.snapshots.Snapshot;
import it.units.erallab.hmsrobots.viewers.drawers.Drawer;

import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.geom.Path2D;
import java.awt.geom.Rectangle2D;

/**
 * @author "Eric Medvet" on 2021/08/29 for 2dhmsr
 */
public class DrawingUtils {

  private DrawingUtils() {
  }

  public static Color linear(final Color c1, final Color c2, final Color c3, float x1, float x2, float x3, float x) {
    if (x < x2) {
      return linear(c1, c2, x1, x2, x);
    }
    return linear(c2, c3, x2, x3, x);
  }

  public static Color linear(final Color c1, final Color c2, float min, float max, float x) {
    x = (x - min) / (max - min);
    x = Float.max(0f, Float.min(1f, x));
    final float r1 = c1.getRed() / 255f;
    final float g1 = c1.getGreen() / 255f;
    final float b1 = c1.getBlue() / 255f;
    final float a1 = c1.getAlpha() / 255f;
    final float r2 = c2.getRed() / 255f;
    final float g2 = c2.getGreen() / 255f;
    final float b2 = c2.getBlue() / 255f;
    final float a2 = c2.getAlpha() / 255f;
    final float r = r1 + (r2 - r1) * x;
    final float g = g1 + (g2 - g1) * x;
    final float b = b1 + (b2 - b1) * x;
    final float a = a1 + (a2 - a1) * x;
    return new Color(r, g, b, a);
  }

  public static Path2D toPath(Poly poly, boolean close) {
    Path2D path = toPath(poly.getVertexes());
    if (close) {
      path.closePath();
    }
    return path;
  }

  public static Path2D toPath(Point2... points) {
    Path2D path = new Path2D.Double();
    path.moveTo(points[0].x, points[0].y);
    for (int i = 1; i < points.length; i++) {
      path.lineTo(points[i].x, points[i].y);
    }
    return path;
  }

  public static Color alphaed(Color color, float alpha) {
    return new Color(
        (float) color.getRed() / 255f,
        (float) color.getGreen() / 255f,
        (float) color.getBlue() / 255f,
        alpha);
  }

  public static void draw(double t, Snapshot snapshot, Graphics2D g, BoundingBox boundingBox, Drawer drawer) {
    AffineTransform originalTransform = g.getTransform();
    Shape shape = g.getClip();
    double clipX = shape.getBounds2D().getX();
    double clipY = shape.getBounds2D().getY();
    double clipW = shape.getBounds2D().getWidth();
    double clipH = shape.getBounds2D().getHeight();
    g.clip(new Rectangle2D.Double(
        clipX + boundingBox.min.x * clipW,
        clipY + boundingBox.min.y * clipH,
        clipW * boundingBox.width(),
        clipH * boundingBox.height()
    ));
    AffineTransform transform = new AffineTransform();
    transform.translate(g.getClip().getBounds2D().getX(), g.getClip().getBounds2D().getY());
    g.setColor(Color.WHITE);
    g.fill(g.getClip());
    g.setTransform(transform);
    //draw
    drawer.draw(t, snapshot, g);
    //restore clip and transform
    g.setTransform(originalTransform);
    g.setClip(shape);
  }

  public static Stroke getScaleIndependentStroke(float thickness, float scale) {
    return new BasicStroke(thickness / scale);
  }
}
