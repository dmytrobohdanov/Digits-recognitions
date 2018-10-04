package com.dbohdanov.opencvbytutorialcppfree.utils;

import com.dbohdanov.opencvbytutorialcppfree.ImgprocUtils;

import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;

/**
 *
 */
public class TempUtils {
    public static ArrayList<MatOfPoint> filterContoures(ArrayList<MatOfPoint> contours) {
        ArrayList<MatOfPoint> fourVerticesShapes = new ArrayList<>();

        for (MatOfPoint contour : contours) {
            double lenght = Imgproc.arcLength(new MatOfPoint2f(contour.toArray()), true);
            MatOfPoint2f shape = ImgprocUtils.approxPolyDP(new MatOfPoint2f(contour.toArray()), 0.02 * lenght, true);

            if (shape.total() == 6) {
                fourVerticesShapes.add(new MatOfPoint(shape.toArray()));
            }
        }

        return fourVerticesShapes;
//        return new ArrayList<>(Collections.singletonList(fourVerticesShapes.get(10)));
    }
}
