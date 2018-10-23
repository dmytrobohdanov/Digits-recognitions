package com.dbohdanov.opencvbytutorialcppfree.utils;

import android.graphics.Bitmap;
import android.util.Log;

import com.dbohdanov.opencvbytutorialcppfree.ImgprocUtils;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;

import static com.dbohdanov.opencvbytutorialcppfree.utils.TextRecognitionUtils.findContours;
import static com.dbohdanov.opencvbytutorialcppfree.utils.TextRecognitionUtils.whatNumberIsThis;
import static org.opencv.imgproc.Imgproc.MORPH_ELLIPSE;
import static org.opencv.imgproc.Imgproc.MORPH_OPEN;
import static org.opencv.imgproc.Imgproc.THRESH_BINARY_INV;
import static org.opencv.imgproc.Imgproc.THRESH_OTSU;

/**
 *
 */
public class TempUt {

//    public static String getNumbersFromShapes(Mat cuttedImgMat) {
//        Bitmap bitmap = Bitmap.createBitmap(cuttedImgMat.cols(), cuttedImgMat.rows(), Bitmap.Config.ARGB_8888);
//        bitmap = ImgprocUtils.matToBitmap(cuttedImgMat, bitmap);
//
//        //looking for contours
//        ArrayList<MatOfPoint> contours = findContours(cuttedImgMat);
//
//        contours = filterSmallOnes(contours);
//
//        StringBuilder stringBuilder = new StringBuilder();
//
//        if (contours.size() == 0) {
//            return "";
//        } else {
//            for (MatOfPoint contour : contours) {
//                MainActivity.digitContoures.add(contour);
//
//                MatOfPoint2f foundedShape = new MatOfPoint2f(contour.toArray());
//
//                Mat tempMat = TempUt.getTransformedAndCutImageByShape(bitmap, foundedShape);
//
//                ArrayList<MatOfPoint> mayBeDigits = findContours(tempMat);
//
//                for (MatOfPoint mayBeDigit : mayBeDigits) {
//                    Rect rect = Imgproc.boundingRect(mayBeDigit);
//
//                    Mat submat = cuttedImgMat.submat(rect);
////                    if (submat.cols() <= 50 || submat.rows() <= 50 ) {
////                    if (submat.cols() <= 50 || submat.rows() <= 50
////                            || ((double)submat.cols()/ submat.rows() >= 0.8)
////                            || ((double)submat.cols()/ submat.rows() <= 0.2)) {
////                        continue;
////                    }
//
//
//                    int number = whatNumberIsThis(submat);
//
//                    MainActivity.digitContoures.add(mayBeDigit);
//                    MainActivity.counter++;
//
//                    if (number >= 0) {
//                        stringBuilder.append(number).append(" ");
//                    } else {
//                        stringBuilder.append(getNumbersFromShapes(submat));
//                    }
//
//                }
//            }
//        }
//
//        return stringBuilder.toString();
//    }

    public static String getNumbersFromShapes(Mat cuttedImgMat) {
        Bitmap bitmap = Bitmap.createBitmap(cuttedImgMat.cols(), cuttedImgMat.rows(), Bitmap.Config.ARGB_8888);
        bitmap = ImgprocUtils.matToBitmap(cuttedImgMat, bitmap);

        //looking for contours
        ArrayList<MatOfPoint> contours = findContours(cuttedImgMat);

        contours = filterSmallOnes(contours);

        StringBuilder stringBuilder = new StringBuilder();

        if (contours.size() == 0) {
            return "";
        } else {
            for (MatOfPoint contour : contours) {

//                MatOfPoint2f foundedShape = new MatOfPoint2f(contour.toArray());
                Mat submat = TempUt.getTransformedAndCutImageByShape(bitmap, new MatOfPoint2f(contour.toArray()));

                if (submat.cols() <= 10 || submat.rows() <= 10) {
//                    if (submat.cols() <= 50 || submat.rows() <= 50
//                            || ((double)submat.cols()/ submat.rows() >= 0.8)
//                            || ((double)submat.cols()/ submat.rows() <= 0.2)) {
                    continue;
                }


                int number = whatNumberIsThis(submat);

                if (number >= 0) {
                    stringBuilder.append(number).append(" ");
                } else {
                    stringBuilder.append(getNumbersFromShapes(submat));
                }

            }
        }
        return stringBuilder.toString();
    }

    private static Mat getTransformedAndCutImageByShape(Bitmap sourceImg, MatOfPoint2f foundedShape) {
        Rect rect = Imgproc.boundingRect(new MatOfPoint(foundedShape.toArray()));

        Point[] sourceArray = new Point[]{new Point(rect.x, rect.y),
                new Point(rect.x, rect.y + rect.height),
                new Point(rect.x + rect.width, rect.y + rect.height),
                new Point(rect.x + rect.width, rect.y)};
        MatOfPoint2f sourceMat = new MatOfPoint2f(sourceArray);

        //creating new array of points for transformation destination
        Point[] resultedArray = new Point[]{
                new Point(0, 0),
                new Point(0, sourceArray[1].y * 0.8),
                new Point(sourceArray[2].x, sourceArray[1].y * 0.8),
                new Point(sourceArray[2].x, 0)};


        //creating destination matrix
        MatOfPoint2f resultedMat = new MatOfPoint2f(resultedArray);

        Log.d("someCoolTag", "res " + resultedMat.toString());
        Log.d("someCoolTag", "founded " + foundedShape.toString());

        //getting PerspectiveTransform matrix
        Mat perspectiveTransform = Imgproc.getPerspectiveTransform(sourceMat, resultedMat);

        //creating output matrix
        Mat cuttedImgMat = new Mat();

        //transformation
        Imgproc.warpPerspective(ImgprocUtils.getMatFromBitmap(sourceImg),
                cuttedImgMat,
                perspectiveTransform,
                new Size(sourceArray[2].x, sourceArray[1].y * 0.8));

        Imgproc.cvtColor(cuttedImgMat, cuttedImgMat, Imgproc.COLOR_BGRA2GRAY);

        Imgproc.threshold(cuttedImgMat, cuttedImgMat,
                0,
                255,
                THRESH_BINARY_INV | THRESH_OTSU);

        Mat kernel = Imgproc.getStructuringElement(MORPH_ELLIPSE, new Size(1, 5));
        Imgproc.morphologyEx(cuttedImgMat, cuttedImgMat, MORPH_OPEN, kernel);

        return cuttedImgMat;
    }


    public static ArrayList<MatOfPoint> filterSmallOnes(ArrayList<MatOfPoint> contours) {
        ArrayList<MatOfPoint> bigOnes = new ArrayList<>();

        for (MatOfPoint contour : contours) {
            if (Imgproc.contourArea(contour) >= 50) {
                bigOnes.add(contour);
            }
        }

        return bigOnes;
    }

    public static int[] changeCOlorsTOOpposite(int[] segArray) {
        for (int i = 0; i < segArray.length; i++) {
            segArray[i] = segArray[i] ^ 255;
        }
        return segArray;
    }
}
