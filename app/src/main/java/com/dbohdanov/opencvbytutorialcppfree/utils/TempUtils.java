package com.dbohdanov.opencvbytutorialcppfree.utils;

import android.util.Log;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;

import static org.opencv.imgproc.Imgproc.MORPH_ELLIPSE;
import static org.opencv.imgproc.Imgproc.MORPH_OPEN;
import static org.opencv.imgproc.Imgproc.THRESH_BINARY_INV;
import static org.opencv.imgproc.Imgproc.THRESH_OTSU;

/**
 *
 */
public class TempUtils {
    public static final String TAG = "taag";

    public static String getNumbersFromShapes(Mat cuttedImgMat) {
        //looking for contours
        ArrayList<MatOfPoint> contours = TextRecognitionUtils.findContours(cuttedImgMat);
        Log.d("someCOolTAg", "contours " + contours.size());

        //filtering too small ones
        contours = filterSmallOnes(contours);

        StringBuilder stringBuilder = new StringBuilder();

        //if there is no more contours return empty string
        if (contours.size() == 0) {
            Log.d(TAG, "returning");
            return "";
        } else {
            //create array with matrixes for next iteration
            ArrayList<Mat> forNextIteration = new ArrayList<>();

            // get matrix
            ArrayList<Mat> potentialDigits = getListOfMatsByShapes(cuttedImgMat, contours);

            for (Mat potentialDigit : potentialDigits) {
                potentialDigit = preprocessSubImg(potentialDigit);

                int recognNum = TextRecognitionUtils.whatNumberIsThis(potentialDigit);

                if (recognNum >= 0) {
                    stringBuilder.append(recognNum).append(" ");
                } else {
                    forNextIteration.add(potentialDigit);
                }
            }

            for (Mat mat : forNextIteration) {
                stringBuilder.append(getNumbersFromShapes(mat));
            }
        }

        return stringBuilder.toString();
    }

    private static Mat preprocessSubImg(Mat potentialDigit) {
        Imgproc.threshold(potentialDigit, potentialDigit,
                0,
                255,
                THRESH_BINARY_INV | THRESH_OTSU);

        Mat kernel = Imgproc.getStructuringElement(MORPH_ELLIPSE, new Size(1, 5));
        Imgproc.morphologyEx(potentialDigit, potentialDigit, MORPH_OPEN, kernel);

        return potentialDigit;
    }


    private static ArrayList<Mat> getListOfMatsByShapes(Mat cuttedImgMat, ArrayList<MatOfPoint> digitContoures) {
        ArrayList<Mat> subMats = new ArrayList<>();
        for (MatOfPoint digitContoure : digitContoures) {
            //get rectangle around contour
            Rect rect = Imgproc.boundingRect(digitContoure);

            //get submatrix bounded this contour
            Mat submat = cuttedImgMat.submat(rect);
            if (submat.cols() <= 20 || submat.rows() <= 20) {
                continue;
            }

            subMats.add(submat);
        }

        return subMats;
    }

    private static ArrayList<MatOfPoint> filterSmallOnes(ArrayList<MatOfPoint> contours) {
        ArrayList<MatOfPoint> bigOnes = new ArrayList<>();

        for (MatOfPoint contour : contours) {
            if (Imgproc.contourArea(contour) >= 50) {
                bigOnes.add(contour);
            }
        }

        return bigOnes;
    }
}
