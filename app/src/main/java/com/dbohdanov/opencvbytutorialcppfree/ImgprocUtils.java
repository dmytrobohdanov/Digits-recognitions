package com.dbohdanov.opencvbytutorialcppfree;

import android.graphics.Bitmap;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;

/**
 *
 */
public class ImgprocUtils {
    public static Bitmap resize(Bitmap img) {
        return Bitmap.createScaledBitmap(img, (int) (img.getWidth() * 0.8), (int) (img.getHeight() * 0.8), true);
    }

    public static Mat getMatFromBitmap(Bitmap bmp) {
        Mat mat = new Mat();
        Bitmap bmp32 = bmp.copy(Bitmap.Config.ARGB_8888, true);
        Utils.bitmapToMat(bmp32, mat);
        return mat;
    }

    public static ArrayList<MatOfPoint> findContours(Mat source, Mat hierarchy, int mode, int method) {
        ArrayList<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(source, contours, hierarchy, mode, method);
        return contours;
    }

    public static Mat cvtColor(Mat source, int code) {
        Imgproc.cvtColor(source, source, code);
        return source;
    }

    public static Mat gaussianBlur(Mat source, Size ksize, double sigma) {
        Imgproc.GaussianBlur(source, source, ksize, sigma);
        return source;
    }

    public static Mat canny(Mat sourse, int threshold1, int threshold2) {
        Imgproc.Canny(sourse, sourse, threshold1, threshold2);
        return sourse;
    }

    public static Bitmap matToBitmap(Mat source, Bitmap resultedBtm) {
        Utils.matToBitmap(source, resultedBtm);
        return resultedBtm;
    }

    public static MatOfPoint2f approxPolyDP(MatOfPoint2f curve, double epsilon, boolean closed) {
        MatOfPoint2f resultedMat = new MatOfPoint2f();
        Imgproc.approxPolyDP(new MatOfPoint2f(curve.toArray()), resultedMat, epsilon, closed);
        return resultedMat;
    }

    public static Mat equalizeHist(Mat source) {
        Imgproc.equalizeHist(source, source);
        return source;
    }
}
