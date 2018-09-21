package com.dbohdanov.opencvbytutorialcppfree;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.widget.ArrayAdapter;
import android.widget.ImageView;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;

public class MainActivity extends AppCompatActivity {
    public static final String TAG = "mainActTaag";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if (OpenCVLoader.initDebug()) {
            Log.i(TAG, "OpenCV initialize success");
        } else {
            Log.i(TAG, "OpenCV initialize failed");
        }


        //#1
        Bitmap img = getBitmapFromAsset(this);

        Mat mat = preprocessImg(img);


        ArrayList<MatOfPoint> contours =
                ImgprocUtils.findContours(mat, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);


        contours = contoursSort(contours);

        ArrayList<MatOfPoint> newContures = new ArrayList<>();
        for (MatOfPoint contour : contours) {
            double lenght = Imgproc.arcLength(new MatOfPoint2f(contour.toArray()), true);
            MatOfPoint2f shape = ImgprocUtils.approxPolyDP(new MatOfPoint2f(contour.toArray()), 0.02 * lenght, true);

//            if (shape.total() == 4) {
            newContures.add(new MatOfPoint(shape.toArray()));
//            }
//            }
        }

        Imgproc.drawContours(mat, newContures, -1, new Scalar(255, 125, 125, 255), 5);

        img = ImgprocUtils.matToBitmap(mat, img);

        ImageView imgView = findViewById(R.id.imageView);
        imgView.setImageBitmap(img);

    }


    private ArrayList<MatOfPoint> contoursSort(ArrayList<MatOfPoint> contours) {
        MatOfPoint[] contoursArray = new MatOfPoint[contours.size()];
        contoursArray = (MatOfPoint[]) contours.toArray();

        Arrays.sort(contoursArray, new Comparator<MatOfPoint>() {
            @Override
            public int compare(MatOfPoint o1, MatOfPoint o2) {
                return Double.compare(Imgproc.contourArea(o1), Imgproc.contourArea(o2));
            }
        });

        return new ArrayList<>(Arrays.asList(contoursArray));
    }

    private Mat preprocessImg(Bitmap img) {
        img = ImgprocUtils.resize(img);
        Mat mat = ImgprocUtils.getMatFromBitmap(img);
        mat = preprocessImg(mat);

        return mat;
    }

    private Mat preprocessImg(Mat mat) {
        ImgprocUtils.cvtColor(mat, Imgproc.COLOR_BGR2GRAY);
        mat = ImgprocUtils.gaussianBlur(mat, new Size(5, 5), 0);
        mat = ImgprocUtils.equalizeHist(mat);

        mat = ImgprocUtils.canny(mat, 20, 80);
//        mat = ImgprocUtils.canny(mat, 50, 200);

        return mat;
    }


    public void someMethods() {
//        double k = (Imgproc.contourArea(shape) / Imgproc.arcLength(shape, true) * Imgproc.arcLength(shape, true));
    }

    public static Bitmap getBitmapFromAsset(Context context) {
        return getBitmapFromAsset(context, "example.jpg");
//        return getBitmapFromAsset(context, "example2.png");
//        return getBitmapFromAsset(context, "dev1.jpg");
//        return getBitmapFromAsset(context, "dev1_cut.png");
//        return getBitmapFromAsset(context, "dev2.jpg");
//        return getBitmapFromAsset(context, "dev3.jpg");
//        return getBitmapFromAsset(context, "dev4.jpg");
//        return getBitmapFromAsset(context, "dev4_cut.png");
    }

    public static Bitmap getBitmapFromAsset(Context context, String filePath) {
        AssetManager assetManager = context.getAssets();

        InputStream istr;
        Bitmap bitmap = null;
        try {
            istr = assetManager.open(filePath);
            bitmap = BitmapFactory.decodeStream(istr);
        } catch (IOException e) {
            Log.d(TAG, "bitmap error: " + e.getMessage());
            e.printStackTrace();
        }

        return bitmap;
    }
}
