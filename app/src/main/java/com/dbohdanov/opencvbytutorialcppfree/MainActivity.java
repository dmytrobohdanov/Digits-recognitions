package com.dbohdanov.opencvbytutorialcppfree;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.widget.ImageView;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;

import static org.opencv.imgproc.Imgproc.MORPH_ELLIPSE;
import static org.opencv.imgproc.Imgproc.MORPH_OPEN;
import static org.opencv.imgproc.Imgproc.THRESH_BINARY_INV;
import static org.opencv.imgproc.Imgproc.THRESH_OTSU;

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

        //getting bitmap
        Bitmap img = getBitmapFromAsset(this);

        //pre processing image, preparing to recognition
        Mat mat = preprocessImg(img);


        //detecting contours
        ArrayList<MatOfPoint> contours =
                ImgprocUtils.findContours(mat, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        //creating holder of all 4-vertices shapes
        ArrayList<MatOfPoint> fourVerticesShapes = new ArrayList<>();

        for (MatOfPoint contour : contours) {
            double lenght = Imgproc.arcLength(new MatOfPoint2f(contour.toArray()), true);
            MatOfPoint2f shape = ImgprocUtils.approxPolyDP(new MatOfPoint2f(contour.toArray()), 0.02 * lenght, true);

            if (shape.total() == 4) {
                fourVerticesShapes.add(new MatOfPoint(shape.toArray()));
            }
        }


        //transformation
        //sorting of contours
        fourVerticesShapes = contoursSort(fourVerticesShapes);

        //getting last element, the biggest area shape, we presume that it is our display
        MatOfPoint2f foundedShape = new MatOfPoint2f(fourVerticesShapes.get(fourVerticesShapes.size() - 1).toArray());

        //getting array of points created shape, we presume there are 4 points
        Point[] sourceArray = foundedShape.toArray();

        //creating new array of points for transformation destination
        Point[] resultedArray = new Point[]{
                new Point(0, 0),
                new Point(0, sourceArray[1].y * 0.8),
                new Point(sourceArray[2].x, sourceArray[1].y * 0.8),
                new Point(sourceArray[2].x, 0)};

        //creating destination matrix
        MatOfPoint2f resultedMat = new MatOfPoint2f(resultedArray);

        //getting PerspectiveTransform matrix
        Mat perspectiveTransform = Imgproc.getPerspectiveTransform(foundedShape, resultedMat);

        //creating output matrix
        Mat outputMat = new Mat();

        //transformation
        Imgproc.warpPerspective(ImgprocUtils.getMatFromBitmap(img),
                outputMat,
                perspectiveTransform,
                new Size(sourceArray[2].x, sourceArray[1].y * 0.8));

        Imgproc.cvtColor(outputMat, outputMat, Imgproc.COLOR_BGRA2GRAY);

        Imgproc.threshold(outputMat, outputMat,
                0,
                255,
                THRESH_BINARY_INV | THRESH_OTSU);

        Mat kernel = Imgproc.getStructuringElement(MORPH_ELLIPSE, new Size(1, 5));
        Imgproc.morphologyEx(outputMat, outputMat, MORPH_OPEN, kernel);

        //creating output bitmap
        Bitmap outputBitmap = Bitmap.createBitmap(outputMat.cols(), outputMat.rows(), Bitmap.Config.ARGB_8888);

        outputBitmap = ImgprocUtils.matToBitmap(outputMat, outputBitmap);

        ImageView imgView = findViewById(R.id.imageView);
        imgView.setImageBitmap(outputBitmap);
    }


    private Mat preprocessImg(Bitmap img) {
//        img = ImgprocUtils.resize(img);
        Mat mat = ImgprocUtils.getMatFromBitmap(img);
        mat = preprocessImg(mat);

        return mat;
    }

    private Mat preprocessImg(Mat mat) {
        ImgprocUtils.cvtColor(mat, Imgproc.COLOR_BGR2GRAY);
        mat = ImgprocUtils.gaussianBlur(mat, new Size(5, 5), 0);
        mat = ImgprocUtils.equalizeHist(mat);

//        mat = ImgprocUtils.canny(mat, 20, 80);
        mat = ImgprocUtils.canny(mat, 50, 200);

        return mat;
    }

    /**
     * sorting contours
     *
     * @param contours array of contours to sort
     * @return sorted array of contours
     */
    private ArrayList<MatOfPoint> contoursSort(ArrayList<MatOfPoint> contours) {
        MatOfPoint[] contoursArray = contours.toArray(new MatOfPoint[contours.size()]);

        Arrays.sort(contoursArray, new Comparator<MatOfPoint>() {
            @Override
            public int compare(MatOfPoint o1, MatOfPoint o2) {
                return Double.compare(Imgproc.contourArea(o1), Imgproc.contourArea(o2));
            }
        });

        return new ArrayList<>(Arrays.asList(contoursArray));
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
