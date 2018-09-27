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
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;

import static org.opencv.imgproc.Imgproc.CHAIN_APPROX_SIMPLE;
import static org.opencv.imgproc.Imgproc.MORPH_ELLIPSE;
import static org.opencv.imgproc.Imgproc.MORPH_OPEN;
import static org.opencv.imgproc.Imgproc.RETR_EXTERNAL;
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
                ImgprocUtils.findContours(mat, new Mat(), RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

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
        fourVerticesShapes = contoursSortAscending(fourVerticesShapes);

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
        Mat cuttedImgMat = new Mat();

        //transformation
        Imgproc.warpPerspective(ImgprocUtils.getMatFromBitmap(img),
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

        ArrayList<MatOfPoint> digitContoures =
                ImgprocUtils.findContours(cuttedImgMat.clone(),
                        new Mat(),
                        RETR_EXTERNAL,
                        CHAIN_APPROX_SIMPLE);

//        drawRectAroundContours(digitContoures, cuttedImgMat);

        //sorting from left to right
        digitContoures = contoursSortLeftToRight(digitContoures);

        //---------------
        //looking through each contour
        for (MatOfPoint digitContoure : digitContoures) {
            //get rectangle around contour
            Rect rect = Imgproc.boundingRect(digitContoure);
            //get submatrix bounded this contour
            Mat submat = cuttedImgMat.submat(rect);
            double[] temp = submat.get(0, 0);
            Log.d(TAG, "sizes: " + submat.cols() + " " + submat.rows());
            if (submat.cols() <= 50 || submat.rows() <= 50 || submat.rows() >= 200) {
                continue;
            }

            int number = whatNumberIsThis(submat);

        }
        //---------------


        //creating output bitmap
        Bitmap outputBitmap = Bitmap.createBitmap(cuttedImgMat.cols(), cuttedImgMat.rows(), Bitmap.Config.ARGB_8888);

        outputBitmap = ImgprocUtils.matToBitmap(cuttedImgMat, outputBitmap);

        ImageView imgView = findViewById(R.id.imageView);
        imgView.setImageBitmap(outputBitmap);
    }

    /**
     * detecting what is number is in the matrix
     * expect 7 segment digit in this matrix
     *
     * @param mat matrix to detect the number
     * @return possible number [0 - 9] or negative value if it is not number in matrix
     */
    private int whatNumberIsThis(Mat mat) {
        //creating 3 control lines:
        // one is in the middle column to detect horizontal segments of number
        // second is row to detect upper vertical segments
        // third one is row to detect down vertical segments
        int ctrlColumnNum = mat.cols() / 2;
        int ctrlRowUpperNum = mat.rows() / 4;
        int ctrlRowDownNum = ctrlRowUpperNum * 3;

        int[] ctrlColArray = getPixelsPatternFromCol(mat, ctrlColumnNum);
        int[] ctrlRowUpArray = getPixelsArrayByRow(mat, ctrlRowUpperNum);
        int[] ctrlRowDownArray = getPixelsArrayByRow(mat, ctrlRowDownNum);

        //plan
        // взять 3 линии пикселей, с обрезанными gap краями
        // перевести пиксели в тру-фолс, с учетом:
        // - считать true/false только в заданных 3 прямыми сегментах
        // перевести тру-фолс в инт-паттерн
        // или сразу 3 паттерна в общий паттерн
        // сравнить с имеющеющимеся паттернами цифр
        // сделать строку

        int v = getSegmentsNumberFromArray(ctrlRowUpArray);

        return 0;
    }

    private int getSegmentsNumberFromArray(int[] array) {
        boolean flag;
        for (int i = 0; i < array.length; i++) {
        }
        return 0;
    }

    private int getPixelsPatternFromCol(Mat mat, int ctrlColumnNum) {

        //gap number to cut border's pixel which are unpredictable
        int gap = 5;

        //init flag value as first in array
        double lastVal = mat.get(gap, ctrlColumnNum)[0];

        for (int i = gap; i < mat.rows() - gap; i++) {
            double current = mat.get(i, ctrlColumnNum)[0];
            if (current != lastVal) {

                lastVal = current;
            }
        }

        return 0;
    }

    private int[] getPixelsArrayByRow(Mat mat, int ctrlRowNum) {
        //gap number to cut border's pixel which are unpredictable
        int gap = 5;

        int[] array = new int[mat.cols() - gap * 2];

        for (int i = gap; i < (mat.cols() - gap); i++) {
            array[i - gap] = (int) mat.get(ctrlRowNum, i)[0];
        }

        return array;
    }


    private void drawRectAroundContours(ArrayList<MatOfPoint> contours, Mat imageMap) {
        MatOfPoint2f approxCurve = new MatOfPoint2f();


        //For each contour found
        for (int i = 0; i < contours.size(); i++) {
            //Convert contours(i) from MatOfPoint to MatOfPoint2f
            MatOfPoint2f contour2f = new MatOfPoint2f(contours.get(i).toArray());

            //Processing on mMOP2f1 which is in type MatOfPoint2f
            double approxDistance = Imgproc.arcLength(contour2f, true) * 0.02;
            Imgproc.approxPolyDP(contour2f, approxCurve, approxDistance, true);

            //Convert back to MatOfPoint
            MatOfPoint points = new MatOfPoint(approxCurve.toArray());

            // Get bounding rect of contour
            Rect rect = Imgproc.boundingRect(points);

            // draw enclosing rectangle (all same color, but you could use variable i to make them unique)
            Imgproc.rectangle(imageMap,
                    new Point(rect.x, rect.y),
                    new Point(rect.x + rect.width, rect.y + rect.height),
                    new Scalar(255, 0, 0, 255),
                    3);
        }

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
     * sorting contours left to right
     *
     * @param contours array of contours to sort
     * @return sorted array of contours
     */
    private ArrayList<MatOfPoint> contoursSortLeftToRight(ArrayList<MatOfPoint> contours) {
        MatOfPoint[] contoursArray = contours.toArray(new MatOfPoint[contours.size()]);

        Arrays.sort(contoursArray, new Comparator<MatOfPoint>() {
            @Override
            public int compare(MatOfPoint o1, MatOfPoint o2) {
                return Double.compare(o1.toArray()[0].x, o2.toArray()[0].x);
            }
        });

        return new ArrayList<>(Arrays.asList(contoursArray));
    }

    /**
     * sorting contours ascending
     *
     * @param contours array of contours to sort
     * @return sorted array of contours
     */
    private ArrayList<MatOfPoint> contoursSortAscending(ArrayList<MatOfPoint> contours) {
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
