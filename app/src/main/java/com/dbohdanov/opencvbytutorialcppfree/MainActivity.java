package com.dbohdanov.opencvbytutorialcppfree;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import com.dbohdanov.opencvbytutorialcppfree.utils.TextRecognitionUtils;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

import static com.dbohdanov.opencvbytutorialcppfree.utils.TextRecognitionUtils.contoursSortAscending;
import static com.dbohdanov.opencvbytutorialcppfree.utils.TextRecognitionUtils.contoursSortLeftToRight;
import static com.dbohdanov.opencvbytutorialcppfree.utils.TextRecognitionUtils.findContours;
import static com.dbohdanov.opencvbytutorialcppfree.utils.TextRecognitionUtils.preprocessImg;

public class MainActivity extends AppCompatActivity {
    public static final String TAG = "taag";

    /**
     * getting bitmap from one of the pre-set images
     */
    public static Bitmap getBitmapFromAsset(Context context) {
//        return getBitmapFromAsset(context, "ex_cond_26_not_cut.jpg");
//        return getBitmapFromAsset(context, "ex_cond_26_cutted.png");
//        return getBitmapFromAsset(context, "ex_cond_cuted.png");

//        return getBitmapFromAsset(context, "example.jpg");
//        return getBitmapFromAsset(context, "example2.png");
//        return getBitmapFromAsset(context, "proto.png");
        return getBitmapFromAsset(context, "proto2.png");

//        return getBitmapFromAsset(context, "dev1.jpg");
//        return getBitmapFromAsset(context, "dev1_cut.png");
//        return getBitmapFromAsset(context, "dev2.jpg");
//        return getBitmapFromAsset(context, "dev3.jpg");
//        return getBitmapFromAsset(context, "dev4.jpg");
//        return getBitmapFromAsset(context, "dev4_cut.png");
//        return getBitmapFromAsset(context, "dev4_cut0.png");
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

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //initializing OpenCV
        if (OpenCVLoader.initDebug()) {
            Log.i(TAG, "OpenCV initialize success");
        } else {
            Log.i(TAG, "OpenCV initialize failed");
            return;
        }

        //getting bitmap
        Bitmap img = getBitmapFromAsset(this);

        //pre processing image, preparing to recognition
        Mat mat = preprocessImg(img);

        //detecting contours
        ArrayList<MatOfPoint> contours = findContours(mat);

        //creating holder of all 4-vertices shapes
        ArrayList<MatOfPoint> fourVerticesShapes = TextRecognitionUtils.filterNot4VerticesContoures(contours);

        //transformation
        //sorting of contours
        fourVerticesShapes = contoursSortAscending(fourVerticesShapes);

        //getting last element, the biggest area shape, we presume that it is our display
        MatOfPoint2f foundedShape = new MatOfPoint2f(fourVerticesShapes.get(fourVerticesShapes.size() - 1).toArray());

        Mat cuttedImgMat = TextRecognitionUtils.getTransformedAndCutImageByShape(img, foundedShape);

        ArrayList<MatOfPoint> digitContoures = findContours(cuttedImgMat);


        //sorting from left to right
        digitContoures = contoursSortLeftToRight(digitContoures);

        String resultedString = TextRecognitionUtils.recognizeDigits(cuttedImgMat, digitContoures);

        TextView textView = findViewById(R.id.text);
        textView.setText(resultedString);


        //creating output bitmap
        Bitmap outputBitmap = Bitmap.createBitmap(cuttedImgMat.cols(), cuttedImgMat.rows(), Bitmap.Config.ARGB_8888);

        outputBitmap = ImgprocUtils.matToBitmap(cuttedImgMat, outputBitmap);


        ImageView imgView = findViewById(R.id.imageView);
        imgView.setImageBitmap(outputBitmap);

//--------------------------------------------------------------------------------------------------
// here are some another variants, but basic idea is the same:
        // - got image
        // - pre process image
        // - find contours
        // - try to recognize is each contour is some digit

        // ------------------------------2--------------------------------

//        ArrayList<MatOfPoint> justToPrint= TextRecognitionUtils.findContours(mat);
//        drawRectAroundContours(justToPrint, mat);
//
//        Bitmap outputBitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
//        outputBitmap = ImgprocUtils.matToBitmap(mat, outputBitmap);
//
//        ImageView imgView = findViewById(R.id.imageView);
//        imgView.setImageBitmap(outputBitmap);

//        String s = TempUt.getNumbersFromShapes(mat);
//
////        digitContoures = TextRecognitionUtils.findContours(mat);
//
//        drawRectAroundContours(digitContoures, mat);
//        Log.d(TAG, "counter " + counter);
//
//        Bitmap outputBitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
//        outputBitmap = ImgprocUtils.matToBitmap(mat, outputBitmap);
//
//        ImageView imgView = findViewById(R.id.imageView);
//        imgView.setImageBitmap(outputBitmap);
//
//        TextView textView = findViewById(R.id.text);
//        textView.setText(s);
//        Log.d(TAG, "result " + s);
//        //--------------------------------------------


        //-----------------------------------3------------------------------------------
//        Mat mat2 = ImgprocUtils.getMatFromBitmap(img);
//
//        Imgproc.cvtColor(mat2, mat2, Imgproc.COLOR_BGRA2GRAY);
//
//        //без этой штуки почемуто работает лчше
////        mat2 = ImgprocUtils.equalizeHist(mat2);
//
////        Imgproc.medianBlur(mat2, mat2, 11);
////        mat2 = ImgprocUtils.gaussianBlur(mat2, new Size( 3,3), 0);
////        mat2 = ImgprocUtils.gaussianBlur(mat2, new Size( 5,5), 0);
////        mat2 = ImgprocUtils.gaussianBlur(mat2, new Size( 5,5), 0);
////        mat2 = ImgprocUtils.gaussianBlur(mat2, new Size(9,9), 0);
////        mat2 = ImgprocUtils.canny(mat2, 50, 200);
//
//
//        Imgproc.threshold(mat2, mat2, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
////
////        Imgproc.adaptiveThreshold(mat2, mat2, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 51, 2);
////        Mat kernel = Imgproc.getStructuringElement(MORPH_ELLIPSE, new Size(1, 5));
////        Imgproc.morphologyEx(mat2, mat2, MORPH_OPEN, kernel);
//
//
//        ArrayList<MatOfPoint> contours = findContours(mat2);
//        Log.d(TAG, "contures all: " + contours.size());
//        contours = TempUt.filterSmallOnes(contours);
//        Log.d(TAG, "contures big ones: " + contours.size());
//
////        contours = contoursSortLeftToRight(contours);
//
//
////        String resultedString = TextRecognitionUtils.recognizeDigits(mat2, new ArrayList<>(contours.subList(4, 5)));
//        String resultedString = TextRecognitionUtils.recognizeDigits(mat2, contours);
//
//        Log.d(TAG, "result: " + resultedString);
//
//        TextView textView = findViewById(R.id.text);
//        textView.setText(resultedString);
//
//
//        //creating output bitmap
//        drawRectAroundContours(contours, mat2);
//
//
//        contours = findContours(mat2);
//        contours = TempUt.filterSmallOnes(contours);
//        contours = contoursSortLeftToRight(contours);
////        drawRectAroundContours(new ArrayList<>(contours.subList(2,3)), mat2);
////        resultedString = resultedString + " " + TextRecognitionUtils.recognizeDigits(mat2, contours);
//        resultedString = TextRecognitionUtils.recognizeDigits(mat2, contours);
//        drawRectAroundContours(contours, mat2);
//
//        textView.setText(resultedString);
//
//
//        Bitmap outputBitmap = Bitmap.createBitmap(mat2.cols(), mat2.rows(), Bitmap.Config.ARGB_8888);
//
//        outputBitmap = ImgprocUtils.matToBitmap(mat2, outputBitmap);
//
//        ImageView imgView = findViewById(R.id.imageView);
//        imgView.setImageBitmap(outputBitmap);
    }
}
