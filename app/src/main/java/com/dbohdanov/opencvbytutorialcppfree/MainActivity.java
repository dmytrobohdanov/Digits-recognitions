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
    public static final String TAG = "mainActTaag";

    public static Bitmap getBitmapFromAsset(Context context) {
        return getBitmapFromAsset(context, "example.jpg");
//        return getBitmapFromAsset(context, "example2.png");
//        return getBitmapFromAsset(context, "dev1.jpg");
//        return getBitmapFromAsset(context, "dev1_cut.png");
//        return getBitmapFromAsset(context, "dev2.jpg");
//        return getBitmapFromAsset(context, "dev3.jpg");
//        return getBitmapFromAsset(context, "dev4.jpg");
//        return getBitmapFromAsset(context, "dev4_cut.png");
//        return getBitmapFromAsset(context, "dev4_cut0.png");
//        return getBitmapFromAsset(context, "ex1.jpg");
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

//        drawRectAroundContours(digitContoures, cuttedImgMat);

        //sorting from left to right
        digitContoures = contoursSortLeftToRight(digitContoures);

        //---------------
        String resultedString = TextRecognitionUtils.recognizeDigits(cuttedImgMat, digitContoures);

        TextView textView = findViewById(R.id.text);
        textView.setText(resultedString);

        //creating output bitmap
        Bitmap outputBitmap = Bitmap.createBitmap(cuttedImgMat.cols(), cuttedImgMat.rows(), Bitmap.Config.ARGB_8888);

        outputBitmap = ImgprocUtils.matToBitmap(cuttedImgMat, outputBitmap);

        ImageView imgView = findViewById(R.id.imageView);
        imgView.setImageBitmap(outputBitmap);
    }
}
