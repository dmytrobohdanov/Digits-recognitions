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

        int[] ctrlColArray = getPixelsArrayFromCol(mat, ctrlColumnNum);
        int[] ctrlRowUpArray = getPixelsArrayByRow(mat, ctrlRowUpperNum);
        int[] ctrlRowDownArray = getPixelsArrayByRow(mat, ctrlRowDownNum);

        try {
            int assumedNumber = getNumberPatternByArrays(ctrlColArray, ctrlColumnNum,
                    ctrlRowUpArray, ctrlRowUpperNum,
                    ctrlRowDownArray, ctrlRowDownNum);

            return getNumberByPattern(assumedNumber);
        } catch (NotNumberException e) {
            return -1;
        }
    }

    /**
     * checking is assumed digit's patter is really digit
     *
     * @param assumedNumber pattern of assumed number to check is this digit
     * @return recognized number of negative int if it is not number
     */
    private int getNumberByPattern(int assumedNumber) {

        switch (assumedNumber) {
            case 63:
                return 0;

            case 6:
                return 1;

            case 91:
                return 2;

            case 79:
                return 3;

            case 102:
                return 4;

            case 109:
                return 5;

            case 125:
                return 6;

            case 7:
                return 7;

            case 127:
                return 8;

            case 111:
                return 9;

            default:
                return -1;
        }
    }

    /**
     * todo
     *
     * @param ctrlColArray
     * @param ctrlColumnNum
     * @param ctrlRowUpArray
     * @param ctrlRowUpperNum
     * @param ctrlRowDownArray
     * @param ctrlRowDownNum
     * @return
     * @throws NotNumberException
     */
    private int getNumberPatternByArrays(int[] ctrlColArray, int ctrlColumnNum,
                                         int[] ctrlRowUpArray, int ctrlRowUpperNum,
                                         int[] ctrlRowDownArray, int ctrlRowDownNum) throws NotNumberException {
        //number of pixels to cut from borders
        int borderPix = 2;

        //getting segment max width, presume that it could not be more then 30% of digit width
        final int segmentMaxWidth = (int) (0.3 * ctrlRowUpArray.length);

        //resulted pattern value
        //int value where each bit represents segment of 7-segment number
        int resultedPattern = 0;

        //segment 0
        int[] seg0Array = Arrays.copyOfRange(ctrlColArray, borderPix, ctrlRowUpperNum);
        resultedPattern = isSegmentOn(seg0Array, 0, segmentMaxWidth) ? resultedPattern | 0b1 : resultedPattern;

        //segment 1
        int[] seg1Array = Arrays.copyOfRange(ctrlRowUpArray, ctrlColumnNum, ctrlRowUpArray.length - borderPix);
        resultedPattern = isSegmentOn(seg1Array, 1, segmentMaxWidth) ? resultedPattern | 0b10 : resultedPattern;

        //segment 2
        int[] seg2Array = Arrays.copyOfRange(ctrlRowDownArray, ctrlColumnNum, ctrlRowDownArray.length - borderPix);
        resultedPattern = isSegmentOn(seg2Array, 2, segmentMaxWidth) ? resultedPattern | 0b100 : resultedPattern;

        //segment 3
        int[] seg3Array = Arrays.copyOfRange(ctrlColArray, ctrlRowDownNum, ctrlColArray.length - borderPix);
        resultedPattern = isSegmentOn(seg3Array, 3, segmentMaxWidth) ? resultedPattern | 0b1000 : resultedPattern;

        //segment 4
        int[] seg4Array = Arrays.copyOfRange(ctrlRowDownArray, borderPix, ctrlColumnNum);
        resultedPattern = isSegmentOn(seg4Array, 4, segmentMaxWidth) ? resultedPattern | 0b10000 : resultedPattern;

        //segment 5
        int[] seg5Array = Arrays.copyOfRange(ctrlRowUpArray, borderPix, ctrlColumnNum);
        resultedPattern = isSegmentOn(seg5Array, 5, segmentMaxWidth) ? resultedPattern | 0b100000 : resultedPattern;

        //segment 6
        int[] seg6Array = Arrays.copyOfRange(ctrlColArray, ctrlRowUpperNum + 1, ctrlRowDownNum);
        resultedPattern = isSegmentOn(seg6Array, 6, segmentMaxWidth) ? resultedPattern | 0b1000000 : resultedPattern;

        return resultedPattern;
    }


    /**
     * check is the segment is ON
     *
     * @param segArray     pixels of segment
     * @param segmentIndex index of segment, counting from the top one clockwise starting from 0 index.
     *                     the central segment has index 6
     * @return is this segment is ON
     * @throws IllegalArgumentException in case of some wrong segment's index
     * @throws NotNumberException       this segment seems like not segment of digit
     */
    private boolean isSegmentOn(int[] segArray, int segmentIndex, final int segmentMaxWidth)
            throws IllegalArgumentException, NotNumberException {
        final String NOT_DIGIT_MESSAGE = "this segment seems like not segment of digit";

        //init is start pixel is ON
        boolean startPixIsOn = segArray[0] >= 128;

        if (segmentIndex == 0 || segmentIndex == 4 || segmentIndex == 5) {
            //cutting starting zero's to prevent appearance of some random spaces
            segArray = cutZeroPixelsFromBegining(segArray);

            //if the segment is ON expect start with TRUE, one color change
            //  and width of segment is no more then segmentMaxWidth
            //if the segment is OFF expect empty array
            if (segArray.length == 0) {
                return false;
            } else if (getNumberOfColorChangesWithCheck(segArray, 1, segmentMaxWidth) == 1) {
                return true;
            } else {
                throw new NotNumberException(NOT_DIGIT_MESSAGE);
            }
        } else if (segmentIndex == 1 || segmentIndex == 2 || segmentIndex == 3) {
            //if the segment is ON expect start with FALSE and one color change
            //if the segment if OFF expect empty array
            //in this case segment can't start from turn-on pixel
            if (startPixIsOn) {
                throw new NotNumberException(NOT_DIGIT_MESSAGE);
            } else {
                segArray = cutZeroPixelsFromEnd(segArray);

                if (segArray.length == 0) {
                    return false;
                } else if (getNumberOfColorChangesWithCheck(segArray, 1, segmentMaxWidth) == 1) {
                    return true;
                } else {
                    throw new NotNumberException(NOT_DIGIT_MESSAGE);
                }
            }
        } else if (segmentIndex == 6) {
            //if the segment is ON expect start with FALSE and two color changes
            //if the segment if OFF expect start with FALSE and zero color changes
            //in this case segment can't start from turn-on pixel

            if (startPixIsOn) {

                throw new NotNumberException(NOT_DIGIT_MESSAGE);
            } else {
                int changes = getNumberOfColorChangesWithCheck(segArray, 2, segmentMaxWidth);

                if (changes == 2) {

                    return true;
                } else if (changes == 0) {

                    return false;
                } else {

                    throw new NotNumberException(NOT_DIGIT_MESSAGE);
                }
            }
        } else {
            throw new IllegalArgumentException("there are only 7 segments in digit, so index of segment must be 0..6");
        }
    }

    private int[] cutZeroPixels(int[] segArray, boolean isFromBegining) {
        int startPixel = isFromBegining ? segArray[0] : segArray[segArray.length - 1];

        //if start pixel is not zero return
        if (startPixel >= 128) {
            return segArray;
        }

        int counter = 0;
        int i = isFromBegining ? 0 : segArray.length - 1;

        while (i >= 0 && i < segArray.length && segArray[i] <= 128) {
            counter++;
            i = isFromBegining ? ++i : --i;
        }


        return isFromBegining
                ? Arrays.copyOfRange(segArray, counter, segArray.length)
                : Arrays.copyOfRange(segArray, 0, segArray.length - counter);
    }

    private int[] cutZeroPixelsFromEnd(int[] segArray) {
        return cutZeroPixels(segArray, false);
    }

    private int[] cutZeroPixelsFromBegining(int[] segArray) {
        return cutZeroPixels(segArray, true);
    }

    /**
     * counting how many times color changes in specified array
     *
     * @param segArray           array to count color's changes
     * @param maxNumberOfChanges maximum number of color changes to expect this segment
     *                           to be a part of digit
     * @param maxLength          max width of segment
     * @return number of changes or negative integer in case it has more colors changes then maximum
     * or in case of width of segment is more then maxLength
     */
    private int getNumberOfColorChangesWithCheck(int[] segArray, int maxNumberOfChanges, int maxLength) {
        //remember start element
        int previousPixel = segArray[0];

        //init counter
        int changeCounter = 0;

        //init segment length counter
        int segmentLengthCounter = 0;

        for (int pixelColor : segArray) {
            if (pixelColor >= 128) {
                segmentLengthCounter++;
                if (segmentLengthCounter >= maxLength) {
                    return -1;
                }
            }

            //check if color changed
            if (pixelColor != previousPixel) {
                changeCounter++;
                segmentLengthCounter = 1;

                //if there was more changes then max number this is not segment of the number
                // so return negative number and end execution of this method
                if (changeCounter > maxNumberOfChanges) {
                    return -1;
                }

                //remember new color
                previousPixel = pixelColor;
            }
        }

        return changeCounter;
    }


    /**
     * getting pixels array from specified column
     *
     * @param mat           matrix to get pixels
     * @param ctrlColumnNum column number
     * @return array of pixel in the column
     */
    private int[] getPixelsArrayFromCol(Mat mat, int ctrlColumnNum) {
        int[] array = new int[mat.rows()];

        for (int i = 0; i < mat.rows(); i++) {
            array[i] = (int) mat.get(i, ctrlColumnNum)[0];
        }

        return array;
    }

    /**
     * getting pixels array from specified row
     *
     * @param mat        matrix to get pixels
     * @param ctrlRowNum row number
     * @return array of pixel in the row
     */
    private int[] getPixelsArrayByRow(Mat mat, int ctrlRowNum) {
        int[] array = new int[mat.cols()];

        for (int i = 0; i < mat.cols(); i++) {
            array[i] = (int) mat.get(ctrlRowNum, i)[0];
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
