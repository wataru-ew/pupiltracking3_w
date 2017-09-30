package com.wataru.pupiltracking3_w;


import android.app.Activity;
import android.content.Context;
import android.graphics.PixelFormat;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager.LayoutParams;
import android.widget.Button;
import android.widget.FrameLayout;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;


public class MainActivity extends Activity implements CvCameraViewListener2 {

    private final String _TAG                       = "MainActivity:";
    private static final Scalar FACE_RECT_COLOR     = new Scalar(0, 255, 0, 255);
    public static final int  JAVA_DETECTOR    = 0;
    private static final int TM_SQDIFF = 0;
    private static final int TM_SQDIFF_NORMED = 1;
    private static final int TM_CCOEFF = 2;
    private static final int TM_CCOEFF_NORMED = 3;
    private static final int TM_CCORR = 4;
    private static final int TM_CCORR_NORMED = 5;
    private int learn_frames = 0;
    private Mat teplateR;
    private Mat teplateL;
    int method = 0;
    // matrix for zooming

    private Mat mRgba;
    private Mat mGray;
    private Mat mRgbaF;
    private Mat mGrayF;
    private Mat mRgbaT;
    private Mat mGrayT;

    private File                   mCascadeFile;
    private File                   mCascadeFileEye;
    private CascadeClassifier mJavaDetector;
    private CascadeClassifier mJavaDetectorEye;

    private int                    mDetectorType       = JAVA_DETECTOR;
    private String[]               mDetectorName;

    private float                  mRelativeFaceSize   = 0.3f;
    private int mAbsoluteFaceSize = 0;

    private CameraBridgeViewBase mOpenCvCameraView;
    double xCenter = -1;
    double yCenter = -1;
    Button button_next, button_recreate;
    boolean bd = false;
    boolean get = false;
    int nums[] = new int[9];
    int pointid=0;
    int eyenum=0;
    int m=nums[0]/3;
    int n=nums[0]%3;
    int mm=m;
    int nn=n;
    int t=0;
    public volatile boolean exit = false;
    int height, width;
    double []a=new double[6];
    double []b=new double[6];
    int r=50;
    double centerx[]=new double[9];
    double centery[]=new double[9];
    int wx,hy;
    Point pupil2eye[][]=new Point[9][9];
    Point screen[]=new Point[9];

    onDrawView myview = null;
    private FrameLayout framelayout;
    private LayoutParams smallwmParams;
    private Point eyecenter;
    private Point pupilcenter;
    double x, y;

    //update
    Mat pupilteplateR;
    Mat pr;
    ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
    Mat hierarchy;
    Rect finalR_pupil_template;
    Mat finalM_pupil_template;
    int typex = 1;
    int row, col;
    int frame;

    /** Called when load the OpenCV **/
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            String TAG = new StringBuilder(_TAG).append("onManagerConnected").toString();

            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    try {
                        // load cascade file from application resources
                        InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();

                        // load cascade file from application resources
                        InputStream ise = getResources().openRawResource(R.raw.haarcascade_lefteye_2splits);
                        File cascadeDirEye = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFileEye = new File(cascadeDirEye, "haarcascade_righteye_2splits.xml");
                        FileOutputStream ose = new FileOutputStream(mCascadeFileEye);

                        while ((bytesRead = ise.read(buffer)) != -1) {
                            ose.write(buffer, 0, bytesRead);
                        }
                        ise.close();
                        ose.close();

                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        mJavaDetector.load(mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

                        mJavaDetectorEye = new CascadeClassifier(mCascadeFileEye.getAbsolutePath());
                        mJavaDetectorEye.load(mCascadeFileEye.getAbsolutePath());

                        if (mJavaDetectorEye.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier for eye");
                            mJavaDetectorEye = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFileEye.getAbsolutePath());

                        cascadeDir.delete();
                        cascadeDirEye.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }
                    mOpenCvCameraView.enableFpsMeter();
                    mOpenCvCameraView.setCameraIndex(1);
                    mOpenCvCameraView.enableView();
                } break;
                default: {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public MainActivity() {
        String TAG = new StringBuilder(_TAG).append("MainActivity").toString();
        mDetectorName = new String[2];
        mDetectorName[JAVA_DETECTOR] = "Java";
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        String TAG = new StringBuilder(_TAG).append("onCreate").toString();
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(LayoutParams.FLAG_KEEP_SCREEN_ON);

        this.myview = new onDrawView(this);
        setScreenMain();
        if (!OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, this,
                mLoaderCallback)) {
        }
    }

    private void setScreenMain(){
        setContentView(R.layout.main_activity);
        DisplayMetrics dm = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(dm);
        height = dm.heightPixels;
        width = dm.widthPixels;

        wx=(width-200)/2;
        hy=(height-200)/2;

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);

        button_recreate = (Button) findViewById(R.id.recreate);
        button_recreate.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v) {
                onRecreateClick(v);
            }
        });

        button_next = (Button) findViewById(R.id.next);
        button_next.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                goScreenSecond();
            }
        });
    }

    @Override
    public void onPause() {
        super.onPause();
        String TAG = new StringBuilder(_TAG).append("onPause").toString();
        Log.i(TAG, "Disabling a camera view");
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        String TAG = new StringBuilder(_TAG).append("onResume").toString();

        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        String TAG = new StringBuilder(_TAG).append("onDestroy").toString();
        Log.i(TAG, "Disabling a camera view");
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat();
        mGray = new Mat();

        //update
//        mGray = new Mat(height, width, CvType.CV_8SC4);
//        mGrayF = new Mat(height, width, CvType.CV_8SC4);
//        mGrayT = new Mat(width, height, CvType.CV_8SC4);
//
//        mRgba = new Mat(height, width, CvType.CV_8SC4);
//        mRgbaF = new Mat(height, width, CvType.CV_8UC4);
//        mRgbaT = new Mat(width, height, CvType.CV_8SC4);
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        String TAG = new StringBuilder(_TAG).append("OnCameraFrame").toString();
        Log.d(TAG, "Cameraframe is coming");
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
        // Rotate 90° counter-clockwise
        // Core.flip(src, dst, int flipCode)
        // flipCode 0: x軸, 1: y軸, -1: x,y軸
//        Core.transpose(mRgba, mRgbaT);
//        Imgproc.resize(mRgbaT, mRgbaF, mRgbaF.size());
//        Core.flip(mRgbaF, mRgba,-1);
//
//        Core.transpose(mGray, mGrayT);
//        Imgproc.resize(mGrayT, mGrayF, mGrayF.size());
//        Core.flip(mGrayF, mGray,-1);

        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
        }

        MatOfRect faces = new MatOfRect();
        if (mDetectorType == JAVA_DETECTOR) {
            if (mJavaDetector != null)
                mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                        new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        }
        else {
            Log.e(TAG, "Detection method is not selected!");
        }
        Log.d(TAG, "Finding the face");
        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++) {
            if (i==0){
                Imgproc.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);
                xCenter = (facesArray[i].x + facesArray[i].width + facesArray[i].x) / 2;
                yCenter = (facesArray[i].y + facesArray[i].y + facesArray[i].height) / 2;
                Point center = new Point(xCenter, yCenter);
                Rect r = facesArray[i];
                Rect eyearea = new Rect(r.x + r.width / 8,
                        (int) (r.y + (r.height / 4.5)),
                        r.width - 2 * r.width / 8,
                        (int) (r.height / 3.0));
                Rect eyearea_right = new Rect(r.x + r.width / 6, // original 16
                        (int) (r.y + (r.height / 3.5)), // original 4.5
                        (r.width - 2 * r.width / 6) / 2, // original 16
                        (int) (r.height / 4.0)); // original 3.0
                Imgproc.rectangle(mRgba, eyearea_right.tl(), eyearea_right.br(), FACE_RECT_COLOR, 3);
                if (learn_frames < 5) { //original 5
                    teplateR = M_get_template(mJavaDetectorEye, eyearea_right);
                    learn_frames++;
                } else {
                    match_eye(eyearea_right, teplateR, method);
                    eye_detect(eyearea_right);
                }
            }
        }
        return mRgba;
    }

    private void CreateAuxiliaryMats() {
        if (mGray.empty())
            return;

        int rows = mGray.rows();
        int cols = mGray.cols();

    }

    private Mat M_get_template(CascadeClassifier clasificator, Rect area) {
        Mat template = new Mat();
        Mat mROI = mGray.submat(area);
        MatOfRect eyes = new MatOfRect();
        Point iris = new Point();
        Rect W_pupil_template;
        clasificator.detectMultiScale(mROI, eyes, 1.15, 2, Objdetect.CASCADE_FIND_BIGGEST_OBJECT | Objdetect.CASCADE_SCALE_IMAGE, new Size(30, 30), new Size());

        Rect[] eyesArray = eyes.toArray();
        for (int i = 0; i < eyesArray.length;) {
            Rect e = eyesArray[i];
            e.x = area.x + e.x;
            e.y = area.y + e.y;
            Rect eye_only_rectangle = new Rect((int) e.tl().x, (int) (e.tl().y + e.height * 0.4), (int) e.width, (int) (e.height * 0.6));
            //Imgproc.rectangle(mRgba, eye_only_rectangle.tl(), eye_only_rectangle.br(), new Scalar(255, 255, 255, 255), 2);
            mROI = mGray.submat(eye_only_rectangle);
            Core.MinMaxLocResult mmG = Core.minMaxLoc(mROI);
            iris.x = mmG.minLoc.x + eye_only_rectangle.x;
            iris.y = mmG.minLoc.y + eye_only_rectangle.y;
            W_pupil_template = new Rect((int) iris.x - 40 / 2, (int) iris.y - 40 / 2, 40, 40);
            Imgproc.rectangle(mRgba, W_pupil_template.tl(), W_pupil_template.br(), new Scalar(255, 255, 0, 255), 2);
            template = (mGray.submat(W_pupil_template)).clone();

            return template;
        }
        return template;
    }

    private Rect R_get_template(CascadeClassifier clasificator, Rect area) {
        Mat mROI = mGray.submat(area);
        MatOfRect eyes = new MatOfRect();
        Point iris = new Point();
        Rect pupil_template = new Rect();

        clasificator.detectMultiScale(mROI, eyes, 1.15, 2, Objdetect.CASCADE_FIND_BIGGEST_OBJECT | Objdetect.CASCADE_SCALE_IMAGE, new Size(30, 30), new Size());
        Rect[] eyesArray = eyes.toArray();
        for (int i = 0; i < eyesArray.length;) {
            Rect e = eyesArray[i];
            e.x = area.x + e.x;
            e.y = area.y + e.y;
            Rect eye_only_rectangle = new Rect((int) e.tl().x, (int) (e.tl().y + e.height * 0.4), (int) e.width, (int) (e.height * 0.6));
            mROI = mGray.submat(eye_only_rectangle);
            Core.MinMaxLocResult mmG = Core.minMaxLoc(mROI);
            iris.x = mmG.minLoc.x + eye_only_rectangle.x;
            iris.y = mmG.minLoc.y + eye_only_rectangle.y;
            pupil_template = new Rect((int) iris.x - 160 / 2, (int) iris.y - 100 / 2 + 10 , 160, 100);
            return pupil_template;
        }
        return pupil_template;
    }

    private void match_eye(Rect area, Mat mTemplate, int type) {
        String TAG = new StringBuilder(_TAG).append("match_eye").toString();
        Log.i(TAG, "called match_eye function");
        Point matchLoc;
        Mat mROI = mGray.submat(area);
        int result_cols = mROI.cols() - mTemplate.cols() + 1;
        int result_rows = mROI.rows() - mTemplate.rows() + 1;
        // Check for bad template size
        if (mTemplate.cols() == 0 || mTemplate.rows() == 0) {
            return ;
        }
        Mat mResult = new Mat(result_cols, result_rows, CvType.CV_8U);

        switch (type) {
            case TM_SQDIFF:
                Imgproc.matchTemplate(mROI, mTemplate, mResult, Imgproc.TM_SQDIFF);
                break;
            case TM_SQDIFF_NORMED:
                Imgproc.matchTemplate(mROI, mTemplate, mResult,
                        Imgproc.TM_SQDIFF_NORMED);
                break;
            case TM_CCOEFF:
                Imgproc.matchTemplate(mROI, mTemplate, mResult, Imgproc.TM_CCOEFF);
                break;
            case TM_CCOEFF_NORMED:
                Imgproc.matchTemplate(mROI, mTemplate, mResult,
                        Imgproc.TM_CCOEFF_NORMED);
                break;
            case TM_CCORR:
                Imgproc.matchTemplate(mROI, mTemplate, mResult, Imgproc.TM_CCORR);
                break;
            case TM_CCORR_NORMED:
                Imgproc.matchTemplate(mROI, mTemplate, mResult,
                        Imgproc.TM_CCORR_NORMED);
                break;
        }

        Core.MinMaxLocResult mmres = Core.minMaxLoc(mResult);
        // there is difference in matching methods - best match is max/min value
        if (type == TM_SQDIFF || type == TM_SQDIFF_NORMED) {
            matchLoc = mmres.minLoc;
        } else {
            matchLoc = mmres.maxLoc;
        }

        Point matchLoc_tx = new Point(matchLoc.x + area.x, matchLoc.y + area.y);
        Point matchLoc_ty = new Point(matchLoc.x + mTemplate.cols() + area.x, matchLoc.y + mTemplate.rows() + area.y);

        Imgproc.rectangle(mRgba, matchLoc_tx, matchLoc_ty, new Scalar(255, 255, 0, 255));
        Rect rec = new Rect(matchLoc_tx,matchLoc_ty);
        pupilcenter = new Point((rec.tl().x + rec.br().x)/2, (rec.tl().y+ rec.br().y)/2);
        Imgproc.circle(mRgba, pupilcenter, 2, new Scalar(255, 255, 255, 255), 2);
    }

    private void eye_detect(Rect area) {
        String TAG = new StringBuilder(_TAG).append("eye_detect").toString();
        Log.d(TAG, "called eye_detect function");
        pr = new Mat();
        hierarchy = new Mat();
        Rect PR = R_get_template(mJavaDetectorEye, area);
        pupilteplateR = mGray.submat(PR).clone();
        Scalar s = Core.mean(pupilteplateR);
        Imgproc.threshold(pupilteplateR, pr, s.val[0], 255, Imgproc.THRESH_BINARY_INV);
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE,
                new Size(2, 2), new Point(1, 1));
        Imgproc.erode(pr, pr, element);
        Imgproc.dilate(pr, pr, element);
        Imgproc.dilate(pr, pr, element);
        Imgproc.erode(pr, pr, element);
        Mat prthis = pr.clone();
        if (!pr.empty()) {
            contours = new ArrayList<MatOfPoint>();
            Imgproc.findContours(prthis, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            double maxArea = -1;
            int maxAreaIdx = -1;
            if (contours.size() > 0) {
                MatOfPoint temp_contour = contours.get(0);  // the largest is at
                                                            // the
                                                            // index 0 for starting
                                                            // point
                MatOfPoint2f approxCurve = new MatOfPoint2f();
                Mat largest_contour = contours.get(0);
                List<MatOfPoint> largest_contours = new ArrayList<MatOfPoint>();
                int a = 0;
                for (int idx = 0; idx < contours.size(); idx++) {
                    temp_contour = contours.get(idx);
                    double contourarea = Imgproc.contourArea(temp_contour);
                    // compare this contour to the previous largest contour
                    // found
                    if (contourarea > maxArea) {
                        // check if this contour is a square
                        MatOfPoint2f new_mat = new MatOfPoint2f(
                                temp_contour.toArray());
                        int contourSize = (int) temp_contour.total();
                        Imgproc.approxPolyDP(new_mat, approxCurve,
                                contourSize * 0.05, true);
                        if (approxCurve.total() != 3
                                && approxCurve.total() != 4) {
                            maxArea = contourarea;
                            maxAreaIdx = idx;
                            largest_contours.add(temp_contour);
                            a++;
                            largest_contour = temp_contour;
                        }
                    }
                }
                if (a > 0) {// largest_contours.size() > 0) {
                    MatOfPoint temp_largest = largest_contours
                            .get(largest_contours.size() - 1);
                    largest_contours = new ArrayList<MatOfPoint>();
                    largest_contours.add(temp_largest);
                }
                Imgproc.cvtColor(prthis, prthis, Imgproc.COLOR_BayerBG2RGB);
                Imgproc.drawContours(prthis, largest_contours, -1, new Scalar(255, 0, 0), 4);
                if (a > 0) {// largest_contours!=null){
                    //Moments moment = Imgproc.moments(largest_contours.get(0), true);
                    //pupilcenter = new Point(moment.get_m10() / moment.get_m00() + PR.x, moment.get_m01() / moment.get_m00() + PR.y); //moment
                    //Imgproc.circle(mRgba, pupilcenter, 2, new Scalar(255, 255, 255, 255),-1); //moment of largest_counter
                    Rect my = Imgproc.boundingRect(largest_contours.get(0));
                    //Imgproc.rectangle(prthis, my.tl(), my.br(), new Scalar(255, 255, 255, 255));
                    Rect ll = new Rect(PR.x + my.x, PR.y + my.y, my.width, my.height);
                    Imgproc.rectangle(mRgba, ll.tl(), ll.br(), new Scalar(255, 0, 0, 255), 2);
                    //Imgproc.rectangle(mRgba, area.tl(), area.br(), new Scalar(0, 255, 0, 255), 2);
                    eyecenter = new Point((ll.tl().x + ll.br().x) / 2, (ll.tl().y + ll.br().y) / 2);
                    Imgproc.circle(mRgba, eyecenter, 2, new Scalar(255, 0, 0, 255), 2);
                }
            }
        }
    }

    public void onRecreateClick(View v) {
        learn_frames = 0;
        finalR_pupil_template = new Rect();
        finalM_pupil_template = new Mat();
    }

    private void goScreenSecond(){
        String TAG = new StringBuilder(_TAG).append("goScreenSecond").toString();
        framelayout = (FrameLayout) findViewById(R.id.framelayout);
        smallwmParams = new LayoutParams();
        smallwmParams.type = LayoutParams.TYPE_PHONE;
        smallwmParams.format = PixelFormat.RGBA_8888;   // picture Format
                                                        // background
                                                        // transparent
        smallwmParams.flags = LayoutParams.FLAG_NOT_TOUCH_MODAL
                | LayoutParams.FLAG_NOT_FOCUSABLE
                | LayoutParams.FLAG_NOT_TOUCHABLE;
        smallwmParams.alpha = 1.0f;
        smallwmParams.gravity = Gravity.RIGHT | Gravity.BOTTOM;// CENTER_VERTICAL;
        smallwmParams.x = 0;
        smallwmParams.y = 0;
        framelayout.addView(myview, smallwmParams);

        myview.setOnTouchListener(new View.OnTouchListener() {

            @Override
            public boolean onTouch(View v, MotionEvent event) {
                // TODO Auto-generated method stub
                //exit=true;
                return true;
            }
        });

        bd = true;
        boolean[] bool = new boolean[9];
        Random random = new Random();
        int rs;
        for (int j = 0; j < 9; j++) {
            do {
                rs = random.nextInt(9);
            } while (bool[rs]);
            bool[rs] = true;
            nums[j] = rs;
        }
        for(int i=0;i<9;i++){
            screen[i]=new Point();
            screen[i].x=wx*(nums[i]%3)+100;
            screen[i].y=hy*(nums[i]/3)+100;
        }
        m=nums[0]/3;
        n=nums[0]%3;
        mm=m;
        nn=n;
        t=0;

        new Thread(new al()).start();
    }

    class al implements Runnable{
        String TAG = new StringBuilder(_TAG).append("al").toString();
        public void run(){
            try{
                while(bd){
                    Message msg=new Message();
                    if(eyenum>8){
                        pointid=pointid+1;
                        eyenum=0;
                        if(pointid>8){
                            bd=false;
                            break;
                        }
                        mm=nums[pointid]/3;
                        nn=nums[pointid]%3;
                        r=50;
                    }
                    if(mm==m&&nn==n){
                        Thread.sleep(40);
                        r=r-1;
                        int ifdo=r%4;
                        if(r<=31){
                            get=true;
                        } else {
                            get=false;
                        }
                        if(r<=5) {
                            r=30;
                        }
                        if(get && ifdo==0){
                            Log.i(TAG, "get pupil to eye");
                            pupil2eye[pointid][eyenum]=new Point();
                            pupil2eye[pointid][eyenum].x=pupilcenter.x-eyecenter.x;
                            pupil2eye[pointid][eyenum].y=pupilcenter.y-eyecenter.y;
                            eyenum=eyenum+1;
                        }
                        x=screen[pointid].x;
                        y=screen[pointid].y;

                    } else{
                        r=50;
                        Thread.sleep(1);
                        t=t+1;
                        x=wx*(nn-n)*t/300+wx*n+2*r;
                        y=hy*(mm-m)*t/300+hy*m+2*r;
                        if(x==wx*nn+2*r&&y==hy*mm+2*r) {
                            m=mm;
                            n=nn;
                            t=0;
                        }
                    }
                    msg.what=1;
                    myHandler.sendMessage(msg);
                }
                Mat abx = new Mat();
                abx.create(9,6,CvType.CV_64FC1);
                Mat xx=new Mat();
                xx.create(6,1,CvType.CV_64FC1);
                Mat bx=new Mat();
                bx.create(9,1,CvType.CV_64FC1);
                Mat xy=new Mat();
                xy.create(6,1,CvType.CV_64FC1);
                Mat by=new Mat();
                by.create(9,1,CvType.CV_64FC1);
                double temp;
                for(int i=0;i<9;i++){
                    for(int j=0;j<9;j++){
                        for(int k=0;k<8;k++){
                            if(pupil2eye[i][k].x>pupil2eye[i][k+1].x){
                                temp=pupil2eye[i][k].x;
                                pupil2eye[i][k].x=pupil2eye[i][k+1].x;
                                pupil2eye[i][k+1].x=temp;
                            }
                            if(pupil2eye[i][k].y>pupil2eye[i][k+1].y){
                                temp=pupil2eye[i][k].y;
                                pupil2eye[i][k].y=pupil2eye[i][k+1].y;
                                pupil2eye[i][k+1].y=temp;
                            }
                        }
                    }
                    //center: pupil中心と目の中心との距離
                    //10個の中の中心を選択する
                    centerx[i]=(float) pupil2eye[i][4].x;
                    centery[i]=(float) pupil2eye[i][4].y;
                }
                for(int i =0;i<9;i++){
                    abx.put(i, 0, 1);
                    abx.put(i, 1, centerx[i]);
                    abx.put(i, 2, centery[i]);
                    abx.put(i, 3, centerx[i]*centerx[i]);
                    abx.put(i, 4, centery[i]*centery[i]);
                    abx.put(i, 5, centerx[i]*centery[i]);
                    bx.put(i, 0, screen[i].x);
                    by.put(i, 0, screen[i].y);
                }
                Core.solve(abx, bx, xx, Core.DECOMP_NORMAL);
                Core.solve(abx, by, xy, Core.DECOMP_NORMAL);
                for(int i=0;i<6;i++){
                    double []xx1=xx.get(i, 0);
                    a[i]=xx1[0];
                    double []xx2=xy.get(i, 0);
                    b[i]=xx2[0];
                }
                while(!exit){
                    frame++;
                    if(frame % 10 == 0){
                        Log.d(TAG, "frame: " + frame);
                        double p2ex=pupilcenter.x-eyecenter.x;
                        double p2ey=pupilcenter.y-eyecenter.y;
                        Message msg=new Message();
                        x=a[0]+a[1]*p2ex+a[2]*p2ey+a[3]*p2ex*p2ex+a[4]*p2ey*p2ey+a[5]*p2ex*p2ey;
                        y=b[0]+b[1]*p2ex+b[2]*p2ey+b[3]*p2ex*p2ex+b[4]*p2ey*p2ey+b[5]*p2ex*p2ey;
                        if(x<0){ x=0; }
                        if(x>width){ x=width; }
                        if(y<0) { y = 0; }
                        if(y>height) { y=height; }
                        switch (typex) {
                            case 0:
                                row = 2;
                                col = 2;
                                break;
                            case 1:
                                row = 2;
                                col = 3;
                                break;
                            case 2:
                                row = 3;
                                col = 3;
                                break;
                            case 3:
                                row = 3;
                                col = 4;
                                break;
                            case 4:
                                row = 4;
                                col = 4;
                                break;
                            case 5:
                                row = 4;
                                col = 5;
                                break;
                        }
                        msg.what=2;
                        myHandler.sendMessage(msg);
                    }
                }

            }catch(InterruptedException e){
                Thread.currentThread().interrupt();
            }
        }
    }
    Handler myHandler =new Handler(){
        public void handleMessage(Message msg){
            switch(msg.what){
                case 1:
                    myview.tran(x, y, r, 1);
                    myview.invalidate();
                    break;
                case 2:
                    myview.tran2(x, y, col, row, height, width, 2);
                    myview.invalidate();
                    break;
            }
            super.handleMessage(msg);
        }
    };

}
