package mariannelinhares.mnistandroid;

/*
   Copyright 2016 Narrative Nights Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   From: https://raw.githubusercontent
   .com/miyosuda/TensorFlowAndroidMNIST/master/app/src/main/java/jp/narr/tensorflowmnist
   /DrawModel.java
*/

//An activity is a single, focused thing that the user can do. Almost all activities interact with the user,
//so the Activity class takes care of creating a window for you in which you can place your UI with setContentView(View)
import android.app.Activity;
//PointF holds two float coordinates
import android.content.res.AssetManager;
import android.content.res.Resources;
import android.graphics.PointF;
//A mapping from String keys to various Parcelable values (interface for data container values, parcels)
import android.os.Bundle;
//Object used to report movement (mouse, pen, finger, trackball) events.
// //Motion events may hold either absolute or relative movements and other data, depending on the type of device.
import android.os.Environment;
import android.view.MotionEvent;
//This class represents the basic building block for user interface components.
// A View occupies a rectangular area on the screen and is responsible for drawing
import android.view.View;
//A user interface element the user can tap or click to perform an action.
import android.widget.Button;
//A user interface element that displays text to the user. To provide user-editable text, see EditText.
import android.widget.LinearLayout;
import android.widget.TextView;
//Resizable-array implementation of the List interface. Implements all optional list operations, and permits all elements,
// including null. In addition to implementing the List interface, this class provides methods to
// //manipulate the size of the array that is used internally to store the list.
import com.jjoe64.graphview.GraphView;
import com.jjoe64.graphview.series.DataPoint;
import com.jjoe64.graphview.series.LineGraphSeries;

import org.tensorflow.Graph;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.FileReader;
import java.util.Scanner;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.reflect.Field;
import java.util.ArrayList;
// basic list
import java.util.List;
//encapsulates a classified image
//public interface to the classification class, exposing a name and the recognize function
import mariannelinhares.mnistandroid.models.Classification;
import mariannelinhares.mnistandroid.models.Classifier;
//contains logic for reading labels, creating classifier, and classifying
import mariannelinhares.mnistandroid.models.TensorFlowClassifier;
//class for drawing MNIST digits by finger
//import mariannelinhares.mnistandroid.views.DrawModel;
//class for drawing the entire app
//import mariannelinhares.mnistandroid.views.DrawView;

public class MainActivity extends Activity implements View.OnClickListener {

    private static final float NOMALIZE = 600;
    private static final int PIXEL_WIDTH = 28;

    // ui elements
    private Button clearBtn, classBtn, chooseFileBtn, nextBtn;
    private TextView resText;
    private LineGraphSeries<DataPoint> series;
    private List<Classifier> mClassifiers = new ArrayList<>();
    public String path = Environment.getExternalStorageDirectory().getAbsolutePath();
    public String folderNameStr = "DCIM/Group13/";

    public String pathdata = "file:///android_assets/";
    private int fileid ;
    private int prevfile = 0;
    float value[][] = new float[15][347];
    float timeaxis;

    // views
    //private DrawModel drawModel;
    //private DrawView drawView;
    //private PointF mTmpPiont = new PointF();

    private GraphView graph;

    //private float mLastX;
   // private float mLastY;


//       <mariannelinhares.mnistandroid.views.DrawView
//    android:id="@+id/draw"
//    android:layout_width="match_parent"
//    android:layout_height="0dp"
//    android:layout_weight="1"
//            />


    @Override
    // In the onCreate() method, you perform basic application startup logic that should happen
    //only once for the entire life of the activity.
    protected void onCreate(Bundle savedInstanceState) {
        //initialization
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

//        //get drawing view from XML (where the finger writes the number)
//        drawView = (DrawView) findViewById(R.id.draw);
//        //get the model object
//        drawModel = new DrawModel(PIXEL_WIDTH, PIXEL_WIDTH);
//        //init the view with the model object
//        drawView.setModel(drawModel);
//        // give it a touch listener to activate when the user taps
//        drawView.setOnTouchListener(this);


        graph = (GraphView) findViewById(R.id.graph);

        //clear button
        //clear the drawing when the user taps
        clearBtn = (Button) findViewById(R.id.btn_clear);
        clearBtn.setOnClickListener(this);

        //class button
        //when tapped, this performs classification on the drawn image
        classBtn = (Button) findViewById(R.id.btn_class);
        classBtn.setOnClickListener(this);

        //Choose file button
        //when tapped, this open dialog to choose files
        chooseFileBtn = (Button) findViewById(R.id.choose_file);
        chooseFileBtn.setOnClickListener(this);

        nextBtn = (Button) findViewById(R.id.btn_next);
        nextBtn.setOnClickListener(this);

        // res text
        //this is the text that shows the output of the classification
        resText = (TextView) findViewById(R.id.tfRes);

        // tensorflow
        //load up our saved model to perform inference from local storage
        loadModel();
    }

    //the activity lifecycle

    @Override
    //OnResume() is called when the user resumes his Activity which he left a while ago,
    // //say he presses home button and then comes back to app, onResume() is called.
    protected void onResume() {
        //drawView.onResume();
        super.onResume();
    }

    @Override
    //OnPause() is called when the user receives an event like a call or a text message,
    // //when onPause() is called the Activity may be partially or completely hidden.
    protected void onPause() {
        //drawView.onPause();
        super.onPause();
    }
    //creates a model object in memory using the saved tensorflow protobuf model file
    //which contains all the learned weights
    private void loadModel() {
        //The Runnable interface is another way in which you can implement multi-threading other than extending the
        // //Thread class due to the fact that Java allows you to extend only one class. Runnable is just an interface,
        // //which provides the method run.
        // //Threads are implementations and use Runnable to call the method run().
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    //add 2 classifiers to our classifier arraylist
                    //the tensorflow classifier and the keras classifier


                    mClassifiers.add(
                            TensorFlowClassifier.create(getAssets(), "Keras",
                                    "frozen_LSTM.pb", "labels1.txt", 347,
                                    "lstm_1_input", "dense_1/Sigmoid", false));

                } catch (final Exception e) {
                    //if they aren't found, throw an error!
                    throw new RuntimeException("Error initializing classifiers!", e);
                }
            }
        }).start();
    }

    @Override
    public void onClick(View view) {
        //when the user clicks something
        if (view.getId() == R.id.btn_clear) {
            //if its the clear button
            //clear the drawing
//            drawModel.clear();
//            drawView.reset();
//            drawView.invalidate();
            //empty the text view
            resText.setText("");
            graph.removeAllSeries();



        } else if (view.getId() == R.id.btn_class) {
            //if the user clicks the classify button
            //get the pixel data and store it in an array
            //float pixels[] = drawView.getPixelData();//byfei
            float unlabelData[] = value[prevfile];

            //init an empty string to fill with the classification output
            String text = "";
            //for each classifier in our array
            for (Classifier classifier : mClassifiers) {
                //perform classification on the image
                //final Classification res = classifier.recognize(pixels);// byfei
                final Classification res = classifier.recognize(unlabelData);
                //if it can't classify, output a question mark
                if (res.getLabel() == null) {
                    text += classifier.name() + ": ?\n";
                } else {
                    //else output its name
                    if(res.getLabel()=="1") {
                        text += String.format("%s: %s, %f, %s \n", classifier.name(), res.getLabel(), res.getConf(),"Seizure");
                    }
                    else{
                        text += String.format("%s: %s, %f, %s \n", classifier.name(), res.getLabel(), res.getConf(),"Normal");
                    }
                }
            }
            resText.setText(text);
        }
        else if (view.getId() == R.id.choose_file) {
            File destDir = new File(path+"/"+folderNameStr);
            File[] childFiles = destDir.listFiles();
            String childFileName = null;
            int fileNo = 0;
            for (File child:childFiles){
                childFileName = child.getName();
                if(childFileName.endsWith(".txt")) {
                    value[fileNo]= readFileByLine(child);
                    for(int k = 0; k <347; k++){
                    value[fileNo][k] = value[fileNo][k]/NOMALIZE;
                    }
                    fileNo += 1;
                    //plot(value[prevfile]);
                }
            }
            plot(value[prevfile]);
        }

        else if (view.getId() == R.id.btn_next) {
            prevfile += 1;
            //GraphView graph = (GraphView) findViewById(R.id.graph);
            graph.removeAllSeries();
            plot(value[prevfile]);
        }
    }


    public void plot(float yaxis[]){
        float y;
        timeaxis = 0;
        series = new LineGraphSeries<DataPoint>();
        for(int i = 0; i< 347; i ++)
        {
            timeaxis += 1;
            y = yaxis[i];///600
            series.appendData(new DataPoint(timeaxis,y),true,347);
        }
        graph.addSeries(series);

    }

    public static float[] readFileByLine(File fileName){
        //File file = new File(filePath);
        // BufferedReader:
        float temparray [] = new float[347];
        int ctr = 0;
        Scanner scan = null;
        BufferedReader buf = null;
        try{
            // FileReader:用来读取字符文件的便捷类。
            buf = new BufferedReader(new FileReader(fileName));
            scan = new Scanner(fileName);
            // buf = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
            String temp = null  ;

            //while ((temp = scan)) != null ){
            while (scan.hasNext()){
                temparray[ctr]=Float.parseFloat(scan.next());
                ctr += 1;
            }

        }catch(Exception e){
            e.getStackTrace();
        }finally{
            if(buf != null){
                try{
                    buf.close();
                } catch (IOException e) {
                    e.getStackTrace();
                }
            }
        }
        return temparray;
    }


//    @Override
//    //this method detects which direction a user is moving
//    //their finger and draws a line accordingly in that
//    //direction
//    public boolean onTouch(View v, MotionEvent event) {
//        //get the action and store it as an int
//        int action = event.getAction() & MotionEvent.ACTION_MASK;
//        //actions have predefined ints, lets match
//        //to detect, if the user has touched, which direction the users finger is
//        //moving, and if they've stopped moving
//
//        //if touched
//        if (action == MotionEvent.ACTION_DOWN) {
//            //begin drawing line
//            processTouchDown(event);
//            return true;
//            //draw line in every direction the user moves
//        } else if (action == MotionEvent.ACTION_MOVE) {
//            processTouchMove(event);
//            return true;
//            //if finger is lifted, stop drawing
//        } else if (action == MotionEvent.ACTION_UP) {
//            processTouchUp();
//            return true;
//        }
//        return false;
//    }
//
//    //draw line down
//
//    private void processTouchDown(MotionEvent event) {
//        //calculate the x, y coordinates where the user has touched
//        mLastX = event.getX();
//        mLastY = event.getY();
//        //user them to calcualte the position
//        drawView.calcPos(mLastX, mLastY, mTmpPiont);
//        //store them in memory to draw a line between the
//        //difference in positions
//        float lastConvX = mTmpPiont.x;
//        float lastConvY = mTmpPiont.y;
//        //and begin the line drawing
//        drawModel.startLine(lastConvX, lastConvY);
//    }
//
//    //the main drawing function
//    //it actually stores all the drawing positions
//    //into the drawmodel object
//    //we actually render the drawing from that object
//    //in the drawrenderer class
//    private void processTouchMove(MotionEvent event) {
//        float x = event.getX();
//        float y = event.getY();
//
//        drawView.calcPos(x, y, mTmpPiont);
//        float newConvX = mTmpPiont.x;
//        float newConvY = mTmpPiont.y;
//        drawModel.addLineElem(newConvX, newConvY);
//
//        mLastX = x;
//        mLastY = y;
//        drawView.invalidate();
//    }
//
//    private void processTouchUp() {
//        drawModel.endLine();
//    }
}