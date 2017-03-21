/*
  This example demonstrates how to use the GRT FFT algorithm in openFrameworks
 */

#pragma once

#include "ofMain.h"
#include "ofxGrt.h"
#include "ofxGui.h"

//#include <algorithm>


//State that we want to use the GRT namespace
using namespace GRT;

class ofApp : public ofBaseApp{

public:
    void setup();
    void update();
    void draw();

    void keyPressed  (int key);
    void keyReleased(int key);
    void mouseMoved(int x, int y );
    void mouseDragged(int x, int y, int button);
    void mousePressed(int x, int y, int button);
    void mouseReleased(int x, int y, int button);
    void windowResized(int w, int h);
    void dragEvent(ofDragInfo dragInfo);
    void gotMessage(ofMessage msg);

    void exit();
    void audioIn(float * input, int bufferSize, int nChannels);
    
    //Create some variables for the demo
    GestureRecognitionPipeline pipeline;
    TimeSeriesClassificationDataStream trainingData;
    MatrixFloat trainingSample;
    ofxGrtTimeseriesPlot magnitudePlot;
    ofxGrtTimeseriesPlot classLikelihoodsPlot;
    unsigned int trainingClassLabel;
    bool record;
    bool processAudio;
    string infoText;
    
    
    //Andreas additions
    bool predictionModeActive;
    bool thresholdMode = false;
    bool singleTrigg = true;
    int triggTimer = 0;
    //int triggTimerThreshold = 0;
    //float volThreshold = 1;
    
    float curVol = 0.0;
    
    float oscCertaintyThreshold = 0.8;
    
    vector<int> predictions;
    
    int predictionsCounter = 0;
    //int predictionSpan = 150;
    
    int maxNumClasses = 10;
    
    
    ofxPanel gui;
    ofxFloatSlider volThreshold;
    ofxIntSlider predictionSpan;
    ofxIntSlider triggTimerThreshold;
    
    
};
