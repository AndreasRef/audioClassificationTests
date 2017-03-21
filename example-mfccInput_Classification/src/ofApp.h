#pragma once

#include "ofMain.h"
#include "ofxAudioAnalyzer.h"
#include "ofxGrt.h"
#include "ofxGui.h"

//State that we want to use the GRT namespace
using namespace GRT;

class ofApp : public ofBaseApp{
    
public:
    
    void setup();
    void update();
    void draw();
    void exit();
    
    void audioIn(ofSoundBuffer &inBuffer);
    
    void trainClassifier();
    void save();
    void load();
    void clear();
    
    void keyPressed(int key);
    void keyReleased(int key);
    void mouseMoved(int x, int y );
    void mouseDragged(int x, int y, int button);
    void mousePressed(int x, int y, int button);
    void mouseReleased(int x, int y, int button);
    void windowResized(int w, int h);
    void dragEvent(ofDragInfo dragInfo);
    void gotMessage(ofMessage msg);
    
    //SOUND STUFF
    ofSoundStream soundStream;
    ofxAudioAnalyzer audioAnalyzer;
    
    vector<float> mfcc;
    float rms;
    
    //GRT STUFF
    ClassificationData trainingData;      		//This will store our training data
    GestureRecognitionPipeline pipeline;        //This is a wrapper for our classifier and any pre/post processing modules
    
    bool trainingModeActive;
    bool predictionModeActive;
    bool drawInfo;
    
    UINT trainingClassLabel;                    //This will hold the current label for when we are training the classifier
    UINT predictedClassLabel;
    string infoText;                            //This string will be used to draw some info messages to the main app window
    ofTrueTypeFont smallFont;
    ofTrueTypeFont hugeFont;
    ofxGrtTimeseriesPlot predictionPlot;
    int trainingInputs;
    
    
    //GUI
    ofxPanel gui;
    ofxFloatSlider volThreshold;
    ofxIntSlider predictionSpan;
    ofxIntSlider triggerTimerThreshold;
    ofxIntSlider sliderClassLabel;
    
    ofxButton  bTrain, bSave, bLoad, bClear;
    ofxToggle tThresholdMode, tRecord; 
    
    //Threshold & triggermode
    bool thresholdMode = false;
    bool singleTrigger = false;
    float predictionAlpha = 255;
    long startTime;
    
};
