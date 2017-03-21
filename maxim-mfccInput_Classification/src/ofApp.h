#pragma once

#include "ofMain.h"
#include "ofxMaxim.h"
#include "ofxGrt.h"
#include "ofxGui.h"
#include "ofxOsc.h"


// where to send osc messages by default
#define DEFAULT_OSC_DESTINATION "localhost"
#define DEFAULT_OSC_ADDRESS "/classification"
#define DEFAULT_OSC_PORT 8000

//State that we want to use the GRT namespace
using namespace GRT;

class ofApp : public ofBaseApp{
    
public:
    
    void setup();
    void update();
    void draw();
    void exit();
    
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
    
    //OSC
    ofxOscSender sender;
    string oscDestination, oscAddress;
    int oscPort;
    
    
    //MAXIM SOUND
    void audioRequested 	(float * input, int bufferSize, int nChannels); /* output method */
    void audioReceived 	(float * input, int bufferSize, int nChannels); /* input method */
    
    float 	* lAudioOut; /* outputs */
    float   * rAudioOut;
    
    float * lAudioIn; /* inputs */
    float * rAudioIn;
    
    int		initialBufferSize; /* buffer size */
    int		sampleRate;
    
    
    /* stick your maximilian stuff below */
    double wave;
    ofxMaxiFFTOctaveAnalyzer oct;
    int nAverages = 12;
    
    ofxMaxiIFFT ifft;
    ofxMaxiFFT mfft;
    int fftSize = 1024;
    int bins, dataSize;
    
    maxiMFCC mfcc;
    double *mfccs;
    float rms = 0;
    
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
