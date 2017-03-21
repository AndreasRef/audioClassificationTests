/*
 To do
 - Output OSC?
 */


#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
    
    ofBackground(34, 34, 34);
    ofSetFrameRate(60);
    
    int sampleRate = 44100;
    int bufferSize = 512;
    int outChannels = 0;
    int inChannels = 2;
    
    // setup the sound stream
    soundStream.setup(this, outChannels, inChannels, sampleRate, bufferSize, 3);
    
    //setup ofxAudioAnalyzer with the SAME PARAMETERS
    audioAnalyzer.setup(sampleRate, bufferSize/2, inChannels);
    
    smallFont.load("arial.ttf", 10, true, true);
    smallFont.setLineHeight(12.0f);
    hugeFont.load("arial.ttf", 36, true, true);
    hugeFont.setLineHeight(38.0f);
    
    infoText = "";
    predictedClassLabel = 0;
    trainingModeActive = false;
    predictionModeActive = false;
    drawInfo = true;
    
    trainingInputs = 13; //Number of mfcc's
    
    trainingData.setNumDimensions( trainingInputs );
    
    //Set classifier
    MinDist minDist; //Other classifiers: AdaBoost adaboost; DecisionTree dtree; KNN knn; GMM gmm; ANBC naiveBayes; MinDist minDist; RandomForests randomForest; Softmax softmax; SVM svm;
    pipeline.setClassifier( minDist );
    
    
    //OSC
    // default settings
    oscDestination = DEFAULT_OSC_DESTINATION;
    oscAddress = DEFAULT_OSC_ADDRESS;
    oscPort = DEFAULT_OSC_PORT;
    sender.setup(oscDestination, oscPort);
    
    
    //GUI
    bTrain.addListener(this, &ofApp::trainClassifier);
    bSave.addListener(this, &ofApp::save);
    bLoad.addListener(this, &ofApp::load);
    bClear.addListener(this, &ofApp::clear);
    
    
    gui.setup();
    gui.add(sliderClassLabel.setup("Class Label", 1, 1, 9));
    gui.add(tRecord.setup("Record", false));
    gui.add(bTrain.setup("Train"));
    gui.add(bSave.setup("Save"));
    gui.add(bLoad.setup("Load"));
    gui.add(bClear.setup("Clear"));
    gui.add(tThresholdMode.setup("Threshold Mode", false));
    gui.add(triggerTimerThreshold.setup("Threshold timer (ms)", 10, 1, 1000));
    gui.add(volThreshold.setup("volThreshold", 0.6, 0.0, 1.0));
    //gui.add(predictionSpan.setup("predictionSpan", 150, 5, 300)); //Maybe comming later
    
    gui.setPosition(10,10);
    
    startTime = ofGetElapsedTimeMillis();
}

//--------------------------------------------------------------
void ofApp::update(){
    
    trainingClassLabel = sliderClassLabel;
    
    float smooth = 0;
    
    //get the analysis values
    rms = audioAnalyzer.getValue(RMS, 0, smooth);
    mfcc = audioAnalyzer.getValues(MFCC, 0, smooth);
    
    long timer = ofGetElapsedTimeMillis() - startTime;
    
    //High volume trigger timer
    if (timer>triggerTimerThreshold) {
        singleTrigger = true;
    }
    
    //GRT STUFF
    VectorFloat trainingSample(trainingInputs);
    VectorFloat inputVector(trainingInputs);
    
    for (int i = 0; i < mfcc.size(); i++) {
        trainingSample[i] = mfcc[i];
    }
    inputVector = trainingSample;
    
    
    if( tRecord && !tThresholdMode){
        trainingData.addSample( trainingClassLabel, trainingSample );
    } else if (tRecord && tThresholdMode && rms > volThreshold && singleTrigger) {
        trainingData.addSample( trainingClassLabel, trainingSample );
        singleTrigger = false;
        startTime = ofGetElapsedTimeMillis();
    }
    
    
    //Update the prediction mode if active
    if( predictionModeActive && !tThresholdMode){
        if( pipeline.predict( inputVector ) ){
            predictedClassLabel = pipeline.getPredictedClassLabel();
            predictionPlot.update( pipeline.getClassLikelihoods() );
            
            // send over OSC
            ofxOscMessage m;
            m.setAddress(oscAddress);
            m.addIntArg(pipeline.getPredictedClassLabel());
            sender.sendMessage(m, false);
            
        }else{
            infoText = "ERROR: Failed to run prediction!";
        }
    } else if (predictionModeActive && tThresholdMode && rms > volThreshold && singleTrigger) {
        if( pipeline.predict( inputVector ) ){
            predictedClassLabel = pipeline.getPredictedClassLabel();
            predictionPlot.update( pipeline.getClassLikelihoods() );
            singleTrigger = false;
            startTime = ofGetElapsedTimeMillis();
            predictionAlpha = 255;
            
            // send over OSC
            ofxOscMessage m;
            m.setAddress(oscAddress);
            m.addIntArg(pipeline.getPredictedClassLabel());
            sender.sendMessage(m, false);
            
        }
    }
    
    if (tThresholdMode && predictionAlpha > 0  ) predictionAlpha-=5;
    if (!tThresholdMode) predictionAlpha = 255;
    
    

    
}

//--------------------------------------------------------------
void ofApp::draw(){
    
    //RMS
    ofFill();
    if (singleTrigger) {
        ofSetColor(255);
    }else {
        ofSetColor(100);
    }
    ofDrawRectangle(10, 210, 200*(rms), 10);
    ofSetColor(255);
    ofDrawBitmapString("RMS: " + ofToString(rms), 10, 240);
    
    //Threshold line
    if (tThresholdMode) {
        ofSetColor(255,0,0);
        ofSetLineWidth(5);
        ofDrawLine(volThreshold*200 + 10, 210, volThreshold*200 + 10, 220);
    }
    
    //MFCC's
    ofSetColor(255);
    int mw = 200;
    int mfccGraphH = 75;
    float bin_w = (float) mw / mfcc.size();
    for (int i = 0; i < mfcc.size(); i++){
        float scaledValue = ofMap(mfcc[i], 0, MFCC_MAX_ESTIMATED_VALUE, 0.0, 1.0, true);//clamped value
        float bin_h = -1 * (scaledValue * mfccGraphH);
        ofDrawRectangle(i*bin_w, 285, bin_w, bin_h);
    }
    
    
    //GRT
    int marginX = 10;
    int marginY = 10;
    int graphX = marginX;
    int graphY = marginY;
    int graphW = ofGetWidth() - graphX*2;
    int graphH = 150;
    ofSetLineWidth(1);
    
    //Draw the info text
    if( drawInfo ){
        float infoX = marginX;
        float infoW = 200;
        float textX = marginX;
        float textY = 300;
        float textSpacer = smallFont.getLineHeight() * 1.5;
        
        ofFill();
        ofSetColor( 255, 255, 255 );
        
        smallFont.drawString( "MFCCS CLASSIFIER EXAMPLE", textX, textY +20); textY += textSpacer*2;
        smallFont.drawString( "Num Samples: " + ofToString( trainingData.getNumSamples() ), textX, textY ); textY += textSpacer;
        textY += textSpacer;
        
        smallFont.drawString( "Total input values: "+ofToString(trainingInputs), textX, textY ); textY += textSpacer;
        ofSetColor(0,255,0);
        smallFont.drawString( infoText, textX, textY ); textY += textSpacer;
        textY += textSpacer;
        
        //Update the graph position
        graphX = infoX + infoW + 15;
        graphW = ofGetWidth() - graphX - 15;
    }
    
    
    //If the model has been trained, then draw this
    if( pipeline.getTrained() ){
        predictionPlot.draw( graphX, graphY, graphW, graphH ); graphY += graphH * 1.1;
        std::string txt = "Predicted Class: " + ofToString( predictedClassLabel );
        ofRectangle bounds = hugeFont.getStringBoundingBox( txt, 0, 0 );
        ofSetColor(255, predictionAlpha);
        hugeFont.drawString( txt, ofGetWidth()/2 - bounds.width*0.5, ofGetHeight() - bounds.height*3 );
    }
    gui.draw();
}
//--------------------------------------------------------------
void ofApp::audioIn(ofSoundBuffer &inBuffer){
    audioAnalyzer.analyze(inBuffer);
}

//--------------------------------------------------------------
void ofApp::exit(){
    ofSoundStreamStop();
    audioAnalyzer.exit();
}


//--------------------------------------------------------------
void ofApp::keyPressed(int key){ //Optional key interactions
    
    switch ( key) {
        case '1':
            sliderClassLabel = 1;
            break;
        case '2':
            sliderClassLabel = 2;
            break;
        case '3':
            sliderClassLabel = 3;
            break;
        case '4':
            sliderClassLabel = 4;
            break;
        case '5':
            sliderClassLabel = 5;
            break;
        case '6':
            sliderClassLabel = 6;
            break;
        case '7':
            sliderClassLabel = 7;
            break;
        case '8':
            sliderClassLabel = 8;
            break;
        case '9':
            sliderClassLabel = 9;
            break;
            
        case 's':
            save();
            break;
        case 'l':
            load();
            break;
        case 't':
            trainClassifier();
            break;
        case 'c':
            clear();
            break;
        case 'r':
            tRecord =! tRecord;
            break;
        case 'm':
            tThresholdMode =! tThresholdMode;
            break;
            
        default:
            break;
    }
}

//--------------------------------------------------------------
void ofApp::trainClassifier() {
    ofLog(OF_LOG_NOTICE, "Training...");
    tRecord = false;
    if( pipeline.train( trainingData ) ){
        infoText = "Pipeline Trained";
        std::cout << "getNumClasses: " << pipeline.getNumClasses() << std::endl;
        predictionPlot.setup( 500, pipeline.getNumClasses(), "prediction likelihoods" );
        predictionPlot.setDrawGrid( true );
        predictionPlot.setDrawInfoText( true );
        predictionPlot.setFont( smallFont );
        predictionPlot.setBackgroundColor( ofColor(50,50,50,255));
        predictionModeActive = true;
    }else infoText = "WARNING: Failed to train pipeline";
    
    
    ofLog(OF_LOG_NOTICE, "Done training...");
}

//--------------------------------------------------------------
void ofApp::save() {
    if( trainingData.save( ofToDataPath("TrainingData.grt") ) ){
        infoText = "Training data saved to file";
    }else infoText = "WARNING: Failed to save training data to file";
    
    
}

//--------------------------------------------------------------
void ofApp::load() {
    if( trainingData.load( ofToDataPath("TrainingData.grt") ) ){
        infoText = "Training data loaded from file";
        trainClassifier();
    }else infoText = "WARNING: Failed to load training data from file";
    
}

//--------------------------------------------------------------
void ofApp::clear() {
    trainingData.clear();
    infoText = "Training data cleared";
    predictionModeActive = false;
}


//--------------------------------------------------------------
void ofApp::keyReleased(int key){
    
}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){
    
}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){
    
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){
    
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){
    
}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){
    
}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){
    
}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 
    
}
