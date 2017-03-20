/*
 To do:
 - Output OSC
 - Make peak threshold adjustable
 - Get running average
 - GUI Controls:
    - predictionSpan
    - volumeThres
    - triggTimer
 
 */

#include "ofApp.h"

#define AUDIO_BUFFER_SIZE 512
#define AUDIO_SAMPLE_RATE 44100
#define FFT_WINDOW_SIZE 2048
#define FFT_HOP_SIZE AUDIO_BUFFER_SIZE

//--------------------------------------------------------------
void ofApp::setup(){
    
    ofSetFrameRate(60);
    
    //Setup the FFT
    FFT fft;
    fft.init(FFT_WINDOW_SIZE,FFT_HOP_SIZE,1,FFT::RECTANGULAR_WINDOW,true,false,DATA_TYPE_MATRIX);
    
    //Setup the classifier
    RandomForests forest;
    forest.setForestSize( 10 );
    forest.setNumRandomSplits( 100 );
    forest.setMaxDepth( 10 );
    forest.setMinNumSamplesPerNode( 10 );
    
    //Add the feature extraction and classifier to the pipeline
    pipeline.addFeatureExtractionModule( fft );
    pipeline.setClassifier( forest );
    
    trainingClassLabel = 0;
    record = false;
    processAudio = true;
    trainingData.setNumDimensions( 1 ); //We are only going to use the data from one microphone channel, so the dimensions are 1
    trainingSample.resize( AUDIO_BUFFER_SIZE, 1 ); //We will set the training matrix to match the audio buffer size
    
    //Setup the audio card
    ofSoundStreamSetup(2, 1, this, AUDIO_SAMPLE_RATE, AUDIO_BUFFER_SIZE, 4);
    
    
    //GUI
    gui.setup();
    gui.add(volThreshold.setup("volThreshold", 1, 0.0, 3.0));
    gui.add(predictionSpan.setup("predictionSpan", 150, 5, 300));
    gui.add(triggTimerThreshold.setup("triggTimerThreshold", 10, 1, 100));
    gui.setPosition(10,620);
    
    
}

//--------------------------------------------------------------
void ofApp::update(){
    //Most of the updates are performed in the audio callback
    
    predictions.resize(predictionSpan);
    
    //High volume trigg timer
    triggTimer++;
    if (triggTimer>triggTimerThreshold) {
        singleTrigg = true;
    }
    
    
    //Avarage calculations
    vector<int> classLikelihoods;
    classLikelihoods.resize(maxNumClasses);
    
    for (int i = 0; i<maxNumClasses; i++) {
        classLikelihoods[i] = 0;
    }
    
    if( pipeline.getTrained() && predictionModeActive ){
        predictions[predictionsCounter] = pipeline.getPredictedClassLabel();
        
        for (int i = 0; i<predictions.size(); i++){
            for (int j = 0; j<maxNumClasses; j++) {
                if (predictions[i] == j) classLikelihoods[j] ++;
            }
        }
        
        double max = *max_element(classLikelihoods.begin(), classLikelihoods.end());
        cout<<"Max value: "<<max<<endl;
        for (int i = 0; i<classLikelihoods.size(); i++) {
            if (max == classLikelihoods[i]) {
                cout<<"CurrentClass value: "<<i<<endl;
            }
        }
        predictionsCounter++;
        
        if (predictionsCounter >= predictions.size()) {
            predictionsCounter = 0;
        }
    }
}

//--------------------------------------------------------------
void ofApp::draw(){
    
    ofBackground(0, 0, 0);
    
    string text;
    const int MARGIN = 20;
    const int graphSpacer = 15;
    int textX = MARGIN;
    int textY = MARGIN;
    
    //If the pipeline has been trained, then draw the plots
    if( pipeline.getTrained() && predictionModeActive ){
        
        //Draw the prediction info
        ofSetColor(255, 255, 255);
        text = "------------------- PredictionInfo -------------------";
        ofDrawBitmapString(text, textX,textY);
        
        textY += 15;
        text = "Pipeline Trained";
        ofDrawBitmapString(text, textX,textY);
        
        textY += 15;
        text = "Predicted Class Label: " + ofToString( pipeline.getPredictedClassLabel() );
        ofDrawBitmapString(text, textX,textY);
        
        textY += 15;
        text = "v: Toogle thresholdMode: (" + ofToString(thresholdMode) + ")";
        ofDrawBitmapString(text, textX,textY);
        
        float margin = 10;
        float x = margin;
        float y = textY += 35;
        float w = ofGetWidth() - margin*2;
        float h = 250;
        
        magnitudePlot.draw( x, y, w, h );
        
        y += h + 15;
        classLikelihoodsPlot.draw( x, y, w, h );
        
        //if (!thresholdMode) cout << pipeline.getClassLikelihoods()[0] << endl;
        
    }
    else{ //Draw the training info
        
        //Draw the training info
        ofSetColor(255, 255, 255);
        text = "------------------- TrainingInfo -------------------";
        ofDrawBitmapString(text, textX,textY);
        
        
        text = "Controls: 1-9: Set class label r: Toggle recording t: Train model l: Load s: Save c: Clear model p: Pause prediction v: Toogle thresholdMode: (" + ofToString(thresholdMode) + ")";
        textY += 15;
        ofDrawBitmapString(text, textX,textY);
        
        
        if( record ) ofSetColor(255, 0, 0);
        else ofSetColor(255, 255, 255);
        textY += 15;
        text = record ? "Current State: RECORDING" : "Current State: Not Recording";
        ofDrawBitmapString(text, textX,textY);
        
        ofSetColor(255, 255, 255);
        textY += 15;
        text = "TrainingClassLabel: " + ofToString(trainingClassLabel);
        ofDrawBitmapString(text, textX,textY);
        
        textY += 15;
        text = "NumTrainingSamples: " + ofToString(trainingData.getNumSamples());
        ofDrawBitmapString(text, textX,textY);
        
        textY += 15;
        text = "Info: " + infoText;
        ofDrawBitmapString(text, textX,textY);
        
        textY += 15;
        text = "Current volume: " + ofToString (curVol);
        ofDrawBitmapString(text, textX,textY);
        
    }
    
    gui.draw();
}

void ofApp::audioIn(float * input, int bufferSize, int nChannels){
    
    if( !processAudio ) return;
    
    //Quick and dirty volume estimate
    curVol = 0;
    for (int i = 0; i < bufferSize; i++){
        curVol += input[i]*input[i];
    }
    curVol = sqrt( curVol );
    
    for (int i=0; i<bufferSize; i++) {
        trainingSample[i][0] = input[i];
    }
    
    if( record && !thresholdMode){
        trainingData.addSample( trainingClassLabel, trainingSample );
    }
    
    if( record && thresholdMode && curVol > volThreshold){
        trainingData.addSample( trainingClassLabel, trainingSample );
    }
    
    
    if( pipeline.getTrained() && predictionModeActive && !thresholdMode){
        
        //Run the prediction using the matrix of audio data
        pipeline.predict( trainingSample );
        
        //Update the FFT plot
        FFT *fft = pipeline.getFeatureExtractionModule< FFT >( 0 );
        if( fft ){
            vector< FastFourierTransform > &results =  fft->getFFTResultsPtr();
            magnitudePlot.setData( results[0].getMagnitudeData() );
        }
        
        //Update the likelihood plot
        classLikelihoodsPlot.update( pipeline.getClassLikelihoods() );
    }
    
    if( pipeline.getTrained() && predictionModeActive && thresholdMode && curVol > volThreshold && singleTrigg){
        
        //Run the prediction using the matrix of audio data
        pipeline.predict( trainingSample );
        
        //Update the FFT plot
        FFT *fft = pipeline.getFeatureExtractionModule< FFT >( 0 );
        if( fft ){
            vector< FastFourierTransform > &results =  fft->getFFTResultsPtr();
            magnitudePlot.setData( results[0].getMagnitudeData() );
        }
        
        //Update the likelihood plot
        classLikelihoodsPlot.update( pipeline.getClassLikelihoods() );
        
        //cout << ofToString(ofGetFrameNum()) + " " + ofToString(pipeline.getPredictedClassLabel()) << endl;
        
        singleTrigg = false; //GET OUT!
        triggTimer = 0;
    }
}

void ofApp::exit(){
    processAudio = false;
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    
    switch ( key) {
        case 'r':
            record = !record;
            break;
        case '0':
            trainingClassLabel = 0;
            break;
        case '1':
            trainingClassLabel = 1;
            break;
        case '2':
            trainingClassLabel = 2;
            break;
        case '3':
            trainingClassLabel = 3;
            break;
        case '4':
            trainingClassLabel = 4;
            break;
        case '5':
            trainingClassLabel = 5;
            break;
        case '6':
            trainingClassLabel = 6;
            break;
        case '7':
            trainingClassLabel = 7;
            break;
        case '8':
            trainingClassLabel = 8;
            break;
        case '9':
            trainingClassLabel = 9;
            break;
        case 'p':
            predictionModeActive =! predictionModeActive;
            break;
            
        case 'v':
            thresholdMode =! thresholdMode;
            break;
            
            
        case 't':
            if( pipeline.train( trainingData ) ){
                infoText = "Pipeline Trained";
                
                predictionModeActive = true;
                
                //Update the plots
                magnitudePlot.setup( FFT_WINDOW_SIZE/2, 1 );
                classLikelihoodsPlot.setup( 60 * 5, pipeline.getNumClasses() );
                classLikelihoodsPlot.setRanges(0,1);
            }else infoText = "WARNING: Failed to train pipeline";
            break;
        case 's':
            if( trainingData.save( ofToDataPath("TrainingData.grt") ) ){
                
                infoText = "Training data saved to file";
            }else infoText = "WARNING: Failed to save training data to file";
            break;
        case 'l':
            if( trainingData.load( ofToDataPath("TrainingData.grt") ) ){
                infoText = "Training data saved to file";
            }else infoText = "WARNING: Failed to load training data from file";
            break;
        case 'c':
            record = false;
            
            trainingData.clear();
            
            predictionModeActive = false;
            
            infoText = "Training data cleared";
            break;
        default:
            break;
    }
    
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