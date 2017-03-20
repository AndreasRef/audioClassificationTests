/* Audio input from Maximilian to Wekinator
  Adapted from Maximilian example code by Rebecca Fiebrink, 2016
 
  You can copy and paste this and use it as a starting example.
 
  Sends OSC to Wekinator. The number of inputs depends on what you select in the GUI.
 
 */


#include "ofApp.h"
#include "maximilian.h"/* include the lib */
#include "time.h"



//-------------------------------------------------------------
testApp::~testApp() {
    
}


//--------------------------------------------------------------
void testApp::setup(){
    
    //Debug test
    ofSleepMillis(1000);
    
    
    sender.setup(HOST, PORT);
    /* some standard setup stuff*/
    
    ofEnableAlphaBlending();
    ofSetupScreen();
    ofBackground(0, 0, 0);
    ofSetFrameRate(60);
    
    /* This is stuff you always need.*/
    
    sampleRate 			= 44100; /* Sampling Rate */
    initialBufferSize	= 512;	/* Buffer Size. you have to fill this buffer with sound*/
    lAudioOut			= new float[initialBufferSize];/* outputs */
    rAudioOut			= new float[initialBufferSize];
    lAudioIn			= new float[initialBufferSize];/* inputs */
    rAudioIn			= new float[initialBufferSize];
    
    
    /* This is a nice safe piece of code */
    memset(lAudioOut, 0, initialBufferSize * sizeof(float));
    memset(rAudioOut, 0, initialBufferSize * sizeof(float));
    
    memset(lAudioIn, 0, initialBufferSize * sizeof(float));
    memset(rAudioIn, 0, initialBufferSize * sizeof(float));
    
    
    fftSize = 1024;
    mfft.setup(fftSize, 512, 256);
    ifft.setup(fftSize, 512, 256);
    
    
    nAverages = 12;
    oct.setup(sampleRate, fftSize/2, nAverages);
    
    mfccs = (double*) malloc(sizeof(double) * 13);
    mfcc.setup(512, 42, 13, 20, 20000, sampleRate);
    
    ofxMaxiSettings::setup(sampleRate, 2, initialBufferSize);
    ofSoundStreamSetup(2,2, this, sampleRate, initialBufferSize, 4);/* Call this last ! */
    
    
    //GRT STUFF
    //Initialize the training and info variables
    infoText = "";
    trainingClassLabel = 1;
    predictedClassLabel = 0;
    trainingModeActive = false;
    recordTrainingData = false;
    predictionModeActive = false;
    drawInfo = true;
    
    trainingInputs = 13; // test with only mfcc's
    
    trainingData.setNumDimensions( trainingInputs );
    
    //set the default classifier
    setClassifier( MINDIST );
    
    
    
    //GUI STUFF
    gui.setup(); // most of the time you don't need a name
    gui.add(fftToggle.setup("FFT bin magnitudes (pitch/timbre/volume) (512 inputs)", true));
    gui.add(mfccToggle.setup("MFCCs (timbre/vocal) (13 inputs)", true));
    gui.add(chromagramToggle.setup("Const-Q analyser (12 bands/oct) (104 inputs)", true));
    gui.add(peakFrequencyToggle.setup("Peak frequency (pitch) (1 input)", true));
    gui.add(centroidToggle.setup("Spectral centroid (timbre) (1 input)", true));
    gui.add(rmsToggle.setup("RMS (volume) (1 input)", true));
    
    gui.setPosition(850, 10);
    
    bHide = true;
    
    myfont.loadFont("arial.ttf", 18); //requires this to be in bin/data/
    myFont2.loadFont("arial.ttf", 12); //requires this to be in bin/data/
    
    
    largeFont.load("arial.ttf", 12, true, true);
    largeFont.setLineHeight(14.0f);
    smallFont.load("arial.ttf", 10, true, true);
    smallFont.setLineHeight(12.0f);
    hugeFont.load("arial.ttf", 36, true, true);
    hugeFont.setLineHeight(38.0f);
    
    
    ofSetVerticalSync(true);
    
}

//--------------------------------------------------------------
void testApp::update(){
    
    ofxOscMessage m;
    m.setAddress("/wek/inputs");
    
    if (fftToggle) {
       // cout << "FFT" << endl;
        //m.setAddress("/fft");
        for (int i = 0; i < fftSize/2; i++) {
            m.addFloatArg(mfft.magnitudes[i]);
            
        }
        // sender.sendMessage(m);
    }
    
    if (mfccToggle) {
        //cout << "mfcc" << endl;
        
        // ofxOscMessage m;
        // m.setAddress("/mfccs");
        for (int i = 0; i < 13; i++) {
            m.addFloatArg(mfccs[i]);
        }
        //sender.sendMessage(m);
    }
    
    
    if (chromagramToggle) {
       // cout << "chrm" << endl;
        
        // ofxOscMessage m;
        // m.setAddress("/octaveBins");
        for (int i = 0; i < oct.nAverages; i++) {
            m.addFloatArg(oct.averages[i]);
            //cout << i << endl;
        }
        // sender.sendMessage(m);
    }
    
    if (peakFrequencyToggle) {
       // cout << "peak" << endl;
        
        //ofxOscMessage m;
        // m.setAddress("/peakFrequency");
        m.addFloatArg(peakFreq);
        // sender.sendMessage(m);
    }
    
    if (centroidToggle) {
       // cout << "centr" << endl;
        
        //// ofxOscMessage m;
        // m.setAddress("/centroid");
        m.addFloatArg(centroid);
        // sender.sendMessage(m);
    }
    if (rmsToggle) {
        
       // cout << "rms" << endl;
        
        // ofxOscMessage m;
        // m.setAddress("/rms");
        m.addFloatArg(RMS);
        // sender.sendMessage(m);
    }
    sender.sendMessage(m);
    
    
    //GRT STUFF
    VectorFloat trainingSample(trainingInputs);
    VectorFloat inputVector(trainingInputs);
    
    //TRY TO SEND THE MFCC's
    for (int i = 0; i < 13; i++) {
        trainingSample[i] = mfccs[i];
        cout << mfccs[i] << endl;
    }
    
    inputVector = trainingSample;
    
    //Update the training mode if needed
    if( trainingModeActive ){
        
        //Check to see if the countdown timer has elapsed, if so then start the recording
        if( !recordTrainingData ){
            if( trainingTimer.timerReached() ){
                recordTrainingData = true;
                trainingTimer.start( RECORDING_TIME );
            }
        }else{
            //We should be recording the training data - check to see if we should stop the recording
            if( trainingTimer.timerReached() ){
                trainingModeActive = false;
                recordTrainingData = false;
            }
        }
        
        if( recordTrainingData ){
            
            if( !trainingData.addSample(trainingClassLabel, trainingSample) ){
                infoText = "WARNING: Failed to add training sample to training data!";
            }
        }
    }
    
    //Update the prediction mode if active
    if( predictionModeActive ){
        
        
        if( pipeline.predict( inputVector ) ){
            predictedClassLabel = pipeline.getPredictedClassLabel();
            predictionPlot.update( pipeline.getClassLikelihoods() );
            
        }else{
            infoText = "ERROR: Failed to run prediction!";
        }
    }
    
    
    
}



//--------------------------------------------------------------
void testApp::draw(){
    
    float horizWidth = 500.;
    float horizOffset = 100;
    float fftTop = 250;
    float mfccTop = 350;
    float chromagramTop = 450;
    
    ofSetColor(255, 0, 0,255);
    
    //FFT magnitudes:
    float xinc = horizWidth / fftSize * 2.0;
    for(int i=0; i < fftSize / 2; i++) {
        float height = mfft.magnitudes[i] * 100;
        ofRect(horizOffset + (i*xinc),250 - height,2, height);
    }
    myfont.drawString("FFT:", 30, 250);
    
    
    //MFCCs:
    ofSetColor(0, 255, 0,200);
    xinc = horizWidth / 13;
    for(int i=0; i < 13; i++) {
        float height = mfccs[i] * 100.0;
        //ofRect(horizOffset + (i*xinc),mfccTop - height,40, height);
        ofDrawRectangle(horizOffset + (i*xinc),mfccTop - height,40, height);
        //		cout << mfccs[i] << ",";
    }
    myfont.drawString("MFCCs:", 12, mfccTop);

    
    //Const-Q:
    ofSetColor(255, 0, 255,200);
    xinc = horizWidth / oct.nAverages;
    for(int i=0; i < oct.nAverages; i++) {
        float height = oct.averages[i] / 20.0 * 100;
        ofRect(horizOffset + (i*xinc),chromagramTop - height,2, height);
    }
    myfont.drawString("ConstQ:", 12, chromagramTop);

    
    ofSetColor(255, 255, 255,255);
    
    char peakString[255]; // an array of chars
    sprintf(peakString, "Peak Frequency: %.2f", peakFreq);
    myfont.drawString(peakString, 12, chromagramTop + 50);
    
    char centroidString[255]; // an array of chars
    sprintf(centroidString, "Spectral Centroid: %f", centroid);
    myfont.drawString(centroidString, 12, chromagramTop + 80);
    
    char rmsString[255]; // an array of chars
    sprintf(rmsString, "RMS: %.2f", RMS);
    myfont.drawString(rmsString, 12, chromagramTop + 110);
    
    int numInputs = 0;
    if (fftToggle) {
        numInputs += fftSize/2;
    }
    if (mfccToggle) {
        numInputs += 13;
    }
    if (chromagramToggle) {
        numInputs += 104;
    }
    if (peakFrequencyToggle) {
        numInputs++;
    }
    if (centroidToggle) {
        numInputs++;
    }
    if (rmsToggle) {
        numInputs++;
    }
    
    char numInputsString[255]; // an array of chars
    sprintf(numInputsString, "Sending %d inputs total", numInputs);
    myfont.drawString(numInputsString, 450, 100);

    
    
    //GRT
    int marginX = 5;
    int marginY = 5;
    int graphX = marginX;
    int graphY = marginY;
    int graphW = ofGetWidth() - graphX*2;
    int graphH = 150;
    
    //Draw the info text
    if( drawInfo ){
        float infoX = marginX;
        float infoW = 250;
        float textX = 10;
        float textY = marginY;
        float textSpacer = smallFont.getLineHeight() * 1.5;
        
        ofFill();
        ofSetColor(100,100,100);
        ofDrawRectangle( infoX, 5, infoW, 365 );
        ofSetColor( 255, 255, 255 );
        
        smallFont.drawString( "MFCCS CLASSIFIER EXAMPLE", textX, textY +20); textY += textSpacer*2;
        
        smallFont.drawString( "[i]: Toogle Info", textX, textY ); textY += textSpacer;
        smallFont.drawString( "[r]: Toggle Recording", textX, textY ); textY += textSpacer;
        smallFont.drawString( "[l]: Load Model", textX, textY ); textY += textSpacer;
        smallFont.drawString( "[s]: Save Model", textX, textY ); textY += textSpacer;
        smallFont.drawString( "[t]: Train Model", textX, textY ); textY += textSpacer;
        smallFont.drawString( "[1,2,3...]: Set Class Label", textX, textY ); textY += textSpacer;
        smallFont.drawString( "Classifier: " + classifierTypeToString( classifierType ), textX, textY ); textY += textSpacer;
        smallFont.drawString( "[n] null rejection: " + ofToString(nullRejection), textX, textY ); textY += textSpacer;
        textY += textSpacer;
        
        smallFont.drawString( "Class Label: " + ofToString( trainingClassLabel ), textX, textY ); textY += textSpacer;
        smallFont.drawString( "Recording: " + ofToString( recordTrainingData ), textX, textY ); textY += textSpacer;
        smallFont.drawString( "Num Samples: " + ofToString( trainingData.getNumSamples() ), textX, textY ); textY += textSpacer;
        textY += textSpacer;
        

        smallFont.drawString( "Total input values: "+ofToString(trainingInputs), textX, textY ); textY += textSpacer;
        textY += textSpacer;
        

        
        //Update the graph position
        graphX = infoX + infoW + 15;
        graphW = ofGetWidth() - graphX - 15;
    }
    
    
    
    
    
    if( trainingModeActive ){
        char strBuffer[1024];
        if( !recordTrainingData ){
            ofSetColor(255,150,0);
            sprintf(strBuffer, "Training Mode Active - Get Ready! Timer: %0.1f",trainingTimer.getSeconds());
        }else{
            ofSetColor(255,0,0);
            sprintf(strBuffer, "Training Mode Active - Recording! Timer: %0.1f",trainingTimer.getSeconds());
        }
        std::string txt = strBuffer;
        ofRectangle bounds = hugeFont.getStringBoundingBox( txt, 0, 0 );
        hugeFont.drawString(strBuffer, ofGetWidth()/2 - bounds.width*0.5, ofGetHeight() - bounds.height*3 );
    }
    
    //If the model has been trained, then draw this
    if( pipeline.getTrained() ){
        predictionPlot.draw( graphX, graphY, graphW, graphH ); graphY += graphH * 1.1;
        
        std::string txt = "Predicted Class: " + ofToString( predictedClassLabel );
        ofRectangle bounds = hugeFont.getStringBoundingBox( txt, 0, 0 );
        ofSetColor(0,0,255);
        hugeFont.drawString( txt, ofGetWidth()/2 - bounds.width*0.5, ofGetHeight() - bounds.height*3 );
    }
    
    
    
    if( bHide ){
        gui.draw();
    }
    
}

//--------------------------------------------------------------
void testApp::audioRequested 	(float * output, int bufferSize, int nChannels){
    for (int i = 0; i < bufferSize; i++){
        wave = lAudioIn[i];
        if (mfft.process(wave)) {

            mfft.magsToDB();
            oct.calculate(mfft.magnitudesDB);
            
            float sum = 0;
            float maxFreq = 0;
            int maxBin = 0;
            
            for (int i = 0; i < fftSize/2; i++) {
                sum += mfft.magnitudes[i];
                if (mfft.magnitudes[i] > maxFreq) {
                    maxFreq=mfft.magnitudes[i];
                    maxBin = i;
                }
            }
            centroid = sum / (fftSize / 2);
            peakFreq = (float)maxBin/fftSize * 44100;
            
            
            mfcc.mfcc(mfft.magnitudes, mfccs);
        }

        lAudioOut[i] = 0;
        rAudioOut[i] = 0;
        
    }
    
    
    
}

//--------------------------------------------------------------
void testApp::audioReceived 	(float * input, int bufferSize, int nChannels){
    
    
    /* You can just grab this input and stick it in a double, then use it above to create output*/
    
    float sum = 0;
    for (int i = 0; i < bufferSize; i++){
        
        /* you can also grab the data out of the arrays*/
        
        lAudioIn[i] = input[i*2];
        rAudioIn[i] = input[i*2+1];
        
        sum += input[i*2] * input[i*2];
        
    }
    RMS = sqrt(sum);
    
}

//--------------------------------------------------------------
void testApp::keyPressed(int key){
    
    infoText = "";
    bool buildTexture = false;
    
    switch ( key) {
        case 'r':
            predictionModeActive = false;
            trainingModeActive = true;
            recordTrainingData = false;
            trainingTimer.start( PRE_RECORDING_COUNTDOWN_TIME );
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
            
            
        case 't':
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
            break;
        case 's':
            if( trainingData.save( ofToDataPath("TrainingData.grt") ) ){
                infoText = "Training data saved to file";
            }else infoText = "WARNING: Failed to save training data to file";
            break;
        case 'l':
            if( trainingData.load( ofToDataPath("TrainingData.grt") ) ){
                infoText = "Training data loaded from file";
            }else infoText = "WARNING: Failed to load training data from file";
            break;
        case 'c':
            trainingData.clear();
            infoText = "Training data cleared";
            break;
        case 'i':
            drawInfo = !drawInfo;
            break;
            

            
            
        case OF_KEY_TAB:
            setClassifier( ++this->classifierType % NUM_CLASSIFIERS );
            break;
            
        default:
            break;
    }
    
    
}

//--------------------------------------------------------------

bool testApp::setClassifier( const int type ){
    
    AdaBoost adaboost;
    DecisionTree dtree;
    KNN knn;
    GMM gmm;
    ANBC naiveBayes;
    MinDist minDist;
    RandomForests randomForest;
    Softmax softmax;
    SVM svm;
    
    this->classifierType = type;
    
    switch( classifierType ){
        case ADABOOST:
            adaboost.enableNullRejection( nullRejection ); // The GRT AdaBoost algorithm does not currently support null rejection, although this will be added at some point in the near future.
            adaboost.setNullRejectionCoeff( 3 );
            pipeline.setClassifier( adaboost );
            break;
        case DECISION_TREE:
            dtree.enableNullRejection( nullRejection );
            dtree.setNullRejectionCoeff( 3 );
            dtree.setMaxDepth( 10 );
            dtree.setMinNumSamplesPerNode( 3 );
            dtree.setRemoveFeaturesAtEachSpilt( false );
            pipeline.setClassifier( dtree );
            break;
        case KKN:
            knn.enableNullRejection( nullRejection );
            knn.setNullRejectionCoeff( 3 );
            pipeline.setClassifier( knn );
            break;
        case GAUSSIAN_MIXTURE_MODEL:
            gmm.enableNullRejection( nullRejection );
            gmm.setNullRejectionCoeff( 3 );
            pipeline.setClassifier( gmm );
            break;
        case NAIVE_BAYES:
            naiveBayes.enableNullRejection( nullRejection );
            naiveBayes.setNullRejectionCoeff( 3 );
            pipeline.setClassifier( naiveBayes );
            break;
        case MINDIST:
            minDist.enableNullRejection( nullRejection );
            minDist.setNullRejectionCoeff( 3 );
            pipeline.setClassifier( minDist );
            break;
        case RANDOM_FOREST_10:
            randomForest.enableNullRejection( nullRejection );
            randomForest.setNullRejectionCoeff( 3 );
            randomForest.setForestSize( 10 );
            randomForest.setNumRandomSplits( 2 );
            randomForest.setMaxDepth( 10 );
            randomForest.setMinNumSamplesPerNode( 3 );
            randomForest.setRemoveFeaturesAtEachSpilt( false );
            pipeline.setClassifier( randomForest );
            break;
        case RANDOM_FOREST_100:
            randomForest.enableNullRejection( nullRejection );
            randomForest.setNullRejectionCoeff( 3 );
            randomForest.setForestSize( 100 );
            randomForest.setNumRandomSplits( 2 );
            randomForest.setMaxDepth( 10 );
            randomForest.setMinNumSamplesPerNode( 3 );
            randomForest.setRemoveFeaturesAtEachSpilt( false );
            pipeline.setClassifier( randomForest );
            break;
        case RANDOM_FOREST_200:
            randomForest.enableNullRejection( nullRejection );
            randomForest.setNullRejectionCoeff( 3 );
            randomForest.setForestSize( 200 );
            randomForest.setNumRandomSplits( 2 );
            randomForest.setMaxDepth( 10 );
            randomForest.setMinNumSamplesPerNode( 3 );
            randomForest.setRemoveFeaturesAtEachSpilt( false );
            pipeline.setClassifier( randomForest );
            break;
        case SOFTMAX:
            softmax.enableNullRejection( false ); //Does not support null rejection
            softmax.setNullRejectionCoeff( 3 );
            pipeline.setClassifier( softmax );
            break;
        case SVM_LINEAR:
            svm.enableNullRejection( nullRejection );
            svm.setNullRejectionCoeff( 3 );
            pipeline.setClassifier( SVM(SVM::LINEAR_KERNEL) );
            break;
        case SVM_RBF:
            svm.enableNullRejection( nullRejection );
            svm.setNullRejectionCoeff( 3 );
            pipeline.setClassifier( SVM(SVM::RBF_KERNEL) );
            break;
        default:
            return false;
            break;
    }
    
    return true;
}

//--------------------------------------------------------------
void testApp::keyReleased(int key){
    
}

//--------------------------------------------------------------
void testApp::mouseMoved(int x, int y ){
    
    
    
}

//--------------------------------------------------------------
void testApp::mouseDragged(int x, int y, int button){
    
}

//--------------------------------------------------------------
void testApp::mousePressed(int x, int y, int button){
    
}

//--------------------------------------------------------------
void testApp::mouseReleased(int x, int y, int button){
    
}

//--------------------------------------------------------------
void testApp::windowResized(int w, int h){
    
}

