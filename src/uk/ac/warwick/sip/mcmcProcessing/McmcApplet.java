package uk.ac.warwick.sip.mcmcProcessing;

import org.ejml.simple.SimpleMatrix;

import g4p_controls.GButton;
import g4p_controls.GEvent;
import processing.core.PApplet;
import uk.ac.warwick.sip.mcmc.TargetDistribution;
import uk.ac.warwick.sip.mcmc.NormalDistribution;

public abstract class McmcApplet extends PApplet{
  
  public static final double NORMAL_TARGET_VARIANCE = 40000.0;
  public static final double SIZE_LEAP_FROG = 10000.0;
  public static final int CIRCLE_SIZE = 10;
  public static final int CHAIN_LENGTH = 1024;
  public static final int N_CONTOUR = 10;
  
  public static final int [] UNPAUSED_COLOUR = {0,0,0};
  public static final int [] PAUSED_COLOUR = {0,33,71};
  
  protected uk.ac.warwick.sip.mcmc.Mcmc chain;
  
  protected boolean isPaused = false;
  protected boolean isToTakeStep = true;
  protected double proposalVariance = 113288.0;
  protected boolean isInit = false;
  protected int chainLength = 1024;
  protected boolean isDrawNormalContour = false;
  
  protected GButton pauseButton;
  protected GButton stepButton;
  
  protected boolean isMouseClickOnGui = false;
  
  @Override
  public void settings() {
    //this.fullScreen();
    this.size(1024, 768);
  }
  
  public void setup() {
    int nFrame = 10;
    String [] label = new String[nFrame];
    for (int i=0; i<nFrame; i++) {
      label[i] = Double.toString((1.0/((double)nFrame)) * (i+1));
      label[i] = label[i].substring(0, 3);
    }
    this.pauseButton = new GButton(this, 10,10,50,50,"pause");
    this.stepButton = new GButton(this, 10,80,50,50,"step");
  }
  
  @Override
  public void draw() {
    if (this.isPaused) {
      this.background(PAUSED_COLOUR[0], PAUSED_COLOUR[1], PAUSED_COLOUR[2]);
    } else {
      this.background(UNPAUSED_COLOUR[0], UNPAUSED_COLOUR[1], UNPAUSED_COLOUR[2]);
    }
    if (this.isDrawNormalContour) {
      this.drawNormalContour();
    }
    this.drawOtherGui();
    if (this.isInit) {
      this.drawMcmc();
      if (this.isToTakeStep) {
        this.takeStep();
        if (this.isPaused) {
          this.isToTakeStep = false;
        }
      }
    }
  }
  
  public double [] drawAllSamples() {
    this.stroke(0,255,0);
    this.fill(0,255,0);
    float x1, x2, y1, y2;
    double [] chainArray = this.chain.getChain();
    x1 = (float) chainArray[0];
    y1 = (float) chainArray[1];
    this.ellipse(x1, y1 , CIRCLE_SIZE, CIRCLE_SIZE);
    for (int i=1; i<=this.chain.getNStep(); i++) {
      x2 = (float) chainArray[i*2];
      y2 = (float) chainArray[i*2+1];
      this.ellipse(x2, y2 , CIRCLE_SIZE, CIRCLE_SIZE);
      this.line(x1, y1, x2, y2);
      x1 = x2;
      y1 = y2;
    }
    
    double [] lastSample = new double[2];
    lastSample[0] = x1;
    lastSample[1] = y1;
    return lastSample;
  }
  
  public double [] drawAllButLastSamples() {
    this.stroke(0,255,0);
    this.fill(0,255,0);
    float x1, x2, y1, y2;
    double [] chainArray = this.chain.getChain();
    x1 = (float) chainArray[0];
    y1 = (float) chainArray[1];
    this.ellipse(x1, y1 , CIRCLE_SIZE, CIRCLE_SIZE);
    for (int i=1; i<this.chain.getNStep(); i++) {
      x2 = (float) chainArray[i*2];
      y2 = (float) chainArray[i*2+1];
      this.ellipse(x2, y2 , CIRCLE_SIZE, CIRCLE_SIZE);
      this.line(x1, y1, x2, y2);
      x1 = x2;
      y1 = y2;
    }
    
    double [] lastSample = new double[2];
    lastSample[0] = x1;
    lastSample[1] = y1;
    return lastSample;
  }
  
  protected void drawOtherGui(){
    //doing nothing
  }
  
  protected SimpleMatrix getProposalCovariance() {
    return SimpleMatrix.identity(2).scale(this.proposalVariance);
  }
  
  protected SimpleMatrix getProposalCovarianceDiag() {
    SimpleMatrix covarianceDiag = new SimpleMatrix(2,1);
    covarianceDiag.set(0, this.proposalVariance);
    covarianceDiag.set(1, this.proposalVariance);
    return covarianceDiag;
  }
  protected abstract void drawMcmc();
  
  protected void takeStep() {
    this.chain.step();
  }
  
  protected abstract void changeProperty();
  
  public void keyReleased() {
    if (this.key == 'p') {
      this.isPaused = !this.isPaused;
      if (!this.isPaused) {
        this.isToTakeStep = true;
      } else {
        this.isToTakeStep = false;
      }
    } else if (this.key == ' ') {
      this.isToTakeStep = true;
    } else if (this.key == 'q') {
      this.exit();
    } else {
      this.checkChangeProperty();
    }
  }
  
  protected void checkChangeProperty() {
    if (this.keyCode == PApplet.UP) {
      this.proposalVariance *= 10;
      if (this.isInit) {
        this.changeProperty();
      }
    } else if (this.keyCode == PApplet.DOWN) {
      this.proposalVariance /= 10;
      if (this.isInit) {
        this.changeProperty();
      }
    }
  }
  
  protected TargetDistribution getNormalDistribution() {
    this.isDrawNormalContour = true;
    SimpleMatrix targetCovariance = new SimpleMatrix(2, 2);
    targetCovariance.set(0, 0, NORMAL_TARGET_VARIANCE);
    targetCovariance.set(1, 1, NORMAL_TARGET_VARIANCE);
    SimpleMatrix mean = new SimpleMatrix(2, 1, true, this.getCentre());
    return new NormalDistribution(2, mean, targetCovariance);
  }
  
  protected double [] getCentre() {
    double [] centre = new double [2];
    centre[0] = ((double)this.width)/2;
    centre[1] = ((double)this.height)/2;
    return centre;
  }
  
  //only for symmetric normal
  protected void drawNormalContour() {
    org.apache.commons.math3.distribution.NormalDistribution normalDistribution =
        new org.apache.commons.math3.distribution.NormalDistribution(0
            , Math.sqrt(NORMAL_TARGET_VARIANCE));
    this.stroke(255,255,255);
    this.noFill();
    double [] meanVector = this.getCentre();
    
    double pSpacing = 0.5 /  ((double)(N_CONTOUR+1));
    double cdf = 0.5 + pSpacing;
    for (int i=0; i<N_CONTOUR; i++) {
      float contourDiameter =
          (float) (normalDistribution.inverseCumulativeProbability(cdf) * 2.0);
      cdf += pSpacing;
      this.ellipse((float)meanVector[0], (float)meanVector[1], contourDiameter, contourDiameter);
    }
    
  }
  
  
  public void handleButtonEvents(GButton button, GEvent event) {
    if (event == GEvent.CLICKED) {
      if (button == this.pauseButton) {
        this.isPaused = !this.isPaused;
        if (!this.isPaused) {
          this.isToTakeStep = true;
        } else {
          this.isToTakeStep = false;
        }
      }
      if (button == this.stepButton) {
        this.isToTakeStep = true;
      }
    }
  }
  
  public void mousePressed() {
    if (this.isMouseOnGui()) {
      this.isMouseClickOnGui = true;
    }
  }
  
  public void mouseReleased() {
    this.isMouseClickOnGui = false;
  }
  
  protected boolean isMouseOnGui() {
    boolean isMouseOnGui = false;
    if (this.pauseButton.isOver(this.mouseX, this.mouseY)) {
      isMouseOnGui = true;
    }
    if (this.stepButton.isOver(this.mouseX, this.mouseY)) {
      isMouseOnGui = true;
    }
    return isMouseOnGui;
  }
  
}
