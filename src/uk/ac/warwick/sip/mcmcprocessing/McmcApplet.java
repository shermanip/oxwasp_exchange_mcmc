package uk.ac.warwick.sip.mcmcprocessing;

import org.ejml.simple.SimpleMatrix;

import processing.core.PApplet;
import uk.ac.warwick.sip.mcmcccfe.TargetDistribution;
import uk.ac.warwick.sip.mcmcccfe.NormalDistribution;

public abstract class McmcApplet extends PApplet{
  
  public static final double NORMAL_TARGET_VARIANCE = 40000.0;
  public static final double SIZE_LEAP_FROG = 10000.0;
  public static final int CIRCLE_SIZE = 10;
  public static final int CHAIN_LENGTH = 1024;
  public static final int N_CONTOUR = 10;
  
  public static final int [] UNPAUSED_COLOUR = {0,0,0};
  public static final int [] PAUSED_COLOUR = {0,33,71};
  
  protected uk.ac.warwick.sip.mcmcccfe.RandomWalkMetropolisHastings chain;
  
  protected boolean isPaused = false;
  protected boolean isToTakeStep = true;
  protected double proposalVariance = 113288.0;
  protected boolean isInit = false;
  protected int chainLength = 1024;
  protected boolean isDrawNormalContour = false;
  
  
  @Override
  public void settings() {
    this.fullScreen();
    //this.size(1024, 768);
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
  protected abstract void takeStep();
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
  
}
