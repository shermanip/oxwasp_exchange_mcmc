package uk.ac.warwick.sip.mcmcProcessing;

import org.apache.commons.math3.random.MersenneTwister;
import org.ejml.simple.SimpleMatrix;

import g4p_controls.GCheckbox;
import g4p_controls.GEvent;
import g4p_controls.GToggleControl;
import processing.core.PApplet;
import processing.core.PVector;
import uk.ac.warwick.sip.mcmc.TargetDistribution;

public class RandomWalkMetropolisHastings extends McmcApplet{
  
  protected TargetDistribution target;
  protected GCheckbox adaptiveCheckBox;
  protected boolean isCreatingChain = false;
  protected double [] clickPosition = new double[2];
  
  @Override
  public void setup() {
    super.setup();
    this.target = this.getNormalDistribution();
    this.adaptiveCheckBox = new GCheckbox(this, 12,150,75,40,"Adaptive");
    this.adaptiveCheckBox.setLocalColorScheme(255);
  }
  
  @Override
  protected void drawMcmc() {
    if (!this.isCreatingChain) {
      double [] lastSample = this.drawAllSamples();
      if (!this.chain.getIsAccepted()) {
        double [] rejectedSample = this.chain.getRejectedSample();
        float x1 = (float) lastSample[0];
        float y1 = (float) lastSample[1];
        float x2 = (float) rejectedSample[0];
        float y2 = (float) rejectedSample[1];
        this.stroke(255, 0, 0);
        this.fill(255, 0, 0);
        this.ellipse(x2, y2 , CIRCLE_SIZE, CIRCLE_SIZE);
        this.line(x1, y1, x2, y2);
      }
    }
  }
  
  @Override
  protected void drawOtherGui() {
    if (this.isCreatingChain) {
      this.stroke(0, 255, 0);
      float radius = this.getMouseClickRadius();
      this.ellipse((float)this.clickPosition[0], (float)this.clickPosition[1],
          radius, radius);
    }
  }
  
  @Override
  protected void changeProperty() {
    uk.ac.warwick.sip.mcmc.RandomWalkMetropolisHastings chain =
        (uk.ac.warwick.sip.mcmc.RandomWalkMetropolisHastings) this.chain;
    chain.setProposalCovariance(this.getProposalCovariance());
  }
  
  public void mousePressed() {
    super.mousePressed();
    if (!this.isMouseClickOnGui) {
      this.isCreatingChain = true;
      this.clickPosition[0] = (double) this.mouseX;
      this.clickPosition[1] = (double) this.mouseY;
    }
  }
  
  public float getMouseClickRadius() {
    PVector r = new PVector();
    r.set((float)this.clickPosition[0]-this.mouseX, (float)this.clickPosition[1]-this.mouseY);
    if (r.mag()<1) {
      return (float) 1.0;
    } else {
      return 2*r.mag();
    }
  }
  
  @Override
  public void mouseReleased() {
    if (!this.isMouseClickOnGui) {
      MersenneTwister rng = new MersenneTwister(this.millis());
      double proposalVariance = Math.pow((double) this.getMouseClickRadius(),2);
      SimpleMatrix proposalCovariance = new SimpleMatrix(2,2);
      proposalCovariance.set(0,proposalVariance);
      proposalCovariance.set(3,proposalVariance);
      this.chain = new uk.ac.warwick.sip.mcmc.MixtureAdaptiveRwmh(this.target, this.chainLength
          , proposalCovariance, rng);
      this.chain.setInitialValue(this.clickPosition);
      uk.ac.warwick.sip.mcmc.MixtureAdaptiveRwmh chain = 
          (uk.ac.warwick.sip.mcmc.MixtureAdaptiveRwmh) this.chain;
      chain.setIsAdaptive(this.adaptiveCheckBox.isSelected());
      this.isInit = true;
    }
    this.isCreatingChain = false;
    super.mouseReleased();
  }
  
  public void handleToggleControlEvents(GToggleControl checkbox, GEvent event) {
    if (checkbox == this.adaptiveCheckBox) {
      if (this.isInit) {
        uk.ac.warwick.sip.mcmc.MixtureAdaptiveRwmh chain =
            (uk.ac.warwick.sip.mcmc.MixtureAdaptiveRwmh) this.chain;
        chain.setIsAdaptive(this.adaptiveCheckBox.isSelected());
      }
    }
  }
  
  protected boolean isMouseOnGui() {
    boolean isMouseOnGui = super.isMouseOnGui();
    if (this.adaptiveCheckBox.isOver(this.mouseX, this.mouseY)) {
      isMouseOnGui = true;
    }
    return isMouseOnGui;
  }
  
  
  public static void main(String[] args) {
    PApplet.main("uk.ac.warwick.sip.mcmcProcessing.RandomWalkMetropolisHastings");
  }
  
}
