package uk.ac.warwick.sip.mcmcProcessing;

import org.apache.commons.math3.random.MersenneTwister;

import g4p_controls.G4P;
import g4p_controls.GEvent;
import g4p_controls.GSlider;
import g4p_controls.GValueControl;
import processing.core.PApplet;
import uk.ac.warwick.sip.mcmc.TargetDistribution;

public class HamiltonianMonteCarlo extends McmcApplet{
  
  protected TargetDistribution target;
  protected int nLeapFrog = 10;
  protected GSlider nLeapFrogSlider;
  
  @Override
  public void setup() {
    super.setup();
    target = this.getNormalDistribution();
    this.nLeapFrogSlider = new GSlider(this, 110, 150, 300, 150, 30);
    this.nLeapFrogSlider.setRotation(HALF_PI);
    this.nLeapFrogSlider.setShowValue(true);
    this.nLeapFrogSlider.setLimits(10, 1,
        uk.ac.warwick.sip.mcmc.HamiltonianMonteCarlo.maxNLeapFrog);
    this.nLeapFrogSlider.setTextOrientation(-1);
    this.nLeapFrogSlider.setNbrTicks(11);
    this.nLeapFrogSlider.setLocalColor(2,-1);
    this.nLeapFrogSlider.setShowTicks(true);
    this.nLeapFrogSlider.setNumberFormat(G4P.INTEGER, 0);

  }
  
  @Override
  protected void drawMcmc() {
    
    double [] secondLastSample = this.drawAllButLastSamples();
    float x1, y1, x2, y2;
    x1 = (float) secondLastSample[0];
    y1 = (float) secondLastSample[1];
    if (this.chain.getIsAccepted()) {
      
      this.stroke(255,255,0);
      this.fill(255,255,0);
      double [] leapFrogPosition;
      uk.ac.warwick.sip.mcmc.HamiltonianMonteCarlo chain = 
          (uk.ac.warwick.sip.mcmc.HamiltonianMonteCarlo) this.chain;
      for (int i=0; i<this.nLeapFrogSlider.getValueI(); i++) {
        leapFrogPosition = chain.getLeapFrogPositions(i);
        x2 = (float) leapFrogPosition[0];
        y2 = (float) leapFrogPosition[1];
        this.ellipse(x2, y2 , CIRCLE_SIZE, CIRCLE_SIZE);
        this.line(x1, y1, x2, y2);
        x1 = x2;
        y1 = y2;
      }
    }
  }
  
  @Override
  protected void changeProperty() {
    uk.ac.warwick.sip.mcmc.HamiltonianMonteCarlo chain = 
        (uk.ac.warwick.sip.mcmc.HamiltonianMonteCarlo) this.chain;
    chain.setNLeapFrog(this.nLeapFrog);
  }
  
  @Override
  public void mouseReleased() {
    if (!this.isMouseClickOnGui) {
      if (!this.isMouseOnGui()) {
        double [] mousePosition = new double [2];
        mousePosition[0] = (double) this.mouseX;
        mousePosition[1] = (double) this.mouseY;
        
        MersenneTwister rng = new MersenneTwister(this.millis());
        this.chain = new uk.ac.warwick.sip.mcmc.HamiltonianMonteCarlo(this.target
            , this.chainLength, this.getProposalCovariance(), SIZE_LEAP_FROG
            , this.nLeapFrogSlider.getValueI(), rng);
        this.chain.setInitialValue(mousePosition);
        this.isInit = true;
      }
    }
    super.mouseReleased();
  }
  
  protected boolean isMouseOnGui() {
    boolean isMouseOnGui = super.isMouseOnGui();
    if (this.nLeapFrogSlider.isOver(this.mouseX, this.mouseY)) {
      isMouseOnGui = true;
    }
    return isMouseOnGui;
  }
  
  public static void main(String[] args) {
    PApplet.main("uk.ac.warwick.sip.mcmcProcessing.HamiltonianMonteCarlo");
  }
  
  public void handleSliderEvents(GValueControl slider, GEvent event) {
    if (slider == this.nLeapFrogSlider) {
      if (event == GEvent.VALUE_STEADY) {
        if (this.isInit) {
          uk.ac.warwick.sip.mcmc.HamiltonianMonteCarlo chain = 
              (uk.ac.warwick.sip.mcmc.HamiltonianMonteCarlo) this.chain;
          chain.setNLeapFrog(this.nLeapFrogSlider.getValueI());
        }
      }
    }
  }
  
}
