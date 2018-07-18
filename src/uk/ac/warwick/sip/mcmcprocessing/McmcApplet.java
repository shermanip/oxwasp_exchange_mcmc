package uk.ac.warwick.sip.mcmcprocessing;

import org.ejml.simple.SimpleMatrix;

import processing.core.PApplet;

public abstract class McmcApplet extends PApplet{
	
	protected uk.ac.warwick.sip.mcmcccfe.RandomWalkMetropolisHastings chain;
	
	protected boolean isPaused = false;
	protected boolean isToTakeStep = true;
	protected double proposalVariance = 100;
	protected boolean isInit = false;
	protected int chainLength = 1024;
	
	protected int [] unpausedColour = {0,0,0};
	protected int [] pausedColour = {0,33,71};
	
	@Override
	public void settings() {
		this.size(1024, 768);
	}
	
	public void draw() {
		if (this.isPaused) {
			this.background(pausedColour[0], pausedColour[1], pausedColour[2]);
		} else {
			this.background(unpausedColour[0], unpausedColour[1], unpausedColour[2]);
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

}
