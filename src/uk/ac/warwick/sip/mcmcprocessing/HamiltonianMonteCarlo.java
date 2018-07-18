package uk.ac.warwick.sip.mcmcprocessing;

import org.apache.commons.math3.random.MersenneTwister;

import processing.core.PApplet;
import uk.ac.warwick.sip.mcmcccfe.TargetDistribution;

public class HamiltonianMonteCarlo extends McmcApplet{
	
	
	protected uk.ac.warwick.sip.mcmcccfe.HamiltonianMonteCarlo chain;
	protected TargetDistribution target;
	protected int nLeapFrog;
	
	
	@Override
	public void setup() {
		this.target = this.getNormalDistribution();
		this.nLeapFrog = 10;
	}
	
	@Override
	protected void drawMcmc() {
		
		
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
		if (this.chain.getIsAccepted()) {
			
			this.stroke(255,255,0);
			this.fill(255,255,0);
			double [] leapFrogPosition;
			for (int i=0; i<this.nLeapFrog; i++) {
				leapFrogPosition = this.chain.getLeapFrogPositions(i);
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
	protected void takeStep() {
		this.chain.step();
	}
	
	@Override
	protected void changeProperty() {
		this.chain.setNLeapFrog(this.nLeapFrog);
	}
	
	@Override
	public void mouseReleased() {
		double [] mousePosition = new double [2];
		mousePosition[0] = (double) this.mouseX;
		mousePosition[1] = (double) this.mouseY;
		
		if (this.mouseButton == PApplet.LEFT) {
			MersenneTwister rng = new MersenneTwister(this.millis());
			this.chain = new uk.ac.warwick.sip.mcmcccfe.HamiltonianMonteCarlo(this.target
					, this.chainLength, this.getProposalCovarianceDiag(), SIZE_LEAP_FROG
					, this.nLeapFrog, rng);
			this.chain.setInitialValue(mousePosition);
			this.isInit = true;
		}
	}
	
	@Override
	protected void checkChangeProperty() {
		if (this.isInit) {
			if (this.keyCode == PApplet.UP) {
				if (this.nLeapFrog != this.chain.getMaxNLeapFrog()) {
					this.nLeapFrog++;
					this.changeProperty();
				}
			} else if (this.keyCode == PApplet.DOWN) {
				if (this.nLeapFrog != 1) {
					this.nLeapFrog--;
					this.changeProperty();
				}
			} else if (this.key == 'm') {
				this.nLeapFrog = this.chain.getMaxNLeapFrog();
				this.changeProperty();
			}
		}
	}

	public static void main(String[] args) {
		PApplet.main("uk.ac.warwick.sip.mcmcprocessing.HamiltonianMonteCarlo");
	}

}
