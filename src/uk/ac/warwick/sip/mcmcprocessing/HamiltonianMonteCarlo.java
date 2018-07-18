package uk.ac.warwick.sip.mcmcprocessing;

import org.apache.commons.math3.random.MersenneTwister;
import org.ejml.simple.SimpleMatrix;

import processing.core.PApplet;
import uk.ac.warwick.sip.mcmcccfe.HomogeneousRwmh;
import uk.ac.warwick.sip.mcmcccfe.NormalDistribution;
import uk.ac.warwick.sip.mcmcccfe.TargetDistribution;

public class HamiltonianMonteCarlo extends McmcApplet{

	
	protected uk.ac.warwick.sip.mcmcccfe.HamiltonianMonteCarlo chain;
	protected TargetDistribution target;
	protected double targetVariance = 1000;
	protected double sizeLeapFrog;
	protected int nLeapFrog;
	
	
	@Override
	public void setup() {
		SimpleMatrix targetCovariance = new SimpleMatrix(2, 2);
		targetCovariance.set(0, 0, this.targetVariance);
		targetCovariance.set(1, 1, this.targetVariance);
		SimpleMatrix mean = new SimpleMatrix(2, 1);
		mean.set(0, ((double)this.width)/2);
		mean.set(1, ((double)this.height)/2);
		this.target = new NormalDistribution(2, mean, targetCovariance);
		this.sizeLeapFrog = 50;
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
		this.ellipse(x1, y1 , 5, 5);
		for (int i=1; i<this.chain.getNStep(); i++) {
			x2 = (float) chainArray[i*2];
			y2 = (float) chainArray[i*2+1];
			this.ellipse(x2, y2 , 5, 5);
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
				this.ellipse(x2, y2 , 5, 5);
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
		//this.chain.setProposalCovariance(this.getProposalCovariance());
	}
	
	@Override
	public void mouseReleased() {
		double [] mousePosition = new double [2];
		mousePosition[0] = (double) this.mouseX;
		mousePosition[1] = (double) this.mouseY;
		
		if (this.mouseButton == PApplet.LEFT) {
			MersenneTwister rng = new MersenneTwister(this.millis());
			this.chain = new uk.ac.warwick.sip.mcmcccfe.HamiltonianMonteCarlo(this.target
					, this.chainLength, this.getProposalCovarianceDiag(), this.sizeLeapFrog
					, this.nLeapFrog, rng);
			this.chain.setInitialValue(mousePosition);
			this.isInit = true;
		}
	}
	
	@Override
	protected void checkChangeProperty() {
		if (this.keyCode == PApplet.UP) {
		} else if (this.keyCode == PApplet.DOWN) {
		}
	}

	public static void main(String[] args) {
		PApplet.main("uk.ac.warwick.sip.mcmcprocessing.HamiltonianMonteCarlo");
	}

}
