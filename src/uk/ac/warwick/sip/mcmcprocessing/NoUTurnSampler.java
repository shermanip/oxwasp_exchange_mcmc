package uk.ac.warwick.sip.mcmcprocessing;

import java.util.Iterator;

import org.apache.commons.math3.random.MersenneTwister;
import org.ejml.simple.SimpleMatrix;

import processing.core.PApplet;
import uk.ac.warwick.sip.mcmcccfe.NormalDistribution;
import uk.ac.warwick.sip.mcmcccfe.TargetDistribution;

public class NoUTurnSampler extends McmcApplet{
	
	protected uk.ac.warwick.sip.mcmcccfe.NoUTurnSampler chain;
	protected TargetDistribution target;
	protected double targetVariance = 1000;
	protected double sizeLeapFrog;
	
	
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
		for (int i=1; i<=this.chain.getNStep(); i++) {
			x2 = (float) chainArray[i*2];
			y2 = (float) chainArray[i*2+1];
			this.ellipse(x2, y2 , 5, 5);
			if (i != this.chain.getNStep()) {
				this.line(x1, y1, x2, y2);
			}
			x1 = x2;
			y1 = y2;
		}
		if (this.chain.getIsAccepted()) {
			
			this.stroke(255,255,0);
			this.fill(255,255,0);
			double [] leapFrogPosition;
			Iterator<SimpleMatrix> leapFrogPositionIterator =
					this.chain.getLeapFrogPositionIterator();
			leapFrogPosition = leapFrogPositionIterator.next().getDDRM().getData();
			x1 = (float) leapFrogPosition[0];
			y1 = (float) leapFrogPosition[1];
			this.ellipse(x1, y1 , 5, 5);
			while (leapFrogPositionIterator.hasNext()) {
				leapFrogPosition = leapFrogPositionIterator.next().getDDRM().getData();
				x2 = (float) leapFrogPosition[0];
				y2 = (float) leapFrogPosition[1];
				this.ellipse(x2, y2 , 5, 5);
				this.line(x1, y1, x2, y2);
				x1 = x2;
				y1 = y2;
			}
			this.stroke(0,0,255);
			this.fill(0,0,255);
			x2 = (float) chainArray[this.chain.getNStep()*2];
			y2 = (float) chainArray[this.chain.getNStep()*2+1];
			this.ellipse(x2, y2 , 5, 5);
		}
	}
	
	@Override
	protected void takeStep() {
		this.chain.step();
	}
	
	@Override
	protected void changeProperty() {
	}
	
	@Override
	public void mouseReleased() {
		double [] mousePosition = new double [2];
		mousePosition[0] = (double) this.mouseX;
		mousePosition[1] = (double) this.mouseY;
		
		if (this.mouseButton == PApplet.LEFT) {
			MersenneTwister rng = new MersenneTwister(this.millis());
			this.chain = new uk.ac.warwick.sip.mcmcccfe.NoUTurnSampler(this.target
					, this.chainLength, this.getProposalCovarianceDiag(), this.sizeLeapFrog
					, rng);
			this.chain.setInitialValue(mousePosition);
			this.isInit = true;
		}
	}
	

	public static void main(String[] args) {
		PApplet.main("uk.ac.warwick.sip.mcmcprocessing.NoUTurnSampler");
	}

}