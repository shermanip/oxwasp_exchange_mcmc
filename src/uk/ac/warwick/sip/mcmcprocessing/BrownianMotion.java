package uk.ac.warwick.sip.mcmcprocessing;

import org.apache.commons.math3.random.MersenneTwister;
import org.ejml.simple.SimpleMatrix;

import processing.core.PApplet;
import uk.ac.warwick.sip.mcmcccfe.BrownianParticle;

public class BrownianMotion extends McmcApplet{
	
	protected BrownianParticle particle;
	protected final double diftMagnitude = 10.0;
	
	@Override
	public void setup() {
		this.proposalVariance = 1000.0;
	}
	
	@Override
	protected void drawMcmc() {
		this.stroke(0,255,0);
		this.fill(0,255,0);
		float x1, x2, y1, y2;
		double [] chainArray = this.particle.getChain();
		x1 = (float) chainArray[0];
		y1 = (float) chainArray[1];
		this.ellipse(x1, y1 , CIRCLE_SIZE, CIRCLE_SIZE);
		for (int i=1; i<=this.particle.getNStep(); i++) {
			x2 = (float) chainArray[i*2];
			y2 = (float) chainArray[i*2+1];
			this.ellipse(x2, y2 , CIRCLE_SIZE, CIRCLE_SIZE);
			this.line(x1, y1, x2, y2);
			x1 = x2;
			y1 = y2;
		}
		if (this.particle.getHasDrift()) {
			this.stroke(255,255,255);
			double [] attractionVector = this.particle.getAttractionVector();
			this.pushMatrix();
			this.translate((float)attractionVector[0], (float)attractionVector[1]);
			this.line(-10, -10, 10, 10);
			this.line(-10, 10, 10, -10);
			this.popMatrix();
		}
	}
	
	@Override
	protected void takeStep() {
		this.particle.step();
	}
	
	@Override
	protected void changeProperty() {
		this.particle.setProposalCovariance(this.getProposalCovariance());
	}
	
	@Override
	public void mouseReleased() {
		double [] mousePosition = new double [2];
		mousePosition[0] = (double) this.mouseX;
		mousePosition[1] = (double) this.mouseY;
		
		if (this.mouseButton == PApplet.LEFT) {
			MersenneTwister rng = new MersenneTwister(this.millis());
			this.particle = new BrownianParticle(2, this.chainLength
					, SimpleMatrix.identity(2).scale(this.proposalVariance)
					, this.diftMagnitude, rng);
			this.particle.setInitialValue(mousePosition);
			this.isInit = true;
		}
		if (this.mouseButton == PApplet.RIGHT) {
			this.particle.setHasDrift(true);
			this.particle.setAttractionVector(mousePosition);
		}
	}
	
	
	public static void main(String[] args) {
		PApplet.main("uk.ac.warwick.sip.mcmcprocessing.BrownianMotion");
	}

}
