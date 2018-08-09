package uk.ac.warwick.sip.mcmc;

import org.ejml.simple.SimpleMatrix;
import seed.minerva.GraphicalModel;

/**CLASS: GRAPH DISTRIBUTION
 * Evaluate the pdf using the MINERVA GraphicalModel object
 */
public class GraphDistribution extends TargetDistribution {
	
	//MINERVA object which can evaluate the log pdf
	protected GraphicalModel graph;
	protected int nMethodCall = 0; //number of times a method is called
	protected double delta = 0.01; //the step size used to numerically calculate the diff
	
	
	/**CONSTRUCTOR
	 * Construct the target pdf using the minerva GraphicalModel object
	 * @param graph GraphicalModel which can evaluate the log pdf
	 */
	public GraphDistribution(GraphicalModel graph) {
		super(graph.getFreeParameters().length);
		this.graph = graph;
	}
	
	/**IMPLEMENTED: GET PDF
	 * @param x Where to evaluate the pdf, column vector
	 * @return The evaluation of the pdf at x up to a constant
	 */
	@Override
	public double getPdf(SimpleMatrix x) {
		return Math.exp(-this.getPotential(x));
	}
	
	/**IMPLEMENTED: GET POTENTIAL
	 * @param x Where to evaluate the potential
	 * @return The evaluation of the potential at x + some constant
	 */
	@Override
	public double getPotential(SimpleMatrix x) {
		this.nMethodCall++;
		this.graph.setFreeParameters(x.getDDRM().getData());
		return -this.graph.logPdf();
	}
	
	/**IMPLEMENTED: GET D POTENTIAL
	 * Returns a vector of partical diff of the potential
	 * Differentation is done numerically for a given step size this.delta
	 * @param x Where to evaluate the potential gradient
	 * @return The evaluation of the potential gradient at x
	 */
	@Override
	public SimpleMatrix getDPotential(SimpleMatrix x) {
		//instantiate a column vector
		SimpleMatrix dPotential = new SimpleMatrix(this.getNDim(), 1);
		//for each dimension
		for (int iDim=0; iDim<this.getNDim(); iDim++) {
			//get the position +/- a small number this.delta
			SimpleMatrix xMinus = new SimpleMatrix(x);
			SimpleMatrix xPlus = new SimpleMatrix(x);
			xMinus.set(iDim, x.get(iDim)-this.delta);
			xPlus.set(iDim, x.get(iDim)+this.delta);
			//calculate the diff potential and save it for this dimension
			double diffIPotential =(this.getPotential(xPlus) 
					- this.getPotential(xMinus)) / (2*this.delta);
			dPotential.set(iDim, diffIPotential);
		}
		return dPotential;
	}

}
