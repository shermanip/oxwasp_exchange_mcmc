package uk.ac.warwick.sip.mcmcccfe;

import org.ejml.simple.SimpleMatrix;

/**CLASS: TARGET DISTRIBUTION
 * Evaluates the target distribution pdf
 */
public abstract class TargetDistribution {
	
	protected int nDim; //number of dimensions of the target distribution
	
	/**CONSTRUCTOR
	 * @param nDim number of dimensions of the target distribution
	 */
	public TargetDistribution(int nDim){
		this.nDim = nDim;
	}
	
	/**METHOD: GET PDF
	 * Evaluate the probability density function at x
	 * The pdf needs not to be normalised, ie integrate to 1
	 * The normalisation constant is not needed for Metropolis-Hastings
	 * @param x Where to evaluate the pdf, column vector
	 * @return The evaluation of the pdf at x up to a constant
	 */
	public abstract double getPdf(SimpleMatrix x);
	
	/**METHOD: GET POTENTIAL
	 * Evaluate the -ln pdf + some constant
	 * The constant comes from the face the pdf is evaluated up to a constant
	 * @param x Where to evaluate the potential
	 * @return The evaluation of the potential at x + some constant
	 */
	public abstract double getPotential(SimpleMatrix x);
	
	/**METHOD: GET D POTENTIAL
	 * Evaluate the differential of -ln pdf
	 * @param x Where to evaluate the potential gradient
	 * @return The evaluation of the potential gradient at x
	 */
	public abstract SimpleMatrix getDPotential(SimpleMatrix x);
	
	/**GET N DIM
	 * @return The number of dimensions this target has
	 */
	public int getNDim() {
		return this.nDim;
	}

}
