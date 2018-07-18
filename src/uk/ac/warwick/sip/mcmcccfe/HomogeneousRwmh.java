package uk.ac.warwick.sip.mcmcccfe;

import org.apache.commons.math3.random.MersenneTwister;
import org.ejml.simple.SimpleMatrix;

/**CLASS: HOMOGENEOUS RANDOM WALK METROPOLIS HASTINGS
 * See superclass RandomWalkMetropolisHastings
 * Runs the Metropolis Hastings algorithm but with a constant proposal covariance
 * The proposal covariance is to be provided via the constructor
 * The run method can be used to run the Metropolis Hastings algorithm for the whole chain
 * @see RandomWalkMetropolisHastings.java
 */
public class HomogeneousRwmh extends RandomWalkMetropolisHastings{
	
	//proposal covariance decomposed  using cholesky
	protected SimpleMatrix proposalCovarianceChol;
	
	/**CONSTRUCTOR
	 * @param target See superclass RandomWalkMetropolisHastings
	 * @param chainLength See superclass RandomWalkMetropolisHastings
	 * @param proposalCovariance proposal covariance use in homogeneous steps
	 * @param rng See superclass RandomWalkMetropolisHastings
	 */
	public HomogeneousRwmh(TargetDistribution target, int chainLength, SimpleMatrix proposalCovariance,
			MersenneTwister rng) {
		//call constructors and assign member variables
		super(target, chainLength, rng);
		this.proposalCovarianceChol = Global.cholesky(proposalCovariance);
	}
	
	/**OVERRIDE: STEP
	 * This chains takes a Metropolis-Hastings step and updates it member variables
	 * this.proposalCovarianceChol lower cholesky decomposition is used for the proposal_covariance
	 */
	@Override
	public void step(){
		//if there are MCMC steps to do...
		if (this.nStep+1 < this.chainLength) {
			this.metropolisHastingsStep(this.proposalCovarianceChol);
			this.updateStatistics();
		}
	}
	
	/**OVERRIDE: RUN
	 * Take multiple Metropolis-Hastings steps to complete the MCMC
	 */
	@Override
	public void run() {
		for (int i=0; i<(this.chainLength-1); i++) {
			this.step();
		}
	}
	
	/**METHOD: SET PROPOSAL COVARIANCE
	 * Set the proposal covariance
	 * @param proposalCovariance proposal covariance use in homogeneous steps
	 */
	public void setProposalCovariance(SimpleMatrix proposalCovariance) {
		this.proposalCovarianceChol = Global.cholesky(proposalCovariance);
	}
}
