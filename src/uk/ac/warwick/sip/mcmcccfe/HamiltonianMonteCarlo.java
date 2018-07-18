package uk.ac.warwick.sip.mcmcccfe;

import org.apache.commons.math3.random.MersenneTwister;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.simple.SimpleMatrix;

/**CLASS: HAMILTONIAN MONTE CARLO
 * Samplier which uses Hamiltonian dynamics.
 * At a step, the momentum vector is proposed using a Gaussian random variable.
 * The particle then moves about obeying Hamiltonian dynamics, this is done using leap frog steps
 * The resulting position of the particle is accept/rejected using the cannonical distribution
 * 
 * Use the method run() to run the chain
 */
public class HamiltonianMonteCarlo extends RandomWalkMetropolisHastings {
	
	//column vector, squared root of the diagonal of the mass matrix, for each dimension
	protected SimpleMatrix momentumScale;
	//column vector, inverse of the diagonal of the mass matrix, for each dimension
	protected SimpleMatrix inverseMass;
	//size of the leap frog step
	protected double sizeLeapFrog;
	protected int nLeapFrog; //number of leap frog step for each mcmc step
	protected int maxNLeapFrog = 100; //maximum number of leap frog steps
	//array of vectors containing position vector of each leapfrog step
	protected SimpleMatrix [] leapFrogPositions;
	
	/**CONSTRUCTOR
	 * @param target Object which has a method to call the pdf
	 * @param chainLength Length of the chain to be obtained
	 * @param massVector column vector, containing diagonal element of the mass matrix 
	 * @param sizeLeapFrog size of the leap frog step
	 * @param nLeapFrog number of leap frog step for each mcmc step
	 * @param rng Random number generator to pre-generate all the random numbers
	 */
	public HamiltonianMonteCarlo(TargetDistribution target, int chainLength,
			SimpleMatrix massVector, double sizeLeapFrog, int nLeapFrog, MersenneTwister rng) {
		//assign member variables
		super(target, chainLength, rng);
		this.momentumScale = new SimpleMatrix(this.getNDim(), 1);
		this.inverseMass = new SimpleMatrix(massVector);
		this.sizeLeapFrog = sizeLeapFrog;
		this.nLeapFrog = nLeapFrog;
		//fill leapFrogPositions with null
		this.leapFrogPositions = new SimpleMatrix [this.maxNLeapFrog];
		//assign the functions of the elements of the mass matrix
		CommonOps_DDRM.elementPower(massVector.getDDRM(), 0.5, this.momentumScale.getDDRM());
		CommonOps_DDRM.divide(1, this.inverseMass.getDDRM());
		this.isAccepted = false;
	}
	
	/**OVERRIDE: RUN
	 * Take multiple HMC steps to complete the MCMC
	 */
	@Override
	public void run() {
		for (int i=0; i<(this.chainLength-1); i++) {
			this.step();
		}
	}
	
	/**OVERRIDE: STEP
	 * Throws exception as HMC cannot do a Metropolis Hastings step
	 * @param proposalCovarianceChol not used
	 */
	@Override
	public void step(SimpleMatrix proposalCovarianceChol){
		throw new RuntimeException("HamiltonianMonteCarlo cannot do a Metropolis Hastings step");
	}
	
	/**OVERRIDE: STEP
	 * Does a HMC step. The position vector is the current position of the chain.
	 * Momentum is generated randomly using Gaussian.
	 * Leap frog steps are used to move the particle obeying Hamiltonian dynamics
	 * The resulting position vector after leap frog steps is the proposal
	 * The proposal is then accepted or rejected using the canonical distribution
	 */
	@Override
	public void step() {
		
		//if there are MCMC steps to do...
		if (this.nStep+1 < this.chainLength) {
		
			//get the position vector from the chain array, and random momentum
			SimpleMatrix position = this.getPosition();
			SimpleMatrix momentum = this.getMomentum();
			
			//instantiate SimpleMatrices for the proposal variables
			SimpleMatrix positionProposal = new SimpleMatrix(position);
			SimpleMatrix momentumProposal = new SimpleMatrix(momentum);
			
			//do the leap frog step
			this.leapFrog(positionProposal, momentumProposal);
			
			//get the canonical distributions given the hamiltonians
			double canonicalCurrent = Math.exp(-this.getHamiltonian(position, momentum));
			double canonicalProposal = Math.exp(-this.getHamiltonian(positionProposal
					,momentumProposal));
			
			//do acceptance step
			double acceptProb = canonicalProposal/canonicalCurrent;
			this.acceptStep(acceptProb, position, positionProposal);
			
			//update the statistics of itself
			this.updateStatistics();
			
		}
	}
	
	/**METHOD: GET POSITION
	 * Return the position vector from the chain array, the chain current position
	 * @return Column vector, the chain current position
	 */
	protected SimpleMatrix getPosition() {
		SimpleMatrix position = this.chainArray.extractVector(true, this.nStep);
		CommonOps_DDRM.transpose(position.getDDRM());
		return position;
	}
	
	/**METHOD: GET MOMENTUM
	 * //generate a random momentum vector
	 * //it is generated using Normal with covariance diag(momentumScale)
	 * //random Normal uses the unused N(0,1) numbers are stored in chain array
	 * @return Column vector, random momentum
	 */
	protected SimpleMatrix getMomentum() {
		SimpleMatrix momentum = this.chainArray.extractVector(true, this.nStep+1);
		CommonOps_DDRM.transpose(momentum.getDDRM());
		CommonOps_DDRM.elementMult(momentum.getDDRM(), this.momentumScale.getDDRM());
		return momentum;
	}
	
	/**METHOD: LEAP FROG
	 * Solves the Hamiltonian equations using this.nLeapFrog steps
	 * The leap frog steps consist of consecutive half momentum update, position update, then
	 * another half momentum update
	 * Two half momentum update is the same as a full momentum update, this is used in the program
	 * The position and momentum proposal vectors ARE MODIFIED
	 * Save the position for each leap frog in the member variable leapFrogPositions
	 * @param positionProposal Column vector containing the position, MODIFIED
	 * @param momentumProposal Column vector containing the momentum, MODIFIED
	 */
	protected void leapFrog(SimpleMatrix positionProposal, SimpleMatrix momentumProposal) {
		this.momentumStep(positionProposal, momentumProposal, true);
		//do leap frog steps
		for (int i=0; i<(this.nLeapFrog-1); i++) {
			this.positionStep(positionProposal, momentumProposal);
			this.momentumStep(positionProposal, momentumProposal, false);
			//save the leap frog position
			this.leapFrogPositions[i] = new SimpleMatrix(positionProposal);
		}
		this.positionStep(positionProposal, momentumProposal);
		this.momentumStep(positionProposal, momentumProposal, true);
		//save the leap frog position
		this.leapFrogPositions[this.nLeapFrog-1] = new SimpleMatrix(positionProposal);
	}
	
	/**METHOD: MOMENTUM STEP
	 * Used for the leap frog step
	 * Updates and MODIFIES the momentum proposal obeying Hamiltonian dynamics
	 * @param positionProposal Column vector, proposed position, not modified
	 * @param momentumProposal Column vector, proposed momentum, MODIFIED
	 * @param isHalfStep true to half the momentum update
	 */
	protected void momentumStep(SimpleMatrix positionProposal, SimpleMatrix momentumProposal,
			boolean isHalfStep) {
		//momentumChange is to be subtracted to momentumProposal
		SimpleMatrix momentumChange = this.target.getDPotential(positionProposal);
		CommonOps_DDRM.scale(this.sizeLeapFrog, momentumChange.getDDRM());
		if (isHalfStep) {
			CommonOps_DDRM.divide(momentumChange.getDDRM(), 2.0);
		}
		CommonOps_DDRM.subtractEquals(momentumProposal.getDDRM(), momentumChange.getDDRM());
	}
	
	/**METHOD: POSITION STEP
	 * Used for the leap frog step
	 * Updates and MODIFIES the position proposal obeying Hamiltonian dynamics
	 * @param positionProposal Column vector, proposed position, MODIFIED
	 * @param momentumProposal Column vector, proposed momentum, not modified
	 */
	protected void positionStep(SimpleMatrix positionProposal, SimpleMatrix momentumProposal) {
		SimpleMatrix positionChange = momentumProposal.elementMult(this.inverseMass);
		CommonOps_DDRM.scale(this.sizeLeapFrog, positionChange.getDDRM());
		CommonOps_DDRM.addEquals(positionProposal.getDDRM(), positionChange.getDDRM());
	}
	
	
	/**METHOD: GET HAMILTONIAN
	 * Returns the Hamiltonian (or energy of the system) given the position and momentum
	 * of the particle
	 * @param position Column vector, position of the particle
	 * @param momentum Column vector, momentum of the particle
	 * @return
	 */
	protected double getHamiltonian(SimpleMatrix position, SimpleMatrix momentum) {
		//evaluate the kinetic energy
		SimpleMatrix inverseMassTimesMomentum = this.inverseMass.elementMult(momentum);
		double kineticEnergy = 0.5 * momentum.dot(inverseMassTimesMomentum);
		//evaluate the potential
		double potentialEnergy = this.target.getPotential(position);
		//add all of the energies
		return kineticEnergy + potentialEnergy;
	}
	
	/**METHOD: GET LEAP FROG POSITIONS
	 * Return the vector of a leap frog step of the last HMC step
	 * @param index which leap frog step to be requested
	 * @return double [] representing a position vector
	 */
	public double [] getLeapFrogPositions(int index) {
		SimpleMatrix leapFrogPosition = this.leapFrogPositions[index];
		if (leapFrogPosition == null) {
			return this.getLeapFrogPositions(index-1);
		} else {
			return leapFrogPosition.getDDRM().getData();
		}
	}
	
	/**METHOD: CHANGE NUMBER OF LEAP FROG STEPS
	 * Change the number of leap frog steps to do
	 * @param nLeapFrog Number of leap frog steps to do in a HMC step
	 */
	public void setNLeapFrog(int nLeapFrog) {
		//if there are MCMC steps to do...
		if (this.nStep+1 < this.chainLength) {
			this.nLeapFrog = nLeapFrog;
		}
	}
	
	/**METHOD: GET MAX N LEAP FROG
	 * @return maximum number of leap frog steps possible
	 */
	public int getMaxNLeapFrog() {
		return this.maxNLeapFrog;
	}
	
}
