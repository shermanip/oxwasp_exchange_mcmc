package uk.ac.warwick.sip.mcmc;

import org.apache.commons.math3.random.MersenneTwister;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.simple.SimpleMatrix;

/**CLASS: RANDOM WALK METROPOLIS HASTINGS
 * Runs the metropolis hastings algorithm to sample a provided target distribution
 * The target distribution is provided via the constructor.
 * The chain length and a MersenneTwister is to be provided via the constructor.
 * Run the Metropolis Hastings algorithm using the step method, this will be one iteration
 * The samples are stored in the member variable chainArray as a design matrix format
 */
public class RandomWalkMetropolisHastings {
  
  protected TargetDistribution target; //target distribution for metropolis hastings
  //matrix containing the value of the chain at each step
  //matrix is of size chainLength X nDim, EJML is row major
  //initially it will contain random Gaussian
  //the intial value is at the origin, this can be set using the method setInitialStep
  protected SimpleMatrix chainArray;
  protected SimpleMatrix chainMean; //the mean of the chain at the current step (column vector)
  protected SimpleMatrix chainCovariance; //covariance of the chain at the current step
  
  //array of acceptance rate at each step
  protected double [] acceptanceArray;
  
  protected int nStep = 0; //number of steps taken so far
  protected int chainLength; //the total length of the chain
  protected int nAccept = 0; //the number of acceptance steps taken
  
  protected MersenneTwister rng; //random number generator
  
  
  /**CONSTRUCTOR
   * Metropolis Hastings algorithm which targets a provided distribution using Gaussian random walk
   * @param target Object which has a method to call the pdf
   * @param chainLength Length of the chain to be obtained
   * @param rng Random number generator all the random numbers
   */
  public RandomWalkMetropolisHastings(TargetDistribution target, int chainLength,
      MersenneTwister rng){
    //assign member variables
    this.target = target;
    this.chainLength = chainLength;
    this.chainArray = new SimpleMatrix(this.chainLength, this.getNDim());
    this.chainMean = new SimpleMatrix(this.getNDim(), 1);
    this.chainCovariance = new SimpleMatrix(this.getNDim(), this.getNDim());
    this.acceptanceArray = new double [this.chainLength-1];
    this.rng = rng;
    
  }
  
  /**CONSTRUCTOR
   * Constructor for extending the length of the chain and resume running it
   * Does a shallow copy of the provided chain and extending the member variable chainArray
   * @param chain Chain to be extended
   * @param nMoreSteps Number of steps to be extended
   */
  public RandomWalkMetropolisHastings(RandomWalkMetropolisHastings chain, int nMoreSteps){
    //do a shallow copy of chain
    this.target = chain.target;
    this.chainLength = chain.chainLength + nMoreSteps;
    this.chainArray = new SimpleMatrix(this.chainLength + nMoreSteps, this.getNDim());
    this.chainMean = chain.chainMean;
    this.chainCovariance = chain.chainCovariance;
    this.acceptanceArray = chain.acceptanceArray;
    this.nStep = chain.nStep;
    this.nAccept = chain.nAccept;
    this.rng = chain.rng;
    
    //deep copy the content of chainArray the old chain to the new chain
    //these contain the values of the MCMC
    for (int i=0; i<chain.chainArray.getNumElements(); i++) {
      this.chainArray.set(i, chain.chainArray.get(i));
    }
    
  }
  
  /**METHOD: STEP
   * This chains takes a Metropolis-Hastings step and updates it member variables-
   * @param proposalCovarianceChol lower cholesky decomposition of the proposal_covariance
   */
  public void step(SimpleMatrix proposalCovarianceChol){
    this.metropolisHastingsStep(proposalCovarianceChol);
    this.updateStatistics();
  }
  
  
  /**METHOD: STEP
   * This chain does a Metropolis-Hastings step
   * Instances created using RandomWalkMetropolisHastings will throw an exception because a
   * proposal covariance needs to be provided at every step
   * Subclasses should override this if a proposal covariance isn't required at every step
   */
  public void step() {
    throw new RuntimeException("RandomWalkMetropolisHastings requires a proposal covariance in"
        + " the method step()");
  }
  
  
  /**METHOD: RUN
   * Throws exception
   * Subclasses should override this if required
   * Overridden methods will use this to run the entire chain
   */
  public void run() {
    throw new RuntimeException("RandomWalkMetropolisHastings cannot use the method run()"
        + " because a proposal covariance is required for every step");
  }
  
  
  /**METHOD: METROPOLIS HASTINGS STEP
   * Does a Metropolis-Hastings step
   * Saves the new value to the chain array given a proposal covariance
   * It does not increment nStep when this method is called
   * The method updateStatistics will increment nStep
   * @param proposalCovarianceChol lower cholesky decomposition of the proposal_covariance
   */
  protected void metropolisHastingsStep(SimpleMatrix proposalCovarianceChol){
    
    //instantiate column vector for the current value of the chain
    SimpleMatrix x = this.chainArray.extractVector(true, this.nStep);
    CommonOps_DDRM.transpose(x.getDDRM());
    
    //instantiate vector of N(0,1) using rng
    SimpleMatrix z = new SimpleMatrix(this.getNDim(), 1);
    for (int i=0; i<this.getNDim(); i++) {
      z.set(i, this.rng.nextGaussian());
    }
    
    //transform z using proposalCovarianceChol and x, assign it to y, y is a proposal
    SimpleMatrix y = proposalCovarianceChol.mult(z);
    CommonOps_DDRM.addEquals(y.getDDRM(), x.getDDRM());
    
    //declare variable for the acceptance probability, work it out using the ratio of target pdf
    //if it larger than one, then an acceptance step will always be taken
    double acceptProb = (this.target.getPdf(y)) / (this.target.getPdf(x));
    this.acceptStep(acceptProb, x, y);
  }
  
  /**METHOD: ACCEPT STEP
   * The chainArray is updated at this.nStep+1.
   * With probability acceptProb, the chain takes the value proposal, otherwise current
   * The random uniform numbers in random01Array are used
   * @param acceptProb Acceptance probability
   * @param current Column vector containing the value of the chain now
   * @param proposal Column vector of the proposal step
   */
  protected void acceptStep(double acceptProb, SimpleMatrix current, SimpleMatrix proposal) {
    //get a random number between 0 and 1 from random01Array
    //with acceptProb chance, accept the sample
    SimpleMatrix acceptedStep = current;
    if (this.rng.nextDouble() < acceptProb){
      acceptedStep = proposal;
      //increment the number of acceptance steps
      this.nAccept++;
    }
    //save x to the chain array
    for (int iDim = 0; iDim<this.getNDim(); iDim++) {
      this.chainArray.set(this.nStep+1, iDim, acceptedStep.get(iDim));
    }
  }
  
  /**METHOD: UPDATE STATISTICS
   * Update the acceptance array, nStep, chainMean and chainCovariance
   */
  protected void updateStatistics(){
    
    //work out the acceptance rate at this stage
    //acceptance rate = (number of acceptance steps) / (number of steps)
    //the acceptance rate keeps track of the acceptance rate from and including the 1st step
    //(not from the initial value)
    //note that this.nStep has been incremented when metropolisHastingsStep was called
    this.acceptanceArray[this.nStep] = ((double)(this.nAccept)) / ((double)(this.nStep+1));
    
    //increment the number of steps taken
    this.nStep++;
    //n is the chain length, so it is the number of steps + 1 (from the initial value)
    double n = (double) (this.nStep+1);
    
    //instantiate vector of the current value of the chain
    SimpleMatrix x = this.chainArray.extractVector(true, this.nStep);
    //transpose it
    CommonOps_DDRM.transpose(x.getDDRM());
    
    //update the mean using the previous mean
    CommonOps_DDRM.scale(n-1, this.chainMean.getDDRM());
    CommonOps_DDRM.addEquals(this.chainMean.getDDRM(), x.getDDRM());
    CommonOps_DDRM.divide(this.chainMean.getDDRM(), n);
    
    
    //if this is the first step
    if (this.nStep == 1){
      //get the initial value of the chain and transpose it to be a column vector
      SimpleMatrix x1 = this.chainArray.extractVector(true, 0);
      CommonOps_DDRM.transpose(x1.getDDRM());
      //calculate the difference between the initial and the mean
      SimpleMatrix r1 = x1.minus(this.chainMean);
      //calculate the difference between the most recent value and the mean
      SimpleMatrix r2 = x.minus(this.chainMean);
      //calculate the covariance from scratch
      CommonOps_DDRM.multOuter(r1.getDDRM(), this.chainCovariance.getDDRM());
      DMatrixRMaj r2Outer = new DMatrixRMaj(this.getNDim(), this.getNDim());
      CommonOps_DDRM.multOuter(r2.getDDRM(), r2Outer);
      CommonOps_DDRM.addEquals(this.chainCovariance.getDDRM(), r2Outer);
      
    }
    
    //else update the covariance recursively
    else{
      //calculate the difference between the most recent value and the mean
      SimpleMatrix r2 = x.minus(this.chainMean);
      //update the covariance using the previous covariance
      CommonOps_DDRM.scale(n-2, this.chainCovariance.getDDRM());
      DMatrixRMaj r2Outer = new DMatrixRMaj(this.getNDim(), this.getNDim());
      CommonOps_DDRM.multOuter(r2.getDDRM(), r2Outer);
      CommonOps_DDRM.scale(n/(n-1), r2Outer);
      CommonOps_DDRM.addEquals(this.chainCovariance.getDDRM(), r2Outer);
      CommonOps_DDRM.divide(this.chainCovariance.getDDRM(), n-1);
    }
    
  }
  
  
  /**METHOD: GET N DIM
   * @return The number of dimensions the target distribution has
   */
  public int getNDim() {
    return this.target.getNDim();
  }
  
  /**METHOD: GET CHAIN
   * @return double array of the chain, row major
   */
  public double [] getChain() {
    return this.chainArray.getDDRM().getData();
  }
  
  /**METHOD: GET END OF CHAIN
   * @return double array, vector of the last postion of the chain
   */
  public double [] getEndOfChain() {
    return this.chainArray.extractVector(true, this.nStep).getDDRM().getData();
  }
  
  /**METHOD: GET CHAIN MEAN
   * @return The chain mean at the current step
   */
  public double[] getChainMean() {
    return this.chainMean.getDDRM().getData();
  }
  
  /**METHOD: GET ACCEPTANCE RATE
   * @return The estimate acceptance rate at each step
   */
  public double[] getAcceptanceRate() {
    return this.acceptanceArray;
  }
  
  /**METHOD: SET INITIAL VALUE
   * Set the initial value of the chain, to be called before running the chain
   * @param initialValue double [] containing the values of the initial position
   */
  public void setInitialValue(double [] initialValue) {
    for (int i=0; i<initialValue.length; i++) {
      this.chainArray.set(i, initialValue[i]);
    }
  }
}
