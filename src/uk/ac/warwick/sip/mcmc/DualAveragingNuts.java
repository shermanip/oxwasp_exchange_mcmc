package uk.ac.warwick.sip.mcmc;

import org.apache.commons.math3.random.MersenneTwister;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.simple.SimpleMatrix;

/**CLASS: DUAL AVERAGING NO U TURN SAMPLER
 * An implementation of the dual averaging no u turn sampler
 * See reference Hoffman, M.D., and Gelman, A., 2014.
 * The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo,
 * Journal of Machine Learning Research, 15(1), pp.1593-1623
 * An adaptive HMC which adapts the number of leap frog steps so that no u turns are made
 * It also adapts the size of the leap frog steps, requires the number of adaptive steps
 */
public class DualAveragingNuts extends NoUTurnSampler {
  
  protected double targetAcceptProb = 0.65; // \delta in the reference
  protected int nAdaptive; // M_adapt in the reference
  protected boolean isAdaptive = true;// boolean, true to do adaptive step
  protected double shrinkCentre; // \mu in the reference
  protected double logSizeLeapFrog; // log of the member variable this.sizeLeapFrog
  protected double logDualAveSizeLeapFrog = 1.0; // \log\bar{\epsilon}_0 in the reference
  protected double objective = 0.0; // \bar{H}_0 in the reference
  protected double shrinkage = 0.05; // \gamma in the reference
  protected double timeBias = 10.0; // t_0 in the reference
  protected double decayParameter = 0.75; // \kappa in the reference
  protected double currentHamiltonian; //the hamiltonian of the position-momentum pair currently
  
  /**CONSTRUCTOR
   * See super class NoUTurnSampler
   * @param target Object which has a method to call the pdf
   * @param chainLength Length of the chain to be obtained
   * @param massVector column vector, containing diagonal element of the mass matrix
   * @parm nAdaptive number of adaptive steps to tune the step size
   * @param rng Random number generator to generate all the random numbers
   */
  public DualAveragingNuts(TargetDistribution target, int chainLength, SimpleMatrix massVector,
      int nAdaptive, MersenneTwister rng) {
    //call superclass constructor
    //set the parameter sizeLeapFrog = 1.0
    super(target, chainLength, massVector, 1.0, rng);
    //assign sizeLeapFrog
    this.setInitialStepSize();
    //assign the rest of the member variables
    this.nAdaptive = nAdaptive;
    this.logSizeLeapFrog = Math.log(this.sizeLeapFrog);
    this.shrinkCentre = Math.log(10.0) + this.logSizeLeapFrog;
  }
  
  /**CONSTRUCTOR
   * Constructor for extending the length of the chain and resume running it
   * Does a shallow copy of the provided chain and extending the member variable chainArray
   * @param chain Chain to be extended
   * @param nMoreSteps Number of steps to be extended
   */
  public DualAveragingNuts(DualAveragingNuts chain, int nMoreSteps) {
    //call superconstructor to do a shallow copy and extend the chain
    //also shallow copy the chain's member variables
    super(chain, nMoreSteps);
    this.targetAcceptProb = chain.targetAcceptProb;
    this.nAdaptive = chain.nAdaptive;
    this.isAdaptive = chain.isAdaptive;
    this.shrinkCentre = chain.shrinkCentre;
    this.logSizeLeapFrog = chain.logSizeLeapFrog;
    this.logDualAveSizeLeapFrog = chain.logDualAveSizeLeapFrog;
    this.objective = chain.objective;
    this.shrinkage = chain.shrinkage;
    this.timeBias = chain.timeBias;
    this.decayParameter = chain.decayParameter;
    this.currentHamiltonian = chain.currentHamiltonian;
  }
  
  /**OVERRIDE: SET INITIAL VALUE
   * Set the initial value of the chain, then set the initial sizeLeapFrog
   * @param initialValue double [] containing the values of the initial position
   */
  @Override
  public void setInitialValue(double [] initialValue) {
    super.setInitialValue(initialValue);
    this.setInitialStepSize();
  }
  
  /**METHOD: SET INITIAL STEP SIZE
   * Called FindReasonableEpsilon in the reference
   * Set the member variable this.sizeLeapFrog during construction
   * @param position Column vector of the current step of the MCMC, to be modified
   */
  protected void setInitialStepSize() {
    SimpleMatrix position = this.chainArray.extractVector(true, this.nSample - 1);
    CommonOps_DDRM.transpose(position.getDDRM());
    SimpleMatrix momentum = this.getMomentum();
    double acceptProb;
    double canonicalCurrent;
    double canonicalProposal;
    double a = 0.0;
    
    //keep halving or doubling sizeLeapFrog till Langevin proposal crosses 0.5
    do {
      //half or double sizeLeapFrog
      this.sizeLeapFrog *= Math.pow(2, a);
      
      //get the position-momentum pair and take a leap frog step
      SimpleMatrix positionProposal = new SimpleMatrix(position);
      SimpleMatrix momentumProposal = new SimpleMatrix(momentum);
      this.leapFrog(positionProposal, momentumProposal);
      
      //get the canonical distributions given the hamiltonians
      canonicalCurrent = Math.exp(-this.getHamiltonian(position, momentum));
      canonicalProposal = Math.exp(-this.getHamiltonian(positionProposal
          ,momentumProposal));
      //get acceptance probability
      acceptProb = canonicalProposal/canonicalCurrent;
      
      //if a hasn't been set it, set it here
      if (a == 0.0) {
        if (acceptProb > 0.5) {
          a = 1.0;
        } else {
          a = -1.0;
        }
      }
      //while till Langevin proposal crosses 0.5
    } while (Math.pow(acceptProb, a) > Math.pow(2, -a));
    
  }
  
  /**OVERRIDE: SAMPLE SLICE VARIABLE
   * Set the member variable sliceVariable given the current hamiltonian
   * Also sets the member variable currentHamiltonian
   * @parm hamiltonian The hamiltonian at the current position-momentum state
   */
  @Override
  protected void sampleSliceVariable(double hamiltonian) {
    //call the superclass method which sets the member variable sliceVariable
    super.sampleSliceVariable(hamiltonian);
    //save the current hamiltonian
    this.currentHamiltonian = hamiltonian;
  }
  
  /**OVERRIDE: ADAPTIVE STEP
   * Calls the method adaptiveStep (Tree tree) which is implemented in this class
   * The superclass will pass a Tree object but with a NoUTurnSampler.Tree reference
   * Cast the NoUTurnSampler.Tree reference to a Tree reference
   * @parm tree The tree after a u turn has been made
   */
  @Override
  protected void adaptiveStep (NoUTurnSampler.Tree superTree) {
    Tree tree = (Tree) superTree;
    this.adaptiveStep(tree);
  }
  
  /**OVERLOAD: ADAPTIVE STEP
   * In a step of HMC, which consist of multiple leap frog steps, update the member variables
   * which includes sizeLeapFrog
   * @parm tree The tree after a u turn has been made
   */
  protected void adaptiveStep (Tree tree) {
    //if this is the adaptive step, adjust the member variables accordingly using the dual
    //averaging procedure
    if (isAdaptive) {
      //if this is the adaptive step
      if (this.nStep <= this.nAdaptive) {
        //set the sub tree, 1/(m+t_0), m^{-\kappa} and \alpha/(n\alpha)
        Tree subTree = tree.subTree;
        double nStepBiasInverse = 1/((double)this.nStep + this.timeBias);
        double decay = Math.pow((double)this.nStep,-this.decayParameter);
        double currentPropAccept = subTree.sumProbAccept/((double)subTree.nAcceptReject);
        
        //update the objective
        this.objective *= 1 - nStepBiasInverse;
        this.objective += nStepBiasInverse * ( this.targetAcceptProb - currentPropAccept);
        
        //adjust logSizeLeapFrog and sizeLeapFrog
        this.logSizeLeapFrog = this.shrinkCentre
            - Math.sqrt((double)this.nStep)*this.objective/this.shrinkage;
        this.sizeLeapFrog = Math.exp(this.logSizeLeapFrog);
        
        //take a weighted average of logSizeLeapFrog and logDualAveSizeLeapFrog
        this.logDualAveSizeLeapFrog *= (1.0 - decay);
        this.logDualAveSizeLeapFrog += decay * this.logSizeLeapFrog;
        
        //for the end of the adaptive stage, set the final value for sizeLeapFrog
      } else {
        this.sizeLeapFrog = Math.exp(this.logDualAveSizeLeapFrog);
        this.isAdaptive = false;
      }
    }
  }
  
  /**OVERRIDE: NEW TREE
   * See constructor in subclass Tree
   */
  @Override
  protected Tree newTree (SimpleMatrix position, SimpleMatrix momentum) {
    return this.new Tree(position, momentum);
  }
  
  /**OVERRIDE: NEW TREE
   * See constructor in subclass Tree
   */
  @Override
  protected Tree newTree (SimpleMatrix position, SimpleMatrix momentum,
      boolean isNegative, int treeHeight) {
    return this.new Tree(position, momentum, isNegative, treeHeight);
  }
  
  /**INNER CLASS: TREE
   * See superclass NoUTurnSampler.Tree
   * Extended for the dual averaging algorithm
   * Sum the Metropolis-Hastings acceptance rate at every node
   */
  protected class Tree extends NoUTurnSampler.Tree {
    
    //temporary variable, resulting subTree after calling the method grow()
    //hides the superclass version
    protected Tree subTree;
    //sum of the metropolis hastings acceptance probability at every step
    protected double sumProbAccept;
    //number terms in sumProbAccept
    protected double nAcceptReject;
    
    /**CONSTRUCTOR
     * See superclass NoUTurnSampler.Tree
     * Instantiate a tree of height 0, no leap frog step taken and grows from the given
     * position and momentum pair 
     * @param position Column vector, position vector to grow from
     * @param momentum Column vector, momentum vector to grow from
     */
    public Tree(SimpleMatrix position, SimpleMatrix momentum) {
      super(position, momentum);
    }
    
    /**CONSTRUCTOR
     * See superclass NoUTurnSampler.Tree
     * Instantiate a tree of height treeHeight, each node correspond to a leap frog step
     * The tree grows from the given position and momentum pair 
     * Note: treeHeight = 0 will instantiate a tree of height 0 and a leap frog step is taken
     * from the given position-momentum pair
     * @param position Column vector, position vector to grow from
     * @param momentum Column vector, momentum vector to grow from
     * @param isNegative Direction to grow the tree
     * @param treeHeight Requested instantiated tree height
     */
    public Tree(SimpleMatrix position, SimpleMatrix momentum,
        boolean isNegative, int treeHeight) {
      super(position, momentum, isNegative, treeHeight);
    }
    
    /**OVERRIDE: SET USING HAMILTONIAN
     * See superclass NoUTurnSampler.Tree
     * Set the member variables this.sumPropAccept and nAcceptReject in the base case
     * construction on the tree
     * @param hamiltonian Hamiltonian of the proposal
     */
    @Override
    protected void setUsingHamiltonian(double hamiltonian) {
      //class superclass method, this set the member variables nSliceAccept and hasNoUTurn
      super.setUsingHamiltonian(hamiltonian);
      //set the Metropolis-Hastings acceptance probability
      this.sumProbAccept = Math.exp(-hamiltonian
          + DualAveragingNuts.this.currentHamiltonian);
      //take the minimum to make it a probability
      if (this.sumProbAccept > 1.0) {
        this.sumProbAccept = 1.0;
      }
      //set the member variable this.nAcceptRejct, this increases when combining trees
      //when using the method buildTree
      this.nAcceptReject = 1;
    }
    
    /**OVERRIDE: GROW
     * Calls the superclass method grow as usual but cast the hidden superclass
     * member variable super.subTree to a DualAveragingNuts.Tree reference
     * @param isNegative boolean, true of going back in time, otherwise forward
     */
    @Override
    protected void grow(boolean isNegative) {
      //superclass method modifies the hidden superclass member variable super.subTree
      super.grow(isNegative);
      //cast the superclass member variable super.subTree to a
      //DualAveragingNuts.Tree reference
      //save the casted reference to the member variable this.subTree
      this.subTree = (Tree) super.subTree;
    }
    
    /**OVERRIDE: BLOOM
     * See superclass NoUTurnSampler.Tree
     * To be called after calling the method grow()
     * Update the member variables nSliceAccept, hasNoUTurn, height
     * sumProbAccept and nAcceptReject
     */
    @Override
    protected void bloom() {
      //call superclass method which updates nSliceAccept, hasNoUTurn, height
      super.bloom();
      //update the member variables sumProbAccept and nAcceptReject
      this.sumProbAccept += this.subTree.sumProbAccept;
      this.nAcceptReject += this.subTree.nAcceptReject;
    }
  }
}
