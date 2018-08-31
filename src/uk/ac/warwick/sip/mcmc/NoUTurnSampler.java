/*
 *    Copyright 2018 Sherman Ip

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */

package uk.ac.warwick.sip.mcmc;

import org.apache.commons.math3.random.MersenneTwister;
import org.ejml.simple.SimpleMatrix;

/**CLASS: NO U TURN SAMPLER
 * Adaptive HMC which adapts the number of leap frog steps so that no u turns are made
 * Reference: Hoffman, M.D., and Gelman, A., (2014)
 *  The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo,
 *  Journal of Machine Learning Research, 15(1), pp.1593-1623
 */
public class NoUTurnSampler extends HamiltonianMonteCarlo {
  
  //after a leapfrog step, threshold of comparing the hamiltonian with the slice variable
  //used to determine if a u turn has been made
  protected double deltaMax;
  //temporary variable for storing the slice variable
  //a random double times the cannonical distribution
  protected double sliceVariable;
  
  /**CONSTRUCTOR
   * Adaptive HMC which adapts the number of leap frog steps so that no u turns are made
   * @param target Object which has a method to call the pdf
   * @param chainLength Length of the chain to be obtained
   * @param mass matrix, determines the variance of the momentum
   * @param sizeLeapFrog size of the leap frog step
   * @param rng Random number generator for all the random numbers
   * also used to further random number generation on the fly
   */
  public NoUTurnSampler(TargetDistribution target, int chainLength, SimpleMatrix massMatrix,
      double sizeLeapFrog, MersenneTwister rng) {
    //set the number of leap frog steps to be one
    //this is so that calling the method leapFrog() will only take on leap frog step
    super(target, chainLength, massMatrix, sizeLeapFrog, 1, rng);
    this.deltaMax = 1000;
  }
  
  /**CONSTRUCTOR
   * Constructor for extending the length of the chain and resume running it
   * Does a shallow copy of the provided chain and extending the member variable chainArray
   * @param chain Chain to be extended
   * @param nMoreSteps Number of steps to be extended
   */
  public NoUTurnSampler(NoUTurnSampler chain, int nMoreSteps) {
    //call superconstructor to do a shallow copy and extend the chain
    //also shallow copy the chain's member variables
    super(chain, nMoreSteps);
    this.deltaMax = chain.deltaMax;
  }
  
  
  /**OVERRIDE: STEP
   * Does a step using the No U Turn Sampler
   * The position vector is the current position of the chain.
   * Momentum is generated randomly using Gaussian.
   * Leap frogs are done until the particle does a U turn
   * @param position Column vector of the current step of the MCMC, to be modified
   */
  @Override
  public void step(SimpleMatrix position) {
    
    //get the position vector from the chain array, and random momentum
    SimpleMatrix momentum = this.getMomentum();
    
    //sample the slice variable
    this.sampleSliceVariable(this.getHamiltonian(position, momentum));
    
    //instantiate a tree with the variables
    //backward position and momentum = position and momentum
    //forward position and momentum = position and momentum
    //proposal position = position
    //nSliceAccept = 1 (referred to as n in the reference)
    //hasNoUTurn = true (referred to as s in the reference)
    Tree tree = this.newTree(position, momentum);
    tree.nSliceAccept++;
    
    //declare variable for flagging if an acceptance step has been taken
    boolean hasAccept = false;
    
    //while no u turn has been made
    while (tree.hasNoUTurn) {
      
      //sample Uniform({-1,1}), this controls the direction of the tree growth
      boolean isNegative = this.rng.nextBoolean();
      
      //grow the tree in the direction of isNegative
      //either the backward or forward position-momentum pairs are updated
      //the new part of the tree is the subtree and is a member variable
      tree.grow(isNegative);
      
      //the subtree contains a new proposal position
      //the subtree is the same height of the tree before the method grow() has been called
      Tree subTree = tree.subTree;
      
      //if the subtree hasn't made a u turn
      if (subTree.hasNoUTurn) {
        //accept the proposal position with a probability
        double probAccept = ((double) subTree.nSliceAccept)
            / ((double) tree.nSliceAccept);
        if (this.rng.nextDouble() < probAccept) {
          tree.positionProposal = subTree.positionProposal;
          hasAccept = true;
        }
      }
      
      //adjust member variables of the main tree, this includes tree.hasNoUTurn
      tree.bloom();
      
    }
    
    //if an acceptance step has been taken, increment the number of acceptance steps
    if (hasAccept) {
      this.nAccept++;
    }
    
    //copy the proposal position
    position.set(tree.positionProposal);
    
    //update the statistics of itself
    this.updateStatistics(position);
    
    //call the method adaptiveStep, in this class it does nothing
    //subclasses may override adaptiveStep method
    this.adaptiveStep(tree);
    
  }
  
  /**METHOD: SAMPLE SLICE VARIABLE
   * Set the member variable sliceVariable given the current hamiltonian
   * The slice varaible is a random number Uniform(0,Math.exp(-hamiltonian))
   * The uniform random number is obtained from the array this.random01Array
   * @parm hamiltonian The hamiltonian at the current position-momentum state
   */
  protected void sampleSliceVariable(double hamiltonian) {
    //sample the slice variable
    this.sliceVariable = this.rng.nextDouble();
    this.sliceVariable *= Math.exp(-hamiltonian);
  }
  
  /**METHOD: ADAPTIVE STEP
   * Does nothing, to be overridden if an adaptive step, which depends on the tree,
   * is to be implemented
   */
  protected void adaptiveStep(Tree tree) {
    //do nothing
  }
  
  /**METHOD: NEW TREE
   * See constructor in subclass Tree
   */
  protected Tree newTree (SimpleMatrix position, SimpleMatrix momentum) {
    return this.new Tree(position, momentum);
  }
  
  /**METHOD: NEW TREE
   * See constructor in subclass Tree
   */
  protected Tree newTree (SimpleMatrix position, SimpleMatrix momentum,
      boolean isNegative, int treeHeight) {
    return this.new Tree(position, momentum, isNegative, treeHeight);
  }
  
  /**INNER CLASS: TREE
   * Stores the position/momentum backward/forward column vectors
   * Stores the proposed position column vector
   * Stores the number of accepted steps
   * Stores the boolean hasNoUTurn
   * 
   * Two constructors are provided
   * 
   * Tree(SimpleMatrix position, SimpleMatrix momentum) only stores the variables. A tree of
   * height 0 is instantiated, no leap frog step is taken
   * 
   * Tree(SimpleMatrix position, SimpleMatrix momentum, boolean isNegative, int treeHeight)
   * instantiate a tree of height treeHeight. It grows from the given position-momentum pair
   * and each node (or height) correspond to a leap frog step. Setting treeHeight = 0 will
   * instantiate a tree of height 0, a leap frog step is taken from the given
   * position-momentum pair. This is different from the Tree(SimpleMatrix position,
   * SimpleMatrix momentum) constructor.
   * 
   * Source code makes use of inner classes, instances are owned by (and can access)
   * the parent NoUTurnSampler object
   */
  protected class Tree {
    
    protected SimpleMatrix positionBackward; //column vector, position backwards in time
    protected SimpleMatrix momentumBackward; //column vector, momentum backwards in time
    protected SimpleMatrix positionForward; //column vector, position forwards in time
    protected SimpleMatrix momentumForward; //column vector, momentum forwards in time
    protected SimpleMatrix positionProposal; //column vector, proposed position
    //number of times slice variable (u) < exp(-H) (n in the reference)
    protected int nSliceAccept = 0; 
    //determines if this sampler has not made a u turn
    protected boolean hasNoUTurn = true;
    //height of the tree
    protected int height = 0;
    //temporary variable, resulting subTree after calling the method grow()
    protected Tree subTree;
    
    /**CONSTRUCTOR
     * Instantiate a tree of height 0, no leap frog step taken and grows from the given
     * position and momentum pair 
     * @param position Column vector, position vector to grow from
     * @param momentum Column vector, momentum vector to grow from
     */
    public Tree(SimpleMatrix position, SimpleMatrix momentum) {
      this.plantSeed(position, momentum);
    }
    
    /**CONSTRUCTOR
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
      this.buildTree(position, momentum, isNegative, treeHeight);
    }
    
    /**METHOD: PLANT SEED
     * Only used by the constructor and the method build tree
     * This assign the member variables given the starting point of the sampler
     * @param position Column vector, position vector to grow from
     * @param momentum Column vector, momentum vector to grow from
     */
    private void plantSeed(SimpleMatrix position, SimpleMatrix momentum) {
      this.positionBackward = position;
      this.momentumBackward = momentum;
      this.positionForward = position;
      this.momentumForward = momentum;
      this.positionProposal = position;
    }
    
    /**METHOD: BUILD TREE
     * See reference (Hoffman 2014), function buildTree()
     * This method is only used by the constructor and itself recursively
     * Modifies itself to grow itself to a height of treeHeight
     * Each node (or height) correspond to a leap frog step
     * @param position Column vector, position vector to grow from
     * @param momentum Column vector, momentum vector to grow from
     * @param isNegative boolean, true if to run the HMC backwards in time
     * @param treeHeight the height of the tree requested
     */
    private void buildTree(SimpleMatrix position, SimpleMatrix momentum,
        boolean isNegative, int treeHeight) {
      //if the tree height is 0, this is the base case
      //take one leapfrog step in corresponding direction indiciated by isNegative
      if (treeHeight == 0) {
        //copy the position and momentum vector
        //change the sign if sizeLeapFrog if to go back in time
        //it shall be changed back to how it was after the leap frog step
        SimpleMatrix positionProposal = new SimpleMatrix(position);
        SimpleMatrix momentumProposal = new SimpleMatrix(momentum);
        if (isNegative) {
          NoUTurnSampler.this.sizeLeapFrog *= -1.0;
        }
        //method leapFrog modifies positionProposal and momentumProposal
        NoUTurnSampler.this.leapFrog(positionProposal, momentumProposal); 
        if (isNegative) {
          NoUTurnSampler.this.sizeLeapFrog *= -1.0;
        }
        
        //prepare variables of the new instantiated Tree object
        this.plantSeed(positionProposal, momentumProposal);
        
        //get the hamiltonian to set variables such as nSliceAccept and hasNoUTurn
        double hamiltonian = NoUTurnSampler.this.getHamiltonian(positionProposal,
            momentumProposal);
        this.setUsingHamiltonian(hamiltonian);
        
        
      } else { //else this is not a base case
        
        //recursion - implicitly build the left and right subtrees
        //the recursion will keep on stacking until the base case treeHeight = 0
        //then treeHeight will increment for each removal from the stack
        //it should be noted that this.treeHeight is set to 0 in the base case
        //this.treeHeight increments everytime this.grow and this.bloom is called
        this.buildTree(position, momentum, isNegative, treeHeight-1);
        
        //if no u turn has been made
        if (this.hasNoUTurn) {
          
          //grow the tree in the direction of build tree
          //this instantiate a tree, grew from the far backward/forward in time
          //the new instantiate tree is the subtree
          this.grow(isNegative);
          
          //the subtree contains a new proposal
          //accept the new proposal with a probability
          double probAccept = ((double)(this.subTree.nSliceAccept))
              / ((double)(this.nSliceAccept+this.subTree.nSliceAccept));
          if (NoUTurnSampler.this.rng.nextDouble() < probAccept) {
            this.positionProposal = this.subTree.positionProposal;
          }
          
          //update the member variables of the main tree
          this.bloom();
          
        }
      }
      
    }
    
    /**METHOD: SET USING HAMILTONIAN
     * Given the hamiltonian of the proposal, set the member variables nSliceAccept and
     * hasNoUTurn
     * This method is only used in the base case of the method buildTree
     * @param hamiltonian Hamiltonian of the proposal
     */
    protected void setUsingHamiltonian(double hamiltonian) {
      //the slice variable is referred to as u in the reference
      if (NoUTurnSampler.this.sliceVariable < Math.exp(-hamiltonian)) {
        this.nSliceAccept++;
      }
      
      //determine if a u turn has been made or not
      if (-hamiltonian <= (Math.log(NoUTurnSampler.this.sliceVariable)
          -NoUTurnSampler.this.deltaMax)) {
        this.hasNoUTurn = false;
      }
    }
    
    /**METHOD: GROW
     * Add the subtree to the tree, given the state of the system
     * This method MODIFIES itself
     * The method bloom() should be called after this to update the member variables
     * nSliceAccept, hasNoUTurn and height
     * The instantiated sub tree is saved in the member variable subTree
     * @param isNegative boolean, true of going back in time, otherwise forward
     */
    protected void grow(boolean isNegative) {
      //declare a pointer for the sub tree
      Tree subTree = null;
      //if going back in time
      if (isNegative) {
        //build a subtree of the same height
        //grow the subtree from the most backward state
        //replace all the backward member variables in the main tree
        subTree = NoUTurnSampler.this.newTree(this.positionBackward, this.momentumBackward,
            isNegative, this.height);
        this.momentumBackward = subTree.momentumBackward;
        this.positionBackward = subTree.positionBackward;
      } else {
        //else going forward in time
        //build a subtree of the same height
        //grow the subtree from the most forward state
        //replace all the forward member variables in the main tree
        subTree = NoUTurnSampler.this.newTree(this.positionForward, this.momentumForward,
            isNegative, this.height);
        this.momentumForward = subTree.momentumForward;
        this.positionForward = subTree.positionForward;
      }
      //save the sub tree
      this.subTree = subTree;
    }
    
    /**METHOD: BLOOM
     * To be called after calling the method grow()
     * Update the member variables nSliceAccept, hasNoUTurn, height
     */
    protected void bloom() {
      this.nSliceAccept += this.subTree.nSliceAccept;
      this.hasNoUTurn = this.subTree.hasNoUTurn && this.checkHasNoUTurn();
      this.height++;
    }
    
    /**METHOD: CHECK HAS NO U TURN
     * Check if a u turn has been made
     * @param tree Uses the variable stored in tree to check for u turns
     * @return boolean, true if no u turns has been made
     */
    protected boolean checkHasNoUTurn() {
      //declare booleans for no u turn indication using the forward and backward variables
      boolean hasNoBackwardUTurn = false;
      boolean hasNoForwardUTurn = false;
      
      //instantiate a column vector, difference in the position vectors
      SimpleMatrix positionDifference = this.positionForward.minus(this.positionBackward);
      
      //check for u turns using the forward and backward momentum
      if (positionDifference.dot(this.momentumBackward) >= 0) {
        hasNoBackwardUTurn = true;
      }
      if (positionDifference.dot(this.momentumForward) >= 0) {
        hasNoForwardUTurn = true;
      }
      
      //return the no u turn boolean
      return hasNoBackwardUTurn & hasNoForwardUTurn;
    }
  }
  
}
