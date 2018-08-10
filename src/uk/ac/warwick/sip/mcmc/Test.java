package uk.ac.warwick.sip.mcmc;

import org.apache.commons.math3.random.MersenneTwister;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.simple.SimpleMatrix;

public class Test {
  
  static public void main(String[] args) {
    testChain(10, 1000, 3, 1262924406, "Test 1.1");
    testChain(10, 1000, 100, 1262924406, "Test 1.2");
    testChain(10, 1000, 999, 1262924406, "Test 1.3");
    //testChain(100, 1000, 999, -178151448, "Test 1.4");
    testCholesky(20, 409534955, "Test 2");
    testAcceptStep(10, 100, 990390580, "Test 3");
    testCopyExtendConstructor(32, 100, 20, 355497382, "Test 4.1");
    testCopyExtendConstructor(32, 1000, 20, 355497382, "Test 4.2");
    testCopyExtendConstructor(32, 1000, 100, 355497382, "Test 4.3");
    testCopyExtendConstructor(64, 1000, 500, 355497382, "Test 4.4");
    testThin(32, 100, 5, 438709310, "Test 5.1");
    testThin(32, 100, 10, 438709310, "Test 5.2");
    testThin(32, 1000, 10, 438709310, "Test 5.3");
    testThin(32, 1000, 50, 438709310, "Test 5.4");
  }
  
  /**FUNCTION: GET CHAIN
   * Return one implementation of a MCMC
   * @param iChain Integer pointing to which mcmc to instantiate
   * @param nDim Number of dimensions
   * @param chainLength Length of the chain
   * @param rng MersenneTwister to give to the mcmc object
   * @return Instantiate chain
   */
  static public Mcmc getChain(int iChain, int nDim, int chainLength, MersenneTwister rng) {
    //simple Normal target with identity covariance
    SimpleMatrix targetCovariance = SimpleMatrix.identity(nDim);
    TargetDistribution target = new NormalDistribution(nDim, targetCovariance);
    
    //rwmh parameters
    SimpleMatrix proposalCovariance = targetCovariance.scale( Math.pow(2.38,2) / ((double)nDim) );
    
    //hmc parameters
    SimpleMatrix massMatrix = SimpleMatrix.identity(nDim);
    int nLeapFrog = 20;
    double sizeLeapFrog = 0.5;
    int nAdaptive = 100;
      
    //instantiate the chain
    Mcmc chain = null;
    switch (iChain) {
      case 0:
        chain = new RandomWalkMetropolisHastings(target, chainLength, proposalCovariance, rng);
        break;
      case 1:
        chain = new AdaptiveRwmh(target, chainLength, proposalCovariance, rng);
        break;
      case 2:
        chain = new MixtureAdaptiveRwmh(target, chainLength, proposalCovariance, rng);
        break;
      case 3:
        chain = new HamiltonianMonteCarlo(target, chainLength, massMatrix, sizeLeapFrog,
            nLeapFrog, rng);
        break;
      case 4:
        chain = new NoUTurnSampler(target, chainLength, massMatrix, sizeLeapFrog, rng);
        break;
      case 5:
        chain = new DualAveragingNuts(target, chainLength, massMatrix, nAdaptive, rng);
        break;
      default:
        break;
    }
    return chain;
  }
  
  /**FUNCTION: COPY CONSTRUCTOR
   * Calls the chain's copy and extend constructor
   * @param iChain Pointer to the class of the object
   * @param chain The chain to call the copy and extend constructor
   * @param nMoreSteps Number of steps to extend
   * @return The chain copied and extended
   */
  static Mcmc copyConstructor(int iChain, Mcmc chain, int nMoreSteps) {
    //call the copy and extend constructor
    switch (iChain) {
      case 0:
        chain = new RandomWalkMetropolisHastings((RandomWalkMetropolisHastings)chain, nMoreSteps);
        break;
      case 1:
        chain = new AdaptiveRwmh((AdaptiveRwmh)chain, nMoreSteps);
        break;
      case 2:
        chain = new MixtureAdaptiveRwmh((MixtureAdaptiveRwmh)chain, nMoreSteps);
        break;
      case 3:
        chain = new HamiltonianMonteCarlo((HamiltonianMonteCarlo)chain, nMoreSteps);
        break;
      case 4:
        chain = new NoUTurnSampler((NoUTurnSampler)chain, nMoreSteps);
        break;
      case 5:
        chain = new DualAveragingNuts((DualAveragingNuts)chain, nMoreSteps);
        break;
      default:
        break;
    }
    return chain;
  }
  
  /**FUNCTION: TEST CHAIN
   * Test the member variables chainMean and chainCovariance, these are recursive estimators.
   * The recursive estimators are comapred with the non-recursive estimators.
   * The non-recursive estimators uses the data after the chain finished running.
   * The stanard Normal distribution is sampled using the optimial proposal.
   * Test if nStep increments at every step
   * @param nDim Number of dimensions the target has
   * @param chainLength The length of the chain
   * @param nStep Number of steps to take in the chain,
   *     the test will be done when the steps are taken
   * @param seed Random seed for the MersenneTwister
   * @param name Name of the test, this will be printed
   */
  static void testChain(int nDim, int chainLength, int nStep,
      int seed, String name) {
    
    System.out.println("==========");
    System.out.println(name);
    
    //for each mcmc class
    for (int iMcmc=0; iMcmc<6; iMcmc++) {
      
      boolean isNStep = true; //tests if nStep increments
      boolean isNSample = true; //tests if nSample increments
      boolean isAllocation = true; //tests if the sample has been allocated to chainArray correctly
      
      //random number generator
      MersenneTwister rng = new MersenneTwister(seed);
      
      Mcmc chain = getChain(iMcmc, nDim, chainLength, rng);
      
      //set random initial value
      double [] initial = getRandomVector(nDim, rng).getDDRM().getData();
      chain.setInitialValue(initial);
      
      //instantiate column vector for the current value of the chain
      SimpleMatrix x = chain.chainArray.extractVector(true, 0);
      CommonOps_DDRM.transpose(x.getDDRM());
      
      //run the chain for nStep
      for (int i=0; i<nStep; i++) {
        
        //copy x to for allocation test
        SimpleMatrix xBefore = new SimpleMatrix(x);
        
        //do mcmc step
        chain.step(x);
        //save x to the chain array
        chain.setCurrentStep(x);
        
        //check if chain.nStep has been incremented
        if (chain.nStep != (i+1)) {
          isNStep = false;
        }
        //check if chain nSample increments
        if (chain.nSample != (i+2)) {
          isNSample = false;
        }
        //check if x has been copied to chainArray
        //also check if the next sample in chainArray are all zeros
        //also check if the sample before is left unchanged
        for (int iDim=0; iDim<chain.getNDim(); iDim++) {
          if (chain.chainArray.get(i+1, iDim)!=x.get(iDim)) {
            isAllocation = false;
          }
          if (i!=(nStep-1)) {
            if (chain.chainArray.get(i+2, iDim)!=0) {
              isAllocation = false;
            }
          }
          if (i!=0) {
            if (chain.chainArray.get(i, iDim)!=xBefore.get(iDim)) {
              isAllocation = false;
            }
          }
          
        }
        
      }
      
      //declare array for storing squared error
      //entry one for the mean
      //entry two for the covariance
      double [] squaredErrorArray = new double [2];
      
      //get the mcmc samples after nStep
      SimpleMatrix designMatrix = chain.chainArray.extractMatrix(0, nStep+1, 0, nDim);
      
      //work out the sample mean and get and save the squared error
      SimpleMatrix sampleMean = getSampleMean(designMatrix);
      SimpleMatrix squaredError = (chain.chainMean.minus(sampleMean)).elementPower(2);
      squaredErrorArray[0] = squaredError.elementSum();
      
      //work out the sample covariance, get and save the squared error
      SimpleMatrix sampleCovariance = getSampleCovariance(designMatrix, sampleMean);
      squaredError = (chain.chainCovariance.minus(sampleCovariance)).elementPower(2);
      squaredErrorArray[1] = squaredError.elementSum();
      
      //print the squared error and return it
      System.out.println(chain.getClass().getName());
      System.out.println("Squared error in mean = "+squaredErrorArray[0]);
      System.out.println("Squared error in covariance = "+squaredErrorArray[1]);
      
      System.out.println("pass nStep test = "+isNStep);
      System.out.println("pass nSample test = "+isNSample);
      System.out.println("pass allocation test = "+isAllocation);
    }
  }
  
  /**FUNCTION: GET SAMPLE MEAN
   * Returns the unbiased estimator of the mean
   * @param designMatrix Collection of data, rows represent each data
   *     columns represent each dimension
   * @return Unbiased estimator of the sample mean (column vector)
   */
  static public SimpleMatrix getSampleMean(SimpleMatrix designMatrix) {
    int n = designMatrix.numRows(); //number of datapoints
    int p = designMatrix.numCols(); //number of dimensions
    //instantiate column vector for the sample mean
    SimpleMatrix sampleMean = new SimpleMatrix(p, 1);
    //for each dimension, work out the sample mean
    //combine all the dimensions together and return it
    for (int i=0; i<p; i++) {
      sampleMean.set(i, designMatrix.extractVector(false, i).elementSum());
    }
    return sampleMean.divide((double)n);
  }
  
  /**FUNCTION: GET SAMPLE COVARIANCE
   * Returns the unbiased estimator of the covariance
   * @param designMatrix Collection of data, rows represent each data,
   *     columns represent each dimension
   * @param sampleMean The mean estimator in a column vector format
   * @return Unbiased estimator of the covariance
   */
  static public SimpleMatrix getSampleCovariance(SimpleMatrix designMatrix,
      SimpleMatrix sampleMean) {
    int n = designMatrix.numRows(); //number of datapoints
    int p = designMatrix.numCols(); //number of dimensions
    //instantiate a matrix for the sample covariance
    SimpleMatrix sampleCovariance = new SimpleMatrix(p,p);
    //for each data point, update the sample covariance
    //this is the sum of squared difference from the mean
    for (int i=0; i<n; i++) {
      SimpleMatrix r = (designMatrix.extractVector(true, i).transpose()).minus(sampleMean);
      sampleCovariance = sampleCovariance.plus(r.mult(r.transpose()));
    }
    //normalise the sum of squared difference using n-1
    //this is the unbiased estimator
    return sampleCovariance.divide((double)(n - 1));
  }
  
  /**FUNCTION: TEST CHOLESKY
   * Check if Global.cholesky() does not modify the parameter
   * Check if Global.cholesky() returns null
   * Check if 
   * @param nDim Number of dimension
   * @param seed Seed for Mersenne Twister
   * @param name Name of the test
   * @return true if pass the test
   */
  static void testCholesky(int nDim, int seed, String name) {
    System.out.println("==========");
    System.out.println(name);
    int nTest = 10; // number of tests
    MersenneTwister rng = new MersenneTwister(seed);
    boolean isNoNull = true;
    boolean isNull = true;
    boolean isNotModify = true;
    //for n tests
    for (int iTest=0; iTest<nTest; iTest++) {
      //get a random covariance and clone it
      SimpleMatrix cov = Global.getRandomCovariance(nDim, rng);
      SimpleMatrix covClone = new SimpleMatrix(cov);
      SimpleMatrix chol = Global.cholesky(cov); //do cholesky decomposition
      //check if chol is null
      if (chol==null) {
        isNoNull = false;
      }
      //check if the difference of cov before and after chol is 0
      double sumDiffSquared = cov.minus(covClone).elementPower(2).elementSum();
      if (sumDiffSquared!=0) {
        isNotModify = false;
      };
      //test is L*LT is similar
      SimpleMatrix cholCholT = chol.mult(chol.transpose());
      sumDiffSquared = cholCholT.minus(cov).elementPower(2).elementSum();
      System.out.println("Squared error in Cholesky decomposition = "+sumDiffSquared);
      
      //check if cholesky of chol of N(0,1) outputs null
      SimpleMatrix randGaussian = new SimpleMatrix(nDim, nDim);
      for (int i=0; i<randGaussian.getNumElements(); i++) {
        randGaussian.set(i, rng.nextGaussian());
      }
      chol = Global.cholesky(randGaussian); //do cholesky decomposition
      if (chol!=null) {
        isNull = false;
      }
    }
    System.out.println("pass no null test = "+isNoNull);
    System.out.println("pass null test = "+isNull);
    System.out.println("pass modify test = "+isNotModify);
    
  }
  
  /**FUNCTION: GET RANDOM VECTOR
   * @param nDim
   * @param rng
   * @return Column vector, random Gaussian
   */
  static SimpleMatrix getRandomVector(int nDim, MersenneTwister rng) {
    SimpleMatrix x = new SimpleMatrix(nDim, 1);
    for (int iDim=0; iDim<nDim; iDim++) {
      x.set(iDim, rng.nextGaussian());
    }
    return x;
  }
  
  /**FUNCTION: TEST ACCEPT STEP
   * Checks if the method acceptStep modifies its parameters accordingly
   * Checks if nAccept is incremented correctly
   * @param nDim
   * @param chainLength
   * @param seed
   * @param name
   */
  static void testAcceptStep(int nDim, int chainLength, int seed, String name) {
    System.out.println("==========");
    System.out.println(name);
    MersenneTwister rng = new MersenneTwister(seed);
    int nTest = 100; //number of times to repeat the test
    boolean isModifyTest = true;
    boolean isNAcceptTest = true;
    
    //for the rwmh family of mcmc
    for (int iChain=0; iChain<3; iChain++) {
      
      Mcmc chain = null;
      for (int i=0; i<nTest; i++) {
        
        chain = getChain(iChain, nDim, chainLength, rng);
        
        //check with accept probability 0
        SimpleMatrix position = getRandomVector(nDim, rng);
        SimpleMatrix proposal = getRandomVector(nDim, rng);
        SimpleMatrix positionBefore = new SimpleMatrix(position);
        SimpleMatrix proposalBefore = new SimpleMatrix(proposal);
        chain.acceptStep(0.0, position, proposal);
        //check if both parameters are not modified
        if (positionBefore.minus(position).elementPower(2).elementSum() != 0) {
          isModifyTest = false;
        }
        if (proposalBefore.minus(proposal).elementPower(2).elementSum() != 0) {
          isModifyTest = false;
        }
        //test if nAccept = 0
        if (chain.nAccept != 0) {
          isNAcceptTest = false;
        }
        
        //check with accept probability 1.0
        position = getRandomVector(nDim, rng);
        proposal = getRandomVector(nDim, rng);
        positionBefore = new SimpleMatrix(position);
        proposalBefore = new SimpleMatrix(proposal);
        chain.acceptStep(1.0, position, proposal);
        //check if position is modified and proposal is not modified
        if (proposal.minus(position).elementPower(2).elementSum() != 0) {
          isModifyTest = false;
        }
        if (proposalBefore.minus(proposal).elementPower(2).elementSum() != 0) {
          isModifyTest = false;
        }
        //test if nAccept = 0
        if (chain.nAccept != 1) {
          isNAcceptTest = false;
        }
        
        //check with accept probability 1.0
        position = getRandomVector(nDim, rng);
        proposal = getRandomVector(nDim, rng);
        positionBefore = new SimpleMatrix(position);
        proposalBefore = new SimpleMatrix(proposal);
        chain.acceptStep(0.5, position, proposal);
        //check if the proposal is not modified
        if (proposalBefore.minus(proposal).elementPower(2).elementSum() != 0) {
          isModifyTest = false;
        }
        //check if the position has been modified with an accept step
        if (chain.nAccept == 2) {
          if (proposal.minus(position).elementPower(2).elementSum() != 0) {
            isModifyTest = false;
          }
        //check if the position has not been modified with a reject step
        } else {
          if (positionBefore.minus(position).elementPower(2).elementSum() != 0) {
            isModifyTest = false;
          }
        }
      }
      
      System.out.println(chain.getClass().getName());
      System.out.println("pass modify test = "+isModifyTest);
      System.out.println("pass nAccept test = "+isNAcceptTest);
      
    }
    
  }
  
  /**FUNCTION: TEST COPY AND EXTEND CONSTRUCTOR
   * Test the copy and extend constructor, runs 2 chains of chainLength, 1 stops at subChainLength
   * and then calls the copy and extend constructor to run the remaining steps
   * Checks the 2 chains has the same nStep
   * Checks the 2 chains has the same nSample
   * Checks the 2 chains has the exact same chainArray
   * @param nDim Number of dimensions
   * @param chainLength Length of chain
   * @param subChainLength Length of the first run
   * @param seed for rng
   * @param name
   */
  static void testCopyExtendConstructor(int nDim, int chainLength, int subChainLength,
      int seed, String name) {
    
    System.out.println("==========");
    System.out.println(name);
    
    //for the rwmh family of mcmc
    for (int iChain=0; iChain<6; iChain++) {
      
      //boolean for the tests
      boolean isNStep = true;
      boolean isNSample = true;
      boolean isSame = true;
      
      //get the chain and run it all the way
      MersenneTwister rng = new MersenneTwister(seed);
      Mcmc chain = getChain(iChain, nDim, chainLength, rng);
      chain.run();
      
      //get the chain, run it for subChainLength, copy and extend, then run the remaining steps
      rng = new MersenneTwister(seed);
      Mcmc chainUseCopyConstructor = getChain(iChain, nDim, subChainLength, rng);
      chainUseCopyConstructor.run();
      chainUseCopyConstructor = copyConstructor(iChain, chainUseCopyConstructor,
          chainLength - subChainLength);
      chainUseCopyConstructor.run();
      
      //check if nStep are the same
      if (chain.nStep != chainUseCopyConstructor.nStep) {
        isNStep = false;
      }
      //check if nSample are the same
      if (chain.nSample != chainUseCopyConstructor.nSample) {
        isNSample = false;
      }
      //check if chainArray are the same
      if (chain.chainArray.minus(chainUseCopyConstructor.chainArray).elementPower(2).elementSum()
          != 0) {
        isSame = false;
      };
      
      System.out.println(chain.getClass().getName());
      System.out.println("pass nStep test = "+isNStep);
      System.out.println("pass nSample test = "+isNSample);
      System.out.println("pass isSame test = "+isSame);
      
    }
  }
  
  /**FUNCTION: TEST THIN
   * Test how the chain behaves with thinning
   * Checks if the number of rows in chainArray is correct
   * Checks if nStep is correct, this is the number of MCMC steps
   * Checks if nSample is correct
   * @param nDim
   * @param chainLength
   * @param nThin
   * @param seed
   * @param name
   */
  static void testThin(int nDim, int chainLength, int nThin, int seed, String name) {
    System.out.println("==========");
    System.out.println(name);
    
    //for the rwmh family of mcmc
    for (int iChain=0; iChain<6; iChain++) {
      //boolean for the tests
      boolean isArrayHeight = true;
      boolean isNStep = true;
      boolean isNSample = true;
      
      //get the chain and run it all the way
      MersenneTwister rng = new MersenneTwister(seed);
      Mcmc chain = getChain(iChain, nDim, chainLength, rng);
      chain.setNThin(nThin);
      chain.run();
      
      if (chain.chainArray.numRows() != chainLength) {
        isArrayHeight = false;
      }
      if (chain.nStep != ( (chainLength-1)*nThin ) ) {
        isNStep = false;
      }
      if (chain.nSample != (chainLength)) {
        isNSample = false;
      }
      System.out.println(chain.getClass().getName());
      System.out.println("pass chainArray height test = "+isArrayHeight);
      System.out.println("pass nStep test = "+isNStep);
      System.out.println("pass nSample test = "+isNSample);
    }
  }
  
}
