package uk.ac.warwick.sip.mcmc;

import org.apache.commons.math3.random.MersenneTwister;
import org.ejml.simple.SimpleMatrix;

public class Test {
  
  static public void main(String[] args) {
    testRwmh();
  }
  
  static public void testRwmh() {
    testRecursiveEstimators(10, 1000, 3, 1262924406, "Test 1");
    testRecursiveEstimators(10, 1000, 100, 1262924406, "Test 2");
    testRecursiveEstimators(10, 1000, 999, 1262924406, "Test 3");
    testRecursiveEstimators(100, 1000, 999, -178151448, "Test 4");
  }
  
  /**TEST RECURSIVE ESTIMATORS
   * Test the member variables chainMean and chainCovariance, these are recursive estimators.
   * The recursive estimators are comapred with the non-recursive estimators.
   * The non-recursive estimators uses the data after the chain finished running.
   * The stanard Normal distribution is sampled using the optimial proposal.
   * @param nDim Number of dimensions the target has
   * @param chainLength The length of the chain
   * @param nStep Number of steps to take in the chain, the test will be done when the steps are taken
   * @param seed Random seed for the MersenneTwister
   * @param name Name of the test, this will be printed
   * @return Squared error of the sample mean and sample covariance
   */
  static double [] testRecursiveEstimators(int nDim, int chainLength, int nStep, int seed, String name) {
    
    //random number generator
    MersenneTwister rng = new MersenneTwister(seed);
    //instantiate the non-recursive estimators
    SimpleMatrix targetCovariance = SimpleMatrix.identity(nDim);
    SimpleMatrix proposalCovariance = targetCovariance.scale( Math.pow(2.38,2) / ((double)nDim) );
    //instantiate the target distribution
    TargetDistribution target = new NormalDistribution(nDim, targetCovariance);
    //instantiate the chain
    HomogeneousRwmh chain = new HomogeneousRwmh(target, chainLength, proposalCovariance, rng);
    
    //run the chain for nStep
    for (int i=0; i<nStep; i++) {
      chain.step();
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
    System.out.println("==========");
    System.out.println(name);
    System.out.println("Squared error in mean = "+squaredErrorArray[0]);
    System.out.println("Squared error in covariance = "+squaredErrorArray[1]);
    return squaredErrorArray;
  }
  
  /**FUNCTION: GET SAMPLE MEAN
   * Returns the unbiased estimator of the mean
   * @param designMatrix Collection of data, rows represent each data, columns represent each dimension
   * @return Unbiased estimator of the sample mean (column vector)
   */
  static public SimpleMatrix getSampleMean(SimpleMatrix designMatrix) {
    int n = designMatrix.numRows(); //number of datapoints
    int p = designMatrix.numCols(); //number of dimensions
    SimpleMatrix sampleMean = new SimpleMatrix(p, 1); //instantiate column vector for the sample mean
    //for each dimension, work out the sample mean
    //combine all the dimensions together and return it
    for (int i=0; i<p; i++) {
      sampleMean.set(i, designMatrix.extractVector(false, i).elementSum());
    }
    return sampleMean.divide((double)n);
  }
  
  /**FUNCTION: GET SAMPLE COVARIANCE
   * Returns the unbiased estimator of the covariance
   * @param designMatrix Collection of data, rows represent each data, columns represent each dimension
   * @param sampleMean The mean estimator in a column vector format
   * @return Unbiased estimator of the covariance
   */
  static public SimpleMatrix getSampleCovariance(SimpleMatrix designMatrix, SimpleMatrix sampleMean) {
    int n = designMatrix.numRows(); //number of datapoints
    int p = designMatrix.numCols(); //number of dimensions
    SimpleMatrix sampleCovariance = new SimpleMatrix(p,p); //instantiate a matrix for the sample covariance
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
  
}
