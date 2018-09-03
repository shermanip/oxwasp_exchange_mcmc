package jet.apps.lappel.voEqui.magnetics;

import static seed.matrix.Mat.*;


import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.PrintWriter;
import java.lang.reflect.Field;
import java.time.Instant;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.List;
import java.util.Properties;
import java.util.Vector;

import org.apache.commons.math3.random.MersenneTwister;
import org.ejml.simple.SimpleMatrix;

import accessors.jet.equi2d.JetEqui2DModel4;
import algorithmrepository.Algorithms;
import algorithmrepository.DynamicDoubleArray;
import aliceinnets.python.Parser;
import aliceinnets.python.jyplot.JyPlot;
import jafama.FastMath;
import jet.apps.equi2d.DryRuns;
import oneLiners.OneLiners;
import otherSupport.TruncatedUnivarGauss;
import seed.deepend.compilers.AccessorCompiler;
import seed.deepend.spring.SpringSerializer;
import seed.minerva.GraphicalModel;
import seed.minerva.LinearGaussianInversion;
import seed.minerva.MAPInversion;
import seed.minerva.MCMCInversion;
import seed.minerva.MultivariateNormal;
import seed.minerva.ProbabilityNode;
import seed.minerva.deepend.MinervaSettings;
import seed.minerva.diagnostics.ChannelDataEnable;
import seed.minerva.diagnostics.UnitManager;
import seed.minerva.diagnostics.magnetics.JETMagneticDiagnosticsDataSource;
import seed.minerva.diagnostics.magnetics.PickupCoils2DDataSourceJET2;
import seed.minerva.util.GraphUtil;
import seed.minerva.util.ReportingUtil;
import uk.ac.warwick.sip.mcmc.AdaptiveRwmh;
import uk.ac.warwick.sip.mcmc.BiasAdaptiveRwmh;
import uk.ac.warwick.sip.mcmc.GelmanRubinF;
import uk.ac.warwick.sip.mcmc.GraphDistribution;
import uk.ac.warwick.sip.mcmc.Mcmc;
import uk.ac.warwick.sip.mcmc.MixtureAdaptiveRwmh;
import uk.ac.warwick.sip.mcmc.RandomWalkMetropolisHastings;
import uk.ac.warwick.sip.mcmc.TargetDistribution;

public class RunVoEqui {
	
	static double ffpUnit = UnitManager.get("ffPrime");
	static double ppUnit = UnitManager.get("pPrime");
	static double pressureUnit = UnitManager.get("Pressure");	
	static final double currentUnit = UnitManager.get("Current");
	
	public static double[] sampleTruncatedDiagMvn(double sigma, int dim) {
		double[] ret = new double[dim];
		TruncatedUnivarGauss truncGauss = new TruncatedUnivarGauss();
		
		for(int i=0;i<dim;++i) {
			ret[i] = truncGauss.nextTruncatedGaussian(0.0, sigma, 0.0, Double.POSITIVE_INFINITY);
		}
		
		return ret;
	}
	
	
	public static void run(Run run) {	
		JetEqui2DModel4 m = new JetEqui2DModel4();
		String basePath = MinervaSettings.getAppsOutputPath()+"/"+m.graph.getName()+"/";				
		String outputFolder = createTimestampedOutputFolder(basePath+"/"+run.runName+"/"+run.pulse+"/"+String.format("%.3f", run.time)+"/", run.postfix);
		run.writeToPropertiesFile(outputFolder+"run.txt");
		OneLiners.makeDir(outputFolder);		

		// set the independent variable used for the flux functions
		m.equi.p_1d.set("rhoIsSqrtPsiN", run.rhoIsSqrtPsiN);
		m.equi.f_1d.set("rhoIsSqrtPsiN", run.rhoIsSqrtPsiN);
		m.equi.equi.set("rhoIsSqrtPsiN", run.rhoIsSqrtPsiN);
		
		Results results = null;
		
		if (run.initialModelXml != null && run.initialModelXml != "") {
			try {
				AccessorCompiler.populateAccessorFromXML(m, run.initialModelXml);
			} catch (Exception ex) {
				System.err.println("Could not populate accessor from xml");
				ex.printStackTrace();
				stop();
			}
			results = new Results(m);			
			
		} else {
	
			// table of configuration used for plotting
			synchronized (RunVoEqui.class) {	// Because of problems when reading from magnetics cache in parallel
				m.pulse.setValue(run.pulse);
				m.time.setValue(run.time);
				results = new Results(m);
				
				
				m.equi.npsi.setTo(run.maxPsiForProfiles);
				m.equi.numProfileValues.setValue(run.numProfileValues);
						
				m.efit.efit_ds.set("efitDda", run.efitDda);
				if (run.efitSeq != 0) m.efit.efit_ds.set("seqNo", run.efitSeq);			
				
				// if we are not using EFIT magnetics, we base the diagnostic set on the most recent dry run prior to this discharge.
				int[][] badDryRunMagnetics=null;				
				if (!run.useEfitMagnetics) badDryRunMagnetics=setupMagnetics(m,run.pulse);
					
				m.currents.poloidalFlux.set("grid_numr", run.fluxGridNumR);
				m.currents.poloidalFlux.set("grid_numz", run.fluxGridNumZ);
						
				//mag.setFancyBeamSet(50, 50, false, common.firstwall.getRZ(), 0, false, null, 0, 0, 0, false);			
				m.currents.jtor.plasmaBeamGrid.setGridResolution(run.plasmaBeamGridNumR, run.plasmaBeamGridNumZ);
				
				if (run.subdivideLCFS) {
					System.out.println("Num cells before subdivide: "+m.currents.jtor.plasmaBeamGrid.numCells());
					m.currents.jtor.plasmaBeamGrid.set("subdivideLine", m.efit.psiNOps.getLCFS()); // Set it always to the same
					m.currents.jtor.plasmaBeamGrid.set("subdivideProximityLHS", run.subdivideProximityLHS);
					m.currents.jtor.plasmaBeamGrid.set("subdivideProximityRHS", run.subdivideProximityRHS);
					m.currents.jtor.plasmaBeamGrid.set("subdivideMaxGridSize", run.subdivideMaxGridSize);
					System.out.println("Num cells after subdivide: "+m.currents.jtor.plasmaBeamGrid.numCells());		
				}
											
				m.currents.jtor.sigmaR.setValue(run.initCTSigmaR);
				m.currents.jtor.sigmaf.setValue(run.initCTSigmaf);
							
				m.currents.jtor.plasmaBeamCurrentDensities.setFullValue(m.currents.jtor.plasmaBeamCurrentDensities.getFullMean().clone());
				
				// assign value for ff-prime=100; pprime linearly 1e4 at mag axis to 0 at edge.
				m.equi.ffprime.values.setFullValue(fillArray(100.0/ffpUnit, m.equi.numProfileValues.getValue()));
				m.equi.pprime.values.setFullValue(linspace(10000.0/pressureUnit, 0, m.equi.numProfileValues.getValue()));
													
				m.equi.equi.set("coreEvalRhoCutoff", run.equi_coreEvalRhoCutOff); 
				m.equi.equi.setEquiCurrentSigma(run.equi_equiCurrentSigma);
				m.equi.equi.setSeparatrixCurrentSigma(run.equi_separatrixCurrentSigma);
				m.equi.equi.setSOLBaseCurrentSigma(run.equi_solBaseCurrentSigma);
				m.equi.equi.setSOLSigmaFalloff(run.equi_solSigmaFalloff);
				m.equi.equi.setEvalResolution(run.equi_numEvalR, run.equi_numEvalZ);
				
				m.currents.jtor.boundaryBeamsConstraint.set(MultivariateNormal.COV, 1.0); // For CT
				
				// record the magnetics setup
				dumpMagneticsSetup(m,badDryRunMagnetics,outputFolder+"/setup/");
				// plot beam grid (with CHAIN-1 EFIT data, shows  grid, hence in setup)
				plotBeamGrid(m, outputFolder+"/setup/","png",false);
								
				/*
				* fit currents, CT only
				*/				
				System.out.println("Doing first CT init");
				GraphUtil.disableAll(m.graph);
				m.currents.jtor.plasmaBeamCurrentDensities.setActive(true);
				m.currents.iron.ironCurrents.setActive(true);
				m.diagnostics.magnetics.pickups_obs.setActive(true);
				m.diagnostics.magnetics.fluxloops_obs.setActive(true);
				m.diagnostics.magnetics.saddles_obs.setActive(true);
				m.currents.jtor.boundaryBeamsConstraint.setActive(true);
				LinearGaussianInversion linv;
				linv = new LinearGaussianInversion(m.graph, false, "MagsCT");
				linv.refine();	
				tic();
				String dirname="initCT/";
				if (run.doInitCtHyperParameterOptimization) dirname ="initCT_withoutHyperOptimization/";
				results.dumpResults(outputFolder+dirname,true);	
				System.out.println("End dumping results time taken="+String.format("%.0fs",toc()/1000.));
			}
			
			
	        /*
	         *  Optimize hyperparameters using CT
	         */
			if (run.doInitCtHyperParameterOptimization) 
				hyperParameterOptimization(m,outputFolder+"initCT/",results);
			else {
				m.currents.jtor.sigmaf.setValue(485.);
				m.currents.jtor.sigmaR.setValue(0.45);
			}
			
			 // Set the covariance for the boundary beam back to the values set in run (Run3.java)
			m.currents.jtor.boundaryBeamsConstraint.set(MultivariateNormal.COV, run.boundaryBeamsConstraintSigma);
			
			/*
			 * Fit ff' to match equi constraint
			 */
			GraphUtil.disableAll(m.graph);
			m.equi.equiConstraint.setActive(true);
			m.equi.ffprime.values.setActive(true);	
			m.equi.pprime.values.setActive(true);	
			LinearGaussianInversion linv = null;
			linv = new LinearGaussianInversion(m.graph);
			linv.refine();
			dump("ffprime sigmaf="+Double.toString(m.equi.ffprime.sigmaf.getValue()));
			results.dumpResults(outputFolder+"initProfiles",false);
			
			//need to disconnect GP and connect a diagonal at this point	
			if (run.useEquiConstaint) {
				int numPlasmaBeams = m.currents.jtor.plasmaBeamCurrentDensities.mean().length;       
				double plasmaPriorMean[] = new double[numPlasmaBeams]; 
		        double plasmaPriorCovDiag[] = OneLiners.fillArray(Math.pow(300e9/currentUnit, 2.0), numPlasmaBeams);         
		        m.currents.jtor.plasmaBeamCurrentDensities.disconnect(MultivariateNormal.COV);
		        m.currents.jtor.plasmaBeamCurrentDensities.disconnect(MultivariateNormal.INVCOV);
		        m.currents.jtor.plasmaBeamCurrentDensities.disconnect(MultivariateNormal.MEAN);        
		        m.currents.jtor.plasmaBeamCurrentDensities.set(MultivariateNormal.COV, plasmaPriorCovDiag);
		        m.currents.jtor.plasmaBeamCurrentDensities.set(MultivariateNormal.MEAN, plasmaPriorMean);
			}
	        
		
	        
	        // and now iterate everything
			GraphUtil.disableAll(m.graph);
			m.currents.jtor.plasmaBeamCurrentDensities.setActive(true);
			m.currents.jtor.boundaryBeamsConstraint.setActive(run.useBoundaryBeamConstraint);
			m.currents.iron.ironCurrents.setActive(true);
			m.diagnostics.magnetics.pickups_obs.setActive(true);
			m.diagnostics.magnetics.fluxloops_obs.setActive(true);
			m.diagnostics.magnetics.saddles_obs.setActive(true);
			m.equi.equiConstraint.setActive(run.useEquiConstaint);
			m.equi.ffprime.values.setActive(run.ffprimeFree);
			m.equi.pprime.values.setActive(run.pprimeFree);
			if (run.pprimeFree && run.pprimeFreeHyperParameters) {
				m.equi.pprime.l1.setActive(true);
				m.equi.pprime.l2.setActive(true);
				m.equi.pprime.x0.setActive(true);
				m.equi.pprime.xw.setActive(true);
				m.equi.pprime.sigmaf.setActive(true);
			}
			if (run.ffprimeFree && run.ffprimeFreeHyperParameters) {
				m.equi.ffprime.l1.setActive(true);
				m.equi.ffprime.l2.setActive(true);
				m.equi.ffprime.x0.setActive(true);
				m.equi.ffprime.xw.setActive(true);
				m.equi.ffprime.sigmaf.setActive(true);
			}
		 		
			GraphUtil.printFreeParametersAndObservations(m.graph);
			dumpFreeParametersAndObservations(m.graph, outputFolder+"freeParametersAndObservations.txt");			
			dump(SpringSerializer.toSpringXML(m.graph), outputFolder+"model-init.xml");
		}
		

		if(run.map) {
			dump("Now starting MAP calculation");
			MAPInversion inv = new MAPInversion(m.graph);
			inv.getObjectiveFunction().setDebug(true);
			DynamicDoubleArray logPdfs = new DynamicDoubleArray();
			for(int i=0;i<10000;++i) {			
				String iterationFolder = outputFolder+"/iteration_"+i+"/";
				inv.refine();
				results.dumpResults(iterationFolder,false);
				logPdfs.add(m.graph.logPdf());
				dumpAppend(logPdfs.getLast(), outputFolder+"logPdf.txt");
				JyPlot p = new JyPlot();
				p.figure();
				p.plot(logPdfs.getTrimmedArray());
				p.xlabel("Iteration");
				p.ylabel("logPdf");
				p.title("logPdf");
				dump(p.getScript(), outputFolder+"scripts/logPdf.py");
				p.savefig("r'"+outputFolder+"logPdf.png"+"'");
				p.exec();			
			}
		}
		if(run.mcmc) {
			dump("Now starting MCMC");
			
			int sampleCount=10000;
			//index of the dimension, this is the centre of the cross section
			int dimOfInterest = 3;
			
			//instantiate objects to be pass onto the mcmc object
			TargetDistribution target = new GraphDistribution(m.graph); //contains the pdf
			MersenneTwister rng = new MersenneTwister(-280845742); //random number generator
			//set the proposal covariance, in adaptive methods this is the inital proposal
			SimpleMatrix proposalCovariance = SimpleMatrix.identity(target.getNDim())
					.scale(Math.pow(10, -6)/((double)target.getNDim()));
			
			//instantiate the chain and set the initial values
			Mcmc chain = new BiasAdaptiveRwmh(target, sampleCount, proposalCovariance, rng );
			chain.setInitialValue(m.graph.getFreeParameters());
			System.out.println("Number of dimensions = "+chain.getNDim());
			//run the chain
			chain.run();
			
			//for rubin-gelman, need to run additional chains
			int nChain = 5;
			//declare array of chains
			Mcmc [] mcmcArray = new Mcmc [nChain];
			mcmcArray[0] = chain; //save the first chain
			//declare variables for different initial values
			double [] initialValue = new double[chain.getNDim()];
			//initial value uses random points from the first chain
			int nIndex;
			//for each chain
			for (int iChain=1; iChain<nChain; iChain++) {
				//instantiate a chain
				Mcmc chainSub = new BiasAdaptiveRwmh(target, sampleCount, proposalCovariance, rng );
				//use random point from the first chain as the initial value
				nIndex = rng.nextInt(sampleCount);
				for (int iDim=0; iDim<chain.getNDim(); iDim++) {
					initialValue[iDim] = chain.getChain()[nIndex*chain.getNDim()+iDim];
				}
				chainSub.setInitialValue(initialValue);
				//run the chain and save it
				chainSub.run();
				mcmcArray[iChain] = chainSub;
			}
			//plot trace plot for each chain
			for (int i=0; i<nChain; i++) {
				JyPlot tracePlot = new JyPlot();
				tracePlot.figure();
				tracePlot.plot(mcmcArray[i].getChain(dimOfInterest));
				tracePlot.xlabel("number of iterations");
				tracePlot.ylabel("current density (kA.m^{-2})");
				tracePlot.show();
				tracePlot.exec();
			}
			
			//get the gelman rubin statistic
			//plot 2,3,...,nBurnInMax vs F
			int nBurnInMax = 2000;
			GelmanRubinF fStat = new GelmanRubinF(mcmcArray);
			double [] nBurnIn = new double [nBurnInMax-1];
			for (int i=0; i<nBurnIn.length; i++) {
				nBurnIn[i] = (double)(i+2);
			}
			JyPlot fPlot = new JyPlot();
			fPlot.figure();
			fPlot.plot(nBurnIn, fStat.getGelmanRubinFArray(dimOfInterest, nBurnInMax));
			fPlot.xlabel("burn-in");
			fPlot.ylabel("F statistic");
			fPlot.show();
			fPlot.exec();
			
			//plot acceptance rate
			JyPlot acceptancePlot = new JyPlot();
			acceptancePlot.figure();
			acceptancePlot.plot(chain.getAcceptanceRate());
			acceptancePlot.xlabel("number of iterations");
			acceptancePlot.ylabel("acceptance rate");
			acceptancePlot.show();
			acceptancePlot.exec();
			
			
			//plot autocorrelation
			int nLag = 100;
			double [] lag = new double[nLag];
			for (int i=0; i<nLag; i++) {
				lag[i] = (double)(i);
			}
			JyPlot autoCorrelationPlot = new JyPlot();
			autoCorrelationPlot.figure();
			autoCorrelationPlot.stem(lag, chain.getAcf(dimOfInterest, nLag));
			autoCorrelationPlot.hlines(1/Math.sqrt(sampleCount), 0, nLag-1, "r");
			autoCorrelationPlot.hlines(-1/Math.sqrt(sampleCount), 0, nLag-1, "r");
			autoCorrelationPlot.xlabel("lag");
			autoCorrelationPlot.ylabel("autocorrelation");
			autoCorrelationPlot.show();
			autoCorrelationPlot.exec();
			
			//print the efficiency for all chains
			for (int i=0; i<nChain; i++) {
				//print the efficiency
				System.out.println("efficiency = "+mcmcArray[i].getEfficiency(dimOfInterest));
				
				//calculate the posterior statistics using burn in
				mcmcArray[i].calculatePosteriorStatistics(410);
				//print the monte carlo error
				System.out.println("log precision = "+mcmcArray[i].getDifferenceLnError()[dimOfInterest]);
				//print the mean
				System.out.println("mean = "+mcmcArray[i].getPosteriorExpectation()[dimOfInterest]);
				//print the variance
				SimpleMatrix posteriorCovariance = new SimpleMatrix(mcmcArray[i].getNDim(), mcmcArray[i].getNDim(),
						true, mcmcArray[i].getPosteriorCovariance());
				System.out.println("error = "+Math.sqrt(posteriorCovariance.get(dimOfInterest, dimOfInterest)));
				System.out.println("units = kA.m^{-2}");
				
			}
			
			//plot autocorrelation of the batch
			nLag = 10;
			lag = new double[nLag];
			for (int i=0; i<nLag; i++) {
				lag[i] = (double)(i);
			}
			JyPlot batchAcfPlot = new JyPlot();
			batchAcfPlot.figure();
			batchAcfPlot.stem(lag, chain.getBatchAcf(nLag));
			batchAcfPlot.hlines(1/Math.sqrt(Math.sqrt(sampleCount)), 0, nLag-1, "r");
			batchAcfPlot.hlines(-1/Math.sqrt(Math.sqrt(sampleCount)), 0, nLag-1, "r");
			batchAcfPlot.xlabel("lag");
			batchAcfPlot.ylabel("autocorrelation");
			batchAcfPlot.show();
			batchAcfPlot.exec();
			
		}
	}
	private static void plotMcmc(long sampleCount,String outputFolder,long timeValLong,double[] sampleVec) {
		double timeVal = timeValLong/1000.; // convert from ms to sec
		String timeString=null;
		if(timeVal<60) timeString=String.format("%.0fs",timeVal);
		timeVal /= 60;   // convert sec to min
		if(timeVal<60)
			timeString=String.format("%.0fmin",timeVal);
		else {
			timeVal /= 60;   // convert from min to hours
			timeString=String.format("%.0fhours",timeVal);
		}
		String mess=Long.toString(sampleCount)+" evaluations; "+timeString;
		System.out.println(mess);
		JyPlot p = new JyPlot();
		p.figure();
		p.plot(sampleVec);
		p.xlabel("Sampled values");
		p.ylabel("Total current [MA]");
		mess = "'MCMC Jtor: "+mess+"'";
		p.title(mess);
		dump(p.getScript(), outputFolder+"scripts/mcmc.py");
		p.savefig("r'"+outputFolder+"mcmc.png"+"'");
		p.exec();			
	}
	
	public static String createTimestampedOutputFolder(String baseFolder, String postfix) {
		OffsetDateTime odt = OffsetDateTime.ofInstant(Instant.now() , ZoneOffset.UTC );
		String dateTime = odt.format(DateTimeFormatter.ofPattern("yyyyMMdd'T'HHmm"));
		
		String baseFolderPrefix = baseFolder+"/"+dateTime;
		int i=1;
		String folderName = baseFolderPrefix+postfix;
		while (new File(folderName).exists()) {
			folderName = baseFolderPrefix+postfix+"_"+i+"/";
			++i;
		}
		
		OneLiners.makeDir(folderName);
		return folderName+"/";
	}
	
	
	public static int numTrue(boolean[] a) {
		int num = 0;
		for(int i=0;i<a.length;++i) {
			if (a[i]) ++num;
		}
		
		return num;
	}
	
	public static double[] cutoff(double[] a, double minValue, double maxValue) {
		double[] ret = a.clone();
		for(int i=0;i<ret.length;++i) {
			if (ret[i] > maxValue) {
				ret[i] = maxValue;
			} else if (ret[i] < minValue) {
				ret[i] = minValue;
			}
		}
		
		return ret;
	}
	
	public static void plotBeamGrid(JetEqui2DModel4 m, String folder,String plotFormat, boolean plotLCFS) {		
			JyPlot p = new JyPlot();
			p.figure();
			p.xlabel("R [m]");
			p.ylabel("Z [m]");
			p.plot(m.firstwall.getR(), m.firstwall.getZ(), "k");
			double[] coilsR = m.currents.jtor.plasmaBeamGrid.getx();
			double[] coilsZ = m.currents.jtor.plasmaBeamGrid.gety();
			double[] coilsdR = m.currents.jtor.plasmaBeamGrid.getdx();
			double[] coilsdZ = m.currents.jtor.plasmaBeamGrid.getdy();
			boolean[] isOuterMostCell = m.currents.jtor.plasmaBeamGrid.isOuterMostCell();
			for(int j=0;j<coilsR.length;++j) {
				double R1 = coilsR[j] - coilsdR[j]/2.0;
				double R2 = coilsR[j] + coilsdR[j]/2.0;
				double Z1 = coilsZ[j] - coilsdZ[j]/2.0;
				double Z2 = coilsZ[j] + coilsdZ[j]/2.0;
				p.plot(new double[] {  R1, R2, R2, R1, R1}, new double[] { Z1, Z1, Z2, Z2, Z1  }, "k", "linewidth=0.01");	
				if (isOuterMostCell[j]) p.fill_between(new double[] { R1, R2 }, Z1, Z2, "color='y'", "alpha=0.4");
			}						
			
						
			
			double[][][] npsi0_9 = m.efit.psiNOps.getClosedContoursNormPsiAccurate(new double[] { 0.9 });
			p.plot(m.efit.psiNOps.getLCFSR(), m.efit.psiNOps.getLCFSZ(), "c--", "label='separatrix (EFIT)'");
			p.plot(npsi0_9[0][0], npsi0_9[0][1], "c--", "linewidth=0.2","label=r'$\\bar\\psi$=0.9 (EFIT)'");			
			if (plotLCFS) p.plot(m.currents.psiNOps.getLCFSR(), m.currents.psiNOps.getLCFSZ(), "r--", "label='Minerva'");					
			p.plot(m.efit.psiNOps.getAccurateMagneticAxisR(), m.efit.psiNOps.getAccurateMagneticAxisZ(), "co","label='Mag axis (EFIT)'");
			if (plotLCFS) p.plot(m.currents.psiNOps.getAccurateMagneticAxisR(), m.currents.psiNOps.getAccurateMagneticAxisZ(), "ro","label='Mag axis'");
			if (plotLCFS) p.plot(new double[] { 1.8, 4.1 }, new double[] { m.currents.psiNOps.getAccurateMagneticAxisZ(), m.currents.psiNOps.getAccurateMagneticAxisZ() }, "k");
			p.legend("numpoints=1");
			p.axis("equal");
			dump(p.getScript(), folder+"scripts/beamGrid.py");
			String filename="beamGrid."+plotFormat;
			p.savefig("r'"+folder+filename+"'");
			p.axis("auto");
			p.xlim(new double[] { 3.5, 4.0 });
			p.ylim(new double[] { -0.5, 1.5 });		
			dump(p.getScript(), folder+"scripts/beamGridZoomed.py");
			filename="beamGridZoomed."+plotFormat;
			p.savefig("r'"+folder+filename+"'");
			p.exec();		
	}
	

	/**
	 * Performs optimization of sigmaf, sigmaR, the current beam hyperparametes using CT.
	 * 
	 * @author lynton
	 *                                            
	 */
	private static void hyperParameterOptimization(JetEqui2DModel4 m,String outputFolder, Results results) {
		System.out.println("Doing hyperparameter optimization for initial CT");
		double current_sigmaf = m.currents.jtor.sigmaf.getValue();
		double current_sigmaR = m.currents.jtor.sigmaR.getValue();
	
		double[] sigmafGrid = linspace(100.0, 700.0, 15);
		double[] sigmaRGrid = linspace(0.2, 1.0, 20);
		double[][] logEvidence = new double[sigmafGrid.length][sigmaRGrid.length];
		LinearGaussianInversion linv = new LinearGaussianInversion(m.graph);
		System.out.println("Initial logEvidence "+linv.logEvidence());
		for (int i = 0; i < logEvidence.length; i++) {
			System.out.println("Hyperoptimization, doing "+i+"/"+(logEvidence.length-1)+", each "+logEvidence[0].length+" iterations");
			m.currents.jtor.sigmaf.setValue(sigmafGrid[i]);
			for (int j = 0; j < logEvidence[0].length; j++) {
				m.currents.jtor.sigmaR.setValue(sigmaRGrid[j]);
				logEvidence[i][j] = linv.logEvidence(linv.createCovPrior(), false);
			}
		}
		int[] maxi = Algorithms.maxi(logEvidence);
	
		dump(sigmafGrid, outputFolder+"textFiles/current_sigmafGrid0.txt");
		dump(sigmaRGrid, outputFolder+"textFiles/current_sigmaRGrid0.txt");
		dump(logEvidence, outputFolder+"textFiles/current_logEvidence.txt");
		dump(maxi, outputFolder+"textFiles/current_logEvidenceMaxi.txt");
	
		if(maxi[0] != -1) current_sigmaf = sigmafGrid[maxi[0]];
		if(maxi[1] != -1) current_sigmaR = sigmaRGrid[maxi[1]];
	
		m.currents.jtor.sigmaf.setValue(current_sigmaf);
		m.currents.jtor.sigmaR.setValue(current_sigmaR);
		
		dump(sigmaRGrid, outputFolder+"textFiles/logEvidence_sigmaR.txt");
		dump(sigmafGrid, outputFolder+"textFiles/logEvidence_sigmaf.txt");
		dump(new double[] { current_sigmaR, current_sigmaf }, outputFolder+"textFiles/max_sigmaR_sigmaf.txt");
		dump(new int[] { maxi[1], maxi[0] }, outputFolder+"textFiles/maxIndex_sigmaR_sigmaf.txt");
		
		JyPlot p = new JyPlot();
		p.figure();
		p.contourf(sigmaRGrid, sigmafGrid, exp(subtract(logEvidence, Algorithms.max(logEvidence))));
		p.xlabel("$\\sigma_R$ [m]");
		p.ylabel("$\\sigma_f$ [kA/m$^2$]");
		p.savefig("r'"+outputFolder+"evidence.pdf"+"'");
		p.exec();
	
		System.out.println("Ready hyperoptimization");
		
		linv = new LinearGaussianInversion(m.graph, false, "MagsCT");
		linv.refine();	
		
		// write out results, but without a report of all the random variables.
		results.dumpResults(outputFolder,false);
	}
	
	/**
	 * Uses a dry run to work out which detectors can be used, and returns an array containing the bad detectors for magnetic pickups, saddle loops and flux loops.
	 * 
	 * @author lynton
	 *                                            
	 */
	private static int[][] setupMagnetics(JetEqui2DModel4 m, int pulseNumber)
	{				
		// Switch all on, some are default off from the data source. The findBadMagnetics() method checks all
		// diagnostics, also those that are default off in the data source.
		m.diagnostics.magnetics.pickups_ds.setEnableRequest(ChannelDataEnable.ON);
		m.diagnostics.magnetics.saddles_ds.setEnableRequest(ChannelDataEnable.ON);
		m.diagnostics.magnetics.fluxloops_ds.setEnableRequest(ChannelDataEnable.ON);
		int[][] badDryRunMagnetics = DryRuns.findBadMagnetics(pulseNumber);
		for(int i=0;i<badDryRunMagnetics[0].length;++i) {
			m.diagnostics.magnetics.pickups_ds.setEnableRequest(badDryRunMagnetics[0][i], ChannelDataEnable.OFF);
		}
		if (m.pulse.getValue() == 87562) m.diagnostics.magnetics.pickups_ds.setEnableRequest(135, ChannelDataEnable.OFF);
		if (m.pulse.getValue() == 85300) m.diagnostics.magnetics.pickups_ds.setEnableRequest(134, ChannelDataEnable.OFF);
		
		for(int i=0;i<badDryRunMagnetics[1].length;++i) {
			m.diagnostics.magnetics.saddles_ds.setEnableRequest(badDryRunMagnetics[1][i], ChannelDataEnable.OFF);
		}
		for(int i=0;i<badDryRunMagnetics[2].length;++i) {
			m.diagnostics.magnetics.fluxloops_ds.setEnableRequest(badDryRunMagnetics[2][i], ChannelDataEnable.OFF);
		}
		
		m.diagnostics.magnetics.pickups_obs.disconnect(MultivariateNormal.ENABLE);
		m.diagnostics.magnetics.fluxloops_obs.disconnect(MultivariateNormal.ENABLE);
		m.diagnostics.magnetics.saddles_obs.disconnect(MultivariateNormal.ENABLE);
		
		m.diagnostics.magnetics.pickups_obs.set(MultivariateNormal.ENABLE, m.diagnostics.magnetics.pickups_ds, "getEnable");
		m.diagnostics.magnetics.saddles_obs.set(MultivariateNormal.ENABLE, m.diagnostics.magnetics.saddles_ds, "getEnable");
		m.diagnostics.magnetics.fluxloops_obs.set(MultivariateNormal.ENABLE, m.diagnostics.magnetics.fluxloops_ds, "getEnable");				

		// These pickups have been switched off by Jakob!!!
		m.diagnostics.magnetics.pickups_ds.setEnableRequest(93, ChannelDataEnable.OFF);
		m.diagnostics.magnetics.pickups_ds.setEnableRequest(94, ChannelDataEnable.OFF);

		return badDryRunMagnetics;
	}
	
	
	/**
	 * Provides customized report of graph				 * 
	 * @author lynton
	 *                                            
	 */
	private static void dumpMagneticsSetup(JetEqui2DModel4 m,int[][] badDryRunMagnetics, String outputFolder) {

		// write out magnetic diagnostics used in this model.
		{
			String filePickups = "magnetic_pickups.txt";
			String fileSaddles = "magnetic_saddles.txt";
			String fileFluxloops = "magnetic_fluxloops.txt";
			dump("Magnetic pickups", outputFolder+filePickups);
			dump("Magnetic saddle loops", outputFolder+fileSaddles);
			dump("Magnetic flux loops", outputFolder+fileFluxloops);
			if(badDryRunMagnetics != null) {
				String str1=String.format("%16s", "Used in this run");
				String str2=String.format("%18s", "Total available");
				String str3=String.format("%24s", "From Dry-run analysis");
				String str4=String.format("%23s", "Used by EFIT(CHAIN-1)");
				dumpAppend(str1+str2+str3+str4, outputFolder+filePickups);
				dumpAppend(str1+str2+str3+str4, outputFolder+fileSaddles);
				dumpAppend(str1+str2+str3+str4, outputFolder+fileFluxloops);
				int mag_avail=m.diagnostics.magnetics.pickups_ds.getNames().length;
				int sad_avail=m.diagnostics.magnetics.saddles_ds.getNames().length;
				int flx_avail=m.diagnostics.magnetics.fluxloops_ds.getNames().length;
				int[] mag = new int[] { numTrue(m.diagnostics.magnetics.pickups_obs.getEnable()),  mag_avail, mag_avail-badDryRunMagnetics[0].length,numTrue(m.efit.efit_ds.getMagneticProbesEnabled())};
				int[] sad = new int[] { numTrue(m.diagnostics.magnetics.saddles_obs.getEnable()),  sad_avail, sad_avail-badDryRunMagnetics[1].length,numTrue(m.efit.efit_ds.getSaddleLoopsEnabled())};
				int[] flx = new int[] { numTrue(m.diagnostics.magnetics.fluxloops_obs.getEnable()),flx_avail, flx_avail-badDryRunMagnetics[2].length,numTrue(m.efit.efit_ds.getFluxLoopsEnabled())};			
				dumpAppend(String.format("%16d", mag[0])+String.format("%18d", mag[1])+String.format("%24d", mag[2])+String.format("%23d", mag[3]), outputFolder+filePickups);
				dumpAppend(String.format("%16d", sad[0])+String.format("%18d", sad[1])+String.format("%24d", sad[2])+String.format("%23d", sad[3]), outputFolder+fileSaddles);
				dumpAppend(String.format("%16d", flx[0])+String.format("%18d", flx[1])+String.format("%24d", flx[2])+String.format("%23d", flx[3]), outputFolder+fileFluxloops);
			} 
			else {
				String str1=String.format("%16s", "Used in this run");
				String str2=String.format("%18s", "Total available");
				String str3=String.format("%23s", "Used by EFIT(CHAIN-1)");
				dumpAppend(str1+str2+str3, outputFolder+filePickups);
				dumpAppend(str1+str2+str3, outputFolder+fileSaddles);
				dumpAppend(str1+str2+str3, outputFolder+fileFluxloops);
				int mag_avail=m.diagnostics.magnetics.pickups_ds.getNames().length;
				int sad_avail=m.diagnostics.magnetics.saddles_ds.getNames().length;
				int flx_avail=m.diagnostics.magnetics.fluxloops_ds.getNames().length;
				int[] mag = new int[] { numTrue(m.diagnostics.magnetics.pickups_obs.getEnable()),  mag_avail,numTrue(m.efit.efit_ds.getMagneticProbesEnabled())};
				int[] sad = new int[] { numTrue(m.diagnostics.magnetics.saddles_obs.getEnable()),  sad_avail,numTrue(m.efit.efit_ds.getSaddleLoopsEnabled())};
				int[] flx = new int[] { numTrue(m.diagnostics.magnetics.fluxloops_obs.getEnable()),flx_avail,numTrue(m.efit.efit_ds.getFluxLoopsEnabled())};			
				dumpAppend(String.format("%16d", mag[0])+String.format("%18d", mag[1])+String.format("%23d", mag[2]), outputFolder+filePickups);
				dumpAppend(String.format("%16d", sad[0])+String.format("%18d", sad[1])+String.format("%23d", sad[2]), outputFolder+fileSaddles);
				dumpAppend(String.format("%16d", flx[0])+String.format("%18d", flx[1])+String.format("%23d", flx[2]), outputFolder+fileFluxloops);				
			}
		
			// Now write out details of each detector
			// names
	        String[] pickup_names = m.diagnostics.magnetics.pickups_ds.getNames();
	        String[] saddle_names = m.diagnostics.magnetics.saddles_ds.getNames();
	        String[] flux_names = m.diagnostics.magnetics.fluxloops_ds.getNames();
			// EFIT-enabled detectors (1)
			boolean[] enable_pickups_efit = m.diagnostics.magnetics.pickups_enable.getEnable();     
			boolean[] enable_saddles_efit = m.diagnostics.magnetics.saddles_enable.getEnable();
			boolean[] enable_fluxloops_efit = m.diagnostics.magnetics.fluxloops_enable.getEnable();
			// Detectors used in this run (2)
			boolean[] enable_pickups = m.diagnostics.magnetics.pickups_obs.getEnable();
			boolean[] enable_saddles = m.diagnostics.magnetics.saddles_obs.getEnable();
			boolean[] enable_fluxloops = m.diagnostics.magnetics.fluxloops_obs.getEnable();
			// magnetic pickup coords
			double[] rcoord= m.diagnostics.magnetics.pickups_ds.getR();
			double[] zcoord= m.diagnostics.magnetics.pickups_ds.getZ();
	        String str0=String.format("%n%n%s%n", "--------------------------------------");
	        String str1=String.format("%5s", "index");
			String str2=String.format("%14s", "signal name");
			String str3=String.format("%15s", "EFIT-enabled");
			String str4=String.format("%19s", "Used in this run");
			String str5=String.format("%10s%10s", "R-coord","Z-coord");
			dumpAppend(str0+str1+str2+str3+str4+str5, outputFolder+filePickups);
			dumpAppend(str0+str1+str2+str3+str4, outputFolder+fileSaddles);
			dumpAppend(str0+str1+str2+str3+str4, outputFolder+fileFluxloops);
			for (int i=0;i<pickup_names.length;i++) {
				String coord = String.format("%10.3f,%10.3f", rcoord[i],zcoord[i]);
				String enable = enable_pickups[i] ? String.format("%19d",1):String.format("%19s"," ");
				String enable_efit = enable_pickups_efit[i] ? String.format("%15d",1):String.format("%15s"," ");
				dumpAppend(String.format("%5d",i)+String.format("%14s",pickup_names[i])+enable_efit+enable+coord, outputFolder+filePickups);
			}
			for (int i=0;i<saddle_names.length;i++) {
				String enable = enable_saddles[i] ? String.format("%19d",1):String.format("%19s"," ");
				String enable_efit = enable_saddles_efit[i] ? String.format("%15d",1):String.format("%15s"," ");
				dumpAppend(String.format("%5d",i)+String.format("%14s",saddle_names[i])+enable_efit+enable, outputFolder+fileSaddles);
			}
			for (int i=0;i<flux_names.length;i++) {
				String enable = enable_fluxloops[i] ? String.format("%19d",1):String.format("%19s"," ");
				String enable_efit = enable_fluxloops_efit[i] ? String.format("%15d",1):String.format("%15s"," ");
				dumpAppend(String.format("%5d",i)+String.format("%14s",flux_names[i])+enable_efit+enable, outputFolder+fileFluxloops);
			}		
		}		
		
		//write XML file  (LCA)
//		dump(m.getLocalXMLConfiguration(), outputFolder+"model.xml");
//		dump(m.toConfigurationXML(), outputFolder+"model.xml");
		
		
	}	
	

	public static class Run {
		int pulse;
		double time;
		String runName = "";
		String postfix = "";
		String initialModelXml = "";
		
		boolean useBoundaryBeamConstraint = true;
		boolean useEquiConstaint = true;
		
		boolean pprimeFree = true;
		boolean ffprimeFree = true;
		boolean hrts_shiftFree = true;
		
		boolean pprimeFreeHyperParameters = true;
		boolean ffprimeFreeHyperParameters = true;
		// controls whether the flux functions are defined with psi or root psi as independent variable.
		boolean rhoIsSqrtPsiN = false;
		
		String efitDda = "EFIT"; // "EFTF"
		int efitSeq = 0; // 164						
	
		boolean useEfitMagnetics = false;
		
		int fluxGridNumR = 60;
		int fluxGridNumZ = 60;
		
		int plasmaBeamGridNumR = 50;
		int plasmaBeamGridNumZ = 50;
		boolean subdivideLCFS = true;
		double subdivideProximityLHS = 0.05;
		double subdivideProximityRHS = 0.025;
		double subdivideMaxGridSize = 0.02;
		
		int numProfileValues = 50;
		double maxPsiForProfiles = 1.2;
					
		double initCTSigmaR = 0.2;
		double initCTSigmaf = 2000;
		boolean doInitCtHyperParameterOptimization = true;
		
		double equi_coreEvalRhoCutOff = 0.0;
		double equi_equiCurrentSigma = 50e3;
		double equi_separatrixCurrentSigma = 50e3;
		double equi_solBaseCurrentSigma = 50e3;
		double equi_solSigmaFalloff = 10.0;
		int equi_numEvalR = 3;
		int equi_numEvalZ = 3;
		
		double boundaryBeamsConstraintSigma = 20.0; // kA/m2

		boolean map=true;
		boolean mcmc=false;
		/**
		 * In A/m2 it seems
		 * @param equiCurrentSigma
		 */
		public void setEquiCurrentSigma(double equiCurrentSigma) {
			equi_equiCurrentSigma = equiCurrentSigma;
			equi_separatrixCurrentSigma = equiCurrentSigma;
			equi_solBaseCurrentSigma = equiCurrentSigma;			
		}	
		
		/**
		 * Writes the properties of a class into a text file.				 * 
		 * @author lynton
		 *                                            
		 */
		public void writeToPropertiesFile(String filename) {
			Field[] fields = this.getClass().getDeclaredFields();
			SortedProperties properties = new SortedProperties();
		
			Arrays.sort(fields, (Comparator<Field>) (Field o1, Field o2)-> {
				return o1.getName().compareTo(o2.getName());
			});
			
			try {
				for(Field field : fields) {			
					Object value = field.get(this);
					properties.setProperty(field.getName(), value == null ? "" : field.get(this).toString());				 
				} 
				FileOutputStream out = new FileOutputStream(filename);
				properties.store(out,  null);
			} catch(Exception ex) {
				ex.printStackTrace();
			}			
		}

		/**
		 * Reads the properties of a class into a text file.				 * 
		 * @author lynton
		 *                                            
		 */
		public void readFromPropertiesFile(String filename) {
			try {
					// read in all properties in the file
					Properties props = new Properties();
					InputStream input =	new FileInputStream(filename);
					props.load(input);
					Enumeration<?> e = props.propertyNames();
					// for each property, set the value in the class
					while (e.hasMoreElements()) {
					      String key = (String) e.nextElement();
					      String member=props.getProperty(key);
					      Field field= this.getClass().getDeclaredField(key);
					      Class<?> type = field.getType();
					      if(type.isAssignableFrom(Boolean.TYPE)) field.set(this,Boolean.parseBoolean(member));
					      else if(type.isAssignableFrom(Double.TYPE)) field.set(this,Double.parseDouble(member));
					      else if(type.isAssignableFrom(Integer.TYPE)) field.set(this,Integer.parseInt(member));
					      else if(type.isAssignableFrom(key.getClass())) field.set(this,member);
					}
			} catch(Exception ex) {
					ex.printStackTrace();
			}			
		}		
	}
	
	public static class SortedProperties extends Properties {
		private static final long serialVersionUID = 6985856099156029926L;
		public Enumeration keys() {
		     Enumeration<Object> keysEnum = super.keys();
		     Vector<String> keyList = new Vector<String>();
		     while(keysEnum.hasMoreElements()){
		       keyList.add((String)keysEnum.nextElement());
		     }
		     Collections.sort(keyList);
		     return keyList.elements();
		  }
	}
	
	public static double[] getObsLogPdf(GraphicalModel g) {
		List<ProbabilityNode> nodes = g.getObservedNodes(); 
		double[] ret = new double[nodes.size()];
		int index = 0;
		for(ProbabilityNode n : nodes) {
			ret[index++] = n.logpdf();
		}
		
		return ret;
	}

	public static double[] getFreeLogPdf(GraphicalModel g) {
		List<ProbabilityNode> nodes = g.getUnobservedNodes(); 
		double[] ret = new double[nodes.size()];
		int index = 0;
		for(ProbabilityNode n : nodes) {
			ret[index++] = n.logpdf();
		}
		
		return ret;
	}

	
	public static class Results {
		JetEqui2DModel4 m; 
		int pulse;
		double time;
		double[] curObsLogPdf = null;
		double[] curFreeLogPdf= null;
		double[] prevObsLogPdf= null;
		double[] prevFreeLogPdf= null;;
		
		/**
		 * Class writes out all results.
		 * 
		 * @author lynton
		 *                                            
		 */
		public Results(JetEqui2DModel4 m) {
			this.m = m;
			this.pulse = m.pulse.getValue();
			this.time = m.time.getValue();
		}
		
		/**
		 * Method dumps all results (plots and text files).
		 * 
		 * @author lynton
		 *                                            
		 */
		public void dumpResults(String folder,boolean makeRandomVariableReport) {

			// get summary of the prior and likelihood pdfs
			List<Object> pdfSummary=getPdfSummaries(curObsLogPdf,curFreeLogPdf,prevObsLogPdf,prevFreeLogPdf);
			curObsLogPdf=(double[]) pdfSummary.get(2);
			curFreeLogPdf=(double[])pdfSummary.get(3);	
			prevObsLogPdf=(double[])pdfSummary.get(4);
			prevFreeLogPdf=(double[])pdfSummary.get(5);

			// get current profile across mid-plane from the current beams and from the equilibrium virtual observations [p'+1/Rmu_0)ff']
			List<Object> jtorProfiles=getJtorProfiles(m);

			dumpTextOutput(folder,pdfSummary,jtorProfiles);
			
			dumpPlots(folder,pdfSummary,jtorProfiles);
			
			if(makeRandomVariableReport) ReportingUtil.report(m.graph, folder+"randomVariableReport");

		}
			
		/**
		 * Method generates all plotted results.
		 * 
		 * @author lynton
		 *                                            
		 */
		private void dumpPlots(String folder, List<Object> pdfSummary, List<Object> jtorProfiles) {

			folder = folder+"/plots/";
			double[][] pickups_predobs = new double[][] { m.diagnostics.magnetics.pickups_obs.getFullMean(), sqrt((double[]) m.diagnostics.magnetics.pickups_obs.getFullCov()), m.diagnostics.magnetics.pickups_obs.getFullValue() };
			double[] pickups_normdiff = m.diagnostics.magnetics.pickups_obs.diffNormalisedFull();
			double[][] saddles_predobs = new double[][] { m.diagnostics.magnetics.saddles_obs.getFullMean(), sqrt((double[])m.diagnostics.magnetics.saddles_obs.getFullCov()), m.diagnostics.magnetics.saddles_obs.getFullValue() };
			double[] saddles_normdiff = m.diagnostics.magnetics.saddles_obs.diffNormalisedFull();
			double[][] fluxloops_predobs = new double[][] { m.diagnostics.magnetics.fluxloops_obs.getFullMean(), sqrt((double[])m.diagnostics.magnetics.fluxloops_obs.getFullCov()), m.diagnostics.magnetics.fluxloops_obs.getFullValue() };
			double[] fluxloops_normdiff = m.diagnostics.magnetics.fluxloops_obs.diffNormalisedFull();
			boolean[] pickups_enable = m.diagnostics.magnetics.pickups_obs.getEnable();
			boolean[] saddles_enable = m.diagnostics.magnetics.saddles_obs.getEnable();
			boolean[] fluxloops_enable = m.diagnostics.magnetics.fluxloops_obs.getEnable();
			int profileCount=m.equi.numProfileValues.getValue();
			double [] npsiValues= m.equi.npsi.getValue();
			// get the units of the independent variable used by the flux functions
			boolean rhoIsSqrtPsiN = (boolean) m.equi.equi.get("rhoIsSqrtPsiN");
			// note that the following variable represents a linearly space vector either normalized psi values of sqrt of these values,
			// depending on the setting of rhoIsSqrtPsiN
			double[] npsiGrid = linspace(npsiValues[0],npsiValues[profileCount-1], 100);
			double[] pprime = mulElem(UnitManager.getInstance().getUnit("pPrime"), m.equi.pprime.pprime1d.evalScalar1D(npsiGrid)); // Pa/[Wb/rad], which is what I suppose is written to EFIT PPF
			double[] ffprime = mulElem(UnitManager.getInstance().getUnit("ffPrime"), m.equi.ffprime.ffprime1d.evalScalar1D(npsiGrid));
			String[] obsNames = (String[]) pdfSummary.get(0);
			String[] freeNames = (String[]) pdfSummary.get(1);
			int[] beamIndices = (int[]) jtorProfiles.get(0);
			double[] Raxis = (double[]) jtorProfiles.get(1);
			double[] npsi = (double[]) jtorProfiles.get(3);
			double[] beamCurrentDensities = (double[]) jtorProfiles.get(4);
			double[] beamCurrentDensitiesEqui= (double[]) jtorProfiles.get(5);
			double[]jtorMidplane = (double[]) jtorProfiles.get(6);
			double[] jtorMidplaneEqui = (double[]) jtorProfiles.get(7);
			// convert to kA/m^2
			double[] beamCurrentDensitiesKa=OneLiners.arrayMultiply(beamCurrentDensities, 1e-3);
			double[] beamCurrentDensitiesEquiKa=OneLiners.arrayMultiply(beamCurrentDensitiesEqui, 1e-3);
			double[] jtorMidplaneKa=OneLiners.arrayMultiply(jtorMidplane, 1e-3);
			double[] jtorMidplaneEquiKa=OneLiners.arrayMultiply(jtorMidplaneEqui, 1e-3);

			String plotFormat="png";
			String filename = null;

			// assemble the geometric information for the magnetic pickup coils
			Object[] magneticsTable = getPickupsTable(m.diagnostics.magnetics.mag_ds, m.diagnostics.magnetics.pickups_ds);			
//			ModelReport.report(magneticsTable, m.pulse, m.time, m.firstwall, m.currents.magneticModel, m.currents.psiNOps, m.currents.poloidalFlux, folder);

			plotBeamGrid(m,folder,plotFormat,true);

			plotColoredGrid(beamCurrentDensitiesKa, folder,"plasmaBeamCurrentDensities",plotFormat, true,"'$kA/m^2$'","'Plasma Beam Current Density'");
			plotColoredGrid(beamCurrentDensitiesEquiKa, folder, "plasmaBeamEquiCurrentDensities",plotFormat, true,"'$kA/m^2$'","'Equi Current Density'");
			plotColoredGrid(m.equi.equiConstraint.diffNormalisedFull(), folder, "plasmaBeamDiffEquiCurrentDensities",plotFormat, true,"'[-]'","'Equi constraint: (Jbeam[i]-Jequi[i])/sigma'");
			
			/*
			 *  Covariance scale length
			 */
			{
				double l1=m.equi.ffprime.l1.getValue();
				double l2=m.equi.ffprime.l2.getValue();
				double x0=m.equi.ffprime.x0.getValue();
				double xw=m.equi.ffprime.xw.getValue();
				plotCovarianceScaleLength(m.equi.npsi.getValue(), l1, l2, xw, x0,"$ff^\\prime$ covariance scale length",folder, "ffprimeCovarianceScaleLength", plotFormat);
			}
			{
				double l1=m.equi.pprime.l1.getValue();
				double l2=m.equi.pprime.l2.getValue();
				double x0=m.equi.pprime.x0.getValue();
				double xw=m.equi.pprime.xw.getValue();
				plotCovarianceScaleLength(m.equi.npsi.getValue(), l1, l2, xw, x0,"$p^\\prime$ covariance scale length",folder, "pprimeCovarianceScaleLength", plotFormat);
			}
			
			/*
			 * Currents scatter plot
			 */
			JyPlot p = new JyPlot();
			p.figure("figsize=(12, 9)");
			p.subplot(2,1,1);
			p.plot(beamCurrentDensitiesEquiKa, "-o", "label=r'$Rp^\\prime+\\mu_0/R ff_\\prime$'");
			p.plot(beamCurrentDensitiesKa, "-o", "label='Beam current density'", "markersize=3");
			p.title(pulse+"/"+time);
			p.xlabel("Position");
			p.ylabel("Current density ($kA/m^2$)");
			p.grid();			
			p.legend("loc='best'");			
			p.subplot(2,1,2);
			p.plot(beamCurrentDensitiesKa, beamCurrentDensitiesEquiKa, "o");
			p.xlabel("Beam current density ($kA/m^2$)");
			p.ylabel("r'$Rp^\\prime+\\mu_0/R ff_\\prime$ ($kA/m^2$)'");
			p.grid();			
			p.tight_layout();
			dump(p.getScript(), folder+"scripts/equiVsBeamCurrentDensities.py");
			filename="equiVsBeamCurrentDensities."+plotFormat;
			p.savefig("r'"+folder+filename+"'");			
			p.exec();

			// plot poloidal flux surfaces
			String poloidalFluxTitle="'Poloidal flux contours for #"+pulse+"; t="+time+"'";
			plotFluxSurfaces(false,0,magneticsTable,pickups_enable,poloidalFluxTitle,plotFormat,folder,"poloidalFlux");
			plotFluxSurfaces(false,1,magneticsTable,pickups_enable,poloidalFluxTitle,plotFormat,folder,"poloidalFlux2");

			// plot poloidal flux surfaces (and overplot the CHAIN-1 EFIT surfaces)
			poloidalFluxTitle="'Comparison of MINERVA and CHAIN-1 EFIT for #"+pulse+"; t="+time+"'";
			plotFluxSurfaces(true,0,magneticsTable,pickups_enable,poloidalFluxTitle,plotFormat,folder,"poloidalFluxComparison");
			
			// pickups
		    plot_magnetics(folder+"scripts/pickups",folder+"pickups",plotFormat,pickups_predobs, pickups_normdiff, pickups_enable,
			"pickups #"+pulse+", time="+time+"s","Channel","Magnetic field [T]","Normalized fit [-]");

			// saddles			
		    plot_magnetics(folder+"scripts/saddles",folder+"saddles",plotFormat, saddles_predobs, saddles_normdiff, saddles_enable,
			"saddles #"+pulse+", time="+time+"s","Channel","Poloidal Flux [Wb/rad]","Normalized fit [-]");
			
			// flux loops			
		    plot_magnetics(folder+"scripts/fluxloops",folder+"fluxloops",plotFormat, fluxloops_predobs, fluxloops_normdiff, fluxloops_enable,
			"fluxloops #"+pulse+", time="+time+"s","Channel","Poloidal Flux [Wb/rad]","Normalized fit [-]");
			
			// Pressure flux profiles 
			p = new JyPlot();			
			p.figure("figsize=(12, 9)");			
			p.plot(npsiGrid, m.equi.p_1d.evalScalar1D(npsiGrid), "label='$p_{ne}$'");
			p.title(pulse+"/"+time);
			if (rhoIsSqrtPsiN)
				p.xlabel("$\\sqrt {\\psi_N}$");
			else
				p.xlabel("$\\psi_N$");
			p.ylabel("$Pressure$ [kPa]");					
			p.legend("loc='best'");
			p.grid();			
			dump(p.getScript(), folder+"scripts/pressureProfiles.py");
			filename="pressureProfiles."+plotFormat;
			p.savefig("r'"+folder+filename+"'");	
			p.exec();
			
			// pprime flux profile
			p = new JyPlot();			
			p.figure("figsize=(12, 9)");			
			// plot the inference MAP profile of pprime
			p.plot(npsiGrid, pprime, "label='$pprime$ Minerva'");
			// plot the EFIT pprime values; note that the x-values, psiPvalues, are  in units of normalized poloidal flux 
			double[] psiPvalues = m.efit.efit_ds.getPsiPprime();
			double[] psiPvalues2 = linspace(psiPvalues[0],psiPvalues[psiPvalues.length-1], 100);
			double[] psiPvalues3 = new double[psiPvalues2.length];
			if (rhoIsSqrtPsiN)
				for(int i=0;i<psiPvalues2.length;i++) 
					psiPvalues3[i]=Math.sqrt(psiPvalues2[i]);
			else
				psiPvalues3=psiPvalues2;
			p.plot(psiPvalues3, m.efit.efit_ds.evalPprime(psiPvalues2), "label='$pprime$ EFIT'");
			// now the title etc
			p.title(pulse+"/"+time);
			if (rhoIsSqrtPsiN)
				p.xlabel("$\\sqrt {\\psi_N}$");
			else
				p.xlabel("$\\psi_N$");
			p.ylabel("$Pprime$");					
			p.legend("loc='best'");
			p.grid();		
			dump(p.getScript(), folder+"scripts/pprime.py");
			filename="pprime."+plotFormat;
			p.savefig("r'"+folder+filename+"'");	
			p.exec();
			
			
			// ffprime flux profile
			p = new JyPlot();			
			p.figure("figsize=(12, 9)");			
			// plot the inference MAP profile of ffprime
			p.plot(npsiGrid, ffprime, "label='$ffprime$ Minerva'");
			// plot the EFIT ffprime values; note that the x-values, psiFvalues, are  in units of normalized poloidal flux 
			double[] psiFfvalues = m.efit.efit_ds.getPsiFfprime();
			double[] psiFfvalues2 = linspace(psiFfvalues[0],psiFfvalues[psiFfvalues.length-1], 100);
			double[] psiFfvalues3 = new double[psiFfvalues2.length];
			if (rhoIsSqrtPsiN)
				for(int i=0;i<psiFfvalues2.length;i++) 
					psiFfvalues3[i]=Math.sqrt(psiFfvalues2[i]);
			else
				psiFfvalues3=psiFfvalues2;
			p.plot(psiFfvalues3, m.efit.efit_ds.evalFfprime(psiFfvalues2), "label='$ffprime$ EFIT'");
			// now the title etc
			p.title(pulse+"/"+time);
			if (rhoIsSqrtPsiN)
				p.xlabel("$\\sqrt {\\psi_N}$");
			else
				p.xlabel("$\\psi_N$");
			p.ylabel("$Ffprime$");					
			p.legend("loc='best'");
			p.grid();		
			dump(p.getScript(), folder+"scripts/ffrime.py");
			filename="ffprime."+plotFormat;
			p.savefig("r'"+folder+filename+"'");	
			p.exec();
							
					
			// logPdfs
			double barWidth = 0.1;
			p = new JyPlot();			
			p.figure("figsize=(12, 9)");			
			p.barh(linspace(0, curFreeLogPdf.length-1, curFreeLogPdf.length), curFreeLogPdf, barWidth, "label='Current'");
			p.yticks(linspace(0, curFreeLogPdf.length-1, curFreeLogPdf.length), freeNames);							
			p.barh(add(barWidth, linspace(0, curFreeLogPdf.length-1, curFreeLogPdf.length)), curFreeLogPdf, barWidth, "label='Previous'"); 						
			p.title("LogPdf free parameters "+pulse+"/"+time);
			p.xlabel("$logPdf$");			
			p.grid();
			p.legend();
			p.tight_layout();
			dump(p.getScript(), folder+"scripts/logPdfFree.py");		
			filename="logPdfFree."+plotFormat;
			p.savefig("r'"+folder+filename+"'");			
			p.exec();
			
			
			p = new JyPlot();			
			p.figure("figsize=(12, 9)");
			p.barh(linspace(0, curObsLogPdf.length-1, curObsLogPdf.length), curObsLogPdf, barWidth, "label='Current'");
			p.yticks(linspace(0, curObsLogPdf.length-1, curObsLogPdf.length), obsNames);							
			p.barh(add(barWidth, linspace(0, curObsLogPdf.length-1, curObsLogPdf.length)), prevObsLogPdf, barWidth, "label='Previous'"); 						
			p.title("LogPdf observations "+pulse+"/"+time);
			p.xlabel("$logPdf$");			
			p.grid();
			p.legend();
			p.tight_layout();
			dump(p.getScript(), folder+"scripts/logPdfObs.py");	
			filename="logPdfObs."+plotFormat;
			p.savefig("r'"+folder+filename+"'");			
			p.exec();	
			
						
			p = new JyPlot();			
			p.figure("figsize=(12, 9)");
			p.yticks(linspace(0, curObsLogPdf.length-1, curObsLogPdf.length), obsNames);							
			p.barh(linspace(0, curObsLogPdf.length-1, curObsLogPdf.length), subtract(curObsLogPdf, prevObsLogPdf), barWidth, "label='Diff from previous'"); 						
			p.title("LogPdf observations "+pulse+"/"+time);
			p.xlabel("$logPdf$");			
			p.grid();
			p.legend();
			p.tight_layout();
			dump(p.getScript(), folder+"scripts/logPdfObsDiffs.py");				
			filename="logPdfObsDiffs."+plotFormat;
			p.savefig("r'"+folder+filename+"'");			
			p.exec();
						
			p = new JyPlot();			
			p.figure("figsize=(12, 9)");
			p.yticks(linspace(0, curFreeLogPdf.length-1, curFreeLogPdf.length), freeNames);							
			p.barh(linspace(0, curFreeLogPdf.length-1, curFreeLogPdf.length), subtract(curFreeLogPdf, prevFreeLogPdf), barWidth, "label='Diff from previous'"); 						
			p.title("LogPdf Free nodes "+pulse+"/"+time);
			p.xlabel("$logPdf$");			
			p.grid();
			p.legend();
			p.tight_layout();
			dump(p.getScript(), folder+"scripts/logPdfFreeDiffs.py");		
			filename="lofPdfFreeDiffs."+plotFormat;
			p.savefig("r'"+folder+filename+"'");			
			p.exec(); 
			

			// jtor
			p = new JyPlot();
			p.figure();
			p.suptitle("Current density midplane "+pulse+"/"+time);			
			p.subplot(2,1,1);
			p.plot(Raxis, jtorMidplaneKa, "b", "label='jtor'");
			p.plot(Raxis, jtorMidplaneEquiKa, "r", "label='jtor_equi'");
			p.xlabel("$R$ [m]");
			p.ylabel("Current density [kA/m$^2$]");
			p.grid();		
			p.legend();
			p.subplot(2,1,2);
			p.plot(npsi, jtorMidplaneKa, "b", "label='jtor'");
			p.plot(npsi, jtorMidplaneEquiKa, "r", "label='jtor_equi'");
			p.xlabel("$\\psi_\\mathrm{norm}$");
			p.ylabel("Current density [kA/m$^2$]");			
			p.grid();
			p.legend();
			p.tight_layout();
			dump(p.getScript(), folder+"scripts/currentDensityMidplane.py");	
			filename="currentDensityMidplane."+plotFormat;
			p.savefig("r'"+folder+filename+"'");			
			p.exec();
			
			
			// Profiles at edge along mid-plane, together with beam limits
			p = new JyPlot();
			// create two sub-plots, the right-hand one is for the legend
			p.subplots("ncols=2", "figsize=(6.8,4.1)","gridspec_kw={'width_ratios':[2.7,1]}"); 
			// set  to the first (lh) region.
			p.write("plt.axes(plt.gcf().get_axes()[0])");
			String titleString= "'Flux Function profile for #"+pulse+", t="+time+"s'";
			p.title(titleString);
			p.plot(npsi, jtorMidplaneKa,"b", "label=r'$j_\\phi$ (beam)'");
			p.plot(npsi, jtorMidplaneEquiKa, "r", "label=r'$j_\\phi$ (equi)'");
			p.xlabel("$\\psi_\\mathrm{norm}$");
			p.ylabel("Current density [kA/m$^2$]");
//			double[] xlim = new double[] { Algorithms.min(npsiGrid), Algorithms.max(npsiGrid) };
			double[] xlim = new double[] { 0.8, 1.05 }; 
			int firstIndex = OneLiners.getNearestIndex(npsi, xlim[0]);
			double ylim0 =  min(Algorithms.min(jtorMidplaneKa), Algorithms.min(jtorMidplaneEquiKa));
			double ylim1 =  max(Algorithms.max(jtorMidplaneKa), Algorithms.max(jtorMidplaneEquiKa));
			double[] ylim = {ylim0,ylim1};
			p.xlim(xlim);
			p.ylim(ylim);
			// create a second axis on the right-hand side of the figure (for pressure)
			p.twinx();
			double[] ptot = m.equi.p_1d.evalScalar1D(npsi);
			p.plot(npsi, OneLiners.arrayMultiply(ptot, 1e-3), "g--", "label='plasma pressure'");
			p.ylabel("Pressure [kPa/m$^2$]");
			double ymin=Algorithms.min(extractElems(ptot, firstIndex, 1, npsi.length-1))*1e-3;
			double ymax=Algorithms.max(extractElems(ptot, firstIndex, 1, npsi.length-1))*1.2 *1e-3;
			ylim = new double[] {ymin,ymax}; 
			p.ylim(ylim);
			p.write("plt.gca().yaxis.label.set_color('green')");
			p.write("plt.gca().tick_params(axis='y', colors='green')");
			// plot locations of flux-coordinates
			int lastBeamIndex = beamIndices[0];
			String labelString1="label='plot resolution'";
			String labelString2="label='flux-function ordinates'";
			for(int i=0;i<beamIndices.length;++i) {
				if (beamIndices[i] != -1) {
					if (npsi[i] >= xlim[0]) {
						if (beamIndices[i] != lastBeamIndex) {
							p.axvline("x="+npsi[i]+", color='m', linestyle='--', linewidth=0.1, alpha=0.5,"+labelString1);
							labelString1="label=''";
						}
					}
				}
			}
			double[] npsiProfiles = m.equi.npsi.getArray();
			for(int i=0;i<npsiProfiles.length;++i) {
				if (npsiProfiles[i] >= xlim[0]) {					
					p.axvline("x="+npsiProfiles[i]+", color='c', linestyle='--', linewidth=0.1, alpha=0.5,"+labelString2);
					labelString2="label=''";
				}
			}
			// set  to the second (right hand) plot region and place the legend there
			p.write("plt.axes(plt.gcf().get_axes()[1])");
			p.write("plt.gca().axis('off')");
			p.write("plt.gcf().legend(loc=7, prop={'size': 9})");
			p.tight_layout();
			//p.grid();
			dump(p.getScript(), folder+"scripts/profilesAtEdge.py");	
			filename="profilesAtEdge."+plotFormat;
			p.savefig("r'"+folder+filename+"'");			
			p.exec();
			

		}

		/**
		 * plots covariance scale length
		 * 
		 * @author lynton
		 *                                            
		 */
		private void plotCovarianceScaleLength(double[] xvals, double l1, double l2, double xw, double x0,String title,String folder, String filename, String plotFormat) {

			double[] xvec=linSpace(xvals[0], xvals[xvals.length-1], 300);
			double[] yvec=new double[xvec.length];
			for(int i=0;i<xvec.length;i++)
				yvec[i]=(l1+l2)/2.-(l1-l2)/2.*FastMath.tanh((xvec[i]-x0)/xw);
			JyPlot p = new JyPlot();
		    p.figure();
		    p.plot(xvec, yvec,"'-',label=r'$l(x)$'");
			// get the units of the independent variable used by the flux functions
			boolean rhoIsSqrtPsiN = (boolean) m.equi.equi.get("rhoIsSqrtPsiN");
			if (rhoIsSqrtPsiN)
			    p.xlabel("r'$\\sqrt {\\hat\\psi_p}$'");
//				p.xlabel("$\\sqrt {\\psi_N}$");
			else
			    p.xlabel("r'$\\hat\\psi_p$'");
//				p.xlabel("$\\psi_N$");

	    	p.axvline("x="+xvals[0]+",color='gray',linestyle='--',alpha=0.7,label='node position'");
		    for (int i=1;i<xvals.length;i++)
		    	p.axvline("x="+xvals[i]+",color='gray',linestyle='--',alpha=0.7");
		    p.axhline("y="+l1+",linestyle='--',color='c',label=r'core saturation value, l1=$"+l1+"$'");
		    p.axhline("y="+l2+",linestyle=':',color='c',label=r'edge saturation value, l2=$"+l2+"$'");
		    p.axvline("x="+x0+",linestyle='--',color='g',label=r'x0=$"+x0+"$'");
		    double diffVal=x0-xw;
		    p.axvline("x="+diffVal+",linestyle='--',color='r',label=r'x0-xw=$"+diffVal+"$'");
		    double addVal=x0+xw;
		    p.axvline("x="+addVal+",linestyle=':',color='r',label=r'x0+xw=$"+addVal+"$'");
		    p.legend();
		    p.title("'"+title+"'");
			dump(p.getScript(), folder+"/scripts/"+filename+"_0.py");
			p.savefig("r'"+folder+filename+"_0."+plotFormat+"'");
			p.exec();			

			// plot again without the node positions plotted (as these can obscure the plot).
			p.figure();
		    p.plot(xvec, yvec,"'-',label=r'$l(x)$'");
			if (rhoIsSqrtPsiN)
			    p.xlabel("r'$\\sqrt {\\hat\\psi_p}$'");
//				p.xlabel("$\\sqrt {\\psi_N}$");
			else
			    p.xlabel("r'$\\hat\\psi_p$'");
//				p.xlabel("$\\psi_N$");
		    p.axhline("y="+l1+",linestyle='--',color='c',label=r'core saturation value, l1=$"+l1+"$'");
		    p.axhline("y="+l2+",linestyle=':',color='c',label=r'edge saturation value, l2=$"+l2+"$'");
		    p.axvline("x="+x0+",linestyle='--',color='g',label=r'x0=$"+x0+"$'");
		    diffVal=x0-xw;
		    p.axvline("x="+diffVal+",linestyle='--',color='r',label=r'x0-xw=$"+diffVal+"$'");
		    addVal=x0+xw;
		    p.axvline("x="+addVal+",linestyle=':',color='r',label=r'x0+xw=$"+addVal+"$'");
		    p.legend();
		    p.title("'"+title+"'");
			dump(p.getScript(), folder+"/scripts/"+filename+".py");
			p.savefig("r'"+folder+filename+"."+plotFormat+"'");
			p.exec();			
		}
	
	
		/*
		 * Plots poloidal flux conotours.  If plotEfit=.true. then EFIT flux contours are also plotted.
		   magneticOptions for int(abcde) : 
		             e=1    plot (R,Z) location of probes
		             d=1    plot text info for each probes
		   
		 */
		
		private void plotFluxSurfaces(boolean plotEfit, int magneticOptions, Object[] magneticsTable, boolean[] pickupsEnabled, String title,String plotFormat,String folder, String filename) {
			JyPlot p = new JyPlot();
			p.figure("figsize=(12, 9)");
			p.title(title);	
			p.plot(m.firstwall.getR(), m.firstwall.getZ(),"k");		
			p.plot(m.currents.psiNOps.getLCFS()[0], m.currents.psiNOps.getLCFS()[1], "b", "label=r'$\\psi_p$[Minerva]'");
			p.plot(m.currents.psiNOps.getFOFS()[0], m.currents.psiNOps.getFOFS()[1], "b");
			
			double[][][] c = m.currents.psiNOps.getClosedContoursNormPsiAccurate(linspace(0.0, 1.0, 20));	
			for (int j = 0; j < c.length; ++j) {
				if (c[j][0] != null && c[j][0].length != 0) p.plot(c[j][0], c[j][1], "b");
			}	
			if(plotEfit) {
				p.plot(m.efit.psiNOps.getLCFS()[0], m.efit.psiNOps.getLCFS()[1], "g", "label=r'$\\psi_p$[EFIT]'");
				p.plot(m.efit.psiNOps.getFOFS()[0], m.efit.psiNOps.getFOFS()[1], "g");
				c = m.efit.psiNOps.getClosedContoursNormPsiAccurate(linspace(0.0, 1.0, 20));	
				for (int j = 0; j < c.length; ++j) {
					if (c[j][0] != null && c[j][0].length != 0) p.plot(c[j][0], c[j][1], "g", "linewidth=0.5", "alpha=0.5");
				}	
				p.plot(m.efit.psiNOps.getAccurateMagneticAxisR(), m.efit.psiNOps.getAccurateMagneticAxisZ(), "go");
			}
			p.plot(m.currents.psiNOps.getAccurateMagneticAxisR(), m.currents.psiNOps.getAccurateMagneticAxisZ(), "bo");
			if(magneticOptions >0) {
				String[] shortIds = (String[]) magneticsTable[0];
				double[] R = (double[]) magneticsTable[1];
				double[] Z = (double[]) magneticsTable[2];
				double[] poloidalAngles = (double[]) magneticsTable[3];
				double[] toroidalAngles = (double[]) magneticsTable[4];
				double[] ZEnabled = insertNansInArray(true, Z, pickupsEnabled);
				// if the unit  digit is 1
				if(magneticOptions % 2  ==1) p.plot(R, ZEnabled, "C3o");
				// if the tens digit is 1
				if(magneticOptions/10 % 2 ==1) {
					for (int i = 0; i < shortIds.length; i++) {
						p.text(R[i] + poloidalAngles[i]/900.0, Z[i] + toroidalAngles[i]/3600.0, shortIds[i]);
					}
				}
			}		
			p.xlabel("$R$ [m]");
			p.ylabel("$Z$ [m]");
			p.legend("loc='best'");
			p.grid();
			p.axis("equal");
			dump(p.getScript(), folder+"scripts/"+filename+".py");
			filename=filename+"."+plotFormat;
			p.savefig("r'"+folder+filename+"'");			
			p.exec();
		}

		
		private void plotColoredGrid(double[] values, String folder, String filename, String plotFormat, boolean plotLCFS,String colorbarLabel, String title) {
			JyPlot p = new JyPlot();
			p.figure();
			p.xlabel("R [m]");
			p.ylabel("Z [m]");
			p.plot(m.firstwall.getR(), m.firstwall.getZ(), "k");
			p.write("m=plt.get_cmap('plasma')");
			double[] coilsR = m.currents.jtor.plasmaBeamGrid.getx();
			double[] coilsZ = m.currents.jtor.plasmaBeamGrid.gety();
			double[] coilsdR = m.currents.jtor.plasmaBeamGrid.getdx();
			double[] coilsdZ = m.currents.jtor.plasmaBeamGrid.getdy();
			double minValue = Algorithms.min(values);
			double maxValue = Algorithms.max(values);
			double[] scaledValues = divide(subtract(values, minValue), maxValue-minValue);
			p.write("values="+Parser.toPythonExpression(scaledValues));
			for(int j=0;j<coilsR.length;++j) {
				double R1 = coilsR[j] - coilsdR[j]/2.0;
				double R2 = coilsR[j] + coilsdR[j]/2.0;
				double Z1 = coilsZ[j] - coilsdZ[j]/2.0;
				double Z2 = coilsZ[j] + coilsdZ[j]/2.0;
				p.plot(new double[] {  R1, R2, R2, R1, R1}, new double[] { Z1, Z1, Z2, Z2, Z1  }, "k", "linewidth=0.01");
				p.write("c=m("+scaledValues[j]+")");
				p.fill_between(new double[] { R1, R2 }, Z1, Z2, "color=c", "alpha=0.6");
			}											
			
			if (plotLCFS) {
				double[][][] npsi0_9 = m.currents.psiNOps.getClosedContoursNormPsiAccurate(new double[] { 0.9 });
				double[][][] npsi0_95 = m.currents.psiNOps.getClosedContoursNormPsiAccurate(new double[] { 0.95 });
				p.plot(m.currents.psiNOps.getLCFSR(), m.currents.psiNOps.getLCFSZ(), "b");
				p.plot(npsi0_9[0][0], npsi0_9[0][1], "b--", "linewidth=0.2");			
				p.plot(npsi0_95[0][0], npsi0_95[0][1], "b--", "linewidth=0.2");						
				p.plot(m.currents.psiNOps.getAccurateMagneticAxisR(), m.currents.psiNOps.getAccurateMagneticAxisZ(), "ro");			
			}
			
			p.write("import matplotlib as mpl");
			p.write("norm = mpl.colors.Normalize(vmin="+minValue+", vmax="+maxValue+")");
			p.write("sm = plt.cm.ScalarMappable(norm=norm, cmap=m)");
			p.write("sm._A = []");
			p.write("cbar=plt.colorbar(sm)");
			p.write("cbar.set_label("+colorbarLabel+", rotation=270)");
			p.title(title);			
			dump(p.getScript(), folder+"/scripts/"+filename+".py");
			p.savefig("r'"+folder+filename+"."+plotFormat+"'");
			p.axis("auto");
			p.xlim(new double[] { 3.5, 4.0 });
			p.ylim(new double[] { -0.5, 1.5 });		
			dump(p.getScript(), folder+"/scripts/"+filename+"_zoomed.py");
			p.savefig("r'"+folder+filename+"_zoomed."+plotFormat+"'");
			p.exec();			
		}


		/**
		 * Gets current profile across mid-plane through the magnetic axis, from the current beams and from the equilibrium virtual observations [p'+1/Rmu_0)ff']
		 *  return arguments: 
		 *		 beamIndicies	   -  Indices of current beams
		 *       Raxis             -  r-coordinates                          (m)
		 *       Zaxis             -  z-coordinates (constant values)        (m)
		 *       beamCurrentDensities     -  toroidal current density    (only enabled currents beams)          (A/m^2)
		 *       beamCurrentDensitiesEqui -  "equi" current density (ie  J=p'+1/Rmu_0)ff'] (array size is same as enabled beams) (A/m^2)
		 *       jtorMidplane             -  midplane toroidal current, using "Full" collection   (A/m^2)
		 *       jtorMidplaneEqui]        -  midplane toroidal current from [p'+1/Rmu_0)ff'] (A/m^2)
		 * 
		 * @author lynton
		 *                                            
		 */
		private static List<Object> getJtorProfiles(JetEqui2DModel4 m) {
			
			double zMidplane = m.currents.psiNOps.getAccurateMagneticAxisZ();
			if (Double.isNaN(zMidplane)) {
				zMidplane = 0.0;
			}
			double[] Raxis = linspace(Algorithms.min(m.firstwall.getR()), Algorithms.max(m.firstwall.getR()), 200);
			double[] Zaxis = fillArray(zMidplane, Raxis.length);
			int[] beamIndicies = m.currents.jtor.plasmaBeamGrid.getCellIndiciesForPos(Raxis, Zaxis);
			double[] jtorMidplane = new double[Raxis.length];
			double[] jtorMidplaneEqui = new double[Raxis.length];
			double[] beamCurrentDensities = OneLiners.arrayMultiply(m.currents.jtor.plasmaBeamCurrentDensities.getEnabledValue(), currentUnit);
			double[] beamCurrentDensitiesEqui = OneLiners.arrayMultiply(m.equi.equi.getEquiCurrentDensity(), currentUnit);
			for(int i=0;i<beamIndicies.length;++i) {
				if (beamIndicies[i] != -1) {
					jtorMidplane[i] = beamCurrentDensities[beamIndicies[i]];
					jtorMidplaneEqui[i] = beamCurrentDensitiesEqui[beamIndicies[i]];
				} else {
					jtorMidplane[i] = 0;
					jtorMidplaneEqui[i] = 0;
				}
			}
			// compute poloidal flux
			double[] npsi = m.currents.psiNOps.toFluxCoord(Raxis, fillArray(0.0, Raxis.length), Zaxis)[0];			
			int indexMinFlux = Algorithms.argmin(npsi);
			// The next bit make the locations inside of the magnetic axis negative so that the npsi profile is monotonic 
			for(int j=0;j<=indexMinFlux;++j) {
				npsi[j] = -npsi[j];
			}
			List<Object> retList = new ArrayList<Object>();
			retList.add(beamIndicies);
			retList.add(Raxis);
			retList.add(Zaxis);
			retList.add(npsi);
			retList.add(beamCurrentDensities);
			retList.add(beamCurrentDensitiesEqui);
			retList.add(jtorMidplane);
			retList.add(jtorMidplaneEqui);
			return retList;
		}
	
		/**
		 * Assembles summaries for the prior (free) and likelihood (obs) PDFs.
		 * 
		 * @author lynton
		 *                                            
		 */
		private List<Object> getPdfSummaries(double[] curObsLogPdf, double[] curFreeLogPdf, double[]prevObsLogPdf,double[]prevFreeLogPdf) {

			List<ProbabilityNode> freeNodes = m.graph.getUnobservedNodes();
			String[] freeNames = new String[freeNodes.size()];
			int index = 0;
			for(ProbabilityNode n : freeNodes) {
				freeNames[index++] = n.getPath();
			}
			List<ProbabilityNode> obsNodes = m.graph.getObservedNodes();
			String[] obsNames = new String[obsNodes.size()];
			index = 0;
			for(ProbabilityNode n : obsNodes) {
				obsNames[index++] = n.getPath();
			}
			if (curObsLogPdf == null) {
				curObsLogPdf = getObsLogPdf(m.graph);
				curFreeLogPdf = getFreeLogPdf(m.graph);
				prevObsLogPdf = curObsLogPdf;
				prevFreeLogPdf = curFreeLogPdf;

			} else {
				prevObsLogPdf = curObsLogPdf;
				prevFreeLogPdf = curFreeLogPdf;
				curObsLogPdf = getObsLogPdf(m.graph);
				curFreeLogPdf = getFreeLogPdf(m.graph);
				
				if ((curObsLogPdf.length != prevObsLogPdf.length) || (curFreeLogPdf.length != prevFreeLogPdf.length)) {
					prevObsLogPdf = curObsLogPdf;
					prevFreeLogPdf = curFreeLogPdf;
				}
			}
			List<Object> retList = new ArrayList<Object>();
			retList.add(obsNames);
			retList.add(freeNames);
			retList.add(curObsLogPdf);
			retList.add(curFreeLogPdf);
			retList.add(prevObsLogPdf);
			retList.add(prevFreeLogPdf);
			return retList;
		}

		
		/**
		 * Assembles of table of the pickup coil geometry.
		 * 
		 * @author lynton
		 *                                            
		 */
		public static Object[] getPickupsTable(JETMagneticDiagnosticsDataSource mag_ds, PickupCoils2DDataSourceJET2 pickups_ds) {
			String[] shortIds = mag_ds.getPickupsShortIds();
			double[] R = pickups_ds.getR();
			double[] Z = pickups_ds.getZ();
			double[] poloidalAngles = mulElem(pickups_ds.getPoloidalAngle(), 180.0/Math.PI);
			double[] toroidalAngles = mulElem(pickups_ds.getToroidalAngle(), 180.0/Math.PI);
			for (int i = 0; i < toroidalAngles.length; i++) {
				if (toroidalAngles[i] < 0.0) {
					toroidalAngles[i] += 360.0;
				}
			}
			String[] Ids = mag_ds.getPickupIds();
			
			return new Object[] { shortIds, R, Z, poloidalAngles, toroidalAngles, Ids };
		}

		
		/**
		 * Method writes out all text-based results.
		 * 
		 * @author lynton
		 *                                            
		 */
		private void dumpTextOutput(String folder, List<Object> pdfSummary, List<Object> jtorProfiles) {
	
			folder = folder+"/textFiles/";

			dump(SpringSerializer.toSpringXML(m.graph), folder+"model-current.xml");			
			
			dumpTextInformativeOutput(folder+"details.txt",pdfSummary,jtorProfiles);

			{
				String[] names = m.diagnostics.magnetics.pickups_ds.getNames();
				boolean[] enabled = m.diagnostics.magnetics.pickups_obs.getEnable();
				double[] predicted =  m.diagnostics.magnetics.pickups_obs.getFullMean();
				double[] observed=  m.diagnostics.magnetics.pickups_obs.getFullValue();
				double[] normdiff = m.diagnostics.magnetics.pickups_obs.diffNormalisedFull();
				dumpTextMagnetics(folder+"pickups.txt","magnetic pickups", names,enabled,observed,predicted,normdiff);
			}		
			{
				String[] names = m.diagnostics.magnetics.saddles_ds.getNames();
				boolean[] enabled = m.diagnostics.magnetics.saddles_obs.getEnable();
				double[] predicted =  m.diagnostics.magnetics.saddles_obs.getFullMean();
				double[] observed=  m.diagnostics.magnetics.saddles_obs.getFullValue();
				double[] normdiff = m.diagnostics.magnetics.saddles_obs.diffNormalisedFull();
				dumpTextMagnetics(folder+"saddles.txt","magnetic saddles", names,enabled,observed,predicted,normdiff);
			}		
			{
				String[] names = m.diagnostics.magnetics.fluxloops_ds.getNames();
				boolean[] enabled = m.diagnostics.magnetics.fluxloops_obs.getEnable();
				double[] predicted =  m.diagnostics.magnetics.fluxloops_obs.getFullMean();
				double[] observed=  m.diagnostics.magnetics.fluxloops_obs.getFullValue();
				double[] normdiff = m.diagnostics.magnetics.fluxloops_obs.diffNormalisedFull();
				dumpTextMagnetics(folder+"fluxloops.txt","magnetic flux loops", names,enabled,observed,predicted,normdiff);
			}		
		}
	
		/**
		 *  Writes out informative summary information.
		 *
		 * @author lynton
		 *                                            
		 */
		private void dumpTextInformativeOutput(String filename, List<Object> pdfSummary, List<Object> jtorProfiles) {
			
			String[] obsNames = (String[]) pdfSummary.get(0);
			String[] freeNames = (String[]) pdfSummary.get(1);
			double [] curObsLogPdf = (double[]) pdfSummary.get(2);
			double [] curFreeLogPdf= (double[]) pdfSummary.get(3);
			double [] prevObsLogPdf= (double[]) pdfSummary.get(4);
			double [] prevFreeLogPdf= (double[]) pdfSummary.get(5);
			double [] Raxis = (double[]) jtorProfiles.get(1);
			double [] Zaxis = (double[]) jtorProfiles.get(2);
			double [] npsi = (double[]) jtorProfiles.get(3);
			double [] jtorMidplane = (double[]) jtorProfiles.get(6);
			double [] jtorMidplaneEqui = (double[]) jtorProfiles.get(7);
			// write out informative output
			{
				// information about the grid 
				double[] R = m.currents.jtor.plasmaBeamGrid.xKnots();
				double[] Z = m.currents.jtor.plasmaBeamGrid.yKnots();
				dump(R, filename);
				dump("poloidalFlux grid", filename);
				String rmin=String.format("%.3fm", R[0]);
				String rmax=String.format("%.3fm", R[R.length-1]);
				String dr=String.format("%.3fcm", (R[1]-R[0])*100.);
				String zmin=String.format("%.3fm", Z[0]);
				String zmax=String.format("%.3fm", Z[Z.length-1]);
				String dz=String.format("%.3fcm", (Z[1]-Z[0])*100.);
				dumpAppend("Rmin="+rmin+", Rmax="+rmax+", dR="+dr, filename);
				dumpAppend("Zmin="+zmin+", Rmax="+zmax+", dZ="+dz, filename);
				
				// information about the prior PDFs (free parameters)
				{
					String str0=String.format("%n%n%s%n", "--------------------------------------");
					dumpAppend(str0, filename);
					dumpAppend("Free parameter PDFs:", filename);
					int maxstr=freeNames[0].length(); 
					for (int i=1;i<freeNames.length;i++) {
						maxstr=Math.max(maxstr,freeNames[i].length()); 
					}
					String fmt=String.format("%%%ds",maxstr);
					String str1=String.format(fmt, " ");
					String str2=String.format("%15s", "prev pdf");
					String str3=String.format("%15s", "current pdf");
					String str4=String.format("%16s", "change in pdf");
					dumpAppend(str1+str2+str3+str4, filename);
					for (int i=0;i<freeNames.length;i++) {
						str1=String.format(fmt, freeNames[i]);
						str2=String.format("%15.5f", prevFreeLogPdf[i]);
						str3=String.format("%15.5f", curFreeLogPdf[i]);
						str4=String.format(" %15.5f", curFreeLogPdf[i]-prevFreeLogPdf[i]);
						dumpAppend(str1+str2+str3+str4, filename);
					}
				}
				// information about the likelihood PDFs (observations)
				{
					String str0=String.format("%n%n%s%n", "--------------------------------------");
					dumpAppend(str0, filename);
					dumpAppend("Observation PDFs:", filename);
					int maxstr=obsNames[0].length(); 
					for (int i=1;i<obsNames.length;i++) {
						maxstr=Math.max(maxstr,obsNames[i].length()); 
					}
					String fmt=String.format("%%%ds",maxstr);
					String str1=String.format(fmt, " ");
					String str2=String.format("%15s", "prev pdf");
					String str3=String.format("%15s", "current pdf");
					String str4=String.format("%16s", "change in pdf");
					dumpAppend(str1+str2+str3+str4, filename);
					for (int i=0;i<obsNames.length;i++) {
						str1=String.format(fmt, obsNames[i]);
						str2=String.format("%15.5f", prevObsLogPdf[i]);
						str3=String.format("%15.5f", curObsLogPdf[i]);
						str4=String.format(" %15.5f", curObsLogPdf[i]-prevObsLogPdf[i]);
						dumpAppend(str1+str2+str3+str4, filename);
					}
				}
			
		
				// toroidal current profile across mid-plane from the current beams and from the equilibrium virtual observations [p'+1/Rmu_0)ff']
				{
					String str0=String.format("%n%n%s%n", "--------------------------------------");
					dumpAppend(str0, filename);
					dumpAppend(String.format("Radial profile of toroidal Current density at Z_mag=%7.3fm",Zaxis[0]), filename);
					String str1=String.format("%12s","R-coordinate");
					String str2=String.format("%16s","Poloidal flux");
					String str3=String.format("%15s", "J_tor(beams)");
					String str4=String.format("%22s", "J_tor=Rp'+1/Rmu0ff'");
					String str5=String.format("%30s", "delta_J=Beams-Rp'+1/Rmu0ff'");
					dumpAppend(str1+str2+str3+str4+str5, filename);
					for (int i=0;i<Raxis.length;i++) {
						str1=String.format("     %7.3f", Raxis[i]);
						str2=String.format("         %7.3f", npsi[i]);
						str3=String.format("%15.5f", jtorMidplane[i]);
						str4=String.format("   %15.5f", jtorMidplaneEqui[i]);
						str5=String.format("         %15.5f", jtorMidplane[i]-jtorMidplaneEqui[i]);
						dumpAppend(str1+str2+str3+str4+str5, filename);
					}
		
				}
				
	//			dump(pickups_normdiff, folder+"pickups_normdiff.txt");
	//			dump(saddles_normdiff, folder+"saddles_normdiff.txt");
	//			dump(fluxloops_normdiff, folder+"fluxloops_normdiff.txt");
	//			dump(npsiGrid, folder+"npsiGrid.txt");
	//			dump(pprime, folder+"pprime.txt");
	//			dump(ffprime, folder+"ffprime.txt");			
	//			dump(m.equi.equi.getR(), folder+"equi_evalR.txt");
	//			dump(m.equi.equi.getZ(), folder+"equi_evalZ.txt");
	//			dump(m.equi.equi.getEquiCurrentDensity(), folder+"equi_jtor.txt");
	//			dump(m.currents.jtor.plasmaBeamGrid.getx(), folder+"equi_beamR.txt");
	//			dump(m.currents.jtor.plasmaBeamGrid.gety(), folder+"equi_beamZ.txt");
	//			dump(m.currents.psiNOps.getLCFS(), folder+"lcfs.txt");			
	//			dump(m.currents.psiNOps.getFOFS(), folder+"fofs.txt");
	//			dump(m.equi.equi.getEquiCurrentDensity(), folder+"equiCurrents.txt");
	//			dump(m.currents.jtor.plasmaBeamCurrentDensities.getEnabledValue(), folder+"beamCurrents.txt");
	//			dump(prevFreeLogPdf, folder+"logPdfPrevFree.txt");
	//			dump(curFreeLogPdf, folder+"logPdfCurFree.txt");
	//			dump(subtract(curFreeLogPdf, prevFreeLogPdf), folder+"logPdfFreeDiffs.txt");
	
			}
		}
	
	
		/**
		 *  Writes out details of fit for a magnetic-type diagnostics.
		 *
		 * @author lynton
		 *                                            
		 */
		private static void dumpTextMagnetics(String filename,String title, String[] signalNames,boolean[] enabled,
				double[] observed,double[] predicted, double[]normdiff) {
	
			dump(title, filename);
	        String str0=String.format("%n%n%s%n", "--------------------------------------");
	        String str1=String.format("%5s", "index");
			String str2=String.format("%14s", "signal name");
			String str3=String.format("%10s", "Enabled");
			String str4=String.format("%11s", "Observed");
			String str5=String.format("%12s", "Predicted");
			String str6=String.format("%14s", "(M-P)/sigma");
			dumpAppend(str0+str1+str2+str3+str4+str5+str6, filename);
			for (int i=0;i<signalNames.length;i++) {
				String name = String.format("%14s",signalNames[i]);
				String enable = enabled[i] ? String.format("%10d",1):String.format("%10s"," ");
				String obs = String.format("%11f",observed[i]);
				String pred = String.format("%12f",predicted[i]);
				String norm = String.format("%14f",normdiff[i]);
				dumpAppend(String.format("%5d",i)+name+enable+obs+pred+norm, filename);
			}				
		}	
		
		/**
		 * Generates a set of plot for magnetic-type diagnostics.
		 *  if there are more than 50 channels, then plots are generated in groups of 50 as well.
		 *  
		 * @author lynton
		 *                                            
		 */
		private static void plot_magnetics(String scriptFilename, String plotFilename, String plotFormat, double[][] predobs, double[] normdiff, boolean[]enabled,
				String titleString, String xlabel, String ylabel1, String ylabel2) {
			int channelMin=-1;
			int channelMax=-1;			
			plot_magnetics(0,channelMin,channelMax,scriptFilename, plotFilename, plotFormat, predobs, normdiff, enabled, titleString, xlabel, ylabel1, ylabel2);
			int channelCount=normdiff.length;
			if (channelCount >= 50) {
				for (int i=0;i <= channelCount/50;i++) {
					channelMin=i*50;
					channelMax=Math.min((i+1)*50-1,channelCount-1);
					String chan="_"+Integer.toString(channelMin)+'_'+Integer.toString(channelMax);
					plot_magnetics(0, channelMin,channelMax,scriptFilename+chan, plotFilename+chan, plotFormat, predobs, normdiff, enabled, titleString, xlabel, ylabel1, ylabel2);
				}
			}
		}
		
		/**
		 * Generates a plot for magnetic-type diagnostics.
		 *  if plotOption =1 then y-range (sigma) of lower plot is between -2<sigma<2, and for the upper plot it over the enabled detectors 
		 *  if channelMin !=-1 then plot in channel range channelMin:channelMax
		 * @author lynton
		 *                                            
		 */
		private static void plot_magnetics(int plotOption,int channelMin, int channelMax, String scriptFilename, String plotFilename, String plotFormat, double[][] predobs, double[] normdiff, boolean[]enabled,
				String titleString, String xlabel, String ylabel1, String ylabel2) {
			double[] predicted=predobs[0];
			double[] sigma=predobs[1];
			double[] observed=predobs[2];
			double[] channels = linspace(0., predicted.length-1, predicted.length);
			JyPlot p = new JyPlot();
			// set up the plot space
			p.write("gs=plt.GridSpec(2,2,width_ratios=[3,1])");
			p.write("f = plt.figure(figsize=(6.8,4.1))");
			p.write("ax1=f.add_subplot(gs[0,0])");
			p.write("ax2=f.add_subplot(gs[1,0],sharex=ax1)");
			p.write("ax3=f.add_subplot(gs[:,1])");
			// make the upper plot
			p.write("plt.axes(ax1)");
			p.errorbar(channels,predicted,sigma,"marker='o',markersize=4,color='r',linestyle='none',label='pred with errors',zorder=-32");  // note zorder param ensurs this is rendered first. 
			double[] observedEnabled = insertNansInArray(true, observed, enabled);
			double[] observedDisabled = insertNansInArray(false, observed, enabled);
			p.plot(channels,observedEnabled,"marker='o',markerfacecolor='g',markeredgecolor='g',markersize=4,color='grey',linestyle='--',linewidth=1,label='observations (enabled)'");
			// the y-range encompasses only the enable detectors
			p.write("ymin,ymax =ax1.get_ylim()");
			p.plot(channels,observedDisabled,"marker='o',markerfacecolor='b',markeredgecolor='b',markersize=4,color='grey',linestyle='--',linewidth=1,label='observations (disabled)'");
			// the y-range encompasses only both enabled and disabled detectors
			if(plotOption != 1) p.write("ymin,ymax =ax1.get_ylim()");
			p.title("'"+titleString+"'");
			p.xlabel(xlabel);
			p.ylabel(ylabel1);
			p.write("ax1.set_ylim(ymin-(ymax-ymin)*0.1,ymax+(ymax-ymin)*0.1)");
			if(channelMin != -1) {
				int xmin= channelMin-1;
				int xmax= channelMax+1;
				p.write("ax1.set_xlim("+Integer.toString(xmin)+","+Integer.toString(xmax)+")");
				p.write("ax1.grid(which='major', axis='x',linestyle='--')");
			}
			p.axhline(0.,"linestyle='-',color='k',linewidth=1");
			// now the lower plot
			p.write("plt.axes(ax2)");
			p.plot(channels,normdiff,"markerfacecolor='m',markeredgecolor='m',marker='o',color='grey',linewidth=1,linestyle='--',markersize=4,label=r'(M-P)/$\\sigma$'");
			p.write("ymin,ymax =ax2.get_ylim()");
			double ymin=Algorithms.min(normdiff);
			double ymax=Algorithms.max(normdiff);
			if(plotOption ==1) {
				// restrict plot range
				ymin=max(ymin,-2);
				ymax=min(ymax,+2);
				p.write("ymin=max(ymin,-2.)");
				p.write("ymax=min(ymax,+2.)");
			}
			p.write("ax2.set_ylim(ymin-(ymax-ymin)*0.1,ymax+(ymax-ymin)*0.1)");
			if(channelMin != -1) {
				int xmin= channelMin-1;
				int xmax= channelMax+1;
				p.write("ax2.set_xlim("+Integer.toString(xmin)+","+Integer.toString(xmax)+")");
				p.write("ax2.grid(which='major', axis='x',linestyle='--')");
			}
			p.xlabel(xlabel);
			p.ylabel(ylabel2);
			for (long i=Math.round(Math.floor(ymin-1));i<Math.round(Math.ceil(ymax+1));i++) p.axhline(i,"linestyle='-',color='k',linewidth=1");
			// now the label
			p.write("plt.axes(ax3)");
			p.gca().axis("off");
			p.write("plt.gcf().legend(loc=7,prop= {'size':9})");
			p.tight_layout();
			dump(p.getScript(), scriptFilename+".py");
			p.savefig("'"+plotFilename+"."+plotFormat+"'");
			p.exec();
			// if the lower plot has a y-range greater then |2 sigma| we plot this again.
			boolean doDetail=false;
			if(plotOption != 1) {
				double ymin1=Algorithms.min(predicted);
				double ymax1=Algorithms.max(predicted);
				double ymin2=Algorithms.min(observedEnabled);
				double ymax2=Algorithms.max(observedEnabled);
				double ymin3=Algorithms.min(observedDisabled);
				double ymax3=Algorithms.max(observedDisabled);
	            if(Math.min(ymin1,Math.min(ymin2,ymin3))< Math.min(ymin1,ymin2) || Math.max(ymax1,Math.max(ymax2,ymax3)) > Math.max(ymax1,ymax2) ) doDetail=true;
			}		
			if(ymax >2 || ymin < -2|doDetail) plot_magnetics(1,channelMin,channelMax,scriptFilename+"_detail",plotFilename+"_detail",plotFormat,predobs,normdiff,enabled,titleString,xlabel,ylabel1,ylabel2);
	
		}		
	}
	
	
	public static void dumpFreeParametersAndObservations(GraphicalModel g, String filename) {
		try(PrintWriter out = new PrintWriter(filename)) {
		
			out.println("Free parameters");
			int index = 0;
			for(ProbabilityNode node : g.getUnobservedNodes()) {
				out.println(node.getName()+" (dim: "+node.dim()+"), indicies: "+(index)+"->"+(index+node.dim()-1));
				index += node.dim();
			}
			
			out.println("Observations");
			index = 0;
			for(ProbabilityNode node : g.getObservedNodes()) {
				out.println(node.getName()+" (dim: "+node.dim()+"), indicies: "+(index)+"->"+(index+node.dim()-1));
				index += node.dim();
			}
		} catch (Exception ex) {
			ex.printStackTrace();
		}
				
	}
	
	public static boolean hasData(boolean[] enable) {
		for(int i=0;i<enable.length;++i) {
			if (enable[i]) return true;
		}
		return false;
	}

	
	/**
	 * Inserts NaN  in element 'k' if either (1) flag[k]=true and option=true, or (2) flag[k]=false and option=false.
	 * Based on the unshuffleArray method defined in the Multivariate class
	 *
	 * @author lynton
	 *                                            
	 */
	private static double[] insertNansInArray(boolean option, double[] array, boolean[] flag) {
		double[] ret = new double[array.length];	
		for(int i = 0; i < flag.length; ++i) {
		    ret[i] = flag[i]==option ? array[i] : Double.NaN;
		}
		return ret;
	}
}
