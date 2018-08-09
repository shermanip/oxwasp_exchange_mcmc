package jet.apps.lappel.voEqui.magnetics;

import seed.minerva.equilibrium.EquilibriumModelOnBeamSet;
import static seed.matrix.Mat.stop;
import java.util.stream.IntStream;

public class Runs3 {
	
	public static void main(String[] args) throws Exception {

		//  if there are any input arguments and more than one is supplied print help message and stop. 
		if (args != null && args.length > 1) {
			System.out.println("Usage: java -jar jar-file-name <filename>");
			stop();
		}
		String propertiesFilename=null;
		if(args.length == 1) propertiesFilename=args[0];
		//runOne(92274, 51.0);
		//runMany();
		runOne(propertiesFilename);
	}
	
	public static void runOne(String propertiesFilename)  throws Exception {

		
		// instantiate a RuNVoEquiRun object.
		RunVoEqui.Run run = new RunVoEqui.Run();
		
		// initialize explicitly members of the run class.
		run.pulse = 84600;
		run.time = 51.7723;	
		run.runName = "testo1";
		run.useEfitMagnetics = false;
		run.postfix = "-1_efit";
		run.doInitCtHyperParameterOptimization = true;
		run.initCTSigmaR = 0.5789473684210527;	
		run.initCTSigmaf = 614.2857142857142;
		run.plasmaBeamGridNumR = 50;
		run.plasmaBeamGridNumZ = 70;
		run.useBoundaryBeamConstraint = true;
		run.boundaryBeamsConstraintSigma = 50.0; // kA/m2
		
		run.useEquiConstaint = true;
		run.subdivideLCFS = false;
		run.subdivideProximityLHS = 0.08;
		run.subdivideProximityRHS = 0.08;
		run.subdivideMaxGridSize = 0.025;		
		run.setEquiCurrentSigma(50e3);
		
		run.ffprimeFree = true;
		run.ffprimeFreeHyperParameters = true;

		run.pprimeFree = true;
		run.pprimeFreeHyperParameters = true;

		run.numProfileValues = 100;
		
		// temporary values (LCA)
		run.doInitCtHyperParameterOptimization = true;
		run.plasmaBeamGridNumR = 5;
		run.plasmaBeamGridNumZ = 5;
		run.fluxGridNumR = 5;
		run.fluxGridNumZ = 5;
		run.numProfileValues = 75;
		run.useEquiConstaint = true;
		run.ffprimeFreeHyperParameters = false;
		run.pprimeFreeHyperParameters = false;
		run.map=false;
		run.mcmc=true;
		// controls whether the flux functions are defined with psi or root psi as independent variable.
		run.rhoIsSqrtPsiN = true;

		// overide the values with those provided in the properties file 
		if(propertiesFilename != null)
			run.readFromPropertiesFile(propertiesFilename);

		// now get on with the inference calculation
		RunVoEqui.run(run);
	}
	
	public static void runMany()  throws Exception {
		//int[] pulses = new int[] {  85109, 85219, 85220, 85222, 85224, 85228, 85229, 85231, 85262, 85263, 85264, 85265, 85266, 85267, 85268, 85269, 85270, 85272, 85273, 85274, 85275, 85276, 85277, 85278, 85290, 85300, 85371, 85383, 85384, 85385, 85387, 85393, 85406, 85407, 85408, 85410, 85411, 85412, 85415, 85423, 85424, 85425, 85428, 85438 };
		int[] pulses = new int[] { 86465, 86470, 86529, 86625, 86631 };
		//final int pulse = 87562;
		//final double[] times = linspace(42, 58, 7);
		
		//IntStream.range(0, times.length).parallel().forEach((i) -> {
		IntStream.range(0, pulses.length).parallel().forEach((i) -> {
			System.out.println("Trying to start thread "+i);
			try {
				//Thread.sleep(30000);
			} catch (Exception ex) {
				ex.printStackTrace();
			}
			System.out.println("Starting thread "+i);
			try {
				RunVoEqui.run(createRun(pulses[i], 51.0));
			} catch (Exception ex) {
				ex.printStackTrace();
			}
		});
	}
	
	public static RunVoEqui.Run createRun(int pulse, double time) {
		RunVoEqui.Run run = new RunVoEqui.Run();
		run.pulse = pulse;
		run.time = time;
		
		run.runName = "autos";
		run.useEfitMagnetics = false;
		run.postfix = "-1_longrun";
		run.doInitCtHyperParameterOptimization = true;
		run.initCTSigmaR = 0.5789473684210527;	
		run.initCTSigmaf = 614.2857142857142;
		run.plasmaBeamGridNumR = 50;
		run.plasmaBeamGridNumZ = 70;
		run.useBoundaryBeamConstraint = true;
		run.boundaryBeamsConstraintSigma = 50.0; // kA/m2
		
		run.useEquiConstaint = true;
		run.subdivideLCFS = false;
		run.subdivideProximityLHS = 0.08;
		run.subdivideProximityRHS = 0.08;
		run.subdivideMaxGridSize = 0.025;		
		run.setEquiCurrentSigma(50e3);
		
		run.numProfileValues = 100;
		
		return run;
	}

}
