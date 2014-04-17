package main;

import classify.Accuracy;
import classify.LabeledDatum;
import classify.SoftMaxClassifier;
import classify.StratifiedCrossValidation;
import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;
import math.*;
import org.jblas.DoubleMatrix;
import rae.FineTunableTheta;
import rae.RAECost;
import rae.RAEFeatureExtractor;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.List;

public class FullRun {

    public static void main(final String[] args) throws Exception {
        Arguments params = new Arguments();
        params.parseArguments(args);

        if (params.exitOnReturn)
            return;

        RAECost RAECost = null;
        FineTunableTheta initialTheta = null;
        RAEFeatureExtractor FeatureExtractor = null;
        DifferentiableMatrixFunction f = new Norm1Tanh();

        System.out.printf("%d\n%d\n", params.dictionarySize, params.hiddenSize);

        StratifiedCrossValidation<LabeledDatum<Integer, Integer>, Integer, Integer> cv
                = new StratifiedCrossValidation<LabeledDatum<Integer, Integer>, Integer, Integer>(params.numFolds, params.Dataset);
        FineTunableTheta tunedTheta = null;

        for (int foldNumber = 0; foldNumber < params.numFolds; foldNumber++) {
            long startTime = System.nanoTime();
            initialTheta = new FineTunableTheta(params.embeddingSize, params.embeddingSize,
                    params.catSize, params.dictionarySize, true);

            List<LabeledDatum<Integer, Integer>> trainingData = cv.getTrainingData(foldNumber); //,numFolds);
            List<LabeledDatum<Integer, Integer>> testData = cv.getValidationData(foldNumber);

            if (params.trainModel) {
                RAECost = new RAECost(params.AlphaCat, params.catSize, params.Beta, params.dictionarySize,
                        params.hiddenSize, params.visibleSize, params.Lambda, initialTheta.We, trainingData, null, f);
                Minimizer<DifferentiableFunction> minFunc = new QNMinimizer(10, params.maxIterations);

                double[] minTheta = minFunc.minimize(RAECost, 1e-6, initialTheta.theta, params.maxIterations);
                tunedTheta = new FineTunableTheta(minTheta, params.hiddenSize,
                        params.visibleSize, params.catSize, params.dictionarySize);
            } else {
                System.out.println("Reading in the pre-computed RAE ...");

                FileInputStream fis = new FileInputStream(params.dir + "/opttheta.dat");
                ObjectInputStream ois = new ObjectInputStream(fis);
                tunedTheta = (FineTunableTheta) ois.readObject();
                ois.close();

                initialTheta = new FineTunableTheta(params.embeddingSize, params.embeddingSize,
                        params.catSize, params.dictionarySize, true);
                initialTheta.setWe(DoubleMatrix.zeros(params.hiddenSize, params.dictionarySize));
            }

            // Important step
            tunedTheta.setWe(tunedTheta.We.add(initialTheta.We));
//			tunedTheta.dump(params.dir + "/" + params.ModelFile + params.AlphaCat + "." + params.Beta + ".rae");
            tunedTheta.dump(params.ModelFile);

            System.out.println("Extracting features ...");

            FeatureExtractor = new RAEFeatureExtractor(params.embeddingSize, tunedTheta,
                    params.AlphaCat, params.Beta, params.catSize, params.dictionarySize, f);

            List<LabeledDatum<Double, Integer>> classifierTrainingData
                    = FeatureExtractor.extractFeaturesIntoArray(trainingData);

            List<LabeledDatum<Double, Integer>> classifierTestingData
                    = FeatureExtractor.extractFeaturesIntoArray(testData);

            SoftMaxClassifier<Double, Integer> classifier = new SoftMaxClassifier<Double, Integer>();

            Accuracy TrainAccuracy = classifier.train(classifierTrainingData);
            Accuracy TestAccuracy = classifier.test(classifierTestingData);
            System.out.println("Train Accuracy :" + TrainAccuracy.toString());
            System.out.println("Test Accuracy :" + TestAccuracy.toString());
            long endTime = System.nanoTime();
            long duration = endTime - startTime;
            System.out.println("Fold " + foldNumber + " took " + duration / (1000 * 1000) + "ms ");
        }
    }

    public static DoubleMatrix ReadMatrix(String file, String var) throws IOException {
        MatFileReader mfr = new MatFileReader(file);
        MLArray mlArrayRetrived = mfr.getMLArray(var);
        return new DoubleMatrix(((MLDouble) mlArrayRetrived).getArray());
    }

}
