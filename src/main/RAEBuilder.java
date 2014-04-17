package main;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.PrintStream;
import java.util.Collection;
import java.util.List;

import org.jblas.DoubleMatrix;

import math.DifferentiableFunction;
import math.DifferentiableMatrixFunction;
import math.Minimizer;
import math.Norm1Tanh;
import math.QNMinimizer;

import classify.Accuracy;
import classify.ClassifierTheta;
import classify.LabeledDatum;

import classify.SoftmaxClassifier;

import rae.FineTunableTheta;
import rae.RAECost;
import rae.RAEFeatureExtractor;

public class RAEBuilder {
    FineTunableTheta initialTheta;
    RAEFeatureExtractor FeatureExtractor;
    DifferentiableMatrixFunction f;

    public RAEBuilder() {
        initialTheta = null;
        FeatureExtractor = null;
        f = new Norm1Tanh();
    }

    public static void main(final String[] args) throws Exception {
        RAEBuilder rae = new RAEBuilder();

        Arguments params = new Arguments();
        params.parseArguments(args);
        if (params.exitOnReturn)
            return;

        System.out.printf("dictionarySize : %d\nembeddingSize : %d\n",
                params.dictionarySize, params.hiddenSize);

        if (params.trainModel) {
            System.out.println("Training the RAE. Model file will be saved in "
                    + params.ModelFile);
            FineTunableTheta tunedTheta = rae.train(params);
            tunedTheta.dump(params.ModelFile);

            System.out.println("RAE trained. The model file is saved in "
                    + params.ModelFile);

            RAEFeatureExtractor fe = new RAEFeatureExtractor(
                    params.embeddingSize, tunedTheta, params.AlphaCat,
                    params.Beta, params.catSize, params.Dataset.Vocab.size(),
                    rae.f);

            List<LabeledDatum<Double, Integer>> classifierTrainingData
                    = fe.extractFeaturesIntoArray(params.Dataset, params.Dataset.Data, params.treeDumpDir);

            SoftmaxClassifier<Double, Integer> classifier = new SoftmaxClassifier<Double, Integer>();
            Accuracy trainAccuracy = classifier.train(classifierTrainingData);
            System.out.println("Train Accuracy :" + trainAccuracy.toString());

            System.out.println("Classifier trained. The model file is saved in "
                    + params.ClassifierFile);
            classifier.dump(params.ClassifierFile);

            if (params.featuresOutputFile != null)
                rae.dumpFeatures(params.featuresOutputFile,
                        classifierTrainingData);

            if (params.ProbabilitiesOutputFile != null)
                rae.dumpProbabilities(params.ProbabilitiesOutputFile,
                        classifier.getTrainScores());

//			if (params.treeDumpDir != null)
//				rae.dumpTrees(Trees , params.treeDumpDir, params.Dataset, params.Dataset.Data);

            System.out.println("Dumping complete");

        } else {
            System.out.println
                    ("Using the trained RAE. Model file retrieved from " + params.ModelFile
                            + "\nNote that this overrides all RAE specific arguments you passed.");

            FineTunableTheta tunedTheta = rae.loadRAE(params);
            if (params.matrixDumpDir != null) {
                tunedTheta.dumpAsCSV(params.matrixDumpDir);
                tunedTheta.dumpWords(params.matrixDumpDir+"/wordsVector.txt", params.matrixDumpDir+"/wordmap.map");
                System.out.println("Theta dumped to " + params.matrixDumpDir + "/finedTunedTheta.csv");
            }
            assert tunedTheta.getNumCategories() == params.Dataset.getCatSize();

            SoftmaxClassifier<Double, Integer> classifier = rae.loadClassifier(params);

            RAEFeatureExtractor fe = new RAEFeatureExtractor(
                    params.embeddingSize, tunedTheta, params.AlphaCat,
                    params.Beta, params.catSize, params.Dataset.Vocab.size(),
                    rae.f);

            if (params.Dataset.Data.size() > 0) {
                System.err.println("There is training data in the directory.");
                System.err.println("It will be ignored when you are not in the training mode.");
            }

            List<LabeledDatum<Double, Integer>> classifierTestingData
                    = fe.extractFeaturesIntoArray(params.Dataset, params.Dataset.TestData, params.treeDumpDir);

            Accuracy TestAccuracy = classifier.test(classifierTestingData);
            if (params.isTestLabelsKnown) {
                System.out.println("Test Accuracy : " + TestAccuracy);
            }

            if (params.featuresOutputFile != null)
                rae.dumpFeatures(params.featuresOutputFile,
                        classifierTestingData);

            if (params.ProbabilitiesOutputFile != null)
                rae.dumpProbabilities(params.ProbabilitiesOutputFile,
                        classifier.getTestScores());

//			if (params.treeDumpDir != null)
//				rae.DumpTrees(testTrees, params.treeDumpDir, params.Dataset, params.Dataset.TestData);
        }

        System.exit(0);
    }

    public void dumpFeatures(String featuresOutputFile,
                             List<LabeledDatum<Double, Integer>> Features)
            throws FileNotFoundException {

        PrintStream out = new PrintStream(featuresOutputFile);
        for (LabeledDatum<Double, Integer> data : Features) {
            Collection<Double> features = data.getFeatures();
            for (Double f : features)
                out.printf("%.8f ", f.doubleValue());
            out.println();
        }
        out.close();
    }

    public void dumpProbabilities(String ProbabilitiesOutputFile,
                                  DoubleMatrix classifierScores) throws IOException {

        PrintStream out = new PrintStream(ProbabilitiesOutputFile);
        for (int dataIndex = 0; dataIndex < classifierScores.columns; dataIndex++) {
            // params.Dataset.getLabelString(l.intValue())
            for (int classNum = 0; classNum < classifierScores.rows; classNum++)
                out.printf("%d : %.3f, ", classNum, classifierScores.get(
                        classNum, dataIndex));
            out.println();
        }
        out.close();
    }

    private FineTunableTheta train(Arguments params) throws IOException,
            ClassNotFoundException {

        initialTheta = new FineTunableTheta(params.embeddingSize,
                params.embeddingSize, params.catSize, params.dictionarySize,
                true);

        FineTunableTheta tunedTheta = null;

        RAECost RAECost = new RAECost(params.AlphaCat, params.catSize,
                params.Beta, params.dictionarySize, params.hiddenSize,
                params.visibleSize, params.Lambda, initialTheta.We,
                params.Dataset.Data, null, f);

        Minimizer<DifferentiableFunction> minFunc = new QNMinimizer(10,
                params.maxIterations);

        double[] minTheta = minFunc.minimize(RAECost, 1e-6, initialTheta.Theta,
                params.maxIterations);

        tunedTheta = new FineTunableTheta(minTheta, params.hiddenSize,
                params.visibleSize, params.catSize, params.dictionarySize);

        // Important step
        tunedTheta.setWe(tunedTheta.We.add(initialTheta.We));
        return tunedTheta;
    }

    private FineTunableTheta loadRAE(Arguments params) throws IOException,
            ClassNotFoundException {
        FineTunableTheta tunedTheta = null;
        FileInputStream fis = new FileInputStream(params.ModelFile);
        ObjectInputStream ois = new ObjectInputStream(fis);
        tunedTheta = (FineTunableTheta) ois.readObject();
        ois.close();
        return tunedTheta;
    }

    private SoftmaxClassifier<Double, Integer> loadClassifier
            (Arguments params) throws IOException, ClassNotFoundException {
        SoftmaxClassifier<Double, Integer> classifier = null;
        FileInputStream fis = new FileInputStream(params.ClassifierFile);
        ObjectInputStream ois = new ObjectInputStream(fis);
        ClassifierTheta ClassifierTheta = (ClassifierTheta) ois.readObject();
        ois.close();
        classifier = new SoftmaxClassifier<Double, Integer>
                (ClassifierTheta, params.Dataset.getLabelSet());
        return classifier;
    }
}
