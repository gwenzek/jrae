package classify;

import math.DifferentiableFunction;
import math.Minimizer;
import math.QNMinimizer;
import org.jblas.DoubleMatrix;
import util.Counter;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Collection;
import java.util.List;
import java.util.Set;

/**
 * TODO Make it more generic later
 * Currently works only for <Double,L>, type-casts inside
 * that may go wrong if you try with something else.
 *
 * @author ssanjeev
 */
public class SoftMaxClassifier<F, L>
        implements ProbabilisticClassifier<F, L>, Serializable {

    private static final long serialVersionUID = 3043693828283322518L;
    private final int MaxIterations = 100;
    private final double Lambda = 1e-6;
    private Counter<L> LabelSet;
    private DoubleMatrix trainScores, testScores;
    private SoftmaxCost costFunction;
    int CatSize;

    ClassifierTheta ClassifierTheta;
    Minimizer<DifferentiableFunction> minFunc;

    Accuracy trainAccuracy, testAccuracy;

    public SoftMaxClassifier() {
        LabelSet = new Counter<L>();
        minFunc = new QNMinimizer(10, MaxIterations);
    }

    public SoftMaxClassifier(ClassifierTheta ClassifierParams, Set<L> labelSet) {
        CatSize = 0;

        LabelSet = new Counter<L>();
        minFunc = new QNMinimizer(10, MaxIterations);
        for (L label : labelSet)
            LabelSet.setCount(label, CatSize++);

        ClassifierTheta = ClassifierParams;
        assert CatSize == ClassifierTheta.CatSize;
    }

    public DoubleMatrix getTrainingResults(List<LabeledDatum<F, L>> Data) {
        populateLabels(Data);
        DoubleMatrix Features = makeFeatureMatrix(Data);
        int FeatureLength = Features.rows;

        int[] Labels = makeLabelVector(Data);
        costFunction = new SoftmaxCost(Features, Labels, CatSize, Lambda);

        ClassifierTheta = new ClassifierTheta(FeatureLength, CatSize);
        double[] InitialTheta = ClassifierTheta.Theta;

        double[] OptimalTheta = minFunc.minimize(costFunction, 1e-6, InitialTheta);
        ClassifierTheta = new ClassifierTheta(OptimalTheta, FeatureLength, CatSize);
        return costFunction.getPredictions(ClassifierTheta, Features);
    }

    public Accuracy train(List<LabeledDatum<F, L>> Data) {
        trainScores = getTrainingResults(Data);
        int[] GTLabels = makeLabelVector(Data);
        int[] Predictions = trainScores.columnArgmaxs();
        trainAccuracy = new Accuracy(Predictions, GTLabels, CatSize);
        return trainAccuracy;
    }

    public Accuracy test(List<LabeledDatum<F, L>> Data) {
        DoubleMatrix Features = makeFeatureMatrix(Data);
        int[] Labels = makeLabelVector(Data);
        costFunction = new SoftmaxCost(CatSize, ClassifierTheta.FeatureLength, Lambda);
        testScores = costFunction.getPredictions(ClassifierTheta, Features);
        int[] Predictions = testScores.columnArgmaxs();
        testAccuracy = new Accuracy(Predictions, Labels, CatSize);
        return testAccuracy;
    }

    public DoubleMatrix getTrainScores() {
        if (trainScores == null)
            System.err.println("Train scores polled before training! Will be null.");
        return trainScores;
    }

    public DoubleMatrix getTestScores() {
        if (testScores == null)
            System.err.println("Test scores polled before training! Will be null.");
        return testScores;
    }

    private void populateLabels(List<LabeledDatum<F, L>> Data) {
        CatSize = 0;
        for (LabeledDatum<F, L> datum : Data) {
            L label = datum.getLabel();
            if (!LabelSet.containsKey(label)) {
                System.err.println("Label " + label);
                LabelSet.setCount(label, CatSize++);
            }
        }
        CatSize = LabelSet.keySet().size();
    }

    private int[] makeLabelVector(List<LabeledDatum<F, L>> Data) {
        int[] Labels = new int[Data.size()];
        int i = 0;
        for (LabeledDatum<F, L> datum : Data)
            Labels[i++] = (int) LabelSet.getCount(datum.getLabel());
        return Labels;
    }

    private DoubleMatrix makeFeatureMatrix(List<LabeledDatum<F, L>> Data) {
        int NumExamples = Data.size();
        if (NumExamples == 0)
            return null;

        int FeatureLength = Data.get(0).getFeatures().size();
        double[][] features = new double[FeatureLength][NumExamples];
        for (int i = 0; i < NumExamples; i++) {
            Collection<F> tf = Data.get(i).getFeatures();
            int j = 0;
            for (F f : tf) {
                features[j][i] = (Double) f;
                j++;
            }
        }
        return new DoubleMatrix(features);
    }

    @Override
    public L getLabel(Datum<F> datum) {
        Counter<L> probabilities = getProbabilities(datum);
        return probabilities.argMax();
    }

    @Override
    public Counter<L> getProbabilities(Datum<F> datum) {
        Collection<F> f = datum.getFeatures();
        Counter<L> probabilities = new Counter<L>();
        DoubleMatrix Features = DoubleMatrix.zeros(ClassifierTheta.FeatureLength, 1);
        int i = 0;
        for (F feature : f) {
            Features.put(i, 0, (Double) feature);
            i++;
        }

        if (costFunction == null)
            costFunction = new SoftmaxCost(CatSize, ClassifierTheta.FeatureLength, Lambda);

        DoubleMatrix Scores = costFunction.getPredictions(ClassifierTheta, Features);

        for (L label : LabelSet.keySet()) {
            int labelIndex = (int) LabelSet.getCount(label);
            probabilities.setCount(label, Scores.get(labelIndex, 0));
        }
        return probabilities;
    }

    @Override
    public Counter<L> getLogProbabilities(Datum<F> datum) {
        Counter<L> probablities = getProbabilities(datum);
        Counter<L> logProbablities = new Counter<L>();
        for (L label : probablities.keySet()) {
            double logProb = Math.log(probablities.getCount(label));
            logProbablities.setCount(label, logProb);
        }
        return logProbablities;
    }

    public void dump(String fileName) throws IOException {
        FileOutputStream fos = new FileOutputStream(fileName);
        ObjectOutputStream oos = new ObjectOutputStream(fos);
        oos.writeObject(ClassifierTheta);
        oos.flush();
        oos.close();
    }
}
