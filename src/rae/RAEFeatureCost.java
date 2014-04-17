package rae;

import classify.LabeledDatum;
import math.DifferentiableMatrixFunction;
import math.DoubleArrays;
import org.jblas.DoubleMatrix;
import util.ArraysHelper;
import util.DoubleMatrixFunctions;

public class RAEFeatureCost extends OnePassCost {
    DoubleMatrix WeOrig;
    Theta theta;

    public RAEFeatureCost(double AlphaCat, double Beta,
                          int DictionaryLength, int HiddenSize, double[] Lambda,
                          DifferentiableMatrixFunction f, DoubleMatrix WeOrig, Theta Theta) {
        super(AlphaCat, Beta, DictionaryLength, HiddenSize, Lambda, f);
        this.WeOrig = WeOrig;
        this.theta = Theta;
        Propagator = new RAEPropagation(Theta, AlphaCat, Beta, HiddenSize, DictionaryLength, f);
    }

    public LabeledRAETree Compute(LabeledDatum<Integer, Integer> Data)
            throws Exception {
        int[] WordIndices = ArraysHelper.getIntArray(Data.getFeatures());
        DoubleMatrix L = theta.We.getColumns(WordIndices);
        DoubleMatrix WordsEmbedded = (WeOrig.getColumns(WordIndices)).addi(L);

        int CurrentLabel = Data.getLabel();
        int SentenceLength = WordsEmbedded.columns;

        if (SentenceLength == 1)
            return null;

        LabeledRAETree Tree = Propagator.ForwardPropagate
                (theta, WordsEmbedded, null, CurrentLabel, SentenceLength);
        Propagator.BackPropagate(Tree, theta, WordIndices);

        lock.lock();
        {
            cost += Tree.totalScore;
            num_nodes += SentenceLength - 1;
        }
        lock.unlock();
        return Tree;
    }

    @Override
    public double getCost() {
        calculateCosts(theta);
        return cost;
    }

    protected void calculateCosts(Theta theta) {
        double WNormSquared = DoubleMatrixFunctions.SquaredNorm(theta.W1) + DoubleMatrixFunctions.SquaredNorm(theta.W2) +
                DoubleMatrixFunctions.SquaredNorm(theta.W3) + DoubleMatrixFunctions.SquaredNorm(theta.W4);

        cost = (1.0f / num_nodes) * cost + 0.5 * LambdaW * WNormSquared
                + 0.5 * LambdaL * DoubleMatrixFunctions.SquaredNorm(theta.We);

        double[] CalcGrad = (new Theta(Propagator.GW1, Propagator.GW2,
                Propagator.GW3, Propagator.GW4, Propagator.GWe_total,
                Propagator.Gb1, Propagator.Gb2, Propagator.Gb3)).theta;

        DoubleMatrix b0 = DoubleMatrix.zeros(HiddenSize, 1);
        double[] WeightedGrad = (new Theta(theta.W1.mul(LambdaW), theta.W2.mul(LambdaW), theta.W3.mul(LambdaW),
                theta.W4.mul(LambdaW), theta.We.mul(LambdaL), b0, b0, b0)).theta;

        DoubleArrays.scale(CalcGrad, (1.0f / num_nodes));
        Gradient = DoubleArrays.add(CalcGrad, WeightedGrad);
    }
}
